"""
Training script for the Lagrangian Multiplier GNN.
"""
import os
import json
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import yaml
import torch
import torch.optim as optim
from torch_geometric.data import Batch

# Resolve paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(BASE_DIR)

from src_ml.input_graph import create_bipartite_graph
from src_ml.model import LagrangianMultiplierModel
from src_ml.loss_mc import BatchedSubgradientLoss, worker_init
from src_ml.feature_extractor import extract_features_single

# Config

# Load config
config = {}
for conf_file in ["config/config_general.yaml", "config/config_ml.yaml"]:
    path = os.path.join(BASE_DIR, conf_file)
    if os.path.exists(path):
        with open(path, 'r') as f:
            config.update(yaml.safe_load(f))

# Training paths
DATA_DIR       = config['general_settings']['data_dir']

LP_DIR         = os.path.join(DATA_DIR, "lpfiles")         # contains training/, val/, test/ subdirs
DEC_DIR        = os.path.join(DATA_DIR, "decfiles")         # .dec files (same stem as .lp)
FEATURE_DIR    = os.path.join(DATA_DIR, "featureinf")     # .json ML input files land here
BOUNDS_DIR     = os.path.join(DATA_DIR, "bounds")
LABELS_PATH    = os.path.join(BOUNDS_DIR, "lagrangian_bounds.json")
LP_BOUNDS_PATH = os.path.join(BOUNDS_DIR, "lp_bounds.json")
WEIGHTS_PATH   = os.path.join(DATA_DIR, "weights", config['general_settings'].get('weights_filename', 'gaptest.pth'))
RESULTS_PATH   = os.path.join(DATA_DIR, "results.json")
PLOT_PATH      = os.path.join(DATA_DIR, "training_plots", "training_loss_plot.png")
BOXPLOT_PATH   = os.path.join(DATA_DIR, "training_plots", "best_model_boxplots.png")


TRAIN_LP_DIR   = os.path.join(LP_DIR, "training")
VAL_LP_DIR     = os.path.join(LP_DIR, "val")

# Model architecture
FEATURE_EMBEDDING_SIZE = config['model_architecture']['feature_embedding_size']
ENCODER_MLP_DIMS       = config['model_architecture']['encoder_mlp_dims']
GNN_MLP_DIMS           = config['model_architecture']['gnn_mlp_dims']
DECODER_MLP_DIMS       = config['model_architecture']['decoder_mlp_dims']
NUM_MESSAGE_PASSING_LAYERS = config['model_architecture']['num_message_passing_layers']
DROPOUT                = config['model_architecture'].get('dropout', 0.1)

# Training parameters
LOAD_PRETRAINED                    = config['training_parameters']['load_pretrained']
LEARNING_RATE                      = float(config['training_parameters']['learning_rate'])
EPOCHS                             = config['training_parameters']['epochs']
USE_RANDOM_SUBSET                  = config['training_parameters'].get('use_random_subset', False)
RANDOM_SUBSET_RATIO                = float(config['training_parameters'].get('random_subset_ratio', 1.0))
RANDOM_SEED                        = config['general_settings']['random_seed']
PLOT_EVERY_N_EPOCHS                = config['training_parameters']['plot_every_n_epochs']
EVALUATE_VALIDATION_LOSS           = config['training_parameters']['evaluate_validation_loss']
EVALUATE_VALIDATION_EVERY_N_EPOCHS = config['training_parameters'].get('evaluate_validation_every_n_epochs', 1)
PLOT_MIN_GAP_CLOSED                = config['training_parameters']['plot_min_gap_closed']
USE_GRADIENT_CLIPPING              = config['training_parameters']['use_gradient_clipping']
GRADIENT_CLIPPING_MAX_NORM         = config['training_parameters']['gradient_clipping_max_norm']
USE_LR_SCHEDULER                   = config['training_parameters']['use_lr_scheduler']
LR_SCHEDULER_FACTOR                = config['training_parameters']['lr_scheduler_factor']
LR_SCHEDULER_PATIENCE              = config['training_parameters']['lr_scheduler_patience']

# Parallelization
NUM_CORES  = config['general_settings']['num_cores']
BATCH_SIZE = config['general_settings']['gpu_batch_size']

# Caching settings
CACHE_GRAPH              = config['training_parameters']['caching_settings']['cache_graph']
CACHE_LAG_RELAX          = config['training_parameters']['caching_settings'].get('cache_lag_relax', False)
RESHUFFLE_EVERY_N_EPOCHS = config['training_parameters']['caching_settings'].get('reshuffle_allocation_every_n_epochs', 0)
WARMUP_EPOCHS            = config['training_parameters']['caching_settings'].get('warmup_epochs_without_cache', 0)
SMA_ALPHA                = config['training_parameters']['caching_settings'].get('solve_time_sma_alpha', 0.5)
TRACK_AFTER_WARMUP       = config['training_parameters']['caching_settings'].get('track_solve_times_after_warmup', False)

# Feature JSON generation

def _gen_json_worker(args):
    """Extracts features for one instance and writes to JSON."""
    lp_path, dec_path, out_json = args
    maxrounds = config['feature_extraction'].get('presolving_maxrounds', 0)
    feature_dict, lp_obj_val = extract_features_single(lp_path, dec_path, maxrounds=maxrounds, quiet=True, extract_lp_bound=True)
    import json as _json
    with open(out_json, 'w') as fh:
        _json.dump(feature_dict, fh, indent=4)
    return out_json, lp_obj_val

def ensure_feature_jsons(lp_subdir: str, label: str):
    """Ensures feature JSONs exist for all LP files."""
    os.makedirs(FEATURE_DIR, exist_ok=True)

    lp_files = sorted(f for f in os.listdir(lp_subdir) if f.endswith(".lp"))
    if not lp_files:
        raise FileNotFoundError(f"No .lp files found in '{lp_subdir}'")

    existing_lp_bounds = {}
    if LP_BOUNDS_PATH and os.path.exists(LP_BOUNDS_PATH):
        import json as _json
        try:
            with open(LP_BOUNDS_PATH, 'r') as f:
                existing_lp_bounds = _json.load(f)
        except _json.JSONDecodeError:
            pass

    missing = []
    for lp_name in lp_files:
        stem = os.path.splitext(lp_name)[0]
        json_path = os.path.join(FEATURE_DIR, stem + ".json")
        lp_path  = os.path.join(lp_subdir, lp_name)
        dec_path = os.path.join(DEC_DIR, stem + ".dec")
        
        needs_gen = False
        if not os.path.exists(json_path):
            needs_gen = True
        elif LP_BOUNDS_PATH and stem not in existing_lp_bounds:
            needs_gen = True
            
        if needs_gen:
            if not os.path.exists(dec_path):
                print(f"  [!] Skipping {stem}: no matching .dec file at {dec_path}")
                continue
            missing.append((lp_path, dec_path, json_path))

    if missing:
        lp_bounds_updates = {}
        print(f"  [{label}] Generating {len(missing)} missing .json feature file(s) "
              f"using {NUM_CORES} core(s) ...")
        workers = min(NUM_CORES, len(missing))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_gen_json_worker, args): args[0] for args in missing}
            done = 0
            for future in as_completed(futures):
                done += 1
                try:
                    out, lp_obj_val = future.result()
                    stem = os.path.basename(out).replace(".json", "")
                    lp_bounds_updates[stem] = lp_obj_val
                    print(f"    [{done}/{len(missing)}] generated: {os.path.basename(out)}")
                except Exception as exc:
                    print(f"    [!] FAILED for {os.path.basename(futures[future])}: {exc}")
                    
        if lp_bounds_updates and LP_BOUNDS_PATH:
            import json as _json
            existing_lp_bounds = {}
            if os.path.exists(LP_BOUNDS_PATH):
                with open(LP_BOUNDS_PATH, 'r') as f:
                    try:
                        existing_lp_bounds = _json.load(f)
                    except _json.JSONDecodeError:
                        pass
            existing_lp_bounds.update(lp_bounds_updates)
            os.makedirs(os.path.dirname(LP_BOUNDS_PATH), exist_ok=True)
            with open(LP_BOUNDS_PATH, 'w') as f:
                _json.dump(existing_lp_bounds, f, indent=4)
    else:
        print(f"  [{label}] All feature .json files already present.")

    # Return json basenames that exist (and whose .lp is in the subdir)
    available = []
    for lp_name in lp_files:
        stem = os.path.splitext(lp_name)[0]
        json_path = os.path.join(FEATURE_DIR, stem + ".json")
        if os.path.exists(json_path):
            available.append(stem + ".json")
    return available


def get_train_val_files():
    """Returns lists of train and val JSON basenames."""
    print("--- Checking / generating feature .json files ---")
    train_files = ensure_feature_jsons(TRAIN_LP_DIR, "training")
    val_files   = ensure_feature_jsons(VAL_LP_DIR,   "val")

    if not train_files:
        raise FileNotFoundError(f"No usable training instances found under '{TRAIN_LP_DIR}'")
    if not val_files:
        print(f"Warning: No usable validation instances found under '{VAL_LP_DIR}'. "
              "Validation will be skipped.")

    return train_files, val_files


# Helper functions

def compute_batch_gap_closed(actual_files, bounds, labels_dict, lp_bounds_dict):
    """Calculates gap closed and relative closeness for a batch."""
    if not labels_dict or not lp_bounds_dict:
        return 0.0, 0, 0.0, [], []

    gap_sum = 0.0
    gap_count = 0
    rel_close_sum = 0.0
    gap_list = []
    rel_close_list = []
    for file_name, calculated_lagb in zip(actual_files, bounds):
        instance_name = file_name.replace(".json", "")
        
        opt_lagb = None
        lpb = None
        
        if instance_name in labels_dict:
            if isinstance(labels_dict[instance_name], dict):
                opt_lagb = labels_dict[instance_name].get("lagb")
                lpb = labels_dict[instance_name].get("lpb")
            else:
                opt_lagb = labels_dict[instance_name]
                if instance_name in lp_bounds_dict:
                    lpb = lp_bounds_dict[instance_name]
                    
        if opt_lagb is not None and lpb is not None:
            denominator = lpb - opt_lagb
            if abs(denominator) > 1e-9:
                gap = (lpb - calculated_lagb) / denominator * 100
            else:
                gap = 100.0
            gap_sum += gap
            gap_list.append(gap)
            gap_count += 1
            
            if abs(opt_lagb) > 1e-9:
                rel_close = abs(calculated_lagb - opt_lagb) / abs(opt_lagb) * 100
            else:
                rel_close = 0.0
            rel_close_sum += rel_close
            rel_close_list.append(rel_close)
            
    return gap_sum, gap_count, rel_close_sum, gap_list, rel_close_list


def process_batch(batch_files, model, executors, file_to_worker_map, graph_cache, cons_names_cache, device, is_training=True):
    """Handles forward pass and parallel solver execution for a batch."""
    if not batch_files:
        return None, [], []

    # Load graphs
    graph_list = []
    batch_cons_names = []
    for file_name in batch_files:
        json_path = os.path.join(FEATURE_DIR, file_name)
        if CACHE_GRAPH and file_name in graph_cache:
            graph_data = graph_cache[file_name]
            all_cons_names = cons_names_cache[file_name]
        else:
            graph_data = create_bipartite_graph(json_path)
            with open(json_path, 'r') as f:
                data_dict = json.load(f)
            all_cons_names = list(data_dict.get("constraints", {}).keys())

            if CACHE_GRAPH:
                graph_cache[file_name] = graph_data
                cons_names_cache[file_name] = all_cons_names

        graph_list.append(graph_data)
        batch_cons_names.append(all_cons_names)

    # Batched forward pass
    batch_data = Batch.from_data_list(graph_list).pin_memory().to(device, non_blocking=True)
    lambda_raw_batch, mask_batch, eq_flags_batch = model(batch_data)

    # Prepare solver args
    batch_sizes = []
    worker_args_list = []

    for i, file_name in enumerate(batch_files):
        stem = file_name.replace(".json", "")
        lp_path = os.path.join(TRAIN_LP_DIR, stem + ".lp")
        if not os.path.exists(lp_path):
            lp_path = os.path.join(VAL_LP_DIR, stem + ".lp")
        dec_path = os.path.join(DEC_DIR, stem + ".dec")

        # Access the dualized mask (index 2) directly from CPU graph data
        instance_mask = graph_list[i]['constraint'].x[:, 2].bool().tolist()
        all_cons_names = batch_cons_names[i]

        dualized_cons_names = [name for j, name in enumerate(all_cons_names) if instance_mask[j]]
        batch_sizes.append(len(dualized_cons_names))
        worker_args_list.append((file_name, lp_path, dec_path, dualized_cons_names))

    current_mip_gap = config['training_parameters']['scip_settings'].get('mip_gap', 0.0) if is_training else 0.0

    # Solver execution
    loss, bounds, solve_times = BatchedSubgradientLoss.apply(
        lambda_raw_batch, eq_flags_batch, batch_sizes,
        worker_args_list, executors, file_to_worker_map, current_mip_gap
    )

    return loss, batch_files, bounds, solve_times


# Main loop

def train():
    """Main training loop."""
    # Ensure plots directories exist
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    if BOXPLOT_PATH:
        os.makedirs(os.path.dirname(BOXPLOT_PATH), exist_ok=True)

    # --- Ensure all feature .json files exist; generate missing ones ---
    train_files, val_files = get_train_val_files()

    # Determine validation metric direction based on the first LP file
    # The validation metric is the opposite of the original problem sense (dual bound)
    IS_MINIMIZATION = True
    if train_files:
        first_lp = os.path.join(TRAIN_LP_DIR, train_files[0].replace('.json', '.lp'))
        if os.path.exists(first_lp):
            with open(first_lp, 'r') as f:
                for line in f:
                    line_lower = line.strip().lower()
                    if line_lower:
                        if line_lower.startswith('max'):
                            IS_MINIMIZATION = True
                            break
                        elif line_lower.startswith('min'):
                            IS_MINIMIZATION = False
                            break


    # Load labels for Gap Closed calculation
    labels_dict = {}
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, 'r', encoding='utf-8') as f:
            labels_dict = json.load(f)
    else:
        print(f"Warning: Labels file not found at {LABELS_PATH}. Gap Closed metric will not be computed.\n")
        
    lp_bounds_dict = {}
    if LP_BOUNDS_PATH and os.path.exists(LP_BOUNDS_PATH):
        with open(LP_BOUNDS_PATH, 'r', encoding='utf-8') as f:
            lp_bounds_dict = json.load(f)
    else:
        print(f"Warning: LP bounds file not found at {LP_BOUNDS_PATH}. Gap Closed metric will not be computed for instances missing it.\n")

    has_labels_and_bounds = bool(labels_dict and lp_bounds_dict)

    print(f"\n--- Dataset Loaded ---")
    print(f"Training instances:   {len(train_files)}")
    print(f"Validation instances: {len(val_files)}\n")

    # Cache capacity: how many models one worker might hold
    total_files = len(train_files) + len(val_files)
    cache_capacity_per_worker = (total_files + NUM_CORES - 1) // NUM_CORES
    if CACHE_LAG_RELAX:
        print(f"[*] Setting worker cache capacity to {cache_capacity_per_worker} models.")

    # Initialize caches locally in the main execution thread
    graph_cache = {}
    cons_names_cache = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LagrangianMultiplierModel(
        feature_embedding_size=FEATURE_EMBEDDING_SIZE,
        encoder_mlp_dims=ENCODER_MLP_DIMS,
        gnn_mlp_dims=GNN_MLP_DIMS,
        num_message_passing_layers=NUM_MESSAGE_PASSING_LAYERS,
        decoder_mlp_dims=DECODER_MLP_DIMS,
        dropout=DROPOUT
    ).to(device)

    print(f"Device used for GNN inference: {next(model.parameters()).device}\n")

    if LOAD_PRETRAINED:
        if os.path.exists(WEIGHTS_PATH):
            model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
            print(f"[*] Successfully loaded pretrained weights from '{WEIGHTS_PATH}'\n")

    optimizer = optim.RAdam(model.parameters(), lr=LEARNING_RATE)
    if USE_LR_SCHEDULER:
        scheduler_mode = 'min' if IS_MINIMIZATION else 'max'
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=scheduler_mode,
            factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE
        )

    print("--- Starting Training ---")
    train_losses = []
    val_losses = []
    train_gaps = []
    val_gaps = []
    train_rel_closes = []
    val_rel_closes = []
    val_epochs = []

    best_val_loss   = float('inf') if IS_MINIMIZATION else -float('inf')
    best_train_loss = float('inf')

    # ---------- Executor management ----------
    executors = []
    file_to_worker_map = {}
    solve_times_ema = {}
    current_cache_state = False

    def init_executors(enable_cache, capacity):
        nonlocal current_cache_state
        for exc in executors:
            exc.shutdown(wait=True)
        executors.clear()
        current_cache_state = enable_cache
        if enable_cache:
            for _ in range(NUM_CORES):
                executors.append(ProcessPoolExecutor(
                    max_workers=1, initializer=worker_init, initargs=(True, capacity)
                ))
        else:
            executors.append(ProcessPoolExecutor(
                max_workers=NUM_CORES, initializer=worker_init, initargs=(False, 0)
            ))

    def update_solve_times(files, times, is_warmup_phase):
        if not CACHE_LAG_RELAX:
            return
        if not (is_warmup_phase or TRACK_AFTER_WARMUP):
            return
        for f_name, t in zip(files, times):
            if f_name in solve_times_ema:
                solve_times_ema[f_name] = SMA_ALPHA * t + (1.0 - SMA_ALPHA) * solve_times_ema[f_name]
            else:
                solve_times_ema[f_name] = t

    def allocate_instances(files):
        file_to_worker_map.clear()
        if not solve_times_ema:
            shuffled = files.copy()
            random.shuffle(shuffled)
            for i, f in enumerate(shuffled):
                file_to_worker_map[f] = i % NUM_CORES
            return

        capacity = (len(files) + NUM_CORES - 1) // NUM_CORES
        worker_loads  = [0.0] * NUM_CORES
        worker_counts = [0]   * NUM_CORES

        sorted_files = sorted(
            files, key=lambda x: solve_times_ema.get(x, 0.0) + random.uniform(0, 1e-5), reverse=True
        )
        for f in sorted_files:
            valid_workers = [w for w in range(NUM_CORES) if worker_counts[w] < capacity]
            if not valid_workers:
                valid_workers = list(range(NUM_CORES))
            min_worker = min(valid_workers, key=lambda w: worker_loads[w])
            file_to_worker_map[f] = min_worker
            worker_loads[min_worker]  += solve_times_ema.get(f, 0.0)
            worker_counts[min_worker] += 1

    def get_batches(files):
        if current_cache_state:
            active_files_per_worker = {w: [] for w in range(NUM_CORES)}
            for f in files:
                active_files_per_worker[file_to_worker_map.get(f, 0)].append(f)
            for w in range(NUM_CORES):
                random.shuffle(active_files_per_worker[w])
            per_worker = max(1, BATCH_SIZE // NUM_CORES)
            iters = {w: iter(active_files_per_worker[w]) for w in range(NUM_CORES)}
            batches = []
            while True:
                current_batch = []
                for w in range(NUM_CORES):
                    for _ in range(per_worker):
                        try:
                            current_batch.append(next(iters[w]))
                        except StopIteration:
                            pass
                if not current_batch:
                    break
                batches.append(current_batch)
            return batches
        else:
            shuffled = files.copy()
            random.shuffle(shuffled)
            return [shuffled[i:i+BATCH_SIZE] for i in range(0, len(shuffled), BATCH_SIZE)]

    initial_cache_enabled = CACHE_LAG_RELAX and (WARMUP_EPOCHS <= 0)
    init_executors(initial_cache_enabled, cache_capacity_per_worker)
    if initial_cache_enabled:
        allocate_instances(train_files + val_files)

    try:
        # ------------------------------------------------------------------
        # Initial Validation Step
        # ------------------------------------------------------------------
        if EVALUATE_VALIDATION_LOSS and val_files:
            print("--- Initial Validation ---")
            model.eval()
            val_epoch_loss = 0.0
            val_epoch_gap_sum = 0.0
            val_epoch_gap_count = 0
            val_epoch_rel_close_sum = 0.0
            val_processed_count = 0
            all_val_gaps = []
            all_val_rel_closes = []
            with torch.no_grad():
                for batch_files in get_batches(val_files):
                    v_sum_loss, actual_files, bounds, solve_times = process_batch(
                        batch_files, model, executors, file_to_worker_map,
                        graph_cache, cons_names_cache, device, is_training=False
                    )
                    if v_sum_loss is not None:
                        update_solve_times(actual_files, solve_times, is_warmup_phase=True)
                        val_epoch_loss += sum(bounds)
                        gap_sum, gap_count, rel_close_sum, gap_list, rel_close_list = compute_batch_gap_closed(actual_files, bounds, labels_dict, lp_bounds_dict)
                        val_epoch_gap_sum   += gap_sum
                        val_epoch_gap_count += gap_count
                        val_epoch_rel_close_sum += rel_close_sum
                        all_val_gaps.extend(gap_list)
                        all_val_rel_closes.extend(rel_close_list)
                        val_processed_count += len(actual_files)
            if val_processed_count > 0:
                avg_val_loss = val_epoch_loss / val_processed_count
                is_better = (avg_val_loss < best_val_loss) if IS_MINIMIZATION else (avg_val_loss > best_val_loss)
                if is_better:
                    best_val_loss = avg_val_loss
                    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
                    torch.save(model.state_dict(), WEIGHTS_PATH)
                    print(f"[*] Initial best validation score: {best_val_loss:.4f}. Model saved to '{WEIGHTS_PATH}'")
                    
                    if BOXPLOT_PATH and all_val_gaps and all_val_rel_closes:
                        import matplotlib.pyplot as plt
                        import numpy as np
                        fig, (ax_gap, ax_rel) = plt.subplots(1, 2, figsize=(12, 6))
                        ax_gap.boxplot(all_val_gaps)
                        gap_mean, gap_median = np.mean(all_val_gaps), np.median(all_val_gaps)
                        ax_gap.set_title(f"Validation Gap Closed\nMean: {gap_mean:.2f}%, Median: {gap_median:.2f}%")
                        ax_gap.set_ylabel("Gap Closed (%)")
                        ax_rel.boxplot(all_val_rel_closes)
                        rel_mean, rel_median = np.mean(all_val_rel_closes), np.median(all_val_rel_closes)
                        ax_rel.set_title(f"Validation Relative Closeness\nMean: {rel_mean:.2f}%, Median: {rel_median:.2f}%")
                        ax_rel.set_ylabel("Relative Closeness (%)")
                        plt.tight_layout()
                        plt.savefig(BOXPLOT_PATH)
                        plt.close()

                if has_labels_and_bounds:
                    val_rel_close = val_epoch_rel_close_sum / val_epoch_gap_count if val_epoch_gap_count > 0 else 0.0
                    gap_str = f"{val_epoch_gap_sum / val_epoch_gap_count:.2f}%" if val_epoch_gap_count > 0 else "N/A"
                    print(f"[*] Initial Validation Score: {avg_val_loss:.4f} | Gap Closed: {gap_str} | Rel Closeness: {val_rel_close:.2f}%\n")
                else:
                    print(f"[*] Initial Validation Score: {avg_val_loss:.4f}\n")

        # Training loop
        for epoch in range(EPOCHS):
            is_warmup   = CACHE_LAG_RELAX and (epoch < WARMUP_EPOCHS)
            should_cache = CACHE_LAG_RELAX and not is_warmup

            if should_cache and not current_cache_state:
                print("[*] Warmup phase complete. Initializing caches and allocating instances sensibly...")
                init_executors(True, cache_capacity_per_worker)
                allocate_instances(train_files + val_files)
            elif (current_cache_state and RESHUFFLE_EVERY_N_EPOCHS > 0
                  and (epoch - WARMUP_EPOCHS) > 0
                  and (epoch - WARMUP_EPOCHS) % RESHUFFLE_EVERY_N_EPOCHS == 0):
                print("[*] Reshuffling instance allocation based on updated solve times...")
                allocate_instances(train_files + val_files)

            model.train()
            epoch_loss      = 0.0
            epoch_gap_sum   = 0.0
            epoch_gap_count = 0
            epoch_rel_close_sum = 0.0
            processed_count = 0

            current_train_files = train_files.copy()
            if USE_RANDOM_SUBSET:
                random.shuffle(current_train_files)
                num_subset = max(1, int(len(current_train_files) * RANDOM_SUBSET_RATIO)) if current_train_files else 0
                current_train_files = current_train_files[:num_subset]

            for batch_files in get_batches(current_train_files):
                optimizer.zero_grad()

                sum_loss, actual_files, bounds, solve_times = process_batch(
                    batch_files, model, executors, file_to_worker_map,
                    graph_cache, cons_names_cache, device, is_training=True
                )
                if sum_loss is None:
                    continue
                update_solve_times(actual_files, solve_times, is_warmup_phase=is_warmup)

                actual_bs = len(actual_files)
                avg_batch_loss = sum_loss / actual_bs
                avg_batch_loss.backward()
                if USE_GRADIENT_CLIPPING:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIPPING_MAX_NORM)
                optimizer.step()

                epoch_loss      += sum(bounds)
                gap_sum, gap_count, rel_close_sum, _, _ = compute_batch_gap_closed(actual_files, bounds, labels_dict, lp_bounds_dict)
                epoch_gap_sum   += gap_sum
                epoch_gap_count += gap_count
                epoch_rel_close_sum += rel_close_sum
                processed_count += actual_bs

            if processed_count > 0:
                avg_loss = epoch_loss / processed_count
                train_losses.append(avg_loss)

                if not EVALUATE_VALIDATION_LOSS and avg_loss < best_train_loss:
                    best_train_loss = avg_loss
                    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
                    torch.save(model.state_dict(), WEIGHTS_PATH)

                train_gap = epoch_gap_sum / epoch_gap_count if epoch_gap_count > 0 else 0.0
                train_gaps.append(train_gap)
                train_rel_close = epoch_rel_close_sum / epoch_gap_count if epoch_gap_count > 0 else 0.0
                train_rel_closes.append(train_rel_close)

                if has_labels_and_bounds:
                    gap_str = f"{train_gap:.2f}%" if epoch_gap_count > 0 else "N/A"
                    print(f"Epoch {epoch+1:02d}/{EPOCHS} | LR: {optimizer.param_groups[0]['lr']:.1e} | Avg Lagrangian Bound (Train): {avg_loss:.4f} | Gap Closed: {gap_str} | Rel Closeness: {train_rel_close:.2f}%")
                else:
                    print(f"Epoch {epoch+1:02d}/{EPOCHS} | LR: {optimizer.param_groups[0]['lr']:.1e} | Avg Lagrangian Bound (Train): {avg_loss:.4f}")

                if USE_LR_SCHEDULER:
                    scheduler.step(avg_loss)

            # ------ Validation ------
            if val_files and EVALUATE_VALIDATION_LOSS and (epoch + 1) % EVALUATE_VALIDATION_EVERY_N_EPOCHS == 0:
                model.eval()
                val_epoch_loss      = 0.0
                val_epoch_gap_sum   = 0.0
                val_epoch_gap_count = 0
                val_epoch_rel_close_sum = 0.0
                val_processed_count = 0
                all_val_gaps = []
                all_val_rel_closes = []
                with torch.no_grad():
                    for batch_files in get_batches(val_files):
                        v_sum_loss, actual_files, bounds, solve_times = process_batch(
                            batch_files, model, executors, file_to_worker_map,
                            graph_cache, cons_names_cache, device, is_training=False
                        )
                        if v_sum_loss is not None:
                            update_solve_times(actual_files, solve_times, is_warmup_phase=is_warmup)
                            val_epoch_loss      += sum(bounds)
                            gap_sum, gap_count, rel_close_sum, gap_list, rel_close_list = compute_batch_gap_closed(actual_files, bounds, labels_dict, lp_bounds_dict)
                            val_epoch_gap_sum   += gap_sum
                            val_epoch_gap_count += gap_count
                            val_epoch_rel_close_sum += rel_close_sum
                            all_val_gaps.extend(gap_list)
                            all_val_rel_closes.extend(rel_close_list)
                            val_processed_count += len(actual_files)

                if val_processed_count > 0:
                    avg_val_loss = val_epoch_loss / val_processed_count
                    val_losses.append(avg_val_loss)

                    is_better = (avg_val_loss < best_val_loss) if IS_MINIMIZATION else (avg_val_loss > best_val_loss)
                    if is_better:
                        best_val_loss = avg_val_loss
                        os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
                        torch.save(model.state_dict(), WEIGHTS_PATH)
                        print(f"[*] New best validation score: {best_val_loss:.4f}. Model saved to '{WEIGHTS_PATH}'")
                        
                        if BOXPLOT_PATH and all_val_gaps and all_val_rel_closes:
                            import matplotlib.pyplot as plt
                            import numpy as np
                            fig, (ax_gap, ax_rel) = plt.subplots(1, 2, figsize=(12, 6))
                            ax_gap.boxplot(all_val_gaps)
                            gap_mean, gap_median = np.mean(all_val_gaps), np.median(all_val_gaps)
                            ax_gap.set_title(f"Validation Gap Closed\nMean: {gap_mean:.2f}%, Median: {gap_median:.2f}%")
                            ax_gap.set_ylabel("Gap Closed (%)")
                            ax_rel.boxplot(all_val_rel_closes)
                            rel_mean, rel_median = np.mean(all_val_rel_closes), np.median(all_val_rel_closes)
                            ax_rel.set_title(f"Validation Relative Closeness\nMean: {rel_mean:.2f}%, Median: {rel_median:.2f}%")
                            ax_rel.set_ylabel("Relative Closeness (%)")
                            plt.tight_layout()
                            plt.savefig(BOXPLOT_PATH)
                            plt.close()

                    val_gap = val_epoch_gap_sum / val_epoch_gap_count if val_epoch_gap_count > 0 else 0.0
                    val_gaps.append(val_gap)
                    val_rel_close = val_epoch_rel_close_sum / val_epoch_gap_count if val_epoch_gap_count > 0 else 0.0
                    val_rel_closes.append(val_rel_close)
                    val_epochs.append(epoch + 1)

                    if has_labels_and_bounds:
                        gap_str = f"{val_gap:.2f}%" if val_epoch_gap_count > 0 else "N/A"
                        print(f"Epoch {epoch+1:02d}/{EPOCHS} | LR: {optimizer.param_groups[0]['lr']:.1e} | Avg Lagrangian Bound (Val):   {avg_val_loss:.4f} | Gap Closed: {gap_str} | Rel Closeness: {val_rel_close:.2f}%")
                    else:
                        print(f"Epoch {epoch+1:02d}/{EPOCHS} | LR: {optimizer.param_groups[0]['lr']:.1e} | Avg Lagrangian Bound (Val):   {avg_val_loss:.4f}")

            # ------ Plotting ------
            if (epoch + 1) % PLOT_EVERY_N_EPOCHS == 0:
                import matplotlib.pyplot as plt
                if has_labels_and_bounds:
                    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
                else:
                    fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
                
                # Plot 1: Losses
                ax1.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o', color='blue')
                if EVALUATE_VALIDATION_LOSS and len(val_losses) > 0:
                    ax1.plot(val_epochs, val_losses, label='Validation Loss', marker='s', color='orange')
                    ax1.set_title(f'Training and Validation Loss (Epoch {epoch + 1})')
                else:
                    ax1.set_title(f'Training Loss (Epoch {epoch + 1})')
                ax1.set_ylabel('Avg Lagrangian Bound')
                if has_labels_and_bounds:
                    ax1.legend()
                else:
                    ax1.set_xlabel('Epoch')
                    ax1.legend()
                ax1.grid(True, linestyle='--', alpha=0.7)
                
                if has_labels_and_bounds:
                    # Plot 2: Gap Closed
                    filtered_train_gaps = [g if g >= 0 else float('nan') for g in train_gaps]
                    ax2.plot(range(1, len(filtered_train_gaps) + 1), filtered_train_gaps, label='Train Gap Closed', marker='o', color='green')
                    if EVALUATE_VALIDATION_LOSS and len(val_gaps) > 0:
                        filtered_val_gaps = [g if g >= 0 else float('nan') for g in val_gaps]
                        ax2.plot(val_epochs, filtered_val_gaps, label='Validation Gap Closed', marker='s', color='red')
                    ax2.set_title('Gap Closed Percentage')
                    ax2.set_ylabel('Gap Closed (%)')
                    ax2.set_ylim(bottom=PLOT_MIN_GAP_CLOSED)
                    ax2.legend()
                    ax2.grid(True, linestyle='--', alpha=0.7)
                    
                    # Plot 3: Relative Closeness
                    filtered_train_rel = [g if g >= 0 else float('nan') for g in train_rel_closes]
                    ax3.plot(range(1, len(filtered_train_rel) + 1), filtered_train_rel, label='Train Rel Closeness', marker='o', color='purple')
                    if EVALUATE_VALIDATION_LOSS and len(val_rel_closes) > 0:
                        filtered_val_rel = [g if g >= 0 else float('nan') for g in val_rel_closes]
                        ax3.plot(val_epochs, filtered_val_rel, label='Validation Rel Closeness', marker='s', color='brown')
                    ax3.set_title('Relative Closeness to Optimal')
                    ax3.set_xlabel('Epoch')
                    ax3.set_ylabel('Relative Closeness (%)')
                    ax3.legend()
                    ax3.grid(True, linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                plt.savefig(PLOT_PATH)
                plt.close()

        # Final Validation
        last_epoch_evaluated = val_files and EVALUATE_VALIDATION_LOSS and (EPOCHS % EVALUATE_VALIDATION_EVERY_N_EPOCHS == 0)
        if not last_epoch_evaluated and val_files:
            print("\n--- Starting Final Validation ---")
            model.eval()
            val_loss      = 0.0
            val_gap_sum   = 0.0
            val_gap_count = 0
            val_processed = 0

            with torch.no_grad():
                for batch_files in get_batches(val_files):
                    v_sum_loss, actual_files, bounds, solve_times = process_batch(
                        batch_files, model, executors, file_to_worker_map,
                        graph_cache, cons_names_cache, device, is_training=False
                    )
                    if v_sum_loss is not None:
                        val_loss    += sum(bounds)
                        gap_sum, gap_count, _, _, _ = compute_batch_gap_closed(actual_files, bounds, labels_dict, lp_bounds_dict)
                        val_gap_sum   += gap_sum
                        val_gap_count += gap_count
                        val_processed += len(actual_files)

            if val_processed > 0:
                avg_val_loss = val_loss / val_processed
                if has_labels_and_bounds:
                    gap_str = f"{val_gap_sum / val_gap_count:.2f}%" if val_gap_count > 0 else "N/A"
                    print(f"Final Validation | Avg Lagrangian Bound: {avg_val_loss:.4f} | Gap Closed: {gap_str}")
                else:
                    print(f"Final Validation | Avg Lagrangian Bound: {avg_val_loss:.4f}")

                is_better = (avg_val_loss < best_val_loss) if IS_MINIMIZATION else (avg_val_loss > best_val_loss)
                if EVALUATE_VALIDATION_LOSS and is_better:
                    best_val_loss = avg_val_loss
                    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
                    torch.save(model.state_dict(), WEIGHTS_PATH)
                    print(f"[*] New best validation score: {best_val_loss:.4f}. Model saved to '{WEIGHTS_PATH}'")

        print("\nPipeline Complete.")
        if os.path.exists(WEIGHTS_PATH):
            model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
            print(f"Loaded best model weights from '{WEIGHTS_PATH}' for final evaluation.")
        else:
            os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
            torch.save(model.state_dict(), WEIGHTS_PATH)
            print(f"Model saved to '{WEIGHTS_PATH}'")

        # Export Results to JSON
        if val_files:
            print("\n--- Generating Results JSON ---")
            results_dict = {}

            model.eval()
            with torch.no_grad():
                for batch_files in get_batches(val_files):
                    loss, actual_files, bounds, solve_times = process_batch(
                        batch_files, model, executors, file_to_worker_map,
                        graph_cache, cons_names_cache, device, is_training=False
                    )
                    if loss is not None:
                        for file_name, bound in zip(actual_files, bounds):
                            base_name = file_name.replace(".json", "")
                            results_dict[base_name] = bound

            os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
            with open(RESULTS_PATH, "w") as f:
                json.dump(results_dict, f, indent=4)
            print(f"Results successfully saved to '{RESULTS_PATH}'")

        # Deactivated import
        # print("\n--- Running Final Evaluation ---")
        # evaluate_main()

    finally:
        for exc in executors:
            exc.shutdown(wait=False)


if __name__ == "__main__":
    # Using 'spawn' prevents CUDA context errors when mixing PyTorch with multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    train()
