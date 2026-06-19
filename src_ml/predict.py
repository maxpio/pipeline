"""
Full end-to-end pipeline for batch instance processing:
  1. Scan lp_dir for .lp files, sample by instance_ratio, union with required_instances
  2. For each instance: extract LP relaxation features (PySCIPOpt)
  3. Build the bipartite graph and predict dual values with the trained GNN model
  4. Save the predicted duals to a file
  5. Run GCG with the custom pricing plugin that uses the predicted duals

Usage:
    python -m src_ml.full_pipeline [--lp-dir <dir>] [--dec-dir <dir>] [--instance-ratio 0.1] [--required-instances e10100-1.lp]
"""

import os
import sys
import json
import time
import yaml
import argparse
import torch
from pathlib import Path
from pyscipopt import Model as SCIPModel
from torch_geometric.data import Batch

# ---------------------------------------------------------------------------
# Resolve project paths so imports work regardless of cwd
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(BASE_DIR)

from src_ml.input_graph import create_bipartite_graph
from src_ml.model import LagrangianMultiplierModel
from src_ml.single_runner import run_single_instance

# ---------------------------------------------------------------------------
# Load config.yaml (model architecture, defaults, etc.)
# ---------------------------------------------------------------------------
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# Model Architecture from config
FEATURE_EMBEDDING_SIZE = config['model_architecture']['feature_embedding_size']
ENCODER_MLP_DIMS = config['model_architecture']['encoder_mlp_dims']
GNN_MLP_DIMS = config['model_architecture']['gnn_mlp_dims']
DECODER_MLP_DIMS = config['model_architecture']['decoder_mlp_dims']
NUM_LAYERS = config['model_architecture']['num_layers']
DROPOUT = config['model_architecture'].get('dropout', 0.0)

data_dir = Path(config['general_settings']['data_dir']).resolve()
DEFAULT_WEIGHTS_PATH = data_dir / "weights" / config['general_settings'].get('weights_filename', 'gaptest.pth')
DEFAULT_GCG_EXECUTABLE = Path(config['general_settings'].get('gcg_executable', "/home/max/gcg/build/gcg-linux-release/bin/gcg")).resolve()


# ========================== STEP 1: Feature Extraction ==========================
from src_ml.feature_extractor import extract_features_single

# ========================== STEP 2: ML Prediction ==========================

def predict_duals(feature_dict: dict, weights_path: str, quiet: bool = False) -> tuple[dict, str]:
    """
    Builds the bipartite graph from the feature dictionary, loads the trained
    GNN model, and predicts Lagrangian dual multipliers.

    Returns:
        (mult_dict, dual_txt)  — a dict {constraint_name: value} and a
        pre-formatted string ready to be written to a .txt file.
    """
    if not quiet:
        print(f"[Step 2] Predicting dual values with ML model ({weights_path})")
    start = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Write feature dict to a temp JSON so create_bipartite_graph can read it ---
    import tempfile
    fd, tmp_json = tempfile.mkstemp(suffix=".json", text=True)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(feature_dict, f)

        # Build graph
        graph_data = create_bipartite_graph(tmp_json).to(device)
    finally:
        if os.path.exists(tmp_json):
            os.remove(tmp_json)

    # --- Load model ---
    model = LagrangianMultiplierModel(
        feature_embedding_size=FEATURE_EMBEDDING_SIZE,
        encoder_mlp_dims=ENCODER_MLP_DIMS,
        gnn_mlp_dims=GNN_MLP_DIMS,
        num_layers=NUM_LAYERS,
        decoder_mlp_dims=DECODER_MLP_DIMS,
        dropout=DROPOUT
    ).to(device)

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # --- Predict ---
    all_cons_names = list(feature_dict.get("constraints", {}).keys())

    with torch.no_grad():
        batch_data = Batch.from_data_list([graph_data]).to(device)
        lambda_raw, dualized_mask, eq_flags = model(batch_data)

        # Enforce non-negativity for inequality constraints
        actual_multipliers = torch.where(eq_flags == 1.0, lambda_raw, torch.relu(lambda_raw))

    actual_multipliers = actual_multipliers.cpu().tolist()
    mask_list = dualized_mask.cpu().tolist()

    dualized_cons_names = [name for j, name in enumerate(all_cons_names) if mask_list[j]]
    mult_dict = dict(zip(dualized_cons_names, actual_multipliers))

    # Build the text representation matching the format GCG's plugin expects
    lines = []
    for k, v in mult_dict.items():
        lines.append(f'"{k}": {v},')
    dual_txt = "\n".join(lines) + "\n"

    elapsed = time.time() - start
    if not quiet:
        print(f"  Predicted {len(mult_dict)} dual multipliers in {elapsed:.2f}s")

    return mult_dict, dual_txt


# ========================== STEP 3: Run GCG ==========================

def run_gcg_with_duals(
    lp_file: Path,
    dec_file: Path,
    dual_file: Path,
    gcg_executable: Path,
    experiment_params: dict,
    timeout: int = 3600,
    log_file: Path | None = None,
    save_logs: bool = True,
    print_solver_output: bool = True,
    quiet: bool = False
) -> tuple[str, dict]:
    """
    Constructs the GCG command string with the custom pricing plugin settings
    and runs the solver.  Returns (status, metrics_dict).
    """
    if not quiet:
        print(f"[Step 3] Running GCG with predicted duals")
        print(f"  LP:   {lp_file}")
        print(f"  DEC:  {dec_file}")
        print(f"  DUAL: {dual_file}")

    cmd_string = (
        ("set heuristics emphasis off\n" if experiment_params.get('disable_heuristics', True) else "") +
        ("set presolving maxrounds 0\n" if experiment_params.get('disable_presolving', True) else "") +
        ("set separating clique freq -1\n" if experiment_params.get('disable_separating_cliques', True) else "") +
        ("set limits nodes 1\n" if experiment_params.get('solve_root_only', False) else "") +
        f"set pricingcb initduals duals_file {dual_file}\n"
        f"set pricingcb initduals use_custom_duals {experiment_params.get('use_custom_duals', 'TRUE')}\n"
        f"set pricingcb initduals use_bigm_artificials {experiment_params.get('use_bigm_artificials', 'FALSE')}\n"
        f"set pricingcb initduals n_perturbation_rounds {experiment_params.get('n_perturbation_rounds', 0)}\n"
        f"set pricingcb initduals perturbation_percent {experiment_params.get('perturbation_percent', 0.05)}\n"
        f"set pricingcb initduals add_round_0_columns {experiment_params.get('add_round_0_columns', 'TRUE')}\n"
        f"set pricingcb initduals bigm_value {experiment_params.get('bigm_value', 1000000.0)}\n"
        f"set pricingcb initduals large_log {experiment_params.get('large_log', 'FALSE')}\n"
        f"set pricingcb initduals use_smoothing {experiment_params.get('use_smoothing', 'FALSE')}\n"
        f"set pricingcb initduals smoothing_weight_start {experiment_params.get('smoothing_weight_start', 0.99)}\n"
        f"set pricingcb initduals smoothing_weight_factor {experiment_params.get('smoothing_weight_factor', 0.4)}\n"
        f"set pricingcb initduals smoothing_weight_min {experiment_params.get('smoothing_weight_min', 0.01)}\n"
        f"set pricingcb initduals smoothing_improvement_threshold {experiment_params.get('smoothing_improvement_threshold', 200.0)}\n"
        f"set pricingcb initduals disable_subprob_presolve_heur {experiment_params.get('disable_subprob_presolve_heur', 'FALSE')}\n"
        f"set pricing masterpricer stabilization {experiment_params.get('masterpricer_stabilization', 'FALSE')}\n"
        f"read {lp_file}\n"
        f"read {dec_file}\n"
        "optimize\nquit\n"
    )

    return run_single_instance(
        lp_file=lp_file,
        dec_file=dec_file,
        dual_file=dual_file,
        gcg_executable=gcg_executable,
        cmd_string=cmd_string,
        timeout=timeout,
        log_file=log_file,
        save_logs=save_logs,
        print_solver_output=print_solver_output
    )


# ========================== Parallel Workers ==========================

def _extract_wrapper(args_tuple):
    """Worker for ProcessPoolExecutor — extracts features for one instance."""
    lp_path, dec_path = args_tuple
    maxrounds = config['feature_extraction'].get('presolving_maxrounds', 0)
    return extract_features_single(str(lp_path), str(dec_path), maxrounds=maxrounds, quiet=True, extract_lp_bound=False)


def _gcg_wrapper(args_tuple):
    """Worker for ProcessPoolExecutor — runs GCG for one instance."""
    lp_f, dec_f, dual_f, gcg_exe, exp_params, tout, log_f, save_l = args_tuple
    return run_gcg_with_duals(
        lp_file=lp_f, dec_file=dec_f, dual_file=dual_f,
        gcg_executable=gcg_exe, experiment_params=exp_params,
        timeout=tout, log_file=log_f, save_logs=save_l,
        print_solver_output=False, quiet=True
    )


# ========================== Main ==========================

# (resolve_instance_list has been removed as it was unused and deprecated)


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: extract features → predict duals → run GCG (batch mode)"
    )
    parser.add_argument("--data-dir", type=Path,
                        default=data_dir,
                        help="Absolute path to the data directory")
    parser.add_argument("--lpfile-subdir", type=str,
                        default=config['prediction_parameters'].get('lpfile_subdir', 'test'),
                        help="Subdirectory of lpfiles to evaluate (e.g. val, test)")
    parser.add_argument("--dualvalue-type", type=str,
                        default=config['prediction_parameters'].get('dualvalue_type', 'predicted'),
                        help="Type of dual values to use (e.g., predicted, optimal, random)")
    parser.add_argument("--skip-prediction", action="store_true",
                        default=config['prediction_parameters'].get('skip_prediction', False),
                        help="Skip prediction step and use saved duals (only applies if dualvalue-type is 'predicted')")
    parser.add_argument("--weights", type=Path, default=Path(DEFAULT_WEIGHTS_PATH),
                        help="Path to trained model weights (.pth)")
    parser.add_argument("--gcg", type=Path, default=Path(DEFAULT_GCG_EXECUTABLE),
                        help="Path to the GCG executable")
    parser.add_argument("--timeout", type=int, default=config['prediction_parameters'].get('timeout', 3600),
                        help=f"Max solving time in seconds (default: {config['prediction_parameters'].get('timeout', 3600)})")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for logs and experiment outputs")
    parser.add_argument("--no-logs", action="store_true",
                        help="Do not save GCG log output")
    parser.add_argument("--quiet", action="store_true",
                        help="Do not print GCG solver output to console")
    parser.add_argument("--seed", type=int,
                        default=config['general_settings'].get('random_seed', 42),
                        help="Random seed for instance sampling")
    parser.add_argument("--num-cores", type=int,
                        default=config['general_settings'].get('num_cores', 1),
                        help="CPU cores for parallel feature extraction and GCG solving")
    parser.add_argument("--gpu-batch-size", type=int,
                        default=config['general_settings'].get('gpu_batch_size', 16),
                        help="Number of instances to predict duals for in one GPU batch")
    args = parser.parse_args()
    
    data_dir_path = args.data_dir
    lp_dir = data_dir_path / "lpfiles"
    if args.lpfile_subdir:
        lp_dir = lp_dir / args.lpfile_subdir
        
    dec_dir = data_dir_path / "decfiles"
    
    dual_dir = data_dir_path / "dualvalues" / args.dualvalue_type
    dual_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = (args.output_dir or data_dir_path) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    exp_dir = (args.output_dir or data_dir_path) / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    if not lp_dir.is_dir():
        sys.exit(f"Error: LP directory not found: {lp_dir}")
    if not dec_dir.is_dir():
        sys.exit(f"Error: DEC directory not found: {dec_dir}")
    if not args.weights.exists():
        sys.exit(f"Error: Model weights not found: {args.weights}")
    if not args.gcg.exists():
        sys.exit(f"Error: GCG executable not found: {args.gcg}")

    # Resolve which instances to run
    instance_files = sorted(list(lp_dir.glob("*.lp")))
    if not instance_files:
        sys.exit(f"Error: No .lp files found in {lp_dir}")

    # Build (lp_file, dec_file) pairs, filtering out missing .dec files
    instance_pairs = []
    for lp_file in instance_files:
        dec_file = dec_dir / f"{lp_file.stem}.dec"
        if not dec_file.exists():
            print(f"  [!] Skipping {lp_file.stem}: no matching .dec file at {dec_file}")
            continue
        instance_pairs.append((lp_file, dec_file))

    n_instances = len(instance_pairs)
    is_batch = n_instances > 1
    num_cores = min(args.num_cores, n_instances) if is_batch else 1

    print("=" * 70)
    print(f"  Full Pipeline — Batch Mode")
    print(f"  LP dir:    {lp_dir}")
    print(f"  DEC dir:   {dec_dir}")
    print(f"  Instances: {n_instances} selected from {args.lpfile_subdir} set")
    if is_batch:
        print(f"  Cores:     {num_cores}  |  GPU batch size: {args.gpu_batch_size}")
    print("=" * 70)

    experiment_params = config.get('prediction_parameters', {}).get('gcg_settings', {})
    pipeline_start = time.time()

    # ====================================================================
    #  SINGLE-INSTANCE PATH  (no parallelization overhead)
    # ====================================================================
    if not is_batch:
        lp_file, dec_file = instance_pairs[0]
        instance_name = lp_file.stem

        print(f"\n{'=' * 70}")
        print(f"  Instance: {instance_name}")
        print("=" * 70)

        use_custom_duals = str(experiment_params.get('use_custom_duals', 'TRUE')).upper() == 'TRUE'
        dual_file = dual_dir / f"{instance_name}.txt"

        if use_custom_duals:
            if args.dualvalue_type != "predicted" or args.skip_prediction:
                if dual_file.exists():
                    if not args.quiet:
                        if args.dualvalue_type != "predicted":
                            print(f"  Using precalculated {args.dualvalue_type} duals from: {dual_file}")
                        else:
                            print(f"  Skipped prediction, using saved predicted duals from: {dual_file}")
                else:
                    print(f"  [!] Saved duals not found at {dual_file}. Creating empty file.")
                    dual_file.touch()
            else:
                feature_dict = extract_features_single(
                    str(lp_file), 
                    str(dec_file), 
                    maxrounds=config['feature_extraction'].get('presolving_maxrounds', 0),
                    quiet=args.quiet,
                    extract_lp_bound=False
                )
            mult_dict, dual_txt = predict_duals(feature_dict, str(args.weights), quiet=args.quiet)
            with open(dual_file, 'w') as f:
                f.write(dual_txt)
            if not args.quiet:
                print(f"  Predicted duals saved to: {dual_file}")
        else:
            dual_file.touch()
            if not args.quiet:
                print("  Skipped feature extraction and prediction (use_custom_duals is False)")

        log_file = log_dir / f"{instance_name}_gcg.log"
        status, metrics = run_gcg_with_duals(
            lp_file=lp_file, dec_file=dec_file, dual_file=dual_file,
            gcg_executable=args.gcg, experiment_params=experiment_params,
            timeout=args.timeout, log_file=log_file,
            save_logs=not args.no_logs, print_solver_output=not args.quiet,
            quiet=args.quiet
        )

        pipeline_elapsed = time.time() - pipeline_start
        print(f"\n{'=' * 70}")
        print("  Pipeline Results")
        print("=" * 70)
        print(f"  Instance:        {instance_name}")
        print(f"  Status:          {status}")
        print(f"  Execution Time:  {metrics['solving_time']:.2f}s")
        print(f"  Objective Value: {metrics['final_obj_val']}")
        print(f"\n  Total wall-clock time: {pipeline_elapsed:.2f}s")
        print("=" * 70)

        all_results = [{
            "instance": instance_name,
            "status": status,
            **metrics
        }]
        
    else:
        # ====================================================================
        #  BATCH PATH  (parallel feature extraction, batched GPU, parallel GCG)
        # ====================================================================
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import tempfile as _tempfile

        use_custom_duals = str(experiment_params.get('use_custom_duals', 'TRUE')).upper() == 'TRUE'

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model once
        is_prediction_needed = use_custom_duals and args.dualvalue_type == "predicted" and not args.skip_prediction
        if is_prediction_needed:
            model = LagrangianMultiplierModel(
                feature_embedding_size=FEATURE_EMBEDDING_SIZE,
                encoder_mlp_dims=ENCODER_MLP_DIMS,
                gnn_mlp_dims=GNN_MLP_DIMS,
                num_layers=NUM_LAYERS,
                decoder_mlp_dims=DECODER_MLP_DIMS,
                dropout=DROPOUT
            ).to(device)
            model.load_state_dict(torch.load(str(args.weights), map_location=device))
            model.eval()

        all_results = []
        batch_size = args.gpu_batch_size

        if device.type == 'cpu' or not is_prediction_needed:
            batch_size = len(instance_pairs)

        for batch_start in range(0, len(instance_pairs), batch_size):
            batch_end = min(batch_start + batch_size, len(instance_pairs))
            chunk_pairs = instance_pairs[batch_start:batch_end]
            chunk_names = [lp.stem for lp, _ in chunk_pairs]
            n_chunk = len(chunk_pairs)

            print(f"\n{'=' * 50}")
            print(f"  Processing Batch {(batch_start//batch_size) + 1} "
                  f"(Instances {batch_start+1}–{batch_end} / {len(instance_pairs)})")
            print(f"{'=' * 50}")

            dual_files: dict[str, Path] = {}

            if use_custom_duals and not is_prediction_needed:
                print(f"  Phase 1 & 2: Skipped (using precalculated {args.dualvalue_type} duals or skip_prediction is True) ...")
                for lp_file, _ in chunk_pairs:
                    name = lp_file.stem
                    d_file = dual_dir / f"{name}.txt"
                    if not d_file.exists():
                        print(f"    [!] Saved duals not found at {d_file}")
                        d_file.touch()
                    dual_files[name] = d_file
            elif use_custom_duals:
                # ---- Phase 1: Parallel feature extraction ----
                print(f"  Phase 1: Extracting features ({num_cores} cores) ...")
                phase1_start = time.time()
                feature_dicts: dict[str, dict] = {}
                with ProcessPoolExecutor(max_workers=num_cores) as pool:
                    future_to_name = {
                        pool.submit(_extract_wrapper, pair): pair[0].stem
                        for pair in chunk_pairs
                    }
                    done_count = 0
                    for future in as_completed(future_to_name):
                        name = future_to_name[future]
                        done_count += 1
                        try:
                            feature_dicts[name] = future.result()
                            # print(f"    [{done_count}/{n_chunk}] extracted: {name}")
                        except Exception as exc:
                            print(f"    [!] FAILED extraction: {name} — {exc}")

                phase1_elapsed = time.time() - phase1_start
                print(f"    Done in {phase1_elapsed:.2f}s  ({len(feature_dicts)}/{n_chunk} succeeded)")

                # ---- Phase 2: Batched GPU prediction ----
                print(f"  Phase 2: Predicting duals (GPU) ...")
                phase2_start = time.time()
                
                ordered_names = [n for n in chunk_names if n in feature_dicts]
                graphs = []
                cons_names_per_instance = []
                for name in ordered_names:
                    fd_dict = feature_dicts[name]
                    fd, tmp_json = _tempfile.mkstemp(suffix=".json", text=True)
                    try:
                        with os.fdopen(fd, 'w') as f:
                            json.dump(fd_dict, f)
                        graph_data = create_bipartite_graph(tmp_json).to(device)
                    finally:
                        if os.path.exists(tmp_json):
                            os.remove(tmp_json)
                    graphs.append(graph_data)
                    cons_names_per_instance.append(list(fd_dict.get("constraints", {}).keys()))

                if graphs:
                    with torch.no_grad():
                        batch_data = Batch.from_data_list(graphs).to(device)
                        lambda_raw, dualized_mask, eq_flags = model(batch_data)
                        actual_multipliers = torch.where(eq_flags == 1.0, lambda_raw, torch.relu(lambda_raw))

                    # Un-batch predictions
                    actual_multipliers_cpu = actual_multipliers.cpu()
                    mask_cpu = dualized_mask.cpu()
                    
                    if hasattr(batch_data['constraint'], 'batch') and batch_data['constraint'].batch is not None:
                        cons_batch_indices = batch_data['constraint'].batch.cpu()
                    else:
                        cons_batch_indices = torch.zeros(len(mask_cpu), dtype=torch.long)
                        
                    dualized_batch_indices = cons_batch_indices[mask_cpu]

                    for i, name in enumerate(ordered_names):
                        idx_mask = (cons_batch_indices == i)
                        idx_mask_dual = (dualized_batch_indices == i)
                        
                        inst_mults = actual_multipliers_cpu[idx_mask_dual].tolist()
                        inst_mask = mask_cpu[idx_mask].tolist()
                        all_cons = cons_names_per_instance[i]

                        dualized_names = [cname for j, cname in enumerate(all_cons) if inst_mask[j]]
                        mult_dict = dict(zip(dualized_names, inst_mults))

                        lines = [f'"{k}": {v},' for k, v in mult_dict.items()]
                        dual_txt = "\n".join(lines) + "\n"

                        dual_file = dual_dir / f"{name}.txt"
                        with open(dual_file, 'w') as f:
                            f.write(dual_txt)
                        dual_files[name] = dual_file

                phase2_elapsed = time.time() - phase2_start
                print(f"    Done in {phase2_elapsed:.2f}s  ({len(dual_files)} predicted)")
            else:
                print("  Phase 1 & 2: Skipped (use_custom_duals is False)")
                for lp_file, _ in chunk_pairs:
                    name = lp_file.stem
                    empty_dual = dual_dir / f"{name}.txt"
                    empty_dual.touch()
                    dual_files[name] = empty_dual

            # ---- Phase 3: Parallel GCG solving ----
            print(f"  Phase 3: Running GCG ({num_cores} cores) ...")
            phase3_start = time.time()

            gcg_tasks = []
            for lp_file, dec_file in chunk_pairs:
                name = lp_file.stem
                if name not in dual_files:
                    continue
                log_file = log_dir / f"{name}_gcg.log"
                gcg_tasks.append((
                    lp_file, dec_file, dual_files[name], args.gcg,
                    experiment_params, args.timeout, log_file, not args.no_logs
                ))

            with ProcessPoolExecutor(max_workers=num_cores) as pool:
                future_to_name = {
                    pool.submit(_gcg_wrapper, task): task[0].stem
                    for task in gcg_tasks
                }
                done_count = 0
                for future in as_completed(future_to_name):
                    name = future_to_name[future]
                    done_count += 1
                    try:
                        status, metrics = future.result()
                        all_results.append({
                            "instance": name, "status": status,
                            **metrics
                        })
                        print(f"    [{done_count}/{len(gcg_tasks)}] {name:30s}  "
                              f"status={status:10s}  time={metrics['solving_time']:8.2f}s  obj={metrics['final_obj_val']}")
                    except Exception as exc:
                        all_results.append({
                            "instance": name, "status": "CRASH",
                            "solving_time": 0.0, "final_obj_val": float('inf'),
                            "cols_needed_for_rmp_feasibility": 0,
                            "slp_iterations_main_loop": 0,
                            "slp_iterations_custom_pricing": 0,
                        })
                        print(f"    [{done_count}/{len(gcg_tasks)}] {name:30s}  CRASHED: {exc}")

            phase3_elapsed = time.time() - phase3_start
            print(f"    Done in {phase3_elapsed:.2f}s")


        pipeline_elapsed = time.time() - pipeline_start

        # ---- Final Summary ----
        all_results.sort(key=lambda r: r['instance'])

        print(f"\n{'=' * 70}")
        print("  Batch Pipeline Results")
        print("=" * 70)
        for r in all_results:
            print(f"  {r['instance']:30s}  status={r['status']:10s}  "
                  f"time={r.get('solving_time', 0.0):8.2f}s  obj={r.get('final_obj_val', 'inf')}")
        print(f"\n  Total instances processed: {len(all_results)}")
        print(f"  Total wall-clock time:         {pipeline_elapsed:8.2f}s")
        print("=" * 70)

    exp_json_name = config['prediction_parameters'].get('experiment_json_name', 'run_results.json')
    exp_json_path = exp_dir / exp_json_name
    
    output_data = {
        "total_wall_clock_time": pipeline_elapsed,
        "instances": all_results
    }
    with open(exp_json_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"\nSaved experiment results to {exp_json_path}")


if __name__ == '__main__':
    main()
