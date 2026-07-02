"""
Generates optimal dual values for the continuous relaxation (LP relaxation) of MILP instances.
Reads LP files, relaxes integrality, solves the LP using SCIP, and extracts dual values for master constraints.
"""
import os
import sys
import glob
import json
import concurrent.futures
import yaml
from pyscipopt import Model

# Allow importing from src_ml
_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_BASE, "..", "src_ml"))
from feature_extractor import get_master_constraints

# Paths and Settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config_full = {}
for conf_file in ["config/config_general.yaml", "config/config_data.yaml", "config/config_test_base.yaml"]:
    config_path = os.path.abspath(os.path.join(BASE_DIR, "..", conf_file))
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_full.update(yaml.safe_load(f))

config = config_full.get("duals_generation_settings", {})

DATA_DIR = config_full.get("general_settings", {}).get("data_dir")
if not DATA_DIR:
    raise ValueError("general_settings -> data_dir not found in config.yaml")

LP_DIR = os.path.join(DATA_DIR, "lpfiles")
DEC_DIR = os.path.join(DATA_DIR, "decfiles")
DUAL_BASE_DIR = os.path.join(DATA_DIR, "dualvalues")

CR_DUAL_VALUES_DIR = os.path.join(DUAL_BASE_DIR, "cr_optimal")

BOUNDS_DIR = os.path.join(DATA_DIR, "bounds")
RESULTS_JSON_FILE = os.path.join(BOUNDS_DIR, "cr_bounds.json")

# Generation Options
PROCESS_SPLITS = config.get("process_splits", False)
PREFIX_FILTER = config.get("prefix_filter", "gap")

# Execution Options
MULTICORE_SOLVING = config.get("multicore_solving", False)
MAX_WORKERS = config_full.get("general_settings", {}).get("num_cores", 1)
SHORT_LOG_OUTPUT = MULTICORE_SOLVING

def process_instance(lp_file):
    """Processes a single LP instance: finds optimal dual values for its continuous relaxation."""
    filename = os.path.basename(lp_file)
    base_name = os.path.splitext(filename)[0]
    dec_file = os.path.join(DEC_DIR, f"{base_name}.dec")
    
    if not SHORT_LOG_OUTPUT:
        print(f"\n{'='*40}")
        print(f"Processing {filename}...")
        print(f"{'='*40}")
    
    if not os.path.exists(dec_file):
        if not SHORT_LOG_OUTPUT:
            print(f"No .dec file found for {filename} at {dec_file}. Skipping...")
        return filename, None
        
    master_conss = get_master_constraints(dec_file)
    
    if not master_conss:
        if not SHORT_LOG_OUTPUT:
            print(f"No master constraints found in {filename}. Skipping...")
        return filename, None
        
    model = Model("LP_Relaxation")
    model.hideOutput()
    model.readProblem(lp_file)
    
    # Relax integrality
    for v in model.getVars():
        model.chgVarType(v, "CONTINUOUS")
        
    model.optimize()
    
    if model.getStatus() not in ("optimal", "gaplimit"):
        if not SHORT_LOG_OUTPUT:
            print(f"Could not solve {filename} to optimality. Status: {model.getStatus()}")
        return filename, None
        
    best_obj = model.getObjVal()
    
    if not SHORT_LOG_OUTPUT:
        print(f"Optimal LP Objective: {best_obj:.4f}")
        
    optimal_multipliers = {}
    for cons in model.getConss():
        if cons.name in master_conss:
            try:
                # SCIP requires constraint to be in the original problem or similar?
                # In PySCIPOpt, getting duals of linear constraints:
                dual_val = model.getDualsolLinear(cons)
                optimal_multipliers[cons.name] = dual_val
            except Exception as e:
                if not SHORT_LOG_OUTPUT:
                    print(f"Warning: Could not get dual value for constraint {cons.name}: {e}")
                optimal_multipliers[cons.name] = 0.0

    out_file_opt = os.path.join(CR_DUAL_VALUES_DIR, f"{base_name}.txt")
    with open(out_file_opt, "w") as f:
        for cons_name, dual_val in optimal_multipliers.items():
            f.write(f'"{cons_name}": {abs(dual_val)},\n')
    if not SHORT_LOG_OUTPUT:
        print(f"Saved optimal CR multipliers to: {out_file_opt}")

    if not SHORT_LOG_OUTPUT:
        print(f"\n--- Finished {filename} ---")
        
    return base_name, best_obj

if __name__ == "__main__":
    os.makedirs(CR_DUAL_VALUES_DIR, exist_ok=True)
        
    lpfile_subdir = config_full.get("prediction_parameters", {}).get("lpfile_subdir")
    
    if lpfile_subdir == "":
        lp_files = glob.glob(os.path.join(LP_DIR, "*.lp"))
        if not lp_files:
            print(f"No .lp files found in {LP_DIR}")
    elif lpfile_subdir is not None:
        target_dir = os.path.join(LP_DIR, lpfile_subdir)
        lp_files = glob.glob(os.path.join(target_dir, "*.lp"))
        if not lp_files:
            print(f"No .lp files found in {target_dir}")
    else:
        if PROCESS_SPLITS:
            lp_files = []
            for split in ['training', 'val', 'test']:
                lp_files.extend(glob.glob(os.path.join(LP_DIR, split, "*.lp")))
            if not lp_files:
                print(f"No .lp files found in subdirectories (training, val, test) of {LP_DIR}")
        else:
            lp_files = glob.glob(os.path.join(LP_DIR, "*.lp"))
            if not lp_files:
                print(f"No .lp files found in {LP_DIR}")
        
    if PREFIX_FILTER:
        lp_files = [f for f in lp_files if os.path.basename(f).startswith(PREFIX_FILTER)]
        
    results = {}
    
    if MULTICORE_SOLVING:
        print(f"Starting multicore solving with {MAX_WORKERS} workers...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_instance, lp_file): os.path.basename(lp_file) for lp_file in lp_files}
                
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                filename = futures[future]
                completed += 1
                try:
                    fn, best_obj = future.result()
                    if best_obj is not None:
                        results[fn] = best_obj
                        if SHORT_LOG_OUTPUT:
                            print(f"[{completed}/{len(lp_files)}] Finished {fn} (Best CR Objective: {best_obj:.4f})")
                    else:
                        if SHORT_LOG_OUTPUT:
                            print(f"[{completed}/{len(lp_files)}] Skipped {fn} (No master constraints or error)")
                except Exception as exc:
                    print(f"[{completed}/{len(lp_files)}] Exception in {filename}: {exc}")
    else:
        for i, lp_file in enumerate(lp_files):
            filename = os.path.basename(lp_file)
            
            fn, best_obj = process_instance(lp_file)
            if best_obj is not None:
                results[fn] = best_obj
                if SHORT_LOG_OUTPUT:
                    print(f"[{i+1}/{len(lp_files)}] Finished {fn} (Best CR Objective: {best_obj:.4f})")
            else:
                if SHORT_LOG_OUTPUT:
                    print(f"[{i+1}/{len(lp_files)}] Skipped {fn} (No master constraints or error)")
            
    os.makedirs(os.path.dirname(RESULTS_JSON_FILE), exist_ok=True)
    with open(RESULTS_JSON_FILE, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved optimal CR dual values summary to {RESULTS_JSON_FILE}")
