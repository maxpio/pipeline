"""
Generates zero dual values for MILP instances.
Reads .dec files to extract master constraints, and sets all dual values to 0.0.
"""
import os
import sys
import glob
import concurrent.futures
import yaml
import re

# Allow importing from src_ml
_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_BASE, "..", "src_ml"))
from feature_extractor import get_master_constraints

# Paths and Settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config_full = {}
for conf_file in ["config/config_general.yaml", "config/config_data.yaml"]:
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

ZERO_DUAL_VALUES_DIR = os.path.join(DUAL_BASE_DIR, "zero")

# Generation Options
PROCESS_SPLITS = config.get("process_splits", False)
PREFIX_FILTER = config.get("prefix_filter", "gap")

# Execution Options
MULTICORE_SOLVING = config.get("multicore_solving", False)
MAX_WORKERS = config_full.get("general_settings", {}).get("num_cores", 1)
SHORT_LOG_OUTPUT = MULTICORE_SOLVING

def process_instance(lp_file):
    """Processes a single LP instance: finds master constraints and sets duals to zero."""
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
        return filename
        
    master_conss = get_master_constraints(dec_file)
    
    if not master_conss:
        if not SHORT_LOG_OUTPUT:
            print(f"No master constraints found in {filename}. Skipping...")
        return filename

    out_file_opt = os.path.join(ZERO_DUAL_VALUES_DIR, f"{base_name}.txt")
    with open(out_file_opt, "w") as f:
        for cons_name in sorted(master_conss, key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]):
            f.write(f'"{cons_name}": 0.0,\n')
            
    if not SHORT_LOG_OUTPUT:
        print(f"Saved zero multipliers to: {out_file_opt}")
        print(f"\n--- Finished {filename} ---")
        
    return base_name

if __name__ == "__main__":
    os.makedirs(ZERO_DUAL_VALUES_DIR, exist_ok=True)
        
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
        
    if MULTICORE_SOLVING:
        print(f"Starting multicore solving with {MAX_WORKERS} workers...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_instance, lp_file): os.path.basename(lp_file) for lp_file in lp_files}
                
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                filename = futures[future]
                completed += 1
                try:
                    fn = future.result()
                    if SHORT_LOG_OUTPUT:
                        print(f"[{completed}/{len(lp_files)}] Finished {fn}")
                except Exception as exc:
                    print(f"[{completed}/{len(lp_files)}] Exception in {filename}: {exc}")
    else:
        for i, lp_file in enumerate(lp_files):
            filename = os.path.basename(lp_file)
            fn = process_instance(lp_file)
            if SHORT_LOG_OUTPUT:
                print(f"[{i+1}/{len(lp_files)}] Finished {fn}")
            
    print(f"\nFinished saving zero dual values.")
