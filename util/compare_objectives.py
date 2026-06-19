import json
import yaml
from pathlib import Path

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    data_dir = Path(config.get('general_settings', {}).get('data_dir', '.'))
    exp_dir = data_dir / "experiments"
    files = config.get('visualization_settings', {}).get('experiment_files', [])
    
    if not files:
        print("No experiment files defined in config.yaml under visualization_settings.experiment_files")
        return

    # run_name -> {inst_name: final_obj_val}
    data_by_run = {}
    
    for fname in files:
        fpath = exp_dir / fname
        if not fpath.exists():
            print(f"Warning: file {fpath} does not exist, skipping.")
            continue
        with open(fpath, "r") as f:
            d = json.load(f)
            
        run_name = fname.replace(".json", "")
        # Parse instances into a dict keyed by instance name
        inst_dict = {
            inst["instance"]: inst.get("final_obj_val")
            for inst in d.get("instances", []) 
            if inst.get("status") == "SUCCESS"
        }
        data_by_run[run_name] = inst_dict

    if not data_by_run:
        print("No data loaded. Exiting.")
        return

    run_names = list(data_by_run.keys())
    
    # Find all instances that exist in at least one run
    all_instances = set()
    for run in run_names:
        all_instances.update(data_by_run[run].keys())

    all_instances = sorted(list(all_instances))
    
    differences_found = 0
    tolerance = 1e-2  # Allow small floating point variations

    for inst in all_instances:
        values = []
        for run in run_names:
            val = data_by_run[run].get(inst)
            if val is not None:
                values.append((run, val))
        
        if len(values) < 2:
            continue
            
        # Compare all values against the first one
        base_run, base_val = values[0]
        for other_run, other_val in values[1:]:
            # Handle 'inf' or missing floats safely
            if base_val == float('inf') and other_val == float('inf'):
                continue
            if base_val == float('inf') or other_val == float('inf') or abs(base_val - other_val) > tolerance:
                differences_found += 1
                print(f"Mismatch in {inst}:")
                for run, val in values:
                    print(f"  - {run}: {val}")
                print("-" * 40)
                break  # Only print mismatch for this instance once

    if differences_found == 0:
        print(f"Success! No objective value differences found across {len(all_instances)} instances.")
    else:
        print(f"Found {differences_found} instances with mismatched objective values.")

if __name__ == "__main__":
    main()
