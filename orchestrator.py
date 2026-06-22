"""
Runs experiments based on YAML config files and visualizes the results.
"""
import os
import sys
import yaml
import subprocess
from pathlib import Path

def load_config(base_dir):
    """Loads configuration from config_data.yaml."""
    config = {}
    config_path = base_dir / "config" / "config_data.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            config.update(yaml.safe_load(f) or {})
    return config

def main():
    """Finds experiment files, runs them sequentially, and triggers visualization."""
    base_dir = Path(__file__).resolve().parent
    config = load_config(base_dir)
    tests_dir_name = config.get("orchestrator_settings", {}).get("tests_dir", str(base_dir / "tests"))
    tests_dir = Path(tests_dir_name).resolve()
    
    if not tests_dir.exists():
        print(f"Tests directory not found at {tests_dir}. Creating it...")
        tests_dir.mkdir(parents=True, exist_ok=True)
        
    experiment_files = sorted(tests_dir.glob("*.yaml"))
    
    if not experiment_files:
        print(f"No experiment yaml files found in {tests_dir}.")
        print(f"Please create at least one experiment yaml file (e.g., experiment1.yaml) in {tests_dir}.")
        sys.exit(0)
        
    print(f"Found {len(experiment_files)} experiment files in {tests_dir}.")
    print("Running them sequentially...\n")
    
    for exp_file in experiment_files:
        print("="*80)
        print(f"Running experiment: {exp_file.name}")
        print("="*80)
        
        env = os.environ.copy()
        env["EXPERIMENT_CONFIG"] = str(exp_file)
        
        # Run experiment module
        cmd = [sys.executable, "-m", "src_ml.solve_instances"]
        
        try:
            # Run and stream output
            subprocess.run(cmd, env=env, cwd=base_dir, check=True)
            print(f"\n[+] Experiment {exp_file.name} completed successfully.\n")
        except subprocess.CalledProcessError as e:
            print(f"\n[!] Experiment {exp_file.name} failed with return code {e.returncode}.\n")
            # Continue on failure
            continue

    print("All experiments finished.")

    # Run visualize_results.py
    print("\n" + "="*80)
    print("Running visualize_results.py on all generated experiment results")
    print("="*80)
    
    # Find JSON files
    base_config_path = base_dir / "config" / "config_test_base.yaml"
    base_json = "opt.json"
    if base_config_path.exists():
        with open(base_config_path, 'r') as f:
            base_data = yaml.safe_load(f) or {}
            base_json = base_data.get('prediction_parameters', {}).get('experiment_json_name', 'opt.json')
            
    json_files = []
    for exp_file in experiment_files:
        with open(exp_file, 'r') as f:
            exp_data = yaml.safe_load(f) or {}
            json_name = exp_data.get('prediction_parameters', {}).get('experiment_json_name', base_json)
            if json_name not in json_files:
                json_files.append(json_name)
                
    cmd = [sys.executable, "-m", "util.visualize_results"] + json_files
    try:
        subprocess.run(cmd, cwd=base_dir, check=True)
        print("\nVisualization completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\nVisualization failed with return code {e.returncode}.")

if __name__ == "__main__":
    main()
