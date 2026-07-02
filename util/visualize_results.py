"""
Generates comparison plots and boxplots from experiment JSON results.
"""
import json
import yaml
import statistics
from pathlib import Path
import matplotlib.pyplot as plt

def load_config():
    """Loads settings from configuration YAML files."""
    config = {}
    for conf_file in ["config/config_general.yaml", "config/config_data.yaml"]:
        if Path(conf_file).exists():
            with open(conf_file, "r") as f:
                config.update(yaml.safe_load(f))
    return config

def main():
    """Reads results and generates visualizations."""
    config = load_config()
    data_dir = Path(config.get('general_settings', {}).get('data_dir', '.'))
    exp_dir = data_dir / "experiments"
    vis_dir = data_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    import sys
    vis_settings = config.get('visualization_settings', {})
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        if vis_settings.get('process_all', False):
            files = [f.name for f in exp_dir.glob("*.json")]
        else:
            files = vis_settings.get('experiment_files', [])
    
    if not files:
        print("No experiment files provided via args or in config.yaml under visualization_settings.experiment_files")
        return

    # Map runs to data
    data_by_run = {}
    
    for fname in files:
        fpath = exp_dir / fname
        if not fpath.exists():
            print(f"Warning: file {fpath} does not exist, skipping.")
            continue
        with open(fpath, "r") as f:
            d = json.load(f)
            
        run_name = fname.replace(".json", "")
        # Parse instances
        inst_dict = {
            inst["instance"]: inst 
            for inst in d.get("instances", []) 
            if inst.get("status") == "SUCCESS"
        }
        data_by_run[run_name] = {
            "total_time": d.get("total_wall_clock_time", 0.0),
            "instances": inst_dict
        }

    if not data_by_run:
        print("No data loaded. Exiting.")
        return

    run_names = list(data_by_run.keys())
    run_names.sort(key=lambda r: data_by_run[r]["total_time"])
    plot_width = max(10, len(run_names) * 1.5)
    
    # Plot total time
    plt.figure(figsize=(plot_width, 6))
    times = [data_by_run[r]["total_time"] for r in run_names]
    plt.bar(run_names, times, color='skyblue')
    plt.ylabel("Total Wall-Clock Time (s)")
    plt.title("Total Pipeline Execution Time Comparison")
    plt.xticks(rotation=90, ha='right')
    plt.tight_layout()
    plt.savefig(vis_dir / "total_wall_clock_time.png")
    plt.close()
    print(f"Saved {vis_dir / 'total_wall_clock_time.png'}")

    # Set metrics
    metrics_to_plot = [
        ("solving_time", "Runtime per Instance (s)"),
        ("slp_iterations_non_pricing", "SLP Iterations Non-Pricing"),
        ("slp_iterations_pricing", "SLP Iterations Pricing"),
        ("slp_total_slp_iters", "SLP Iterations Total"),
        ("cols_needed_for_rmp_feasibility", "Columns Needed for RMP Feasibility")
    ]

    # Calc derived metrics
    for run in run_names:
        for inst_name, inst_data in data_by_run[run]["instances"].items():
            slp_non_pricing = inst_data.get("slp_iterations_non_pricing", 0)
            slp_pricing = inst_data.get("slp_iterations_pricing", 0)
            inst_data["slp_total_slp_iters"] = slp_non_pricing + slp_pricing

    # Get common instances
    all_instances = set()
    for i, run in enumerate(run_names):
        insts = set(data_by_run[run]["instances"].keys())
        if i == 0:
            all_instances = insts
        else:
            all_instances = all_instances.intersection(insts)

    all_instances = sorted(list(all_instances))
    print(f"Plotting metrics for {len(all_instances)} common successful instances.")

    if not all_instances:
        print("No common instances found across runs. Cannot create boxplots.")
        return

    # Create boxplots
    for metric_key, metric_name in metrics_to_plot:
        
        abs_data = []
        for run in run_names:
            vals = [data_by_run[run]["instances"][inst].get(metric_key, 0.0) for inst in all_instances]
            abs_data.append(vals)
            
        safe_name = metric_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
        
        # Abs boxplot
        means_abs = [sum(vals)/len(vals) if len(vals) > 0 else 0 for vals in abs_data]
        medians_abs = [statistics.median(vals) if len(vals) > 0 else 0 for vals in abs_data]
        sorted_indices = sorted(range(len(means_abs)), key=lambda k: means_abs[k])
        
        sorted_abs_data = [abs_data[i] for i in sorted_indices]
        sorted_run_names_abs = [f"{run_names[i]}\nMean: {means_abs[i]:.2f}\nMed: {medians_abs[i]:.2f}" for i in sorted_indices]
        
        plt.figure(figsize=(plot_width, 8))
        plt.boxplot(sorted_abs_data, tick_labels=sorted_run_names_abs, showmeans=True)
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} - Absolute Values")
        plt.xticks(rotation=90, ha='center')
        plt.tight_layout()
        plt.savefig(vis_dir / f"{safe_name}_absolute.png")
        plt.close()
        print(f"Saved {vis_dir / f'{safe_name}_absolute.png'}")
        
        # Norm boxplot
        norm_data = [[] for _ in run_names]
        normalized_by_file = vis_settings.get('normalized_by', "")
        norm_run_name = normalized_by_file.replace(".json", "") if normalized_by_file else ""
        
        for inst_idx, _ in enumerate(all_instances):
            vals_for_inst = [abs_data[run_idx][inst_idx] for run_idx in range(len(run_names))]
            
            if norm_run_name and norm_run_name in run_names:
                base_val = abs_data[run_names.index(norm_run_name)][inst_idx]
            else:
                base_val = max(vals_for_inst)
            
            for run_idx in range(len(run_names)):
                val = vals_for_inst[run_idx]
                if base_val > 0:
                    norm_val = val / base_val
                else:
                    norm_val = 0.0
                norm_data[run_idx].append(norm_val)
                
        means_norm = [sum(vals)/len(vals) if len(vals) > 0 else 0 for vals in norm_data]
        medians_norm = [statistics.median(vals) if len(vals) > 0 else 0 for vals in norm_data]
        sorted_indices_norm = sorted(range(len(means_norm)), key=lambda k: means_norm[k])
        
        sorted_norm_data = [norm_data[i] for i in sorted_indices_norm]
        sorted_run_names_norm = [f"{run_names[i]}\nMean: {means_norm[i]:.2f}\nMed: {medians_norm[i]:.2f}" for i in sorted_indices_norm]
                
        plt.figure(figsize=(plot_width, 8))
        plt.boxplot(sorted_norm_data, tick_labels=sorted_run_names_norm, showmeans=True)
        
        if norm_run_name and norm_run_name in run_names:
            plt.ylabel(f"Normalized {metric_name} (relative to {normalized_by_file})")
        else:
            plt.ylabel(f"Normalized {metric_name} (relative to worst)")
        plt.title(f"{metric_name} - Normalized")
        plt.xticks(rotation=90, ha='center')
        plt.tight_layout()
        plt.savefig(vis_dir / f"{safe_name}_normalized.png")
        plt.close()
        print(f"Saved {vis_dir / f'{safe_name}_normalized.png'}")

if __name__ == "__main__":
    main()
