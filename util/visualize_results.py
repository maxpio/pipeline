import json
import yaml
import statistics
from pathlib import Path
import matplotlib.pyplot as plt

def load_config():
    config = {}
    for conf_file in ["config/config_general.yaml", "config/config_data.yaml"]:
        if Path(conf_file).exists():
            with open(conf_file, "r") as f:
                config.update(yaml.safe_load(f))
    return config

def main():
    config = load_config()
    data_dir = Path(config.get('general_settings', {}).get('data_dir', '.'))
    exp_dir = data_dir / "experiments"
    vis_dir = data_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    import sys
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = config.get('visualization_settings', {}).get('experiment_files', [])
    
    if not files:
        print("No experiment files provided via args or in config.yaml under visualization_settings.experiment_files")
        return

    # run_name -> {"total_time": float, "instances": {inst_name: dict}}
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
    
    # 1. Plot total wall clock time
    plt.figure(figsize=(10, 6))
    times = [data_by_run[r]["total_time"] for r in run_names]
    plt.bar(run_names, times, color='skyblue')
    plt.ylabel("Total Wall-Clock Time (s)")
    plt.title("Total Pipeline Execution Time Comparison")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(vis_dir / "total_wall_clock_time.png")
    plt.close()
    print(f"Saved {vis_dir / 'total_wall_clock_time.png'}")

    # Metrics to plot: (JSON key, Display Name)
    metrics_to_plot = [
        ("solving_time", "Runtime per Instance (s)"),
        ("slp_iterations_main_loop", "SLP Iterations Main Loop"),
        ("total_slp_iters", "SLP Iterations Main Loop + Custom Pricing"),
        ("cols_needed_for_rmp_feasibility", "Columns Needed for RMP Feasibility")
    ]

    # Pre-process derived metrics
    for run in run_names:
        for inst_name, inst_data in data_by_run[run]["instances"].items():
            slp_main = inst_data.get("slp_iterations_main_loop", 0)
            slp_custom = inst_data.get("slp_iterations_custom_pricing", 0)
            inst_data["total_slp_iters"] = slp_main + slp_custom

    # Find intersection of successful instances
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

    # Boxplots per metric
    for metric_key, metric_name in metrics_to_plot:
        
        abs_data = []
        for run in run_names:
            vals = [data_by_run[run]["instances"][inst].get(metric_key, 0.0) for inst in all_instances]
            abs_data.append(vals)
            
        safe_name = metric_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
        
        # Absolute Boxplot
        means_abs = [sum(vals)/len(vals) if len(vals) > 0 else 0 for vals in abs_data]
        medians_abs = [statistics.median(vals) if len(vals) > 0 else 0 for vals in abs_data]
        sorted_indices = sorted(range(len(means_abs)), key=lambda k: means_abs[k])
        
        sorted_abs_data = [abs_data[i] for i in sorted_indices]
        sorted_run_names_abs = [f"{run_names[i]}\nMean: {means_abs[i]:.2f}\nMed: {medians_abs[i]:.2f}" for i in sorted_indices]
        
        plt.figure(figsize=(10, 8))
        plt.boxplot(sorted_abs_data, tick_labels=sorted_run_names_abs, showmeans=True)
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} - Absolute Values")
        plt.xticks(rotation=45, ha='center')
        plt.tight_layout()
        plt.savefig(vis_dir / f"{safe_name}_absolute.png")
        plt.close()
        print(f"Saved {vis_dir / f'{safe_name}_absolute.png'}")
        
        # Normalized Boxplot (divided by the worst/max value per instance)
        norm_data = [[] for _ in run_names]
        for inst_idx, _ in enumerate(all_instances):
            vals_for_inst = [abs_data[run_idx][inst_idx] for run_idx in range(len(run_names))]
            worst_val = max(vals_for_inst)
            
            for run_idx in range(len(run_names)):
                val = vals_for_inst[run_idx]
                if worst_val > 0:
                    norm_val = val / worst_val
                else:
                    norm_val = 0.0
                norm_data[run_idx].append(norm_val)
                
        means_norm = [sum(vals)/len(vals) if len(vals) > 0 else 0 for vals in norm_data]
        medians_norm = [statistics.median(vals) if len(vals) > 0 else 0 for vals in norm_data]
        sorted_indices_norm = sorted(range(len(means_norm)), key=lambda k: means_norm[k])
        
        sorted_norm_data = [norm_data[i] for i in sorted_indices_norm]
        sorted_run_names_norm = [f"{run_names[i]}\nMean: {means_norm[i]:.2f}\nMed: {medians_norm[i]:.2f}" for i in sorted_indices_norm]
                
        plt.figure(figsize=(10, 8))
        plt.boxplot(sorted_norm_data, tick_labels=sorted_run_names_norm, showmeans=True)
        plt.ylabel(f"Normalized {metric_name} (relative to worst)")
        plt.title(f"{metric_name} - Normalized")
        plt.xticks(rotation=45, ha='center')
        plt.tight_layout()
        plt.savefig(vis_dir / f"{safe_name}_normalized.png")
        plt.close()
        print(f"Saved {vis_dir / f'{safe_name}_normalized.png'}")

if __name__ == "__main__":
    main()
