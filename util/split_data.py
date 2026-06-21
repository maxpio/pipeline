import os
import glob
import random
import shutil
import yaml

def read_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config = {}
    for conf_file in ["config/config_general.yaml", "config/config_data.yaml"]:
        config_path = os.path.join(script_dir, "..", conf_file)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Could not locate {conf_file} at {config_path}")
        with open(config_path, "r") as f:
            config.update(yaml.safe_load(f))
    return config

def main():
    config = read_config()
    try:
        data_dir = config["general_settings"]["data_dir"]
        lpfiles_dir = os.path.join(data_dir, "lpfiles")
        random_seed = config["general_settings"].get("random_seed", 42)
    except KeyError:
        raise ValueError("general_settings -> data_dir is not defined in config.yaml")
        
    split_config = config.get("split_parameters", {})
    train_ratio = split_config.get("train_ratio", 0.8)
    val_ratio = split_config.get("val_ratio", 0.1)
    test_ratio = split_config.get("test_ratio", 0.1)
    
    # Verify split ratios sum to approximately 1.0
    if not abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-9:
        print(f"Warning: Split ratios do not sum to 1.0 (sum: {train_ratio + val_ratio + test_ratio})")

    if not os.path.isdir(lpfiles_dir):
        raise FileNotFoundError(f"lp_dir not found: {lpfiles_dir}")
        
    # Get all .lp files
    lp_files = glob.glob(os.path.join(lpfiles_dir, "*.lp"))
    if not lp_files:
        print(f"No .lp files found in {lpfiles_dir}")
        return
        
    print(f"Found {len(lp_files)} .lp files in {lpfiles_dir}")
    
    # Shuffle files with a seed for reproducibility
    random.seed(random_seed)
    random.shuffle(lp_files)
    
    # Define split ratios
    n_files = len(lp_files)
    train_end = int(n_files * train_ratio)
    val_end = train_end + int(n_files * val_ratio)
    
    splits = {
        "training": lp_files[:train_end],
        "val": lp_files[train_end:val_end],
        "test": lp_files[val_end:]
    }
    
    # Copy files to destination directories inside lpfiles
    for split_name, files in splits.items():
        split_dir = os.path.join(lpfiles_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        # Clean any existing files in directory to start fresh (optional but recommended for clean runs)
        for existing_file in glob.glob(os.path.join(split_dir, "*.lp")):
            os.remove(existing_file)
            
        print(f"Copying {len(files)} files to {split_dir}...")
        for filepath in files:
            shutil.copy2(filepath, split_dir)
            
    print("Splitting complete successfully!")

if __name__ == "__main__":
    main()

