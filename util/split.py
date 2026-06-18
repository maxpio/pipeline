import os
import glob
import random
import shutil

# Split ratios (must sum to 1.0)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

import yaml

def read_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not locate config.yaml at {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    config = read_config()
    try:
        lpfiles_dir = config["training_paths"]["lp_dir"]
    except KeyError:
        raise ValueError("training_paths -> lp_dir is not defined in config.yaml")
        
    if not os.path.isdir(lpfiles_dir):
        raise FileNotFoundError(f"lp_dir not found: {lpfiles_dir}")
        
    # Get all .lp files
    lp_files = glob.glob(os.path.join(lpfiles_dir, "*.lp"))
    if not lp_files:
        print(f"No .lp files found in {lpfiles_dir}")
        return
        
    print(f"Found {len(lp_files)} .lp files in {lpfiles_dir}")
    
    # Shuffle files with a fixed seed for reproducibility
    random.seed(RANDOM_SEED)
    random.shuffle(lp_files)
    
    # Define split ratios
    n_files = len(lp_files)
    train_end = int(n_files * TRAIN_RATIO)
    val_end = train_end + int(n_files * VAL_RATIO)
    
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

