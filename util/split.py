import os
import glob
import random
import shutil

# Split ratios (must sum to 1.0)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

def read_config():
    # Find config.txt in parent directory or current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(script_dir, "..", "config.txt"),
        os.path.join(script_dir, "config.txt"),
        "config.txt"
    ]
    
    config_path = None
    for path in possible_paths:
        if os.path.exists(path):
            config_path = path
            break
            
    if not config_path:
        raise FileNotFoundError("Could not locate config.txt file.")
        
    config = {}
    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                config[key.strip()] = val.strip()
    return config

def main():
    config = read_config()
    data_dir = config.get("data_dir")
    if not data_dir:
        raise ValueError("data_dir is not defined in config.txt")
        
    lpfiles_dir = os.path.join(data_dir, "lpfiles")
    if not os.path.isdir(lpfiles_dir):
        raise FileNotFoundError(f"Subdirectory 'lpfiles' not found in data directory: {lpfiles_dir}")
        
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
        "train": lp_files[:train_end],
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

