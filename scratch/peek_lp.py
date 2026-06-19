import os
lp_dir = '/home/max/master_ths/data/GAD/lpfiles/training'
files = os.listdir(lp_dir)
if files:
    with open(os.path.join(lp_dir, files[0]), 'r') as f:
        print(f.read(200))
