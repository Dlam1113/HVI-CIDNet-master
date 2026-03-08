import os
import random
import shutil

def reduce_dataset(dataset_dir, target_count=1400):
    train_high_dir = os.path.join(dataset_dir, 'high')
    train_low_dir = os.path.join(dataset_dir, 'low')
    
    if not os.path.exists(train_high_dir) or not os.path.exists(train_low_dir):
        print(f"Skipping {dataset_dir}: Directories not found.")
        return
        
    high_files = sorted(os.listdir(train_high_dir))
    low_files = sorted(os.listdir(train_low_dir))
    
    # Check pairing
    if len(high_files) != len(low_files):
        print(f"Warning: {dataset_dir} has mismatched high/low counts ({len(high_files)} vs {len(low_files)}).")
        
    current_count = len(high_files)
    print(f"[{os.path.basename(dataset_dir)}] Current pairs: {current_count}")
    
    if current_count <= target_count:
        print(f"  -> Already <= {target_count}, no need to reduce.")
        return
        
    # Create an unused directory to move excess files rather than deleting them immediately
    unused_high_dir = os.path.join(dataset_dir, 'unused_high')
    unused_low_dir = os.path.join(dataset_dir, 'unused_low')
    os.makedirs(unused_high_dir, exist_ok=True)
    os.makedirs(unused_low_dir, exist_ok=True)
    
    # Randomly select which ones to KEEP
    random.seed(42) # For reproducibility
    files_to_keep = set(random.sample(high_files, target_count))
    files_to_move = [f for f in high_files if f not in files_to_keep]
    
    print(f"  -> Moving {len(files_to_move)} pairs to unused directories...")
    
    moved_count = 0
    for filename in files_to_move:
        high_src = os.path.join(train_high_dir, filename)
        high_dst = os.path.join(unused_high_dir, filename)
        low_src = os.path.join(train_low_dir, filename)
        low_dst = os.path.join(unused_low_dir, filename)
        
        # We need to make sure the low file exists with same name (as previously established, they have identical names)
        if os.path.exists(low_src):
            shutil.move(high_src, high_dst)
            shutil.move(low_src, low_dst)
            moved_count += 1
        else:
            print(f"  Warning: Low image {filename} corresponding to high image not found. Kept high image as is.")
            
    print(f"  -> Done. Reduced to {target_count} pairs. (Moved {moved_count} pairs)")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Reduce dataset sizes.")
    parser.add_argument('--base_dir', type=str, default=r'd:\HVI-CIDNet-master\new_datasets\filtered',
                        help='Base directory of the datasets')
    args = parser.parse_args()
    
    base_dir = args.base_dir
    loli_dir = os.path.join(base_dir, 'loli_pedestrian')
    foggy_dir = os.path.join(base_dir, 'cityscapes_foggy_pedestrian')
    
    print(f"--- Reducing Datasets in {base_dir} ---")
    reduce_dataset(loli_dir, target_count=1400)
    print("-" * 30)
    reduce_dataset(foggy_dir, target_count=1400)
    print("Finished.")
