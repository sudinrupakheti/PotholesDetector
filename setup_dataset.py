#!/usr/bin/env python3
import os
import shutil
import random
import subprocess
import zipfile
from pathlib import Path
from collections import defaultdict

DATASET_NAME = "rajdalsaniya/pothole-detection-dataset"
TRAIN_PCT = 0.70
VAL_PCT = 0.15
TEST_PCT = 0.15

def download_dataset():
    """Download dataset from Kaggle"""
    print("Downloading dataset...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", DATASET_NAME],
            check=True
        )
        print("✓ Download complete")
    except subprocess.CalledProcessError:
        print("✗ Download failed. Check Kaggle API credentials.")
        return False
    return True

def extract_dataset():
    """Extract zip file"""
    print("Extracting dataset...")
    zip_file = "pothole-detection-dataset.zip"

    if not Path(zip_file).exists():
        print(f"✗ {zip_file} not found")
        return False

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall()

    print("✓ Extraction complete")
    return True

def find_dataset_folder():
    """Find extracted dataset folder"""
    possible_names = ["Dataset3Class", "pothole-detection-dataset", "data"]

    for name in possible_names:
        if Path(name).exists() and Path(name).is_dir():
            return Path(name)

    print("✗ Dataset folder not found")
    return None

def organize_dataset(data_path):
    """Organize YOLO dataset into train/val/test splits"""
    print("Organizing dataset...")

    # Get all files
    all_files = sorted(os.listdir(data_path))

    # Group files by number (pair image + txt)
    pairs = defaultdict(dict)
    for filename in all_files:
        if filename.startswith('.'):
            continue

        if '.' in filename:
            name_part = filename.rsplit('.', 1)[0]
            ext = filename.rsplit('.', 1)[1]

            if '_' in name_part:
                parts = name_part.rsplit('_', 1)
                prefix = parts[0]
                number = parts[1]
            else:
                continue

            pair_key = f"{prefix}_{number}"
            pairs[pair_key][ext] = filename

    # Filter valid pairs
    valid_pairs = []
    for pair_key, files in pairs.items():
        if ('jpg' in files or 'jpeg' in files) and 'txt' in files:
            valid_pairs.append((pair_key, files))

    print(f"Found {len(valid_pairs)} valid pairs")

    # Random shuffle and split
    random.shuffle(valid_pairs)

    total = len(valid_pairs)
    train_count = int(total * TRAIN_PCT)
    val_count = int(total * VAL_PCT)

    train_pairs = valid_pairs[:train_count]
    val_pairs = valid_pairs[train_count:train_count + val_count]
    test_pairs = valid_pairs[train_count + val_count:]

    print(f"Split: Train {len(train_pairs)} | Val {len(val_pairs)} | Test {len(test_pairs)}")

    # Create directory structure
    dirs = [
        "images/train", "images/val", "images/test",
        "labels/train", "labels/val", "labels/test"
    ]

    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Move files
    def move_pairs(pairs, split_name):
        for pair_key, files in pairs:
            img_ext = 'jpg' if 'jpg' in files else 'jpeg'
            img_src = data_path / files[img_ext]
            img_dst = Path(f"images/{split_name}") / files[img_ext]
            shutil.copy2(img_src, img_dst)

            txt_src = data_path / files['txt']
            txt_dst = Path(f"labels/{split_name}") / files['txt']
            shutil.copy2(txt_src, txt_dst)

    move_pairs(train_pairs, "train")
    move_pairs(val_pairs, "val")
    move_pairs(test_pairs, "test")

    print("✓ Files organized")

def create_yaml():
    """Create data.yaml for YOLOv8"""
    print("Creating data.yaml...")

    abs_path = Path.cwd().absolute()
    yaml_content = f"""path: {abs_path}
train: images/train
val: images/val
test: images/test
nc: 1
names: ['pothole']
"""

    with open("data.yaml", "w") as f:
        f.write(yaml_content)

    print("✓ data.yaml created")

def verify():
    """Verify counts match"""
    print("\n=== Verification ===")
    for split in ["train", "val", "test"]:
        img_count = len(os.listdir(f"images/{split}"))
        lbl_count = len(os.listdir(f"labels/{split}"))
        match = "✓" if img_count == lbl_count else "✗"
        print(f"{split}: {img_count} images, {lbl_count} labels {match}")

def cleanup(data_path):
    """Remove original dataset folder and zip"""
    print("\nCleaning up...")
    shutil.rmtree(data_path, ignore_errors=True)
    zip_file = "pothole-detection-dataset.zip"
    if Path(zip_file).exists():
        Path(zip_file).unlink()
    print("✓ Cleanup complete")

def main():
    print("=== Pothole Dataset Setup ===\n")

    # Download
    if not download_dataset():
        return

    # Extract
    if not extract_dataset():
        return

    # Find dataset
    data_path = find_dataset_folder()
    if not data_path:
        return

    # Organize
    organize_dataset(data_path)

    # Create yaml
    create_yaml()

    # Verify
    verify()

    # Cleanup
    cleanup(data_path)

    print("\n✓✓✓ READY FOR TRAINING ✓✓✓")
    print("Use: yolo detect train data=data.yaml model=yolov8n.pt epochs=50")

if __name__ == "__main__":
    main()
