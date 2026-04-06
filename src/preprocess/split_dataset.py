"""
Split a dataset of sign language videos/npy files into train/val/test
with balanced class distribution AND move actual files accordingly.

Supports:
- Video files: .mp4, .avi, .mov, .mkv, .webm
- Numpy files: .npy

Input structure (two modes):

1) Nested mode (default):
    data_dir/
        GLOSS1/*.mp4 (or *.npy)
        GLOSS2/*.mp4 (or *.npy)

2) Flat mode (use --flat):
    data_dir/*.mp4 (or *.npy)
    + WLASL JSON required for mapping video_id -> gloss
"""

import os, glob, argparse, shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from src.utils.logger import *
from src.config.config import NPY_DIR, SPLIT_DIR

# Supported file extensions
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
NPY_EXTENSIONS = ('.npy',)
ALL_EXTENSIONS = VIDEO_EXTENSIONS + NPY_EXTENSIONS


def is_valid_file(filename):
    """Check if file has a valid extension"""
    return filename.lower().endswith(ALL_EXTENSIONS)


def collect_nested(data_dir, file_type='auto'):
    """
    Collect files from gloss folders: data_dir/GLOSS/*.{mp4,npy}.
    
    Args:
        data_dir: Root directory
        file_type: 'video', 'npy', or 'auto' (detect automatically)
    """
    rows = []
    
    for label in sorted(os.listdir(data_dir)):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        
        # Collect files based on type
        if file_type == 'video':
            patterns = [os.path.join(label_dir, f"*{ext}") for ext in VIDEO_EXTENSIONS]
        elif file_type == 'npy':
            patterns = [os.path.join(label_dir, "*.npy")]
        else:  # auto
            patterns = [os.path.join(label_dir, f"*{ext}") for ext in ALL_EXTENSIONS]
        
        for pattern in patterns:
            for p in glob.glob(pattern):
                rows.append({"path": p, "label": label})
    
    return rows


def collect_flat(data_dir, json_path, file_type='auto'):
    """
    Collect from data_dir/*.{mp4,npy} using WLASL JSON to assign gloss labels.
    
    Args:
        data_dir: Directory containing files
        json_path: Path to WLASL JSON mapping
        file_type: 'video', 'npy', or 'auto'
    """
    import json
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Create mapping video_id -> gloss
    vid2gloss = {}
    for entry in data:
        gloss = entry["gloss"]
        for inst in entry["instances"]:
            vid = inst["video_id"]
            vid2gloss[vid] = gloss

    # Collect files
    rows = []
    
    if file_type == 'video':
        patterns = [os.path.join(data_dir, f"*{ext}") for ext in VIDEO_EXTENSIONS]
    elif file_type == 'npy':
        patterns = [os.path.join(data_dir, "*.npy")]
    else:  # auto
        patterns = [os.path.join(data_dir, f"*{ext}") for ext in ALL_EXTENSIONS]
    
    for pattern in patterns:
        for p in glob.glob(pattern):
            vid = Path(p).stem  # filename without extension
            if vid not in vid2gloss:
                logger.warning(f"Cannot find gloss for {vid}")
                continue
            rows.append({"path": p, "label": vid2gloss[vid]})
    
    return rows


def move_files(df, split_name, output_dir, copy_mode=True):
    """
    Move/copy files into output_dir/split_name/label/.
    
    Args:
        df: DataFrame with 'path' and 'label' columns
        split_name: 'train', 'val', or 'test'
        output_dir: Output directory
        copy_mode: If True, copy files. If False, move files.
    """
    action = "Copying" if copy_mode else "Moving"
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{action} {split_name}"):
        src = row["path"]
        label = row["label"]

        dst_dir = os.path.join(output_dir, split_name, label)
        os.makedirs(dst_dir, exist_ok=True)

        dst = os.path.join(dst_dir, os.path.basename(src))
        
        if copy_mode:
            shutil.copy2(src, dst)
        else:
            shutil.move(src, dst)


def detect_file_type(data_dir, flat=False):
    """
    Auto-detect whether directory contains videos or npy files.
    
    Returns:
        'video', 'npy', or 'mixed'
    """
    if flat:
        # Check files directly in data_dir
        all_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    else:
        # Check files in subdirectories
        all_files = []
        for root, dirs, files in os.walk(data_dir):
            all_files.extend(files)
            if len(all_files) > 100:  # Sample
                break
    
    video_count = sum(1 for f in all_files if f.lower().endswith(VIDEO_EXTENSIONS))
    npy_count = sum(1 for f in all_files if f.lower().endswith(NPY_EXTENSIONS))
    
    if npy_count > 0 and video_count == 0:
        return 'npy'
    elif video_count > 0 and npy_count == 0:
        return 'video'
    elif video_count > 0 and npy_count > 0:
        return 'mixed'
    else:
        return None


def split_dataset(data_dir, output_dir, json_path=None, flat=False,
                  train_ratio=0.7, val_ratio=0.15, seed=42, 
                  file_type='auto', move_files_flag=False):
    """
    Split dataset into train/val/test and organize files.
    
    Args:
        data_dir: Input directory
        output_dir: Output directory
        json_path: Path to WLASL JSON (required for flat mode)
        flat: Whether input is flat structure
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        seed: Random seed
        file_type: 'video', 'npy', or 'auto'
        move_files_flag: If True, move files. If False, copy files.
    """
    
    # Auto-detect file type if needed
    if file_type == 'auto':
        detected_type = detect_file_type(data_dir, flat)
        if detected_type is None:
            raise ValueError(f"No valid files found in {data_dir}")
        elif detected_type == 'mixed':
            logger.warning("Found both video and npy files. Specify --file_type to process only one type.")
            file_type = 'auto'  # Process all
        else:
            file_type = detected_type
            logger.info(f"Auto-detected file type: {file_type}")
    
    # Collect files
    if flat:
        if json_path is None:
            raise ValueError("Flat mode requires --json mapping file")
        rows = collect_flat(data_dir, json_path, file_type)
    else:
        rows = collect_nested(data_dir, file_type)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise ValueError(f"No valid files found under {data_dir}")

    # Determine file extension for output message
    sample_file = df['path'].iloc[0]
    file_ext = Path(sample_file).suffix
    
    logger.info(f"Found {len(df)} {file_ext} files")
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Number of classes: {df['label'].nunique()}")
    
    # Show class distribution summary
    class_counts = df['label'].value_counts()
    logger.info(f"\nClass distribution:")
    logger.info(f"  Min samples per class: {class_counts.min()}")
    logger.info(f"  Max samples per class: {class_counts.max()}")
    logger.info(f"  Mean samples per class: {class_counts.mean():.1f}")
    logger.info(f"  Median samples per class: {class_counts.median():.1f}")

    # Check for classes too small
    if class_counts.min() < 3:
        logger.warning("Some classes have <3 samples → stratification may fail")
        logger.info(f"Classes with <3 samples:\n{class_counts[class_counts < 3].to_string()}")

    test_ratio = 1.0 - train_ratio - val_ratio

    # 1) Split train+val vs test
    try:
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_ratio,
            random_state=seed,
            stratify=df["label"]
        )
    except ValueError as e:
        logger.error(f"Stratified split failed: {e}")
        logger.info("Attempting split without stratification...")
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_ratio,
            random_state=seed,
            stratify=None
        )

    # 2) Split train vs val
    val_adj = val_ratio / (train_ratio + val_ratio)

    try:
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_adj,
            random_state=seed,
            stratify=train_val_df["label"]
        )
    except ValueError as e:
        logger.error(f"Stratified split failed: {e}")
        logger.info("Attempting split without stratification...")
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_adj,
            random_state=seed,
            stratify=None
        )

    # Create output tree
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    # Move/copy files
    copy_mode = not move_files_flag
    move_files(train_df, "train", output_dir, copy_mode=copy_mode)
    move_files(val_df, "val", output_dir, copy_mode=copy_mode)
    move_files(test_df, "test", output_dir, copy_mode=copy_mode)

    # Summary
    action = "Moved" if move_files_flag else "Copied"
    
    summary = f"""
{'='*60}
SPLIT SUMMARY
{'='*60}
{action} files to: {output_dir}

Total:  {len(df):4d} files
Train:  {len(train_df):4d} ({len(train_df)/len(df)*100:.1f}%)
Val:    {len(val_df):4d} ({len(val_df)/len(df)*100:.1f}%)
Test:   {len(test_df):4d} ({len(test_df)/len(df)*100:.1f}%)

Classes per split:
  Train: {train_df['label'].nunique()} classes
  Val:   {val_df['label'].nunique()} classes
  Test:  {test_df['label'].nunique()} classes

Train distribution (top 10):
{train_df['label'].value_counts().head(10).to_string()}

Val distribution (top 10):
{val_df['label'].value_counts().head(10).to_string()}

Test distribution (top 10):
{test_df['label'].value_counts().head(10).to_string()}
{'='*60}
"""
    logger.info(summary)
    
    # Save split info to CSV
    info_path = os.path.join(output_dir, 'split_info.csv')
    split_info = pd.concat([
        train_df.assign(split='train'),
        val_df.assign(split='val'),
        test_df.assign(split='test')
    ])
    split_info.to_csv(info_path, index=False)
    logger.info(f"Split information saved to: {info_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Split dataset and organize files into train/val/test folders.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data_dir", default=NPY_DIR, 
                   help="Directory containing videos/npy files")
    p.add_argument("--output_dir", default=SPLIT_DIR, 
                   help="Output split dataset directory")
    p.add_argument("--json", default=None, 
                   help="JSON mapping path (required for flat mode)")
    p.add_argument("--flat", action="store_true", 
                   help="Use flat mode (files directly inside data_dir)")
    p.add_argument("--file_type", default='auto', choices=['auto', 'video', 'npy'],
                   help="Type of files to process (default: auto-detect)")
    p.add_argument("--train_ratio", type=float, default=0.7,
                   help="Training set ratio (default: 0.7)")
    p.add_argument("--val_ratio", type=float, default=0.15,
                   help="Validation set ratio (default: 0.15)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--move", action="store_true",
                   help="Move files instead of copying (saves disk space)")
    args = p.parse_args()

    logger = setup_logger("split_dataset")
    log_arguments(logger=logger, args=args)

    split_dataset(
        args.data_dir,
        args.output_dir,
        json_path=args.json,
        flat=args.flat,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        file_type=args.file_type,
        move_files_flag=args.move
    )