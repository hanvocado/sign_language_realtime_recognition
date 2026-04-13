"""
Create a WLASL subset and organize videos into:
train/<gloss>/, val/<gloss>/, test/<gloss>/

Uses split information from WLASL_v0.3.json:
    'split' ∈ ['train', 'val', 'test']

Usage:

  # Default (with train/val/test split):
  python -m src.utils.reorganize_wlasl --subset 100

  # Without splitting (videos go directly into <gloss>/ folders):
  python -m src.utils.reorganize_wlasl --subset 100 --no_split

  With --no_split, output is <dst_dir>/<gloss>/<video_id>.npy instead of <dst_dir>/<split>/<gloss>/<video_id>.npy.
"""

import os
import json
import shutil
from tqdm import tqdm
import argparse


def create_split_folders(dst_root, glosses):
    """Create train/val/test and gloss subfolders."""
    for split in ["train", "val", "test"]:
        for gloss in glosses:
            os.makedirs(os.path.join(dst_root, split, gloss), exist_ok=True)


def create_gloss_folders(dst_root, glosses):
    """Create gloss subfolders directly (no split)."""
    for gloss in glosses:
        os.makedirs(os.path.join(dst_root, gloss), exist_ok=True)


def main(subset, json_path, src_dir, dst_dir, data_type, no_split=False):
    # Load JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Only keep first `subset` glosses
    selected_entries = data[:subset]
    gloss_names = [entry["gloss"] for entry in selected_entries]

    # Prepare directory tree
    if no_split:
        create_gloss_folders(dst_dir, gloss_names)
    else:
        create_split_folders(dst_dir, gloss_names)

    print(f"\nOrganizing WLASL subset of {subset} glosses...")
    missing = 0
    copied = 0

    for entry in tqdm(selected_entries):
        gloss = entry["gloss"]
        instances = entry["instances"]

        for inst in instances:
            vid = inst["video_id"]
            src = os.path.join(src_dir, f"{vid}.{data_type}")

            if no_split:
                dst = os.path.join(dst_dir, gloss, f"{vid}.{data_type}")
            else:
                split_dir = inst["split"]  # 'train', 'val', or 'test'
                dst = os.path.join(dst_dir, split_dir, gloss, f"{vid}.{data_type}")

            if not os.path.exists(src):
                print(f"[WARNING] Missing video: {src}")
                missing += 1
                continue

            shutil.copy2(src, dst)
            copied += 1

    print("\n[SUMMARY]")
    print(f"Glosses processed: {subset}")
    print(f"Videos copied:     {copied}")
    print(f"Missing videos:    {missing}")
    print(f"Output structure:  {dst_dir}")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=100, help="Number of glosses to include")
    parser.add_argument("--json", default="data/wlasl/WLASL_v0.3.json", help="Path to WLASL json")
    parser.add_argument("--src_dir", default="data/wlasl/videos", help="Directory with all WLASL files (npy or mp4)")
    parser.add_argument("--dst_dir", default="data/wlasl/wlasl100_npy", help="Destination directory for the subset")
    parser.add_argument("--data_type", default="npy", choices=["npy", "mp4"], help="Type of source data files")
    parser.add_argument("--no_split", action="store_true", help="Place videos into <gloss>/ folders without train/val/test splitting")

    args = parser.parse_args()

    main(args.subset, args.json, args.src_dir, args.dst_dir, args.data_type, args.no_split)
