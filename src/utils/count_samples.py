import os
from collections import defaultdict
import argparse
from src.utils.logger import *

def main(dir, extension):
    counts = defaultdict(int)

    # Walk through label folders
    for label in sorted(os.listdir(dir)):
        label_dir = os.path.join(dir, label)

        if not os.path.isdir(label_dir):
            continue  # skip non-folders

        # Count files
        num_videos = len([f for f in os.listdir(label_dir) if f.endswith(extension)])
        counts[label] = num_videos

    for lbl, num in counts.items():
        logger.info(f"{lbl}: {num}")
    
    logger.info(f"Total labels: {len(counts)}")
    logger.info(f"Total videos: {sum(counts.values())}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/raw")
    parser.add_argument("--extension", default="npy")
    args = parser.parse_args()
    dir = args.dir
    extension = args.extension

    logger = setup_logger(f"count_samples_{dir.replace('/', '_')}")
    log_arguments(logger=logger, args=args)

    main(dir, extension)
