"""
Convert videos to .npy sequences of keypoints (pose + hands) with 
frame sampling (no padding/truncate).

Sampling:
- Apply frame sampling, ensure uniform coverage of the signing motion, regardless of video length.
- Always returns exactly seq_len frames.

Supports any directory structure:
- Flat: input_dir/video.mp4 → output_dir/video.npy
- Nested: input_dir/label/video.mp4 → output_dir/label/video.npy
- Deeply nested: input_dir/a/b/c/video.mp4 → output_dir/a/b/c/video.npy
"""

import os, cv2, numpy as np, argparse
from pathlib import Path
import mediapipe as mp
from src.config.config import SEQ_LEN, PREPROCESSED_DIR, NPY_DIR
from src.utils.logger import *
from src.utils.common_functions import *

mp_holistic = mp.solutions.holistic

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

# =====================================================
# Main conversion function
# =====================================================

def convert_video_to_npy(video_path, output_path, seq_len=SEQ_LEN,
                         sampling_mode="2", skip_existing=False):

    if skip_existing and os.path.exists(output_path):
        logger.info(f"Skip {video_path}: already exists")
        return "skipped"

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total == 0:
        logger.warning(f"Skip {video_path}: cannot read frames (total=0)")
        cap.release()
        return "failed"

    # sample frame indices
    indices = sample_frames(total, seq_len, mode=sampling_mode)

    seq = []
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                seq.append(np.zeros(225, dtype=np.float32))
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(image)
            seq.append(extract_keypoints(res))

    cap.release()
    arr = np.stack(seq).astype(np.float32)        

    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    np.save(output_path, arr)
    logger.info(f"Saved {output_path}, shape={arr.shape}")
    return "converted"


def replace_video_extension(filename):
    """Replace video extension with .npy"""
    for ext in VIDEO_EXTENSIONS:
        if filename.lower().endswith(ext):
            return filename[:-len(ext)] + ".npy"
    return filename + ".npy"


def is_video_file(filename):
    """Check if file is a video based on extension"""
    return filename.lower().endswith(VIDEO_EXTENSIONS)


def find_all_videos(input_dir):
    """
    Recursively find all video files in input_dir.
    
    Returns:
        List of tuples: (absolute_path, relative_path)
        - absolute_path: full path to video file
        - relative_path: path relative to input_dir (preserves directory structure)
    """
    videos = []
    input_dir = os.path.abspath(input_dir)
    
    for root, dirs, files in os.walk(input_dir):
        # Sort for consistent ordering
        dirs.sort()
        files.sort()
        
        for f in files:
            if is_video_file(f):
                abs_path = os.path.join(root, f)
                rel_path = os.path.relpath(abs_path, input_dir)
                videos.append((abs_path, rel_path))
    
    return videos


def get_directory_structure_info(input_dir):
    """Analyze and describe the directory structure"""
    max_depth = 0
    total_dirs = 0
    
    for root, dirs, files in os.walk(input_dir):
        depth = root.replace(input_dir, '').count(os.sep)
        max_depth = max(max_depth, depth)
        total_dirs += len(dirs)
    
    if max_depth == 0:
        return "flat (no subdirectories)"
    elif max_depth == 1:
        return f"nested (1 level, {total_dirs} subdirectories)"
    else:
        return f"deeply nested ({max_depth} levels, {total_dirs} subdirectories)"


def batch_convert(input_dir, output_dir, seq_len=SEQ_LEN, sampling_mode="2", skip_existing=False):
    """
    Convert all videos in input_dir to .npy files in output_dir.
    
    Preserves directory structure:
    - input_dir/video.mp4 → output_dir/video.npy
    - input_dir/a/video.mp4 → output_dir/a/video.npy
    - input_dir/a/b/c/video.mp4 → output_dir/a/b/c/video.npy
    """
    
    if not os.path.exists(input_dir):
        logger.error(f"❌ Input directory does not exist: {input_dir}")
        return
    
    # Analyze directory structure
    structure = get_directory_structure_info(input_dir)
    logger.info(f"📁 Directory structure: {structure}")
    
    # Find all videos recursively
    videos = find_all_videos(input_dir)
    
    if len(videos) == 0:
        logger.warning(f"⚠️ No video files found in {input_dir}")
        logger.info(f"   Supported extensions: {VIDEO_EXTENSIONS}")
        return
    
    logger.info(f"🎬 Found {len(videos)} video files")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Convert each video
    converted = 0
    skipped = 0
    failed = 0
    
    for i, (abs_path, rel_path) in enumerate(videos, 1):
        # Build output path preserving directory structure
        out_rel_path = replace_video_extension(rel_path)
        out_abs_path = os.path.join(output_dir, out_rel_path)
        
        logger.info(f"[{i}/{len(videos)}] Processing: {rel_path}")
        
        result = convert_video_to_npy(
            abs_path, out_abs_path,
            seq_len=seq_len,
            sampling_mode=sampling_mode,
            skip_existing=skip_existing
        )
        
        if result == "converted":
            converted += 1
        elif result == "skipped":
            skipped += 1
        else:
            failed += 1
    
    # Summary
    logger.info("=" * 50)
    logger.info(f"📊 Summary:")
    logger.info(f"   Total videos: {len(videos)}")
    logger.info(f"   Converted: {converted}")
    if skipped > 0:
        logger.info(f"   Skipped (already exists): {skipped}")
    if failed > 0:
        logger.info(f"   Failed: {failed}")
    logger.info(f"   Output directory: {output_dir}")
    logger.info("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert videos to .npy keypoint sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Directory Structure Support:
  The script automatically handles any directory structure:
  
  Flat:
    input/video1.mp4        → output/video1.npy
    input/video2.mp4        → output/video2.npy
  
  Nested (1 level - typical for labeled data):
    input/hello/vid1.mp4    → output/hello/vid1.npy
    input/thanks/vid2.mp4   → output/thanks/vid2.npy
  
  Deeply nested (multiple levels):
    input/train/hello/v1.mp4  → output/train/hello/v1.npy
    input/test/thanks/v2.mp4  → output/test/thanks/v2.npy

Examples:
  # Basic usage
    python -m src.preprocess.video2npy --input_dir data/wlasl/videos/preprocessed --output_dir data/wlasl/npy/25

  # With custom sequence length
    python -m src.preprocess.video2npy --input_dir data/wlasl/videos/preprocessed --output_dir data/wlasl/npy/30 --seq_len 30

  # Skip already converted files
    python -m src.preprocess.video2npy --input_dir data/wlasl/videos/preprocessed --output_dir data/wlasl/npy/25 --skip_existing
        """
    )
    parser.add_argument("--input_dir", default=PREPROCESSED_DIR, 
                        help="Input directory containing videos (any structure)")
    parser.add_argument("--output_dir", default=NPY_DIR, 
                        help="Output directory for .npy files (structure preserved)")
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN, 
                        help=f"Sequence length (default: {SEQ_LEN})")
    parser.add_argument("--sampling_mode", default="2", choices=["1", "2"], 
                        help="Sampling mode: 1=uniform, 2=smart (default: 2)")
    parser.add_argument("--skip_existing", action="store_true", 
                        help="Skip already converted files")
    args = parser.parse_args()

    logger = setup_logger("video2npy")
    log_arguments(logger, args)

    batch_convert(
        args.input_dir,
        args.output_dir,
        seq_len=args.seq_len,
        sampling_mode=args.sampling_mode,
        skip_existing=args.skip_existing
    )