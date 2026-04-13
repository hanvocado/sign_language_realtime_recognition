"""
Preprocess raw sign language videos:
- Convert fps -> 30
- Resize to 1280x720 (16:9)
- Pixel value normalization to [0, 1] range using min-max normalization
- Detect motion and only keep segments with significant movement
- Optionally split long videos into smaller clips (2-5s)
"""

import os
import glob
import cv2
import numpy as np
from pathlib import Path
import argparse
from src.utils.logger import *
from src.config.config import RAW_DIR, PREPROCESSED_DIR

def detect_motion(frame1, frame2, threshold=25, min_area=500):
    """
    Detect motion between two consecutive frames.
    
    Args:
        frame1: Previous frame (grayscale)
        frame2: Current frame (grayscale)
        threshold: Pixel difference threshold
        min_area: Minimum contour area to consider as motion
    
    Returns:
        bool: True if significant motion detected
    """
    # Compute absolute difference
    diff = cv2.absdiff(frame1, frame2)
    
    # Threshold the difference
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Dilate to fill gaps
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if any contour is large enough
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            return True
    
    return False


def preprocess_video(input_path, output_path, fps=30, width=1280, height=720, 
                     max_duration=5, motion_threshold=25, min_motion_area=500,
                     min_motion_frames=5, motion_buffer=10, skip_existing=False):
    """
    Preprocess video with motion detection to crop only active segments.
    
    Args:
        motion_threshold: Pixel difference threshold for motion detection
        min_motion_area: Minimum contour area to consider as motion
        min_motion_frames: Minimum consecutive frames with motion to start recording
        motion_buffer: Number of frames to keep before/after motion (padding)
    """
    output_dir = os.path.dirname(output_path)
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    existing_segments = glob.glob(os.path.join(output_dir, f"{base_name}_*.mp4"))

    if skip_existing and existing_segments:
        logger.info(f"⏭️  Skip {input_path}: existing preprocessed segments found")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"❌ Cannot open {input_path}")
        return

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare output directory
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Motion detection state
    prev_gray = None
    motion_count = 0
    is_recording = False
    frame_buffer = []
    seg_idx = 0
    out = None
    frames_written = 0
    segment_len = int(fps * max_duration) if max_duration > 0 else float('inf')

    print(f"🔍 Processing {input_path} with motion detection...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize
        frame = cv2.resize(frame, (width, height))
        
        # Convert to grayscale for motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect motion
        has_motion = False
        if prev_gray is not None:
            has_motion = detect_motion(prev_gray, gray, motion_threshold, min_motion_area)
        
        prev_gray = gray.copy()
        
        # Update motion counter
        if has_motion:
            motion_count += 1
        else:
            motion_count = max(0, motion_count - 1)
        
        # Start recording if enough consecutive motion frames
        if not is_recording and motion_count >= min_motion_frames:
            is_recording = True
            # Add buffered frames before motion started
            buffered_start = max(0, len(frame_buffer) - motion_buffer)
            for buffered_frame in frame_buffer[buffered_start:]:
                if out is None or frames_written >= segment_len:
                    if out is not None:
                        out.release()
                        frames_written = 0
                    seg_name = os.path.splitext(os.path.basename(output_path))[0] + f"_{seg_idx}.mp4"
                    seg_path = os.path.join(output_dir, seg_name)
                    out = cv2.VideoWriter(seg_path, fourcc, fps, (width, height))
                    seg_idx += 1
                
                out.write(buffered_frame)
                frames_written += 1
            
            frame_buffer.clear()
        
        # Stop recording if motion has stopped
        if is_recording and motion_count == 0:
            is_recording = False
            # Add a few more frames after motion stops (buffer)
            for _ in range(motion_buffer):
                ret_buf, frame_buf = cap.read()
                if not ret_buf:
                    break
                frame_buf = cv2.resize(frame_buf, (width, height))
                normalized_frame = normalize_frame(frame_buf)
                if out is not None:
                    out.write(normalized_frame)
                    frames_written += 1
            continue
        
        # Normalize pixel values
        normalized_frame = normalize_frame(frame)
        
        # Write or buffer the frame
        if is_recording:
            if out is None or frames_written >= segment_len:
                if out is not None:
                    out.release()
                    frames_written = 0
                seg_name = os.path.splitext(os.path.basename(output_path))[0] + f"_{seg_idx}.mp4"
                seg_path = os.path.join(output_dir, seg_name)
                out = cv2.VideoWriter(seg_path, fourcc, fps, (width, height))
                seg_idx += 1
            
            out.write(normalized_frame)
            frames_written += 1
        else:
            # Keep a rolling buffer of recent frames
            frame_buffer.append(normalized_frame)
            if len(frame_buffer) > motion_buffer * 2:
                frame_buffer.pop(0)

    if out is not None:
        out.release()
    cap.release()
    
    if seg_idx > 0:
        logger.info(f"✅ Preprocessed {input_path} -> {seg_idx} segment(s) in {os.path.dirname(output_path)}")
    else:
        logger.warning(f"⚠️  No motion detected in {input_path}, no output created")


def normalize_frame(frame):
    """
    Normalize pixel values to [0, 1] range using min-max normalization,
    then convert back to uint8 for video writing.
    """
    frame_min = frame.min()
    frame_max = frame.max()
    
    if frame_max > frame_min:
        frame = (frame - frame_min) / (frame_max - frame_min)
    else:
        frame = frame.astype(np.float32) / 255.0
    
    return (frame * 255).astype(np.uint8)


def batch_preprocess(input_dir, output_dir, fps=30, width=1280, height=720, 
                     max_duration=5, motion_threshold=25, min_motion_area=500,
                     min_motion_frames=5, motion_buffer=10, skip_existing=False):
    for label in sorted(os.listdir(input_dir)):
        in_label_dir = os.path.join(input_dir, label)
        if not os.path.isdir(in_label_dir):
            continue
        out_label_dir = os.path.join(output_dir, label)
        Path(out_label_dir).mkdir(parents=True, exist_ok=True)
        
        for f in sorted(os.listdir(in_label_dir)):
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                in_path = os.path.join(in_label_dir, f)
                out_path = os.path.join(out_label_dir, f)
                preprocess_video(in_path, out_path, fps=fps, width=width, height=height,
                               max_duration=max_duration, motion_threshold=motion_threshold,
                               min_motion_area=min_motion_area, min_motion_frames=min_motion_frames,
                               motion_buffer=motion_buffer, skip_existing=skip_existing)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=RAW_DIR, help="Thư mục chứa video gốc")
    parser.add_argument("--output_dir", default=PREPROCESSED_DIR, help="Thư mục lưu video đã chuẩn hóa")
    parser.add_argument("--fps", type=int, default=30, help="Tần số khung hình sau chuẩn hóa")
    parser.add_argument("--width", type=int, default=1280, help="Chiều rộng video output")
    parser.add_argument("--height", type=int, default=720, help="Chiều cao video output")
    parser.add_argument("--max_duration", type=int, default=10, help="Thời lượng tối đa mỗi clip (giây), 0 = không cắt")
    parser.add_argument("--motion_threshold", type=int, default=25, help="Ngưỡng khác biệt pixel để phát hiện chuyển động")
    parser.add_argument("--min_motion_area", type=int, default=100, help="Diện tích tối thiểu của vùng chuyển động")
    parser.add_argument("--min_motion_frames", type=int, default=5, help="Số frame liên tiếp có chuyển động để bắt đầu ghi")
    parser.add_argument("--motion_buffer", type=int, default=10, help="Số frame buffer trước/sau chuyển động")
    parser.add_argument("--skip_existing", action="store_true", help="Bỏ qua video nếu đã có segment output")
    
    args = parser.parse_args()

    logger = setup_logger("preprocess_video")
    log_arguments(logger=logger, args=args)

    batch_preprocess(args.input_dir, args.output_dir,
                     fps=args.fps, width=args.width, height=args.height,
                     max_duration=args.max_duration, motion_threshold=args.motion_threshold,
                     min_motion_area=args.min_motion_area, min_motion_frames=args.min_motion_frames,
                     motion_buffer=args.motion_buffer, skip_existing=args.skip_existing)