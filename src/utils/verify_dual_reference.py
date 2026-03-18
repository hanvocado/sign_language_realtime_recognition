"""
Kiểm chứng Dual-Reference Normalization Strategy
================================================

Script này chứng minh chiến lược chuẩn hóa tham chiếu kép cho nhận dạng ngôn ngữ ký hiệu
thông qua case study cụ thể: "MOTHER" vs "FATHER"

Giả thuyết:
-----------
1. WRIST-CENTRIC (Morphology): Hai từ có CÙNG handshape "5" 
   → Biểu diễn wrist-centric NÊN TƯƠNG TỰ
   
2. FACE-CENTRIC (Trajectory): Hai từ ở VỊ TRÍ KHÁC NHAU
   - MOTHER: chạm cằm (chin)
   - FATHER: chạm trán (forehead)
   → Biểu diễn face-centric NÊN KHÁC BIỆT

Usage:
------
    python verify_dual_reference.py --glosses "mother father" --output output.png
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from tqdm import tqdm


# ============================================================================
# MediaPipe Landmark Indices Reference
# ============================================================================
WRIST_INDICES = {
    'left': (45, 48),   # pose[15] × 3 → [45:48]
    'right': (48, 51),  # pose[16] × 3 → [48:51]
}

FACE_INDICES = {
    'nose': (0, 3),          # pose[0] × 3
    'mouth_left': (27, 30),  # pose[9] × 3
    'mouth_right': (30, 33), # pose[10] × 3
}

HAND_INDICES = {
    'left': (99, 162),   # 21 landmarks × 3 = 63
    'right': (162, 225), # 21 landmarks × 3 = 63
}


# ============================================================================
# Main Verifier Class
# ============================================================================
class DualReferenceVerifier:
    """Class chính để verify dual-reference normalization strategy"""
    
    def __init__(self, videos_dir: str, metadata_path: str):
        self.videos_dir = Path(videos_dir)
        self.metadata_path = Path(metadata_path)
        
        # Initialize MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        print("✓ DualReferenceVerifier initialized")
    
    def _load_metadata(self) -> List[Dict]:
        """Load WLASL metadata từ JSON file"""
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_path}\n"
                f"Please download WLASL_v0.3.json from:\n"
                f"https://github.com/dxli94/WLASL"
            )
        
        print(f"Loading metadata from: {self.metadata_path}")
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✓ Loaded {len(data)} glosses from metadata")
        return data
    
    def get_video_for_gloss(self, gloss: str, instance_id: int = 0) -> Optional[str]:
        """Tìm video_id cho gloss cụ thể"""
        gloss_lower = gloss.strip().lower()
        
        for entry in self.metadata:
            if entry['gloss'].lower() == gloss_lower:
                instances = entry.get('instances', [])
                
                if not instances:
                    print(f"⚠ Gloss '{gloss}' found but has no instances")
                    return None
                
                if instance_id >= len(instances):
                    print(f"⚠ Instance {instance_id} not found, using 0")
                    instance_id = 0
                
                video_id = instances[instance_id]['video_id']
                print(f"✓ Found '{gloss}': video_id={video_id} (instance {instance_id}/{len(instances)-1})")
                return video_id
        
        print(f"✗ Gloss '{gloss}' not found in metadata")
        return None
    
    def extract_landmarks(self, video_path: Path, max_frames: Optional[int] = None) -> Tuple[np.ndarray, List[int]]:
        """Extract MediaPipe Holistic landmarks từ video"""
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if max_frames is not None:
            total_frames = min(total_frames, max_frames)
        
        print(f"\nProcessing: {video_path.name}")
        print(f"  Total frames: {total_frames}, FPS: {fps:.2f}")
        
        landmarks_list = []
        valid_frames = []
        
        with tqdm(total=total_frames, desc="  Extracting", unit="frame") as pbar:
            frame_idx = 0
            
            while cap.isOpened() and (max_frames is None or frame_idx < max_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                # Process with MediaPipe
                results = self.holistic.process(image)
                
                # Extract keypoints
                landmarks = self._extract_keypoints(results)
                
                if landmarks is not None:
                    landmarks_list.append(landmarks)
                    valid_frames.append(frame_idx)
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        
        if not landmarks_list:
            raise ValueError(f"No valid landmarks extracted from {video_path.name}")
        
        landmarks_array = np.array(landmarks_list, dtype=np.float32)
        
        print(f"  ✓ Extracted {len(landmarks_array)}/{total_frames} frames "
              f"({100*len(landmarks_array)/total_frames:.1f}%)")
        
        return landmarks_array, valid_frames
    
    def _extract_keypoints(self, results) -> Optional[np.ndarray]:
        """Extract keypoints từ MediaPipe results"""
        pose_kp, left_hand_kp, right_hand_kp = [], [], []
        
        # Pose (33 × 3 = 99)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                pose_kp.extend([lm.x, lm.y, lm.z])
        else:
            pose_kp = [0.0] * 99
        
        # Left hand (21 × 3 = 63)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                left_hand_kp.extend([lm.x, lm.y, lm.z])
        else:
            left_hand_kp = [0.0] * 63
        
        # Right hand (21 × 3 = 63)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                right_hand_kp.extend([lm.x, lm.y, lm.z])
        else:
            right_hand_kp = [0.0] * 63
        
        # Require pose + ít nhất 1 hand
        if results.pose_landmarks and (results.left_hand_landmarks or results.right_hand_landmarks):
            return np.array(pose_kp + left_hand_kp + right_hand_kp, dtype=np.float32)
        
        return None
    
    def wrist_centric_normalization(self, sequence: np.ndarray) -> np.ndarray:
        """
        Chuẩn hóa theo cổ tay (Wrist-centric Morphological Frame)
        
        Returns: (T, 126) - Wrist-centered hand keypoints
        """
        # Extract wrists
        left_wrist = sequence[:, WRIST_INDICES['left'][0]:WRIST_INDICES['left'][1]]
        right_wrist = sequence[:, WRIST_INDICES['right'][0]:WRIST_INDICES['right'][1]]
        
        # Extract hands
        left_hand = sequence[:, HAND_INDICES['left'][0]:HAND_INDICES['left'][1]].reshape(-1, 21, 3)
        right_hand = sequence[:, HAND_INDICES['right'][0]:HAND_INDICES['right'][1]].reshape(-1, 21, 3)
        
        # Center at wrists
        left_hand_centered = left_hand - left_wrist[:, np.newaxis, :]
        right_hand_centered = right_hand - right_wrist[:, np.newaxis, :]
        
        # Concatenate
        morphology = np.concatenate([
            left_hand_centered.reshape(-1, 63),
            right_hand_centered.reshape(-1, 63)
        ], axis=1)

        return morphology

    def wrist_centric_normalization_scaled(self, sequence: np.ndarray) -> np.ndarray:
        """
        Chuẩn hóa theo cổ tay VỚI scale normalization

        Returns: (T, 126) - Wrist-centered and scale-normalized hand keypoints
        """
        # Extract wrists
        left_wrist = sequence[:, WRIST_INDICES['left'][0]:WRIST_INDICES['left'][1]]
        right_wrist = sequence[:, WRIST_INDICES['right'][0]:WRIST_INDICES['right'][1]]

        # Extract hands
        left_hand = sequence[:, HAND_INDICES['left'][0]:HAND_INDICES['left'][1]].reshape(-1, 21, 3)
        right_hand = sequence[:, HAND_INDICES['right'][0]:HAND_INDICES['right'][1]].reshape(-1, 21, 3)

        # Center at wrists
        left_hand_centered = left_hand - left_wrist[:, np.newaxis, :]
        right_hand_centered = right_hand - right_wrist[:, np.newaxis, :]

        # Compute hand scale (max distance from wrist to any finger tip)
        # Finger tips: indices 4, 8, 12, 16, 20
        finger_tips = [4, 8, 12, 16, 20]

        left_scale = np.max([np.linalg.norm(left_hand_centered[:, tip, :], axis=1) for tip in finger_tips], axis=0)
        right_scale = np.max([np.linalg.norm(right_hand_centered[:, tip, :], axis=1) for tip in finger_tips], axis=0)

        left_scale = np.where(left_scale < 1e-6, 1.0, left_scale)[:, np.newaxis, np.newaxis]
        right_scale = np.where(right_scale < 1e-6, 1.0, right_scale)[:, np.newaxis, np.newaxis]

        # Scale normalize
        left_hand_normalized = left_hand_centered / left_scale
        right_hand_normalized = right_hand_centered / right_scale

        # Concatenate
        morphology = np.concatenate([
            left_hand_normalized.reshape(-1, 63),
            right_hand_normalized.reshape(-1, 63)
        ], axis=1)

        return morphology

    def face_centric_normalization(self, sequence: np.ndarray) -> np.ndarray:
        """
        Chuẩn hóa theo khuôn mặt (Face-centric Semantic Frame)
        
        Returns: (T, 6) - Normalized wrist positions
        """
        # Extract wrists
        left_wrist = sequence[:, WRIST_INDICES['left'][0]:WRIST_INDICES['left'][1]]
        right_wrist = sequence[:, WRIST_INDICES['right'][0]:WRIST_INDICES['right'][1]]
        
        # Extract face points
        nose = sequence[:, FACE_INDICES['nose'][0]:FACE_INDICES['nose'][1]]
        mouth_left = sequence[:, FACE_INDICES['mouth_left'][0]:FACE_INDICES['mouth_left'][1]]
        mouth_right = sequence[:, FACE_INDICES['mouth_right'][0]:FACE_INDICES['mouth_right'][1]]
        
        # Face center
        face_center = (nose + mouth_left + mouth_right) / 3.0
        
        # Face scale (mouth width)
        face_scale = np.linalg.norm(mouth_right - mouth_left, axis=1, keepdims=True)
        face_scale = np.where(face_scale < 1e-6, 1.0, face_scale)
        
        # Normalize
        left_wrist_norm = (left_wrist - face_center) / face_scale
        right_wrist_norm = (right_wrist - face_center) / face_scale
        
        trajectory = np.concatenate([left_wrist_norm, right_wrist_norm], axis=1)

        return trajectory

    def face_centric_normalization_unscaled(self, sequence: np.ndarray) -> np.ndarray:
        """
        Chuẩn hóa theo khuôn mặt KHÔNG scale (chỉ centering)

        Returns: (T, 6) - Face-centered wrist positions (không scale)
        """
        # Extract wrists
        left_wrist = sequence[:, WRIST_INDICES['left'][0]:WRIST_INDICES['left'][1]]
        right_wrist = sequence[:, WRIST_INDICES['right'][0]:WRIST_INDICES['right'][1]]

        # Extract face points
        nose = sequence[:, FACE_INDICES['nose'][0]:FACE_INDICES['nose'][1]]
        mouth_left = sequence[:, FACE_INDICES['mouth_left'][0]:FACE_INDICES['mouth_left'][1]]
        mouth_right = sequence[:, FACE_INDICES['mouth_right'][0]:FACE_INDICES['mouth_right'][1]]

        # Face center
        face_center = (nose + mouth_left + mouth_right) / 3.0

        # Center only (no scaling)
        left_wrist_centered = left_wrist - face_center
        right_wrist_centered = right_wrist - face_center

        trajectory = np.concatenate([left_wrist_centered, right_wrist_centered], axis=1)

        return trajectory

    def compute_metrics(self, seq1: np.ndarray, seq2: np.ndarray) -> Dict[str, float]:
        """Tính similarity metrics"""
        # Resample
        T_min = min(len(seq1), len(seq2))
        
        if len(seq1) != T_min:
            indices = np.linspace(0, len(seq1) - 1, T_min).astype(int)
            seq1 = seq1[indices]
        if len(seq2) != T_min:
            indices = np.linspace(0, len(seq2) - 1, T_min).astype(int)
            seq2 = seq2[indices]
        
        # L2 distance
        l2_distances = np.linalg.norm(seq1 - seq2, axis=1)
        mean_l2 = float(np.mean(l2_distances))
        
        # Cosine similarity
        seq1_flat = seq1.flatten()
        seq2_flat = seq2.flatten()
        
        dot = np.dot(seq1_flat, seq2_flat)
        norm1 = np.linalg.norm(seq1_flat)
        norm2 = np.linalg.norm(seq2_flat)
        cosine_sim = float(dot / (norm1 * norm2 + 1e-8))
        
        # Correlation
        correlation = float(np.corrcoef(seq1_flat, seq2_flat)[0, 1])
        
        return {
            'mean_l2': mean_l2,
            'cosine_sim': cosine_sim,
            'correlation': correlation
        }
    
    def visualize_results(
        self,
        gloss1: str,
        gloss2: str,
        video_id1: str,
        video_id2: str,
        morph1: np.ndarray,
        morph2: np.ndarray,
        morph1_scaled: np.ndarray,
        morph2_scaled: np.ndarray,
        traj1_scaled: np.ndarray,
        traj2_scaled: np.ndarray,
        traj1_unscaled: np.ndarray,
        traj2_unscaled: np.ndarray,
        output_path: str = "dual_reference_verification.png"
    ):
        """Tạo visualization với 4 plots (2x2)"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        color1, color2 = '#2E86AB', '#A23B72'

        # ===== ROW 1: MORPHOLOGY (Wrist-centric) =====
        # Left: Without scale normalization
        ax1 = axes[0, 0]
        morph1_mag = np.linalg.norm(morph1, axis=1)
        morph2_mag = np.linalg.norm(morph2, axis=1)

        ax1.plot(morph1_mag, color=color1, linewidth=2, label=gloss1.upper(), alpha=0.8)
        ax1.plot(morph2_mag, color=color2, linewidth=2, label=gloss2.upper(), alpha=0.8, linestyle='--')
        ax1.set_xlabel('Frame', fontweight='bold')
        ax1.set_ylabel('||Morphology||', fontweight='bold')
        ax1.set_title('Morphology (no scale)', fontweight='bold', pad=10)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Right: With scale normalization
        ax2 = axes[0, 1]
        morph1_scaled_mag = np.linalg.norm(morph1_scaled, axis=1)
        morph2_scaled_mag = np.linalg.norm(morph2_scaled, axis=1)

        ax2.plot(morph1_scaled_mag, color=color1, linewidth=2, label=gloss1.upper(), alpha=0.8)
        ax2.plot(morph2_scaled_mag, color=color2, linewidth=2, label=gloss2.upper(), alpha=0.8, linestyle='--')
        ax2.set_xlabel('Frame', fontweight='bold')
        ax2.set_ylabel('||Morphology||', fontweight='bold')
        ax2.set_title('Morphology (with scale)', fontweight='bold', pad=10)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # ===== ROW 2: TRAJECTORY (Face-centric) =====
        # Left: Without scale normalization
        ax3 = axes[1, 0]
        traj1_unscaled_y = traj1_unscaled[:, 4]  # Right wrist Y
        traj2_unscaled_y = traj2_unscaled[:, 4]

        ax3.plot(traj1_unscaled_y, color=color1, linewidth=2, label=gloss1.upper(), alpha=0.8)
        ax3.plot(traj2_unscaled_y, color=color2, linewidth=2, label=gloss2.upper(), alpha=0.8, linestyle='--')
        ax3.axhline(y=0, color='black', linestyle=':', alpha=0.5, label='Face center')
        ax3.set_xlabel('Frame', fontweight='bold')
        ax3.set_ylabel('Wrist Y (raw)', fontweight='bold')
        ax3.set_title('Trajectory (no scale)', fontweight='bold', pad=10)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Right: With scale normalization
        ax4 = axes[1, 1]
        traj1_scaled_y = traj1_scaled[:, 4]  # Right wrist Y
        traj2_scaled_y = traj2_scaled[:, 4]

        ax4.plot(traj1_scaled_y, color=color1, linewidth=2, label=gloss1.upper(), alpha=0.8)
        ax4.plot(traj2_scaled_y, color=color2, linewidth=2, label=gloss2.upper(), alpha=0.8, linestyle='--')
        ax4.axhline(y=0, color='black', linestyle=':', alpha=0.5, label='Face center')
        ax4.set_xlabel('Frame', fontweight='bold')
        ax4.set_ylabel('Wrist Y (normalized)', fontweight='bold')
        ax4.set_title('Trajectory (with scale)', fontweight='bold', pad=10)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Set same y-axis scale for all 4 plots
        all_values = np.concatenate([
            morph1_mag, morph2_mag, morph1_scaled_mag, morph2_scaled_mag,
            traj1_scaled_y, traj2_scaled_y, traj1_unscaled_y, traj2_unscaled_y
        ])
        y_min, y_max = all_values.min(), all_values.max()
        y_margin = (y_max - y_min) * 0.05
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_ylim(y_min - y_margin, y_max + y_margin)

        fig.suptitle(f'{gloss1.upper()} ({video_id1}) vs {gloss2.upper()} ({video_id2})', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved visualization: {output_path}")

        # Compute metrics for return value
        morph_metrics = self.compute_metrics(morph1, morph2)
        traj_metrics = self.compute_metrics(traj1_scaled, traj2_scaled)
        morph_pass = morph_metrics['correlation'] > 0.6
        traj_pass = traj_metrics['correlation'] < 0.7
        overall_pass = morph_pass and traj_pass

        return morph_metrics, traj_metrics, overall_pass


def main():
    parser = argparse.ArgumentParser(
        description='Verify dual-reference normalization for SLR'
    )
    parser.add_argument('--glosses', type=str, default='mother father',
                       help='Space-separated glosses (default: "mother father")')
    parser.add_argument('--videos', type=str, default='data/wlasl/videos',
                       help='Videos directory')
    parser.add_argument('--metadata', type=str, default='data/wlasl/WLASL_v0.3.json',
                       help='Metadata JSON path')
    parser.add_argument('--output', type=str, default='dual_reference_verification.png',
                       help='Output visualization path')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Max frames to process')
    
    args = parser.parse_args()
    
    glosses = args.glosses.strip().split()
    
    if len(glosses) != 2:
        print(f"Error: Expected 2 glosses, got {len(glosses)}")
        return 1
    
    gloss1, gloss2 = glosses
    
    print("\n" + "="*80)
    print("DUAL-REFERENCE NORMALIZATION VERIFICATION")
    print("="*80)
    print(f"Case Study: {gloss1.upper()} vs {gloss2.upper()}")
    print("="*80 + "\n")
    
    try:
        verifier = DualReferenceVerifier(args.videos, args.metadata)
        
        # Get videos
        print("-"*80)
        print("STEP 1: Finding videos")
        print("-"*80)
        
        video_id1 = verifier.get_video_for_gloss(gloss1)
        video_id2 = verifier.get_video_for_gloss(gloss2)
        
        if not video_id1 or not video_id2:
            print("\n✗ Error: Missing video IDs")
            return 1
        
        video_path1 = verifier.videos_dir / f"{video_id1}.mp4"
        video_path2 = verifier.videos_dir / f"{video_id2}.mp4"
        
        # Extract
        print("\n" + "-"*80)
        print("STEP 2: Extracting landmarks")
        print("-"*80)
        
        landmarks1, _ = verifier.extract_landmarks(video_path1, args.max_frames)
        landmarks2, _ = verifier.extract_landmarks(video_path2, args.max_frames)
        
        # Normalize
        print("\n" + "-"*80)
        print("STEP 3: Applying normalization")
        print("-"*80)

        # Wrist-centric (morphology)
        morph1 = verifier.wrist_centric_normalization(landmarks1)
        morph2 = verifier.wrist_centric_normalization(landmarks2)
        print(f"  ✓ Morphology (no scale): {gloss1}={morph1.shape}, {gloss2}={morph2.shape}")

        morph1_scaled = verifier.wrist_centric_normalization_scaled(landmarks1)
        morph2_scaled = verifier.wrist_centric_normalization_scaled(landmarks2)
        print(f"  ✓ Morphology (with scale): {gloss1}={morph1_scaled.shape}, {gloss2}={morph2_scaled.shape}")

        # Face-centric (trajectory)
        traj1_scaled = verifier.face_centric_normalization(landmarks1)
        traj2_scaled = verifier.face_centric_normalization(landmarks2)
        print(f"  ✓ Trajectory (with scale): {gloss1}={traj1_scaled.shape}, {gloss2}={traj2_scaled.shape}")

        traj1_unscaled = verifier.face_centric_normalization_unscaled(landmarks1)
        traj2_unscaled = verifier.face_centric_normalization_unscaled(landmarks2)
        print(f"  ✓ Trajectory (no scale): {gloss1}={traj1_unscaled.shape}, {gloss2}={traj2_unscaled.shape}")

        # Visualize
        print("\n" + "-"*80)
        print("STEP 4: Generating visualization")
        print("-"*80)

        verifier.visualize_results(
            gloss1, gloss2,
            video_id1, video_id2,
            morph1, morph2,
            morph1_scaled, morph2_scaled,
            traj1_scaled, traj2_scaled,
            traj1_unscaled, traj2_unscaled,
            args.output
        )
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())