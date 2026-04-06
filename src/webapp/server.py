"""
Vietnamese Sign Language Recognition - Web App (PROPERLY OPTIMIZED)

KEY IMPROVEMENTS:
1. Dedicated MediaPipe thread (thread-safe keypoint extraction)
2. Sliding buffer (10 frames) for real-time response
3. Batch inference every 150ms
4. NO blocking in SocketIO handler
5. Confidence voting + smoothing
6. Non-blocking frame queue
"""

import os
import sys
import base64
import json
import threading
import time
import re
import numpy as np
import torch
import cv2
import logging
from pathlib import Path
from collections import deque
import queue

# Flask
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename

# MinIO
from minio import Minio
from minio.error import S3Error

# MediaPipe
import mediapipe as mp

# Project imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.model.train import build_model
from src.utils.utils import load_label_map, load_checkpoint
from src.utils.common_functions import extract_keypoints, normalize_keypoints, sample_frames
from src.config.config import (
    DEVICE, MODEL_TYPE, INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, BIDIRECTIONAL, SEQ_LEN
)
from src.webapp.config import ModelConfig

# Suppress MediaPipe warnings
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# ===================================================================
# CONFIGURATION
# ===================================================================

BUFFER_SIZE = 15  # Reduced from 15 - faster buffer fill (480ms @ 25FPS)
MIN_CONFIDENCE = 0.52  # Slightly higher to filter borderline predictions
INFERENCE_INTERVAL = 0.15  # Run inference every 150ms
SMOOTHING_WINDOW = 4  # Voting window size
MIN_VOTES_FOR_RESULT = 3  # Increased from 2 to 3 - need stronger consensus (75% in window of 4)

# Upload pipeline config
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "data")
MINIO_UPLOAD_PREFIX = os.environ.get(
    "MINIO_UPLOAD_PREFIX", "lakehouse/bronze/wlasl/videos/user_uploads"
).strip("/")
MAX_UPLOAD_FILES = int(os.environ.get("MAX_UPLOAD_FILES", "30"))
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# ===================================================================
# FLASK APP
# ===================================================================

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', logger=False)

# ===================================================================
# LOGGING
# ===================================================================

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/webapp.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===================================================================
# GLOBAL STATE
# ===================================================================

model = None
label_list = None
minio_client = None

# MediaPipe (single instance in dedicated thread)
mp_holistic = mp.solutions.holistic
holistic = None

# Frame processing
frame_queue = queue.Queue(maxsize=2)  # Only keep 1-2 frames pending
frame_buffer = deque(maxlen=BUFFER_SIZE)
prediction_history = deque(maxlen=SMOOTHING_WINDOW)

# Inference state
last_inference_time = 0
is_inferring = False
inference_lock = threading.Lock()

latest_prediction = {
    'label': 'Waiting...',
    'confidence': 0.0,
    'timestamp': 0
}

last_emitted_label = None  
last_emit_time = 0.0
DUPLICATE_PREVENTION_TIMEOUT = 0.5  
inference_blocked_until = 0  


def _normalize_minio_endpoint(endpoint: str) -> str:
    if endpoint.startswith("http://"):
        return endpoint.replace("http://", "", 1)
    if endpoint.startswith("https://"):
        return endpoint.replace("https://", "", 1)
    return endpoint


def _sanitize_label(label: str) -> str:
    value = (label or "").strip()
    value = re.sub(r"[/\\]+", "_", value)
    value = re.sub(r"\s+", " ", value)
    return value


def _label_folder(label: str) -> str:
    return _sanitize_label(label).replace(" ", "_")


def _allowed_video(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_VIDEO_EXTENSIONS


def init_minio_client():
    global minio_client

    raw_endpoint = os.environ.get("MINIO_ENDPOINT", "minio:9000")
    endpoint = _normalize_minio_endpoint(raw_endpoint)
    access_key = os.environ.get("MINIO_ACCESS_KEY") or os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")
    secret_key = os.environ.get("MINIO_SECRET_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin")

    minio_secure_env = os.environ.get("MINIO_SECURE")
    if minio_secure_env is not None:
        secure = minio_secure_env.lower() in ("1", "true", "yes")
    else:
        secure = raw_endpoint.startswith("https://")

    minio_client = Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure,
    )

    if not minio_client.bucket_exists(MINIO_BUCKET):
        minio_client.make_bucket(MINIO_BUCKET)

    logger.info(
        f"✅ MinIO ready (bucket={MINIO_BUCKET}, prefix={MINIO_UPLOAD_PREFIX}, endpoint={endpoint})"
    )


def list_available_labels():
    labels = set(label_list or [])

    if minio_client is not None:
        try:
            prefix = f"{MINIO_UPLOAD_PREFIX}/"
            for obj in minio_client.list_objects(MINIO_BUCKET, prefix=prefix, recursive=True):
                rel = obj.object_name[len(prefix):]
                parts = rel.split("/")
                # Preferred layout: <yyyymm>/<yyyymmdd>/<label>/<file>
                if len(parts) >= 4 and parts[2]:
                    labels.add(parts[2].replace("_", " "))
                # Backward compatible layout with hour: <yyyymm>/<yyyymmdd>/<label>/<hh>/<file>
                elif len(parts) >= 5 and parts[2]:
                    labels.add(parts[2].replace("_", " "))
                # Backward compatible layout: <yyyymm>/<label>/<file>
                elif len(parts) >= 3 and parts[1]:
                    labels.add(parts[1].replace("_", " "))
                # Backward compatibility: <label>/<file>
                elif len(parts) >= 2 and parts[0]:
                    labels.add(parts[0].replace("_", " "))
        except Exception as exc:
            logger.warning(f"Cannot read labels from MinIO uploads: {exc}")

    return sorted(labels)

# ===================================================================
# MODEL LOADING
# ===================================================================

def load_model_and_weights(model_path, label_map_path):
    """Load PyTorch model and label map"""
    logger.info(f"Loading model from {model_path}")
    
    label_list = load_label_map(label_map_path)
    
    model = build_model(
        MODEL_TYPE,
        INPUT_DIM,
        HIDDEN_DIM,
        len(label_list),
        NUM_LAYERS,
        DROPOUT,
        BIDIRECTIONAL,
    ).to(DEVICE)
    
    ck = load_checkpoint(model_path, device=DEVICE)
    model.load_state_dict(ck['model_state'], strict=False)
    model.eval()
    
    logger.info(f"✅ Model loaded: {len(label_list)} classes")
    logger.info(f"   Classes: {label_list}")
    
    return model, label_list

# ===================================================================
# MEDIAPIPE PROCESSING (Dedicated Thread)
# ===================================================================

def process_frame_mediapipe(frame_bgr):
    """Extract keypoints using MediaPipe"""
    try:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        keypoints = extract_keypoints(results)
        return keypoints
    except Exception as e:
        logger.error(f"MediaPipe error: {str(e)}")
        return None

def mediapipe_worker():
    """Background thread that processes frames and extracts keypoints"""
    global frame_queue, frame_buffer
    
    while True:
        try:
            # Wait for frame with timeout
            frame_bgr = frame_queue.get(timeout=1)
            
            # Extract keypoints
            keypoints = process_frame_mediapipe(frame_bgr)
            
            if keypoints is not None:
                frame_buffer.append(keypoints)
        
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Worker error: {str(e)}")

# ===================================================================
# INFERENCE
# ===================================================================

def run_inference():
    """Run model inference on buffered frames"""
    global is_inferring, frame_buffer, prediction_history, latest_prediction, last_emitted_label, last_emit_time, inference_blocked_until
    
    is_inferring = True
    
    try:
        # Skip inference if blocked (after recent emission)
        current_time = time.time()
        if current_time < inference_blocked_until:
            is_inferring = False
            logger.debug(f"Inference blocked for {inference_blocked_until - current_time:.2f}s")
            return
        
        if len(frame_buffer) < BUFFER_SIZE:
            is_inferring = False
            return
        
        # Get frames from buffer
        frames_array = np.array(list(frame_buffer), dtype=np.float32)
        
        # Normalize keypoints
        frames_array = normalize_keypoints(frames_array)
        
        # Sample to SEQ_LEN (25)
        indices = sample_frames(len(frames_array), SEQ_LEN, mode="1")
        sampled = frames_array[indices]
        
        # Inference
        X = torch.from_numpy(sampled).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits = model(X)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        pred_idx = int(probs.argmax())
        pred_label = label_list[pred_idx]
        pred_conf = float(probs[pred_idx])
        
        # Log top predictions
        top3 = np.argsort(probs)[-3:][::-1]
        top3_str = " | ".join([f"{label_list[i]}: {probs[i]:.3f}" for i in top3])
        logger.info(f"🔍 Raw: {pred_label} ({pred_conf:.3f}) | Top3: {top3_str}")
        
        # Smoothing: vote over last predictions (STRICT VOTING)
        if pred_conf >= MIN_CONFIDENCE:
            prediction_history.append((pred_label, pred_conf))
            
            # If we have enough votes for decision
            if len(prediction_history) >= MIN_VOTES_FOR_RESULT:
                # Get most common label
                labels = [l for l, c in prediction_history]
                counts = {}
                for lbl in labels:
                    counts[lbl] = counts.get(lbl, 0) + 1
                
                voted_label = max(counts, key=counts.get)
                vote_count = counts[voted_label]
                
                if vote_count >= MIN_VOTES_FOR_RESULT:
                    current_time = time.time()
                    is_duplicate = (voted_label == last_emitted_label and 
                                  (current_time - last_emit_time) < DUPLICATE_PREVENTION_TIMEOUT)
                    
                    if not is_duplicate:
                        # Get average confidence for voted label
                        voted_confs = [c for l, c in prediction_history if l == voted_label]
                        avg_conf = np.mean(voted_confs)
                        
                        logger.info(f"VOTED: {voted_label} ({avg_conf:.3f}) | Votes: {vote_count}/{len(prediction_history)}")
                        
                        with inference_lock:
                            latest_prediction['label'] = voted_label
                            latest_prediction['confidence'] = avg_conf
                            latest_prediction['timestamp'] = current_time
                        
                        # Emit prediction
                        socketio.emit('prediction', {
                            'label': voted_label,
                            'confidence': float(avg_conf),
                            'votes': vote_count,
                            'buffer_size': len(frame_buffer),
                        })
                        
                        # Update last emitted info
                        last_emitted_label = voted_label
                        last_emit_time = current_time
                        
                        # Block inference for 1 second to prevent spam from continuous buffer updates
                        inference_blocked_until = current_time + 1.0
                        
                        # Clear BOTH history and buffer to prevent spam from repeated inference
                        prediction_history.clear()
                        frame_buffer.clear()
                        logger.info(f"RESULT EMITTED - Blocked inference for 1.0s to prevent spam")
                    else:
                        # Suppress duplicate
                        logger.debug(f"Duplicate suppressed: {voted_label} (time since last: {current_time - last_emit_time:.2f}s)")
                        prediction_history.clear()
                else:
                    logger.debug(f"⏳ Waiting for consensus: {voted_label} has {vote_count}/{MIN_VOTES_FOR_RESULT} votes")
        else:
            # Low confidence - clear history
            prediction_history.clear()
            logger.info(f"⚠️  Low confidence ({pred_conf:.3f})")
    
    except Exception as e:
        logger.error(f"❌ Inference error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        is_inferring = False

# ===================================================================
# SOCKET.IO HANDLERS
# ===================================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload-options', methods=['GET'])
def upload_options():
    return jsonify({
        'labels': list_available_labels(),
        'max_upload_files': MAX_UPLOAD_FILES,
        'allowed_extensions': sorted(ALLOWED_VIDEO_EXTENSIONS),
    })


@app.route('/api/upload-videos', methods=['POST'])
def upload_videos():
    if minio_client is None:
        return jsonify({'ok': False, 'message': 'MinIO is not ready'}), 503

    selected_label = request.form.get('selected_label', '')
    new_label = request.form.get('new_label', '')
    label = _sanitize_label(new_label if new_label.strip() else selected_label)
    label_folder = _label_folder(label)

    if not label:
        return jsonify({'ok': False, 'message': 'Label is required'}), 400

    files = request.files.getlist('videos')
    if not files:
        return jsonify({'ok': False, 'message': 'No files were uploaded'}), 400

    if len(files) > MAX_UPLOAD_FILES:
        return jsonify({
            'ok': False,
            'message': f'Too many files. Maximum is {MAX_UPLOAD_FILES} files per request.',
        }), 400

    uploaded = []
    skipped = []
    failed = []

    upload_month = time.strftime("%Y%m")
    upload_day = time.strftime("%Y%m%d")

    for file_storage in files:
        original_name = file_storage.filename or ''
        safe_name = secure_filename(original_name)
        suffix = Path(safe_name).suffix.lower()

        if not safe_name:
            failed.append({'file': original_name, 'reason': 'Invalid file name'})
            continue
        if not _allowed_video(safe_name):
            skipped.append({'file': original_name, 'reason': 'Unsupported extension'})
            continue

        try:
            file_storage.stream.seek(0, 2)
            size = file_storage.stream.tell()
            if size == 0:
                failed.append({'file': original_name, 'reason': 'Empty file'})
                continue

            file_storage.stream.seek(0)
            object_name = (
                f"{MINIO_UPLOAD_PREFIX}/{upload_month}/{upload_day}/{label_folder}/"
                f"{int(time.time() * 1000)}_{safe_name}"
            )

            minio_client.put_object(
                MINIO_BUCKET,
                object_name,
                data=file_storage.stream,
                length=size,
                content_type=file_storage.content_type or 'video/mp4',
            )

            uploaded.append(
                {
                    'file': original_name,
                    'object_name': object_name,
                }
            )
        except S3Error as exc:
            failed.append({'file': original_name, 'reason': str(exc)})
        except Exception as exc:
            failed.append({'file': original_name, 'reason': str(exc)})

    status_code = 200 if uploaded else 400
    return jsonify({
        'ok': bool(uploaded),
        'label': label,
        'uploaded_count': len(uploaded),
        'skipped_count': len(skipped),
        'failed_count': len(failed),
        'uploaded': uploaded,
        'skipped': skipped,
        'failed': failed,
    }), status_code

@socketio.on('connect')
def handle_connect():
    logger.info("✅ Client connected")
    emit('connect_response', {'data': 'Connected'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info("❌ Client disconnected")

@socketio.on('frame')
def handle_frame(data):
    """Receive frame from client - add to queue for processing"""
    global last_inference_time, is_inferring, frame_buffer
    
    if model is None:
        logger.warning("Model not loaded yet, skipping frame")
        return
    
    try:
        # Decode image
        img_data = base64.b64decode(data['image'].split(',')[1])
        frame_bgr = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        
        if frame_bgr is None:
            logger.warning("Failed to decode frame")
            return
        
        # Debug log every 10 frames
        frame_num = data.get('frame_num', 0)
        if frame_num % 10 == 0:
            logger.info(f"📤 Received frame #{frame_num}, buffer_size={len(frame_buffer)}")
        
        # Add to queue (non-blocking - drops if queue full)
        try:
            frame_queue.put_nowait(frame_bgr)
        except queue.Full:
            pass  # Drop frame if queue full
        
        # Trigger inference if buffer full
        current_time = time.time()
        if (len(frame_buffer) >= BUFFER_SIZE and 
            not is_inferring and 
            current_time - last_inference_time >= INFERENCE_INTERVAL):
            
            last_inference_time = current_time
            logger.info(f"🔄 Buffer full ({len(frame_buffer)} frames), triggering inference...")
            
            # Run inference in background
            thread = threading.Thread(target=run_inference)
            thread.daemon = True
            thread.start()
        
        # Send status every 10 frames
        if frame_num % 10 == 0:
            socketio.emit('status', {
                'buffer_size': len(frame_buffer),
                'is_ready': len(frame_buffer) == BUFFER_SIZE,
                'is_inferring': is_inferring,
                'last_prediction': latest_prediction['label'],
            }, skip_sid=True)
    
    except Exception as e:
        logger.error(f"❌ Frame error: {str(e)}")

@socketio.on('reset')
def handle_reset():
    """Clear buffer"""
    global frame_buffer, prediction_history
    frame_buffer.clear()
    prediction_history.clear()
    logger.info("🔄 Buffer cleared")
    emit('reset_response', {'status': 'cleared'})

# ===================================================================
# INITIALIZATION
# ===================================================================

def initialize_app():
    """Initialize app"""
    global model, label_list, holistic, minio_client
    
    logger.info("=" * 70)
    logger.info("WEBAPP STARTING - Vietnamese Sign Language Recognition")
    logger.info("=" * 70)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Buffer: {BUFFER_SIZE} | Confidence: {MIN_CONFIDENCE} | Smoothing: {SMOOTHING_WINDOW}")
    
    # Load model from production manifest or fallback
    config = ModelConfig.get_active_model_paths()
    model_path = str(config['model_path'])
    label_map_path = str(config['label_map_path'])
    
    if config['production']:
        logger.info("✅ Using production model from manifest (production.json)")
    else:
        logger.warning("⚠️ Using fallback model path (no production manifest found)")
    
    model, label_list = load_model_and_weights(model_path, label_map_path)

    # Initialize MinIO upload client for user videos
    try:
        init_minio_client()
    except Exception as exc:
        logger.warning(f"⚠️ MinIO unavailable at startup (uploads disabled): {exc}")
        minio_client = None
    
    # Initialize MediaPipe (shared instance)
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False,
    )
    
    # Start MediaPipe worker thread
    worker_thread = threading.Thread(target=mediapipe_worker, daemon=True)
    worker_thread.start()
    logger.info("✅ MediaPipe worker started")
    
    logger.info("🚀 Server ready at http://0.0.0.0:5000")

# ===================================================================
# MAIN
# ===================================================================

if __name__ == '__main__':
    initialize_app()
    dev_mode = os.environ.get("FLASK_DEV", "false").lower() == "true"
    allow_unsafe_werkzeug = os.environ.get("ALLOW_UNSAFE_WERKZEUG", str(dev_mode).lower()).lower() == "true"
    socketio.run(
        app,
        host='0.0.0.0',
        port=5000,
        debug=dev_mode,
        allow_unsafe_werkzeug=allow_unsafe_werkzeug,
    )
