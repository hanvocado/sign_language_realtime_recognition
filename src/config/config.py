import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
CKPT_DIR = os.path.join(MODEL_DIR, "checkpoints")

DATASET_NAME = os.environ.get("DATASET_NAME", "wlasl").strip("/")

# ensure directories exist
for p in [DATA_DIR, MODEL_DIR, CKPT_DIR]:
    os.makedirs(p, exist_ok=True)

# Feature dimensions after extracting Face(468*3) + LHand(21*3) + RHand(21*3)
# FEATURE_DIM = 1530

# Feature dimensions after extracting Pose(33*3) + LHand(21*3) + RHand(21*3)
# Updated: removed face landmarks, added pose landmarks
FEATURE_DIM = 225  # 99 + 63 + 63 = 225
SEQ_LEN = int(os.environ.get("SEQ_LEN", "25"))

DATASET_DIR = os.path.join(DATA_DIR, DATASET_NAME)
RAW_DIR = os.path.join(DATASET_DIR, "videos", "raw")
PREPROCESSED_DIR = os.path.join(DATASET_DIR, "videos", "preprocessed")
STAGING_DIR = os.path.join(DATASET_DIR, "videos", "staging")
NPY_ROOT_DIR = os.path.join(DATASET_DIR, "npy")
NPY_DIR = os.path.join(NPY_ROOT_DIR, str(SEQ_LEN))
SPLIT_DIR = os.path.join(DATASET_DIR, "splits")

for p in [DATASET_DIR, RAW_DIR, PREPROCESSED_DIR, STAGING_DIR, NPY_ROOT_DIR, NPY_DIR, SPLIT_DIR]:
    os.makedirs(p, exist_ok=True)

BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 200
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_TYPE = 'gru'
INPUT_DIM = FEATURE_DIM
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.6
BIDIRECTIONAL = False