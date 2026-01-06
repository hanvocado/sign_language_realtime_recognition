"""
Web Application Configuration
"""

from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


class ModelConfig:
    """Model paths and parameters for Vietnamese Sign Language recognition"""
    
    MODELS_DIR = PROJECT_ROOT / 'models'
    CHECKPOINTS_DIR = MODELS_DIR / 'checkpoints'
    
    MODEL_PATH = CHECKPOINTS_DIR / 'vsl_v1' / 'best.pth'
    LABEL_MAP_PATH = CHECKPOINTS_DIR / 'vsl_v1' / 'label_map.json'  

