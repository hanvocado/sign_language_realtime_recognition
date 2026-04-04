"""
Web Application Configuration
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


class ModelConfig:
    """Model paths and parameters for Vietnamese Sign Language recognition"""
    
    MODELS_DIR = PROJECT_ROOT / 'models'
    CHECKPOINTS_DIR = MODELS_DIR / 'checkpoints'
    
    MODEL_PATH = CHECKPOINTS_DIR / 'vsl_v1' / 'best.pth'
    LABEL_MAP_PATH = CHECKPOINTS_DIR / 'vsl_v1' / 'label_map.json'
    
    @classmethod
    def get_active_model_paths(cls):
        """
        Get active model paths from production.json manifest (created by training DAG)
        Falls back to default paths if manifest not found
        """
        manifest_path = cls.CHECKPOINTS_DIR / 'production.json'
        
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                
                model_path = manifest.get('model_path', str(cls.MODEL_PATH))
                
                # Resolve relative paths for model_path
                if not Path(model_path).is_absolute():
                    model_path = PROJECT_ROOT / model_path
                model_path = Path(model_path)

                if not model_path.exists():
                    logger.warning(
                        f"⚠️ production.json references missing model file: {model_path}. "
                        "Falling back to default checkpoint."
                    )
                    return {
                        'model_path': cls.MODEL_PATH,
                        'label_map_path': cls.LABEL_MAP_PATH,
                        'production': False,
                    }

                # Determine label_map_path:
                # 1) from manifest if provided
                # 2) from the model directory (model_path.parent / 'label_map.json') if it exists
                # 3) fall back to the default LABEL_MAP_PATH
                label_map_path = manifest.get('label_map_path')
                if label_map_path and str(label_map_path).strip():
                    if not Path(label_map_path).is_absolute():
                        label_map_path = PROJECT_ROOT / label_map_path
                    label_map_path = Path(label_map_path)
                else:
                    candidate_label_map = model_path.parent / 'label_map.json'
                    if candidate_label_map.exists():
                        label_map_path = candidate_label_map
                    else:
                        label_map_path = cls.LABEL_MAP_PATH

                return {
                    'model_path': model_path,
                    'label_map_path': label_map_path,
                    'production': True,
                }
            except Exception as e:
                logger.warning(f"⚠️ Error reading production manifest: {e}")
                # Fall back to defaults
        
        # Default fallback
        return {
            'model_path': cls.MODEL_PATH,
            'label_map_path': cls.LABEL_MAP_PATH,
            'production': False,
        }


