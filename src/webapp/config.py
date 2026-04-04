"""
Web Application Configuration
"""

import json
from pathlib import Path

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
                
                # Resolve relative paths
                if not Path(model_path).is_absolute():
                    model_path = PROJECT_ROOT / model_path
                
                return {
                    'model_path': Path(model_path),
                    'label_map_path': cls.LABEL_MAP_PATH,
                    'production': True,
                }
            except Exception as e:
                print(f"⚠️ Error reading production manifest: {e}")
                # Fall back to defaults
        
        # Default fallback
        return {
            'model_path': cls.MODEL_PATH,
            'label_map_path': cls.LABEL_MAP_PATH,
            'production': False,
        }


