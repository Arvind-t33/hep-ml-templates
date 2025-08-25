"""
Modular ML pipeline blocks.

Import blocks individually to avoid unnecessary dependencies:
    from mlpipe.blocks.ingest.csv_loader import UniversalCSVLoader
    from mlpipe.blocks.preprocessing.standard_scaler import StandardScalerBlock
    from mlpipe.blocks.model.xgb_classifier import XGBClassifierBlock  # requires xgboost
"""

# Function to register commonly used blocks for testing/demo purposes
def register_all_available_blocks():
    """
    Register all available blocks. Use this for testing or when you want
    to use the registry-based block access pattern.
    
    This function only imports blocks that have their dependencies available.
    """
    try:
        from . import ingest
        from .ingest import csv_loader
    except ImportError:
        pass
    
    try:
        from . import preprocessing
        from .preprocessing import standard_scaler
    except ImportError:
        pass
    
    try:
        from . import feature_eng
        from .feature_eng import column_selector
    except ImportError:
        pass
    
    try:
        from . import training
        from .training import sklearn_trainer
    except ImportError:
        pass
    
    try:
        from . import evaluation
        from .evaluation import classification_metrics
    except ImportError:
        pass
    
    # Optional blocks that require specific dependencies
    try:
        from . import model
        from .model import xgb_classifier
    except ImportError:
        pass  # XGBoost not available

# For backward compatibility, you can call register_all_available_blocks()
# But users should prefer importing blocks directly for better modularity
