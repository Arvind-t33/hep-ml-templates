# preprocess_higgs_uci.py
# Template preprocessor for HIGGS UCI dataset using hep-ml-templates

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
# Import from local blocks - this will be modified during local installation
from blocks.ingest.csv_loader import UniversalCSVLoader

# HIGGS UCI Dataset Configuration
HIGGS_CONFIG = {
    "file_path": "data/HIGGS_100k.csv",
    "target_column": "label",
    "auto_download": True,
    "download_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz",
    "has_header": True,
    "nrows": 100000,  # Adjust as needed
    "target_type": "binary",
    "positive_class_labels": [1],
    "missing_values": ["", "NA", "NaN", "null", "NULL", "-999"],
    "verbose": True
}

def load_and_preprocess_data(config=None, test_size=0.2, val_size=0.1, random_state=42):
    """
    Load and preprocess HIGGS UCI dataset using hep-ml-templates.
    
    This function demonstrates the power of modular dataset loading:
    - No need for custom CSV parsing
    - Automatic data download and validation
    - Consistent preprocessing across different datasets
    
    Args:
        config (dict, optional): Custom dataset configuration. Uses HIGGS_CONFIG if None.
        test_size (float): Proportion for test set
        val_size (float): Proportion for validation set
        random_state (int): Random state for reproducibility
        
    Returns:
        dict: Dictionary with train/val/test splits and metadata
    """
    print("🚀 Loading HIGGS UCI dataset using hep-ml-templates...")
    
    # Use provided config or default
    dataset_config = config or HIGGS_CONFIG
    
    # Load data using the modular library
    loader = UniversalCSVLoader(dataset_config)
    X, y, metadata = loader.load()
    
    print(f"✅ Dataset loaded: {X.shape} features, {y.shape} targets")
    print(f"📊 Task type: {metadata['target_info']['task_type']}")
    print(f"🎯 Classes: {metadata['target_info']['classes']}")
    
    # The library already handles missing values and basic preprocessing!
    # This is the key benefit of the modular approach
    
    # Split data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size / (1 - test_size),
        random_state=random_state, stratify=y_train_val
    )

    print(f"📈 Data splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Feature scaling
    print("🔧 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for later use
    scaler_path = "models/scaler.joblib"
    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"💾 Scaler saved to {scaler_path}")

    return {
        "X_train": X_train_scaled, "y_train": y_train.values,
        "X_val": X_val_scaled, "y_val": y_val.values,
        "X_test": X_test_scaled, "y_test": y_test.values,
        "feature_names": list(X.columns),
        "metadata": metadata,
        "scaler": scaler
    }

# Example usage
if __name__ == "__main__":
    print("Testing HIGGS UCI preprocessor...")
    data = load_and_preprocess_data()
    print(f"✅ Preprocessing complete!")
    print(f"📊 Feature names: {data['feature_names'][:5]}...")
    print(f"🎯 Train target distribution: {np.bincount(data['y_train'].astype(int))}")
