"""
HIGGS dataset-specific management and preprocessing.
"""

from typing import Dict, Any, Callable
from mlpipe.blocks.ingest.csv_loader import UniversalCSVLoader


class HiggsDatasetManager:
    """Manager for HIGGS UCI dataset with specialized preprocessing."""
    
    def __init__(self):
        self.dataset_name = "higgs_uci"
    
    def get_config(self) -> Dict[str, Any]:
        """Get HIGGS-specific configuration."""
        return {
            "file_path": "data/HIGGS_100k.csv",
            "target_column": "label",
            "auto_download": True,
            "download_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz",
            "has_header": True,
            "nrows": 100000,
            "target_type": "binary",
            "positive_class_labels": [1],
            "missing_values": ["", "NA", "NaN", "null", "NULL", "-999"],
            "verbose": True
        }
    
    def get_features(self) -> list:
        """Get HIGGS dataset feature column names."""
        return [
            'lepton_pT', 'lepton_eta', 'lepton_phi', 'missing_energy_magnitude', 'missing_energy_phi',
            'jet_1_pt', 'jet_1_eta', 'jet_1_phi', 'jet_1_b_tag', 'jet_2_pt', 'jet_2_eta', 'jet_2_phi',
            'jet_2_b_tag', 'jet_3_pt', 'jet_3_eta', 'jet_3_phi', 'jet_3_b_tag', 'jet_4_pt', 'jet_4_eta',
            'jet_4_phi', 'jet_4_b_tag', 'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'
        ]
    
    def create_loader(self, **overrides) -> UniversalCSVLoader:
        """Create configured HIGGS data loader."""
        config = self.get_config()
        config.update(overrides)
        return UniversalCSVLoader(config)


def get_higgs_preprocessor() -> Callable:
    """Get HIGGS-specific preprocessor function."""
    
    def higgs_preprocessor(test_split_size=0.2, validation_split_size=0.1, random_state=42, **kwargs):
        """HIGGS-specific preprocessing pipeline."""
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        import joblib
        
        # Create HIGGS loader
        manager = HiggsDatasetManager()
        loader = manager.create_loader(**kwargs)
        
        # Load data
        X, y, metadata = loader.load()
        
        print(f"✅ HIGGS dataset loaded: {X.shape} features, {y.shape} targets")
        print(f"Task type: {metadata['target_info']['task_type']}")
        
        # Split data
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_split_size, random_state=random_state, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=validation_split_size / (1 - test_split_size),
            random_state=random_state, stratify=y_train_val
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler
        scaler_path = kwargs.get('scaler_path', 'models/scaler.joblib')
        joblib.dump(scaler, scaler_path)
        
        return {
            "X_train": X_train_scaled, "y_train": y_train.values,
            "X_val": X_val_scaled, "y_val": y_val.values,
            "X_test": X_test_scaled, "y_test": y_test.values
        }
    
    return higgs_preprocessor
