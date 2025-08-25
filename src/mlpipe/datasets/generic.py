"""
Generic dataset preprocessing functions.
"""

from typing import Callable


def get_generic_preprocessor(dataset_name: str) -> Callable:
    """Get a generic preprocessor for any dataset."""
    
    def generic_preprocessor(test_split_size=0.2, validation_split_size=0.1, random_state=42, **kwargs):
        """Generic preprocessing pipeline that works for most datasets."""
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from mlpipe.datasets.registry import create_dataset_loader
        import joblib
        
        # Create dataset loader
        loader = create_dataset_loader(dataset_name, **kwargs)
        
        # Load data
        X, y, metadata = loader.load()
        
        print(f"✅ {dataset_name} dataset loaded: {X.shape} features, {y.shape} targets")
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
    
    return generic_preprocessor
