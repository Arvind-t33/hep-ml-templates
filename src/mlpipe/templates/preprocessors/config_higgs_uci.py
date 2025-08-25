# config_higgs_uci.py
# Configuration template for HIGGS UCI dataset

import torch

# Dataset-specific settings for HIGGS UCI
DATA_FILE = "data/HIGGS_100k.csv"
MODEL_PATH = "models/best_model.ckpt"
SCALER_PATH = "models/scaler.joblib"
OUTPUT_DIR = "outputs"

# HIGGS UCI dataset features (28 kinematic features)
FEATURES = [
    'lepton_pT', 'lepton_eta', 'lepton_phi', 'missing_energy_magnitude', 'missing_energy_phi',
    'jet_1_pt', 'jet_1_eta', 'jet_1_phi', 'jet_1_b_tag', 'jet_2_pt', 'jet_2_eta', 'jet_2_phi',
    'jet_2_b_tag', 'jet_3_pt', 'jet_3_eta', 'jet_3_phi', 'jet_3_b_tag', 'jet_4_pt', 'jet_4_eta',
    'jet_4_phi', 'jet_4_b_tag', 'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'
]
LABEL_COLUMN = "label"
MISSING_VALUE = -999.0

# Library configuration for UniversalCSVLoader
HIGGS_LOADER_CONFIG = {
    "file_path": DATA_FILE,
    "target_column": LABEL_COLUMN,
    "auto_download": True,
    "download_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz",
    "has_header": True,
    "nrows": 100000,  # Adjust based on your computational resources
    "target_type": "binary",
    "positive_class_labels": [1],
    "missing_values": ["", "NA", "NaN", "null", "NULL", "-999"],
    "verbose": True
}

# Model hyperparameters (adjusted for HIGGS features)
INPUT_SIZE = len(FEATURES)  # 28 features
HIDDEN_LAYERS = [256, 128, 64]  # Can be adjusted
OUTPUT_SIZE = 1
LEARNING_RATE = 1e-4
DROPOUT_PROB = 0.2

# Training parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1024
EPOCHS = 10
TEST_SPLIT_SIZE = 0.2
VALIDATION_SPLIT_SIZE = 0.1
RANDOM_STATE = 42
