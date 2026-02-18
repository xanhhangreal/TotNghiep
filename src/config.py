"""
Configuration and constants for Stress Detection project
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PUBLIC_DATA_DIR = DATA_DIR / "public"
OWN_DATA_DIR = DATA_DIR / "own"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for dir_path in [PUBLIC_DATA_DIR, OWN_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Physiological signals sampling rates (Hz)
SAMPLING_RATES = {
    "eda": 4,      # EDA/GSR - 4 Hz for consumer devices
    "bvp": 64,     # Blood Volume Pulse - 64 Hz
    "temp": 4,     # Skin Temperature - 4 Hz
    "heart_rate": 1,  # Derived heart rate - 1 Hz
}

# Feature extraction parameters
WINDOW_SIZE = 60  # seconds - temporal window for feature extraction
WINDOW_SHIFT = 0.25  # seconds - shift step
OVERLAP = 0.5  # 50% overlap

# Stress labels
STRESS_LABELS = {
    0: "Relaxed",
    1: "Stressed"
}

# Model hyperparameters
MODEL_PARAMS = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
    },
    "logistic_regression": {
        "max_iter": 1000,
        "random_state": 42,
        "solver": "lbfgs",
    },
    "svm": {
        "kernel": "rbf",
        "C": 1.0,
        "random_state": 42,
    }
}

# Cross-validation
CV_FOLDS = 5
RANDOM_STATE = 42

# SHAP analysis
SHAP_SAMPLES = "auto"  # Number of samples to use for SHAP
SHAP_MAX_DISPLAY = 10  # Top N features to display

# Dataset URLs & info
WESAD_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00465/WESAD.zip"
AFFECTIVEROAD_URL = "https://dataverse.harvard.edu/api/access/datafile/4127260"

# Evaluation metrics
METRICS = ["accuracy", "precision", "recall", "f1", "auc_roc", "confusion_matrix"]

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
