<<<<<<< HEAD
"""Project configuration and constants."""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
WESAD_DIR = DATA_DIR / "WESAD"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

for _d in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── WESAD dataset ─────────────────────────────────────────────────────────────
WESAD_SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
WESAD_LABELS = {
    0: "not defined", 1: "baseline", 2: "stress",
    3: "amusement", 4: "meditation",
}
WESAD_BINARY_MAP = {1: 0, 2: 1}       # baseline → relaxed(0), stress → stressed(1)
WESAD_KEEP_LABELS = [1, 2]

WESAD_3CLASS_MAP = {1: 0, 2: 1, 3: 2}  # baseline→0, stress→1, amusement→2
WESAD_KEEP_LABELS_3CLASS = [1, 2, 3]

WESAD_LABEL_SR = 700                    # Hz – labels at chest-device rate

# Empatica E4 wrist sampling rates (Hz)
SAMPLING_RATES = {"EDA": 4, "BVP": 64, "TEMP": 4, "ACC": 32}

# RespiBAN chest sampling rates (Hz) – all sensors at 700 Hz
CHEST_SAMPLING_RATES = {
    "ECG": 700, "EMG": 700, "EDA": 700,
    "Temp": 700, "Resp": 700, "ACC": 700,
}

# ── Signal processing ─────────────────────────────────────────────────────────
TARGET_SR = 4          # Hz – unified output rate for EDA / TEMP
WINDOW_SIZE = 60       # seconds
OVERLAP = 0.5
WINDOW_STEP = int(WINDOW_SIZE * (1 - OVERLAP))

# ── Classification ────────────────────────────────────────────────────────────
STRESS_LABELS = {0: "Relaxed", 1: "Stressed"}
STRESS_LABELS_3CLASS = {0: "Baseline", 1: "Stress", 2: "Amusement"}

DEFAULT_MODELS = {
    "random_forest": {
        "n_estimators": 100, "max_depth": 15,
        "min_samples_split": 5, "min_samples_leaf": 2, "random_state": 42,
    },
    "logistic_regression": {
        "max_iter": 1000, "solver": "lbfgs", "random_state": 42,
    },
    "svm": {
        "kernel": "rbf", "C": 1.0, "gamma": "scale", "random_state": 42,
    },
    "decision_tree": {
        "max_depth": 10, "min_samples_split": 5, "random_state": 42,
    },
}

CV_FOLDS = 5
RANDOM_STATE = 42

# ── Deep Learning ─────────────────────────────────────────────────────────────
DL_BATCH_SIZE = 64
DL_EPOCHS = 100
DL_LEARNING_RATE = 1e-3
DL_WEIGHT_DECAY = 1e-4
DL_PATIENCE = 10          # early-stopping patience
DL_LR_PATIENCE = 5        # ReduceLROnPlateau patience
DL_MIN_LR = 1e-6
DL_DROPOUT = 0.3
DL_MODELS = ["cnn1d", "unet1d", "resnet1d"]
=======
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
>>>>>>> 8c60043252f731287d32a5c7796a7892fc60cb19
