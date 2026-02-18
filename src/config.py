"""Project configuration and constants."""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
WESAD_DIR = DATA_DIR / "public" / "WESAD"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

for _d in [DATA_DIR / "public", DATA_DIR / "own", MODELS_DIR, RESULTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── WESAD dataset ─────────────────────────────────────────────────────────────
WESAD_SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
WESAD_LABELS = {
    0: "not defined", 1: "baseline", 2: "stress",
    3: "amusement", 4: "meditation",
}
WESAD_BINARY_MAP = {1: 0, 2: 1}   # baseline → relaxed(0), stress → stressed(1)
WESAD_KEEP_LABELS = [1, 2]
WESAD_LABEL_SR = 700               # Hz – labels at chest-device rate

# Empatica E4 wrist sampling rates (Hz)
SAMPLING_RATES = {"EDA": 4, "BVP": 64, "TEMP": 4, "ACC": 32}

# ── Signal processing ─────────────────────────────────────────────────────────
TARGET_SR = 4          # Hz – unified output rate for EDA / TEMP
WINDOW_SIZE = 60       # seconds
OVERLAP = 0.5
WINDOW_STEP = int(WINDOW_SIZE * (1 - OVERLAP))

# ── Classification ────────────────────────────────────────────────────────────
STRESS_LABELS = {0: "Relaxed", 1: "Stressed"}

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
