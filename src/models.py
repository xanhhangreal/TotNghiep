"""ML model wrappers for binary stress classification."""
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Optional

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate as sklearn_cv
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve,
)
import logging

logger = logging.getLogger(__name__)

_CONSTRUCTORS = {
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
    "svm": lambda **kw: SVC(probability=True, **kw),
    "decision_tree": DecisionTreeClassifier,
}


class StressModel:
    """Train / evaluate / save a single stress-detection classifier."""

    def __init__(self, model_type: str = "random_forest", params: Dict = None):
        if model_type not in _CONSTRUCTORS:
            raise ValueError(f"Unknown model: {model_type}. Choose from {list(_CONSTRUCTORS)}")
        self.model_type = model_type
        self.params = params or {}
        self.model = _CONSTRUCTORS[model_type](**self.params)
        self.scaler = StandardScaler()
        self.is_fitted = False

    # ── training ──────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        X, y = self._drop_nan(X, y)
        if verbose:
            logger.info("Training %s on %d samples (%d features)",
                        self.model_type, len(X), X.shape[1])
        self.scaler.fit(X)
        self.model.fit(self.scaler.transform(X), y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(X))

    # ── cross-validation (leak-free) ─────────────────────────────────────────
    def cross_validate(self, X: np.ndarray, y: np.ndarray,
                       cv: int = 5, verbose: bool = True) -> Dict:
        """K-fold CV using Pipeline(Scaler → Model) to avoid data leakage."""
        X, y = self._drop_nan(X, y)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", _CONSTRUCTORS[self.model_type](**self.params)),
        ])
        scoring = {
            "accuracy": "accuracy",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
            "f1": "f1_weighted",
            "roc_auc": "roc_auc",
        }
        res = sklearn_cv(pipe, X, y, cv=cv, scoring=scoring, return_train_score=True)
        if verbose:
            for m in scoring:
                t = res[f"test_{m}"]
                logger.info("  %s: %.4f (±%.4f)", m.upper(), t.mean(), t.std())
        return res

    # ── evaluation ────────────────────────────────────────────────────────────
    def evaluate(self, X: np.ndarray, y: np.ndarray,
                 verbose: bool = True) -> Dict:
        yp = self.predict(X)
        ypr = self.predict_proba(X)
        m: Dict = {
            "accuracy": accuracy_score(y, yp),
            "precision": precision_score(y, yp, average="weighted", zero_division=0),
            "recall": recall_score(y, yp, average="weighted", zero_division=0),
            "f1": f1_score(y, yp, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(y, yp),
            "report": classification_report(y, yp, output_dict=True),
        }
        if len(np.unique(y)) == 2:
            try:
                m["roc_auc"] = roc_auc_score(y, ypr[:, 1])
                fpr, tpr, _ = roc_curve(y, ypr[:, 1])
                m["roc_fpr"], m["roc_tpr"] = fpr, tpr
            except Exception:
                m["roc_auc"] = np.nan
        if verbose:
            logger.info("  Acc=%.4f  F1=%.4f  AUC=%.4f",
                        m["accuracy"], m["f1"], m.get("roc_auc", float("nan")))
        return m

    # ── feature importance ────────────────────────────────────────────────────
    def feature_importance(self) -> Optional[np.ndarray]:
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        if hasattr(self.model, "coef_"):
            return np.abs(self.model.coef_[0])
        return None

    # ── persistence ───────────────────────────────────────────────────────────
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model, "scaler": self.scaler,
            "model_type": self.model_type, "params": self.params,
            "is_fitted": self.is_fitted,
        }, path)
        logger.info("Saved model → %s", path)

    @classmethod
    def load(cls, path: str) -> "StressModel":
        d = joblib.load(path)
        obj = cls(d["model_type"], d["params"])
        obj.model, obj.scaler, obj.is_fitted = d["model"], d["scaler"], d["is_fitted"]
        return obj

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _drop_nan(X, y):
        ok = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        return X[ok], y[ok]
