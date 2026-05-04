"""ML model wrappers for stress classification."""
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_validate as sklearn_cv
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import logging

logger = logging.getLogger(__name__)

_CONSTRUCTORS = {
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
    "svm": lambda **kw: SVC(probability=True, **kw),
    "decision_tree": DecisionTreeClassifier,
    "adaboost": AdaBoostClassifier,
    "lda": LinearDiscriminantAnalysis,
    "knn": KNeighborsClassifier,
}


class StressModel:
    """Train, evaluate, save a single stress-detection classifier."""

    def __init__(self, model_type: str = "random_forest",
                 params: Optional[Dict] = None):
        if model_type not in _CONSTRUCTORS:
            raise ValueError(
                f"Unknown model: {model_type}. Choose from {list(_CONSTRUCTORS)}"
            )
        self.model_type = model_type
        self.params = params or {}
        self.model = _CONSTRUCTORS[model_type](**self.params)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        X, y = self._drop_invalid(X, y)
        if verbose:
            logger.info(
                "Training %s on %d samples (%d features)",
                self.model_type, len(X), X.shape[1]
            )
        self.scaler.fit(X)
        self.model.fit(self.scaler.transform(X), y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._sanitize_x(X)
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self._sanitize_x(X)
        return self.model.predict_proba(self.scaler.transform(X))

    def cross_validate(self, X: np.ndarray, y: np.ndarray,
                       cv: int = 5, verbose: bool = True) -> Dict:
        """K-fold CV using Pipeline(Scaler -> Model) to avoid data leakage."""
        X, y = self._drop_invalid(X, y)
        unique_classes = np.unique(y)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", _CONSTRUCTORS[self.model_type](**self.params)),
        ])
        scoring = {
            "accuracy": "accuracy",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
            "f1": "f1_weighted",
            "roc_auc": ("roc_auc" if len(unique_classes) == 2
                        else "roc_auc_ovr_weighted"),
        }
        res = sklearn_cv(
            pipe, X, y, cv=cv, scoring=scoring, return_train_score=True
        )
        if verbose:
            for metric_name in scoring:
                scores = res[f"test_{metric_name}"]
                logger.info(
                    "  %s: %.4f (+/-%.4f)",
                    metric_name.upper(), scores.mean(), scores.std()
                )
        return res

    def evaluate(self, X: np.ndarray, y: np.ndarray,
                 verbose: bool = True) -> Dict:
        X, y = self._drop_invalid(X, y)
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)
        unique_classes = np.unique(y)

        metrics: Dict = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(
                y, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(
                y, y_pred, average="weighted", zero_division=0
            ),
            "f1": f1_score(
                y, y_pred, average="weighted", zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y, y_pred),
            "report": classification_report(
                y, y_pred, output_dict=True, zero_division=0
            ),
        }

        if len(unique_classes) == 2:
            try:
                metrics["roc_auc"] = roc_auc_score(y, y_prob[:, 1])
                fpr, tpr, _ = roc_curve(y, y_prob[:, 1])
                metrics["roc_fpr"] = fpr
                metrics["roc_tpr"] = tpr
            except Exception:
                metrics["roc_auc"] = np.nan
        elif len(unique_classes) > 2:
            try:
                metrics["roc_auc"] = roc_auc_score(
                    y, y_prob, multi_class="ovr", average="weighted"
                )
            except Exception:
                metrics["roc_auc"] = np.nan

        if verbose:
            logger.info(
                "  Acc=%.4f  F1=%.4f  AUC=%.4f",
                metrics["accuracy"],
                metrics["f1"],
                metrics.get("roc_auc", float("nan")),
            )
        return metrics

    def feature_importance(self) -> Optional[np.ndarray]:
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        if hasattr(self.model, "coef_"):
            coef = self.model.coef_
            return np.abs(coef).mean(axis=0) if coef.ndim == 2 else np.abs(coef)
        return None

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "model_type": self.model_type,
            "params": self.params,
            "is_fitted": self.is_fitted,
        }, path)
        logger.info("Saved model -> %s", path)

    @classmethod
    def load(cls, path: str) -> "StressModel":
        obj_data = joblib.load(path)
        obj = cls(obj_data["model_type"], obj_data["params"])
        obj.model = obj_data["model"]
        obj.scaler = obj_data["scaler"]
        obj.is_fitted = obj_data["is_fitted"]
        return obj

    @staticmethod
    def _sanitize_x(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if np.isfinite(X).all():
            return X
        X = X.copy()
        X[~np.isfinite(X)] = 0.0
        return X

    @staticmethod
    def _drop_invalid(X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        valid_rows = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X_ok, y_ok = X[valid_rows], y[valid_rows]
        if len(X_ok) == 0:
            raise ValueError("No valid samples after removing NaN/Inf rows.")
        return X_ok, y_ok
