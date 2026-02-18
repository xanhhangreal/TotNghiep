"""SHAP-based model interpretability for stress detection."""
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """Compute and visualize SHAP explanations."""

    def __init__(self, model, X_background: np.ndarray,
                 feature_names: list = None):
        self.model = model
        self.X_bg = X_background
        self.names = feature_names or [f"f{i}" for i in range(X_background.shape[1])]
        self.explainer = None
        self.shap_values = None
        self._X = None  # samples that were explained

    # ── explainer creation ────────────────────────────────────────────────────
    def create_explainer(self, method: str = "auto"):
        cls_name = self.model.__class__.__name__
        if method == "auto":
            if "Forest" in cls_name or "Tree" in cls_name:
                method = "tree"
            elif "Logistic" in cls_name or "Linear" in cls_name:
                method = "linear"
            else:
                method = "kernel"

        if method == "tree":
            self.explainer = shap.TreeExplainer(self.model)
        elif method == "linear":
            self.explainer = shap.LinearExplainer(self.model, self.X_bg)
        elif method == "kernel":
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, shap.sample(self.X_bg, 100))
        else:
            raise ValueError(f"Unknown SHAP method: {method}")

    def compute(self, X: np.ndarray):
        """Compute SHAP values for *X*."""
        if self.explainer is None:
            self.create_explainer()
        self.shap_values = self.explainer.shap_values(X)
        self._X = X

    # ── helpers ───────────────────────────────────────────────────────────────
    def _vals(self):
        """Return SHAP values for the positive class (stressed)."""
        sv = self.shap_values
        return sv[1] if isinstance(sv, list) else sv

    # ── plots ─────────────────────────────────────────────────────────────────
    def plot_summary(self, plot_type="bar", max_display=10, *,
                     save: str = None, show=True):
        plt.figure(figsize=(10, 6))
        shap.summary_plot(self._vals(), self._X,
                          feature_names=self.names,
                          plot_type=plot_type, max_display=max_display,
                          show=False)
        fig = plt.gcf()
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    def plot_importance(self, top_n=10, *, save: str = None, show=True):
        imp, names = self.feature_importance()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(names[:top_n][::-1], imp[:top_n][::-1])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"Top-{top_n} Feature Importance (SHAP)")
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    def plot_dependence(self, feature_idx=0, *, save: str = None):
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature_idx, self._vals(), self._X,
                             feature_names=self.names, show=False)
        fig = plt.gcf()
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=150, bbox_inches="tight")
        return fig

    # ── importance ranking ────────────────────────────────────────────────────
    def feature_importance(self) -> Tuple[np.ndarray, np.ndarray]:
        """Mean |SHAP| per feature, sorted descending."""
        sv = self._vals()
        imp = np.abs(sv).mean(axis=0)
        idx = np.argsort(imp)[::-1]
        return imp[idx], np.array(self.names)[idx]


# ── convenience pipeline ──────────────────────────────────────────────────────

def run_shap_analysis(model, X_train: np.ndarray, X_test: np.ndarray,
                      feature_names: list = None,
                      save_dir: str = None) -> Dict:
    """Full SHAP pipeline: compute values, plot summary + importance."""
    n_bg = min(200, len(X_train))
    sa = SHAPAnalyzer(model, X_train[:n_bg], feature_names)
    sa.create_explainer()
    sa.compute(X_test)

    imp, names = sa.feature_importance()
    result: Dict = {"analyzer": sa,
                    "top_features": names[:10].tolist(),
                    "top_importance": imp[:10].tolist()}

    if save_dir:
        d = Path(save_dir)
        d.mkdir(parents=True, exist_ok=True)
        sa.plot_summary("bar", save=str(d / "shap_bar.png"), show=False)
        sa.plot_summary("dot", save=str(d / "shap_dot.png"), show=False)
        sa.plot_importance(save=str(d / "shap_importance.png"), show=False)
        plt.close("all")

    return result
