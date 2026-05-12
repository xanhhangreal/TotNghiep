"""Build tracked benchmark artifacts from JSON results.

Outputs to ``results_summary/``:
  - ml_binary_loso_summary.csv
  - ml_3class_loso_summary.csv
  - dl_binary_loso_summary.csv
  - dl_3class_loso_summary.csv
  - final_benchmark_summary.csv
  - final_benchmark_summary.md
  - figures/model_comparison_binary.png
  - figures/model_comparison_3class.png
  - figures/confusion_matrix_rf_binary.png
  - figures/confusion_matrix_resnet_3class.png
  - figures/shap_top_features.png
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from ml_models import StressModel
from training import extract_subject_features
from wesad_loader import load_wesad

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
OUT_DIR = ROOT_DIR / "results_summary"
FIG_DIR = OUT_DIR / "figures"


@dataclass
class SourceFiles:
    final_summary_csv: Path
    rf_subject_independent_json: Path
    resnet_3cls_loso_json: Path


def _read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_latest_subject_independent_with_rf_cm(results_dir: Path) -> Path:
    files = sorted(results_dir.glob("subject_independent_*.json"))
    files.reverse()
    for p in files:
        try:
            data = _read_json(p)
            cm = data.get("models", {}).get("random_forest", {}).get("cm")
            if not cm:
                continue
            arr = np.asarray(cm)
            if arr.shape == (2, 2):
                return p
        except Exception:
            continue
    raise FileNotFoundError(
        "No subject_independent_*.json with random_forest 2x2 confusion matrix found."
    )


def _resolve_sources(results_dir: Path) -> SourceFiles:
    final_summary_csv = results_dir / "final_benchmark_summary.csv"
    if not final_summary_csv.exists():
        raise FileNotFoundError(f"Missing file: {final_summary_csv}")

    df = pd.read_csv(final_summary_csv)
    row = df[
        (df["Task"] == "Tri-class (3-class)")
        & (df["Family"] == "DL")
        & (df["Model"] == "ResNet1D")
    ]
    if row.empty:
        raise ValueError("Could not find DL ResNet1D 3-class row in final summary CSV.")
    resnet_3cls_loso_json = results_dir / str(row.iloc[0]["SourceFile"])
    if not resnet_3cls_loso_json.exists():
        raise FileNotFoundError(f"Missing file: {resnet_3cls_loso_json}")

    rf_subject_independent_json = _find_latest_subject_independent_with_rf_cm(results_dir)
    return SourceFiles(
        final_summary_csv=final_summary_csv,
        rf_subject_independent_json=rf_subject_independent_json,
        resnet_3cls_loso_json=resnet_3cls_loso_json,
    )


def _write_subset_csvs(df: pd.DataFrame, out_dir: Path) -> None:
    mapping = {
        ("Binary (2-class)", "ML"): "ml_binary_loso_summary.csv",
        ("Tri-class (3-class)", "ML"): "ml_3class_loso_summary.csv",
        ("Binary (2-class)", "DL"): "dl_binary_loso_summary.csv",
        ("Tri-class (3-class)", "DL"): "dl_3class_loso_summary.csv",
    }
    cols = [
        "Task",
        "Family",
        "Model",
        "Accuracy_mean",
        "Accuracy_std",
        "F1_mean",
        "F1_std",
        "Accuracy_mean_std",
        "F1_mean_std",
        "SourceFile",
    ]
    for (task, family), filename in mapping.items():
        part = df[(df["Task"] == task) & (df["Family"] == family)].copy()
        part = part.sort_values(by="F1_mean", ascending=False)
        part.to_csv(out_dir / filename, index=False, columns=cols)


def _write_final_markdown(df: pd.DataFrame, out_path: Path) -> None:
    lines: List[str] = ["# Final Benchmark Summary (WESAD, LOSO 15 subjects)", ""]
    for task in ("Binary (2-class)", "Tri-class (3-class)"):
        lines.append(f"## {task}")
        lines.append("| Family | Model | Accuracy (mean ± std) | F1 (mean ± std) |")
        lines.append("|---|---|---:|---:|")
        part = df[df["Task"] == task].sort_values(by="F1_mean", ascending=False)
        for _, row in part.iterrows():
            lines.append(
                f"| {row['Family']} | {row['Model']} | "
                f"{row['Accuracy_mean_std']} | {row['F1_mean_std']} |"
            )
        lines.append("")

    lines.append("## Source files used")
    lines.append("")
    for _, row in df.iterrows():
        lines.append(f"- {row['Task']} / {row['Family']} / {row['Model']}: {row['SourceFile']}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _plot_comparison(df: pd.DataFrame, task: str, out_path: Path) -> None:
    part = df[df["Task"] == task].copy().sort_values(by="F1_mean", ascending=False)
    labels = [f"{fam}-{mdl}" for fam, mdl in zip(part["Family"], part["Model"])]
    x = np.arange(len(part))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, part["Accuracy_mean"], width=width, label="Accuracy", color="#2563eb")
    ax.bar(x + width / 2, part["F1_mean"], width=width, label="F1", color="#059669")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title(f"Model Comparison ({task})")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_cm(cm: np.ndarray, labels: List[str], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _aggregate_resnet_3class_cm(path: Path) -> np.ndarray:
    data = _read_json(path)
    cms = []
    for fold in data.get("per_subject", []):
        cm = fold.get("confusion_matrix")
        if cm:
            arr = np.asarray(cm, dtype=int)
            if arr.shape == (3, 3):
                cms.append(arr)
    if not cms:
        raise ValueError(f"No 3x3 confusion matrices found in {path}")
    return np.sum(np.stack(cms, axis=0), axis=0)


def _load_rf_binary_cm(path: Path) -> np.ndarray:
    data = _read_json(path)
    cm = data.get("models", {}).get("random_forest", {}).get("cm")
    if not cm:
        raise ValueError(f"random_forest.cm not found in {path}")
    arr = np.asarray(cm, dtype=int)
    if arr.shape != (2, 2):
        raise ValueError(f"Expected 2x2 cm, got {arr.shape} in {path}")
    return arr


def _compute_shap_top_features(out_path: Path, model_path: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file for SHAP: {model_path}")

    data = load_wesad(subject_ids=[2, 3, 4, 5], device="both", n_classes=2)
    if not data["subjects"]:
        raise RuntimeError("No subjects loaded for SHAP analysis.")

    X_parts: List[np.ndarray] = []
    feature_names: Optional[List[str]] = None
    for subj in data["subjects"]:
        X, y, names = extract_subject_features(subj, window_sec=60, step_sec=30)
        if len(X):
            X_parts.append(X)
            if feature_names is None:
                feature_names = names

    if not X_parts or not feature_names:
        raise RuntimeError("No feature windows extracted for SHAP analysis.")

    X_all = np.vstack(X_parts)
    model = StressModel.load(str(model_path))
    X_scaled = model.scaler.transform(X_all)

    rng = np.random.RandomState(42)
    bg_n = min(120, len(X_scaled))
    ex_n = min(360, len(X_scaled))
    bg_idx = rng.choice(len(X_scaled), size=bg_n, replace=False)
    ex_idx = rng.choice(len(X_scaled), size=ex_n, replace=False)
    X_bg = X_scaled[bg_idx]
    X_ex = X_scaled[ex_idx]

    explainer = shap.TreeExplainer(model.model, data=X_bg)
    raw_vals = explainer.shap_values(X_ex)

    if isinstance(raw_vals, list):
        vals = np.asarray(raw_vals[1], dtype=float)
    else:
        vals_arr = np.asarray(raw_vals, dtype=float)
        if vals_arr.ndim == 3 and vals_arr.shape[-1] >= 2:
            vals = vals_arr[:, :, 1]
        else:
            vals = vals_arr

    importance = np.abs(vals).mean(axis=0)
    order = np.argsort(importance)[::-1][:12]
    top_vals = importance[order][::-1]
    top_names = [feature_names[i] for i in order][::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_names, top_vals, color="#7c3aed")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top SHAP Features (Random Forest, Binary)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    src = _resolve_sources(RESULTS_DIR)
    logger.info("Using summary CSV: %s", src.final_summary_csv.name)
    logger.info("Using RF cm source: %s", src.rf_subject_independent_json.name)
    logger.info("Using ResNet-3cls cm source: %s", src.resnet_3cls_loso_json.name)

    df = pd.read_csv(src.final_summary_csv)
    df = df.sort_values(by=["Task", "Family", "Model"]).reset_index(drop=True)
    df.to_csv(OUT_DIR / "final_benchmark_summary.csv", index=False)
    _write_final_markdown(df, OUT_DIR / "final_benchmark_summary.md")
    _write_subset_csvs(df, OUT_DIR)

    _plot_comparison(df, "Binary (2-class)", FIG_DIR / "model_comparison_binary.png")
    _plot_comparison(df, "Tri-class (3-class)", FIG_DIR / "model_comparison_3class.png")

    rf_cm = _load_rf_binary_cm(src.rf_subject_independent_json)
    _plot_cm(
        rf_cm,
        labels=["Non-Stress", "Stress"],
        title=f"RandomForest Confusion Matrix (Subject-Independent)\n{src.rf_subject_independent_json.name}",
        out_path=FIG_DIR / "confusion_matrix_rf_binary.png",
    )

    resnet_cm = _aggregate_resnet_3class_cm(src.resnet_3cls_loso_json)
    _plot_cm(
        resnet_cm,
        labels=["Baseline", "Stress", "Amusement"],
        title=f"ResNet1D Confusion Matrix (LOSO aggregate)\n{src.resnet_3cls_loso_json.name}",
        out_path=FIG_DIR / "confusion_matrix_resnet_3class.png",
    )

    _compute_shap_top_features(
        FIG_DIR / "shap_top_features.png",
        ROOT_DIR / "models" / "random_forest_independent.joblib",
    )

    logger.info("Artifacts written to %s", OUT_DIR)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    build()


if __name__ == "__main__":
    main()
