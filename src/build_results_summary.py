"""Build benchmark artifacts from the latest LOSO result JSON files.

Outputs to ``results_summary/``:
  - ml_binary_loso_summary.csv
  - ml_3class_loso_summary.csv
  - dl_binary_loso_summary.csv
  - dl_3class_loso_summary.csv
  - final_benchmark_summary.csv
  - final_benchmark_summary.md
  - figures/model_comparison_binary.png
  - figures/model_comparison_3class.png
  - figures/confusion_matrix_binary.png
  - figures/confusion_matrix_3class.png
  - figures/confusion_matrix_rf_binary.png
  - figures/confusion_matrix_resnet_3class.png
  - figures/shap_top_features.png
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from config import DEFAULT_MODELS
from ml_models import StressModel
from training import extract_subject_features
from wesad_loader import load_wesad

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
OUT_DIR = ROOT_DIR / "results_summary"
FIG_DIR = OUT_DIR / "figures"

TASK_BINARY = "Binary (2-class)"
TASK_3CLASS = "Tri-class (3-class)"

ML_NAME_MAP = {
    "random_forest": "RandomForest",
    "logistic_regression": "LogisticRegression",
    "svm": "SVM",
    "decision_tree": "DecisionTree",
    "adaboost": "AdaBoost",
    "lda": "LDA",
    "knn": "KNN",
}

DL_NAME_MAP = {
    "cnn1d": "CNN1D",
    "unet1d": "UNet1D",
    "resnet1d": "ResNet1D",
}


def _read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_ts(path: Path) -> Tuple[int, int]:
    match = re.search(r"_(\d{8})_(\d{6})$", path.stem)
    if match:
        return int(match.group(1)), int(match.group(2))
    # Fallback if filename does not embed timestamp
    return 0, int(path.stat().st_mtime)


def _fmt_mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f} +/- {std:.4f}"


def _discover_ml_loso_files(results_dir: Path) -> Tuple[Path, Path]:
    """Return (binary_path, tri_class_path) from the two latest ML LOSO files."""
    files = sorted(results_dir.glob("loso_*.json"), key=_extract_ts)
    if len(files) < 2:
        raise FileNotFoundError("Need at least 2 files matching results/loso_*.json")

    latest_two = files[-2:]
    scored: List[Tuple[float, Path]] = []
    for p in latest_two:
        data = _read_json(p)
        f1_vals = [
            float(v.get("f1_mean"))
            for v in data.values()
            if isinstance(v, dict) and "f1_mean" in v
        ]
        if not f1_vals:
            raise ValueError(f"File has no model-level f1_mean metrics: {p}")
        scored.append((float(np.mean(f1_vals)), p))

    # Heuristic: binary task usually has higher F1 than tri-class.
    scored.sort(key=lambda x: x[0], reverse=True)
    binary_path = scored[0][1]
    tri_path = scored[-1][1]
    return binary_path, tri_path


def _discover_dl_loso_files(results_dir: Path) -> Dict[Tuple[int, str], Path]:
    """Return latest DL LOSO file for each (n_classes, arch)."""
    found: Dict[Tuple[int, str], Path] = {}
    for arch in DL_NAME_MAP:
        for n_classes, cls_tag in ((2, "2cls"), (3, "3cls")):
            pattern = f"dl_loso_{arch}_{cls_tag}_*.json"
            files = sorted(results_dir.glob(pattern), key=_extract_ts)
            if not files:
                raise FileNotFoundError(f"Missing file for pattern: results/{pattern}")
            found[(n_classes, arch)] = files[-1]
    return found


def _rows_from_ml(path: Path, task: str) -> List[Dict]:
    data = _read_json(path)
    rows: List[Dict] = []
    for model_key, metrics in data.items():
        if not isinstance(metrics, dict):
            continue
        if "accuracy_mean" not in metrics or "f1_mean" not in metrics:
            continue
        model_name = ML_NAME_MAP.get(model_key, model_key)
        rows.append(
            {
                "Task": task,
                "Family": "ML",
                "Model": model_name,
                "Accuracy_mean": float(metrics["accuracy_mean"]),
                "Accuracy_std": float(metrics["accuracy_std"]),
                "F1_mean": float(metrics["f1_mean"]),
                "F1_std": float(metrics["f1_std"]),
                "SourceFile": path.name,
            }
        )
    if not rows:
        raise ValueError(f"No valid ML rows parsed from {path}")
    return rows


def _rows_from_dl(path: Path) -> List[Dict]:
    data = _read_json(path)
    arch = str(data.get("arch", "")).lower()
    n_classes = int(data.get("n_classes", 2))
    task = TASK_BINARY if n_classes == 2 else TASK_3CLASS
    model_name = DL_NAME_MAP.get(arch, arch or "DLModel")
    rows = [
        {
            "Task": task,
            "Family": "DL",
            "Model": model_name,
            "Accuracy_mean": float(data["accuracy_mean"]),
            "Accuracy_std": float(data["accuracy_std"]),
            "F1_mean": float(data["f1_mean"]),
            "F1_std": float(data["f1_std"]),
            "SourceFile": path.name,
        }
    ]
    return rows


def _write_subset_csvs(df: pd.DataFrame, out_dir: Path) -> None:
    mapping = {
        (TASK_BINARY, "ML"): "ml_binary_loso_summary.csv",
        (TASK_3CLASS, "ML"): "ml_3class_loso_summary.csv",
        (TASK_BINARY, "DL"): "dl_binary_loso_summary.csv",
        (TASK_3CLASS, "DL"): "dl_3class_loso_summary.csv",
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
    for task in (TASK_BINARY, TASK_3CLASS):
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

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(
        x - width / 2,
        part["Accuracy_mean"],
        width=width,
        label="Accuracy",
        color="#1d4ed8",
    )
    ax.bar(
        x + width / 2,
        part["F1_mean"],
        width=width,
        label="F1",
        color="#059669",
    )
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
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
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


def _aggregate_dl_cm(path: Path, shape: Tuple[int, int]) -> np.ndarray:
    data = _read_json(path)
    cms = []
    for fold in data.get("per_subject", []):
        cm = fold.get("confusion_matrix")
        if cm:
            arr = np.asarray(cm, dtype=int)
            if arr.shape == shape:
                cms.append(arr)
    if not cms:
        raise ValueError(f"No confusion matrices with shape={shape} in {path}")
    return np.sum(np.stack(cms, axis=0), axis=0)


def _compute_rf_binary_cm() -> np.ndarray:
    """Compute LOSO aggregate RF confusion matrix for binary task."""
    data = load_wesad(subject_ids=None, device="both", n_classes=2)
    if not data["subjects"]:
        raise RuntimeError("Could not load WESAD for RF confusion matrix.")

    per_subject: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for subj in data["subjects"]:
        X, y, _ = extract_subject_features(subj, window_sec=60, step_sec=30)
        if len(X):
            per_subject[int(subj["subject_id"])] = (X, y)

    sids = sorted(per_subject)
    if len(sids) < 2:
        raise RuntimeError("Need at least 2 subjects for RF LOSO confusion matrix.")

    cm_total = np.zeros((2, 2), dtype=int)
    params = DEFAULT_MODELS["random_forest"]
    for test_sid in sids:
        X_te, y_te = per_subject[test_sid]
        X_tr = np.vstack([per_subject[s][0] for s in sids if s != test_sid])
        y_tr = np.concatenate([per_subject[s][1] for s in sids if s != test_sid])
        mdl = StressModel("random_forest", params)
        mdl.fit(X_tr, y_tr, verbose=False)
        metrics = mdl.evaluate(X_te, y_te, verbose=False)
        cm_total += np.asarray(metrics["confusion_matrix"], dtype=int)
    return cm_total


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
    ax.barh(top_names, top_vals, color="#1f77b4")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top SHAP Features (Random Forest, Binary)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ml_binary_path, ml_tri_path = _discover_ml_loso_files(RESULTS_DIR)
    dl_files = _discover_dl_loso_files(RESULTS_DIR)

    logger.info("ML binary source : %s", ml_binary_path.name)
    logger.info("ML tri-class src : %s", ml_tri_path.name)
    for key, path in sorted(dl_files.items()):
        n_classes, arch = key
        logger.info("DL %s (%d-class): %s", arch, n_classes, path.name)

    rows: List[Dict] = []
    rows.extend(_rows_from_ml(ml_binary_path, TASK_BINARY))
    rows.extend(_rows_from_ml(ml_tri_path, TASK_3CLASS))
    for _, path in sorted(dl_files.items()):
        rows.extend(_rows_from_dl(path))

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No benchmark rows produced.")

    df["Accuracy_mean_std"] = df.apply(
        lambda r: _fmt_mean_std(r["Accuracy_mean"], r["Accuracy_std"]), axis=1
    )
    df["F1_mean_std"] = df.apply(
        lambda r: _fmt_mean_std(r["F1_mean"], r["F1_std"]), axis=1
    )

    task_order = {TASK_BINARY: 0, TASK_3CLASS: 1}
    family_order = {"ML": 0, "DL": 1}
    df["_task_order"] = df["Task"].map(task_order)
    df["_fam_order"] = df["Family"].map(family_order)
    df = df.sort_values(
        by=["_task_order", "_fam_order", "F1_mean"],
        ascending=[True, True, False],
    ).drop(columns=["_task_order", "_fam_order"])
    df = df.reset_index(drop=True)

    df.to_csv(OUT_DIR / "final_benchmark_summary.csv", index=False)
    _write_final_markdown(df, OUT_DIR / "final_benchmark_summary.md")
    _write_subset_csvs(df, OUT_DIR)
    _plot_comparison(df, TASK_BINARY, FIG_DIR / "model_comparison_binary.png")
    _plot_comparison(df, TASK_3CLASS, FIG_DIR / "model_comparison_3class.png")

    # Binary confusion matrix: aggregate best DL LOSO model (available in JSON).
    dl_binary = (
        df[(df["Task"] == TASK_BINARY) & (df["Family"] == "DL")]
        .sort_values(by="F1_mean", ascending=False)
        .iloc[0]
    )
    dl_bin_path = RESULTS_DIR / str(dl_binary["SourceFile"])
    dl_bin_cm = _aggregate_dl_cm(dl_bin_path, shape=(2, 2))
    _plot_cm(
        dl_bin_cm,
        labels=["Non-Stress", "Stress"],
        title=f"Binary Confusion Matrix (LOSO aggregate)\n{dl_binary['Model']} - {dl_bin_path.name}",
        out_path=FIG_DIR / "confusion_matrix_binary.png",
    )

    # Keep compatibility filename expected by earlier docs/checklists.
    _plot_cm(
        dl_bin_cm,
        labels=["Non-Stress", "Stress"],
        title=f"Binary Confusion Matrix (LOSO aggregate)\n{dl_binary['Model']} - {dl_bin_path.name}",
        out_path=FIG_DIR / "confusion_matrix_rf_binary.png",
    )

    # Tri-class confusion matrix: best DL LOSO model.
    dl_tri = (
        df[(df["Task"] == TASK_3CLASS) & (df["Family"] == "DL")]
        .sort_values(by="F1_mean", ascending=False)
        .iloc[0]
    )
    dl_tri_path = RESULTS_DIR / str(dl_tri["SourceFile"])
    dl_tri_cm = _aggregate_dl_cm(dl_tri_path, shape=(3, 3))
    _plot_cm(
        dl_tri_cm,
        labels=["Baseline", "Stress", "Amusement"],
        title=f"3-class Confusion Matrix (LOSO aggregate)\n{dl_tri['Model']} - {dl_tri_path.name}",
        out_path=FIG_DIR / "confusion_matrix_3class.png",
    )

    # Also export ResNet 3-class confusion matrix for continuity.
    resnet_3cls = dl_files.get((3, "resnet1d"))
    if resnet_3cls is not None:
        resnet_cm = _aggregate_dl_cm(resnet_3cls, shape=(3, 3))
        _plot_cm(
            resnet_cm,
            labels=["Baseline", "Stress", "Amusement"],
            title=f"ResNet1D 3-class Confusion Matrix (LOSO aggregate)\n{resnet_3cls.name}",
            out_path=FIG_DIR / "confusion_matrix_resnet_3class.png",
        )

    # Recompute RF LOSO confusion matrix from data to keep ML confusion artifact.
    try:
        rf_cm = _compute_rf_binary_cm()
        _plot_cm(
            rf_cm,
            labels=["Non-Stress", "Stress"],
            title="RandomForest Binary Confusion Matrix (LOSO aggregate)",
            out_path=FIG_DIR / "confusion_matrix_rf_binary.png",
        )
    except Exception as exc:  # pragma: no cover - robustness fallback
        logger.warning("Could not compute RF confusion matrix: %s", exc)

    try:
        _compute_shap_top_features(
            FIG_DIR / "shap_top_features.png",
            ROOT_DIR / "models" / "random_forest_independent.joblib",
        )
    except Exception as exc:  # pragma: no cover - robustness fallback
        logger.warning("Could not generate SHAP figure: %s", exc)

    logger.info("Artifacts written to %s", OUT_DIR)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build()


if __name__ == "__main__":
    main()
