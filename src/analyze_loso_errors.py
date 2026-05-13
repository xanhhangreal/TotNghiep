"""Analyze LOSO subject-level errors from latest ML/DL result JSON files.

Outputs to ``results_summary/``:
  - subject_level_metrics.csv
  - subject_error_analysis.csv
  - subject_error_analysis.md
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
OUT_DIR = ROOT_DIR / "results_summary"

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
    m = re.search(r"_(\d{8})_(\d{6})$", path.stem)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 0, int(path.stat().st_mtime)


def _discover_ml_loso_files() -> Tuple[Path, Path]:
    files = sorted(RESULTS_DIR.glob("loso_*.json"), key=_extract_ts)
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
            raise ValueError(f"No model-level F1 found in {p.name}")
        scored.append((float(np.mean(f1_vals)), p))
    scored.sort(key=lambda x: x[0], reverse=True)
    # Binary usually has higher F1 than tri-class.
    return scored[0][1], scored[1][1]


def _discover_dl_loso_files() -> Dict[Tuple[int, str], Path]:
    out: Dict[Tuple[int, str], Path] = {}
    for arch in DL_NAME_MAP:
        for n_classes, cls_tag in ((2, "2cls"), (3, "3cls")):
            files = sorted(
                RESULTS_DIR.glob(f"dl_loso_{arch}_{cls_tag}_*.json"),
                key=_extract_ts,
            )
            if not files:
                raise FileNotFoundError(f"Missing results for {arch} {cls_tag}")
            out[(n_classes, arch)] = files[-1]
    return out


def _collect_ml_rows(path: Path, task: str) -> List[Dict]:
    data = _read_json(path)
    rows: List[Dict] = []
    for model_key, metrics in data.items():
        if not isinstance(metrics, dict):
            continue
        for fold in metrics.get("per_subject", []):
            rows.append(
                {
                    "Task": task,
                    "Family": "ML",
                    "Model": ML_NAME_MAP.get(model_key, model_key),
                    "TestSubject": int(fold["test_subject"]),
                    "Accuracy": float(fold["accuracy"]),
                    "F1": float(fold["f1"]),
                    "SourceFile": path.name,
                }
            )
    return rows


def _collect_dl_rows(path: Path) -> List[Dict]:
    data = _read_json(path)
    n_classes = int(data.get("n_classes", 2))
    task = TASK_BINARY if n_classes == 2 else TASK_3CLASS
    arch = str(data.get("arch", "")).lower()
    model = DL_NAME_MAP.get(arch, arch or "DLModel")
    rows: List[Dict] = []
    for fold in data.get("per_subject", []):
        rows.append(
            {
                "Task": task,
                "Family": "DL",
                "Model": model,
                "TestSubject": int(fold["test_subject"]),
                "Accuracy": float(fold["accuracy"]),
                "F1": float(fold["f1"]),
                "SourceFile": path.name,
            }
        )
    return rows


def _write_markdown(df_agg: pd.DataFrame, out_path: Path) -> None:
    lines = ["# Subject Error Analysis (LOSO)", ""]
    for task in (TASK_BINARY, TASK_3CLASS):
        lines.append(f"## {task}")
        lines.append("| Subject | Mean Accuracy | Mean F1 | Num Models |")
        lines.append("|---:|---:|---:|---:|")
        part = (
            df_agg[df_agg["Task"] == task]
            .sort_values(by=["MeanF1", "MeanAccuracy"], ascending=[True, True])
        )
        for _, row in part.iterrows():
            lines.append(
                f"| S{int(row['TestSubject'])} | "
                f"{row['MeanAccuracy']:.4f} | {row['MeanF1']:.4f} | {int(row['NumModels'])} |"
            )
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ml_binary, ml_tri = _discover_ml_loso_files()
    dl_files = _discover_dl_loso_files()

    rows: List[Dict] = []
    rows.extend(_collect_ml_rows(ml_binary, TASK_BINARY))
    rows.extend(_collect_ml_rows(ml_tri, TASK_3CLASS))
    for _, path in sorted(dl_files.items()):
        rows.extend(_collect_dl_rows(path))

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No subject-level rows collected from LOSO files.")

    df = df.sort_values(by=["Task", "Family", "Model", "TestSubject"]).reset_index(drop=True)
    df.to_csv(OUT_DIR / "subject_level_metrics.csv", index=False)

    agg = (
        df.groupby(["Task", "TestSubject"], as_index=False)
        .agg(
            MeanAccuracy=("Accuracy", "mean"),
            MeanF1=("F1", "mean"),
            NumModels=("Model", "count"),
        )
        .sort_values(by=["Task", "MeanF1", "MeanAccuracy"], ascending=[True, True, True])
    )
    agg.to_csv(OUT_DIR / "subject_error_analysis.csv", index=False)
    _write_markdown(agg, OUT_DIR / "subject_error_analysis.md")

    print("Saved:")
    print(f"  - {OUT_DIR / 'subject_level_metrics.csv'}")
    print(f"  - {OUT_DIR / 'subject_error_analysis.csv'}")
    print(f"  - {OUT_DIR / 'subject_error_analysis.md'}")


if __name__ == "__main__":
    build()
