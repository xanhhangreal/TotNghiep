"""Build device-ablation summary from latest LOSO JSON results.

Outputs to ``results_summary/``:
  - device_ablation_summary.csv
  - device_ablation_summary.md

This script expects result files produced by:
  - src/training.py (ML feature-based LOSO)
  - src/dl_training.py (DL feature-based LOSO)
  - src/raw_dl_training.py (DL raw-signal LOSO)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
OUT_DIR = ROOT_DIR / "results_summary"

TASK_MAP = {2: "Binary (2-class)", 3: "Tri-class (3-class)"}

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
    "rawcnn1d": "RawCNN1D",
}


def _read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_ts(path: Path) -> Tuple[int, int]:
    m = re.search(r"_(\d{8})_(\d{6})$", path.stem)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 0, int(path.stat().st_mtime)


def _task_name(n_classes: int) -> str:
    return TASK_MAP.get(int(n_classes), "Unknown")


def _rows_from_ml(path: Path, data: Dict) -> List[Dict]:
    meta = data.get("_meta", {}) if isinstance(data, dict) else {}
    n_classes = int(meta.get("n_classes", 0))
    if n_classes not in (2, 3):
        return []

    device = str(meta.get("device", "unknown"))
    if device not in ("wrist", "chest", "both"):
        return []
    task = _task_name(n_classes)

    rows: List[Dict] = []
    for key, metrics in data.items():
        if key == "_meta" or not isinstance(metrics, dict):
            continue
        if "accuracy_mean" not in metrics or "f1_mean" not in metrics:
            continue

        rows.append(
            {
                "Task": task,
                "Device": device,
                "Family": "ML",
                "InputMode": "feature_based",
                "Model": ML_NAME_MAP.get(key, key),
                "Accuracy_mean": float(metrics["accuracy_mean"]),
                "Accuracy_std": float(metrics.get("accuracy_std", 0.0)),
                "F1_mean": float(metrics["f1_mean"]),
                "F1_std": float(metrics.get("f1_std", 0.0)),
                "SourceFile": path.name,
            }
        )
    return rows


def _rows_from_dl(path: Path, data: Dict) -> List[Dict]:
    if not isinstance(data, dict):
        return []
    if "accuracy_mean" not in data or "f1_mean" not in data:
        return []

    meta = data.get("_meta", {}) if isinstance(data.get("_meta", {}), dict) else {}
    n_classes = int(data.get("n_classes", meta.get("n_classes", 0)))
    if n_classes not in (2, 3):
        return []

    task = _task_name(n_classes)
    device = str(data.get("device", meta.get("device", "unknown")))
    if device not in ("wrist", "chest", "both"):
        return []
    arch = str(data.get("arch", meta.get("arch", "dl"))).lower()
    model = DL_NAME_MAP.get(arch, arch or "DLModel")

    input_mode = str(data.get("input_mode", "feature_based")).lower()
    if input_mode not in ("feature_based", "raw_signal"):
        input_mode = "feature_based"

    return [
        {
            "Task": task,
            "Device": device,
            "Family": "DL",
            "InputMode": input_mode,
            "Model": model,
            "Accuracy_mean": float(data["accuracy_mean"]),
            "Accuracy_std": float(data.get("accuracy_std", 0.0)),
            "F1_mean": float(data["f1_mean"]),
            "F1_std": float(data.get("f1_std", 0.0)),
            "SourceFile": path.name,
        }
    ]


def _collect_latest_rows() -> pd.DataFrame:
    latest: Dict[Tuple[str, str, str, str, str], Dict] = {}

    for path in sorted(RESULTS_DIR.glob("loso_*.json"), key=_extract_ts):
        data = _read_json(path)
        ts = _extract_ts(path)
        for row in _rows_from_ml(path, data):
            key = (row["Task"], row["Device"], row["Family"], row["InputMode"], row["Model"])
            prev = latest.get(key)
            if prev is None or ts > prev["_ts"]:
                latest[key] = {**row, "_ts": ts}

    for path in sorted(RESULTS_DIR.glob("dl_loso_*.json"), key=_extract_ts):
        data = _read_json(path)
        ts = _extract_ts(path)
        for row in _rows_from_dl(path, data):
            key = (row["Task"], row["Device"], row["Family"], row["InputMode"], row["Model"])
            prev = latest.get(key)
            if prev is None or ts > prev["_ts"]:
                latest[key] = {**row, "_ts": ts}

    if not latest:
        return pd.DataFrame()

    df = pd.DataFrame(list(latest.values())).drop(columns=["_ts"])
    return df


def _write_markdown(df: pd.DataFrame, out_path: Path) -> None:
    lines: List[str] = ["# Device Ablation Summary (LOSO)", ""]

    for task in ("Binary (2-class)", "Tri-class (3-class)"):
        part_task = df[df["Task"] == task]
        if part_task.empty:
            continue
        lines.append(f"## {task}")
        for device in ("wrist", "chest", "both"):
            part_dev = part_task[part_task["Device"] == device].copy()
            if part_dev.empty:
                continue
            part_dev = part_dev.sort_values(by=["Family", "InputMode", "F1_mean"], ascending=[True, True, False])
            lines.append(f"### Device: {device}")
            lines.append("| Family | Input | Model | Accuracy (mean +/- std) | F1 (mean +/- std) | Source |")
            lines.append("|---|---|---|---:|---:|---|")
            for _, row in part_dev.iterrows():
                acc = f"{row['Accuracy_mean']:.4f} +/- {row['Accuracy_std']:.4f}"
                f1 = f"{row['F1_mean']:.4f} +/- {row['F1_std']:.4f}"
                lines.append(
                    f"| {row['Family']} | {row['InputMode']} | {row['Model']} | {acc} | {f1} | {row['SourceFile']} |"
                )
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = _collect_latest_rows()
    if df.empty:
        raise RuntimeError(
            "No compatible LOSO result files found. "
            "Run training.py / dl_training.py / raw_dl_training.py first."
        )

    df = df.sort_values(
        by=["Task", "Device", "Family", "InputMode", "F1_mean"],
        ascending=[True, True, True, True, False],
    ).reset_index(drop=True)

    csv_path = OUT_DIR / "device_ablation_summary.csv"
    md_path = OUT_DIR / "device_ablation_summary.md"

    df.to_csv(csv_path, index=False)
    _write_markdown(df, md_path)

    print("Saved:")
    print(f"  - {csv_path}")
    print(f"  - {md_path}")


if __name__ == "__main__":
    build()
