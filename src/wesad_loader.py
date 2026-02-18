"""WESAD dataset loader.

Handles loading pickle files, aligning signals/labels, and binary mapping.

WESAD pickle structure:
    data['signal']['wrist']  →  {'ACC','BVP','EDA','TEMP'}
    data['signal']['chest']  →  {'ACC','ECG','EMG','EDA','Temp','Resp'}
    data['label']            →  ndarray at 700 Hz

Labels: 0=undefined, 1=baseline, 2=stress, 3=amusement, 4=meditation
"""
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

from config import (
    WESAD_DIR, WESAD_SUBJECTS, WESAD_BINARY_MAP,
    WESAD_KEEP_LABELS, WESAD_LABEL_SR, SAMPLING_RATES,
)

logger = logging.getLogger(__name__)


# ── single subject ────────────────────────────────────────────────────────────

def load_subject(subject_id: int, wesad_dir: str = None,
                 device: str = "wrist") -> Dict:
    """Load one WESAD subject → signals dict + raw labels."""
    wesad_dir = Path(wesad_dir or WESAD_DIR)
    pkl = wesad_dir / f"S{subject_id}" / f"S{subject_id}.pkl"
    if not pkl.exists():
        raise FileNotFoundError(f"Not found: {pkl}")

    with open(pkl, "rb") as f:
        raw = pickle.load(f, encoding="latin1")

    signals: Dict[str, np.ndarray] = {}
    if device == "wrist":
        for key in ("ACC", "BVP", "EDA", "TEMP"):
            arr = raw["signal"]["wrist"].get(key)
            if arr is not None:
                signals[key] = np.asarray(arr).flatten()
        sr_dict = {k: v for k, v in SAMPLING_RATES.items() if k in signals}
    elif device == "chest":
        for key in ("ACC", "ECG", "EMG", "EDA", "Temp", "Resp"):
            arr = raw["signal"]["chest"].get(key)
            if arr is not None:
                s = np.asarray(arr)
                signals[key] = s.flatten() if s.ndim > 1 and s.shape[1] == 1 else s
        sr_dict = {k: 700 for k in signals}
    else:
        raise ValueError(f"Unknown device: {device}")

    labels = np.asarray(raw["label"]).flatten()
    logger.info("  S%d: %d label samples, signals=%s", subject_id, len(labels), list(signals))
    return {"signals": signals, "labels": labels,
            "subject_id": subject_id, "sampling_rates": sr_dict}


# ── label helpers ─────────────────────────────────────────────────────────────

def downsample_labels(labels: np.ndarray, label_sr: int = 700,
                      target_sr: int = 4) -> np.ndarray:
    """Majority-vote downsample from *label_sr* to *target_sr*."""
    ratio = label_sr // target_sr
    n = len(labels) // ratio
    out = np.zeros(n, dtype=int)
    for i in range(n):
        seg = labels[i * ratio:(i + 1) * ratio]
        vals, cnts = np.unique(seg, return_counts=True)
        out[i] = vals[np.argmax(cnts)]
    return out


def to_binary(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map WESAD labels → binary (0=relaxed, 1=stressed).

    Returns (binary_labels, valid_mask).  Invalid samples get label -1.
    """
    valid = np.isin(labels, WESAD_KEEP_LABELS)
    binary = np.full_like(labels, -1)
    for src, dst in WESAD_BINARY_MAP.items():
        binary[labels == src] = dst
    return binary, valid


# ── align signals + labels ────────────────────────────────────────────────────

def _align(signals: Dict[str, np.ndarray], sr_dict: Dict[str, int],
           labels: np.ndarray, label_sr: int) -> Tuple[Dict, np.ndarray]:
    """Trim to shortest common duration, downsample labels to 4 Hz."""
    durations = {k: len(v) / sr_dict.get(k, 4) for k, v in signals.items()}
    durations["labels"] = len(labels) / label_sr
    min_dur = min(durations.values())

    aligned = {k: v[:int(min_dur * sr_dict.get(k, 4))] for k, v in signals.items()}
    labels_4hz = downsample_labels(labels[:int(min_dur * label_sr)], label_sr, 4)
    return aligned, labels_4hz


# ── main loader ───────────────────────────────────────────────────────────────

def load_wesad(subject_ids: List[int] = None, wesad_dir: str = None,
               device: str = "wrist") -> Dict:
    """Load WESAD for multiple subjects with binary labels.

    Returns ``{'subjects': [<per-subject dict>, ...]}``.
    Each dict has: subject_id, signals, binary_labels, valid_mask, sampling_rates.
    """
    subject_ids = subject_ids or WESAD_SUBJECTS
    subjects = []

    for sid in subject_ids:
        try:
            raw = load_subject(sid, wesad_dir, device)
            signals, labels_4hz = _align(
                raw["signals"], raw["sampling_rates"],
                raw["labels"], WESAD_LABEL_SR,
            )
            binary, valid = to_binary(labels_4hz)
            subjects.append({
                "subject_id": sid,
                "signals": signals,
                "binary_labels": binary,
                "valid_mask": valid,
                "sampling_rates": raw["sampling_rates"],
            })
            logger.info("  S%d: %.0fs, %d valid 4-Hz samples",
                        sid, len(labels_4hz) / 4, valid.sum())
        except Exception as e:
            logger.error("Failed to load S%d: %s", sid, e)

    logger.info("Loaded %d/%d subjects", len(subjects), len(subject_ids))
    return {"subjects": subjects}


# ── CLI helper ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    wesad_dir = Path(WESAD_DIR)
    print(f"\nWESAD Dataset ({wesad_dir})")
    print("=" * 50)
    for sid in WESAD_SUBJECTS:
        pkl = wesad_dir / f"S{sid}" / f"S{sid}.pkl"
        print(f"  S{sid:2d}: {'OK' if pkl.exists() else 'MISSING'}")
    found = sum(1 for s in WESAD_SUBJECTS
                if (wesad_dir / f"S{s}" / f"S{s}.pkl").exists())
    print(f"\n{found}/{len(WESAD_SUBJECTS)} subjects available")
