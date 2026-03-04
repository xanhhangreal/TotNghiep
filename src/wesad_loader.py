"""WESAD dataset loader.

Handles loading pickle files, aligning signals/labels, and binary/3-class mapping.

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
    CHEST_SAMPLING_RATES, WESAD_3CLASS_MAP, WESAD_KEEP_LABELS_3CLASS,
)

logger = logging.getLogger(__name__)


# ── single subject ────────────────────────────────────────────────────────────

def load_subject(subject_id: int, wesad_dir: str = None,
                 device: str = "wrist") -> Dict:
    """Load one WESAD subject → signals dict + raw labels.

    Args:
        device: "wrist", "chest", or "both" (all modalities).
    """
    wesad_dir = Path(wesad_dir or WESAD_DIR)
    pkl = wesad_dir / f"S{subject_id}" / f"S{subject_id}.pkl"
    if not pkl.exists():
        raise FileNotFoundError(f"Not found: {pkl}")

    with open(pkl, "rb") as f:
        raw = pickle.load(f, encoding="latin1")

    signals: Dict[str, np.ndarray] = {}
    sr_dict: Dict[str, int] = {}

    # ── wrist signals ─────────────────────────────────────────────────────
    if device in ("wrist", "both"):
        for key in ("ACC", "BVP", "EDA", "TEMP"):
            arr = raw["signal"]["wrist"].get(key)
            if arr is not None:
                s = np.asarray(arr)
                pref = "wrist_" if device == "both" else ""
                skey = f"{pref}{key}"
                signals[skey] = s.flatten() if s.ndim == 1 or (s.ndim > 1 and s.shape[1] == 1) else s
                sr_dict[skey] = SAMPLING_RATES.get(key, 4)

    # ── chest signals ─────────────────────────────────────────────────────
    if device in ("chest", "both"):
        for key in ("ACC", "ECG", "EMG", "EDA", "Temp", "Resp"):
            arr = raw["signal"]["chest"].get(key)
            if arr is not None:
                s = np.asarray(arr)
                pref = "chest_" if device == "both" else ""
                skey = f"{pref}{key}"
                signals[skey] = s.flatten() if s.ndim == 1 or (s.ndim > 1 and s.shape[1] == 1) else s
                sr_dict[skey] = CHEST_SAMPLING_RATES.get(key, 700)

    if not signals:
        raise ValueError(f"No signals found for device={device}")

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


def to_3class(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map WESAD labels → 3-class (0=baseline, 1=stress, 2=amusement).

    Returns (mapped_labels, valid_mask).  Invalid samples get label -1.
    """
    valid = np.isin(labels, WESAD_KEEP_LABELS_3CLASS)
    mapped = np.full_like(labels, -1)
    for src, dst in WESAD_3CLASS_MAP.items():
        mapped[labels == src] = dst
    return mapped, valid


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
               device: str = "wrist", n_classes: int = 2) -> Dict:
    """Load WESAD for multiple subjects with binary or 3-class labels.

    Args:
        device:    "wrist", "chest", or "both" (all modalities).
        n_classes: 2 for binary (relaxed/stressed), 3 for 3-class
                   (baseline/stress/amusement).

    Returns ``{'subjects': [<per-subject dict>, ...]}``.
    Each dict has: subject_id, signals, binary_labels (or mapped_labels),
    valid_mask, sampling_rates.
    """
    subject_ids = subject_ids or WESAD_SUBJECTS
    label_fn = to_binary if n_classes == 2 else to_3class
    subjects = []

    for sid in subject_ids:
        try:
            raw = load_subject(sid, wesad_dir, device)
            signals, labels_4hz = _align(
                raw["signals"], raw["sampling_rates"],
                raw["labels"], WESAD_LABEL_SR,
            )
            mapped, valid = label_fn(labels_4hz)
            subjects.append({
                "subject_id": sid,
                "signals": signals,
                "binary_labels": mapped,
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
