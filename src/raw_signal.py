"""Utilities for building raw-signal windows for DL baselines."""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

from preprocessing import preprocess_wesad_signal

logger = logging.getLogger(__name__)

RAW_CHANNELS: Dict[str, List[str]] = {
    "wrist": ["acc", "bvp", "eda", "temp"],
    "chest": ["acc", "ecg", "emg", "eda", "temp", "resp"],
    "both": [
        "wrist_acc",
        "wrist_bvp",
        "wrist_eda",
        "wrist_temp",
        "chest_acc",
        "chest_ecg",
        "chest_emg",
        "chest_eda",
        "chest_temp",
        "chest_resp",
    ],
}


def _to_float_1d(signal: np.ndarray) -> np.ndarray:
    """Convert input to 1-D float signal, using magnitude for 3-axis ACC."""
    arr = np.asarray(signal, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return np.sqrt((arr ** 2).sum(axis=1))
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr.reshape(-1)
    return arr.reshape(-1)


def _prepare_signals(
    signals: Dict[str, np.ndarray],
    sampling_rates: Dict[str, int],
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Lowercase keys and normalize signal shapes for preprocessing."""
    sig_lower: Dict[str, np.ndarray] = {}
    sr_lower: Dict[str, int] = {}
    for key, sig in signals.items():
        lk = key.lower()
        sig_lower[lk] = _to_float_1d(sig)
        sr_lower[lk] = int(sampling_rates.get(key, sampling_rates.get(lk, 4)))
    return sig_lower, sr_lower


def _expected_channels(device_mode: str, available: List[str]) -> List[str]:
    """Stable channel order for each device mode."""
    if device_mode in RAW_CHANNELS:
        return list(RAW_CHANNELS[device_mode])
    return sorted(available)


def _pad_or_trim(sig: np.ndarray, target_len: int) -> np.ndarray:
    """Pad with edge value or trim to an exact length."""
    out = np.asarray(sig, dtype=float).reshape(-1)
    if len(out) == target_len:
        return out
    if len(out) > target_len:
        return out[:target_len]
    if len(out) == 0:
        return np.zeros(target_len, dtype=float)
    pad_len = target_len - len(out)
    return np.pad(out, (0, pad_len), mode="edge")


def extract_subject_raw_windows(
    subject: Dict,
    *,
    device_mode: str = "both",
    window_sec: float = 60.0,
    step_sec: float = 30.0,
    target_sr: int = 32,
    min_valid_ratio: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build raw-signal windows with majority-vote labels.

    Returns:
        X: (N, C, T) float32 raw windows
        y: (N,) int labels
        channels: channel names (C)
    """
    labels = np.asarray(subject["binary_labels"], dtype=int)
    valid_mask = np.asarray(subject["valid_mask"], dtype=bool)

    sig_lower, sr_lower = _prepare_signals(
        subject["signals"],
        subject["sampling_rates"],
    )
    preprocessed = preprocess_wesad_signal(sig_lower, sr_lower, target_sr=target_sr)
    if not preprocessed:
        return np.array([]), np.array([]), []

    available = sorted(preprocessed.keys())
    channels = _expected_channels(device_mode, available)

    sig_durations = [len(preprocessed[k]) / float(target_sr) for k in available if len(preprocessed[k])]
    if not sig_durations:
        return np.array([]), np.array([]), channels

    min_dur = min(min(sig_durations), len(labels) / 4.0)
    if min_dur <= 0:
        return np.array([]), np.array([]), channels

    n_sig_samples = int(min_dur * target_sr)
    trimmed: Dict[str, np.ndarray] = {}
    for ch in channels:
        sig = preprocessed.get(ch)
        if sig is None:
            logger.warning("Missing channel '%s' for S%s; filling zeros", ch, subject.get("subject_id", "?"))
            arr = np.zeros(n_sig_samples, dtype=float)
        else:
            arr = _pad_or_trim(sig, n_sig_samples)
            finite = np.isfinite(arr)
            if not finite.any():
                arr = np.zeros_like(arr)
            elif not finite.all():
                fill_val = float(np.nanmedian(arr[finite]))
                arr = arr.copy()
                arr[~finite] = fill_val
        trimmed[ch] = arr.astype(float)

    win_len = int(round(window_sec * target_sr))
    n_win = int((min_dur - window_sec) / step_sec) + 1
    if win_len <= 0 or n_win <= 0:
        return np.array([]), np.array([]), channels

    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for wi in range(n_win):
        t0 = wi * step_sec
        t1 = t0 + window_sec

        i0 = int(round(t0 * target_sr))
        i1 = i0 + win_len
        if i0 >= n_sig_samples:
            continue

        win_channels: List[np.ndarray] = []
        for ch in channels:
            seg = _pad_or_trim(trimmed[ch][i0:i1], win_len)
            win_channels.append(seg)

        l0 = int(round(t0 * 4.0))
        l1 = int(round(t1 * 4.0))
        seg_lbl = labels[l0:l1]
        seg_val = valid_mask[l0:l1]
        if len(seg_lbl) == 0 or len(seg_val) == 0:
            continue
        if float(seg_val.mean()) < min_valid_ratio:
            continue

        vals, cnts = np.unique(seg_lbl[seg_val], return_counts=True)
        if len(vals) == 0:
            continue
        y = int(vals[np.argmax(cnts)])
        if y < 0:
            continue

        X_list.append(np.stack(win_channels, axis=0).astype(np.float32))
        y_list.append(y)

    if not X_list:
        return np.array([]), np.array([]), channels

    return np.stack(X_list, axis=0), np.asarray(y_list, dtype=int), channels
