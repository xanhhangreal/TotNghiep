"""Signal preprocessing pipeline for physiological data.

Supports both wrist (Empatica E4) and chest (RespiBAN) modalities.
"""
import numpy as np
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SignalPreprocessor:
    """Preprocess physiological signals (resample, filter, normalize, fill NaN)."""

    def __init__(self, target_sr: int = 4):
        self.target_sr = target_sr

    # ── resampling ────────────────────────────────────────────────────────────
    def resample(self, signal: np.ndarray, original_sr: int,
                 target_sr: int = None) -> np.ndarray:
        """Resample *signal* from *original_sr* to *target_sr* via linear interpolation."""
        target_sr = target_sr or self.target_sr
        if original_sr == target_sr:
            return signal.copy()

        valid = ~np.isnan(signal)
        if not valid.any():
            return np.full(int(len(signal) * target_sr / original_sr), np.nan)

        t_orig = np.arange(len(signal)) / original_sr
        t_new = np.arange(int(len(signal) * target_sr / original_sr)) / target_sr

        f = interp1d(t_orig, signal, kind="linear",
                     fill_value=(signal[valid][0], signal[valid][-1]),
                     bounds_error=False)
        return f(t_new)

    # ── artifact removal ──────────────────────────────────────────────────────
    def remove_artifacts(self, signal: np.ndarray,
                         z_threshold: float = 3.0) -> np.ndarray:
        """Replace outliers (|z-score| > *z_threshold*) with NaN."""
        out = signal.copy()
        valid = out[~np.isnan(out)]
        if len(valid) < 2:
            return out
        mean, std = np.mean(valid), np.std(valid)
        bad = np.abs((out - mean) / (std + 1e-8)) > z_threshold
        out[bad] = np.nan
        return out

    # ── digital filtering ─────────────────────────────────────────────────────
    def apply_filter(self, signal: np.ndarray, sr: int,
                     filter_type: str = "lowpass", cutoff=1.0,
                     order: int = 4) -> np.ndarray:
        """Butterworth filter with automatic Nyquist clamping.

        Args:
            signal:      input signal (may contain NaN)
            sr:          sampling rate (Hz)
            filter_type: 'lowpass' | 'highpass' | 'bandpass'
            cutoff:      scalar or (low, high) for bandpass
            order:       filter order
        Returns:
            Filtered signal (NaN positions preserved).
        """
        out = signal.copy()
        valid = ~np.isnan(out)
        if valid.sum() < 3 * order + 1:
            return out

        nyq = sr / 2.0
        try:
            if filter_type == "bandpass":
                lo, hi = cutoff
                hi = min(hi, nyq * 0.95)
                if lo >= hi:
                    return out
                wn = [lo / nyq, hi / nyq]
                sos = scipy_signal.butter(order, wn, btype="band", output="sos")
            elif filter_type == "lowpass":
                wn = min(cutoff, nyq * 0.95) / nyq
                sos = scipy_signal.butter(order, wn, btype="low", output="sos")
            elif filter_type == "highpass":
                if cutoff >= nyq:
                    return out
                sos = scipy_signal.butter(order, cutoff / nyq, btype="high",
                                          output="sos")
            else:
                raise ValueError(f"Unknown filter type: {filter_type}")

            out[valid] = scipy_signal.sosfiltfilt(sos, out[valid])
            # Guard downstream code from occasional numeric explosions.
            out[~np.isfinite(out)] = np.nan
        except Exception as e:
            logger.warning("Filter failed (%s), returning original signal", e)
        return out

    # ── normalization ─────────────────────────────────────────────────────────
    def normalize(self, signal: np.ndarray, method: str = "zscore") -> np.ndarray:
        out = signal.copy()
        valid = ~np.isnan(out)
        v = out[valid]
        if len(v) == 0:
            return out
        if method == "zscore":
            m, s = np.mean(v), np.std(v)
            out[valid] = (v - m) / (s if s > 0 else 1)
        elif method == "minmax":
            lo, hi = np.min(v), np.max(v)
            r = hi - lo
            out[valid] = (v - lo) / (r if r > 0 else 1)
        return out

    # ── missing-value imputation ──────────────────────────────────────────────
    def fillna(self, signal: np.ndarray,
               method: str = "interpolate") -> np.ndarray:
        out = signal.copy()
        mask = np.isnan(out)
        if not mask.any():
            return out

        good = np.where(~mask)[0]
        if len(good) == 0:
            return out

        if method == "forward":
            idx = np.where(~mask, np.arange(len(mask)), 0)
            np.maximum.accumulate(idx, out=idx)
            idx[:good[0]] = good[0]
            return out[idx]

        if method == "interpolate":
            x_nan = mask.nonzero()[0]
            x_ok = (~mask).nonzero()[0]
            out[x_nan] = np.interp(x_nan, x_ok, out[x_ok])
            return out

        if method == "mean":
            out[mask] = np.mean(out[~mask])
            return out

        return out

    # ── convenience pipeline ──────────────────────────────────────────────────
    def pipeline(self, signal: np.ndarray, sr: int, *,
                 remove_outliers: bool = True,
                 lowpass: Optional[float] = None,
                 do_normalize: bool = True) -> np.ndarray:
        """Artifact removal → lowpass → fillna → normalize."""
        if remove_outliers:
            signal = self.remove_artifacts(signal)
        if lowpass is not None:
            signal = self.apply_filter(signal, sr, "lowpass", lowpass)
        signal = self.fillna(signal)
        if do_normalize:
            signal = self.normalize(signal)
        return signal


# ── WESAD-specific preprocessing ──────────────────────────────────────────────

def preprocess_wesad_signal(signal_dict: Dict[str, np.ndarray],
                            sr_dict: Dict[str, int],
                            target_sr: int = 4) -> Dict[str, np.ndarray]:
    """Preprocess WESAD signals → all outputs at *target_sr*.

    Handles both wrist (EDA, BVP, TEMP) and chest (ECG, EMG, EDA, Temp,
    Resp, ACC) signals.  Keys may be prefixed with ``wrist_`` or ``chest_``
    when ``device='both'`` is used in the loader.

    BVP/ECG are bandpass-filtered at their native SR **before** downsampling.
    """
    pp = SignalPreprocessor(target_sr)
    result: Dict[str, np.ndarray] = {}

    for name, sig in signal_dict.items():
        if sig is None or len(sig) == 0:
            continue
        sr = sr_dict.get(name, target_sr)
        low_name = name.lower()

        # ── BVP (wrist, 64 Hz) ────────────────────────────────────────
        if low_name in ("bvp", "wrist_bvp"):
            s = pp.remove_artifacts(sig)
            s = pp.apply_filter(s, sr, "bandpass", (0.7, 4.0))
            s = pp.fillna(s)
            s = pp.resample(s, sr, target_sr)
            s = pp.normalize(s)

        # ── EDA (wrist 4 Hz or chest 700 Hz) ──────────────────────────
        elif low_name in ("eda", "gsr", "wrist_eda", "chest_eda"):
            s = pp.resample(sig, sr, target_sr)
            s = pp.pipeline(s, target_sr, lowpass=1.5)

        # ── TEMP (wrist 4 Hz or chest 700 Hz) ────────────────────────
        elif low_name in ("temp", "temperature", "wrist_temp",
                          "chest_temp"):
            s = pp.resample(sig, sr, target_sr)
            s = pp.pipeline(s, target_sr, lowpass=1.0)

        # ── ECG (chest, 700 Hz) ───────────────────────────────────────
        elif low_name in ("ecg", "chest_ecg"):
            s = pp.remove_artifacts(sig, z_threshold=4.0)
            s = pp.apply_filter(s, sr, "bandpass", (0.5, 40.0))
            s = pp.fillna(s)
            s = pp.resample(s, sr, target_sr)
            s = pp.normalize(s)

        # ── EMG (chest, 700 Hz) ───────────────────────────────────────
        elif low_name in ("emg", "chest_emg"):
            s = pp.remove_artifacts(sig, z_threshold=4.0)
            s = pp.apply_filter(s, sr, "bandpass", (20.0, 250.0))
            s = pp.fillna(s)
            # rectify → envelope (lowpass at 10 Hz)
            s = np.abs(s)
            s = pp.apply_filter(s, sr, "lowpass", 10.0)
            s = pp.resample(s, sr, target_sr)
            s = pp.normalize(s)

        # ── Resp (chest, 700 Hz) ──────────────────────────────────────
        elif low_name in ("resp", "chest_resp"):
            s = pp.remove_artifacts(sig)
            s = pp.apply_filter(s, sr, "bandpass", (0.1, 0.5))
            s = pp.fillna(s)
            s = pp.resample(s, sr, target_sr)
            s = pp.normalize(s)

        # ── ACC (wrist 32 Hz or chest 700 Hz, 3-axis) ────────────────
        elif low_name in ("acc", "wrist_acc", "chest_acc"):
            if sig.ndim == 2 and sig.shape[1] == 3:
                # compute magnitude
                mag = np.sqrt((sig ** 2).sum(axis=1))
                s = pp.resample(mag, sr, target_sr)
                s = pp.pipeline(s, target_sr, lowpass=1.5)
            else:
                s = pp.resample(sig, sr, target_sr)
                s = pp.pipeline(s, target_sr)

        # ── fallback ──────────────────────────────────────────────────
        else:
            s = pp.resample(sig, sr, target_sr)
            s = pp.pipeline(s, target_sr)

        result[name] = s
    return result
