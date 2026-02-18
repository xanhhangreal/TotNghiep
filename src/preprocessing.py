"""Signal preprocessing pipeline for physiological data."""
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
                b, a = scipy_signal.butter(order, wn, btype="band")
            elif filter_type == "lowpass":
                wn = min(cutoff, nyq * 0.95) / nyq
                b, a = scipy_signal.butter(order, wn, btype="low")
            elif filter_type == "highpass":
                if cutoff >= nyq:
                    return out
                b, a = scipy_signal.butter(order, cutoff / nyq, btype="high")
            else:
                raise ValueError(f"Unknown filter type: {filter_type}")

            out[valid] = scipy_signal.filtfilt(b, a, out[valid])
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

    BVP is bandpass-filtered at its native 64 Hz **before** downsampling
    (the 0.7–4 Hz passband requires Nyquist > 4 Hz).
    """
    pp = SignalPreprocessor(target_sr)
    result: Dict[str, np.ndarray] = {}

    for name, sig in signal_dict.items():
        if sig is None or len(sig) == 0:
            continue
        sr = sr_dict.get(name, target_sr)

        if name == "bvp":
            # filter at native SR, THEN downsample
            s = pp.remove_artifacts(sig)
            s = pp.apply_filter(s, sr, "bandpass", (0.7, 4.0))
            s = pp.fillna(s)
            s = pp.resample(s, sr, target_sr)
            s = pp.normalize(s)
        elif name in ("eda", "gsr"):
            s = pp.resample(sig, sr, target_sr)
            s = pp.pipeline(s, target_sr, lowpass=1.5)
        elif name in ("temp", "temperature"):
            s = pp.resample(sig, sr, target_sr)
            s = pp.pipeline(s, target_sr, lowpass=1.0)
        else:
            s = pp.resample(sig, sr, target_sr)
            s = pp.pipeline(s, target_sr)

        result[name] = s
    return result
