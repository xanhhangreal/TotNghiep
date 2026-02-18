"""Feature extraction from EDA, BVP, and temperature signals."""
import numpy as np
from scipy import signal as scipy_signal
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Features returned when a signal segment is too short
_EDA_NAN = {f"eda_{k}": np.nan for k in
            ["mean", "std", "min", "max", "range",
             "scr_peaks", "scr_mean_amp", "scr_max_amp", "scl_mean"]}

_BVP_NAN = {f"bvp_{k}": np.nan for k in
            ["mean", "std", "min", "max",
             "hr_mean", "hr_std", "hr_min", "hr_max",
             "hrv_sdnn", "hrv_rmssd", "hrv_lf", "hrv_hf", "hrv_lf_hf"]}

_TEMP_NAN = {f"temp_{k}": np.nan for k in
             ["mean", "std", "min", "max", "range", "slope"]}


class FeatureExtractor:
    """Static methods for per-window feature computation."""

    # ── EDA ───────────────────────────────────────────────────────────────────
    @staticmethod
    def eda_features(eda: np.ndarray, sr: int = 4) -> Dict:
        v = eda[~np.isnan(eda)]
        if len(v) < sr * 2:
            return dict(_EDA_NAN)

        feat: Dict = {
            "eda_mean": np.mean(v), "eda_std": np.std(v),
            "eda_min": np.min(v), "eda_max": np.max(v),
            "eda_range": np.ptp(v),
        }

        # Phasic (SCR) via high-pass 0.05 Hz
        phasic = _safe_filter(eda, sr, "highpass", 0.05)
        if phasic is not None:
            p = phasic[~np.isnan(phasic)]
            thresh = np.std(p) * 0.5
            peaks, props = find_peaks(p, height=thresh, distance=max(sr, 1))
            feat["eda_scr_peaks"] = len(peaks)
            feat["eda_scr_mean_amp"] = float(np.mean(props["peak_heights"])) if peaks.size else 0.0
            feat["eda_scr_max_amp"] = float(np.max(props["peak_heights"])) if peaks.size else 0.0
        else:
            feat.update({k: 0.0 for k in ("eda_scr_peaks", "eda_scr_mean_amp", "eda_scr_max_amp")})

        # Tonic (SCL) via low-pass 0.05 Hz
        tonic = _safe_filter(eda, sr, "lowpass", 0.05)
        feat["eda_scl_mean"] = float(np.nanmean(tonic)) if tonic is not None else np.nan
        return feat

    # ── BVP / HRV ────────────────────────────────────────────────────────────
    @staticmethod
    def bvp_features(bvp: np.ndarray, sr: int = 64) -> Dict:
        v = bvp[~np.isnan(bvp)]
        if len(v) < sr * 2:
            return dict(_BVP_NAN)

        feat: Dict = {
            "bvp_mean": np.mean(v), "bvp_std": np.std(v),
            "bvp_min": np.min(v), "bvp_max": np.max(v),
        }

        try:
            sig = _safe_filter(bvp, sr, "bandpass", (0.7, 4.0)) if sr > 8 else bvp.copy()
            if sig is None:
                raise ValueError("filter failed")
            clean = sig[~np.isnan(sig)]

            peaks, _ = find_peaks(clean, distance=max(int(sr * 0.3), 1))
            if len(peaks) < 3:
                raise ValueError("too few peaks")

            ibi = np.diff(peaks) / sr * 1000.0          # ms
            ibi = ibi[(ibi >= 300) & (ibi <= 1500)]     # 40–200 bpm
            if len(ibi) < 2:
                raise ValueError("no plausible IBIs")

            hr = 60_000.0 / ibi
            feat.update({
                "bvp_hr_mean": np.mean(hr), "bvp_hr_std": np.std(hr),
                "bvp_hr_min": np.min(hr), "bvp_hr_max": np.max(hr),
                "bvp_hrv_sdnn": np.std(ibi),
                "bvp_hrv_rmssd": np.sqrt(np.mean(np.diff(ibi) ** 2)),
            })

            # Frequency domain (needs ≥10 IBIs)
            if len(ibi) >= 10:
                feat.update(FeatureExtractor._hrv_freq(ibi))
            else:
                feat.update({"bvp_hrv_lf": np.nan, "bvp_hrv_hf": np.nan, "bvp_hrv_lf_hf": np.nan})

        except Exception:
            for k in ("hr_mean", "hr_std", "hr_min", "hr_max",
                      "hrv_sdnn", "hrv_rmssd", "hrv_lf", "hrv_hf", "hrv_lf_hf"):
                feat.setdefault(f"bvp_{k}", np.nan)
        return feat

    @staticmethod
    def _hrv_freq(ibi_ms: np.ndarray) -> Dict:
        """LF / HF power from IBI series, resampled to uniform 4 Hz grid."""
        times = np.cumsum(ibi_ms) / 1000.0
        times -= times[0]
        t_uni = np.arange(0, times[-1], 0.25)  # 4 Hz
        if len(t_uni) < 8:
            return {"bvp_hrv_lf": np.nan, "bvp_hrv_hf": np.nan, "bvp_hrv_lf_hf": np.nan}

        interp = interp1d(times, ibi_ms, kind="linear", fill_value="extrapolate")
        uniform = interp(t_uni) - np.mean(ibi_ms)

        f, pxx = scipy_signal.welch(uniform, fs=4.0, nperseg=min(256, len(uniform)))
        lf = np.trapz(pxx[(f >= 0.04) & (f <= 0.15)], f[(f >= 0.04) & (f <= 0.15)]) if (f <= 0.15).any() else 0.0
        hf = np.trapz(pxx[(f >= 0.15) & (f <= 0.4)], f[(f >= 0.15) & (f <= 0.4)]) if (f <= 0.4).any() else 0.0
        return {"bvp_hrv_lf": lf, "bvp_hrv_hf": hf,
                "bvp_hrv_lf_hf": lf / (hf + 1e-8)}

    # ── Temperature ───────────────────────────────────────────────────────────
    @staticmethod
    def temp_features(temp: np.ndarray, sr: int = 4) -> Dict:
        v = temp[~np.isnan(temp)]
        if len(v) < 2:
            return dict(_TEMP_NAN)
        feat: Dict = {
            "temp_mean": np.mean(v), "temp_std": np.std(v),
            "temp_min": np.min(v), "temp_max": np.max(v),
            "temp_range": np.ptp(v),
        }
        if len(v) > 2:
            feat["temp_slope"] = np.polyfit(np.arange(len(v)), v, 1)[0] * sr * 60
        else:
            feat["temp_slope"] = np.nan
        return feat

    # ── windowed extraction (main entry point) ───────────────────────────────
    @staticmethod
    def extract_windows(signals: Dict[str, np.ndarray],
                        sampling_rates: Dict[str, int],
                        window_sec: int = 60,
                        step_sec: int = 30) -> List[Dict]:
        """Slide a window over all signals and extract features per window.

        Each signal is sliced according to its own sampling rate so that
        all windows cover the same time interval.
        """
        durations = {k: len(v) / sampling_rates.get(k, 4) for k, v in signals.items()}
        min_dur = min(durations.values())
        n_win = int((min_dur - window_sec) / step_sec) + 1
        if n_win <= 0:
            return []

        rows: List[Dict] = []
        for w in range(n_win):
            t0 = w * step_sec
            t1 = t0 + window_sec
            row: Dict = {"window": w, "t0": t0, "t1": t1}

            for name, sig in signals.items():
                sr = sampling_rates.get(name, 4)
                seg = sig[int(t0 * sr):int(t1 * sr)]

                if name in ("eda", "gsr"):
                    row.update(FeatureExtractor.eda_features(seg, sr))
                elif name == "bvp":
                    row.update(FeatureExtractor.bvp_features(seg, sr))
                elif name in ("temp", "temperature"):
                    row.update(FeatureExtractor.temp_features(seg, sr))
            rows.append(row)
        return rows


# ── module-level filter helper (avoids circular import with preprocessing) ────

def _safe_filter(signal: np.ndarray, sr: int,
                 ftype: str, cutoff, order: int = 4) -> Optional[np.ndarray]:
    """Lightweight Butterworth filter; returns None on failure."""
    out = signal.copy()
    valid = ~np.isnan(out)
    if valid.sum() < 3 * order + 1:
        return None
    nyq = sr / 2.0
    try:
        if ftype == "bandpass":
            lo, hi = cutoff
            hi = min(hi, nyq * 0.95)
            if lo >= hi:
                return None
            wn = [lo / nyq, hi / nyq]
            b, a = scipy_signal.butter(order, wn, btype="band")
        elif ftype == "lowpass":
            b, a = scipy_signal.butter(order, min(cutoff, nyq * 0.95) / nyq, btype="low")
        elif ftype == "highpass":
            if cutoff >= nyq:
                return None
            b, a = scipy_signal.butter(order, cutoff / nyq, btype="high")
        else:
            return None
        out[valid] = scipy_signal.filtfilt(b, a, out[valid])
        return out
    except Exception:
        return None
