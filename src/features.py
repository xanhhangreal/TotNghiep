"""Feature extraction from all physiological modalities.

Wrist (Empatica E4): EDA, BVP, TEMP, ACC
Chest (RespiBAN):    ECG, EMG, EDA, Temp, Resp, ACC
"""
import numpy as np
from scipy import signal as scipy_signal
from scipy.signal import find_peaks, welch
from scipy.interpolate import interp1d
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# ── NaN templates for each modality ───────────────────────────────────────────

_EDA_NAN = {f"eda_{k}": np.nan for k in
            ["mean", "std", "min", "max", "range",
             "scr_peaks", "scr_mean_amp", "scr_max_amp", "scl_mean"]}

_BVP_NAN = {f"bvp_{k}": np.nan for k in
            ["mean", "std", "min", "max",
             "hr_mean", "hr_std", "hr_min", "hr_max",
             "hrv_sdnn", "hrv_rmssd", "hrv_lf", "hrv_hf", "hrv_lf_hf"]}

_TEMP_NAN = {f"temp_{k}": np.nan for k in
             ["mean", "std", "min", "max", "range", "slope"]}

_ECG_NAN = {f"ecg_{k}": np.nan for k in
            ["hr_mean", "hr_std", "hr_min", "hr_max",
             "hrv_sdnn", "hrv_rmssd", "hrv_pnn50",
             "hrv_lf", "hrv_hf", "hrv_lf_hf",
             "rr_mean", "rr_std", "amplitude",
             "qrs_count", "qrs_rate"]}

_EMG_NAN = {f"emg_{k}": np.nan for k in
            ["mean", "std", "min", "max", "rms",
             "median_freq", "mean_freq", "peak_count"]}

_RESP_NAN = {f"resp_{k}": np.nan for k in
             ["mean", "std", "min", "max", "range",
              "rate", "depth_mean", "depth_std", "ie_ratio"]}

_ACC_NAN = {f"acc_{k}": np.nan for k in
            ["mag_mean", "mag_std", "mag_min", "mag_max",
             "sma", "energy", "entropy"]}


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

    # ── ECG ───────────────────────────────────────────────────────────────────
    @staticmethod
    def ecg_features(ecg: np.ndarray, sr: int = 700) -> Dict:
        """Extract ECG / HRV features from an ECG segment.

        Uses R-peak detection via ``scipy.signal.find_peaks`` on bandpass-
        filtered ECG.  Returns heart-rate statistics, time- and frequency-
        domain HRV metrics.
        """
        v = ecg[~np.isnan(ecg)]
        if len(v) < sr * 2:
            return dict(_ECG_NAN)

        feat: Dict = {}
        try:
            # Bandpass 1–40 Hz for R-peak detection
            sig = _safe_filter(ecg, sr, "bandpass", (1.0, 40.0))
            if sig is None:
                raise ValueError("ECG filter failed")
            clean = sig[~np.isnan(sig)]

            # R-peak detection
            min_dist = max(int(sr * 0.3), 1)  # ≥300 ms between R-peaks
            height = np.mean(clean) + 0.5 * np.std(clean)
            peaks, _ = find_peaks(clean, distance=min_dist, height=height)
            if len(peaks) < 3:
                raise ValueError("Too few R-peaks")

            # RR intervals in ms
            rr = np.diff(peaks) / sr * 1000.0
            rr = rr[(rr >= 300) & (rr <= 1500)]  # 40–200 bpm
            if len(rr) < 2:
                raise ValueError("No plausible RR intervals")

            hr = 60_000.0 / rr
            feat.update({
                "ecg_hr_mean": np.mean(hr), "ecg_hr_std": np.std(hr),
                "ecg_hr_min": np.min(hr), "ecg_hr_max": np.max(hr),
                "ecg_hrv_sdnn": np.std(rr),
                "ecg_hrv_rmssd": np.sqrt(np.mean(np.diff(rr) ** 2)),
                "ecg_hrv_pnn50": np.sum(np.abs(np.diff(rr)) > 50) / max(len(rr) - 1, 1) * 100,
                "ecg_rr_mean": np.mean(rr),
                "ecg_rr_std": np.std(rr),
                "ecg_amplitude": np.mean(clean[peaks]) if len(peaks) else np.nan,
                "ecg_qrs_count": len(peaks),
                "ecg_qrs_rate": len(peaks) / (len(ecg) / sr) * 60,
            })

            # Frequency-domain HRV
            if len(rr) >= 10:
                feat.update(FeatureExtractor._hrv_freq_ecg(rr))
            else:
                feat.update({"ecg_hrv_lf": np.nan, "ecg_hrv_hf": np.nan,
                             "ecg_hrv_lf_hf": np.nan})

        except Exception:
            for k in _ECG_NAN:
                feat.setdefault(k, np.nan)
        return feat

    @staticmethod
    def _hrv_freq_ecg(rr_ms: np.ndarray) -> Dict:
        """LF / HF power from RR series, resampled to 4 Hz uniform grid."""
        times = np.cumsum(rr_ms) / 1000.0
        times -= times[0]
        t_uni = np.arange(0, times[-1], 0.25)  # 4 Hz
        if len(t_uni) < 8:
            return {"ecg_hrv_lf": np.nan, "ecg_hrv_hf": np.nan,
                    "ecg_hrv_lf_hf": np.nan}

        interp = interp1d(times, rr_ms, kind="linear", fill_value="extrapolate")
        uniform = interp(t_uni) - np.mean(rr_ms)

        f, pxx = welch(uniform, fs=4.0, nperseg=min(256, len(uniform)))
        lf_band = (f >= 0.04) & (f <= 0.15)
        hf_band = (f >= 0.15) & (f <= 0.4)
        lf = np.trapz(pxx[lf_band], f[lf_band]) if lf_band.any() else 0.0
        hf = np.trapz(pxx[hf_band], f[hf_band]) if hf_band.any() else 0.0
        return {"ecg_hrv_lf": lf, "ecg_hrv_hf": hf,
                "ecg_hrv_lf_hf": lf / (hf + 1e-8)}

    # ── EMG ───────────────────────────────────────────────────────────────────
    @staticmethod
    def emg_features(emg: np.ndarray, sr: int = 700) -> Dict:
        """EMG features: RMS, spectral (median/mean frequency), peak count."""
        v = emg[~np.isnan(emg)]
        if len(v) < sr:
            return dict(_EMG_NAN)

        feat: Dict = {
            "emg_mean": np.mean(v),
            "emg_std": np.std(v),
            "emg_min": np.min(v),
            "emg_max": np.max(v),
            "emg_rms": np.sqrt(np.mean(v ** 2)),
        }

        # Spectral features
        try:
            f, pxx = welch(v, fs=sr, nperseg=min(512, len(v)))
            total_power = np.sum(pxx)
            if total_power > 0:
                cum_power = np.cumsum(pxx) / total_power
                feat["emg_median_freq"] = float(f[np.searchsorted(cum_power, 0.5)])
                feat["emg_mean_freq"] = float(np.sum(f * pxx) / total_power)
            else:
                feat["emg_median_freq"] = np.nan
                feat["emg_mean_freq"] = np.nan
        except Exception:
            feat["emg_median_freq"] = np.nan
            feat["emg_mean_freq"] = np.nan

        # Activation count (threshold-based)
        thresh = np.mean(np.abs(v)) + 1.5 * np.std(np.abs(v))
        peaks, _ = find_peaks(np.abs(v), height=thresh,
                              distance=max(int(sr * 0.1), 1))
        feat["emg_peak_count"] = len(peaks)
        return feat

    # ── Respiration ───────────────────────────────────────────────────────────
    @staticmethod
    def resp_features(resp: np.ndarray, sr: int = 700) -> Dict:
        """Respiration features: rate, depth, inhalation/exhalation ratio."""
        v = resp[~np.isnan(resp)]
        if len(v) < sr * 2:
            return dict(_RESP_NAN)

        feat: Dict = {
            "resp_mean": np.mean(v),
            "resp_std": np.std(v),
            "resp_min": np.min(v),
            "resp_max": np.max(v),
            "resp_range": np.ptp(v),
        }

        try:
            # Filter for breathing frequencies (0.1–0.5 Hz → 6–30 breaths/min)
            sig = _safe_filter(resp, sr, "bandpass", (0.1, 0.5))
            if sig is None:
                raise ValueError("Resp filter failed")
            clean = sig[~np.isnan(sig)]

            # Peak detection (inhalation peaks)
            min_dist = max(int(sr * 1.5), 1)  # ≥1.5 s between breaths
            peaks, _ = find_peaks(clean, distance=min_dist)
            troughs, _ = find_peaks(-clean, distance=min_dist)

            if len(peaks) >= 2:
                breath_intervals = np.diff(peaks) / sr  # seconds
                feat["resp_rate"] = 60.0 / np.mean(breath_intervals)

                # Depth: peak-to-trough amplitude
                n_cycles = min(len(peaks), len(troughs))
                if n_cycles >= 2:
                    depths = []
                    for i in range(n_cycles):
                        depths.append(abs(clean[peaks[i]] - clean[troughs[i]]))
                    feat["resp_depth_mean"] = np.mean(depths)
                    feat["resp_depth_std"] = np.std(depths)
                else:
                    feat["resp_depth_mean"] = np.nan
                    feat["resp_depth_std"] = np.nan

                # I:E ratio (inhalation vs exhalation duration)
                if len(peaks) >= 2 and len(troughs) >= 1:
                    ie_ratios = []
                    for i in range(min(len(peaks) - 1, len(troughs))):
                        t_peak = peaks[i] / sr
                        t_trough = troughs[i] / sr if i < len(troughs) else None
                        t_next_peak = peaks[i + 1] / sr
                        if t_trough is not None and t_trough > t_peak:
                            insp = t_trough - t_peak
                            expi = t_next_peak - t_trough
                            if expi > 0:
                                ie_ratios.append(insp / expi)
                    feat["resp_ie_ratio"] = np.mean(ie_ratios) if ie_ratios else np.nan
                else:
                    feat["resp_ie_ratio"] = np.nan
            else:
                feat.update({"resp_rate": np.nan, "resp_depth_mean": np.nan,
                             "resp_depth_std": np.nan, "resp_ie_ratio": np.nan})
        except Exception:
            for k in _RESP_NAN:
                feat.setdefault(k, np.nan)
        return feat

    # ── Accelerometer ─────────────────────────────────────────────────────────
    @staticmethod
    def acc_features(acc: np.ndarray, sr: int = 4,
                     prefix: str = "acc") -> Dict:
        """Accelerometer features from magnitude signal (or 3-axis → magnitude).

        If *acc* is 2-D (N×3), magnitude is computed first.
        """
        nan_dict = {f"{prefix}_{k}": np.nan for k in
                    ["mag_mean", "mag_std", "mag_min", "mag_max",
                     "sma", "energy", "entropy"]}
        if acc is None or len(acc) == 0:
            return nan_dict

        if acc.ndim == 2 and acc.shape[1] == 3:
            mag = np.sqrt((acc ** 2).sum(axis=1)).astype(float)
        else:
            mag = acc.flatten().astype(float)

        v = mag[~np.isnan(mag)]
        if len(v) < 4:
            return nan_dict

        feat: Dict = {
            f"{prefix}_mag_mean": np.mean(v),
            f"{prefix}_mag_std": np.std(v),
            f"{prefix}_mag_min": np.min(v),
            f"{prefix}_mag_max": np.max(v),
        }

        # Signal Magnitude Area (mean absolute value)
        feat[f"{prefix}_sma"] = np.mean(np.abs(v))

        # Energy (sum of squared values / N)
        feat[f"{prefix}_energy"] = np.sum(v ** 2) / len(v)

        # Spectral entropy
        try:
            f_psd, pxx = welch(v, fs=sr, nperseg=min(64, len(v)))
            pxx_norm = pxx / (np.sum(pxx) + 1e-12)
            pxx_norm = pxx_norm[pxx_norm > 0]
            feat[f"{prefix}_entropy"] = -np.sum(pxx_norm * np.log2(pxx_norm))
        except Exception:
            feat[f"{prefix}_entropy"] = np.nan

        return feat

    # ── Chest EDA (same logic as wrist EDA with different prefix) ─────────
    @staticmethod
    def chest_eda_features(eda: np.ndarray, sr: int = 4) -> Dict:
        """Chest EDA features – mirrors wrist EDA with ``chest_eda_`` prefix."""
        raw = FeatureExtractor.eda_features(eda, sr)
        return {k.replace("eda_", "chest_eda_"): v for k, v in raw.items()}

    # ── Chest Temp ────────────────────────────────────────────────────────
    @staticmethod
    def chest_temp_features(temp: np.ndarray, sr: int = 4) -> Dict:
        """Chest temperature features with ``chest_temp_`` prefix."""
        raw = FeatureExtractor.temp_features(temp, sr)
        return {k.replace("temp_", "chest_temp_"): v for k, v in raw.items()}

    # ── windowed extraction (main entry point) ───────────────────────────
    @staticmethod
    def extract_windows(signals: Dict[str, np.ndarray],
                        sampling_rates: Dict[str, int],
                        window_sec: int = 60,
                        step_sec: int = 30) -> List[Dict]:
        """Slide a window over all signals and extract features per window.

        Automatically dispatches to the correct feature extractor based on
        the signal key name.  Supports both wrist-only keys (``eda``,
        ``bvp``, ``temp``) and ``device='both'`` prefixed keys
        (``wrist_EDA``, ``chest_ECG``, etc.).
        """
        durations = {k: len(v) / sampling_rates.get(k, 4)
                     for k, v in signals.items()}
        if not durations:
            return []
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
                low = name.lower()

                # ── wrist / generic EDA ──────────────────────────────
                if low in ("eda", "gsr", "wrist_eda"):
                    row.update(FeatureExtractor.eda_features(seg, sr))
                # ── BVP ──────────────────────────────────────────────
                elif low in ("bvp", "wrist_bvp"):
                    row.update(FeatureExtractor.bvp_features(seg, sr))
                # ── wrist / generic TEMP ─────────────────────────────
                elif low in ("temp", "temperature", "wrist_temp"):
                    row.update(FeatureExtractor.temp_features(seg, sr))
                # ── ECG ──────────────────────────────────────────────
                elif low in ("ecg", "chest_ecg"):
                    row.update(FeatureExtractor.ecg_features(seg, sr))
                # ── EMG ──────────────────────────────────────────────
                elif low in ("emg", "chest_emg"):
                    row.update(FeatureExtractor.emg_features(seg, sr))
                # ── Respiration ──────────────────────────────────────
                elif low in ("resp", "chest_resp"):
                    row.update(FeatureExtractor.resp_features(seg, sr))
                # ── Chest EDA ────────────────────────────────────────
                elif low == "chest_eda":
                    row.update(FeatureExtractor.chest_eda_features(seg, sr))
                # ── Chest Temp ───────────────────────────────────────
                elif low == "chest_temp":
                    row.update(FeatureExtractor.chest_temp_features(seg, sr))
                # ── ACC (with device prefix) ─────────────────────────
                elif low in ("wrist_acc",):
                    row.update(FeatureExtractor.acc_features(seg, sr, prefix="wrist_acc"))
                elif low in ("chest_acc",):
                    row.update(FeatureExtractor.acc_features(seg, sr, prefix="chest_acc"))
                elif low == "acc":
                    row.update(FeatureExtractor.acc_features(seg, sr, prefix="acc"))

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
            sos = scipy_signal.butter(order, wn, btype="band", output="sos")
        elif ftype == "lowpass":
            sos = scipy_signal.butter(
                order, min(cutoff, nyq * 0.95) / nyq, btype="low", output="sos"
            )
        elif ftype == "highpass":
            if cutoff >= nyq:
                return None
            sos = scipy_signal.butter(order, cutoff / nyq, btype="high",
                                      output="sos")
        else:
            return None
        out[valid] = scipy_signal.sosfiltfilt(sos, out[valid])
        out[~np.isfinite(out)] = np.nan
        return out
    except Exception:
        return None
