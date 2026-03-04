"""
Data preprocessing pipeline for stress detection
"""
import numpy as np
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d
import pandas as pd
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class SignalPreprocessor:
    """Preprocess physiological signals"""
    
    def __init__(self, target_sr: int = 4):
        """
        Args:
            target_sr: Target sampling rate (Hz)
        """
        self.target_sr = target_sr
    
    def resample(self, signal: np.ndarray, original_sr: int, target_sr: int = None) -> np.ndarray:
        """Resample signal to target sampling rate
        
        Args:
            signal: Input signal
            original_sr: Original sampling rate (Hz)
            target_sr: Target sampling rate (Hz), uses self.target_sr if None
        
        Returns:
            Resampled signal
        """
        if target_sr is None:
            target_sr = self.target_sr
        
        if original_sr == target_sr:
            return signal
        
        # Handle NaN values
        valid_idx = ~np.isnan(signal)
        if not valid_idx.any():
            return np.full(int(len(signal) * target_sr / original_sr), np.nan)
        
        # Create time arrays
        t_original = np.arange(len(signal)) / original_sr
        t_target = np.arange(int(len(signal) * target_sr / original_sr)) / target_sr
        
        # Interpolate
        f = interp1d(t_original, signal, kind='linear', fill_value='extrapolate')
        resampled = f(t_target)
        
        return resampled
    
    def remove_artifacts(self, signal: np.ndarray, z_score_threshold: float = 3.0) -> np.ndarray:
        """Remove outliers/artifacts using z-score method
        
        Args:
            signal: Input signal
            z_score_threshold: Z-score threshold for outlier detection
        
        Returns:
            Signal with artifacts replaced by NaN
        """
        signal = signal.copy()
        valid_data = signal[~np.isnan(signal)]
        
        if len(valid_data) < 2:
            return signal
        
        mean = np.mean(valid_data)
        std = np.std(valid_data)
        
        z_scores = np.abs((signal - mean) / (std + 1e-8))
        artifacts = z_scores > z_score_threshold
        
        signal[artifacts] = np.nan
        logger.debug(f"Removed {np.sum(artifacts)} artifacts")
        
        return signal
    
    def apply_filter(self, signal: np.ndarray, sr: int, 
                    filter_type: str = "lowpass", 
                    cutoff_freq: float = 1.0, 
                    filter_order: int = 4) -> np.ndarray:
        """Apply digital filter to signal
        
        Args:
            signal: Input signal
            sr: Sampling rate (Hz)
            filter_type: 'lowpass', 'highpass', 'bandpass'
            cutoff_freq: Cutoff frequency (Hz) or tuple for bandpass
            filter_order: Filter order
        
        Returns:
            Filtered signal
        """
        signal = signal.copy()
        valid_idx = ~np.isnan(signal)
        
        if not valid_idx.any():
            return signal
        
        try:
            # Design filter
            nyquist = sr / 2
            
            if filter_type == "lowpass":
                normalized_cutoff = cutoff_freq / nyquist
                b, a = scipy_signal.butter(filter_order, normalized_cutoff, btype='low')
            elif filter_type == "highpass":
                normalized_cutoff = cutoff_freq / nyquist
                b, a = scipy_signal.butter(filter_order, normalized_cutoff, btype='high')
            elif filter_type == "bandpass":
                normalized_cutoff = [f / nyquist for f in cutoff_freq]
                b, a = scipy_signal.butter(filter_order, normalized_cutoff, btype='band')
            else:
                raise ValueError(f"Unknown filter type: {filter_type}")
            
            # Apply filter only to valid data
            filtered_signal = signal.copy()
            filtered_signal[valid_idx] = scipy_signal.filtfilt(b, a, signal[valid_idx])
            
            return filtered_signal
        
        except Exception as e:
            logger.warning(f"Filter failed: {e}. Returning original signal.")
            return signal
    
    def normalize(self, signal: np.ndarray, method: str = "zscore") -> np.ndarray:
        """Normalize signal
        
        Args:
            signal: Input signal
            method: 'zscore' or 'minmax'
        
        Returns:
            Normalized signal
        """
        signal = signal.copy()
        valid_data = signal[~np.isnan(signal)]
        
        if len(valid_data) == 0:
            return signal
        
        if method == "zscore":
            mean = np.mean(valid_data)
            std = np.std(valid_data)
            std = std if std > 0 else 1
            signal[~np.isnan(signal)] = (signal[~np.isnan(signal)] - mean) / std
        
        elif method == "minmax":
            data_min = np.min(valid_data)
            data_max = np.max(valid_data)
            data_range = data_max - data_min
            data_range = data_range if data_range > 0 else 1
            signal[~np.isnan(signal)] = (signal[~np.isnan(signal)] - data_min) / data_range
        
        return signal
    
    def fillna(self, signal: np.ndarray, method: str = "interpolate", limit: int = 10) -> np.ndarray:
        """Fill NaN values
        
        Args:
            signal: Input signal
            method: 'forward', 'backward', 'interpolate', 'mean'
            limit: Max consecutive NaN to fill
        
        Returns:
            Signal with NaN values filled
        """
        signal = signal.copy()
        
        if method == "forward":
            mask = np.isnan(signal)
            idx = np.where(~mask, np.arange(mask.size), 0)
            np.maximum.accumulate(idx, axis=0, out=idx)
            signal = signal[idx]
        
        elif method == "backward":
            mask = np.isnan(signal)
            idx = np.where(~mask, np.arange(mask.size), mask.size-1)
            idx = np.minimum.accumulate(idx[::-1], axis=0)[::-1]
            signal = signal[idx]
        
        elif method == "interpolate":
            nans = np.isnan(signal)
            x = lambda z: z.nonzero()[0]
            
            if nans.any() and (~nans).any():
                signal[nans] = np.interp(x(nans), x(~nans), signal[~nans])
        
        elif method == "mean":
            valid_data = signal[~np.isnan(signal)]
            if len(valid_data) > 0:
                signal[np.isnan(signal)] = np.mean(valid_data)
        
        return signal
    
    def preprocess_pipeline(self, signal: np.ndarray, sr: int,
                          remove_outliers: bool = True,
                          lowpass_freq: Optional[float] = None,
                          normalize: bool = True) -> np.ndarray:
        """Complete preprocessing pipeline
        
        Args:
            signal: Input signal
            sr: Sampling rate
            remove_outliers: Whether to remove outliers
            lowpass_freq: Lowpass filter frequency (None to skip)
            normalize: Whether to normalize signal
        
        Returns:
            Preprocessed signal
        """
        # Remove outliers
        if remove_outliers:
            signal = self.remove_artifacts(signal)
        
        # Apply lowpass filter
        if lowpass_freq is not None:
            signal = self.apply_filter(signal, sr, 'lowpass', lowpass_freq)
        
        # Fill NaN values
        signal = self.fillna(signal, method='interpolate')
        
        # Normalize
        if normalize:
            signal = self.normalize(signal, method='zscore')
        
        return signal


def preprocess_wesad_signal(signal_dict: Dict[str, np.ndarray], 
                           sr_dict: Dict[str, int],
                           target_sr: int = 4) -> Dict[str, np.ndarray]:
    """Preprocess WESAD physiological signals
    
    Args:
        signal_dict: Dictionary of signals (eda, bvp, temp, etc.)
        sr_dict: Dictionary of sampling rates
        target_sr: Target sampling rate (Hz)
    
    Returns:
        Dictionary of preprocessed signals
    """
    preprocessor = SignalPreprocessor(target_sr=target_sr)
    preprocessed = {}
    
    for signal_name, signal in signal_dict.items():
        if signal is None or len(signal) == 0:
            continue
        
        sr = sr_dict.get(signal_name, target_sr)
        
        logger.info(f"Preprocessing {signal_name} (SR: {sr} Hz)")
        
        # Resample to target SR
        resampled = preprocessor.resample(signal, sr, target_sr)
        
        # Apply signal-specific preprocessing
        if signal_name in ['eda', 'gsr']:
            # EDA: lowpass at 5 Hz
            processed = preprocessor.preprocess_pipeline(resampled, target_sr, 
                                                        lowpass_freq=5.0)
        elif signal_name in ['bvp']:
            # BVP: bandpass 0.7-4 Hz
            processed = preprocessor.apply_filter(resampled, target_sr, 
                                                 'bandpass', (0.7, 4.0))
            processed = preprocessor.fillna(processed)
            processed = preprocessor.normalize(processed)
        elif signal_name in ['temp', 'temperature']:
            # Temperature: lowpass at 1 Hz
            processed = preprocessor.preprocess_pipeline(resampled, target_sr,
                                                        lowpass_freq=1.0)
        else:
            processed = preprocessor.preprocess_pipeline(resampled, target_sr)
        
        preprocessed[signal_name] = processed
    
    return preprocessed
