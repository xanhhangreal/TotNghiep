"""
Feature extraction from physiological signals
"""
import numpy as np
from scipy import signal as scipy_signal
from scipy.signal import find_peaks
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extract features from physiological signals"""
    
    @staticmethod
    def extract_eda_features(eda_signal: np.ndarray, sr: int = 4) -> Dict:
        """Extract features from EDA/GSR signal
        
        Reference: Schmidt et al. 2018 (WESAD paper)
        
        Args:
            eda_signal: EDA signal array
            sr: Sampling rate (Hz)
        
        Returns:
            Dictionary of EDA features
        """
        features = {}
        
        # Handle NaN
        valid_data = eda_signal[~np.isnan(eda_signal)]
        if len(valid_data) < 2:
            return {f'eda_{k}': np.nan for k in ['mean', 'std', 'min', 'max', 'scr_peaks', 
                                                    'scr_mean_amp', 'scr_rise_time', 'scr_recovery_time']}
        
        # Basic statistics
        features['eda_mean'] = np.mean(valid_data)
        features['eda_std'] = np.std(valid_data)
        features['eda_min'] = np.min(valid_data)
        features['eda_max'] = np.max(valid_data)
        features['eda_range'] = features['eda_max'] - features['eda_min']
        
        # Phasic component (SCR) - high-pass filter
        # Use highpass at 0.05 Hz to extract phasic response
        eda_filtered = FeatureExtractor._apply_filter(eda_signal, sr, 'highpass', 0.05)
        if eda_filtered is not None:
            eda_phasic = eda_filtered[~np.isnan(eda_filtered)]
            
            # Find SCR peaks
            if len(eda_phasic) > 0:
                # Adaptive threshold based on signal amplitude
                threshold = np.std(eda_phasic) * 0.5
                peaks, properties = find_peaks(eda_phasic, height=threshold, distance=sr)
                
                features['eda_scr_peaks'] = len(peaks)  # Number of peaks
                
                if len(peaks) > 0:
                    features['eda_scr_mean_amplitude'] = np.mean(properties['peak_heights'])
                    features['eda_scr_max_amplitude'] = np.max(properties['peak_heights'])
                else:
                    features['eda_scr_mean_amplitude'] = 0
                    features['eda_scr_max_amplitude'] = 0
            else:
                features['eda_scr_peaks'] = 0
                features['eda_scr_mean_amplitude'] = 0
                features['eda_scr_max_amplitude'] = 0
        else:
            features['eda_scr_peaks'] = np.nan
            features['eda_scr_mean_amplitude'] = np.nan
            features['eda_scr_max_amplitude'] = np.nan
        
        # Tonic component (SCL) - low-pass filter
        eda_tonic = FeatureExtractor._apply_filter(eda_signal, sr, 'lowpass', 0.05)
        if eda_tonic is not None:
            valid_tonic = eda_tonic[~np.isnan(eda_tonic)]
            if len(valid_tonic) > 0:
                features['eda_scl_mean'] = np.mean(valid_tonic)
            else:
                features['eda_scl_mean'] = np.nan
        else:
            features['eda_scl_mean'] = np.nan
        
        return features
    
    @staticmethod
    def extract_bvp_features(bvp_signal: np.ndarray, sr: int = 64) -> Dict:
        """Extract features from BVP/PPG signal
        
        Reference: Schmidt et al. 2018 (WESAD paper)
        
        Args:
            bvp_signal: BVP signal array
            sr: Sampling rate (Hz)
        
        Returns:
            Dictionary of BVP/HRV features
        """
        features = {}
        
        # Handle NaN
        valid_data = bvp_signal[~np.isnan(bvp_signal)]
        if len(valid_data) < 2:
            return {f'bvp_{k}': np.nan for k in ['mean', 'std', 'hr_mean', 'hr_std', 
                                                    'hrv_sdnn', 'hrv_rmssd', 'hrv_lf', 'hrv_hf']}
        
        # Basic statistics
        features['bvp_mean'] = np.mean(valid_data)
        features['bvp_std'] = np.std(valid_data)
        features['bvp_min'] = np.min(valid_data)
        features['bvp_max'] = np.max(valid_data)
        
        # Extract IBI (Inter-Beat Interval) from BVP signal
        # Find local maxima (peaks) which correspond to heartbeats
        try:
            # Bandpass filter 0.7-4 Hz (normal heartbeat range)
            bvp_filtered = FeatureExtractor._apply_filter(bvp_signal, sr, 'bandpass', (0.7, 4.0))
            if bvp_filtered is not None:
                bvp_filtered = bvp_filtered[~np.isnan(bvp_filtered)]
                
                # Find peaks
                peaks, _ = find_peaks(bvp_filtered, distance=sr//4)  # Min distance between beats
                
                if len(peaks) >= 2:
                    # Calculate IBI (convert to seconds, then to ms)
                    ibi = np.diff(peaks) / sr * 1000  # milliseconds
                    
                    # Convert IBI to heart rate
                    hr = 60000 / ibi  # bpm
                    
                    features['bvp_hr_mean'] = np.mean(hr)
                    features['bvp_hr_std'] = np.std(hr)
                    features['bvp_hr_min'] = np.min(hr)
                    features['bvp_hr_max'] = np.max(hr)
                    
                    # HRV - Time domain
                    features['bvp_hrv_sdnn'] = np.std(ibi)  # Standard deviation of IBI
                    features['bvp_hrv_rmssd'] = np.sqrt(np.mean(np.diff(ibi)**2))  # Root mean square of successive differences
                    
                    # HRV - Frequency domain
                    try:
                        # Compute power spectral density using Welch method
                        f, pxx = scipy_signal.welch(ibi, fs=1000/np.mean(ibi), nperseg=min(256, len(ibi)))
                        
                        # LF (0.04-0.15 Hz) and HF (0.15-0.4 Hz) bands
                        lf_mask = (f >= 0.04) & (f <= 0.15)
                        hf_mask = (f >= 0.15) & (f <= 0.4)
                        
                        lf_power = np.sum(pxx[lf_mask])
                        hf_power = np.sum(pxx[hf_mask])
                        
                        features['bvp_hrv_lf'] = lf_power
                        features['bvp_hrv_hf'] = hf_power
                        features['bvp_hrv_lf_hf_ratio'] = lf_power / (hf_power + 1e-8)
                    except:
                        features['bvp_hrv_lf'] = np.nan
                        features['bvp_hrv_hf'] = np.nan
                        features['bvp_hrv_lf_hf_ratio'] = np.nan
                else:
                    # Not enough peaks to extract HRV
                    features['bvp_hr_mean'] = np.nan
                    features['bvp_hr_std'] = np.nan
                    features['bvp_hr_min'] = np.nan
                    features['bvp_hr_max'] = np.nan
                    features['bvp_hrv_sdnn'] = np.nan
                    features['bvp_hrv_rmssd'] = np.nan
                    features['bvp_hrv_lf'] = np.nan
                    features['bvp_hrv_hf'] = np.nan
                    features['bvp_hrv_lf_hf_ratio'] = np.nan
            else:
                raise Exception("Filter failed")
        except Exception as e:
            logger.warning(f"BVP feature extraction failed: {e}")
            features['bvp_hr_mean'] = np.nan
            features['bvp_hr_std'] = np.nan
            features['bvp_hr_min'] = np.nan
            features['bvp_hr_max'] = np.nan
            features['bvp_hrv_sdnn'] = np.nan
            features['bvp_hrv_rmssd'] = np.nan
            features['bvp_hrv_lf'] = np.nan
            features['bvp_hrv_hf'] = np.nan
            features['bvp_hrv_lf_hf_ratio'] = np.nan
        
        return features
    
    @staticmethod
    def extract_temperature_features(temp_signal: np.ndarray, sr: int = 4) -> Dict:
        """Extract features from skin temperature signal
        
        Args:
            temp_signal: Temperature signal array
            sr: Sampling rate (Hz)
        
        Returns:
            Dictionary of temperature features
        """
        features = {}
        
        # Handle NaN
        valid_data = temp_signal[~np.isnan(temp_signal)]
        if len(valid_data) < 2:
            return {f'temp_{k}': np.nan for k in ['mean', 'std', 'min', 'max', 'slope']}
        
        # Basic statistics
        features['temp_mean'] = np.mean(valid_data)
        features['temp_std'] = np.std(valid_data)
        features['temp_min'] = np.min(valid_data)
        features['temp_max'] = np.max(valid_data)
        features['temp_range'] = features['temp_max'] - features['temp_min']
        
        # Slope (rate of change)
        # Linear regression of temperature over time
        if len(valid_data) > 2:
            x = np.arange(len(valid_data))
            coeffs = np.polyfit(x, valid_data, 1)
            features['temp_slope'] = coeffs[0]  # Rate of temperature change per sample
            features['temp_slope_permin'] = coeffs[0] * sr * 60  # Per minute
        else:
            features['temp_slope'] = np.nan
            features['temp_slope_permin'] = np.nan
        
        return features
    
    @staticmethod
    def _apply_filter(signal: np.ndarray, sr: int, 
                     filter_type: str = "lowpass", 
                     cutoff_freq: float = 1.0, 
                     filter_order: int = 4) -> Optional[np.ndarray]:
        """Apply digital filter to signal
        
        Args:
            signal: Input signal
            sr: Sampling rate (Hz)
            filter_type: 'lowpass', 'highpass', 'bandpass'
            cutoff_freq: Cutoff frequency (Hz) or tuple for bandpass
            filter_order: Filter order
        
        Returns:
            Filtered signal or None if filter fails
        """
        try:
            signal = signal.copy()
            valid_idx = ~np.isnan(signal)
            
            if not valid_idx.any():
                return None
            
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
                return None
            
            # Apply filter
            filtered_signal = signal.copy()
            filtered_signal[valid_idx] = scipy_signal.filtfilt(b, a, signal[valid_idx])
            
            return filtered_signal
        except Exception as e:
            logger.warning(f"Filter application failed: {e}")
            return None
    
    @staticmethod
    def extract_all_features(eda_signal: np.ndarray,
                           bvp_signal: np.ndarray,
                           temp_signal: np.ndarray,
                           sr_eda: int = 4,
                           sr_bvp: int = 64,
                           sr_temp: int = 4) -> Dict:
        """Extract all features from physiological signals
        
        Args:
            eda_signal: EDA signal
            bvp_signal: BVP signal
            temp_signal: Temperature signal
            sr_eda: EDA sampling rate
            sr_bvp: BVP sampling rate
            sr_temp: Temperature sampling rate
        
        Returns:
            Combined feature dictionary
        """
        all_features = {}
        
        # Extract from each signal
        all_features.update(FeatureExtractor.extract_eda_features(eda_signal, sr_eda))
        all_features.update(FeatureExtractor.extract_bvp_features(bvp_signal, sr_bvp))
        all_features.update(FeatureExtractor.extract_temperature_features(temp_signal, sr_temp))
        
        return all_features


# Convenience function
def extract_features_from_window(window_dict: Dict[str, np.ndarray],
                                sr_dict: Dict[str, int]) -> Dict:
    """Extract features from a window of physiological data
    
    Args:
        window_dict: Dictionary with keys 'eda', 'bvp', 'temp' containing signal windows
        sr_dict: Dictionary with sampling rates
    
    Returns:
        Feature dictionary
    """
    extractor = FeatureExtractor()
    
    eda_features = extractor.extract_eda_features(
        window_dict.get('eda', np.array([])),
        sr_dict.get('eda', 4)
    )
    
    bvp_features = extractor.extract_bvp_features(
        window_dict.get('bvp', np.array([])),
        sr_dict.get('bvp', 64)
    )
    
    temp_features = extractor.extract_temperature_features(
        window_dict.get('temp', np.array([])),
        sr_dict.get('temp', 4)
    )
    
    all_features = {**eda_features, **bvp_features, **temp_features}
    
    return all_features
