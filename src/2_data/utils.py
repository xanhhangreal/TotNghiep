"""
Utility functions for stress detection project
"""
import logging
import os
from pathlib import Path
from typing import Dict, Tuple, List
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# Setup logging
def setup_logging(log_level="INFO", log_file=None):
    """Setup logging configuration"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    if log_file:
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format,
        )

def get_logger(name):
    """Get logger instance"""
    return logging.getLogger(name)

logger = get_logger(__name__)

# Load/Save utilities
def load_pickle(filepath: str):
    """Load pickle file with Python 2/3 compatibility"""
    with open(filepath, 'rb') as f:
        return pickle.load(f, encoding='latin1')

def save_pickle(obj, filepath: str):
    """Save object as pickle"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    logger.info(f"Saved to {filepath}")

def load_csv(filepath: str, **kwargs) -> pd.DataFrame:
    """Load CSV file"""
    return pd.read_csv(filepath, **kwargs)

def save_csv(data: pd.DataFrame, filepath: str, **kwargs):
    """Save DataFrame to CSV"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data.to_csv(filepath, **kwargs)
    logger.info(f"Saved to {filepath}")

# Data validation
def validate_signal(signal: np.ndarray, name: str = "signal") -> bool:
    """Validate signal array"""
    if signal is None or len(signal) == 0:
        logger.warning(f"{name} is empty or None")
        return False
    if np.isnan(signal).all():
        logger.warning(f"{name} contains only NaN values")
        return False
    return True

def check_missing_values(data: np.ndarray, threshold: float = 0.5) -> float:
    """Check percentage of missing values"""
    if len(data) == 0:
        return 1.0
    missing_pct = np.isnan(data).sum() / len(data)
    if missing_pct > threshold:
        logger.warning(f"Missing values: {missing_pct*100:.1f}%")
    return missing_pct

# Time utilities
def get_timestamp():
    """Get current timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def seconds_to_formatted(seconds: int) -> str:
    """Convert seconds to HH:MM:SS format"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# Array utilities
def sliding_window(data: np.ndarray, window_size: int, step: int) -> np.ndarray:
    """Create sliding windows from data
    
    Args:
        data: Input array (1D or 2D)
        window_size: Size of window
        step: Step size
    
    Returns:
        Array of windows shape (n_windows, window_size) or (n_windows, window_size, n_features)
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    n_windows = (len(data) - window_size) // step + 1
    windows = []
    
    for i in range(n_windows):
        start = i * step
        end = start + window_size
        windows.append(data[start:end])
    
    return np.array(windows)

def normalize(data: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Normalize data
    
    Args:
        data: Input array
        method: "zscore" or "minmax"
    """
    if method == "zscore":
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        std[std == 0] = 1  # Avoid division by zero
        return (data - mean) / std
    elif method == "minmax":
        data_min = np.nanmin(data, axis=0)
        data_max = np.nanmax(data, axis=0)
        data_range = data_max - data_min
        data_range[data_range == 0] = 1
        return (data - data_min) / data_range
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def fillna(data: np.ndarray, method: str = "forward") -> np.ndarray:
    """Fill NaN values
    
    Args:
        data: Input array
        method: "forward", "backward", or "interpolate"
    """
    data = data.copy()
    
    if method == "forward":
        mask = np.isnan(data)
        idx = np.where(~mask, np.arange(mask.size), 0)
        np.maximum.accumulate(idx, axis=0, out=idx)
        return data[idx]
    elif method == "backward":
        mask = np.isnan(data)
        idx = np.where(~mask, np.arange(mask.size), mask.size-1)
        idx = np.minimum.accumulate(idx[::-1], axis=0)[::-1]
        return data[idx]
    elif method == "interpolate":
        from scipy.interpolate import interpolate
        nans = np.isnan(data)
        x = lambda z: z.nonzero()[0]
        data[nans] = np.interp(x(nans), x(~nans), data[~nans])
        return data
    else:
        raise ValueError(f"Unknown fillna method: {method}")

# Stats utilities
def describe_signal(signal: np.ndarray, name: str = "") -> Dict:
    """Get descriptive statistics of signal"""
    valid_data = signal[~np.isnan(signal)]
    
    if len(valid_data) == 0:
        return None
    
    stats = {
        'name': name,
        'count': len(valid_data),
        'mean': np.mean(valid_data),
        'std': np.std(valid_data),
        'min': np.min(valid_data),
        'max': np.max(valid_data),
        'median': np.median(valid_data),
        'missing': np.isnan(signal).sum(),
        'missing_pct': np.isnan(signal).sum() / len(signal) * 100,
    }
    return stats

def print_signal_summary(signals_dict: Dict[str, np.ndarray]):
    """Print summary statistics for multiple signals"""
    print("\n" + "="*80)
    print("SIGNAL SUMMARY STATISTICS")
    print("="*80)
    
    for name, signal in signals_dict.items():
        stats = describe_signal(signal, name)
        if stats:
            print(f"\n{name.upper()}:")
            print(f"  Count: {stats['count']}")
            print(f"  Mean: {stats['mean']:.4f} | Std: {stats['std']:.4f}")
            print(f"  Min: {stats['min']:.4f} | Max: {stats['max']:.4f}")
            print(f"  Median: {stats['median']:.4f}")
            print(f"  Missing: {stats['missing']} ({stats['missing_pct']:.2f}%)")

# Label utilities
def balance_labels(data: np.ndarray, labels: np.ndarray, 
                  method: str = "oversample") -> Tuple[np.ndarray, np.ndarray]:
    """Balance imbalanced labels
    
    Args:
        data: Features array
        labels: Label array
        method: "oversample" or "undersample"
    
    Returns:
        Balanced (data, labels)
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    if method == "oversample":
        max_count = np.max(counts)
        balanced_data = []
        balanced_labels = []
        
        for label in unique:
            mask = labels == label
            class_data = data[mask]
            
            # Oversample to max_count
            indices = np.random.choice(len(class_data), size=max_count, replace=True)
            balanced_data.append(class_data[indices])
            balanced_labels.extend([label] * max_count)
        
        return np.vstack(balanced_data), np.array(balanced_labels)
    
    elif method == "undersample":
        min_count = np.min(counts)
        balanced_data = []
        balanced_labels = []
        
        for label in unique:
            mask = labels == label
            class_data = data[mask]
            
            # Undersample to min_count
            indices = np.random.choice(len(class_data), size=min_count, replace=False)
            balanced_data.append(class_data[indices])
            balanced_labels.extend([label] * min_count)
        
        return np.vstack(balanced_data), np.array(balanced_labels)
    
    else:
        raise ValueError(f"Unknown balance method: {method}")

logger.info("Utilities loaded successfully")
