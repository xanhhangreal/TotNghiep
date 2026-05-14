"""Tests for raw-signal window extraction."""

import numpy as np

from raw_signal import extract_subject_raw_windows


def test_extract_subject_raw_windows_wrist_shape_and_labels():
    duration_sec = 10
    labels_4hz = np.concatenate([
        np.zeros(20, dtype=int),
        np.ones(20, dtype=int),
    ])
    valid = np.ones_like(labels_4hz, dtype=bool)

    subject = {
        "subject_id": 1,
        "signals": {
            "ACC": np.tile(np.array([[1.0, 0.0, 0.0]]), (32 * duration_sec, 1)),
            "BVP": np.sin(np.linspace(0, 20 * np.pi, 64 * duration_sec)).astype(float),
            "EDA": np.linspace(0.0, 1.0, 4 * duration_sec).astype(float),
            "TEMP": np.linspace(36.2, 36.8, 4 * duration_sec).astype(float),
        },
        "sampling_rates": {
            "ACC": 32,
            "BVP": 64,
            "EDA": 4,
            "TEMP": 4,
        },
        "binary_labels": labels_4hz,
        "valid_mask": valid,
    }

    X, y, channels = extract_subject_raw_windows(
        subject,
        device_mode="wrist",
        window_sec=4,
        step_sec=2,
        target_sr=8,
    )

    assert channels == ["acc", "bvp", "eda", "temp"]
    assert X.shape == (4, 4, 32)
    assert y.shape == (4,)
    assert set(np.unique(y)).issubset({0, 1})
    assert np.isfinite(X).all()
