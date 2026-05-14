"""Tests for feature extraction helpers."""

import numpy as np

from features import FeatureExtractor


def test_eda_features_short_segment_returns_nan_template():
    eda = np.array([0.1, 0.2, 0.3], dtype=float)
    feat = FeatureExtractor.eda_features(eda, sr=4)

    assert feat
    assert all(k.startswith("eda_") for k in feat)
    assert all(np.isnan(v) for v in feat.values())


def test_acc_features_from_3axis_signal_has_expected_keys():
    t = np.linspace(0, 4 * np.pi, 256)
    acc = np.stack([np.sin(t), np.cos(t), 0.5 * np.sin(2 * t)], axis=1)

    feat = FeatureExtractor.acc_features(acc, sr=32, prefix="wrist_acc")

    expected = {
        "wrist_acc_mag_mean",
        "wrist_acc_mag_std",
        "wrist_acc_mag_min",
        "wrist_acc_mag_max",
        "wrist_acc_sma",
        "wrist_acc_energy",
        "wrist_acc_entropy",
    }
    assert expected.issubset(feat.keys())
    assert np.isfinite(feat["wrist_acc_mag_mean"])
    assert np.isfinite(feat["wrist_acc_sma"])


def test_extract_windows_chest_prefixed_signals():
    sr = 4
    n = 40  # 10 seconds
    signals = {
        "chest_eda": np.linspace(0.0, 1.0, n).astype(float),
        "chest_temp": np.linspace(36.0, 36.5, n).astype(float),
    }
    sampling_rates = {"chest_eda": sr, "chest_temp": sr}

    rows = FeatureExtractor.extract_windows(
        signals,
        sampling_rates,
        window_sec=4,
        step_sec=2,
    )

    assert len(rows) == 4
    row0 = rows[0]
    assert "chest_eda_mean" in row0
    assert "chest_temp_mean" in row0
    assert row0["window"] == 0
    assert row0["t0"] == 0
    assert row0["t1"] == 4
