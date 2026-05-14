"""Tests for preprocessing utilities."""

import numpy as np

from preprocessing import SignalPreprocessor, preprocess_wesad_signal


def test_resample_all_nan_returns_nan_output():
    pp = SignalPreprocessor(target_sr=2)
    x = np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)

    y = pp.resample(x, original_sr=4, target_sr=2)

    assert y.shape == (2,)
    assert np.isnan(y).all()


def test_fillna_methods_forward_and_interpolate():
    pp = SignalPreprocessor()
    x = np.array([np.nan, 1.0, np.nan, 3.0, np.nan], dtype=float)

    y_forward = pp.fillna(x, method="forward")
    y_interp = pp.fillna(x, method="interpolate")

    np.testing.assert_allclose(y_forward, np.array([1.0, 1.0, 1.0, 3.0, 3.0]))
    np.testing.assert_allclose(y_interp, np.array([1.0, 1.0, 2.0, 3.0, 3.0]))


def test_preprocess_wesad_signal_acc_and_bvp_paths_produce_finite_arrays():
    sr = {"wrist_acc": 32, "wrist_bvp": 64, "wrist_eda": 4}
    signals = {
        "wrist_acc": np.tile(np.array([[1.0, 0.0, 0.0]]), (320, 1)),
        "wrist_bvp": np.sin(np.linspace(0, 20 * np.pi, 640)).astype(float),
        "wrist_eda": np.linspace(0.0, 1.0, 40).astype(float),
    }

    out = preprocess_wesad_signal(signals, sr, target_sr=4)

    assert set(["wrist_acc", "wrist_bvp", "wrist_eda"]).issubset(out.keys())
    assert out["wrist_acc"].ndim == 1
    assert out["wrist_bvp"].ndim == 1
    assert len(out["wrist_bvp"]) == 40
    assert np.isfinite(out["wrist_acc"]).all()
    assert np.isfinite(out["wrist_bvp"]).all()
