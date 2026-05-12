"""Tests for window label assignment in training.extract_subject_features."""

import numpy as np

import training


def test_extract_subject_features_discards_low_valid_windows(monkeypatch):
    # Two windows: first valid, second below 80% valid-label threshold.
    fake_rows = [
        {"window": 0, "t0": 0.0, "t1": 60.0, "f_eda": 1.0, "f_temp": 2.0},
        {"window": 1, "t0": 60.0, "t1": 120.0, "f_eda": 1.1, "f_temp": 2.1},
    ]

    monkeypatch.setattr(
        training,
        "preprocess_wesad_signal",
        lambda signal_dict, sr_dict, target_sr=4: signal_dict,
    )
    monkeypatch.setattr(
        training.FeatureExtractor,
        "extract_windows",
        staticmethod(lambda signals, srs, window_sec, step_sec: fake_rows),
    )

    labels = np.concatenate(
        [
            np.zeros(240, dtype=int),  # 60s * 4Hz
            np.ones(240, dtype=int),
        ]
    )
    valid = np.concatenate(
        [
            np.ones(240, dtype=bool),  # 100% valid
            np.zeros(60, dtype=bool),  # 75% valid for window 2
            np.ones(180, dtype=bool),
        ]
    )

    subject = {
        "subject_id": 99,
        "signals": {"EDA": np.linspace(0, 1, 480)},
        "sampling_rates": {"EDA": 4},
        "binary_labels": labels,
        "valid_mask": valid,
    }

    X, y, feat_names = training.extract_subject_features(
        subject, window_sec=60, step_sec=30
    )

    assert X.shape == (1, 2)
    assert y.tolist() == [0]
    assert feat_names == ["f_eda", "f_temp"]
