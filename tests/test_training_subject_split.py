"""Tests for subject-level split behavior in subject-independent training."""

import numpy as np

import training


def test_subject_independent_split_has_no_leakage(monkeypatch, tmp_path):
    def fake_extract_subject_features(subject, window_sec=60, step_sec=30):
        sid = subject["subject_id"]
        X = np.array(
            [
                [sid * 1.0, 0.0],
                [sid * 1.0 + 0.1, 1.0],
                [sid * 1.0 + 0.2, 0.0],
                [sid * 1.0 + 0.3, 1.0],
            ],
            dtype=float,
        )
        y = np.array([0, 1, 0, 1], dtype=int)
        return X, y, ["f1", "f2"]

    monkeypatch.setattr(
        training, "extract_subject_features", fake_extract_subject_features
    )
    monkeypatch.setattr(
        training,
        "DEFAULT_MODELS",
        {"logistic_regression": {"max_iter": 200, "random_state": 42}},
    )
    monkeypatch.setattr(training, "MODELS_DIR", tmp_path)

    data = {"subjects": [{"subject_id": sid} for sid in [2, 3, 4, 5, 6]]}
    results, feat_names = training.train_subject_independent(data, test_ratio=0.4)

    train_subjects = set(results["train_subjects"])
    test_subjects = set(results["test_subjects"])

    assert train_subjects
    assert test_subjects
    assert train_subjects.isdisjoint(test_subjects)
    assert train_subjects.union(test_subjects) == {2, 3, 4, 5, 6}
    assert feat_names == ["f1", "f2"]
    assert "logistic_regression" in results["models"]
    assert (tmp_path / "logistic_regression_independent.joblib").exists()
