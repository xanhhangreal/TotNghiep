"""Tests for save/load behavior of StressModel."""

import numpy as np

from ml_models import StressModel


def test_stress_model_save_and_load_roundtrip(tmp_path):
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [0.2, 0.1],
            [1.2, 1.1],
            [0.1, 0.2],
            [1.1, 1.3],
        ],
        dtype=float,
    )
    y = np.array([0, 1, 0, 1, 0, 1], dtype=int)

    model = StressModel("logistic_regression", {"max_iter": 300, "random_state": 42})
    model.fit(X, y, verbose=False)

    out_path = tmp_path / "stress_lr.joblib"
    model.save(str(out_path))

    loaded = StressModel.load(str(out_path))
    pred_original = model.predict(X)
    pred_loaded = loaded.predict(X)

    np.testing.assert_array_equal(pred_original, pred_loaded)
