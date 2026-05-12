"""Tests for label mapping helpers in wesad_loader.py."""

import numpy as np

from wesad_loader import to_3class, to_binary


def test_to_binary_mapping():
    labels = np.array([0, 1, 2, 3, 4, 2, 1])
    mapped, valid = to_binary(labels)

    assert mapped.tolist() == [-1, 0, 1, 0, -1, 1, 0]
    assert valid.tolist() == [False, True, True, True, False, True, True]


def test_to_3class_mapping():
    labels = np.array([0, 1, 2, 3, 4, 3, 2, 1])
    mapped, valid = to_3class(labels)

    assert mapped.tolist() == [-1, 0, 1, 2, -1, 2, 1, 0]
    assert valid.tolist() == [False, True, True, True, False, True, True, True]
