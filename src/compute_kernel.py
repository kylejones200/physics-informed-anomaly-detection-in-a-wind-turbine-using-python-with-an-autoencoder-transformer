"""Reconstruction error magnitudes for anomaly detection."""

from __future__ import annotations

import numpy as np


def reconstruction_errors(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    return np.abs(a - p)
