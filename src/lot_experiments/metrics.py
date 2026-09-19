"""Small target-aware numerical metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def l1_error(first: ArrayLike, second: ArrayLike) -> float:
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if first_array.shape != second_array.shape:
        raise ValueError("metric inputs must have identical shapes")
    return float(np.abs(first_array - second_array).sum())


def assert_probability(values: ArrayLike, *, tolerance: float = 1e-12) -> None:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or np.min(vector, initial=0.0) < -tolerance:
        raise AssertionError("values are not nonnegative probabilities")
    if not np.isclose(vector.sum(), 1.0, atol=tolerance, rtol=0.0):
        raise AssertionError("probabilities do not sum to one")

