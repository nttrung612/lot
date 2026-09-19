"""Deterministic top-mass pruning and its sharp certificate."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


BoolArray = NDArray[np.bool_]


def top_mass_mask(weights: ArrayLike, alpha: float) -> BoolArray:
    vector = np.asarray(weights, dtype=np.float64)
    if vector.ndim != 1 or np.any(vector < 0.0) or not np.isclose(vector.sum(), 1.0):
        raise ValueError("weights must be a probability vector")
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must lie in [0, 1)")
    order = np.argsort(-vector, kind="stable")
    cumulative = np.cumsum(vector[order])
    count = int(np.searchsorted(cumulative, 1.0 - alpha, side="left") + 1)
    mask = np.zeros(len(vector), dtype=bool)
    mask[order[:count]] = True
    return mask


def sharp_pruning_error(alpha: float, span: float, T0: float) -> float:
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must lie in [0, 1)")
    if span < 0.0 or T0 <= 0.0:
        raise ValueError("span must be nonnegative and T0 positive")
    if alpha == 0.0:
        return 0.0
    log_ratio = span / T0 + math.log(alpha) - math.log1p(-alpha)
    return T0 * float(np.logaddexp(0.0, log_ratio))


def admissible_omitted_mass(span: float, tolerance: float, T0: float) -> float:
    if span < 0.0 or tolerance < 0.0 or T0 <= 0.0:
        raise ValueError("span/tolerance must be nonnegative and T0 positive")
    if tolerance == 0.0:
        return 0.0
    numerator = math.expm1(tolerance / T0)
    log_numerator = math.log(numerator)
    log_denominator = float(np.logaddexp(span / T0, log_numerator))
    return math.exp(log_numerator - log_denominator)
