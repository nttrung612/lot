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


def sharp_pruning_error_from_masses(
    retained_mass: float, omitted_mass: float, span: float, T0: float
) -> float:
    """Stable sharp error using both masses instead of forming ``1 - alpha``.

    This is algebraically identical to :func:`sharp_pruning_error`, but remains
    finite when a tiny positive retained mass is below float64 resolution
    relative to one, so that ``1 - retained_mass`` rounds to exactly one.
    """

    if (
        retained_mass <= 0.0
        or omitted_mass < 0.0
        or not math.isfinite(retained_mass)
        or not math.isfinite(omitted_mass)
    ):
        raise ValueError("retained mass must be positive and omitted mass nonnegative")
    if not math.isclose(
        retained_mass + omitted_mass, 1.0, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("retained and omitted masses must sum to one")
    if span < 0.0 or not math.isfinite(span) or T0 <= 0.0 or not math.isfinite(T0):
        raise ValueError("span must be finite/nonnegative and T0 finite/positive")
    if omitted_mass == 0.0:
        return 0.0
    log_ratio = span / T0 + math.log(omitted_mass) - math.log(retained_mass)
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


def poisson_backup_error_bound(beta_r: float, span: float, T0: float) -> float:
    """Return the sharp heat-mixture backup envelope ``delta_r(span)``.

    For the normalized Poisson head, the exact heat column decomposes as
    ``(1 - beta_r) * H_tr + beta_r * R_r``.  Equation (34) of the paper then
    bounds the absolute one-backup error by

    ``T0 * log(1 + beta_r * (exp(span / T0) - 1))``.

    The log-mixture form below remains stable when ``span / T0`` is large.
    """

    if not 0.0 <= beta_r <= 1.0 or not math.isfinite(beta_r):
        raise ValueError("beta_r must lie in [0, 1]")
    if span < 0.0 or not math.isfinite(span) or T0 <= 0.0 or not math.isfinite(T0):
        raise ValueError("span must be finite/nonnegative and T0 finite/positive")
    if beta_r == 0.0 or span == 0.0:
        return 0.0
    if beta_r == 1.0:
        return float(span)
    log_value = float(
        np.logaddexp(math.log1p(-beta_r), math.log(beta_r) + span / T0)
    )
    return T0 * log_value
