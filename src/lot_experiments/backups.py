"""Stable full, pruned, cost-based, and direct-heat LOT backups."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BackupResult:
    value: float
    policy: FloatArray
    anchor_policies: FloatArray


def _probability_vector(values: ArrayLike, name: str) -> FloatArray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector")
    if np.min(vector, initial=0.0) < -1e-14:
        raise ValueError(f"{name} must be nonnegative")
    vector = np.maximum(vector, 0.0)
    total = float(vector.sum())
    if total <= 0.0 or not np.isclose(total, 1.0, atol=1e-12, rtol=1e-12):
        raise ValueError(f"{name} must sum to one")
    return vector / total


def _kernel_matrix(weights: ArrayLike, K: int) -> FloatArray:
    matrix = np.asarray(weights, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[0] != K:
        raise ValueError("weights must have shape (candidate actions, anchors)")
    if not np.all(np.isfinite(matrix)) or np.min(matrix, initial=0.0) < -1e-14:
        raise ValueError("weights must be finite and nonnegative")
    matrix = np.maximum(matrix, 0.0)
    totals = matrix.sum(axis=0)
    if np.any(~np.isclose(totals, 1.0, atol=1e-12, rtol=1e-12)):
        raise ValueError("every kernel column must sum to one")
    return matrix / totals


def lot_backup(
    action_values: ArrayLike,
    weights: ArrayLike,
    anchor_distribution: ArrayLike,
    T0: float,
    *,
    offsets: ArrayLike | None = None,
    retained: ArrayLike | None = None,
) -> BackupResult:
    """Evaluate Equation (19), optionally with unnormalized pruning.

    ``weights[i, j]`` is the candidate distribution for anchor ``j``. When a
    retained mask is supplied, omitted weights stay omitted in the value; the
    policy is the derivative and therefore normalizes within each retained set.
    """

    q = np.asarray(action_values, dtype=np.float64)
    if q.ndim != 1 or not np.all(np.isfinite(q)):
        raise ValueError("action_values must be a finite vector")
    if T0 <= 0.0 or not np.isfinite(T0):
        raise ValueError("T0 must be finite and positive")
    kernel = _kernel_matrix(weights, len(q))
    mu = _probability_vector(anchor_distribution, "anchor_distribution")
    if kernel.shape[1] != len(mu):
        raise ValueError("one anchor weight is required for every kernel column")
    if offsets is None:
        offset_vector = np.zeros(len(mu), dtype=np.float64)
    else:
        offset_vector = np.asarray(offsets, dtype=np.float64)
        if offset_vector.shape != mu.shape or not np.all(np.isfinite(offset_vector)):
            raise ValueError("offsets must be a finite vector indexed by anchor")
    if retained is None:
        mask = kernel > 0.0
    else:
        mask = np.asarray(retained, dtype=bool)
        if mask.shape != kernel.shape:
            raise ValueError("retained mask must match the kernel shape")
        mask &= kernel > 0.0
    if np.any(mask.sum(axis=0) == 0):
        raise ValueError("every anchor must retain at least one positive-weight action")

    log_kernel = np.full_like(kernel, -np.inf)
    positive = kernel > 0.0
    log_kernel[positive] = np.log(kernel[positive])
    scores = log_kernel + q[:, None] / T0
    scores = np.where(mask, scores, -np.inf)
    log_partitions = logsumexp(scores, axis=0)
    anchor_policies = np.exp(scores - log_partitions[None, :])
    anchor_policies[~mask] = 0.0
    policy = anchor_policies @ mu
    policy /= policy.sum()
    value = float(mu @ (offset_vector + T0 * log_partitions))
    return BackupResult(value=value, policy=policy, anchor_policies=anchor_policies)


def kernel_from_cost(
    costs: ArrayLike,
    candidate_prior: ArrayLike,
    tau: float,
    lambda_: float,
) -> tuple[FloatArray, FloatArray]:
    """Convert costs to normalized columns and Equation (19) offsets."""

    cost_matrix = np.asarray(costs, dtype=np.float64)
    if cost_matrix.ndim != 2 or not np.all(np.isfinite(cost_matrix)):
        raise ValueError("costs must be a finite matrix")
    if np.min(cost_matrix, initial=0.0) < 0.0:
        raise ValueError("costs must be nonnegative")
    if tau <= 0.0 or lambda_ <= 0.0:
        raise ValueError("tau and lambda_ must be positive")
    prior = _probability_vector(candidate_prior, "candidate_prior")
    if np.any(prior <= 0.0):
        raise ValueError("candidate_prior must have full support for cost-based LOT")
    if cost_matrix.shape[0] != len(prior):
        raise ValueError("cost rows must be indexed by candidate action")
    log_unnormalized = np.log(prior)[:, None] - cost_matrix / lambda_
    log_kappa = logsumexp(log_unnormalized, axis=0)
    weights = np.exp(log_unnormalized - log_kappa[None, :])
    T0 = tau * lambda_
    offsets = T0 * log_kappa
    return weights, offsets


def cost_based_lot_backup(
    action_values: ArrayLike,
    costs: ArrayLike,
    candidate_prior: ArrayLike,
    anchor_distribution: ArrayLike,
    tau: float,
    lambda_: float,
    *,
    retained: ArrayLike | None = None,
) -> BackupResult:
    weights, offsets = kernel_from_cost(costs, candidate_prior, tau, lambda_)
    return lot_backup(
        action_values,
        weights,
        anchor_distribution,
        tau * lambda_,
        offsets=offsets,
        retained=retained,
    )


def prior_weighted_maxent(
    action_values: ArrayLike, candidate_prior: ArrayLike, T0: float
) -> BackupResult:
    prior = _probability_vector(candidate_prior, "candidate_prior")
    return lot_backup(action_values, prior[:, None], np.ones(1), T0)


def heat_backup(
    action_values: ArrayLike,
    heat_columns: ArrayLike,
    anchor_distribution: ArrayLike,
    T0: float,
    *,
    retained: ArrayLike | None = None,
) -> BackupResult:
    """Centered direct-heat backup from Equation (31)."""

    return lot_backup(
        action_values,
        heat_columns,
        anchor_distribution,
        T0,
        offsets=None,
        retained=retained,
    )
