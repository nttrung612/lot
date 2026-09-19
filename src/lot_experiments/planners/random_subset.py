"""Uniform-action Monte Carlo and equal-size random-subset baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp

from lot_experiments.backups import BackupResult, lot_backup
from lot_experiments.counters import OperationCounters


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class SampledBackupResult:
    value: float
    policy: FloatArray
    sampled_actions: tuple[NDArray[np.int64], ...]
    method: str
    target: str


def random_subset_masks(
    action_count: int,
    anchor_count: int,
    subset_size: int,
    rng: np.random.Generator,
) -> BoolArray:
    """Draw an independent equal-cardinality subset for every anchor."""

    if action_count < 1 or anchor_count < 1:
        raise ValueError("action_count and anchor_count must be positive")
    if not 1 <= subset_size <= action_count:
        raise ValueError("subset_size must lie in [1, action_count]")
    mask = np.zeros((action_count, anchor_count), dtype=bool)
    for anchor in range(anchor_count):
        chosen = rng.choice(action_count, size=subset_size, replace=False)
        mask[chosen, anchor] = True
    return mask


def random_subset_backup(
    action_values: ArrayLike,
    weights: ArrayLike,
    anchor_distribution: ArrayLike,
    T0: float,
    *,
    subset_size: int,
    rng: np.random.Generator,
    counter: OperationCounters | None = None,
) -> BackupResult:
    """Unnormalized pruning on random subsets of the exact target kernel.

    This is the equal-cardinality random-subset baseline, not an importance
    sampler.  Consequently its value has the same signed omitted-mass bias as
    any fixed-set pruned backup.
    """

    kernel = np.asarray(weights, dtype=np.float64)
    if kernel.ndim != 2:
        raise ValueError("weights must be a matrix")
    retained = random_subset_masks(
        kernel.shape[0], kernel.shape[1], subset_size, rng
    )
    if counter is not None:
        actions = set(map(int, np.flatnonzero(retained.any(axis=1))))
        for action in actions:
            counter.touch_action(action)
    return lot_backup(
        action_values,
        kernel,
        anchor_distribution,
        T0,
        retained=retained,
    )


def uniform_action_mc_backup(
    action_values: ArrayLike,
    weights: ArrayLike,
    anchor_distribution: ArrayLike,
    T0: float,
    *,
    samples_per_anchor: int,
    rng: np.random.Generator,
    replace: bool = True,
    target: str = "exact_heat",
    counter: OperationCounters | None = None,
) -> SampledBackupResult:
    """Estimate each kernel partition from uniformly sampled actions.

    The factor ``K`` is the exact importance correction for the uniform
    proposal.  The logarithm still gives the usual finite-sample plug-in bias;
    this baseline is therefore empirical and carries no deterministic pruning
    guarantee.
    """

    q = np.asarray(action_values, dtype=np.float64)
    kernel = np.asarray(weights, dtype=np.float64)
    mu = np.asarray(anchor_distribution, dtype=np.float64)
    if q.ndim != 1 or not np.all(np.isfinite(q)):
        raise ValueError("action_values must be a finite vector")
    if kernel.shape[0] != len(q) or kernel.ndim != 2:
        raise ValueError("weights must have shape (actions, anchors)")
    if mu.shape != (kernel.shape[1],) or np.any(mu < 0.0) or not np.isclose(
        mu.sum(), 1.0
    ):
        raise ValueError("anchor_distribution must match anchors and sum to one")
    if T0 <= 0.0 or not math.isfinite(T0):
        raise ValueError("T0 must be finite and positive")
    if samples_per_anchor < 1 or (not replace and samples_per_anchor > len(q)):
        raise ValueError("invalid samples_per_anchor")
    if not target:
        raise ValueError("target must be nonempty")

    shift = float(np.max(q))
    scaled = (q - shift) / T0
    policy = np.zeros(len(q), dtype=np.float64)
    anchor_values = np.empty(kernel.shape[1], dtype=np.float64)
    sampled: list[NDArray[np.int64]] = []
    for anchor in range(kernel.shape[1]):
        actions = np.asarray(
            rng.choice(len(q), size=samples_per_anchor, replace=replace),
            dtype=np.int64,
        )
        sampled.append(actions)
        log_terms = (
            math.log(len(q))
            + np.log(kernel[actions, anchor])
            + scaled[actions]
        )
        log_partition = float(logsumexp(log_terms) - math.log(samples_per_anchor))
        if not np.isfinite(log_partition):
            raise FloatingPointError("uniform MC estimated a zero partition")
        anchor_values[anchor] = shift + T0 * log_partition
        normalized = np.exp(log_terms - logsumexp(log_terms))
        np.add.at(policy, actions, mu[anchor] * normalized)
        if counter is not None:
            for action in actions:
                counter.touch_action(int(action))
    policy /= policy.sum()
    return SampledBackupResult(
        value=float(mu @ anchor_values),
        policy=policy,
        sampled_actions=tuple(sampled),
        method="uniform_action_mc",
        target=target,
    )
