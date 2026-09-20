"""Reward-free transition geometry from Theorem 10 of the paper.

The learned objects in this module are deliberately limited to geometry.  In
particular, the empirical transitions returned by :func:`sample_reward_free_batch`
are never exposed as an MDP model to a Bellman planner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm

from lot_experiments.environments.ring_control import RingControlMDP, cyclic_distance


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class TransitionGeometry:
    """Shared state Laplacian and action-specific signed transition means."""

    signatures: FloatArray
    laplacian: FloatArray


@dataclass(frozen=True)
class RewardFreeBatch:
    """Nested reward-free samples, indexed by action and draw number."""

    origins: IntArray
    successors: IntArray

    @property
    def action_count(self) -> int:
        return int(self.origins.shape[0])

    @property
    def sample_capacity(self) -> int:
        return int(self.origins.shape[1])


def probability_vector(values: ArrayLike, size: int, *, name: str = "nu") -> FloatArray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector of length {size}")
    if np.min(vector, initial=0.0) < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    total = float(vector.sum())
    if total <= 0.0:
        raise ValueError(f"{name} must have positive mass")
    return vector / total


def probe_distribution(
    K: int,
    *,
    kind: str = "spike_uniform",
    uniform_mass: float = 0.1,
    center: int = 0,
) -> FloatArray:
    """Build a task-independent starting-state distribution for geometry data.

    Uniform ``nu`` is supported as a diagnostic.  It is degenerate for the
    translation-invariant ring because every action preserves uniform mass.
    The primary full-support probe therefore mixes a fixed state with uniform
    mass; its center is fixed independently of every downstream reward goal.
    """

    if not isinstance(K, (int, np.integer)) or K < 3:
        raise ValueError("K must be an integer at least three")
    if kind == "uniform":
        return np.full(K, 1.0 / K, dtype=np.float64)
    if kind != "spike_uniform":
        raise ValueError(f"unknown probe distribution: {kind}")
    if not 0.0 < float(uniform_mass) <= 1.0:
        raise ValueError("uniform_mass must lie in (0, 1]")
    if not 0 <= int(center) < K:
        raise ValueError("probe center must lie in Z_K")
    distribution = np.full(K, float(uniform_mass) / K, dtype=np.float64)
    distribution[int(center)] += 1.0 - float(uniform_mass)
    return distribution


def _add_edge_laplacian(
    laplacian: FloatArray, origin: int, successor: int, weight: float
) -> None:
    if origin == successor or weight == 0.0:
        return
    half_weight = 0.5 * float(weight)
    laplacian[origin, origin] += half_weight
    laplacian[successor, successor] += half_weight
    laplacian[origin, successor] -= half_weight
    laplacian[successor, origin] -= half_weight


def population_transition_geometry(
    mdp: RingControlMDP, nu: ArrayLike
) -> TransitionGeometry:
    """Compute ``z_a`` and ``L_nu`` exactly from the true ring dynamics."""

    distribution = probability_vector(nu, mdp.K)
    signatures = np.zeros((mdp.K, mdp.K), dtype=np.float64)
    laplacian = np.zeros((mdp.K, mdp.K), dtype=np.float64)
    for action in range(mdp.K):
        next_mass = np.zeros(mdp.K, dtype=np.float64)
        for state, state_mass in enumerate(distribution):
            successors, probabilities = mdp.transition_outcomes(state, action)
            next_mass[successors] += state_mass * probabilities
            for successor, probability in zip(successors, probabilities, strict=True):
                _add_edge_laplacian(
                    laplacian,
                    state,
                    int(successor),
                    state_mass * float(probability) / mdp.K,
                )
        signatures[action] = next_mass - distribution
    return _validated_geometry(signatures, laplacian)


def sample_reward_free_batch(
    mdp: RingControlMDP,
    nu: ArrayLike,
    samples_per_action: int,
    rng: np.random.Generator,
) -> RewardFreeBatch:
    """Draw independent ``S~nu, S'~P(.|S,a)`` samples for every action."""

    if not isinstance(samples_per_action, (int, np.integer)) or samples_per_action < 1:
        raise ValueError("samples_per_action must be a positive integer")
    distribution = probability_vector(nu, mdp.K)
    n = int(samples_per_action)
    origins = rng.choice(mdp.K, size=(mdp.K, n), p=distribution).astype(np.int64)
    noise_indices = rng.choice(
        len(mdp.noise_offsets),
        size=(mdp.K, n),
        p=np.asarray(mdp.noise_probabilities, dtype=np.float64),
    )
    actions = np.arange(mdp.K, dtype=np.int64)[:, None]
    offsets = np.asarray(mdp.noise_offsets, dtype=np.int64)[noise_indices]
    successors = (origins + actions + offsets) % mdp.K
    return RewardFreeBatch(origins=origins, successors=successors.astype(np.int64))


def empirical_transition_geometry(
    batch: RewardFreeBatch, samples_per_action: int | None = None
) -> TransitionGeometry:
    """Estimate ``z_a`` and the shared ``L_nu`` from a nested batch prefix."""

    n = batch.sample_capacity if samples_per_action is None else int(samples_per_action)
    if n < 1 or n > batch.sample_capacity:
        raise ValueError("requested prefix lies outside the reward-free batch")
    K = batch.action_count
    origins = batch.origins[:, :n]
    successors = batch.successors[:, :n]
    signatures = np.empty((K, K), dtype=np.float64)
    laplacian = np.zeros((K, K), dtype=np.float64)
    for action in range(K):
        origin_counts = np.bincount(origins[action], minlength=K)
        successor_counts = np.bincount(successors[action], minlength=K)
        signatures[action] = (successor_counts - origin_counts) / n
        for origin, successor in zip(
            origins[action], successors[action], strict=True
        ):
            _add_edge_laplacian(laplacian, int(origin), int(successor), 1.0 / (K * n))
    return _validated_geometry(signatures, laplacian)


def _validated_geometry(signatures: FloatArray, laplacian: FloatArray) -> TransitionGeometry:
    signatures = np.asarray(signatures, dtype=np.float64)
    laplacian = np.asarray(laplacian, dtype=np.float64)
    K, D = signatures.shape
    if laplacian.shape != (D, D) or K < 1 or not np.all(np.isfinite(signatures)):
        raise ValueError("incompatible or nonfinite transition geometry")
    if not np.allclose(laplacian, laplacian.T, atol=1e-13, rtol=0.0):
        raise FloatingPointError("transition Laplacian is not symmetric")
    if not np.allclose(laplacian.sum(axis=1), 0.0, atol=1e-13, rtol=0.0):
        raise FloatingPointError("transition Laplacian rows do not sum to zero")
    if float(np.linalg.eigvalsh(laplacian).min(initial=0.0)) < -1e-12:
        raise FloatingPointError("transition Laplacian is not positive semidefinite")
    return TransitionGeometry(signatures=signatures, laplacian=laplacian)


def transition_diffusion_cost(
    geometry: TransitionGeometry, diffusion_time: float, kappa_C: float
) -> FloatArray:
    """Return ``C_ab^t = kappa_C ||exp(-t L_nu)(z_a-z_b)||_2^2``."""

    if diffusion_time < 0.0 or not math.isfinite(diffusion_time):
        raise ValueError("diffusion_time must be finite and nonnegative")
    if kappa_C <= 0.0 or not math.isfinite(kappa_C):
        raise ValueError("kappa_C must be finite and positive")
    state_heat = expm(-float(diffusion_time) * geometry.laplacian)
    embeddings = (state_heat @ geometry.signatures.T).T
    squared_norms = np.sum(embeddings * embeddings, axis=1)
    costs = squared_norms[:, None] + squared_norms[None, :] - 2.0 * (
        embeddings @ embeddings.T
    )
    np.maximum(costs, 0.0, out=costs)
    costs *= float(kappa_C)
    np.fill_diagonal(costs, 0.0)
    return 0.5 * (costs + costs.T)


def raw_action_coordinate_cost(K: int, kappa_C: float = 1.0) -> FloatArray:
    """Squared normalized cyclic-coordinate cost used as a distinct baseline."""

    actions = np.arange(K, dtype=np.int64)
    distances = cyclic_distance(actions[:, None], actions[None, :], K)
    return float(kappa_C) * (distances / (K / 2.0)) ** 2


def theorem10_cost_certificate(
    *,
    state_count: int,
    action_count: int,
    samples_per_action: int,
    delta: float,
    diffusion_time: float,
    kappa_C: float,
) -> float:
    """Explicit high-probability ``epsilon_(n,delta)`` from Theorem 10."""

    D, K, n = int(state_count), int(action_count), int(samples_per_action)
    if D < 1 or K < 1 or n < 1:
        raise ValueError("state_count, action_count, and samples_per_action must be positive")
    if not 0.0 < float(delta) < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    if diffusion_time < 0.0 or kappa_C <= 0.0:
        raise ValueError("diffusion_time must be nonnegative and kappa_C positive")
    total = K * n
    log_l = math.log(4.0 * D / float(delta))
    eta_l = math.sqrt(log_l / (2.0 * total)) + 2.0 * log_l / (3.0 * total)
    eta_z = math.sqrt(2.0 / n) + 2.0 * math.sqrt(math.log(2.0 * K / float(delta)) / n)
    return float(kappa_C) * (16.0 * float(diffusion_time) * eta_l + 8.0 * math.sqrt(2.0) * eta_z)


def max_cost_error(estimate: ArrayLike, reference: ArrayLike) -> float:
    estimate_array = np.asarray(estimate, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if estimate_array.shape != reference_array.shape:
        raise ValueError("cost matrices must have the same shape")
    return float(np.max(np.abs(estimate_array - reference_array), initial=0.0))
