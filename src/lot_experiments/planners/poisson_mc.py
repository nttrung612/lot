"""Poisson random-walk endpoint Monte Carlo baseline."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp

from lot_experiments.counters import OperationCounters
from lot_experiments.graphs import ActionGraph
from lot_experiments.planners.random_subset import SampledBackupResult


def sample_poisson_endpoint(
    graph: ActionGraph,
    anchor: int,
    diffusion_time: float,
    rng: np.random.Generator,
    *,
    nu_u: float | None = None,
    counter: OperationCounters | None = None,
) -> int:
    """Sample exactly from one heat column via uniformization.

    If ``N ~ Poisson(nu_u t)`` and ``P_G = I-L/nu_u``, the endpoint of an
    ``N``-step walk has law ``exp(-tL)(:, anchor)``.
    """

    if not 0 <= anchor < graph.K:
        raise IndexError(anchor)
    if diffusion_time < 0.0 or not math.isfinite(diffusion_time):
        raise ValueError("diffusion_time must be finite and nonnegative")
    max_degree = graph.max_weighted_degree
    rate = max_degree if nu_u is None and max_degree > 0.0 else (
        1.0 if nu_u is None else float(nu_u)
    )
    if rate <= 0.0 or not math.isfinite(rate) or rate + 1e-12 < max_degree:
        raise ValueError("nu_u must be finite, positive, and at least d_max")
    steps = int(rng.poisson(rate * diffusion_time))
    vertex = int(anchor)
    for _ in range(steps):
        neighbors, edge_weights = graph.neighbors(vertex, counter)
        move_mass = float(edge_weights.sum()) / rate
        draw = float(rng.random())
        if draw < move_mass:
            probabilities = edge_weights / float(edge_weights.sum())
            vertex = int(rng.choice(neighbors, p=probabilities))
    if counter is not None:
        counter.touch_action(vertex)
    return vertex


def poisson_endpoint_mc_backup(
    action_values: ArrayLike,
    graph: ActionGraph,
    anchor_distribution: ArrayLike,
    T0: float,
    diffusion_time: float,
    *,
    endpoints_per_anchor: int,
    rng: np.random.Generator,
    nu_u: float | None = None,
    counter: OperationCounters | None = None,
) -> SampledBackupResult:
    """Plug-in exact-heat backup using sampled Poisson-walk endpoints."""

    q = np.asarray(action_values, dtype=np.float64)
    mu = np.asarray(anchor_distribution, dtype=np.float64)
    if q.shape != (graph.K,) or not np.all(np.isfinite(q)):
        raise ValueError("action_values must be a finite vector matching the graph")
    if mu.shape != (graph.K,) or np.any(mu < 0.0) or not np.isclose(mu.sum(), 1.0):
        raise ValueError("anchor_distribution must match the graph and sum to one")
    if T0 <= 0.0 or not math.isfinite(T0):
        raise ValueError("T0 must be finite and positive")
    if endpoints_per_anchor < 1:
        raise ValueError("endpoints_per_anchor must be positive")
    shift = float(np.max(q))
    policy = np.zeros(graph.K, dtype=np.float64)
    anchor_values = np.empty(graph.K, dtype=np.float64)
    samples: list[np.ndarray] = []
    for anchor in range(graph.K):
        endpoints = np.fromiter(
            (
                sample_poisson_endpoint(
                    graph,
                    anchor,
                    diffusion_time,
                    rng,
                    nu_u=nu_u,
                    counter=counter,
                )
                for _ in range(endpoints_per_anchor)
            ),
            dtype=np.int64,
            count=endpoints_per_anchor,
        )
        samples.append(endpoints)
        scores = (q[endpoints] - shift) / T0
        log_partition = float(logsumexp(scores) - math.log(endpoints_per_anchor))
        anchor_values[anchor] = shift + T0 * log_partition
        normalized = np.exp(scores - logsumexp(scores))
        np.add.at(policy, endpoints, mu[anchor] * normalized)
    policy /= policy.sum()
    return SampledBackupResult(
        value=float(mu @ anchor_values),
        policy=policy,
        sampled_actions=tuple(samples),
        method="poisson_endpoint_mc",
        target="exact_heat",
    )
