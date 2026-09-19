"""Graph-local normalized finite-walk heat planner.

Only one heat column is requested at a time from ``LazyTruncatedHeat``; this
module never stacks those columns into a dense ``K x K`` heat matrix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp

from lot_experiments.counters import OperationCounters
from lot_experiments.environments.ring_control import RingControlMDP
from lot_experiments.graphs import ActionGraph
from lot_experiments.heat_local import LazyTruncatedHeat
from lot_experiments.planners.dense import PlanningResult


@dataclass(frozen=True)
class LocalBackupResult:
    """Local backup output without a dense action-by-anchor allocation."""

    value: float
    policy: np.ndarray


def local_truncated_heat_backup(
    action_values: ArrayLike,
    anchor_distribution: ArrayLike,
    T0: float,
    heat: LazyTruncatedHeat,
    *,
    counter: OperationCounters | None = None,
) -> LocalBackupResult:
    """Evaluate a truncated-heat backup by sparse column supports."""

    q = np.asarray(action_values, dtype=np.float64)
    mu = np.asarray(anchor_distribution, dtype=np.float64)
    if q.shape != (heat.graph.K,) or not np.all(np.isfinite(q)):
        raise ValueError("action_values must be a finite vector matching the graph")
    if mu.shape != (heat.graph.K,) or np.any(mu < 0.0) or not np.isclose(
        mu.sum(), 1.0
    ):
        raise ValueError("anchor_distribution must match anchors and sum to one")
    if T0 <= 0.0 or not math.isfinite(T0):
        raise ValueError("T0 must be finite and positive")
    policy = np.zeros(heat.graph.K, dtype=np.float64)
    value = 0.0
    evaluated: set[int] = set()
    for anchor in np.flatnonzero(mu > 0.0):
        support, weights = heat.sparse_column(int(anchor))
        scores = np.log(weights) + q[support] / T0
        log_partition = float(logsumexp(scores))
        probabilities = np.exp(scores - log_partition)
        policy[support] += mu[anchor] * probabilities
        value += mu[anchor] * T0 * log_partition
        if counter is not None:
            for action in map(int, support):
                if action not in evaluated:
                    counter.touch_action(action)
                    evaluated.add(action)
    policy /= policy.sum()
    return LocalBackupResult(value=float(value), policy=policy)


def local_truncated_heat_value_iteration(
    mdp: RingControlMDP,
    graph: ActionGraph,
    *,
    gamma: float,
    T0: float,
    diffusion_time: float,
    radius: int,
    nu_u: float | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
    initial_values: ArrayLike | None = None,
    counter: OperationCounters | None = None,
) -> PlanningResult:
    """Solve the finite-walk fixed point without forming its dense heat matrix.

    Because the configured ring anchor distribution has full support, an exact
    deterministic sweep eventually touches the union of all local supports.
    The root-only recursive estimator introduced by M5 is what avoids that
    exhaustive outer-anchor average; this function is its local correctness
    reference.
    """

    if graph.K != mdp.K:
        raise ValueError("action graph and MDP must have the same action count")
    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must lie in [0, 1)")
    if tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("tolerance and max_iterations must be positive")
    counters = OperationCounters() if counter is None else counter
    lazy = LazyTruncatedHeat(
        graph=graph,
        diffusion_time=diffusion_time,
        radius=radius,
        nu_u=nu_u,
        counter=counters,
    )
    values = (
        np.zeros(mdp.K, dtype=np.float64)
        if initial_values is None
        else np.asarray(initial_values, dtype=np.float64).copy()
    )
    if values.shape != (mdp.K,) or not np.all(np.isfinite(values)):
        raise ValueError("initial_values must be a finite vector of length K")
    geometry_before = counters.geometry_preprocess_seconds
    start = perf_counter()
    residual = math.inf
    policy = np.empty((mdp.K, mdp.K), dtype=np.float64)
    for iteration in range(1, int(max_iterations) + 1):
        q_values = mdp.expected_action_values(values, gamma, counter=counters)
        updated = np.empty(mdp.K, dtype=np.float64)
        for state in range(mdp.K):
            backup = local_truncated_heat_backup(
                q_values[state],
                mdp.anchor_distributions[state],
                T0,
                lazy,
            )
            updated[state] = backup.value
            policy[state] = backup.policy
        residual = float(np.max(np.abs(updated - values)))
        values = updated
        if residual <= tolerance:
            converged = True
            break
    else:
        iteration = int(max_iterations)
        converged = False
    q_values = mdp.expected_action_values(values, gamma, counter=counters)
    bellman_values = np.empty(mdp.K, dtype=np.float64)
    for state in range(mdp.K):
        backup = local_truncated_heat_backup(
            q_values[state], mdp.anchor_distributions[state], T0, lazy
        )
        bellman_values[state] = backup.value
        policy[state] = backup.policy
    residual = float(np.max(np.abs(bellman_values - values)))
    elapsed = perf_counter() - start
    geometry_elapsed = counters.geometry_preprocess_seconds - geometry_before
    counters.online_seconds += max(0.0, elapsed - geometry_elapsed)
    counters.observe_memory()
    return PlanningResult(
        values=values,
        q_values=q_values,
        policy=policy,
        iterations=iteration,
        bellman_residual=residual,
        converged=converged,
        method="truncated_heat_local",
        target="truncated_heat",
        counters=counters,
    )
