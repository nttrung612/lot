"""Dense Bellman references for the ring-control experiment.

These routines deliberately evaluate every action.  They are correctness
references and same-/different-target baselines, not implementations of the
localized recursive estimator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp

from lot_experiments.counters import OperationCounters
from lot_experiments.environments.ring_control import RingControlMDP
from lot_experiments.graphs import ActionGraph, cycle_graph
from lot_experiments.kernels import (
    diffusion_distance_gibbs_kernel,
    exact_heat_kernel,
    truncated_heat_kernel,
)
from lot_experiments.pruning import top_mass_mask


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class PlanningResult:
    """A converged tabular solution and its fully accounted computation."""

    values: FloatArray
    q_values: FloatArray
    policy: FloatArray
    iterations: int
    bellman_residual: float
    converged: bool
    method: str
    target: str
    counters: OperationCounters = field(compare=False)

    def root_value(self, root_state: int = 0) -> float:
        if not 0 <= root_state < len(self.values):
            raise IndexError(root_state)
        return float(self.values[root_state])


def _probability_rows(values: ArrayLike, rows: int, columns: int) -> FloatArray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (rows, columns) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"anchor_distributions must have shape {(rows, columns)}")
    if np.min(matrix, initial=0.0) < -1e-14:
        raise ValueError("anchor distributions must be nonnegative")
    matrix = np.maximum(matrix, 0.0)
    totals = matrix.sum(axis=1)
    if np.any(~np.isclose(totals, 1.0, atol=1e-12, rtol=1e-12)):
        raise ValueError("anchor distributions must sum to one by row")
    return matrix / totals[:, None]


def _kernel_and_mask(
    weights: ArrayLike, retained: ArrayLike | None
) -> tuple[FloatArray, BoolArray]:
    kernel = np.asarray(weights, dtype=np.float64)
    if (
        kernel.ndim != 2
        or kernel.shape[0] == 0
        or not np.all(np.isfinite(kernel))
        or np.min(kernel, initial=0.0) < -1e-14
    ):
        raise ValueError("weights must be a finite nonnegative matrix")
    kernel = np.maximum(kernel, 0.0)
    totals = kernel.sum(axis=0)
    if np.any(~np.isclose(totals, 1.0, atol=1e-12, rtol=1e-12)):
        raise ValueError("kernel columns must sum to one")
    kernel = kernel / totals[None, :]
    if retained is None:
        mask = kernel > 0.0
    else:
        mask = np.asarray(retained, dtype=bool)
        if mask.shape != kernel.shape:
            raise ValueError("retained must match the kernel shape")
        mask = mask & (kernel > 0.0)
    if np.any(mask.sum(axis=0) == 0):
        raise ValueError("every anchor must retain a positive-weight action")
    return kernel, mask


def batched_lot_backup(
    action_values: ArrayLike,
    weights: ArrayLike,
    anchor_distributions: ArrayLike,
    T0: float,
    *,
    offsets: ArrayLike | None = None,
    retained: ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Evaluate the same LOT backup for a batch of states.

    The implementation is algebraically equivalent to calling ``lot_backup``
    state by state, but uses two matrix products and avoids an ``S x K x K``
    score tensor.  Retained kernel weights remain unnormalized in the value.
    """

    q_values = np.asarray(action_values, dtype=np.float64)
    if q_values.ndim != 2 or not np.all(np.isfinite(q_values)):
        raise ValueError("action_values must be a finite (states, actions) matrix")
    if T0 <= 0.0 or not math.isfinite(T0):
        raise ValueError("T0 must be finite and positive")
    state_count, action_count = q_values.shape
    kernel, mask = _kernel_and_mask(weights, retained)
    if kernel.shape[0] != action_count:
        raise ValueError("kernel rows must index the action values")
    anchor_count = kernel.shape[1]
    mu = _probability_rows(anchor_distributions, state_count, anchor_count)
    if offsets is None:
        offset_vector = np.zeros(anchor_count, dtype=np.float64)
    else:
        offset_vector = np.asarray(offsets, dtype=np.float64)
        if offset_vector.shape != (anchor_count,) or not np.all(
            np.isfinite(offset_vector)
        ):
            raise ValueError("offsets must be a finite vector indexed by anchor")

    retained_kernel = np.where(mask, kernel, 0.0)
    row_maxima = np.max(q_values, axis=1)
    exponentials = np.exp((q_values - row_maxima[:, None]) / T0)
    partitions = exponentials @ retained_kernel
    if np.any(partitions <= 0.0) or not np.all(np.isfinite(partitions)):
        raise FloatingPointError("LOT partition is zero or nonfinite")
    log_partitions = np.log(partitions) + row_maxima[:, None] / T0
    values = mu @ offset_vector + T0 * np.sum(mu * log_partitions, axis=1)

    # pi_si = exp((q_si-m_s)/T0) sum_j w_ij mu_sj / Z_sj.
    policy = exponentials * ((mu / partitions) @ retained_kernel.T)
    policy /= policy.sum(axis=1, keepdims=True)
    return values, policy


def hard_max_backup(action_values: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Hard-max reference with uniform mass across exact ties."""

    q_values = np.asarray(action_values, dtype=np.float64)
    if q_values.ndim != 2 or not np.all(np.isfinite(q_values)):
        raise ValueError("action_values must be a finite (states, actions) matrix")
    values = np.max(q_values, axis=1)
    maximizers = q_values == values[:, None]
    policy = maximizers / maximizers.sum(axis=1, keepdims=True)
    return values, policy.astype(np.float64)


def dense_value_iteration(
    mdp: RingControlMDP,
    *,
    gamma: float,
    method: str,
    target: str,
    weights: ArrayLike | None = None,
    T0: float | None = None,
    anchor_distributions: ArrayLike | None = None,
    offsets: ArrayLike | None = None,
    retained: ArrayLike | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
    initial_values: ArrayLike | None = None,
    counter: OperationCounters | None = None,
) -> PlanningResult:
    """Solve a dense tabular Bellman fixed point by value iteration."""

    if not 0.0 <= gamma < 1.0 or not math.isfinite(gamma):
        raise ValueError("gamma must be finite and lie in [0, 1)")
    if tolerance <= 0.0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and positive")
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    is_hard_max = target == "hard_max"
    if is_hard_max:
        if weights is not None or retained is not None or offsets is not None:
            raise ValueError("hard max does not accept a kernel, offsets, or retained mask")
    elif weights is None or T0 is None:
        raise ValueError("regularized value iteration requires weights and T0")
    mu = (
        mdp.anchor_distributions
        if anchor_distributions is None
        else np.asarray(anchor_distributions, dtype=np.float64)
    )
    values = (
        np.zeros(mdp.K, dtype=np.float64)
        if initial_values is None
        else np.asarray(initial_values, dtype=np.float64).copy()
    )
    if values.shape != (mdp.K,) or not np.all(np.isfinite(values)):
        raise ValueError("initial_values must be a finite vector of length K")
    counters = OperationCounters() if counter is None else counter
    residual = math.inf
    policy = np.empty((mdp.K, mdp.K), dtype=np.float64)
    start = perf_counter()
    for iteration in range(1, int(max_iterations) + 1):
        q_values = mdp.expected_action_values(values, gamma, counter=counters)
        if is_hard_max:
            updated, policy = hard_max_backup(q_values)
        else:
            updated, policy = batched_lot_backup(
                q_values,
                weights,
                mu,
                float(T0),
                offsets=offsets,
                retained=retained,
            )
        residual = float(np.max(np.abs(updated - values)))
        values = updated
        if residual <= tolerance:
            converged = True
            break
    else:
        iteration = int(max_iterations)
        converged = False

    # Return Q and policy for the returned value iterate.  This final diagnostic
    # Bellman application is accounted like every other online query.
    q_values = mdp.expected_action_values(values, gamma, counter=counters)
    if is_hard_max:
        bellman_values, policy = hard_max_backup(q_values)
    else:
        bellman_values, policy = batched_lot_backup(
            q_values,
            weights,
            mu,
            float(T0),
            offsets=offsets,
            retained=retained,
        )
    residual = float(np.max(np.abs(bellman_values - values)))
    counters.online_seconds += perf_counter() - start
    counters.observe_memory()
    return PlanningResult(
        values=values,
        q_values=q_values,
        policy=policy,
        iterations=iteration,
        bellman_residual=residual,
        converged=converged,
        method=method,
        target=target,
        counters=counters,
    )


def full_exact_heat_reference(
    mdp: RingControlMDP,
    *,
    gamma: float,
    T0: float,
    diffusion_time: float,
    graph: ActionGraph | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
    counter: OperationCounters | None = None,
) -> PlanningResult:
    """Reference solution for the centered exact-heat target."""

    action_graph = cycle_graph(mdp.K) if graph is None else graph
    if action_graph.K != mdp.K:
        raise ValueError("action graph and MDP must have the same action count")
    counters = OperationCounters() if counter is None else counter
    start = perf_counter()
    heat = exact_heat_kernel(action_graph.laplacian, diffusion_time)
    counters.geometry_preprocess_seconds += perf_counter() - start
    counters.dense_linear_algebra_operations += 1
    counters.observe_memory()
    return dense_value_iteration(
        mdp,
        gamma=gamma,
        method="full_exact_heat",
        target="exact_heat",
        weights=heat,
        T0=T0,
        tolerance=tolerance,
        max_iterations=max_iterations,
        counter=counters,
    )


def dense_second_order_reference(
    mdp: RingControlMDP,
    *,
    gamma: float,
    T0: float,
    diffusion_time: float,
    graph: ActionGraph | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
) -> PlanningResult:
    """Full-action oracle baseline, numerically equal to ``FullExactHeat``.

    This deterministic implementation supplies the full-action limit against
    which finite-sample recursive implementations are validated.  It does not
    claim the transition complexity of the stochastic second-order estimator.
    """

    result = full_exact_heat_reference(
        mdp,
        gamma=gamma,
        T0=T0,
        diffusion_time=diffusion_time,
        graph=graph,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return PlanningResult(
        values=result.values,
        q_values=result.q_values,
        policy=result.policy,
        iterations=result.iterations,
        bellman_residual=result.bellman_residual,
        converged=result.converged,
        method="dense_second_order",
        target="exact_heat",
        counters=result.counters,
    )


def exact_heat_topmass_reference(
    mdp: RingControlMDP,
    *,
    gamma: float,
    T0: float,
    diffusion_time: float,
    alpha: float,
    graph: ActionGraph | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
) -> PlanningResult:
    """Solve the unnormalized exact-heat top-mass pruned operator."""

    action_graph = cycle_graph(mdp.K) if graph is None else graph
    if action_graph.K != mdp.K:
        raise ValueError("action graph and MDP must have the same action count")
    counters = OperationCounters()
    start = perf_counter()
    heat = exact_heat_kernel(action_graph.laplacian, diffusion_time)
    retained = np.column_stack(
        [top_mass_mask(heat[:, anchor], alpha) for anchor in range(mdp.K)]
    )
    counters.geometry_preprocess_seconds += perf_counter() - start
    counters.dense_linear_algebra_operations += 1
    counters.observe_memory()
    return dense_value_iteration(
        mdp,
        gamma=gamma,
        method="exact_heat_topmass",
        target="exact_heat",
        weights=heat,
        retained=retained,
        T0=T0,
        tolerance=tolerance,
        max_iterations=max_iterations,
        counter=counters,
    )


def truncated_heat_reference(
    mdp: RingControlMDP,
    *,
    gamma: float,
    T0: float,
    diffusion_time: float,
    radius: int,
    graph: ActionGraph | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
) -> PlanningResult:
    """Dense correctness reference for the normalized finite-walk target."""

    action_graph = cycle_graph(mdp.K) if graph is None else graph
    if action_graph.K != mdp.K:
        raise ValueError("action graph and MDP must have the same action count")
    counters = OperationCounters()
    start = perf_counter()
    heat = truncated_heat_kernel(action_graph.laplacian, diffusion_time, radius)
    counters.geometry_preprocess_seconds += perf_counter() - start
    counters.dense_linear_algebra_operations += max(1, int(radius))
    counters.observe_memory()
    return dense_value_iteration(
        mdp,
        gamma=gamma,
        method="full_truncated_heat",
        target="truncated_heat",
        weights=heat,
        T0=T0,
        tolerance=tolerance,
        max_iterations=max_iterations,
        counter=counters,
    )


def uniform_maxent_reference(
    mdp: RingControlMDP,
    *,
    gamma: float,
    T0: float,
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
) -> PlanningResult:
    prior = np.full((mdp.K, 1), 1.0 / mdp.K, dtype=np.float64)
    anchors = np.ones((mdp.K, 1), dtype=np.float64)
    return dense_value_iteration(
        mdp,
        gamma=gamma,
        method="uniform_maxent",
        target="maxent",
        weights=prior,
        anchor_distributions=anchors,
        T0=T0,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )


def hard_max_reference(
    mdp: RingControlMDP,
    *,
    gamma: float,
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
) -> PlanningResult:
    return dense_value_iteration(
        mdp,
        gamma=gamma,
        method="hard_max",
        target="hard_max",
        tolerance=tolerance,
        max_iterations=max_iterations,
    )


def diffusion_gibbs_reference(
    mdp: RingControlMDP,
    *,
    gamma: float,
    T0: float,
    diffusion_time: float,
    lambda_: float,
    graph: ActionGraph | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
) -> PlanningResult:
    action_graph = cycle_graph(mdp.K) if graph is None else graph
    if action_graph.K != mdp.K:
        raise ValueError("action graph and MDP must have the same action count")
    counters = OperationCounters()
    start = perf_counter()
    weights, _, _ = diffusion_distance_gibbs_kernel(
        action_graph.laplacian,
        diffusion_time,
        lambda_,
        batch_size=min(128, mdp.K),
    )
    counters.geometry_preprocess_seconds += perf_counter() - start
    counters.dense_linear_algebra_operations += 1
    counters.observe_memory()
    return dense_value_iteration(
        mdp,
        gamma=gamma,
        method="diffusion_distance_gibbs",
        target="diffusion_gibbs",
        weights=weights,
        T0=T0,
        tolerance=tolerance,
        max_iterations=max_iterations,
        counter=counters,
    )


def permuted_action_graph_reference(
    mdp: RingControlMDP,
    *,
    gamma: float,
    T0: float,
    diffusion_time: float,
    permutation: ArrayLike,
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
) -> PlanningResult:
    """Exact heat on a relabeled, deliberately misaligned cycle graph."""

    order = np.asarray(permutation, dtype=np.int64)
    if order.shape != (mdp.K,) or not np.array_equal(np.sort(order), np.arange(mdp.K)):
        raise ValueError("permutation must contain every action exactly once")
    base = cycle_graph(mdp.K)
    counters = OperationCounters()
    start = perf_counter()
    adjacency = base.adjacency[order, :][:, order]
    graph = ActionGraph(adjacency=adjacency, family="permuted_cycle")
    heat = exact_heat_kernel(graph.laplacian, diffusion_time)
    counters.geometry_preprocess_seconds += perf_counter() - start
    counters.dense_linear_algebra_operations += 1
    counters.observe_memory()
    return dense_value_iteration(
        mdp,
        gamma=gamma,
        method="permuted_action_graph",
        target="permuted_exact_heat",
        weights=heat,
        T0=T0,
        tolerance=tolerance,
        max_iterations=max_iterations,
        counter=counters,
    )


def stable_logmeanexp(values: ArrayLike) -> float:
    """Utility used by sampled baselines."""

    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or len(vector) == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("values must be a nonempty finite vector")
    return float(logsumexp(vector) - math.log(len(vector)))
