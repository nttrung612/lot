"""End-to-end planning with frozen empirical LOT Bellman operators.

The sampled anchors and candidate measures are drawn once per paired run and
then held fixed.  Each resulting operator remains a monotone,
translation-equivariant gamma contraction, so ordinary value iteration has a
well-defined per-seed fixed point.  These are empirical baselines; no
finite-sample certificate is inferred from their plug-in construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp

from lot_experiments.counters import OperationCounters
from lot_experiments.environments.ring_control import RingControlMDP
from lot_experiments.planners.dense import PlanningResult


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class EmpiricalComponent:
    """One sampled anchor contribution ``T0 log sum_i weight_i exp(q_i/T0)``."""

    actions: IntArray
    log_weights: FloatArray

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=np.int64)
        log_weights = np.asarray(self.log_weights, dtype=np.float64)
        if actions.ndim != 1 or len(actions) == 0 or actions.shape != log_weights.shape:
            raise ValueError("component actions and log_weights must be aligned vectors")
        if not np.all(np.isfinite(log_weights)):
            raise ValueError("component log_weights must be finite")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "log_weights", log_weights)


def sample_anchor_table(
    mdp: RingControlMDP,
    samples_per_state: int,
    rng: np.random.Generator,
) -> IntArray:
    """Draw paired outer anchors for every state."""

    if samples_per_state < 1:
        raise ValueError("samples_per_state must be positive")
    anchors = np.empty((mdp.K, samples_per_state), dtype=np.int64)
    for state in range(mdp.K):
        anchors[state] = rng.choice(
            mdp.K,
            size=samples_per_state,
            replace=True,
            p=mdp.anchor_distributions[state],
        )
    return anchors


def component_from_weights(
    weights: ArrayLike,
    *,
    actions: ArrayLike | None = None,
) -> EmpiricalComponent:
    """Construct a component, preserving unnormalized retained mass."""

    vector = np.asarray(weights, dtype=np.float64)
    if vector.ndim != 1 or np.any(vector < 0.0) or not np.any(vector > 0.0):
        raise ValueError("weights must be a nonzero nonnegative vector")
    selected = (
        np.flatnonzero(vector > 0.0).astype(np.int64)
        if actions is None
        else np.asarray(actions, dtype=np.int64)
    )
    selected_weights = vector[selected]
    positive = selected_weights > 0.0
    selected = selected[positive]
    selected_weights = selected_weights[positive]
    if len(selected) == 0:
        raise ValueError("the selected component has zero mass")
    return EmpiricalComponent(selected, np.log(selected_weights))


def component_from_log_weights(
    log_weights: ArrayLike,
    *,
    actions: ArrayLike | None = None,
) -> EmpiricalComponent:
    """Construct a component without exponentiating tiny positive weights."""

    vector = np.asarray(log_weights, dtype=np.float64)
    if vector.ndim != 1 or np.any(np.isnan(vector)) or np.any(np.isposinf(vector)):
        raise ValueError("log_weights must contain finite values or negative infinity")
    selected = (
        np.flatnonzero(np.isfinite(vector)).astype(np.int64)
        if actions is None
        else np.asarray(actions, dtype=np.int64)
    )
    selected_logs = vector[selected]
    positive = np.isfinite(selected_logs)
    selected = selected[positive]
    selected_logs = selected_logs[positive]
    if len(selected) == 0:
        raise ValueError("the selected component has zero mass")
    return EmpiricalComponent(selected, selected_logs)


def component_from_samples(
    actions: ArrayLike,
    unnormalized_weights: ArrayLike,
) -> EmpiricalComponent:
    """Aggregate duplicate sampled actions into one discrete measure."""

    sampled = np.asarray(actions, dtype=np.int64)
    weights = np.asarray(unnormalized_weights, dtype=np.float64)
    if sampled.ndim != 1 or sampled.shape != weights.shape or len(sampled) == 0:
        raise ValueError("sampled actions and weights must be aligned nonempty vectors")
    if np.any(weights < 0.0) or not np.any(weights > 0.0):
        raise ValueError("sample weights must be nonnegative with positive mass")
    unique, inverse = np.unique(sampled, return_inverse=True)
    aggregated = np.zeros(len(unique), dtype=np.float64)
    np.add.at(aggregated, inverse, weights)
    positive = aggregated > 0.0
    return EmpiricalComponent(unique[positive], np.log(aggregated[positive]))


def component_from_log_samples(
    actions: ArrayLike,
    log_unnormalized_weights: ArrayLike,
) -> EmpiricalComponent:
    """Aggregate duplicate sampled actions while preserving subnormal tails."""

    sampled = np.asarray(actions, dtype=np.int64)
    log_weights = np.asarray(log_unnormalized_weights, dtype=np.float64)
    if sampled.ndim != 1 or sampled.shape != log_weights.shape or len(sampled) == 0:
        raise ValueError("sampled actions and log weights must be aligned vectors")
    if np.any(np.isnan(log_weights)) or np.any(np.isposinf(log_weights)):
        raise ValueError("sample log weights must be finite or negative infinity")
    unique, inverse = np.unique(sampled, return_inverse=True)
    aggregated = np.array(
        [
            logsumexp(log_weights[inverse == index])
            for index in range(len(unique))
        ],
        dtype=np.float64,
    )
    positive = np.isfinite(aggregated)
    if not np.any(positive):
        raise ValueError("sample weights have zero mass")
    return EmpiricalComponent(unique[positive], aggregated[positive])


def empirical_value_iteration(
    mdp: RingControlMDP,
    components: list[tuple[EmpiricalComponent, ...]],
    *,
    gamma: float,
    T0: float,
    method: str,
    target: str,
    tolerance: float = 1e-9,
    max_iterations: int = 10_000,
    counter: OperationCounters | None = None,
) -> PlanningResult:
    """Solve the fixed point of a frozen empirical LOT operator."""

    if len(components) != mdp.K or any(len(row) == 0 for row in components):
        raise ValueError("every state needs at least one empirical component")
    if not 0.0 <= gamma < 1.0 or T0 <= 0.0:
        raise ValueError("gamma must lie in [0,1) and T0 must be positive")
    if tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("tolerance and max_iterations must be positive")
    counters = OperationCounters() if counter is None else counter
    state_actions = [
        np.unique(np.concatenate([component.actions for component in row]))
        for row in components
    ]
    max_query_count = max(map(len, state_actions))
    query_actions = np.zeros((mdp.K, max_query_count), dtype=np.int64)
    query_mask = np.zeros((mdp.K, max_query_count), dtype=bool)
    for state, actions in enumerate(state_actions):
        query_actions[state, : len(actions)] = actions
        query_mask[state, : len(actions)] = True

    max_component_count = max(map(len, components))
    max_component_size = max(
        len(component.actions) for row in components for component in row
    )
    component_actions = np.zeros(
        (mdp.K, max_component_count, max_component_size), dtype=np.int64
    )
    component_log_weights = np.full(
        (mdp.K, max_component_count, max_component_size),
        -math.inf,
        dtype=np.float64,
    )
    component_present = np.zeros((mdp.K, max_component_count), dtype=bool)
    component_mass = np.zeros((mdp.K, max_component_count), dtype=np.float64)
    for state, row in enumerate(components):
        component_present[state, : len(row)] = True
        component_mass[state, : len(row)] = 1.0 / len(row)
        for index, component in enumerate(row):
            size = len(component.actions)
            component_actions[state, index, :size] = component.actions
            component_log_weights[state, index, :size] = component.log_weights

    state_index = np.arange(mdp.K, dtype=np.int64)
    query_states = np.broadcast_to(state_index[:, None], query_actions.shape)
    component_states = np.broadcast_to(
        state_index[:, None, None], component_actions.shape
    )
    touched_actions = np.unique(query_actions[query_mask])
    for action in touched_actions:
        counters.touch_action(int(action), evaluations=0)
    queries_per_application = int(query_mask.sum())
    values = np.zeros(mdp.K, dtype=np.float64)
    policy = np.zeros((mdp.K, mdp.K), dtype=np.float64)
    q_values = np.full((mdp.K, mdp.K), np.nan, dtype=np.float64)

    def apply(
        current: FloatArray, *, diagnostics: bool = False
    ) -> tuple[FloatArray, FloatArray | None, FloatArray | None]:
        base = (query_states + query_actions) % mdp.K
        expectation = np.zeros(query_actions.shape, dtype=np.float64)
        for offset, probability in zip(
            mdp.noise_offsets, mdp.noise_probabilities, strict=True
        ):
            expectation += float(probability) * current[(base + int(offset)) % mdp.K]
        queried = (
            mdp.reward_matrix[query_states, query_actions] + gamma * expectation
        )
        current_q = np.full((mdp.K, mdp.K), np.nan, dtype=np.float64)
        current_q[query_states[query_mask], query_actions[query_mask]] = queried[
            query_mask
        ]
        counters.transition_calls += queries_per_application
        counters.action_evaluations += queries_per_application

        gathered = current_q[component_states, component_actions]
        valid_weights = np.isfinite(component_log_weights)
        scores = np.where(
            valid_weights,
            component_log_weights + gathered / T0,
            -math.inf,
        )
        log_partitions = logsumexp(scores, axis=2)
        weighted_partitions = component_mass * np.where(
            component_present, log_partitions, 0.0
        )
        updated = T0 * weighted_partitions.sum(axis=1)
        if not diagnostics:
            return updated, None, None

        normalizers = np.where(component_present, log_partitions, 0.0)
        probabilities = np.where(
            valid_weights,
            np.exp(scores - normalizers[:, :, None])
            * component_mass[:, :, None],
            0.0,
        )
        current_policy = np.zeros((mdp.K, mdp.K), dtype=np.float64)
        np.add.at(
            current_policy,
            (component_states.ravel(), component_actions.ravel()),
            probabilities.ravel(),
        )
        current_policy /= current_policy.sum(axis=1, keepdims=True)
        return updated, current_policy, current_q

    start = perf_counter()
    residual = math.inf
    for iteration in range(1, int(max_iterations) + 1):
        updated, _, _ = apply(values)
        residual = float(np.max(np.abs(updated - values)))
        values = updated
        if residual <= tolerance:
            converged = True
            break
    else:
        iteration = int(max_iterations)
        converged = False
    bellman_values, final_policy, final_q_values = apply(values, diagnostics=True)
    assert final_policy is not None and final_q_values is not None
    policy = final_policy
    q_values = final_q_values
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
