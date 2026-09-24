"""M8 Pendulum baselines, fitted planners, evaluation, and summaries."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import multiprocessing
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd
import psutil
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp

from lot_experiments.config import config_json
from lot_experiments.counters import OperationCounters
from lot_experiments.environments.pendulum_discrete import (
    PendulumDiscreteEnv,
    PendulumStateGrid,
    normalize_angle,
)
from lot_experiments.graphs import path_graph
from lot_experiments.heat_local import LazyTruncatedHeat
from lot_experiments.pendulum_reference import EVALUATION_STATES, PathHeatOperator
from lot_experiments.planners.poisson_mc import sample_poisson_endpoint
from lot_experiments.pruning import top_mass_mask
from lot_experiments.reproducibility import rng_for, run_metadata
from lot_experiments.results import result_row, validate_results, write_results_atomic


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
LOGGER = logging.getLogger(__name__)

SAME_TARGET_METHODS = (
    "full_exact_heat",
    "exact_heat_topmass",
    "truncated_heat_local",
    "poisson_endpoint_mc",
    "uniform_random_subset",
    "local_uniform_neighborhood",
    "local_rbf_neighborhood",
)
DIFFERENT_TARGET_METHODS = (
    "uniform_maxent",
    "hard_max",
    "squared_torque_lot",
    "permuted_path_graph",
)


@dataclass
class EvaluationCounters:
    transition_calls: int = 0
    action_evaluations: int = 0
    graph_neighbor_accesses: int = 0
    backups: int = 0


class BellmanOperator:
    method: str
    target: str
    reference_target = "exact_heat"
    geometry_preprocess_seconds: float = 0.0
    graph_neighbor_accesses: int = 0

    def evaluate(
        self,
        environment: PendulumDiscreteEnv,
        grid: PendulumStateGrid,
        values: FloatArray,
        theta: FloatArray,
        theta_dot: FloatArray,
        *,
        gamma: float,
        T0: float,
        angle_gain: float,
        velocity_gain: float,
        return_policy: bool,
        counters: EvaluationCounters,
    ) -> tuple[FloatArray, FloatArray | None]:
        raise NotImplementedError


@dataclass
class DenseColumnOperator(BellmanOperator):
    method: str
    target: str
    weights: FloatArray

    def evaluate(
        self,
        environment: PendulumDiscreteEnv,
        grid: PendulumStateGrid,
        values: FloatArray,
        theta: FloatArray,
        theta_dot: FloatArray,
        *,
        gamma: float,
        T0: float,
        angle_gain: float,
        velocity_gain: float,
        return_policy: bool,
        counters: EvaluationCounters,
    ) -> tuple[FloatArray, FloatArray | None]:
        q = environment.q_values(values, grid, theta, theta_dot, gamma)
        anchors = environment.nominal_action_indices(
            theta,
            theta_dot,
            angle_gain=angle_gain,
            velocity_gain=velocity_gain,
        )
        selected = self.weights[:, anchors].T
        maxima = np.max(q, axis=1)
        exponentials = np.exp((q - maxima[:, None]) / T0)
        partitions = np.sum(exponentials * selected, axis=1)
        if np.any(partitions <= 0.0):
            raise FloatingPointError(f"{self.method} produced a zero partition")
        backed_up = maxima + T0 * np.log(partitions)
        counters.transition_calls += q.size
        counters.action_evaluations += q.size
        counters.backups += len(theta)
        if not return_policy:
            return backed_up, None
        policy = exponentials * selected / partitions[:, None]
        policy /= policy.sum(axis=1, keepdims=True)
        return backed_up, policy


@dataclass
class HardMaxOperator(BellmanOperator):
    method: str = "hard_max"
    target: str = "hard_max"

    def evaluate(self, environment, grid, values, theta, theta_dot, *, gamma, T0,
                 angle_gain, velocity_gain, return_policy, counters):
        del T0, angle_gain, velocity_gain
        q = environment.q_values(values, grid, theta, theta_dot, gamma)
        maxima = np.max(q, axis=1)
        counters.transition_calls += q.size
        counters.action_evaluations += q.size
        counters.backups += len(theta)
        if not return_policy:
            return maxima, None
        maximizers = q == maxima[:, None]
        return maxima, maximizers / maximizers.sum(axis=1, keepdims=True)


@dataclass
class SparseColumnOperator(BellmanOperator):
    method: str
    target: str
    supports: list[IntArray | None]
    weights: list[FloatArray | None]
    geometry_preprocess_seconds: float = 0.0
    graph_neighbor_accesses: int = 0
    provider: Callable[[int], tuple[IntArray, FloatArray]] | None = None
    graph_counter: OperationCounters | None = None

    def _column(self, anchor: int) -> tuple[IntArray, FloatArray]:
        support = self.supports[anchor]
        weights = self.weights[anchor]
        if support is None or weights is None:
            if self.provider is None:
                raise RuntimeError(f"{self.method} has no column provider")
            start = perf_counter()
            support, weights = self.provider(anchor)
            self.geometry_preprocess_seconds += perf_counter() - start
            self.supports[anchor] = support
            self.weights[anchor] = weights
        return support, weights

    def evaluate(
        self,
        environment: PendulumDiscreteEnv,
        grid: PendulumStateGrid,
        values: FloatArray,
        theta: FloatArray,
        theta_dot: FloatArray,
        *,
        gamma: float,
        T0: float,
        angle_gain: float,
        velocity_gain: float,
        return_policy: bool,
        counters: EvaluationCounters,
    ) -> tuple[FloatArray, FloatArray | None]:
        anchors = environment.nominal_action_indices(
            theta,
            theta_dot,
            angle_gain=angle_gain,
            velocity_gain=velocity_gain,
        )
        backed_up = np.empty(len(theta), dtype=np.float64)
        policy = (
            np.zeros((len(theta), environment.K), dtype=np.float64)
            if return_policy
            else None
        )
        for anchor in np.unique(anchors):
            rows = np.flatnonzero(anchors == anchor)
            support, weights = self._column(int(anchor))
            q = environment.q_values(
                values,
                grid,
                theta[rows],
                theta_dot[rows],
                gamma,
                action_indices=support,
            )
            scores = q / T0 + np.log(weights)[None, :]
            log_partitions = logsumexp(scores, axis=1)
            backed_up[rows] = T0 * log_partitions
            queries = int(q.size)
            counters.transition_calls += queries
            counters.action_evaluations += queries
            counters.backups += len(rows)
            if policy is not None:
                probabilities = np.exp(scores - log_partitions[:, None])
                policy[np.ix_(rows, support)] = probabilities
        return backed_up, policy


@dataclass(frozen=True)
class FittedSolution:
    values: FloatArray
    evaluation_values: FloatArray
    evaluation_policies: FloatArray
    iterations: int
    residual: float
    residual_mean: float
    converged: bool
    online_seconds: float
    peak_memory_mb: float
    counters: EvaluationCounters


def _solve_operator(
    environment: PendulumDiscreteEnv,
    grid: PendulumStateGrid,
    operator: BellmanOperator,
    evaluation_states: FloatArray,
    *,
    gamma: float,
    T0: float,
    tolerance: float,
    max_iterations: int,
    chunk_size: int,
    angle_gain: float,
    velocity_gain: float,
    progress_interval: int,
    anderson_depth: int,
    initial_values: ArrayLike | None = None,
) -> FittedSolution:
    values = (
        np.zeros(grid.shape, dtype=np.float64)
        if initial_values is None
        else np.asarray(initial_values, dtype=np.float64).copy()
    )
    if values.shape != grid.shape:
        raise ValueError("initial fitted values do not match the state grid")
    reference_index = (grid.angle_count // 2, grid.velocity_count // 2)
    values -= values[reference_index]
    all_theta, all_velocity = grid.states()
    counters = EvaluationCounters(graph_neighbor_accesses=operator.graph_neighbor_accesses)
    start = perf_counter()
    residual = math.inf
    residual_mean = math.inf
    converged = False
    previous_values: FloatArray | None = None
    previous_fixed_residual: FloatArray | None = None
    value_differences: list[FloatArray] = []
    residual_differences: list[FloatArray] = []
    for iteration in range(1, max_iterations + 1):
        updated_flat = np.empty(grid.size, dtype=np.float64)
        for first in range(0, grid.size, chunk_size):
            last = min(first + chunk_size, grid.size)
            updated_flat[first:last], _ = operator.evaluate(
                environment,
                grid,
                values,
                all_theta[first:last],
                all_velocity[first:last],
                gamma=gamma,
                T0=T0,
                angle_gain=angle_gain,
                velocity_gain=velocity_gain,
                return_policy=False,
                counters=counters,
            )
        updated = updated_flat.reshape(grid.shape)
        difference = updated - values
        shift = float(difference[reference_index])
        centered = updated - updated[reference_index]
        centered_residual = float(np.max(np.abs(centered - values)))
        if progress_interval and (iteration == 1 or iteration % progress_interval == 0):
            LOGGER.info(
                "Pendulum M8 method=%s K=%s iteration=%s residual=%.3e",
                operator.method,
                environment.K,
                iteration,
                centered_residual,
            )
        if centered_residual <= tolerance:
            values = values + shift / (1.0 - gamma)
            check_flat = np.empty(grid.size, dtype=np.float64)
            check_counters = EvaluationCounters()
            for first in range(0, grid.size, chunk_size):
                last = min(first + chunk_size, grid.size)
                check_flat[first:last], _ = operator.evaluate(
                    environment,
                    grid,
                    values,
                    all_theta[first:last],
                    all_velocity[first:last],
                    gamma=gamma,
                    T0=T0,
                    angle_gain=angle_gain,
                    velocity_gain=velocity_gain,
                    return_policy=False,
                    counters=check_counters,
                )
            counters.transition_calls += check_counters.transition_calls
            counters.action_evaluations += check_counters.action_evaluations
            counters.backups += check_counters.backups
            residual_values = np.abs(check_flat.reshape(grid.shape) - values)
            residual = float(np.max(residual_values))
            residual_mean = float(np.mean(residual_values))
            converged = residual <= tolerance * 1.05
            if converged:
                break
        fixed_residual = centered - values
        if previous_values is not None and previous_fixed_residual is not None:
            value_differences.append((values - previous_values).ravel())
            residual_differences.append(
                (fixed_residual - previous_fixed_residual).ravel()
            )
            if len(value_differences) > anderson_depth:
                value_differences.pop(0)
                residual_differences.pop(0)
        next_values = centered
        if anderson_depth and residual_differences:
            delta_values = np.column_stack(value_differences)
            delta_residuals = np.column_stack(residual_differences)
            gram = delta_residuals.T @ delta_residuals
            right = delta_residuals.T @ fixed_residual.ravel()
            ridge = 1e-12 * max(1.0, float(np.trace(gram)))
            coefficients = np.linalg.solve(
                gram + ridge * np.eye(gram.shape[0]), right
            )
            accelerated = centered.ravel() - (
                delta_values + delta_residuals
            ) @ coefficients
            accelerated = accelerated.reshape(grid.shape)
            accelerated -= accelerated[reference_index]
            if np.all(np.isfinite(accelerated)) and np.max(np.abs(accelerated)) <= (
                10.0 * max(1.0, float(np.max(np.abs(centered))))
            ):
                next_values = accelerated
        previous_values = values.copy()
        previous_fixed_residual = fixed_residual.copy()
        values = next_values
    elapsed = perf_counter() - start
    if isinstance(operator, SparseColumnOperator) and operator.graph_counter is not None:
        counters.graph_neighbor_accesses = operator.graph_counter.graph_neighbor_accesses
    evaluation_counters = EvaluationCounters()
    evaluation_values, policies = operator.evaluate(
        environment,
        grid,
        values,
        evaluation_states[:, 0],
        evaluation_states[:, 1],
        gamma=gamma,
        T0=T0,
        angle_gain=angle_gain,
        velocity_gain=velocity_gain,
        return_policy=True,
        counters=evaluation_counters,
    )
    counters.transition_calls += evaluation_counters.transition_calls
    counters.action_evaluations += evaluation_counters.action_evaluations
    counters.backups += evaluation_counters.backups
    assert policies is not None
    return FittedSolution(
        values=values,
        evaluation_values=evaluation_values,
        evaluation_policies=policies,
        iterations=iteration,
        residual=residual,
        residual_mean=residual_mean,
        converged=converged,
        online_seconds=elapsed,
        peak_memory_mb=psutil.Process().memory_info().rss / (1024.0**2),
        counters=counters,
    )


def _dense_to_sparse(
    method: str,
    target: str,
    kernel: FloatArray,
    masks: NDArray[np.bool_],
    *,
    geometry_seconds: float = 0.0,
) -> SparseColumnOperator:
    supports: list[IntArray] = []
    weights: list[FloatArray] = []
    for anchor in range(kernel.shape[1]):
        support = np.flatnonzero(masks[:, anchor] & (kernel[:, anchor] > 0.0))
        if not len(support):
            raise ValueError(f"{method} retained no action for anchor {anchor}")
        supports.append(support.astype(np.int64))
        weights.append(kernel[support, anchor].astype(np.float64))
    return SparseColumnOperator(
        method=method,
        target=target,
        supports=list(supports),
        weights=list(weights),
        geometry_preprocess_seconds=geometry_seconds,
    )


def _build_operators(
    environment: PendulumDiscreteEnv,
    *,
    heat_scaling: str,
    diffusion_time: float,
    radius: int,
    alpha: float,
    endpoint_samples: int,
    random_subset_size: int,
    rbf_sigma: float,
    squared_torque_lambda: float,
    seed: int,
    methods: Sequence[str],
) -> dict[str, BellmanOperator]:
    edge_weight = 1.0 if heat_scaling == "index_heat" else 1.0 / environment.action_spacing**2
    start = perf_counter()
    exact = PathHeatOperator(environment.K, diffusion_time, edge_weight).dense_kernel
    assert exact is not None
    exact_seconds = perf_counter() - start
    operators: dict[str, BellmanOperator] = {}
    if "full_exact_heat" in methods:
        operator = DenseColumnOperator("full_exact_heat", "exact_heat", exact)
        operator.geometry_preprocess_seconds = exact_seconds
        operators[operator.method] = operator
    if "exact_heat_topmass" in methods:
        masks = np.column_stack(
            [top_mass_mask(exact[:, anchor], alpha) for anchor in range(environment.K)]
        )
        operators["exact_heat_topmass"] = _dense_to_sparse(
            "exact_heat_topmass", "exact_heat", exact, masks, geometry_seconds=exact_seconds
        )
    graph = path_graph(environment.K, edge_weight=edge_weight)
    if any(
        method in methods
        for method in (
            "truncated_heat_local",
            "local_uniform_neighborhood",
            "local_rbf_neighborhood",
        )
    ):
        empty_supports: list[IntArray | None] = [None] * environment.K
        empty_weights: list[FloatArray | None] = [None] * environment.K
        if "truncated_heat_local" in methods:
            heat_counter = OperationCounters()
            heat_lazy = LazyTruncatedHeat(
                graph, diffusion_time, radius, counter=heat_counter
            )
            operators["truncated_heat_local"] = SparseColumnOperator(
                "truncated_heat_local",
                "truncated_heat",
                empty_supports.copy(),
                empty_weights.copy(),
                provider=lambda anchor: heat_lazy.sparse_column(anchor),
                graph_counter=heat_counter,
            )
        if "local_uniform_neighborhood" in methods:
            uniform_counter = OperationCounters()
            uniform_lazy = LazyTruncatedHeat(
                graph, diffusion_time, radius, counter=uniform_counter
            )
            def uniform_provider(anchor: int) -> tuple[IntArray, FloatArray]:
                support, _ = uniform_lazy.sparse_column(anchor)
                return support, np.full(len(support), 1.0 / len(support))

            operators["local_uniform_neighborhood"] = SparseColumnOperator(
                "local_uniform_neighborhood", "local_uniform",
                empty_supports.copy(), empty_weights.copy(),
                provider=uniform_provider, graph_counter=uniform_counter
            )
        if "local_rbf_neighborhood" in methods:
            rbf_counter = OperationCounters()
            rbf_lazy = LazyTruncatedHeat(
                graph, diffusion_time, radius, counter=rbf_counter
            )
            def rbf_provider(anchor: int) -> tuple[IntArray, FloatArray]:
                support, _ = rbf_lazy.sparse_column(anchor)
                distances = (environment.actions[support] - environment.actions[anchor]) / rbf_sigma
                weights = np.exp(-0.5 * distances**2)
                return support, weights / weights.sum()

            operators["local_rbf_neighborhood"] = SparseColumnOperator(
                "local_rbf_neighborhood", "local_rbf",
                empty_supports.copy(), empty_weights.copy(),
                provider=rbf_provider, graph_counter=rbf_counter
            )
    if "poisson_endpoint_mc" in methods:
        rng = rng_for(seed, "pendulum_poisson_endpoints")
        counter = OperationCounters()
        start = perf_counter()
        supports = []
        weights = []
        for anchor in range(environment.K):
            endpoints = np.asarray(
                [
                    sample_poisson_endpoint(
                        graph, anchor, diffusion_time, rng, counter=counter
                    )
                    for _ in range(endpoint_samples)
                ],
                dtype=np.int64,
            )
            support, counts = np.unique(endpoints, return_counts=True)
            supports.append(support)
            weights.append(counts.astype(np.float64) / endpoint_samples)
        operators["poisson_endpoint_mc"] = SparseColumnOperator(
            "poisson_endpoint_mc", "exact_heat", supports, weights,
            perf_counter() - start, counter.graph_neighbor_accesses
        )
    if "uniform_random_subset" in methods:
        rng = rng_for(seed, "pendulum_random_subsets")
        masks = np.zeros_like(exact, dtype=bool)
        for anchor in range(environment.K):
            subset_size = min(random_subset_size, environment.K)
            selected = rng.choice(
                environment.K,
                size=subset_size,
                replace=False,
            )
            if anchor not in selected:
                selected[0] = anchor
            masks[selected, anchor] = True
        operators["uniform_random_subset"] = _dense_to_sparse(
            "uniform_random_subset", "exact_heat", exact, masks,
            geometry_seconds=exact_seconds
        )
    if "uniform_maxent" in methods:
        uniform = np.full_like(exact, 1.0 / environment.K)
        operators["uniform_maxent"] = DenseColumnOperator(
            "uniform_maxent", "maxent", uniform
        )
    if "hard_max" in methods:
        operators["hard_max"] = HardMaxOperator()
    if "squared_torque_lot" in methods:
        delta = environment.actions[:, None] - environment.actions[None, :]
        squared = np.exp(-(delta**2) / squared_torque_lambda)
        squared /= squared.sum(axis=0, keepdims=True)
        operators["squared_torque_lot"] = DenseColumnOperator(
            "squared_torque_lot", "squared_torque_lot", squared
        )
    if "permuted_path_graph" in methods:
        permutation = rng_for(seed, "pendulum_permuted_path").permutation(environment.K)
        permuted = exact[np.ix_(permutation, permutation)]
        operators["permuted_path_graph"] = DenseColumnOperator(
            "permuted_path_graph", "permuted_heat", permuted
        )
    return operators


def _behavior_metrics(
    solution: FittedSolution,
    operator: BellmanOperator,
    environment: PendulumDiscreteEnv,
    grid: PendulumStateGrid,
    initial_states: FloatArray,
    *,
    gamma: float,
    T0: float,
    angle_gain: float,
    velocity_gain: float,
    horizon: int,
    seed: int,
) -> dict[str, float]:
    # The same uniform draws are reused across methods; differing policies map
    # them to actions differently while preserving paired episode randomness.
    rng = rng_for(seed, "pendulum_rollout")
    returns: list[float] = []
    angle_errors: list[float] = []
    successes: list[float] = []
    torques: list[float] = []
    for episode, (initial_theta, initial_velocity) in enumerate(initial_states):
        del episode
        theta = float(initial_theta)
        velocity = float(initial_velocity)
        raw_return = 0.0
        for _ in range(horizon):
            _, policy = operator.evaluate(
                environment,
                grid,
                solution.values,
                np.asarray([theta]),
                np.asarray([velocity]),
                gamma=gamma,
                T0=T0,
                angle_gain=angle_gain,
                velocity_gain=velocity_gain,
                return_policy=True,
                counters=EvaluationCounters(),
            )
            assert policy is not None
            action = int(rng.choice(environment.K, p=policy[0]))
            torque = float(environment.actions[action])
            raw_return += float(environment.raw_reward(theta, velocity, torque))
            angle = float(normalize_angle(theta))
            angle_errors.append(angle**2)
            successes.append(float(abs(angle) <= 0.2 and abs(velocity) <= 1.0))
            torques.append(abs(torque))
            next_theta, next_velocity = environment.transition(theta, velocity, torque)
            theta, velocity = float(next_theta), float(next_velocity)
        returns.append(raw_return)
    return {
        "raw_return_mean": float(np.mean(returns)),
        "mean_squared_angle_error": float(np.mean(angle_errors)),
        "upright_fraction": float(np.mean(successes)),
        "mean_absolute_torque": float(np.mean(torques)),
    }


def resolve_pendulum_experiment_config(config: Mapping[str, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "experiment": "pendulum",
        "seed": 0,
        "K": [51, 101, 201, 401, 801],
        "heat_scalings": ["index_heat", "physical_heat"],
        "diffusion_time": {"index_heat": 2.0, "physical_heat": 0.02},
        "temperature": [0.02, 0.05, 0.1],
        "gamma": 0.99,
        "reward_scale": 16.2736044,
        "development_grid": [129, 129],
        "evaluation_reset_states": 100,
        "execution": {"workers": 1},
        "methods": [*SAME_TARGET_METHODS, *DIFFERENT_TARGET_METHODS],
        "radii": [2, 4, 8, 16],
        "primary_radius": 16,
        "physical_radius_reference_K": 101,
        "topmass_alpha": 0.01,
        "endpoint_samples": 16,
        "random_subset_size": 0,
        "rbf_sigma": 0.25,
        "squared_torque_lambda": 0.25,
        "planning": {
            "tolerance": 1e-6,
            "max_iterations": 3000,
            "state_chunk_size": 256,
            "progress_interval": 100,
            "anderson_depth": 5,
        },
        "anchor": {"angle_gain": 2.0, "velocity_gain": 0.5},
        "behavior": {
            "enabled": True,
            "episodes": 100,
            "horizon": 200,
            "K": 101,
            "heat_scaling": "physical_heat",
            "temperature": 0.05,
        },
        "radius_ablation": {
            "enabled": True,
            "K": 101,
            "heat_scaling": "physical_heat",
            "temperature": 0.05,
        },
        "figure": {"K": 101, "heat_scaling": "physical_heat", "temperature": 0.05},
        "raw_output": "outputs/raw/pendulum.parquet",
        "summary_output": "outputs/summaries/pendulum.csv",
        "figure_png": "outputs/figures/figure4_pendulum.png",
    }
    resolved = copy.deepcopy(defaults)
    for key, value in config.items():
        if key in {
            "planning",
            "anchor",
            "behavior",
            "radius_ablation",
            "figure",
            "diffusion_time",
            "execution",
        } and isinstance(value, Mapping):
            resolved[key].update(value)
        else:
            resolved[key] = copy.deepcopy(value)
    methods = list(map(str, resolved["methods"]))
    unknown = set(methods) - set(SAME_TARGET_METHODS) - set(DIFFERENT_TARGET_METHODS)
    if unknown:
        raise ValueError(f"unknown Pendulum methods: {sorted(unknown)}")
    if "full_exact_heat" not in methods:
        raise ValueError("M8 requires FullExactHeat in every case")
    if int(resolved["primary_radius"]) < 0:
        raise ValueError("primary_radius must be nonnegative")
    workers = resolved["execution"]["workers"]
    if (
        isinstance(workers, bool)
        or not isinstance(workers, (int, np.integer))
        or int(workers) < 1
    ):
        raise ValueError("execution.workers must be a positive integer")
    return resolved


def _run_id(parts: tuple[Any, ...], resolved_json: str) -> str:
    payload = json.dumps([parts, resolved_json], sort_keys=True).encode()
    return hashlib.blake2b(payload, digest_size=10).hexdigest()


def _run_pendulum_experiment_serial(config: Mapping[str, Any]) -> pd.DataFrame:
    resolved = resolve_pendulum_experiment_config(config)
    resolved_json = config_json(resolved)
    metadata = run_metadata(resolved)
    fixed = np.asarray(EVALUATION_STATES, dtype=np.float64)
    reset_count = int(resolved["evaluation_reset_states"])
    reset_states = rng_for(int(resolved["seed"]), "pendulum_evaluation_states").uniform(
        low=(-np.pi, -1.0), high=(np.pi, 1.0), size=(reset_count, 2)
    )
    evaluation_states = np.vstack((fixed, reset_states))
    raw_path = Path(resolved["raw_output"])
    existing = pd.DataFrame()
    if raw_path.exists():
        existing = (
            pd.read_parquet(raw_path)
            if raw_path.suffix == ".parquet"
            else pd.read_csv(raw_path)
        )
        if "config_json" in existing and not existing.empty:
            existing = existing.loc[existing["config_json"] == resolved_json].copy()
    rows: list[dict[str, Any]] = existing.to_dict("records")
    grid = PendulumStateGrid(*map(int, resolved["development_grid"]))
    planning = resolved["planning"]
    angle_gain = float(resolved["anchor"]["angle_gain"])
    velocity_gain = float(resolved["anchor"]["velocity_gain"])
    for K in map(int, resolved["K"]):
        environment = PendulumDiscreteEnv(K, reward_scale=float(resolved["reward_scale"]))
        for scaling in map(str, resolved["heat_scalings"]):
            diffusion_time = float(resolved["diffusion_time"][scaling])
            for temperature in map(float, resolved["temperature"]):
                case_radius = int(resolved["primary_radius"])
                if scaling == "physical_heat":
                    reference_K = int(resolved["physical_radius_reference_K"])
                    case_radius = max(
                        1,
                        int(
                            math.ceil(
                                case_radius * (K - 1) / max(1, reference_K - 1)
                            )
                        ),
                    )
                random_subset_size = int(resolved["random_subset_size"])
                if random_subset_size <= 0:
                    random_subset_size = min(K, 2 * case_radius + 1)
                completed = existing.loc[
                    (existing.get("K", pd.Series(dtype=float)) == K)
                    & (existing.get("heat_scaling", pd.Series(dtype=object)) == scaling)
                    & np.isclose(
                        existing.get("temperature", pd.Series(dtype=float)).astype(float),
                        temperature,
                    )
                ]
                expected_methods = set(map(str, resolved["methods"]))
                ablation = resolved["radius_ablation"]
                expects_ablation = bool(ablation["enabled"]) and (
                    K == int(ablation["K"])
                    and scaling == str(ablation["heat_scaling"])
                    and np.isclose(temperature, float(ablation["temperature"]))
                )
                completed_methods = set(completed.get("method", pd.Series(dtype=str)))
                completed_radii = set(
                    completed.loc[
                        completed.get("method", pd.Series(dtype=str))
                        == "truncated_heat_local_radius",
                        "radius",
                    ].dropna().astype(int)
                ) if not completed.empty else set()
                if expected_methods.issubset(completed_methods) and (
                    not expects_ablation
                    or set(map(int, resolved["radii"])).issubset(completed_radii)
                ):
                    LOGGER.info(
                        "Skipping completed Pendulum case K=%s scaling=%s T0=%s",
                        K,
                        scaling,
                        temperature,
                    )
                    continue
                if not completed.empty:
                    completed_ids = set(completed["run_id"].astype(str))
                    rows = [
                        row for row in rows if str(row.get("run_id")) not in completed_ids
                    ]
                    existing = existing.drop(index=completed.index)
                operators = _build_operators(
                    environment,
                    heat_scaling=scaling,
                    diffusion_time=diffusion_time,
                    radius=case_radius,
                    alpha=float(resolved["topmass_alpha"]),
                    endpoint_samples=int(resolved["endpoint_samples"]),
                    random_subset_size=random_subset_size,
                    rbf_sigma=float(resolved["rbf_sigma"]),
                    squared_torque_lambda=float(resolved["squared_torque_lambda"]),
                    seed=int(resolved["seed"]),
                    methods=resolved["methods"],
                )
                solutions: dict[str, FittedSolution] = {}
                for method in resolved["methods"]:
                    operator = operators[method]
                    solution = _solve_operator(
                        environment,
                        grid,
                        operator,
                        evaluation_states,
                        gamma=float(resolved["gamma"]),
                        T0=temperature,
                        tolerance=float(planning["tolerance"]),
                        max_iterations=int(planning["max_iterations"]),
                        chunk_size=int(planning["state_chunk_size"]),
                        angle_gain=angle_gain,
                        velocity_gain=velocity_gain,
                        progress_interval=int(planning["progress_interval"]),
                        anderson_depth=int(planning["anderson_depth"]),
                    )
                    solutions[method] = solution
                reference = solutions["full_exact_heat"]
                for method in resolved["methods"]:
                    operator = operators[method]
                    solution = solutions[method]
                    value_errors = np.abs(solution.evaluation_values - reference.evaluation_values)
                    policy_errors = np.sum(
                        np.abs(solution.evaluation_policies - reference.evaluation_policies), axis=1
                    )
                    behavior = {
                        "raw_return_mean": np.nan,
                        "mean_squared_angle_error": np.nan,
                        "upright_fraction": np.nan,
                        "mean_absolute_torque": np.nan,
                    }
                    behavior_config = resolved["behavior"]
                    behavior_case = (
                        K == int(behavior_config["K"])
                        and scaling == str(behavior_config["heat_scaling"])
                        and np.isclose(temperature, float(behavior_config["temperature"]))
                    )
                    if bool(behavior_config["enabled"]) and behavior_case:
                        behavior_states = reset_states[: int(resolved["behavior"]["episodes"])]
                        behavior = _behavior_metrics(
                            solution,
                            operator,
                            environment,
                            grid,
                            behavior_states,
                            gamma=float(resolved["gamma"]),
                            T0=temperature,
                            angle_gain=angle_gain,
                            velocity_gain=velocity_gain,
                            horizon=int(behavior_config["horizon"]),
                            seed=int(resolved["seed"]),
                        )
                    actions_per_backup = solution.counters.action_evaluations / max(
                        1, solution.counters.backups
                    )
                    row = result_row(
                        experiment="pendulum",
                        run_id=_run_id((K, scaling, temperature, method), resolved_json),
                        seed=int(resolved["seed"]),
                        method=method,
                        target=operator.target,
                        graph_family="path",
                        K=K,
                        radius=(case_radius if "local" in method else np.nan),
                        diffusion_time=diffusion_time,
                        temperature=temperature,
                        gamma=float(resolved["gamma"]),
                        value_estimate=float(np.mean(solution.evaluation_values)),
                        reference_value=float(np.mean(reference.evaluation_values)),
                        absolute_value_error=float(np.mean(value_errors)),
                        policy_l1_error=float(np.mean(policy_errors)),
                        transition_calls=solution.counters.transition_calls,
                        action_evaluations=solution.counters.action_evaluations,
                        unique_actions_touched=K,
                        graph_neighbor_accesses=solution.counters.graph_neighbor_accesses,
                        geometry_preprocess_seconds=operator.geometry_preprocess_seconds,
                        online_seconds=solution.online_seconds,
                        peak_memory_mb=solution.peak_memory_mb,
                        status="complete" if solution.converged else "not_converged",
                        git_commit=metadata["git_commit"],
                        config_json=resolved_json,
                    )
                    row.update(
                        {
                            "reference_target": "exact_heat",
                            "heat_scaling": scaling,
                            "mean_value_error": float(np.mean(value_errors)),
                            "max_value_error": float(np.max(value_errors)),
                            "mean_policy_l1_error": float(np.mean(policy_errors)),
                            "max_policy_l1_error": float(np.max(policy_errors)),
                            "bellman_residual_mean": solution.residual_mean,
                            "bellman_residual_max": solution.residual,
                            "bellman_iterations": solution.iterations,
                            "converged": solution.converged,
                            "actions_per_backup": actions_per_backup,
                            "action_fraction": actions_per_backup / K,
                            "time_per_sweep": solution.online_seconds / max(1, solution.iterations),
                            "regularized_value_mean": float(
                                np.mean(solution.evaluation_values[len(fixed) :])
                            ),
                            **behavior,
                        }
                    )
                    rows.append(row)
                if (
                    bool(ablation["enabled"])
                    and K == int(ablation["K"])
                    and scaling == str(ablation["heat_scaling"])
                    and np.isclose(temperature, float(ablation["temperature"]))
                ):
                    for radius in map(int, resolved["radii"]):
                        if radius == case_radius:
                            local_solution = solutions.get("truncated_heat_local")
                            local_operator = operators.get("truncated_heat_local")
                        else:
                            ablation_operators = _build_operators(
                                environment,
                                heat_scaling=scaling,
                                diffusion_time=diffusion_time,
                                radius=radius,
                                alpha=float(resolved["topmass_alpha"]),
                                endpoint_samples=int(resolved["endpoint_samples"]),
                                random_subset_size=random_subset_size,
                                rbf_sigma=float(resolved["rbf_sigma"]),
                                squared_torque_lambda=float(resolved["squared_torque_lambda"]),
                                seed=int(resolved["seed"]),
                                methods=["full_exact_heat", "truncated_heat_local"],
                            )
                            local_operator = ablation_operators["truncated_heat_local"]
                            local_solution = _solve_operator(
                                environment,
                                grid,
                                local_operator,
                                evaluation_states,
                                gamma=float(resolved["gamma"]),
                                T0=temperature,
                                tolerance=float(planning["tolerance"]),
                                max_iterations=int(planning["max_iterations"]),
                                chunk_size=int(planning["state_chunk_size"]),
                                angle_gain=angle_gain,
                                velocity_gain=velocity_gain,
                                progress_interval=int(planning["progress_interval"]),
                                anderson_depth=int(planning["anderson_depth"]),
                            )
                        if local_solution is None or local_operator is None:
                            continue
                        value_errors = np.abs(
                            local_solution.evaluation_values - reference.evaluation_values
                        )
                        policy_errors = np.sum(
                            np.abs(
                                local_solution.evaluation_policies
                                - reference.evaluation_policies
                            ),
                            axis=1,
                        )
                        actions_per_backup = (
                            local_solution.counters.action_evaluations
                            / max(1, local_solution.counters.backups)
                        )
                        row = result_row(
                            experiment="pendulum",
                            run_id=_run_id(
                                (K, scaling, temperature, "radius_ablation", radius),
                                resolved_json,
                            ),
                            seed=int(resolved["seed"]),
                            method="truncated_heat_local_radius",
                            target="truncated_heat",
                            graph_family="path",
                            K=K,
                            radius=radius,
                            diffusion_time=diffusion_time,
                            temperature=temperature,
                            gamma=float(resolved["gamma"]),
                            value_estimate=float(np.mean(local_solution.evaluation_values)),
                            reference_value=float(np.mean(reference.evaluation_values)),
                            absolute_value_error=float(np.mean(value_errors)),
                            policy_l1_error=float(np.mean(policy_errors)),
                            transition_calls=local_solution.counters.transition_calls,
                            action_evaluations=local_solution.counters.action_evaluations,
                            unique_actions_touched=K,
                            graph_neighbor_accesses=local_solution.counters.graph_neighbor_accesses,
                            geometry_preprocess_seconds=local_operator.geometry_preprocess_seconds,
                            online_seconds=local_solution.online_seconds,
                            peak_memory_mb=local_solution.peak_memory_mb,
                            status="complete" if local_solution.converged else "not_converged",
                            git_commit=metadata["git_commit"],
                            config_json=resolved_json,
                        )
                        row.update(
                            {
                                "reference_target": "exact_heat",
                                "heat_scaling": scaling,
                                "mean_value_error": float(np.mean(value_errors)),
                                "max_value_error": float(np.max(value_errors)),
                                "mean_policy_l1_error": float(np.mean(policy_errors)),
                                "max_policy_l1_error": float(np.max(policy_errors)),
                                "bellman_residual_mean": local_solution.residual_mean,
                                "bellman_residual_max": local_solution.residual,
                                "bellman_iterations": local_solution.iterations,
                                "converged": local_solution.converged,
                                "actions_per_backup": actions_per_backup,
                                "action_fraction": actions_per_backup / K,
                                "time_per_sweep": local_solution.online_seconds
                                / max(1, local_solution.iterations),
                                "regularized_value_mean": float(
                                    np.mean(local_solution.evaluation_values[len(fixed) :])
                                ),
                                "raw_return_mean": np.nan,
                                "mean_squared_angle_error": np.nan,
                                "upright_fraction": np.nan,
                                "mean_absolute_torque": np.nan,
                            }
                        )
                        rows.append(row)
                write_results_atomic(rows, resolved["raw_output"])
    frame = pd.DataFrame(rows)
    write_results_atomic(frame, resolved["raw_output"])
    write_results_atomic(frame, resolved["summary_output"])
    return frame


@dataclass(frozen=True)
class PendulumCase:
    K: int
    heat_scaling: str
    temperature: float


def _pendulum_cases(resolved: Mapping[str, Any]) -> list[PendulumCase]:
    return [
        PendulumCase(int(K), str(scaling), float(temperature))
        for K in resolved["K"]
        for scaling in resolved["heat_scalings"]
        for temperature in resolved["temperature"]
    ]


def _case_has_ablation(case: PendulumCase, resolved: Mapping[str, Any]) -> bool:
    ablation = resolved["radius_ablation"]
    return bool(ablation["enabled"]) and (
        case.K == int(ablation["K"])
        and case.heat_scaling == str(ablation["heat_scaling"])
        and np.isclose(case.temperature, float(ablation["temperature"]))
    )


def _case_run_ids(
    case: PendulumCase, resolved: Mapping[str, Any], resolved_json: str
) -> set[str]:
    identifiers = {
        _run_id(
            (case.K, case.heat_scaling, case.temperature, str(method)),
            resolved_json,
        )
        for method in resolved["methods"]
    }
    if _case_has_ablation(case, resolved):
        identifiers.update(
            _run_id(
                (
                    case.K,
                    case.heat_scaling,
                    case.temperature,
                    "radius_ablation",
                    int(radius),
                ),
                resolved_json,
            )
            for radius in resolved["radii"]
        )
    return identifiers


def _row_matches_case(row: Mapping[str, Any], case: PendulumCase) -> bool:
    try:
        return (
            int(row["K"]) == case.K
            and str(row["heat_scaling"]) == case.heat_scaling
            and np.isclose(float(row["temperature"]), case.temperature)
        )
    except (KeyError, TypeError, ValueError):
        return False


def _merge_rows(
    rows: list[dict[str, Any]], updates: Sequence[Mapping[str, Any]]
) -> None:
    indexed = {str(row["run_id"]): dict(row) for row in rows}
    indexed.update({str(row["run_id"]): dict(row) for row in updates})
    rows[:] = [indexed[run_id] for run_id in sorted(indexed)]


def _execute_pendulum_case(
    resolved: Mapping[str, Any],
    resolved_json: str,
    metadata: Mapping[str, Any],
    case: PendulumCase,
) -> tuple[dict[str, Any], ...]:
    """Run one independent M8 case without writing shared artifacts."""

    case_config = copy.deepcopy(dict(resolved))
    case_config.update(
        {
            "K": [case.K],
            "heat_scalings": [case.heat_scaling],
            "temperature": [case.temperature],
            "execution": {"workers": 1},
        }
    )
    with tempfile.TemporaryDirectory(prefix="lot-pendulum-case-") as directory:
        temporary = Path(directory)
        case_config.update(
            {
                "raw_output": str(temporary / "raw.parquet"),
                "summary_output": str(temporary / "summary.csv"),
            }
        )
        frame = _run_pendulum_experiment_serial(case_config)

    execution_workers = int(resolved["execution"]["workers"])
    timing_mode = "isolated" if execution_workers == 1 else "concurrent"
    records = frame.to_dict(orient="records")
    for row in records:
        method = str(row["method"])
        if method == "truncated_heat_local_radius":
            run_parts = (
                case.K,
                case.heat_scaling,
                case.temperature,
                "radius_ablation",
                int(row["radius"]),
            )
        else:
            run_parts = (
                case.K,
                case.heat_scaling,
                case.temperature,
                method,
            )
        row.update(
            {
                "run_id": _run_id(run_parts, resolved_json),
                "git_commit": str(metadata["git_commit"]),
                "config_json": resolved_json,
                "execution_workers": execution_workers,
                "timing_mode": timing_mode,
                "error_message": "",
            }
        )
    return tuple(records)


def _case_failure_row(
    case: PendulumCase,
    error: Exception,
    resolved: Mapping[str, Any],
    resolved_json: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    workers = int(resolved["execution"]["workers"])
    row = result_row(
        experiment="pendulum",
        run_id=_run_id(
            (
                case.K,
                case.heat_scaling,
                case.temperature,
                "experiment_failure",
            ),
            resolved_json,
        ),
        seed=int(resolved["seed"]),
        method="experiment_failure",
        target="exact_heat",
        graph_family="path",
        K=case.K,
        diffusion_time=float(resolved["diffusion_time"][case.heat_scaling]),
        temperature=case.temperature,
        gamma=float(resolved["gamma"]),
        status="failed",
        git_commit=str(metadata["git_commit"]),
        config_json=resolved_json,
    )
    row.update(
        {
            "reference_target": "exact_heat",
            "heat_scaling": case.heat_scaling,
            "execution_workers": workers,
            "timing_mode": "isolated" if workers == 1 else "concurrent",
            "error_message": f"{type(error).__name__}: {error}",
        }
    )
    return row


def run_pendulum_experiment(config: Mapping[str, Any]) -> pd.DataFrame:
    """Run or resume M8, optionally distributing independent cases."""

    resolved = resolve_pendulum_experiment_config(config)
    resolved_json = config_json(resolved)
    metadata = run_metadata(resolved)
    output = Path(resolved["raw_output"])
    if output.exists():
        existing = validate_results(
            pd.read_parquet(output)
            if output.suffix == ".parquet"
            else pd.read_csv(output)
        )
        configurations = set(existing["config_json"].astype(str))
        if configurations and configurations != {resolved_json}:
            raise ValueError(
                "raw_output contains a different resolved configuration; "
                "choose a distinct output path instead of overwriting it"
            )
        rows = existing.to_dict(orient="records")
    else:
        rows = []

    completed = {
        str(row["run_id"]) for row in rows if str(row["status"]) == "complete"
    }
    cases = _pendulum_cases(resolved)
    pending: list[PendulumCase] = []
    for case in cases:
        expected = _case_run_ids(case, resolved, resolved_json)
        if expected <= completed:
            continue
        pending.append(case)
        # A case is the checkpoint unit. Drop incomplete/stale rows in memory;
        # the on-disk checkpoint is replaced only after the rerun finishes.
        rows = [row for row in rows if not _row_matches_case(row, case)]

    workers = int(resolved["execution"]["workers"])
    if workers > 1:
        # Schedule expensive action grids first to reduce the straggler tail.
        pending.sort(
            key=lambda item: (
                item.K,
                item.heat_scaling == "physical_heat",
                -item.temperature,
            ),
            reverse=True,
        )
    effective_workers = min(workers, max(1, len(pending)))
    LOGGER.info(
        "Pendulum M8 start workers=%d effective_workers=%d total_cases=%d "
        "resumed_cases=%d pending_cases=%d checkpoint_rows=%d",
        workers,
        effective_workers,
        len(cases),
        len(cases) - len(pending),
        len(pending),
        len(rows),
    )

    failures: list[tuple[PendulumCase, str]] = []

    def commit() -> pd.DataFrame:
        frame = validate_results(pd.DataFrame(rows))
        write_results_atomic(frame, resolved["raw_output"])
        write_results_atomic(frame, resolved["summary_output"])
        return frame

    def accept(case: PendulumCase, case_rows: Sequence[Mapping[str, Any]]) -> None:
        _merge_rows(rows, case_rows)
        commit()
        incomplete = [
            row for row in case_rows if str(row.get("status")) != "complete"
        ]
        if incomplete:
            failures.append(
                (
                    case,
                    ", ".join(
                        f"{row.get('method')}={row.get('status')}" for row in incomplete
                    ),
                )
            )
        LOGGER.info(
            "Pendulum checkpoint K=%d scaling=%s T0=%g rows=%d",
            case.K,
            case.heat_scaling,
            case.temperature,
            len(rows),
        )

    if workers == 1:
        for case in pending:
            try:
                accept(
                    case,
                    _execute_pendulum_case(
                        resolved, resolved_json, metadata, case
                    ),
                )
            except Exception as error:
                accept(
                    case,
                    (
                        _case_failure_row(
                            case, error, resolved, resolved_json, metadata
                        ),
                    ),
                )
    elif pending:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=effective_workers, mp_context=context
        ) as executor:
            future_cases = {
                executor.submit(
                    _execute_pendulum_case,
                    resolved,
                    resolved_json,
                    metadata,
                    case,
                ): case
                for case in pending
            }
            outstanding = set(future_cases)
            accepted = 0
            while outstanding:
                finished, outstanding = wait(
                    outstanding,
                    timeout=30.0,
                    return_when=FIRST_COMPLETED,
                )
                if not finished:
                    LOGGER.info(
                        "Pendulum M8 heartbeat completed_cases=%d/%d "
                        "outstanding_cases=%d checkpoint_rows=%d",
                        accepted,
                        len(future_cases),
                        len(outstanding),
                        len(rows),
                    )
                    continue
                for future in finished:
                    case = future_cases[future]
                    try:
                        accept(case, future.result())
                    except Exception as error:
                        accept(
                            case,
                            (
                                _case_failure_row(
                                    case, error, resolved, resolved_json, metadata
                                ),
                            ),
                        )
                    accepted += 1

    frame = commit()
    if failures:
        examples = "; ".join(
            f"K={case.K}/{case.heat_scaling}/T0={case.temperature:g}: {message}"
            for case, message in failures[:3]
        )
        raise RuntimeError(
            f"{len(failures)} Pendulum case(s) failed; raw failures were "
            f"checkpointed. First failures: {examples}"
        )
    return frame
