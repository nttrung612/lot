"""Converged full-exact-heat fitted-value reference for ``Pendulum-v1``."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import psutil
from numpy.typing import ArrayLike, NDArray
from scipy.fft import dct, idct
from scipy.stats import poisson

from lot_experiments.config import config_json
from lot_experiments.environments.pendulum_discrete import (
    PendulumDiscreteEnv,
    PendulumStateGrid,
)
from lot_experiments.reproducibility import rng_for, run_metadata
from lot_experiments.results import result_row, write_results_atomic


FloatArray = NDArray[np.float64]
LOGGER = logging.getLogger(__name__)

EVALUATION_STATES = (
    (math.pi, 0.0),
    (math.pi / 2.0, 0.0),
    (-math.pi / 2.0, 0.0),
    (0.0, 0.0),
    (math.pi, 2.0),
)

@dataclass(frozen=True)
class PathHeatOperator:
    """Exact path heat semigroup with stable dense and spectral backends."""

    K: int
    diffusion_time: float
    edge_weight: float = 1.0
    dense_threshold: int = 256

    def __post_init__(self) -> None:
        if self.K < 2:
            raise ValueError("path heat requires K >= 2")
        if self.diffusion_time < 0.0 or not math.isfinite(self.diffusion_time):
            raise ValueError("diffusion_time must be finite and nonnegative")
        if self.edge_weight <= 0.0 or not math.isfinite(self.edge_weight):
            raise ValueError("edge_weight must be finite and positive")
        if self.dense_threshold < 0:
            raise ValueError("dense_threshold must be nonnegative")

    @property
    def multipliers(self) -> FloatArray:
        modes = np.arange(self.K, dtype=np.float64)
        eigenvalues = self.edge_weight * (
            2.0 - 2.0 * np.cos(np.pi * modes / self.K)
        )
        return np.exp(-self.diffusion_time * eigenvalues)

    @cached_property
    def dense_kernel(self) -> FloatArray | None:
        """Materialize a tail-stable numerical exact kernel for dense backups.

        Odd action counts such as 101 have comparatively expensive prime-size
        DCTs.  The dense backend is only a reference optimization; local M8
        methods must continue to construct columns lazily.
        """

        if self.K > self.dense_threshold:
            identity = np.eye(self.K, dtype=np.float64)
            coefficients = dct(identity, type=2, norm="ortho", axis=-1)
            kernel = idct(
                coefficients * self.multipliers,
                type=2,
                norm="ortho",
                axis=-1,
            )
            np.maximum(kernel, 0.0, out=kernel)
            kernel /= kernel.sum(axis=0, keepdims=True)
            return kernel
        theta = 2.0 * self.edge_weight * self.diffusion_time
        if theta == 0.0:
            return np.eye(self.K, dtype=np.float64)
        # A nonnegative uniformization sum preserves the very small path-heat
        # tails needed inside log partitions. Spectral reconstruction loses
        # those tails to cancellation, even though it is accurate in absolute
        # matrix norm, and can therefore corrupt a FullExactHeat LOT backup.
        poisson_quantile = int(poisson.ppf(1.0 - 1e-15, theta))
        maximum_walks = max(
            poisson_quantile,
            self.K - 1 + int(math.ceil(theta + 12.0 * math.sqrt(theta + self.K))),
        )
        power = np.eye(self.K, dtype=np.float64)
        coefficient = math.exp(-theta)
        kernel = coefficient * power
        for walk_count in range(1, maximum_walks + 1):
            next_power = np.zeros_like(power)
            next_power[:, 1:] += 0.5 * power[:, :-1]
            next_power[:, :-1] += 0.5 * power[:, 1:]
            next_power[:, 0] += 0.5 * power[:, 0]
            next_power[:, -1] += 0.5 * power[:, -1]
            power = next_power
            coefficient *= theta / walk_count
            kernel += coefficient * power
        if np.any(kernel <= 0.0):
            raise FloatingPointError(
                "uniformized path heat underflowed; use a log-domain backend"
            )
        kernel /= kernel.sum(axis=0, keepdims=True)
        return kernel

    def apply(self, values: ArrayLike) -> FloatArray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape[-1] != self.K or not np.all(np.isfinite(array)):
            raise ValueError(f"values must be finite with final dimension {self.K}")
        if self.dense_kernel is not None:
            return array @ self.dense_kernel
        coefficients = dct(array, type=2, norm="ortho", axis=-1)
        coefficients *= self.multipliers
        result = idct(coefficients, type=2, norm="ortho", axis=-1)
        # DCT roundoff can create tiny negative values when applying heat to a
        # nonnegative vector.  Callers use the operator on such vectors only.
        return result


def full_exact_heat_backup(
    q_values: ArrayLike,
    heat: PathHeatOperator,
    T0: float,
    *,
    return_policy: bool = True,
    anchor_indices: ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray | None]:
    """Evaluate the uniform-anchor FullExactHeat backup for many states."""

    q = np.asarray(q_values, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != heat.K or not np.all(np.isfinite(q)):
        raise ValueError("q_values must be a finite (states, K) matrix")
    if T0 <= 0.0 or not math.isfinite(T0):
        raise ValueError("T0 must be finite and positive")
    maxima = np.max(q, axis=1)
    exponentials = np.exp((q - maxima[:, None]) / T0)
    anchors = None if anchor_indices is None else np.asarray(anchor_indices, dtype=np.int64)
    if anchors is not None:
        if anchors.shape != (len(q),) or np.any(anchors < 0) or np.any(anchors >= heat.K):
            raise ValueError("anchor_indices must contain one valid anchor per state")
        kernel = heat.dense_kernel
        assert kernel is not None
        selected_columns = kernel[:, anchors].T
        partitions = np.sum(exponentials * selected_columns, axis=1)
    else:
        partitions = heat.apply(exponentials)
    tiny = np.finfo(np.float64).tiny
    if np.min(partitions) < -1e-12 or not np.all(np.isfinite(partitions)):
        raise FloatingPointError("path heat produced invalid partition values")
    np.maximum(partitions, tiny, out=partitions)
    values = (
        maxima + T0 * np.log(partitions)
        if anchors is not None
        else maxima + T0 * np.mean(np.log(partitions), axis=1)
    )
    if not return_policy:
        return values, None

    if anchors is not None:
        policy = exponentials * selected_columns / partitions[:, None]
        policy /= policy.sum(axis=1, keepdims=True)
        return values, policy
    inverse_partitions = (1.0 / heat.K) / partitions
    mixture = heat.apply(inverse_partitions)
    if np.min(mixture) < -1e-12 or not np.all(np.isfinite(mixture)):
        raise FloatingPointError("path heat produced an invalid policy mixture")
    np.maximum(mixture, 0.0, out=mixture)
    policy = exponentials * mixture
    policy /= policy.sum(axis=1, keepdims=True)
    return values, policy


@dataclass(frozen=True)
class PendulumReferenceSolution:
    grid: PendulumStateGrid
    values: FloatArray
    evaluation_values: FloatArray
    evaluation_policies: FloatArray
    iterations: int
    bellman_residual: float
    converged: bool
    geometry_preprocess_seconds: float
    online_seconds: float
    peak_memory_mb: float
    action_evaluations: int


def _bellman_values(
    environment: PendulumDiscreteEnv,
    grid: PendulumStateGrid,
    values: FloatArray,
    *,
    gamma: float,
    T0: float,
    heat: PathHeatOperator,
    state_chunk_size: int,
    anchor_angle_gain: float,
    anchor_velocity_gain: float,
) -> FloatArray:
    theta, theta_dot = grid.states()
    updated = np.empty(grid.size, dtype=np.float64)
    for start in range(0, grid.size, state_chunk_size):
        stop = min(start + state_chunk_size, grid.size)
        q_values = environment.q_values(
            values, grid, theta[start:stop], theta_dot[start:stop], gamma
        )
        anchors = environment.nominal_action_indices(
            theta[start:stop],
            theta_dot[start:stop],
            angle_gain=anchor_angle_gain,
            velocity_gain=anchor_velocity_gain,
        )
        updated[start:stop], _ = full_exact_heat_backup(
            q_values, heat, T0, return_policy=False, anchor_indices=anchors
        )
    return updated.reshape(grid.shape)


def _initial_values(
    initial_values: ArrayLike | None,
    initial_grid: PendulumStateGrid | None,
    grid: PendulumStateGrid,
) -> FloatArray:
    if initial_values is None:
        return np.zeros(grid.shape, dtype=np.float64)
    values = np.asarray(initial_values, dtype=np.float64)
    if initial_grid is None:
        if values.shape != grid.shape:
            raise ValueError("initial_values shape does not match target grid")
        return values.copy()
    theta, theta_dot = grid.states()
    return initial_grid.interpolate(values, theta, theta_dot).reshape(grid.shape)


def solve_pendulum_reference(
    environment: PendulumDiscreteEnv,
    grid: PendulumStateGrid,
    *,
    gamma: float,
    T0: float,
    diffusion_time: float,
    heat_scaling: str,
    tolerance: float,
    max_iterations: int,
    state_chunk_size: int,
    anderson_depth: int = 5,
    progress_interval: int = 25,
    anchor_angle_gain: float = 2.0,
    anchor_velocity_gain: float = 0.5,
    evaluation_states: Sequence[tuple[float, float]] = EVALUATION_STATES,
    initial_values: ArrayLike | None = None,
    initial_grid: PendulumStateGrid | None = None,
) -> PendulumReferenceSolution:
    """Solve the discounted fitted Bellman equation on one periodic grid.

    Iterates are centered at one grid cell.  Translation equivariance removes
    the slow constant mode induced by ``gamma=0.99``; the final scalar shift is
    restored analytically, and convergence is still checked with the ordinary
    (uncentered) Bellman residual.
    """

    if heat_scaling not in {"index_heat", "physical_heat"}:
        raise ValueError("heat_scaling must be index_heat or physical_heat")
    if tolerance <= 0.0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and positive")
    if (
        max_iterations < 1
        or state_chunk_size < 1
        or anderson_depth < 0
        or progress_interval < 0
    ):
        raise ValueError("max_iterations and state_chunk_size must be positive")
    edge_weight = (
        1.0 if heat_scaling == "index_heat" else 1.0 / environment.action_spacing**2
    )
    preprocess_start = perf_counter()
    heat = PathHeatOperator(environment.K, diffusion_time, edge_weight)
    # Force any dense reference construction outside the online Bellman timer.
    _ = heat.dense_kernel
    geometry_preprocess_seconds = perf_counter() - preprocess_start
    values = _initial_values(initial_values, initial_grid, grid)
    values -= values[grid.angle_count // 2, grid.velocity_count // 2]
    start_time = perf_counter()
    residual = math.inf
    converged = False
    reference_index = (grid.angle_count // 2, grid.velocity_count // 2)
    sweeps = 0
    previous_values: FloatArray | None = None
    previous_fixed_point_residual: FloatArray | None = None
    value_differences: list[FloatArray] = []
    residual_differences: list[FloatArray] = []
    for iteration in range(1, max_iterations + 1):
        updated = _bellman_values(
            environment,
            grid,
            values,
            gamma=gamma,
            T0=T0,
            heat=heat,
            state_chunk_size=state_chunk_size,
            anchor_angle_gain=anchor_angle_gain,
            anchor_velocity_gain=anchor_velocity_gain,
        )
        differences = updated - values
        reference_difference = float(differences[reference_index])
        centered_update = updated - updated[reference_index]
        fixed_point_residual = centered_update - values
        centered_residual = float(np.max(np.abs(fixed_point_residual)))
        sweeps += 1
        if progress_interval and (
            iteration == 1 or iteration % progress_interval == 0
        ):
            LOGGER.info(
                "Pendulum reference grid=%sx%s iteration=%s centered_residual=%.3e",
                grid.angle_count,
                grid.velocity_count,
                iteration,
                centered_residual,
            )
        if centered_residual <= tolerance:
            fixed_values = values + reference_difference / (1.0 - gamma)
            checked = _bellman_values(
                environment,
                grid,
                fixed_values,
                gamma=gamma,
                T0=T0,
                heat=heat,
                state_chunk_size=state_chunk_size,
                anchor_angle_gain=anchor_angle_gain,
                anchor_velocity_gain=anchor_velocity_gain,
            )
            sweeps += 1
            residual = float(np.max(np.abs(checked - fixed_values)))
            converged = residual <= max(tolerance * 1.05, 32.0 * np.finfo(float).eps)
            if converged:
                values = fixed_values
                break
        if previous_values is not None and previous_fixed_point_residual is not None:
            value_differences.append((values - previous_values).ravel())
            residual_differences.append(
                (fixed_point_residual - previous_fixed_point_residual).ravel()
            )
            if len(value_differences) > anderson_depth:
                value_differences.pop(0)
                residual_differences.pop(0)
        next_values = centered_update
        if anderson_depth and residual_differences:
            delta_values = np.column_stack(value_differences)
            delta_residuals = np.column_stack(residual_differences)
            gram = delta_residuals.T @ delta_residuals
            right_hand_side = delta_residuals.T @ fixed_point_residual.ravel()
            ridge = 1e-12 * max(1.0, float(np.trace(gram)))
            coefficients = np.linalg.solve(
                gram + ridge * np.eye(gram.shape[0]), right_hand_side
            )
            accelerated = centered_update.ravel() - (
                delta_values + delta_residuals
            ) @ coefficients
            accelerated = accelerated.reshape(grid.shape)
            accelerated -= accelerated[reference_index]
            # A conservative safeguard keeps the globally convergent centered
            # Bellman step whenever the small least-squares problem is ill-conditioned.
            if (
                np.all(np.isfinite(accelerated))
                and np.max(np.abs(accelerated))
                <= 10.0 * max(1.0, float(np.max(np.abs(centered_update))))
            ):
                next_values = accelerated
        previous_values = values.copy()
        previous_fixed_point_residual = fixed_point_residual.copy()
        values = next_values
    elapsed = perf_counter() - start_time

    evaluation = np.asarray(evaluation_states, dtype=np.float64)
    if evaluation.ndim != 2 or evaluation.shape[1] != 2:
        raise ValueError("evaluation_states must be a sequence of (theta, theta_dot)")
    q_values = environment.q_values(
        values, grid, evaluation[:, 0], evaluation[:, 1], gamma
    )
    evaluation_anchors = environment.nominal_action_indices(
        evaluation[:, 0],
        evaluation[:, 1],
        angle_gain=anchor_angle_gain,
        velocity_gain=anchor_velocity_gain,
    )
    evaluation_values, policies = full_exact_heat_backup(
        q_values, heat, T0, anchor_indices=evaluation_anchors
    )
    assert policies is not None
    action_evaluations = sweeps * grid.size * environment.K + len(evaluation) * environment.K
    return PendulumReferenceSolution(
        grid=grid,
        values=values,
        evaluation_values=evaluation_values,
        evaluation_policies=policies,
        iterations=iteration,
        bellman_residual=residual,
        converged=converged,
        geometry_preprocess_seconds=geometry_preprocess_seconds,
        online_seconds=elapsed,
        peak_memory_mb=psutil.Process().memory_info().rss / (1024.0**2),
        action_evaluations=action_evaluations,
    )


def resolve_pendulum_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve and validate the M7 reference configuration."""

    defaults: dict[str, Any] = {
        "experiment": "pendulum",
        "seed": 0,
        "environment": "Pendulum-v1",
        "K": [51, 101, 201, 401, 801],
        "heat_scalings": ["index_heat", "physical_heat"],
        "diffusion_time": {"index_heat": 2.0, "physical_heat": 0.02},
        "gamma": 0.99,
        "reward_scale": 16.2736044,
        "temperature": [0.02, 0.05, 0.1],
        "development_grid": [129, 129],
        "validation_grid": [257, 257],
        "reference_tolerance": 1e-3,
        "reference_policy_statistic_tolerance": 1e-2,
        "evaluation_reset_states": 100,
        "reference": {
            "K": 101,
            "heat_scaling": "physical_heat",
            "temperature": 0.05,
            "bellman_tolerance": 1e-7,
            "max_iterations": 5000,
            "state_chunk_size": 256,
            "anderson_depth": 5,
            "progress_interval": 25,
            "anchor_prior": "nominal_pd",
            "anchor_angle_gain": 2.0,
            "anchor_velocity_gain": 0.5,
            "require_grid_convergence": False,
        },
        "raw_output": "outputs/raw/pendulum_reference.parquet",
        "solution_output": "outputs/raw/pendulum_reference_solution.npz",
    }
    resolved = copy.deepcopy(defaults)
    for key, value in config.items():
        if key == "reference" and isinstance(value, Mapping):
            resolved["reference"].update(value)
        elif key == "diffusion_time" and isinstance(value, Mapping):
            resolved["diffusion_time"].update(value)
        else:
            resolved[key] = copy.deepcopy(value)
    if resolved["environment"] != "Pendulum-v1":
        raise ValueError("M7 primary environment must be Pendulum-v1")
    gamma = float(resolved["gamma"])
    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must lie in [0, 1)")
    if float(resolved["reward_scale"]) <= 0.0:
        raise ValueError("reward_scale must be positive")
    for name in ("development_grid", "validation_grid"):
        dimensions = resolved[name]
        if not isinstance(dimensions, Sequence) or len(dimensions) != 2:
            raise ValueError(f"{name} must contain angle and velocity counts")
        if any(int(value) < 2 for value in dimensions):
            raise ValueError(f"{name} dimensions are too small")
    development = tuple(map(int, resolved["development_grid"]))
    validation = tuple(map(int, resolved["validation_grid"]))
    if validation[0] < development[0] or validation[1] < development[1]:
        raise ValueError("validation_grid must not be coarser than development_grid")
    reference = resolved["reference"]
    K = int(reference["K"])
    if K not in {int(value) for value in resolved["K"]}:
        raise ValueError("reference.K must be included in the declared action grids")
    scaling = str(reference["heat_scaling"])
    if scaling not in resolved["heat_scalings"] or scaling not in resolved["diffusion_time"]:
        raise ValueError("reference heat scaling is not declared")
    if reference["anchor_prior"] != "nominal_pd":
        raise ValueError("Pendulum experiments require the explicit nominal_pd anchor prior")
    if float(reference["temperature"]) <= 0.0:
        raise ValueError("reference.temperature must be positive")
    if float(resolved["diffusion_time"][scaling]) < 0.0:
        raise ValueError("diffusion time must be nonnegative")
    if float(resolved["reference_tolerance"]) <= 0.0:
        raise ValueError("reference_tolerance must be positive")
    if float(resolved["reference_policy_statistic_tolerance"]) <= 0.0:
        raise ValueError("reference_policy_statistic_tolerance must be positive")
    if int(resolved["evaluation_reset_states"]) < 0:
        raise ValueError("evaluation_reset_states must be nonnegative")
    return resolved


def _stable_run_id(resolved_json: str) -> str:
    return hashlib.blake2b(resolved_json.encode("utf-8"), digest_size=10).hexdigest()


def _policy_statistics(policy: FloatArray, actions: FloatArray) -> tuple[FloatArray, ...]:
    mean = policy @ actions
    mean_absolute = policy @ np.abs(actions)
    entropy = -np.sum(policy * np.log(np.maximum(policy, np.finfo(float).tiny)), axis=1)
    return mean, mean_absolute, entropy


def _write_solution_atomic(
    destination: str | Path,
    development: PendulumReferenceSolution,
    validation: PendulumReferenceSolution,
    evaluation_states: FloatArray,
    evaluation_sets: NDArray[np.str_],
    resolved_json: str,
) -> Path:
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(
                stream,
                development_values=development.values,
                validation_values=validation.values,
                development_angles=development.grid.angles,
                development_angular_velocities=development.grid.angular_velocities,
                validation_angles=validation.grid.angles,
                validation_angular_velocities=validation.grid.angular_velocities,
                evaluation_values=validation.evaluation_values,
                evaluation_policies=validation.evaluation_policies,
                evaluation_states=evaluation_states,
                evaluation_sets=evaluation_sets,
                config_json=np.asarray(resolved_json),
            )
        os.replace(temporary_name, output_path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return output_path


def run_pendulum_reference(config: Mapping[str, Any]) -> pd.DataFrame:
    """Build development/validation references and persist reusable artifacts."""

    resolved = resolve_pendulum_config(config)
    resolved_json = config_json(resolved)
    reference = resolved["reference"]
    K = int(reference["K"])
    scaling = str(reference["heat_scaling"])
    diffusion_time = float(resolved["diffusion_time"][scaling])
    temperature = float(reference["temperature"])
    environment = PendulumDiscreteEnv(K=K, reward_scale=float(resolved["reward_scale"]))
    development_grid = PendulumStateGrid(*map(int, resolved["development_grid"]))
    validation_grid = PendulumStateGrid(*map(int, resolved["validation_grid"]))
    fixed_states = np.asarray(EVALUATION_STATES, dtype=np.float64)
    evaluation_rng = rng_for(int(resolved["seed"]), "pendulum_evaluation_states")
    reset_count = int(resolved["evaluation_reset_states"])
    reset_states = evaluation_rng.uniform(
        low=(-np.pi, -1.0), high=(np.pi, 1.0), size=(reset_count, 2)
    )
    evaluation_states = np.vstack((fixed_states, reset_states))
    evaluation_sets = np.asarray(
        ["fixed_diagnostic"] * len(fixed_states) + ["standard_reset"] * reset_count
    )
    solver_kwargs = {
        "gamma": float(resolved["gamma"]),
        "T0": temperature,
        "diffusion_time": diffusion_time,
        "heat_scaling": scaling,
        "tolerance": float(reference["bellman_tolerance"]),
        "max_iterations": int(reference["max_iterations"]),
        "state_chunk_size": int(reference["state_chunk_size"]),
        "anderson_depth": int(reference["anderson_depth"]),
        "progress_interval": int(reference["progress_interval"]),
        "evaluation_states": evaluation_states,
        "anchor_angle_gain": float(reference["anchor_angle_gain"]),
        "anchor_velocity_gain": float(reference["anchor_velocity_gain"]),
    }
    development = solve_pendulum_reference(environment, development_grid, **solver_kwargs)
    validation = solve_pendulum_reference(
        environment,
        validation_grid,
        initial_values=development.values,
        initial_grid=development.grid,
        **solver_kwargs,
    )
    value_errors = np.abs(
        development.evaluation_values - validation.evaluation_values
    )
    actions = environment.actions
    development_stats = _policy_statistics(development.evaluation_policies, actions)
    validation_stats = _policy_statistics(validation.evaluation_policies, actions)
    policy_statistic_errors = np.max(
        np.column_stack(
            [np.abs(first - second) for first, second in zip(development_stats, validation_stats)]
        ),
        axis=1,
    )
    accepted = bool(
        development.converged
        and validation.converged
        and float(value_errors.max(initial=0.0)) <= float(resolved["reference_tolerance"])
        and float(policy_statistic_errors.max(initial=0.0))
        <= float(resolved["reference_policy_statistic_tolerance"])
    )
    metadata = run_metadata(resolved)
    run_id = _stable_run_id(resolved_json)
    edge_weight = 1.0 if scaling == "index_heat" else 1.0 / environment.action_spacing**2
    rows: list[dict[str, Any]] = []
    for index, ((theta, theta_dot), evaluation_set, value_error) in enumerate(
        zip(evaluation_states, evaluation_sets, value_errors, strict=True)
    ):
        for stage, solution in (("development", development), ("validation", validation)):
            mean, mean_abs, entropy = _policy_statistics(
                solution.evaluation_policies[index : index + 1], actions
            )
            row = result_row(
                experiment="pendulum_reference",
                run_id=f"{run_id}-{stage}-{index}",
                seed=int(resolved["seed"]),
                method="full_exact_heat",
                target="exact_heat",
                graph_family="path",
                K=K,
                diffusion_time=diffusion_time,
                temperature=temperature,
                gamma=float(resolved["gamma"]),
                root_state=index,
                value_estimate=float(solution.evaluation_values[index]),
                reference_value=float(validation.evaluation_values[index]),
                absolute_value_error=(float(value_error) if stage == "development" else 0.0),
                policy_l1_error=(
                    float(
                        np.linalg.norm(
                            development.evaluation_policies[index]
                            - validation.evaluation_policies[index],
                            ord=1,
                        )
                    )
                    if stage == "development"
                    else 0.0
                ),
                transition_calls=solution.action_evaluations,
                action_evaluations=solution.action_evaluations,
                unique_actions_touched=K,
                graph_neighbor_accesses=0,
                geometry_preprocess_seconds=solution.geometry_preprocess_seconds,
                online_seconds=solution.online_seconds,
                peak_memory_mb=solution.peak_memory_mb,
                status="complete" if solution.converged else "not_converged",
                git_commit=metadata["git_commit"],
                config_json=resolved_json,
            )
            row.update(
                {
                    "reference_stage": stage,
                    "grid_angle_count": solution.grid.angle_count,
                    "grid_velocity_count": solution.grid.velocity_count,
                    "evaluation_theta": theta,
                    "evaluation_theta_dot": theta_dot,
                    "evaluation_set": str(evaluation_set),
                    "evaluation_state_id": index,
                    "development_value": float(development.evaluation_values[index]),
                    "validation_value": float(validation.evaluation_values[index]),
                    "grid_discretization_error": float(value_error),
                    "policy_mean_torque": float(mean[0]),
                    "policy_mean_abs_torque": float(mean_abs[0]),
                    "policy_entropy": float(entropy[0]),
                    "development_policy_mean_torque": float(development_stats[0][index]),
                    "policy_statistic_error": float(policy_statistic_errors[index]),
                    "bellman_iterations": solution.iterations,
                    "bellman_residual": solution.bellman_residual,
                    "fixed_point_error_bound": solution.bellman_residual / (1.0 - float(resolved["gamma"])),
                    "reference_accepted": accepted,
                    "reference_value_tolerance": float(resolved["reference_tolerance"]),
                    "reference_policy_statistic_tolerance": float(
                        resolved["reference_policy_statistic_tolerance"]
                    ),
                    "heat_scaling": scaling,
                    "action_spacing": environment.action_spacing,
                    "edge_weight": edge_weight,
                    "state_chunk_size": int(reference["state_chunk_size"]),
                    "python": metadata["python"],
                    "platform": metadata["platform"],
                    "machine": metadata["machine"],
                    "processor": metadata["processor"],
                    "cpu_count": metadata["cpu_count"],
                    "thread_environment": json.dumps(metadata["thread_environment"], sort_keys=True),
                    "package_versions": json.dumps(metadata["package_versions"], sort_keys=True),
                }
            )
            rows.append(row)
    frame = pd.DataFrame(rows)
    write_results_atomic(frame, resolved["raw_output"])
    _write_solution_atomic(
        resolved["solution_output"],
        development,
        validation,
        evaluation_states,
        evaluation_sets,
        resolved_json,
    )
    if bool(reference["require_grid_convergence"]) and not accepted:
        maximum = float(value_errors.max(initial=0.0))
        raise RuntimeError(
            "Pendulum reference failed the declared grid-convergence check: "
            f"max evaluation-value change={maximum:.6g}, "
            f"tolerance={float(resolved['reference_tolerance']):.6g}; "
            "artifacts were retained for diagnosis"
        )
    if not accepted:
        LOGGER.warning(
            "Pendulum fixed points converged, but state-grid acceptance failed: "
            "max value change=%.6g (tol %.6g), max policy-statistic change=%.6g "
            "(tol %.6g). The refined-grid artifact is retained with "
            "reference_accepted=false.",
            float(value_errors.max(initial=0.0)),
            float(resolved["reference_tolerance"]),
            float(policy_statistic_errors.max(initial=0.0)),
            float(resolved["reference_policy_statistic_tolerance"]),
        )
    return frame
