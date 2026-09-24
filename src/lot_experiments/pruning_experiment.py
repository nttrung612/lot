"""M3 sharp-pruning and normalized Poisson-truncation experiments."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import psutil
from numpy.typing import NDArray
from scipy.sparse import csgraph
from scipy.sparse.linalg import expm_multiply

from lot_experiments.backups import heat_backup
from lot_experiments.config import config_json
from lot_experiments.counters import OperationCounters
from lot_experiments.graphs import ActionGraph, graph_from_config
from lot_experiments.heat_local import LazyTruncatedHeat
from lot_experiments.kernels import (
    exact_heat_column,
    exact_heat_columns,
    poisson_tail,
    uniformized_random_walk,
)
from lot_experiments.pruning import (
    poisson_backup_error_bound,
    sharp_pruning_error,
    sharp_pruning_error_from_masses,
    top_mass_mask,
)
from lot_experiments.reproducibility import rng_for, run_metadata
from lot_experiments.results import result_row, validate_results, write_results_atomic


FloatArray = NDArray[np.float64]

PRUNING_COLUMNS = (
    "experiment_part",
    "reference_target",
    "replicate",
    "nominal_alpha",
    "selection_radius",
    "retained_count",
    "retained_mass",
    "omitted_mass",
    "span",
    "span_over_temperature",
    "q_construction",
    "observed_pruning_error",
    "theoretical_pruning_error",
    "pruning_formula_gap",
    "poisson_tail",
    "kernel_l1_bound",
    "backup_error",
    "backup_error_bound",
    "fixed_point_value_error",
    "fixed_point_value_bound",
    "fixed_point_q_error",
    "fixed_point_q_bound",
    "exact_q_span",
    "graph_ball_size",
    "kernel_certificate_holds",
    "backup_certificate_holds",
    "fixed_point_certificate_holds",
    "fixed_point_iterations",
    "fixed_point_residual",
    "metadata_json",
    "error_message",
)

SUMMARY_METRICS = (
    "retained_count",
    "retained_mass",
    "omitted_mass",
    "span",
    "observed_pruning_error",
    "theoretical_pruning_error",
    "pruning_formula_gap",
    "poisson_tail",
    "heat_approximation_error",
    "kernel_l1_bound",
    "backup_error",
    "backup_error_bound",
    "fixed_point_value_error",
    "fixed_point_value_bound",
    "fixed_point_q_error",
    "fixed_point_q_bound",
    "exact_q_span",
    "graph_ball_size",
    "graph_neighbor_accesses",
    "geometry_preprocess_seconds",
    "online_seconds",
)


def _deep_defaults(destination: dict[str, Any], defaults: Mapping[str, Any]) -> None:
    for key, value in defaults.items():
        if key not in destination:
            destination[key] = copy.deepcopy(value)
        elif isinstance(value, Mapping) and isinstance(destination[key], dict):
            _deep_defaults(destination[key], value)


def resolve_pruning_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate M3 settings and inject reproducible defaults."""

    resolved = copy.deepcopy(dict(config))
    defaults: dict[str, Any] = {
        "experiment": "pruning",
        "seed": 0,
        "temperature": 0.1,
        "span_over_temperature": [1, 2, 4, 8],
        "numerical_tolerance": 1e-10,
        "benchmark_threads": 1,
        "raw_output": "outputs/raw/pruning.parquet",
        "summary_output": "outputs/summaries/pruning.csv",
        "figure_png": "outputs/figures/figure2_pruning.png",
        "worst_case": {
            "graph": {"family": "cycle", "K": 41},
            "poisson_mean": 2.0,
            "anchor": 0,
            "alpha": [0.4, 0.2, 0.1, 0.05, 0.02, 0.01],
        },
        "typical": {
            "graphs": [
                {"family": "cycle", "K": 25, "anchor": 0},
                {"family": "grid", "rows": 6, "columns": 6},
            ],
            "poisson_mean": 2.0,
            "subset_radii": [0, 1, 2, 3, 4, 5],
            "repetitions": 100,
            "field_smoothing_time": 1.0,
        },
        "poisson_truncation": {
            "graphs": [
                {"family": "path", "K": 25},
                {"family": "cycle", "K": 25},
                {"family": "grid", "rows": 5, "columns": 5},
            ],
            "poisson_mean": [0.5, 2.0, 8.0],
            "r_max": 20,
            "fixed_point": {
                "state_count": 5,
                "gamma": 0.9,
                "tolerance": 1e-12,
                "max_iterations": 10000,
                "root_state": 0,
            },
        },
        "figure": {
            "typical_span_over_temperature": 4.0,
            "truncation_graph_family": "cycle",
            "truncation_poisson_mean": 2.0,
        },
    }
    _deep_defaults(resolved, defaults)
    if resolved["experiment"] != "pruning":
        raise ValueError("experiment must be 'pruning'")
    if float(resolved["temperature"]) <= 0.0:
        raise ValueError("temperature must be positive")
    ratios = [float(value) for value in resolved["span_over_temperature"]]
    if not ratios or any(value < 0.0 for value in ratios):
        raise ValueError("span_over_temperature must be nonempty and nonnegative")
    alphas = [float(value) for value in resolved["worst_case"]["alpha"]]
    if not alphas or any(not 0.0 < value < 1.0 for value in alphas):
        raise ValueError("worst_case.alpha values must lie in (0, 1)")
    subset_radii = [int(value) for value in resolved["typical"]["subset_radii"]]
    if not subset_radii or any(value < 0 for value in subset_radii):
        raise ValueError("typical.subset_radii must be nonempty and nonnegative")
    if int(resolved["typical"]["repetitions"]) < 1:
        raise ValueError("typical.repetitions must be positive")
    truncation = resolved["poisson_truncation"]
    if int(truncation["r_max"]) < 0:
        raise ValueError("poisson_truncation.r_max must be nonnegative")
    if any(float(value) < 0.0 for value in truncation["poisson_mean"]):
        raise ValueError("poisson means must be nonnegative")
    fixed = truncation["fixed_point"]
    if not 0.0 <= float(fixed["gamma"]) < 1.0:
        raise ValueError("fixed_point.gamma must lie in [0, 1)")
    if int(fixed["state_count"]) < 2 or int(fixed["max_iterations"]) < 1:
        raise ValueError("invalid fixed-point dimensions or iteration limit")
    if float(fixed["tolerance"]) <= 0.0:
        raise ValueError("fixed_point.tolerance must be positive")
    if int(resolved["benchmark_threads"]) != 1:
        raise ValueError("M3 runtime measurements require benchmark_threads: 1")
    # Construct once during validation so malformed graph cases fail before output.
    graph_from_config(dict(resolved["worst_case"]["graph"]))
    for case in [*resolved["typical"]["graphs"], *truncation["graphs"]]:
        graph_from_config(dict(case))
    return resolved


def _stable_run_id(parts: Sequence[Any], resolved_json: str) -> str:
    payload = json.dumps([*parts, resolved_json], separators=(",", ":"), default=str)
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=10).hexdigest()
    return f"pruning-{digest}"


def _anchor_for_case(case: Mapping[str, Any], graph: ActionGraph) -> int:
    if "anchor" in case:
        anchor = int(case["anchor"])
    elif graph.family in {"grid", "torus"}:
        rows, columns = int(case["rows"]), int(case["columns"])
        anchor = (rows // 2) * columns + columns // 2
    else:
        anchor = graph.K // 2
    if not 0 <= anchor < graph.K:
        raise ValueError(f"anchor {anchor} is outside graph with K={graph.K}")
    return anchor


def _base_row(
    *,
    part: str,
    run_id: str,
    seed: int,
    method: str,
    target: str,
    graph: ActionGraph,
    resolved_json: str,
    metadata: Mapping[str, Any],
    T0: float,
    alpha: float | None = None,
    radius: int | None = None,
    diffusion_time: float | None = None,
    theta: float | None = None,
    gamma: float | None = None,
    status: str = "complete",
) -> dict[str, Any]:
    row = result_row(
        experiment="pruning",
        run_id=run_id,
        seed=seed,
        method=method,
        target=target,
        graph_family=graph.family,
        K=graph.K,
        alpha=np.nan if alpha is None else alpha,
        radius=np.nan if radius is None else radius,
        diffusion_time=np.nan if diffusion_time is None else diffusion_time,
        poisson_mean=np.nan if theta is None else theta,
        temperature=T0,
        gamma=np.nan if gamma is None else gamma,
        status=status,
        git_commit=str(metadata["git_commit"]),
        config_json=resolved_json,
    )
    row.update({column: np.nan for column in PRUNING_COLUMNS})
    row.update(
        {
            "experiment_part": part,
            "reference_target": "exact_heat",
            "metadata_json": json.dumps(metadata, sort_keys=True, separators=(",", ":")),
            "error_message": "",
        }
    )
    return row


def _top_cardinality_mask(weights: FloatArray, cardinality: int) -> NDArray[np.bool_]:
    if not 1 <= cardinality <= len(weights):
        raise ValueError("cardinality must lie in [1, K]")
    order = np.argsort(-weights, kind="stable")
    mask = np.zeros(len(weights), dtype=bool)
    mask[order[:cardinality]] = True
    return mask


def _distance_order(graph: ActionGraph, anchor: int, *, farthest: bool) -> NDArray[np.int64]:
    distances = np.asarray(
        csgraph.shortest_path(
            graph.adjacency, directed=False, unweighted=True, indices=anchor
        ),
        dtype=np.float64,
    )
    indices = np.arange(graph.K, dtype=np.int64)
    primary = -distances if farthest else distances
    return np.lexsort((indices, primary)).astype(np.int64)


def equal_cardinality_masks(
    graph: ActionGraph,
    anchor: int,
    radius: int,
    weights: FloatArray,
    rng: np.random.Generator,
) -> dict[str, NDArray[np.bool_]]:
    """Build all four plan baselines with the exact graph-ball cardinality."""

    ball_vertices = graph.ball(anchor, radius)
    cardinality = len(ball_vertices)
    ball = np.zeros(graph.K, dtype=bool)
    ball[list(ball_vertices)] = True
    random_mask = np.zeros(graph.K, dtype=bool)
    random_mask[rng.choice(graph.K, size=cardinality, replace=False)] = True
    farthest = np.zeros(graph.K, dtype=bool)
    farthest[_distance_order(graph, anchor, farthest=True)[:cardinality]] = True
    return {
        "top_kernel_weights": _top_cardinality_mask(weights, cardinality),
        "smallest_graph_ball": ball,
        "uniform_random_subset": random_mask,
        "farthest_action_subset": farthest,
    }


def _rescale_to_span(values: FloatArray, span: float) -> FloatArray:
    values = np.asarray(values, dtype=np.float64)
    width = float(np.ptp(values))
    if span == 0.0:
        return np.zeros_like(values)
    if width <= np.finfo(np.float64).eps:
        raise FloatingPointError("cannot rescale a constant field to positive span")
    return span * (values - float(values.min())) / width


def _typical_values(
    graph: ActionGraph,
    run_seed: int,
    span: float,
    smoothing_time: float,
) -> tuple[FloatArray, str]:
    rng = rng_for(run_seed, f"typical-values:{graph.family}:{graph.K}")
    if graph.family == "cycle":
        angles = 2.0 * np.pi * np.arange(graph.K) / graph.K
        values = np.sin(angles + rng.uniform(0.0, 2.0 * np.pi))
        values += 0.25 * np.sin(2.0 * angles + rng.uniform(0.0, 2.0 * np.pi))
        construction = "smooth_sinusoid"
    elif graph.family in {"grid", "torus"}:
        white = rng.standard_normal(graph.K)
        values = np.asarray(
            expm_multiply(-smoothing_time * graph.laplacian, white), dtype=np.float64
        )
        construction = "smooth_gaussian_random_field"
    else:
        raise ValueError("typical-value study supports cycle and grid/torus graphs")
    return _rescale_to_span(values, span), construction


@dataclass(frozen=True)
class FixedPointSolution:
    action_values: FloatArray
    state_values: FloatArray
    iterations: int
    residual: float


def solve_heat_fixed_point(
    rewards: FloatArray,
    transitions: FloatArray,
    heat_columns: FloatArray,
    anchor_distributions: FloatArray,
    T0: float,
    gamma: float,
    *,
    tolerance: float,
    max_iterations: int,
) -> FixedPointSolution:
    """Solve a small tabular centered-heat Bellman fixed point."""

    rewards = np.asarray(rewards, dtype=np.float64)
    transitions = np.asarray(transitions, dtype=np.float64)
    heat_columns = np.asarray(heat_columns, dtype=np.float64)
    anchor_distributions = np.asarray(anchor_distributions, dtype=np.float64)
    state_count, action_count = rewards.shape
    if transitions.shape != (state_count, action_count, state_count):
        raise ValueError("transitions must have shape (states, actions, states)")
    if heat_columns.shape != (action_count, action_count):
        raise ValueError("heat_columns must have shape (actions, anchors)")
    if anchor_distributions.shape != (state_count, action_count):
        raise ValueError("anchor distributions must have shape (states, anchors)")
    if not np.allclose(transitions.sum(axis=2), 1.0, atol=1e-13, rtol=0.0):
        raise ValueError("transition rows must sum to one")
    state_values = np.zeros(state_count, dtype=np.float64)
    residual = math.inf
    for iteration in range(1, max_iterations + 1):
        action_values = rewards + gamma * np.einsum(
            "sak,k->sa", transitions, state_values, optimize=True
        )
        updated = np.array(
            [
                heat_backup(
                    action_values[state],
                    heat_columns,
                    anchor_distributions[state],
                    T0,
                ).value
                for state in range(state_count)
            ],
            dtype=np.float64,
        )
        residual = float(np.max(np.abs(updated - state_values)))
        state_values = updated
        if residual <= tolerance:
            break
    else:
        raise RuntimeError("heat fixed-point iteration did not converge")
    action_values = rewards + gamma * np.einsum(
        "sak,k->sa", transitions, state_values, optimize=True
    )
    return FixedPointSolution(action_values, state_values, iteration, residual)


def _toy_tabular_mdp(
    graph: ActionGraph, state_count: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Construct the deterministic small MDP used only for M3 propagation checks."""

    K = graph.K
    anchors = (np.arange(state_count, dtype=np.int64) * K // state_count).astype(int)
    anchor_distributions = np.zeros((state_count, K), dtype=np.float64)
    anchor_distributions[np.arange(state_count), anchors] = 1.0
    transitions = np.zeros((state_count, K, state_count), dtype=np.float64)
    action_steps = np.floor(np.arange(K) * state_count / K).astype(int)
    for state in range(state_count):
        destinations = (state + action_steps) % state_count
        for action, destination in enumerate(destinations):
            transitions[state, action, destination] += 0.8
            transitions[state, action, (destination - 1) % state_count] += 0.1
            transitions[state, action, (destination + 1) % state_count] += 0.1
    rewards = np.empty((state_count, K), dtype=np.float64)
    state_scale = max(1, state_count // 2)
    for state, anchor in enumerate(anchors):
        distances = np.asarray(
            csgraph.shortest_path(
                graph.adjacency, directed=False, unweighted=True, indices=int(anchor)
            ),
            dtype=np.float64,
        )
        distance_scale = max(1.0, float(distances.max()))
        state_distance = min(state, state_count - state) / state_scale
        rewards[state] = -0.3 * state_distance**2 - 0.7 * (distances / distance_scale) ** 2
    return rewards, transitions, anchor_distributions


def _max_ball_size(graph: ActionGraph, radius: int) -> int:
    return max(len(graph.ball(anchor, radius)) for anchor in range(graph.K))


def _write_current(rows: list[dict[str, Any]], output_path: Path) -> None:
    write_results_atomic(rows, output_path)


def _run_pruning_experiment_impl(
    config: Mapping[str, Any], *, repository: str | Path = "."
) -> pd.DataFrame:
    """Run or resume all deterministic and paired-seed M3 components."""

    resolved = resolve_pruning_config(config)
    resolved_json = config_json(resolved)
    metadata = run_metadata(resolved, repository)
    output_path = Path(resolved["raw_output"])
    if output_path.exists():
        existing = validate_results(pd.read_parquet(output_path))
        existing = existing.loc[existing["config_json"] == resolved_json].copy()
        rows = existing.to_dict(orient="records")
    else:
        rows = []
    complete_ids = {
        str(row["run_id"]) for row in rows if row.get("status") == "complete"
    }
    base_seed = int(resolved["seed"])
    T0 = float(resolved["temperature"])
    span_ratios = [float(value) for value in resolved["span_over_temperature"]]
    numerical_tolerance = float(resolved["numerical_tolerance"])

    def commit(new_rows: list[dict[str, Any]]) -> None:
        nonlocal rows, complete_ids
        identifiers = {str(row["run_id"]) for row in new_rows}
        rows = [row for row in rows if str(row["run_id"]) not in identifiers]
        rows.extend(new_rows)
        _write_current(rows, output_path)
        complete_ids.update(
            str(row["run_id"]) for row in new_rows if row["status"] == "complete"
        )

    # Part 1: the mask-dependent adversarial Q attains Lemma 2 exactly.
    worst = resolved["worst_case"]
    worst_graph = graph_from_config(dict(worst["graph"]))
    worst_anchor = _anchor_for_case({**worst["graph"], "anchor": worst["anchor"]}, worst_graph)
    _, worst_nu = uniformized_random_walk(worst_graph.laplacian)
    worst_theta = float(worst["poisson_mean"])
    worst_time = worst_theta / worst_nu
    worst_weights = exact_heat_column(worst_graph.laplacian, worst_time, worst_anchor)
    worst_rows: list[dict[str, Any]] = []
    for nominal_alpha in map(float, worst["alpha"]):
        retained = top_mass_mask(worst_weights, nominal_alpha)
        omitted_mass = float(worst_weights[~retained].sum())
        for ratio in span_ratios:
            run_id = _stable_run_id(
                ("worst_case", nominal_alpha, ratio), resolved_json
            )
            if run_id in complete_ids:
                continue
            span = ratio * T0
            action_values = np.where(retained, 0.0, span)
            start = perf_counter()
            full = heat_backup(action_values, worst_weights[:, None], [1.0], T0)
            pruned = heat_backup(
                action_values,
                worst_weights[:, None],
                [1.0],
                T0,
                retained=retained[:, None],
            )
            elapsed = perf_counter() - start
            observed = full.value - pruned.value
            formula = sharp_pruning_error(omitted_mass, span, T0)
            row = _base_row(
                part="worst_case",
                run_id=run_id,
                seed=base_seed,
                method="exact_heat_topmass",
                target="exact_heat",
                graph=worst_graph,
                resolved_json=resolved_json,
                metadata=metadata,
                T0=T0,
                alpha=omitted_mass,
                diffusion_time=worst_time,
                theta=worst_theta,
            )
            row.update(
                {
                    "nominal_alpha": nominal_alpha,
                    "retained_count": int(retained.sum()),
                    "retained_mass": 1.0 - omitted_mass,
                    "omitted_mass": omitted_mass,
                    "span": span,
                    "span_over_temperature": ratio,
                    "q_construction": "adversarial_retained_zero_omitted_span",
                    "observed_pruning_error": observed,
                    "theoretical_pruning_error": formula,
                    "pruning_formula_gap": abs(observed - formula),
                    "value_estimate": pruned.value,
                    "reference_value": full.value,
                    "absolute_value_error": abs(observed),
                    "action_evaluations": int(retained.sum()),
                    "unique_actions_touched": int(retained.sum()),
                    "online_seconds": elapsed,
                    "backup_certificate_holds": abs(observed - formula)
                    <= numerical_tolerance,
                    "peak_memory_mb": psutil.Process().memory_info().rss / (1024.0**2),
                }
            )
            worst_rows.append(row)
    if worst_rows:
        commit(worst_rows)

    # Part 2: smooth values and four equal-cardinality fixed-set baselines.
    typical = resolved["typical"]
    typical_theta = float(typical["poisson_mean"])
    repetitions = int(typical["repetitions"])
    smoothing_time = float(typical["field_smoothing_time"])
    for graph_case_raw in typical["graphs"]:
        graph_case = dict(graph_case_raw)
        graph = graph_from_config(graph_case)
        anchor = _anchor_for_case(graph_case, graph)
        _, nu_u = uniformized_random_walk(graph.laplacian)
        diffusion_time = typical_theta / nu_u
        weights = exact_heat_column(graph.laplacian, diffusion_time, anchor)
        graph_rows: list[dict[str, Any]] = []
        for replicate in range(repetitions):
            run_seed = base_seed + replicate
            for ratio in span_ratios:
                span = ratio * T0
                action_values, construction = _typical_values(
                    graph, run_seed, span, smoothing_time
                )
                full = heat_backup(action_values, weights[:, None], [1.0], T0)
                for selection_radius in map(int, typical["subset_radii"]):
                    masks = equal_cardinality_masks(
                        graph,
                        anchor,
                        selection_radius,
                        weights,
                        rng_for(
                            run_seed,
                            f"subset:{graph.family}:{graph.K}:{selection_radius}",
                        ),
                    )
                    for method, retained in masks.items():
                        run_id = _stable_run_id(
                            (
                                "typical",
                                graph.family,
                                graph.K,
                                replicate,
                                ratio,
                                selection_radius,
                                method,
                            ),
                            resolved_json,
                        )
                        if run_id in complete_ids:
                            continue
                        start = perf_counter()
                        pruned = heat_backup(
                            action_values,
                            weights[:, None],
                            [1.0],
                            T0,
                            retained=retained[:, None],
                        )
                        elapsed = perf_counter() - start
                        retained_mass = float(weights[retained].sum())
                        omitted_mass = float(weights[~retained].sum())
                        observed = full.value - pruned.value
                        certificate = sharp_pruning_error_from_masses(
                            retained_mass, omitted_mass, span, T0
                        )
                        row = _base_row(
                            part="typical",
                            run_id=run_id,
                            seed=run_seed,
                            method=method,
                            target="exact_heat",
                            graph=graph,
                            resolved_json=resolved_json,
                            metadata=metadata,
                            T0=T0,
                            alpha=omitted_mass,
                            diffusion_time=diffusion_time,
                            theta=typical_theta,
                        )
                        row.update(
                            {
                                "replicate": replicate,
                                "selection_radius": selection_radius,
                                "retained_count": int(retained.sum()),
                                "retained_mass": retained_mass,
                                "omitted_mass": omitted_mass,
                                "span": span,
                                "span_over_temperature": ratio,
                                "q_construction": construction,
                                "observed_pruning_error": observed,
                                "theoretical_pruning_error": certificate,
                                "pruning_formula_gap": certificate - observed,
                                "value_estimate": pruned.value,
                                "reference_value": full.value,
                                "absolute_value_error": abs(observed),
                                "action_evaluations": int(retained.sum()),
                                "unique_actions_touched": int(retained.sum()),
                                "graph_ball_size": len(graph.ball(anchor, selection_radius)),
                                "online_seconds": elapsed,
                                "backup_certificate_holds": observed
                                <= certificate + numerical_tolerance,
                                "peak_memory_mb": psutil.Process().memory_info().rss
                                / (1024.0**2),
                            }
                        )
                        graph_rows.append(row)
        if graph_rows:
            commit(graph_rows)

    # Part 3: exact heat versus locally constructed normalized Poisson heads.
    truncation = resolved["poisson_truncation"]
    fixed = truncation["fixed_point"]
    state_count = int(fixed["state_count"])
    gamma = float(fixed["gamma"])
    fixed_tolerance = float(fixed["tolerance"])
    max_iterations = int(fixed["max_iterations"])
    root_state = int(fixed["root_state"])
    for graph_case_raw in truncation["graphs"]:
        graph_case = dict(graph_case_raw)
        graph = graph_from_config(graph_case)
        rewards, transitions, anchor_distributions = _toy_tabular_mdp(
            graph, state_count
        )
        _, nu_u = uniformized_random_walk(graph.laplacian)
        for theta_value in truncation["poisson_mean"]:
            theta = float(theta_value)
            diffusion_time = theta / nu_u
            expected_ids = {
                _stable_run_id(
                    ("poisson_truncation", graph.family, graph.K, theta, radius),
                    resolved_json,
                )
                for radius in range(int(truncation["r_max"]) + 1)
            }
            exact_id = _stable_run_id(
                ("poisson_reference", graph.family, graph.K, theta), resolved_json
            )
            if expected_ids | {exact_id} <= complete_ids:
                continue
            start = perf_counter()
            exact_kernel = exact_heat_columns(graph.laplacian, diffusion_time)
            exact_preprocess = perf_counter() - start
            start = perf_counter()
            exact_solution = solve_heat_fixed_point(
                rewards,
                transitions,
                exact_kernel,
                anchor_distributions,
                T0,
                gamma,
                tolerance=fixed_tolerance,
                max_iterations=max_iterations,
            )
            exact_online = perf_counter() - start
            exact_span = float(
                max(np.ptp(exact_solution.action_values[state]) for state in range(state_count))
            )
            combo_rows: list[dict[str, Any]] = []
            if exact_id not in complete_ids:
                exact_row = _base_row(
                    part="poisson_reference",
                    run_id=exact_id,
                    seed=base_seed,
                    method="full_exact_heat",
                    target="exact_heat",
                    graph=graph,
                    resolved_json=resolved_json,
                    metadata=metadata,
                    T0=T0,
                    diffusion_time=diffusion_time,
                    theta=theta,
                    gamma=gamma,
                )
                exact_row.update(
                    {
                        "root_state": root_state,
                        "value_estimate": exact_solution.state_values[root_state],
                        "reference_value": exact_solution.state_values[root_state],
                        "absolute_value_error": 0.0,
                        "heat_approximation_error": 0.0,
                        "backup_error": 0.0,
                        "fixed_point_value_error": 0.0,
                        "fixed_point_q_error": 0.0,
                        "exact_q_span": exact_span,
                        "graph_ball_size": graph.K,
                        "action_evaluations": exact_solution.iterations
                        * state_count
                        * graph.K,
                        "unique_actions_touched": graph.K,
                        "graph_neighbor_accesses": graph.adjacency.nnz,
                        "geometry_preprocess_seconds": exact_preprocess,
                        "online_seconds": exact_online,
                        "fixed_point_iterations": exact_solution.iterations,
                        "fixed_point_residual": exact_solution.residual,
                        "kernel_certificate_holds": True,
                        "backup_certificate_holds": True,
                        "fixed_point_certificate_holds": True,
                        "peak_memory_mb": psutil.Process().memory_info().rss
                        / (1024.0**2),
                    }
                )
                combo_rows.append(exact_row)
            exact_backups = np.array(
                [
                    heat_backup(
                        exact_solution.action_values[state],
                        exact_kernel,
                        anchor_distributions[state],
                        T0,
                    ).value
                    for state in range(state_count)
                ]
            )
            for radius in range(int(truncation["r_max"]) + 1):
                run_id = _stable_run_id(
                    ("poisson_truncation", graph.family, graph.K, theta, radius),
                    resolved_json,
                )
                if run_id in complete_ids:
                    continue
                counter = OperationCounters()
                lazy = LazyTruncatedHeat(
                    graph,
                    diffusion_time,
                    radius,
                    nu_u=nu_u,
                    counter=counter,
                )
                start = perf_counter()
                truncated_kernel = np.column_stack(
                    [lazy.column(anchor) for anchor in range(graph.K)]
                )
                construction_seconds = perf_counter() - start
                beta_r = poisson_tail(theta, radius)
                kernel_error = float(
                    np.max(np.abs(exact_kernel - truncated_kernel).sum(axis=0))
                )
                backup_bound = poisson_backup_error_bound(beta_r, exact_span, T0)
                truncated_backups = np.array(
                    [
                        heat_backup(
                            exact_solution.action_values[state],
                            truncated_kernel,
                            anchor_distributions[state],
                            T0,
                        ).value
                        for state in range(state_count)
                    ]
                )
                backup_error = float(np.max(np.abs(exact_backups - truncated_backups)))
                start = perf_counter()
                truncated_solution = solve_heat_fixed_point(
                    rewards,
                    transitions,
                    truncated_kernel,
                    anchor_distributions,
                    T0,
                    gamma,
                    tolerance=fixed_tolerance,
                    max_iterations=max_iterations,
                )
                online_seconds = perf_counter() - start
                value_error = float(
                    np.max(
                        np.abs(
                            exact_solution.state_values - truncated_solution.state_values
                        )
                    )
                )
                q_error = float(
                    np.max(
                        np.abs(
                            exact_solution.action_values
                            - truncated_solution.action_values
                        )
                    )
                )
                value_bound = backup_bound / (1.0 - gamma)
                q_bound = gamma * backup_bound / (1.0 - gamma)
                supports = [
                    set(np.flatnonzero(truncated_kernel[:, anchor] > 0.0))
                    for anchor in np.flatnonzero(anchor_distributions.sum(axis=0) > 0.0)
                ]
                actions_touched = len(set().union(*supports)) if supports else 0
                evaluations_per_sweep = sum(len(support) for support in supports)
                row = _base_row(
                    part="poisson_truncation",
                    run_id=run_id,
                    seed=base_seed,
                    method="truncated_heat_local",
                    target="truncated_heat",
                    graph=graph,
                    resolved_json=resolved_json,
                    metadata=metadata,
                    T0=T0,
                    radius=radius,
                    diffusion_time=diffusion_time,
                    theta=theta,
                    gamma=gamma,
                )
                row.update(
                    {
                        "root_state": root_state,
                        "value_estimate": truncated_solution.state_values[root_state],
                        "reference_value": exact_solution.state_values[root_state],
                        "absolute_value_error": abs(
                            truncated_solution.state_values[root_state]
                            - exact_solution.state_values[root_state]
                        ),
                        "statistical_error": 0.0,
                        "heat_approximation_error": kernel_error,
                        "poisson_tail": beta_r,
                        "kernel_l1_bound": 2.0 * beta_r,
                        "backup_error": backup_error,
                        "backup_error_bound": backup_bound,
                        "fixed_point_value_error": value_error,
                        "fixed_point_value_bound": value_bound,
                        "fixed_point_q_error": q_error,
                        "fixed_point_q_bound": q_bound,
                        "exact_q_span": exact_span,
                        "graph_ball_size": _max_ball_size(graph, radius),
                        "action_evaluations": truncated_solution.iterations
                        * evaluations_per_sweep,
                        "unique_actions_touched": actions_touched,
                        "graph_neighbor_accesses": counter.graph_neighbor_accesses,
                        "geometry_preprocess_seconds": construction_seconds,
                        "online_seconds": online_seconds,
                        "fixed_point_iterations": truncated_solution.iterations,
                        "fixed_point_residual": truncated_solution.residual,
                        "kernel_certificate_holds": kernel_error
                        <= 2.0 * beta_r + numerical_tolerance,
                        "backup_certificate_holds": backup_error
                        <= backup_bound + numerical_tolerance,
                        "fixed_point_certificate_holds": value_error
                        <= value_bound + numerical_tolerance
                        and q_error <= q_bound + numerical_tolerance,
                        "peak_memory_mb": max(
                            counter.peak_memory_mb,
                            psutil.Process().memory_info().rss / (1024.0**2),
                        ),
                    }
                )
                combo_rows.append(row)
            if combo_rows:
                commit(combo_rows)
    return validate_results(pd.DataFrame(rows))


def run_pruning_experiment(
    config: Mapping[str, Any], *, repository: str | Path = "."
) -> pd.DataFrame:
    """Run M3 and leave an explicit raw failure marker before re-raising."""

    resolved = resolve_pruning_config(config)
    resolved_json = config_json(resolved)
    failure_id = _stable_run_id(("experiment_failure",), resolved_json)
    output_path = Path(resolved["raw_output"])
    try:
        frame = _run_pruning_experiment_impl(resolved, repository=repository)
    except Exception as error:
        metadata = run_metadata(resolved, repository)
        graph = graph_from_config(dict(resolved["worst_case"]["graph"]))
        failed = _base_row(
            part="experiment_failure",
            run_id=failure_id,
            seed=int(resolved["seed"]),
            method="experiment_failure",
            target="exact_heat",
            graph=graph,
            resolved_json=resolved_json,
            metadata=metadata,
            T0=float(resolved["temperature"]),
            status="failed",
        )
        failed["error_message"] = f"{type(error).__name__}: {error}"
        if output_path.exists():
            existing = validate_results(pd.read_parquet(output_path))
            existing = existing.loc[existing["config_json"] == resolved_json].copy()
            rows = existing.to_dict(orient="records")
        else:
            rows = []
        rows = [row for row in rows if str(row["run_id"]) != failure_id]
        rows.append(failed)
        write_results_atomic(rows, output_path)
        raise
    if failure_id in set(frame["run_id"]):
        frame = frame.loc[frame["run_id"] != failure_id].copy()
        write_results_atomic(frame, output_path)
    return validate_results(frame)


def summarize_pruning(
    raw: pd.DataFrame, *, resolved_config: Mapping[str, Any] | None = None
) -> pd.DataFrame:
    """Aggregate paired smooth-value runs while retaining deterministic checks."""

    frame = validate_results(raw)
    frame = frame.loc[
        (frame["experiment"] == "pruning") & (frame["status"] == "complete")
    ].copy()
    if resolved_config is not None:
        expected = config_json(resolve_pruning_config(resolved_config))
        frame = frame.loc[frame["config_json"] == expected]
    identifiers = [
        "experiment_part",
        "method",
        "target",
        "reference_target",
        "graph_family",
        "K",
        "nominal_alpha",
        "selection_radius",
        "radius",
        "diffusion_time",
        "poisson_mean",
        "temperature",
        "gamma",
        "span_over_temperature",
        "q_construction",
        "config_json",
    ]
    records: list[dict[str, Any]] = []
    for keys, group in frame.groupby(identifiers, dropna=False, sort=True):
        record = dict(zip(identifiers, keys, strict=True))
        record["n_runs"] = len(group)
        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            record[f"{metric}_median"] = float(values.median()) if len(values) else np.nan
            record[f"{metric}_q25"] = float(values.quantile(0.25)) if len(values) else np.nan
            record[f"{metric}_q75"] = float(values.quantile(0.75)) if len(values) else np.nan
        for flag in (
            "kernel_certificate_holds",
            "backup_certificate_holds",
            "fixed_point_certificate_holds",
        ):
            values = group[flag].dropna()
            record[flag] = bool(values.astype(bool).all()) if len(values) else np.nan
        records.append(record)
    return pd.DataFrame(records)


def write_pruning_summary_atomic(
    summary: pd.DataFrame, destination: str | Path
) -> Path:
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        summary.to_csv(temporary_path, index=False)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path
