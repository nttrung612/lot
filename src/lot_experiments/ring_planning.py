"""M5 end-to-end synthetic ring-control planning experiment."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import poisson

from lot_experiments.config import config_json
from lot_experiments.counters import OperationCounters
from lot_experiments.environments.ring_control import RingControlMDP
from lot_experiments.graphs import ActionGraph, cycle_graph, random_regular_expander
from lot_experiments.heat_local import LazyTruncatedHeat
from lot_experiments.kernels import (
    cycle_exact_heat_kernel,
    cycle_exact_heat_log_kernel,
    exact_heat_kernel,
    poisson_tail,
    truncated_heat_kernel,
    uniformized_random_walk,
)
from lot_experiments.planners.dense import (
    PlanningResult,
    dense_value_iteration,
    diffusion_gibbs_reference,
    hard_max_reference,
    uniform_maxent_reference,
)
from lot_experiments.planners.empirical import (
    EmpiricalComponent,
    component_from_log_samples,
    component_from_log_weights,
    component_from_samples,
    component_from_weights,
    empirical_value_iteration,
    sample_anchor_table,
)
from lot_experiments.planners.poisson_mc import sample_poisson_endpoint
from lot_experiments.pruning import admissible_omitted_mass, top_mass_mask
from lot_experiments.reproducibility import derive_seed, rng_for, run_metadata
from lot_experiments.results import result_row, validate_results, write_results_atomic


METHODS = frozenset(
    {
        "dense_second_order",
        "exact_heat_topmass",
        "truncated_heat_local",
        "poisson_endpoint_mc",
        "uniform_action_mc",
        "random_subset",
    }
)
LOGGER = logging.getLogger(__name__)

RING_COLUMNS = (
    "experiment_part",
    "reference_target",
    "replicate",
    "requested_epsilon",
    "coverage_success",
    "anchor_samples",
    "inner_samples",
    "subset_size",
    "retained_mass_min",
    "deterministic_approximation_error",
    "poisson_tail",
    "graph_ball_size",
    "fixed_point_iterations",
    "fixed_point_residual",
    "cache_hits",
    "cache_misses",
    "metadata_json",
    "error_message",
)

SUMMARY_METRICS = (
    "value_estimate",
    "absolute_value_error",
    "statistical_error",
    "heat_approximation_error",
    "deterministic_approximation_error",
    "policy_l1_error",
    "transition_calls",
    "action_evaluations",
    "unique_actions_touched",
    "graph_neighbor_accesses",
    "dense_linear_algebra_operations",
    "geometry_preprocess_seconds",
    "online_seconds",
    "peak_memory_mb",
    "radius",
    "poisson_tail",
    "graph_ball_size",
    "fixed_point_iterations",
    "fixed_point_residual",
)


def _deep_defaults(destination: dict[str, Any], defaults: Mapping[str, Any]) -> None:
    for key, value in defaults.items():
        if key not in destination:
            destination[key] = copy.deepcopy(value)
        elif isinstance(value, Mapping) and isinstance(destination[key], dict):
            _deep_defaults(destination[key], value)


def resolve_ring_planning_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve and validate the complete M5 experiment configuration."""

    resolved = copy.deepcopy(dict(config))
    _deep_defaults(
        resolved,
        {
            "experiment": "ring_planning",
            "seed": 0,
            "K": [64, 128, 256, 512, 1024],
            "gamma": 0.95,
            "action_cost": 0.05,
            "goal_fraction": 0.0,
            "root_state_fraction": 0.5,
            "temperature": [0.05, 0.1, 0.2],
            "poisson_mean": [0.5, 2.0, 8.0],
            "epsilon": [0.2, 0.1, 0.05],
            "delta": 0.05,
            "anchor_uniform_mass": 0.05,
            "paired_seeds": 30,
            "benchmark_threads": 1,
            "methods": sorted(METHODS),
            "reference": {"tolerance": 1e-9, "max_iterations": 5000},
            "sampling": {
                "anchor_samples": 8,
                "sample_scale": 0.05,
                "minimum_inner_samples": 8,
                "maximum_inner_samples": 128,
            },
            "localization": {
                "heat_tail_scale": 0.25,
                "minimum_tail_budget": 1e-12,
                "maximum_radius": 96,
                "pruning_span_bound": 2.0,
            },
            "different_target_references": {
                "enabled": True,
                "diffusion_gibbs_lambda": 1.0,
            },
            "geometry_controls": {
                "enabled": True,
                "K": 256,
                "temperature": 0.1,
                "poisson_mean": 2.0,
                "epsilon": 0.1,
                "graph_families": ["cycle", "random_regular_expander", "permuted_cycle"],
                "expander_degree": 4,
                "paired_seeds": 30,
            },
            "bootstrap_repetitions": 2000,
            "raw_output": "outputs/raw/ring_planning.parquet",
            "summary_output": "outputs/summaries/ring_planning.csv",
            "figure_png": "outputs/figures/figure3_ring_planning.png",
            "figure_pdf": "outputs/figures/figure3_ring_planning.pdf",
            "figure": {"temperature": 0.1, "poisson_mean": 2.0, "epsilon": 0.1},
        },
    )
    if resolved["experiment"] != "ring_planning":
        raise ValueError("experiment must be 'ring_planning'")
    if int(resolved["benchmark_threads"]) != 1:
        raise ValueError("M5 runtime comparisons require benchmark_threads: 1")
    if not resolved["K"] or any(int(value) < 3 for value in resolved["K"]):
        raise ValueError("K must be a nonempty list of integers >= 3")
    if not 0.0 <= float(resolved["gamma"]) < 1.0:
        raise ValueError("gamma must lie in [0, 1)")
    if not 0.0 < float(resolved["delta"]) < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    for key in ("temperature", "epsilon"):
        if not resolved[key] or any(float(value) <= 0.0 for value in resolved[key]):
            raise ValueError(f"{key} must be nonempty and positive")
    if not resolved["poisson_mean"] or any(
        float(value) < 0.0 for value in resolved["poisson_mean"]
    ):
        raise ValueError("poisson_mean must be nonempty and nonnegative")
    methods = list(map(str, resolved["methods"]))
    unknown = set(methods) - METHODS
    if not methods or unknown:
        raise ValueError(f"unknown or empty method selection: {sorted(unknown)}")
    if int(resolved["paired_seeds"]) < 1:
        raise ValueError("paired_seeds must be positive")
    sampling = resolved["sampling"]
    if int(sampling["anchor_samples"]) < 1:
        raise ValueError("sampling.anchor_samples must be positive")
    if not 1 <= int(sampling["minimum_inner_samples"]) <= int(
        sampling["maximum_inner_samples"]
    ):
        raise ValueError("invalid inner-sample limits")
    localization = resolved["localization"]
    if int(localization["maximum_radius"]) < 0:
        raise ValueError("localization.maximum_radius must be nonnegative")
    if int(resolved["bootstrap_repetitions"]) < 1:
        raise ValueError("bootstrap_repetitions must be positive")
    figure = resolved["figure"]
    selectors = (
        ("temperature", float(figure["temperature"])),
        ("poisson_mean", float(figure["poisson_mean"])),
        ("epsilon", float(figure["epsilon"])),
    )
    for key, selected in selectors:
        if not any(np.isclose(selected, float(value), atol=1e-12, rtol=0.0) for value in resolved[key]):
            raise ValueError(f"figure.{key}={selected:g} is not present in the {key} grid")
    return resolved


def _stable_run_id(parts: Sequence[Any], resolved_json: str) -> str:
    payload = json.dumps([*parts, resolved_json], separators=(",", ":"), default=str)
    digest = hashlib.blake2b(payload.encode(), digest_size=10).hexdigest()
    return f"ring-{digest}"


def _target_for_graph(family: str, base: str) -> str:
    if family == "cycle":
        return base
    if family == "random_regular_expander":
        return f"{base}_expander"
    if family == "permuted_cycle":
        return f"{base}_permuted"
    return f"{base}_{family}"


def _action_graph(family: str, K: int, seed: int, degree: int = 4) -> ActionGraph:
    if family == "cycle":
        return cycle_graph(K)
    if family == "random_regular_expander":
        return random_regular_expander(K, degree=degree, seed=seed)
    if family == "permuted_cycle":
        permutation = rng_for(seed, "permuted_action_graph").permutation(K)
        base = cycle_graph(K).adjacency
        return ActionGraph(base[permutation, :][:, permutation], "permuted_cycle")
    raise ValueError(f"unknown geometry-control graph: {family}")


def _exact_heat(graph: ActionGraph, diffusion_time: float) -> np.ndarray:
    if graph.family == "cycle":
        return cycle_exact_heat_kernel(graph.K, diffusion_time)
    return exact_heat_kernel(graph.laplacian, diffusion_time)


def _heat_radius(theta: float, epsilon: float, gamma: float, T0: float, config: Mapping[str, Any]) -> int:
    tail_budget = float(config["heat_tail_scale"]) * epsilon * (1.0 - gamma) / T0
    tail_budget = min(0.5, max(float(config["minimum_tail_budget"]), tail_budget))
    radius = 0
    maximum = int(config["maximum_radius"])
    while radius < maximum and poisson_tail(theta, radius) > tail_budget:
        radius += 1
    return radius


def _inner_samples(epsilon: float, delta: float, config: Mapping[str, Any]) -> int:
    requested = math.ceil(
        float(config["sample_scale"]) * math.log(2.0 / delta) / epsilon**2
    )
    return max(
        int(config["minimum_inner_samples"]),
        min(int(config["maximum_inner_samples"]), requested),
    )


def _max_ball_size(graph: ActionGraph, radius: int) -> int:
    return max(len(graph.ball(anchor, radius)) for anchor in range(graph.K))


def _build_components(
    method: str,
    mdp: RingControlMDP,
    graph: ActionGraph,
    exact_heat: np.ndarray,
    exact_log_heat: np.ndarray | None,
    anchor_table: np.ndarray,
    *,
    diffusion_time: float,
    radius: int,
    alpha: float,
    inner_samples: int,
    subset_size: int,
    rng: np.random.Generator,
    counter: OperationCounters,
) -> tuple[list[tuple[EmpiricalComponent, ...]], float, int, int]:
    """Build one frozen empirical operator and return cache diagnostics."""

    top_masks = None
    if method == "exact_heat_topmass":
        top_masks = (
            np.column_stack(
                [
                    np.roll(top_mass_mask(exact_heat[:, 0], alpha), anchor)
                    for anchor in range(graph.K)
                ]
            )
            if graph.family == "cycle"
            else np.column_stack(
                [top_mass_mask(exact_heat[:, anchor], alpha) for anchor in range(graph.K)]
            )
        )
    lazy = (
        LazyTruncatedHeat(graph, diffusion_time, radius, counter=counter)
        if method == "truncated_heat_local"
        else None
    )
    rows: list[tuple[EmpiricalComponent, ...]] = []
    retained_mass_min = 1.0
    start = perf_counter()
    endpoint_counter = OperationCounters()
    for state in range(mdp.K):
        components: list[EmpiricalComponent] = []
        for anchor_value in anchor_table[state]:
            anchor = int(anchor_value)
            if method == "dense_second_order":
                component = (
                    component_from_log_weights(exact_log_heat[:, anchor])
                    if exact_log_heat is not None
                    else component_from_weights(exact_heat[:, anchor])
                )
            elif method == "exact_heat_topmass":
                assert top_masks is not None
                actions = np.flatnonzero(top_masks[:, anchor])
                retained_mass_min = min(
                    retained_mass_min, float(exact_heat[actions, anchor].sum())
                )
                component = (
                    component_from_log_weights(
                        exact_log_heat[:, anchor], actions=actions
                    )
                    if exact_log_heat is not None
                    else component_from_weights(
                        exact_heat[:, anchor], actions=actions
                    )
                )
            elif method == "truncated_heat_local":
                assert lazy is not None
                actions, weights = lazy.sparse_column(anchor)
                component = EmpiricalComponent(actions, np.log(weights))
            elif method == "poisson_endpoint_mc":
                endpoints = np.fromiter(
                    (
                        sample_poisson_endpoint(
                            graph,
                            anchor,
                            diffusion_time,
                            rng,
                            counter=endpoint_counter,
                        )
                        for _ in range(inner_samples)
                    ),
                    dtype=np.int64,
                    count=inner_samples,
                )
                component = component_from_samples(
                    endpoints, np.full(inner_samples, 1.0 / inner_samples)
                )
            elif method == "uniform_action_mc":
                actions = rng.integers(0, graph.K, size=inner_samples, dtype=np.int64)
                if exact_log_heat is not None:
                    log_importance = (
                        math.log(graph.K)
                        + exact_log_heat[actions, anchor]
                        - math.log(inner_samples)
                    )
                    component = component_from_log_samples(actions, log_importance)
                else:
                    importance = (
                        graph.K * exact_heat[actions, anchor] / inner_samples
                    )
                    component = component_from_samples(actions, importance)
            elif method == "random_subset":
                actions = rng.choice(graph.K, size=subset_size, replace=False)
                retained_mass_min = min(
                    retained_mass_min, float(exact_heat[actions, anchor].sum())
                )
                component = (
                    component_from_log_weights(
                        exact_log_heat[:, anchor], actions=actions
                    )
                    if exact_log_heat is not None
                    else component_from_weights(
                        exact_heat[:, anchor], actions=actions
                    )
                )
            else:
                raise ValueError(method)
            components.append(component)
        rows.append(tuple(components))
    construction_seconds = perf_counter() - start
    counter.graph_neighbor_accesses += endpoint_counter.graph_neighbor_accesses
    if method != "truncated_heat_local":
        counter.online_seconds += construction_seconds
    cache_hits = 0 if lazy is None else lazy.cache_hits
    cache_misses = 0 if lazy is None else lazy.cache_misses
    return rows, retained_mass_min, cache_hits, cache_misses


def _base_row(
    *,
    run_id: str,
    seed: int,
    method: str,
    target: str,
    graph: ActionGraph,
    resolved_json: str,
    metadata_json: str,
    part: str,
    T0: float,
    theta: float,
    diffusion_time: float,
    gamma: float,
    epsilon: float,
    delta: float,
    radius: int | None,
    root_state: int,
    status: str = "complete",
) -> dict[str, Any]:
    row = result_row(
        experiment="ring_planning",
        run_id=run_id,
        seed=seed,
        method=method,
        target=target,
        graph_family=graph.family,
        K=graph.K,
        radius=np.nan if radius is None else radius,
        diffusion_time=diffusion_time,
        poisson_mean=theta,
        temperature=T0,
        gamma=gamma,
        epsilon=epsilon,
        delta=delta,
        root_state=root_state,
        status=status,
        git_commit=json.loads(metadata_json)["git_commit"],
        config_json=resolved_json,
    )
    row.update({column: np.nan for column in RING_COLUMNS})
    row.update(
        {
            "experiment_part": part,
            "metadata_json": metadata_json,
            "error_message": "",
        }
    )
    return row


def _fill_result(
    row: dict[str, Any],
    result: PlanningResult,
    exact: PlanningResult,
    *,
    reference_target: str,
    epsilon: float,
    statistical_reference: PlanningResult | None = None,
) -> None:
    root = int(row["root_state"])
    estimate = result.root_value(root)
    reference = exact.root_value(root)
    total_error = abs(estimate - reference)
    statistical_error = (
        total_error
        if statistical_reference is None
        else abs(estimate - statistical_reference.root_value(root))
    )
    heat_error = (
        0.0
        if statistical_reference is None
        else abs(statistical_reference.root_value(root) - reference)
    )
    row.update(
        {
            "reference_target": reference_target,
            "value_estimate": estimate,
            "reference_value": reference,
            "absolute_value_error": total_error,
            "statistical_error": statistical_error,
            "heat_approximation_error": heat_error,
            "policy_l1_error": float(
                np.abs(result.policy[root] - exact.policy[root]).sum()
            ),
            "coverage_success": total_error <= epsilon,
            "fixed_point_iterations": result.iterations,
            "fixed_point_residual": result.bellman_residual,
            **result.counters.as_dict(),
        }
    )


def _run_case(
    *,
    resolved: Mapping[str, Any],
    resolved_json: str,
    metadata_json: str,
    rows: list[dict[str, Any]],
    completed: set[str],
    part: str,
    graph: ActionGraph,
    K: int,
    T0: float,
    theta: float,
    epsilon: float,
    repetitions: int,
) -> None:
    LOGGER.info(
        "ring case part=%s graph=%s K=%d T0=%g theta=%g epsilon=%g reps=%d",
        part,
        graph.family,
        K,
        T0,
        theta,
        epsilon,
        repetitions,
    )
    gamma = float(resolved["gamma"])
    delta = float(resolved["delta"])
    _, nu_u = uniformized_random_walk(graph.laplacian)
    diffusion_time = theta / nu_u
    target = _target_for_graph(graph.family, "exact_heat")
    truncated_target = _target_for_graph(graph.family, "truncated_heat")
    goal = int(round(float(resolved["goal_fraction"]) * K)) % K
    root_state = int(round(float(resolved["root_state_fraction"]) * K)) % K
    mdp = RingControlMDP(
        K,
        goal=goal,
        action_cost=float(resolved["action_cost"]),
        anchor_uniform_mass=float(resolved["anchor_uniform_mass"]),
    )
    reference_config = resolved["reference"]
    tolerance = float(reference_config["tolerance"])
    max_iterations = int(reference_config["max_iterations"])
    geometry_start = perf_counter()
    if graph.family == "cycle":
        log_heat = cycle_exact_heat_log_kernel(graph.K, diffusion_time)
        heat = np.exp(log_heat)
    else:
        heat = _exact_heat(graph, diffusion_time)
        log_heat = None
    geometry_seconds = perf_counter() - geometry_start
    exact_counter = OperationCounters(
        geometry_preprocess_seconds=geometry_seconds,
        dense_linear_algebra_operations=1,
    )
    exact = dense_value_iteration(
        mdp,
        gamma=gamma,
        method="full_exact_heat",
        target=target,
        weights=heat,
        T0=T0,
        tolerance=tolerance,
        max_iterations=max_iterations,
        counter=exact_counter,
        circulant_kernel=graph.family == "cycle",
    )
    radius = _heat_radius(theta, epsilon, gamma, T0, resolved["localization"])
    beta_r = poisson_tail(theta, radius)
    subset_size = min(K, _max_ball_size(graph, radius))
    truncated_start = perf_counter()
    truncated_kernel = truncated_heat_kernel(
        graph.laplacian, diffusion_time, radius, nu_u=nu_u
    )
    truncated_geometry_seconds = perf_counter() - truncated_start
    truncated_counter = OperationCounters(
        geometry_preprocess_seconds=truncated_geometry_seconds,
        dense_linear_algebra_operations=max(1, radius),
    )
    truncated = dense_value_iteration(
        mdp,
        gamma=gamma,
        method="full_truncated_heat",
        target=truncated_target,
        weights=truncated_kernel,
        T0=T0,
        tolerance=tolerance,
        max_iterations=max_iterations,
        circulant_kernel=graph.family == "cycle",
        counter=truncated_counter,
    )
    alpha = admissible_omitted_mass(
        float(resolved["localization"]["pruning_span_bound"]),
        (1.0 - gamma) * epsilon / 2.0,
        T0,
    )
    topmass_reference = None
    if part == "primary" and "exact_heat_topmass" in resolved["methods"]:
        top_masks = np.column_stack(
            [np.roll(top_mass_mask(heat[:, 0], alpha), anchor) for anchor in range(K)]
        )
        topmass_reference = dense_value_iteration(
            mdp,
            gamma=gamma,
            method="full_exact_heat_topmass",
            target=target,
            weights=heat,
            retained=top_masks,
            T0=T0,
            tolerance=tolerance,
            max_iterations=max_iterations,
            circulant_kernel=graph.family == "cycle",
        )
    inner = _inner_samples(epsilon, delta, resolved["sampling"])
    anchor_count = int(resolved["sampling"]["anchor_samples"])

    reference_specs = ((exact, "full_exact_heat", target, None), (truncated, "full_truncated_heat", truncated_target, radius))
    for result, method, method_target, method_radius in reference_specs:
        run_id = _stable_run_id((part, graph.family, K, T0, theta, epsilon, method, -1), resolved_json)
        if run_id in completed:
            continue
        row = _base_row(
            run_id=run_id,
            seed=-1,
            method=method,
            target=method_target,
            graph=graph,
            resolved_json=resolved_json,
            metadata_json=metadata_json,
            part=part,
            T0=T0,
            theta=theta,
            diffusion_time=diffusion_time,
            gamma=gamma,
            epsilon=epsilon,
            delta=delta,
            radius=method_radius,
            root_state=root_state,
        )
        _fill_result(
            row,
            result,
            exact,
            reference_target=target,
            epsilon=epsilon,
            statistical_reference=truncated if method == "full_truncated_heat" else None,
        )
        row.update(
            {
                "replicate": -1,
                "requested_epsilon": epsilon,
                "poisson_tail": beta_r if method_radius is not None else 0.0,
                "graph_ball_size": subset_size if method_radius is not None else K,
            }
        )
        rows.append(row)
        completed.add(run_id)

    methods = list(map(str, resolved["methods"]))
    if part == "geometry_control":
        methods = [method for method in methods if method == "truncated_heat_local"]
    for replicate in range(repetitions):
        run_seed = derive_seed(
            int(resolved["seed"]),
            f"{part}:{graph.family}:{K}:{T0}:{theta}:{epsilon}:{replicate}",
        ) % (2**63 - 1)
        anchor_table = sample_anchor_table(
            mdp, anchor_count, rng_for(run_seed, "anchors")
        )
        for method in methods:
            run_id = _stable_run_id((part, graph.family, K, T0, theta, epsilon, method, replicate), resolved_json)
            if run_id in completed:
                continue
            method_target = truncated_target if method == "truncated_heat_local" else target
            row = _base_row(
                run_id=run_id,
                seed=run_seed,
                method=method,
                target=method_target,
                graph=graph,
                resolved_json=resolved_json,
                metadata_json=metadata_json,
                part=part,
                T0=T0,
                theta=theta,
                diffusion_time=diffusion_time,
                gamma=gamma,
                epsilon=epsilon,
                delta=delta,
                radius=radius if method == "truncated_heat_local" else None,
                root_state=root_state,
            )
            row.update(
                {
                    "replicate": replicate,
                    "requested_epsilon": epsilon,
                    "anchor_samples": anchor_count,
                    "inner_samples": inner if method in {"poisson_endpoint_mc", "uniform_action_mc"} else np.nan,
                    "subset_size": subset_size if method == "random_subset" else np.nan,
                    "poisson_tail": beta_r if method == "truncated_heat_local" else np.nan,
                    "graph_ball_size": subset_size if method == "truncated_heat_local" else np.nan,
                }
            )
            try:
                counter = OperationCounters()
                if method in {
                    "dense_second_order",
                    "exact_heat_topmass",
                    "uniform_action_mc",
                    "random_subset",
                }:
                    counter.geometry_preprocess_seconds += geometry_seconds
                    counter.dense_linear_algebra_operations += 1
                if method == "dense_second_order":
                    empirical_anchors = np.zeros((K, K), dtype=np.float64)
                    state_indices = np.repeat(np.arange(K), anchor_count)
                    np.add.at(
                        empirical_anchors,
                        (state_indices, anchor_table.ravel()),
                        1.0 / anchor_count,
                    )
                    result = dense_value_iteration(
                        mdp,
                        gamma=gamma,
                        method=method,
                        target=method_target,
                        weights=heat,
                        T0=T0,
                        anchor_distributions=empirical_anchors,
                        tolerance=tolerance,
                        max_iterations=max_iterations,
                        counter=counter,
                        circulant_kernel=True,
                    )
                    retained_mass, cache_hits, cache_misses = 1.0, 0, 0
                else:
                    components, retained_mass, cache_hits, cache_misses = (
                        _build_components(
                            method,
                            mdp,
                            graph,
                            heat,
                            log_heat,
                            anchor_table,
                            diffusion_time=diffusion_time,
                            radius=radius,
                            alpha=alpha,
                            inner_samples=inner,
                            subset_size=subset_size,
                            rng=rng_for(run_seed, method),
                            counter=counter,
                        )
                    )
                    result = empirical_value_iteration(
                        mdp,
                        components,
                        gamma=gamma,
                        T0=T0,
                        method=method,
                        target=method_target,
                        tolerance=tolerance,
                        max_iterations=max_iterations,
                        counter=counter,
                    )
                if not result.converged:
                    raise RuntimeError(
                        f"value iteration did not converge; residual={result.bellman_residual:g}"
                    )
                _fill_result(
                    row,
                    result,
                    exact,
                    reference_target=target,
                    epsilon=epsilon,
                    statistical_reference=(
                        truncated
                        if method == "truncated_heat_local"
                        else topmass_reference
                        if method == "exact_heat_topmass"
                        else None
                    ),
                )
                if method == "exact_heat_topmass":
                    row["deterministic_approximation_error"] = row[
                        "heat_approximation_error"
                    ]
                    row["heat_approximation_error"] = 0.0
                row.update(
                    {
                        "retained_mass_min": retained_mass,
                        "cache_hits": cache_hits,
                        "cache_misses": cache_misses,
                    }
                )
            except Exception as error:
                row["status"] = "timeout" if isinstance(error, TimeoutError) else "failed"
                row["error_message"] = f"{type(error).__name__}: {error}"
            rows[:] = [
                existing
                for existing in rows
                if str(existing["run_id"]) != run_id
            ]
            rows.append(row)
            completed.add(run_id)
            LOGGER.debug(
                "ring run method=%s replicate=%d status=%s value_error=%s "
                "actions=%s failure=%s",
                method,
                replicate,
                row["status"],
                row["absolute_value_error"],
                row["action_evaluations"],
                row["error_message"] or "none",
            )

    if part == "primary" and bool(resolved["different_target_references"]["enabled"]):
        different = [
            uniform_maxent_reference(
                mdp, gamma=gamma, T0=T0, tolerance=tolerance, max_iterations=max_iterations
            ),
            hard_max_reference(
                mdp, gamma=gamma, tolerance=tolerance, max_iterations=max_iterations
            ),
            diffusion_gibbs_reference(
                mdp,
                gamma=gamma,
                T0=T0,
                diffusion_time=diffusion_time,
                lambda_=float(resolved["different_target_references"]["diffusion_gibbs_lambda"]),
                graph=graph,
                tolerance=tolerance,
                max_iterations=max_iterations,
            ),
        ]
        for result in different:
            run_id = _stable_run_id((part, graph.family, K, T0, theta, epsilon, result.method, -1), resolved_json)
            if run_id in completed:
                continue
            row = _base_row(
                run_id=run_id,
                seed=-1,
                method=result.method,
                target=result.target,
                graph=graph,
                resolved_json=resolved_json,
                metadata_json=metadata_json,
                part="different_target_reference",
                T0=T0,
                theta=theta,
                diffusion_time=diffusion_time,
                gamma=gamma,
                epsilon=epsilon,
                delta=delta,
                radius=None,
                root_state=root_state,
            )
            row.update(
                {
                    "replicate": -1,
                    "requested_epsilon": epsilon,
                    "value_estimate": result.root_value(root_state),
                    "fixed_point_iterations": result.iterations,
                    "fixed_point_residual": result.bellman_residual,
                    **result.counters.as_dict(),
                }
            )
            rows.append(row)
            completed.add(run_id)


def run_ring_planning(
    config: Mapping[str, Any], *, repository: str | Path = "."
) -> pd.DataFrame:
    """Run or resume the complete M5 raw experiment."""

    resolved = resolve_ring_planning_config(config)
    resolved_json = config_json(resolved)
    metadata_json = json.dumps(
        run_metadata(resolved, repository), sort_keys=True, separators=(",", ":")
    )
    output = Path(resolved["raw_output"])
    if output.exists():
        existing = validate_results(pd.read_parquet(output))
        existing = existing.loc[existing["config_json"] == resolved_json].copy()
        rows = existing.to_dict(orient="records")
    else:
        rows = []
    completed = {
        str(row["run_id"]) for row in rows if str(row["status"]) == "complete"
    }

    def commit() -> None:
        write_results_atomic(rows, output)

    def run_case_or_record(**case: Any) -> None:
        try:
            _run_case(
                resolved=resolved,
                resolved_json=resolved_json,
                metadata_json=metadata_json,
                rows=rows,
                completed=completed,
                **case,
            )
        except Exception as error:
            graph = case["graph"]
            _, nu_u = uniformized_random_walk(graph.laplacian)
            failure_id = _stable_run_id(
                (
                    case["part"],
                    graph.family,
                    case["K"],
                    case["T0"],
                    case["theta"],
                    case["epsilon"],
                    "experiment_failure",
                ),
                resolved_json,
            )
            rows[:] = [row for row in rows if str(row["run_id"]) != failure_id]
            failed = _base_row(
                run_id=failure_id,
                seed=int(resolved["seed"]),
                method="experiment_failure",
                target=_target_for_graph(graph.family, "exact_heat"),
                graph=graph,
                resolved_json=resolved_json,
                metadata_json=metadata_json,
                part=case["part"],
                T0=case["T0"],
                theta=case["theta"],
                diffusion_time=case["theta"] / nu_u,
                gamma=float(resolved["gamma"]),
                epsilon=case["epsilon"],
                delta=float(resolved["delta"]),
                radius=None,
                root_state=int(
                    round(float(resolved["root_state_fraction"]) * case["K"])
                )
                % case["K"],
                status="timeout" if isinstance(error, TimeoutError) else "failed",
            )
            failed["error_message"] = f"{type(error).__name__}: {error}"
            rows.append(failed)
            commit()
            raise

    for K_value in resolved["K"]:
        K = int(K_value)
        graph = cycle_graph(K)
        for T0_value in resolved["temperature"]:
            for theta_value in resolved["poisson_mean"]:
                for epsilon_value in resolved["epsilon"]:
                    run_case_or_record(
                        part="primary",
                        graph=graph,
                        K=K,
                        T0=float(T0_value),
                        theta=float(theta_value),
                        epsilon=float(epsilon_value),
                        repetitions=int(resolved["paired_seeds"]),
                    )
                    commit()

    controls = resolved["geometry_controls"]
    if bool(controls["enabled"]):
        K = int(controls["K"])
        for family in map(str, controls["graph_families"]):
            graph = _action_graph(
                family,
                K,
                int(resolved["seed"]),
                degree=int(controls["expander_degree"]),
            )
            run_case_or_record(
                part="geometry_control",
                graph=graph,
                K=K,
                T0=float(controls["temperature"]),
                theta=float(controls["poisson_mean"]),
                epsilon=float(controls["epsilon"]),
                repetitions=int(controls["paired_seeds"]),
            )
            commit()
    return validate_results(pd.DataFrame(rows))


def _bootstrap_interval(values: np.ndarray, repetitions: int, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    medians = np.median(values[indices], axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def summarize_ring_planning(
    raw: pd.DataFrame, *, resolved_config: Mapping[str, Any] | None = None
) -> pd.DataFrame:
    """Median/IQR summary with bootstrap 95% CIs for stochastic results."""

    frame = validate_results(raw)
    frame = frame.loc[
        (frame["experiment"] == "ring_planning") & (frame["status"] == "complete")
    ].copy()
    resolved = None
    if resolved_config is not None:
        resolved = resolve_ring_planning_config(resolved_config)
        frame = frame.loc[frame["config_json"] == config_json(resolved)]
    identifiers = [
        "experiment_part",
        "method",
        "target",
        "reference_target",
        "graph_family",
        "K",
        "radius",
        "diffusion_time",
        "poisson_mean",
        "temperature",
        "gamma",
        "epsilon",
        "delta",
        "root_state",
        "anchor_samples",
        "inner_samples",
        "subset_size",
        "config_json",
    ]
    repetitions = 2000 if resolved is None else int(resolved["bootstrap_repetitions"])
    records: list[dict[str, Any]] = []
    for keys, group in frame.groupby(identifiers, dropna=False, sort=True):
        record = dict(zip(identifiers, keys, strict=True))
        record["n_runs"] = len(group)
        seed = derive_seed(0, json.dumps(list(map(str, keys))))
        rng = np.random.default_rng(seed)
        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            record[f"{metric}_median"] = float(np.median(values)) if len(values) else np.nan
            record[f"{metric}_q25"] = float(np.quantile(values, 0.25)) if len(values) else np.nan
            record[f"{metric}_q75"] = float(np.quantile(values, 0.75)) if len(values) else np.nan
            low, high = _bootstrap_interval(values, repetitions, rng)
            record[f"{metric}_ci_low"] = low
            record[f"{metric}_ci_high"] = high
        coverage = group["coverage_success"].dropna()
        record["empirical_coverage"] = (
            float(coverage.astype(bool).mean()) if len(coverage) else np.nan
        )
        records.append(record)
    return pd.DataFrame(records)


def write_ring_summary_atomic(summary: pd.DataFrame, destination: str | Path) -> Path:
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        summary.to_csv(temporary, index=False)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
