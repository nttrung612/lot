"""Kernel design-map experiment (milestone M2)."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import psutil
from numpy.typing import NDArray
from scipy.sparse import csgraph
from scipy.stats import poisson

from lot_experiments.config import config_json
from lot_experiments.counters import OperationCounters
from lot_experiments.graphs import (
    ActionGraph,
    cycle_graph,
    grid_graph,
    path_graph,
    random_regular_expander,
    torus_graph,
)
from lot_experiments.heat_local import (
    local_truncated_heat_column,
    local_truncated_heat_columns,
)
from lot_experiments.kernels import (
    diffusion_distance_gibbs_kernel,
    exact_heat_column,
    exact_heat_columns,
    poisson_tail,
    uniformized_random_walk,
)
from lot_experiments.reproducibility import derive_seed, run_metadata
from lot_experiments.results import result_row, validate_results, write_results_atomic


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
LOGGER = logging.getLogger(__name__)

METHOD_TARGETS = {
    "uniform_maxent": "maxent",
    "diffusion_distance_gibbs": "diffusion_gibbs",
    "exact_heat": "exact_heat",
    "truncated_heat": "truncated_heat",
}

KERNEL_MAP_COLUMNS = (
    "requested_K",
    "graph_rows",
    "graph_columns",
    "graph_degree",
    "gibbs_lambda",
    "kernel_effective_count",
    "kernel_effective_fraction",
    "high_mass_ball_radius",
    "exact_support_size",
    "poisson_tail",
    "certificate_l1_bound",
    "certificate_holds",
    "vertices_accessed",
    "neighbor_entries_accessed",
    "one_column_seconds",
    "cached_one_column_seconds",
    "analysis_seconds",
    "one_column_memory_bytes",
    "full_matrix_memory_bytes",
    "diffusion_cost_max",
    "gibbs_count_lower_bound",
    "metadata_json",
    "error_message",
)


def resolve_kernel_map_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the plan grid and inject all experiment defaults."""

    resolved = copy.deepcopy(dict(config))
    defaults: dict[str, Any] = {
        "experiment": "kernel_map",
        "expander_degree": 4,
        "gibbs_lambda": 1.0,
        "timing_repeats": 3,
        "benchmark_threads": 1,
        "dense_batch_size": 128,
        "raw_output": "outputs/raw/kernel_map.parquet",
        "summary_output": "outputs/summaries/kernel_map.csv",
        "figure_png": "outputs/figures/figure1_kernel_map.png",
        "figure_pdf": "outputs/figures/figure1_kernel_map.pdf",
        "figure": {
            "alpha": 0.01,
            "poisson_mean": 2.0,
            "primary_graph_family": "path",
            "comparison_requested_K": 1024,
        },
    }
    for key, value in defaults.items():
        resolved.setdefault(key, value)
    required = ("graph_families", "K", "alpha", "poisson_mean")
    missing = [key for key in required if key not in resolved]
    if missing:
        raise ValueError(f"kernel-map configuration is missing: {missing}")
    if resolved["experiment"] != "kernel_map":
        raise ValueError("experiment must be 'kernel_map'")
    if any(int(K) < 3 for K in resolved["K"]):
        raise ValueError("all requested K values must be at least three")
    if any(not 0.0 < float(alpha) < 1.0 for alpha in resolved["alpha"]):
        raise ValueError("alpha values must lie strictly between zero and one")
    if any(float(theta) < 0.0 for theta in resolved["poisson_mean"]):
        raise ValueError("poisson_mean values must be nonnegative")
    if float(resolved["gibbs_lambda"]) <= 0.0:
        raise ValueError("gibbs_lambda must be positive")
    if int(resolved["benchmark_threads"]) != 1:
        raise ValueError("M2 runtime benchmarks require benchmark_threads: 1")
    return resolved


def _graph_for_request(
    family: str, requested_K: int, *, degree: int, seed: int
) -> tuple[ActionGraph, int | None, int | None]:
    if family == "path":
        return path_graph(requested_K), None, None
    if family == "cycle":
        return cycle_graph(requested_K), None, None
    if family in {"grid", "torus"}:
        side = max(2 if family == "grid" else 3, int(round(math.sqrt(requested_K))))
        graph = grid_graph(side, side) if family == "grid" else torus_graph(side, side)
        return graph, side, side
    if family == "random_regular_expander":
        graph_seed = derive_seed(seed, f"kernel-map:{family}:{requested_K}") % (2**32)
        return (
            random_regular_expander(requested_K, degree, seed=int(graph_seed)),
            None,
            None,
        )
    raise ValueError(f"unsupported graph family: {family}")


def certified_poisson_radius(theta: float, tail_budget: float) -> int:
    """Smallest integer radius with ``Pr(Pois(theta) > r) <= tail_budget``."""

    if theta < 0.0 or not math.isfinite(theta):
        raise ValueError("theta must be finite and nonnegative")
    if not 0.0 < tail_budget < 1.0:
        raise ValueError("tail_budget must lie in (0, 1)")
    radius = int(poisson.ppf(1.0 - tail_budget, theta))
    radius = max(radius, 0)
    while poisson_tail(theta, radius) > tail_budget:
        radius += 1
    while radius > 0 and poisson_tail(theta, radius - 1) <= tail_budget:
        radius -= 1
    return radius


def kernel_effective_counts(kernel: FloatArray, alphas: Sequence[float]) -> IntArray:
    """Worst-anchor top-mass cardinalities for several exact tail budgets."""

    matrix = np.asarray(kernel, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("kernel must be a nonempty matrix")
    if np.min(matrix, initial=0.0) < -1e-13:
        raise ValueError("kernel must be nonnegative")
    if not np.allclose(matrix.sum(axis=0), 1.0, atol=1e-11, rtol=0.0):
        raise ValueError("kernel columns must sum to one")
    alpha_array = np.asarray(alphas, dtype=np.float64)
    counts = np.zeros(len(alpha_array), dtype=np.int64)
    targets = 1.0 - alpha_array
    for anchor in range(matrix.shape[1]):
        descending = np.sort(matrix[:, anchor])[::-1]
        cumulative = np.cumsum(descending)
        column_counts = np.searchsorted(cumulative, targets, side="left") + 1
        counts = np.maximum(counts, column_counts)
    return counts


def _distance_matrix(graph: ActionGraph) -> NDArray[np.int16]:
    distances = csgraph.shortest_path(
        graph.adjacency, directed=False, unweighted=True, method="D"
    )
    if not np.all(np.isfinite(distances)):
        raise ValueError("kernel-map graphs must be connected")
    if np.max(distances) >= np.iinfo(np.int16).max:
        raise ValueError("graph diameter exceeds diagnostic integer storage")
    return distances.astype(np.int16)


def high_mass_ball_radii(
    kernel: FloatArray | None,
    distances: NDArray[np.int16],
    alphas: Sequence[float],
) -> IntArray:
    """Smallest worst-anchor graph-ball radius retaining mass ``1-alpha``.

    Passing ``kernel=None`` computes the uniform-kernel result without forming
    a dense uniform matrix.
    """

    K = distances.shape[0]
    if distances.shape != (K, K):
        raise ValueError("distances must be square")
    if kernel is not None and np.asarray(kernel).shape != (K, K):
        raise ValueError("kernel and distance matrix shapes must match")
    radii = np.zeros(len(alphas), dtype=np.int64)
    targets = 1.0 - np.asarray(alphas, dtype=np.float64)
    for anchor in range(K):
        anchor_distances = distances[:, anchor].astype(np.int64, copy=False)
        weights = None if kernel is None else np.asarray(kernel)[:, anchor]
        mass_by_radius = np.bincount(
            anchor_distances,
            weights=weights,
            minlength=int(anchor_distances.max()) + 1,
        )
        if weights is None:
            mass_by_radius = mass_by_radius / K
        cumulative = np.cumsum(mass_by_radius)
        anchor_radii = np.searchsorted(cumulative, targets, side="left")
        radii = np.maximum(radii, anchor_radii)
    return radii


def _median_seconds(function: Callable[[], Any], repeats: int) -> float:
    timings = []
    for _ in range(repeats):
        start = perf_counter()
        function()
        timings.append(perf_counter() - start)
    return float(np.median(timings))


def _stable_run_id(parts: Sequence[Any], resolved_json: str) -> str:
    payload = json.dumps([*parts, resolved_json], separators=(",", ":"), default=str)
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=10).hexdigest()
    return f"kernel-map-{digest}"


def _base_row(
    *,
    resolved: Mapping[str, Any],
    resolved_json: str,
    metadata_json: str,
    git_commit: str,
    seed: int,
    method: str,
    family: str,
    requested_K: int,
    graph: ActionGraph,
    rows: int | None,
    columns: int | None,
    degree: int | None,
    alpha: float,
    theta: float,
    diffusion_time: float,
    radius: int | float,
    status: str = "complete",
) -> dict[str, Any]:
    run_id = _stable_run_id(
        (family, requested_K, graph.K, theta, alpha, method), resolved_json
    )
    row = result_row(
        experiment="kernel_map",
        run_id=run_id,
        seed=seed,
        method=method,
        target=METHOD_TARGETS[method],
        graph_family=family,
        K=graph.K,
        alpha=alpha,
        radius=radius,
        diffusion_time=diffusion_time,
        poisson_mean=theta,
        status=status,
        git_commit=git_commit,
        config_json=resolved_json,
    )
    row.update(
        {
            "requested_K": requested_K,
            "graph_rows": rows if rows is not None else np.nan,
            "graph_columns": columns if columns is not None else np.nan,
            "graph_degree": degree if degree is not None else np.nan,
            "gibbs_lambda": float(resolved["gibbs_lambda"]),
            "metadata_json": metadata_json,
            "error_message": "",
        }
    )
    return row


def _method_rows(
    base: Mapping[str, Any],
    alphas: Sequence[float],
    counts: Sequence[int],
    radii: Sequence[int],
    **metrics: Any,
) -> list[dict[str, Any]]:
    rows = []
    for alpha, count, ball_radius in zip(alphas, counts, radii, strict=True):
        row = dict(base)
        row["alpha"] = float(alpha)
        row["run_id"] = _stable_run_id(
            (
                row["graph_family"],
                row["requested_K"],
                row["K"],
                row["poisson_mean"],
                alpha,
                row["method"],
            ),
            str(row["config_json"]),
        )
        row.update(metrics)
        row["kernel_effective_count"] = int(count)
        row["kernel_effective_fraction"] = float(count) / int(row["K"])
        row["high_mass_ball_radius"] = int(ball_radius)
        rows.append(row)
    return rows


def run_kernel_map(
    config: Mapping[str, Any], *, repository: str | Path = "."
) -> pd.DataFrame:
    """Run or resume the configured kernel design-map study."""

    resolved = resolve_kernel_map_config(config)
    resolved_json = config_json(resolved)
    metadata = run_metadata(resolved, repository)
    metadata_json = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    seed = int(resolved["seed"])
    alphas = [float(value) for value in resolved["alpha"]]
    repeats = int(resolved["timing_repeats"])
    batch_size = int(resolved["dense_batch_size"])
    lambda_ = float(resolved["gibbs_lambda"])
    output_path = Path(resolved["raw_output"])
    if output_path.exists():
        existing = pd.read_parquet(output_path)
        existing = validate_results(existing)
        existing = existing.loc[existing["config_json"] == resolved_json].copy()
    else:
        existing = pd.DataFrame()
    completed_ids = (
        set(existing.loc[existing["status"] == "complete", "run_id"])
        if not existing.empty
        else set()
    )
    all_rows = [] if existing.empty else existing.to_dict(orient="records")

    for family in map(str, resolved["graph_families"]):
        for requested_K_value in resolved["K"]:
            requested_K = int(requested_K_value)
            graph, graph_rows, graph_columns = _graph_for_request(
                family,
                requested_K,
                degree=int(resolved["expander_degree"]),
                seed=seed,
            )
            LOGGER.info(
                "kernel-map graph family=%s requested_K=%d actual_K=%d",
                family,
                requested_K,
                graph.K,
            )
            graph_degree = (
                int(resolved["expander_degree"])
                if family == "random_regular_expander"
                else None
            )
            graph_expected_ids = {
                _stable_run_id(
                    (
                        family,
                        requested_K,
                        graph.K,
                        float(theta),
                        alpha,
                        method,
                    ),
                    resolved_json,
                )
                for theta in resolved["poisson_mean"]
                for method in METHOD_TARGETS
                for alpha in alphas
            }
            if graph_expected_ids <= completed_ids:
                LOGGER.info(
                    "skip completed graph family=%s requested_K=%d actual_K=%d",
                    family,
                    requested_K,
                    graph.K,
                )
                continue
            distances = _distance_matrix(graph)
            _, nu_u = uniformized_random_walk(graph.laplacian)
            representative_anchor = graph.K // 2
            for theta_value in resolved["poisson_mean"]:
                theta = float(theta_value)
                diffusion_time = theta / nu_u
                expected_ids = {
                    _stable_run_id(
                        (family, requested_K, graph.K, theta, alpha, method),
                        resolved_json,
                    )
                    for method in METHOD_TARGETS
                    for alpha in alphas
                }
                if expected_ids <= completed_ids:
                    LOGGER.info(
                        "skip completed family=%s K=%d theta=%g",
                        family,
                        graph.K,
                        theta,
                    )
                    continue
                LOGGER.info(
                    "run family=%s K=%d theta=%g (%d rows)",
                    family,
                    graph.K,
                    theta,
                    len(expected_ids),
                )
                combo_rows: list[dict[str, Any]] = []
                process = psutil.Process()
                try:
                    # Uniform MaxEnt is analytic and uses no graph preprocessing.
                    uniform_counts = np.ceil((1.0 - np.asarray(alphas)) * graph.K).astype(int)
                    uniform_radii = high_mass_ball_radii(None, distances, alphas)
                    uniform_time = _median_seconds(
                        lambda: np.full(graph.K, 1.0 / graph.K), repeats
                    )
                    uniform_base = _base_row(
                        resolved=resolved,
                        resolved_json=resolved_json,
                        metadata_json=metadata_json,
                        git_commit=str(metadata["git_commit"]),
                        seed=seed,
                        method="uniform_maxent",
                        family=family,
                        requested_K=requested_K,
                        graph=graph,
                        rows=graph_rows,
                        columns=graph_columns,
                        degree=graph_degree,
                        alpha=alphas[0],
                        theta=theta,
                        diffusion_time=diffusion_time,
                        radius=np.nan,
                    )
                    combo_rows.extend(
                        _method_rows(
                            uniform_base,
                            alphas,
                            uniform_counts,
                            uniform_radii,
                            exact_support_size=graph.K,
                            vertices_accessed=graph.K,
                            neighbor_entries_accessed=0,
                            one_column_seconds=uniform_time,
                            cached_one_column_seconds=uniform_time,
                            analysis_seconds=0.0,
                            geometry_preprocess_seconds=0.0,
                            one_column_memory_bytes=graph.K * 8,
                            full_matrix_memory_bytes=graph.K * graph.K * 8,
                            peak_memory_mb=process.memory_info().rss / (1024.0**2),
                        )
                    )

                    # Exact heat is the dense correctness reference.
                    start = perf_counter()
                    exact = exact_heat_columns(
                        graph.laplacian,
                        diffusion_time,
                        batch_size=batch_size,
                    )
                    exact_analysis_seconds = perf_counter() - start
                    exact_counts = kernel_effective_counts(exact, alphas)
                    exact_radii = high_mass_ball_radii(exact, distances, alphas)
                    exact_time = _median_seconds(
                        lambda: exact_heat_column(
                            graph.laplacian, diffusion_time, representative_anchor
                        ),
                        repeats,
                    )
                    exact_cached_time = _median_seconds(
                        lambda: exact[:, representative_anchor].copy(), repeats
                    )
                    exact_base = _base_row(
                        resolved=resolved,
                        resolved_json=resolved_json,
                        metadata_json=metadata_json,
                        git_commit=str(metadata["git_commit"]),
                        seed=seed,
                        method="exact_heat",
                        family=family,
                        requested_K=requested_K,
                        graph=graph,
                        rows=graph_rows,
                        columns=graph_columns,
                        degree=graph_degree,
                        alpha=alphas[0],
                        theta=theta,
                        diffusion_time=diffusion_time,
                        radius=np.nan,
                    )
                    combo_rows.extend(
                        _method_rows(
                            exact_base,
                            alphas,
                            exact_counts,
                            exact_radii,
                            exact_support_size=graph.K,
                            heat_approximation_error=0.0,
                            vertices_accessed=graph.K,
                            neighbor_entries_accessed=graph.adjacency.nnz,
                            one_column_seconds=exact_time,
                            cached_one_column_seconds=exact_cached_time,
                            analysis_seconds=exact_analysis_seconds,
                            geometry_preprocess_seconds=0.0,
                            one_column_memory_bytes=graph.K * 8,
                            full_matrix_memory_bytes=graph.K * graph.K * 8,
                            peak_memory_mb=process.memory_info().rss / (1024.0**2),
                        )
                    )

                    # Diffusion-distance Gibbs is a distinct, dense target.
                    start = perf_counter()
                    gibbs, cost_max, heat_twice_diagonal = diffusion_distance_gibbs_kernel(
                        graph.laplacian,
                        diffusion_time,
                        lambda_,
                        batch_size=batch_size,
                    )
                    gibbs_preprocess_seconds = perf_counter() - start
                    gibbs_counts = kernel_effective_counts(gibbs, alphas)
                    gibbs_radii = high_mass_ball_radii(gibbs, distances, alphas)

                    def uncached_gibbs_column() -> FloatArray:
                        heat_column = exact_heat_column(
                            graph.laplacian,
                            2.0 * diffusion_time,
                            representative_anchor,
                        )
                        cost = (
                            heat_twice_diagonal
                            + heat_twice_diagonal[representative_anchor]
                            - 2.0 * heat_column
                        )
                        weights = np.exp(-np.maximum(cost, 0.0) / lambda_)
                        return weights / weights.sum()

                    gibbs_time = _median_seconds(uncached_gibbs_column, repeats)
                    gibbs_cached_time = _median_seconds(
                        lambda: gibbs[:, representative_anchor].copy(), repeats
                    )
                    gibbs_base = _base_row(
                        resolved=resolved,
                        resolved_json=resolved_json,
                        metadata_json=metadata_json,
                        git_commit=str(metadata["git_commit"]),
                        seed=seed,
                        method="diffusion_distance_gibbs",
                        family=family,
                        requested_K=requested_K,
                        graph=graph,
                        rows=graph_rows,
                        columns=graph_columns,
                        degree=graph_degree,
                        alpha=alphas[0],
                        theta=theta,
                        diffusion_time=diffusion_time,
                        radius=np.nan,
                    )
                    gibbs_rows = _method_rows(
                        gibbs_base,
                        alphas,
                        gibbs_counts,
                        gibbs_radii,
                        exact_support_size=graph.K,
                        vertices_accessed=graph.K,
                        neighbor_entries_accessed=graph.adjacency.nnz,
                        one_column_seconds=gibbs_time,
                        cached_one_column_seconds=gibbs_cached_time,
                        analysis_seconds=gibbs_preprocess_seconds,
                        geometry_preprocess_seconds=gibbs_preprocess_seconds,
                        one_column_memory_bytes=graph.K * 8,
                        full_matrix_memory_bytes=graph.K * graph.K * 8,
                        diffusion_cost_max=cost_max,
                        peak_memory_mb=process.memory_info().rss / (1024.0**2),
                    )
                    for row in gibbs_rows:
                        row["gibbs_count_lower_bound"] = graph.K * max(
                            0.0,
                            1.0
                            - float(row["alpha"])
                            * math.exp(cost_max / lambda_),
                        )
                    combo_rows.extend(gibbs_rows)
                    del gibbs, heat_twice_diagonal

                    # Truncated heat stays column-lazy. Walk powers are reused
                    # across radii, while beta_r remains distinct from alpha.
                    truncated_radii = [
                        certified_poisson_radius(theta, alpha) for alpha in alphas
                    ]
                    truncated_counts = np.zeros(len(alphas), dtype=np.int64)
                    truncated_ball_radii = np.zeros(len(alphas), dtype=np.int64)
                    truncated_support_sizes = np.zeros(len(alphas), dtype=np.int64)
                    truncated_max_l1 = np.zeros(len(alphas), dtype=np.float64)
                    targets = 1.0 - np.asarray(alphas, dtype=np.float64)
                    start = perf_counter()
                    for anchor in range(graph.K):
                        columns_by_radius = local_truncated_heat_columns(
                            graph,
                            diffusion_time,
                            truncated_radii,
                            anchor,
                            nu_u=nu_u,
                        )
                        anchor_distances = distances[:, anchor].astype(
                            np.int64, copy=False
                        )
                        for index, radius in enumerate(truncated_radii):
                            column = columns_by_radius[radius]
                            cumulative = np.cumsum(np.sort(column)[::-1])
                            truncated_counts[index] = max(
                                truncated_counts[index],
                                int(
                                    np.searchsorted(
                                        cumulative, targets[index], side="left"
                                    )
                                    + 1
                                ),
                            )
                            mass_by_radius = np.bincount(
                                anchor_distances, weights=column
                            )
                            truncated_ball_radii[index] = max(
                                truncated_ball_radii[index],
                                int(
                                    np.searchsorted(
                                        np.cumsum(mass_by_radius),
                                        targets[index],
                                        side="left",
                                    )
                                ),
                            )
                            truncated_support_sizes[index] = max(
                                truncated_support_sizes[index],
                                int(np.count_nonzero(column)),
                            )
                            truncated_max_l1[index] = max(
                                truncated_max_l1[index],
                                float(np.abs(exact[:, anchor] - column).sum()),
                            )
                    truncated_analysis_seconds = perf_counter() - start

                    for index, alpha in enumerate(alphas):
                        radius = truncated_radii[index]
                        beta_r = poisson_tail(theta, radius)
                        counter = OperationCounters()
                        truncated_time = _median_seconds(
                            lambda: local_truncated_heat_column(
                                graph,
                                diffusion_time,
                                radius,
                                representative_anchor,
                                nu_u=nu_u,
                            ),
                            repeats,
                        )
                        representative_column = local_truncated_heat_column(
                            graph,
                            diffusion_time,
                            radius,
                            representative_anchor,
                            nu_u=nu_u,
                            counter=counter,
                        )
                        base = _base_row(
                            resolved=resolved,
                            resolved_json=resolved_json,
                            metadata_json=metadata_json,
                            git_commit=str(metadata["git_commit"]),
                            seed=seed,
                            method="truncated_heat",
                            family=family,
                            requested_K=requested_K,
                            graph=graph,
                            rows=graph_rows,
                            columns=graph_columns,
                            degree=graph_degree,
                            alpha=alpha,
                            theta=theta,
                            diffusion_time=diffusion_time,
                            radius=radius,
                        )
                        base.update(
                            {
                                "kernel_effective_count": int(
                                    truncated_counts[index]
                                ),
                                "kernel_effective_fraction": float(
                                    truncated_counts[index]
                                )
                                / graph.K,
                                "high_mass_ball_radius": int(
                                    truncated_ball_radii[index]
                                ),
                                "exact_support_size": int(
                                    truncated_support_sizes[index]
                                ),
                                "poisson_tail": beta_r,
                                "certificate_l1_bound": 2.0 * beta_r,
                                "certificate_holds": truncated_max_l1[index]
                                <= 2.0 * beta_r + 2e-12,
                                "heat_approximation_error": truncated_max_l1[index],
                                "vertices_accessed": counter.unique_actions_touched,
                                "neighbor_entries_accessed": counter.graph_neighbor_accesses,
                                "graph_neighbor_accesses": counter.graph_neighbor_accesses,
                                "one_column_seconds": truncated_time,
                                "cached_one_column_seconds": np.nan,
                                "analysis_seconds": truncated_analysis_seconds,
                                "geometry_preprocess_seconds": 0.0,
                                "one_column_memory_bytes": int(
                                    np.count_nonzero(representative_column) * 16
                                ),
                                "full_matrix_memory_bytes": int(
                                    truncated_support_sizes[index] * graph.K * 16
                                ),
                                "peak_memory_mb": process.memory_info().rss / (1024.0**2),
                            }
                        )
                        combo_rows.append(base)
                    del exact
                except Exception as error:  # failures must remain visible in raw data
                    LOGGER.exception(
                        "kernel-map failure family=%s K=%d theta=%g",
                        family,
                        graph.K,
                        theta,
                    )
                    combo_rows = []
                    for method in METHOD_TARGETS:
                        for alpha in alphas:
                            failed = _base_row(
                                resolved=resolved,
                                resolved_json=resolved_json,
                                metadata_json=metadata_json,
                                git_commit=str(metadata["git_commit"]),
                                seed=seed,
                                method=method,
                                family=family,
                                requested_K=requested_K,
                                graph=graph,
                                rows=graph_rows,
                                columns=graph_columns,
                                degree=graph_degree,
                                alpha=alpha,
                                theta=theta,
                                diffusion_time=diffusion_time,
                                radius=np.nan,
                                status="failed",
                            )
                            failed["error_message"] = f"{type(error).__name__}: {error}"
                            combo_rows.append(failed)
                combo_ids = {row["run_id"] for row in combo_rows}
                all_rows = [row for row in all_rows if row["run_id"] not in combo_ids]
                all_rows.extend(combo_rows)
                write_results_atomic(all_rows, output_path)
                completed_ids.update(
                    row["run_id"] for row in combo_rows if row["status"] == "complete"
                )
                LOGGER.info(
                    "wrote family=%s K=%d theta=%g status=%s",
                    family,
                    graph.K,
                    theta,
                    sorted({row["status"] for row in combo_rows}),
                )
    return validate_results(pd.DataFrame(all_rows))


def summarize_kernel_map(
    raw: pd.DataFrame, *, resolved_config: Mapping[str, Any] | None = None
) -> pd.DataFrame:
    frame = validate_results(raw)
    frame = frame.loc[
        (frame["experiment"] == "kernel_map") & (frame["status"] == "complete")
    ].copy()
    if resolved_config is not None:
        expected_json = config_json(resolve_kernel_map_config(resolved_config))
        frame = frame.loc[frame["config_json"] == expected_json]
    columns = [
        "run_id",
        "seed",
        "method",
        "target",
        "graph_family",
        "requested_K",
        "K",
        "alpha",
        "radius",
        "diffusion_time",
        "poisson_mean",
        *KERNEL_MAP_COLUMNS[4:-2],
        "config_json",
    ]
    columns = list(dict.fromkeys(columns))
    summary = frame.loc[:, columns].sort_values(
        ["graph_family", "K", "poisson_mean", "alpha", "method"]
    )
    return summary.reset_index(drop=True)


def write_summary_atomic(summary: pd.DataFrame, destination: str | Path) -> Path:
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
