"""Lazy graph-local construction of normalized finite-walk heat columns."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from lot_experiments.counters import OperationCounters
from lot_experiments.graphs import ActionGraph
from lot_experiments.kernels import poisson_head_weights


FloatArray = NDArray[np.float64]


def _validate_local_request(
    graph: ActionGraph,
    diffusion_time: float,
    anchor: int,
    nu_u: float | None,
) -> tuple[FloatArray, float]:
    if diffusion_time < 0.0 or not math.isfinite(diffusion_time):
        raise ValueError("diffusion_time must be finite and nonnegative")
    if not 0 <= anchor < graph.K:
        raise IndexError(anchor)
    degrees = graph.weighted_degree
    max_degree = float(degrees.max(initial=0.0))
    resolved_rate = (
        max_degree
        if nu_u is None and max_degree > 0.0
        else (1.0 if nu_u is None else float(nu_u))
    )
    if (
        resolved_rate <= 0.0
        or not math.isfinite(resolved_rate)
        or resolved_rate + 1e-12 < max_degree
    ):
        raise ValueError(
            "nu_u must be finite, positive, and at least the maximum weighted degree"
        )
    return degrees, resolved_rate


def _local_walk_powers(
    graph: ActionGraph,
    radius: int,
    anchor: int,
    degrees: FloatArray,
    resolved_rate: float,
    counter: OperationCounters | None,
) -> tuple[NDArray[np.int64], list[FloatArray]]:
    distances = {anchor: 0}
    frontier = [anchor]
    adjacency_cache: dict[int, tuple[NDArray[np.int32], FloatArray]] = {}
    if counter is not None:
        counter.touch_action(anchor, evaluations=0)
    for depth in range(radius):
        next_frontier: list[int] = []
        for source in frontier:
            neighbors, weights = graph.neighbors(source, counter)
            adjacency_cache[source] = (neighbors, weights)
            for destination in neighbors:
                destination_int = int(destination)
                if destination_int not in distances:
                    distances[destination_int] = depth + 1
                    next_frontier.append(destination_int)
                    if counter is not None:
                        counter.touch_action(destination_int, evaluations=0)
        frontier = next_frontier
        if not frontier:
            # The ball has saturated; cache remaining vertices once because
            # later walk powers may revisit them.
            for source in distances:
                if source not in adjacency_cache:
                    adjacency_cache[source] = graph.neighbors(source, counter)
            break

    nodes = np.fromiter(distances, dtype=np.int64)
    local_index = np.full(graph.K, -1, dtype=np.int64)
    local_index[nodes] = np.arange(len(nodes))
    walk_rows: list[int] = []
    walk_columns: list[int] = []
    walk_values: list[float] = []
    for source, (neighbors, weights) in adjacency_cache.items():
        source_local = int(local_index[source])
        self_probability = 1.0 - degrees[source] / resolved_rate
        if self_probability > 0.0:
            walk_rows.append(source_local)
            walk_columns.append(source_local)
            walk_values.append(float(self_probability))
        for destination, edge_weight in zip(neighbors, weights, strict=True):
            destination_local = int(local_index[int(destination)])
            if destination_local >= 0:
                walk_rows.append(destination_local)
                walk_columns.append(source_local)
                walk_values.append(float(edge_weight) / resolved_rate)
    local_walk = sparse.csc_matrix(
        (walk_values, (walk_rows, walk_columns)), shape=(len(nodes), len(nodes))
    )
    current = np.zeros(len(nodes), dtype=np.float64)
    current[int(local_index[anchor])] = 1.0
    powers = [current]
    for order in range(1, radius + 1):
        current = np.asarray(local_walk @ current).ravel()
        powers.append(current)
    return nodes, powers


def local_truncated_heat_columns(
    graph: ActionGraph,
    diffusion_time: float,
    radii: list[int] | tuple[int, ...],
    anchor: int,
    *,
    nu_u: float | None = None,
    counter: OperationCounters | None = None,
) -> dict[int, FloatArray]:
    """Build several normalized finite-walk columns from one local traversal."""

    if not radii:
        raise ValueError("radii must be nonempty")
    unique_radii = sorted(set(radii))
    if any(
        not isinstance(radius, (int, np.integer)) or radius < 0
        for radius in unique_radii
    ):
        raise ValueError("every radius must be a nonnegative integer")
    degrees, resolved_rate = _validate_local_request(
        graph, diffusion_time, anchor, nu_u
    )
    nodes, powers = _local_walk_powers(
        graph,
        int(unique_radii[-1]),
        anchor,
        degrees,
        resolved_rate,
        counter,
    )
    columns: dict[int, FloatArray] = {}
    theta = resolved_rate * diffusion_time
    for radius in unique_radii:
        coefficients = poisson_head_weights(theta, int(radius))
        accumulated = np.zeros(len(nodes), dtype=np.float64)
        for order in range(int(radius) + 1):
            accumulated += coefficients[order] * powers[order]
        result = np.zeros(graph.K, dtype=np.float64)
        result[nodes] = accumulated
        result[result < 0.0] = 0.0
        total = float(result.sum())
        if total <= 0.0:
            raise FloatingPointError("local heat construction returned zero mass")
        columns[int(radius)] = result / total
    return columns


def local_truncated_heat_column(
    graph: ActionGraph,
    diffusion_time: float,
    radius: int,
    anchor: int,
    *,
    nu_u: float | None = None,
    counter: OperationCounters | None = None,
) -> FloatArray:
    """Build one column from adjacency access, never a dense heat matrix."""

    return local_truncated_heat_columns(
        graph,
        diffusion_time,
        [radius],
        anchor,
        nu_u=nu_u,
        counter=counter,
    )[int(radius)]


@dataclass
class LazyTruncatedHeat:
    """Sparse per-column cache exposing cold-start and amortized statistics.

    Cached columns store only positive support entries.  ``column`` reconstructs
    a dense vector for compatibility; local planners should use
    :meth:`sparse_column` to avoid accumulating a hidden ``K x K`` allocation.
    """

    graph: ActionGraph
    diffusion_time: float
    radius: int
    nu_u: float | None = None
    counter: OperationCounters = field(default_factory=OperationCounters)
    _cache: dict[int, tuple[NDArray[np.int64], FloatArray]] = field(
        default_factory=dict, init=False, repr=False
    )
    cache_hits: int = 0
    cache_misses: int = 0

    def column(self, anchor: int) -> FloatArray:
        support, probabilities = self.sparse_column(anchor)
        column = np.zeros(self.graph.K, dtype=np.float64)
        column[support] = probabilities
        return column

    def sparse_column(self, anchor: int) -> tuple[NDArray[np.int64], FloatArray]:
        """Return positive indices and weights without a dense column."""

        if anchor in self._cache:
            self.cache_hits += 1
            support, probabilities = self._cache[anchor]
            return support.copy(), probabilities.copy()
        self.cache_misses += 1
        with self.counter.time_geometry():
            column = local_truncated_heat_column(
                self.graph,
                self.diffusion_time,
                self.radius,
                anchor,
                nu_u=self.nu_u,
                counter=self.counter,
            )
        support = np.flatnonzero(column > 0.0).astype(np.int64, copy=False)
        probabilities = column[support].copy()
        self._cache[anchor] = (support.copy(), probabilities.copy())
        return support, probabilities

    @property
    def cached_columns(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
