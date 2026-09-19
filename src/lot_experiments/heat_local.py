"""Lazy graph-local construction of normalized finite-walk heat columns."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from lot_experiments.counters import OperationCounters
from lot_experiments.graphs import ActionGraph
from lot_experiments.kernels import poisson_head_weights


FloatArray = NDArray[np.float64]


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

    if diffusion_time < 0.0 or radius < 0:
        raise ValueError("diffusion_time and radius must be nonnegative")
    if not 0 <= anchor < graph.K:
        raise IndexError(anchor)
    degrees = graph.weighted_degree
    max_degree = float(degrees.max(initial=0.0))
    resolved_rate = max_degree if nu_u is None and max_degree > 0.0 else (1.0 if nu_u is None else float(nu_u))
    if resolved_rate <= 0.0 or resolved_rate + 1e-12 < max_degree:
        raise ValueError("nu_u must be positive and at least the maximum weighted degree")
    coefficients = poisson_head_weights(resolved_rate * diffusion_time, radius)
    current: dict[int, float] = {anchor: 1.0}
    accumulated: dict[int, float] = {anchor: float(coefficients[0])}
    if counter is not None:
        counter.touch_action(anchor, evaluations=0)

    for order in range(1, radius + 1):
        next_values: dict[int, float] = {}
        for source, mass in current.items():
            self_probability = 1.0 - degrees[source] / resolved_rate
            if self_probability > 0.0:
                next_values[source] = next_values.get(source, 0.0) + mass * self_probability
            neighbors, weights = graph.neighbors(source, counter)
            for destination, edge_weight in zip(neighbors, weights, strict=True):
                destination_int = int(destination)
                contribution = mass * float(edge_weight) / resolved_rate
                next_values[destination_int] = next_values.get(destination_int, 0.0) + contribution
                if counter is not None:
                    counter.touch_action(destination_int, evaluations=0)
        coefficient = float(coefficients[order])
        for destination, mass in next_values.items():
            accumulated[destination] = accumulated.get(destination, 0.0) + coefficient * mass
        current = next_values

    result = np.zeros(graph.K, dtype=np.float64)
    for action, mass in accumulated.items():
        result[action] = mass
    result[result < 0.0] = 0.0
    total = float(result.sum())
    if total <= 0.0:
        raise FloatingPointError("local heat construction returned zero mass")
    return result / total


@dataclass
class LazyTruncatedHeat:
    """Per-column cache exposing cold-start and amortized cache statistics."""

    graph: ActionGraph
    diffusion_time: float
    radius: int
    nu_u: float | None = None
    counter: OperationCounters = field(default_factory=OperationCounters)
    _cache: dict[int, FloatArray] = field(default_factory=dict, init=False, repr=False)
    cache_hits: int = 0
    cache_misses: int = 0

    def column(self, anchor: int) -> FloatArray:
        if anchor in self._cache:
            self.cache_hits += 1
            return self._cache[anchor].copy()
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
        self._cache[anchor] = column
        return column.copy()

    @property
    def cached_columns(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

