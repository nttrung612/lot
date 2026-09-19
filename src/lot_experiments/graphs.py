"""Sparse undirected action-graph constructors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse import csgraph

from lot_experiments.counters import OperationCounters


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ActionGraph:
    """An undirected weighted action graph with adjacency-list access."""

    adjacency: sparse.csr_matrix
    family: str

    def __post_init__(self) -> None:
        adjacency = sparse.csr_matrix(self.adjacency, dtype=np.float64)
        adjacency.sum_duplicates()
        adjacency.eliminate_zeros()
        adjacency.sort_indices()
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency must be square")
        if adjacency.nnz and np.min(adjacency.data) < 0:
            raise ValueError("adjacency weights must be nonnegative")
        if np.any(adjacency.diagonal() != 0):
            raise ValueError("action graphs must not contain explicit self-edges")
        difference = adjacency - adjacency.T
        if difference.nnz and np.max(np.abs(difference.data)) > 1e-12:
            raise ValueError("adjacency must be symmetric")
        object.__setattr__(self, "adjacency", adjacency)

    @property
    def K(self) -> int:
        return self.adjacency.shape[0]

    @property
    def weighted_degree(self) -> FloatArray:
        return np.asarray(self.adjacency.sum(axis=0)).ravel()

    @property
    def max_weighted_degree(self) -> float:
        return float(self.weighted_degree.max(initial=0.0))

    @property
    def laplacian(self) -> sparse.csr_matrix:
        return sparse.csr_matrix(csgraph.laplacian(self.adjacency, normed=False))

    def neighbors(
        self, vertex: int, counter: OperationCounters | None = None
    ) -> tuple[NDArray[np.int32], FloatArray]:
        if not 0 <= vertex < self.K:
            raise IndexError(vertex)
        start, stop = self.adjacency.indptr[vertex : vertex + 2]
        indices = self.adjacency.indices[start:stop]
        weights = self.adjacency.data[start:stop]
        if counter is not None:
            counter.graph_neighbor_accesses += len(indices)
        return indices, weights

    def ball(
        self, anchor: int, radius: int, counter: OperationCounters | None = None
    ) -> set[int]:
        if radius < 0:
            raise ValueError("radius must be nonnegative")
        visited = {int(anchor)}
        frontier = {int(anchor)}
        for _ in range(radius):
            next_frontier: set[int] = set()
            for vertex in frontier:
                indices, _ = self.neighbors(vertex, counter)
                next_frontier.update(map(int, indices))
            next_frontier.difference_update(visited)
            visited.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
        return visited


def _graph_from_edges(
    K: int, edges: set[tuple[int, int]], family: str, edge_weight: float
) -> ActionGraph:
    if K < 1:
        raise ValueError("K must be positive")
    if edge_weight <= 0:
        raise ValueError("edge_weight must be positive")
    rows: list[int] = []
    cols: list[int] = []
    for first, second in sorted(edges):
        if first == second:
            continue
        rows.extend((first, second))
        cols.extend((second, first))
    data = np.full(len(rows), edge_weight, dtype=np.float64)
    adjacency = sparse.csr_matrix((data, (rows, cols)), shape=(K, K))
    return ActionGraph(adjacency=adjacency, family=family)


def path_graph(K: int, *, edge_weight: float = 1.0) -> ActionGraph:
    edges = {(vertex, vertex + 1) for vertex in range(K - 1)}
    return _graph_from_edges(K, edges, "path", edge_weight)


def cycle_graph(K: int, *, edge_weight: float = 1.0) -> ActionGraph:
    if K < 3:
        raise ValueError("cycle graphs require K >= 3")
    edges = {(min(vertex, (vertex + 1) % K), max(vertex, (vertex + 1) % K)) for vertex in range(K)}
    return _graph_from_edges(K, edges, "cycle", edge_weight)


def grid_graph(
    rows: int,
    columns: int,
    *,
    periodic: bool = False,
    edge_weight: float = 1.0,
) -> ActionGraph:
    if rows < 1 or columns < 1:
        raise ValueError("grid dimensions must be positive")
    edges: set[tuple[int, int]] = set()

    def vertex(row: int, column: int) -> int:
        return row * columns + column

    for row in range(rows):
        for column in range(columns):
            candidates = []
            if row + 1 < rows:
                candidates.append((row + 1, column))
            elif periodic and rows > 1:
                candidates.append((0, column))
            if column + 1 < columns:
                candidates.append((row, column + 1))
            elif periodic and columns > 1:
                candidates.append((row, 0))
            for next_row, next_column in candidates:
                endpoints = sorted((vertex(row, column), vertex(next_row, next_column)))
                edges.add((endpoints[0], endpoints[1]))
    family = "torus" if periodic else "grid"
    return _graph_from_edges(rows * columns, edges, family, edge_weight)


def torus_graph(rows: int, columns: int, *, edge_weight: float = 1.0) -> ActionGraph:
    return grid_graph(rows, columns, periodic=True, edge_weight=edge_weight)


def random_regular_expander(
    K: int,
    degree: int = 4,
    *,
    seed: int = 0,
    max_attempts: int = 10_000,
) -> ActionGraph:
    """Sample a simple regular graph with the configuration model.

    No expansion constant is asserted; this is the standard random-regular
    family used as the experiment's expander control.
    """

    if K < 1 or degree < 0 or degree >= K or (K * degree) % 2:
        raise ValueError("require 0 <= degree < K and K * degree even")
    if degree == 0:
        return _graph_from_edges(K, set(), "random_regular_expander", 1.0)
    rng = np.random.default_rng(seed)
    stubs = np.repeat(np.arange(K, dtype=np.int64), degree)
    for _ in range(max_attempts):
        rng.shuffle(stubs)
        edges: set[tuple[int, int]] = set()
        valid = True
        for index in range(0, len(stubs), 2):
            first, second = sorted((int(stubs[index]), int(stubs[index + 1])))
            edge = (first, second)
            if first == second or edge in edges:
                valid = False
                break
            edges.add(edge)
        if valid:
            graph = _graph_from_edges(K, edges, "random_regular_expander", 1.0)
            if degree >= 2 and csgraph.connected_components(graph.adjacency)[0] != 1:
                continue
            return graph
    raise RuntimeError(f"failed to sample a simple {degree}-regular graph in {max_attempts} attempts")


def graph_from_config(config: dict[str, object]) -> ActionGraph:
    family = str(config["family"])
    edge_weight = float(config.get("edge_weight", 1.0))
    if family == "path":
        return path_graph(int(config["K"]), edge_weight=edge_weight)
    if family == "cycle":
        return cycle_graph(int(config["K"]), edge_weight=edge_weight)
    if family in {"grid", "torus"}:
        return grid_graph(
            int(config["rows"]),
            int(config["columns"]),
            periodic=family == "torus",
            edge_weight=edge_weight,
        )
    if family == "random_regular_expander":
        return random_regular_expander(
            int(config["K"]), int(config.get("degree", 4)), seed=int(config.get("seed", 0))
        )
    raise ValueError(f"unknown graph family: {family}")
