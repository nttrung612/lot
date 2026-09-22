"""M9 effective-resistance identity for the LOT policy regularizer.

The input action graph is used only to construct a finite cost matrix.  The
curvature graph is the distinct, value-dependent co-occurrence graph induced
by the optimal LOT coupling.
"""

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
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.linalg import pinvh
from scipy.sparse import csgraph
from scipy.special import logsumexp

from lot_experiments.config import config_json
from lot_experiments.graphs import ActionGraph
from lot_experiments.reproducibility import run_metadata
from lot_experiments.results import result_row, validate_results, write_results_atomic


FloatArray = NDArray[np.float64]
TARGET = "lot_policy_curvature"

RESISTANCE_COLUMNS = (
    "record_type",
    "reference_target",
    "q_name",
    "q_index",
    "action_i",
    "action_k",
    "pair_name",
    "pair_type",
    "q_value",
    "policy_mass",
    "input_edge_weight",
    "input_cost",
    "cooccurrence_weight",
    "effective_resistance",
    "step_size",
    "omega_minus",
    "omega_base",
    "omega_plus",
    "finite_difference_curvature",
    "theoretical_curvature",
    "absolute_curvature_error",
    "relative_curvature_error",
    "coupling_row_error",
    "coupling_column_error",
    "laplacian_rank",
    "identity_holds",
    "metadata_json",
    "error_message",
)


@dataclass(frozen=True)
class CouplingSolution:
    """A balanced entropic-OT solution and its numerical diagnostics."""

    coupling: FloatArray
    regularizer_value: float
    iterations: int
    row_error: float
    column_error: float


@dataclass(frozen=True)
class CooccurrenceGeometry:
    adjacency: FloatArray
    laplacian: FloatArray
    laplacian_pseudoinverse: FloatArray
    resistance: FloatArray
    rank: int


def _deep_defaults(destination: dict[str, Any], defaults: Mapping[str, Any]) -> None:
    for key, value in defaults.items():
        if key not in destination:
            destination[key] = copy.deepcopy(value)
        elif isinstance(value, Mapping) and isinstance(destination[key], dict):
            _deep_defaults(destination[key], value)


def _probability_vector(values: ArrayLike, name: str) -> FloatArray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)) or np.any(vector <= 0.0):
        raise ValueError(f"{name} must be a finite, strictly positive vector")
    total = float(vector.sum())
    if total <= 0.0:
        raise ValueError(f"{name} must have positive mass")
    return vector / total


def two_cluster_weak_bridge_graph(
    cluster_size: int, *, within_weight: float, bridge_weight: float
) -> ActionGraph:
    """Return two cliques connected by one weak edge between their endpoints."""

    if isinstance(cluster_size, bool) or int(cluster_size) < 2:
        raise ValueError("cluster_size must be an integer at least two")
    if not math.isfinite(within_weight) or within_weight <= 0.0:
        raise ValueError("within_weight must be finite and positive")
    if not math.isfinite(bridge_weight) or bridge_weight <= 0.0:
        raise ValueError("bridge_weight must be finite and positive")
    size = int(cluster_size)
    K = 2 * size
    adjacency = np.zeros((K, K), dtype=np.float64)
    for start in (0, size):
        block = slice(start, start + size)
        adjacency[block, block] = within_weight
        adjacency[np.arange(start, start + size), np.arange(start, start + size)] = 0.0
    adjacency[size - 1, size] = bridge_weight
    adjacency[size, size - 1] = bridge_weight
    return ActionGraph(sparse.csr_matrix(adjacency), "two_cluster_weak_bridge")


def effective_resistance_matrix(laplacian: ArrayLike, *, rtol: float = 1e-12) -> FloatArray:
    """Compute all pairwise effective resistances of a connected Laplacian."""

    matrix = np.asarray(laplacian, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("laplacian must be square")
    if not np.all(np.isfinite(matrix)) or not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError("laplacian must be finite and symmetric")
    pseudoinverse = pinvh(matrix, rtol=rtol, check_finite=True)
    diagonal = np.diag(pseudoinverse)
    resistance = diagonal[:, None] + diagonal[None, :] - 2.0 * pseudoinverse
    resistance = 0.5 * (resistance + resistance.T)
    resistance[np.abs(resistance) < 1e-13] = 0.0
    if np.min(resistance, initial=0.0) < -1e-10:
        raise FloatingPointError("effective resistance is materially negative")
    resistance = np.maximum(resistance, 0.0)
    np.fill_diagonal(resistance, 0.0)
    return resistance


def graph_cost_matrix(
    graph: ActionGraph, *, kind: str = "input_effective_resistance", scale: float = 1.0
) -> FloatArray:
    """Construct the finite graph-derived action cost used by M9."""

    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("cost scale must be finite and positive")
    if kind == "input_effective_resistance":
        if csgraph.connected_components(graph.adjacency, directed=False)[0] != 1:
            raise ValueError("effective-resistance cost requires a connected input graph")
        cost = effective_resistance_matrix(graph.laplacian.toarray())
    elif kind == "shortest_path":
        lengths = graph.adjacency.copy()
        lengths.data = 1.0 / lengths.data
        cost = np.asarray(csgraph.shortest_path(lengths, directed=False), dtype=np.float64)
    else:
        raise ValueError(f"unknown cost kind: {kind}")
    cost *= scale
    if not np.all(np.isfinite(cost)):
        raise ValueError("graph-derived costs must be finite")
    np.fill_diagonal(cost, 0.0)
    return cost


def lot_optimal_coupling(
    action_values: ArrayLike,
    costs: ArrayLike,
    candidate_prior: ArrayLike,
    anchor_distribution: ArrayLike,
    tau: float,
    lambda_: float,
) -> FloatArray:
    """Closed-form semi-relaxed LOT coupling ``Gamma_Q``.

    Columns have prescribed masses ``mu[j]`` and rows sum to the LOT policy.
    The implementation is log-domain stable and assumes the paper's
    kernel-relative entropic transport penalty.
    """

    q = np.asarray(action_values, dtype=np.float64)
    cost = np.asarray(costs, dtype=np.float64)
    rho = _probability_vector(candidate_prior, "candidate_prior")
    mu = _probability_vector(anchor_distribution, "anchor_distribution")
    if q.ndim != 1 or not np.all(np.isfinite(q)):
        raise ValueError("action_values must be a finite vector")
    if cost.shape != (len(q), len(mu)) or not np.all(np.isfinite(cost)):
        raise ValueError("costs must have shape (candidate actions, anchors)")
    if len(rho) != len(q):
        raise ValueError("candidate_prior must match action_values")
    if np.min(cost, initial=0.0) < 0.0:
        raise ValueError("costs must be nonnegative")
    if tau <= 0.0 or lambda_ <= 0.0 or not np.isfinite([tau, lambda_]).all():
        raise ValueError("tau and lambda_ must be finite and positive")
    T0 = tau * lambda_
    log_scores = np.log(rho)[:, None] + (q[:, None] - tau * cost) / T0
    anchor_policies = np.exp(log_scores - logsumexp(log_scores, axis=0, keepdims=True))
    return anchor_policies * mu[None, :]


def cooccurrence_geometry(
    coupling: ArrayLike, anchor_distribution: ArrayLike, *, rtol: float = 1e-12
) -> CooccurrenceGeometry:
    """Build ``A_Q``, ``L_Q``, its pseudoinverse, and all resistances."""

    gamma = np.asarray(coupling, dtype=np.float64)
    mu = _probability_vector(anchor_distribution, "anchor_distribution")
    if gamma.ndim != 2 or gamma.shape[1] != len(mu):
        raise ValueError("coupling columns must match anchor_distribution")
    if not np.all(np.isfinite(gamma)) or np.any(gamma < 0.0):
        raise ValueError("coupling must be finite and nonnegative")
    if not np.allclose(gamma.sum(axis=0), mu, atol=1e-11, rtol=1e-11):
        raise ValueError("coupling must have anchor_distribution as column marginal")
    adjacency = (gamma / mu[None, :]) @ gamma.T
    adjacency = 0.5 * (adjacency + adjacency.T)
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    eigenvalues = np.linalg.eigvalsh(laplacian)
    cutoff = rtol * max(np.finfo(np.float64).tiny, float(eigenvalues[-1]))
    rank = int(np.count_nonzero(eigenvalues > cutoff))
    pseudoinverse = pinvh(laplacian, rtol=rtol, check_finite=True)
    diagonal = np.diag(pseudoinverse)
    resistance = diagonal[:, None] + diagonal[None, :] - 2.0 * pseudoinverse
    resistance = np.maximum(0.5 * (resistance + resistance.T), 0.0)
    np.fill_diagonal(resistance, 0.0)
    return CooccurrenceGeometry(
        adjacency=adjacency,
        laplacian=laplacian,
        laplacian_pseudoinverse=pseudoinverse,
        resistance=resistance,
        rank=rank,
    )


def regularizer_value(
    policy: ArrayLike,
    costs: ArrayLike,
    candidate_prior: ArrayLike,
    anchor_distribution: ArrayLike,
    tau: float,
    lambda_: float,
    *,
    tolerance: float = 3e-15,
    max_iterations: int = 100_000,
) -> CouplingSolution:
    """Evaluate ``Omega(pi)`` by log-domain balanced Sinkhorn scaling.

    Unlike the semi-relaxed backup, both row and column marginals are fixed
    here.  This distinction is essential for the policy-space finite
    difference in Theorem 5.
    """

    pi = _probability_vector(policy, "policy")
    rho = _probability_vector(candidate_prior, "candidate_prior")
    mu = _probability_vector(anchor_distribution, "anchor_distribution")
    cost = np.asarray(costs, dtype=np.float64)
    if cost.shape != (len(pi), len(mu)) or len(rho) != len(pi):
        raise ValueError("cost and prior dimensions do not match policy/anchors")
    if not np.all(np.isfinite(cost)) or np.min(cost, initial=0.0) < 0.0:
        raise ValueError("costs must be finite and nonnegative")
    if tau <= 0.0 or lambda_ <= 0.0:
        raise ValueError("tau and lambda_ must be positive")
    if tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("invalid Sinkhorn stopping rule")

    log_reference = (
        np.log(rho)[:, None] + np.log(mu)[None, :] - cost / float(lambda_)
    )
    log_u = np.zeros(len(pi), dtype=np.float64)
    log_v = np.zeros(len(mu), dtype=np.float64)
    row_error = math.inf
    column_error = math.inf
    coupling = np.empty_like(cost)
    for iteration in range(1, int(max_iterations) + 1):
        log_u = np.log(pi) - logsumexp(log_reference + log_v[None, :], axis=1)
        log_v = np.log(mu) - logsumexp(log_reference + log_u[:, None], axis=0)
        # Fix the irrelevant scaling gauge to avoid a long-run drift.
        shift = float(np.mean(log_u))
        log_u -= shift
        log_v += shift
        if iteration == 1 or iteration % 5 == 0:
            log_coupling = log_reference + log_u[:, None] + log_v[None, :]
            coupling = np.exp(log_coupling)
            row_error = float(np.max(np.abs(coupling.sum(axis=1) - pi)))
            column_error = float(np.max(np.abs(coupling.sum(axis=0) - mu)))
            if max(row_error, column_error) <= tolerance:
                break
    else:
        raise RuntimeError(
            "balanced Sinkhorn scaling did not converge: "
            f"row_error={row_error:.3e}, column_error={column_error:.3e}"
        )

    log_coupling = log_reference + log_u[:, None] + log_v[None, :]
    coupling = np.exp(log_coupling)
    log_ratio = log_coupling - np.log(rho)[:, None] - np.log(mu)[None, :]
    omega = float(tau * (np.sum(coupling * cost) + lambda_ * np.sum(coupling * log_ratio)))
    return CouplingSolution(coupling, omega, iteration, row_error, column_error)


def finite_difference_curvature(
    policy: ArrayLike,
    action_i: int,
    action_k: int,
    step_size: float,
    costs: ArrayLike,
    candidate_prior: ArrayLike,
    anchor_distribution: ArrayLike,
    tau: float,
    lambda_: float,
    *,
    tolerance: float = 3e-15,
    max_iterations: int = 100_000,
) -> tuple[float, CouplingSolution, CouplingSolution, CouplingSolution]:
    """Central policy-space second difference along ``e_i - e_k``."""

    pi = _probability_vector(policy, "policy")
    if action_i == action_k or not 0 <= action_i < len(pi) or not 0 <= action_k < len(pi):
        raise ValueError("finite-difference actions must be distinct valid indices")
    if not math.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("step_size must be finite and positive")
    direction = np.zeros(len(pi), dtype=np.float64)
    direction[action_i] = 1.0
    direction[action_k] = -1.0
    plus_policy = pi + step_size * direction
    minus_policy = pi - step_size * direction
    if np.any(plus_policy <= 0.0) or np.any(minus_policy <= 0.0):
        raise ValueError("step_size leaves the simplex interior")
    kwargs = {"tolerance": tolerance, "max_iterations": max_iterations}
    base = regularizer_value(pi, costs, candidate_prior, anchor_distribution, tau, lambda_, **kwargs)
    plus = regularizer_value(
        plus_policy, costs, candidate_prior, anchor_distribution, tau, lambda_, **kwargs
    )
    minus = regularizer_value(
        minus_policy, costs, candidate_prior, anchor_distribution, tau, lambda_, **kwargs
    )
    curvature = (plus.regularizer_value - 2.0 * base.regularizer_value + minus.regularizer_value) / (
        step_size**2
    )
    return float(curvature), minus, base, plus


def resolve_resistance_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve and validate the complete deterministic M9 configuration."""

    resolved = copy.deepcopy(dict(config))
    # These mappings are experiment grids, not nested option blocks: an
    # explicit user mapping replaces the default grid in full.
    supplied_q_vectors = copy.deepcopy(resolved.get("q_vectors"))
    supplied_pairs = copy.deepcopy(resolved.get("pairs"))
    _deep_defaults(
        resolved,
        {
            "experiment": "resistance",
            "seed": 0,
            "graph": {
                "family": "two_cluster_weak_bridge",
                "cluster_size": 4,
                "within_weight": 1.0,
                "bridge_weight": 0.08,
            },
            "cost": {"kind": "input_effective_resistance", "scale": 0.2},
            "regularization": {"tau": 0.1, "lambda_": 1.0},
            "candidate_prior": "uniform",
            "anchor_distribution": "uniform",
            "q_vectors": {
                "balanced": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "left_favored": [0.08, 0.04, 0.02, 0.0, -0.02, -0.04, -0.06, -0.08],
                "bridge_favored": [0.0, 0.0, 0.02, 0.08, 0.08, 0.02, 0.0, 0.0],
            },
            "pairs": {
                "within_cluster": {"actions": [0, 1], "type": "within_cluster"},
                "bridge_endpoints": {"actions": [3, 4], "type": "bridge"},
                "cross_cluster": {"actions": [0, 7], "type": "cross_cluster"},
            },
            "finite_difference_steps": [0.01, 0.003, 0.001, 0.0003, 0.0001],
            "sinkhorn": {"tolerance": 3e-15, "max_iterations": 100000},
            "numerical_tolerance": 1e-6,
            "visualization_q": "bridge_favored",
            "raw_output": "outputs/raw/resistance.parquet",
            "summary_output": "outputs/summaries/resistance.csv",
            "figure_png": "outputs/figures/effective_resistance.png",
            "figure_pdf": "outputs/figures/effective_resistance.pdf",
        },
    )
    if supplied_q_vectors is not None:
        resolved["q_vectors"] = supplied_q_vectors
    if supplied_pairs is not None:
        resolved["pairs"] = supplied_pairs
    if resolved["experiment"] != "resistance":
        raise ValueError("experiment must be 'resistance'")
    graph_config = resolved["graph"]
    if graph_config["family"] != "two_cluster_weak_bridge":
        raise ValueError("M9 requires graph.family='two_cluster_weak_bridge'")
    graph = two_cluster_weak_bridge_graph(
        int(graph_config["cluster_size"]),
        within_weight=float(graph_config["within_weight"]),
        bridge_weight=float(graph_config["bridge_weight"]),
    )
    graph_cost_matrix(
        graph, kind=str(resolved["cost"]["kind"]), scale=float(resolved["cost"]["scale"])
    )
    tau = float(resolved["regularization"]["tau"])
    lambda_ = float(resolved["regularization"]["lambda_"])
    if tau <= 0.0 or lambda_ <= 0.0:
        raise ValueError("regularization tau and lambda_ must be positive")
    if resolved["candidate_prior"] != "uniform" or resolved["anchor_distribution"] != "uniform":
        raise ValueError("M9 currently supports explicit uniform full-support priors")
    q_vectors = resolved["q_vectors"]
    if not isinstance(q_vectors, Mapping) or not q_vectors:
        raise ValueError("q_vectors must be a nonempty mapping")
    for name, values in q_vectors.items():
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (graph.K,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"q_vectors.{name} must contain K finite values")
    if str(resolved["visualization_q"]) not in q_vectors:
        raise ValueError("visualization_q must name one configured Q vector")
    pairs = resolved["pairs"]
    if not isinstance(pairs, Mapping) or not pairs:
        raise ValueError("pairs must be a nonempty mapping")
    for name, specification in pairs.items():
        actions = list(specification["actions"])
        if len(actions) != 2 or actions[0] == actions[1] or any(
            not 0 <= int(action) < graph.K for action in actions
        ):
            raise ValueError(f"pairs.{name}.actions must be two distinct valid actions")
    steps = [float(value) for value in resolved["finite_difference_steps"]]
    if not steps or any(value <= 0.0 or not math.isfinite(value) for value in steps):
        raise ValueError("finite_difference_steps must be finite and positive")
    if len(set(steps)) != len(steps):
        raise ValueError("finite_difference_steps must be unique")
    if float(resolved["sinkhorn"]["tolerance"]) <= 0.0 or int(
        resolved["sinkhorn"]["max_iterations"]
    ) < 1:
        raise ValueError("invalid Sinkhorn configuration")
    if float(resolved["numerical_tolerance"]) <= 0.0:
        raise ValueError("numerical_tolerance must be positive")
    return resolved


def _stable_run_id(parts: Sequence[Any], resolved_json: str) -> str:
    payload = json.dumps([*parts, resolved_json], separators=(",", ":"), default=str)
    return "resistance-" + hashlib.blake2b(payload.encode(), digest_size=10).hexdigest()


def _base_row(
    *,
    resolved: Mapping[str, Any],
    resolved_json: str,
    metadata_json: str,
    run_id: str,
    method: str,
    record_type: str,
    q_name: str = "",
    status: str = "complete",
) -> dict[str, Any]:
    graph_config = resolved["graph"]
    row = result_row(
        experiment="resistance",
        run_id=run_id,
        seed=int(resolved["seed"]),
        method=method,
        target=TARGET,
        graph_family=str(graph_config["family"]),
        K=2 * int(graph_config["cluster_size"]),
        temperature=float(resolved["regularization"]["tau"])
        * float(resolved["regularization"]["lambda_"]),
        status=status,
        git_commit=str(json.loads(metadata_json)["git_commit"]),
        config_json=resolved_json,
    )
    row.update({column: np.nan for column in RESISTANCE_COLUMNS})
    row.update(
        {
            "record_type": record_type,
            "reference_target": TARGET,
            "q_name": q_name,
            "metadata_json": metadata_json,
            "error_message": "",
        }
    )
    return row


def _run_resistance_impl(
    resolved: Mapping[str, Any], *, repository: str | Path
) -> pd.DataFrame:
    resolved_json = config_json(resolved)
    metadata_json = json.dumps(run_metadata(resolved, repository), sort_keys=True)
    graph_config = resolved["graph"]
    graph = two_cluster_weak_bridge_graph(
        int(graph_config["cluster_size"]),
        within_weight=float(graph_config["within_weight"]),
        bridge_weight=float(graph_config["bridge_weight"]),
    )
    costs = graph_cost_matrix(
        graph, kind=str(resolved["cost"]["kind"]), scale=float(resolved["cost"]["scale"])
    )
    K = graph.K
    rho = np.full(K, 1.0 / K, dtype=np.float64)
    mu = np.full(K, 1.0 / K, dtype=np.float64)
    tau = float(resolved["regularization"]["tau"])
    lambda_ = float(resolved["regularization"]["lambda_"])
    T0 = tau * lambda_
    sinkhorn_tolerance = float(resolved["sinkhorn"]["tolerance"])
    sinkhorn_iterations = int(resolved["sinkhorn"]["max_iterations"])
    identity_tolerance = float(resolved["numerical_tolerance"])
    rows: list[dict[str, Any]] = []

    adjacency = graph.adjacency.toarray()
    for action_i in range(K):
        for action_k in range(K):
            row = _base_row(
                resolved=resolved,
                resolved_json=resolved_json,
                metadata_json=metadata_json,
                run_id=_stable_run_id(("input", action_i, action_k), resolved_json),
                method="input_action_graph",
                record_type="input_matrix",
            )
            row.update(
                {
                    "action_i": action_i,
                    "action_k": action_k,
                    "input_edge_weight": adjacency[action_i, action_k],
                    "input_cost": costs[action_i, action_k],
                }
            )
            rows.append(row)

    for q_index, (q_name, q_values) in enumerate(resolved["q_vectors"].items()):
        q = np.asarray(q_values, dtype=np.float64)
        coupling = lot_optimal_coupling(q, costs, rho, mu, tau, lambda_)
        policy = coupling.sum(axis=1)
        column_error = float(np.max(np.abs(coupling.sum(axis=0) - mu)))
        geometry = cooccurrence_geometry(coupling, mu)
        if geometry.rank != K - 1:
            raise FloatingPointError(
                f"co-occurrence Laplacian for {q_name} has rank {geometry.rank}, expected {K - 1}"
            )
        if not np.allclose(geometry.adjacency.sum(axis=1), policy, atol=1e-12, rtol=1e-12):
            raise FloatingPointError("A_Q 1 does not equal pi_Q")

        for action_i in range(K):
            node = _base_row(
                resolved=resolved,
                resolved_json=resolved_json,
                metadata_json=metadata_json,
                run_id=_stable_run_id(("node", q_name, action_i), resolved_json),
                method="lot_optimal_coupling",
                record_type="node",
                q_name=q_name,
            )
            node.update(
                {
                    "q_index": q_index,
                    "action_i": action_i,
                    "q_value": q[action_i],
                    "policy_mass": policy[action_i],
                    "coupling_column_error": column_error,
                    "laplacian_rank": geometry.rank,
                }
            )
            rows.append(node)
            for action_k in range(K):
                matrix_row = _base_row(
                    resolved=resolved,
                    resolved_json=resolved_json,
                    metadata_json=metadata_json,
                    run_id=_stable_run_id(
                        ("geometry", q_name, action_i, action_k), resolved_json
                    ),
                    method="transport_cooccurrence",
                    record_type="geometry_matrix",
                    q_name=q_name,
                )
                matrix_row.update(
                    {
                        "q_index": q_index,
                        "action_i": action_i,
                        "action_k": action_k,
                        "cooccurrence_weight": geometry.adjacency[action_i, action_k],
                        "effective_resistance": geometry.resistance[action_i, action_k],
                        "laplacian_rank": geometry.rank,
                    }
                )
                rows.append(matrix_row)

        for pair_name, specification in resolved["pairs"].items():
            action_i, action_k = map(int, specification["actions"])
            theoretical = T0 * geometry.resistance[action_i, action_k]
            if theoretical <= 0.0:
                raise FloatingPointError("distinct actions must have positive effective resistance")
            if max(float(value) for value in resolved["finite_difference_steps"]) >= min(
                policy[action_i], policy[action_k]
            ):
                raise ValueError(
                    f"finite-difference step leaves simplex interior for {q_name}/{pair_name}"
                )
            for step in resolved["finite_difference_steps"]:
                h = float(step)
                observed, minus, base, plus = finite_difference_curvature(
                    policy,
                    action_i,
                    action_k,
                    h,
                    costs,
                    rho,
                    mu,
                    tau,
                    lambda_,
                    tolerance=sinkhorn_tolerance,
                    max_iterations=sinkhorn_iterations,
                )
                absolute_error = abs(observed - theoretical)
                relative_error = absolute_error / theoretical
                row = _base_row(
                    resolved=resolved,
                    resolved_json=resolved_json,
                    metadata_json=metadata_json,
                    run_id=_stable_run_id(("curvature", q_name, pair_name, h), resolved_json),
                    method="central_finite_difference",
                    record_type="curvature",
                    q_name=q_name,
                )
                row.update(
                    {
                        "q_index": q_index,
                        "action_i": action_i,
                        "action_k": action_k,
                        "pair_name": pair_name,
                        "pair_type": str(specification["type"]),
                        "effective_resistance": geometry.resistance[action_i, action_k],
                        "step_size": h,
                        "omega_minus": minus.regularizer_value,
                        "omega_base": base.regularizer_value,
                        "omega_plus": plus.regularizer_value,
                        "finite_difference_curvature": observed,
                        "theoretical_curvature": theoretical,
                        "absolute_curvature_error": absolute_error,
                        "relative_curvature_error": relative_error,
                        "coupling_row_error": max(
                            minus.row_error, base.row_error, plus.row_error
                        ),
                        "coupling_column_error": max(
                            minus.column_error, base.column_error, plus.column_error
                        ),
                        "laplacian_rank": geometry.rank,
                        "identity_holds": relative_error <= identity_tolerance,
                    }
                )
                rows.append(row)
    return validate_results(pd.DataFrame(rows))


def run_resistance_experiment(
    config: Mapping[str, Any], *, repository: str | Path = "."
) -> pd.DataFrame:
    """Run M9 and atomically write the complete long-form raw artifact.

    A failed numerical run leaves one explicit target-labeled failure row
    before the exception is re-raised, matching the suite's result policy.
    """

    resolved = resolve_resistance_config(config)
    resolved_json = config_json(resolved)
    failure_id = _stable_run_id(("experiment_failure",), resolved_json)
    output_path = Path(resolved["raw_output"])
    try:
        frame = _run_resistance_impl(resolved, repository=repository)
    except Exception as error:
        metadata_json = json.dumps(run_metadata(resolved, repository), sort_keys=True)
        failed = _base_row(
            resolved=resolved,
            resolved_json=resolved_json,
            metadata_json=metadata_json,
            run_id=failure_id,
            method="experiment_failure",
            record_type="experiment_failure",
            status="failed",
        )
        failed["error_message"] = f"{type(error).__name__}: {error}"
        write_results_atomic([failed], output_path)
        raise
    write_results_atomic(frame, output_path)
    return frame


def summarize_resistance(
    raw: pd.DataFrame, *, resolved_config: Mapping[str, Any] | None = None
) -> pd.DataFrame:
    """Return the compact deterministic finite-difference convergence table."""

    frame = validate_results(raw)
    frame = frame.loc[
        (frame["experiment"] == "resistance")
        & (frame["status"] == "complete")
        & (frame["record_type"] == "curvature")
    ].copy()
    if resolved_config is not None:
        expected = config_json(resolve_resistance_config(resolved_config))
        frame = frame.loc[frame["config_json"] == expected]
    columns = [
        "q_name",
        "pair_name",
        "pair_type",
        "action_i",
        "action_k",
        "step_size",
        "effective_resistance",
        "temperature",
        "finite_difference_curvature",
        "theoretical_curvature",
        "absolute_curvature_error",
        "relative_curvature_error",
        "coupling_row_error",
        "coupling_column_error",
        "laplacian_rank",
        "identity_holds",
        "target",
        "reference_target",
        "config_json",
    ]
    return frame.loc[:, columns].sort_values(
        ["q_name", "pair_name", "step_size"], ascending=[True, True, False]
    ).reset_index(drop=True)


def write_resistance_summary_atomic(
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
