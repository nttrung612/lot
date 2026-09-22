from __future__ import annotations

import numpy as np
import pandas as pd

from lot_experiments.plotting.resistance import plot_resistance_figure
from lot_experiments.resistance import (
    TARGET,
    cooccurrence_geometry,
    finite_difference_curvature,
    graph_cost_matrix,
    lot_optimal_coupling,
    regularizer_value,
    resolve_resistance_config,
    run_resistance_experiment,
    summarize_resistance,
    two_cluster_weak_bridge_graph,
)


def _small_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    graph = two_cluster_weak_bridge_graph(2, within_weight=1.0, bridge_weight=0.2)
    costs = graph_cost_matrix(graph, scale=0.15)
    prior = np.full(graph.K, 1.0 / graph.K)
    return costs, prior, prior.copy(), 0.2, 0.7


def test_lot_coupling_and_cooccurrence_invariants() -> None:
    costs, rho, mu, tau, lambda_ = _small_problem()
    q = np.array([0.04, -0.01, 0.02, -0.03])
    coupling = lot_optimal_coupling(q, costs, rho, mu, tau, lambda_)
    policy = coupling.sum(axis=1)
    geometry = cooccurrence_geometry(coupling, mu)

    np.testing.assert_allclose(coupling.sum(axis=0), mu, atol=1e-14)
    np.testing.assert_allclose(policy.sum(), 1.0, atol=1e-14)
    assert np.all(coupling > 0.0)
    np.testing.assert_allclose(geometry.adjacency, geometry.adjacency.T, atol=1e-14)
    np.testing.assert_allclose(geometry.adjacency.sum(axis=1), policy, atol=1e-14)
    np.testing.assert_allclose(geometry.laplacian.sum(axis=1), 0.0, atol=1e-14)
    np.testing.assert_allclose(np.diag(geometry.resistance), 0.0, atol=1e-14)
    assert geometry.rank == len(policy) - 1
    assert np.all(geometry.resistance[np.triu_indices(len(policy), 1)] > 0.0)


def test_balanced_regularizer_recovers_semirelaxed_optimum() -> None:
    costs, rho, mu, tau, lambda_ = _small_problem()
    q = np.array([0.04, -0.01, 0.02, -0.03])
    coupling = lot_optimal_coupling(q, costs, rho, mu, tau, lambda_)
    solution = regularizer_value(
        coupling.sum(axis=1), costs, rho, mu, tau, lambda_, tolerance=5e-15
    )

    np.testing.assert_allclose(solution.coupling, coupling, atol=2e-13, rtol=2e-13)
    assert solution.row_error < 5e-15
    assert solution.column_error < 5e-15


def test_policy_curvature_converges_to_scaled_effective_resistance() -> None:
    costs, rho, mu, tau, lambda_ = _small_problem()
    coupling = lot_optimal_coupling(
        np.array([0.03, -0.01, 0.01, -0.03]), costs, rho, mu, tau, lambda_
    )
    policy = coupling.sum(axis=1)
    geometry = cooccurrence_geometry(coupling, mu)
    theoretical = tau * lambda_ * geometry.resistance[0, 3]
    errors = []
    for step in (1e-2, 3e-3, 1e-3):
        observed, *_ = finite_difference_curvature(
            policy,
            0,
            3,
            step,
            costs,
            rho,
            mu,
            tau,
            lambda_,
            tolerance=5e-15,
        )
        errors.append(abs(observed - theoretical) / theoretical)
    assert errors[2] < errors[1] < errors[0]
    assert errors[-1] < 2e-5


def test_zero_cost_reduces_to_fisher_rao() -> None:
    K = 4
    rho = np.full(K, 1.0 / K)
    mu = rho.copy()
    tau, lambda_ = 0.2, 0.5
    q = np.array([0.08, -0.03, 0.01, -0.06])
    coupling = lot_optimal_coupling(q, np.zeros((K, K)), rho, mu, tau, lambda_)
    policy = coupling.sum(axis=1)
    resistance = cooccurrence_geometry(coupling, mu).resistance

    expected = 1.0 / policy[0] + 1.0 / policy[2]
    np.testing.assert_allclose(resistance[0, 2], expected, rtol=2e-12, atol=2e-12)


def test_resistance_experiment_writes_complete_target_labeled_rows(tmp_path) -> None:
    config = resolve_resistance_config(
        {
            "experiment": "resistance",
            "graph": {
                "cluster_size": 2,
                "within_weight": 1.0,
                "bridge_weight": 0.2,
            },
            "cost": {"scale": 0.15},
            "q_vectors": {
                "balanced": [0.0, 0.0, 0.0, 0.0],
                "tilted": [0.03, -0.01, 0.01, -0.03],
            },
            "pairs": {
                "within": {"actions": [0, 1], "type": "within_cluster"},
                "bridge": {"actions": [1, 2], "type": "bridge"},
                "cross": {"actions": [0, 3], "type": "cross_cluster"},
            },
            "finite_difference_steps": [0.003, 0.001],
            "visualization_q": "tilted",
            "numerical_tolerance": 2e-5,
            "raw_output": str(tmp_path / "raw.parquet"),
            "summary_output": str(tmp_path / "summary.csv"),
            "figure_png": str(tmp_path / "figure.png"),
            "figure_pdf": str(tmp_path / "figure.pdf"),
        }
    )
    raw = run_resistance_experiment(config)
    summary = summarize_resistance(raw, resolved_config=config)

    assert (tmp_path / "raw.parquet").exists()
    assert set(raw["target"]) == {TARGET}
    assert set(raw["reference_target"]) == {TARGET}
    assert set(raw["record_type"]) == {
        "input_matrix",
        "node",
        "geometry_matrix",
        "curvature",
    }
    assert len(summary) == 2 * 3 * 2
    finest = summary.groupby(["q_name", "pair_name"]).last()
    assert finest["relative_curvature_error"].max() < 2e-5
    assert finest["identity_holds"].astype(bool).all()
    assert pd.to_numeric(raw["laplacian_rank"], errors="coerce").dropna().eq(3).all()
    png, pdf = plot_resistance_figure(
        raw,
        config,
        png_path=config["figure_png"],
        pdf_path=config["figure_pdf"],
    )
    assert png.exists() and png.stat().st_size > 0
    assert pdf.exists() and pdf.stat().st_size > 0
