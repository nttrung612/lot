import numpy as np
import pytest

from lot_experiments.graphs import (
    cycle_graph,
    grid_graph,
    path_graph,
    random_regular_expander,
    torus_graph,
)
from lot_experiments.kernels import (
    cycle_exact_heat_kernel,
    cycle_exact_heat_log_kernel,
    exact_heat_column,
    exact_heat_kernel,
    heat_approximation_certificate,
    poisson_head_weights,
    truncated_heat_kernel,
    uniformized_random_walk,
)


def test_fast_cycle_heat_matches_generic_dense_reference():
    graph = cycle_graph(9)
    generic = exact_heat_kernel(graph.laplacian, 0.73)
    fast = cycle_exact_heat_kernel(graph.K, 0.73)
    np.testing.assert_allclose(fast, generic, atol=4e-15, rtol=0.0)


def test_cycle_log_heat_preserves_tiny_positive_tail_mass():
    log_heat = cycle_exact_heat_log_kernel(64, 0.25)
    heat = cycle_exact_heat_kernel(64, 0.25)

    assert np.all(np.isfinite(log_heat))
    assert np.all(heat > 0.0)
    np.testing.assert_allclose(np.exp(log_heat), heat, atol=0.0, rtol=2e-15)
    np.testing.assert_allclose(heat.sum(axis=0), 1.0, atol=2e-15, rtol=0.0)


@pytest.mark.parametrize(
    "graph",
    [path_graph(7), cycle_graph(8), grid_graph(3, 4), torus_graph(3, 4)],
)
def test_exact_heat_is_nonnegative_and_column_stochastic(graph):
    heat = exact_heat_kernel(graph.laplacian, 0.7)
    assert np.min(heat) >= 0.0
    np.testing.assert_allclose(heat.sum(axis=0), 1.0, atol=1e-13, rtol=0.0)
    for anchor in range(graph.K):
        np.testing.assert_allclose(
            exact_heat_column(graph.laplacian, 0.7, anchor),
            heat[:, anchor],
            atol=2e-13,
            rtol=0.0,
        )


def test_uniformized_walk_is_a_nonnegative_column_stochastic_action_walk():
    graph = path_graph(6)
    walk, nu_u = uniformized_random_walk(graph.laplacian)
    assert nu_u == 2.0
    dense = walk.toarray()
    assert np.min(dense) >= 0.0
    np.testing.assert_allclose(dense.sum(axis=0), 1.0, atol=1e-14, rtol=0.0)


def test_truncated_heat_is_nonnegative_and_column_stochastic():
    graph = cycle_graph(9)
    heat = truncated_heat_kernel(graph.laplacian, 1.2, radius=4)
    assert np.min(heat) >= 0.0
    np.testing.assert_allclose(heat.sum(axis=0), 1.0, atol=1e-13, rtol=0.0)


def test_random_regular_constructor_uses_generator_and_has_requested_degree():
    first = random_regular_expander(12, 3, seed=17).adjacency
    second = random_regular_expander(12, 3, seed=17).adjacency
    np.testing.assert_array_equal(first.toarray(), second.toarray())
    np.testing.assert_allclose(first.sum(axis=0), 3.0)


def test_uniformization_rejects_rate_below_maximum_weighted_degree():
    with pytest.raises(ValueError, match="below"):
        uniformized_random_walk(path_graph(5).laplacian, nu_u=1.5)


@pytest.mark.parametrize(
    ("laplacian", "message"),
    [
        (np.array([[1.0, -1.0], [0.0, 0.0]]), "symmetric"),
        (np.array([[1.0, 0.1], [0.1, 1.0]]), "off-diagonal"),
        (np.array([[1.0, -0.5], [-0.5, 1.0]]), "sum to zero"),
        (np.array([[np.nan]]), "finite"),
    ],
)
def test_heat_rejects_invalid_laplacians(laplacian, message):
    with pytest.raises(ValueError, match=message):
        exact_heat_kernel(laplacian, 0.5)


def test_zero_time_exact_and_truncated_heat_are_identity():
    graph = path_graph(5)
    identity = np.eye(graph.K)
    np.testing.assert_array_equal(exact_heat_kernel(graph.laplacian, 0.0), identity)
    np.testing.assert_array_equal(
        truncated_heat_kernel(graph.laplacian, 0.0, radius=4), identity
    )


def test_poisson_head_is_stable_and_radius_must_be_integral():
    weights = poisson_head_weights(theta=1e6, radius=8)
    assert np.all(np.isfinite(weights))
    assert np.all(weights >= 0.0)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-14, rtol=0.0)
    with pytest.raises(ValueError, match="integer"):
        poisson_head_weights(theta=1.0, radius=2.5)


def test_certificate_reports_each_column_and_detects_violation():
    exact = np.eye(3)
    approximate = exact.copy()
    approximate[:, 0] = [0.8, 0.2, 0.0]
    certificate = heat_approximation_certificate(exact, approximate, beta_r=0.2)
    np.testing.assert_allclose(certificate.column_l1_errors, [0.4, 0.0, 0.0])
    assert certificate.holds
    assert certificate.l1_bound == pytest.approx(0.4)
    failed = heat_approximation_certificate(exact, approximate, beta_r=0.1)
    assert not failed.holds
