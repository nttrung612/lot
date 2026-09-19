import numpy as np

from lot_experiments.counters import OperationCounters
from lot_experiments.graphs import cycle_graph, path_graph
from lot_experiments.heat_local import LazyTruncatedHeat, local_truncated_heat_column
from lot_experiments.kernels import (
    exact_heat_column,
    poisson_tail,
    truncated_heat_column,
    truncated_heat_kernel,
    uniformized_random_walk,
)


def test_truncated_support_is_inside_the_graph_ball():
    graph = cycle_graph(11)
    radius = 3
    anchor = 1
    column = local_truncated_heat_column(graph, 0.9, radius, anchor)
    support = set(np.flatnonzero(column > 1e-15))
    assert support <= graph.ball(anchor, radius)
    assert np.all(column >= 0.0)
    np.testing.assert_allclose(column.sum(), 1.0, atol=1e-14, rtol=0.0)


def test_poisson_tail_l1_certificate_holds_on_small_graphs():
    graph = path_graph(8)
    diffusion_time = 0.85
    _, nu_u = uniformized_random_walk(graph.laplacian)
    theta = nu_u * diffusion_time
    for radius in range(6):
        for anchor in range(graph.K):
            exact = exact_heat_column(graph.laplacian, diffusion_time, anchor)
            truncated = truncated_heat_column(
                graph.laplacian, diffusion_time, radius, anchor, nu_u=nu_u
            )
            error = np.abs(exact - truncated).sum()
            assert error <= 2.0 * poisson_tail(theta, radius) + 2e-13


def test_local_and_dense_finite_walk_implementations_agree_when_ball_covers_graph():
    graph = cycle_graph(7)
    radius = 4
    dense = truncated_heat_kernel(graph.laplacian, 0.7, radius)
    for anchor in range(graph.K):
        local = local_truncated_heat_column(graph, 0.7, radius, anchor)
        np.testing.assert_allclose(local, dense[:, anchor], atol=2e-14, rtol=0.0)


def test_local_finite_walk_converges_to_exact_heat_for_large_radius():
    graph = path_graph(6)
    diffusion_time = 0.4
    radius = 24
    _, nu_u = uniformized_random_walk(graph.laplacian)
    assert poisson_tail(nu_u * diffusion_time, radius) < 1e-25
    for anchor in range(graph.K):
        local = local_truncated_heat_column(graph, diffusion_time, radius, anchor)
        exact = exact_heat_column(graph.laplacian, diffusion_time, anchor)
        np.testing.assert_allclose(local, exact, atol=5e-14, rtol=0.0)


def test_lazy_columns_are_cached_and_charge_only_cold_graph_accesses():
    graph = path_graph(9)
    counter = OperationCounters()
    lazy = LazyTruncatedHeat(graph, diffusion_time=0.5, radius=3, counter=counter)
    first = lazy.column(4)
    accesses_after_cold_start = counter.graph_neighbor_accesses
    second = lazy.column(4)
    np.testing.assert_array_equal(first, second)
    assert accesses_after_cold_start > 0
    assert counter.graph_neighbor_accesses == accesses_after_cold_start
    assert lazy.cache_misses == 1
    assert lazy.cache_hits == 1
    assert lazy.cached_columns == 1

