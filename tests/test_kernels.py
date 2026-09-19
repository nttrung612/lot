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
    exact_heat_column,
    exact_heat_kernel,
    truncated_heat_kernel,
    uniformized_random_walk,
)


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

