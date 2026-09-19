import numpy as np

from lot_experiments.backups import heat_backup
from lot_experiments.counters import OperationCounters
from lot_experiments.environments.ring_control import RingControlMDP
from lot_experiments.graphs import cycle_graph
from lot_experiments.heat_local import LazyTruncatedHeat
from lot_experiments.kernels import exact_heat_kernel
from lot_experiments.planners.dense import (
    batched_lot_backup,
    dense_second_order_reference,
    full_exact_heat_reference,
    hard_max_reference,
    truncated_heat_reference,
    uniform_maxent_reference,
)
from lot_experiments.planners.local_heat import (
    local_truncated_heat_backup,
    local_truncated_heat_value_iteration,
)
from lot_experiments.planners.poisson_mc import poisson_endpoint_mc_backup
from lot_experiments.planners.random_subset import (
    random_subset_backup,
    uniform_action_mc_backup,
)


def test_batched_dense_backup_matches_scalar_reference_with_pruning():
    rng = np.random.default_rng(2)
    q = rng.normal(size=(4, 7))
    weights = rng.uniform(0.1, 1.0, size=(7, 5))
    weights /= weights.sum(axis=0, keepdims=True)
    mu = rng.uniform(0.1, 1.0, size=(4, 5))
    mu /= mu.sum(axis=1, keepdims=True)
    retained = rng.random((7, 5)) > 0.35
    retained[0, :] = True
    values, policies = batched_lot_backup(
        q, weights, mu, 0.3, retained=retained
    )
    for state in range(len(q)):
        expected = heat_backup(q[state], weights, mu[state], 0.3, retained=retained)
        np.testing.assert_allclose(values[state], expected.value, atol=3e-15)
        np.testing.assert_allclose(policies[state], expected.policy, atol=3e-15)


def test_dense_references_converge_and_keep_targets_distinct():
    mdp = RingControlMDP(K=7, goal=3)
    kwargs = dict(
        mdp=mdp,
        gamma=0.8,
        T0=0.15,
        diffusion_time=0.6,
        tolerance=1e-11,
    )
    exact = full_exact_heat_reference(**kwargs)
    second_order_limit = dense_second_order_reference(**kwargs)
    maxent = uniform_maxent_reference(
        mdp, gamma=0.8, T0=0.15, tolerance=1e-11
    )
    hard = hard_max_reference(mdp, gamma=0.8, tolerance=1e-11)
    assert exact.converged and maxent.converged and hard.converged
    assert exact.target == second_order_limit.target == "exact_heat"
    assert maxent.target == "maxent"
    assert hard.target == "hard_max"
    np.testing.assert_allclose(exact.values, second_order_limit.values, atol=1e-13)
    for result in (exact, maxent, hard):
        assert np.min(result.policy) >= 0.0
        np.testing.assert_allclose(result.policy.sum(axis=1), 1.0, atol=1e-14)
        assert result.counters.action_evaluations > 0


def test_local_planner_matches_dense_truncated_target_without_dense_heat():
    mdp = RingControlMDP(K=7, goal=2)
    graph = cycle_graph(mdp.K)
    kwargs = dict(
        mdp=mdp,
        gamma=0.75,
        T0=0.2,
        diffusion_time=0.7,
        radius=4,
        graph=graph,
        tolerance=1e-11,
    )
    dense = truncated_heat_reference(**kwargs)
    local = local_truncated_heat_value_iteration(**kwargs)
    assert dense.target == local.target == "truncated_heat"
    np.testing.assert_allclose(local.values, dense.values, atol=3e-12, rtol=0.0)
    np.testing.assert_allclose(local.policy, dense.policy, atol=3e-12, rtol=0.0)
    assert local.counters.graph_neighbor_accesses > 0


def test_local_one_backup_matches_dense_for_same_finite_walk_columns():
    graph = cycle_graph(9)
    q = np.linspace(-0.8, 1.1, graph.K)
    mu = RingControlMDP(9, goal=4).anchor_distribution(2)
    counter = OperationCounters()
    lazy = LazyTruncatedHeat(graph, diffusion_time=0.5, radius=3, counter=counter)
    local = local_truncated_heat_backup(q, mu, 0.25, lazy, counter=counter)
    columns = np.column_stack([lazy.column(anchor) for anchor in range(graph.K)])
    dense = heat_backup(q, columns, mu, 0.25)
    np.testing.assert_allclose(local.value, dense.value, atol=2e-15)
    np.testing.assert_allclose(local.policy, dense.policy, atol=2e-15)
    assert counter.unique_actions_touched == graph.K


def test_full_size_random_and_uniform_mc_baselines_reduce_to_dense_backup():
    graph = cycle_graph(7)
    weights = exact_heat_kernel(graph.laplacian, 0.8)
    q = np.array([-0.2, 0.4, 0.7, -0.5, 0.1, 0.8, -0.1])
    mu = RingControlMDP(7, goal=2).anchor_distribution(3)
    exact = heat_backup(q, weights, mu, 0.3)
    random_full = random_subset_backup(
        q,
        weights,
        mu,
        0.3,
        subset_size=graph.K,
        rng=np.random.default_rng(5),
    )
    uniform_full = uniform_action_mc_backup(
        q,
        weights,
        mu,
        0.3,
        samples_per_anchor=graph.K,
        rng=np.random.default_rng(6),
        replace=False,
    )
    np.testing.assert_allclose(random_full.value, exact.value, atol=2e-15)
    np.testing.assert_allclose(uniform_full.value, exact.value, atol=2e-15)
    np.testing.assert_allclose(uniform_full.policy, exact.policy, atol=2e-15)


def test_poisson_endpoint_mc_estimates_exact_heat_backup():
    graph = cycle_graph(5)
    q = np.array([-0.3, 0.2, 0.8, -0.1, 0.4])
    mu = RingControlMDP(5, goal=1).anchor_distribution(4)
    exact = heat_backup(q, exact_heat_kernel(graph.laplacian, 0.6), mu, 0.4)
    estimated = poisson_endpoint_mc_backup(
        q,
        graph,
        mu,
        0.4,
        0.6,
        endpoints_per_anchor=20_000,
        rng=np.random.default_rng(91),
    )
    assert abs(estimated.value - exact.value) < 0.012
    assert np.linalg.norm(estimated.policy - exact.policy, ord=1) < 0.035
    np.testing.assert_allclose(estimated.policy.sum(), 1.0, atol=1e-14)

