import numpy as np
import pytest

from lot_experiments.backups import (
    cost_based_lot_backup,
    heat_backup,
    lot_backup,
    prior_weighted_maxent,
)
from lot_experiments.graphs import cycle_graph
from lot_experiments.kernels import exact_heat_kernel


def assert_policy(policy):
    assert np.min(policy) >= 0.0
    np.testing.assert_allclose(policy.sum(), 1.0, atol=1e-14, rtol=0.0)


def test_zero_cost_lot_agrees_with_prior_weighted_maxent():
    q = np.array([-0.4, 0.2, 1.1, -0.1])
    rho = np.array([0.1, 0.2, 0.3, 0.4])
    mu = np.array([0.2, 0.5, 0.3])
    costs = np.zeros((len(q), len(mu)))
    lot = cost_based_lot_backup(q, costs, rho, mu, tau=0.4, lambda_=0.5)
    maxent = prior_weighted_maxent(q, rho, T0=0.2)
    np.testing.assert_allclose(lot.value, maxent.value, atol=1e-14, rtol=0.0)
    np.testing.assert_allclose(lot.policy, maxent.policy, atol=1e-14, rtol=0.0)


@pytest.mark.parametrize("backup_kind", ["heat", "cost", "pruned"])
def test_translation_equivariance_and_policy_validity(backup_kind):
    q = np.array([-0.2, 0.7, 1.4, -0.8, 0.3])
    shift = 123.25
    mu = np.full(5, 0.2)
    if backup_kind == "heat":
        heat = exact_heat_kernel(cycle_graph(5).laplacian, 0.6)
        first = heat_backup(q, heat, mu, T0=0.3)
        shifted = heat_backup(q + shift, heat, mu, T0=0.3)
    elif backup_kind == "cost":
        costs = np.abs(np.arange(5)[:, None] - np.arange(5)[None, :]).astype(float)
        rho = np.array([0.05, 0.1, 0.2, 0.25, 0.4])
        first = cost_based_lot_backup(q, costs, rho, mu, tau=0.6, lambda_=0.5)
        shifted = cost_based_lot_backup(q + shift, costs, rho, mu, tau=0.6, lambda_=0.5)
    else:
        heat = exact_heat_kernel(cycle_graph(5).laplacian, 0.6)
        retained = heat >= np.sort(heat, axis=0)[-3, :][None, :]
        first = lot_backup(q, heat, mu, T0=0.3, retained=retained)
        shifted = lot_backup(q + shift, heat, mu, T0=0.3, retained=retained)
    np.testing.assert_allclose(shifted.value, first.value + shift, atol=2e-13, rtol=0.0)
    np.testing.assert_allclose(shifted.policy, first.policy, atol=2e-13, rtol=0.0)
    assert_policy(first.policy)
    np.testing.assert_allclose(first.anchor_policies.sum(axis=0), 1.0, atol=1e-14)


def test_stable_log_sum_exp_handles_extreme_action_values():
    q = np.array([10_000.0, -10_000.0, 0.0])
    weights = np.full((3, 2), 1.0 / 3.0)
    result = lot_backup(q, weights, [0.25, 0.75], T0=1e-3)
    assert np.isfinite(result.value)
    assert_policy(result.policy)
    assert result.policy[0] == pytest.approx(1.0)

