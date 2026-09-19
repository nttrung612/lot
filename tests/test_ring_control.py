import numpy as np

from lot_experiments.counters import OperationCounters
from lot_experiments.environments.ring_control import (
    RingControlMDP,
    cyclic_distance,
    signed_cyclic_displacement,
)


def test_ring_reward_anchor_and_transition_kernel_match_specification():
    mdp = RingControlMDP(K=8, goal=3, action_cost=0.05, anchor_uniform_mass=0.05)
    assert cyclic_distance(7, 1, 8) == 2
    assert signed_cyclic_displacement(7, 3, 8) == 4
    assert mdp.nominal_action(7) == 4
    expected_reward = -(2.0**2) / (8 / 2) ** 2 - 0.05 * (2.0**2) / (8 / 2) ** 2
    assert mdp.reward(1, 2) == expected_reward

    mu = mdp.anchor_distribution(7)
    assert np.all(mu > 0.0)
    np.testing.assert_allclose(mu.sum(), 1.0, atol=1e-15)
    np.testing.assert_allclose(mu[4], 0.95 + 0.05 / 8, atol=1e-15)

    P = mdp.transition_kernel()
    assert P.shape == (8, 8, 8)
    np.testing.assert_allclose(P.sum(axis=2), 1.0, atol=1e-15)
    np.testing.assert_allclose(P[2, 4, [5, 6, 7]], [0.1, 0.8, 0.1])


def test_expected_values_equal_explicit_P_and_account_queries():
    mdp = RingControlMDP(K=7, goal=2)
    values = np.linspace(-0.5, 0.7, mdp.K)
    counter = OperationCounters()
    actual = mdp.expected_action_values(
        values,
        0.8,
        states=[1, 3],
        actions=[0, 2, 5],
        counter=counter,
    )
    P = mdp.transition_kernel()
    expected = mdp.reward_matrix[np.ix_([1, 3], [0, 2, 5])] + 0.8 * np.einsum(
        "sak,k->sa", P[np.ix_([1, 3], [0, 2, 5], np.arange(mdp.K))], values
    )
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    assert counter.transition_calls == 6
    assert counter.action_evaluations == 6
    assert counter.unique_actions_touched == 3


def test_sampling_is_seeded_and_charges_one_oracle_call():
    mdp = RingControlMDP(K=9)
    first_counter = OperationCounters()
    first = mdp.sample_transition(
        3, 4, np.random.default_rng(11), counter=first_counter
    )
    second = mdp.sample_transition(3, 4, np.random.default_rng(11))
    assert first == second
    assert first_counter.transition_calls == 1
    assert first_counter.action_evaluations == 1
    assert first_counter.unique_actions_touched == 1

