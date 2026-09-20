import numpy as np

from lot_experiments.environments.ring_control import RingControlMDP
from lot_experiments.learning.transition_geometry import (
    empirical_transition_geometry,
    population_transition_geometry,
    probe_distribution,
    sample_reward_free_batch,
    theorem10_cost_certificate,
    transition_diffusion_cost,
)


def test_population_geometry_matches_definition_and_is_a_laplacian():
    mdp = RingControlMDP(7)
    nu = probe_distribution(7, kind="spike_uniform", uniform_mass=0.2, center=2)
    geometry = population_transition_geometry(mdp, nu)
    transition_kernel = mdp.transition_kernel()
    expected = np.stack(
        [transition_kernel[:, action, :].T @ nu - nu for action in range(mdp.K)]
    )
    np.testing.assert_allclose(geometry.signatures, expected, atol=1e-14)
    np.testing.assert_allclose(geometry.laplacian, geometry.laplacian.T, atol=1e-14)
    np.testing.assert_allclose(geometry.laplacian.sum(axis=1), 0.0, atol=1e-14)
    assert np.linalg.eigvalsh(geometry.laplacian).min() >= -1e-12

    costs = transition_diffusion_cost(geometry, diffusion_time=0.8, kappa_C=1.3)
    np.testing.assert_allclose(costs, costs.T, atol=1e-14)
    np.testing.assert_allclose(np.diag(costs), 0.0, atol=1e-14)
    assert np.min(costs) >= 0.0
    assert np.max(costs) > 0.0


def test_uniform_probe_exposes_ring_signature_degeneracy():
    mdp = RingControlMDP(8)
    geometry = population_transition_geometry(
        mdp, probe_distribution(mdp.K, kind="uniform")
    )
    np.testing.assert_allclose(geometry.signatures, 0.0, atol=1e-14)
    np.testing.assert_allclose(
        transition_diffusion_cost(geometry, 1.0, 1.0), 0.0, atol=1e-14
    )


def test_reward_free_batches_are_reproducible_and_prefix_nested():
    mdp = RingControlMDP(6)
    nu = probe_distribution(6, uniform_mass=0.15)
    first = sample_reward_free_batch(mdp, nu, 32, np.random.default_rng(11))
    second = sample_reward_free_batch(mdp, nu, 32, np.random.default_rng(11))
    np.testing.assert_array_equal(first.origins, second.origins)
    np.testing.assert_array_equal(first.successors, second.successors)

    prefix = empirical_transition_geometry(first, 8)
    explicit = empirical_transition_geometry(
        type(first)(origins=first.origins[:, :8], successors=first.successors[:, :8])
    )
    np.testing.assert_allclose(prefix.signatures, explicit.signatures)
    np.testing.assert_allclose(prefix.laplacian, explicit.laplacian)
    assert theorem10_cost_certificate(
        state_count=6,
        action_count=6,
        samples_per_action=32,
        delta=0.05,
        diffusion_time=1.0,
        kappa_C=1.0,
    ) > 0.0
