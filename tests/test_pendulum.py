import numpy as np

from lot_experiments.environments.pendulum_discrete import (
    PendulumDiscreteEnv,
    PendulumStateGrid,
    torque_grid,
)
from lot_experiments.graphs import path_graph
from lot_experiments.kernels import exact_heat_kernel
from lot_experiments.pendulum_reference import (
    PathHeatOperator,
    full_exact_heat_backup,
    solve_pendulum_reference,
)
from lot_experiments.planners.dense import batched_lot_backup


def test_pendulum_model_matches_official_gymnasium_dynamics():
    model = PendulumDiscreteEnv(K=5)
    np.testing.assert_allclose(torque_grid(5), [-2.0, -1.0, 0.0, 1.0, 2.0])
    indices = model.nominal_action_indices([0.0, 1.0], [0.0, 0.0])
    np.testing.assert_array_equal(indices, [2, 0])
    theta, theta_dot, torque = 0.73, -1.4, 1.1
    next_theta, next_theta_dot = model.transition(theta, theta_dot, torque)
    raw_reward = model.raw_reward(theta, theta_dot, torque)

    gym_env = model.make_gymnasium_env(remove_time_limit=True)
    try:
        gym_env.reset(seed=4)
        gym_env.unwrapped.state = np.array([theta, theta_dot], dtype=np.float64)
        _, expected_reward, terminated, truncated, _ = gym_env.step(
            np.array([torque], dtype=np.float32)
        )
        assert not terminated and not truncated
        np.testing.assert_allclose(
            gym_env.unwrapped.state, [next_theta, next_theta_dot], atol=2e-8
        )
        np.testing.assert_allclose(raw_reward, expected_reward, atol=2e-8)
    finally:
        gym_env.close()


def test_periodic_grid_interpolation_wraps_and_is_exact_on_grid_nodes():
    grid = PendulumStateGrid(9, 7)
    theta, theta_dot = np.meshgrid(
        grid.angles, grid.angular_velocities, indexing="ij"
    )
    values = np.sin(theta) + 0.2 * theta_dot
    actual = grid.interpolate(values, theta, theta_dot)
    np.testing.assert_allclose(actual, values, atol=2e-15)
    left = grid.interpolate(values, [-np.pi + 1e-8], [0.3])
    right = grid.interpolate(values, [np.pi + 1e-8], [0.3])
    np.testing.assert_allclose(left, right, atol=3e-8)


def test_spectral_path_heat_and_backup_match_dense_reference():
    rng = np.random.default_rng(19)
    K = 8
    diffusion_time = 0.31
    edge_weight = 2.7
    operator = PathHeatOperator(K, diffusion_time, edge_weight)
    dense_heat = exact_heat_kernel(
        path_graph(K, edge_weight=edge_weight).laplacian, diffusion_time
    )
    nonnegative = rng.uniform(size=(4, K))
    np.testing.assert_allclose(
        operator.apply(nonnegative), nonnegative @ dense_heat, atol=2e-15
    )

    q_values = rng.normal(size=(4, K))
    uniform_anchors = np.full((4, K), 1.0 / K)
    actual_values, actual_policy = full_exact_heat_backup(
        q_values, operator, 0.23
    )
    expected_values, expected_policy = batched_lot_backup(
        q_values, dense_heat, uniform_anchors, 0.23
    )
    np.testing.assert_allclose(actual_values, expected_values, atol=3e-15)
    np.testing.assert_allclose(actual_policy, expected_policy, atol=3e-15)

    anchors = np.array([0, 2, 5, 7])
    selected_values, selected_policy = full_exact_heat_backup(
        q_values, operator, 0.23, anchor_indices=anchors
    )
    one_hot = np.zeros((len(q_values), K))
    one_hot[np.arange(len(q_values)), anchors] = 1.0
    expected_selected_values, expected_selected_policy = batched_lot_backup(
        q_values, dense_heat, one_hot, 0.23
    )
    np.testing.assert_allclose(selected_values, expected_selected_values, atol=3e-15)
    np.testing.assert_allclose(selected_policy, expected_selected_policy, atol=3e-15)


def test_small_fitted_value_reference_converges_with_valid_policy():
    environment = PendulumDiscreteEnv(K=7)
    solution = solve_pendulum_reference(
        environment,
        PendulumStateGrid(13, 13),
        gamma=0.8,
        T0=0.1,
        diffusion_time=0.02,
        heat_scaling="physical_heat",
        tolerance=2e-8,
        max_iterations=500,
        state_chunk_size=32,
    )
    assert solution.converged
    assert solution.bellman_residual <= 2.1e-8
    assert solution.action_evaluations > 0
    assert np.all(solution.evaluation_policies >= 0.0)
    np.testing.assert_allclose(solution.evaluation_policies.sum(axis=1), 1.0, atol=1e-14)
