"""Exact ``Pendulum-v1`` dynamics with a discrete torque interface.

The adapter mirrors Gymnasium's unmodified deterministic dynamics but exposes
vectorized model queries for fitted value iteration.  The external ``TimeLimit``
is deliberately absent from these model queries; episodic evaluation can still
use :meth:`make_gymnasium_env` with the standard wrapper intact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def normalize_angle(angle: ArrayLike) -> FloatArray:
    """Map angles to Gymnasium Pendulum's ``[-pi, pi)`` convention."""

    values = np.asarray(angle, dtype=np.float64)
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def torque_grid(K: int, max_torque: float = 2.0) -> FloatArray:
    """Return the inclusive ordered action grid on ``[-max_torque, max_torque]``."""

    if not isinstance(K, (int, np.integer)) or int(K) < 2:
        raise ValueError("K must be an integer at least two")
    if max_torque <= 0.0 or not math.isfinite(max_torque):
        raise ValueError("max_torque must be finite and positive")
    return np.linspace(-float(max_torque), float(max_torque), int(K), dtype=np.float64)


@dataclass(frozen=True)
class PendulumStateGrid:
    """Tensor grid with a periodic angle axis and bounded velocity axis."""

    angle_count: int
    velocity_count: int
    max_speed: float = 8.0

    def __post_init__(self) -> None:
        if self.angle_count < 3 or self.velocity_count < 2:
            raise ValueError("state grid requires at least 3 angles and 2 velocities")
        if self.max_speed <= 0.0 or not math.isfinite(self.max_speed):
            raise ValueError("max_speed must be finite and positive")

    @property
    def angles(self) -> FloatArray:
        # Centering the periodic cells at zero includes the upright state for
        # odd grids without duplicating the -pi/pi endpoint.
        indices = np.arange(self.angle_count, dtype=np.float64)
        return (indices - self.angle_count // 2) * (2.0 * np.pi / self.angle_count)

    @property
    def angular_velocities(self) -> FloatArray:
        return np.linspace(
            -self.max_speed,
            self.max_speed,
            self.velocity_count,
            dtype=np.float64,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.angle_count, self.velocity_count

    @property
    def size(self) -> int:
        return self.angle_count * self.velocity_count

    def states(self) -> tuple[FloatArray, FloatArray]:
        theta, theta_dot = np.meshgrid(
            self.angles, self.angular_velocities, indexing="ij"
        )
        return theta.ravel(), theta_dot.ravel()

    def interpolate(
        self,
        values: ArrayLike,
        theta: ArrayLike,
        theta_dot: ArrayLike,
    ) -> FloatArray:
        """Bilinearly interpolate, wrapping angle and clipping velocity."""

        table = np.asarray(values, dtype=np.float64)
        if table.shape != self.shape or not np.all(np.isfinite(table)):
            raise ValueError(f"values must be a finite array with shape {self.shape}")
        query_theta, query_velocity = np.broadcast_arrays(
            np.asarray(theta, dtype=np.float64),
            np.asarray(theta_dot, dtype=np.float64),
        )
        if not np.all(np.isfinite(query_theta)) or not np.all(np.isfinite(query_velocity)):
            raise ValueError("interpolation coordinates must be finite")

        angle_step = 2.0 * np.pi / self.angle_count
        angle_position = (
            (query_theta - self.angles[0]) / angle_step
        ) % self.angle_count
        angle_lower = np.floor(angle_position).astype(np.int64)
        angle_upper = (angle_lower + 1) % self.angle_count
        angle_fraction = angle_position - angle_lower

        velocities = self.angular_velocities
        velocity_step = velocities[1] - velocities[0]
        velocity_position = (
            np.clip(query_velocity, -self.max_speed, self.max_speed) - velocities[0]
        ) / velocity_step
        velocity_lower = np.minimum(
            np.floor(velocity_position).astype(np.int64), self.velocity_count - 2
        )
        velocity_upper = velocity_lower + 1
        velocity_fraction = np.clip(velocity_position - velocity_lower, 0.0, 1.0)

        lower = (
            (1.0 - angle_fraction) * table[angle_lower, velocity_lower]
            + angle_fraction * table[angle_upper, velocity_lower]
        )
        upper = (
            (1.0 - angle_fraction) * table[angle_lower, velocity_upper]
            + angle_fraction * table[angle_upper, velocity_upper]
        )
        return (1.0 - velocity_fraction) * lower + velocity_fraction * upper


@dataclass(frozen=True)
class PendulumDiscreteEnv:
    """Discrete-action, exact-model adapter for Gymnasium ``Pendulum-v1``."""

    K: int
    reward_scale: float = 16.2736044
    gravity: float = 10.0
    mass: float = 1.0
    length: float = 1.0
    time_step: float = 0.05
    max_speed: float = 8.0
    max_torque: float = 2.0

    def __post_init__(self) -> None:
        torque_grid(self.K, self.max_torque)
        positive = (
            self.reward_scale,
            self.gravity,
            self.mass,
            self.length,
            self.time_step,
            self.max_speed,
            self.max_torque,
        )
        if any(value <= 0.0 or not math.isfinite(value) for value in positive):
            raise ValueError("Pendulum constants and reward_scale must be positive and finite")

    @property
    def actions(self) -> FloatArray:
        return torque_grid(self.K, self.max_torque)

    @property
    def action_spacing(self) -> float:
        return 2.0 * self.max_torque / (self.K - 1)

    def transition(
        self, theta: ArrayLike, theta_dot: ArrayLike, torque: ArrayLike
    ) -> tuple[FloatArray, FloatArray]:
        """Evaluate the official deterministic next-state equations."""

        angle, velocity, action = np.broadcast_arrays(
            np.asarray(theta, dtype=np.float64),
            np.asarray(theta_dot, dtype=np.float64),
            np.asarray(torque, dtype=np.float64),
        )
        if not (np.all(np.isfinite(angle)) and np.all(np.isfinite(velocity))):
            raise ValueError("states must be finite")
        action = np.clip(action, -self.max_torque, self.max_torque)
        next_velocity = velocity + (
            3.0 * self.gravity / (2.0 * self.length) * np.sin(angle)
            + 3.0 / (self.mass * self.length**2) * action
        ) * self.time_step
        next_velocity = np.clip(next_velocity, -self.max_speed, self.max_speed)
        next_angle = angle + next_velocity * self.time_step
        return normalize_angle(next_angle), next_velocity

    def raw_reward(
        self, theta: ArrayLike, theta_dot: ArrayLike, torque: ArrayLike
    ) -> FloatArray:
        """Return Gymnasium's unnormalized reward at the current state/action."""

        angle, velocity, action = np.broadcast_arrays(
            np.asarray(theta, dtype=np.float64),
            np.asarray(theta_dot, dtype=np.float64),
            np.asarray(torque, dtype=np.float64),
        )
        action = np.clip(action, -self.max_torque, self.max_torque)
        costs = normalize_angle(angle) ** 2 + 0.1 * velocity**2 + 0.001 * action**2
        return -costs

    def planning_reward(
        self, theta: ArrayLike, theta_dot: ArrayLike, torque: ArrayLike
    ) -> FloatArray:
        return self.raw_reward(theta, theta_dot, torque) / self.reward_scale

    def q_values(
        self,
        values: ArrayLike,
        grid: PendulumStateGrid,
        theta: ArrayLike,
        theta_dot: ArrayLike,
        gamma: float,
    ) -> FloatArray:
        """Evaluate all discrete actions for a batch of continuous states."""

        if not 0.0 <= gamma < 1.0 or not math.isfinite(gamma):
            raise ValueError("gamma must be finite and lie in [0, 1)")
        angles = np.asarray(theta, dtype=np.float64).reshape(-1, 1)
        velocities = np.asarray(theta_dot, dtype=np.float64).reshape(-1, 1)
        if angles.shape != velocities.shape:
            raise ValueError("theta and theta_dot must contain the same number of states")
        actions = self.actions[None, :]
        next_angle, next_velocity = self.transition(angles, velocities, actions)
        continuation = grid.interpolate(values, next_angle, next_velocity)
        return self.planning_reward(angles, velocities, actions) + gamma * continuation

    @staticmethod
    def reset_state(seed: int) -> tuple[float, float]:
        """Draw the standard reset distribution reproducibly."""

        rng = np.random.default_rng(int(seed))
        state = rng.uniform(low=(-np.pi, -1.0), high=(np.pi, 1.0))
        return float(state[0]), float(state[1])

    @staticmethod
    def make_gymnasium_env(*, remove_time_limit: bool = False) -> gym.Env:
        """Create the official environment, optionally removing only TimeLimit."""

        environment = gym.make("Pendulum-v1")
        if remove_time_limit:
            if not isinstance(environment, gym.wrappers.TimeLimit):
                environment.close()
                raise RuntimeError("Pendulum-v1 no longer has the expected outer TimeLimit")
            return environment.env
        return environment
