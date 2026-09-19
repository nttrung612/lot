"""Finite stochastic ring-control environment used by the synthetic study.

The environment dynamics are deliberately kept separate from the action graph:
``RingControlMDP`` represents the MDP transition kernel ``P`` only.  Planners
receive an :class:`~lot_experiments.graphs.ActionGraph` (and hence ``P_G`` or a
heat kernel) as a separate argument.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import ArrayLike, NDArray

from lot_experiments.counters import OperationCounters


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def cyclic_distance(first: ArrayLike, second: ArrayLike, K: int) -> FloatArray:
    """Return the unsigned shortest distance on ``Z_K`` elementwise."""

    if not isinstance(K, (int, np.integer)) or K < 2:
        raise ValueError("K must be an integer at least two")
    delta = np.mod(
        np.asarray(first, dtype=np.int64) - np.asarray(second, dtype=np.int64), K
    )
    return np.minimum(delta, K - delta).astype(np.float64)


def signed_cyclic_displacement(start: int, destination: int, K: int) -> int:
    """Shortest signed displacement, with the positive tie chosen for even K."""

    if K < 2:
        raise ValueError("K must be at least two")
    forward = (int(destination) - int(start)) % K
    backward = forward - K
    return int(forward if forward <= abs(backward) else backward)


@dataclass(frozen=True)
class RingControlMDP:
    """Tabular MDP on ``S = A = Z_K``.

    A transition follows ``s' = (s + a + xi) mod K``.  Rewards are
    deterministic and use the current state, as specified in the experiment
    plan.  Noise offsets with the same residue are merged, which also makes
    the implementation correct for very small rings.
    """

    K: int
    goal: int = 0
    action_cost: float = 0.05
    anchor_uniform_mass: float = 0.05
    noise_offsets: tuple[int, ...] = (0, 1, -1)
    noise_probabilities: tuple[float, ...] = (0.8, 0.1, 0.1)

    def __post_init__(self) -> None:
        if not isinstance(self.K, (int, np.integer)) or self.K < 3:
            raise ValueError("ring control requires integer K >= 3")
        if not 0 <= int(self.goal) < int(self.K):
            raise ValueError("goal must lie in Z_K")
        if self.action_cost < 0.0 or not math.isfinite(self.action_cost):
            raise ValueError("action_cost must be finite and nonnegative")
        if not 0.0 <= self.anchor_uniform_mass <= 1.0:
            raise ValueError("anchor_uniform_mass must lie in [0, 1]")
        if len(self.noise_offsets) != len(self.noise_probabilities) or not self.noise_offsets:
            raise ValueError("noise offsets and probabilities must be nonempty and aligned")
        probabilities = np.asarray(self.noise_probabilities, dtype=np.float64)
        if (
            not np.all(np.isfinite(probabilities))
            or np.min(probabilities) < 0.0
            or not np.isclose(probabilities.sum(), 1.0, atol=1e-14, rtol=0.0)
        ):
            raise ValueError("noise_probabilities must be a probability vector")

    @property
    def state_count(self) -> int:
        return int(self.K)

    @property
    def action_count(self) -> int:
        return int(self.K)

    @cached_property
    def reward_matrix(self) -> FloatArray:
        """Dense ``r(s,a)`` table; this is not a transition-kernel tensor."""

        states = np.arange(self.K, dtype=np.int64)
        actions = np.arange(self.K, dtype=np.int64)
        scale = (self.K / 2.0) ** 2
        state_penalty = cyclic_distance(states, self.goal, self.K) ** 2 / scale
        action_penalty = cyclic_distance(actions, 0, self.K) ** 2 / scale
        rewards = -state_penalty[:, None] - self.action_cost * action_penalty[None, :]
        rewards.setflags(write=False)
        return rewards

    def reward(self, state: int, action: int) -> float:
        self._validate_state_action(state, action)
        return float(self.reward_matrix[int(state), int(action)])

    def nominal_action(self, state: int) -> int:
        """Action residue giving the shortest deterministic move to the goal."""

        self._validate_state(state)
        return signed_cyclic_displacement(state, self.goal, self.K) % self.K

    def anchor_distribution(self, state: int) -> FloatArray:
        """Return ``(1-zeta)e_j(s) + zeta Uniform(A)``."""

        self._validate_state(state)
        distribution = np.full(
            self.K, self.anchor_uniform_mass / self.K, dtype=np.float64
        )
        distribution[self.nominal_action(state)] += 1.0 - self.anchor_uniform_mass
        return distribution

    @cached_property
    def anchor_distributions(self) -> FloatArray:
        distributions = np.vstack(
            [self.anchor_distribution(state) for state in range(self.K)]
        )
        distributions.setflags(write=False)
        return distributions

    def transition_outcomes(
        self, state: int, action: int
    ) -> tuple[IntArray, FloatArray]:
        """Return unique successor states and their probabilities."""

        self._validate_state_action(state, action)
        mass: dict[int, float] = {}
        for offset, probability in zip(
            self.noise_offsets, self.noise_probabilities, strict=True
        ):
            successor = (int(state) + int(action) + int(offset)) % self.K
            mass[successor] = mass.get(successor, 0.0) + float(probability)
        successors = np.fromiter(mass, dtype=np.int64)
        probabilities = np.fromiter(mass.values(), dtype=np.float64)
        return successors, probabilities

    def transition_probabilities(self, state: int, action: int) -> FloatArray:
        """Return one dense column of the MDP kernel ``P(· | s,a)``."""

        successors, probabilities = self.transition_outcomes(state, action)
        column = np.zeros(self.K, dtype=np.float64)
        column[successors] = probabilities
        return column

    def transition_kernel(self) -> FloatArray:
        """Materialize ``P[s,a,s']`` for small validation cases.

        Production planners use :meth:`expected_action_values` and never need
        this cubic tensor.  The explicit method exists as a dense correctness
        reference and makes the ``P``/``P_G`` distinction testable.
        """

        kernel = np.zeros((self.K, self.K, self.K), dtype=np.float64)
        for state in range(self.K):
            for action in range(self.K):
                successors, probabilities = self.transition_outcomes(state, action)
                kernel[state, action, successors] = probabilities
        return kernel

    def expected_action_values(
        self,
        values: ArrayLike,
        gamma: float,
        *,
        states: ArrayLike | None = None,
        actions: ArrayLike | None = None,
        counter: OperationCounters | None = None,
    ) -> FloatArray:
        """Compute exact ``r(s,a) + gamma E_P[V(S')]`` on a Cartesian set.

        One transition-oracle call and one action-value evaluation are charged
        per requested state-action pair.  Repeated action indices are retained
        and charged repeatedly, as they would be for separate oracle queries.
        """

        value = np.asarray(values, dtype=np.float64)
        if value.shape != (self.K,) or not np.all(np.isfinite(value)):
            raise ValueError("values must be a finite vector of length K")
        if not 0.0 <= gamma < 1.0 or not math.isfinite(gamma):
            raise ValueError("gamma must be finite and lie in [0, 1)")
        state_indices = self._indices(states, "states")
        action_indices = self._indices(actions, "actions")
        base = (state_indices[:, None] + action_indices[None, :]) % self.K
        expectation = np.zeros(base.shape, dtype=np.float64)
        for offset, probability in zip(
            self.noise_offsets, self.noise_probabilities, strict=True
        ):
            expectation += float(probability) * value[(base + int(offset)) % self.K]
        result = (
            self.reward_matrix[np.ix_(state_indices, action_indices)]
            + gamma * expectation
        )
        if counter is not None:
            queries = int(result.size)
            counter.transition_calls += queries
            counter.action_evaluations += queries
            for action in action_indices:
                counter.touch_action(int(action), evaluations=0)
        return result

    def sample_transition(
        self,
        state: int,
        action: int,
        rng: np.random.Generator,
        *,
        counter: OperationCounters | None = None,
    ) -> tuple[float, int]:
        """Draw one independent generative-model sample."""

        self._validate_state_action(state, action)
        noise_index = int(rng.choice(len(self.noise_offsets), p=self.noise_probabilities))
        successor = (
            int(state) + int(action) + int(self.noise_offsets[noise_index])
        ) % self.K
        if counter is not None:
            counter.transition_calls += 1
            counter.touch_action(int(action))
        return self.reward(state, action), successor

    def _indices(self, values: ArrayLike | None, name: str) -> IntArray:
        if values is None:
            return np.arange(self.K, dtype=np.int64)
        indices = np.asarray(values, dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= self.K):
            raise ValueError(f"{name} must be one-dimensional indices in Z_K")
        return indices

    def _validate_state(self, state: int) -> None:
        if not isinstance(state, (int, np.integer)) or not 0 <= int(state) < self.K:
            raise IndexError(state)

    def _validate_state_action(self, state: int, action: int) -> None:
        self._validate_state(state)
        if not isinstance(action, (int, np.integer)) or not 0 <= int(action) < self.K:
            raise IndexError(action)


# A concise alias is convenient in experiment configuration and notebooks.
RingControl = RingControlMDP
