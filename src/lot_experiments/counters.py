"""Shared operation accounting for all future planners."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from time import perf_counter
from typing import Iterator

import psutil


@dataclass
class OperationCounters:
    transition_calls: int = 0
    action_evaluations: int = 0
    graph_neighbor_accesses: int = 0
    dense_linear_algebra_operations: int = 0
    geometry_preprocess_seconds: float = 0.0
    online_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    _actions_touched: set[int] = field(default_factory=set, repr=False)

    @property
    def unique_actions_touched(self) -> int:
        return len(self._actions_touched)

    def touch_action(self, action: int, *, evaluations: int = 1) -> None:
        if evaluations < 0:
            raise ValueError("evaluations must be nonnegative")
        self._actions_touched.add(int(action))
        self.action_evaluations += evaluations

    def touch_actions(self, actions: set[int] | list[int] | tuple[int, ...]) -> None:
        action_set = {int(action) for action in actions}
        self._actions_touched.update(action_set)
        self.action_evaluations += len(action_set)

    def observe_memory(self) -> float:
        rss_mb = psutil.Process().memory_info().rss / (1024.0**2)
        self.peak_memory_mb = max(self.peak_memory_mb, rss_mb)
        return rss_mb

    @contextmanager
    def time_geometry(self) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            self.geometry_preprocess_seconds += perf_counter() - start
            self.observe_memory()

    @contextmanager
    def time_online(self) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            self.online_seconds += perf_counter() - start
            self.observe_memory()

    def merge(self, other: "OperationCounters") -> None:
        for item in fields(self):
            if item.name in {"_actions_touched", "peak_memory_mb"}:
                continue
            setattr(self, item.name, getattr(self, item.name) + getattr(other, item.name))
        self._actions_touched.update(other._actions_touched)
        self.peak_memory_mb = max(self.peak_memory_mb, other.peak_memory_mb)

    def as_dict(self) -> dict[str, int | float]:
        return {
            "transition_calls": self.transition_calls,
            "action_evaluations": self.action_evaluations,
            "unique_actions_touched": self.unique_actions_touched,
            "graph_neighbor_accesses": self.graph_neighbor_accesses,
            "dense_linear_algebra_operations": self.dense_linear_algebra_operations,
            "geometry_preprocess_seconds": self.geometry_preprocess_seconds,
            "online_seconds": self.online_seconds,
            "peak_memory_mb": self.peak_memory_mb,
        }

