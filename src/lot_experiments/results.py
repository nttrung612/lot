"""Common run-level result schema and atomic persistence."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RESULT_COLUMNS = (
    "experiment",
    "run_id",
    "seed",
    "method",
    "target",
    "graph_family",
    "K",
    "alpha",
    "radius",
    "diffusion_time",
    "poisson_mean",
    "temperature",
    "gamma",
    "epsilon",
    "delta",
    "root_state",
    "value_estimate",
    "reference_value",
    "absolute_value_error",
    "statistical_error",
    "heat_approximation_error",
    "policy_l1_error",
    "transition_calls",
    "action_evaluations",
    "unique_actions_touched",
    "graph_neighbor_accesses",
    "dense_linear_algebra_operations",
    "geometry_preprocess_seconds",
    "online_seconds",
    "peak_memory_mb",
    "status",
    "git_commit",
    "config_json",
)

RECOMMENDED_TARGETS = frozenset(
    {"exact_heat", "truncated_heat", "maxent", "diffusion_gibbs", "hard_max"}
)


class ResultSchemaError(ValueError):
    pass


def result_row(**values: Any) -> dict[str, Any]:
    """Construct a complete row; unspecified nullable measurements use NaN."""

    row = {column: np.nan for column in RESULT_COLUMNS}
    row.update(values)
    unknown = set(row) - set(RESULT_COLUMNS)
    if unknown:
        raise ResultSchemaError(f"unknown result fields: {sorted(unknown)}")
    missing_identity = [
        field
        for field in ("experiment", "run_id", "seed", "method", "target", "status", "config_json")
        if pd.isna(row[field]) or row[field] == ""
    ]
    if missing_identity:
        raise ResultSchemaError(f"missing identity fields: {missing_identity}")
    return row


def validate_results(rows: pd.DataFrame | Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    missing = set(RESULT_COLUMNS) - set(frame.columns)
    extra = set(frame.columns) - set(RESULT_COLUMNS)
    if missing or extra:
        raise ResultSchemaError(f"schema mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    if frame["target"].isna().any() or (frame["target"] == "").any():
        raise ResultSchemaError("every result row must have a target")
    return frame.loc[:, RESULT_COLUMNS]


def require_common_target(
    rows: pd.DataFrame | Sequence[Mapping[str, Any]], *, expected: str | None = None
) -> str:
    """Reject error comparisons that silently mix different estimands."""

    frame = validate_results(rows)
    targets = set(frame["target"])
    if not targets:
        raise ResultSchemaError("comparison contains no result rows")
    if len(targets) != 1:
        raise ResultSchemaError(f"comparison mixes targets: {sorted(targets)}")
    target = next(iter(targets))
    if expected is not None and target != expected:
        raise ResultSchemaError(f"comparison target is {target!r}, expected {expected!r}")
    return target


def write_results_atomic(
    rows: pd.DataFrame | Sequence[Mapping[str, Any]], destination: str | Path
) -> Path:
    frame = validate_results(rows)
    output_path = Path(destination)
    if output_path.suffix not in {".csv", ".parquet"}:
        raise ValueError("results must be written as .csv or .parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        if output_path.suffix == ".parquet":
            frame.to_parquet(temporary_path, index=False)
        else:
            frame.to_csv(temporary_path, index=False)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path
