"""Run M3 and write raw Parquet plus the compact CSV summary."""

from __future__ import annotations

import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

from lot_experiments.cli import configured_parser, resolve_cli
from lot_experiments.pruning_experiment import (
    resolve_pruning_config,
    run_pruning_experiment,
    summarize_pruning,
    write_pruning_summary_atomic,
)


def main() -> None:
    parser = configured_parser("Run sharp-pruning and Poisson-truncation experiments")
    _, config = resolve_cli(parser)
    resolved = resolve_pruning_config(config)
    raw = run_pruning_experiment(resolved)
    summary = summarize_pruning(raw, resolved_config=resolved)
    write_pruning_summary_atomic(summary, resolved["summary_output"])


if __name__ == "__main__":
    main()
