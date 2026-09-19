"""Run M2 and write raw Parquet plus the compact CSV summary."""

from __future__ import annotations

import os

# Runtime comparisons default to one process and single-threaded numerical
# libraries. These variables must be set before importing NumPy/SciPy through
# the package.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

from lot_experiments.cli import configured_parser, resolve_cli
from lot_experiments.kernel_map import (
    resolve_kernel_map_config,
    run_kernel_map,
    summarize_kernel_map,
    write_summary_atomic,
)


def main() -> None:
    parser = configured_parser("Run the kernel design-map experiment")
    _, config = resolve_cli(parser)
    resolved = resolve_kernel_map_config(config)
    raw = run_kernel_map(resolved)
    summary = summarize_kernel_map(raw, resolved_config=resolved)
    write_summary_atomic(summary, resolved["summary_output"])


if __name__ == "__main__":
    main()
