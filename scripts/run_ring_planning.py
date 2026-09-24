"""Run M5 ring planning and write raw, summary, and Figure 3 artifacts."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

_cache_root = Path(tempfile.gettempdir()) / "lot-experiments-cache"
_cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root))

from lot_experiments.cli import configured_parser, resolve_cli
from lot_experiments.plotting.planning import plot_ring_planning_figure
from lot_experiments.ring_planning import (
    resolve_ring_planning_config,
    run_ring_planning,
    summarize_ring_planning,
    write_ring_summary_atomic,
)


def main() -> None:
    parser = configured_parser("Run the end-to-end stochastic ring-control study")
    _, config = resolve_cli(parser)
    import logging

    logging.getLogger("fontTools").setLevel(logging.WARNING)
    resolved = resolve_ring_planning_config(config)
    raw = run_ring_planning(resolved)
    summary = summarize_ring_planning(raw, resolved_config=resolved)
    write_ring_summary_atomic(summary, resolved["summary_output"])
    plot_ring_planning_figure(
        summary,
        resolved,
        png_path=resolved["figure_png"],
    )


if __name__ == "__main__":
    main()
