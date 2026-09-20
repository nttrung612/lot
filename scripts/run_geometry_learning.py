"""Run M6 reward-free geometry learning and write all planned artifacts."""

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
from lot_experiments.geometry_learning import (
    geometry_threshold_table,
    resolve_geometry_learning_config,
    run_geometry_learning,
    summarize_geometry_learning,
    write_geometry_summary_atomic,
    write_geometry_table_atomic,
)
from lot_experiments.plotting.learning import plot_geometry_learning_figure


def main() -> None:
    parser = configured_parser("Run reward-free learned geometry and reward transfer")
    _, config = resolve_cli(parser)
    import logging

    logging.getLogger("fontTools").setLevel(logging.WARNING)
    resolved = resolve_geometry_learning_config(config)
    raw = run_geometry_learning(resolved)
    summary = summarize_geometry_learning(raw, resolved_config=resolved)
    table = geometry_threshold_table(raw, resolved["value_error_thresholds"])
    write_geometry_summary_atomic(summary, resolved["summary_output"])
    write_geometry_table_atomic(table, resolved["table_output"])
    plot_geometry_learning_figure(
        summary,
        table,
        resolved,
        png_path=resolved["figure_png"],
        pdf_path=resolved["figure_pdf"],
    )


if __name__ == "__main__":
    main()
