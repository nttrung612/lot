"""Run M9 and write the effective-resistance appendix artifacts."""

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
from lot_experiments.plotting.resistance import plot_resistance_figure
from lot_experiments.resistance import (
    resolve_resistance_config,
    run_resistance_experiment,
    summarize_resistance,
    write_resistance_summary_atomic,
)


def main() -> None:
    parser = configured_parser("Run the LOT effective-resistance identity visualization")
    _, config = resolve_cli(parser)
    import logging

    logging.getLogger("fontTools").setLevel(logging.WARNING)
    resolved = resolve_resistance_config(config)
    raw = run_resistance_experiment(resolved)
    summary = summarize_resistance(raw, resolved_config=resolved)
    write_resistance_summary_atomic(summary, resolved["summary_output"])
    plot_resistance_figure(
        raw,
        resolved,
        png_path=resolved["figure_png"],
    )


if __name__ == "__main__":
    main()
