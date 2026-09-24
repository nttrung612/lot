"""Run M8 Pendulum baselines and render Figure 4."""

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
from lot_experiments.pendulum_experiment import (
    resolve_pendulum_experiment_config,
    run_pendulum_experiment,
)
from lot_experiments.plotting.pendulum import plot_pendulum_figure


def main() -> None:
    parser = configured_parser("Run M8 Pendulum baselines and Figure 4")
    _, config = resolve_cli(parser)
    import logging

    logging.getLogger("fontTools").setLevel(logging.WARNING)
    resolved = resolve_pendulum_experiment_config(config)
    summary = run_pendulum_experiment(resolved)
    plot_pendulum_figure(
        summary,
        resolved,
        png_path=resolved["figure_png"],
    )


if __name__ == "__main__":
    main()
