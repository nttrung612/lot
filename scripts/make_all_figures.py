"""Regenerate implemented paper figures from summaries only."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

# Keep plotting caches writable in sandboxed and clean-checkout environments.
_cache_root = Path(tempfile.gettempdir()) / "lot-experiments-cache"
_cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root))

import pandas as pd

from lot_experiments.config import load_config
from lot_experiments.kernel_map import resolve_kernel_map_config
from lot_experiments.plotting.kernel_map import plot_kernel_map_figure
from lot_experiments.plotting.planning import plot_ring_planning_figure
from lot_experiments.plotting.pruning import plot_pruning_figure
from lot_experiments.pruning_experiment import resolve_pruning_config
from lot_experiments.ring_planning import resolve_ring_planning_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-map-config", type=Path, default=Path("configs/kernel_map.yaml"))
    parser.add_argument("--pruning-config", type=Path, default=Path("configs/pruning.yaml"))
    parser.add_argument(
        "--ring-planning-config",
        type=Path,
        default=Path("configs/ring_planning.yaml"),
    )
    arguments = parser.parse_args()
    resolved = resolve_kernel_map_config(load_config(arguments.kernel_map_config))
    summary = pd.read_csv(resolved["summary_output"])
    plot_kernel_map_figure(
        summary,
        resolved,
        png_path=resolved["figure_png"],
        pdf_path=resolved["figure_pdf"],
    )
    pruning = resolve_pruning_config(load_config(arguments.pruning_config))
    pruning_summary = pd.read_csv(pruning["summary_output"])
    plot_pruning_figure(
        pruning_summary,
        pruning,
        png_path=pruning["figure_png"],
        pdf_path=pruning["figure_pdf"],
    )
    ring = resolve_ring_planning_config(load_config(arguments.ring_planning_config))
    ring_summary = pd.read_csv(ring["summary_output"])
    plot_ring_planning_figure(
        ring_summary,
        ring,
        png_path=ring["figure_png"],
        pdf_path=ring["figure_pdf"],
    )


if __name__ == "__main__":
    main()
