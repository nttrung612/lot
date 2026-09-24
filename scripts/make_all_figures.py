"""Regenerate implemented paper figures from stored result artifacts only."""

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
from lot_experiments.geometry_learning import resolve_geometry_learning_config
from lot_experiments.kernel_map import resolve_kernel_map_config
from lot_experiments.plotting.kernel_map import plot_kernel_map_figure
from lot_experiments.plotting.learning import plot_geometry_learning_figure
from lot_experiments.plotting.planning import plot_ring_planning_figure
from lot_experiments.plotting.pendulum import plot_pendulum_figure
from lot_experiments.pendulum_experiment import resolve_pendulum_experiment_config
from lot_experiments.plotting.pruning import plot_pruning_figure
from lot_experiments.plotting.resistance import plot_resistance_figure
from lot_experiments.pruning_experiment import resolve_pruning_config
from lot_experiments.resistance import resolve_resistance_config
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
    parser.add_argument(
        "--geometry-learning-config",
        type=Path,
        default=Path("configs/geometry_learning.yaml"),
    )
    parser.add_argument(
        "--pendulum-config", type=Path, default=Path("configs/pendulum.yaml")
    )
    parser.add_argument(
        "--resistance-config", type=Path, default=Path("configs/resistance.yaml")
    )
    arguments = parser.parse_args()
    resolved = resolve_kernel_map_config(load_config(arguments.kernel_map_config))
    summary = pd.read_csv(resolved["summary_output"])
    plot_kernel_map_figure(
        summary,
        resolved,
        png_path=resolved["figure_png"],
    )
    pruning = resolve_pruning_config(load_config(arguments.pruning_config))
    pruning_summary = pd.read_csv(pruning["summary_output"])
    plot_pruning_figure(
        pruning_summary,
        pruning,
        png_path=pruning["figure_png"],
    )
    ring = resolve_ring_planning_config(load_config(arguments.ring_planning_config))
    ring_summary = pd.read_csv(ring["summary_output"])
    plot_ring_planning_figure(
        ring_summary,
        ring,
        png_path=ring["figure_png"],
    )
    geometry = resolve_geometry_learning_config(
        load_config(arguments.geometry_learning_config)
    )
    geometry_raw = pd.read_parquet(geometry["raw_output"])
    geometry_table = pd.read_csv(geometry["table_output"])
    plot_geometry_learning_figure(
        geometry_raw,
        geometry_table,
        geometry,
        png_path=geometry["figure_png"],
    )
    pendulum = resolve_pendulum_experiment_config(
        load_config(arguments.pendulum_config)
    )
    pendulum_summary = pd.read_csv(pendulum["summary_output"])
    plot_pendulum_figure(
        pendulum_summary,
        pendulum,
        png_path=pendulum["figure_png"],
    )
    resistance = resolve_resistance_config(load_config(arguments.resistance_config))
    resistance_raw = pd.read_parquet(resistance["raw_output"])
    plot_resistance_figure(
        resistance_raw,
        resistance,
        png_path=resistance["figure_png"],
    )


if __name__ == "__main__":
    main()
