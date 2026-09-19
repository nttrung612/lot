"""Regenerate implemented paper figures from summaries only."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lot_experiments.config import load_config
from lot_experiments.kernel_map import resolve_kernel_map_config
from lot_experiments.plotting.kernel_map import plot_kernel_map_figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-map-config", type=Path, default=Path("configs/kernel_map.yaml"))
    arguments = parser.parse_args()
    resolved = resolve_kernel_map_config(load_config(arguments.kernel_map_config))
    summary = pd.read_csv(resolved["summary_output"])
    plot_kernel_map_figure(
        summary,
        resolved,
        png_path=resolved["figure_png"],
        pdf_path=resolved["figure_pdf"],
    )


if __name__ == "__main__":
    main()

