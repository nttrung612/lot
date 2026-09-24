"""Figure 1 draft for the kernel design-map study."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lot_experiments.plotting.common import (
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_LINESTYLES,
    apply_paper_style,
)


METHOD_ORDER = (
    "exact_heat",
    "truncated_heat",
    "uniform_maxent",
    "diffusion_distance_gibbs",
)

FAMILY_LABELS = {
    "path": "Path",
    "cycle": "Cycle",
    "grid": "Grid",
    "torus": "Torus",
    "random_regular_expander": "Random regular",
}


def _near(values: pd.Series, target: float) -> pd.Series:
    return np.isclose(values.astype(float), float(target), rtol=0.0, atol=1e-12)


def _atomic_save(figure: plt.Figure, destination: str | Path) -> Path:
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=output_path.suffix, dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        figure.savefig(temporary_path, bbox_inches="tight")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


def _plot_method_lines(
    axis: plt.Axes,
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
) -> None:
    for method in METHOD_ORDER:
        method_frame = frame.loc[frame["method"] == method].sort_values(x)
        if method_frame.empty:
            continue
        axis.plot(
            method_frame[x],
            method_frame[y],
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            marker="o",
            label=METHOD_LABELS[method],
        )


def plot_kernel_map_figure(
    summary: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    png_path: str | Path,
) -> Path:
    """Create Figure 1 from an existing summary; never rerun simulation."""

    if summary.empty:
        raise ValueError("kernel-map summary is empty")
    required = {
        "method",
        "target",
        "graph_family",
        "requested_K",
        "K",
        "alpha",
        "poisson_mean",
        "kernel_effective_count",
        "one_column_seconds",
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"kernel-map summary is missing columns: {sorted(missing)}")
    figure_config = dict(config["figure"])
    alpha = float(figure_config["alpha"])
    theta = float(figure_config["poisson_mean"])
    primary_family = str(figure_config["primary_graph_family"])
    comparison_K = int(figure_config["comparison_requested_K"])

    apply_paper_style()
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.5))

    fixed = summary.loc[
        (summary["graph_family"] == primary_family)
        & _near(summary["alpha"], alpha)
        & _near(summary["poisson_mean"], theta)
    ]
    _plot_method_lines(
        axes[0, 0], fixed, x="K", y="kernel_effective_count"
    )
    axes[0, 0].set_xscale("log", base=2)
    axes[0, 0].set_yscale("log", base=2)
    axes[0, 0].set_xlabel("Number of actions, K")
    axes[0, 0].set_ylabel(r"Effective count, $d_{\mathrm{ker}}(\alpha)$")
    axes[0, 0].set_title(
        f"A  Scaling on {FAMILY_LABELS.get(primary_family, primary_family)}"
    )

    primary = summary.loc[
        (summary["graph_family"] == primary_family)
        & _near(summary["poisson_mean"], theta)
    ].copy()
    largest_K = int(primary["K"].max())
    primary = primary.loc[primary["K"] == largest_K]
    primary["log_inverse_alpha"] = np.log(1.0 / primary["alpha"].astype(float))
    _plot_method_lines(
        axes[0, 1], primary, x="log_inverse_alpha", y="kernel_effective_count"
    )
    axes[0, 1].set_yscale("log", base=2)
    axes[0, 1].set_xlabel(r"$\log(1/\alpha)$")
    axes[0, 1].set_ylabel(r"Effective count, $d_{\mathrm{ker}}(\alpha)$")
    axes[0, 1].set_title(f"B  Accuracy at K={largest_K}")

    comparison = summary.loc[
        (summary["requested_K"] == comparison_K)
        & _near(summary["alpha"], alpha)
        & _near(summary["poisson_mean"], theta)
    ].copy()
    family_order = [
        family
        for family in ("path", "grid", "torus", "random_regular_expander")
        if family in set(comparison["graph_family"])
    ]
    positions = np.arange(len(family_order), dtype=float)
    width = 0.19
    for method_index, method in enumerate(METHOD_ORDER):
        values = []
        for family in family_order:
            selected = comparison.loc[
                (comparison["graph_family"] == family)
                & (comparison["method"] == method),
                "kernel_effective_fraction",
            ]
            values.append(float(selected.iloc[0]) if len(selected) else np.nan)
        axes[1, 0].bar(
            positions + (method_index - 1.5) * width,
            values,
            width=width,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    axes[1, 0].set_xticks(
        positions,
        [FAMILY_LABELS.get(family, family) for family in family_order],
        rotation=15,
        ha="right",
    )
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].set_ylabel(r"Retained fraction, $d_{\mathrm{ker}}/K$")
    axes[1, 0].set_title(f"C  Geometry control near K={comparison_K}")

    _plot_method_lines(
        axes[1, 1], fixed, x="K", y="one_column_seconds"
    )
    axes[1, 1].set_xscale("log", base=2)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("Number of actions, K")
    axes[1, 1].set_ylabel("Uncached column time (s)")
    axes[1, 1].set_title("D  One-column construction")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=False,
    )
    figure.text(
        0.5,
        -0.005,
        (
            "Methods define different kernel targets; panels compare concentration and "
            "construction cost, not exact-heat estimation error. "
            f"Fixed panels use alpha={alpha:g}, theta={theta:g}."
        ),
        ha="center",
        va="top",
        fontsize=7,
    )
    figure.tight_layout(rect=(0.0, 0.045, 1.0, 0.94))
    png = _atomic_save(figure, png_path)
    plt.close(figure)
    return png
