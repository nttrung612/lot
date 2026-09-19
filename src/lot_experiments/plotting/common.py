"""Shared publication plotting style."""

from __future__ import annotations

import matplotlib as mpl


METHOD_COLORS = {
    "exact_heat": "#1f77b4",
    "truncated_heat": "#2ca02c",
    "uniform_maxent": "#7f7f7f",
    "diffusion_distance_gibbs": "#d62728",
}

METHOD_LABELS = {
    "exact_heat": "Exact heat",
    "truncated_heat": "Truncated local heat",
    "uniform_maxent": "Uniform MaxEnt",
    "diffusion_distance_gibbs": "Diffusion Gibbs",
}

METHOD_LINESTYLES = {
    "exact_heat": "-",
    "truncated_heat": "-",
    "uniform_maxent": "--",
    "diffusion_distance_gibbs": "--",
}


def apply_paper_style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.6,
            "lines.markersize": 4,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

