"""Figure 2 for sharp pruning and normalized Poisson truncation."""

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

from lot_experiments.plotting.common import apply_paper_style
from lot_experiments.pruning import sharp_pruning_error


BASELINE_COLORS = {
    "top_kernel_weights": "#1f77b4",
    "smallest_graph_ball": "#2ca02c",
    "uniform_random_subset": "#ff7f0e",
    "farthest_action_subset": "#d62728",
}
BASELINE_LABELS = {
    "top_kernel_weights": "Top heat weights",
    "smallest_graph_ball": "Graph ball",
    "uniform_random_subset": "Uniform random subset",
    "farthest_action_subset": "Farthest subset",
}
ERROR_COLORS = {
    "kernel": "#1f77b4",
    "backup": "#2ca02c",
    "fixed_point": "#9467bd",
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


def _positive_line(
    axis: plt.Axes,
    x: pd.Series,
    y: pd.Series,
    *,
    label: str,
    color: str,
    linestyle: str = "-",
    marker: str | None = "o",
) -> None:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    keep = np.isfinite(x_values) & np.isfinite(y_values) & (y_values > 0.0)
    if np.any(keep):
        axis.plot(
            x_values[keep],
            y_values[keep],
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
        )


def plot_pruning_figure(
    summary: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    png_path: str | Path,
    pdf_path: str | Path,
) -> tuple[Path, Path]:
    """Create Figure 2 from the M3 summary without rerunning experiments."""

    required = {
        "experiment_part",
        "method",
        "target",
        "reference_target",
        "graph_family",
        "span_over_temperature",
        "observed_pruning_error_median",
        "theoretical_pruning_error_median",
        "omitted_mass_median",
        "radius",
        "poisson_mean",
        "heat_approximation_error_median",
        "kernel_l1_bound_median",
        "backup_error_median",
        "backup_error_bound_median",
        "fixed_point_value_error_median",
        "fixed_point_value_bound_median",
        "graph_ball_size_median",
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"pruning summary is missing columns: {sorted(missing)}")
    if summary.empty:
        raise ValueError("pruning summary is empty")
    pruning_rows = summary.loc[
        summary["experiment_part"].isin(["worst_case", "typical"])
    ]
    if set(pruning_rows["target"].dropna()) != {"exact_heat"}:
        raise ValueError("fixed-set pruning panels must retain the exact-heat target")
    truncation_rows = summary.loc[
        summary["experiment_part"] == "poisson_truncation"
    ]
    if set(truncation_rows["target"].dropna()) != {"truncated_heat"} or set(
        truncation_rows["reference_target"].dropna()
    ) != {"exact_heat"}:
        raise ValueError("Poisson rows must identify truncated and exact-heat targets")

    figure_config = dict(config["figure"])
    typical_ratio = float(figure_config["typical_span_over_temperature"])
    truncation_family = str(figure_config["truncation_graph_family"])
    truncation_theta = float(figure_config["truncation_poisson_mean"])

    apply_paper_style()
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))

    worst = summary.loc[summary["experiment_part"] == "worst_case"]
    axes[0, 0].scatter(
        worst["theoretical_pruning_error_median"],
        worst["observed_pruning_error_median"],
        c=worst["span_over_temperature"],
        cmap="viridis",
        s=23,
        zorder=3,
    )
    maximum = float(
        np.nanmax(
            np.concatenate(
                [
                    worst["theoretical_pruning_error_median"].to_numpy(float),
                    worst["observed_pruning_error_median"].to_numpy(float),
                ]
            )
        )
    )
    axes[0, 0].plot([0.0, maximum], [0.0, maximum], "k--", label="Identity")
    axes[0, 0].set_xlabel("Sharp formula")
    axes[0, 0].set_ylabel("Measured pruning error")
    axes[0, 0].set_title("A  Adversarial construction")
    axes[0, 0].legend(frameon=False)

    typical = summary.loc[
        (summary["experiment_part"] == "typical")
        & _near(summary["span_over_temperature"], typical_ratio)
    ]
    markers = {"cycle": "o", "grid": "s", "torus": "D"}
    for family in sorted(typical["graph_family"].unique()):
        for method, color in BASELINE_COLORS.items():
            selected = typical.loc[
                (typical["graph_family"] == family) & (typical["method"] == method)
            ].sort_values("omitted_mass_median")
            if selected.empty:
                continue
            axes[0, 1].plot(
                selected["omitted_mass_median"],
                selected["observed_pruning_error_median"],
                color=color,
                marker=markers.get(family, "o"),
                linestyle="-" if family == "cycle" else ":",
                label=f"{BASELINE_LABELS[method]} ({family})",
            )
    positive_mass = typical.loc[typical["omitted_mass_median"] > 0, "omitted_mass_median"]
    if len(positive_mass):
        maximum_alpha = min(float(positive_mass.max()), 1.0 - 1e-12)
        alpha_grid = np.geomspace(float(positive_mass.min()), maximum_alpha, 200)
        T0 = float(config["temperature"])
        certificate = [
            sharp_pruning_error(alpha, typical_ratio * T0, T0) for alpha in alpha_grid
        ]
        axes[0, 1].plot(
            alpha_grid,
            certificate,
            color="black",
            linestyle="--",
            label="Worst-case certificate",
        )
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("Actual omitted heat mass")
    axes[0, 1].set_ylabel("One-backup error")
    axes[0, 1].set_title(f"B  Smooth values, S/T0={typical_ratio:g}")
    axes[0, 1].legend(frameon=False, fontsize=5.5, ncol=2)

    truncated = truncation_rows.loc[
        (truncation_rows["graph_family"] == truncation_family)
        & _near(truncation_rows["poisson_mean"], truncation_theta)
    ].sort_values("radius")
    metrics = (
        ("kernel", "heat_approximation_error_median", "kernel_l1_bound_median", "Kernel L1"),
        ("backup", "backup_error_median", "backup_error_bound_median", "One backup"),
        (
            "fixed_point",
            "fixed_point_value_error_median",
            "fixed_point_value_bound_median",
            "Fixed-point value",
        ),
    )
    for name, observed_column, bound_column, label in metrics:
        _positive_line(
            axes[1, 0],
            truncated["radius"],
            truncated[observed_column],
            label=label,
            color=ERROR_COLORS[name],
        )
        _positive_line(
            axes[1, 0],
            truncated["radius"],
            truncated[bound_column],
            label=f"{label} bound",
            color=ERROR_COLORS[name],
            linestyle="--",
            marker=None,
        )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("Poisson truncation radius, r")
    axes[1, 0].set_ylabel("Error to exact heat")
    axes[1, 0].set_title(
        f"C  Error propagation ({truncation_family}, theta={truncation_theta:g})"
    )
    axes[1, 0].legend(frameon=False, fontsize=6, ncol=2)

    for name, observed_column, _, label in metrics:
        _positive_line(
            axes[1, 1],
            truncated["graph_ball_size_median"],
            truncated[observed_column],
            label=label,
            color=ERROR_COLORS[name],
        )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("Maximum graph-ball size")
    axes[1, 1].set_ylabel("Error to exact heat")
    axes[1, 1].set_title("D  Accuracy versus local action count")
    axes[1, 1].legend(frameon=False)

    figure.text(
        0.5,
        -0.002,
        (
            "A-B prune unnormalized exact-heat weights (same target). C-D compare "
            "the truncated-heat fixed point with FullExactHeat; beta_r is distinct "
            "from the pruning mass alpha."
        ),
        ha="center",
        va="top",
        fontsize=7,
    )
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    png = _atomic_save(figure, png_path)
    pdf = _atomic_save(figure, pdf_path)
    plt.close(figure)
    return png, pdf
