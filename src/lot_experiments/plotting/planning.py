"""Figure 3 for end-to-end synthetic ring-control planning."""

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


COLORS = {
    "full_exact_heat": "#1f77b4",
    "dense_second_order": "#4c78a8",
    "exact_heat_topmass": "#2ca02c",
    "truncated_heat_local": "#17becf",
    "poisson_endpoint_mc": "#9467bd",
    "uniform_action_mc": "#ff7f0e",
    "random_subset": "#d62728",
}
LABELS = {
    "full_exact_heat": "Full exact heat",
    "dense_second_order": "Dense full-action",
    "exact_heat_topmass": "Exact-heat top mass",
    "truncated_heat_local": "Truncated local heat",
    "poisson_endpoint_mc": "Poisson endpoint MC",
    "uniform_action_mc": "Uniform action MC",
    "random_subset": "Random subset",
}
LINESTYLES = {
    "full_exact_heat": "-",
    "dense_second_order": "-",
    "exact_heat_topmass": "-",
    "truncated_heat_local": "-",
    "poisson_endpoint_mc": "--",
    "uniform_action_mc": "--",
    "random_subset": "--",
}


def _atomic_save(figure: plt.Figure, destination: str | Path) -> Path:
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=output.suffix, dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(temporary, bbox_inches="tight")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _near(series: pd.Series, value: float) -> pd.Series:
    return np.isclose(series.astype(float), value, atol=1e-12, rtol=0.0)


def _method_lines(
    axis: plt.Axes,
    frame: pd.DataFrame,
    x: str,
    y: str,
    *,
    log_x: bool = False,
    log_y: bool = False,
) -> None:
    for method in LABELS:
        selected = frame.loc[frame["method"] == method].sort_values(x)
        if selected.empty:
            continue
        x_values = selected[x].to_numpy(float)
        y_values = selected[y].to_numpy(float)
        keep = np.isfinite(x_values) & np.isfinite(y_values)
        if log_x:
            keep &= x_values > 0.0
        if log_y:
            keep &= y_values > 0.0
        if not np.any(keep):
            continue
        axis.plot(
            x_values[keep],
            y_values[keep],
            marker="o",
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            label=LABELS[method],
        )
    if log_x:
        axis.set_xscale("log")
    if log_y:
        axis.set_yscale("log")


def plot_ring_planning_figure(
    summary: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    png_path: str | Path,
    pdf_path: str | Path,
) -> tuple[Path, Path]:
    """Render Figure 3 from summary data only."""

    required = {
        "experiment_part",
        "method",
        "target",
        "reference_target",
        "graph_family",
        "K",
        "temperature",
        "poisson_mean",
        "epsilon",
        "radius_median",
        "absolute_value_error_median",
        "statistical_error_median",
        "heat_approximation_error_median",
        "transition_calls_median",
        "action_evaluations_median",
        "unique_actions_touched_median",
        "online_seconds_median",
        "graph_ball_size_median",
        "timing_mode",
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"ring summary is missing columns: {sorted(missing)}")
    if summary.empty:
        raise ValueError("ring summary is empty")
    primary = summary.loc[summary["experiment_part"] == "primary"].copy()
    if primary.empty:
        raise ValueError("ring summary has no primary rows")
    exact_estimators = primary.loc[primary["target"] == "exact_heat"]
    if set(exact_estimators["reference_target"].dropna()) != {"exact_heat"}:
        raise ValueError("exact-heat estimator rows must reference FullExactHeat")
    local = primary.loc[primary["method"].isin(["truncated_heat_local", "full_truncated_heat"])]
    if not local.empty and set(local["target"].dropna()) != {"truncated_heat"}:
        raise ValueError("local finite-walk rows must retain the truncated-heat target")

    figure_config = config["figure"]
    selected = primary.loc[
        _near(primary["temperature"], float(figure_config["temperature"]))
        & _near(primary["poisson_mean"], float(figure_config["poisson_mean"]))
        & _near(primary["epsilon"], float(figure_config["epsilon"]))
    ].copy()
    if selected.empty:
        raise ValueError("figure selection contains no primary rows")

    apply_paper_style()
    figure, axes = plt.subplots(2, 3, figsize=(10.4, 6.2))
    _method_lines(
        axes[0, 0], selected, "transition_calls_median", "absolute_value_error_median", log_x=True, log_y=True
    )
    axes[0, 0].set_xlabel("Transition-oracle calls")
    axes[0, 0].set_ylabel("Root-value error to exact heat")
    axes[0, 0].set_title("A  Error versus transition calls")
    axes[0, 0].axhline(
        float(figure_config["epsilon"]), color="black", linestyle=":", label="Requested epsilon"
    )

    _method_lines(
        axes[0, 1], selected, "action_evaluations_median", "absolute_value_error_median", log_x=True, log_y=True
    )
    axes[0, 1].set_xlabel("Action-value evaluations")
    axes[0, 1].set_ylabel("Root-value error to exact heat")
    axes[0, 1].set_title("B  Error versus action evaluations")
    axes[0, 1].axhline(float(figure_config["epsilon"]), color="black", linestyle=":")

    _method_lines(axes[0, 2], selected, "K", "unique_actions_touched_median", log_x=True, log_y=True)
    axes[0, 2].set_xlabel("Number of actions, K")
    axes[0, 2].set_ylabel("Unique actions touched")
    axes[0, 2].set_title("C  Action scaling")

    isolated_timing = selected.loc[selected["timing_mode"] == "isolated"]
    if isolated_timing.empty:
        axes[1, 0].text(
            0.5,
            0.5,
            "Concurrent run: use workers=1\nfor paper-ready runtime",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
        )
    else:
        _method_lines(
            axes[1, 0],
            isolated_timing,
            "K",
            "online_seconds_median",
            log_x=True,
            log_y=True,
        )
    axes[1, 0].set_xlabel("Number of actions, K")
    axes[1, 0].set_ylabel("Online planning time (s)")
    axes[1, 0].set_title("D  Runtime scaling")

    decomposition = primary.loc[primary["method"] == "truncated_heat_local"].sort_values("radius_median")
    for column, label, color in (
        ("statistical_error_median", "Empirical estimation", "#9467bd"),
        ("heat_approximation_error_median", "Heat bias", "#2ca02c"),
        ("absolute_value_error_median", "Total error", "#1f77b4"),
    ):
        values = decomposition[column].to_numpy(float)
        radii = decomposition["radius_median"].to_numpy(float)
        keep = np.isfinite(values) & np.isfinite(radii) & (values > 0.0)
        if np.any(keep):
            axes[1, 1].plot(radii[keep], values[keep], marker="o", label=label, color=color)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("Finite-walk radius, r")
    axes[1, 1].set_ylabel("Root-value error")
    axes[1, 1].set_title("E  Statistical/heat decomposition")

    controls = summary.loc[
        (summary["experiment_part"] == "geometry_control")
        & (summary["method"] == "truncated_heat_local")
    ].sort_values("graph_family")
    if not controls.empty:
        positions = np.arange(len(controls))
        axes[1, 2].bar(
            positions,
            controls["graph_ball_size_median"],
            color=["#2ca02c", "#d62728", "#7f7f7f"][: len(controls)],
        )
        axes[1, 2].set_xticks(positions, controls["graph_family"], rotation=20, ha="right")
    axes[1, 2].set_ylabel("Maximum local ball size")
    axes[1, 2].set_title("F  Geometry controls (distinct targets)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.925),
            ncol=4,
            frameon=False,
        )
    figure.suptitle(
        "Ring planning: primary errors reference FullExactHeat; geometry controls use distinct targets",
        y=0.985,
        fontsize=9,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
    png = _atomic_save(figure, png_path)
    pdf = _atomic_save(figure, pdf_path)
    plt.close(figure)
    return png, pdf
