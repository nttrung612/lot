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
MARKERS = {
    "full_exact_heat": "o",
    "dense_second_order": "o",
    "exact_heat_topmass": "s",
    "truncated_heat_local": "D",
    "poisson_endpoint_mc": "^",
    "uniform_action_mc": "v",
    "random_subset": "P",
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
            marker=MARKERS[method],
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            label=LABELS[method],
        )
    if log_x:
        axis.set_xscale("log")
    if log_y:
        axis.set_yscale("log")


def _method_points_with_iqr(
    axis: plt.Axes,
    frame: pd.DataFrame,
    x: str,
    y: str,
) -> None:
    """Plot one fixed-problem cost/error point per method with IQR whiskers."""

    for method in LABELS:
        selected = frame.loc[frame["method"] == method]
        if selected.empty:
            continue
        row = selected.iloc[0]
        x_value = float(row[x])
        y_value = float(row[y])
        if not (np.isfinite(x_value) and np.isfinite(y_value)) or min(
            x_value, y_value
        ) <= 0.0:
            continue
        low = float(row.get(f"{y.removesuffix('_median')}_q25", np.nan))
        high = float(row.get(f"{y.removesuffix('_median')}_q75", np.nan))
        yerr = None
        if np.isfinite(low) and np.isfinite(high) and low > 0.0:
            yerr = np.array([[y_value - low], [high - y_value]])
        axis.errorbar(
            x_value,
            y_value,
            yerr=yerr,
            marker=MARKERS[method],
            color=COLORS[method],
            linestyle="none",
            markerfacecolor=(
                "white" if LINESTYLES[method] == "--" else COLORS[method]
            ),
            markeredgewidth=1.2,
            capsize=2.5,
            elinewidth=1.0,
            label=LABELS[method],
            zorder=3,
        )


def _actions_per_state_per_backup(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive the typical state-local branching cost from summary medians.

    Every planner charges one transition call per queried state-action pair and
    performs one accounted diagnostic backup after value iteration.  Dividing
    by ``iterations + 1`` therefore removes convergence length, while dividing
    by ``K`` exposes the candidate actions queried at one state in one backup.
    The ratio uses medians already present in the paper summary; raw runs retain
    the exact integer accounting.
    """

    result = frame.copy()
    calls = pd.to_numeric(result["transition_calls_median"], errors="coerce")
    iterations = pd.to_numeric(
        result["fixed_point_iterations_median"], errors="coerce"
    )
    action_count = pd.to_numeric(result["K"], errors="coerce")
    result["actions_per_state_per_backup"] = calls / (
        (iterations + 1.0) * action_count
    )
    return result


def plot_ring_planning_figure(
    summary: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    png_path: str | Path,
) -> Path:
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
        "graph_ball_size_median",
        "fixed_point_iterations_median",
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
    local = primary.loc[
        primary["method"].isin(["truncated_heat_local", "full_truncated_heat"])
    ]
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
    figure = plt.figure(figsize=(10.8, 6.2))
    grid = figure.add_gridspec(
        2, 3, height_ratios=(1.12, 1.0), hspace=0.48, wspace=0.38
    )
    axis_cost = figure.add_subplot(grid[0, :])
    axis_scaling = figure.add_subplot(grid[1, 0])
    axis_decomposition = figure.add_subplot(grid[1, 1])
    axis_controls = figure.add_subplot(grid[1, 2])

    # Compare methods at one fixed problem size.  Connecting configurations
    # with different K turns a scaling sweep into a misleading cost/error
    # trajectory and compresses the competitive methods into an unreadable
    # bundle.  Vertical whiskers show paired-seed IQRs.
    comparison_K = int(pd.to_numeric(selected["K"]).max())
    fixed_problem = selected.loc[pd.to_numeric(selected["K"]) == comparison_K]
    _method_points_with_iqr(
        axis_cost,
        fixed_problem,
        "transition_calls_median",
        "absolute_value_error_median",
    )
    axis_cost.set_xscale("log")
    axis_cost.set_yscale("log")
    axis_cost.set_xlabel("Transition-oracle calls")
    axis_cost.set_ylabel("Root-value error to FullExactHeat")
    axis_cost.set_title(
        f"A  Cost–accuracy at K={comparison_K} (median and IQR)",
        loc="left",
    )
    axis_cost.axhline(
        float(figure_config["epsilon"]),
        color="black",
        linewidth=1.2,
        linestyle=":",
        label=r"Requested $\epsilon$",
        zorder=1,
    )
    dense = fixed_problem.loc[fixed_problem["method"] == "dense_second_order"]
    local_fixed = fixed_problem.loc[
        fixed_problem["method"] == "truncated_heat_local"
    ]
    if not dense.empty and not local_fixed.empty:
        saving = float(dense.iloc[0]["transition_calls_median"]) / float(
            local_fixed.iloc[0]["transition_calls_median"]
        )
        axis_cost.text(
            0.02,
            0.07,
            f"Local heat uses {saving:.0f}× fewer calls than dense full-action",
            transform=axis_cost.transAxes,
            fontsize=7.5,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "0.8",
                "alpha": 0.9,
            },
        )

    # Global ``unique_actions_touched`` saturates at K because the experiment
    # solves all K states, even when every individual backup is local.  Expose
    # the state-local branching cost instead; this is the quantity that the
    # action geometry is intended to reduce.
    action_scaling = _actions_per_state_per_backup(selected)
    action_scaling = action_scaling.loc[
        action_scaling["method"]
        .isin(
            [
                "dense_second_order",
                "exact_heat_topmass",
                "truncated_heat_local",
                "poisson_endpoint_mc",
                "uniform_action_mc",
                "random_subset",
            ]
        )
    ]
    _method_lines(
        axis_scaling,
        action_scaling,
        "K",
        "actions_per_state_per_backup",
        log_x=True,
        log_y=True,
    )
    K_values = np.sort(
        pd.to_numeric(action_scaling["K"], errors="coerce").dropna().unique()
    )
    if len(K_values):
        axis_scaling.set_xscale("log", base=2)
        axis_scaling.set_xticks(K_values)
        axis_scaling.set_xticklabels([f"{value:g}" for value in K_values])
        axis_scaling.tick_params(axis="x", which="minor", bottom=False)
        axis_scaling.plot(
            K_values,
            K_values,
            color="0.25",
            linestyle=":",
            linewidth=1.2,
            zorder=0,
        )
        axis_scaling.text(
            0.97,
            0.94,
            "full action = K",
            ha="right",
            va="top",
            color="0.3",
            transform=axis_scaling.transAxes,
        )
    axis_scaling.set_xlabel("Number of actions, K")
    axis_scaling.set_ylabel("Candidates / state / backup")
    axis_scaling.set_title("B  Local branching versus K", loc="left")

    # Hold K, temperature, and requested epsilon fixed.  The previous plot
    # joined every configuration after sorting only by radius, producing
    # vertical zig-zags between unrelated K/theta/epsilon cases.
    decomposition_K = int(pd.to_numeric(selected["K"]).max())
    decomposition = primary.loc[
        (primary["method"] == "truncated_heat_local")
        & (pd.to_numeric(primary["K"]) == decomposition_K)
        & _near(primary["temperature"], float(figure_config["temperature"]))
        & _near(primary["epsilon"], float(figure_config["epsilon"]))
    ].sort_values(["radius_median", "poisson_mean"])
    x_positions = np.arange(len(decomposition), dtype=float)
    for column, label, color, marker, offset, zorder in (
        (
            "absolute_value_error_median",
            "Total error to exact heat",
            "#1f77b4",
            "o",
            -0.09,
            3,
        ),
        (
            "statistical_error_median",
            "Empirical error to truncated heat",
            "#9467bd",
            "x",
            0.09,
            4,
        ),
        (
            "heat_approximation_error_median",
            "Heat truncation bias",
            "#2ca02c",
            "^",
            0.0,
            2,
        ),
    ):
        values = decomposition[column].to_numpy(float)
        keep = np.isfinite(values) & (values > 0.0)
        if np.any(keep):
            axis_decomposition.plot(
                x_positions[keep] + offset,
                values[keep],
                marker=marker,
                linestyle="none",
                label=label,
                color=color,
                markerfacecolor="white" if marker != "x" else color,
                markeredgewidth=1.2,
                zorder=zorder,
            )
    radii = decomposition["radius_median"].to_numpy(float)
    theta = decomposition["poisson_mean"].to_numpy(float)
    axis_decomposition.set_xticks(
        x_positions,
        [
            f"$\\theta$={value:g}\n$r$={radius:g}"
            for value, radius in zip(theta, radii, strict=True)
        ],
    )
    axis_decomposition.set_yscale("log")
    axis_decomposition.set_xlabel("Heat scale and certified radius")
    axis_decomposition.set_ylabel("Root-value error")
    axis_decomposition.set_title("C  Local error decomposition", loc="left")
    axis_decomposition.legend(loc="best", frameon=False, fontsize=6.2)

    controls = summary.loc[
        (summary["experiment_part"] == "geometry_control")
        & (summary["method"] == "truncated_heat_local")
    ].copy()
    if not controls.empty:
        order = {"cycle": 0, "permuted_cycle": 1, "random_regular_expander": 2}
        controls["_order"] = controls["graph_family"].map(order).fillna(len(order))
        controls = controls.sort_values("_order")
        control_styles = {
            "cycle": ("Aligned cycle", "#2ca02c", "o", (5, 7)),
            "permuted_cycle": ("Permuted cycle", "#d62728", "s", (5, -13)),
            "random_regular_expander": (
                "Random regular\nexpander",
                "#7f7f7f",
                "^",
                (-7, 7),
            ),
        }
        for _, row in controls.iterrows():
            name = str(row["graph_family"])
            label, color, marker, offset = control_styles.get(
                name, (name, "0.3", "o", (5, 5))
            )
            support_fraction = float(row["graph_ball_size_median"]) / float(row["K"])
            bias = float(row["heat_approximation_error_median"])
            axis_controls.scatter(
                support_fraction,
                bias,
                color=color,
                marker=marker,
                s=42,
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )
            axis_controls.annotate(
                label,
                (support_fraction, bias),
                xytext=offset,
                textcoords="offset points",
                ha="left" if offset[0] >= 0 else "right",
                va="bottom" if offset[1] >= 0 else "top",
                fontsize=6.5,
            )
        axis_controls.set_xscale("log")
        axis_controls.set_yscale("log")
        axis_controls.margins(x=0.05, y=0.16)
    axis_controls.set_xlabel("Local support fraction, |ball| / K")
    axis_controls.set_ylabel("Root-value heat bias")
    axis_controls.set_title("D  Geometry controls (graph-specific targets)", loc="left")

    handles, labels = axis_cost.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles, strict=True))
        legend_order = [
            label for label in LABELS.values() if label in by_label
        ] + [r"Requested $\epsilon$"]
        figure.legend(
            [by_label[label] for label in legend_order],
            legend_order,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=7,
            frameon=False,
            fontsize=6.8,
        )
    figure.text(
        0.995,
        0.008,
        "A–C: FullExactHeat reference; D: each graph uses its own exact-heat target.",
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="0.35",
    )
    figure.subplots_adjust(top=0.88, bottom=0.12, left=0.075, right=0.985)
    png = _atomic_save(figure, png_path)
    plt.close(figure)
    return png
