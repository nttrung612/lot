"""Paper Figure 4 for the Pendulum accuracy/computation frontier."""

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
    "exact_heat_topmass": "#2ca02c",
    "truncated_heat_local": "#17becf",
    "poisson_endpoint_mc": "#9467bd",
    "uniform_random_subset": "#d62728",
    "local_uniform_neighborhood": "#ff7f0e",
    "local_rbf_neighborhood": "#8c564b",
    "uniform_maxent": "#7f7f7f",
    "hard_max": "#111111",
    "squared_torque_lot": "#e377c2",
    "permuted_path_graph": "#bcbd22",
}
LABELS = {method: method.replace("_", " ").title() for method in COLORS}
SAME_PANEL = list(COLORS)[:7]


def _atomic_save(figure: plt.Figure, destination: str | Path) -> Path:
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=output.suffix, dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        figure.savefig(temporary, bbox_inches="tight")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _scatter_methods(axis: plt.Axes, frame: pd.DataFrame, x: str, y: str) -> None:
    for method in SAME_PANEL:
        selected = frame.loc[frame["method"] == method]
        if selected.empty:
            continue
        axis.scatter(
            selected[x],
            selected[y],
            s=25,
            color=COLORS[method],
            marker="o" if method in {"full_exact_heat", "truncated_heat_local"} else "s",
            label=LABELS[method],
        )


def plot_pendulum_figure(
    summary: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    png_path: str | Path,
    pdf_path: str | Path,
) -> tuple[Path, Path]:
    required = {
        "method",
        "target",
        "reference_target",
        "K",
        "radius",
        "heat_scaling",
        "temperature",
        "mean_value_error",
        "mean_policy_l1_error",
        "actions_per_backup",
        "online_seconds",
        "raw_return_mean",
        "regularized_value_mean",
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Pendulum summary is missing columns: {sorted(missing)}")
    if set(summary["reference_target"].dropna()) != {"exact_heat"}:
        raise ValueError("Pendulum error rows must reference FullExactHeat")
    primary = summary.loc[
        summary["method"].isin(SAME_PANEL)
        & (summary["status"] == "complete")
    ].copy()
    if primary.empty:
        raise ValueError("Pendulum figure has no completed primary rows")

    apply_paper_style()
    figure, axes = plt.subplots(2, 3, figsize=(10.4, 6.2))
    figure_config = config["figure"]
    representative = primary.loc[
        (primary["K"] == int(figure_config["K"]))
        & (primary["heat_scaling"] == str(figure_config["heat_scaling"]))
        & np.isclose(
            primary["temperature"].astype(float),
            float(figure_config["temperature"]),
        )
    ]
    if representative.empty:
        representative = primary
    _scatter_methods(axes[0, 0], representative, "actions_per_backup", "mean_value_error")
    axes[0, 0].set_yscale("symlog", linthresh=1e-8)
    axes[0, 0].set_xlabel("Actions evaluated per backup")
    axes[0, 0].set_ylabel("Mean value error to FullExactHeat")
    axes[0, 0].set_title("A  Value accuracy/computation")

    _scatter_methods(
        axes[0, 1], representative, "actions_per_backup", "mean_policy_l1_error"
    )
    axes[0, 1].set_yscale("symlog", linthresh=1e-8)
    axes[0, 1].set_xlabel("Actions evaluated per backup")
    axes[0, 1].set_ylabel("Mean policy L1 error")
    axes[0, 1].set_title("B  Policy accuracy/computation")

    for scaling, linestyle in (("index_heat", "-"), ("physical_heat", "--")):
        selected_scaling = primary.loc[
            (primary["heat_scaling"] == scaling)
            & np.isclose(
                primary["temperature"].astype(float),
                float(figure_config["temperature"]),
            )
            & (primary["method"].isin(["full_exact_heat", "truncated_heat_local"]))
        ]
        for method in ("full_exact_heat", "truncated_heat_local"):
            selected = selected_scaling.loc[selected_scaling["method"] == method].sort_values("K")
            if selected.empty:
                continue
            axes[0, 2].plot(
                selected["K"],
                selected["online_seconds"],
                marker="o",
                linestyle=linestyle,
                color=COLORS[method],
                label=f"{LABELS[method]} ({scaling.replace('_', ' ')})",
            )
    axes[0, 2].set_xscale("log")
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_xlabel("Action-grid size, K")
    axes[0, 2].set_ylabel("Total planning time (s)")
    axes[0, 2].set_title("C  Index vs physical heat scaling")

    behavioral = summary.loc[
        np.isfinite(summary["raw_return_mean"].astype(float))
        & (summary["K"] == int(figure_config["K"]))
        & (summary["heat_scaling"] == str(figure_config["heat_scaling"]))
        & np.isclose(
            summary["temperature"].astype(float),
            float(figure_config["temperature"]),
        )
    ]
    for method in COLORS:
        selected = behavioral.loc[behavioral["method"] == method]
        if selected.empty:
            continue
        axes[1, 0].scatter(
            selected["regularized_value_mean"],
            selected["raw_return_mean"],
            color=COLORS[method],
            label=LABELS[method],
        )
    axes[1, 0].set_xlabel("Regularized discounted value")
    axes[1, 0].set_ylabel("Raw episodic return")
    axes[1, 0].set_title("D  Behavior (objectives labeled separately)")

    ablation = summary.loc[
        summary["method"] == "truncated_heat_local_radius"
    ].sort_values("radius")
    if not ablation.empty:
        axes[1, 1].plot(
            ablation["radius"],
            ablation["mean_value_error"],
            marker="o",
            color=COLORS["truncated_heat_local"],
        )
        computation_axis = axes[1, 1].twinx()
        computation_axis.plot(
            ablation["radius"],
            ablation["actions_per_backup"],
            marker="s",
            linestyle="--",
            color="#555555",
        )
        computation_axis.set_ylabel("Actions per backup")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("Finite-walk radius, r")
    axes[1, 1].set_ylabel("Mean value error")
    axes[1, 1].set_title("E  Radius bias/computation")

    different = summary.loc[
        summary["method"].isin(list(COLORS)[7:])
        & (summary["K"] == int(figure_config["K"]))
        & (summary["heat_scaling"] == str(figure_config["heat_scaling"]))
        & np.isclose(
            summary["temperature"].astype(float),
            float(figure_config["temperature"]),
        )
    ]
    if not different.empty:
        positions = np.arange(len(different))
        axes[1, 2].bar(
            positions,
            different["regularized_value_mean"],
            color=[COLORS[item] for item in different["method"]],
        )
        axes[1, 2].set_xticks(
            positions,
            [LABELS[item] for item in different["method"]],
            rotation=25,
            ha="right",
        )
    axes[1, 2].set_ylabel("Own-objective regularized value")
    axes[1, 2].set_title("F  Different-target references")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=4,
            frameon=False,
        )
    figure.suptitle(
        "Pendulum-v1: errors are diagnostic to FullExactHeat; target identities remain explicit",
        y=0.995,
        fontsize=9,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    png = _atomic_save(figure, png_path)
    pdf = _atomic_save(figure, pdf_path)
    plt.close(figure)
    return png, pdf
