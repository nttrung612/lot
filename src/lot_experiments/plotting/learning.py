"""Reward-free geometry learning figure."""

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

from lot_experiments.geometry_learning import PRIMARY_TARGET
from lot_experiments.plotting.common import apply_paper_style


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


def _positive_curve(frame: pd.DataFrame, x: str, y: str) -> tuple[np.ndarray, np.ndarray]:
    selected = frame.sort_values(x)
    x_values = selected[x].to_numpy(float)
    y_values = selected[y].to_numpy(float)
    keep = np.isfinite(x_values) & np.isfinite(y_values) & (x_values > 0.0) & (y_values > 0.0)
    return x_values[keep], y_values[keep]


def plot_geometry_learning_figure(
    summary: pd.DataFrame,
    threshold_table: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    png_path: str | Path,
    pdf_path: str | Path,
) -> tuple[Path, Path]:
    """Render all planned M6 panels from summaries, without rerunning data."""

    required = {
        "experiment_part",
        "method",
        "target",
        "samples_per_action",
        "reward_goal",
        "cost_max_error_median",
        "fixed_point_q_error_median",
        "propagation_envelope_median",
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"geometry summary is missing columns: {sorted(missing)}")
    primary = summary.loc[
        (summary["experiment_part"] == "primary")
        & (summary["method"] == "learned_transition_geometry")
    ].copy()
    if primary.empty:
        raise ValueError("geometry summary has no primary learned-geometry rows")
    if set(primary["target"].dropna()) != {PRIMARY_TARGET}:
        raise ValueError("primary learned geometry must retain the population-cost target")

    task_curves = primary.groupby(
        ["samples_per_action", "reward_goal"], as_index=False
    ).agg(
        cost_error=("cost_max_error_median", "median"),
        q_error=("fixed_point_q_error_median", "median"),
        root_error=("absolute_value_error_median", "median"),
        envelope=("propagation_envelope_median", "median"),
    )
    curves = task_curves.groupby("samples_per_action", as_index=False).agg(
        cost_error=("cost_error", "median"),
        q_error=("q_error", "median"),
        q_error_worst=("q_error", "max"),
        envelope=("envelope", "median"),
    )
    n_values, cost_values = _positive_curve(curves, "samples_per_action", "cost_error")
    if len(n_values) == 0:
        raise ValueError("primary cost-error curve has no positive values")

    apply_paper_style()
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))

    axes[0, 0].plot(n_values, cost_values, marker="o", color="#1f77b4", label="Learned cost")
    reference = cost_values[0] * np.sqrt(n_values[0] / n_values)
    axes[0, 0].plot(n_values, reference, linestyle=":", color="black", label=r"$n^{-1/2}$ reference")
    axes[0, 0].set(xscale="log", yscale="log", xlabel="Transitions per action, n", ylabel=r"$\|\widehat C_t-C_t\|_{\max}$", title="A  Reward-free cost learning")
    axes[0, 0].legend(frameon=False)

    q_n, q_values = _positive_curve(curves, "samples_per_action", "q_error")
    env_n, env_values = _positive_curve(curves, "samples_per_action", "envelope")
    axes[0, 1].plot(q_n, q_values, marker="o", color="#2ca02c", label="Fixed-point Q error")
    axes[0, 1].plot(env_n, env_values, marker="s", linestyle="--", color="#d62728", label="Propagation envelope")
    axes[0, 1].set(xscale="log", yscale="log", xlabel="Transitions per action, n", ylabel=r"$\|Q^*_{\widehat C_t}-Q^*_{C_t}\|_\infty$", title="B  Error propagation")
    axes[0, 1].legend(frameon=False)

    for goal, group in task_curves.groupby("reward_goal", sort=True):
        x_values, y_values = _positive_curve(group, "samples_per_action", "root_error")
        axes[1, 0].plot(x_values, y_values, color="#9ecae1", linewidth=0.9, alpha=0.8)
    transfer_curves = task_curves.groupby("samples_per_action", as_index=False).agg(
        root_error=("root_error", "median"),
        root_error_worst=("root_error", "max"),
    )
    median_n, median_values = _positive_curve(transfer_curves, "samples_per_action", "root_error")
    worst_n, worst_values = _positive_curve(transfer_curves, "samples_per_action", "root_error_worst")
    axes[1, 0].plot(median_n, median_values, marker="o", color="#1f77b4", linewidth=2.0, label="Task median")
    axes[1, 0].plot(worst_n, worst_values, marker="s", color="#ff7f0e", linewidth=2.0, label="Worst task")
    axes[1, 0].set(xscale="log", yscale="log", xlabel="Transitions per action, n", ylabel="Root-value error", title="C  Transfer across reward goals")
    axes[1, 0].legend(frameon=False)

    if threshold_table.empty:
        axes[1, 1].text(0.5, 0.5, "No threshold reached", ha="center", va="center")
    else:
        thresholds = sorted(threshold_table["value_error_threshold"].unique(), reverse=True)
        x = np.arange(len(thresholds))
        width = 0.36
        for offset, aggregation, color, label in (
            (-width / 2, "median", "#1f77b4", "Task median"),
            (width / 2, "worst_case", "#ff7f0e", "Worst task"),
        ):
            selected = threshold_table.loc[
                threshold_table["task_aggregation"] == aggregation
            ].set_index("value_error_threshold")
            values = np.array(
                [selected.loc[value, "total_reward_free_samples"] for value in thresholds],
                dtype=float,
            )
            axes[1, 1].bar(x + offset, values, width=width, color=color, label=label)
        axes[1, 1].set_xticks(x, [f"{value:g}" for value in thresholds])
        finite = threshold_table["total_reward_free_samples"].dropna()
        if len(finite) and np.all(finite > 0.0):
            axes[1, 1].set_yscale("log")
    axes[1, 1].set(xlabel="Fixed-point error threshold", ylabel="Total reward-free transitions", title="D  Samples to threshold")
    axes[1, 1].legend(frameon=False)

    figure.suptitle(
        "Reward-free geometry is frozen and reused; all downstream solves use the true transition kernel",
        fontsize=9,
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    png = _atomic_save(figure, png_path)
    pdf = _atomic_save(figure, pdf_path)
    plt.close(figure)
    return png, pdf
