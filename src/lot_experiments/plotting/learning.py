"""Reward-free geometry learning figure."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lot_experiments.geometry_learning import PRIMARY_TARGET
from lot_experiments.plotting.common import apply_paper_style
from lot_experiments.reproducibility import derive_seed
from lot_experiments.results import validate_results


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


def _paired_task_statistic(matrix: np.ndarray, aggregation: str) -> float:
    """Aggregate seed-paired errors after taking each task's seed median."""

    task_medians = np.median(matrix, axis=0)
    if aggregation == "median":
        return float(np.median(task_medians))
    if aggregation == "worst_goal":
        return float(np.max(task_medians))
    raise ValueError(f"unknown task aggregation: {aggregation}")


def _bootstrap_curve(
    frame: pd.DataFrame,
    *,
    metric: str,
    statistic: Callable[[np.ndarray], float],
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    """Pointwise paired bootstrap intervals over reward-free replicates."""

    records: list[dict[str, float]] = []
    for n, group in frame.groupby("samples_per_action", sort=True):
        matrix = group.pivot(index="replicate", columns="reward_goal", values=metric)
        if matrix.isna().any(axis=None):
            raise ValueError(f"incomplete paired reward-task data for n={n}")
        values = matrix.to_numpy(dtype=float)
        point = statistic(values)
        rng = np.random.default_rng(derive_seed(seed, f"plot:{metric}:{n}"))
        indices = rng.integers(0, len(values), size=(repetitions, len(values)))
        bootstrap = np.asarray([statistic(values[index]) for index in indices])
        records.append(
            {
                "samples_per_action": float(n),
                "estimate": point,
                "ci_low": float(np.quantile(bootstrap, 0.025)),
                "ci_high": float(np.quantile(bootstrap, 0.975)),
            }
        )
    return pd.DataFrame(records)


def _geometry_curve(
    frame: pd.DataFrame,
    *,
    metric: str,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap a reward-independent geometry metric once per replicate."""

    spread = frame.groupby(["samples_per_action", "replicate"])[metric].agg(
        lambda values: float(np.max(values) - np.min(values))
    )
    if (spread > 1e-12).any():
        raise ValueError(f"reward-independent metric {metric!r} varies across goals")
    reduced = frame.groupby(
        ["samples_per_action", "replicate"], as_index=False, sort=True
    )[metric].first()
    reduced["reward_goal"] = 0
    return _bootstrap_curve(
        reduced,
        metric=metric,
        statistic=lambda values: float(np.median(values[:, 0])),
        repetitions=repetitions,
        seed=seed,
    )


def _plot_curve_with_ci(
    axis: plt.Axes,
    curve: pd.DataFrame,
    *,
    color: str,
    label: str,
    marker: str,
    linestyle: str = "-",
    linewidth: float = 1.8,
) -> None:
    x = curve["samples_per_action"].to_numpy(float)
    estimate = curve["estimate"].to_numpy(float)
    low = curve["ci_low"].to_numpy(float)
    high = curve["ci_high"].to_numpy(float)
    axis.fill_between(x, low, high, color=color, alpha=0.14, linewidth=0.0)
    axis.plot(
        x,
        estimate,
        color=color,
        label=label,
        marker=marker,
        linestyle=linestyle,
        linewidth=linewidth,
        markersize=4.0,
    )


def plot_geometry_learning_figure(
    raw: pd.DataFrame,
    threshold_table: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    png_path: str | Path,
) -> Path:
    """Render the M6 paper figure from raw results, without rerunning data."""

    frame = validate_results(raw)
    primary = frame.loc[
        (frame["experiment"] == "geometry_learning")
        & (frame["status"] == "complete")
        & (frame["experiment_part"] == "primary")
        & (frame["method"] == "learned_transition_geometry")
    ].copy()
    if primary.empty:
        raise ValueError("geometry results have no primary learned-geometry rows")
    if set(primary["target"].dropna()) != {PRIMARY_TARGET}:
        raise ValueError("primary learned geometry must retain the population-cost target")
    required = {
        "replicate",
        "samples_per_action",
        "reward_goal",
        "cost_max_error",
        "fixed_point_q_error",
        "propagation_envelope",
    }
    missing = required - set(primary.columns)
    if missing:
        raise ValueError(f"geometry results are missing columns: {sorted(missing)}")

    repetitions = int(config["bootstrap_repetitions"])
    seed = int(config["seed"])
    cost_curve = _geometry_curve(
        primary,
        metric="cost_max_error",
        repetitions=repetitions,
        seed=seed,
    )
    envelope_curve = _geometry_curve(
        primary,
        metric="propagation_envelope",
        repetitions=repetitions,
        seed=derive_seed(seed, "envelope"),
    )
    median_q_curve = _bootstrap_curve(
        primary,
        metric="fixed_point_q_error",
        statistic=lambda values: _paired_task_statistic(values, "median"),
        repetitions=repetitions,
        seed=derive_seed(seed, "median-q"),
    )
    worst_q_curve = _bootstrap_curve(
        primary,
        metric="fixed_point_q_error",
        statistic=lambda values: _paired_task_statistic(values, "worst_goal"),
        repetitions=repetitions,
        seed=derive_seed(seed, "worst-q"),
    )
    goal_curves = (
        primary.groupby(["samples_per_action", "reward_goal"], as_index=False)[
            "fixed_point_q_error"
        ]
        .median()
        .sort_values(["reward_goal", "samples_per_action"])
    )

    apply_paper_style()
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))

    _plot_curve_with_ci(
        axes[0, 0],
        cost_curve,
        color="#1f77b4",
        label="Median (95% bootstrap CI)",
        marker="o",
    )
    n_values = cost_curve["samples_per_action"].to_numpy(float)
    cost_values = cost_curve["estimate"].to_numpy(float)
    reference = cost_values[0] * np.sqrt(n_values[0] / n_values)
    axes[0, 0].plot(
        n_values,
        reference,
        linestyle=":",
        color="#333333",
        linewidth=1.4,
        label=r"$n^{-1/2}$ reference",
    )
    axes[0, 0].set(
        xscale="log",
        yscale="log",
        xlabel="Transitions per action, $n$",
        ylabel=r"$\|\widehat C_t-C_t\|_{\max}$",
        title="A  Reward-free cost learning",
    )
    axes[0, 0].legend(frameon=False, fontsize=7)

    _plot_curve_with_ci(
        axes[0, 1],
        median_q_curve,
        color="#2ca02c",
        label="Median goal (95% CI)",
        marker="o",
    )
    _plot_curve_with_ci(
        axes[0, 1],
        envelope_curve,
        color="#d62728",
        label="Propagation envelope",
        marker="s",
        linestyle="--",
    )
    axes[0, 1].set(
        xscale="log",
        yscale="log",
        xlabel="Transitions per action, $n$",
        ylabel=r"$\|Q^*_{\widehat C_t}-Q^*_{C_t}\|_\infty$",
        title="B  Error propagation",
    )
    axes[0, 1].legend(frameon=False, fontsize=7)

    goal_count = primary["reward_goal"].nunique()
    goal_colors = plt.get_cmap("Blues")(np.linspace(0.35, 0.72, goal_count))
    markers = ("o", "s", "^", "D", "v", "P", "X")
    for index, (goal, group) in enumerate(goal_curves.groupby("reward_goal", sort=True)):
        axes[1, 0].plot(
            group["samples_per_action"],
            group["fixed_point_q_error"],
            color=goal_colors[index],
            marker=markers[index % len(markers)],
            markersize=2.8,
            linewidth=0.85,
            alpha=0.9,
            label=f"$g={int(goal)}$",
        )
    _plot_curve_with_ci(
        axes[1, 0],
        median_q_curve,
        color="#174a7e",
        label="Median goal",
        marker="o",
        linewidth=2.0,
    )
    _plot_curve_with_ci(
        axes[1, 0],
        worst_q_curve,
        color="#ff7f0e",
        label="Worst-goal envelope",
        marker="s",
        linestyle="--",
        linewidth=1.8,
    )
    axes[1, 0].set(
        xscale="log",
        yscale="log",
        xlabel="Transitions per action, $n$",
        ylabel=r"$\|Q^*_{\widehat C_t}-Q^*_{C_t}\|_\infty$",
        title="C  Transfer across reward goals",
    )
    axes[1, 0].legend(frameon=False, fontsize=6.3, ncol=2, columnspacing=0.8)

    thresholds = sorted(
        threshold_table["value_error_threshold"].dropna().unique(), reverse=True
    )
    positions = np.arange(len(thresholds), dtype=float)
    plotted = False
    for offset, aggregation, color, marker, label, text_offset, alignment in (
        (-0.08, "median", "#174a7e", "o", "Median goal", -3, "right"),
        (0.08, "worst_case", "#ff7f0e", "s", "Worst goal", 3, "left"),
    ):
        selected = threshold_table.loc[
            threshold_table["task_aggregation"] == aggregation
        ].set_index("value_error_threshold")
        values = np.asarray(
            [
                selected.loc[value, "total_reward_free_samples"]
                if value in selected.index
                else np.nan
                for value in thresholds
            ],
            dtype=float,
        )
        finite = np.isfinite(values) & (values > 0.0)
        if finite.any():
            plotted = True
            axes[1, 1].plot(
                positions[finite] + offset,
                values[finite],
                color=color,
                marker=marker,
                markersize=5.0,
                linewidth=1.4,
                label=label,
            )
            for x_value, sample_count in zip(
                positions[finite] + offset, values[finite], strict=True
            ):
                axes[1, 1].annotate(
                    f"{int(sample_count):,}",
                    (x_value, sample_count),
                    xytext=(text_offset, 5),
                    textcoords="offset points",
                    ha=alignment,
                    va="bottom",
                    fontsize=6.5,
                    color=color,
                )
    axes[1, 1].set_xticks(positions, [f"{value:g}" for value in thresholds])
    if plotted:
        axes[1, 1].set_yscale("log")
        axes[1, 1].legend(frameon=False, fontsize=7)
    else:
        axes[1, 1].text(0.5, 0.5, "No threshold reached", ha="center", va="center")
    axes[1, 1].set(
        xlabel=r"Fixed-point $Q$-error threshold, $\epsilon$",
        ylabel="Total reward-free transitions, $Kn$",
        title="D  Samples to target accuracy",
    )

    figure.suptitle(
        "One reward-free geometry reused across goals; downstream dynamics remain exact",
        fontsize=9,
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    png = _atomic_save(figure, png_path)
    plt.close(figure)
    return png
