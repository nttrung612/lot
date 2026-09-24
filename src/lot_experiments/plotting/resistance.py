"""Appendix figure for the LOT curvature/effective-resistance identity."""

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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from lot_experiments.config import config_json
from lot_experiments.plotting.common import apply_paper_style
from lot_experiments.resistance import TARGET
from lot_experiments.results import validate_results


PAIR_COLORS = {
    "within_cluster": "#1f77b4",
    "bridge": "#d62728",
    "cross_cluster": "#9467bd",
}
PAIR_LABELS = {
    "within_cluster": "Within cluster",
    "bridge": "Bridge endpoints",
    "cross_cluster": "Cross cluster",
}


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


def _positions(cluster_size: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, cluster_size, endpoint=False)
    positions = np.empty((2 * cluster_size, 2), dtype=np.float64)
    # Put the designated bridge endpoints (cluster_size - 1, cluster_size)
    # opposite one another so the weak cut edge is visually unambiguous.
    left_angles = np.roll(angles, 1)
    positions[:cluster_size, 0] = -1.15 + 0.58 * np.cos(left_angles)
    positions[:cluster_size, 1] = 0.58 * np.sin(left_angles)
    positions[cluster_size:, 0] = 1.15 - 0.58 * np.cos(angles)
    positions[cluster_size:, 1] = 0.58 * np.sin(angles)
    return positions


def _draw_graph(
    axis: plt.Axes,
    adjacency: np.ndarray,
    positions: np.ndarray,
    *,
    node_values: np.ndarray | None = None,
    complete: bool = False,
) -> None:
    K = adjacency.shape[0]
    off_diagonal = adjacency.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    maximum = float(np.max(off_diagonal, initial=0.0))
    # Full-support LOT costs induce a complete graph. Draw every positive edge;
    # alpha and width carry its weight instead of silently thresholding it.
    visibility = 0.0
    for action_i in range(K):
        for action_k in range(action_i + 1, K):
            weight = float(adjacency[action_i, action_k])
            if weight <= 0.0 or (complete and weight < visibility):
                continue
            normalized = weight / maximum if maximum > 0.0 else 0.0
            weak_input_edge = not complete and weight < 0.5 * maximum
            axis.plot(
                positions[[action_i, action_k], 0],
                positions[[action_i, action_k], 1],
                color="#d62728" if weak_input_edge else ("#555555" if complete else "#6f6f6f"),
                linewidth=1.0 if weak_input_edge else 0.25 + 2.8 * normalized,
                alpha=0.9 if weak_input_edge else 0.15 + 0.75 * normalized,
                zorder=1,
            )
    if node_values is None:
        colors = "#d9dee7"
        sizes = np.full(K, 72.0)
        edgecolor = "#3f4854"
    else:
        colors = np.asarray(node_values, dtype=float)
        sizes = 45.0 + 360.0 * colors
        edgecolor = "white"
    axis.scatter(
        positions[:, 0],
        positions[:, 1],
        c=colors,
        cmap=None if node_values is None else "viridis",
        s=sizes,
        edgecolor=edgecolor,
        linewidth=0.6,
        zorder=2,
    )
    for index, (x_value, y_value) in enumerate(positions):
        axis.text(x_value, y_value, str(index), ha="center", va="center", fontsize=6, zorder=3)
    axis.set_aspect("equal")
    axis.set_xlim(-2.0, 2.0)
    axis.set_ylim(-1.1, 1.0)
    axis.axis("off")


def _matrix_from_rows(
    rows: pd.DataFrame, column: str, K: int, *, q_name: str | None = None
) -> np.ndarray:
    selected = rows
    if q_name is not None:
        selected = selected.loc[selected["q_name"] == q_name]
    matrix = np.full((K, K), np.nan, dtype=np.float64)
    for row in selected.itertuples(index=False):
        matrix[int(row.action_i), int(row.action_k)] = float(getattr(row, column))
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"raw resistance results do not contain a complete {column} matrix")
    return matrix


def plot_resistance_figure(
    raw: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    png_path: str | Path,
) -> Path:
    """Plot M9 entirely from stored run-level results."""

    frame = validate_results(raw)
    frame = frame.loc[
        (frame["experiment"] == "resistance") & (frame["status"] == "complete")
    ].copy()
    if frame.empty:
        raise ValueError("resistance results are empty")
    expected_config = config_json(config)
    if set(frame["config_json"]) != {expected_config}:
        raise ValueError("resistance results do not match the resolved plotting config")
    if set(frame["target"]) != {TARGET} or set(frame["reference_target"]) != {TARGET}:
        raise ValueError("resistance figure requires the LOT policy-curvature target")
    required = {
        "record_type",
        "q_name",
        "action_i",
        "action_k",
        "input_edge_weight",
        "cooccurrence_weight",
        "effective_resistance",
        "policy_mass",
        "pair_type",
        "step_size",
        "relative_curvature_error",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"resistance results are missing columns: {sorted(missing)}")

    graph_config = config["graph"]
    cluster_size = int(graph_config["cluster_size"])
    K = 2 * cluster_size
    q_name = str(config["visualization_q"])
    positions = _positions(cluster_size)
    input_matrix = _matrix_from_rows(
        frame.loc[frame["record_type"] == "input_matrix"], "input_edge_weight", K
    )
    geometry_rows = frame.loc[frame["record_type"] == "geometry_matrix"]
    cooccurrence = _matrix_from_rows(
        geometry_rows, "cooccurrence_weight", K, q_name=q_name
    )
    resistance = _matrix_from_rows(
        geometry_rows, "effective_resistance", K, q_name=q_name
    )
    nodes = frame.loc[
        (frame["record_type"] == "node") & (frame["q_name"] == q_name)
    ].sort_values("action_i")
    if len(nodes) != K:
        raise ValueError(f"expected {K} node records for visualization_q={q_name!r}")
    policy = nodes["policy_mass"].to_numpy(dtype=float)

    apply_paper_style()
    figure = plt.figure(figsize=(7.2, 4.65))
    outer = figure.add_gridspec(
        2,
        1,
        height_ratios=(1.0, 1.48),
        hspace=0.42,
    )
    top = outer[0].subgridspec(
        1,
        3,
        width_ratios=(1.08, 1.08, 0.90),
        wspace=0.34,
    )
    graph_axis = figure.add_subplot(top[0, 0])
    cooccurrence_axis = figure.add_subplot(top[0, 1])
    resistance_axis = figure.add_subplot(top[0, 2])
    bottom = outer[1].subgridspec(
        1,
        2,
        width_ratios=(4.9, 1.45),
        wspace=0.08,
    )
    curvature_axis = figure.add_subplot(bottom[0, 0])
    legend_axis = figure.add_subplot(bottom[0, 1])
    legend_axis.axis("off")

    _draw_graph(graph_axis, input_matrix, positions)
    graph_axis.set_title("A  Input action graph", loc="left", fontweight="bold")
    graph_axis.text(
        0.0,
        -1.02,
        f"weak bridge weight={float(graph_config['bridge_weight']):g}",
        ha="center",
        fontsize=6.5,
    )

    _draw_graph(
        cooccurrence_axis,
        cooccurrence,
        positions,
        node_values=policy,
        complete=True,
    )
    cooccurrence_axis.set_title(
        f"B  Induced $A_Q$  ({q_name.replace('_', ' ')})",
        loc="left",
        fontweight="bold",
    )
    cooccurrence_axis.text(
        0.0,
        -1.02,
        "node size and color = $\\pi_Q$",
        ha="center",
        fontsize=6.5,
    )

    image = resistance_axis.imshow(resistance, cmap="magma", interpolation="nearest")
    resistance_axis.set_title(
        "C  Effective resistance $R_Q(i,k)$",
        loc="left",
        fontweight="bold",
    )
    resistance_axis.set_xlabel("action $k$")
    resistance_axis.set_ylabel("action $i$")
    resistance_axis.set_xticks(range(K))
    resistance_axis.set_yticks(range(K))
    for specification in config["pairs"].values():
        action_i, action_k = map(int, specification["actions"])
        color = PAIR_COLORS.get(str(specification["type"]), "#333333")
        for row, column in ((action_i, action_k), (action_k, action_i)):
            resistance_axis.add_patch(
                Rectangle(
                    (column - 0.46, row - 0.46),
                    0.92,
                    0.92,
                    fill=False,
                    edgecolor=color,
                    linewidth=1.25,
                )
            )
    colorbar = figure.colorbar(
        image,
        ax=resistance_axis,
        fraction=0.046,
        pad=0.04,
        label="resistance",
    )
    colorbar.ax.tick_params(labelsize=6)

    curvature = frame.loc[frame["record_type"] == "curvature"].copy()
    q_names = [str(name) for name in config["q_vectors"]]
    marker_cycle = ("o", "s", "^", "D", "v", "P")
    markers = {
        name: marker_cycle[index % len(marker_cycle)]
        for index, name in enumerate(q_names)
    }
    curvature["curvature_ratio"] = (
        curvature["finite_difference_curvature"]
        / curvature["theoretical_curvature"]
    )
    curvature["signed_deviation_milli"] = 1.0e3 * (
        curvature["curvature_ratio"] - 1.0
    )
    for (pair_type, current_q), group in curvature.groupby(
        ["pair_type", "q_name"], sort=True
    ):
        group = group.sort_values("step_size")
        curvature_axis.plot(
            group["step_size"],
            group["signed_deviation_milli"],
            color=PAIR_COLORS.get(str(pair_type), "#333333"),
            marker=markers.get(str(current_q), "o"),
            linestyle="-",
            markerfacecolor="white",
            markeredgewidth=0.9,
            linewidth=1.25,
            markersize=4.3,
            alpha=0.9,
        )
    steps = np.sort(curvature["step_size"].unique())
    largest_step = float(steps[-1])
    largest_error = float(
        curvature.loc[
            np.isclose(curvature["step_size"], largest_step),
            "signed_deviation_milli",
        ].abs().max()
    )
    envelope = largest_error * (steps / largest_step) ** 2
    curvature_axis.fill_between(
        steps,
        -envelope,
        envelope,
        color="#9ca3af",
        alpha=0.18,
        linewidth=0.0,
        zorder=0,
    )
    curvature_axis.plot(
        steps,
        envelope,
        color="#6b7280",
        linestyle="--",
        linewidth=0.9,
        zorder=0,
    )
    curvature_axis.plot(
        steps,
        -envelope,
        color="#6b7280",
        linestyle="--",
        linewidth=0.9,
        zorder=0,
    )
    curvature_axis.axhline(0.0, color="black", linewidth=1.0, zorder=1)
    curvature_axis.set_xscale("log")
    curvature_axis.set_xlabel("finite-difference step $h$  ($\\leftarrow$ smaller $h$)")
    curvature_axis.set_ylabel("relative deviation from $T_0R_Q$  ($10^{-3}$)")
    curvature_axis.set_title(
        "D  Finite differences recover the effective-resistance curvature",
        loc="left",
        fontweight="bold",
    )
    curvature_axis.text(
        float(steps[0]),
        0.04 * max(1.0, largest_error),
        "exact identity",
        fontsize=6.5,
        ha="left",
        va="bottom",
    )
    curvature_axis.margins(x=0.03, y=0.12)

    pair_handles = [
        Line2D([0], [0], color=color, label=PAIR_LABELS[pair_type])
        for pair_type, color in PAIR_COLORS.items()
    ]
    pair_legend = legend_axis.legend(
        handles=pair_handles,
        title="Transfer pair",
        frameon=False,
        fontsize=6.5,
        title_fontsize=7,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        borderaxespad=0.0,
    )
    legend_axis.add_artist(pair_legend)
    q_handles = [
        Line2D(
            [0],
            [0],
            color="#555555",
            marker=marker,
            linestyle="none",
            label=q_name.replace("_", " "),
        )
        for q_name, marker in markers.items()
    ]
    q_legend = legend_axis.legend(
        handles=q_handles,
        title="Value profile $Q$",
        frameon=False,
        fontsize=6.5,
        title_fontsize=7,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.54),
        borderaxespad=0.0,
        handletextpad=0.3,
    )
    legend_axis.add_artist(q_legend)
    legend_axis.legend(
        handles=[
            Patch(
                facecolor="#9ca3af",
                edgecolor="#6b7280",
                alpha=0.25,
                label="$O(h^2)$ envelope",
            )
        ],
        frameon=False,
        fontsize=6.5,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.22),
        borderaxespad=0.0,
    )

    finest_step = float(curvature["step_size"].min())
    finest_error = float(
        curvature.loc[
            np.isclose(curvature["step_size"], finest_step),
            "relative_curvature_error",
        ].max()
    )
    legend_axis.text(
        0.0,
        0.0,
        f"Worst case at $h={finest_step:g}$:\nrelative error = {finest_error:.1e}",
        transform=legend_axis.transAxes,
        fontsize=6.5,
        ha="left",
        va="bottom",
    )

    figure.text(
        0.01,
        0.012,
        (
            "The input graph and the $Q$-dependent co-occurrence graph are distinct. "
            "Colored boxes in C identify the transfer pairs used in D. "
            "This verifies a local identity; it does not claim an optimization improvement."
        ),
        ha="left",
        va="bottom",
        fontsize=6.7,
    )
    figure.subplots_adjust(left=0.075, right=0.985, top=0.94, bottom=0.125)
    png = _atomic_save(figure, png_path)
    plt.close(figure)
    return png
