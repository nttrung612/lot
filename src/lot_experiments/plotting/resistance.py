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
        colors = np.full(K, 0.5)
        sizes = np.full(K, 72.0)
    else:
        colors = np.asarray(node_values, dtype=float)
        sizes = 45.0 + 360.0 * colors
    axis.scatter(
        positions[:, 0],
        positions[:, 1],
        c=colors,
        cmap="viridis",
        s=sizes,
        edgecolor="white",
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
    pdf_path: str | Path,
) -> tuple[Path, Path]:
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
    figure, axes = plt.subplots(1, 4, figsize=(7.2, 2.35))
    _draw_graph(axes[0], input_matrix, positions)
    axes[0].set_title("A  Input action graph")
    axes[0].text(
        0.0,
        -1.02,
        f"weak bridge weight={float(graph_config['bridge_weight']):g}",
        ha="center",
        fontsize=6.5,
    )

    _draw_graph(axes[1], cooccurrence, positions, node_values=policy, complete=True)
    axes[1].set_title(f"B  Induced $A_Q$\n({q_name.replace('_', ' ')})")
    axes[1].text(0.0, -1.02, "node area = policy mass", ha="center", fontsize=6.5)

    image = axes[2].imshow(resistance, cmap="magma", interpolation="nearest")
    axes[2].set_title("C  $R_Q(i,k)$")
    axes[2].set_xlabel("action $k$")
    axes[2].set_ylabel("action $i$")
    axes[2].set_xticks(range(K))
    axes[2].set_yticks(range(K))
    colorbar = figure.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(labelsize=6)

    curvature = frame.loc[frame["record_type"] == "curvature"].copy()
    markers = {"balanced": "o", "left_favored": "s", "bridge_favored": "^"}
    for (pair_type, current_q), group in curvature.groupby(
        ["pair_type", "q_name"], sort=True
    ):
        group = group.sort_values("step_size")
        axes[3].plot(
            group["step_size"],
            group["relative_curvature_error"],
            color=PAIR_COLORS.get(str(pair_type), "#333333"),
            marker=markers.get(str(current_q), "o"),
            linestyle="-",
            alpha=0.78,
        )
    positive = curvature.loc[
        (curvature["step_size"] > 0.0) & (curvature["relative_curvature_error"] > 0.0)
    ]
    if not positive.empty:
        steps = np.sort(positive["step_size"].unique())
        reference = float(positive["relative_curvature_error"].median()) * (
            steps / float(np.median(positive["step_size"]))
        ) ** 2
        axes[3].plot(steps, reference, "k--", linewidth=1.0, label="$O(h^2)$")
    axes[3].set_xscale("log")
    axes[3].set_yscale("log")
    axes[3].set_xlabel("finite-difference step $h$")
    axes[3].set_ylabel("relative error")
    axes[3].set_title("D  Curvature convergence")
    pair_handles = [
        Line2D([0], [0], color=color, label=PAIR_LABELS[pair_type])
        for pair_type, color in PAIR_COLORS.items()
    ]
    pair_handles.append(
        Line2D([0], [0], color="black", linestyle="--", label="$O(h^2)$")
    )
    pair_legend = axes[3].legend(
        handles=pair_handles, frameon=False, fontsize=5.2, loc="upper left"
    )
    axes[3].add_artist(pair_legend)
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
    axes[3].legend(
        handles=q_handles,
        frameon=False,
        fontsize=4.7,
        loc="lower right",
        handletextpad=0.3,
    )

    figure.text(
        0.5,
        -0.01,
        (
            "Input geometry and the Q-dependent co-occurrence graph are distinct. "
            "Panel D verifies the exact local identity $d^2\\Omega=T_0 R_Q$; "
            "it does not assert an optimization improvement."
        ),
        ha="center",
        va="top",
        fontsize=6.7,
    )
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 1.0), w_pad=1.0)
    png = _atomic_save(figure, png_path)
    pdf = _atomic_save(figure, pdf_path)
    plt.close(figure)
    return png, pdf
