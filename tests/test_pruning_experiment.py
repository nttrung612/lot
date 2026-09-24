from pathlib import Path

import numpy as np
import pandas as pd

from lot_experiments.graphs import cycle_graph
from lot_experiments.plotting.pruning import plot_pruning_figure
from lot_experiments.pruning_experiment import (
    equal_cardinality_masks,
    resolve_pruning_config,
    run_pruning_experiment,
    summarize_pruning,
    write_pruning_summary_atomic,
)


def test_fixed_set_baselines_have_equal_cardinality_and_exact_ball():
    graph = cycle_graph(11)
    weights = np.linspace(1.0, 2.0, graph.K)
    weights /= weights.sum()
    masks = equal_cardinality_masks(
        graph, anchor=2, radius=2, weights=weights, rng=np.random.default_rng(4)
    )
    assert {int(mask.sum()) for mask in masks.values()} == {5}
    assert set(np.flatnonzero(masks["smallest_graph_ball"])) == graph.ball(2, 2)


def test_small_m3_run_is_resumable_certified_and_plot_only(tmp_path):
    config = resolve_pruning_config(
        {
            "experiment": "pruning",
            "seed": 7,
            "temperature": 0.2,
            "span_over_temperature": [1, 4],
            "numerical_tolerance": 2e-10,
            "worst_case": {
                "graph": {"family": "cycle", "K": 9},
                "poisson_mean": 1.0,
                "anchor": 0,
                "alpha": [0.2, 0.05],
            },
            "typical": {
                "graphs": [
                    {"family": "cycle", "K": 9, "anchor": 0},
                    {"family": "grid", "rows": 3, "columns": 3},
                ],
                "poisson_mean": 1.0,
                "subset_radii": [0, 1],
                "repetitions": 3,
                "field_smoothing_time": 0.5,
            },
            "poisson_truncation": {
                "graphs": [{"family": "cycle", "K": 9}],
                "poisson_mean": [1.0],
                "r_max": 5,
                "fixed_point": {
                    "state_count": 3,
                    "gamma": 0.75,
                    "tolerance": 1e-12,
                    "max_iterations": 5000,
                    "root_state": 0,
                },
            },
            "raw_output": str(tmp_path / "raw.parquet"),
            "summary_output": str(tmp_path / "summary.csv"),
            "figure_png": str(tmp_path / "figure.png"),
            "figure": {
                "typical_span_over_temperature": 4.0,
                "truncation_graph_family": "cycle",
                "truncation_poisson_mean": 1.0,
            },
        }
    )
    first = run_pruning_experiment(config)
    second = run_pruning_experiment(config)
    assert len(first) == len(second)
    assert second["run_id"].is_unique
    assert set(second["status"]) == {"complete"}

    worst = second.loc[second["experiment_part"] == "worst_case"]
    assert len(worst) == 4
    np.testing.assert_allclose(
        worst["observed_pruning_error"],
        worst["theoretical_pruning_error"],
        atol=2e-12,
        rtol=0.0,
    )
    assert set(worst["target"]) == {"exact_heat"}

    typical = second.loc[second["experiment_part"] == "typical"]
    assert typical["backup_certificate_holds"].all()
    equal_sizes = typical.groupby(
        ["graph_family", "replicate", "span_over_temperature", "selection_radius"]
    )["retained_count"].nunique()
    assert (equal_sizes == 1).all()

    truncated = second.loc[second["experiment_part"] == "poisson_truncation"]
    assert set(truncated["target"]) == {"truncated_heat"}
    assert set(truncated["reference_target"]) == {"exact_heat"}
    assert truncated["kernel_certificate_holds"].all()
    assert truncated["backup_certificate_holds"].all()
    assert truncated["fixed_point_certificate_holds"].all()

    summary = summarize_pruning(second, resolved_config=config)
    write_pruning_summary_atomic(summary, config["summary_output"])
    png = plot_pruning_figure(
        pd.read_csv(config["summary_output"]),
        config,
        png_path=config["figure_png"],
    )
    assert Path(png).stat().st_size > 0
