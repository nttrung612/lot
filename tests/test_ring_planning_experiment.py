from pathlib import Path

import numpy as np

from lot_experiments.planners.empirical import (
    component_from_log_samples,
    component_from_log_weights,
)
from lot_experiments.plotting.planning import plot_ring_planning_figure
from lot_experiments.ring_planning import (
    resolve_ring_planning_config,
    run_ring_planning,
    summarize_ring_planning,
    write_ring_summary_atomic,
)


def test_log_components_preserve_weights_below_linear_float_range():
    direct = component_from_log_weights(
        np.array([-1000.0, -np.inf, -1002.0]), actions=np.array([0, 2])
    )
    np.testing.assert_array_equal(direct.actions, [0, 2])
    np.testing.assert_allclose(direct.log_weights, [-1000.0, -1002.0])

    sampled = component_from_log_samples(
        np.array([4, 4, 7]), np.array([-1000.0, -1001.0, -1200.0])
    )
    np.testing.assert_array_equal(sampled.actions, [4, 7])
    np.testing.assert_allclose(
        sampled.log_weights,
        [np.logaddexp(-1000.0, -1001.0), -1200.0],
    )


def test_small_m5_run_resumes_aggregates_and_plots(tmp_path):
    config = resolve_ring_planning_config(
        {
            "experiment": "ring_planning",
            "seed": 4,
            "K": [5],
            "gamma": 0.65,
            "action_cost": 0.05,
            "temperature": [0.2],
            "poisson_mean": [1.0],
            "epsilon": [0.3],
            "delta": 0.1,
            "anchor_uniform_mass": 0.05,
            "paired_seeds": 2,
            "benchmark_threads": 1,
            "reference": {"tolerance": 1e-8, "max_iterations": 1000},
            "sampling": {
                "anchor_samples": 2,
                "sample_scale": 0.01,
                "minimum_inner_samples": 2,
                "maximum_inner_samples": 3,
            },
            "localization": {
                "heat_tail_scale": 0.25,
                "minimum_tail_budget": 1e-8,
                "maximum_radius": 2,
                "pruning_span_bound": 1.0,
            },
            "different_target_references": {"enabled": False},
            "geometry_controls": {"enabled": False},
            "bootstrap_repetitions": 20,
            "raw_output": str(tmp_path / "raw.parquet"),
            "summary_output": str(tmp_path / "summary.csv"),
            "figure_png": str(tmp_path / "figure.png"),
            "figure_pdf": str(tmp_path / "figure.pdf"),
            "figure": {"temperature": 0.2, "poisson_mean": 1.0, "epsilon": 0.3},
        }
    )
    first = run_ring_planning(config)
    second = run_ring_planning(config)
    assert len(first) == len(second) == 14
    assert second["run_id"].is_unique
    assert set(second["status"]) == {"complete"}
    assert set(second["method"]) == {
        "full_exact_heat",
        "full_truncated_heat",
        "dense_second_order",
        "exact_heat_topmass",
        "truncated_heat_local",
        "poisson_endpoint_mc",
        "uniform_action_mc",
        "random_subset",
    }

    exact_methods = second.loc[
        second["method"].isin(
            [
                "full_exact_heat",
                "dense_second_order",
                "exact_heat_topmass",
                "poisson_endpoint_mc",
                "uniform_action_mc",
                "random_subset",
            ]
        )
    ]
    assert set(exact_methods["target"]) == {"exact_heat"}
    local = second.loc[second["method"] == "truncated_heat_local"]
    assert set(local["target"]) == {"truncated_heat"}
    assert set(local["reference_target"]) == {"exact_heat"}
    heat_bias = abs(
        second.loc[second["method"] == "full_truncated_heat", "value_estimate"].iloc[0]
        - second.loc[second["method"] == "full_exact_heat", "value_estimate"].iloc[0]
    )
    np.testing.assert_allclose(local["heat_approximation_error"], heat_bias)
    assert local["graph_neighbor_accesses"].gt(0).all()

    summary = summarize_ring_planning(second, resolved_config=config)
    stochastic = summary.loc[summary["method"] == "truncated_heat_local"].iloc[0]
    assert stochastic["n_runs"] == 2
    assert 0.0 <= stochastic["empirical_coverage"] <= 1.0
    assert stochastic["absolute_value_error_ci_low"] <= stochastic[
        "absolute_value_error_ci_high"
    ]
    write_ring_summary_atomic(summary, config["summary_output"])
    png, pdf = plot_ring_planning_figure(
        summary,
        config,
        png_path=config["figure_png"],
        pdf_path=config["figure_pdf"],
    )
    assert Path(png).stat().st_size > 0
    assert Path(pdf).stat().st_size > 0
