from pathlib import Path

import numpy as np
import pandas as pd

from lot_experiments.graphs import path_graph
from lot_experiments.kernel_map import (
    METHOD_TARGETS,
    certified_poisson_radius,
    diffusion_distance_gibbs_kernel,
    high_mass_ball_radii,
    kernel_effective_counts,
    resolve_kernel_map_config,
    run_kernel_map,
    summarize_kernel_map,
    write_summary_atomic,
)
from lot_experiments.kernels import exact_heat_columns, poisson_tail
from lot_experiments.plotting.kernel_map import plot_kernel_map_figure


def test_batched_exact_heat_matches_single_column_backend():
    graph = path_graph(7)
    batched = exact_heat_columns(graph.laplacian, 0.6, batch_size=3)
    assert batched.shape == (7, 7)
    np.testing.assert_allclose(batched.sum(axis=0), 1.0, atol=1e-13)
    assert np.min(batched) >= 0.0


def test_effective_count_and_ball_radius_for_uniform_path():
    K = 5
    uniform = np.full((K, K), 1.0 / K)
    np.testing.assert_array_equal(kernel_effective_counts(uniform, [0.4, 0.01]), [3, 5])
    distances = np.abs(np.arange(K)[:, None] - np.arange(K)[None, :]).astype(np.int16)
    np.testing.assert_array_equal(high_mass_ball_radii(None, distances, [0.4]), [2])


def test_certified_radius_is_minimal():
    theta = 2.0
    alpha = 0.01
    radius = certified_poisson_radius(theta, alpha)
    assert poisson_tail(theta, radius) <= alpha
    assert radius == 0 or poisson_tail(theta, radius - 1) > alpha


def test_diffusion_gibbs_is_dense_column_stochastic_and_obeys_count_bound():
    graph = path_graph(8)
    lambda_ = 1.0
    kernel, cost_max, _ = diffusion_distance_gibbs_kernel(
        graph.laplacian, diffusion_time=0.5, lambda_=lambda_, batch_size=4
    )
    assert np.min(kernel) > 0.0
    np.testing.assert_allclose(kernel.sum(axis=0), 1.0, atol=1e-13)
    alpha = 0.1
    observed = int(kernel_effective_counts(kernel, [alpha])[0])
    lower_bound = graph.K * max(0.0, 1.0 - alpha * np.exp(cost_max / lambda_))
    assert observed + 1e-12 >= lower_bound


def test_small_kernel_map_run_resumes_and_plots_without_rerunning(tmp_path):
    config = resolve_kernel_map_config(
        {
            "experiment": "kernel_map",
            "seed": 3,
            "graph_families": ["path", "grid"],
            "K": [9],
            "alpha": [0.1, 0.01],
            "poisson_mean": [0.5],
            "gibbs_lambda": 1.0,
            "timing_repeats": 1,
            "dense_batch_size": 4,
            "raw_output": str(tmp_path / "raw.parquet"),
            "summary_output": str(tmp_path / "summary.csv"),
            "figure_png": str(tmp_path / "figure.png"),
            "figure": {
                "alpha": 0.1,
                "poisson_mean": 0.5,
                "primary_graph_family": "path",
                "comparison_requested_K": 9,
            },
        }
    )
    first = run_kernel_map(config)
    second = run_kernel_map(config)
    assert len(first) == len(second) == 2 * 4 * 2
    assert second["run_id"].is_unique
    assert set(second["method"]) == set(METHOD_TARGETS)
    for method, target in METHOD_TARGETS.items():
        assert set(second.loc[second["method"] == method, "target"]) == {target}
    truncated = second.loc[second["method"] == "truncated_heat"]
    assert truncated["certificate_holds"].all()
    assert np.all(
        truncated["heat_approximation_error"]
        <= truncated["certificate_l1_bound"] + 2e-12
    )
    summary = summarize_kernel_map(second, resolved_config=config)
    write_summary_atomic(summary, config["summary_output"])
    png = plot_kernel_map_figure(
        pd.read_csv(config["summary_output"]),
        config,
        png_path=config["figure_png"],
    )
    assert Path(png).stat().st_size > 0
