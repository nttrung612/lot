import copy

import numpy as np
import pandas as pd
import pytest

from lot_experiments.pendulum_experiment import (
    resolve_pendulum_experiment_config,
    run_pendulum_experiment,
)
from lot_experiments.plotting.pendulum import plot_pendulum_figure


def _small_config(tmp_path):
    return {
        "seed": 3,
        "K": [7],
        "heat_scalings": ["physical_heat"],
        "diffusion_time": {"physical_heat": 0.02},
        "temperature": [0.1],
        "gamma": 0.8,
        "development_grid": [9, 9],
        "evaluation_reset_states": 3,
        "methods": ["full_exact_heat", "truncated_heat_local", "hard_max"],
        "primary_radius": 2,
        "radii": [1, 2],
        "planning": {
            "tolerance": 2e-6,
            "max_iterations": 500,
            "state_chunk_size": 32,
            "progress_interval": 0,
            "anderson_depth": 3,
        },
        "behavior": {"enabled": False},
        "radius_ablation": {"enabled": False},
        "figure": {"K": 7, "heat_scaling": "physical_heat", "temperature": 0.1},
        "raw_output": str(tmp_path / "raw.parquet"),
        "summary_output": str(tmp_path / "summary.csv"),
        "figure_png": str(tmp_path / "figure.png"),
    }


def test_small_pendulum_experiment_preserves_targets_and_local_cost(tmp_path):
    config = _small_config(tmp_path)
    rows = run_pendulum_experiment(config)
    assert set(rows["method"]) == {
        "full_exact_heat",
        "truncated_heat_local",
        "hard_max",
    }
    assert set(rows["status"]) == {"complete"}
    assert set(rows["execution_workers"]) == {1}
    assert set(rows["timing_mode"]) == {"isolated"}
    targets = rows.set_index("method")["target"].to_dict()
    assert targets == {
        "full_exact_heat": "exact_heat",
        "truncated_heat_local": "truncated_heat",
        "hard_max": "hard_max",
    }
    costs = rows.set_index("method")["actions_per_backup"]
    assert costs["truncated_heat_local"] < costs["full_exact_heat"]
    assert np.isclose(rows.loc[rows.method == "full_exact_heat", "mean_value_error"].iloc[0], 0.0)
    resumed = run_pendulum_experiment(config)
    assert resumed["run_id"].is_unique
    assert set(resumed["run_id"]) == set(rows["run_id"])
    png = plot_pendulum_figure(
        rows,
        config,
        png_path=config["figure_png"],
    )
    assert png.exists()


def test_parallel_pendulum_matches_serial_and_resumes(tmp_path):
    common = _small_config(tmp_path / "unused")
    common["temperature"] = [0.1, 0.2]

    serial_config = copy.deepcopy(common)
    serial_config.update(
        {
            "execution": {"workers": 1},
            "raw_output": str(tmp_path / "serial" / "raw.parquet"),
            "summary_output": str(tmp_path / "serial" / "summary.csv"),
        }
    )
    parallel_config = copy.deepcopy(common)
    parallel_config.update(
        {
            "execution": {"workers": 2},
            "raw_output": str(tmp_path / "parallel" / "raw.parquet"),
            "summary_output": str(tmp_path / "parallel" / "summary.csv"),
            "figure_png": str(tmp_path / "parallel" / "figure.png"),
        }
    )

    serial = run_pendulum_experiment(serial_config)
    parallel = run_pendulum_experiment(parallel_config)
    resumed = run_pendulum_experiment(parallel_config)

    assert len(serial) == len(parallel) == len(resumed) == 6
    assert parallel["run_id"].is_unique
    assert set(parallel["execution_workers"]) == {2}
    assert set(parallel["timing_mode"]) == {"concurrent"}
    assert set(resumed["run_id"]) == set(parallel["run_id"])

    excluded = {
        "run_id",
        "config_json",
        "geometry_preprocess_seconds",
        "online_seconds",
        "time_per_sweep",
        "peak_memory_mb",
        "execution_workers",
        "timing_mode",
    }
    comparable = [column for column in serial.columns if column not in excluded]
    sort_by = ["temperature", "method", "radius"]
    serial_values = serial.sort_values(sort_by)[comparable].reset_index(drop=True)
    parallel_values = parallel.sort_values(sort_by)[comparable].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        serial_values,
        parallel_values,
        check_dtype=False,
        check_exact=False,
        rtol=1e-13,
        atol=1e-13,
    )
    png = plot_pendulum_figure(
        parallel,
        parallel_config,
        png_path=parallel_config["figure_png"],
    )
    assert png.exists()


@pytest.mark.parametrize("workers", [0, -1, 1.5, True])
def test_pendulum_config_rejects_invalid_worker_count(workers):
    with pytest.raises(ValueError, match="execution.workers"):
        resolve_pendulum_experiment_config({"execution": {"workers": workers}})
