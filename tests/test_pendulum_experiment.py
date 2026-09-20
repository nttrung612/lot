import numpy as np

from lot_experiments.pendulum_experiment import run_pendulum_experiment
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
        "figure_pdf": str(tmp_path / "figure.pdf"),
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
    png, pdf = plot_pendulum_figure(
        rows,
        config,
        png_path=config["figure_png"],
        pdf_path=config["figure_pdf"],
    )
    assert png.exists() and pdf.exists()
