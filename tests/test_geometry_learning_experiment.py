from pathlib import Path

import numpy as np

from lot_experiments.geometry_learning import (
    PRIMARY_TARGET,
    geometry_threshold_table,
    resolve_geometry_learning_config,
    run_geometry_learning,
    summarize_geometry_learning,
    write_geometry_summary_atomic,
    write_geometry_table_atomic,
)
from lot_experiments.plotting.learning import plot_geometry_learning_figure


def _small_config(tmp_path: Path):
    return resolve_geometry_learning_config(
        {
            "experiment": "geometry_learning",
            "seed": 7,
            "K": 5,
            "samples_per_action": [4, 8],
            "goal_fractions": [0.0, 0.4],
            "paired_seeds": 1,
            "gamma": 0.7,
            "reference": {"tolerance": 1e-8, "max_iterations": 1000},
            "uniform_probe_ablation": {"enabled": True},
            "bootstrap_repetitions": 20,
            "raw_output": str(tmp_path / "raw.parquet"),
            "summary_output": str(tmp_path / "summary.csv"),
            "table_output": str(tmp_path / "table.csv"),
            "figure_png": str(tmp_path / "figure.png"),
        }
    )


def test_small_m6_run_resumes_reuses_geometry_and_checks_envelope(tmp_path):
    config = _small_config(tmp_path)
    first = run_geometry_learning(config)
    second = run_geometry_learning(config)
    assert len(first) == len(second) == 20
    assert second["run_id"].is_unique
    assert set(second["status"]) == {"complete"}

    primary = second.loc[second["experiment_part"] == "primary"]
    learned = primary.loc[primary["method"] == "learned_transition_geometry"]
    assert set(learned["target"]) == {PRIMARY_TARGET}
    assert learned["envelope_holds"].astype(bool).all()
    assert learned["certificate_holds"].astype(bool).all()
    assert set(learned["transition_calls"]) == {20, 40}
    assert learned["downstream_transition_calls"].gt(0).all()
    assert learned["anchor_reference_goal"].nunique() == 1

    # One frozen reward-free geometry is reused by both reward goals.
    reuse_counts = learned.groupby(["replicate", "samples_per_action"])[
        "geometry_id"
    ].nunique()
    assert (reuse_counts == 1).all()
    goal_counts = learned.groupby(["replicate", "samples_per_action"])[
        "reward_goal"
    ].nunique()
    assert (goal_counts == 2).all()

    references = second.loc[second["experiment_part"] == "reference"]
    assert set(references["target"]) == {
        PRIMARY_TARGET,
        "raw_action_coordinate_cost",
        "maxent",
    }
    ablation = second.loc[second["experiment_part"] == "uniform_probe_ablation"]
    assert len(ablation) == 2
    assert ablation["fixed_point_q_error"].isna().all()

    summary = summarize_geometry_learning(second, resolved_config=config)
    table = geometry_threshold_table(second, config["value_error_thresholds"])
    assert len(table) == 2 * len(config["value_error_thresholds"])
    write_geometry_summary_atomic(summary, config["summary_output"])
    write_geometry_table_atomic(table, config["table_output"])
    png = plot_geometry_learning_figure(
        second,
        table,
        config,
        png_path=config["figure_png"],
    )
    assert Path(png).stat().st_size > 0


def test_geometry_error_is_reward_independent_within_a_frozen_batch(tmp_path):
    config = _small_config(tmp_path)
    raw = run_geometry_learning(config)
    learned = raw.loc[
        (raw["experiment_part"] == "primary")
        & (raw["method"] == "learned_transition_geometry")
    ]
    for _, group in learned.groupby(["replicate", "samples_per_action"]):
        np.testing.assert_allclose(group["cost_max_error"], group["cost_max_error"].iloc[0])
        np.testing.assert_allclose(
            group["theoretical_cost_certificate"],
            group["theoretical_cost_certificate"].iloc[0],
        )
