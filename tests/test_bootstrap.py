from pathlib import Path

from lot_experiments.config import config_json, load_config
from lot_experiments.counters import OperationCounters
from lot_experiments.reproducibility import derive_seed, rng_for
from lot_experiments.results import result_row, validate_results, write_results_atomic


def test_configuration_overrides_are_typed_and_deterministic():
    config = load_config(
        Path("configs/numerics.yaml"),
        ["graph.K=15", "heat.diffusion_time=1.25", "enabled=true"],
    )
    assert config["graph"]["K"] == 15
    assert config["heat"]["diffusion_time"] == 1.25
    assert config["enabled"] is True
    assert config_json(config) == config_json(config)


def test_component_seeds_and_generators_are_reproducible_and_separate():
    assert derive_seed(7, "graph") == derive_seed(7, "graph")
    assert derive_seed(7, "graph") != derive_seed(7, "planner")
    assert rng_for(7, "graph").integers(2**31) == rng_for(7, "graph").integers(2**31)


def test_counters_merge_unique_actions_without_double_counting():
    first = OperationCounters(transition_calls=2)
    second = OperationCounters(transition_calls=3)
    first.touch_actions([1, 2])
    second.touch_actions([2, 3])
    first.merge(second)
    assert first.transition_calls == 5
    assert first.action_evaluations == 4
    assert first.unique_actions_touched == 3


def test_atomic_csv_result_round_trip(tmp_path):
    row = result_row(
        experiment="unit",
        run_id="atomic",
        seed=0,
        method="full_exact_heat",
        target="exact_heat",
        status="complete",
        git_commit="deadbeef",
        config_json=config_json({"seed": 0}),
    )
    destination = write_results_atomic([row], tmp_path / "rows.csv")
    assert destination.exists()
    import pandas as pd

    frame = pd.read_csv(destination)
    validate_results(frame)

