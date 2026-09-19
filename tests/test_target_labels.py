import pandas as pd
import pytest

from lot_experiments.config import config_json
from lot_experiments.results import (
    RESULT_COLUMNS,
    ResultSchemaError,
    require_common_target,
    result_row,
    validate_results,
)


def make_row(method, target):
    return result_row(
        experiment="unit",
        run_id=f"unit-{method}",
        seed=0,
        method=method,
        target=target,
        status="complete",
        git_commit="deadbeef",
        config_json=config_json({"seed": 0}),
    )


def test_result_rows_have_the_complete_common_schema():
    frame = validate_results([make_row("full_exact_heat", "exact_heat")])
    assert tuple(frame.columns) == RESULT_COLUMNS
    assert frame.loc[0, "target"] == "exact_heat"


def test_cross_target_error_comparison_is_rejected():
    rows = [
        make_row("full_exact_heat", "exact_heat"),
        make_row("uniform_maxent", "maxent"),
    ]
    with pytest.raises(ResultSchemaError, match="mixes targets"):
        require_common_target(rows)


def test_missing_target_is_rejected_even_if_dataframe_has_column():
    row = make_row("full_exact_heat", "exact_heat")
    row["target"] = ""
    with pytest.raises(ResultSchemaError, match="must have a target"):
        validate_results(pd.DataFrame([row]))


def test_experiment_specific_columns_are_preserved():
    row = make_row("exact_heat", "exact_heat")
    row["kernel_effective_count"] = 7
    frame = validate_results(pd.DataFrame([row]))
    assert frame.loc[0, "kernel_effective_count"] == 7
