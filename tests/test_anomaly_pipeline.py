"""Tests for modules/anomaly_pipeline.py."""

import numpy as np
import pandas as pd
import pytest

from models.isolation_forest import run_isolation_forest
from models.lof import run_lof
from modules.anomaly_analysis import (
    HIGH_CONFIDENCE,
    IF_RESULT_COL,
    LOF_RESULT_COL,
    NORMAL,
    STATUS_COL,
    combine_anomaly_results,
)
from modules.anomaly_dashboard import RECORD_ID_COL, build_anomaly_table, get_anomaly_summary
from modules.anomaly_pipeline import AnomalyPipelineResult, run_anomaly_pipeline
from modules.feature_selection import select_features
from modules.preprocessing import preprocess_for_ml

N = 200


def make_dataset(index_kind="default", n=N, seed=0):
    """Normal cluster + identifier columns + ONE obvious outlier in the last row."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Customer ID": np.arange(1001, 1001 + n),
        "Name": [f"Person {i}" for i in range(n)],
        "Email": [f"person{i}@example.com" for i in range(n)],
        "Age": rng.normal(40, 5, n).round(1),
        "Income": rng.normal(50000, 5000, n).round(2),
        "Gender": rng.choice(["Male", "Female"], n),
        "City": rng.choice(["Pune", "Mumbai", "Nagpur"], n),
    })
    df.iloc[n - 1, df.columns.get_loc("Age")] = 95.0
    df.iloc[n - 1, df.columns.get_loc("Income")] = 5_000_000.0

    if index_kind == "string":
        df.index = [f"r{i}" for i in range(n)]
    elif index_kind == "shuffled":
        df.index = rng.permutation(n) + 500
    return df


@pytest.fixture
def df():
    return make_dataset()


@pytest.fixture(params=["default", "string", "shuffled"])
def indexed_df(request):
    return make_dataset(request.param)


# ----------------------------------------------------------------------
# Structure
# ----------------------------------------------------------------------
def test_returns_pipeline_result_with_dashboard_outputs(df):
    result = run_anomaly_pipeline(df)

    assert isinstance(result, AnomalyPipelineResult)
    assert result.n_records == len(df)
    assert set(result.summary) == {
        "total_records", "if_anomalies", "lof_anomalies", "common_anomalies",
    }
    assert list(result.table.columns) == [RECORD_ID_COL, IF_RESULT_COL, LOF_RESULT_COL, STATUS_COL]
    assert list(result.analysis.columns) == [IF_RESULT_COL, LOF_RESULT_COL, STATUS_COL]
    assert isinstance(result.warning_messages, list)


def test_summary_and_table_come_from_the_dashboard_module(df):
    result = run_anomaly_pipeline(df)
    assert result.summary == get_anomaly_summary(result.analysis)
    pd.testing.assert_frame_equal(result.table, build_anomaly_table(result.analysis))


def test_summary_is_consistent_with_detectors_and_status(df):
    result = run_anomaly_pipeline(df)
    status = result.analysis[STATUS_COL]

    assert result.summary["total_records"] == len(df)
    assert result.summary["if_anomalies"] == result.isolation_forest.n_anomalies
    assert result.summary["lof_anomalies"] == result.lof.n_anomalies
    assert result.summary["common_anomalies"] == int((status == HIGH_CONFIDENCE).sum())


# ----------------------------------------------------------------------
# Reuse of the existing implementations (no duplicated logic)
# ----------------------------------------------------------------------
def test_matches_calling_the_existing_modules_directly(df):
    result = run_anomaly_pipeline(df)

    selection = select_features(df)
    prep = preprocess_for_ml(df, columns=selection.selected_columns)
    if_out = run_isolation_forest(prep.X)
    lof_out = run_lof(prep.X)
    expected = combine_anomaly_results(if_out.results, lof_out.results)

    assert result.feature_selection.selected_columns == selection.selected_columns
    pd.testing.assert_frame_equal(result.preprocessing.X, prep.X)
    pd.testing.assert_frame_equal(result.isolation_forest.results, if_out.results)
    pd.testing.assert_frame_equal(result.lof.results, lof_out.results)
    pd.testing.assert_frame_equal(result.analysis, expected)


def test_is_deterministic(df):
    first = run_anomaly_pipeline(df)
    second = run_anomaly_pipeline(df)
    pd.testing.assert_frame_equal(first.analysis, second.analysis)
    pd.testing.assert_frame_equal(first.scores, second.scores)


# ----------------------------------------------------------------------
# Feature selection is really applied
# ----------------------------------------------------------------------
def test_identifier_columns_are_not_used_as_features(df):
    result = run_anomaly_pipeline(df)
    features = result.preprocessing.feature_names

    assert "Age" in features and "Income" in features
    assert not any(f.startswith(("Customer ID", "Name", "Email")) for f in features)

    excluded = set(result.feature_selection.excluded_columns["column"])
    assert excluded == {"Customer ID", "Name", "Email"}


def test_force_include_and_exclude_columns_are_passed_through(df):
    forced = run_anomaly_pipeline(df, force_include=["Customer ID"])
    assert "Customer ID" in forced.preprocessing.feature_names

    excluded = run_anomaly_pipeline(df, exclude_columns=["Income"])
    assert "Income" not in excluded.preprocessing.feature_names


def test_columns_argument_restricts_features(df):
    result = run_anomaly_pipeline(df, columns=["Age", "Income"])
    assert result.preprocessing.feature_names == ["Age", "Income"]


# ----------------------------------------------------------------------
# Row identity and index preservation
# ----------------------------------------------------------------------
def test_original_index_is_preserved_at_every_stage(indexed_df):
    result = run_anomaly_pipeline(indexed_df)
    original = indexed_df.index

    assert result.preprocessing.X.index.equals(original)
    assert result.isolation_forest.results.index.equals(original)
    assert result.lof.results.index.equals(original)
    assert result.analysis.index.equals(original)
    assert result.table.index.equals(original)
    assert result.scores.index.equals(original)


def test_record_id_is_the_original_index_label(indexed_df):
    result = run_anomaly_pipeline(indexed_df)
    assert result.table[RECORD_ID_COL].tolist() == indexed_df.index.tolist()


def test_obvious_outlier_is_flagged_under_its_original_label(indexed_df):
    result = run_anomaly_pipeline(indexed_df)
    outlier_label = indexed_df.index[-1]

    assert outlier_label in result.flagged.index
    assert result.analysis.loc[outlier_label, STATUS_COL] != NORMAL


# ----------------------------------------------------------------------
# Anomalies are never deleted; input is never modified
# ----------------------------------------------------------------------
def test_no_record_is_removed(df):
    result = run_anomaly_pipeline(df)

    assert len(result.analysis) == len(df)
    assert len(result.table) == len(df)
    assert set(result.analysis.index) == set(df.index)
    assert len(result.flagged) < len(df)          # flagged is only a filtered view
    assert set(result.flagged.index) <= set(df.index)
    assert (result.flagged[STATUS_COL] != NORMAL).all()


def test_input_dataframe_is_not_modified(df):
    before = df.copy(deep=True)
    run_anomaly_pipeline(df)
    pd.testing.assert_frame_equal(df, before)


def test_scores_property(df):
    result = run_anomaly_pipeline(df)
    assert list(result.scores.columns) == ["if_score", "lof_score"]
    assert len(result.scores) == len(df)


# ----------------------------------------------------------------------
# Parameters are passed through to the stages
# ----------------------------------------------------------------------
def test_detector_parameters_are_passed_through(df):
    result = run_anomaly_pipeline(
        df,
        n_estimators=50,
        if_contamination=0.05,
        random_state=7,
        n_neighbors=10,
        lof_contamination=0.05,
    )
    assert result.isolation_forest.n_estimators == 50
    assert result.isolation_forest.contamination == 0.05
    assert result.isolation_forest.random_state == 7
    assert result.lof.n_neighbors == 10
    assert result.lof.contamination == 0.05

    # about 5 % of 200 records are flagged by each detector
    assert 5 <= result.summary["if_anomalies"] <= 15
    assert 5 <= result.summary["lof_anomalies"] <= 15


def test_preprocessing_options_are_passed_through(df):
    freq = run_anomaly_pipeline(df, encoding="frequency")
    assert any(name.endswith("_freq") for name in freq.preprocessing.feature_names)
    assert freq.preprocessing.encoding == "frequency"

    standard = run_anomaly_pipeline(df, scaler="standard")
    assert standard.preprocessing.scaler == "standard"


# ----------------------------------------------------------------------
# Robustness
# ----------------------------------------------------------------------
def test_missing_values_are_handled_and_rows_kept(indexed_df):
    messy = indexed_df.copy()
    messy.iloc[3, messy.columns.get_loc("Age")] = np.nan
    messy.iloc[5, messy.columns.get_loc("City")] = None
    messy.iloc[7, messy.columns.get_loc("City")] = "   "

    result = run_anomaly_pipeline(messy)

    assert len(result.analysis) == len(messy)
    assert result.analysis.index.equals(messy.index)
    assert not result.preprocessing.X.isna().any().any()


def test_small_dataset_runs_end_to_end():
    small = pd.DataFrame({
        "Age": [25, 30, 28, 31, 29, 27, 26, 90, 30, 33],
        "Income": [40, 42, 41, 45, 43, 39, 44, 400, 46, 47],
    })
    result = run_anomaly_pipeline(small)

    assert len(result.table) == 10
    assert result.summary["total_records"] == 10
    assert result.analysis.index.equals(small.index)


# ----------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------
def test_not_a_dataframe_raises_type_error():
    with pytest.raises(TypeError):
        run_anomaly_pipeline([1, 2, 3])


def test_empty_dataframe_raises_value_error():
    with pytest.raises(ValueError):
        run_anomaly_pipeline(pd.DataFrame())


def test_duplicate_index_raises_clear_error():
    dup = pd.DataFrame({"Age": [20, 25, 31, 40, 22, 28]}, index=[0, 0, 1, 2, 3, 4])
    with pytest.raises(ValueError, match="duplicate"):
        run_anomaly_pipeline(dup)


def test_no_usable_features_raises_clear_error():
    ids_only = pd.DataFrame({
        "Name": [f"P{i}" for i in range(10)],
        "Customer ID": range(10),
    })
    with pytest.raises(ValueError, match="No features"):
        run_anomaly_pipeline(ids_only)


def test_invalid_stage_parameters_raise_value_error(df):
    with pytest.raises(ValueError):
        run_anomaly_pipeline(df, scaler="bogus")
    with pytest.raises(ValueError):
        run_anomaly_pipeline(df, n_neighbors=10_000)
    with pytest.raises(ValueError):
        run_anomaly_pipeline(df, if_contamination=0.9)
    with pytest.raises(ValueError):
        run_anomaly_pipeline(df, lof_contamination=0.9)
