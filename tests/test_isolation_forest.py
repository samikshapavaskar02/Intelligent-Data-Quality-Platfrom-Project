"""Tests for modules/isolation_forest.py. Run from the project root:

    python -m pytest tests/test_preprocessing.py tests/test_isolation_forest.py -v
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest

from models.isolation_forest import (
    COL_PREDICTION, COL_SCORE, COL_STATUS,
    STATUS_ANOMALY, STATUS_NORMAL,
    IsolationForestResult, run_isolation_forest,
)
from modules.preprocessing import preprocess_for_ml

RESULT_COLUMNS = [COL_PREDICTION, COL_STATUS, COL_SCORE]


@pytest.fixture
def blob():
    """200 normal 2-D points + 3 obvious outliers at index 200, 201, 202."""
    rng = np.random.default_rng(0)
    normal = rng.normal(0, 1, size=(200, 2))
    outliers = np.array([[15.0, 15.0], [-14.0, 16.0], [18.0, -17.0]])
    return pd.DataFrame(np.vstack([normal, outliers]), columns=["a", "b"])


# ------------------------------------------------ 1. basic execution
def test_basic_run_returns_result_for_every_record(blob):
    out = run_isolation_forest(blob)
    assert isinstance(out, IsolationForestResult)
    assert isinstance(out.model, IsolationForest)
    assert len(out.results) == len(blob) == out.n_records
    assert list(out.results.columns) == RESULT_COLUMNS
    assert out.feature_names == ["a", "b"]
    assert out.n_estimators == 100 and out.random_state == 42


def test_custom_parameters_are_used(blob):
    out = run_isolation_forest(blob, n_estimators=25, contamination=0.1, random_state=7)
    assert out.model.n_estimators == 25
    assert out.model.contamination == 0.1
    assert out.model.random_state == 7


# ------------------------------------------------ 2. prediction output
def test_prediction_values_are_minus_one_or_one(blob):
    pred = run_isolation_forest(blob).results[COL_PREDICTION]
    assert set(pred.unique()) <= {-1, 1}
    assert pd.api.types.is_integer_dtype(pred)


# ------------------------------------------------ 3. status output
def test_status_matches_prediction_and_uses_agreed_wording(blob):
    res = run_isolation_forest(blob, contamination=0.05).results
    assert set(res[COL_STATUS].unique()) <= {STATUS_NORMAL, STATUS_ANOMALY}
    assert STATUS_ANOMALY == "Potential Anomaly" and STATUS_NORMAL == "Normal"
    assert (res.loc[res[COL_PREDICTION] == -1, COL_STATUS] == STATUS_ANOMALY).all()
    assert (res.loc[res[COL_PREDICTION] == 1, COL_STATUS] == STATUS_NORMAL).all()
    assert not res[COL_STATUS].str.contains("error", case=False).any()


def test_contamination_controls_number_flagged(blob):
    out = run_isolation_forest(blob, contamination=0.1)
    expected = 0.1 * len(blob)
    assert abs(out.n_anomalies - expected) <= 2
    assert len(out.anomalies) == out.n_anomalies


# ------------------------------------------------ 4. score output
def test_score_is_finite_float_and_consistent_with_threshold(blob):
    out = run_isolation_forest(blob, contamination=0.05)
    score = out.results[COL_SCORE]
    assert pd.api.types.is_float_dtype(score)
    assert np.isfinite(score).all()
    assert ((score > 0) & (score <= 1)).all()
    flagged = out.results[COL_PREDICTION] == -1
    assert (score[flagged] > out.score_threshold).all()
    assert (score[~flagged] <= out.score_threshold).all()


def test_higher_score_means_more_unusual(blob):
    score = run_isolation_forest(blob).results[COL_SCORE]
    assert score.iloc[200:].min() > score.iloc[:200].median()


# ------------------------------------------------ 5. index preservation
def test_custom_unsorted_and_string_index_preserved(blob):
    df = blob.copy()
    df.index = [f"row_{i}" for i in range(len(df) - 1, -1, -1)]
    out = run_isolation_forest(df)
    assert out.results.index.equals(df.index)

    df2 = blob.copy()
    df2.index = np.arange(len(df2))[::-1] + 1000
    assert run_isolation_forest(df2).results.index.equals(df2.index)


def test_row_order_matches_input_order(blob):
    plain = run_isolation_forest(blob).results
    shuffled_index = blob.copy()
    shuffled_index.index = np.random.default_rng(1).permutation(len(blob))
    out = run_isolation_forest(shuffled_index).results
    np.testing.assert_allclose(out[COL_SCORE].to_numpy(), plain[COL_SCORE].to_numpy())


# ------------------------------------------------ 6. duplicate index labels
def test_duplicate_index_labels_preserved(blob):
    df = blob.copy()
    df.index = [i // 2 for i in range(len(df))]          # every label appears twice
    out = run_isolation_forest(df)
    assert len(out.results) == len(df)
    assert out.results.index.equals(df.index)
    assert not out.results.index.is_unique
    # same numbers as the unique-index run -> no row was merged or reordered
    plain = run_isolation_forest(blob).results
    np.testing.assert_allclose(out.results[COL_SCORE].to_numpy(), plain[COL_SCORE].to_numpy())


# ------------------------------------------------ 7. reproducibility
def test_same_random_state_gives_identical_results(blob):
    a = run_isolation_forest(blob, random_state=123).results
    b = run_isolation_forest(blob, random_state=123).results
    pd.testing.assert_frame_equal(a, b)


def test_default_random_state_is_deterministic(blob):
    pd.testing.assert_frame_equal(run_isolation_forest(blob).results,
                                  run_isolation_forest(blob).results)


def test_different_random_state_changes_scores(blob):
    a = run_isolation_forest(blob, random_state=1).results[COL_SCORE]
    b = run_isolation_forest(blob, random_state=2).results[COL_SCORE]
    assert not np.allclose(a, b)


# ------------------------------------------------ 8. input not modified
def test_input_dataframe_is_not_modified(blob):
    df = blob.copy()
    df.index = [10 + i for i in range(len(df))]
    before = df.copy(deep=True)
    run_isolation_forest(df)
    pd.testing.assert_frame_equal(df, before)


# ------------------------------------------------ 9. invalid parameters
@pytest.mark.parametrize("bad", [0, -5, 1.5, "100", True, None])
def test_invalid_n_estimators(blob, bad):
    with pytest.raises(ValueError):
        run_isolation_forest(blob, n_estimators=bad)


@pytest.mark.parametrize("bad", [0, 0.0, -0.1, 0.51, 1, "bad", True, None, float("nan")])
def test_invalid_contamination(blob, bad):
    with pytest.raises(ValueError):
        run_isolation_forest(blob, contamination=bad)


@pytest.mark.parametrize("bad", ["seed", 1.5, True])
def test_invalid_random_state(blob, bad):
    with pytest.raises(ValueError):
        run_isolation_forest(blob, random_state=bad)


def test_valid_boundary_parameters(blob):
    assert run_isolation_forest(blob, contamination=0.5, n_estimators=1).n_records == len(blob)
    assert run_isolation_forest(blob, contamination="auto", random_state=None).n_records == len(blob)


# ------------------------------------------------ 10. obvious outliers
def test_obvious_outliers_get_top_scores_and_are_flagged(blob):
    out = run_isolation_forest(blob, contamination=3 / len(blob))
    top3 = set(out.results[COL_SCORE].nlargest(3).index)
    assert top3 == {200, 201, 202}
    flagged = set(out.anomalies.index)
    assert {200, 201, 202} <= flagged
    assert len(flagged) <= 6


def test_obvious_outliers_flagged_with_auto_contamination(blob):
    out = run_isolation_forest(blob)
    assert {200, 201, 202} <= set(out.anomalies.index)
    assert out.n_anomalies < len(blob) / 4


def test_identical_rows_flag_nothing_and_do_not_crash():
    df = pd.DataFrame(np.ones((30, 3)), columns=list("xyz"))
    out = run_isolation_forest(df, contamination=0.1)
    assert out.n_anomalies == 0
    assert (out.results[COL_STATUS] == STATUS_NORMAL).all()


def test_constant_column_is_allowed(blob):
    df = blob.copy()
    df["const"] = 5.0
    assert run_isolation_forest(df).n_records == len(df)


def test_two_row_dataset_runs():
    out = run_isolation_forest(pd.DataFrame({"x": [1.0, 2.0]}))
    assert out.n_records == 2


# ------------------------------------------------ edge cases: bad input
def test_empty_dataframe_rejected():
    with pytest.raises(ValueError):
        run_isolation_forest(pd.DataFrame())
    with pytest.raises(ValueError):
        run_isolation_forest(pd.DataFrame({"x": pd.Series([], dtype=float)}))


def test_no_feature_columns_rejected():
    with pytest.raises(ValueError):
        run_isolation_forest(pd.DataFrame(index=[0, 1, 2]))


def test_one_row_dataset_rejected():
    with pytest.raises(ValueError, match="at least"):
        run_isolation_forest(pd.DataFrame({"x": [1.0]}))


def test_nan_rejected_with_column_name():
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="x"):
        run_isolation_forest(df)


def test_nullable_integer_with_na_rejected():
    df = pd.DataFrame({"x": pd.array([1, None, 3], dtype="Int64")})
    with pytest.raises(ValueError, match="missing"):
        run_isolation_forest(df)


@pytest.mark.parametrize("bad_value", [np.inf, -np.inf])
def test_infinite_values_rejected(bad_value):
    df = pd.DataFrame({"x": [1.0, bad_value, 3.0], "y": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="infinite"):
        run_isolation_forest(df)


def test_non_numeric_column_rejected():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "city": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="city"):
        run_isolation_forest(df)


@pytest.mark.parametrize("bad_input", [[[1.0, 2.0], [3.0, 4.0]], np.ones((5, 2)), None])
def test_non_dataframe_rejected(bad_input):
    with pytest.raises(TypeError):
        run_isolation_forest(bad_input)


def test_integer_dtype_input_is_accepted():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100], "y": [2, 3, 4, 5, 6, 200]})
    assert run_isolation_forest(df).n_records == 6


# ------------------------------------------------ 11. compatibility with preprocessing.py
@pytest.fixture
def customers():
    rng = np.random.default_rng(3)
    n = 300
    df = pd.DataFrame({
        "Customer_ID": [f"C{i:04d}" for i in range(n)],
        "Age": rng.integers(20, 60, n).astype(float),
        "Income": rng.normal(50_000, 8_000, n),
        "City": rng.choice(["Pune", "Mumbai", "Nagpur"], n),
        "Purchase": rng.normal(600, 80, n),
    })
    df.loc[rng.choice(n, 15, replace=False), "Age"] = np.nan      # missing numbers
    df.loc[rng.choice(n, 10, replace=False), "City"] = None       # missing categories
    df.loc[123, "Income"] = 5_000_000                             # injected unusual record
    df.index = np.arange(n) + 500                                 # non-default index
    return df


@pytest.mark.parametrize("scaler", ["robust", "standard", "none"])
@pytest.mark.parametrize("encoding", ["onehot", "frequency"])
def test_works_on_preprocessing_output(customers, scaler, encoding):
    prep = preprocess_for_ml(customers, scaler=scaler, encoding=encoding)
    out = run_isolation_forest(prep.X)
    assert out.results.index.equals(customers.index)
    assert out.feature_names == prep.feature_names
    assert out.n_records == len(customers)


def test_injected_unusual_customer_is_top_scored(customers):
    prep = preprocess_for_ml(customers)
    out = run_isolation_forest(prep.X, contamination=0.02)
    assert out.results[COL_SCORE].idxmax() == 123 + 500
    assert (123 + 500) in out.anomalies.index
    row = out.results.loc[123 + 500]
    assert row[COL_STATUS] == STATUS_ANOMALY and row[COL_PREDICTION] == -1


def test_preprocessing_result_is_not_modified(customers):
    prep = preprocess_for_ml(customers)
    before = prep.X.copy(deep=True)
    run_isolation_forest(prep.X)
    pd.testing.assert_frame_equal(prep.X, before)
