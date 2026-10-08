"""Tests for models/lof.py. Run from the project root:

    python -m pytest tests/test_preprocessing.py tests/test_isolation_forest.py tests/test_lof.py -v
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import LocalOutlierFactor

from models.lof import (
    COL_PREDICTION, COL_SCORE, COL_STATUS,
    STATUS_ANOMALY, STATUS_NORMAL,
    LOFResult, run_lof,
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
def test_basic_run_returns_result(blob):
    out = run_lof(blob)
    assert isinstance(out, LOFResult)
    assert isinstance(out.model, LocalOutlierFactor)
    assert list(out.results.columns) == RESULT_COLUMNS
    assert out.feature_names == ["a", "b"]
    assert out.n_neighbors == 20
    assert out.contamination == "auto"
    assert out.warning_messages == []


def test_custom_parameters_are_used(blob):
    out = run_lof(blob, n_neighbors=10, contamination=0.1)
    assert out.model.n_neighbors == 10 and out.n_neighbors == 10
    assert out.model.contamination == 0.1 and out.contamination == 0.1


# ------------------------------------------------ 2. prediction output
def test_prediction_values_are_minus_one_or_one(blob):
    pred = run_lof(blob).results[COL_PREDICTION]
    assert set(pred.unique()) <= {-1, 1}
    assert pd.api.types.is_integer_dtype(pred)


# ------------------------------------------------ 3. status output
def test_status_matches_prediction_and_uses_agreed_wording(blob):
    res = run_lof(blob, contamination=0.05).results
    assert STATUS_ANOMALY == "Potential Anomaly" and STATUS_NORMAL == "Normal"
    assert set(res[COL_STATUS].unique()) <= {"Normal", "Potential Anomaly"}
    assert (res.loc[res[COL_PREDICTION] == -1, COL_STATUS] == "Potential Anomaly").all()
    assert (res.loc[res[COL_PREDICTION] == 1, COL_STATUS] == "Normal").all()
    assert not res[COL_STATUS].str.contains("error", case=False).any()


def test_contamination_controls_number_flagged(blob):
    out = run_lof(blob, contamination=0.1)
    assert abs(out.n_anomalies - 0.1 * len(blob)) <= 2
    assert len(out.anomalies) == out.n_anomalies
    assert (out.anomalies[COL_STATUS] == STATUS_ANOMALY).all()


# ------------------------------------------------ 4. score output
def test_score_is_finite_float_and_consistent_with_threshold(blob):
    out = run_lof(blob, contamination=0.05)
    score = out.results[COL_SCORE]
    assert pd.api.types.is_float_dtype(score)
    assert np.isfinite(score).all()
    assert (score > 0).all()
    flagged = out.results[COL_PREDICTION] == -1
    assert (score[flagged] > out.score_threshold).all()
    assert (score[~flagged] <= out.score_threshold).all()


def test_auto_threshold_is_one_point_five(blob):
    assert run_lof(blob).score_threshold == pytest.approx(1.5)


def test_score_is_sklearn_negative_outlier_factor_times_minus_one(blob):
    out = run_lof(blob, n_neighbors=15)
    reference = LocalOutlierFactor(n_neighbors=15).fit(blob.to_numpy())
    np.testing.assert_allclose(out.results[COL_SCORE].to_numpy(),
                               -reference.negative_outlier_factor_)


def test_normal_records_score_near_one_and_outliers_much_higher(blob):
    score = run_lof(blob).results[COL_SCORE]
    assert 0.9 < score.iloc[:200].median() < 1.3
    assert score.iloc[200:].min() > 3 * score.iloc[:200].median()


# ------------------------------------------------ 5. index preservation
def test_custom_unsorted_and_string_index_preserved(blob):
    df = blob.copy()
    df.index = [f"row_{i}" for i in range(len(df) - 1, -1, -1)]
    assert run_lof(df).results.index.equals(df.index)

    df2 = blob.copy()
    df2.index = np.arange(len(df2))[::-1] + 1000
    assert run_lof(df2).results.index.equals(df2.index)


def test_scores_follow_the_records_not_the_row_order(blob):
    """Shuffling the rows must give every record the same score."""
    plain = run_lof(blob).results[COL_SCORE]
    shuffled = blob.sample(frac=1.0, random_state=5)
    out = run_lof(shuffled).results[COL_SCORE]
    np.testing.assert_allclose(out.loc[plain.index].to_numpy(), plain.to_numpy())


# ------------------------------------------------ 6. duplicate index labels
def test_duplicate_index_labels_preserved(blob):
    df = blob.copy()
    df.index = [i // 2 for i in range(len(df))]          # every label appears twice
    out = run_lof(df)
    assert len(out.results) == len(df)
    assert out.results.index.equals(df.index)
    assert not out.results.index.is_unique
    plain = run_lof(blob).results
    np.testing.assert_allclose(out.results[COL_SCORE].to_numpy(), plain[COL_SCORE].to_numpy())


# ------------------------------------------------ 7. input not modified
def test_input_dataframe_is_not_modified(blob):
    df = blob.copy()
    df.index = [10 + i for i in range(len(df))]
    before = df.copy(deep=True)
    run_lof(df)
    pd.testing.assert_frame_equal(df, before)


# ------------------------------------------------ 8. invalid n_neighbors
@pytest.mark.parametrize("bad", [0, -3, 1.5, "5", True, [5]])
def test_invalid_n_neighbors(blob, bad):
    with pytest.raises(ValueError):
        run_lof(blob, n_neighbors=bad)


def test_n_neighbors_none_uses_default(blob):
    assert run_lof(blob, n_neighbors=None).n_neighbors == 20


def test_minimum_n_neighbors_is_accepted(blob):
    assert run_lof(blob, n_neighbors=1).n_records == len(blob)


def test_numpy_integer_n_neighbors_is_accepted(blob):
    assert run_lof(blob, n_neighbors=np.int64(12)).n_neighbors == 12


# ------------------------------------------------ 9. invalid contamination
@pytest.mark.parametrize("bad", [0, 0.0, -0.1, 0.51, 1, "bad", True, None, float("nan")])
def test_invalid_contamination(blob, bad):
    with pytest.raises(ValueError):
        run_lof(blob, contamination=bad)


def test_valid_contamination_boundary(blob):
    assert run_lof(blob, contamination=0.5).n_records == len(blob)


# ------------------------------------------------ 10-12. bad input values
def test_nan_rejected_with_column_name():
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ValueError, match="x"):
        run_lof(df)


def test_nullable_integer_with_na_rejected():
    df = pd.DataFrame({"x": pd.array([1, None, 3, 4], dtype="Int64")})
    with pytest.raises(ValueError, match="missing"):
        run_lof(df)


@pytest.mark.parametrize("bad_value", [np.inf, -np.inf])
def test_infinite_values_rejected(bad_value):
    df = pd.DataFrame({"x": [1.0, bad_value, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ValueError, match="infinite"):
        run_lof(df)


def test_non_numeric_column_rejected():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "city": ["a", "b", "c", "d"]})
    with pytest.raises(ValueError, match="city"):
        run_lof(df)


@pytest.mark.parametrize("bad_input", [[[1.0, 2.0], [3.0, 4.0]], np.ones((5, 2)), None])
def test_non_dataframe_rejected(bad_input):
    with pytest.raises(TypeError):
        run_lof(bad_input)


def test_integer_dtype_input_is_accepted():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100], "y": [2, 3, 4, 5, 6, 200]})
    assert run_lof(df).n_records == 6


# ------------------------------------------------ 13-14. empty / one row
def test_empty_dataframe_rejected():
    with pytest.raises(ValueError):
        run_lof(pd.DataFrame())
    with pytest.raises(ValueError):
        run_lof(pd.DataFrame({"x": pd.Series([], dtype=float)}))


def test_no_feature_columns_rejected():
    with pytest.raises(ValueError):
        run_lof(pd.DataFrame(index=[0, 1, 2]))


def test_one_row_dataset_rejected():
    with pytest.raises(ValueError, match="at least"):
        run_lof(pd.DataFrame({"x": [1.0]}))


# ------------------------------------------------ 15. n_neighbors vs number of rows
def test_too_large_n_neighbors_is_rejected_not_reduced(blob):
    with pytest.raises(ValueError, match="too large"):
        run_lof(blob, n_neighbors=len(blob))          # needs at most n - 1
    with pytest.raises(ValueError, match="too large"):
        run_lof(blob, n_neighbors=10_000)


def test_largest_valid_n_neighbors_is_accepted():
    df = pd.DataFrame({"x": np.arange(12, dtype=float)})
    out = run_lof(df, n_neighbors=11)
    assert out.n_neighbors == 11 and out.n_records == 12


def test_default_n_neighbors_adapts_to_small_datasets():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 50.0]})
    out = run_lof(df)                                  # 6 rows -> default 5, not 20
    assert out.n_neighbors == 5
    assert out.model.n_neighbors == 5
    assert out.n_records == 6


def test_two_row_dataset_runs_with_one_neighbor():
    out = run_lof(pd.DataFrame({"x": [1.0, 2.0]}))
    assert out.n_neighbors == 1 and out.n_records == 2


# ------------------------------------------------ 16. obvious unusual observations
def test_obvious_outliers_get_top_scores_and_are_flagged(blob):
    out = run_lof(blob, contamination=3 / len(blob))
    assert set(out.results[COL_SCORE].nlargest(3).index) == {200, 201, 202}
    flagged = set(out.anomalies.index)
    assert {200, 201, 202} <= flagged
    assert len(flagged) <= 6


def test_obvious_outliers_flagged_with_auto_contamination(blob):
    out = run_lof(blob)
    assert {200, 201, 202} <= set(out.anomalies.index)
    assert out.n_anomalies < len(blob) / 4


def test_lof_finds_a_local_outlier_next_to_a_dense_cluster():
    """The point of LOF: a point only 1.0 away from a very tight cluster is
    unusual locally, although 1.0 is a normal spacing in the sparse cluster."""
    rng = np.random.default_rng(1)
    dense = rng.normal([0.0, 0.0], 0.1, size=(100, 2))
    sparse = rng.normal([30.0, 0.0], 3.0, size=(100, 2))
    local_outlier = np.array([[1.0, 0.0]])
    df = pd.DataFrame(np.vstack([dense, sparse, local_outlier]), columns=["a", "b"])
    out = run_lof(df, n_neighbors=20)
    assert out.results[COL_SCORE].idxmax() == 200
    assert out.results.loc[200, COL_SCORE] > 3 * out.results[COL_SCORE].iloc[:200].median()
    assert 200 in out.anomalies.index


def test_identical_rows_flag_nothing_and_do_not_crash():
    df = pd.DataFrame(np.ones((30, 3)), columns=list("xyz"))
    out = run_lof(df, n_neighbors=5)
    assert out.n_anomalies == 0
    assert out.results[COL_SCORE].to_numpy() == pytest.approx(1.0)


def test_many_duplicate_rows_produce_a_visible_warning():
    """More identical rows than n_neighbors makes LOF scores unreliable.
    That must be reported, not hidden."""
    rng = np.random.default_rng(2)
    df = pd.DataFrame(np.vstack([np.ones((30, 2)), rng.normal(3, 1, size=(5, 2))]),
                      columns=["a", "b"])
    with pytest.warns(UserWarning, match="Duplicate"):
        out = run_lof(df, n_neighbors=5)
    assert any("Duplicate" in m for m in out.warning_messages)
    assert np.isfinite(out.results[COL_SCORE]).all()


def test_constant_column_is_allowed(blob):
    df = blob.copy()
    df["const"] = 5.0
    assert run_lof(df).n_records == len(df)


# ------------------------------------------------ 17. compatibility with preprocessing.py
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
    out = run_lof(prep.X)
    assert out.results.index.equals(customers.index)
    assert out.feature_names == prep.feature_names
    assert out.n_records == len(customers)


def test_injected_unusual_customer_is_top_scored(customers):
    prep = preprocess_for_ml(customers)
    out = run_lof(prep.X, contamination=0.02)
    assert out.results[COL_SCORE].idxmax() == 123 + 500
    assert (123 + 500) in out.anomalies.index
    row = out.results.loc[123 + 500]
    assert row[COL_STATUS] == STATUS_ANOMALY and row[COL_PREDICTION] == -1


def test_preprocessing_result_is_not_modified(customers):
    prep = preprocess_for_ml(customers)
    before = prep.X.copy(deep=True)
    run_lof(prep.X)
    pd.testing.assert_frame_equal(prep.X, before)


# ------------------------------------------------ 18. repeated execution
def test_repeated_runs_give_identical_results(blob):
    pd.testing.assert_frame_equal(run_lof(blob).results, run_lof(blob).results)
    pd.testing.assert_frame_equal(run_lof(blob, n_neighbors=7, contamination=0.05).results,
                                  run_lof(blob, n_neighbors=7, contamination=0.05).results)


def test_repeated_runs_on_preprocessing_output_are_identical(customers):
    prep = preprocess_for_ml(customers)
    pd.testing.assert_frame_equal(run_lof(prep.X).results, run_lof(prep.X).results)


# ------------------------------------------------ 19. one row per input record
@pytest.mark.parametrize("n_rows", [2, 3, 10, 57, 200])
def test_one_result_row_per_input_record(n_rows):
    rng = np.random.default_rng(n_rows)
    df = pd.DataFrame(rng.normal(size=(n_rows, 3)), columns=list("abc"))
    out = run_lof(df)
    assert out.n_records == len(out.results) == n_rows
    assert out.results.index.equals(df.index)
    assert not out.results.isna().any().any()
