"""Tests for modules/preprocessing.py. Run from the project root:

    python -m pytest tests/test_preprocessing.py -v
"""

import numpy as np
import pandas as pd
import pytest

from modules.preprocessing import identify_column_types, preprocess_for_ml


@pytest.fixture
def customers():
    """Small customer dataset (the example from the project documents)."""
    return pd.DataFrame({
        "Customer_ID": ["C001", "C002", "C003", "C004", "C005", "C006"],
        "Age": [22, 25, np.nan, 250, 27, 30],
        "Income": [35000, 42000, 39000, 45000, -5000, 40000],
        "City": ["Pune", "Mumbai", "Pune", "Mumbai", "Pune", None],
        "Purchase": [500, 700, 600, 800, 550, 650],
    })


# ---------------------------------------------------------------- types
def test_identify_types_basic():
    df = pd.DataFrame({
        "n_int": [1, 2], "n_float": [1.5, 2.5], "n_nullable": pd.array([1, None], dtype="Int64"),
        "text": ["a", "b"], "flag": [True, False],
        "cat": pd.Categorical(["x", "y"]),
        "when": pd.to_datetime(["2025-01-01", "2025-01-02"]),
    })
    num, cat, ign = identify_column_types(df)
    assert num == ["n_int", "n_float", "n_nullable"]
    assert cat == ["text", "flag", "cat"]
    assert ign == ["when"]


def test_identify_types_old_object_dtype():
    df = pd.DataFrame({"t": pd.Series(["a", "b"], dtype="object")})
    assert identify_column_types(df)[1] == ["t"]


# ------------------------------------------------- matrix / index mapping
def test_output_is_numeric_without_nan_or_inf(customers):
    res = preprocess_for_ml(customers, columns=["Age", "Income", "City", "Purchase"])
    assert not res.X.isna().any().any()
    assert np.isfinite(res.X.to_numpy()).all()
    assert all(pd.api.types.is_float_dtype(t) for t in res.X.dtypes)
    assert res.feature_names == list(res.X.columns)


def test_default_index_preserved(customers):
    res = preprocess_for_ml(customers, columns=["Age", "Income"])
    assert res.X.index.equals(customers.index)
    assert len(res.X) == len(customers)


def test_custom_unsorted_index_preserved(customers):
    df = customers.copy()
    df.index = [105, 7, 42, 3, 99, 1]
    res = preprocess_for_ml(df, columns=["Age", "Income", "City"])
    assert list(res.X.index) == [105, 7, 42, 3, 99, 1]
    # row 42 is the one with the missing Age -> imputed with the median (27.0)
    unscaled = preprocess_for_ml(df, columns=["Age"], scaler="none")
    assert unscaled.X.loc[42, "Age"] == 27.0


def test_duplicate_index_labels_preserved():
    df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "c": ["x", "y", "x", None]},
                      index=[0, 0, 1, 1])
    res = preprocess_for_ml(df, scaler="none")
    assert list(res.X.index) == [0, 0, 1, 1]
    assert res.X["a"].tolist() == [1.0, 2.0, 2.0, 4.0]
    assert res.X["c_x"].tolist() == [1.0, 0.0, 1.0, 1.0]  # None -> mode "x"


def test_original_dataframe_not_modified(customers):
    before = customers.copy(deep=True)
    preprocess_for_ml(customers)
    pd.testing.assert_frame_equal(customers, before)


# ------------------------------------------------------- missing report
def test_missing_report_shows_original_counts(customers):
    res = preprocess_for_ml(customers, columns=["Age", "City"])
    rep = res.missing_report.set_index("column")
    assert rep.loc["Age", "missing_count"] == 1
    assert rep.loc["Age", "missing_pct"] == round(100 / 6, 2)
    assert rep.loc["Age", "fill_strategy"] == "median"
    assert rep.loc["City", "missing_count"] == 1
    assert rep.loc["City", "fill_strategy"] == "mode"
    assert rep.loc["City", "fill_value"] == "Pune"


def test_blank_strings_count_as_missing():
    df = pd.DataFrame({"c": ["a", "", "  ", "a", None, "b"]})
    res = preprocess_for_ml(df)
    rep = res.missing_report.iloc[0]
    assert rep["missing_count"] == 3
    assert rep["fill_value"] == "a"


def test_inf_treated_as_missing_and_reported():
    df = pd.DataFrame({"x": [1.0, 2.0, np.inf, 3.0, np.nan]})
    res = preprocess_for_ml(df, scaler="none")
    rep = res.missing_report.iloc[0]
    assert rep["missing_count"] == 2
    assert rep["non_finite_count"] == 1
    assert res.X["x"].tolist() == [1.0, 2.0, 2.0, 3.0, 2.0]


def test_nullable_int_column():
    df = pd.DataFrame({"x": pd.array([1, None, 3, 5], dtype="Int64")})
    res = preprocess_for_ml(df, scaler="none")
    assert res.X["x"].tolist() == [1.0, 3.0, 3.0, 5.0]
    assert res.missing_report.iloc[0]["missing_count"] == 1


# ------------------------------------------------------------- imputation
def test_median_and_mode_imputation_values():
    df = pd.DataFrame({"n": [10.0, 20.0, np.nan, 1000.0],
                       "c": ["u", "v", "v", None]})
    res = preprocess_for_ml(df, scaler="none")
    assert res.X["n"].tolist() == [10.0, 20.0, 20.0, 1000.0]   # median 20
    assert res.X["c_v"].tolist() == [0.0, 1.0, 1.0, 1.0]       # mode "v"


# --------------------------------------------------------------- encoding
def test_onehot_columns_and_source_mapping(customers):
    res = preprocess_for_ml(customers, columns=["Age", "City"], scaler="none")
    assert set(res.feature_names) == {"Age", "City_Mumbai", "City_Pune"}
    assert res.feature_source["City_Pune"] == "City"
    assert res.feature_source["Age"] == "Age"
    assert set(np.unique(res.X[["City_Mumbai", "City_Pune"]].to_numpy())) <= {0.0, 1.0}


def test_frequency_encoding():
    df = pd.DataFrame({"c": ["a", "a", "a", "b"]})
    res = preprocess_for_ml(df, encoding="frequency")
    assert res.feature_names == ["c_freq"]
    assert res.X["c_freq"].tolist() == [0.75, 0.75, 0.75, 0.25]


def test_boolean_column_is_categorical():
    df = pd.DataFrame({"flag": [True, False, True]})
    res = preprocess_for_ml(df)
    assert res.categorical_columns == ["flag"]
    assert res.X.shape == (3, 2)


# ---------------------------------------------------------------- scaling
def test_robust_scaling_centres_on_median():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 1000.0]})
    res = preprocess_for_ml(df, scaler="robust")
    assert res.X["x"].median() == pytest.approx(0.0)


def test_standard_scaling_mean_zero_std_one():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    res = preprocess_for_ml(df, scaler="standard")
    assert res.X["x"].mean() == pytest.approx(0.0)
    assert res.X["x"].std(ddof=0) == pytest.approx(1.0)


def test_scaler_none_keeps_values():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    assert preprocess_for_ml(df, scaler="none").X["x"].tolist() == [1.0, 2.0, 3.0]


def test_onehot_columns_are_not_scaled():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "c": ["a", "a", "a", "b"]})
    res = preprocess_for_ml(df, scaler="standard")
    assert res.X["c_b"].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_constant_column_does_not_produce_nan():
    df = pd.DataFrame({"x": [5.0, 5.0, 5.0, 5.0], "y": [1.0, 2.0, 3.0, 4.0]})
    for sc in ("robust", "standard"):
        res = preprocess_for_ml(df, scaler=sc)
        assert np.isfinite(res.X.to_numpy()).all()


# ------------------------------------------------------------- edge cases
def test_all_missing_column_is_dropped_and_reported():
    df = pd.DataFrame({"x": [1.0, 2.0], "dead": [np.nan, np.nan]})
    res = preprocess_for_ml(df)
    assert res.feature_names == ["x"]
    assert res.dropped_columns.iloc[0]["column"] == "dead"


def test_datetime_column_ignored_and_reported():
    df = pd.DataFrame({"x": [1.0, 2.0], "when": pd.to_datetime(["2025-01-01", "2025-01-02"])})
    res = preprocess_for_ml(df)
    assert res.feature_names == ["x"]
    assert "when" in res.dropped_columns["column"].tolist()


def test_high_cardinality_column_skipped():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "email": ["a@x", "b@x", "c@x", "d@x"]})
    res = preprocess_for_ml(df, max_categories=3)
    assert res.feature_names == ["x"]
    assert "email" in res.dropped_columns["column"].tolist()


def test_single_row_dataframe():
    res = preprocess_for_ml(pd.DataFrame({"x": [3.0], "c": ["a"]}))
    assert res.X.shape == (1, 2)
    assert np.isfinite(res.X.to_numpy()).all()


def test_mixed_type_column_handled_as_categorical():
    df = pd.DataFrame({"age": pd.Series([20, 21, "Twenty", 25], dtype="object")})
    res = preprocess_for_ml(df)
    assert res.categorical_columns == ["age"]  # type problem is a data-quality issue, not ours


def test_integer_column_labels_are_converted_to_text():
    df = pd.DataFrame({0: [1.0, 2.0], 1: ["a", "b"]})
    res = preprocess_for_ml(df)
    assert all(isinstance(n, str) for n in res.feature_names)


def test_errors_on_bad_input():
    good = pd.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(ValueError):
        preprocess_for_ml(pd.DataFrame())
    with pytest.raises(ValueError):
        preprocess_for_ml(good, scaler="minmax")
    with pytest.raises(ValueError):
        preprocess_for_ml(good, encoding="label")
    with pytest.raises(ValueError):
        preprocess_for_ml(good, columns=["nope"])
    with pytest.raises(ValueError):
        preprocess_for_ml(pd.DataFrame({"d": [np.nan, np.nan]}))
    with pytest.raises(TypeError):
        preprocess_for_ml([1, 2, 3])


# ------------------------------------------------------ model compatibility
def test_matrix_is_accepted_by_isolation_forest_and_lof(customers):
    """Smoke test only: proves X is model-ready. The real models come later."""
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor

    res = preprocess_for_ml(customers, columns=["Age", "Income", "City", "Purchase"])
    IsolationForest(random_state=0).fit(res.X)
    labels = LocalOutlierFactor(n_neighbors=3).fit_predict(res.X)
    assert len(labels) == len(customers)
