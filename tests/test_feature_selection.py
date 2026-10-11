"""Tests for modules/feature_selection.py."""

import numpy as np
import pandas as pd
import pytest

from modules.feature_selection import FeatureSelectionResult, select_features


def make_df(n=30):
    """Deterministic dataset: 4 useful features + 3 identifier-like columns."""
    idx = np.arange(n)
    return pd.DataFrame({
        "Customer ID": 1001 + idx,                               # sequential + ID name
        "Name": [f"Person {i}" for i in range(n)],               # name column
        "Email": [f"person{i}@example.com" for i in range(n)],   # email column
        "Age": 20 + (idx * 7) % 45,
        "Income": 30000 + ((idx ** 2) % 97) * 500.5,
        "Gender": ["Male", "Female"] * (n // 2),
        "City": ["Pune", "Mumbai", "Nagpur"] * (n // 3),
    })


def reasons(result):
    return dict(zip(result.excluded_columns["column"], result.excluded_columns["reason"]))


# ----------------------------------------------------------------------
# The PDF example: keep Age/Income, drop Customer ID / Name / Email
# ----------------------------------------------------------------------
def test_selects_meaningful_features_and_excludes_identifiers():
    result = select_features(make_df())

    assert isinstance(result, FeatureSelectionResult)
    assert result.selected_columns == ["Age", "Income", "Gender", "City"]
    assert set(reasons(result)) == {"Customer ID", "Name", "Email"}
    assert result.n_selected == 4


def test_numeric_and_categorical_split():
    result = select_features(make_df())
    assert result.numeric_columns == ["Age", "Income"]
    assert result.categorical_columns == ["Gender", "City"]


def test_excluded_columns_table_has_column_and_reason():
    result = select_features(make_df())
    assert list(result.excluded_columns.columns) == ["column", "reason"]
    assert all(isinstance(r, str) and r for r in result.excluded_columns["reason"])


def test_input_is_not_modified():
    df = make_df()
    before = df.copy(deep=True)
    select_features(df)
    pd.testing.assert_frame_equal(df, before)


def test_works_with_non_default_index():
    df = make_df()
    df.index = [f"r{i}" for i in range(len(df))]
    result = select_features(df)
    assert result.selected_columns == ["Age", "Income", "Gender", "City"]


def test_non_string_column_names():
    idx = np.arange(30)
    df = pd.DataFrame({0: (idx * 5) % 17, 1: (idx % 4) * 1.5 + 0.1 * (idx % 3)})
    result = select_features(df)
    assert result.selected_columns == [0, 1]


def test_no_features_selected_returns_empty_selection():
    df = make_df()[["Name", "Email"]]
    result = select_features(df)
    assert result.selected_columns == []
    assert result.numeric_columns == []
    assert result.categorical_columns == []
    assert set(reasons(result)) == {"Name", "Email"}


# ----------------------------------------------------------------------
# Name-based rules
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name", [
    "ID", "id", "CustomerID", "Customer ID", "customer_id", "user-id", "UserID",
    "Unnamed: 0", "Unnamed: 0.1", "Index", "S.No", "Serial No", "Row Number",
    "Full Name", "Email", "Phone Number", "PIN Code", "Zip", "Address",
])
def test_identifier_like_names_are_excluded(name):
    n = 30
    df = pd.DataFrame({
        name: (np.arange(n) * 37) % 101,   # not constant, not sequential
        "Age": 20 + (np.arange(n) * 7) % 45,
    })
    result = select_features(df)
    assert name not in result.selected_columns
    assert "identifier" in reasons(result)[name]
    assert "Age" in result.selected_columns


@pytest.mark.parametrize("name", [
    "Paid", "Valid", "Width", "Hidden", "Credit Score", "Body Mass Index",
    "Purchase Amount", "Rating",
])
def test_ordinary_names_are_not_mistaken_for_identifiers(name):
    n = 30
    df = pd.DataFrame({
        name: (np.arange(n) * 37) % 101,
        "Age": 20 + (np.arange(n) * 7) % 45,
    })
    result = select_features(df)
    assert name in result.selected_columns


def test_date_like_name_excluded_for_text_columns_only():
    n = 30
    df = pd.DataFrame({
        "Signup Date": ["2024-01-01", "2024-02-01", "2024-03-01"] * (n // 3),
        "Date Score": (np.arange(n) * 37) % 101,   # numeric: name rule does not apply
    })
    result = select_features(df)
    assert "Signup Date" not in result.selected_columns
    assert "date-like" in reasons(result)["Signup Date"]
    assert "Date Score" in result.selected_columns


# ----------------------------------------------------------------------
# Value-based rules
# ----------------------------------------------------------------------
def test_sequential_integers_are_excluded():
    n = 30
    df = pd.DataFrame({"Seq": np.arange(5, 5 + n), "Age": 20 + (np.arange(n) * 7) % 45})
    result = select_features(df)
    assert "Seq" not in result.selected_columns
    assert "sequential" in reasons(result)["Seq"]


def test_descending_sequence_is_excluded():
    n = 30
    df = pd.DataFrame({"Seq": np.arange(n, 0, -1), "Age": 20 + (np.arange(n) * 7) % 45})
    result = select_features(df)
    assert "Seq" not in result.selected_columns


def test_unique_but_not_constant_step_integers_are_kept():
    n = 30
    df = pd.DataFrame({"Squares": np.arange(n) ** 2, "Age": 20 + (np.arange(n) * 7) % 45})
    result = select_features(df)
    assert "Squares" in result.selected_columns


def test_constant_columns_are_excluded():
    n = 30
    df = pd.DataFrame({
        "Const Number": [7] * n,
        "Const Text": ["Yes"] * n,
        "Age": 20 + (np.arange(n) * 7) % 45,
    })
    result = select_features(df)
    assert result.selected_columns == ["Age"]
    assert "constant" in reasons(result)["Const Number"]
    assert "constant" in reasons(result)["Const Text"]


def test_all_missing_columns_are_excluded():
    n = 30
    df = pd.DataFrame({
        "Empty Number": [np.nan] * n,
        "Blank Text": ["   "] * n,
        "Age": 20 + (np.arange(n) * 7) % 45,
    })
    result = select_features(df)
    assert result.selected_columns == ["Age"]
    assert "missing" in reasons(result)["Empty Number"]
    assert "missing" in reasons(result)["Blank Text"]


def test_datetime_columns_are_excluded():
    n = 30
    df = pd.DataFrame({
        "Created": pd.date_range("2024-01-01", periods=n),
        "Age": 20 + (np.arange(n) * 7) % 45,
    })
    result = select_features(df)
    assert result.selected_columns == ["Age"]
    assert "datetime" in reasons(result)["Created"]


def test_high_cardinality_text_is_excluded():
    n = 60
    df = pd.DataFrame({
        "Label": [f"Item {i}" for i in range(n)],        # 60 distinct > 50
        "Age": 20 + (np.arange(n) * 7) % 45,
    })
    result = select_features(df)
    assert "Label" not in result.selected_columns
    assert "too many distinct" in reasons(result)["Label"]


def test_all_unique_text_is_excluded_even_in_small_data():
    df = pd.DataFrame({
        "Label": ["a", "b", "c", "d", "e", "f"],
        "Age": [20, 25, 21, 40, 22, 23],
    })
    result = select_features(df)
    assert "Label" not in result.selected_columns
    assert "every value is different" in reasons(result)["Label"]


def test_distinct_to_rows_ratio_rule():
    n = 40
    df = pd.DataFrame({
        "Label": [f"c{i % 25}" for i in range(n)],       # 25 distinct of 40 rows
        "Age": 20 + (np.arange(n) * 7) % 45,
    })
    result = select_features(df)
    assert "Label" not in result.selected_columns
    assert "relative to rows" in reasons(result)["Label"]


def test_small_categorical_columns_are_kept():
    df = pd.DataFrame({
        "Group": ["a", "b", "a", "b", "a"],
        "Size": ["x", "y", "z", "x", "y"],
        "Age": [20, 25, 21, 40, 22],
    })
    result = select_features(df)
    assert result.selected_columns == ["Group", "Size", "Age"]


def test_tiny_dataset_does_not_exclude_by_uniqueness():
    df = pd.DataFrame({"Label": ["a", "b", "c", "a"], "Age": [20, 25, 21, 40]})
    result = select_features(df)
    assert "Label" in result.selected_columns


def test_max_categories_is_configurable():
    n = 30
    df = pd.DataFrame({
        "Label": [f"c{i % 10}" for i in range(n)],       # 10 distinct
        "Age": 20 + (np.arange(n) * 7) % 45,
    })
    assert "Label" in select_features(df).selected_columns
    assert "Label" not in select_features(df, max_categories=5).selected_columns


# ----------------------------------------------------------------------
# Caller control: columns / exclude_columns / force_include
# ----------------------------------------------------------------------
def test_columns_restricts_the_candidates():
    result = select_features(make_df(), columns=["Age", "Gender"])
    assert result.selected_columns == ["Age", "Gender"]
    assert result.excluded_columns.empty   # non-candidates are not listed


def test_exclude_columns():
    result = select_features(make_df(), exclude_columns=["Age"])
    assert "Age" not in result.selected_columns
    assert "caller" in reasons(result)["Age"]


def test_single_label_instead_of_list_is_accepted():
    result = select_features(make_df(), exclude_columns="Age")
    assert "Age" not in result.selected_columns


def test_force_include_overrides_heuristics():
    result = select_features(make_df(), force_include=["Customer ID"])
    assert "Customer ID" in result.selected_columns
    assert "Customer ID" not in reasons(result)


def test_force_include_is_added_to_candidates():
    result = select_features(make_df(), columns=["Age"], force_include=["Customer ID"])
    assert result.selected_columns == ["Customer ID", "Age"]


def test_force_include_cannot_keep_datetime_or_all_missing_columns():
    n = 30
    df = pd.DataFrame({
        "Created": pd.date_range("2024-01-01", periods=n),
        "Empty": [np.nan] * n,
        "Age": 20 + (np.arange(n) * 7) % 45,
    })
    result = select_features(df, force_include=["Created", "Empty"])
    assert result.selected_columns == ["Age"]


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
def test_not_a_dataframe_raises_type_error():
    with pytest.raises(TypeError):
        select_features([1, 2, 3])


def test_empty_dataframe_raises():
    with pytest.raises(ValueError):
        select_features(pd.DataFrame())
    with pytest.raises(ValueError):
        select_features(pd.DataFrame({"a": []}))


def test_duplicate_column_names_raise():
    df = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"])
    with pytest.raises(ValueError, match="duplicate column names"):
        select_features(df)


@pytest.mark.parametrize("kwargs", [
    {"columns": ["Nope"]},
    {"exclude_columns": ["Nope"]},
    {"force_include": ["Nope"]},
])
def test_unknown_columns_raise(kwargs):
    with pytest.raises(ValueError, match="not found"):
        select_features(make_df(), **kwargs)


def test_column_in_both_force_include_and_exclude_raises():
    with pytest.raises(ValueError, match="both"):
        select_features(make_df(), force_include=["Age"], exclude_columns=["Age"])


def test_empty_columns_list_raises():
    with pytest.raises(ValueError, match="empty"):
        select_features(make_df(), columns=[])


@pytest.mark.parametrize("kwargs", [
    {"max_categories": 0},
    {"max_categories": 2.5},
    {"max_categories": True},
    {"max_unique_ratio": 0},
    {"max_unique_ratio": 1.5},
    {"max_unique_ratio": "half"},
])
def test_invalid_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        select_features(make_df(), **kwargs)
