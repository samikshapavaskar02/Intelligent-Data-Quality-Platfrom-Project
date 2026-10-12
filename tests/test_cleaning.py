"""Tests for modules/cleaning.py (Data Cleaning phase).

Run from the project root:

    pytest tests/test_cleaning.py

The tests use only pandas, numpy and pytest, plus the project's own
``modules.statistical_outliers`` detector. The Isolation Forest test needs
scikit-learn (already a project dependency) and is skipped if it is missing.
"""

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

# Make "modules" importable however pytest is started.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.cleaning import (  # noqa: E402
    CHANGE_COLUMNS,
    INVALID_CLIP,
    INVALID_KEEP,
    INVALID_MISSING,
    MISSING_DROP,
    MISSING_FILL,
    MISSING_KEEP,
    SUMMARY_COLUMNS,
    clean_dataset,
)
from modules.statistical_outliers import detect_iqr_outliers  # noqa: E402

RULES = {"Age": (0, 120)}


# ----------------------------------------------------------------------
# Duplicate rows
# ----------------------------------------------------------------------
def test_remove_duplicates_keeps_first_occurrence():
    df = pd.DataFrame(
        {
            "Name": ["Asha", "Ravi", "Asha", "Meera", "Asha"],
            "Age": [25, 30, 25, 41, 25],
        },
        index=["r1", "r2", "r3", "r4", "r5"],
    )

    result = clean_dataset(df, remove_duplicates=True)

    # The first "Asha" row (r1) survives; r3 and r5 are the copies.
    assert list(result.cleaned.index) == ["r1", "r2", "r4"]
    assert result.rows_before == 5
    assert result.rows_after == 3
    assert result.has_changes

    # Removed rows are reported with their 1-based original position.
    assert result.removed_rows["Original Row"].tolist() == [3, 5]
    assert (
        result.removed_rows["Reason"] == "Exact duplicate of an earlier row"
    ).all()


def test_duplicates_are_kept_when_the_option_is_off():
    df = pd.DataFrame({"A": [1, 1, 2]})

    result = clean_dataset(df, remove_duplicates=False)

    assert len(result.cleaned) == 3
    assert not result.has_changes


# ----------------------------------------------------------------------
# Invalid values
# ----------------------------------------------------------------------
def test_invalid_values_can_be_clipped_to_the_allowed_range():
    df = pd.DataFrame({"Age": [25, 150, -5, 40]})

    result = clean_dataset(
        df, invalid_action=INVALID_CLIP, invalid_rules=RULES
    )

    assert result.cleaned["Age"].tolist() == [25, 120, 0, 40]
    assert result.cells_changed == 2
    assert result.changes["Original Row"].tolist() == [2, 3]
    assert result.changes["Before"].tolist() == ["150", "-5"]
    assert result.changes["After"].tolist() == ["120", "0"]
    assert (result.changes["Column"] == "Age").all()


def test_invalid_values_can_be_set_to_missing():
    df = pd.DataFrame({"Age": [25, 150, -5, 40]})

    result = clean_dataset(
        df, invalid_action=INVALID_MISSING, invalid_rules=RULES
    )

    age = result.cleaned["Age"]

    assert age.isna().tolist() == [False, True, True, False]
    assert age.iloc[0] == 25
    assert age.iloc[3] == 40
    assert result.changes["After"].tolist() == ["(missing)", "(missing)"]


def test_text_that_is_not_a_number_is_set_to_missing_even_when_clipping():
    df = pd.DataFrame({"Age": ["20", "Twenty", "150"]})

    result = clean_dataset(
        df, invalid_action=INVALID_CLIP, invalid_rules=RULES
    )

    age = result.cleaned["Age"]

    assert age.iloc[0] == "20"          # valid text is left alone
    assert pd.isna(age.iloc[1])         # "Twenty" cannot be clipped
    assert age.iloc[2] == 120           # "150" is a number in text form
    assert result.cells_changed == 2


def test_invalid_rules_for_columns_not_in_the_dataset_are_ignored():
    df = pd.DataFrame({"Income": [100, 200]})

    result = clean_dataset(
        df, invalid_action=INVALID_CLIP, invalid_rules=RULES
    )

    assert_frame_equal(result.cleaned, df)
    assert not result.has_changes
    assert any("No invalid-value rules apply" in n for n in result.notes)


# ----------------------------------------------------------------------
# Missing values
# ----------------------------------------------------------------------
def test_missing_numeric_values_are_filled_with_the_median():
    df = pd.DataFrame({"Income": [10.0, 20.0, np.nan, 40.0, np.nan]})

    result = clean_dataset(df, missing_action=MISSING_FILL)

    # Median of the present values [10, 20, 40] is 20.
    assert result.cleaned["Income"].tolist() == [10.0, 20.0, 20.0, 40.0, 20.0]
    assert result.cells_changed == 2
    assert result.changes["Original Row"].tolist() == [3, 5]
    assert result.changes["Operation"].str.contains("median").all()


def test_missing_categorical_values_are_filled_with_the_most_frequent_value():
    df = pd.DataFrame(
        {"City": ["Pune", "Pune", "Mumbai", None, "  "]}
    )

    result = clean_dataset(df, missing_action=MISSING_FILL)

    # Both the None and the whitespace-only cell count as missing.
    assert result.cleaned["City"].tolist() == [
        "Pune", "Pune", "Mumbai", "Pune", "Pune"
    ]
    assert result.changes["Before"].tolist() == ["(missing)", "(blank)"]
    assert result.changes["Operation"].str.contains("most frequent").all()


def test_a_column_with_no_values_cannot_be_filled_and_is_reported():
    df = pd.DataFrame({"A": [1.0, 2.0], "B": [np.nan, np.nan]})

    result = clean_dataset(df, missing_action=MISSING_FILL)

    assert result.cleaned["B"].isna().all()
    assert result.cells_changed == 0
    assert any("'B'" in note for note in result.notes)


def test_rows_containing_missing_values_can_be_removed():
    df = pd.DataFrame(
        {"A": [1.0, np.nan, 3.0, 4.0], "B": ["x", "y", "", "w"]}
    )

    result = clean_dataset(df, missing_action=MISSING_DROP)

    # Row 2 has a NaN, row 3 has a blank string.
    assert result.cleaned["A"].tolist() == [1.0, 4.0]
    assert result.rows_after == 2
    assert result.removed_rows["Original Row"].tolist() == [2, 3]
    assert (
        result.removed_rows["Reason"] == "Contains a missing or blank value"
    ).all()


def test_invalid_values_set_to_missing_can_then_be_filled():
    df = pd.DataFrame({"Age": [30, 150, 50]})

    result = clean_dataset(
        df,
        invalid_action=INVALID_MISSING,
        invalid_rules=RULES,
        missing_action=MISSING_FILL,
    )

    # 150 -> missing -> median of [30, 50] = 40.
    assert result.cleaned["Age"].tolist() == [30.0, 40.0, 50.0]
    assert result.cells_changed == 2


def test_rows_that_become_duplicates_after_filling_are_reported_not_removed():
    df = pd.DataFrame({"A": [1, 1], "B": [5.0, np.nan]})

    result = clean_dataset(
        df, remove_duplicates=True, missing_action=MISSING_FILL
    )

    assert len(result.cleaned) == 2
    assert any("became exact duplicates" in n for n in result.notes)


def test_removing_every_row_is_reported():
    df = pd.DataFrame({"A": [np.nan, np.nan]})

    result = clean_dataset(df, missing_action=MISSING_DROP)

    assert result.cleaned.empty
    assert result.rows_after == 0
    assert any("empty" in note for note in result.notes)


# ----------------------------------------------------------------------
# The input is never modified
# ----------------------------------------------------------------------
def test_original_dataframe_is_preserved():
    df = pd.DataFrame(
        {
            "Name": ["Asha", "Ravi", "Asha", "Meera", "Kiran"],
            "Age": [25, 150, 25, 41, np.nan],
            "Income": [50000.0, np.nan, 50000.0, 62000.0, 58000.0],
            "City": ["Pune", "Mumbai", "Pune", None, "Pune"],
        }
    )
    snapshot = df.copy(deep=True)

    result = clean_dataset(
        df,
        remove_duplicates=True,
        invalid_action=INVALID_CLIP,
        invalid_rules=RULES,
        missing_action=MISSING_FILL,
    )

    assert result.has_changes
    assert_frame_equal(df, snapshot)

    # Editing the cleaned copy afterwards must not leak into the original.
    result.cleaned.loc[result.cleaned.index[0], "Income"] = -1.0
    assert_frame_equal(df, snapshot)


def test_cleaned_result_is_always_a_separate_object():
    df = pd.DataFrame({"A": [1, 2, 3]})

    result = clean_dataset(df)   # no operations selected

    assert result.cleaned is not df
    assert_frame_equal(result.cleaned, df)


# ----------------------------------------------------------------------
# Empty datasets, no issues, no operations
# ----------------------------------------------------------------------
def test_empty_datasets_are_handled():
    for empty in (pd.DataFrame(), pd.DataFrame(columns=["Age", "City"])):

        result = clean_dataset(
            empty,
            remove_duplicates=True,
            invalid_action=INVALID_CLIP,
            invalid_rules=RULES,
            missing_action=MISSING_FILL,
        )

        assert result.cleaned.empty
        assert result.rows_before == 0
        assert result.rows_after == 0
        assert not result.has_changes
        assert any("empty" in note for note in result.notes)


def test_dataset_without_issues_is_returned_unchanged():
    df = pd.DataFrame(
        {
            "Name": ["A", "B", "C"],
            "Age": [20, 30, 40],
            "City": ["X", "Y", "Z"],
        }
    )

    result = clean_dataset(
        df,
        remove_duplicates=True,
        invalid_action=INVALID_CLIP,
        invalid_rules=RULES,
        missing_action=MISSING_FILL,
    )

    assert_frame_equal(result.cleaned, df)
    assert not result.has_changes
    assert result.changes.empty
    assert result.removed_rows.empty
    assert list(result.changes.columns) == CHANGE_COLUMNS
    assert result.summary["Effect"].tolist() == [
        "0 row(s) removed",
        "0 cell(s) corrected",
        "0 cell(s) filled",
    ]


def test_no_operations_selected_changes_nothing():
    df = pd.DataFrame({"A": [1, 1, np.nan], "Age": [10, 500, 20]})

    result = clean_dataset(df)

    assert_frame_equal(result.cleaned, df)
    assert not result.has_changes
    assert result.summary.empty
    assert list(result.summary.columns) == SUMMARY_COLUMNS


def test_invalid_inputs_are_rejected():
    df = pd.DataFrame({"A": [1, 2]})

    with pytest.raises(TypeError):
        clean_dataset([1, 2, 3])

    with pytest.raises(ValueError):
        clean_dataset(df, invalid_action="delete")

    with pytest.raises(ValueError):
        clean_dataset(df, missing_action="delete")

    repeated_names = pd.DataFrame([[1, 2]], columns=["A", "A"])

    with pytest.raises(ValueError):
        clean_dataset(repeated_names)


# ----------------------------------------------------------------------
# Anomalies are review suggestions: cleaning never removes them
# ----------------------------------------------------------------------
def _dataset_with_an_iqr_outlier():
    # Row 8 (Income 5000) is the IQR outlier. Row 9 is an exact copy of
    # row 6. Row 4 has an invalid Age (150).
    return pd.DataFrame(
        {
            "Income": [50, 52, 48, 51, 49, 50, 53, 5000, 50],
            "Age": [27, 31, 29, 150, 28, 30, 32, 33, 30],
        }
    )


def test_cleaning_has_no_anomaly_or_outlier_input():
    parameters = inspect.signature(clean_dataset).parameters

    for name in parameters:
        assert "outlier" not in name.lower()
        assert "anomal" not in name.lower()


def test_iqr_outlier_rows_are_not_removed_by_any_operation():
    df = _dataset_with_an_iqr_outlier()

    # The detector really does flag the 5000 income...
    outliers = detect_iqr_outliers(df)
    flagged_income = outliers[outliers["Column"] == "Income"]["Value"].tolist()
    assert 5000 in flagged_income

    # ...but cleaning with every operation switched on keeps that record.
    result = clean_dataset(
        df,
        remove_duplicates=True,
        invalid_action=INVALID_CLIP,
        invalid_rules=RULES,
        missing_action=MISSING_FILL,
    )

    assert 5000 in result.cleaned["Income"].tolist()
    assert len(result.cleaned) == len(df) - 1          # only the true copy
    assert result.removed_rows["Original Row"].tolist() == [9]
    assert result.cleaned["Age"].max() == 120          # invalid, via the rule


def test_a_valid_value_flagged_as_an_outlier_is_never_changed():
    # Age 100 is an IQR outlier here but is inside the allowed 0-120 range.
    df = pd.DataFrame({"Age": [30, 31, 29, 32, 28, 30, 31, 100]})

    outliers = detect_iqr_outliers(df)
    assert 100 in outliers["Value"].tolist()

    result = clean_dataset(
        df,
        remove_duplicates=False,
        invalid_action=INVALID_CLIP,
        invalid_rules=RULES,
        missing_action=MISSING_FILL,
    )

    assert_frame_equal(result.cleaned, df)
    assert not result.has_changes


def test_isolation_forest_anomalies_are_not_removed_by_cleaning():
    ensemble = pytest.importorskip("sklearn.ensemble")

    rng = np.random.default_rng(0)
    values = rng.normal(0, 1, size=(200, 2))
    values = np.vstack([values, [[60.0, 60.0]]])        # obvious anomaly
    df = pd.DataFrame(values, columns=["x", "y"])

    model = ensemble.IsolationForest(contamination=0.01, random_state=0)
    predictions = model.fit_predict(df)

    assert predictions[-1] == -1                        # flagged as anomaly

    result = clean_dataset(
        df,
        remove_duplicates=True,
        missing_action=MISSING_DROP,
    )

    assert len(result.cleaned) == len(df)
    assert result.cleaned.iloc[-1].tolist() == [60.0, 60.0]
    assert not result.has_changes
