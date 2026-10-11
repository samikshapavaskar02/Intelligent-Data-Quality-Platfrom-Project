"""Tests for modules/unified_results.py.

The first part uses hand-built result frames in the exact formats the
existing detection modules return. The last part (integration) runs the real
detection modules and the real anomaly pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from modules.anomaly_analysis import (
    HIGH_CONFIDENCE,
    NEEDS_REVIEW,
    NORMAL,
    POTENTIAL_ANOMALY,
    combine_anomaly_results,
)
from modules.unified_results import (
    CATEGORY_ANOMALY,
    CATEGORY_OUTLIER,
    CATEGORY_QUALITY,
    SUPPORTED_SOURCES,
    UNIFIED_COLUMNS,
    combine_quality_results,
)


# ----------------------------------------------------------------------
# Helpers that build results in the real modules' formats
# ----------------------------------------------------------------------
def missing_result(counts):
    """detect_missing_values format: Column | Missing Values | Missing Percentage."""
    return pd.DataFrame({
        "Column": list(counts),
        "Missing Values": list(counts.values()),
        "Missing Percentage": [round(v * 100 / 8, 2) for v in counts.values()],
    })


def cell_result(rows, status):
    """invalid_values / data_type format: Row | Column | Value | Status."""
    return pd.DataFrame(
        [{"Row": r, "Column": c, "Value": v, "Status": status} for r, c, v in rows],
        columns=["Row", "Column", "Value", "Status"],
    )


def format_result(rows):
    """detect_format_problems format (has an extra 'Format' column)."""
    return pd.DataFrame([
        {"Row": r, "Column": c, "Value": v, "Format": f, "Status": "Invalid Format"}
        for r, c, v, f in rows
    ])


def iqr_result(rows):
    """detect_iqr_outliers format."""
    return pd.DataFrame([
        {"Row": r, "Column": c, "Value": v, "Lower Bound": lo, "Upper Bound": hi,
         "Method": "IQR", "Status": "Potential Outlier"}
        for r, c, v, lo, hi in rows
    ])


def model_result(prefix, statuses, scores, index=None):
    """if_* / lof_* per-record format."""
    pred = [-1 if s == POTENTIAL_ANOMALY else 1 for s in statuses]
    return pd.DataFrame({
        f"{prefix}_prediction": pred,
        f"{prefix}_status": statuses,
        f"{prefix}_score": scores,
    }, index=index)


def one(frame):
    """Assert there is exactly one finding and return it as a Series."""
    assert len(frame) == 1
    return frame.iloc[0]


# ----------------------------------------------------------------------
# Output format and input validation
# ----------------------------------------------------------------------
def test_empty_dict_returns_standard_empty_table():
    result = combine_quality_results({})
    assert result.empty
    assert list(result.columns) == UNIFIED_COLUMNS


def test_none_and_empty_sources_are_skipped():
    result = combine_quality_results({
        "missing_values": None,
        "format_problems": pd.DataFrame(),   # what the detector returns when clean
        "iqr_outliers": pd.DataFrame(),
        "inconsistent_values": pd.DataFrame(),
        "duplicates": pd.DataFrame(),
    })
    assert result.empty
    assert list(result.columns) == UNIFIED_COLUMNS


def test_non_dict_raises_type_error():
    with pytest.raises(TypeError):
        combine_quality_results([pd.DataFrame()])
    with pytest.raises(TypeError):
        combine_quality_results(None)


def test_non_dataframe_result_raises_type_error():
    with pytest.raises(TypeError):
        combine_quality_results({"missing_values": [1, 2, 3]})
    with pytest.raises(TypeError):
        combine_quality_results({"duplicates": "not a frame"})


def test_unknown_source_name_raises_value_error():
    with pytest.raises(ValueError, match="Unknown source"):
        combine_quality_results({"Custom Check": cell_result([(1, "A", 1)], "x")})


def test_invalid_index_argument_raises_type_error():
    with pytest.raises(TypeError):
        combine_quality_results({}, index=[1, 2, 3])


def test_missing_required_columns_raise_value_error():
    with pytest.raises(ValueError, match="required columns"):
        combine_quality_results({"invalid_values": pd.DataFrame({"Row": [1]})})
    with pytest.raises(ValueError, match="required columns"):
        combine_quality_results({"missing_values": pd.DataFrame({"Column": ["A"]})})
    with pytest.raises(ValueError, match="required columns"):
        combine_quality_results({"isolation_forest": pd.DataFrame({"if_score": [0.1]})})
    with pytest.raises(ValueError, match="required columns"):
        combine_quality_results({"anomaly_analysis": pd.DataFrame({"IF Result": [NORMAL]})})


def test_supported_sources_cover_all_ten_result_types():
    assert set(SUPPORTED_SOURCES) == {
        "missing_values", "duplicates", "invalid_values", "data_type_problems",
        "format_problems", "inconsistent_values", "iqr_outliers",
        "isolation_forest", "lof", "anomaly_analysis",
    }


# ----------------------------------------------------------------------
# Missing values (column-level)
# ----------------------------------------------------------------------
def test_missing_values_are_column_level_findings():
    result = combine_quality_results({"missing_values": missing_result({"Age": 2, "City": 1})})
    assert len(result) == 2
    row = result.iloc[0]
    assert row["Category"] == CATEGORY_QUALITY
    assert row["Issue Type"] == "Missing Value"
    assert row["Scope"] == "Column"
    assert row["Column"] == "Age"
    assert row["Record ID"] is None
    assert pd.isna(row["Row Position"])
    assert "2 missing" in row["Details"] and "25.0%" in row["Details"]


def test_columns_with_zero_missing_values_are_not_findings():
    result = combine_quality_results({
        "missing_values": missing_result({"Age": 0, "Email": 0, "City": 3}),
    })
    assert result["Column"].tolist() == ["City"]


def test_all_columns_complete_gives_no_findings():
    result = combine_quality_results({"missing_values": missing_result({"A": 0, "B": 0})})
    assert result.empty


def test_non_numeric_missing_count_raises():
    bad = pd.DataFrame({"Column": ["A"], "Missing Values": ["many"]})
    with pytest.raises(ValueError, match="non-numeric"):
        combine_quality_results({"missing_values": bad})


# ----------------------------------------------------------------------
# Duplicates (record-level, index labels)
# ----------------------------------------------------------------------
def duplicate_result():
    """detect_duplicates format: the duplicated data rows + Status, original index."""
    return pd.DataFrame(
        {"Name": ["A", "A"], "Age": [30, 30], "Status": ["Duplicate", "Duplicate"]},
        index=[102, 105],
    )


def test_duplicates_keep_original_index_labels():
    result = combine_quality_results({"duplicates": duplicate_result()})
    assert result["Record ID"].tolist() == [102, 105]
    assert result["Issue Type"].unique().tolist() == ["Duplicate Record"]
    assert result["Scope"].unique().tolist() == ["Record"]
    assert result["Status"].unique().tolist() == ["Duplicate"]
    assert result["Row Position"].isna().all()          # not derivable without index


def test_duplicates_row_position_from_index():
    dataset_index = pd.Index([100, 101, 102, 103, 104, 105])
    result = combine_quality_results({"duplicates": duplicate_result()}, index=dataset_index)
    assert result["Record ID"].tolist() == [102, 105]
    assert result["Row Position"].tolist() == [3, 6]


def test_duplicates_with_duplicate_dataset_index_have_no_position():
    dataset_index = pd.Index([1, 1, 2, 3, 4, 5])
    frame = pd.DataFrame({"A": [1, 1], "Status": ["Duplicate"] * 2}, index=[1, 1])
    result = combine_quality_results({"duplicates": frame}, index=dataset_index)
    assert result["Record ID"].tolist() == [1, 1]
    assert result["Row Position"].isna().all()           # ambiguous, so not guessed


def test_duplicate_data_columns_named_like_unified_columns_are_not_misread():
    frame = pd.DataFrame(
        {"Row": [7, 7], "Column": ["x", "x"], "Value": [1, 1], "Status": ["Duplicate"] * 2},
        index=[10, 11],
    )
    result = combine_quality_results({"duplicates": frame})
    assert result["Record ID"].tolist() == [10, 11]
    assert result["Column"].isna().all()


# ----------------------------------------------------------------------
# Cell-level sources and the Row-meaning problem
# ----------------------------------------------------------------------
def test_invalid_values_row_is_a_position_not_a_label():
    frame = cell_result([(3, "Age", 150)], "Invalid Value")

    without_index = one(combine_quality_results({"invalid_values": frame}))
    assert without_index["Row Position"] == 3
    assert without_index["Record ID"] is None            # label unknown, not guessed

    dataset_index = pd.Index([100, 101, 102, 103])
    with_index = one(combine_quality_results({"invalid_values": frame}, index=dataset_index))
    assert with_index["Row Position"] == 3
    assert with_index["Record ID"] == 102                # position 3 -> label 102


def test_data_type_problem_row_is_a_position_not_a_label():
    frame = cell_result([(2, "Age", "Twenty")], "Data Type Problem")
    dataset_index = pd.Index(["a", "b", "c"])
    finding = one(combine_quality_results({"data_type_problems": frame}, index=dataset_index))
    assert finding["Record ID"] == "b"
    assert finding["Row Position"] == 2
    assert finding["Issue Type"] == "Data Type Problem"
    assert finding["Value"] == "Twenty"


def test_format_problem_row_is_label_plus_one():
    frame = format_result([(102, "Email", "bad@", "Email")])

    without_index = one(combine_quality_results({"format_problems": frame}))
    assert without_index["Record ID"] == 101              # 102 - 1 = the index label
    assert pd.isna(without_index["Row Position"])        # position unknown

    dataset_index = pd.Index([100, 101, 102, 103])
    with_index = one(combine_quality_results({"format_problems": frame}, index=dataset_index))
    assert with_index["Record ID"] == 101
    assert with_index["Row Position"] == 2               # NOT 102
    assert with_index["Details"] == "Expected format: Email"


def test_iqr_outlier_row_is_label_plus_one():
    frame = iqr_result([(108, "Income", 12500, 9.5, 13.5)])
    dataset_index = pd.Index(range(100, 108))
    finding = one(combine_quality_results({"iqr_outliers": frame}, index=dataset_index))
    assert finding["Record ID"] == 107
    assert finding["Row Position"] == 8
    assert finding["Category"] == CATEGORY_OUTLIER
    assert finding["Issue Type"] == "IQR Outlier"
    assert finding["Method"] == "IQR"
    assert finding["Status"] == "Potential Outlier"
    assert "9.5" in finding["Details"] and "13.5" in finding["Details"]


def test_default_range_index_gives_matching_label_and_position():
    dataset_index = pd.RangeIndex(5)
    invalid = one(combine_quality_results(
        {"invalid_values": cell_result([(3, "Age", 150)], "Invalid Value")}, index=dataset_index))
    fmt = one(combine_quality_results(
        {"format_problems": format_result([(3, "Email", "x", "Email")])}, index=dataset_index))
    assert (invalid["Record ID"], invalid["Row Position"]) == (2, 3)
    assert (fmt["Record ID"], fmt["Row Position"]) == (2, 3)


def test_label_plus_one_with_duplicate_index_has_no_position():
    dataset_index = pd.Index([0, 1, 1, 2])
    frame = format_result([(2, "Email", "x", "Email")])    # label 1 appears twice
    finding = one(combine_quality_results({"format_problems": frame}, index=dataset_index))
    assert finding["Record ID"] == 1
    assert pd.isna(finding["Row Position"])


def test_position_with_duplicate_index_keeps_exact_position():
    dataset_index = pd.Index([0, 1, 1, 2])
    frame = cell_result([(3, "Age", 150)], "Invalid Value")
    finding = one(combine_quality_results({"invalid_values": frame}, index=dataset_index))
    assert finding["Record ID"] == 1
    assert finding["Row Position"] == 3                   # disambiguates the duplicated label


def test_row_outside_supplied_index_raises():
    dataset_index = pd.Index([10, 20])
    with pytest.raises(ValueError, match="row position"):
        combine_quality_results(
            {"invalid_values": cell_result([(5, "A", 1)], "Invalid Value")}, index=dataset_index)
    with pytest.raises(ValueError, match="not in the supplied index"):
        combine_quality_results(
            {"iqr_outliers": iqr_result([(99, "A", 1, 0, 1)])}, index=dataset_index)


def test_invalid_row_values_raise():
    with pytest.raises(ValueError, match="non-integer"):
        combine_quality_results({"invalid_values": cell_result([("x", "A", 1)], "Invalid Value")})
    with pytest.raises(ValueError, match="non-integer"):
        combine_quality_results({"invalid_values": cell_result([(1.5, "A", 1)], "Invalid Value")})
    with pytest.raises(ValueError, match="invalid Row position"):
        combine_quality_results({"invalid_values": cell_result([(0, "A", 1)], "Invalid Value")})


def test_missing_row_value_gives_unknown_identity():
    frame = pd.DataFrame({"Row": [None], "Column": ["A"], "Value": [1], "Status": ["Invalid Value"]})
    finding = one(combine_quality_results({"invalid_values": frame}))
    assert finding["Record ID"] is None
    assert pd.isna(finding["Row Position"])


def test_optional_status_column_defaults_to_issue_type():
    frame = pd.DataFrame({"Row": [1], "Column": ["A"], "Value": [5]})
    finding = one(combine_quality_results({"invalid_values": frame}))
    assert finding["Status"] == "Invalid Value"


def test_same_cell_flagged_by_two_checks_is_kept_twice():
    invalid = cell_result([(3, "Age", "Twenty")], "Invalid Value")
    dtype = cell_result([(3, "Age", "Twenty")], "Data Type Problem")
    result = combine_quality_results({"invalid_values": invalid, "data_type_problems": dtype})
    assert result["Issue Type"].tolist() == ["Invalid Value", "Data Type Problem"]


# ----------------------------------------------------------------------
# Consistency (column-level)
# ----------------------------------------------------------------------
def test_inconsistent_values_are_column_level_findings():
    frame = pd.DataFrame([{
        "Column": "City", "Normalized Value": "pune",
        "Different Representations": "PUNE, Pune, pune",
        "Status": "Inconsistent Representation",
    }])
    finding = one(combine_quality_results({"inconsistent_values": frame}))
    assert finding["Category"] == CATEGORY_QUALITY
    assert finding["Issue Type"] == "Inconsistent Representation"
    assert finding["Scope"] == "Column"
    assert finding["Column"] == "City"
    assert finding["Record ID"] is None
    assert finding["Details"] == "'pune' appears as: PUNE, Pune, pune"


def test_inconsistent_values_without_optional_columns():
    frame = pd.DataFrame({"Column": ["City"], "Different Representations": ["A, a"]})
    finding = one(combine_quality_results({"inconsistent_values": frame}))
    assert finding["Details"] == "Appears as: A, a"
    assert finding["Status"] == "Inconsistent Representation"


# ----------------------------------------------------------------------
# Isolation Forest / LOF (per-record, all records)
# ----------------------------------------------------------------------
def test_isolation_forest_only_flagged_records_become_findings():
    frame = model_result("if", [NORMAL, POTENTIAL_ANOMALY, NORMAL], [0.4, 0.71234, 0.5],
                         index=[10, 20, 30])
    finding = one(combine_quality_results({"isolation_forest": frame}))
    assert finding["Category"] == CATEGORY_ANOMALY
    assert finding["Issue Type"] == "Isolation Forest Anomaly"
    assert finding["Method"] == "Isolation Forest"
    assert finding["Record ID"] == 20
    assert finding["Status"] == POTENTIAL_ANOMALY
    assert "0.7123" in finding["Details"]
    assert finding["Value"] is None and finding["Column"] is None


def test_lof_only_flagged_records_become_findings():
    frame = model_result("lof", [POTENTIAL_ANOMALY, NORMAL], [2.5, 1.0], index=["a", "b"])
    finding = one(combine_quality_results({"lof": frame}))
    assert finding["Issue Type"] == "LOF Anomaly"
    assert finding["Record ID"] == "a"
    assert "LOF score: 2.5000" in finding["Details"]


def test_model_results_without_score_column():
    frame = pd.DataFrame({"if_status": [POTENTIAL_ANOMALY]}, index=[5])
    finding = one(combine_quality_results({"isolation_forest": frame}))
    assert finding["Details"] is None


def test_model_row_position_from_dataset_index():
    dataset_index = pd.Index([10, 20, 30])
    frame = model_result("lof", [NORMAL, NORMAL, POTENTIAL_ANOMALY], [1, 1, 3], index=dataset_index)
    finding = one(combine_quality_results({"lof": frame}, index=dataset_index))
    assert finding["Record ID"] == 30
    assert finding["Row Position"] == 3


def test_model_results_with_duplicate_index_labels_use_exact_positions():
    dataset_index = pd.Index([7, 7, 8, 8])
    frame = model_result("if", [NORMAL, POTENTIAL_ANOMALY, NORMAL, POTENTIAL_ANOMALY],
                         [0.1, 0.9, 0.2, 0.8], index=dataset_index)
    result = combine_quality_results({"isolation_forest": frame}, index=dataset_index)
    assert result["Record ID"].tolist() == [7, 8]
    assert result["Row Position"].tolist() == [2, 4]      # exact even though labels repeat


def test_model_results_with_duplicate_index_without_dataset_index():
    frame = model_result("if", [POTENTIAL_ANOMALY, POTENTIAL_ANOMALY], [0.9, 0.8], index=[7, 7])
    result = combine_quality_results({"isolation_forest": frame})
    assert result["Record ID"].tolist() == [7, 7]
    assert result["Row Position"].isna().all()


def test_model_result_objects_with_results_attribute_are_accepted():
    class FakeResult:
        results = model_result("if", [POTENTIAL_ANOMALY], [0.9], index=[1])

    finding = one(combine_quality_results({"isolation_forest": FakeResult()}))
    assert finding["Record ID"] == 1


def test_result_object_is_rejected_for_other_sources():
    class FakeResult:
        results = pd.DataFrame({"Row": [1]})

    with pytest.raises(TypeError):
        combine_quality_results({"invalid_values": FakeResult()})


def test_anomaly_alias_and_invalid_model_labels():
    ok = pd.DataFrame({"lof_status": ["Anomaly"]}, index=[1])
    assert one(combine_quality_results({"lof": ok}))["Status"] == POTENTIAL_ANOMALY
    for bad in ("Weird", None):
        frame = pd.DataFrame({"if_status": [bad]}, index=[1])
        with pytest.raises(ValueError, match="invalid label"):
            combine_quality_results({"isolation_forest": frame})


def test_all_normal_model_results_give_no_findings():
    frame = model_result("if", [NORMAL, NORMAL], [0.1, 0.2], index=[1, 2])
    assert combine_quality_results({"isolation_forest": frame}).empty


# ----------------------------------------------------------------------
# Combined anomaly status
# ----------------------------------------------------------------------
def combined_result(index):
    return pd.DataFrame({
        "IF Result": [NORMAL, POTENTIAL_ANOMALY, POTENTIAL_ANOMALY, NORMAL],
        "LOF Result": [NORMAL, POTENTIAL_ANOMALY, NORMAL, POTENTIAL_ANOMALY],
        "Status": [NORMAL, HIGH_CONFIDENCE, NEEDS_REVIEW, NEEDS_REVIEW],
    }, index=index)


def test_combined_normal_records_are_not_findings():
    result = combine_quality_results({"anomaly_analysis": combined_result([101, 102, 103, 104])})
    assert result["Record ID"].tolist() == [102, 103, 104]
    assert NORMAL not in result["Status"].tolist()


def test_combined_statuses_and_details():
    result = combine_quality_results({"anomaly_analysis": combined_result([101, 102, 103, 104])})
    assert result["Status"].tolist() == [HIGH_CONFIDENCE, NEEDS_REVIEW, NEEDS_REVIEW]
    assert result["Details"].tolist() == [
        "IF: Potential Anomaly; LOF: Potential Anomaly",
        "IF: Potential Anomaly; LOF: Normal",
        "IF: Normal; LOF: Potential Anomaly",
    ]
    assert result["Issue Type"].unique().tolist() == ["Combined Anomaly"]
    assert result["Method"].unique().tolist() == ["Isolation Forest + LOF"]


def test_combined_dashboard_table_uses_record_id_column():
    table = combined_result([0, 1, 2, 3])
    table.insert(0, "Record ID", ["cust-A", "cust-B", "cust-C", "cust-D"])
    result = combine_quality_results({"anomaly_analysis": table})
    assert result["Record ID"].tolist() == ["cust-B", "cust-C", "cust-D"]


def test_combined_row_position_from_dataset_index():
    dataset_index = pd.Index([101, 102, 103, 104])
    result = combine_quality_results(
        {"anomaly_analysis": combined_result(dataset_index)}, index=dataset_index)
    assert result["Row Position"].tolist() == [2, 3, 4]


def test_combined_without_optional_result_columns():
    frame = pd.DataFrame({"Status": [NEEDS_REVIEW]}, index=[9])
    finding = one(combine_quality_results({"anomaly_analysis": frame}))
    assert finding["Details"] is None


def test_combined_unknown_status_raises():
    frame = pd.DataFrame({"Status": ["Suspicious"]}, index=[1])
    with pytest.raises(ValueError, match="invalid status"):
        combine_quality_results({"anomaly_analysis": frame})


def test_anomalies_are_never_classified_as_data_quality_errors():
    result = combine_quality_results({
        "isolation_forest": model_result("if", [POTENTIAL_ANOMALY], [0.9], index=[1]),
        "lof": model_result("lof", [POTENTIAL_ANOMALY], [2.0], index=[1]),
        "anomaly_analysis": combined_result([101, 102, 103, 104]),
        "iqr_outliers": iqr_result([(2, "Income", 9999, 0.0, 10.0)]),
    })
    assert CATEGORY_QUALITY not in result["Category"].tolist()
    assert "Invalid Value" not in result["Issue Type"].tolist()
    anomaly_rows = result[result["Category"] == CATEGORY_ANOMALY]
    assert len(anomaly_rows) == 5
    assert (result[result["Method"] == "IQR"]["Category"] == CATEGORY_OUTLIER).all()


# ----------------------------------------------------------------------
# Whole-table behaviour
# ----------------------------------------------------------------------
def full_results():
    return {
        # Deliberately NOT in canonical order.
        "anomaly_analysis": combined_result([101, 102, 103, 104]),
        "lof": model_result("lof", [NORMAL, POTENTIAL_ANOMALY, NORMAL, NORMAL], [1, 2, 1, 1],
                            index=[101, 102, 103, 104]),
        "isolation_forest": model_result("if", [NORMAL, POTENTIAL_ANOMALY, NORMAL, NORMAL],
                                         [0.1, 0.9, 0.1, 0.1], index=[101, 102, 103, 104]),
        "iqr_outliers": iqr_result([(104, "Income", 500, 1.0, 20.0)]),
        "inconsistent_values": pd.DataFrame({
            "Column": ["City"], "Normalized Value": ["pune"],
            "Different Representations": ["PUNE, pune"], "Status": ["Inconsistent Representation"]}),
        "format_problems": format_result([(102, "Email", "bad@", "Email")]),
        "data_type_problems": cell_result([(3, "Age", "Twenty")], "Data Type Problem"),
        "invalid_values": cell_result([(3, "Age", "Twenty")], "Invalid Value"),
        "duplicates": pd.DataFrame({"A": [1, 1], "Status": ["Duplicate"] * 2}, index=[101, 103]),
        "missing_values": missing_result({"Age": 1}),
    }


def test_all_ten_sources_combine_in_canonical_order():
    dataset_index = pd.Index([101, 102, 103, 104])
    result = combine_quality_results(full_results(), index=dataset_index)

    assert list(result.columns) == UNIFIED_COLUMNS
    expected_order = [
        "Missing Value", "Duplicate Record", "Duplicate Record", "Invalid Value",
        "Data Type Problem", "Format Problem", "Inconsistent Representation", "IQR Outlier",
        "Isolation Forest Anomaly", "LOF Anomaly",
        "Combined Anomaly", "Combined Anomaly", "Combined Anomaly",
    ]
    assert result["Issue Type"].tolist() == expected_order


def test_result_has_fresh_index_and_stable_dtypes():
    dataset_index = pd.Index([101, 102, 103, 104])
    result = combine_quality_results(full_results(), index=dataset_index)
    assert result.index.tolist() == list(range(len(result)))
    assert str(result["Row Position"].dtype) == "Int64"
    # Record IDs stay integers (no 102.0 caused by missing values in the column).
    assert 102 in result["Record ID"].tolist()
    assert not any(isinstance(v, float) for v in result["Record ID"].tolist())


def test_dict_order_does_not_change_the_result():
    a = combine_quality_results(full_results())
    reversed_results = dict(reversed(list(full_results().items())))
    b = combine_quality_results(reversed_results)
    pd.testing.assert_frame_equal(a, b)


def test_every_finding_has_category_issue_method_and_scope():
    result = combine_quality_results(full_results())
    for column in ("Category", "Issue Type", "Method", "Scope", "Status"):
        assert result[column].notna().all()


def test_inputs_and_index_are_not_modified():
    results = full_results()
    before = {k: v.copy(deep=True) for k, v in results.items()}
    dataset_index = pd.Index([101, 102, 103, 104])
    index_before = dataset_index.copy()

    combine_quality_results(results, index=dataset_index)

    for key, frame in results.items():
        pd.testing.assert_frame_equal(frame, before[key])
    assert dataset_index.equals(index_before)


# ----------------------------------------------------------------------
# Regression: a duplicated label must not hide the position of a unique one
# ----------------------------------------------------------------------
DUP_INDEX = pd.Index([1, 1, 2, 3])      # label 1 is ambiguous; 2 and 3 are unique


def test_position_of_label_unit():
    from modules.unified_results import _position_of_label

    assert _position_of_label(2, DUP_INDEX) == 3          # unique label, dup elsewhere
    assert _position_of_label(3, DUP_INDEX) == 4
    assert _position_of_label(1, DUP_INDEX) is None       # the label itself is ambiguous
    assert _position_of_label(5, pd.Index([5, 6])) == 1   # fully unique index
    assert _position_of_label(6, pd.Index([6, 5, 6])) is None
    assert _position_of_label(5, pd.Index([6, 5, 6])) == 2
    with pytest.raises(ValueError, match="not in the supplied index"):
        _position_of_label(9, DUP_INDEX)
    assert _position_of_label(9, DUP_INDEX, on_missing="none") is None


def test_label_plus_one_unique_label_keeps_position_when_index_has_duplicates():
    # Row 3 means label 2, which occurs once, at position 3.
    unique = one(combine_quality_results(
        {"format_problems": format_result([(3, "Email", "x", "Email")])}, index=DUP_INDEX))
    assert (unique["Record ID"], unique["Row Position"]) == (2, 3)

    # Row 4 means label 3, which occurs once, at position 4.
    last = one(combine_quality_results(
        {"iqr_outliers": iqr_result([(4, "Income", 9, 0.0, 1.0)])}, index=DUP_INDEX))
    assert (last["Record ID"], last["Row Position"]) == (3, 4)


def test_label_plus_one_duplicated_label_has_no_position_in_same_index():
    # Row 2 means label 1, which occurs twice.
    ambiguous = one(combine_quality_results(
        {"format_problems": format_result([(2, "Email", "x", "Email")])}, index=DUP_INDEX))
    assert ambiguous["Record ID"] == 1
    assert pd.isna(ambiguous["Row Position"])


def test_mixed_unique_and_duplicated_labels_in_one_result():
    frame = iqr_result([(2, "A", 9, 0.0, 1.0), (3, "A", 9, 0.0, 1.0), (4, "A", 9, 0.0, 1.0)])
    result = combine_quality_results({"iqr_outliers": frame}, index=DUP_INDEX)
    assert result["Record ID"].tolist() == [1, 2, 3]
    assert result["Row Position"].isna().tolist() == [True, False, False]
    assert result["Row Position"].dropna().tolist() == [3, 4]


def test_position_based_sources_are_exact_with_duplicated_labels():
    frame = cell_result([(1, "A", 9), (2, "A", 9), (3, "A", 9)], "Invalid Value")
    result = combine_quality_results({"invalid_values": frame}, index=DUP_INDEX)
    assert result["Record ID"].tolist() == [1, 1, 2]
    assert result["Row Position"].tolist() == [1, 2, 3]   # positions tell the two 1s apart


def test_duplicates_frame_labels_with_duplicated_dataset_labels():
    frame = pd.DataFrame({"A": [5, 5, 7], "Status": ["Duplicate"] * 3}, index=[1, 2, 3])
    result = combine_quality_results({"duplicates": frame}, index=DUP_INDEX)
    assert result["Record ID"].tolist() == [1, 2, 3]
    assert result["Row Position"].isna().tolist() == [True, False, False]
    assert result["Row Position"].dropna().tolist() == [3, 4]


def test_record_level_subset_frame_with_duplicated_dataset_labels():
    frame = model_result("if", [POTENTIAL_ANOMALY, POTENTIAL_ANOMALY, POTENTIAL_ANOMALY],
                         [0.9, 0.8, 0.7], index=[1, 2, 3])        # a subset, not row-aligned
    result = combine_quality_results({"isolation_forest": frame}, index=DUP_INDEX)
    assert result["Record ID"].tolist() == [1, 2, 3]
    assert result["Row Position"].dropna().tolist() == [3, 4]
    assert pd.isna(result["Row Position"].iloc[0])


def test_row_aligned_frame_with_duplicated_labels_gets_exact_positions():
    frame = model_result("lof", [POTENTIAL_ANOMALY, POTENTIAL_ANOMALY, NORMAL, POTENTIAL_ANOMALY],
                         [3, 3, 1, 3], index=DUP_INDEX)
    result = combine_quality_results({"lof": frame}, index=DUP_INDEX)
    assert result["Record ID"].tolist() == [1, 1, 3]
    assert result["Row Position"].tolist() == [1, 2, 4]


def test_label_missing_from_dup_index_still_raises():
    with pytest.raises(ValueError, match="not in the supplied index"):
        combine_quality_results(
            {"format_problems": format_result([(10, "Email", "x", "Email")])}, index=DUP_INDEX)


# ----------------------------------------------------------------------
# Record ID column vs index labels (dashboard table / combined results)
# ----------------------------------------------------------------------
def dashboard_table(index, record_ids, statuses=None):
    """Shape of build_anomaly_table output: Record ID | IF | LOF | Status."""
    if statuses is None:                                  # the 4-record example
        statuses = [NORMAL, HIGH_CONFIDENCE, NEEDS_REVIEW, NEEDS_REVIEW]
    flag = {NORMAL: NORMAL, HIGH_CONFIDENCE: POTENTIAL_ANOMALY, NEEDS_REVIEW: POTENTIAL_ANOMALY}
    return pd.DataFrame({
        "Record ID": record_ids,
        "IF Result": [flag[s] for s in statuses],
        "LOF Result": [POTENTIAL_ANOMALY if s == HIGH_CONFIDENCE else NORMAL for s in statuses],
        "Status": statuses,
    }, index=index)


def test_dashboard_table_that_copies_its_index_uses_labels_for_positions():
    dataset_index = pd.Index([101, 102, 103, 104])
    table = dashboard_table(dataset_index, list(dataset_index))      # real pipeline shape
    result = combine_quality_results({"anomaly_analysis": table}, index=dataset_index)
    assert result["Record ID"].tolist() == [102, 103, 104]
    assert result["Row Position"].tolist() == [2, 3, 4]


def test_dashboard_table_subset_copying_its_index_looks_labels_up():
    dataset_index = pd.Index([101, 102, 103, 104, 105])
    subset = dashboard_table([102, 104], [102, 104], [HIGH_CONFIDENCE, NEEDS_REVIEW])
    result = combine_quality_results({"anomaly_analysis": subset}, index=dataset_index)
    assert result["Record ID"].tolist() == [102, 104]
    assert result["Row Position"].tolist() == [2, 4]


def test_dashboard_table_index_copy_not_in_dataset_index_gives_no_position_and_no_error():
    table = dashboard_table([201, 202, 203, 204], [201, 202, 203, 204])
    result = combine_quality_results(
        {"anomaly_analysis": table}, index=pd.Index([101, 102, 103, 104]))
    assert result["Record ID"].tolist() == [202, 203, 204]
    assert result["Row Position"].isna().all()


def test_dashboard_table_index_copy_with_duplicated_dataset_labels():
    dataset_index = pd.Index([1, 1, 2, 3])
    table = dashboard_table([2, 3], [2, 3], [NEEDS_REVIEW, HIGH_CONFIDENCE])
    ambiguous = dashboard_table([1], [1], [NEEDS_REVIEW])

    result = combine_quality_results({"anomaly_analysis": table}, index=dataset_index)
    assert result["Row Position"].tolist() == [3, 4]
    result = combine_quality_results({"anomaly_analysis": ambiguous}, index=dataset_index)
    assert result["Record ID"].tolist() == [1]
    assert result["Row Position"].isna().all()


def test_separate_record_ids_row_aligned_with_dataset_get_exact_positions():
    # Customer numbers as Record ID; the table's own index equals the dataset index.
    dataset_index = pd.Index([0, 1, 2, 3])
    table = dashboard_table(dataset_index, ["cust-A", "cust-B", "cust-C", "cust-D"])
    result = combine_quality_results({"anomaly_analysis": table}, index=dataset_index)
    assert result["Record ID"].tolist() == ["cust-B", "cust-C", "cust-D"]
    assert result["Row Position"].tolist() == [2, 3, 4]


def test_separate_record_ids_not_matching_any_label_do_not_raise():
    table = dashboard_table([0, 1, 2, 3], ["cust-A", "cust-B", "cust-C", "cust-D"])
    result = combine_quality_results(
        {"anomaly_analysis": table}, index=pd.Index([100, 101, 102, 103]))
    assert result["Record ID"].tolist() == ["cust-B", "cust-C", "cust-D"]
    assert result["Row Position"].isna().all()                        # no invented position


def test_separate_record_ids_that_equal_other_labels_are_not_used_as_labels():
    # IDs 30/20/10/40 happen to be labels of the dataset, but they are separate
    # identifiers (they differ from the table's own index), so no position is invented.
    dataset_index = pd.Index([10, 20, 30, 40])
    table = dashboard_table([0, 1, 2, 3], [30, 20, 10, 40])
    result = combine_quality_results({"anomaly_analysis": table}, index=dataset_index)
    assert result["Record ID"].tolist() == [20, 10, 40]
    assert result["Row Position"].isna().all()


def test_record_id_column_without_dataset_index_never_gets_a_position():
    table = dashboard_table([0, 1, 2, 3], ["cust-A", "cust-B", "cust-C", "cust-D"])
    result = combine_quality_results({"anomaly_analysis": table})
    assert result["Record ID"].tolist() == ["cust-B", "cust-C", "cust-D"]
    assert result["Row Position"].isna().all()


def test_frame_index_labels_not_in_dataset_index_still_raise_without_record_id_column():
    frame = combined_result([201, 202, 203, 204])                     # no Record ID column
    with pytest.raises(ValueError, match="not in the supplied index"):
        combine_quality_results({"anomaly_analysis": frame}, index=pd.Index([101, 102, 103, 104]))


# ----------------------------------------------------------------------
# Integration with the REAL detection modules
# ----------------------------------------------------------------------
def build_real_dataset():
    df = pd.DataFrame({
        "Age": [25, 30, "Twenty", 150, 28, 27, 26, None],
        "Email": ["a@x.com", "bad@", "c@y.com", "c@y.com", "d@z.com", "e@z.com", "f@z.com", "g@z.com"],
        "City": ["Pune", "pune", "PUNE", "Mumbai", "Mumbai", "Pune", "Pune", "Pune"],
        "Income": [10, 12, 11, 13, 12, 11, 12, 12500],
    }, index=[100, 101, 102, 103, 104, 105, 106, 107])
    df.loc[103] = df.loc[102]                              # makes 102 and 103 exact duplicates
    return df


def run_real_checks(df):
    from modules.consistency import detect_inconsistent_values
    from modules.data_type_validation import detect_data_type_problems
    from modules.duplicates import detect_duplicates
    from modules.format_validation import detect_format_problems
    from modules.invalid_values import detect_invalid_values
    from modules.missing_values import detect_missing_values
    from modules.statistical_outliers import detect_iqr_outliers

    return {
        "missing_values": detect_missing_values(df),
        "duplicates": detect_duplicates(df),
        "invalid_values": detect_invalid_values(df, {"Age": (0, 120)}),
        "data_type_problems": detect_data_type_problems(df),
        "format_problems": detect_format_problems(df),
        "inconsistent_values": detect_inconsistent_values(df),
        "iqr_outliers": detect_iqr_outliers(df),
    }


def test_integration_real_quality_checks_on_non_default_index():
    df = build_real_dataset()
    before = df.copy(deep=True)
    raw = run_real_checks(df)

    result = combine_quality_results(raw, index=df.index)

    assert list(result.columns) == UNIFIED_COLUMNS
    pd.testing.assert_frame_equal(df, before)

    # Only the Age column has a missing value; clean columns are not findings.
    missing = result[result["Issue Type"] == "Missing Value"]
    assert missing["Column"].tolist() == ["Age"]

    # Duplicates: the two identical rows, reported by index label.
    dups = result[result["Issue Type"] == "Duplicate Record"]
    assert dups["Record ID"].tolist() == [102, 103]
    assert dups["Row Position"].tolist() == [3, 4]

    # Cell-level position source: "Twenty" sits at labels 102 and 103.
    invalid = result[result["Issue Type"] == "Invalid Value"]
    assert invalid["Record ID"].tolist() == [102, 103]
    assert invalid["Row Position"].tolist() == [3, 4]

    # Label+1 sources: "bad@" has label 101 (position 2), the outlier label 107 (position 8).
    fmt = one(result[result["Issue Type"] == "Format Problem"])
    assert (fmt["Record ID"], fmt["Row Position"], fmt["Value"]) == (101, 2, "bad@")
    iqr = one(result[result["Issue Type"] == "IQR Outlier"])
    assert (iqr["Record ID"], iqr["Row Position"], iqr["Value"]) == (107, 8, 12500)

    # Consistency is a column-level finding for City.
    inconsistent = result[result["Issue Type"] == "Inconsistent Representation"]
    assert inconsistent["Column"].tolist() == ["City"]

    # Same label and position regardless of which Row convention the source used.
    for _, finding in result[result["Scope"] != "Column"].iterrows():
        assert df.index[finding["Row Position"] - 1] == finding["Record ID"]


def test_integration_real_quality_checks_without_index_never_mix_labels_and_positions():
    df = build_real_dataset()
    result = combine_quality_results(run_real_checks(df))

    invalid = result[result["Issue Type"] == "Invalid Value"]
    assert invalid["Row Position"].tolist() == [3, 4]
    assert invalid["Record ID"].tolist() == [None, None]

    fmt = one(result[result["Issue Type"] == "Format Problem"])
    assert fmt["Record ID"] == 101
    assert pd.isna(fmt["Row Position"])


def test_integration_real_quality_checks_on_default_index_dataset():
    df = build_real_dataset().reset_index(drop=True)
    result = combine_quality_results(run_real_checks(df), index=df.index)
    for _, finding in result[result["Scope"] != "Column"].iterrows():
        assert finding["Record ID"] == finding["Row Position"] - 1


def test_integration_real_anomaly_pipeline_results():
    from modules.anomaly_pipeline import run_anomaly_pipeline

    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "Age": rng.normal(40, 3, 60),
        "Income": rng.normal(50000, 2000, 60),
    }, index=pd.Index(range(1000, 1060), name="customer"))
    df.iloc[5] = [95.0, 400000.0]                          # extreme record, label 1005
    before = df.copy(deep=True)

    pipeline = run_anomaly_pipeline(df)
    result = combine_quality_results({
        "isolation_forest": pipeline.isolation_forest,     # result objects
        "lof": pipeline.lof,
        "anomaly_analysis": pipeline.analysis,
    }, index=df.index)
    from_table = combine_quality_results({"anomaly_analysis": pipeline.table}, index=df.index)

    pd.testing.assert_frame_equal(df, before)
    assert (result["Category"] == CATEGORY_ANOMALY).all()

    combined = result[result["Issue Type"] == "Combined Anomaly"]
    assert len(combined) == len(pipeline.flagged)
    assert combined["Record ID"].tolist() == pipeline.flagged.index.tolist()
    pd.testing.assert_frame_equal(combined.reset_index(drop=True), from_table)

    if_rows = result[result["Issue Type"] == "Isolation Forest Anomaly"]
    lof_rows = result[result["Issue Type"] == "LOF Anomaly"]
    assert len(if_rows) == pipeline.summary["if_anomalies"]
    assert len(lof_rows) == pipeline.summary["lof_anomalies"]
    assert 1005 in combined["Record ID"].tolist()
    assert set(combined["Status"]) <= {NEEDS_REVIEW, HIGH_CONFIDENCE}
    for _, finding in result.iterrows():
        assert df.index[finding["Row Position"] - 1] == finding["Record ID"]


def test_integration_quality_checks_and_anomalies_in_one_table():
    from modules.anomaly_pipeline import run_anomaly_pipeline

    df = build_real_dataset()[["Income"]].astype(float)
    raw = {"iqr_outliers": run_real_checks(build_real_dataset())["iqr_outliers"]}
    pipeline = run_anomaly_pipeline(df)
    raw["anomaly_analysis"] = pipeline.analysis

    result = combine_quality_results(raw, index=df.index)

    assert set(result["Category"]) <= {CATEGORY_OUTLIER, CATEGORY_ANOMALY}
    assert result["Row Position"].notna().all()
