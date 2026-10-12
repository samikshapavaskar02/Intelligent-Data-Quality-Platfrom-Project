"""Tests for modules/quality_score.py.

Most tests use the REAL detection modules (with the same rules app.py uses)
on small datasets with known problems, so the score is checked against what
the detectors really return. A few hand-built result frames cover cases the
detectors cannot produce on demand (for example a column that is 30 % missing
inside a wide dataset, or a check whose result is malformed).
"""

import copy

import numpy as np
import pandas as pd
import pytest

from modules.anomaly_analysis import HIGH_CONFIDENCE, NEEDS_REVIEW, NORMAL, POTENTIAL_ANOMALY
from modules.consistency import detect_inconsistent_values
from modules.data_type_validation import detect_data_type_problems
from modules.duplicates import detect_duplicates
from modules.format_validation import detect_format_problems
from modules.invalid_values import detect_invalid_values
from modules.missing_values import detect_missing_values
from modules.quality_score import (
    CAP_CRITICAL,
    CAP_HIGH,
    DIMENSION_COLUMNS,
    RATING_NOT_SCORED,
    SEV_CRITICAL,
    SEV_HIGH,
    SEV_LOW,
    SEV_MEDIUM,
    SEV_NOT_RATED,
    SEV_REVIEW,
    TREAT_BLANK,
    TREAT_BLANK_GROUP,
    TREAT_COMPLETENESS,
    TREAT_REVIEW,
    TREAT_UNIDENTIFIABLE,
    TREAT_UNIQUENESS,
    TREAT_VALIDITY,
    WEIGHTS,
    QualityScoreResult,
    _round1,
    add_severity_columns,
    calculate_quality_score,
    classify_severity,
    get_rating,
)
from modules.statistical_outliers import detect_iqr_outliers
from modules.unified_dashboard import build_unified_table
from modules.unified_results import (
    CATEGORY_ANOMALY,
    CATEGORY_OUTLIER,
    CATEGORY_QUALITY,
    SRC_ANOMALY,
    SRC_CONSISTENCY,
    SRC_DATA_TYPE,
    SRC_DUPLICATES,
    SRC_FORMAT,
    SRC_IF,
    SRC_INVALID,
    SRC_IQR,
    SRC_MISSING,
    UNIFIED_COLUMNS,
    combine_quality_results,
)

RULES = {"Age": (0, 120), "Marks": (0, 100), "Rating": (1, 5)}   # same as app.py


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def run_detectors(df):
    """What app.py computes, using the real detectors."""
    return {
        SRC_MISSING: detect_missing_values(df),
        SRC_DUPLICATES: detect_duplicates(df),
        SRC_INVALID: detect_invalid_values(df, RULES),
        SRC_DATA_TYPE: detect_data_type_problems(df),
        SRC_FORMAT: detect_format_problems(df),
        SRC_CONSISTENCY: detect_inconsistent_values(df),
        SRC_IQR: detect_iqr_outliers(df),
    }


def score(df):
    return calculate_quality_score(df, run_detectors(df))


def dim(result, name):
    return result.dimensions.set_index("Dimension").loc[name]


def view_of(result):
    return add_severity_columns(result.unified, result)


def rows_at(view, issue, position):
    sub = view[view["Issue Type"] == issue]
    return sub[sub["Row Position"].fillna(-1) == position]


def clean_df(n=20):
    """Valid, complete, unique, consistent, no outliers."""
    return pd.DataFrame({
        "Name": [f"P{i}" for i in range(n)],
        "Age": 20 + np.arange(n),
        "City": ["Pune", "Mumbai"] * (n // 2),
        "Email": [f"p{i}@x.com" for i in range(n)],
    })


def overlap_df():
    """Age has: "150" (invalid), "Twenty" (invalid AND data type), "  " (blank).

    "  " is missing, invalid and a data type problem for the detectors.
    """
    return pd.DataFrame({
        "Age": ["30", "150", "Twenty", "  ", "25", "28", "31", "29", "27", "26"],
        "Name": [f"P{i}" for i in range(10)],
    })


def wide_missing_case(missing_in_first_column, n_rows=200, n_cols=10):
    """Hand-built case: only column c0 has missing values; only that check ran."""
    columns = [f"c{i}" for i in range(n_cols)]
    df = pd.DataFrame(np.zeros((n_rows, n_cols)), columns=columns)
    frame = pd.DataFrame({
        "Column": columns,
        "Missing Values": [missing_in_first_column] + [0] * (n_cols - 1),
        "Missing Percentage": [100.0 * missing_in_first_column / n_rows] + [0.0] * (n_cols - 1),
    })
    return df, {SRC_MISSING: frame}


# ----------------------------------------------------------------------
# Constants, rating bands, rounding, severity ladder
# ----------------------------------------------------------------------
def test_weights_and_caps_follow_the_agreed_methodology():
    assert WEIGHTS == {
        "Completeness": 0.30, "Validity": 0.25, "Consistency": 0.25, "Uniqueness": 0.20,
    }
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)
    assert CAP_CRITICAL == 74.9
    assert CAP_HIGH == 89.9


@pytest.mark.parametrize("value, expected", [
    (100.0, "Excellent"), (90.0, "Excellent"), (89.9, "Good"), (75.0, "Good"),
    (74.9, "Fair"), (50.0, "Fair"), (49.9, "Poor"), (0.0, "Poor"),
    (None, RATING_NOT_SCORED),
])
def test_get_rating_bands(value, expected):
    assert get_rating(value) == expected


@pytest.mark.parametrize("value, expected", [
    (89.95, 90.0), (74.95, 75.0), (89.94, 89.9), (0.05, 0.1), (100, 100.0), (0, 0.0),
])
def test_rounding_is_half_up(value, expected):
    assert _round1(value) == expected


@pytest.mark.parametrize("issue, share, expected", [
    # Invalid / data type / duplicate: any occurrence is at least Medium.
    ("Invalid Value", 0.001, SEV_MEDIUM), ("Invalid Value", 0.0199, SEV_MEDIUM),
    ("Invalid Value", 0.02, SEV_HIGH), ("Invalid Value", 0.0999, SEV_HIGH),
    ("Invalid Value", 0.10, SEV_CRITICAL),
    ("Data Type Problem", 0.02, SEV_HIGH),
    ("Duplicate Record", 0.0199, SEV_MEDIUM), ("Duplicate Record", 0.02, SEV_HIGH),
    ("Duplicate Record", 0.10, SEV_CRITICAL),
    # Format / inconsistent.
    ("Format Problem", 0.0199, SEV_LOW), ("Format Problem", 0.02, SEV_MEDIUM),
    ("Format Problem", 0.0999, SEV_MEDIUM), ("Format Problem", 0.10, SEV_HIGH),
    ("Format Problem", 0.2499, SEV_HIGH), ("Format Problem", 0.25, SEV_CRITICAL),
    ("Inconsistent Representation", 0.0199, SEV_LOW),
    ("Inconsistent Representation", 0.25, SEV_CRITICAL),
    # Missing.
    ("Missing Value", 0.0499, SEV_LOW), ("Missing Value", 0.05, SEV_MEDIUM),
    ("Missing Value", 0.1999, SEV_MEDIUM), ("Missing Value", 0.20, SEV_HIGH),
    ("Missing Value", 0.4999, SEV_HIGH), ("Missing Value", 0.50, SEV_CRITICAL),
    ("Missing Value", 1.0, SEV_CRITICAL),
])
def test_severity_ladder_boundaries(issue, share, expected):
    assert classify_severity(issue, share) == expected


def test_classify_severity_rejects_bad_input():
    with pytest.raises(ValueError, match="No severity ladder"):
        classify_severity("IQR Outlier", 0.5)
    with pytest.raises(ValueError):
        classify_severity("Missing Value", float("nan"))


# ----------------------------------------------------------------------
# Clean, empty and all-missing datasets
# ----------------------------------------------------------------------
def test_clean_dataset_scores_100_with_no_findings_and_no_notes():
    result = score(clean_df())

    assert isinstance(result, QualityScoreResult)
    assert result.overall_score == 100.0
    assert result.raw_score == 100.0
    assert result.rating == "Excellent"
    assert result.cap_limit is None and result.cap_applied is False
    assert result.cap_reason is None
    assert result.severity_summary.empty
    assert result.unscored_impact is None
    assert result.notes == []
    assert list(result.dimensions.columns) == DIMENSION_COLUMNS
    assert (result.dimensions["Status"] == "Assessed").all()
    assert (result.dimensions["Score"] == 100.0).all()


@pytest.mark.parametrize("df", [pd.DataFrame(), pd.DataFrame({"Age": []})])
def test_empty_dataset_is_not_scored_and_does_not_raise(df):
    result = calculate_quality_score(df, {})

    assert result.overall_score is None
    assert result.raw_score is None
    assert result.rating == RATING_NOT_SCORED
    assert (result.dimensions["Status"] == "Not assessed").all()
    assert result.dimensions["Score"].isna().all()
    assert result.severity_summary.empty
    assert result.findings.empty
    assert any("empty" in note for note in result.notes)


def test_empty_dataset_still_labels_review_rows():
    iqr = pd.DataFrame({
        "Row": [1], "Column": ["Age"], "Value": [5], "Lower Bound": [0],
        "Upper Bound": [1], "Method": ["IQR"], "Status": ["Potential Outlier"],
    })
    unified = combine_quality_results({SRC_IQR: iqr})

    result = calculate_quality_score(pd.DataFrame({"Age": []}), {}, unified=unified)

    assert result.overall_score is None
    assert result.findings.loc[0, "Severity"] == SEV_REVIEW
    assert result.findings.loc[0, "Score Treatment"] == TREAT_REVIEW


def test_all_missing_dataset():
    df = pd.DataFrame({"Name": [None, "  ", ""], "Age": [np.nan] * 3})

    result = score(df)

    assert dim(result, "Completeness")["Score"] == 0.0
    assert dim(result, "Completeness")["Issues Counted"] == 6
    # Nothing is left to check for validity or consistency.
    assert dim(result, "Validity")["Status"] == "Not assessed"
    assert dim(result, "Consistency")["Status"] == "Not assessed"
    assert dim(result, "Uniqueness")["Status"] == "Assessed"
    assert result.rating == "Poor"
    assert result.overall_score < 50
    assert (result.severity_summary["Severity"] == SEV_CRITICAL).all()
    assert any("rescaled" in note for note in result.notes)
    # The blank-only spelling group is a Completeness matter, not Consistency.
    view = view_of(result)
    blank_group = view[view["Issue Type"] == "Inconsistent Representation"]
    assert (blank_group["Score Treatment"] == TREAT_BLANK_GROUP).all()
    assert (blank_group["Severity"] == SEV_NOT_RATED).all()


# ----------------------------------------------------------------------
# Empty result frames: "the check ran and found nothing"
# ----------------------------------------------------------------------
def test_the_real_detectors_return_columnless_frames_when_nothing_is_found():
    """Why the score must accept ``pd.DataFrame()`` as a clean result."""
    df = clean_df()

    for detector in (detect_inconsistent_values, detect_format_problems, detect_iqr_outliers):
        frame = detector(df)
        assert frame.empty and len(frame.columns) == 0


@pytest.mark.parametrize("source, name, status", [
    (SRC_MISSING, "Completeness", "Assessed"),
    (SRC_CONSISTENCY, "Consistency", "Assessed"),
    (SRC_DUPLICATES, "Uniqueness", "Assessed"),
    (SRC_INVALID, "Validity", "Partial"),
    (SRC_DATA_TYPE, "Validity", "Partial"),
    (SRC_FORMAT, "Validity", "Partial"),
])
def test_a_columnless_empty_frame_means_ran_and_clean_for_every_check(source, name, status):
    result = calculate_quality_score(clean_df(), {source: pd.DataFrame()})

    row = dim(result, name)
    assert row["Status"] == status
    assert row["Score"] == 100.0
    assert row["Issues Counted"] == 0
    assert not any("could not be used" in note for note in result.notes)
    assert result.overall_score == 100.0


# ----------------------------------------------------------------------
# Overlapping findings
# ----------------------------------------------------------------------
def test_overlapping_findings_are_counted_once():
    result = score(overlap_df())
    view = view_of(result)

    # 20 cells; 1 blank. Bad non-blank cells: "150" and "Twenty" (2), not 5.
    assert dim(result, "Completeness")["Issues Counted"] == 1
    assert dim(result, "Validity")["Issues Counted"] == 2
    assert dim(result, "Validity")["Items Checked"] == 19
    assert dim(result, "Validity")["Score"] == pytest.approx(89.5)

    assert rows_at(view, "Invalid Value", 2).iloc[0]["Score Treatment"] == TREAT_VALIDITY
    assert rows_at(view, "Invalid Value", 3).iloc[0]["Score Treatment"] == TREAT_VALIDITY
    twenty_as_type = rows_at(view, "Data Type Problem", 3).iloc[0]
    assert twenty_as_type["Score Treatment"].startswith("Counted once")
    assert "Invalid Value" in twenty_as_type["Score Treatment"]
    assert twenty_as_type["Severity"] == SEV_NOT_RATED

    # The blank cell is counted under Completeness only.
    assert rows_at(view, "Invalid Value", 4).iloc[0]["Score Treatment"] == TREAT_BLANK
    assert rows_at(view, "Data Type Problem", 4).iloc[0]["Score Treatment"] == TREAT_BLANK
    missing_row = view[view["Issue Type"] == "Missing Value"].iloc[0]
    assert missing_row["Score Treatment"] == TREAT_COMPLETENESS

    # Findings stay visible: every detector row is still in the table.
    assert len(view) == 6


def test_severity_and_cap_for_the_overlap_dataset():
    result = score(overlap_df())

    # 2 counted cells in 9 non-blank Age cells = 22 % -> Critical.
    assert result.severity_summary.iloc[0]["Severity"] == SEV_CRITICAL
    assert result.severity_summary.iloc[0]["Issue Type"] == "Invalid Value"
    assert result.raw_score > CAP_CRITICAL
    assert result.cap_limit == CAP_CRITICAL
    assert result.cap_applied is True
    assert result.overall_score == CAP_CRITICAL
    assert result.rating == "Fair"
    assert "Critical" in result.cap_reason


def test_blank_cells_are_not_excluded_when_the_missing_check_did_not_run():
    df = overlap_df()
    qr = run_detectors(df)
    del qr[SRC_MISSING]

    result = calculate_quality_score(df, qr)

    # "150", "Twenty" and the blank "  " are now three distinct bad cells.
    assert dim(result, "Validity")["Issues Counted"] == 3
    assert dim(result, "Validity")["Items Checked"] == 20       # all cells
    assert dim(result, "Completeness")["Status"] == "Not assessed"
    assert any("all cells" in note for note in result.notes)


# ----------------------------------------------------------------------
# Duplicates
# ----------------------------------------------------------------------
def test_duplicates_count_redundant_copies_only():
    df = pd.DataFrame({
        "A": [1, 1, 1, 2, 2, 3, 4, 5],
        "B": ["x", "x", "x", "y", "y", "z", "w", "v"],
    })
    assert df.duplicated().sum() == 3            # 2 + 1 redundant copies

    result = score(df)
    view = view_of(result)

    assert dim(result, "Uniqueness")["Issues Counted"] == 3
    assert dim(result, "Uniqueness")["Items Checked"] == 8
    assert dim(result, "Uniqueness")["Score"] == 62.5
    assert result.severity_summary.iloc[0]["Severity"] == SEV_CRITICAL   # 37.5 %

    # All five group members are listed with the same treatment: the
    # original row is not guessed.
    duplicates = view[view["Issue Type"] == "Duplicate Record"]
    assert len(duplicates) == 5
    assert (duplicates["Score Treatment"] == TREAT_UNIQUENESS).all()
    assert (duplicates["Severity"] == SEV_CRITICAL).all()


def test_duplicate_severity_starts_high_at_two_percent():
    n = 50                                        # 1 redundant copy = 2 %
    df = pd.DataFrame({"A": list(range(n - 1)) + [0], "B": ["q"] * n})

    result = score(df)

    assert dim(result, "Uniqueness")["Issues Counted"] == 1
    assert result.severity_summary.iloc[0]["Severity"] == SEV_HIGH
    assert result.cap_limit == CAP_HIGH


def test_duplicate_index_labels_with_duplicate_rows():
    df = pd.DataFrame({"A": [1, 1, 2], "B": ["x", "x", "y"]}, index=[0, 0, 1])

    result = score(df)

    assert dim(result, "Uniqueness")["Issues Counted"] == 1
    assert dim(result, "Uniqueness")["Score"] == pytest.approx(66.7, abs=0.05)
    assert result.cap_applied is True
    assert result.overall_score == CAP_CRITICAL
    assert result.rating == "Fair"


# ----------------------------------------------------------------------
# Findings whose row cannot be identified
# ----------------------------------------------------------------------
def test_unidentifiable_findings_are_shown_but_not_scored_and_impact_is_bounded():
    df = pd.DataFrame(
        {"Email": ["a@b.com", "bad", "a2@b.com", "c@d.com", "e@f.in", "g@h.in"],
         "Age": [30, 31, 32, 33, 34, 35]},
        index=[0, 0, 1, 2, 3, 4],                 # label 0 occurs twice
    )

    result = score(df)
    view = view_of(result)

    finding = view[view["Issue Type"] == "Format Problem"].iloc[0]
    assert pd.isna(finding["Row Position"])        # never guessed
    assert finding["Score Treatment"] == TREAT_UNIDENTIFIABLE
    assert finding["Severity"] == SEV_NOT_RATED

    assert dim(result, "Validity")["Issues Counted"] == 0
    assert result.overall_score == 100.0

    # 1 possible bad cell in 12: at most 8.3 Validity points, 2.1 overall.
    assert result.unscored_impact == {
        "unidentifiable_findings": 1,
        "max_validity_points": 8.3,
        "max_overall_points": 2.1,
    }
    assert any("could not be tied to an exact row" in note for note in result.notes)


def test_several_unidentifiable_findings_are_never_merged_or_scored():
    df = pd.DataFrame(
        {"Email": ["a@b.com", "bad1", "bad2", "c@d.com", "e@f.in", "g@h.in"],
         "Age": [30, 31, 32, 33, 34, 35]},
        index=[0, 0, 0, 1, 2, 3],                 # label 0 occurs three times
    )

    result = score(df)
    view = view_of(result)

    findings = view[view["Issue Type"] == "Format Problem"]
    assert len(findings) == 2                      # both stay visible, not merged
    assert findings["Row Position"].isna().all()
    assert (findings["Score Treatment"] == TREAT_UNIDENTIFIABLE).all()
    assert dim(result, "Validity")["Issues Counted"] == 0
    assert result.overall_score == 100.0
    assert result.unscored_impact["unidentifiable_findings"] == 2
    assert result.unscored_impact["max_validity_points"] == 16.7     # 2 of 12 cells
    assert result.unscored_impact["max_overall_points"] == 4.2       # x 25 % weight


# ----------------------------------------------------------------------
# Consistency: minority spellings, normalisation like consistency.py
# ----------------------------------------------------------------------
def test_consistency_counts_minority_spellings():
    df = pd.DataFrame({
        "City": ["Pune"] * 5 + ["pune"] * 2 + ["PUNE"] + ["Mumbai"] * 2,
        "Id": range(1, 11),
    })

    result = score(df)

    # "Pune" (5) is the largest spelling; the other 3 cells are counted.
    assert dim(result, "Consistency")["Issues Counted"] == 3
    assert dim(result, "Consistency")["Items Checked"] == 20
    assert dim(result, "Consistency")["Score"] == pytest.approx(85.0)
    summary = result.severity_summary.iloc[0]
    assert (summary["Issue Type"], summary["Column"]) == ("Inconsistent Representation", "City")
    assert summary["Severity"] == SEV_CRITICAL                      # 3 of 10 = 30 %
    finding = view_of(result).query("`Issue Type` == 'Inconsistent Representation'").iloc[0]
    assert finding["Severity"] == SEV_CRITICAL
    assert "Consistency" in finding["Score Treatment"]


def test_consistency_tie_between_spellings_does_not_guess_the_right_one():
    df = pd.DataFrame({"City": ["A", "A", "a", "a", "b", "c"], "Id": range(6)})

    result = score(df)

    assert dim(result, "Consistency")["Issues Counted"] == 2        # 4 cells - largest 2
    assert dim(result, "Consistency")["Score"] == pytest.approx(100 * (1 - 2 / 12), abs=0.05)


def test_consistency_uses_strip_and_lower_like_the_detector():
    df = pd.DataFrame({
        "Gender": ["Male", "Male ", "male", "Female", "Female", "Female"],
        "Place": ["New York", "New  York", "Paris", "Paris", "Paris", "Paris"],
    })
    detector = detect_inconsistent_values(df)

    # The detector groups "Male", "Male " and "male" (strip + lower) and does
    # NOT group "New York" with "New  York" (inner spaces are kept).
    assert list(detector["Column"]) == ["Gender"]

    result = calculate_quality_score(df, run_detectors(df))

    assert dim(result, "Consistency")["Issues Counted"] == 2        # 3 cells - 1


def test_consistency_is_zero_when_the_detector_reports_nothing():
    result = score(clean_df())
    assert dim(result, "Consistency")["Issues Counted"] == 0
    assert dim(result, "Consistency")["Status"] == "Assessed"


# ----------------------------------------------------------------------
# Severity caps and rating consistency
# ----------------------------------------------------------------------
def test_high_severity_caps_an_otherwise_excellent_score():
    df, qr = wide_missing_case(60)               # c0 is 30 % missing -> High
    result = calculate_quality_score(df, qr)

    assert dim(result, "Completeness")["Score"] == pytest.approx(97.0)
    assert result.raw_score == 97.0
    assert result.severity_summary.iloc[0]["Severity"] == SEV_HIGH
    assert result.cap_limit == CAP_HIGH and result.cap_applied is True
    assert result.overall_score == CAP_HIGH
    assert result.rating == "Good"               # never "Excellent" with a High issue


def test_critical_severity_cap():
    df, qr = wide_missing_case(120)              # c0 is 60 % missing -> Critical
    result = calculate_quality_score(df, qr)

    assert result.raw_score == 94.0
    assert result.cap_limit == CAP_CRITICAL
    assert result.overall_score == CAP_CRITICAL
    assert result.rating == "Fair"


def test_medium_severity_does_not_cap():
    df, qr = wide_missing_case(10)               # 5 % -> Medium
    result = calculate_quality_score(df, qr)

    assert result.severity_summary.iloc[0]["Severity"] == SEV_MEDIUM
    assert result.cap_limit is None and result.cap_applied is False
    assert result.overall_score == 99.5
    assert result.rating == "Excellent"


def test_a_cap_that_is_not_binding_is_not_reported_as_applied():
    df, qr = wide_missing_case(40, n_rows=100, n_cols=2)   # 40 % -> High, score 80.0

    result = calculate_quality_score(df, qr)

    assert result.severity_summary.iloc[0]["Severity"] == SEV_HIGH
    assert result.cap_limit == CAP_HIGH
    assert result.overall_score == 80.0 and result.raw_score == 80.0
    assert result.cap_applied is False
    assert result.cap_reason is None
    assert result.rating == "Good"


def test_a_raw_score_that_rounds_up_to_excellent_is_still_capped_to_good():
    # 3012 of 30000 cells missing: raw 89.96 rounds to 90.0 (Excellent), but
    # the 30 % missing column is High, so the shown score must be 89.9 / Good.
    df, qr = wide_missing_case(3012, n_rows=10000, n_cols=3)

    result = calculate_quality_score(df, qr)

    assert result.severity_summary.iloc[0]["Severity"] == SEV_HIGH
    assert result.raw_score == 90.0
    assert result.overall_score == CAP_HIGH
    assert result.cap_applied is True
    assert result.rating == "Good" == get_rating(result.overall_score)


def test_a_raw_score_that_rounds_up_to_good_is_still_capped_to_fair():
    # 7512 of 30000 cells missing: raw 74.96 rounds to 75.0 (Good), but a
    # 75 % missing column is Critical, so the shown score is 74.9 / Fair.
    df, qr = wide_missing_case(7512, n_rows=10000, n_cols=3)

    result = calculate_quality_score(df, qr)

    assert result.severity_summary.iloc[0]["Severity"] == SEV_CRITICAL
    assert result.raw_score == 75.0
    assert result.overall_score == CAP_CRITICAL
    assert result.rating == "Fair" == get_rating(result.overall_score)


# ----------------------------------------------------------------------
# IQR outliers and ML anomalies are review only
# ----------------------------------------------------------------------
def test_iqr_and_ml_findings_are_visible_but_never_scored():
    df = pd.DataFrame({
        "Age": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 500],
        "Name": [f"P{i}" for i in range(12)],
    })
    qr = run_detectors(df)
    ml = pd.DataFrame({
        "IF Result": [NORMAL] * 12, "LOF Result": [NORMAL] * 12, "Status": [NORMAL] * 12,
    }, index=df.index)
    ml.loc[3, ["IF Result", "Status"]] = [POTENTIAL_ANOMALY, NEEDS_REVIEW]
    ml.loc[11, ["IF Result", "LOF Result", "Status"]] = [
        POTENTIAL_ANOMALY, POTENTIAL_ANOMALY, HIGH_CONFIDENCE]
    unified, _ = build_unified_table(df, qr, {SRC_ANOMALY: ml})

    with_review = calculate_quality_score(df, qr, unified=unified)
    view = view_of(with_review)

    review_rows = view[view["Category"].isin([CATEGORY_OUTLIER, CATEGORY_ANOMALY])]
    assert (view["Category"] == CATEGORY_ANOMALY).sum() == 2
    assert (view["Category"] == CATEGORY_OUTLIER).sum() >= 1
    assert (review_rows["Severity"] == SEV_REVIEW).all()
    assert (review_rows["Score Treatment"] == TREAT_REVIEW).all()

    # The score is identical with and without them.
    without_iqr = {k: v for k, v in qr.items() if k != SRC_IQR}
    plain = calculate_quality_score(df, without_iqr)
    assert with_review.overall_score == plain.overall_score
    pd.testing.assert_frame_equal(with_review.dimensions, plain.dimensions)

    # Age 500 is also an INVALID value (rule 0-120): that finding is scored,
    # 1 of 12 Age cells = 8.3 % -> High, which caps the score at 89.9.
    invalid = view[view["Issue Type"] == "Invalid Value"].iloc[0]
    assert invalid["Score Treatment"] == TREAT_VALIDITY
    assert invalid["Severity"] == SEV_HIGH
    assert with_review.overall_score == CAP_HIGH
    assert with_review.rating == "Good"


def test_an_ml_result_that_cannot_be_added_is_noted_and_never_scored():
    df = clean_df()
    qr = run_detectors(df)
    # Labels 500.. are not in the dataset: the unified table cannot map them.
    qr[SRC_IF] = pd.DataFrame({
        "if_prediction": [-1], "if_status": [POTENTIAL_ANOMALY], "if_score": [0.9],
    }, index=[500])

    result = calculate_quality_score(df, qr)

    assert any("Isolation Forest could not be added to the findings table" in n
               for n in result.notes)
    assert not (result.unified["Category"] == CATEGORY_ANOMALY).any()
    assert result.overall_score == 100.0


# ----------------------------------------------------------------------
# Partial checks and malformed results
# ----------------------------------------------------------------------
def test_absent_checks_make_validity_partial_and_rescale_weights():
    df = clean_df()
    qr = run_detectors(df)
    del qr[SRC_FORMAT]
    del qr[SRC_INVALID]
    qr[SRC_CONSISTENCY] = None                   # None = did not run

    result = calculate_quality_score(df, qr)

    assert dim(result, "Validity")["Status"] == "Partial"
    assert dim(result, "Validity")["Checks Run"] == "1 of 3"
    assert dim(result, "Consistency")["Status"] == "Not assessed"
    weights = result.dimensions.set_index("Dimension")["Effective Weight (%)"]
    assert weights["Completeness"] == pytest.approx(40.0)
    assert weights["Validity"] == pytest.approx(33.3)
    assert weights["Uniqueness"] == pytest.approx(26.7)
    assert weights["Consistency"] == 0.0
    assert result.overall_score == 100.0
    assert any("partial" in note for note in result.notes)
    assert any("rescaled" in note for note in result.notes)


def test_no_checks_at_all_gives_no_score():
    for qr in ({}, {SRC_MISSING: None, SRC_DUPLICATES: None}):
        result = calculate_quality_score(clean_df(), qr)
        assert result.overall_score is None
        assert result.rating == RATING_NOT_SCORED
        assert (result.dimensions["Status"] == "Not assessed").all()
        assert any("No dimension could be assessed" in note for note in result.notes)


def test_findings_of_a_check_that_did_not_run_are_explained_not_scored():
    df = overlap_df()
    qr = run_detectors(df)
    unified, _ = build_unified_table(df, qr, None)
    qr[SRC_DATA_TYPE] = None                     # did not run, but the table has its rows

    result = calculate_quality_score(df, qr, unified=unified)

    rows = view_of(result).query("`Issue Type` == 'Data Type Problem'")
    assert len(rows) == 2
    assert (rows["Score Treatment"] == "Not scored - the check did not run").all()
    assert (rows["Severity"] == SEV_NOT_RATED).all()
    assert dim(result, "Validity")["Checks Run"] == "2 of 3"


def test_an_empty_frame_means_the_check_ran_and_found_nothing():
    df = clean_df()

    result = calculate_quality_score(df, {SRC_INVALID: pd.DataFrame()})

    assert dim(result, "Validity")["Status"] == "Partial"
    assert dim(result, "Validity")["Score"] == 100.0
    assert dim(result, "Validity")["Issues Counted"] == 0


def test_a_malformed_result_is_reported_and_the_rest_is_still_scored():
    df = clean_df()
    qr = run_detectors(df)
    qr[SRC_MISSING] = pd.DataFrame({"x": [1]})   # not a missing-values result

    result = calculate_quality_score(df, qr)

    assert dim(result, "Completeness")["Status"] == "Not assessed"
    assert any("Missing values could not be used" in note for note in result.notes)
    assert dim(result, "Uniqueness")["Status"] == "Assessed"
    # Without the missing-value check Validity uses all cells as its base.
    assert dim(result, "Validity")["Items Checked"] == 80
    assert result.overall_score == 100.0


def test_a_consistency_result_for_an_unknown_column_is_reported_not_raised():
    df = clean_df()
    qr = run_detectors(df)
    qr[SRC_CONSISTENCY] = pd.DataFrame({
        "Column": ["Nope"], "Normalized Value": ["x"],
        "Different Representations": ["x, X"], "Status": ["Inconsistent Representation"],
    })

    result = calculate_quality_score(df, qr)

    assert dim(result, "Consistency")["Status"] == "Not assessed"
    assert any("Inconsistent values could not be used" in note for note in result.notes)
    rows = view_of(result).query("`Issue Type` == 'Inconsistent Representation'")
    assert (rows["Score Treatment"] == "Not scored - the results of this check could not be processed").all()
    assert dim(result, "Uniqueness")["Status"] == "Assessed"


def test_an_incomplete_unified_table_makes_that_check_unusable_instead_of_clean():
    df = pd.DataFrame({"Age": [30, 31, 150, 32, 33], "Name": list("abcde")})
    qr = run_detectors(df)
    assert len(qr[SRC_INVALID]) == 1
    stale = combine_quality_results({}, index=df.index)      # has no findings at all

    result = calculate_quality_score(df, qr, unified=stale)

    assert dim(result, "Validity")["Checks Run"] == "2 of 3"
    assert dim(result, "Validity")["Status"] == "Partial"
    assert any("Invalid values could not be used" in note for note in result.notes)
    assert result.unified is stale


# ----------------------------------------------------------------------
# Row identity with different kinds of index
# ----------------------------------------------------------------------
def make_indexed(kind):
    df = pd.DataFrame({
        "Age": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 150],
        "Email": [f"p{i}@x.com" for i in range(12)],
    })
    df.loc[4, "Email"] = "bad"
    if kind == "offset":
        df.index = range(100, 112)
    elif kind == "shuffled":
        df.index = np.random.default_rng(1).permutation(12) + 50
    return df


@pytest.mark.parametrize("kind", ["default", "offset", "shuffled"])
def test_row_identity_is_exact_for_any_unique_integer_index(kind):
    df = make_indexed(kind)

    result = score(df)
    view = view_of(result)

    assert dim(result, "Validity")["Issues Counted"] == 2     # 150 and "bad"
    assert result.unscored_impact is None
    fmt = view[view["Issue Type"] == "Format Problem"].iloc[0]
    assert fmt["Row Position"] == 5
    assert fmt["Record ID"] == df.index[4]
    assert fmt["Score Treatment"] == TREAT_VALIDITY
    # The same data gives the same score whatever the labels are.
    assert result.overall_score == score(make_indexed("default")).overall_score


def test_a_supplied_unified_table_gives_the_same_result_as_a_built_one():
    df = overlap_df()
    qr = run_detectors(df)
    unified, _ = build_unified_table(df, qr, None)

    built = calculate_quality_score(df, qr)
    supplied = calculate_quality_score(df, qr, unified=unified)

    assert supplied.overall_score == built.overall_score
    pd.testing.assert_frame_equal(supplied.dimensions, built.dimensions)
    pd.testing.assert_frame_equal(supplied.findings, built.findings)
    assert supplied.unified is unified


# ----------------------------------------------------------------------
# Invariants on many different datasets
# ----------------------------------------------------------------------
def random_dataset(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(30, 70))
    df = pd.DataFrame({
        "Age": rng.integers(18, 70, n).astype(float),
        "City": rng.choice(["Pune", "pune", "PUNE", "Mumbai"], n),
        "Email": [f"user{i}@example.com" for i in range(n)],
    })
    for pos in rng.choice(n, size=int(rng.integers(0, n // 3)), replace=False):
        df.iloc[pos, 0] = 150.0
    for pos in rng.choice(n, size=int(rng.integers(0, n // 4)), replace=False):
        df.iloc[pos, 0] = np.nan
    for pos in rng.choice(n, size=int(rng.integers(0, n // 4)), replace=False):
        df.iloc[pos, 1] = None
    for pos in rng.choice(n, size=int(rng.integers(0, n // 3)), replace=False):
        df.iloc[pos, 2] = "not-an-email"
    return pd.concat([df, df.iloc[: int(rng.integers(0, 6))]], ignore_index=True)


def _blank(value):
    return value is None or (isinstance(value, str) and not value.strip()) or bool(pd.isna(value))


@pytest.mark.parametrize("seed", range(15))
def test_invariants_hold_on_varied_datasets(seed):
    df = random_dataset(seed)

    result = score(df)

    # Score, rating and caps agree after rounding.
    assert 0.0 <= result.overall_score <= 100.0
    assert result.overall_score <= result.raw_score
    assert result.rating == get_rating(result.overall_score)
    severities = set(result.severity_summary["Severity"])
    if SEV_CRITICAL in severities:
        assert result.overall_score <= CAP_CRITICAL
        assert result.rating in ("Fair", "Poor")
    elif SEV_HIGH in severities:
        assert result.overall_score <= CAP_HIGH
        assert result.rating != "Excellent"
    else:
        assert result.cap_limit is None

    # Every finding is annotated; nothing is dropped from the table.
    assert len(result.findings) == len(result.unified)
    assert result.findings["Severity"].notna().all()
    assert result.findings["Score Treatment"].notna().all()
    scored = result.findings["Score Treatment"].str.startswith("Scored")
    assert result.findings.loc[scored, "Severity"].isin(
        [SEV_LOW, SEV_MEDIUM, SEV_HIGH, SEV_CRITICAL]).all()

    # Validity counts each identifiable, non-blank bad cell exactly once.
    u = result.unified
    cells = u[(u["Category"] == CATEGORY_QUALITY)
              & u["Issue Type"].isin(["Invalid Value", "Data Type Problem", "Format Problem"])]
    distinct = {
        (int(r["Row Position"]), r["Column"])
        for _, r in cells.iterrows()
        if not _blank(r["Value"]) and not pd.isna(r["Row Position"])
    }
    assert dim(result, "Validity")["Issues Counted"] == len(distinct)


# ----------------------------------------------------------------------
# Validation, display helper, no side effects
# ----------------------------------------------------------------------
def test_invalid_inputs_raise_clear_errors():
    df = clean_df()
    with pytest.raises(TypeError):
        calculate_quality_score([1, 2], {})
    with pytest.raises(TypeError):
        calculate_quality_score(df, [])
    with pytest.raises(ValueError, match="Unknown source"):
        calculate_quality_score(df, {"nonsense": pd.DataFrame()})
    with pytest.raises(TypeError):
        calculate_quality_score(df, {SRC_MISSING: "not a frame"})
    with pytest.raises(TypeError):
        calculate_quality_score(df, {}, unified=[1])
    with pytest.raises(ValueError, match="missing required columns"):
        calculate_quality_score(df, {}, unified=pd.DataFrame({"x": [1]}))


def test_add_severity_columns_is_display_only():
    df = overlap_df()
    result = score(df)
    unified_before = result.unified.copy(deep=True)

    view = add_severity_columns(result.unified, result)

    assert list(view.columns) == UNIFIED_COLUMNS + ["Severity", "Score Treatment"]
    assert list(result.unified.columns) == UNIFIED_COLUMNS             # schema untouched
    pd.testing.assert_frame_equal(result.unified, unified_before)
    assert len(view) == len(result.unified)


def test_add_severity_columns_rejects_a_different_table():
    result = score(overlap_df())
    other = result.unified.iloc[:2].reset_index(drop=True)
    with pytest.raises(ValueError, match="index differs"):
        add_severity_columns(other, result)
    with pytest.raises(TypeError):
        add_severity_columns(result.unified, "not a result")


def test_inputs_are_not_modified():
    df = overlap_df()
    qr = run_detectors(df)
    df_before = df.copy(deep=True)
    qr_before = copy.deepcopy(qr)

    calculate_quality_score(df, qr)

    pd.testing.assert_frame_equal(df, df_before)
    assert set(qr) == set(qr_before)
    for key in qr:
        pd.testing.assert_frame_equal(qr[key], qr_before[key])
