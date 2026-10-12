"""Tests for modules/report_builder.py (quality report and anomaly report).

Run from the project root:

    python -m pytest tests/test_report_builder.py -v

The first sections use hand-built result objects (edge cases, empty and
missing inputs). The last section is an integration test: it runs the real
detectors, the real data quality score and the real anomaly sources on a small
dataset with a non-default index and checks that the reports match them.
"""

import ast
import copy
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modules import report_builder
from modules.anomaly_analysis import (
    HIGH_CONFIDENCE,
    NEEDS_REVIEW,
    NORMAL,
    POTENTIAL_ANOMALY,
)
from modules.quality_score import (
    SEVERITY_COLUMNS,
    QualityScoreResult,
    calculate_quality_score,
)
from modules.report_builder import (
    ANOMALY_REPORT_COLUMNS,
    QUALITY_REPORT_COLUMNS,
    build_anomaly_report,
    build_quality_report,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def make_score_result(rows=None, overall_score=85.0, severity_summary=None):
    """A QualityScoreResult whose severity_summary holds ``rows``.

    ``rows`` are (Dimension, Issue Type, Column, Severity, Affected, Total,
    Share (%)) tuples, i.e. the real severity_summary format.
    """
    if severity_summary is None:
        severity_summary = pd.DataFrame(rows or [], columns=SEVERITY_COLUMNS)
    return QualityScoreResult(
        overall_score=overall_score,
        raw_score=overall_score,
        rating="Good",
        cap_limit=None,
        cap_applied=False,
        cap_reason=None,
        dimensions=pd.DataFrame(),
        severity_summary=severity_summary,
        findings=pd.DataFrame(),
        unified=pd.DataFrame(),
    )


SAMPLE_ROWS = [
    ("Validity", "Invalid Value", "Age", "High", 2, 39, 5.13),
    ("Uniqueness", "Duplicate Record", "(whole row)", "High", 1, 40, 2.5),
    ("Validity", "Format Problem", "Email", "Medium", 1, 40, 2.5),
    ("Completeness", "Missing Value", "Age", "Low", 1, 40, 2.5),
]


def make_sources(index, if_status, lof_status, with_scores=True):
    """Anomaly sources in the format returned by run_anomaly_sources."""
    n = len(index)
    if_df = pd.DataFrame({
        "if_prediction": [-1 if s == POTENTIAL_ANOMALY else 1 for s in if_status],
        "if_status": if_status,
        "if_score": [0.40 + 0.01 * i for i in range(n)],
    }, index=index)
    lof_df = pd.DataFrame({
        "lof_prediction": [-1 if s == POTENTIAL_ANOMALY else 1 for s in lof_status],
        "lof_status": lof_status,
        "lof_score": [1.0 + 0.1 * i for i in range(n)],
    }, index=index)

    def combined(a, b):
        if a == POTENTIAL_ANOMALY and b == POTENTIAL_ANOMALY:
            return HIGH_CONFIDENCE
        if a == NORMAL and b == NORMAL:
            return NORMAL
        return NEEDS_REVIEW

    analysis = pd.DataFrame({
        "IF Result": if_status,
        "LOF Result": lof_status,
        "Status": [combined(a, b) for a, b in zip(if_status, lof_status)],
    }, index=index)

    sources = {"anomaly_analysis": analysis}
    if with_scores:
        sources["isolation_forest"] = if_df
        sources["lof"] = lof_df
    return sources


def sample_sources():
    """Records 101-104, as in the project PDF example."""
    return make_sources(
        pd.Index([101, 102, 103, 104], name="record_id"),
        [NORMAL, POTENTIAL_ANOMALY, POTENTIAL_ANOMALY, NORMAL],
        [NORMAL, POTENTIAL_ANOMALY, NORMAL, POTENTIAL_ANOMALY],
    )
def csv_text(report):
    """Return CSV text with consistent Unix-style line endings."""
    return report.to_csv(index=False).replace("\r\n", "\n")
def csv_round_trip(report):
    text = report.to_csv(index=False)
    return text, pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)


# ----------------------------------------------------------------------
# Quality report: mapping
# ----------------------------------------------------------------------
def test_quality_report_maps_severity_summary_fields():
    report, notes = build_quality_report(make_score_result(SAMPLE_ROWS))

    assert notes == []
    assert list(report.columns) == QUALITY_REPORT_COLUMNS
    assert report["Issue"].tolist() == [
        "Invalid Value", "Duplicate Record", "Format Problem", "Missing Value",
    ]
    assert report["Column"].tolist() == ["Age", "(whole row)", "Email", "Age"]
    assert report["Count"].tolist() == [2, 1, 1, 1]          # the score's "Affected"
    assert report["Percentage"].tolist() == [5.13, 2.5, 2.5, 2.5]  # "Share (%)"
    assert report["Severity"].tolist() == ["High", "High", "Medium", "Low"]


def test_quality_report_keeps_the_score_order_and_values_untouched():
    # Same severities, deliberately not sorted: the report must not re-rank.
    rows = [SAMPLE_ROWS[3], SAMPLE_ROWS[0]]
    report, _ = build_quality_report(make_score_result(rows))
    assert report["Severity"].tolist() == ["Low", "High"]


def test_quality_report_types_and_fresh_index():
    report, _ = build_quality_report(make_score_result(SAMPLE_ROWS))
    assert report["Count"].dtype == np.dtype("int64")
    assert report["Percentage"].dtype == np.dtype("float64")
    assert report.index.tolist() == [0, 1, 2, 3]


def test_quality_report_does_not_modify_the_score_result():
    result = make_score_result(SAMPLE_ROWS)
    before = result.severity_summary.copy(deep=True)
    summary_object = result.severity_summary

    build_quality_report(result)

    pd.testing.assert_frame_equal(result.severity_summary, before)
    assert result.severity_summary is summary_object


def test_quality_report_shares_no_data_with_the_score_result():
    result = make_score_result(SAMPLE_ROWS)
    before = result.severity_summary.copy(deep=True)
    report, _ = build_quality_report(result)

    for column, new_value in (("Issue", "X"), ("Column", "X"), ("Count", 99),
                              ("Percentage", 99.9), ("Severity", "X")):
        report.loc[0, column] = new_value

    pd.testing.assert_frame_equal(result.severity_summary, before)


# ----------------------------------------------------------------------
# Quality report: empty, missing and invalid inputs
# ----------------------------------------------------------------------
def test_quality_report_without_rated_issues_is_header_only():
    report, notes = build_quality_report(make_score_result([]))

    assert report.empty
    assert list(report.columns) == QUALITY_REPORT_COLUMNS
    assert report["Count"].dtype == np.dtype("int64")
    assert len(notes) == 1 and "no rated issue groups" in notes[0].lower()


def test_quality_report_when_no_score_could_be_calculated():
    report, notes = build_quality_report(make_score_result([], overall_score=None))

    assert report.empty
    assert list(report.columns) == QUALITY_REPORT_COLUMNS
    assert "no quality score could be calculated" in notes[0].lower()


def test_quality_report_accepts_a_summary_frame_without_columns():
    result = make_score_result(severity_summary=pd.DataFrame())
    report, notes = build_quality_report(result)
    assert report.empty and list(report.columns) == QUALITY_REPORT_COLUMNS
    assert notes


def test_quality_report_missing_required_column_is_rejected():
    summary = pd.DataFrame(SAMPLE_ROWS, columns=SEVERITY_COLUMNS).drop(columns=["Severity"])
    with pytest.raises(ValueError, match="missing required columns"):
        build_quality_report(make_score_result(severity_summary=summary))


def test_quality_report_extra_columns_are_ignored():
    summary = pd.DataFrame(SAMPLE_ROWS, columns=SEVERITY_COLUMNS)
    summary["Note"] = "x"
    report, _ = build_quality_report(make_score_result(severity_summary=summary))
    assert list(report.columns) == QUALITY_REPORT_COLUMNS


def test_quality_report_non_numeric_or_missing_counts_are_rejected():
    for column, bad in (("Affected", "many"), ("Share (%)", None), ("Affected", np.nan)):
        summary = pd.DataFrame(SAMPLE_ROWS, columns=SEVERITY_COLUMNS)
        summary[column] = summary[column].astype(object)
        summary.loc[0, column] = bad
        with pytest.raises(ValueError, match="non-numeric"):
            build_quality_report(make_score_result(severity_summary=summary))


def test_quality_report_invalid_input_types_are_rejected():
    for bad in (None, {"severity_summary": pd.DataFrame()}, pd.DataFrame(), [1, 2]):
        with pytest.raises(TypeError, match="QualityScoreResult"):
            build_quality_report(bad)

    with pytest.raises(TypeError, match="severity_summary"):
        build_quality_report(make_score_result(severity_summary=[("a", "b")]))


# ----------------------------------------------------------------------
# Anomaly report: mapping
# ----------------------------------------------------------------------
def test_anomaly_report_has_one_row_per_record_with_all_fields():
    sources = sample_sources()
    report, notes = build_anomaly_report(sources)

    assert notes == []
    assert list(report.columns) == ANOMALY_REPORT_COLUMNS
    assert len(report) == 4
    assert report["Record ID"].tolist() == [101, 102, 103, 104]
    assert report["Isolation Forest Result"].tolist() == [
        NORMAL, POTENTIAL_ANOMALY, POTENTIAL_ANOMALY, NORMAL]
    assert report["LOF Result"].tolist() == [
        NORMAL, POTENTIAL_ANOMALY, NORMAL, POTENTIAL_ANOMALY]
    assert report["Status"].tolist() == [
        NORMAL, HIGH_CONFIDENCE, NEEDS_REVIEW, NEEDS_REVIEW]


def test_anomaly_report_uses_the_real_scores_of_each_model_separately():
    sources = sample_sources()
    report, _ = build_anomaly_report(sources)

    assert report["Isolation Forest Score"].tolist() == pytest.approx(
        sources["isolation_forest"]["if_score"].tolist())
    assert report["LOF Score"].tolist() == pytest.approx(
        sources["lof"]["lof_score"].tolist())
    # No combined "Anomaly Score" is invented.
    assert "Anomaly Score" not in report.columns


def test_anomaly_report_keeps_normal_records_and_flags_nothing_itself():
    sources = make_sources(pd.Index(["a", "b", "c"]), [NORMAL] * 3, [NORMAL] * 3)
    report, _ = build_anomaly_report(sources)
    assert report["Status"].tolist() == [NORMAL, NORMAL, NORMAL]
    assert report["Record ID"].tolist() == ["a", "b", "c"]


def test_anomaly_report_all_records_flagged():
    sources = make_sources(
        pd.Index([1, 2]), [POTENTIAL_ANOMALY] * 2, [POTENTIAL_ANOMALY] * 2)
    report, _ = build_anomaly_report(sources)
    assert report["Status"].tolist() == [HIGH_CONFIDENCE, HIGH_CONFIDENCE]


def test_anomaly_report_preserves_record_ids_and_their_types():
    for labels in ([30, 10, 20], ["x-3", "x-1", "x-2"], [0.5, 1.5, 2.5]):
        sources = make_sources(
            pd.Index(labels), [NORMAL, POTENTIAL_ANOMALY, NORMAL], [NORMAL] * 3)
        report, _ = build_anomaly_report(sources)
        assert report["Record ID"].tolist() == labels          # same order, not sorted
        assert [type(v) for v in report["Record ID"]] == [type(v) for v in labels]


def test_anomaly_report_has_a_fresh_index_and_record_id_carries_the_label():
    report, _ = build_anomaly_report(sample_sources())
    assert report.index.tolist() == [0, 1, 2, 3]
    assert report["Record ID"].tolist() == [101, 102, 103, 104]


def test_anomaly_report_ignores_other_keys_and_runs_no_model():
    sources = sample_sources()
    sources["something_else"] = "ignored"
    report, _ = build_anomaly_report(sources)
    assert len(report) == 4


def test_report_builder_only_reads_results_and_stays_separate_from_ml_evaluation():
    tree = ast.parse(Path(report_builder.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module)
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)

    assert "modules.ml_evaluation" not in imported
    for runner in ("run_isolation_forest", "run_lof", "run_anomaly_pipeline",
                   "run_anomaly_sources", "run_ml_evaluation", "calculate_quality_score"):
        assert runner not in imported


# ----------------------------------------------------------------------
# Anomaly report: missing and optional results
# ----------------------------------------------------------------------
def test_anomaly_report_without_score_sources_leaves_scores_blank():
    sources = make_sources(
        pd.Index([1, 2]), [NORMAL, POTENTIAL_ANOMALY], [NORMAL, NORMAL], with_scores=False)
    report, notes = build_anomaly_report(sources)

    assert report["Status"].tolist() == [NORMAL, NEEDS_REVIEW]
    assert report["Isolation Forest Score"].isna().all()
    assert report["LOF Score"].isna().all()
    assert any("Isolation Forest scores are not available" in n for n in notes)
    assert any("LOF scores are not available" in n for n in notes)


def test_anomaly_report_with_only_one_score_source():
    sources = sample_sources()
    del sources["lof"]
    report, notes = build_anomaly_report(sources)

    assert report["Isolation Forest Score"].notna().all()
    assert report["LOF Score"].isna().all()
    assert len(notes) == 1 and notes[0].startswith("LOF")


def test_anomaly_report_score_frame_without_score_column_is_blank_with_a_note():
    sources = sample_sources()
    sources["isolation_forest"] = sources["isolation_forest"].drop(columns=["if_score"])
    report, notes = build_anomaly_report(sources)

    assert report["Isolation Forest Score"].isna().all()
    assert report["LOF Score"].notna().all()
    assert any("if_score" in n for n in notes)


def test_anomaly_report_misaligned_score_frame_is_not_used():
    sources = sample_sources()
    # Same labels in a different order: row i would be matched to the wrong record.
    sources["lof"] = sources["lof"].iloc[::-1]
    report, notes = build_anomaly_report(sources)

    assert report["LOF Score"].isna().all()
    assert any("do not line up" in n for n in notes)
    assert report["Record ID"].tolist() == [101, 102, 103, 104]


def test_anomaly_report_score_frame_of_another_dataset_is_not_used():
    sources = sample_sources()
    sources["isolation_forest"] = sources["isolation_forest"].set_axis([1, 2, 3, 4])
    report, notes = build_anomaly_report(sources)
    assert report["Isolation Forest Score"].isna().all()
    assert any("do not line up" in n for n in notes)


def test_anomaly_report_non_numeric_scores_are_blank_with_a_note():
    sources = sample_sources()
    sources["lof"] = sources["lof"].astype({"lof_score": object})
    sources["lof"].loc[101, "lof_score"] = "high"
    report, notes = build_anomaly_report(sources)
    assert report["LOF Score"].isna().all()
    assert any("not numeric" in n for n in notes)


def test_anomaly_report_score_value_that_is_missing_stays_blank():
    sources = sample_sources()
    sources["lof"].loc[102, "lof_score"] = np.nan
    report, notes = build_anomaly_report(sources)
    assert report["LOF Score"].isna().tolist() == [False, True, False, False]
    assert notes == []


def test_anomaly_report_requires_the_combined_results():
    for sources in ({}, {"isolation_forest": pd.DataFrame()},
                    {"anomaly_analysis": None}, {"anomaly_analysis": "x"}):
        with pytest.raises(ValueError, match="not available"):
            build_anomaly_report(sources)


def test_anomaly_report_invalid_input_types_are_rejected():
    for bad in (None, [], "sources", pd.DataFrame()):
        with pytest.raises(TypeError, match="dictionary"):
            build_anomaly_report(bad)


def test_anomaly_report_missing_required_columns_are_rejected():
    for column in ("IF Result", "LOF Result", "Status"):
        sources = sample_sources()
        sources["anomaly_analysis"] = sources["anomaly_analysis"].drop(columns=[column])
        with pytest.raises(ValueError, match="missing required columns"):
            build_anomaly_report(sources)


def test_anomaly_report_duplicate_record_labels_are_rejected():
    sources = make_sources(pd.Index([1, 1, 2]), [NORMAL] * 3, [NORMAL] * 3)
    with pytest.raises(ValueError, match="duplicate record labels"):
        build_anomaly_report(sources)


def test_anomaly_report_with_no_records_is_header_only():
    empty = pd.DataFrame({"IF Result": [], "LOF Result": [], "Status": []})
    report, notes = build_anomaly_report({"anomaly_analysis": empty})

    assert report.empty
    assert list(report.columns) == ANOMALY_REPORT_COLUMNS
    assert len(notes) == 1 and "no records" in notes[0].lower()


def test_anomaly_report_with_a_completely_empty_frame_is_header_only():
    report, notes = build_anomaly_report({"anomaly_analysis": pd.DataFrame()})
    assert report.empty and list(report.columns) == ANOMALY_REPORT_COLUMNS
    assert notes


def test_anomaly_report_does_not_modify_the_sources():
    sources = sample_sources()
    before = {key: frame.copy(deep=True) for key, frame in sources.items()}
    keys_before = list(sources)

    build_anomaly_report(sources)

    assert list(sources) == keys_before
    for key, frame in sources.items():
        pd.testing.assert_frame_equal(frame, before[key])


def test_anomaly_report_shares_no_data_with_the_sources():
    sources = sample_sources()
    before = {key: frame.copy(deep=True) for key, frame in sources.items()}
    report, _ = build_anomaly_report(sources)

    for column in ANOMALY_REPORT_COLUMNS:
        report.loc[0, column] = 12345 if "Score" in column or column == "Record ID" else "X"

    for key, frame in sources.items():
        pd.testing.assert_frame_equal(frame, before[key])


# ----------------------------------------------------------------------
# CSV output
# ----------------------------------------------------------------------
def test_quality_report_csv_header_and_round_trip():
    report, _ = build_quality_report(make_score_result(SAMPLE_ROWS))
    text, back = csv_round_trip(report)

    assert text.splitlines()[0] == "Issue,Column,Count,Percentage,Severity"
    assert back["Issue"].tolist() == report["Issue"].tolist()
    assert back["Count"].tolist() == ["2", "1", "1", "1"]
    assert back["Percentage"].tolist() == ["5.13", "2.5", "2.5", "2.5"]


def test_quality_report_csv_survives_awkward_column_names():
    rows = [("Validity", "Invalid Value", 'Price, "USD"', "High", 1, 10, 10.0)]
    report, _ = build_quality_report(make_score_result(rows))
    _, back = csv_round_trip(report)
    assert back["Column"].tolist() == ['Price, "USD"']


def test_empty_quality_report_csv_is_just_the_header():
    report, _ = build_quality_report(make_score_result([]))
    assert csv_text(report) == "Issue,Column,Count,Percentage,Severity\n"


def test_anomaly_report_csv_header_and_round_trip():
    report, _ = build_anomaly_report(sample_sources())
    text, back = csv_round_trip(report)

    assert text.splitlines()[0] == (
        "Record ID,Isolation Forest Result,LOF Result,"
        "Isolation Forest Score,LOF Score,Status"
    )
    assert back["Record ID"].tolist() == ["101", "102", "103", "104"]
    assert back["Status"].tolist() == [NORMAL, HIGH_CONFIDENCE, NEEDS_REVIEW, NEEDS_REVIEW]
    assert len(back) == 4


def test_anomaly_report_csv_blank_scores_are_empty_cells():
    sources = make_sources(pd.Index([1]), [NORMAL], [NORMAL], with_scores=False)
    report, _ = build_anomaly_report(sources)
    assert report.to_csv(index=False).splitlines()[1] == "1,Normal,Normal,,,Normal"


def test_anomaly_report_csv_survives_awkward_record_ids():
    sources = make_sources(
        pd.Index(['id, "1"', "id-2"]), [NORMAL, NORMAL], [NORMAL, NORMAL])
    report, _ = build_anomaly_report(sources)
    _, back = csv_round_trip(report)
    assert back["Record ID"].tolist() == ['id, "1"', "id-2"]


def test_empty_anomaly_report_csv_is_just_the_header():
    report, _ = build_anomaly_report({"anomaly_analysis": pd.DataFrame()})
    assert csv_text(report) == (
    "Record ID,Isolation Forest Result,LOF Result,"
    "Isolation Forest Score,LOF Score,Status\n"
)


# ----------------------------------------------------------------------
# Integration with the REAL detectors, score and anomaly sources
# ----------------------------------------------------------------------
def build_real_dataset():
    rng = np.random.RandomState(1)
    n = 40
    df = pd.DataFrame({
        "Age": rng.randint(20, 60, n).astype(object),
        "Email": [f"user{i}@example.com" for i in range(n)],
        "City": rng.choice(["Pune", "Mumbai"], n),
        "Income": rng.normal(50000, 2000, n),
    }, index=pd.Index(range(500, 500 + n), name="customer"))
    df.loc[505, "Age"] = 150            # invalid (outside 0-120)
    df.loc[506, "Age"] = "Twenty"       # not a number
    df.loc[507, "Email"] = "bad@"       # bad email format
    df.loc[508, "City"] = "pune"        # inconsistent spelling
    df.loc[509, "Age"] = None           # missing
    df.loc[510] = df.loc[511]           # exact duplicate row
    df.loc[520, "Income"] = 900000.0    # extreme record
    return df


def run_real_quality_checks(df):
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


def test_integration_quality_report_matches_the_real_score():
    from modules.unified_dashboard import build_unified_table

    df = build_real_dataset()
    before = df.copy(deep=True)
    quality_results = run_real_quality_checks(df)
    unified, _ = build_unified_table(df, quality_results, None)
    score = calculate_quality_score(df, quality_results, unified)

    report, notes = build_quality_report(score)

    pd.testing.assert_frame_equal(df, before)
    assert notes == []
    assert list(report.columns) == QUALITY_REPORT_COLUMNS
    assert len(report) == len(score.severity_summary) > 0

    # Row for row the same groups, in the same order, with the same numbers.
    summary = score.severity_summary
    assert report["Issue"].tolist() == summary["Issue Type"].tolist()
    assert report["Column"].tolist() == summary["Column"].tolist()
    assert report["Count"].tolist() == summary["Affected"].tolist()
    assert report["Percentage"].tolist() == summary["Share (%)"].tolist()
    assert report["Severity"].tolist() == summary["Severity"].tolist()

    # Known values for this dataset.
    by_group = {(r.Issue, r.Column): r for r in report.itertuples()}
    missing_age = by_group[("Missing Value", "Age")]
    assert missing_age.Count == 1 and missing_age.Percentage == 2.5
    duplicate = by_group[("Duplicate Record", "(whole row)")]
    assert duplicate.Count == 1 and duplicate.Percentage == 2.5
    assert ("Format Problem", "Email") in by_group
    assert ("Inconsistent Representation", "City") in by_group

    # IQR outliers are review items, not quality errors: not in this report.
    assert "IQR Outlier" not in report["Issue"].tolist()
    assert not report["Issue"].str.contains("Anomaly|Outlier").any()
    assert not (report["Severity"] == "Review (not rated)").any()


def test_integration_quality_report_for_a_clean_dataset_is_header_only():
    df = pd.DataFrame({"Age": [20, 30, 40, 50], "City": ["A", "B", "C", "D"]})
    quality_results = run_real_quality_checks(df)
    score = calculate_quality_score(df, quality_results)

    report, notes = build_quality_report(score)

    assert report.empty
    assert list(report.columns) == QUALITY_REPORT_COLUMNS
    assert notes


def test_integration_anomaly_report_matches_the_real_pipeline():
    from modules.anomaly_pipeline import run_anomaly_pipeline
    from modules.unified_dashboard import run_anomaly_sources

    df = build_real_dataset()
    before = df.copy(deep=True)
    sources, source_notes = run_anomaly_sources(df)
    assert source_notes == [] and sources

    report, notes = build_anomaly_report(sources)

    pd.testing.assert_frame_equal(df, before)
    assert notes == []
    assert list(report.columns) == ANOMALY_REPORT_COLUMNS

    # One row per uploaded record, with the original index labels, in order.
    assert len(report) == len(df)
    assert report["Record ID"].tolist() == df.index.tolist()

    # Same results as the (separately run) pipeline's dashboard table and scores.
    pipeline = run_anomaly_pipeline(df)
    assert report["Isolation Forest Result"].tolist() == pipeline.table["IF Result"].tolist()
    assert report["LOF Result"].tolist() == pipeline.table["LOF Result"].tolist()
    assert report["Status"].tolist() == pipeline.table["Status"].tolist()
    assert report["Isolation Forest Score"].tolist() == pytest.approx(
        pipeline.scores["if_score"].tolist())
    assert report["LOF Score"].tolist() == pytest.approx(
        pipeline.scores["lof_score"].tolist())

    # Counts agree with the pipeline summary; the extreme record is flagged.
    summary = pipeline.summary
    assert int((report["Isolation Forest Result"] == POTENTIAL_ANOMALY).sum()) == summary["if_anomalies"]
    assert int((report["LOF Result"] == POTENTIAL_ANOMALY).sum()) == summary["lof_anomalies"]
    assert report.loc[report["Record ID"] == 520, "Status"].iloc[0] == HIGH_CONFIDENCE

    # Nothing is removed: Normal records are still listed.
    assert (report["Status"] == NORMAL).any()
    assert not report.isna().any().any()


def test_integration_anomaly_sources_that_did_not_run_cannot_be_reported():
    from modules.unified_dashboard import run_anomaly_sources

    empty_sources, notes = run_anomaly_sources(pd.DataFrame())
    assert empty_sources == {} and notes

    with pytest.raises(ValueError, match="not available"):
        build_anomaly_report(empty_sources)

    duplicate_index = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=[1, 1, 2])
    sources, notes = run_anomaly_sources(duplicate_index)
    assert sources == {} and notes
    with pytest.raises(ValueError, match="not available"):
        build_anomaly_report(sources)


def test_integration_deep_copy_of_sources_gives_the_same_report():
    from modules.unified_dashboard import run_anomaly_sources

    sources, _ = run_anomaly_sources(build_real_dataset())
    first, _ = build_anomaly_report(sources)
    second, _ = build_anomaly_report(copy.deepcopy(sources))
    pd.testing.assert_frame_equal(first, second)
