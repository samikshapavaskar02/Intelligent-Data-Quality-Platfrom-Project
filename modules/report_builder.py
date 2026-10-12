"""Downloadable CSV reports for the UPLOADED dataset (PDF Phase 15, Steps 30-31).

Builds the two report tables the app offers for download:

* ``quality_report.csv``  - ``Issue | Column | Count | Percentage | Severity``
* ``anomaly_report.csv``  - ``Record ID | Isolation Forest Result | LOF Result |
  Isolation Forest Score | LOF Score | Status``

Both are only a re-arrangement of results the app already has. Nothing is
recalculated: no detector, score, severity rule or model is run here, and the
inputs are never modified. There is no Streamlit code in this module.

Quality report
--------------
Source: ``QualityScoreResult.severity_summary`` from
``modules.quality_score`` (one row per rated issue group, worst first).

=============  ==============================================================
Report column  Taken from ``severity_summary``
=============  ==============================================================
Issue          ``Issue Type`` (for example "Missing Value")
Column         ``Column`` ("(whole row)" for duplicate records)
Count          ``Affected`` - the number counted by the score for that group
               (after the score's overlap rules; for duplicates the redundant
               copies, for inconsistent values the minority spellings)
Percentage     ``Share (%)`` - Count / ``Total`` x 100, where ``Total`` is the
               number of units checked for that group (rows, or non-blank
               cells of that column)
Severity       ``Severity`` exactly as classified by the score
=============  ==============================================================

Not in this report: IQR outliers and ML anomalies (the score labels them
"Review (not rated)"; they are unusual records, not data quality errors) and
findings the score could not locate on an exact row. The Unified Results
table lists all of those.

Anomaly report
--------------
Source: the dictionary returned by ``modules.unified_dashboard.run_anomaly_sources``
(already computed and cached by the app).

* One row per record of the uploaded dataset, in dataset order. Normal
  records are included. Nothing is removed or filtered.
* ``Record ID`` is the dataset's index label, exactly as in the dashboard.
* ``Isolation Forest Result``, ``LOF Result`` and ``Status`` come from the
  combined anomaly analysis (``IF Result``, ``LOF Result``, ``Status``).
* There is NO single "Anomaly Score": Isolation Forest and LOF scores are on
  different scales and must not be averaged. The report therefore carries
  the two real scores in separate columns (higher = more unusual for both).
  A score column is left blank (with a note) when its source is missing.
* A record flagged here is a record to review, not automatically an error.

This is unrelated to ``modules.ml_evaluation``, which works on a synthetic
dataset and must never feed these reports.
"""

import pandas as pd

from models.isolation_forest import COL_SCORE as IF_SCORE_COL
from models.lof import COL_SCORE as LOF_SCORE_COL
from modules.anomaly_analysis import IF_RESULT_COL, LOF_RESULT_COL, STATUS_COL
from modules.quality_score import QualityScoreResult
from modules.unified_results import SRC_ANOMALY, SRC_IF, SRC_LOF

# ----------------------------------------------------------------------
# Report formats
# ----------------------------------------------------------------------
QUALITY_REPORT_COLUMNS = ["Issue", "Column", "Count", "Percentage", "Severity"]

ANOMALY_REPORT_COLUMNS = [
    "Record ID",
    "Isolation Forest Result",
    "LOF Result",
    "Isolation Forest Score",
    "LOF Score",
    "Status",
]

# report column -> column of QualityScoreResult.severity_summary
_QUALITY_SOURCE_COLUMNS = {
    "Issue": "Issue Type",
    "Column": "Column",
    "Count": "Affected",
    "Percentage": "Share (%)",
    "Severity": "Severity",
}

_ANOMALY_REQUIRED_COLUMNS = [IF_RESULT_COL, LOF_RESULT_COL, STATUS_COL]


# ----------------------------------------------------------------------
# Quality report
# ----------------------------------------------------------------------
def build_quality_report(score_result):
    """Build the quality report from a data quality score result.

    Parameters
    ----------
    score_result : modules.quality_score.QualityScoreResult
        Result of ``calculate_quality_score``. Not modified.

    Returns
    -------
    (report, notes) : (pandas.DataFrame, list[str])
        ``report`` has the columns ``QUALITY_REPORT_COLUMNS``, one row per
        rated issue group in the score's own order (worst first), with a
        fresh 0..n-1 index. It has no rows when the score has no rated issue
        groups; ``notes`` then explains why. ``notes`` is empty otherwise.

    Raises
    ------
    TypeError
        If ``score_result`` is not a ``QualityScoreResult`` or its
        ``severity_summary`` is not a DataFrame.
    ValueError
        If ``severity_summary`` lacks a required column or has a missing or
        non-numeric count or percentage.
    """
    if not isinstance(score_result, QualityScoreResult):
        raise TypeError(
            "score_result must be a QualityScoreResult, "
            f"got {type(score_result).__name__}."
        )
    summary = score_result.severity_summary
    if not isinstance(summary, pd.DataFrame):
        raise TypeError(
            "score_result.severity_summary must be a pandas DataFrame, "
            f"got {type(summary).__name__}."
        )

    if summary.empty:
        if score_result.overall_score is None:
            note = (
                "No quality score could be calculated, so there are no rated "
                "issue groups to report. The report contains only the column headers."
            )
        else:
            note = (
                "No rated issue groups were found, so the report contains "
                "only the column headers."
            )
        return _empty_quality_report(), [note]

    missing = [c for c in _QUALITY_SOURCE_COLUMNS.values() if c not in summary.columns]
    if missing:
        raise ValueError(
            f"severity_summary is missing required columns {missing}. "
            f"Available columns: {list(summary.columns)}."
        )

    try:
        count = pd.to_numeric(summary["Affected"])
        percentage = pd.to_numeric(summary["Share (%)"])
        if count.isna().any() or percentage.isna().any():
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(
            "severity_summary has a missing or non-numeric 'Affected' or "
            "'Share (%)' value."
        )

    # Every column is copied explicitly: the report must never share memory
    # with the score result (editing the report must not change the score).
    report = pd.DataFrame({
        "Issue": summary["Issue Type"].to_numpy(dtype=object, copy=True),
        "Column": summary["Column"].to_numpy(dtype=object, copy=True),
        "Count": count.to_numpy(dtype="int64", copy=True),
        "Percentage": percentage.to_numpy(dtype="float64", copy=True),
        "Severity": summary["Severity"].to_numpy(dtype=object, copy=True),
    }, columns=QUALITY_REPORT_COLUMNS)
    return report, []


def _empty_quality_report():
    return pd.DataFrame({
        "Issue": pd.Series(dtype=object),
        "Column": pd.Series(dtype=object),
        "Count": pd.Series(dtype="int64"),
        "Percentage": pd.Series(dtype="float64"),
        "Severity": pd.Series(dtype=object),
    }, columns=QUALITY_REPORT_COLUMNS)


# ----------------------------------------------------------------------
# Anomaly report
# ----------------------------------------------------------------------
def build_anomaly_report(anomaly_sources):
    """Build the anomaly report from the already computed anomaly results.

    Parameters
    ----------
    anomaly_sources : dict
        As returned by ``run_anomaly_sources``: ``"anomaly_analysis"``
        (``IF Result | LOF Result | Status``, indexed by record) is required;
        ``"isolation_forest"`` (with ``if_score``) and ``"lof"`` (with
        ``lof_score``) are optional and only supply the score columns. They
        must have exactly the index of ``anomaly_analysis`` to be used.
        Other keys are ignored. No model is run.

    Returns
    -------
    (report, notes) : (pandas.DataFrame, list[str])
        ``report`` has the columns ``ANOMALY_REPORT_COLUMNS``, one row per
        record of ``anomaly_analysis`` in the same order (a fresh 0..n-1
        index; the index labels are in ``Record ID``). A score column that
        cannot be filled is blank and ``notes`` says why. ``notes`` is empty
        when everything was available.

    Raises
    ------
    TypeError
        If ``anomaly_sources`` is not a dict.
    ValueError
        If the combined analysis is missing (anomaly detection did not run),
        lacks a required column, or has duplicate record labels.
    """
    if not isinstance(anomaly_sources, dict):
        raise TypeError(
            "anomaly_sources must be a dictionary, "
            f"got {type(anomaly_sources).__name__}."
        )

    analysis = anomaly_sources.get(SRC_ANOMALY)
    if not isinstance(analysis, pd.DataFrame):
        raise ValueError(
            "The combined anomaly results are not available, so no anomaly "
            "report can be built (anomaly detection did not run)."
        )

    if len(analysis) == 0:
        return _empty_anomaly_report(), [
            "There are no records in the anomaly results, so the report "
            "contains only the column headers."
        ]

    missing = [c for c in _ANOMALY_REQUIRED_COLUMNS if c not in analysis.columns]
    if missing:
        raise ValueError(
            f"The combined anomaly results are missing required columns {missing}. "
            f"Available columns: {list(analysis.columns)}."
        )
    if analysis.index.has_duplicates:
        raise ValueError(
            "The combined anomaly results have duplicate record labels, so "
            "records cannot be identified uniquely."
        )

    notes = []
    if_scores = _scores(anomaly_sources.get(SRC_IF), IF_SCORE_COL, analysis.index,
                        "Isolation Forest", notes)
    lof_scores = _scores(anomaly_sources.get(SRC_LOF), LOF_SCORE_COL, analysis.index,
                         "LOF", notes)

    # Every column is copied explicitly: the report must never share memory
    # with the anomaly results (editing the report must not change them).
    report = pd.DataFrame({
        "Record ID": pd.Series(analysis.index.to_numpy(dtype=object, copy=True), dtype=object),
        "Isolation Forest Result": analysis[IF_RESULT_COL].to_numpy(dtype=object, copy=True),
        "LOF Result": analysis[LOF_RESULT_COL].to_numpy(dtype=object, copy=True),
        "Isolation Forest Score": if_scores,
        "LOF Score": lof_scores,
        "Status": analysis[STATUS_COL].to_numpy(dtype=object, copy=True),
    }, columns=ANOMALY_REPORT_COLUMNS)
    return report, notes


def _scores(frame, score_column, analysis_index, label, notes):
    """Score values aligned to the analysis, or all-NaN with a note."""
    blank = pd.Series(float("nan"), index=range(len(analysis_index)), dtype="float64")

    if frame is None:
        notes.append(f"{label} scores are not available, so that column is blank.")
        return blank.to_numpy()
    if not isinstance(frame, pd.DataFrame) or score_column not in frame.columns:
        notes.append(
            f"{label} scores could not be read (no '{score_column}' column), "
            "so that column is blank."
        )
        return blank.to_numpy()
    if not frame.index.equals(analysis_index):
        notes.append(
            f"{label} scores do not line up with the records of the combined "
            "results, so that column is blank."
        )
        return blank.to_numpy()
    try:
        return pd.to_numeric(frame[score_column]).to_numpy(dtype="float64", copy=True)
    except (TypeError, ValueError):
        notes.append(f"{label} scores are not numeric, so that column is blank.")
        return blank.to_numpy()


def _empty_anomaly_report():
    return pd.DataFrame({
        "Record ID": pd.Series(dtype=object),
        "Isolation Forest Result": pd.Series(dtype=object),
        "LOF Result": pd.Series(dtype=object),
        "Isolation Forest Score": pd.Series(dtype="float64"),
        "LOF Score": pd.Series(dtype="float64"),
        "Status": pd.Series(dtype=object),
    }, columns=ANOMALY_REPORT_COLUMNS)
