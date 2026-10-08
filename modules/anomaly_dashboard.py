"""Dashboard data preparation for the Anomaly Detection page.

Prepares the numbers and the table the dashboard needs (project PDF,
Step 27): total records, anomalies found by Isolation Forest, anomalies
found by LOF, common anomalies, and a
"Record ID | IF Result | LOF Result | Status" table.

Input is the DataFrame returned by
``modules.anomaly_analysis.combine_anomaly_results`` (columns
"IF Result", "LOF Result", "Status"). No detection logic is repeated here,
no models are run, and there is no UI code: the inputs are never modified.
"""

import pandas as pd

from modules.anomaly_analysis import (
    IF_RESULT_COL,
    LOF_RESULT_COL,
    NORMAL,
    POTENTIAL_ANOMALY,
    STATUS_COL,
)

RECORD_ID_COL = "Record ID"

_REQUIRED_COLUMNS = [IF_RESULT_COL, LOF_RESULT_COL, STATUS_COL]
_VALID_LABELS = {NORMAL, POTENTIAL_ANOMALY}


def _validate(analysis_df):
    """Validate the anomaly analysis result (does not modify it)."""
    if not isinstance(analysis_df, pd.DataFrame):
        raise TypeError(
            "analysis_df must be a pandas DataFrame, "
            f"got {type(analysis_df).__name__}."
        )
    if analysis_df.empty:
        raise ValueError("analysis_df is empty.")
    missing = [c for c in _REQUIRED_COLUMNS if c not in analysis_df.columns]
    if missing:
        raise ValueError(
            f"analysis_df is missing required columns {missing}. "
            f"Available columns: {list(analysis_df.columns)}."
        )
    if analysis_df.index.has_duplicates:
        raise ValueError("analysis_df has duplicate index values.")
    for col in (IF_RESULT_COL, LOF_RESULT_COL):
        bad = ~analysis_df[col].isin(_VALID_LABELS)
        if bad.any():
            raise ValueError(
                f"Column '{col}' contains invalid labels "
                f"{analysis_df[col][bad].unique().tolist()}. "
                f"Allowed labels: {sorted(_VALID_LABELS)}."
            )


def get_total_records(analysis_df):
    """Number of records analysed."""
    _validate(analysis_df)
    return int(len(analysis_df))


def get_if_anomaly_count(analysis_df):
    """Number of records Isolation Forest marked 'Potential Anomaly'."""
    _validate(analysis_df)
    return int((analysis_df[IF_RESULT_COL] == POTENTIAL_ANOMALY).sum())


def get_lof_anomaly_count(analysis_df):
    """Number of records LOF marked 'Potential Anomaly'."""
    _validate(analysis_df)
    return int((analysis_df[LOF_RESULT_COL] == POTENTIAL_ANOMALY).sum())


def get_common_anomaly_count(analysis_df):
    """Number of records flagged by BOTH Isolation Forest and LOF."""
    _validate(analysis_df)
    both = (analysis_df[IF_RESULT_COL] == POTENTIAL_ANOMALY) & (
        analysis_df[LOF_RESULT_COL] == POTENTIAL_ANOMALY
    )
    return int(both.sum())


def build_anomaly_table(analysis_df):
    """Dashboard-ready table: Record ID | IF Result | LOF Result | Status.

    Returns a new DataFrame with the original index preserved. "Record ID"
    holds the index values, so the table can be shown without the index.
    """
    _validate(analysis_df)
    table = analysis_df[_REQUIRED_COLUMNS].copy()
    table.insert(0, RECORD_ID_COL, analysis_df.index.to_numpy())
    return table


def get_anomaly_summary(analysis_df):
    """All dashboard numbers in one dict (plain Python ints)."""
    return {
        "total_records": get_total_records(analysis_df),
        "if_anomalies": get_if_anomaly_count(analysis_df),
        "lof_anomalies": get_lof_anomaly_count(analysis_df),
        "common_anomalies": get_common_anomaly_count(analysis_df),
    }
