"""Anomaly analysis: combine Isolation Forest and LOF results.

This module compares the per-record results of the Isolation Forest and
LOF detectors and produces one unified interpretation per record.

Decision logic (from the project PDF):

    Isolation Forest   LOF                 -> Status
    -----------------  ------------------  -----------------------------------
    Normal             Normal              -> Normal
    Potential Anomaly  Potential Anomaly   -> High-confidence suspicious record
    Potential Anomaly  Normal              -> Needs Review
    Normal             Potential Anomaly   -> Needs Review

An anomaly is NOT automatically a data-quality error and is NEVER deleted
here. This module only labels records so they can be reviewed later.
It does not clean, impute, preprocess, or modify its inputs.
"""

import pandas as pd

# Project terminology
NORMAL = "Normal"
POTENTIAL_ANOMALY = "Potential Anomaly"
HIGH_CONFIDENCE = "High-confidence suspicious record"
NEEDS_REVIEW = "Needs Review"

# Output column names (match the dashboard table in the project PDF)
IF_RESULT_COL = "IF Result"
LOF_RESULT_COL = "LOF Result"
STATUS_COL = "Status"

# Default names of the label columns expected in the detector outputs.
DEFAULT_IF_COLUMN = "if_status"
DEFAULT_LOF_COLUMN = "lof_status"

# "Anomaly" is the wording used in the PDF's example table; it is treated
# as the same thing as "Potential Anomaly".
_LABEL_ALIASES = {
    NORMAL: NORMAL,
    POTENTIAL_ANOMALY: POTENTIAL_ANOMALY,
    "Anomaly": POTENTIAL_ANOMALY,
}


def _validate_frame(df, name, column):
    """Validate one detector result DataFrame (does not modify it)."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"{name} must be a pandas DataFrame, got {type(df).__name__}."
        )
    if df.empty:
        raise ValueError(f"{name} is empty.")
    if column not in df.columns:
        raise ValueError(
            f"{name} is missing required column '{column}'. "
            f"Available columns: {list(df.columns)}."
        )
    if df.index.has_duplicates:
        raise ValueError(f"{name} has duplicate index values.")


def _normalise_labels(series, name):
    """Map labels to project terminology; reject unknown or missing labels."""
    mapped = series.map(_LABEL_ALIASES)
    invalid = mapped.isna()
    if invalid.any():
        bad = series[invalid].unique().tolist()
        raise ValueError(
            f"{name} contains invalid labels {bad}. "
            f"Allowed labels: {sorted(_LABEL_ALIASES)}."
        )
    return mapped


def _combine(if_label, lof_label):
    """Apply the decision logic to one pair of labels."""
    if if_label == POTENTIAL_ANOMALY and lof_label == POTENTIAL_ANOMALY:
        return HIGH_CONFIDENCE
    if if_label == NORMAL and lof_label == NORMAL:
        return NORMAL
    return NEEDS_REVIEW


def combine_anomaly_results(
    if_results,
    lof_results,
    if_column=DEFAULT_IF_COLUMN,
    lof_column=DEFAULT_LOF_COLUMN,
):
    """Combine Isolation Forest and LOF results into one table.

    Parameters
    ----------
    if_results : pandas.DataFrame
        Isolation Forest output, one row per record, containing `if_column`.
    lof_results : pandas.DataFrame
        LOF output, one row per record, containing `lof_column`.
    if_column, lof_column : str
        Names of the label columns ("Normal" / "Potential Anomaly").
        "Anomaly" is also accepted and treated as "Potential Anomaly".

    Returns
    -------
    pandas.DataFrame
        New DataFrame, one row per record, with the original index and
        columns "IF Result", "LOF Result" and "Status". Inputs are not
        modified.

    Raises
    ------
    TypeError
        If an input is not a DataFrame.
    ValueError
        If an input is empty, lacks its label column, has duplicate index
        values, has invalid labels, or the two inputs do not describe the
        same records (different row counts or different indexes).
    """
    _validate_frame(if_results, "if_results", if_column)
    _validate_frame(lof_results, "lof_results", lof_column)

    if len(if_results) != len(lof_results):
        raise ValueError(
            "if_results and lof_results have different row counts "
            f"({len(if_results)} vs {len(lof_results)})."
        )
    if set(if_results.index) != set(lof_results.index):
        raise ValueError(
            "if_results and lof_results have mismatched indexes."
        )

    if_labels = _normalise_labels(if_results[if_column], "if_results")
    lof_labels = _normalise_labels(lof_results[lof_column], "lof_results")

    # Indexes are unique and contain the same records, so aligning by label
    # is safe. The Isolation Forest row order is kept.
    lof_labels = lof_labels.reindex(if_labels.index)

    status = [_combine(a, b) for a, b in zip(if_labels, lof_labels)]

    return pd.DataFrame(
        {
            IF_RESULT_COL: if_labels.to_numpy(),
            LOF_RESULT_COL: lof_labels.to_numpy(),
            STATUS_COL: status,
        },
        index=if_results.index.copy(),
    )
