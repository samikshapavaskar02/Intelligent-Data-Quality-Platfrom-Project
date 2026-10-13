"""Dashboard data preparation for the Unified Results section.

Connects the existing detection modules to ``modules.unified_results`` for
the Streamlit app (project PDF, Phase 11 Step 19 and Phase 14). Like
``anomaly_dashboard``, it contains NO UI code: ``app.py`` only displays what
this module prepares.

What it does
------------
* ``run_anomaly_sources``  - runs the existing ``run_anomaly_pipeline`` and
  returns the three ML sources the unified table understands
  (``isolation_forest``, ``lof``, ``anomaly_analysis``). When anomaly
  detection cannot run (empty data, duplicate index, no usable features, ...)
  it returns no sources and a note instead of raising.
* ``get_dataset_fingerprint`` - a text fingerprint of the WHOLE dataset, so
  the app can cache the (slow) ML results safely between reruns.
* ``build_unified_table``  - merges the quality results and the optional ML
  sources and calls ``combine_quality_results`` with the dataset's index, so
  ``Record ID`` and ``Row Position`` are filled exactly. One failing source
  never hides the others; it becomes a note.
* ``get_unified_summary`` / ``count_by_issue_type`` / ``filter_unified_table``
  / ``prepare_display_table`` - small helpers for the page.

What it does NOT do
-------------------
* It runs no detector itself (the app already has those results), changes no
  detector, cleans nothing and never deletes records.
* ML anomalies stay in their own category (``ML Anomaly``). They are records
  to review, never counted as data-quality errors, and records whose
  combined status is "Normal" are not listed.
* Findings are never de-duplicated by value: two identical-looking rows can
  be two different records (for example duplicate index labels), and the
  same cell can legitimately be reported by two different checks.
"""

import hashlib
import json
import math
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd

from modules.anomaly_pipeline import run_anomaly_pipeline
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
    SRC_LOF,
    SRC_MISSING,
    SUPPORTED_SOURCES,
    UNIFIED_COLUMNS,
    combine_quality_results,
)

# Source keys of the ML results produced together by the anomaly pipeline.
ML_SOURCES = (SRC_IF, SRC_LOF, SRC_ANOMALY)

# Readable names used in notes shown to the user.
SOURCE_LABELS = {
    SRC_MISSING: "Missing values",
    SRC_DUPLICATES: "Duplicates",
    SRC_INVALID: "Invalid values",
    SRC_DATA_TYPE: "Data type problems",
    SRC_FORMAT: "Format problems",
    SRC_CONSISTENCY: "Inconsistent values",
    SRC_IQR: "IQR outliers",
    SRC_IF: "Isolation Forest",
    SRC_LOF: "Local Outlier Factor (LOF)",
    SRC_ANOMALY: "Combined anomaly analysis",
}

_COUNT_COLUMNS = ["Category", "Issue Type", "Method", "Findings"]


# ----------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------
def _validate_dataset(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}.")


def _validate_unified(unified):
    """Check ``unified`` looks like the output of ``combine_quality_results``."""
    if not isinstance(unified, pd.DataFrame):
        raise TypeError(
            f"unified must be a pandas DataFrame, got {type(unified).__name__}."
        )
    missing = [c for c in UNIFIED_COLUMNS if c not in unified.columns]
    if missing:
        raise ValueError(
            f"unified is missing required columns {missing}. "
            f"Available columns: {list(unified.columns)}."
        )


# ----------------------------------------------------------------------
# ML sources
# ----------------------------------------------------------------------
def get_dataset_fingerprint(df):
    """Text fingerprint of the whole dataset: index, columns, dtypes and values.

    Used by the app as a cache key for ``run_anomaly_sources``. It reads every
    value (Streamlit's own DataFrame hashing only samples very large frames,
    so a few corrected cells could go unnoticed). Equal datasets give equal
    fingerprints; changing any value, label, column name or dtype changes it.
    """
    _validate_dataset(df)

    def encode(value):
        """Return a typed, JSON-safe representation of a dataset value."""
        if value is None:
            return ["none"]
        if value is pd.NA:
            return ["pandas.NA"]
        if value is pd.NaT:
            return ["pandas.NaT"]
        if isinstance(value, np.generic):
            return ["numpy", value.dtype.str, encode(value.item())]
        if isinstance(value, bool):
            return ["bool", value]
        if isinstance(value, int):
            return ["int", str(value)]
        if isinstance(value, float):
            if math.isnan(value):
                return ["float", "nan"]
            if math.isinf(value):
                return ["float", "+inf" if value > 0 else "-inf"]
            return ["float", value.hex()]
        if isinstance(value, str):
            return ["str", value]
        if isinstance(value, bytes):
            return ["bytes", value.hex()]
        if isinstance(value, Decimal):
            parts = value.as_tuple()
            return ["decimal", parts.sign, list(parts.digits), parts.exponent]
        if isinstance(value, pd.Timestamp):
            return ["pandas.Timestamp", value.isoformat(), str(value.tz)]
        if isinstance(value, pd.Timedelta):
            return ["pandas.Timedelta", value.value]
        if isinstance(value, datetime):
            return ["datetime", value.isoformat(), getattr(value.tzinfo, "key", None)]
        if isinstance(value, date):
            return ["date", value.isoformat()]
        if isinstance(value, time):
            return ["time", value.isoformat(), getattr(value.tzinfo, "key", None)]
        if isinstance(value, timedelta):
            return ["timedelta", value.days, value.seconds, value.microseconds]
        if isinstance(value, list):
            return ["list", [encode(item) for item in value]]
        if isinstance(value, tuple):
            return ["tuple", [encode(item) for item in value]]
        if isinstance(value, dict):
            items = [[encode(key), encode(item)] for key, item in value.items()]
            items.sort(key=lambda pair: json.dumps(
                pair[0], ensure_ascii=False, separators=(",", ":")
            ))
            return ["dict", items]
        if isinstance(value, set):
            items = [encode(item) for item in value]
            items.sort(key=lambda item: json.dumps(
                item, ensure_ascii=False, separators=(",", ":")
            ))
            return ["set", items]
        if isinstance(value, frozenset):
            items = [encode(item) for item in value]
            items.sort(key=lambda item: json.dumps(
                item, ensure_ascii=False, separators=(",", ":")
            ))
            return ["frozenset", items]
        raise TypeError(
            "Dataset fingerprint does not support values of type "
            f"{type(value).__module__}.{type(value).__qualname__}."
        )

    index_values = [encode(value) for value in df.index.tolist()]
    columns = [encode(value) for value in df.columns.tolist()]
    index_dtypes = getattr(df.index, "dtypes", None)
    if index_dtypes is None:
        index_dtypes = [df.index.dtype]
    rows = [
        [encode(df.iloc[row, column]) for column in range(df.shape[1])]
        for row in range(df.shape[0])
    ]
    identity = [
        "dataframe-v2",
        list(df.shape),
        columns,
        [str(dtype) for dtype in df.dtypes],
        [type(df.index).__module__, type(df.index).__qualname__],
        [str(dtype) for dtype in index_dtypes],
        [encode(name) for name in df.index.names],
        index_values,
        rows,
    ]
    canonical = json.dumps(
        identity, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def run_anomaly_sources(df):
    """Run the anomaly pipeline and return its unified-table sources.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset. Not modified. The pipeline's default settings are used.

    Returns
    -------
    (sources, notes) : (dict, list[str])
        ``sources`` maps ``"isolation_forest"``, ``"lof"`` and
        ``"anomaly_analysis"`` to per-record DataFrames indexed like ``df``;
        it is ``{}`` when anomaly detection could not run. ``notes`` are
        short messages for the user: why it did not run, or the pipeline's
        own warnings. An empty list means there is nothing to report.

    Raises
    ------
    TypeError
        If ``df`` is not a DataFrame.
    """
    _validate_dataset(df)

    if df.shape[0] == 0 or df.shape[1] == 0:
        return {}, ["Anomaly detection was not run: the dataset is empty."]
    if df.index.has_duplicates:
        # The pipeline needs one unique label per record. Resetting the index
        # here would report wrong Record IDs, so ML detection is skipped.
        return {}, [
            "Anomaly detection was not run: the dataset index has duplicate "
            "labels, so records cannot be identified uniquely."
        ]

    try:
        result = run_anomaly_pipeline(df)
    except (ValueError, RuntimeError) as error:
        return {}, [f"Anomaly detection was not run: {error}"]

    sources = {
        SRC_IF: result.isolation_forest.results,
        SRC_LOF: result.lof.results,
        SRC_ANOMALY: result.analysis,
    }
    return sources, list(result.warning_messages)


def _usable_ml_sources(df, anomaly_sources):
    """Split ML sources into (usable, notes).

    Pipeline results are indexed exactly like the dataset. A result whose
    index differs (for example one computed on another dataset) is skipped
    with a note, because its rows could not be mapped to records correctly.
    """
    usable, notes = {}, []
    for source, frame in anomaly_sources.items():
        if isinstance(frame, pd.DataFrame) and not frame.empty \
                and not frame.index.equals(df.index):
            notes.append(
                f"{SOURCE_LABELS[source]} results were not added: they do not "
                "line up with the rows of the dataset."
            )
            continue
        usable[source] = frame
    return usable, notes


# ----------------------------------------------------------------------
# Unified table
# ----------------------------------------------------------------------
def build_unified_table(df, quality_results, anomaly_sources=None):
    """Build the unified findings table for the dataset ``df``.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset the results were computed on. Only its index is used.
    quality_results : dict
        Source name -> result, using the source names of
        ``modules.unified_results`` (``"missing_values"``, ``"duplicates"``,
        ``"invalid_values"``, ``"data_type_problems"``, ``"format_problems"``,
        ``"inconsistent_values"``, ``"iqr_outliers"``). A value of ``None``
        or an empty DataFrame means "no findings".
    anomaly_sources : dict, optional
        Output of ``run_anomaly_sources`` (``"isolation_forest"``, ``"lof"``,
        ``"anomaly_analysis"``). ``None`` or ``{}`` means ML results are not
        available; the table then contains the quality results only.

    Returns
    -------
    (unified, notes) : (pandas.DataFrame, list[str])
        ``unified`` has the columns ``UNIFIED_COLUMNS`` (one row per finding,
        possibly empty). ``notes`` lists sources that could not be added and
        why; empty when everything was added.

    Raises
    ------
    TypeError
        If ``df`` is not a DataFrame or a result container is not a dict.
    ValueError
        If a source name is not supported.
    """
    _validate_dataset(df)
    if not isinstance(quality_results, dict):
        raise TypeError("quality_results must be a dictionary of source name -> result.")
    if anomaly_sources is not None and not isinstance(anomaly_sources, dict):
        raise TypeError("anomaly_sources must be a dictionary of source name -> result or None.")

    unknown = [
        name for name in list(quality_results) + list(anomaly_sources or {})
        if name not in SUPPORTED_SOURCES
    ]
    if unknown:
        raise ValueError(f"Unknown source name(s) {unknown}. Supported: {SUPPORTED_SOURCES}.")

    results = {**quality_results, **(anomaly_sources or {})}
    notes = []

    # ML results must line up with the dataset rows however they were passed.
    ml_results, ml_notes = _usable_ml_sources(
        df, {name: results.pop(name) for name in ML_SOURCES if name in results}
    )
    results.update(ml_results)
    notes.extend(ml_notes)

    # Normal case: everything converts in one call.
    try:
        return combine_quality_results(results, index=df.index), notes
    except (TypeError, ValueError):
        pass

    # Something in one source could not be converted (for example a Row that
    # is not in the dataset index). Find out which, keep the rest.
    frames = []
    for source in SUPPORTED_SOURCES:
        if source not in results:
            continue
        try:
            frame = combine_quality_results({source: results[source]}, index=df.index)
        except (TypeError, ValueError) as error:
            notes.append(
                f"{SOURCE_LABELS[source]} results were not added to the "
                f"unified table: {error}"
            )
            continue
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return combine_quality_results({}, index=df.index), notes
    return pd.concat(frames, ignore_index=True), notes


# ----------------------------------------------------------------------
# Helpers for the page
# ----------------------------------------------------------------------
def get_unified_summary(unified):
    """Counts for the page header, as plain Python ints.

    * ``total_findings``        - every row of the table.
    * ``data_quality``          - real quality problems.
    * ``statistical_outliers``  - IQR outliers (unusual, not necessarily wrong).
    * ``ml_anomaly_findings``   - Isolation Forest + LOF + combined rows.
    * ``ml_records_to_review``  - DISTINCT records behind those ML rows (a
      record flagged by both models appears in several rows but is one
      record). ML anomalies are records to review, not errors.
    """
    _validate_unified(unified)
    category = unified["Category"]
    ml = unified[category == CATEGORY_ANOMALY]
    return {
        "total_findings": int(len(unified)),
        "data_quality": int((category == CATEGORY_QUALITY).sum()),
        "statistical_outliers": int((category == CATEGORY_OUTLIER).sum()),
        "ml_anomaly_findings": int(len(ml)),
        "ml_records_to_review": int(len(ml[["Record ID", "Row Position"]].drop_duplicates())),
    }


def count_by_issue_type(unified):
    """Number of findings per issue type, in the table's source order."""
    _validate_unified(unified)
    if unified.empty:
        return pd.DataFrame(columns=_COUNT_COLUMNS)
    counts = (
        unified.groupby(["Category", "Issue Type", "Method"], sort=False)
        .size()
        .reset_index(name="Findings")
    )
    return counts[_COUNT_COLUMNS]


def filter_unified_table(unified, categories=None, issue_types=None):
    """Rows of ``unified`` in the given categories / issue types.

    ``None`` means "do not filter on this"; an empty list selects nothing.
    Returns a new DataFrame; ``unified`` is not modified.
    """
    _validate_unified(unified)
    keep = pd.Series(True, index=unified.index)
    if categories is not None:
        keep &= unified["Category"].isin(list(categories))
    if issue_types is not None:
        keep &= unified["Issue Type"].isin(list(issue_types))
    return unified[keep].reset_index(drop=True)


def prepare_display_table(unified):
    """Copy of ``unified`` that is safe to show in a Streamlit table.

    ``Record ID`` and ``Value`` mix numbers and text in the same column
    (for example 150 and "Twenty"), which table rendering can reject, so they
    are shown as text (blank when missing). The original is not modified;
    use it, not this copy, for calculations and downloads.
    """
    _validate_unified(unified)
    table = unified.copy()
    for column in ("Record ID", "Value"):
        table[column] = pd.Series(
            ["" if pd.isna(v) else str(v) for v in unified[column]],
            index=table.index,
            dtype=object,
        )
    return table
