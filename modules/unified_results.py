"""Unified Data Quality Results (PDF Phase 11, Step 19).

Turns the differently shaped outputs of the existing detection modules into
ONE "findings" table: one row per finding, always the same columns.

Why a conversion step is needed
-------------------------------
The existing modules do not share a format, and their ``Row`` columns do not
mean the same thing:

=====================  ==============================  ======================
Source                 Shape                           What ``Row`` means
=====================  ==============================  ======================
missing_values         one row per COLUMN              (no Row)
duplicates             the duplicated DATA rows        (no Row; index labels)
invalid_values         one row per bad cell            1-based POSITION
data_type_problems     one row per bad cell            1-based POSITION
format_problems        one row per bad cell            index LABEL + 1
iqr_outliers           one row per outlying cell       index LABEL + 1
inconsistent_values    one row per COLUMN group        (no Row)
isolation_forest       one row per record, all records index = record label
lof                    one row per record, all records index = record label
anomaly_analysis       one row per record, all records index (or Record ID)
=====================  ==============================  ======================

"Index LABEL + 1" and "1-based position" are only the same number when the
DataFrame has the default 0..n-1 index. This module never guesses: it keeps
two separate columns, ``Record ID`` (the DataFrame index label) and
``Row Position`` (1-based position in the dataset), and fills each only when
it can be derived exactly. Pass the dataset's ``index`` to fill both.

Anomalies are not errors
------------------------
Every finding has a ``Category``:

* "Data Quality"        - an actual quality problem (missing, duplicate,
                          invalid, data type, format, inconsistent).
* "Statistical Outlier" - IQR outlier. Unusual, not necessarily wrong.
* "ML Anomaly"          - Isolation Forest / LOF / combined status. A record
                          to REVIEW, never an automatic error, never deleted.

Records whose combined status is "Normal" are not findings and are not listed.

This module only reshapes results. It runs no detection, cleans nothing,
modifies none of its inputs, and has no UI code.
"""

import pandas as pd

from modules.anomaly_analysis import (
    HIGH_CONFIDENCE,
    IF_RESULT_COL,
    LOF_RESULT_COL,
    NEEDS_REVIEW,
    NORMAL,
    POTENTIAL_ANOMALY,
    STATUS_COL,
)

# ----------------------------------------------------------------------
# Output format
# ----------------------------------------------------------------------
UNIFIED_COLUMNS = [
    "Category",      # Data Quality / Statistical Outlier / ML Anomaly
    "Issue Type",    # e.g. Missing Value, Invalid Value, IQR Outlier ...
    "Method",        # Quality Check / IQR / Isolation Forest / LOF / both
    "Scope",         # Column / Record / Cell
    "Record ID",     # DataFrame index label (None if not derivable)
    "Row Position",  # 1-based position in the dataset (<NA> if not derivable)
    "Column",
    "Value",
    "Status",        # the source's own status text
    "Details",
]

CATEGORY_QUALITY = "Data Quality"
CATEGORY_OUTLIER = "Statistical Outlier"
CATEGORY_ANOMALY = "ML Anomaly"

SCOPE_COLUMN = "Column"
SCOPE_RECORD = "Record"
SCOPE_CELL = "Cell"

# Source names (keys of the dict given to combine_quality_results)
SRC_MISSING = "missing_values"
SRC_DUPLICATES = "duplicates"
SRC_INVALID = "invalid_values"
SRC_DATA_TYPE = "data_type_problems"
SRC_FORMAT = "format_problems"
SRC_CONSISTENCY = "inconsistent_values"
SRC_IQR = "iqr_outliers"
SRC_IF = "isolation_forest"
SRC_LOF = "lof"
SRC_ANOMALY = "anomaly_analysis"

# How to read each source's "Row" column.
ROW_POSITION = "position"          # 1-based position in the DataFrame
ROW_LABEL_PLUS_ONE = "label+1"     # index label + 1 (assumes integer labels)

# (category, issue type, method, scope, row meaning)
_SPECS = {
    SRC_MISSING: (CATEGORY_QUALITY, "Missing Value", "Quality Check", SCOPE_COLUMN, None),
    SRC_DUPLICATES: (CATEGORY_QUALITY, "Duplicate Record", "Quality Check", SCOPE_RECORD, None),
    SRC_INVALID: (CATEGORY_QUALITY, "Invalid Value", "Quality Check", SCOPE_CELL, ROW_POSITION),
    SRC_DATA_TYPE: (CATEGORY_QUALITY, "Data Type Problem", "Quality Check", SCOPE_CELL, ROW_POSITION),
    SRC_FORMAT: (CATEGORY_QUALITY, "Format Problem", "Quality Check", SCOPE_CELL, ROW_LABEL_PLUS_ONE),
    SRC_CONSISTENCY: (CATEGORY_QUALITY, "Inconsistent Representation", "Quality Check", SCOPE_COLUMN, None),
    SRC_IQR: (CATEGORY_OUTLIER, "IQR Outlier", "IQR", SCOPE_CELL, ROW_LABEL_PLUS_ONE),
    SRC_IF: (CATEGORY_ANOMALY, "Isolation Forest Anomaly", "Isolation Forest", SCOPE_RECORD, None),
    SRC_LOF: (CATEGORY_ANOMALY, "LOF Anomaly", "LOF", SCOPE_RECORD, None),
    SRC_ANOMALY: (CATEGORY_ANOMALY, "Combined Anomaly", "Isolation Forest + LOF", SCOPE_RECORD, None),
}

# Output order is fixed (not the caller's dict order) so results are stable.
SUPPORTED_SOURCES = list(_SPECS)

_RECORD_ID_COL = "Record ID"
_ANOMALY_LABELS = {NORMAL: NORMAL, POTENTIAL_ANOMALY: POTENTIAL_ANOMALY, "Anomaly": POTENTIAL_ANOMALY}
_COMBINED_STATUSES = {NORMAL, NEEDS_REVIEW, HIGH_CONFIDENCE}


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def _clean(value):
    """Plain Python value, with missing values (None/NaN/NA) as None."""
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return value
    return value.item() if hasattr(value, "item") else value


def _require(df, source, columns):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Result for '{source}' is missing required columns {missing}. "
            f"Available columns: {list(df.columns)}."
        )


def _as_int(value, source):
    """Integer from a Row value, or None if missing."""
    if value is None or pd.isna(value):
        return None
    try:
        as_int = int(value)
        if as_int != value:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(f"Result for '{source}' has a non-integer Row value: {value!r}.")
    return as_int


def _position_of_label(label, index, on_missing="raise"):
    """1-based position of ``label`` in ``index``.

    * label occurs exactly once   -> its exact 1-based position, even when
      OTHER labels in the index are duplicated;
    * label occurs more than once -> None (ambiguous, never guessed);
    * label does not occur        -> ValueError (the wrong index was
      supplied), or None when ``on_missing="none"``.
    """
    found = index.get_indexer_for([label])
    found = found[found >= 0]
    if len(found) == 0:
        if on_missing == "none":
            return None
        raise ValueError(f"Record ID {label!r} is not in the supplied index.")
    if len(found) > 1:
        return None
    return int(found[0]) + 1


def _cell_identity(row_value, row_meaning, index, source):
    """(Record ID, Row Position) for a cell-level finding."""
    number = _as_int(row_value, source)
    if number is None:
        return None, None

    if row_meaning == ROW_POSITION:
        if number < 1:
            raise ValueError(f"Result for '{source}' has an invalid Row position: {number}.")
        if index is None:
            return None, number                       # label unknown
        if number > len(index):
            raise ValueError(
                f"Result for '{source}' refers to row position {number}, but the "
                f"supplied index has only {len(index)} rows."
            )
        return _clean(index[number - 1]), number

    label = number - 1                                # ROW_LABEL_PLUS_ONE
    if index is None:
        return label, None                            # position unknown
    return label, _position_of_label(label, index)


def _record_identities(df, index, use_record_id_column):
    """(labels, positions) lists for a record-level frame.

    ``labels`` are what is reported as ``Record ID``. They come from the
    frame's own ``Record ID`` column when ``use_record_id_column`` is true and
    the column exists (for example the dashboard table), otherwise from the
    frame's index.

    ``positions`` (1-based, in the dataset ``index``) are only set when they
    can be established exactly:

    1. frame index == dataset index (same labels, same order): the frame is
       row-aligned with the dataset, so position = row number in the frame.
       This is exact even with duplicated labels, and holds whether or not a
       ``Record ID`` column exists.
    2. no ``Record ID`` column: the frame's index labels ARE dataset labels,
       so each is looked up; a label absent from ``index`` is an error (wrong
       index supplied) and a duplicated label gives None.
    3. ``Record ID`` column that is an exact copy of the frame's index (the
       dashboard table built by ``build_anomaly_table``): the IDs are index
       labels, so each is looked up. Absent or duplicated labels give None;
       there is no error, because the table may be a subset.
    4. ``Record ID`` column that differs from the frame's index: the IDs are
       SEPARATE identifiers (e.g. customer numbers). They are reported as
       given and are NEVER matched against index labels, because an ID that
       happens to equal some label would give a wrong position.
    """
    has_id_column = use_record_id_column and _RECORD_ID_COL in df.columns
    frame_labels = [_clean(v) for v in df.index]
    labels = [_clean(v) for v in df[_RECORD_ID_COL]] if has_id_column else frame_labels

    if index is None:
        return labels, [None] * len(labels)
    if df.index.equals(index):
        return labels, list(range(1, len(labels) + 1))
    if not has_id_column:
        return labels, [_position_of_label(label, index) for label in labels]
    if labels == frame_labels:
        return labels, [_position_of_label(label, index, on_missing="none") for label in labels]
    return labels, [None] * len(labels)


def _finding(source, record_id=None, row_position=None, column=None,
             value=None, status=None, details=None):
    category, issue, method, scope, _ = _SPECS[source]
    return {
        "Category": category,
        "Issue Type": issue,
        "Method": method,
        "Scope": scope,
        "Record ID": record_id,
        "Row Position": row_position,
        "Column": column,
        "Value": value,
        "Status": status if status is not None else issue,
        "Details": details,
    }


# ----------------------------------------------------------------------
# One converter per source: DataFrame -> list of finding dicts
# ----------------------------------------------------------------------
def _from_missing(df, index, source):
    _require(df, source, ["Column", "Missing Values"])
    out = []
    for _, r in df.iterrows():
        count = pd.to_numeric(r["Missing Values"], errors="coerce")
        if pd.isna(count):
            raise ValueError(f"Result for '{source}' has a non-numeric 'Missing Values' count.")
        if count <= 0:
            continue                                  # a clean column is not a finding
        details = f"{int(count)} missing value(s)"
        if "Missing Percentage" in df.columns and not pd.isna(r["Missing Percentage"]):
            details += f" ({r['Missing Percentage']}%)"
        out.append(_finding(source, column=_clean(r["Column"]), details=details))
    return out


def _from_duplicates(df, index, source):
    labels, positions = _record_identities(df, index, False)
    status = df["Status"] if "Status" in df.columns else [None] * len(df)
    return [
        _finding(source, record_id=label, row_position=pos, status=_clean(st),
                 details="Identical to at least one other record")
        for label, pos, st in zip(labels, positions, status)
    ]


def _from_cell_issues(df, index, source):
    """invalid_values, data_type_problems, format_problems, iqr_outliers."""
    meaning = _SPECS[source][4]
    _require(df, source, ["Row", "Column", "Value"])
    out = []
    for _, r in df.iterrows():
        record_id, position = _cell_identity(_clean(r["Row"]), meaning, index, source)

        details = None
        if source == SRC_FORMAT and "Format" in df.columns:
            details = f"Expected format: {_clean(r['Format'])}"
        elif source == SRC_IQR and "Lower Bound" in df.columns and "Upper Bound" in df.columns:
            details = f"Outside IQR range {_clean(r['Lower Bound'])} to {_clean(r['Upper Bound'])}"

        out.append(_finding(
            source, record_id=record_id, row_position=position,
            column=_clean(r["Column"]), value=_clean(r["Value"]),
            status=_clean(r["Status"]) if "Status" in df.columns else None,
            details=details,
        ))
    return out


def _from_consistency(df, index, source):
    _require(df, source, ["Column", "Different Representations"])
    out = []
    for _, r in df.iterrows():
        normal = _clean(r["Normalized Value"]) if "Normalized Value" in df.columns else None
        prefix = f"'{normal}' appears as: " if normal is not None else "Appears as: "
        out.append(_finding(
            source, column=_clean(r["Column"]),
            status=_clean(r["Status"]) if "Status" in df.columns else None,
            details=prefix + str(r["Different Representations"]),
        ))
    return out


def _from_model(df, index, source):
    """Isolation Forest / LOF per-record results (only flagged records kept)."""
    prefix = "if" if source == SRC_IF else "lof"
    status_col, score_col = f"{prefix}_status", f"{prefix}_score"
    _require(df, source, [status_col])

    labels, positions = _record_identities(df, index, False)
    out = []
    for i, raw in enumerate(df[status_col]):
        label = _ANOMALY_LABELS.get(raw)
        if label is None:
            raise ValueError(
                f"Result for '{source}' has an invalid label {raw!r} in '{status_col}'. "
                f"Allowed labels: {sorted(_ANOMALY_LABELS)}."
            )
        if label != POTENTIAL_ANOMALY:
            continue                                  # Normal records are not findings
        details = None
        if score_col in df.columns:
            score = _clean(df[score_col].iloc[i])
            if score is not None:
                details = f"{prefix.upper()} score: {float(score):.4f} (higher = more unusual)"
        out.append(_finding(source, record_id=labels[i], row_position=positions[i],
                            status=POTENTIAL_ANOMALY, details=details))
    return out


def _from_combined(df, index, source):
    """Output of combine_anomaly_results, or the dashboard table built from it.

    A "Record ID" column is reported as given. Whether it is also used to find
    a Row Position depends on whether it is an index copy or a separate
    identifier; see ``_record_identities``.
    """
    _require(df, source, [STATUS_COL])
    labels, positions = _record_identities(df, index, True)
    out = []
    for i, status in enumerate(df[STATUS_COL]):
        if status not in _COMBINED_STATUSES:
            raise ValueError(
                f"Result for '{source}' has an invalid status {status!r}. "
                f"Allowed statuses: {sorted(_COMBINED_STATUSES)}."
            )
        if status == NORMAL:
            continue
        parts = []
        if IF_RESULT_COL in df.columns:
            parts.append(f"IF: {_clean(df[IF_RESULT_COL].iloc[i])}")
        if LOF_RESULT_COL in df.columns:
            parts.append(f"LOF: {_clean(df[LOF_RESULT_COL].iloc[i])}")
        out.append(_finding(source, record_id=labels[i], row_position=positions[i],
                            status=status, details="; ".join(parts) or None))
    return out


_CONVERTERS = {
    SRC_MISSING: _from_missing,
    SRC_DUPLICATES: _from_duplicates,
    SRC_INVALID: _from_cell_issues,
    SRC_DATA_TYPE: _from_cell_issues,
    SRC_FORMAT: _from_cell_issues,
    SRC_CONSISTENCY: _from_consistency,
    SRC_IQR: _from_cell_issues,
    SRC_IF: _from_model,
    SRC_LOF: _from_model,
    SRC_ANOMALY: _from_combined,
}


def _unwrap(source, value):
    """A DataFrame, or (IF/LOF only) a result object with a ``.results`` DataFrame."""
    if isinstance(value, pd.DataFrame):
        return value
    inner = getattr(value, "results", None)
    if source in (SRC_IF, SRC_LOF) and isinstance(inner, pd.DataFrame):
        return inner
    raise TypeError(
        f"Result for '{source}' must be a pandas DataFrame, got {type(value).__name__}."
    )


def _to_frame(findings):
    """Findings -> DataFrame with fixed columns and dtypes (ints stay ints)."""
    data = {}
    for column in UNIFIED_COLUMNS:
        values = [f[column] for f in findings]
        if column == "Row Position":
            data[column] = pd.array(values, dtype="Int64")
        else:
            data[column] = pd.Series(values, dtype=object)
    return pd.DataFrame(data, columns=UNIFIED_COLUMNS)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def combine_quality_results(results, index=None):
    """Combine detection results into one standardized findings table.

    Parameters
    ----------
    results : dict
        Maps a source name to that source's result. Supported names are in
        ``SUPPORTED_SOURCES``:

        * "missing_values"       <- detect_missing_values(df)
        * "duplicates"           <- detect_duplicates(df)
        * "invalid_values"       <- detect_invalid_values(df, rules)
        * "data_type_problems"   <- detect_data_type_problems(df)
        * "format_problems"      <- detect_format_problems(df)
        * "inconsistent_values"  <- detect_inconsistent_values(df)
        * "iqr_outliers"         <- detect_iqr_outliers(df)
        * "isolation_forest"     <- run_isolation_forest(...) result or its .results
        * "lof"                  <- run_lof(...) result or its .results
        * "anomaly_analysis"     <- combine_anomaly_results(...) or the dashboard table
                                    (a "Record ID" column is reported as given; it is
                                    treated as an index label only when it is an exact
                                    copy of that table's index)

        A value of None, or an empty DataFrame (some detectors return one
        with no columns when nothing is found), means "no findings".
    index : pandas.Index, optional
        The index of the ORIGINAL dataset (``df.index``). With it, both
        ``Record ID`` and ``Row Position`` are filled wherever they can be
        derived. Without it, only what each source states exactly is filled
        (see the module docstring); nothing is guessed. A label that occurs
        more than once in ``index`` gives no ``Row Position``; a label that
        occurs once keeps its exact position even if other labels repeat.

    Returns
    -------
    pandas.DataFrame
        Columns ``UNIFIED_COLUMNS``, a fresh 0..n-1 index, one row per
        finding, ordered by source (``SUPPORTED_SOURCES``) and then by the
        source's own row order. Inputs are not modified.

    Raises
    ------
    TypeError
        ``results`` is not a dict, a result is not a DataFrame, or ``index``
        is not a pandas Index.
    ValueError
        Unknown source name, missing required columns, invalid labels or
        counts, or a Row/Record ID that does not exist in ``index``.
    """
    if not isinstance(results, dict):
        raise TypeError("results must be a dictionary mapping source names to DataFrames.")
    if index is not None and not isinstance(index, pd.Index):
        raise TypeError("index must be a pandas Index (for example df.index) or None.")

    unknown = [name for name in results if name not in _SPECS]
    if unknown:
        raise ValueError(f"Unknown source name(s) {unknown}. Supported: {SUPPORTED_SOURCES}.")

    findings = []
    for source in SUPPORTED_SOURCES:
        if source not in results or results[source] is None:
            continue
        frame = _unwrap(source, results[source])
        if frame.empty:
            continue
        findings.extend(_CONVERTERS[source](frame, index, source))

    return _to_frame(findings)
