"""Data Quality Score and Severity Classification (PDF Phase 13, Steps 22-23).

Turns the existing detector results into

* four dimension scores (Completeness, Validity, Consistency, Uniqueness),
* one explainable 0-100 overall score with a rating (Excellent / Good /
  Fair / Poor), and
* a Severity and a "Score Treatment" for every row of the unified findings
  table (display only; ``UNIFIED_COLUMNS`` is never changed).

This is the project's chosen methodology, not a universal standard. All
weights, caps, bands and severity thresholds are named constants below.

Method in one page
------------------
Dimension score = 100 x (1 - bad / total), with dataset-wide denominators:

=============  ======  ===============================  =========================
Dimension      Weight  Fed by                           bad / total
=============  ======  ===============================  =========================
Completeness   30 %    missing_values                   missing cells / all cells
Validity       25 %    invalid_values, data_type_       DISTINCT bad cells (after
                       problems, format_problems        overlap rules) / non-missing
                                                        cells
Consistency    25 %    inconsistent_values              minority-spelling cells /
                                                        non-missing cells
Uniqueness     20 %    duplicates                       redundant copies / rows
=============  ======  ===============================  =========================

Overall = weighted sum of the ASSESSED dimensions (weights are rescaled to
100 % when a dimension cannot be assessed). The result is then limited by
the worst severity found: a Critical issue caps it at 74.9, a High issue at
89.9. The score is rounded to one decimal (half up) AFTER capping, and the
rating is taken from that rounded number, so the number shown and its band
always agree.

Severity (per issue group = issue type x column; duplicates = whole dataset)
----------------------------------------------------------------------------
Share = affected units / units in that column (rows for duplicates).

* Invalid Value, Data Type Problem, Duplicate Record: any occurrence is at
  least Medium; High from 2 %; Critical from 10 %.
* Format Problem, Inconsistent Representation: Low below 2 %; Medium from
  2 %; High from 10 %; Critical from 25 %.
* Missing Value: Low below 5 %; Medium from 5 %; High from 20 %; Critical
  from 50 %.

A duplicate therefore becomes "High" at a 2 % duplicate rate (our chosen
methodology; it is not forced to High for every duplicate).

IQR outliers and ML anomalies (Isolation Forest, LOF, combined) are labelled
"Review (not rated)" and are NEVER part of the score. They stay visible.

Overlap rules (nothing is counted twice inside one dimension)
------------------------------------------------------------
1. A blank cell (None / NaN / whitespace-only value) belongs to Completeness.
   Validity or Consistency findings on blank cells are not counted again
   (only when the missing-value check ran).
2. The same cell flagged by several validity checks is counted once, in the
   order Invalid Value, Data Type Problem, Format Problem.
3. Cell findings whose exact row position is unknown (for example a label
   that occurs twice in the index) are shown but NOT scored; a warning gives
   an upper bound of their possible effect. Row positions are never guessed.
4. For duplicates, every group member is listed, but only the redundant
   copies (rows in the duplicate frame minus number of distinct groups) are
   counted. Which row is the "original" is not known and not guessed.
5. A cell may count in two different dimensions (for example a redundant
   copy that also has an invalid value). Each dimension answers a different
   question; this is a documented limitation.

Check status
------------
For ``quality_results``: a key that is absent or ``None`` means the check
DID NOT RUN; an empty DataFrame means it ran and found nothing. (This is
deliberately different from ``build_unified_table``, where ``None`` means no
findings.) Some detectors return an empty frame WITHOUT columns when they
find nothing (``detect_inconsistent_values``, ``detect_format_problems``,
``detect_iqr_outliers``); that is handled as "ran, found nothing". A
dimension whose checks did not run, or whose denominator is 0, is "Not
assessed". If only some of a dimension's checks ran it is "Partial".

What this module does NOT do
----------------------------
No detection, no cleaning, no UI code. Inputs are never modified. Existing
detector bugs are not worked around (for example the substring match on
"pin" in format_validation.py): the score reflects what the detectors report.
"""

from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

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

# ----------------------------------------------------------------------
# Methodology constants
# ----------------------------------------------------------------------
DIM_COMPLETENESS = "Completeness"
DIM_VALIDITY = "Validity"
DIM_CONSISTENCY = "Consistency"
DIM_UNIQUENESS = "Uniqueness"

WEIGHTS = {
    DIM_COMPLETENESS: 0.30,
    DIM_VALIDITY: 0.25,
    DIM_CONSISTENCY: 0.25,
    DIM_UNIQUENESS: 0.20,
}

CAP_CRITICAL = 74.9
CAP_HIGH = 89.9

# (lowest score of the band, name), best band first.
RATING_BANDS = (
    (90.0, "Excellent"),
    (75.0, "Good"),
    (50.0, "Fair"),
    (0.0, "Poor"),
)
RATING_NOT_SCORED = "Not scored"

SEV_LOW = "Low"
SEV_MEDIUM = "Medium"
SEV_HIGH = "High"
SEV_CRITICAL = "Critical"
SEV_REVIEW = "Review (not rated)"   # IQR outliers and ML anomalies
SEV_NOT_RATED = "Not rated"         # visible finding that is not scored

_SEVERITY_RANK = {SEV_LOW: 1, SEV_MEDIUM: 2, SEV_HIGH: 3, SEV_CRITICAL: 4}

# Issue types, exactly as written by modules/unified_results.py.
ISSUE_MISSING = "Missing Value"
ISSUE_DUPLICATE = "Duplicate Record"
ISSUE_INVALID = "Invalid Value"
ISSUE_DATA_TYPE = "Data Type Problem"
ISSUE_FORMAT = "Format Problem"
ISSUE_INCONSISTENT = "Inconsistent Representation"

# (medium_from, high_from, critical_from) as fractions. A share at or above
# a limit gets that severity; below medium_from it is Low. medium_from = 0.0
# means "any occurrence is at least Medium".
SEVERITY_LADDERS = {
    ISSUE_INVALID: (0.0, 0.02, 0.10),
    ISSUE_DATA_TYPE: (0.0, 0.02, 0.10),
    ISSUE_DUPLICATE: (0.0, 0.02, 0.10),
    ISSUE_FORMAT: (0.02, 0.10, 0.25),
    ISSUE_INCONSISTENT: (0.02, 0.10, 0.25),
    ISSUE_MISSING: (0.05, 0.20, 0.50),
}

WHOLE_ROW = "(whole row)"

STATUS_ASSESSED = "Assessed"
STATUS_PARTIAL = "Partial"
STATUS_NOT_ASSESSED = "Not assessed"

DIMENSION_COLUMNS = [
    "Dimension", "Weight (%)", "Effective Weight (%)", "Score",
    "Issues Counted", "Items Checked", "Unit", "Checks Run", "Status",
]
SEVERITY_COLUMNS = [
    "Dimension", "Issue Type", "Column", "Severity", "Affected", "Total", "Share (%)",
]

# Score Treatment texts (shown next to every finding).
TREAT_REVIEW = "Excluded - review only (not scored)"
TREAT_COMPLETENESS = "Scored - Completeness"
TREAT_VALIDITY = "Scored - Validity"
TREAT_CONSISTENCY = "Scored - Consistency (minority spellings counted)"
TREAT_UNIQUENESS = ("Scored - Uniqueness (redundant copies of the duplicate group are "
                    "counted; the original row is not identified)")
TREAT_BLANK = "Counted under Completeness (blank cell)"
TREAT_BLANK_GROUP = "Counted under Completeness (blank cells)"
TREAT_ALREADY_COUNTED = "Counted once - same cell already counted as {primary}"
TREAT_UNIDENTIFIABLE = ("Not scored - exact row not identifiable (see the score warning)")
TREAT_NOT_RUN = "Not scored - the check did not run"
TREAT_UNUSABLE = "Not scored - the results of this check could not be processed"
TREAT_EMPTY = "Not scored - the dataset is empty"
TREAT_OTHER = "Not scored"

_SCORED_SOURCES = (
    SRC_MISSING, SRC_DUPLICATES, SRC_INVALID, SRC_DATA_TYPE, SRC_FORMAT, SRC_CONSISTENCY,
)
_CELL_SOURCES = (SRC_INVALID, SRC_DATA_TYPE, SRC_FORMAT)
_ISSUE_OF = {
    SRC_INVALID: ISSUE_INVALID,
    SRC_DATA_TYPE: ISSUE_DATA_TYPE,
    SRC_FORMAT: ISSUE_FORMAT,
}
# Validity findings are de-duplicated in this priority order.
_VALIDITY_ISSUES = (ISSUE_INVALID, ISSUE_DATA_TYPE, ISSUE_FORMAT)
_SOURCE_OF_ISSUE = {
    ISSUE_MISSING: SRC_MISSING,
    ISSUE_DUPLICATE: SRC_DUPLICATES,
    ISSUE_INVALID: SRC_INVALID,
    ISSUE_DATA_TYPE: SRC_DATA_TYPE,
    ISSUE_FORMAT: SRC_FORMAT,
    ISSUE_INCONSISTENT: SRC_CONSISTENCY,
}
_SOURCE_LABELS = {
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


# ----------------------------------------------------------------------
# Result object
# ----------------------------------------------------------------------
@dataclass
class QualityScoreResult:
    """Everything produced by ``calculate_quality_score``.

    Attributes
    ----------
    overall_score : float or None
        0-100, one decimal, after severity caps. ``None`` when nothing could
        be scored (empty dataset, or no assessable dimension).
    raw_score : float or None
        The weighted score before caps (one decimal).
    rating : str
        "Excellent", "Good", "Fair", "Poor" or "Not scored". Always derived
        from ``overall_score`` exactly as shown.
    cap_limit : float or None
        The cap implied by the worst severity (74.9 or 89.9), or None.
    cap_applied : bool
        True only when the cap actually lowered the displayed score.
    cap_reason : str or None
        Plain-language explanation when ``cap_applied`` is true.
    dimensions : pd.DataFrame
        One row per dimension (``DIMENSION_COLUMNS``).
    severity_summary : pd.DataFrame
        One row per rated issue group (``SEVERITY_COLUMNS``), worst first.
    findings : pd.DataFrame
        ``Severity`` and ``Score Treatment`` for every row of ``unified``
        (same index).
    unified : pd.DataFrame
        The unified findings table that was scored (given or built).
    unscored_impact : dict or None
        Upper bound of the effect of findings that could not be scored
        because their row is not identifiable (None if there are none).
    notes : list[str]
        Warnings and explanations for the user.
    """

    overall_score: object
    raw_score: object
    rating: str
    cap_limit: object
    cap_applied: bool
    cap_reason: object
    dimensions: pd.DataFrame
    severity_summary: pd.DataFrame
    findings: pd.DataFrame
    unified: pd.DataFrame
    unscored_impact: object = None
    notes: list = field(default_factory=list)


# ----------------------------------------------------------------------
# Small public helpers
# ----------------------------------------------------------------------
def get_rating(score):
    """Rating band of an (already rounded) score; ``None`` -> "Not scored"."""
    if score is None:
        return RATING_NOT_SCORED
    for lowest, name in RATING_BANDS:
        if score >= lowest:
            return name
    return RATING_BANDS[-1][1]


def classify_severity(issue_type, share):
    """Severity of an issue group from the share of units it affects (0-1)."""
    ladder = SEVERITY_LADDERS.get(issue_type)
    if ladder is None:
        raise ValueError(
            f"No severity ladder for issue type {issue_type!r}. "
            f"Known issue types: {sorted(SEVERITY_LADDERS)}."
        )
    share = float(share)
    if np.isnan(share):
        raise ValueError("share must be a number between 0 and 1.")
    share = min(max(share, 0.0), 1.0)
    medium_from, high_from, critical_from = ladder
    if share >= critical_from:
        return SEV_CRITICAL
    if share >= high_from:
        return SEV_HIGH
    if share >= medium_from:
        return SEV_MEDIUM
    return SEV_LOW


def _round1(value):
    """Round half up to one decimal (not banker's rounding)."""
    return float(Decimal(repr(float(value))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def _is_blank(value):
    """None / NaN / NA / whitespace-only text."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, dict, set, np.ndarray)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


# ----------------------------------------------------------------------
# Per-finding annotation storage
# ----------------------------------------------------------------------
class _Annotations:
    """Severity / Score Treatment for every row of the unified table."""

    def __init__(self, unified):
        self.issue = unified["Issue Type"].to_numpy(dtype=object)
        self.category = unified["Category"].to_numpy(dtype=object)
        self.column = unified["Column"].to_numpy(dtype=object)
        self.value = unified["Value"].to_numpy(dtype=object)
        self.position = unified["Row Position"].to_numpy(dtype=object)
        self.severity = [None] * len(unified)
        self.treatment = [None] * len(unified)
        self._quality_rows = {}
        for i, (category, issue) in enumerate(zip(self.category, self.issue)):
            if category == CATEGORY_QUALITY:
                self._quality_rows.setdefault(issue, []).append(i)

    def rows_of(self, issue_type):
        """Row numbers (0-based, in table order) of one quality issue type."""
        return self._quality_rows.get(issue_type, [])

    def set(self, i, severity, treatment):
        self.severity[i] = severity
        self.treatment[i] = treatment

    def mark_review(self):
        """IQR outliers and ML anomalies: visible, never scored."""
        for i, category in enumerate(self.category):
            if category in (CATEGORY_OUTLIER, CATEGORY_ANOMALY):
                self.set(i, SEV_REVIEW, TREAT_REVIEW)

    def finalize(self, ran, unusable, empty=False):
        """Give every still-unannotated finding an explained 'not scored' text."""
        for i in range(len(self.severity)):
            if self.severity[i] is not None:
                continue
            source = _SOURCE_OF_ISSUE.get(self.issue[i])
            if empty:
                text = TREAT_EMPTY
            elif source is None:
                text = TREAT_OTHER
            elif source in unusable:
                text = TREAT_UNUSABLE
            elif not ran.get(source, False):
                text = TREAT_NOT_RUN
            else:
                text = TREAT_OTHER
            self.set(i, SEV_NOT_RATED, text)


# ----------------------------------------------------------------------
# Validation and unified-table preparation
# ----------------------------------------------------------------------
def _validate_unified(unified):
    if not isinstance(unified, pd.DataFrame):
        raise TypeError(f"unified must be a pandas DataFrame, got {type(unified).__name__}.")
    missing = [c for c in UNIFIED_COLUMNS if c not in unified.columns]
    if missing:
        raise ValueError(
            f"unified is missing required columns {missing}. "
            f"Available columns: {list(unified.columns)}."
        )


def _validate_inputs(df, quality_results, unified):
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}.")
    if not isinstance(quality_results, dict):
        raise TypeError("quality_results must be a dictionary of source name -> DataFrame or None.")
    unknown = [name for name in quality_results if name not in SUPPORTED_SOURCES]
    if unknown:
        raise ValueError(f"Unknown source name(s) {unknown}. Supported: {SUPPORTED_SOURCES}.")
    for name, value in quality_results.items():
        if value is not None and not isinstance(value, pd.DataFrame):
            raise TypeError(
                f"Result for '{name}' must be a pandas DataFrame or None, "
                f"got {type(value).__name__}."
            )
    if unified is not None:
        _validate_unified(unified)


def _prepare_unified(df, quality_results, unified):
    """Return (unified table, {source: reason it could not be used}).

    A supplied table is trusted, except that its cell-level quality findings
    must be complete: if it holds fewer or more rows of a check than that
    check's own result, the check is reported as unusable instead of being
    scored as "no findings". Without a table, each source is converted on its
    own so one malformed result cannot hide the others.
    """
    failed = {}
    if unified is not None:
        for source in _CELL_SOURCES:
            frame = quality_results.get(source)
            if isinstance(frame, pd.DataFrame):
                found = int((unified["Issue Type"] == _ISSUE_OF[source]).sum())
                if found != len(frame):
                    failed[source] = (
                        f"the supplied unified table has {found} finding(s) of this check "
                        f"but the check reported {len(frame)}"
                    )
        return unified, failed

    parts = []
    for source in SUPPORTED_SOURCES:
        frame = quality_results.get(source)
        if not isinstance(frame, pd.DataFrame):
            continue
        try:
            part = combine_quality_results({source: frame}, index=df.index)
        except (TypeError, ValueError) as error:
            failed[source] = str(error)
            continue
        if not part.empty:
            parts.append(part)
    if parts:
        return pd.concat(parts, ignore_index=True), failed
    return combine_quality_results({}, index=df.index), failed


# ----------------------------------------------------------------------
# One step per dimension
# ----------------------------------------------------------------------
def _group(dimension, issue, column, affected, total):
    share = min(affected / total, 1.0) if total > 0 else 1.0
    return {
        "Dimension": dimension,
        "Issue Type": issue,
        "Column": column,
        "Severity": classify_severity(issue, share),
        "Affected": int(affected),
        "Total": int(total),
        "Share (%)": round(100.0 * share, 2),
    }


def _need_columns(frame, columns):
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"it is missing required columns {missing}")


def _step_missing(frame, n_rows, n_cols, ann):
    """Completeness. Returns (missing cells, {column: missing count}, groups)."""
    if frame.empty:                  # the check ran and reported nothing
        return 0, {}, []
    _need_columns(frame, ["Column", "Missing Values"])
    by_column = {}
    for column, raw in zip(frame["Column"], frame["Missing Values"]):
        count = pd.to_numeric(raw, errors="coerce")
        if pd.isna(count):
            raise ValueError("it has a non-numeric 'Missing Values' count")
        by_column[column] = by_column.get(column, 0) + max(int(count), 0)
    by_column = {c: min(v, n_rows) for c, v in by_column.items()}
    total = min(sum(by_column.values()), n_rows * n_cols)

    groups, severity_of = [], {}
    for column, count in by_column.items():
        if count > 0:
            group = _group(DIM_COMPLETENESS, ISSUE_MISSING, column, count, n_rows)
            groups.append(group)
            severity_of[column] = group["Severity"]
    for i in ann.rows_of(ISSUE_MISSING):
        severity = severity_of.get(ann.column[i])
        if severity is not None:
            ann.set(i, severity, TREAT_COMPLETENESS)
    return total, by_column, groups


def _step_validity(ann, issues_ran, non_missing, blank_to_completeness):
    """Validity. Distinct bad cells after the overlap rules."""
    claimed = {}            # (row position, column) -> issue type that counted it
    counted = {}            # (issue type, column) -> number of counted cells
    outcome = {}            # row number -> (kind, primary issue)
    n_unidentifiable = 0
    n_blank = 0

    for issue in _VALIDITY_ISSUES:
        if issue not in issues_ran:
            continue
        for i in ann.rows_of(issue):
            if blank_to_completeness and _is_blank(ann.value[i]):
                outcome[i] = ("blank", None)
                n_blank += 1
                continue
            position = ann.position[i]
            if pd.isna(position):
                outcome[i] = ("unidentifiable", None)
                n_unidentifiable += 1
                continue
            key = (int(position), ann.column[i])
            if key in claimed:
                outcome[i] = ("duplicate", claimed[key])
                continue
            claimed[key] = issue
            group_key = (issue, ann.column[i])
            counted[group_key] = counted.get(group_key, 0) + 1
            outcome[i] = ("counted", None)

    groups, severity_of = [], {}
    for (issue, column), count in counted.items():
        group = _group(DIM_VALIDITY, issue, column, count, max(non_missing(column), count))
        groups.append(group)
        severity_of[(issue, column)] = group["Severity"]

    for i, (kind, primary) in outcome.items():
        if kind == "counted":
            ann.set(i, severity_of[(ann.issue[i], ann.column[i])], TREAT_VALIDITY)
        elif kind == "duplicate":
            ann.set(i, SEV_NOT_RATED, TREAT_ALREADY_COUNTED.format(primary=primary))
        elif kind == "blank":
            ann.set(i, SEV_NOT_RATED, TREAT_BLANK)
        else:
            ann.set(i, SEV_NOT_RATED, TREAT_UNIDENTIFIABLE)

    return {
        "counted": len(claimed),
        "groups": groups,
        "unidentifiable": n_unidentifiable,
        "blank": n_blank,
    }


def _step_consistency(df, frame, non_missing, blank_to_completeness, ann):
    """Consistency. Counts minority spellings; never guesses the 'right' one.

    Uses the same normalisation as modules/consistency.py: missing values are
    dropped, the rest is converted to text, and groups are formed by
    strip + lower-case. Within a flagged group the cells of the most common
    exact spelling are not counted; all other cells are.
    """
    if frame.empty:                  # the check ran and reported nothing
        return 0, []                 # (the detector's empty frame has no columns)
    _need_columns(frame, ["Column", "Normalized Value"])
    cache = {}
    seen = set()
    rows = []                # per result row: (column, cells, is blank group)
    per_column = {}

    for column, normal in zip(frame["Column"], frame["Normalized Value"]):
        key = (column, normal)
        if key in seen:
            rows.append((column, 0, False))
            continue
        seen.add(key)
        if column not in df.columns:
            raise ValueError(f"column {column!r} is not in the dataset")
        if column not in cache:
            series = df[column]
            if isinstance(series, pd.DataFrame):
                raise ValueError(f"column name {column!r} is duplicated in the dataset")
            forms = series.dropna().astype(str)
            cache[column] = (
                forms.to_numpy(dtype=object),
                forms.str.strip().str.lower().to_numpy(dtype=object),
            )
        forms, normalised = cache[column]
        normal = str(normal)

        if normal == "" and blank_to_completeness:
            rows.append((column, 0, True))
            continue

        members = forms[normalised == normal]
        if len(members) == 0:
            cells = 0
        else:
            counts = pd.Series(members).value_counts()
            cells = int(counts.sum() - counts.max())
        rows.append((column, cells, False))
        per_column[column] = per_column.get(column, 0) + cells

    groups, severity_of = [], {}
    for column, cells in per_column.items():
        if cells > 0:
            group = _group(DIM_CONSISTENCY, ISSUE_INCONSISTENT, column, cells,
                           max(non_missing(column), cells))
            groups.append(group)
            severity_of[column] = group["Severity"]

    unified_rows = ann.rows_of(ISSUE_INCONSISTENT)
    if len(unified_rows) == len(rows):          # same order as the result's rows
        for i, (column, cells, blank) in zip(unified_rows, rows):
            if blank:
                ann.set(i, SEV_NOT_RATED, TREAT_BLANK_GROUP)
            elif column in severity_of:
                ann.set(i, severity_of[column], TREAT_CONSISTENCY)

    return sum(per_column.values()), groups


def _step_uniqueness(frame, n_rows, ann):
    """Uniqueness. Redundant copies = frame rows - number of distinct groups."""
    if frame.empty:
        return 0, []
    content = frame.drop(columns=["Status"], errors="ignore")
    if content.shape[1] == 0:
        raise ValueError("it has no data columns")
    try:
        distinct = len(content.drop_duplicates())
    except TypeError:                            # unhashable cells (e.g. lists)
        distinct = len(content.astype(str).drop_duplicates())
    redundant = min(max(len(frame) - distinct, 0), max(n_rows - 1, 0))

    groups = []
    if redundant > 0:
        group = _group(DIM_UNIQUENESS, ISSUE_DUPLICATE, WHOLE_ROW, redundant, n_rows)
        groups.append(group)
        for i in ann.rows_of(ISSUE_DUPLICATE):
            ann.set(i, group["Severity"], TREAT_UNIQUENESS)
    return redundant, groups


# ----------------------------------------------------------------------
# Assembling the result
# ----------------------------------------------------------------------
def _dimension(name, bad, total, run, of, unit):
    """One dimension's numbers. ``score`` is None when it cannot be assessed."""
    if run == 0 or bad is None or total <= 0:
        score, status = None, STATUS_NOT_ASSESSED
        bad_shown = None
    else:
        bad_shown = int(min(bad, total))
        score = max(0.0, min(100.0, 100.0 * (1.0 - bad_shown / total)))
        status = STATUS_ASSESSED if run == of else STATUS_PARTIAL
    return {
        "name": name, "score": score, "bad": bad_shown,
        "total": int(total) if total is not None else None,
        "unit": unit, "run": run, "of": of, "status": status,
    }


def _dimension_table(dims, weights_used):
    rows = []
    for d in dims:
        rows.append({
            "Dimension": d["name"],
            "Weight (%)": round(100.0 * WEIGHTS[d["name"]], 1),
            "Effective Weight (%)": (
                round(100.0 * weights_used[d["name"]], 1) if d["name"] in weights_used else 0.0
            ),
            "Score": _round1(d["score"]) if d["score"] is not None else np.nan,
            "Issues Counted": d["bad"] if d["bad"] is not None else np.nan,
            "Items Checked": d["total"] if d["status"] != STATUS_NOT_ASSESSED else np.nan,
            "Unit": d["unit"],
            "Checks Run": f"{d['run']} of {d['of']}",
            "Status": d["status"],
        })
    table = pd.DataFrame(rows, columns=DIMENSION_COLUMNS)
    for column in ("Issues Counted", "Items Checked"):
        table[column] = table[column].astype("Int64")
    return table


def _severity_table(groups):
    table = pd.DataFrame(groups, columns=SEVERITY_COLUMNS)
    if table.empty:
        return table
    table["_rank"] = table["Severity"].map(_SEVERITY_RANK)
    table = table.sort_values(["_rank", "Affected"], ascending=[False, False], kind="stable")
    return table.drop(columns="_rank").reset_index(drop=True)


def _empty_dataset_result(df, unified):
    if unified is None:
        unified = combine_quality_results({}, index=df.index)
    ann = _Annotations(unified)
    ann.mark_review()
    ann.finalize({}, {}, empty=True)

    dims = [
        _dimension(DIM_COMPLETENESS, None, 0, 0, 1, "cells"),
        _dimension(DIM_VALIDITY, None, 0, 0, 3, "cells"),
        _dimension(DIM_CONSISTENCY, None, 0, 0, 1, "cells"),
        _dimension(DIM_UNIQUENESS, None, 0, 0, 1, "rows"),
    ]
    return QualityScoreResult(
        overall_score=None,
        raw_score=None,
        rating=RATING_NOT_SCORED,
        cap_limit=None,
        cap_applied=False,
        cap_reason=None,
        dimensions=_dimension_table(dims, {}),
        severity_summary=_severity_table([]),
        findings=pd.DataFrame(
            {"Severity": ann.severity, "Score Treatment": ann.treatment}, index=unified.index
        ),
        unified=unified,
        unscored_impact=None,
        notes=["The dataset is empty, so no quality score was calculated."],
    )


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def calculate_quality_score(df, quality_results, unified=None):
    """Calculate the 0-100 data quality score and classify severity.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset the results were computed on (used for row, cell and
        column counts and for the consistency check). Not modified.
    quality_results : dict
        Source name -> detector result, as built in app.py
        (``SRC_MISSING``, ``SRC_DUPLICATES``, ``SRC_INVALID``,
        ``SRC_DATA_TYPE``, ``SRC_FORMAT``, ``SRC_CONSISTENCY``,
        ``SRC_IQR``). A key that is absent or ``None`` = the check did not
        run; an empty DataFrame = it ran and found nothing. ML source keys
        are accepted but never scored.
    unified : pandas.DataFrame, optional
        The unified findings table (output of ``build_unified_table``, may
        include IQR and ML rows). If omitted it is built from
        ``quality_results``. Severity and Score Treatment are returned for
        every row of this table.

    Returns
    -------
    QualityScoreResult

    Raises
    ------
    TypeError / ValueError
        Invalid ``df``, ``quality_results`` or ``unified``, or an unknown
        source key. A malformed individual result does NOT raise: that check
        is reported in ``notes`` and left out of the score.
    """
    _validate_inputs(df, quality_results, unified)
    n_rows, n_cols = df.shape
    if n_rows == 0 or n_cols == 0:
        return _empty_dataset_result(df, unified)

    n_cells = n_rows * n_cols
    unified, unusable = _prepare_unified(df, quality_results, unified)
    ran = {
        source: isinstance(quality_results.get(source), pd.DataFrame) and source not in unusable
        for source in _SCORED_SOURCES
    }
    ann = _Annotations(unified)
    ann.mark_review()
    groups = []
    notes = []

    # ---- Completeness --------------------------------------------------
    missing_cells, missing_by_column = None, {}
    if ran[SRC_MISSING]:
        try:
            missing_cells, missing_by_column, found = _step_missing(
                quality_results[SRC_MISSING], n_rows, n_cols, ann)
            groups.extend(found)
        except (ValueError, TypeError, KeyError) as error:
            unusable[SRC_MISSING] = str(error)
            ran[SRC_MISSING] = False
            missing_cells, missing_by_column = None, {}
    missing_known = missing_cells is not None
    non_missing_total = n_cells - missing_cells if missing_known else n_cells

    def non_missing(column):
        if not missing_known:
            return n_rows
        return n_rows - min(missing_by_column.get(column, 0), n_rows)

    # ---- Validity -------------------------------------------------------
    validity_issues = [_ISSUE_OF[s] for s in _CELL_SOURCES if ran[s]]
    validity = None
    if validity_issues:
        validity = _step_validity(ann, validity_issues, non_missing, missing_known)
        groups.extend(validity["groups"])

    # ---- Consistency ----------------------------------------------------
    consistency_bad = None
    if ran[SRC_CONSISTENCY]:
        try:
            consistency_bad, found = _step_consistency(
                df, quality_results[SRC_CONSISTENCY], non_missing, missing_known, ann)
            groups.extend(found)
        except (ValueError, TypeError, KeyError) as error:
            unusable[SRC_CONSISTENCY] = str(error)
            ran[SRC_CONSISTENCY] = False
            consistency_bad = None

    # ---- Uniqueness -----------------------------------------------------
    redundant = None
    if ran[SRC_DUPLICATES]:
        try:
            redundant, found = _step_uniqueness(quality_results[SRC_DUPLICATES], n_rows, ann)
            groups.extend(found)
        except (ValueError, TypeError, KeyError) as error:
            unusable[SRC_DUPLICATES] = str(error)
            ran[SRC_DUPLICATES] = False
            redundant = None

    dims = [
        _dimension(DIM_COMPLETENESS, missing_cells, n_cells,
                   1 if missing_known else 0, 1, "cells"),
        _dimension(DIM_VALIDITY, validity["counted"] if validity else None,
                   non_missing_total, len(validity_issues), 3, "cells"),
        _dimension(DIM_CONSISTENCY, consistency_bad, non_missing_total,
                   1 if consistency_bad is not None else 0, 1, "cells"),
        _dimension(DIM_UNIQUENESS, redundant, n_rows,
                   1 if redundant is not None else 0, 1, "rows"),
    ]

    # ---- Overall score --------------------------------------------------
    assessed = [d for d in dims if d["score"] is not None]
    weight_sum = sum(WEIGHTS[d["name"]] for d in assessed)
    weights_used = {d["name"]: WEIGHTS[d["name"]] / weight_sum for d in assessed} if assessed else {}

    overall_score = raw_score = cap_limit = cap_reason = None
    cap_applied = False
    severity_table = _severity_table(groups)

    if assessed:
        raw = sum(weights_used[d["name"]] * d["score"] for d in assessed)
        raw_score = _round1(max(0.0, min(100.0, raw)))

        worst = max((_SEVERITY_RANK[g["Severity"]] for g in groups), default=0)
        if worst >= _SEVERITY_RANK[SEV_CRITICAL]:
            cap_limit, worst_name = CAP_CRITICAL, SEV_CRITICAL
        elif worst >= _SEVERITY_RANK[SEV_HIGH]:
            cap_limit, worst_name = CAP_HIGH, SEV_HIGH
        else:
            worst_name = None

        capped = min(raw, cap_limit) if cap_limit is not None else raw
        overall_score = _round1(max(0.0, min(100.0, capped)))
        cap_applied = cap_limit is not None and overall_score < raw_score
        if cap_applied:
            culprits = [
                f"{g['Issue Type']} in {g['Column']}" if g["Column"] != WHOLE_ROW
                else g["Issue Type"]
                for g in groups if g["Severity"] == worst_name
            ]
            shown = ", ".join(culprits[:3]) + (" and more" if len(culprits) > 3 else "")
            cap_reason = (
                f"The score is limited to {cap_limit} because at least one issue group has "
                f"{worst_name} severity ({shown}). Score before the limit: {raw_score}."
            )
            notes.append(cap_reason)
    rating = get_rating(overall_score)

    # ---- Notes: checks that did not run / could not be used ----------------
    for source in _SCORED_SOURCES:
        if source in unusable:
            notes.append(
                f"{_SOURCE_LABELS[source]} could not be used for the score: "
                f"{unusable[source].rstrip('.')}."
            )
        elif not ran[source]:
            notes.append(
                f"The {_SOURCE_LABELS[source].lower()} check did not run, "
                "so it is not part of the score."
            )
    for source, reason in unusable.items():
        if source not in _SCORED_SOURCES:                # IQR and ML sources
            notes.append(
                f"{_SOURCE_LABELS[source]} could not be added to the findings table: "
                f"{reason.rstrip('.')}."
            )
    for d in dims:
        if d["status"] == STATUS_NOT_ASSESSED:
            if d["run"] > 0 and (d["total"] or 0) <= 0:
                reason = "there are no non-missing cells to check"
            else:
                reason = "none of its checks ran"
            notes.append(f"{d['name']} was not assessed: {reason}.")
        elif d["status"] == STATUS_PARTIAL:
            notes.append(
                f"{d['name']} is partial: only {d['run']} of {d['of']} of its checks ran."
            )
    if assessed and len(assessed) < len(dims):
        notes.append("Weights of the assessed dimensions were rescaled to total 100%.")
    if not assessed:
        notes.append("No dimension could be assessed, so no quality score was calculated.")
    if not missing_known and any(d["status"] != STATUS_NOT_ASSESSED
                                 for d in dims if d["name"] in (DIM_VALIDITY, DIM_CONSISTENCY)):
        notes.append(
            "Validity and Consistency use all cells as their base because the "
            "missing-value check did not run."
        )
    if validity and validity["blank"]:
        notes.append(
            f"{validity['blank']} validity finding(s) are on blank cells and are counted once, "
            "under Completeness."
        )

    # ---- Findings that cannot be located: shown, not scored, bounded impact ---
    unscored_impact = None
    unidentifiable = validity["unidentifiable"] if validity else 0
    if unidentifiable:
        validity_dim = next(d for d in dims if d["name"] == DIM_VALIDITY)
        message = (
            f"{unidentifiable} validity finding(s) could not be tied to an exact row "
            "(for example the row label occurs more than once). They are shown but NOT "
            "counted in the score."
        )
        if validity_dim["score"] is not None:
            points = min(100.0 * unidentifiable / validity_dim["total"], validity_dim["score"])
            overall_points = weights_used[DIM_VALIDITY] * points
            unscored_impact = {
                "unidentifiable_findings": int(unidentifiable),
                "max_validity_points": _round1(points),
                "max_overall_points": _round1(overall_points),
            }
            message += (
                f" If every one were a separate bad cell, Validity would be at most "
                f"{unscored_impact['max_validity_points']} points lower and the overall score "
                f"at most {unscored_impact['max_overall_points']} points lower (before any "
                "severity limit; their severity could also be higher than shown)."
            )
        notes.append(message)

    ann.finalize(ran, unusable)
    findings = pd.DataFrame(
        {"Severity": ann.severity, "Score Treatment": ann.treatment}, index=unified.index
    )

    return QualityScoreResult(
        overall_score=overall_score,
        raw_score=raw_score,
        rating=rating,
        cap_limit=cap_limit,
        cap_applied=cap_applied,
        cap_reason=cap_reason,
        dimensions=_dimension_table(dims, weights_used),
        severity_summary=severity_table,
        findings=findings,
        unified=unified,
        unscored_impact=unscored_impact,
        notes=notes,
    )


def add_severity_columns(unified, result):
    """Display copy of ``unified`` with ``Severity`` and ``Score Treatment``.

    ``unified`` must be the table the result was calculated for (same index).
    The original table, and its ``UNIFIED_COLUMNS`` schema, are not changed.
    """
    _validate_unified(unified)
    if not isinstance(result, QualityScoreResult):
        raise TypeError("result must be a QualityScoreResult.")
    if not unified.index.equals(result.findings.index):
        raise ValueError(
            "unified does not match the table the score was calculated for "
            "(its index differs from result.findings)."
        )
    table = unified.copy()
    table["Severity"] = result.findings["Severity"].to_numpy(dtype=object)
    table["Score Treatment"] = result.findings["Score Treatment"].to_numpy(dtype=object)
    return table
