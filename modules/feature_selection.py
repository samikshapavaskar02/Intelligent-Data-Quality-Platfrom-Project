"""Feature selection for anomaly detection (project PDF, Phase 7 / Step 15).

Responsibility
--------------
Decide WHICH columns of the uploaded DataFrame are meaningful inputs for the
anomaly detectors (Isolation Forest and LOF). The PDF says not to use things
like Name, Email or Customer ID as anomaly features "unless there is a
specific reason", and to prefer meaningful attributes such as Age, Income,
Purchase Amount or Credit Score.

The function only returns a list of column names. It never modifies, drops
or re-indexes the DataFrame, so row identity is untouched. The selected
columns are meant to be passed to
``modules.preprocessing.preprocess_for_ml(df, columns=...)``.

Rules (the first matching rule excludes a column)
-------------------------------------------------
1. Excluded by the caller (``exclude_columns``).
2. Datetime columns (preprocessing does not use them either).
3. Columns where every value is missing (blank strings count as missing for
   text columns, NaN / infinity for numeric columns).
4. ``force_include`` columns are KEPT from here on (the caller has a
   specific reason), so the heuristic rules below are skipped for them.
5. Identifier-like column NAMES: ID / UUID / GUID / idx tokens, exact names
   such as "Index", "Unnamed: 0", "S.No", "Serial No", and name / email /
   phone / PIN / zip / address / URL style columns. Date-like names
   ("date", "dob", "timestamp") are excluded for non-numeric columns only.
6. Constant columns (one distinct value): they carry no information.
7. Numeric columns that are a plain integer sequence with a constant step
   (1, 2, 3, ... or 1001, 1002, ...): row numbers / identifiers.
8. Text / categorical columns that look like identifiers or free text:
   more distinct values than ``max_categories``; every value different
   (with at least 5 values); or too many distinct values relative to the
   number of rows (with at least 20 values).

An excluded column is not "bad data". It is only not a useful anomaly
feature. Use ``force_include`` if you do have a reason to use it.

What this module does NOT do
----------------------------
* No data-quality checks (missing / duplicate / invalid values).
* No imputation, encoding or scaling (that is ``modules/preprocessing.py``).
* No anomaly detection and no UI code.
"""

import numbers
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from modules.preprocessing import identify_column_types

DEFAULT_MAX_CATEGORIES = 50
DEFAULT_MAX_UNIQUE_RATIO = 0.5

MIN_ROWS_SEQUENCE = 5       # shortest numeric run treated as a row-number sequence
MIN_ROWS_ALL_UNIQUE = 5     # "every value different" rule needs at least this many values
MIN_ROWS_UNIQUE_RATIO = 20  # "too many distinct values per row" rule needs at least this many

# Splits "CustomerID" -> Customer, ID ; "customer_id" -> customer, id ; "PINCode" -> PIN, Code
_TOKEN_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")

# Whole tokens that mark an identifier / personal-detail column.
_IDENTIFIER_TOKENS = {
    "id", "ids", "uuid", "guid", "idx",
    "name", "email", "phone", "mobile", "telephone",
    "pin", "pincode", "zip", "zipcode", "postal", "postcode",
    "address", "url", "ssn", "aadhaar", "aadhar",
}

# Whole normalised names (lower case, letters and digits only).
_IDENTIFIER_NAMES = {
    "index", "srno", "sno", "slno", "serialno", "serialnumber",
    "rownumber", "rownum", "rowno", "recordnumber", "recordno",
}

# Date-like names: only used for non-numeric columns.
_DATE_TOKENS = {"date", "dob", "timestamp", "datetime"}


@dataclass
class FeatureSelectionResult:
    """Output of ``select_features``.

    Attributes
    ----------
    selected_columns : list
        Columns to use as anomaly features, in the original column order.
    numeric_columns, categorical_columns : list
        The selected columns split by type (same rules as
        ``modules.preprocessing.identify_column_types``).
    excluded_columns : pd.DataFrame
        One row per CANDIDATE column that was not selected, with the columns
        ``column`` and ``reason``. Columns that were never candidates (when
        ``columns=`` was given) do not appear here.
    """

    selected_columns: list
    numeric_columns: list
    categorical_columns: list
    excluded_columns: pd.DataFrame

    @property
    def n_selected(self):
        return len(self.selected_columns)


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def _as_list(value):
    """None -> [], a single label -> [label], list-like -> list."""
    if value is None:
        return []
    if isinstance(value, str) or not pd.api.types.is_list_like(value):
        return [value]
    return list(value)


def _is_int(value):
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _tokens(name):
    return [t.lower() for t in _TOKEN_RE.findall(str(name))]


def _normalised(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _finite_values(series):
    """Finite numeric values of a numeric column, in the original row order."""
    values = series.astype("float64").to_numpy()
    return values[np.isfinite(values)]


def _clean_text(series):
    """Text series where blank / whitespace-only strings are missing.

    Same idea as the cleaning step inside ``modules/preprocessing.py``, so a
    column that preprocessing would see as empty is also empty here.
    """
    text = series.astype("string")
    blank = (text.str.strip() == "").fillna(False)
    return text.mask(blank, pd.NA)


def _name_reason(name, is_numeric):
    tokens = _tokens(name)
    normalised = _normalised(name)
    if (any(t in _IDENTIFIER_TOKENS for t in tokens)
            or normalised in _IDENTIFIER_NAMES
            or re.fullmatch(r"unnamed\d*", normalised)):
        return "identifier-like column name (ID / name / contact detail)"
    if not is_numeric and any(t in _DATE_TOKENS for t in tokens):
        return "date-like column name (not a useful anomaly feature)"
    return None


def _is_sequential(values):
    """True for a unique integer run with a constant step, e.g. 1001, 1002, ..."""
    if values.size < MIN_ROWS_SEQUENCE:
        return False
    if not np.all(values == np.round(values)):
        return False
    steps = np.diff(values)
    return bool(steps[0] != 0 and np.allclose(steps, steps[0]))


def _numeric_reason(values):
    if np.unique(values).size <= 1:
        return "constant column (one distinct value)"
    if _is_sequential(values):
        return "sequential integer values (looks like a row number / identifier)"
    return None


def _text_reason(text, n_values, max_categories, max_unique_ratio):
    n_unique = int(text.nunique())
    if n_unique <= 1:
        return "constant column (one distinct value)"
    if n_unique > max_categories:
        return (f"too many distinct values ({n_unique} > {max_categories}); "
                "looks like an identifier or free text")
    if n_values >= MIN_ROWS_ALL_UNIQUE and n_unique == n_values:
        return "every value is different (looks like an identifier or free text)"
    if n_values >= MIN_ROWS_UNIQUE_RATIO and n_unique / n_values > max_unique_ratio:
        return (f"too many distinct values relative to rows ({n_unique} of {n_values}); "
                "looks like an identifier or free text")
    return None


def _exclusion_reason(series, name, is_numeric, is_datetime, forced,
                      max_categories, max_unique_ratio):
    """Return the reason a column should be excluded, or None to keep it."""
    if is_datetime:
        return "datetime column (not used as an ML feature)"

    if is_numeric:
        values = _finite_values(series)
        n_values = int(values.size)
        text = None
    else:
        text = _clean_text(series)
        n_values = int(text.notna().sum())
        values = None

    if n_values == 0:
        return "all values are missing"

    if forced:
        return None  # the caller has a specific reason to use this column

    reason = _name_reason(name, is_numeric)
    if reason:
        return reason

    if is_numeric:
        return _numeric_reason(values)
    return _text_reason(text, n_values, max_categories, max_unique_ratio)


def _validate_inputs(df, max_categories, max_unique_ratio):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError("The DataFrame is empty (no rows or no columns).")
    if df.columns.has_duplicates:
        duplicated = [str(c) for c in df.columns[df.columns.duplicated()].unique()]
        raise ValueError(f"The DataFrame has duplicate column names: {duplicated}. "
                         "Rename them before feature selection.")
    if not _is_int(max_categories) or max_categories < 1:
        raise ValueError(f"max_categories must be an integer >= 1, got {max_categories!r}.")
    if (isinstance(max_unique_ratio, bool)
            or not isinstance(max_unique_ratio, numbers.Real)
            or not 0 < max_unique_ratio <= 1):
        raise ValueError(f"max_unique_ratio must be a number in (0, 1], got {max_unique_ratio!r}.")


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def select_features(df, columns=None, exclude_columns=None, force_include=None,
                    max_categories=DEFAULT_MAX_CATEGORIES,
                    max_unique_ratio=DEFAULT_MAX_UNIQUE_RATIO):
    """Choose the columns that should be used as anomaly-detection features.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset. It is not modified.
    columns : list, optional
        Candidate columns. Default: all columns. Columns that are not
        candidates are neither selected nor listed in ``excluded_columns``.
    exclude_columns : list, optional
        Columns to exclude no matter what.
    force_include : list, optional
        Columns to keep even if a heuristic rule would exclude them (for
        example a numeric ID that you deliberately want to use). Datetime
        and all-missing columns are still excluded because they cannot be
        used as features. Columns listed here are added to the candidates
        even if ``columns`` does not contain them.
    max_categories : int
        Text columns with more distinct values than this are excluded.
    max_unique_ratio : float in (0, 1]
        Text columns with at least 20 values are excluded when
        ``distinct values / values`` is above this ratio.

    Returns
    -------
    FeatureSelectionResult
        ``selected_columns`` may be empty; the caller decides what to do.

    Raises
    ------
    TypeError / ValueError
        For an invalid DataFrame, unknown column names, duplicate column
        names, a column that is both forced in and excluded, or invalid
        parameters.
    """
    _validate_inputs(df, max_categories, max_unique_ratio)

    exclude = _as_list(exclude_columns)
    forced = _as_list(force_include)

    unknown = [c for c in _as_list(columns) + exclude + forced if c not in df.columns]
    if unknown:
        raise ValueError(f"Columns not found in the DataFrame: {unknown}")
    conflict = [c for c in forced if c in exclude]
    if conflict:
        raise ValueError(f"Columns cannot be both in force_include and exclude_columns: {conflict}")

    if columns is None:
        candidate_set = set(df.columns)
    else:
        candidate_set = set(_as_list(columns)) | set(forced)
        if not candidate_set:
            raise ValueError("columns is empty: there is nothing to select from.")
    candidates = [c for c in df.columns if c in candidate_set]

    exclude_set = set(exclude)
    forced_set = set(forced)

    numeric, _categorical, ignored = identify_column_types(df, candidates)
    numeric_set = set(numeric)
    ignored_set = set(ignored)

    selected = []
    excluded_rows = []
    for col in candidates:
        if col in exclude_set:
            reason = "excluded by caller request"
        else:
            reason = _exclusion_reason(
                df[col], col,
                is_numeric=col in numeric_set,
                is_datetime=col in ignored_set,
                forced=col in forced_set,
                max_categories=max_categories,
                max_unique_ratio=max_unique_ratio,
            )
        if reason is None:
            selected.append(col)
        else:
            excluded_rows.append((col, reason))

    selected_set = set(selected)
    numeric_selected, categorical_selected, _ = identify_column_types(df, selected)
    # identify_column_types keeps the order of `selected`; the sets are only
    # used to make the intent explicit.
    assert set(numeric_selected) | set(categorical_selected) == selected_set

    return FeatureSelectionResult(
        selected_columns=selected,
        numeric_columns=numeric_selected,
        categorical_columns=categorical_selected,
        excluded_columns=pd.DataFrame(excluded_rows, columns=["column", "reason"]),
    )
