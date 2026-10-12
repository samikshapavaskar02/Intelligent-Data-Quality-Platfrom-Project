"""Data cleaning (project roadmap Phase 12, Steps 20-21).

Applies the cleaning operations the user selected to a SEPARATE COPY of the
dataset and reports exactly what changed. The input DataFrame is never
modified.

Supported operations
--------------------
1. Duplicate rows   : remove exact duplicates, keeping the first occurrence.
2. Invalid values   : for the columns that have a validity rule, either set
                      invalid values to missing or clip numbers to the allowed
                      range (text that is not a number is always set to
                      missing, because it cannot be clipped).
3. Missing values   : either fill (median for numeric columns, most frequent
                      value for other columns) or remove rows that contain a
                      missing value. Blank and whitespace-only text counts as
                      missing, the same as in modules/missing_values.py.

Order of the steps
------------------
Duplicates -> invalid values -> missing values. Duplicates are removed first
so they match what the duplicate check reported on the uploaded data and so
medians and most frequent values are not skewed by repeated rows. Invalid
values are handled before missing values, so a value set to missing can then
be filled by the missing-value step.

What this module deliberately does NOT do
-----------------------------------------
* It never removes or changes a record because IQR, Isolation Forest or LOF
  flagged it. Those results are review suggestions only, and they are not
  inputs of this module.
* It does not detect anything itself. Duplicates and invalid values are found
  with the existing ``get_duplicate_count`` and ``detect_invalid_values``.
* It does not fix data type, format or consistency problems (not in scope of
  this step).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from modules.duplicates import get_duplicate_count
from modules.invalid_values import detect_invalid_values

# ----------------------------------------------------------------------
# Option values
# ----------------------------------------------------------------------
INVALID_KEEP = "keep"
INVALID_MISSING = "missing"
INVALID_CLIP = "clip"

MISSING_KEEP = "keep"
MISSING_FILL = "fill"
MISSING_DROP = "drop"

_INVALID_ACTIONS = (INVALID_KEEP, INVALID_MISSING, INVALID_CLIP)
_MISSING_ACTIONS = (MISSING_KEEP, MISSING_FILL, MISSING_DROP)

SUMMARY_COLUMNS = ["Operation", "Setting", "Effect"]
CHANGE_COLUMNS = ["Original Row", "Column", "Operation", "Before", "After"]

SHOWN_MISSING = "(missing)"
SHOWN_BLANK = "(blank)"


# ----------------------------------------------------------------------
# Result object
# ----------------------------------------------------------------------
@dataclass
class CleaningResult:
    """Everything produced by ``clean_dataset``.

    Attributes
    ----------
    cleaned : pd.DataFrame
        The cleaned COPY. It keeps the original index labels.
    summary : pd.DataFrame
        One row per selected operation (``SUMMARY_COLUMNS``).
    changes : pd.DataFrame
        One row per changed cell (``CHANGE_COLUMNS``). ``Original Row`` is the
        1-based position in the uploaded dataset.
    removed_rows : pd.DataFrame
        Rows that were removed, with their ``Original Row`` and a ``Reason``.
    notes : list[str]
        Explanations and limitations for the user.
    rows_before, rows_after, cells_changed : int
    """

    cleaned: pd.DataFrame
    summary: pd.DataFrame
    changes: pd.DataFrame
    removed_rows: pd.DataFrame
    rows_before: int
    rows_after: int
    cells_changed: int
    notes: list = field(default_factory=list)

    @property
    def has_changes(self):
        return self.rows_before != self.rows_after or self.cells_changed > 0


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _missing_mask(frame):
    """True where a cell is missing: NaN/None/NA or blank / whitespace-only text."""
    mask = frame.isna()

    for column in frame.columns:

        series = frame[column]

        if (
            pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_datetime64_any_dtype(series)
        ):
            continue

        blank = series.astype(object).map(
            lambda value: isinstance(value, str) and value.strip() == ""
        )

        mask[column] = (mask[column].to_numpy() | blank.to_numpy(dtype=bool))

    return mask


def _is_numeric(series):
    return (
        pd.api.types.is_numeric_dtype(series)
        and not pd.api.types.is_bool_dtype(series)
    )


def _needs_float(values):
    """True if integer storage cannot hold these values (NaN or fractions)."""
    for value in values:
        if pd.isna(value):
            return True
        if float(value) != round(float(value)):
            return True
    return False


def _removed_frame(working, positions, mask, reason):
    """The rows selected by ``mask``, with their original row numbers."""
    data = working[mask].reset_index(drop=True)
    front = pd.DataFrame({"Original Row": positions[mask]})
    back = pd.DataFrame({"Reason": [reason] * len(data)})
    return pd.concat([front, data, back], axis=1)


# ----------------------------------------------------------------------
# One function per operation
# ----------------------------------------------------------------------
def _correct_invalid(working, positions, rules, action, notes):
    """Correct invalid values in ``working`` (a private copy, changed in place).

    Uses the existing ``detect_invalid_values`` on ``working``; its ``Row``
    column is the 1-based position inside ``working``.
    Returns (change log, number of corrected cells).
    """
    findings = detect_invalid_values(working, rules)

    if findings.empty:
        return pd.DataFrame(columns=CHANGE_COLUMNS), 0

    edits = {}

    for row, column, value in zip(
        findings["Row"], findings["Column"], findings["Value"]
    ):
        numeric = pd.to_numeric(value, errors="coerce")
        minimum, maximum = rules[column][0], rules[column][1]

        if action == INVALID_CLIP and not pd.isna(numeric):
            after = min(max(numeric, minimum), maximum)
            operation = "Clipped invalid value to allowed range"
        else:
            after = np.nan
            operation = "Set invalid value to missing"

        edits.setdefault(column, []).append(
            (int(row) - 1, value, after, operation)
        )

    logged = []

    for column, column_edits in edits.items():

        # Whole-number columns cannot hold NaN or fractions.
        if (
            pd.api.types.is_integer_dtype(working[column])
            and _needs_float([edit[2] for edit in column_edits])
        ):
            working[column] = working[column].astype("float64")

        # Text / category columns can reject a number (strict string dtype in
        # newer pandas, categories that do not include it), so they are held
        # as plain objects while a corrected value is written.
        elif not pd.api.types.is_numeric_dtype(working[column]):
            working[column] = working[column].astype(object)

        location = working.columns.get_loc(column)

        for position, before, after, operation in column_edits:

            try:
                working.iat[position, location] = after
            except (TypeError, ValueError):
                notes.append(
                    f"A value in column '{column}' (original row "
                    f"{int(positions[position])}) could not be corrected "
                    "and was left unchanged."
                )
                continue

            logged.append({
                "Original Row": int(positions[position]),
                "Column": column,
                "Operation": operation,
                "Before": str(before),
                "After": SHOWN_MISSING if pd.isna(after) else str(after)
            })

    return pd.DataFrame(logged, columns=CHANGE_COLUMNS), len(logged)


def _fill_missing(working, positions, numeric_like, notes):
    """Fill missing cells in ``working`` (a private copy, changed in place).

    Numeric columns use the median; every other column uses its most frequent
    value. A column that is entirely missing cannot be filled and is reported.
    Returns (change log, number of filled cells).
    """
    mask = _missing_mask(working)
    frames = []
    filled = 0

    for column in working.columns:

        column_mask = mask[column].to_numpy()

        if not column_mask.any():
            continue

        original = working[column]
        series = original
        numeric = _is_numeric(series)

        # Text columns that hold only numbers are treated as numeric only
        # when the caller says so (the columns that have a validity rule).
        if not numeric and column in numeric_like:

            converted = pd.to_numeric(series, errors="coerce")

            if (~column_mask).any() and converted[~column_mask].notna().all():
                series, numeric = converted, True
                notes.append(
                    f"Column '{column}' was stored as text but holds only "
                    "numbers, so it was converted to numbers before filling."
                )

        present = series[~column_mask]

        if present.empty:
            notes.append(
                f"Column '{column}' has no values to fill from, so its "
                "missing cells were left as they are."
            )
            continue

        if numeric:
            fill_value = present.median()
            method = "median"
        else:
            try:
                modes = present.mode()
            except TypeError:
                # Values that cannot be sorted together (mixed types).
                modes = present.value_counts().index.to_series()

            if modes.empty:
                continue

            fill_value = modes.iloc[0]
            method = "most frequent value"

        try:
            working[column] = series.mask(column_mask, fill_value)
        except (TypeError, ValueError):
            notes.append(
                f"Column '{column}' could not be filled and was left "
                "unchanged."
            )
            continue

        before = [
            SHOWN_BLANK if isinstance(value, str) else SHOWN_MISSING
            for value in original[column_mask]
        ]

        frames.append(pd.DataFrame({
            "Original Row": positions[column_mask],
            "Column": column,
            "Operation": f"Filled missing value ({method})",
            "Before": before,
            "After": str(fill_value)
        }, columns=CHANGE_COLUMNS))

        filled += int(column_mask.sum())

    if not frames:
        return pd.DataFrame(columns=CHANGE_COLUMNS), 0

    return pd.concat(frames, ignore_index=True), filled


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def clean_dataset(
    df,
    remove_duplicates=False,
    invalid_action=INVALID_KEEP,
    invalid_rules=None,
    missing_action=MISSING_KEEP
):
    """Apply the selected cleaning operations to a copy of ``df``.

    Parameters
    ----------
    df : pandas.DataFrame
        The uploaded dataset. It is NOT modified.
    remove_duplicates : bool
        Remove exact duplicate rows (first occurrence kept).
    invalid_action : {"keep", "missing", "clip"}
        What to do with values outside ``invalid_rules``.
    invalid_rules : dict, optional
        ``{column: (minimum, maximum)}``, the same rules the invalid-value
        check uses. Rules for columns that are not in ``df`` are ignored.
    missing_action : {"keep", "fill", "drop"}
        Leave missing values, fill them, or remove rows that contain one.

    Returns
    -------
    CleaningResult

    Raises
    ------
    TypeError / ValueError
        ``df`` is not a DataFrame, an option is unknown, or the dataset has
        duplicate column names.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"df must be a pandas DataFrame, got {type(df).__name__}."
        )

    if invalid_action not in _INVALID_ACTIONS:
        raise ValueError(
            f"Unknown invalid_action {invalid_action!r}. "
            f"Choose from {_INVALID_ACTIONS}."
        )

    if missing_action not in _MISSING_ACTIONS:
        raise ValueError(
            f"Unknown missing_action {missing_action!r}. "
            f"Choose from {_MISSING_ACTIONS}."
        )

    if not df.columns.is_unique:
        raise ValueError(
            "Cleaning needs unique column names, but this dataset has "
            "repeated column names."
        )

    rules = {
        column: limits
        for column, limits in (invalid_rules or {}).items()
        if column in df.columns
    }

    # Everything below works on this private copy.
    working = df.copy()
    positions = np.arange(1, len(working) + 1)

    notes = []
    summary = []
    change_frames = []
    removed_frames = []

    if working.empty:

        notes.append("The dataset is empty, so there is nothing to clean.")

    else:

        # 1. Duplicate rows
        if remove_duplicates:

            duplicate_count = int(get_duplicate_count(working))

            if duplicate_count > 0:

                duplicate_mask = working.duplicated(keep="first").to_numpy()

                removed_frames.append(_removed_frame(
                    working,
                    positions,
                    duplicate_mask,
                    "Exact duplicate of an earlier row"
                ))

                working = working[~duplicate_mask].copy()
                positions = positions[~duplicate_mask]

            summary.append({
                "Operation": "Remove duplicate rows",
                "Setting": "Keep the first occurrence",
                "Effect": f"{duplicate_count} row(s) removed"
            })

        # 2. Invalid values
        corrected = 0

        if invalid_action != INVALID_KEEP:

            if not rules:

                notes.append(
                    "No invalid-value rules apply to the columns in this "
                    "dataset, so no invalid values were corrected."
                )

            else:

                frame, corrected = _correct_invalid(
                    working, positions, rules, invalid_action, notes
                )

                if corrected:
                    change_frames.append(frame)

                summary.append({
                    "Operation": "Correct invalid values",
                    "Setting": (
                        "Clip numbers to the allowed range"
                        if invalid_action == INVALID_CLIP
                        else "Set to missing"
                    ),
                    "Effect": f"{corrected} cell(s) corrected"
                })

        # 3. Missing values
        filled = 0

        if missing_action == MISSING_FILL:

            frame, filled = _fill_missing(
                working, positions, set(rules), notes
            )

            if filled:
                change_frames.append(frame)

            summary.append({
                "Operation": "Handle missing values",
                "Setting": "Fill (median / most frequent value)",
                "Effect": f"{filled} cell(s) filled"
            })

        elif missing_action == MISSING_DROP:

            row_mask = _missing_mask(working).any(axis=1).to_numpy()
            dropped = int(row_mask.sum())

            if dropped:

                removed_frames.append(_removed_frame(
                    working,
                    positions,
                    row_mask,
                    "Contains a missing or blank value"
                ))

                working = working[~row_mask].copy()
                positions = positions[~row_mask]

            summary.append({
                "Operation": "Handle missing values",
                "Setting": "Remove rows with a missing value",
                "Effect": f"{dropped} row(s) removed"
            })

        # Corrections can make two different rows identical.
        if remove_duplicates and (corrected or filled) and len(working):

            new_duplicates = int(get_duplicate_count(working))

            if new_duplicates > 0:
                notes.append(
                    f"{new_duplicates} row(s) became exact duplicates after "
                    "the corrections. They were not removed again; run the "
                    "cleaning once more on the cleaned data if needed."
                )

        if working.empty:
            notes.append(
                "All rows would be removed by the selected options, so the "
                "cleaned dataset is empty."
            )

    changes = (
        pd.concat(change_frames, ignore_index=True)
        if change_frames
        else pd.DataFrame(columns=CHANGE_COLUMNS)
    )

    removed_rows = (
        pd.concat(removed_frames, ignore_index=True)
        if removed_frames
        else pd.DataFrame()
    )

    return CleaningResult(
        cleaned=working,
        summary=pd.DataFrame(summary, columns=SUMMARY_COLUMNS),
        changes=changes,
        removed_rows=removed_rows,
        rows_before=len(df),
        rows_after=len(working),
        cells_changed=len(changes),
        notes=notes
    )
