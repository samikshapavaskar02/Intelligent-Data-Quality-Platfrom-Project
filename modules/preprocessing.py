"""
ML preprocessing for anomaly detection.

Responsibility
--------------
Turn a pandas DataFrame (the uploaded dataset) into a purely numeric matrix
that Isolation Forest and LOF can use.

Steps (in order):
    1. Identify numerical and categorical columns.
    2. REPORT the original missing values (before anything is filled).
    3. Impute: numerical -> median, categorical -> mode.
    4. Encode categorical columns (one-hot or frequency encoding).
    5. Scale numerical columns (robust / standard / none).

Design rules
------------
* The original DataFrame is never modified.
* The row index of the input is preserved on the output matrix, so every row
  of ``result.X`` can be traced back to the original dataset row.
* This module does NOT do data-quality checks (duplicates, invalid values,
  spelling consistency). It only prepares data for ML. It does not import
  any other project module.

Typical use
-----------
    from modules.preprocessing import preprocess_for_ml

    result = preprocess_for_ml(df)
    result.X                # transformed ML matrix (DataFrame)
    result.missing_report   # original missing values, before imputation
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

VALID_SCALERS = ("robust", "standard", "none")
VALID_ENCODINGS = ("onehot", "frequency")


@dataclass
class PreprocessingResult:
    """Everything produced by ``preprocess_for_ml``.

    Attributes
    ----------
    X : pd.DataFrame
        The transformed ML matrix. Its index is the ORIGINAL index of the
        input DataFrame (same order, same labels).
    feature_names : list[str]
        Column names of ``X``.
    feature_source : dict[str, str]
        Maps each column of ``X`` to the original column it came from
        (e.g. ``"City_Pune" -> "City"``). Used later for explanations.
    numeric_columns, categorical_columns : list[str]
        Original columns that ended up in the matrix, by type.
    missing_report : pd.DataFrame
        Original missing values BEFORE imputation, one row per used column.
    dropped_columns : pd.DataFrame
        Columns that were NOT used, with the reason.
    scaler, encoding : str
        The options that were used.
    """

    X: pd.DataFrame
    feature_names: list
    feature_source: dict
    numeric_columns: list
    categorical_columns: list
    missing_report: pd.DataFrame
    dropped_columns: pd.DataFrame
    scaler: str
    encoding: str


# ----------------------------------------------------------------------
# Step 1: column types
# ----------------------------------------------------------------------
def identify_column_types(df, columns=None):
    """Split columns into numerical, categorical and ignored (datetime).

    Rules:
        * numeric dtype (int, float, nullable Int/Float) -> numerical
        * boolean -> categorical (it is a yes/no category, not a quantity)
        * datetime -> ignored (not a usable ML feature as-is)
        * everything else (text, category, mixed) -> categorical

    This works for both the old ``object`` text dtype and the newer pandas
    ``str`` dtype.

    Returns
    -------
    (numeric, categorical, ignored) : three lists of column names
    """
    if columns is None:
        columns = list(df.columns)

    numeric, categorical, ignored = [], [], []
    for col in columns:
        dtype = df[col].dtype
        if pd.api.types.is_bool_dtype(dtype):
            categorical.append(col)
        elif pd.api.types.is_numeric_dtype(dtype):
            numeric.append(col)
        elif pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_timedelta64_dtype(dtype):
            ignored.append(col)
        else:
            categorical.append(col)
    return numeric, categorical, ignored


# ----------------------------------------------------------------------
# Internal helpers (work on one column at a time)
# ----------------------------------------------------------------------
def _clean_numeric(series):
    """Return (float series with inf -> NaN, number of non-finite values)."""
    s = series.astype("float64")  # nullable Int64 / pd.NA become float / NaN
    non_finite = int(np.isinf(s.to_numpy()).sum())
    return s.replace([np.inf, -np.inf], np.nan), non_finite


def _clean_categorical(series):
    """Return a text series where blank / whitespace-only strings are missing."""
    s = series.astype("string")
    blank = s.str.strip() == ""
    return s.mask(blank.fillna(False), pd.NA)


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def preprocess_for_ml(df, columns=None, scaler="robust", encoding="onehot",
                      max_categories=50):
    """Prepare ``df`` for ML anomaly detection.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset. It is not modified.
    columns : list, optional
        Columns to use. Default: all columns. (Phase 3 will choose these
        automatically; here you can pass them yourself.)
    scaler : {"robust", "standard", "none"}
        Scaling for numerical columns. "robust" (median / IQR) is the
        default because mean and standard deviation are themselves
        distorted by the outliers we are trying to find.
    encoding : {"onehot", "frequency"}
        "onehot": one 0/1 column per category (recommended in the project
        PDF). "frequency": one column per categorical column holding how
        common each category is (keeps the matrix narrow, which helps
        distance-based methods such as LOF).
    max_categories : int
        Categorical columns with more distinct values than this are
        skipped (e.g. IDs, emails) so the matrix does not explode.

    Returns
    -------
    PreprocessingResult
    """
    # ---- validate inputs ------------------------------------------------
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError("The DataFrame is empty (no rows or no columns).")
    if scaler not in VALID_SCALERS:
        raise ValueError(f"scaler must be one of {VALID_SCALERS}, got {scaler!r}.")
    if encoding not in VALID_ENCODINGS:
        raise ValueError(f"encoding must be one of {VALID_ENCODINGS}, got {encoding!r}.")
    if columns is None:
        columns = list(df.columns)
    unknown = [c for c in columns if c not in df.columns]
    if unknown:
        raise ValueError(f"Columns not found in the DataFrame: {unknown}")

    # Work on positions, not labels, so duplicate index labels are safe.
    original_index = df.index
    work = df[list(columns)].reset_index(drop=True)

    numeric_cols, categorical_cols, ignored_cols = identify_column_types(work)

    report_rows = []   # missing-value report (original data, before filling)
    dropped = []       # (column, reason)
    for col in ignored_cols:
        dropped.append((col, "datetime column is not used as an ML feature"))

    # ---- numerical columns: report, then impute with the median --------
    num_data = {}
    for col in numeric_cols:
        s, non_finite = _clean_numeric(work[col])
        missing = int(s.isna().sum())
        if missing == len(s):
            dropped.append((col, "all values missing - cannot impute"))
            continue
        fill = float(s.median())
        report_rows.append({
            "column": col, "kind": "numerical",
            "missing_count": missing,
            "missing_pct": round(100 * missing / len(s), 2),
            "non_finite_count": non_finite,
            "fill_strategy": "median", "fill_value": fill,
        })
        num_data[col] = s.fillna(fill)

    # ---- categorical columns: report, then impute with the mode --------
    cat_data = {}
    for col in categorical_cols:
        s = _clean_categorical(work[col])
        missing = int(s.isna().sum())
        if missing == len(s):
            dropped.append((col, "all values missing - cannot impute"))
            continue
        n_unique = int(s.nunique())
        if n_unique > max_categories:
            dropped.append((col, f"too many distinct values ({n_unique} > {max_categories}); "
                                 "looks like an ID/free text"))
            continue
        fill = s.mode(dropna=True).iloc[0]  # ties -> first in sorted order
        report_rows.append({
            "column": col, "kind": "categorical",
            "missing_count": missing,
            "missing_pct": round(100 * missing / len(s), 2),
            "non_finite_count": 0,
            "fill_strategy": "mode", "fill_value": str(fill),
        })
        cat_data[col] = s.fillna(fill)

    if not num_data and not cat_data:
        raise ValueError("No usable columns: every selected column was empty, "
                         "a datetime, or had too many distinct values.")

    # ---- scale numerical columns (only these; 0/1 columns stay 0/1) ----
    parts = []
    feature_source = {}
    if num_data:
        num_df = pd.DataFrame(num_data)
        if scaler == "robust":
            num_df = pd.DataFrame(RobustScaler().fit_transform(num_df),
                                  columns=num_df.columns)
        elif scaler == "standard":
            num_df = pd.DataFrame(StandardScaler().fit_transform(num_df),
                                  columns=num_df.columns)
        parts.append(num_df)
        feature_source.update({str(c): c for c in num_df.columns})

    # ---- encode categorical columns ------------------------------------
    for col, s in cat_data.items():
        if encoding == "onehot":
            enc = pd.get_dummies(s, prefix=str(col), dtype="float64")
        else:  # frequency
            freq = s.value_counts(normalize=True)
            enc = pd.DataFrame({f"{col}_freq": s.map(freq).astype("float64")})
        enc.columns = [str(c) for c in enc.columns]
        parts.append(enc)
        feature_source.update({c: col for c in enc.columns})

    X = pd.concat(parts, axis=1)
    X.columns = [str(c) for c in X.columns]
    X.index = original_index  # restore original row labels

    return PreprocessingResult(
        X=X,
        feature_names=list(X.columns),
        feature_source=feature_source,
        numeric_columns=list(num_data.keys()),
        categorical_columns=list(cat_data.keys()),
        missing_report=pd.DataFrame(report_rows),
        dropped_columns=pd.DataFrame(dropped, columns=["column", "reason"]),
        scaler=scaler,
        encoding=encoding,
    )
