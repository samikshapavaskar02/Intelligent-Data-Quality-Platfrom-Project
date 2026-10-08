"""
Local Outlier Factor (LOF) anomaly detection.

Responsibility
--------------
Run scikit-learn's LocalOutlierFactor on an ML-ready numeric DataFrame (for
example ``PreprocessingResult.X`` from ``modules/preprocessing.py``) and
return one result row per input record.

LOF compares the density around a record with the density around its
nearest neighbours. A record in a much sparser area than its neighbours is
unusual *locally*, even if it is not extreme globally. Because LOF is
distance based, the input should be scaled (preprocessing.py does this).

What this module does NOT do
----------------------------
* It does not clean or impute data. NaN / infinite values are rejected with
  a clear error (do that in preprocessing).
* It does not combine its score with Isolation Forest or any other model
  (no ensemble).
* It contains no Streamlit / UI code and imports no other project module,
  so it can be tested and used on its own.

Terminology
-----------
A flagged record is called a "Potential Anomaly", never a "data error".
An unusual record can be perfectly valid (e.g. a genuine high-income
customer). The status only means: "this record looks unusual to the model".

LOF score (``lof_score``)
-------------------------
``lof_score`` is the Local Outlier Factor of the record:

    HIGHER = MORE UNUSUAL

* about 1  -> the record is about as dense as its neighbours (normal)
* above 1  -> sparser than its neighbours; the larger, the more unusual
* below 1  -> denser than its neighbours (clearly normal)

It is scikit-learn's ``negative_outlier_factor_`` multiplied by -1 (sklearn's
own values are negative, where LOWER means more unusual). The score has no
upper limit and is NOT a probability. Do not average it with other models'
scores without normalisation.

LOF has no random component, so results are deterministic for the same
input (there is no ``random_state``).

Typical use
-----------
    from modules.preprocessing import preprocess_for_ml
    from models.lof import run_lof

    prep = preprocess_for_ml(df)
    out = run_lof(prep.X)

    out.results      # one row per record, indexed by the ORIGINAL index
    out.anomalies    # only the Potential Anomalies
"""

import numbers
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

STATUS_NORMAL = "Normal"
STATUS_ANOMALY = "Potential Anomaly"

# scikit-learn convention, kept as-is: -1 = flagged (anomaly), 1 = normal.
PREDICTION_ANOMALY = -1
PREDICTION_NORMAL = 1

# Column names are prefixed with "lof_" so they can be placed next to the
# "if_*" columns of Isolation Forest later without name clashes.
COL_PREDICTION = "lof_prediction"
COL_STATUS = "lof_status"
COL_SCORE = "lof_score"

DEFAULT_N_NEIGHBORS = 20
DEFAULT_CONTAMINATION = "auto"

MIN_ROWS = 2  # LOF needs at least one neighbour besides the record itself


@dataclass
class LOFResult:
    """Output of ``run_lof``.

    Attributes
    ----------
    results : pd.DataFrame
        One row per input record, in the same order and with the SAME index
        as the input (duplicate index labels are preserved). Columns:

        * ``lof_prediction`` : -1 = flagged, 1 = normal (scikit-learn convention)
        * ``lof_status``     : "Potential Anomaly" or "Normal"
        * ``lof_score``      : Local Outlier Factor. HIGHER = MORE UNUSUAL;
          about 1 is normal. Not a probability. See the module docstring.
    model : LocalOutlierFactor
        The fitted scikit-learn model (``novelty=False``: it describes the
        training records only and cannot score new data).
    score_threshold : float
        A record is flagged exactly when ``lof_score > score_threshold``.
        (1.5 when ``contamination="auto"``.)
    feature_names : list
        Names of the columns that were used (the input's columns).
    n_neighbors : int
        The number of neighbours that was actually used.
    contamination :
        The contamination setting that was used ("auto" or a float).
    warning_messages : list[str]
        Warnings raised by scikit-learn while fitting (also re-issued with
        ``warnings.warn``). The important one: "Duplicate values are leading
        to incorrect results" - it means more identical rows exist than
        ``n_neighbors`` and the scores of such data are not reliable.
        Empty list = no warnings.
    """

    results: pd.DataFrame
    model: LocalOutlierFactor
    score_threshold: float
    feature_names: list
    n_neighbors: int
    contamination: object
    warning_messages: list = field(default_factory=list)

    @property
    def n_records(self):
        return len(self.results)

    @property
    def n_anomalies(self):
        return int((self.results[COL_PREDICTION] == PREDICTION_ANOMALY).sum())

    @property
    def anomalies(self):
        """Only the records flagged as Potential Anomaly."""
        return self.results[self.results[COL_PREDICTION] == PREDICTION_ANOMALY]


# ----------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------
def _is_int(value):
    """True for real integers (Python / numpy), but not for booleans."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _validate_contamination(contamination):
    message = (f"contamination must be 'auto' or a number in (0, 0.5], "
               f"got {contamination!r}.")
    if isinstance(contamination, str):
        if contamination != "auto":
            raise ValueError(message)
    elif (isinstance(contamination, bool)
          or not isinstance(contamination, numbers.Real)
          or not 0 < contamination <= 0.5):
        raise ValueError(message)


def _validate_and_convert(X):
    """Check the input and return it as a float numpy array (same row order)."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame (e.g. PreprocessingResult.X).")
    if X.shape[1] == 0:
        raise ValueError("X has no feature columns.")
    if X.shape[0] == 0:
        raise ValueError("X has no rows.")
    if X.shape[0] < MIN_ROWS:
        raise ValueError(f"LOF needs at least {MIN_ROWS} rows, got {X.shape[0]}.")

    non_numeric = [str(c) for c, t in X.dtypes.items() if not pd.api.types.is_numeric_dtype(t)]
    if non_numeric:
        raise ValueError(f"X must be fully numeric. Non-numeric columns: {non_numeric}. "
                         "Run modules/preprocessing.py first.")

    nan_cols = [str(c) for c in X.columns[X.isna().any().to_numpy()]]
    if nan_cols:
        raise ValueError(f"X contains missing values (NaN/NA) in columns: {nan_cols}. "
                         "Impute them first (modules/preprocessing.py).")

    values = X.to_numpy(dtype="float64")
    inf_mask = np.isinf(values).any(axis=0)
    if inf_mask.any():
        inf_cols = [str(c) for c in X.columns[inf_mask]]
        raise ValueError(f"X contains infinite values in columns: {inf_cols}.")

    return values


def _resolve_n_neighbors(n_neighbors, n_records):
    """Return the number of neighbours to use, or raise a clear error.

    * None  -> default: min(20, n_records - 1). The value actually used is
      stored in the result.
    * given -> must be an integer >= 1 and at most n_records - 1. It is
      never silently reduced (scikit-learn would do that quietly).
    """
    max_allowed = n_records - 1
    if n_neighbors is None:
        return min(DEFAULT_N_NEIGHBORS, max_allowed)
    if not _is_int(n_neighbors) or n_neighbors < 1:
        raise ValueError(f"n_neighbors must be an integer >= 1, got {n_neighbors!r}.")
    if n_neighbors > max_allowed:
        raise ValueError(f"n_neighbors={n_neighbors} is too large for {n_records} records. "
                         f"It must be at most {max_allowed} (number of records - 1).")
    return int(n_neighbors)


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def run_lof(X, n_neighbors=None, contamination=DEFAULT_CONTAMINATION):
    """Fit Local Outlier Factor on ``X`` and score every record.

    Parameters
    ----------
    X : pd.DataFrame
        ML-ready numeric matrix with no NaN / infinite values and at least
        2 rows. Not modified. Its index (including duplicate labels) is
        copied to the result. Should be scaled (preprocessing.py does it).
    n_neighbors : int or None
        Size of the neighbourhood (>= 1 and <= number of records - 1).
        Default None means min(20, number of records - 1). A value that is
        too large for the data raises ValueError instead of being reduced
        silently. Larger values give smoother, more global results; small
        values react to very local structure.
    contamination : "auto" or float in (0, 0.5]
        "auto" flags records with ``lof_score`` above 1.5. A float is the
        expected share of anomalies (e.g. 0.05 flags about 5 % of records).
        The project documents give no fixed rate, so no rate is assumed.

    Returns
    -------
    LOFResult
    """
    _validate_contamination(contamination)
    values = _validate_and_convert(X)
    k = _resolve_n_neighbors(n_neighbors, values.shape[0])

    if not isinstance(contamination, str):
        contamination = float(contamination)

    model = LocalOutlierFactor(n_neighbors=k, contamination=contamination)

    # Capture scikit-learn's warnings so they are stored in the result and
    # can be shown to the user, then re-issue them.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prediction = model.fit_predict(values)             # -1 / 1
    messages = sorted({str(w.message) for w in caught})
    for message in messages:
        warnings.warn(f"LOF: {message}", UserWarning, stacklevel=2)

    score = -model.negative_outlier_factor_                # higher = more unusual
    if not np.isfinite(score).all():
        raise ValueError("LOF produced non-finite scores. This usually means many "
                         "identical rows; remove duplicates or increase n_neighbors.")
    status = np.where(prediction == PREDICTION_ANOMALY, STATUS_ANOMALY, STATUS_NORMAL)

    # Built from arrays + the original index, so row order and duplicate
    # index labels are preserved exactly.
    results = pd.DataFrame(
        {COL_PREDICTION: prediction.astype(int),
         COL_STATUS: status,
         COL_SCORE: score},
        index=X.index.copy(),
    )

    return LOFResult(
        results=results,
        model=model,
        score_threshold=float(-model.offset_),
        feature_names=list(X.columns),
        n_neighbors=k,
        contamination=contamination,
        warning_messages=messages,
    )
