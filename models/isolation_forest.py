"""
Isolation Forest anomaly detection.

Responsibility
--------------
Run scikit-learn's IsolationForest on an ML-ready numeric DataFrame (for
example ``PreprocessingResult.X`` from ``modules/preprocessing.py``) and
return one result row per input record.

What this module does NOT do
----------------------------
* It does not clean or impute data. NaN / infinite values are rejected with
  a clear error instead of being silently "fixed" (do that in preprocessing).
* It does not combine its score with LOF or any other model (no ensemble).
* It contains no Streamlit / UI code and imports no other project module.

Terminology
-----------
A flagged record is called a "Potential Anomaly", never a "data error".
An unusual record can be perfectly valid (e.g. a genuine high-income
customer). The status only means: "this record looks unusual to the model".

Typical use
-----------
    from modules.preprocessing import preprocess_for_ml
    from modules.isolation_forest import run_isolation_forest

    prep = preprocess_for_ml(df)
    out = run_isolation_forest(prep.X)

    out.results      # one row per record, indexed by the ORIGINAL index
    out.anomalies    # only the Potential Anomalies
"""

import numbers
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

STATUS_NORMAL = "Normal"
STATUS_ANOMALY = "Potential Anomaly"

# scikit-learn convention, kept as-is: -1 = flagged (anomaly), 1 = normal.
PREDICTION_ANOMALY = -1
PREDICTION_NORMAL = 1

# Column names are prefixed with "if_" so that results of other models
# (e.g. "lof_score") can later be joined side by side without name clashes.
COL_PREDICTION = "if_prediction"
COL_STATUS = "if_status"
COL_SCORE = "if_score"

DEFAULT_N_ESTIMATORS = 100
DEFAULT_CONTAMINATION = "auto"
DEFAULT_RANDOM_STATE = 42

MIN_ROWS = 2  # one record cannot be "unusual" compared with anything


@dataclass
class IsolationForestResult:
    """Output of ``run_isolation_forest``.

    Attributes
    ----------
    results : pd.DataFrame
        One row per input record, in the same order and with the SAME index
        as the input (duplicate index labels are preserved). Columns:

        * ``if_prediction`` : -1 = flagged, 1 = normal (scikit-learn convention)
        * ``if_status``     : "Potential Anomaly" or "Normal"
        * ``if_score``      : Isolation Forest anomaly score. HIGHER = MORE
          UNUSUAL. It is ``-score_samples`` of scikit-learn, which is the
          anomaly score of the original paper (values in (0, 1]; around 0.5
          or lower is typical for normal records). It is NOT a probability
          and must not be averaged with other models' scores without
          normalisation.
    model : IsolationForest
        The fitted scikit-learn model.
    score_threshold : float
        A record is flagged exactly when ``if_score > score_threshold``.
    feature_names : list
        Names of the columns that were used (the input's columns).
    n_estimators, contamination, random_state :
        The parameters that were used.
    """

    results: pd.DataFrame
    model: IsolationForest
    score_threshold: float
    feature_names: list
    n_estimators: int
    contamination: object
    random_state: object

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


def _validate_parameters(n_estimators, contamination, random_state):
    if not _is_int(n_estimators) or n_estimators < 1:
        raise ValueError(f"n_estimators must be an integer >= 1, got {n_estimators!r}.")

    if isinstance(contamination, str):
        if contamination != "auto":
            raise ValueError(f"contamination must be 'auto' or a number in (0, 0.5], "
                             f"got {contamination!r}.")
    elif (isinstance(contamination, bool)
          or not isinstance(contamination, numbers.Real)
          or not 0 < contamination <= 0.5):
        raise ValueError(f"contamination must be 'auto' or a number in (0, 0.5], "
                         f"got {contamination!r}.")

    if random_state is not None and not _is_int(random_state):
        raise ValueError(f"random_state must be an integer or None, got {random_state!r}.")


def _validate_and_convert(X):
    """Check the input and return it as a float numpy array (same row order)."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame (e.g. PreprocessingResult.X).")
    if X.shape[1] == 0:
        raise ValueError("X has no feature columns.")
    if X.shape[0] == 0:
        raise ValueError("X has no rows.")
    if X.shape[0] < MIN_ROWS:
        raise ValueError(f"Isolation Forest needs at least {MIN_ROWS} rows, got {X.shape[0]}.")

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


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def run_isolation_forest(X, n_estimators=DEFAULT_N_ESTIMATORS,
                         contamination=DEFAULT_CONTAMINATION,
                         random_state=DEFAULT_RANDOM_STATE):
    """Fit an Isolation Forest on ``X`` and score every record.

    Parameters
    ----------
    X : pd.DataFrame
        ML-ready numeric matrix with no NaN / infinite values and at least
        2 rows. Not modified. Its index (including duplicate labels) is
        copied to the result.
    n_estimators : int
        Number of trees (>= 1). Default 100.
    contamination : "auto" or float in (0, 0.5]
        "auto" uses the original paper's score threshold. A float is the
        expected share of anomalies (e.g. 0.05 flags about 5 % of records).
        The project documents give no fixed rate, so no rate is assumed.
    random_state : int or None
        Seed. Default 42, so repeated runs give identical results.
        ``None`` gives non-reproducible results.

    Returns
    -------
    IsolationForestResult
    """
    _validate_parameters(n_estimators, contamination, random_state)
    values = _validate_and_convert(X)

    if not isinstance(contamination, str):
        contamination = float(contamination)

    model = IsolationForest(
        n_estimators=int(n_estimators),
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(values)

    prediction = model.predict(values)       # -1 / 1
    score = -model.score_samples(values)     # higher = more unusual
    status = np.where(prediction == PREDICTION_ANOMALY, STATUS_ANOMALY, STATUS_NORMAL)

    # Built from arrays + the original index, so row order and duplicate
    # index labels are preserved exactly.
    results = pd.DataFrame(
        {COL_PREDICTION: prediction.astype(int),
         COL_STATUS: status,
         COL_SCORE: score},
        index=X.index.copy(),
    )

    return IsolationForestResult(
        results=results,
        model=model,
        score_threshold=float(-model.offset_),
        feature_names=list(X.columns),
        n_estimators=int(n_estimators),
        contamination=contamination,
        random_state=random_state,
    )
