"""ML Evaluation: compare anomaly detection methods on labelled synthetic data.

Responsibility
--------------
PDF Phase 18 (Step 35) and Phase 19 (Step 36): build a controlled dataset in
which the anomalous records are known, run IQR, Z-score, Isolation Forest
and LOF on it, and compare their detections with the known labels.

This is NOT the uploaded dataset of the website and has nothing to do with
the data quality score. The uploaded data is never used or modified here.

Fair setup
----------
* Every method sees the same records and the same feature columns, and is
  scored against the same ground-truth labels.
* IQR and Z-score are record-level here: a record counts as flagged when at
  least one of its feature values is an outlier. IQR reuses
  ``modules.statistical_outliers.detect_iqr_outliers``.
* Isolation Forest and LOF reuse ``models.isolation_forest`` /
  ``models.lof`` on the same preprocessed matrix
  (``modules.preprocessing.preprocess_for_ml``, robust scaling, which LOF
  needs because it uses distances).
* IQR and Z-score are fixed rules and have no contamination setting. For
  Isolation Forest and LOF the default is ``"auto"`` (each model's own
  threshold, as in the platform). A float contamination can be given, for
  example the true anomaly rate, which is information the two rules do not
  get; compare results with that in mind.
* Execution time is the mean of ``n_repeats`` runs of the detection step.
  Shared preprocessing is timed once and reported separately.

Metrics
-------
Positive class = anomaly. Precision = TP / (TP + FP) and recall =
TP / (TP + FN); F1 is their harmonic mean. Any ratio with a zero
denominator (for example a method that flags nothing) is reported as 0.0
and a warning says so.

The module contains no Streamlit code.
"""

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from models.isolation_forest import PREDICTION_ANOMALY as IF_ANOMALY
from models.isolation_forest import run_isolation_forest
from models.lof import PREDICTION_ANOMALY as LOF_ANOMALY
from models.lof import run_lof
from modules.preprocessing import preprocess_for_ml
from modules.statistical_outliers import detect_iqr_outliers

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
METHOD_IQR = "IQR"
METHOD_ZSCORE = "Z-score"
METHOD_IF = "Isolation Forest"
METHOD_LOF = "LOF"
METHODS = (METHOD_IQR, METHOD_ZSCORE, METHOD_IF, METHOD_LOF)

LABEL_COLUMN = "is_anomaly"      # 1 = known anomaly, 0 = normal
TYPE_COLUMN = "anomaly_type"     # "normal", "extreme" or "contextual"
FEATURE_COLUMNS = ["Age", "Income", "Purchase_Amount", "Credit_Score"]

TYPE_NORMAL = "normal"
TYPE_EXTREME = "extreme"         # a value far outside its normal range
TYPE_CONTEXTUAL = "contextual"   # every value normal, the combination is not

DEFAULT_N_SAMPLES = 1000
DEFAULT_ANOMALY_FRACTION = 0.05
DEFAULT_RANDOM_STATE = 42
DEFAULT_EXTREME_SHARE = 0.5
DEFAULT_ZSCORE_THRESHOLD = 3.0
DEFAULT_CONTAMINATION = "auto"
DEFAULT_N_REPEATS = 3
MIN_SAMPLES = 50
MAX_SAMPLES = 20000

METRIC_COLUMNS = [
    "Method", "Detected", "TP", "FP", "FN", "TN",
    "Precision", "Recall", "F1-score", "Execution Time (s)",
]
CONFUSION_INDEX = ["Actual Normal", "Actual Anomaly"]
CONFUSION_COLUMNS = ["Predicted Normal", "Predicted Anomaly"]


@dataclass
class EvaluationResult:
    """Output of ``evaluate_anomaly_methods``.

    ``metrics`` has one row per method that ran successfully;
    ``confusion_matrices`` maps method -> 2x2 DataFrame;
    ``predictions`` holds one boolean column per method (True = flagged)
    plus ``Ground Truth``; ``type_recall`` is the share of each anomaly
    type that each method found; ``warnings`` lists methods that failed or
    ratios that were undefined.
    """

    data: pd.DataFrame
    feature_columns: list
    metrics: pd.DataFrame
    confusion_matrices: dict
    predictions: pd.DataFrame
    type_recall: pd.DataFrame
    settings: dict
    warnings: list = field(default_factory=list)


# ----------------------------------------------------------------------
# Synthetic dataset
# ----------------------------------------------------------------------
def _normal_records(rng, n):
    """Normal customers: income grows with age, spending follows income."""
    age = np.clip(rng.normal(40, 10, n), 18, 75).round()
    income = np.clip(15000 + 900 * age + rng.normal(0, 7000, n), 10000, None).round()
    purchase = np.clip(0.04 * income + rng.normal(0, 300, n), 50, None).round(2)
    credit = np.clip(650 + 0.002 * (income - 50000) + rng.normal(0, 40, n), 300, 850).round()
    return pd.DataFrame({
        "Age": age,
        "Income": income,
        "Purchase_Amount": purchase,
        "Credit_Score": credit,
    })


def generate_evaluation_dataset(n_samples=DEFAULT_N_SAMPLES,
                                anomaly_fraction=DEFAULT_ANOMALY_FRACTION,
                                random_state=DEFAULT_RANDOM_STATE,
                                extreme_share=DEFAULT_EXTREME_SHARE):
    """Create a reproducible dataset with known anomalies.

    Two kinds of anomaly are injected, so the methods can be told apart:

    * ``extreme``: one or two values pushed 4 to 7 standard deviations
      above the normal mean (easy for IQR / Z-score).
    * ``contextual``: a low income with a high purchase amount. Each value
      is common on its own; only the combination is unusual (hard for
      one-column rules, the case multivariate models are meant for).

    Returns
    -------
    pd.DataFrame
        Columns ``FEATURE_COLUMNS`` plus ``is_anomaly`` (0/1) and
        ``anomaly_type``. Rows are shuffled; the index is 0..n-1. The same
        arguments always give the same dataset.
    """
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
        raise ValueError(f"n_samples must be an integer, got {n_samples!r}.")
    if not MIN_SAMPLES <= n_samples <= MAX_SAMPLES:
        raise ValueError(f"n_samples must be between {MIN_SAMPLES} and {MAX_SAMPLES}, got {n_samples}.")
    if not 0 < anomaly_fraction <= 0.5:
        raise ValueError(f"anomaly_fraction must be in (0, 0.5], got {anomaly_fraction!r}.")
    if not 0 <= extreme_share <= 1:
        raise ValueError(f"extreme_share must be between 0 and 1, got {extreme_share!r}.")

    rng = np.random.default_rng(random_state)

    n_anomalies = min(max(1, int(round(n_samples * anomaly_fraction))), n_samples // 2)
    n_extreme = int(round(n_anomalies * extreme_share))
    n_contextual = n_anomalies - n_extreme

    normal = _normal_records(rng, n_samples - n_anomalies)
    means, stds = normal.mean(), normal.std()

    # Extreme anomalies: start from a normal record, push 1-2 values far out.
    extreme = normal.iloc[rng.integers(0, len(normal), n_extreme)].reset_index(drop=True)
    for i in range(n_extreme):
        for column in rng.choice(FEATURE_COLUMNS, size=int(rng.integers(1, 3)), replace=False):
            extreme.loc[i, column] = means[column] + rng.uniform(4, 7) * stds[column]

    # Contextual anomalies: low income + high purchase, both inside normal range.
    contextual = normal.iloc[rng.integers(0, len(normal), n_contextual)].reset_index(drop=True)
    contextual["Income"] = np.percentile(normal["Income"], rng.uniform(5, 15, n_contextual)).round()
    contextual["Purchase_Amount"] = np.percentile(
        normal["Purchase_Amount"], rng.uniform(85, 95, n_contextual)
    ).round(2)

    parts = []
    for frame, kind in ((normal, TYPE_NORMAL), (extreme, TYPE_EXTREME), (contextual, TYPE_CONTEXTUAL)):
        frame = frame.copy()
        frame[TYPE_COLUMN] = kind
        parts.append(frame)
    data = pd.concat(parts, ignore_index=True)

    data["Age"] = data["Age"].round().astype(int)
    data["Credit_Score"] = data["Credit_Score"].round().astype(int)
    data["Income"] = data["Income"].round(0)
    data["Purchase_Amount"] = data["Purchase_Amount"].round(2)
    data[LABEL_COLUMN] = (data[TYPE_COLUMN] != TYPE_NORMAL).astype(int)

    data = data.iloc[rng.permutation(len(data))].reset_index(drop=True)
    return data[FEATURE_COLUMNS + [LABEL_COLUMN, TYPE_COLUMN]]


# ----------------------------------------------------------------------
# Statistical detectors
# ----------------------------------------------------------------------
def detect_zscore_outliers(df, threshold=DEFAULT_ZSCORE_THRESHOLD):
    """Cell-level Z-score outliers, in the same format as ``detect_iqr_outliers``.

    A value is flagged when ``abs(value - mean) / std`` is above
    ``threshold``. Columns with fewer than 4 values or a zero / undefined
    standard deviation are skipped (no division by zero). ``Row`` is the
    index label + 1, like the IQR detector.
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be greater than 0, got {threshold!r}.")

    outliers = []
    for column in df.select_dtypes(include="number").columns:
        values = df[column].dropna()
        if len(values) < 4:
            continue

        mean, std = values.mean(), values.std()
        if pd.isna(std) or std == 0:
            continue

        for index, value in df[column].items():
            if pd.isna(value):
                continue

            z = (value - mean) / std
            if abs(z) > threshold:
                outliers.append({
                    "Row": index + 1,
                    "Column": column,
                    "Value": value,
                    "Z-score": round(float(z), 2),
                    "Method": METHOD_ZSCORE,
                    "Status": "Potential Outlier",
                })

    return pd.DataFrame(outliers)


def _flag_records(n_records, findings):
    """Record-level boolean flags from cell-level findings (``Row`` = label + 1)."""
    flags = np.zeros(n_records, dtype=bool)
    if findings is not None and not findings.empty:
        flags[(findings["Row"].astype(int) - 1).to_numpy()] = True
    return flags


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def _safe_divide(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def compute_metrics(truth, predicted):
    """Confusion counts, precision, recall and F1 for boolean arrays.

    Positive class = anomaly. Ratios with a zero denominator are 0.0.
    """
    truth = np.asarray(truth, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    if truth.shape != predicted.shape:
        raise ValueError("truth and predicted must have the same length.")

    tp = int((truth & predicted).sum())
    fp = int((~truth & predicted).sum())
    fn = int((truth & ~predicted).sum())
    tn = int((~truth & ~predicted).sum())

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    return {
        "Detected": tp + fp,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Precision": precision, "Recall": recall, "F1-score": f1,
    }


def _timed(run, n_repeats):
    """(flags of the last run, mean seconds over ``n_repeats`` runs)."""
    seconds, flags = [], None
    for _ in range(n_repeats):
        start = time.perf_counter()
        flags = run()
        seconds.append(time.perf_counter() - start)
    return np.asarray(flags, dtype=bool), float(np.mean(seconds))


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def evaluate_anomaly_methods(data, feature_columns=None, label_column=LABEL_COLUMN,
                             type_column=TYPE_COLUMN,
                             contamination=DEFAULT_CONTAMINATION,
                             zscore_threshold=DEFAULT_ZSCORE_THRESHOLD,
                             random_state=DEFAULT_RANDOM_STATE,
                             n_repeats=DEFAULT_N_REPEATS):
    """Run IQR, Z-score, Isolation Forest and LOF and score them on ``data``.

    Parameters
    ----------
    data : pd.DataFrame
        Records with a 0/1 (or boolean) ground-truth column. Not modified.
    feature_columns : list, optional
        Columns given to every method. Default: the numeric columns except
        the label column.
    label_column : str
        Ground truth; 1 / True = anomaly.
    type_column : str or None
        Optional anomaly-type column used for ``type_recall``.
    contamination : "auto" or float in (0, 0.5]
        Passed to Isolation Forest and LOF only (see the module docstring).
    zscore_threshold : float
        Z-score cut-off. Default 3.
    random_state : int
        Seed of Isolation Forest (LOF and the rules are deterministic).
    n_repeats : int
        Timing repetitions (>= 1).

    Returns
    -------
    EvaluationResult

    A method that fails is skipped and reported in ``warnings`` so that the
    other methods are still compared.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if data.shape[0] == 0:
        raise ValueError("data has no rows.")
    if label_column not in data.columns:
        raise ValueError(f"Ground-truth column {label_column!r} is not in the data.")
    if isinstance(n_repeats, bool) or not isinstance(n_repeats, (int, np.integer)) or n_repeats < 1:
        raise ValueError(f"n_repeats must be an integer >= 1, got {n_repeats!r}.")

    labels = data[label_column]
    if labels.isna().any() or not set(labels.unique()) <= {0, 1, True, False}:
        raise ValueError(f"Ground-truth column {label_column!r} must contain only 0/1 values.")
    truth = labels.astype(int).to_numpy().astype(bool)

    features_were_explicitly_supplied = feature_columns is not None
    if feature_columns is None:
        feature_columns = [
            c for c in data.select_dtypes(include="number").columns if c != label_column
        ]
    feature_columns = list(feature_columns)
    unknown = [c for c in feature_columns if c not in data.columns]
    if unknown:
        raise ValueError(f"Feature columns not found in the data: {unknown}")
    if features_were_explicitly_supplied:
        ground_truth_columns = {label_column}
        if type_column is not None:
            ground_truth_columns.add(type_column)
        leaked_columns = [c for c in feature_columns if c in ground_truth_columns]
        if leaked_columns:
            raise ValueError(
                f"Ground-truth columns cannot be used as features: {leaked_columns}"
            )
    if not feature_columns:
        raise ValueError("No feature columns to evaluate.")

    # Positions 0..n-1 are the labels the detectors report (Row = label + 1).
    work = data[feature_columns].reset_index(drop=True)
    n_records = len(work)

    runners, notes = {}, []
    runners[METHOD_IQR] = lambda: _flag_records(n_records, detect_iqr_outliers(work))
    runners[METHOD_ZSCORE] = lambda: _flag_records(
        n_records, detect_zscore_outliers(work, zscore_threshold)
    )

    preprocessing_time = 0.0
    try:
        start = time.perf_counter()
        prep = preprocess_for_ml(work, columns=feature_columns, scaler="robust")
        preprocessing_time = time.perf_counter() - start

        def run_if():
            out = run_isolation_forest(prep.X, contamination=contamination, random_state=random_state)
            return (out.results["if_prediction"] == IF_ANOMALY).to_numpy()

        def run_lof_model():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="LOF: ", category=UserWarning)
                out = run_lof(prep.X, contamination=contamination)
            return (out.results["lof_prediction"] == LOF_ANOMALY).to_numpy()

        runners[METHOD_IF] = run_if
        runners[METHOD_LOF] = run_lof_model
    except (TypeError, ValueError) as error:
        notes.append(f"{METHOD_IF} and {METHOD_LOF} were skipped: {error}")

    rows, confusion, flagged = [], {}, {}
    for method in METHODS:
        if method not in runners:
            continue
        try:
            flags, seconds = _timed(runners[method], n_repeats)
        except (TypeError, ValueError) as error:
            notes.append(f"{method} was skipped: {error}")
            continue

        scores = compute_metrics(truth, flags)
        if scores["Detected"] == 0:
            notes.append(f"{method} detected no anomalies, so its precision is reported as 0.0 (undefined).")
        rows.append({"Method": method, **scores, "Execution Time (s)": round(seconds, 5)})
        confusion[method] = pd.DataFrame(
            [[scores["TN"], scores["FP"]], [scores["FN"], scores["TP"]]],
            index=CONFUSION_INDEX, columns=CONFUSION_COLUMNS,
        )
        flagged[method] = flags

    if not truth.any():
        notes.append("The data has no anomalies, so recall is reported as 0.0 (undefined).")

    metrics = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    if not metrics.empty:
        for column in ("Precision", "Recall", "F1-score"):
            metrics[column] = metrics[column].astype(float).round(4)

    predictions = pd.DataFrame(flagged, index=data.index)
    predictions["Ground Truth"] = truth

    type_recall = pd.DataFrame()
    if type_column is not None and type_column in data.columns and flagged:
        kinds = [k for k in sorted(data.loc[truth, type_column].unique())]
        type_recall = pd.DataFrame(
            {
                kind: {
                    method: round(float(flags[(data[type_column] == kind).to_numpy()].mean()), 4)
                    for method, flags in flagged.items()
                }
                for kind in kinds
            }
        )
        type_recall.index.name = "Method"

    settings = {
        "records": n_records,
        "known_anomalies": int(truth.sum()),
        "anomaly_rate": float(truth.mean()),
        "contamination": contamination,
        "zscore_threshold": zscore_threshold,
        "random_state": random_state,
        "n_repeats": int(n_repeats),
        "preprocessing_time_s": round(preprocessing_time, 5),
    }

    return EvaluationResult(
        data=data.copy(),
        feature_columns=feature_columns,
        metrics=metrics,
        confusion_matrices=confusion,
        predictions=predictions,
        type_recall=type_recall,
        settings=settings,
        warnings=notes,
    )


def run_ml_evaluation(n_samples=DEFAULT_N_SAMPLES,
                      anomaly_fraction=DEFAULT_ANOMALY_FRACTION,
                      random_state=DEFAULT_RANDOM_STATE,
                      use_true_rate=False,
                      zscore_threshold=DEFAULT_ZSCORE_THRESHOLD,
                      n_repeats=DEFAULT_N_REPEATS):
    """Generate the synthetic dataset and evaluate all four methods on it.

    With ``use_true_rate`` Isolation Forest and LOF receive the real anomaly
    rate as their contamination; otherwise they use ``"auto"``.
    """
    data = generate_evaluation_dataset(
        n_samples=n_samples,
        anomaly_fraction=anomaly_fraction,
        random_state=random_state,
    )
    contamination = DEFAULT_CONTAMINATION
    if use_true_rate:
        contamination = float(data[LABEL_COLUMN].mean())

    return evaluate_anomaly_methods(
        data,
        feature_columns=FEATURE_COLUMNS,
        contamination=contamination,
        zscore_threshold=zscore_threshold,
        random_state=random_state,
        n_repeats=n_repeats,
    )
