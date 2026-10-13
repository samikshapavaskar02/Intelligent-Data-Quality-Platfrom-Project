import pandas as pd
import pytest

from modules.ml_evaluation import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    METHOD_IQR,
    METHOD_IF,
    METHOD_LOF,
    METHOD_ZSCORE,
    TYPE_COLUMN,
    compute_metrics,
    evaluate_anomaly_methods,
    generate_evaluation_dataset,
)


def test_evaluation_dataset_is_reproducible_and_labels_match_anomaly_types():
    first = generate_evaluation_dataset(
        n_samples=100, anomaly_fraction=0.1, random_state=17
    )
    second = generate_evaluation_dataset(
        n_samples=100, anomaly_fraction=0.1, random_state=17
    )

    pd.testing.assert_frame_equal(first, second)
    assert first[LABEL_COLUMN].sum() == 10
    assert (first[LABEL_COLUMN].astype(bool) == first[TYPE_COLUMN].ne("normal")).all()
    assert list(first.columns) == FEATURE_COLUMNS + [LABEL_COLUMN, TYPE_COLUMN]


def test_compute_metrics_calculates_precision_recall_and_f1():
    metrics = compute_metrics([1, 1, 0, 0], [1, 0, 1, 0])

    assert metrics == {
        "Detected": 2,
        "TP": 1,
        "FP": 1,
        "FN": 1,
        "TN": 1,
        "Precision": 0.5,
        "Recall": 0.5,
        "F1-score": 0.5,
    }


def test_compute_metrics_handles_zero_predicted_anomalies():
    metrics = compute_metrics([1, 0], [0, 0])

    assert metrics["Detected"] == 0
    assert metrics["Precision"] == 0.0
    assert metrics["Recall"] == 0.0
    assert metrics["F1-score"] == 0.0


@pytest.mark.parametrize("ground_truth_column", [LABEL_COLUMN, TYPE_COLUMN])
def test_explicit_features_reject_ground_truth_columns(ground_truth_column):
    data = generate_evaluation_dataset(n_samples=60, random_state=4)
    features = [*FEATURE_COLUMNS, ground_truth_column]

    with pytest.raises(ValueError, match="Ground-truth columns cannot be used as features"):
        evaluate_anomaly_methods(data, feature_columns=features, n_repeats=1)


def test_evaluation_runs_isolation_forest_and_lof_with_separate_timing():
    data = generate_evaluation_dataset(
        n_samples=100, anomaly_fraction=0.1, random_state=23
    )

    result = evaluate_anomaly_methods(data, n_repeats=1, random_state=23)

    assert METHOD_IF in result.metrics["Method"].values
    assert METHOD_LOF in result.metrics["Method"].values
    assert METHOD_IF in result.predictions.columns
    assert METHOD_LOF in result.predictions.columns
    assert "Execution Time (s)" in result.metrics.columns
    assert "preprocessing_time_s" in result.settings
    assert result.settings["preprocessing_time_s"] >= 0
    assert (result.metrics["Execution Time (s)"] >= 0).all()
    assert LABEL_COLUMN not in result.feature_columns
    assert TYPE_COLUMN not in result.feature_columns


def test_controlled_evaluation_returns_results_for_all_four_methods():
    data = generate_evaluation_dataset(
        n_samples=80, anomaly_fraction=0.1, random_state=31
    )
    methods = {METHOD_IQR, METHOD_ZSCORE, METHOD_IF, METHOD_LOF}

    result = evaluate_anomaly_methods(data, n_repeats=1, random_state=31)

    assert set(result.metrics["Method"]) == methods
    assert set(result.predictions.columns) == methods | {"Ground Truth"}
    assert len(result.predictions) == len(data)
    pd.testing.assert_series_equal(
        result.predictions["Ground Truth"],
        data[LABEL_COLUMN].astype(bool).rename("Ground Truth"),
    )

    metric_columns = {"Precision", "Recall", "F1-score", "Detected", "Execution Time (s)"}
    assert metric_columns <= set(result.metrics.columns)
    assert result.metrics["Precision"].between(0, 1).all()
    assert result.metrics["Recall"].between(0, 1).all()
    assert result.metrics["F1-score"].between(0, 1).all()
    assert result.metrics["Detected"].between(0, len(data)).all()
    assert result.metrics["Execution Time (s)"].ge(0).all()

    assert set(result.confusion_matrices) == methods
    for method, matrix in result.confusion_matrices.items():
        assert matrix.shape == (2, 2)
        assert matrix.index.tolist() == ["Actual Normal", "Actual Anomaly"]
        assert matrix.columns.tolist() == ["Predicted Normal", "Predicted Anomaly"]
        metrics = result.metrics.set_index("Method").loc[method]
        assert matrix.loc["Actual Anomaly", "Predicted Anomaly"] == metrics["TP"]
        assert matrix.loc["Actual Normal", "Predicted Anomaly"] == metrics["FP"]
        assert matrix.to_numpy().sum() == len(data)
