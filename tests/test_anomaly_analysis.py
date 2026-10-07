"""Tests for modules/anomaly_analysis.py."""

import pandas as pd
import pytest

from modules.anomaly_analysis import (
    HIGH_CONFIDENCE,
    IF_RESULT_COL,
    LOF_RESULT_COL,
    NEEDS_REVIEW,
    NORMAL,
    POTENTIAL_ANOMALY,
    STATUS_COL,
    combine_anomaly_results,
)


def make_if(labels, index=None):
    return pd.DataFrame({"if_status": labels}, index=index)


def make_lof(labels, index=None):
    return pd.DataFrame({"lof_status": labels}, index=index)


def test_basic_combination():
    idx = [101, 102, 103, 104]
    if_df = make_if([NORMAL, POTENTIAL_ANOMALY, POTENTIAL_ANOMALY, NORMAL], idx)
    lof_df = make_lof([NORMAL, POTENTIAL_ANOMALY, NORMAL, POTENTIAL_ANOMALY], idx)

    result = combine_anomaly_results(if_df, lof_df)

    assert list(result.columns) == [IF_RESULT_COL, LOF_RESULT_COL, STATUS_COL]
    assert result[STATUS_COL].tolist() == [
        NORMAL,
        HIGH_CONFIDENCE,
        NEEDS_REVIEW,
        NEEDS_REVIEW,
    ]
    assert result[IF_RESULT_COL].tolist() == if_df["if_status"].tolist()
    assert result[LOF_RESULT_COL].tolist() == lof_df["lof_status"].tolist()


def test_normal_and_normal():
    result = combine_anomaly_results(make_if([NORMAL]), make_lof([NORMAL]))
    assert result[STATUS_COL].iloc[0] == NORMAL


def test_both_potential_anomaly():
    result = combine_anomaly_results(
        make_if([POTENTIAL_ANOMALY]), make_lof([POTENTIAL_ANOMALY])
    )
    assert result[STATUS_COL].iloc[0] == HIGH_CONFIDENCE


def test_if_anomaly_lof_normal():
    result = combine_anomaly_results(
        make_if([POTENTIAL_ANOMALY]), make_lof([NORMAL])
    )
    assert result[STATUS_COL].iloc[0] == NEEDS_REVIEW


def test_if_normal_lof_anomaly():
    result = combine_anomaly_results(
        make_if([NORMAL]), make_lof([POTENTIAL_ANOMALY])
    )
    assert result[STATUS_COL].iloc[0] == NEEDS_REVIEW


def test_anomaly_alias_is_accepted():
    result = combine_anomaly_results(make_if(["Anomaly"]), make_lof(["Anomaly"]))
    assert result[IF_RESULT_COL].iloc[0] == POTENTIAL_ANOMALY
    assert result[STATUS_COL].iloc[0] == HIGH_CONFIDENCE


def test_index_preserved_exactly():
    idx = pd.Index([40, 7, 99, 3], name="record_id")
    if_df = make_if([NORMAL, POTENTIAL_ANOMALY, NORMAL, NORMAL], idx)
    lof_df = make_lof([NORMAL, NORMAL, NORMAL, POTENTIAL_ANOMALY], idx)

    result = combine_anomaly_results(if_df, lof_df)

    assert result.index.tolist() == [40, 7, 99, 3]
    assert result.index.name == "record_id"
    assert result[STATUS_COL].tolist() == [
        NORMAL,
        NEEDS_REVIEW,
        NORMAL,
        NEEDS_REVIEW,
    ]


def test_same_records_in_different_order_are_aligned_by_index():
    if_df = make_if([NORMAL, POTENTIAL_ANOMALY, NORMAL], [1, 2, 3])
    # By label: record 1 -> LOF Normal, 2 -> Potential Anomaly, 3 -> Normal.
    lof_df = make_lof([NORMAL, POTENTIAL_ANOMALY, NORMAL], [3, 2, 1])

    result = combine_anomaly_results(if_df, lof_df)

    assert result.index.tolist() == [1, 2, 3]
    assert result.loc[1, STATUS_COL] == NORMAL
    assert result.loc[2, STATUS_COL] == HIGH_CONFIDENCE
    assert result.loc[3, STATUS_COL] == NORMAL


def test_duplicate_indexes_rejected():
    if_df = make_if([NORMAL, NORMAL], [1, 1])
    lof_df = make_lof([NORMAL, NORMAL], [1, 1])
    with pytest.raises(ValueError, match="duplicate"):
        combine_anomaly_results(if_df, lof_df)


def test_duplicate_index_in_only_one_input_rejected():
    if_df = make_if([NORMAL, NORMAL], [1, 2])
    lof_df = make_lof([NORMAL, NORMAL], [1, 1])
    with pytest.raises(ValueError, match="duplicate"):
        combine_anomaly_results(if_df, lof_df)


def test_empty_input_rejected():
    empty = pd.DataFrame({"if_status": []})
    with pytest.raises(ValueError, match="empty"):
        combine_anomaly_results(empty, make_lof([NORMAL]))
    empty_lof = pd.DataFrame({"lof_status": []})
    with pytest.raises(ValueError, match="empty"):
        combine_anomaly_results(make_if([NORMAL]), empty_lof)


def test_mismatched_row_counts_rejected():
    if_df = make_if([NORMAL, NORMAL, NORMAL])
    lof_df = make_lof([NORMAL, NORMAL])
    with pytest.raises(ValueError, match="row counts"):
        combine_anomaly_results(if_df, lof_df)


def test_mismatched_indexes_rejected():
    if_df = make_if([NORMAL, NORMAL], [1, 2])
    lof_df = make_lof([NORMAL, NORMAL], [1, 3])
    with pytest.raises(ValueError, match="mismatched indexes"):
        combine_anomaly_results(if_df, lof_df)


def test_missing_required_column_rejected():
    wrong = pd.DataFrame({"something_else": [NORMAL]})
    with pytest.raises(ValueError, match="if_status"):
        combine_anomaly_results(wrong, make_lof([NORMAL]))
    with pytest.raises(ValueError, match="lof_status"):
        combine_anomaly_results(make_if([NORMAL]), wrong)


def test_custom_column_names():
    if_df = pd.DataFrame({"IF": [POTENTIAL_ANOMALY]})
    lof_df = pd.DataFrame({"LOF": [POTENTIAL_ANOMALY]})
    result = combine_anomaly_results(if_df, lof_df, if_column="IF", lof_column="LOF")
    assert result[STATUS_COL].iloc[0] == HIGH_CONFIDENCE


def test_invalid_input_types_rejected():
    good = make_lof([NORMAL])
    with pytest.raises(TypeError):
        combine_anomaly_results([NORMAL], good)
    with pytest.raises(TypeError):
        combine_anomaly_results(make_if([NORMAL]), {"lof_status": [NORMAL]})
    with pytest.raises(TypeError):
        combine_anomaly_results(None, None)


def test_invalid_labels_rejected():
    with pytest.raises(ValueError, match="invalid labels"):
        combine_anomaly_results(make_if(["Weird"]), make_lof([NORMAL]))
    with pytest.raises(ValueError, match="invalid labels"):
        combine_anomaly_results(make_if([NORMAL]), make_lof([None]))


def test_exactly_one_result_per_record():
    idx = list(range(50))
    labels = [NORMAL, POTENTIAL_ANOMALY] * 25
    result = combine_anomaly_results(make_if(labels, idx), make_lof(labels[::-1], idx))
    assert len(result) == 50
    assert result.index.is_unique
    assert result.index.tolist() == idx
    assert result[STATUS_COL].notna().all()


def test_inputs_are_not_mutated():
    idx = [10, 20, 30]
    if_df = pd.DataFrame(
        {"if_status": [NORMAL, "Anomaly", NORMAL], "if_score": [0.1, -0.4, 0.2]},
        index=idx,
    )
    lof_df = pd.DataFrame(
        {"lof_status": [NORMAL, NORMAL, "Anomaly"], "lof_score": [1.0, 1.1, 2.5]},
        index=idx,
    )
    if_before = if_df.copy(deep=True)
    lof_before = lof_df.copy(deep=True)

    combine_anomaly_results(if_df, lof_df)

    pd.testing.assert_frame_equal(if_df, if_before)
    pd.testing.assert_frame_equal(lof_df, lof_before)


def test_integration_with_detector_style_outputs():
    """Detector outputs typically carry extra columns (scores, features).

    Extra columns must be ignored and the record index kept.
    """
    idx = pd.Index([101, 102, 103, 104], name="record_id")
    if_df = pd.DataFrame(
        {
            "if_status": [NORMAL, POTENTIAL_ANOMALY, POTENTIAL_ANOMALY, NORMAL],
            "if_score": [0.12, -0.31, -0.18, 0.09],
        },
        index=idx,
    )
    lof_df = pd.DataFrame(
        {
            "lof_status": [NORMAL, POTENTIAL_ANOMALY, NORMAL, POTENTIAL_ANOMALY],
            "lof_score": [1.01, 2.40, 1.05, 1.90],
        },
        index=idx,
    )

    result = combine_anomaly_results(if_df, lof_df)

    assert result.index.equals(idx)
    assert result[STATUS_COL].tolist() == [
        NORMAL,
        HIGH_CONFIDENCE,
        NEEDS_REVIEW,
        NEEDS_REVIEW,
    ]
    assert set(result.columns) == {IF_RESULT_COL, LOF_RESULT_COL, STATUS_COL}


def test_realistic_detector_output_columns():
    """Full detector-style frames with prediction, status and score columns."""
    idx = pd.Index([0, 1, 2, 3, 4], name="record_id")
    if_df = pd.DataFrame(
        {
            "if_prediction": [1, -1, -1, 1, 1],
            "if_status": [
                NORMAL,
                POTENTIAL_ANOMALY,
                POTENTIAL_ANOMALY,
                NORMAL,
                NORMAL,
            ],
            "if_score": [0.15, -0.30, -0.12, 0.10, 0.20],
        },
        index=idx,
    )
    lof_df = pd.DataFrame(
        {
            "lof_prediction": [1, -1, 1, -1, 1],
            "lof_status": [
                NORMAL,
                POTENTIAL_ANOMALY,
                NORMAL,
                POTENTIAL_ANOMALY,
                NORMAL,
            ],
            "lof_score": [-1.02, -2.50, -1.10, -1.80, -1.00],
        },
        index=idx,
    )
    if_before = if_df.copy(deep=True)
    lof_before = lof_df.copy(deep=True)

    # Default columns must be the real detector columns.
    result = combine_anomaly_results(if_df, lof_df)

    assert result.index.equals(idx)
    assert list(result.columns) == [IF_RESULT_COL, LOF_RESULT_COL, STATUS_COL]
    assert result[IF_RESULT_COL].tolist() == if_df["if_status"].tolist()
    assert result[LOF_RESULT_COL].tolist() == lof_df["lof_status"].tolist()
    assert result[STATUS_COL].tolist() == [
        NORMAL,
        HIGH_CONFIDENCE,
        NEEDS_REVIEW,
        NEEDS_REVIEW,
        NORMAL,
    ]
    pd.testing.assert_frame_equal(if_df, if_before)
    pd.testing.assert_frame_equal(lof_df, lof_before)
