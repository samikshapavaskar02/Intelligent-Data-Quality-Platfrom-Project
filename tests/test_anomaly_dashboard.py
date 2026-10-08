"""Tests for modules/anomaly_dashboard.py."""

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
from modules.anomaly_dashboard import (
    RECORD_ID_COL,
    build_anomaly_table,
    get_anomaly_summary,
    get_common_anomaly_count,
    get_if_anomaly_count,
    get_lof_anomaly_count,
    get_total_records,
)


def make_analysis(if_labels, lof_labels, status, index=None):
    return pd.DataFrame(
        {IF_RESULT_COL: if_labels, LOF_RESULT_COL: lof_labels, STATUS_COL: status},
        index=index,
    )


def sample_analysis():
    """Records 101-104 as in the project PDF example table."""
    return make_analysis(
        [NORMAL, POTENTIAL_ANOMALY, POTENTIAL_ANOMALY, NORMAL],
        [NORMAL, POTENTIAL_ANOMALY, NORMAL, POTENTIAL_ANOMALY],
        [NORMAL, HIGH_CONFIDENCE, NEEDS_REVIEW, NEEDS_REVIEW],
        index=[101, 102, 103, 104],
    )


def test_total_record_count():
    assert get_total_records(sample_analysis()) == 4


def test_if_anomaly_count():
    assert get_if_anomaly_count(sample_analysis()) == 2


def test_lof_anomaly_count():
    assert get_lof_anomaly_count(sample_analysis()) == 2


def test_common_anomaly_count():
    assert get_common_anomaly_count(sample_analysis()) == 1


def test_zero_anomalies():
    df = make_analysis([NORMAL] * 3, [NORMAL] * 3, [NORMAL] * 3)
    assert get_total_records(df) == 3
    assert get_if_anomaly_count(df) == 0
    assert get_lof_anomaly_count(df) == 0
    assert get_common_anomaly_count(df) == 0


def test_all_anomalies():
    df = make_analysis(
        [POTENTIAL_ANOMALY] * 3, [POTENTIAL_ANOMALY] * 3, [HIGH_CONFIDENCE] * 3
    )
    assert get_total_records(df) == 3
    assert get_if_anomaly_count(df) == 3
    assert get_lof_anomaly_count(df) == 3
    assert get_common_anomaly_count(df) == 3


def test_summary_dict():
    assert get_anomaly_summary(sample_analysis()) == {
        "total_records": 4,
        "if_anomalies": 2,
        "lof_anomalies": 2,
        "common_anomalies": 1,
    }


def test_counts_are_plain_ints():
    summary = get_anomaly_summary(sample_analysis())
    assert all(type(v) is int for v in summary.values())


def test_dashboard_result_table():
    table = build_anomaly_table(sample_analysis())
    assert list(table.columns) == [
        RECORD_ID_COL,
        IF_RESULT_COL,
        LOF_RESULT_COL,
        STATUS_COL,
    ]
    assert table[RECORD_ID_COL].tolist() == [101, 102, 103, 104]
    assert table[STATUS_COL].tolist() == [
        NORMAL,
        HIGH_CONFIDENCE,
        NEEDS_REVIEW,
        NEEDS_REVIEW,
    ]
    assert len(table) == 4


def test_table_index_preserved():
    idx = pd.Index([40, 7, 99], name="record_id")
    df = make_analysis(
        [NORMAL, POTENTIAL_ANOMALY, NORMAL],
        [NORMAL, NORMAL, POTENTIAL_ANOMALY],
        [NORMAL, NEEDS_REVIEW, NEEDS_REVIEW],
        index=idx,
    )
    table = build_anomaly_table(df)
    assert table.index.equals(idx)
    assert table.index.name == "record_id"
    assert table[RECORD_ID_COL].tolist() == [40, 7, 99]


def test_table_is_a_copy():
    df = sample_analysis()
    table = build_anomaly_table(df)
    table.loc[101, STATUS_COL] = "changed"
    assert df.loc[101, STATUS_COL] == NORMAL


def test_realistic_detector_outputs():
    """Real detector frames -> combine_anomaly_results -> dashboard."""
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
    analysis = combine_anomaly_results(if_df, lof_df)

    assert get_anomaly_summary(analysis) == {
        "total_records": 5,
        "if_anomalies": 2,
        "lof_anomalies": 2,
        "common_anomalies": 1,
    }
    table = build_anomaly_table(analysis)
    assert table.index.equals(idx)
    assert table[RECORD_ID_COL].tolist() == [0, 1, 2, 3, 4]
    assert table[STATUS_COL].tolist() == [
        NORMAL,
        HIGH_CONFIDENCE,
        NEEDS_REVIEW,
        NEEDS_REVIEW,
        NORMAL,
    ]


def test_invalid_input_types():
    for func in (
        get_total_records,
        get_if_anomaly_count,
        get_lof_anomaly_count,
        get_common_anomaly_count,
        build_anomaly_table,
        get_anomaly_summary,
    ):
        with pytest.raises(TypeError):
            func([NORMAL])
        with pytest.raises(TypeError):
            func(None)


def test_empty_input():
    empty = pd.DataFrame(columns=[IF_RESULT_COL, LOF_RESULT_COL, STATUS_COL])
    for func in (
        get_total_records,
        get_if_anomaly_count,
        get_lof_anomaly_count,
        get_common_anomaly_count,
        build_anomaly_table,
        get_anomaly_summary,
    ):
        with pytest.raises(ValueError, match="empty"):
            func(empty)


def test_missing_columns():
    df = sample_analysis().drop(columns=[LOF_RESULT_COL])
    with pytest.raises(ValueError, match="missing required columns"):
        build_anomaly_table(df)
    with pytest.raises(ValueError, match="missing required columns"):
        get_anomaly_summary(df)


def test_invalid_labels():
    df = make_analysis(["Weird"], [NORMAL], [NORMAL])
    with pytest.raises(ValueError, match="invalid labels"):
        get_if_anomaly_count(df)


def test_duplicate_index_rejected():
    df = make_analysis([NORMAL, NORMAL], [NORMAL, NORMAL], [NORMAL, NORMAL], [1, 1])
    with pytest.raises(ValueError, match="duplicate"):
        build_anomaly_table(df)


def test_input_not_mutated():
    df = sample_analysis()
    before = df.copy(deep=True)
    get_total_records(df)
    get_if_anomaly_count(df)
    get_lof_anomaly_count(df)
    get_common_anomaly_count(df)
    build_anomaly_table(df)
    get_anomaly_summary(df)
    pd.testing.assert_frame_equal(df, before)
