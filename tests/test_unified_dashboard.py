"""Tests for modules/unified_dashboard.py and its wiring in app.py.

The first parts use the REAL detection modules and the REAL anomaly pipeline
on a small dataset with known problems, with several kinds of index. Hand-built
result frames (in the exact formats the real modules return) cover cases the
real models do not produce on demand, such as a record flagged by both models.
The last part checks app.py statically, because Streamlit itself is not
needed (or started) by these tests.
"""

import ast
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import modules.unified_dashboard as unified_dashboard
import modules.unified_results as unified_results
from modules.anomaly_analysis import (
    HIGH_CONFIDENCE,
    NEEDS_REVIEW,
    NORMAL,
    POTENTIAL_ANOMALY,
    combine_anomaly_results,
)
from modules.anomaly_pipeline import run_anomaly_pipeline
from modules.consistency import detect_inconsistent_values
from modules.data_type_validation import detect_data_type_problems
from modules.duplicates import detect_duplicates
from modules.format_validation import detect_format_problems
from modules.invalid_values import detect_invalid_values
from modules.missing_values import detect_missing_values
from modules.statistical_outliers import detect_iqr_outliers
from modules.unified_dashboard import (
    ML_SOURCES,
    build_unified_table,
    count_by_issue_type,
    filter_unified_table,
    get_dataset_fingerprint,
    get_unified_summary,
    prepare_display_table,
    run_anomaly_sources,
)
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

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"

# The same rules the app uses.
RULES = {"Age": (0, 120), "Marks": (0, 100), "Rating": (1, 5)}


# ----------------------------------------------------------------------
# Test data
# ----------------------------------------------------------------------
def make_dataset(index_kind="default", n=60):
    """Dataset with known problems, as positions (0-based) in the frame:

    3  Age = 150               -> invalid value
    5  Age = "Twenty"          -> invalid value AND data type problem
    9  Email = "abc@gmail"     -> format problem
    11 City missing            -> missing value
    13 City = "PUNE"           -> inconsistent with "Pune"
    7  Income/Score extreme    -> IQR outlier
    n  (last row)              -> exact copy of row 0 -> duplicates
    """
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "Name": [f"P{i}" for i in range(n)],
        "Age": rng.integers(20, 60, n).astype(object),
        "Income": rng.normal(50000, 5000, n).round(),
        "Score": rng.normal(70, 5, n).round(1),
        "City": rng.choice(["Pune", "Mumbai"], n),
        "Email": ["a@b.com"] * n,
    })
    df.loc[3, "Age"] = 150
    df.loc[5, "Age"] = "Twenty"
    df.loc[9, "Email"] = "abc@gmail"
    df.loc[11, "City"] = None
    df.loc[13, "City"] = "PUNE"
    df.loc[7, ["Income", "Score"]] = [9_000_000, 400.0]
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    size = len(df)
    if index_kind == "offset":
        df.index = pd.RangeIndex(100, 100 + size)
    elif index_kind == "shuffled":
        df.index = list(np.random.default_rng(1).permutation(size) + 10)
    elif index_kind == "string":
        df.index = [f"r{i}" for i in range(size)]
    elif index_kind == "duplicate":
        df.index = [i % 30 for i in range(size)]
    return df


def quality_results(df, skip=()):
    """What app.py computes, using the real detectors."""
    sources = {
        SRC_MISSING: lambda: detect_missing_values(df),
        SRC_DUPLICATES: lambda: detect_duplicates(df),
        SRC_INVALID: lambda: detect_invalid_values(df, RULES),
        SRC_DATA_TYPE: lambda: detect_data_type_problems(df),
        SRC_FORMAT: lambda: detect_format_problems(df),
        SRC_CONSISTENCY: lambda: detect_inconsistent_values(df),
        SRC_IQR: lambda: detect_iqr_outliers(df),
    }
    return {name: make() for name, make in sources.items() if name not in skip}


def ml_frames(index, if_flagged=(), lof_flagged=()):
    """Hand-built per-record IF / LOF / combined results (real formats)."""
    def model(prefix, flagged):
        status = [POTENTIAL_ANOMALY if label in flagged else NORMAL for label in index]
        return pd.DataFrame({
            f"{prefix}_prediction": [-1 if s == POTENTIAL_ANOMALY else 1 for s in status],
            f"{prefix}_status": status,
            f"{prefix}_score": [0.9 if s == POTENTIAL_ANOMALY else 0.4 for s in status],
        }, index=index)

    if_df, lof_df = model("if", if_flagged), model("lof", lof_flagged)
    return {SRC_IF: if_df, SRC_LOF: lof_df,
            SRC_ANOMALY: combine_anomaly_results(if_df, lof_df)}


def rows_of(unified, issue_type):
    """Set of (Record ID, Row Position, Column) for one issue type."""
    part = unified[unified["Issue Type"] == issue_type]
    return {
        (r["Record ID"], None if pd.isna(r["Row Position"]) else int(r["Row Position"]), r["Column"])
        for _, r in part.iterrows()
    }


def expected(df, position0, column=None):
    """(Record ID, 1-based Row Position, Column) of the 0-based position."""
    return (df.index[position0], position0 + 1, column)


@pytest.fixture(params=["default", "offset", "shuffled"])
def df(request):
    """Integer indexes: every real detector works with all of them."""
    return make_dataset(request.param)


# ----------------------------------------------------------------------
# run_anomaly_sources
# ----------------------------------------------------------------------
def test_run_anomaly_sources_returns_the_three_ml_sources_aligned_with_the_data(df):
    sources, notes = run_anomaly_sources(df)

    assert set(sources) == set(ML_SOURCES) == {SRC_IF, SRC_LOF, SRC_ANOMALY}
    for frame in sources.values():
        assert frame.index.equals(df.index)
    assert isinstance(notes, list)


@pytest.mark.parametrize("kind", ["default", "offset", "shuffled", "string"])
def test_run_anomaly_sources_keeps_the_original_index(kind):
    data = make_dataset(kind)
    sources, _ = run_anomaly_sources(data)

    assert sources[SRC_ANOMALY].index.equals(data.index)
    assert sources[SRC_IF].index.equals(data.index)
    assert sources[SRC_LOF].index.equals(data.index)


def test_run_anomaly_sources_does_not_modify_the_dataset():
    data = make_dataset()
    before = data.copy(deep=True)

    run_anomaly_sources(data)

    pd.testing.assert_frame_equal(data, before)


def test_run_anomaly_sources_matches_the_pipeline_output():
    data = make_dataset()
    sources, _ = run_anomaly_sources(data)
    reference = run_anomaly_pipeline(data)

    pd.testing.assert_frame_equal(sources[SRC_ANOMALY], reference.analysis)
    pd.testing.assert_frame_equal(sources[SRC_IF], reference.isolation_forest.results)
    pd.testing.assert_frame_equal(sources[SRC_LOF], reference.lof.results)


def test_run_anomaly_sources_empty_dataset_gives_a_note_not_an_error():
    for empty in (pd.DataFrame(), pd.DataFrame({"Age": []}), pd.DataFrame(index=[0, 1])):
        sources, notes = run_anomaly_sources(empty)
        assert sources == {}
        assert len(notes) == 1 and "empty" in notes[0]


def test_run_anomaly_sources_duplicate_index_is_skipped_with_a_note():
    data = make_dataset("duplicate")

    sources, notes = run_anomaly_sources(data)

    assert sources == {}
    assert len(notes) == 1 and "duplicate" in notes[0]


def test_run_anomaly_sources_single_row_gives_a_note_not_an_error():
    sources, notes = run_anomaly_sources(pd.DataFrame({"Age": [30], "Income": [1000.0]}))

    assert sources == {}
    assert len(notes) == 1 and notes[0].startswith("Anomaly detection was not run")


def test_run_anomaly_sources_no_usable_features_gives_a_note_not_an_error():
    data = pd.DataFrame({
        "Customer ID": range(1, 11),
        "Email": [f"user{i}@example.com" for i in range(10)],
    })

    sources, notes = run_anomaly_sources(data)

    assert sources == {}
    assert len(notes) == 1 and notes[0].startswith("Anomaly detection was not run")


def test_run_anomaly_sources_rejects_non_dataframe():
    with pytest.raises(TypeError):
        run_anomaly_sources([[1, 2], [3, 4]])


# ----------------------------------------------------------------------
# get_dataset_fingerprint (cache key)
# ----------------------------------------------------------------------
def test_fingerprint_is_stable_for_equal_datasets():
    assert get_dataset_fingerprint(make_dataset()) == get_dataset_fingerprint(make_dataset())
    assert isinstance(get_dataset_fingerprint(make_dataset()), str)


def test_fingerprint_changes_when_a_single_value_changes():
    data = make_dataset()
    changed = data.copy()
    changed.loc[40, "Income"] = changed.loc[40, "Income"] + 1

    assert get_dataset_fingerprint(changed) != get_dataset_fingerprint(data)


def test_fingerprint_changes_with_index_columns_and_dtype():
    data = make_dataset()
    base = get_dataset_fingerprint(data)

    assert get_dataset_fingerprint(make_dataset("offset")) != base
    assert get_dataset_fingerprint(data.rename(columns={"Score": "Marks"})) != base
    assert get_dataset_fingerprint(data.astype({"Income": "float32"})) != base
    assert get_dataset_fingerprint(data.iloc[:-1]) != base
    assert get_dataset_fingerprint(data.iloc[::-1]) != base


def test_fingerprint_handles_mixed_unhashable_and_empty_data():
    mixed = pd.DataFrame({"Age": [1, "Twenty", None]})
    lists = pd.DataFrame({"Tags": [["a"], ["b", "c"], []]})

    assert len(get_dataset_fingerprint(mixed)) == 64
    assert len(get_dataset_fingerprint(lists)) == 64
    assert get_dataset_fingerprint(lists) != get_dataset_fingerprint(
        pd.DataFrame({"Tags": [["a"], ["b", "d"], []]}))
    assert len(get_dataset_fingerprint(pd.DataFrame())) == 64


def test_fingerprint_does_not_modify_the_dataset():
    data = make_dataset()
    before = data.copy(deep=True)

    get_dataset_fingerprint(data)

    pd.testing.assert_frame_equal(data, before)


def test_fingerprint_rejects_non_dataframe():
    with pytest.raises(TypeError):
        get_dataset_fingerprint([[1, 2]])


# ----------------------------------------------------------------------
# build_unified_table: real detectors, several indexes
# ----------------------------------------------------------------------
def test_unified_table_has_the_standard_columns_and_all_quality_sources(df):
    unified, notes = build_unified_table(df, quality_results(df))

    assert list(unified.columns) == UNIFIED_COLUMNS
    assert notes == []
    assert set(unified["Issue Type"]) >= {
        "Missing Value", "Duplicate Record", "Invalid Value", "Data Type Problem",
        "Format Problem", "Inconsistent Representation", "IQR Outlier",
    }


def test_row_mapping_is_exact_for_every_index_type(df):
    """Invalid/data-type Row is a position, format/IQR Row is label + 1."""
    unified, _ = build_unified_table(df, quality_results(df))

    assert rows_of(unified, "Invalid Value") == {
        expected(df, 3, "Age"), expected(df, 5, "Age"),
    }
    assert rows_of(unified, "Data Type Problem") == {expected(df, 5, "Age")}
    assert rows_of(unified, "Format Problem") == {expected(df, 9, "Email")}
    assert expected(df, 7, "Income") in rows_of(unified, "IQR Outlier")
    assert expected(df, 7, "Score") in rows_of(unified, "IQR Outlier")
    assert rows_of(unified, "Duplicate Record") == {
        expected(df, 0), expected(df, len(df) - 1),
    }


def test_column_level_findings_have_no_record_or_position(df):
    unified, _ = build_unified_table(df, quality_results(df))

    for issue in ("Missing Value", "Inconsistent Representation"):
        part = unified[unified["Issue Type"] == issue]
        assert len(part) == 1
        assert part["Column"].iloc[0] == "City"
        assert part["Record ID"].iloc[0] is None
        assert part["Row Position"].isna().all()


def test_ml_findings_use_dataset_labels_and_exact_positions(df):
    sources, _ = run_anomaly_sources(df)
    unified, notes = build_unified_table(df, quality_results(df), sources)

    assert notes == []
    ml = unified[unified["Category"] == CATEGORY_ANOMALY]
    assert len(ml) > 0                                  # the pipeline flags something
    assert set(ml["Issue Type"]) <= {
        "Isolation Forest Anomaly", "LOF Anomaly", "Combined Anomaly"}
    for _, row in ml.iterrows():
        assert df.index[int(row["Row Position"]) - 1] == row["Record ID"]


def test_ml_findings_equal_the_pipeline_flags():
    data = make_dataset("shuffled")
    result = run_anomaly_pipeline(data)
    sources, _ = run_anomaly_sources(data)
    unified, _ = build_unified_table(data, {}, sources)

    flagged = result.analysis[result.analysis["Status"] != NORMAL].index
    combined = unified[unified["Issue Type"] == "Combined Anomaly"]
    assert list(combined["Record ID"]) == list(flagged)
    assert {r for r in combined["Row Position"]} == {
        data.index.get_loc(label) + 1 for label in flagged}


def test_string_index_with_ml_and_the_detectors_that_support_it():
    data = make_dataset("string")
    # detect_format_problems and detect_iqr_outliers add 1 to the index
    # label, so they cannot run on text labels; the rest can.
    quality = quality_results(data, skip=(SRC_FORMAT, SRC_IQR))
    sources, _ = run_anomaly_sources(data)

    unified, notes = build_unified_table(data, quality, sources)

    assert notes == []
    assert rows_of(unified, "Invalid Value") == {
        ("r3", 4, "Age"), ("r5", 6, "Age")}
    assert rows_of(unified, "Duplicate Record") == {
        ("r0", 1, None), ("r60", 61, None)}
    ml = unified[unified["Category"] == CATEGORY_ANOMALY]
    assert len(ml) > 0
    assert all(str(label).startswith("r") for label in ml["Record ID"])


# ----------------------------------------------------------------------
# build_unified_table: duplicate index
# ----------------------------------------------------------------------
def test_duplicate_index_never_guesses_a_position_and_does_not_crash():
    data = make_dataset("duplicate")
    sources, ml_notes = run_anomaly_sources(data)
    assert sources == {} and len(ml_notes) == 1

    unified, notes = build_unified_table(data, quality_results(data), sources)

    # detect_data_type_problems returns an array instead of a row number when
    # a label repeats, so that one source may be rejected (with a note). It
    # must never take the other findings down with it.
    assert all("Data type problems" in note for note in notes)
    assert not (unified["Category"] == CATEGORY_ANOMALY).any()
    # Position-based detectors: exact position, label as stored in the index.
    assert rows_of(unified, "Invalid Value") == {
        (data.index[3], 4, "Age"), (data.index[5], 6, "Age")}
    # Label-based findings with a repeated label: position left empty.
    for issue in ("Duplicate Record", "Format Problem", "IQR Outlier"):
        part = unified[unified["Issue Type"] == issue]
        assert len(part) > 0
        assert part["Row Position"].isna().all()


def test_duplicate_index_keeps_a_label_that_occurs_once_exact():
    data = make_dataset("default").iloc[:30]
    data.index = [0, 0] + list(range(2, 30))      # only label 0 repeats
    quality = {SRC_INVALID: detect_invalid_values(data, RULES)}

    unified, _ = build_unified_table(data, quality)

    assert rows_of(unified, "Invalid Value") == {
        (data.index[3], 4, "Age"), (data.index[5], 6, "Age")}


# ----------------------------------------------------------------------
# build_unified_table: empty / missing optional outputs
# ----------------------------------------------------------------------
@pytest.mark.parametrize("ml", [None, {}])
def test_quality_only_when_ml_results_are_not_available(ml):
    data = make_dataset()
    unified, notes = build_unified_table(data, quality_results(data), ml)

    assert notes == []
    assert not unified.empty
    assert not (unified["Category"] == CATEGORY_ANOMALY).any()


def test_partial_ml_results_are_accepted():
    data = make_dataset()
    sources, _ = run_anomaly_sources(data)

    unified, notes = build_unified_table(
        data, quality_results(data), {SRC_LOF: sources[SRC_LOF]})

    assert notes == []
    ml = unified[unified["Category"] == CATEGORY_ANOMALY]
    assert set(ml["Issue Type"]) <= {"LOF Anomaly"}


def test_clean_dataset_gives_an_empty_table_with_standard_columns():
    clean = pd.DataFrame({"Age": [21, 22, 23, 24, 25, 26], "City": list("ABCDEF")})

    unified, notes = build_unified_table(clean, quality_results(clean))

    assert unified.empty
    assert list(unified.columns) == UNIFIED_COLUMNS
    assert notes == []


def test_none_and_empty_results_are_no_findings():
    data = make_dataset()
    results = {name: None for name in SUPPORTED_SOURCES}
    results[SRC_IQR] = pd.DataFrame()                 # detect_iqr_outliers: no findings

    unified, notes = build_unified_table(data, results)

    assert unified.empty and notes == []
    assert list(unified.columns) == UNIFIED_COLUMNS


def test_no_results_at_all():
    unified, notes = build_unified_table(make_dataset(), {})

    assert unified.empty and notes == []


# ----------------------------------------------------------------------
# build_unified_table: ML anomalies are not data-quality errors
# ----------------------------------------------------------------------
def test_normal_records_are_never_listed():
    data = make_dataset()
    ml = ml_frames(data.index)                          # everything Normal

    unified, _ = build_unified_table(data, {}, ml)

    assert unified.empty


def test_only_flagged_records_are_listed_and_only_as_ml_anomalies():
    data = make_dataset()
    ml = ml_frames(data.index, if_flagged={4, 8}, lof_flagged={8, 12})

    unified, _ = build_unified_table(data, {}, ml)

    assert set(unified["Category"]) == {CATEGORY_ANOMALY}
    assert set(unified["Record ID"]) == {4, 8, 12}      # nobody else
    combined = unified[unified["Issue Type"] == "Combined Anomaly"]
    assert dict(zip(combined["Record ID"], combined["Status"])) == {
        4: NEEDS_REVIEW, 8: HIGH_CONFIDENCE, 12: NEEDS_REVIEW}


def test_adding_ml_results_does_not_change_the_quality_findings():
    data = make_dataset()
    quality = quality_results(data)
    sources, _ = run_anomaly_sources(data)

    without_ml, _ = build_unified_table(data, quality)
    with_ml, _ = build_unified_table(data, quality, sources)

    not_ml = with_ml[with_ml["Category"] != CATEGORY_ANOMALY].reset_index(drop=True)
    pd.testing.assert_frame_equal(not_ml, without_ml)
    assert get_unified_summary(with_ml)["data_quality"] == \
        get_unified_summary(without_ml)["data_quality"]


# ----------------------------------------------------------------------
# build_unified_table: no duplicate findings, inputs untouched
# ----------------------------------------------------------------------
def test_no_finding_is_listed_twice():
    data = make_dataset()
    quality = quality_results(data)
    sources, _ = run_anomaly_sources(data)

    unified, _ = build_unified_table(data, quality, sources)

    assert not unified.duplicated().any()
    # One row per finding of each source, exactly once.
    expected_total = sum(
        len(combine_quality_results({name: frame}, index=data.index))
        for name, frame in {**quality, **sources}.items()
    )
    assert len(unified) == expected_total


def test_building_twice_gives_the_same_table():
    data = make_dataset()
    quality = quality_results(data)
    sources, _ = run_anomaly_sources(data)

    first, _ = build_unified_table(data, quality, sources)
    second, _ = build_unified_table(data, quality, sources)

    pd.testing.assert_frame_equal(first, second)


def test_inputs_are_not_modified():
    data = make_dataset()
    quality = quality_results(data)
    sources, _ = run_anomaly_sources(data)
    data_before = data.copy(deep=True)
    quality_before = copy.deepcopy(quality)
    sources_before = copy.deepcopy(sources)

    build_unified_table(data, quality, sources)

    pd.testing.assert_frame_equal(data, data_before)
    assert list(quality) == list(quality_before)
    for name in quality:
        pd.testing.assert_frame_equal(quality[name], quality_before[name])
    for name in sources:
        pd.testing.assert_frame_equal(sources[name], sources_before[name])


def test_table_order_follows_the_supported_sources_not_the_dict_order():
    data = make_dataset()
    quality = quality_results(data)
    reversed_quality = dict(reversed(list(quality.items())))

    first, _ = build_unified_table(data, quality)
    second, _ = build_unified_table(data, reversed_quality)

    pd.testing.assert_frame_equal(first, second)


# ----------------------------------------------------------------------
# build_unified_table: safe handling of bad results
# ----------------------------------------------------------------------
def test_misaligned_ml_results_are_skipped_with_a_note():
    data = make_dataset()
    other = pd.RangeIndex(0, 5)                        # results of another dataset
    ml = ml_frames(other, if_flagged={1}, lof_flagged={1})

    unified, notes = build_unified_table(data, quality_results(data), ml)

    assert not (unified["Category"] == CATEGORY_ANOMALY).any()
    assert len(notes) == 3
    assert all("do not line up" in note for note in notes)
    assert any("Isolation Forest" in note for note in notes)
    assert not unified.empty                            # quality findings survive


def test_ml_results_passed_in_the_quality_dict_are_checked_too():
    data = make_dataset()
    ml = ml_frames(pd.RangeIndex(0, 5), if_flagged={1}, lof_flagged={1})

    unified, notes = build_unified_table(data, {SRC_IF: ml[SRC_IF]})

    assert unified.empty
    assert len(notes) == 1


def test_a_failing_source_becomes_a_note_and_the_others_are_kept():
    data = make_dataset(n=5)
    quality = {
        SRC_INVALID: pd.DataFrame(
            [{"Row": 2, "Column": "Age", "Value": 150, "Status": "Invalid Value"}]),
        # Row 99 means label 98, which is not in the dataset.
        SRC_FORMAT: pd.DataFrame(
            [{"Row": 99, "Column": "Email", "Value": "x", "Format": "Email",
              "Status": "Invalid Format"}]),
        # Wrong type of result.
        SRC_MISSING: "not a DataFrame",
    }

    unified, notes = build_unified_table(data, quality)

    assert list(unified["Issue Type"]) == ["Invalid Value"]
    assert len(notes) == 2
    assert any("Format problems" in n and "not in the supplied index" in n for n in notes)
    assert any("Missing values" in n for n in notes)


def test_every_source_failing_gives_an_empty_table_and_notes():
    data = make_dataset(n=5)
    quality = {SRC_INVALID: pd.DataFrame(
        [{"Row": 50, "Column": "Age", "Value": 1, "Status": "Invalid Value"}])}

    unified, notes = build_unified_table(data, quality)

    assert unified.empty and list(unified.columns) == UNIFIED_COLUMNS
    assert len(notes) == 1


def test_unknown_source_name_is_rejected():
    data = make_dataset(n=5)
    with pytest.raises(ValueError, match="Unknown source"):
        build_unified_table(data, {"not_a_source": pd.DataFrame()})
    with pytest.raises(ValueError, match="Unknown source"):
        build_unified_table(data, {}, {"not_a_source": pd.DataFrame()})


@pytest.mark.parametrize("bad", [None, [], "x", 5])
def test_quality_results_must_be_a_dict(bad):
    with pytest.raises(TypeError):
        build_unified_table(make_dataset(n=5), bad)


def test_anomaly_sources_must_be_a_dict_or_none():
    with pytest.raises(TypeError):
        build_unified_table(make_dataset(n=5), {}, [1, 2])


def test_dataset_must_be_a_dataframe():
    with pytest.raises(TypeError):
        build_unified_table([[1]], {})


# ----------------------------------------------------------------------
# Summary, counts, filter, display
# ----------------------------------------------------------------------
@pytest.fixture
def unified_mixed():
    data = make_dataset()
    ml = ml_frames(data.index, if_flagged={4, 8}, lof_flagged={8, 12})
    unified, _ = build_unified_table(data, quality_results(data), ml)
    return unified


def test_summary_counts_by_category(unified_mixed):
    summary = get_unified_summary(unified_mixed)
    category = unified_mixed["Category"]

    assert summary["total_findings"] == len(unified_mixed)
    assert summary["data_quality"] == (category == CATEGORY_QUALITY).sum()
    assert summary["statistical_outliers"] == (category == CATEGORY_OUTLIER).sum()
    assert summary["ml_anomaly_findings"] == (category == CATEGORY_ANOMALY).sum()
    assert summary["data_quality"] + summary["statistical_outliers"] \
        + summary["ml_anomaly_findings"] == summary["total_findings"]
    assert all(type(v) is int for v in summary.values())


def test_summary_counts_each_ml_record_once(unified_mixed):
    summary = get_unified_summary(unified_mixed)

    # IF flags 2 + LOF flags 2 + combined 3 = 7 rows, but only 3 records.
    assert summary["ml_anomaly_findings"] == 7
    assert summary["ml_records_to_review"] == 3


def test_summary_of_an_empty_table():
    unified, _ = build_unified_table(make_dataset(n=5), {})

    assert get_unified_summary(unified) == {
        "total_findings": 0, "data_quality": 0, "statistical_outliers": 0,
        "ml_anomaly_findings": 0, "ml_records_to_review": 0,
    }


def test_count_by_issue_type(unified_mixed):
    counts = count_by_issue_type(unified_mixed)

    assert list(counts.columns) == ["Category", "Issue Type", "Method", "Findings"]
    assert counts["Findings"].sum() == len(unified_mixed)
    assert counts["Issue Type"].is_unique
    by_type = dict(zip(counts["Issue Type"], counts["Findings"]))
    assert by_type["Combined Anomaly"] == 3
    assert by_type["Isolation Forest Anomaly"] == 2
    assert by_type["LOF Anomaly"] == 2


def test_count_by_issue_type_of_an_empty_table():
    unified, _ = build_unified_table(make_dataset(n=5), {})

    counts = count_by_issue_type(unified)

    assert counts.empty
    assert list(counts.columns) == ["Category", "Issue Type", "Method", "Findings"]


def test_filter_by_category_and_issue_type(unified_mixed):
    only_ml = filter_unified_table(unified_mixed, categories=[CATEGORY_ANOMALY])
    assert set(only_ml["Category"]) == {CATEGORY_ANOMALY}
    assert len(only_ml) == 7

    only_type = filter_unified_table(unified_mixed, issue_types=["Invalid Value"])
    assert set(only_type["Issue Type"]) == {"Invalid Value"}

    both = filter_unified_table(
        unified_mixed, categories=[CATEGORY_QUALITY], issue_types=["LOF Anomaly"])
    assert both.empty


def test_filter_none_keeps_everything_and_empty_list_selects_nothing(unified_mixed):
    everything = filter_unified_table(unified_mixed)
    nothing = filter_unified_table(unified_mixed, categories=[])

    assert len(everything) == len(unified_mixed)
    assert nothing.empty and list(nothing.columns) == UNIFIED_COLUMNS
    assert list(everything.index) == list(range(len(everything)))


def test_filter_does_not_modify_the_table(unified_mixed):
    before = unified_mixed.copy(deep=True)

    filter_unified_table(unified_mixed, categories=[CATEGORY_ANOMALY])

    pd.testing.assert_frame_equal(unified_mixed, before)


def test_display_table_shows_mixed_values_as_text(unified_mixed):
    before = unified_mixed.copy(deep=True)

    shown = prepare_display_table(unified_mixed)

    assert list(shown.columns) == UNIFIED_COLUMNS
    assert len(shown) == len(unified_mixed)
    for column in ("Record ID", "Value"):
        assert all(isinstance(v, str) for v in shown[column])
    values = dict(zip(shown["Issue Type"] + shown["Record ID"], shown["Value"]))
    assert values["Invalid Value5"] == "Twenty"
    assert values["Invalid Value3"] == "150"
    assert values["Missing Value"] == ""                 # no value / record: blank
    pd.testing.assert_frame_equal(unified_mixed, before)  # original untouched
    assert str(shown["Row Position"].dtype) == "Int64"


def test_helpers_reject_frames_that_are_not_unified_tables():
    for helper in (get_unified_summary, count_by_issue_type,
                   filter_unified_table, prepare_display_table):
        with pytest.raises(TypeError):
            helper([1, 2])
        with pytest.raises(ValueError, match="missing required columns"):
            helper(pd.DataFrame({"Category": ["Data Quality"]}))


# ----------------------------------------------------------------------
# app.py wiring (static checks: Streamlit is not started)
# ----------------------------------------------------------------------
def _app_tree():
    return ast.parse(APP_PATH.read_text(encoding="utf-8"), filename=str(APP_PATH))


def test_app_compiles():
    compile(APP_PATH.read_text(encoding="utf-8"), str(APP_PATH), "exec")


def test_app_imports_only_names_that_exist():
    checked = 0
    for node in ast.walk(_app_tree()):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("modules."):
            module = __import__(node.module, fromlist=["_"])
            for alias in node.names:
                assert hasattr(module, alias.name), f"{node.module}.{alias.name} does not exist"
                checked += 1
    assert checked > 10


def test_app_passes_only_supported_source_keys_to_the_unified_section():
    calls = [
        node for node in ast.walk(_app_tree())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "render_unified_results"
    ]
    assert len(calls) == 1
    sources = calls[0].args[1]
    assert isinstance(sources, ast.Dict)

    keys = []
    for key in sources.keys:
        assert isinstance(key, ast.Name) and key.id.startswith("SRC_")
        keys.append(getattr(unified_results, key.id))   # must be a real constant
    assert len(keys) == len(set(keys))                    # no source passed twice
    assert set(keys) <= set(SUPPORTED_SOURCES)
    # The ML sources are produced by run_anomaly_sources, never passed here.
    assert not set(keys) & set(ML_SOURCES)


def test_app_has_one_clearly_labelled_unified_results_section():
    source = APP_PATH.read_text(encoding="utf-8")

    assert source.count('st.header("🧩 Unified Results")') == 1
    # Existing sections are still there.
    for header in ('"📊 Data Profiling"', '"🔍 Data Quality Detection"',
                   '"Statistical Outlier Detection"', '"ML Preprocessing"'):
        assert f"st.header({header})" in source


def test_unified_dashboard_has_no_ui_code():
    source = Path(unified_dashboard.__file__).read_text(encoding="utf-8")

    assert "streamlit" not in source
