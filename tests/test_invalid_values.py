
import pandas as pd

from modules.invalid_values import detect_invalid_values


def test_values_within_limits_are_valid():
    df = pd.DataFrame({"Age": [18, 25, 60]})

    result = detect_invalid_values(df, {"Age": [0, 120]})

    assert result.empty


def test_values_outside_limits_are_flagged():
    df = pd.DataFrame({"Age": [-5, 25, 150]})

    result = detect_invalid_values(df, {"Age": [0, 120]})

    assert len(result) == 2
    assert result["Value"].tolist() == [-5, 150]


def test_text_in_numeric_range_column_does_not_crash():
    df = pd.DataFrame({"Age": [20, "unknown", 30]})

    result = detect_invalid_values(df, {"Age": [0, 120]})

    assert len(result) == 1
    assert result.iloc[0]["Value"] == "unknown"
    assert result.iloc[0]["Status"] == "Invalid Value"


def test_missing_values_are_skipped():
    df = pd.DataFrame({"Age": [20, None, 30]})

    result = detect_invalid_values(df, {"Age": [0, 120]})

    assert result.empty


def test_unknown_rule_columns_are_ignored():
    df = pd.DataFrame({"Age": [20, 30]})

    result = detect_invalid_values(
        df,
        {"Salary": [0, 100000]},
    )

    assert result.empty
