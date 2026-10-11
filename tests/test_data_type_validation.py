import pandas as pd

from modules.data_type_validation import detect_data_type_problems


def test_valid_text_columns_are_not_flagged():
    df = pd.DataFrame({
        "Name": ["Amit", "Priya", "Rahul"],
        "City": ["Pune", "Mumbai", "Kolhapur"],
    })

    result = detect_data_type_problems(df)

    assert result.empty


def test_invalid_value_in_numeric_column_is_flagged():
    df = pd.DataFrame({
        "Age": [21, 25, "unknown", 30],
    })

    result = detect_data_type_problems(df)

    assert len(result) == 1
    assert result.iloc[0]["Column"] == "Age"
    assert result.iloc[0]["Value"] == "unknown"
    assert result.iloc[0]["Status"] == "Data Type Problem"


def test_missing_values_are_not_flagged():
    df = pd.DataFrame({
        "Age": [21, None, 30],
    })

    result = detect_data_type_problems(df)

    assert result.empty


def test_valid_numeric_values_are_not_flagged():
    df = pd.DataFrame({
        "Age": [18, 21, 25, 30],
    })

    result = detect_data_type_problems(df)

    assert result.empty