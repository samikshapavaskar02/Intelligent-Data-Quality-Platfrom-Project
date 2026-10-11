
import pandas as pd


def detect_data_type_problems(df):
    problems = []

    for column in df.columns:
        series = df[column]

        # Check whether the column contains numeric values.
        converted = pd.to_numeric(series, errors="coerce")

        # If the column has no numeric values, treat it as a text column.
        # Do not flag ordinary text such as names and cities.
        non_missing = series.notna()

        if not non_missing.any():
            continue

        if converted[non_missing].notna().sum() == 0:
            continue

        # Flag values that cannot be converted when the column
        # otherwise contains numeric data.
        invalid_mask = non_missing & converted.isna()

        for index in df.index[invalid_mask]:
            original_value = series.loc[index]

            problems.append({
                "Row": df.index.get_loc(index) + 1,
                "Column": column,
                "Value": original_value,
                "Status": "Data Type Problem",
            })

    return pd.DataFrame(
        problems,
        columns=["Row", "Column", "Value", "Status"],
    )
