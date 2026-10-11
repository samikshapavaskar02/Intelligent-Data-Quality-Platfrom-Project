
import pandas as pd


def detect_invalid_values(df, rules):
    invalid_rows = []

    for column, limits in rules.items():
        if column not in df.columns:
            continue

        minimum = limits[0]
        maximum = limits[1]

        for row_number, (index, value) in enumerate(
            df[column].items(), start=1
        ):
            if pd.isna(value):
                continue

            # Convert values to numeric before checking the limits.
            numeric_value = pd.to_numeric(value, errors="coerce")

            # Non-numeric values are invalid when numeric limits apply.
            if pd.isna(numeric_value):
                invalid_rows.append({
                    "Row": row_number,
                    "Column": column,
                    "Value": value,
                    "Status": "Invalid Value",
                })
                continue

            if numeric_value < minimum or numeric_value > maximum:
                invalid_rows.append({
                    "Row": row_number,
                    "Column": column,
                    "Value": value,
                    "Status": "Invalid Value",
                })

    return pd.DataFrame(
        invalid_rows,
        columns=["Row", "Column", "Value", "Status"],
    )
