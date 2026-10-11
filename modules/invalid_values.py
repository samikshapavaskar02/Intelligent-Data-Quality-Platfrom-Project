import pandas as pd


def detect_invalid_values(df, rules):

    invalid_rows = []

    for column, limits in rules.items():

        if column not in df.columns:
            continue

        minimum = limits[0]
        maximum = limits[1]

        for index, value in df[column].items():

            if pd.isna(value):
                continue

            if value < minimum or value > maximum:

                invalid_rows.append({
                    "Row": index + 1,
                    "Column": column,
                    "Value": value,
                    "Status": "Invalid Value"
                })

    return pd.DataFrame(invalid_rows)