import pandas as pd


def detect_data_type_problems(df):

    problems = []

    for column in df.columns:

        # Try converting the column to numeric
        converted = pd.to_numeric(df[column], errors="coerce")

        for index in df.index:

            original_value = df.loc[index, column]

            if pd.isna(original_value):
                continue

            # If conversion failed, the value is not numeric
            if pd.isna(converted.loc[index]):

                problems.append({
                    "Row": index + 1,
                    "Column": column,
                    "Value": original_value,
                    "Status": "Data Type Problem"
                })

    return pd.DataFrame(problems)