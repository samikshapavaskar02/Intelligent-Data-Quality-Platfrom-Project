import pandas as pd


def detect_inconsistent_values(df):

    inconsistencies = []

    for column in df.select_dtypes(include="object").columns:

        # Remove missing values
        values = df[column].dropna().astype(str)

        # Create normalized versions
        normalized = values.str.strip().str.lower()

        # Check whether multiple original forms exist
        groups = {}

        for original, normal in zip(values, normalized):

            if normal not in groups:
                groups[normal] = set()

            groups[normal].add(original)

        for normal_value, original_values in groups.items():

            if len(original_values) > 1:

                inconsistencies.append({
                    "Column": column,
                    "Normalized Value": normal_value,
                    "Different Representations": ", ".join(
                        sorted(original_values)
                    ),
                    "Status": "Inconsistent Representation"
                })

    return pd.DataFrame(inconsistencies)