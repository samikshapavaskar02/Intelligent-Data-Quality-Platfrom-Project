
import pandas as pd


def detect_iqr_outliers(df):

    outliers = []

    # Select numeric columns only
    numeric_columns = df.select_dtypes(include="number").columns

    for column in numeric_columns:

        # Ignore missing values while calculating bounds
        values = df[column].dropna()

        # Need at least 4 values for a useful IQR check
        if len(values) < 4:
            continue

        # Calculate Q1 and Q3
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)

        # Calculate IQR
        iqr = q3 - q1

        # Calculate lower and upper limits
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        # Find values outside the limits
        for index, value in df[column].items():

            if pd.isna(value):
                continue

            if value < lower_bound or value > upper_bound:

                outliers.append({
                    "Row": index + 1,
                    "Column": column,
                    "Value": value,
                    "Lower Bound": round(lower_bound, 2),
                    "Upper Bound": round(upper_bound, 2),
                    "Method": "IQR",
                    "Status": "Potential Outlier"
                })

    return pd.DataFrame(outliers)
