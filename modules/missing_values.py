import pandas as pd


def detect_missing_values(df):

    # Create a copy so the original dataset is not changed
    data = df.copy()

    # Convert blank and empty values into missing values
    data = data.replace(r'^\s*$', pd.NA, regex=True)

    # Count missing values in each column
    missing_count = data.isnull().sum()

    # Create result table
    result = pd.DataFrame({
        "Column": missing_count.index,
        "Missing Values": missing_count.values
    })

    # Calculate missing percentage
    result["Missing Percentage"] = (
        result["Missing Values"] / len(data) * 100
    ).round(2)

    return result