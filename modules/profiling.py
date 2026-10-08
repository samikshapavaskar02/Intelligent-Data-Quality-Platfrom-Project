import pandas as pd


# Get basic information about the dataset
def get_basic_info(df):

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns)
    }


# Detect the data type of a column
def get_data_type(column):

    if pd.api.types.is_bool_dtype(column):
        return "Boolean"

    elif pd.api.types.is_integer_dtype(column):
        return "Integer"

    elif pd.api.types.is_float_dtype(column):
        return "Float"

    elif pd.api.types.is_datetime64_any_dtype(column):
        return "Date"

    else:
        return "String"


# Get information about every column
def get_column_info(df):

    column_info = []

    for column in df.columns:

        column_info.append({
            "Column": column,
            "Data Type": get_data_type(df[column]),
            "Missing Values": df[column].isnull().sum(),
            "Unique Values": df[column].nunique()
        })

    return pd.DataFrame(column_info)


# Get numerical statistics
def get_numerical_statistics(df):

    numerical_df = df.select_dtypes(include="number")

    # If there are no numerical columns
    if numerical_df.empty:
        return pd.DataFrame()

    statistics = numerical_df.describe().T

    statistics = statistics[
        ["mean", "50%", "min", "max", "std"]
    ]

    statistics = statistics.rename(
        columns={
            "50%": "median"
        }
    )

    return statistics


# Count duplicate rows
def get_duplicate_count(df):

    return df.duplicated().sum()