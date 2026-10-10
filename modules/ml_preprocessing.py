
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------------
# STEP 11: Identify column types
# -----------------------------------

def identify_columns(df):

    numerical_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = [
        column for column in df.columns
        if column not in numerical_columns
    ]

    return numerical_columns, categorical_columns


# -----------------------------------
# STEP 12: Report original missing values
# -----------------------------------

def get_missing_value_report(df):

    data = df.copy()

    # Treat blank and whitespace-only cells as missing
    data = data.replace(r"^\s*$", pd.NA, regex=True)

    missing_count = data.isnull().sum()

    report = pd.DataFrame({
        "Column": data.columns,
        "Missing Values": missing_count.values
    })

    if len(data) > 0:
        report["Missing Percentage"] = (
            report["Missing Values"] / len(data) * 100
        ).round(2)
    else:
        report["Missing Percentage"] = 0.0

    return report


# -----------------------------------
# STEPS 11–14: Complete preprocessing
# -----------------------------------

def preprocess_for_ml(
    df,
    encoding_method="Automatic",
    ordinal_orders=None
):

    if df.empty:
        raise ValueError("The uploaded dataset is empty.")

    data = df.copy()

    # Treat blank strings as missing
    data = data.replace(r"^\s*$", pd.NA, regex=True)

    # Step 11: Identify columns
    numerical_columns, categorical_columns = identify_columns(data)

    # Step 12: Record missing values BEFORE filling
    missing_report = get_missing_value_report(data)

    # Separate the columns
    numerical_data = data[numerical_columns].copy()
    categorical_data = data[categorical_columns].copy()

    # -----------------------------------
    # Step 12A: Numerical missing values
    # -----------------------------------

    if numerical_columns:

        # Handle columns containing only missing values
        for column in numerical_columns:
            if numerical_data[column].isnull().all():
                numerical_data[column] = 0.0

        numerical_imputer = SimpleImputer(strategy="median")

        numerical_array = numerical_imputer.fit_transform(
            numerical_data
        )

        numerical_data = pd.DataFrame(
            numerical_array,
            columns=numerical_columns,
            index=data.index
        )

    # -----------------------------------
    # Step 12B: Categorical missing values
    # -----------------------------------

    if categorical_columns:

        # Convert categorical values consistently
        for column in categorical_columns:
            categorical_data[column] = (
                categorical_data[column].astype("string")
            )

            # Fallback for a completely empty column
            if categorical_data[column].isnull().all():
                categorical_data[column] = (
                    categorical_data[column].fillna("Unknown")
                )

        categorical_imputer = SimpleImputer(
            strategy="most_frequent"
        )

        categorical_array = categorical_imputer.fit_transform(
            categorical_data
        )

        categorical_data = pd.DataFrame(
            categorical_array,
            columns=categorical_columns,
            index=data.index
        ).astype(str)

    # -----------------------------------
    # Step 13: Encode categorical columns
    # -----------------------------------

    encoded_data = pd.DataFrame(index=data.index)
    encoding_details = []

    ordinal_orders = ordinal_orders or {}

    for column in categorical_columns:

        unique_count = categorical_data[column].nunique()

        # Choose encoding method
        if encoding_method == "Automatic":

            if unique_count <= 10:
                method = "One-Hot"
            else:
                method = "Frequency"

        else:
            method = encoding_method

        if method == "One-Hot":

            encoded_column = pd.get_dummies(
                categorical_data[column],
                prefix=column,
                dtype=int
            )

            encoded_data = pd.concat(
                [encoded_data, encoded_column],
                axis=1
            )

        elif method == "Frequency":

            frequencies = (
                categorical_data[column]
                .value_counts(normalize=True)
            )

            encoded_data[column + "_frequency"] = (
                categorical_data[column].map(frequencies)
            )

        elif method == "Ordinal":

            order_text = ordinal_orders.get(column, "").strip()

            if not order_text:
                raise ValueError(
                    f"Enter the category order for '{column}'. "
                    "Separate categories with commas."
                )

            categories = [
                value.strip()
                for value in order_text.split(",")
                if value.strip()
            ]

            if len(categories) != len(set(categories)):
                raise ValueError(
                    f"Duplicate categories found in the order for "
                    f"'{column}'."
                )

            mapping = {
                category: position
                for position, category in enumerate(categories)
            }

            unknown_values = (
                set(categorical_data[column].unique())
                - set(mapping.keys())
            )

            if unknown_values:
                raise ValueError(
                    f"Add these values to the ordinal order for "
                    f"'{column}': {sorted(unknown_values)}"
                )

            encoded_data[column + "_ordinal"] = (
                categorical_data[column].map(mapping)
            )

        encoding_details.append({
            "Column": column,
            "Unique Categories": unique_count,
            "Encoding Method": method
        })

    # -----------------------------------
    # Step 14: Scale numerical columns
    # -----------------------------------

    if numerical_columns:

        scaler = StandardScaler()

        scaled_array = scaler.fit_transform(numerical_data)

        scaled_data = pd.DataFrame(
            scaled_array,
            columns=numerical_columns,
            index=data.index
        )

    else:
        scaled_data = pd.DataFrame(index=data.index)

    # Combine the processed columns
    processed_data = pd.concat(
        [scaled_data, encoded_data],
        axis=1
    )

    return {
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "missing_report": missing_report,
        "encoding_details": pd.DataFrame(encoding_details),
        "processed_data": processed_data
    }
