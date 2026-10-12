import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# STEP 11: Identify column types
def identify_columns(df):
    numerical_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        column for column in df.columns if column not in numerical_columns
    ]
    return numerical_columns, categorical_columns


# STEP 12: Report original missing values
def get_missing_value_report(df):
    data = df.copy()
    data = data.replace(r"^\s*$", pd.NA, regex=True)
    missing_count = data.isna().sum()
    report = pd.DataFrame({
        "Column": data.columns,
        "Missing Values": missing_count.values,
    })
    if len(data) > 0:
        report["Missing Percentage"] = (
            report["Missing Values"] / len(data) * 100
        ).round(2)
    else:
        report["Missing Percentage"] = 0.0
    return report


# STEPS 11-14: Complete preprocessing
def preprocess_for_ml(df, encoding_method="Automatic", ordinal_orders=None):
    if df.empty:
        raise ValueError("The uploaded dataset is empty.")
    if encoding_method not in ("Automatic", "One-Hot", "Frequency", "Ordinal"):
        raise ValueError(f"Unsupported encoding method: {encoding_method}")

    data = df.copy()
    data = data.replace(r"^\s*$", pd.NA, regex=True)
    numerical_columns, categorical_columns = identify_columns(data)
    missing_report = get_missing_value_report(data)

    numerical_data = data[numerical_columns].copy()
    categorical_data = data[categorical_columns].copy()

    # Step 12A: Numerical missing values -> median; all-missing -> 0
    if numerical_columns:
        numerical_data = numerical_data.replace([np.inf, -np.inf], np.nan)
        for column in numerical_columns:
            numeric_series = pd.to_numeric(numerical_data[column], errors="coerce")
            numerical_data[column] = numeric_series.astype("float64")
            if numerical_data[column].isna().all():
                numerical_data[column] = 0.0

        numerical_imputer = SimpleImputer(strategy="median")
        numerical_array = numerical_imputer.fit_transform(numerical_data)
        numerical_data = pd.DataFrame(
            numerical_array, columns=numerical_columns, index=data.index
        )

    # Step 12B: Categorical missing values -> mode; all-missing -> Unknown
    # Avoid passing pandas pd.NA to sklearn SimpleImputer.
    if categorical_columns:
        for column in categorical_columns:
            series = categorical_data[column].astype("string")
            series = series.mask(series.str.strip().eq("").fillna(False), pd.NA)
            modes = series.mode(dropna=True)
            fill_value = modes.iloc[0] if not modes.empty else "Unknown"
            categorical_data[column] = series.fillna(fill_value).astype(str)

    # Step 13: Encode categorical columns
    encoded_data = pd.DataFrame(index=data.index)
    encoding_details = []
    ordinal_orders = ordinal_orders or {}

    for column in categorical_columns:
        unique_count = categorical_data[column].nunique()
        if encoding_method == "Automatic":
            method = "One-Hot" if unique_count <= 10 else "Frequency"
        else:
            method = encoding_method

        if method == "One-Hot":
            encoded_column = pd.get_dummies(
                categorical_data[column], prefix=str(column), dtype=int
            )
            encoded_data = pd.concat([encoded_data, encoded_column], axis=1)

        elif method == "Frequency":
            frequencies = categorical_data[column].value_counts(normalize=True)
            encoded_data[f"{column}_frequency"] = (
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
                value.strip() for value in order_text.split(",") if value.strip()
            ]
            if len(categories) != len(set(categories)):
                raise ValueError(f"Duplicate categories found in the order for '{column}'.")
            mapping = {
                category: position for position, category in enumerate(categories)
            }
            unknown_values = (
                set(categorical_data[column].unique()) - set(mapping.keys())
            )
            if unknown_values:
                raise ValueError(
                    f"Add these values to the ordinal order for "
                    f"'{column}': {sorted(unknown_values)}"
                )
            encoded_data[f"{column}_ordinal"] = (
                categorical_data[column].map(mapping)
            )

        encoding_details.append({
            "Column": column,
            "Unique Categories": unique_count,
            "Encoding Method": method,
        })

    # Step 14: Scale numerical columns
    if numerical_columns:
        scaled_array = StandardScaler().fit_transform(numerical_data)
        scaled_data = pd.DataFrame(
            scaled_array, columns=numerical_columns, index=data.index
        )
    else:
        scaled_data = pd.DataFrame(index=data.index)

    processed_data = pd.concat([scaled_data, encoded_data], axis=1)
    return {
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "missing_report": missing_report,
        "encoding_details": pd.DataFrame(encoding_details),
        "processed_data": processed_data,
    }
