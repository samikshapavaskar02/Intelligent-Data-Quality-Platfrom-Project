import streamlit as st
import pandas as pd

from modules.data_loader import load_dataset

from modules.profiling import (
    get_basic_info,
    get_column_info,
    get_numerical_statistics,
    get_duplicate_count
)

from modules.missing_values import detect_missing_values

from modules.duplicates import detect_duplicates

from modules.invalid_values import detect_invalid_values

from modules.data_type_validation import detect_data_type_problems

from modules.format_validation import detect_format_problems

from modules.consistency import detect_inconsistent_values

from modules.statistical_outliers import detect_iqr_outliers

from modules.ml_preprocessing import (identify_columns,preprocess_for_ml)

# Page Configuration
st.set_page_config(
    page_title="Intelligent Data Quality Platform",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("Intelligent Data Quality & Anomaly Platform")

st.write(
    "Upload your dataset to analyze data quality "
    "and detect potential problems."
)

# File Upload
uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls", "json"]
)

# Process Dataset
if uploaded_file is not None:

    try:

        # Load dataset
        df = load_dataset(uploaded_file)

        if df is None:
            st.error("Unable to load the dataset.")
            st.stop()

        st.success("Dataset uploaded successfully!")


        # PHASE 3 — DATA PROFILING
        st.header("📊 Data Profiling")

        # Basic Information
        basic_info = get_basic_info(df)

        st.subheader("Dataset Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Rows",
                basic_info["rows"]
            )

        with col2:
            st.metric(
                "Columns",
                basic_info["columns"]
            )

        with col3:
            st.metric(
                "Duplicate Rows",
                get_duplicate_count(df)
            )

        # Column Names
        st.subheader("Column Names")

        st.write(
            basic_info["column_names"]
        )

        # Dataset Preview
        st.subheader("Dataset Preview")

        st.dataframe(
            df,
            use_container_width=True
        )

        # Column Information
        st.subheader("Column Information")

        column_info = get_column_info(df)

        st.dataframe(
            column_info,
            use_container_width=True
        )

        # Numerical Statistics
        st.subheader("Numerical Statistics")

        statistics = get_numerical_statistics(df)

        if not statistics.empty:

            st.dataframe(
                statistics,
                use_container_width=True
            )

        else:

            st.info(
                "No numerical columns found."
            )

        # PHASE 4 — DATA QUALITY DETECTION
        st.header("🔍 Data Quality Detection")

        # MISSING VALUE DETECTION
        st.subheader("Missing Value Detection")

        missing_values = detect_missing_values(df)

        st.dataframe(
            missing_values,
            use_container_width=True
        )

        # DUPLICATE DETECTION
        st.subheader("Duplicate Detection")

        duplicate_rows = detect_duplicates(df)

        st.write(
            "Total Duplicate Rows:",
            get_duplicate_count(df)
        )

        if not duplicate_rows.empty:

            st.warning(
                "Duplicate rows detected."
            )

            st.dataframe(
                duplicate_rows,
                use_container_width=True
            )

        else:

            st.success(
                "No duplicate rows found."
            )

        # INVALID VALUE DETECTION
        st.subheader("Invalid Value Detection")

        rules = {
            "Age": (0, 120),
            "Marks": (0, 100),
            "Rating": (1, 5)
        }

        invalid_values = detect_invalid_values(df, rules)

        if not invalid_values.empty:
            st.dataframe(
                invalid_values,
                use_container_width=True
            )
        else:
            st.success("No invalid values found.")

        # DATA TYPE PROBLEMS
        st.subheader("Data Type Problems")

        data_type_problems = detect_data_type_problems(df)

        if not data_type_problems.empty:

            st.warning(
                f"{len(data_type_problems)} data type problem(s) detected."
            )

            st.dataframe(
                data_type_problems,
                use_container_width=True
            )

        else:

            st.success(
                "No data type problems found."
            )
        # FORMAT VALIDATION
        st.subheader("Format Validation")

        format_problems = detect_format_problems(df)

        if not format_problems.empty:
            st.warning(
                f"{len(format_problems)} format problem(s) detected."
            )
            st.dataframe(
                format_problems,
                use_container_width=True
            )
        else:
            st.success("No format problems found.")

        # CONSISTENCY DETECTION
        st.subheader("Consistency Detection")

        inconsistencies = detect_inconsistent_values(df)

        if not inconsistencies.empty:
            st.warning(
                f"{len(inconsistencies)} inconsistent representation(s) detected."
            )
            st.dataframe(
                inconsistencies,
                use_container_width=True
            )
        else:
            st.success("No consistency problems found.")

        # PHASE 5 — STATISTICAL OUTLIER DETECTION
        st.header("Statistical Outlier Detection")
        st.subheader("IQR Outlier Detection")

        outliers = detect_iqr_outliers(df)

        if not outliers.empty:

            st.warning(
                f"{len(outliers)} potential outlier(s) detected."
            )

            st.dataframe(
                outliers,
                use_container_width=True
            )

        else:

            st.success(
                "No potential outliers detected using the IQR method."
            )


        # PHASE 6 — ML PREPROCESSING
        st.header("ML Preprocessing")

        if df.empty:

            st.warning("The uploaded dataset is empty.")

        else:

            # Step 11: Identify column types
            clean_df = df.replace(r"^\s*$", pd.NA, regex=True)

            numerical_columns, categorical_columns = identify_columns(
                clean_df
            )

            st.subheader("Step 11 — Automatic Column Identification")

            col1, col2 = st.columns(2)

            with col1:
                st.write("Numerical Columns")
                st.write(numerical_columns)

            with col2:
                st.write("Categorical Columns")
                st.write(categorical_columns)

            # Step 13: Choose encoding method
            st.subheader("Step 13 — Encoding Strategy")

            encoding_method = st.selectbox(
                "Choose categorical encoding",
                [
                    "Automatic",
                    "One-Hot",
                    "Frequency",
                    "Ordinal"
                ],
                help=(
                    "Automatic uses One-Hot for up to 10 unique categories "
                    "and Frequency Encoding for more than 10."
                ),
                key="ml_encoding_method"
            )

            # Collect category order for ordinal encoding
            ordinal_orders = {}

            if encoding_method == "Ordinal" and categorical_columns:

                st.info(
                    "Enter each column's categories from lowest to highest. "
                    "Use the exact values present in your dataset."
                )

                for column in categorical_columns:
                    ordinal_orders[column] = st.text_input(
                        f"Category order for {column}",
                        placeholder="Example: Small, Medium, Large",
                        key=f"ordinal_order_{column}"
                    )

            # Run preprocessing
            if st.button("Run ML Preprocessing", key="run_ml_preprocessing"):

                try:

                    result = preprocess_for_ml(
                        df,
                        encoding_method=encoding_method,
                        ordinal_orders=ordinal_orders
                    )

                    # Step 12: Show missing values before imputation
                    st.subheader("Step 12 — Original Missing Value Report")

                    st.write(
                        "Missing values below are counted before imputation."
                    )

                    st.dataframe(
                        result["missing_report"],
                        use_container_width=True
                    )

                    # Show the selected encoding for each column
                    st.subheader("Encoding Summary")

                    if not result["encoding_details"].empty:

                        st.dataframe(
                            result["encoding_details"],
                            use_container_width=True
                        )

                    else:

                        st.info("No categorical columns were found.")

                    # Show final processed dataset
                    st.subheader("Processed ML Dataset")

                    st.write(
                        "Missing values have been handled, categorical "
                        "features encoded, and numerical features scaled."
                    )

                    st.dataframe(
                        result["processed_data"],
                        use_container_width=True
                    )

                    st.success("ML preprocessing completed successfully!")

                    # Download the processed data
                    csv_data = result["processed_data"].to_csv(index=False)

                    st.download_button(
                        label="Download Processed Dataset",
                        data=csv_data,
                        file_name="processed_ml_dataset.csv",
                        mime="text/csv",
                        key="download_processed_ml_dataset"
                    )

                except ValueError as error:

                    st.warning(str(error))

    # Error Handling
    except Exception as e:

        st.error(
            f"Error while processing dataset: {e}"
        )