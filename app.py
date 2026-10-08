import streamlit as st

from modules.data_loader import load_dataset

from modules.profiling import (
    get_basic_info,
    get_column_info,
    get_numerical_statistics,
    get_duplicate_count
)

from modules.missing_values import detect_missing_values

from modules.duplicates import detect_duplicates


# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="Intelligent Data Quality Platform",
    page_icon="📊",
    layout="wide"
)


# -----------------------------------
# Title
# -----------------------------------

st.title("Intelligent Data Quality & Anomaly Platform")

st.write(
    "Upload your dataset to analyze data quality "
    "and detect potential problems."
)


# -----------------------------------
# File Upload
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls", "json"]
)


# -----------------------------------
# Process Dataset
# -----------------------------------

if uploaded_file is not None:

    try:

        # Load dataset
        df = load_dataset(uploaded_file)

        if df is None:
            st.error("Unable to load the dataset.")
            st.stop()

        st.success("Dataset uploaded successfully!")


        # ===================================
        # PHASE 3 — DATA PROFILING
        # ===================================

        st.header("📊 Data Profiling")


        # -----------------------------------
        # Basic Information
        # -----------------------------------

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


        # -----------------------------------
        # Column Names
        # -----------------------------------

        st.subheader("Column Names")

        st.write(
            basic_info["column_names"]
        )


        # -----------------------------------
        # Dataset Preview
        # -----------------------------------

        st.subheader("Dataset Preview")

        st.dataframe(
            df,
            use_container_width=True
        )


        # -----------------------------------
        # Column Information
        # -----------------------------------

        st.subheader("Column Information")

        column_info = get_column_info(df)

        st.dataframe(
            column_info,
            use_container_width=True
        )


        # -----------------------------------
        # Numerical Statistics
        # -----------------------------------

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


        # ===================================
        # PHASE 4 — DATA QUALITY DETECTION
        # ===================================

        st.header("🔍 Data Quality Detection")


        # ===================================
        # STEP 4 — MISSING VALUE DETECTION
        # ===================================

        st.subheader("Missing Value Detection")

        missing_values = detect_missing_values(df)

        st.dataframe(
            missing_values,
            use_container_width=True
        )


        # ===================================
        # STEP 5 — DUPLICATE DETECTION
        # ===================================

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


    # -----------------------------------
    # Error Handling
    # -----------------------------------

    except Exception as e:

        st.error(
            f"Error while processing dataset: {e}"
        )