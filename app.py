import streamlit as st

from modules.data_loader import load_dataset

from modules.profiling import (
    get_basic_info,
    get_column_info,
    get_numerical_statistics,
    get_duplicate_count
)


st.set_page_config(
    page_title="Intelligent Data Quality Platform",
    page_icon="📊",
    layout="wide"
)


st.title("Intelligent Data Quality & Anomaly Platform")

st.write(
    "Upload your dataset to analyze data quality and detect anomalies."
)


uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls", "json"]
)


if uploaded_file is not None:

    try:

        # Load dataset
        df = load_dataset(uploaded_file)

        st.success("Dataset uploaded successfully!")


        # Basic information
        basic_info = get_basic_info(df)

        st.subheader("Dataset Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Rows", basic_info["rows"])

        with col2:
            st.metric("Columns", basic_info["columns"])

        with col3:
            st.metric(
                "Duplicate Rows",
                get_duplicate_count(df)
            )


        # Column names
        st.subheader("Column Names")

        st.write(basic_info["column_names"])


        # Dataset preview
        st.subheader("Dataset Preview")

        st.dataframe(
            df,
            use_container_width=True
        )


        # Column information
        st.subheader("Column Information")

        column_info = get_column_info(df)

        st.dataframe(
            column_info,
            use_container_width=True
        )


        # Numerical statistics
        st.subheader("Numerical Statistics")

        statistics = get_numerical_statistics(df)

        if not statistics.empty:

            st.dataframe(
                statistics,
                use_container_width=True
            )

        else:

            st.info("No numerical columns found.")


    except Exception as e:

        st.error(f"Error while processing dataset: {e}")