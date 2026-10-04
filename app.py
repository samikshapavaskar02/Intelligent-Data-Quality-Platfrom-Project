import streamlit as st

from modules.data_loader import load_dataset


# Page configuration
st.set_page_config(
    page_title="Intelligent Data Quality Platform",
    page_icon="📊",
    layout="wide"
)


# Main title
st.title("Intelligent Data Quality & Anomaly Platform")

st.write(
    "Upload your dataset to analyze data quality and detect anomalies."
)


# File uploader
uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls", "json"]
)


# Process uploaded file
if uploaded_file is not None:

    try:

        # Load dataset
        df = load_dataset(uploaded_file)

        st.success("Dataset uploaded successfully!")

        # Dataset information
        st.subheader("Dataset Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Rows", df.shape[0])

        with col2:
            st.metric("Columns", df.shape[1])

        with col3:
            st.metric("Duplicate Rows", df.duplicated().sum())

        # Dataset preview
        st.subheader("Dataset Preview")

        st.dataframe(
            df,
            use_container_width=True
        )

    except Exception as e:

        st.error(f"Error while loading dataset: {e}")