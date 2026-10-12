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

from modules.unified_dashboard import (
    build_unified_table,
    count_by_issue_type,
    filter_unified_table,
    get_dataset_fingerprint,
    get_unified_summary,
    prepare_display_table,
    run_anomaly_sources
)

from modules.unified_results import (
    SRC_MISSING,
    SRC_DUPLICATES,
    SRC_INVALID,
    SRC_DATA_TYPE,
    SRC_FORMAT,
    SRC_CONSISTENCY,
    SRC_IQR
)

from modules.quality_score import (
    CAP_CRITICAL,
    CAP_HIGH,
    RATING_BANDS,
    RATING_NOT_SCORED,
    SEV_CRITICAL,
    SEV_HIGH,
    SEV_LOW,
    SEV_MEDIUM,
    SEV_REVIEW,
    WEIGHTS,
    add_severity_columns,
    calculate_quality_score
)

# Page Configuration
st.set_page_config(
    page_title="Intelligent Data Quality Platform",
    page_icon="📊",
    layout="wide"
)


# Unified Results helpers
@st.cache_data(show_spinner=False)
def get_anomaly_sources(_data, fingerprint):
    """Run Isolation Forest + LOF once per dataset (cached between reruns).

    The cache key is the full-content fingerprint; ``_data`` is not hashed.
    """
    return run_anomaly_sources(_data)


# Data Quality Score helpers

# Rated severities named by modules.quality_score, worst first.
RATED_SEVERITIES_WORST_FIRST = (
    SEV_CRITICAL,
    SEV_HIGH,
    SEV_MEDIUM,
    SEV_LOW
)

# Score notes that are ordinary explanations (shown as neutral information).
# Every other note from the score module describes a scoring limitation, a
# partial result or a cap, and is shown as a warning. A note that matches none
# of these patterns therefore defaults to a warning.
INFORMATIONAL_NOTE_PATTERNS = (
    "were rescaled to total 100%",
    "are on blank cells and are counted once"
)


def get_worst_severity(severity_summary):
    """Worst rated severity named in the score's severity summary.

    modules.quality_score sorts the summary worst-first, but this does not
    rely on that order: the worst severity present is chosen with an explicit
    ranking. Returns None when the summary is missing, empty or names no rated
    severity, so nothing is ever inferred.
    """
    if not isinstance(severity_summary, pd.DataFrame):
        return None

    if severity_summary.empty or "Severity" not in severity_summary.columns:
        return None

    present = set(severity_summary["Severity"].dropna())

    for severity in RATED_SEVERITIES_WORST_FIRST:
        if severity in present:
            return severity

    return None


def is_informational_note(note):
    """True for ordinary explanations; False for limitations and warnings."""
    return any(pattern in note for pattern in INFORMATIONAL_NOTE_PATTERNS)


def render_quality_score(result):
    """Show the Data Quality Score block for a QualityScoreResult."""

    st.subheader("Data Quality Score")

    worst_severity = get_worst_severity(result.severity_summary)

    if worst_severity is not None:
        worst_text = worst_severity
    elif result.overall_score is None:
        worst_text = "Not available"
    else:
        worst_text = "No rated issues"

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Overall Score",
            RATING_NOT_SCORED if result.overall_score is None
            else f"{result.overall_score} / 100"
        )

    with col2:
        st.metric("Rating", result.rating)

    with col3:
        st.metric("Worst Severity", worst_text)

    with col4:
        st.metric(
            "Score Before Limit",
            RATING_NOT_SCORED if result.raw_score is None
            else f"{result.raw_score} / 100"
        )

    # Severity limit (cap)
    if result.cap_applied:

        st.warning(
            result.cap_reason
            or f"The score was limited to {result.cap_limit} because of "
               "the worst severity found."
        )

    elif result.cap_limit is not None:

        st.caption(
            f"A severity limit of {result.cap_limit} applies to this dataset, "
            "but the score before the limit was already within it."
        )

    st.caption(
        f'IQR outliers and ML anomalies are shown as "{SEV_REVIEW}" '
        "and are never part of the score."
    )

    st.subheader("Dimension Breakdown")

    st.dataframe(
        result.dimensions,
        use_container_width=True
    )

    st.subheader("Severity Summary")

    if result.severity_summary.empty:

        st.info("No rated issue groups, so there is no severity summary.")

    else:

        st.dataframe(
            result.severity_summary,
            use_container_width=True
        )

    # Notes: the cap explanation is already shown above, so it is not repeated.
    for note in result.notes:

        if note == result.cap_reason:
            continue

        if is_informational_note(note):
            st.info(note)
        else:
            st.warning(note)

    with st.expander("How the score is calculated"):

        weights_text = ", ".join(
            f"{name} {weight * 100:g}%" for name, weight in WEIGHTS.items()
        )

        band_parts = []

        for position, (lowest, name) in enumerate(RATING_BANDS):

            if position == len(RATING_BANDS) - 1 and lowest <= 0:
                band_parts.append(
                    f"{name}: below {RATING_BANDS[position - 1][0]:g}"
                )
            else:
                band_parts.append(f"{name}: {lowest:g} and above")

        st.markdown(
            "- Each dimension is scored from 0 to 100 as "
            "100 x (1 - problems / items checked).\n"
            "- The overall score is the weighted total of the assessed "
            f"dimensions ({weights_text}). If a dimension cannot be assessed, "
            "its weight is shared among the others.\n"
            "- The worst severity found limits the score: a Critical issue "
            f"caps it at {CAP_CRITICAL}, a High issue at {CAP_HIGH}.\n"
            "- The score is rounded to one decimal after the limit, and the "
            f"rating comes from the rounded score ({'; '.join(band_parts)}).\n"
            f'- IQR outliers and ML anomalies are marked "{SEV_REVIEW}" and '
            "never change the score.\n"
            "- This is the project's own scoring method, not an industry "
            "standard."
        )


def render_unified_results(data, quality_results):
    """Show every finding from every detector in one consistent table."""

    st.write(
        "All findings from the checks above in one table. "
        "Data quality problems are actual errors. IQR outliers and "
        "ML anomalies are unusual records to review; they are not "
        "automatically wrong and nothing is removed."
    )

    if data.empty:
        st.info("The uploaded dataset is empty, so there are no findings to combine.")
        return

    include_ml = st.checkbox(
        "Include ML anomaly detection (Isolation Forest + LOF)",
        value=True,
        help=(
            "Runs both models with their default settings. "
            "Untick to show the data quality and IQR findings only."
        ),
        key="unified_include_ml"
    )

    anomaly_sources = {}

    if include_ml:

        with st.spinner("Running anomaly detection..."):
            anomaly_sources, anomaly_notes = get_anomaly_sources(
                data,
                get_dataset_fingerprint(data)
            )

        for note in anomaly_notes:
            st.info(note)

    unified, build_notes = build_unified_table(
        data,
        quality_results,
        anomaly_sources
    )

    for note in build_notes:
        st.warning(note)

    # Data Quality Score (calculated before the empty-findings check so a
    # dataset without findings still gets its score)
    try:
        score_result = calculate_quality_score(
            data,
            quality_results,
            unified
        )
    except (TypeError, ValueError) as error:
        score_result = None
        st.warning(f"The data quality score could not be calculated: {error}")

    if score_result is not None:
        render_quality_score(score_result)

    if unified.empty:
        st.success("No findings detected by the supported checks.")
        return

    # Severity and Score Treatment are display-only columns. They are added to
    # the FULL table (same index as the score) before any filtering.
    findings_table = unified

    if score_result is not None:

        try:
            findings_table = add_severity_columns(unified, score_result)
        except (TypeError, ValueError) as error:
            st.warning(
                f"Severity columns could not be added to the table: {error}"
            )

    # Summary
    summary = get_unified_summary(unified)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Data Quality Findings", summary["data_quality"])

    with col2:
        st.metric("IQR Outliers", summary["statistical_outliers"])

    with col3:
        st.metric("ML Records to Review", summary["ml_records_to_review"])

    with col4:
        st.metric("Total Findings", summary["total_findings"])

    st.subheader("Findings by Issue Type")

    st.dataframe(
        count_by_issue_type(unified),
        use_container_width=True
    )

    # Filters
    st.subheader("All Findings")

    available_categories = list(unified["Category"].unique())

    selected_categories = st.multiselect(
        "Category",
        available_categories,
        default=available_categories
    )

    available_issue_types = list(
        unified.loc[
            unified["Category"].isin(selected_categories),
            "Issue Type"
        ].unique()
    )

    selected_issue_types = st.multiselect(
        "Issue Type",
        available_issue_types,
        default=available_issue_types
    )

    filtered = filter_unified_table(
        findings_table,
        categories=selected_categories,
        issue_types=selected_issue_types
    )

    if filtered.empty:
        st.info("No findings match the selected filters.")
        return

    st.dataframe(
        prepare_display_table(filtered),
        use_container_width=True
    )

    if "Severity" in filtered.columns:
        st.caption(
            "Severity and Score Treatment are display-only columns. "
            f'IQR outliers and ML anomalies are shown as "{SEV_REVIEW}" '
            "and are excluded from the score."
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
            numerical_columns, categorical_columns = identify_columns(df)

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

        # PHASE 11 — UNIFIED RESULTS
        st.header("🧩 Unified Results")

        render_unified_results(
            df,
            {
                SRC_MISSING: missing_values,
                SRC_DUPLICATES: duplicate_rows,
                SRC_INVALID: invalid_values,
                SRC_DATA_TYPE: data_type_problems,
                SRC_FORMAT: format_problems,
                SRC_CONSISTENCY: inconsistencies,
                SRC_IQR: outliers
            }
        )

    # Error Handling
    except Exception as e:
        st.error(f"Error while processing dataset: {e}")
        st.exception(e)