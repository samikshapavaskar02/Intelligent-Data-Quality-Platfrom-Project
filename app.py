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

from modules.ml_evaluation import (
    DEFAULT_ANOMALY_FRACTION,
    DEFAULT_N_SAMPLES,
    DEFAULT_RANDOM_STATE,
    MIN_SAMPLES,
    run_ml_evaluation
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

from modules.cleaning import (
    INVALID_CLIP,
    INVALID_KEEP,
    INVALID_MISSING,
    MISSING_DROP,
    MISSING_FILL,
    MISSING_KEEP,
    clean_dataset
)

from modules.report_builder import (
    build_anomaly_report,
    build_quality_report
)

from modules.analysis_persistence import (
    get_or_create_analysis_id,
    persist_applied_cleaning,
)

from modules.database import (
    initialize_database,
    create_analysis,
    save_quality_score,
    save_quality_issues,
    save_anomaly_results,
    get_analysis_history,
    get_analysis_details,
    test_connection,
    save_analysis_bundle,
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


def render_report_downloads(score_result, anomaly_sources, include_ml):
    """Download buttons for the quality report and the anomaly report.

    Both reports are built from results the page already has (the data
    quality score and the cached anomaly results of the UPLOADED dataset).
    Nothing is recalculated here, and the synthetic ML Evaluation data is
    never used.
    """

    st.subheader("Download Reports")

    col1, col2 = st.columns(2)

    with col1:

        if score_result is None:

            st.info(
                "The quality report is not available because the data "
                "quality score could not be calculated."
            )

        else:

            try:
                quality_report, quality_notes = build_quality_report(
                    score_result
                )

            except (TypeError, ValueError) as error:
                st.warning(f"The quality report could not be built: {error}")

            else:

                for note in quality_notes:
                    st.info(note)

                st.download_button(
                    label="Download Quality Report",
                    data=quality_report.to_csv(index=False),
                    file_name="quality_report.csv",
                    mime="text/csv",
                    key="download_quality_report"
                )

                st.caption(
                    "One row per rated issue group from the Severity Summary. "
                    "Percentage is the share of the units checked. IQR "
                    "outliers and ML anomalies are not included."
                )

    with col2:

        if not include_ml:

            st.info(
                "The anomaly report is not available because ML anomaly "
                "detection is switched off above."
            )

        elif not anomaly_sources:

            st.info(
                "The anomaly report is not available because anomaly "
                "detection did not run for this dataset."
            )

        else:

            try:
                anomaly_report, anomaly_notes = build_anomaly_report(
                    anomaly_sources
                )

            except (TypeError, ValueError) as error:
                st.warning(f"The anomaly report could not be built: {error}")

            else:

                for note in anomaly_notes:
                    st.info(note)

                st.download_button(
                    label="Download Anomaly Report",
                    data=anomaly_report.to_csv(index=False),
                    file_name="anomaly_report.csv",
                    mime="text/csv",
                    key="download_anomaly_report"
                )

                st.caption(
                    "One row per record. The Isolation Forest and LOF scores "
                    "are on different scales and are not combined; higher "
                    "means more unusual. Flagged records are for review, "
                    "not automatic errors."
                )


def dataframe_to_json_records(dataframe):
    """Convert a DataFrame into JSON-safe records for PostgreSQL."""
    if dataframe is None or dataframe.empty:
        return []

    records = dataframe.astype(object).where(
        pd.notna(dataframe), None
    ).to_dict(orient="records")

    return [
        {
            str(key): (
                value.item() if hasattr(value, "item") else value
            )
            for key, value in record.items()
        }
        for record in records
    ]


def render_analysis_history():
    """Display saved analyses and their stored details."""

    st.header("📚 Saved Analysis History")

    try:
        history = get_analysis_history(limit=50)
    except Exception as error:
        st.error(
            "Could not load analysis history from PostgreSQL. "
            f"Error: {type(error).__name__}: {error}"
        )
        return

    if not history:
        st.info(
            "No saved analyses found. Upload a dataset and click "
            "'Save Analysis to Database' to create your first record."
        )
        return

    # Show recent analyses in a table.
    history_table = pd.DataFrame(history)

    if "created_at" in history_table.columns:
        history_table["created_at"] = pd.to_datetime(
            history_table["created_at"], errors="coerce"
        ).astype(str)

    st.subheader("Recent Analyses")
    if "overall_score" not in history_table.columns:
     history_table["overall_score"] = None
    st.dataframe(
        
        history_table[
            [
                "id",
                "created_at",
                "filename",
                "row_count",
                "column_count",
                "overall_score",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    # Let the user choose one saved analysis.
    labels = {
        int(item["id"]): (
            f'ID {item["id"]} — {item["filename"]} '
            f'({item["row_count"]} rows)'
        )
        for item in history
    }

    selected_id = st.selectbox(
        "Select an analysis to inspect",
        options=list(labels.keys()),
        format_func=lambda analysis_id: labels[analysis_id],
        key="saved_analysis_selected_id",
    )

    if st.button("View Selected Analysis", key="view_saved_analysis"):
        st.session_state["view_analysis_id"] = selected_id

    analysis_id = st.session_state.get("view_analysis_id")

    if analysis_id is None:
        st.caption("Select an analysis and click View Selected Analysis.")
        return

    # If the selected analysis is no longer in the current history, stop.
    if analysis_id not in labels:
        st.info("The selected analysis is no longer in the recent history.")
        return

    try:
        details = get_analysis_details(analysis_id)
    except Exception as error:
        st.error(
            "Could not load the selected analysis. "
            f"Error: {type(error).__name__}: {error}"
        )
        return

    if details is None:
        st.warning("The selected analysis was not found in the database.")
        return

    st.subheader(f"Analysis Details — ID {details['id']}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", details["row_count"])

    with col2:
        st.metric("Columns", details["column_count"])

    with col3:
        score = details.get("quality_score")
        overall_score = score.get("overall_score") if score else None
        st.metric(
            "Quality Score",
            "Not available"
            if overall_score is None
            else f"{overall_score}/100",
        )

    st.write("**Filename:**", details["filename"])
    st.write("**Created at:**", str(details["created_at"]))

    with st.expander("Dataset Metadata"):
        st.json(details.get("metadata") or {})

    st.subheader("Saved Quality Issues")
    issues = details.get("quality_issues", [])
    if issues:
        st.dataframe(pd.DataFrame(issues), use_container_width=True)
    else:
        st.info("No quality issues were saved for this analysis.")

    st.subheader("Saved Anomaly Results")
    anomalies = details.get("anomaly_results", [])
    if anomalies:
        st.dataframe(pd.DataFrame(anomalies), use_container_width=True)
    else:
        st.info("No anomaly results were saved for this analysis.")

    st.subheader("Saved Cleaning Actions")
    actions = details.get("cleaning_actions", [])
    if actions:
        st.dataframe(pd.DataFrame(actions), use_container_width=True)
    else:
        st.info("No cleaning actions have been saved for this analysis.")


def render_unified_results(data, quality_results,filename="uploaded_dataset"):
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

    render_report_downloads(score_result, anomaly_sources, include_ml)

    dataset_fingerprint = get_dataset_fingerprint(data)
    score_data = None

    if score_result is not None:
        score_data = {
            "overall_score": getattr(score_result, "overall_score", None),
            "dimensions": dataframe_to_json_records(
                getattr(score_result, "dimensions", None)
            ),
            "severity_summary": dataframe_to_json_records(
                getattr(score_result, "severity_summary", None)
            ),
        }

    # Keep the save payload available to the cleaning section later on the
    # page, so Apply Cleaning can ensure there is a saved analysis to attach to.
    quality_issues = []

    for _, finding in unified.iterrows():
        if finding.get("Category") != "Data Quality":
            continue

        quality_issues.append({
            "issue_type": str(finding.get("Issue Type", "Unknown")),
            "column_name": (
                None if pd.isna(finding.get("Column"))
                else str(finding.get("Column"))
            ),
            "issue_count": 1,
            "severity": (
                None if pd.isna(finding.get("Severity"))
                else str(finding.get("Severity"))
            ),
            "details": {
                "method": finding.get("Method"),
                "scope": finding.get("Scope"),
                "record_id": finding.get("Record ID"),
                "row_position": finding.get("Row Position"),
                "value": finding.get("Value"),
                "status": finding.get("Status"),
                "details": finding.get("Details"),
            },
        })

    anomaly_results = []
    combined = anomaly_sources.get("anomaly_analysis")

    if combined is not None and not combined.empty:
        for record_id, row in combined.iterrows():
            anomaly_results.append({
                "record_id": str(record_id),
                "isolation_forest_result": row.get("IF Result"),
                "lof_result": row.get("LOF Result"),
                "isolation_forest_score": None,
                "lof_score": None,
                "status": row.get("Status"),
                "details": {},
            })

    analysis_payload = {
        "filename": filename,
        "row_count": len(data),
        "column_count": len(data.columns),
        "metadata": {
            "columns": [str(column) for column in data.columns],
            "ml_detection_enabled": include_ml,
            "dataset_fingerprint": dataset_fingerprint,
        },
        "quality_score": score_data,
        "quality_issues": quality_issues,
        "anomaly_results": anomaly_results,
    }
    analysis_context = {
        "fingerprint": dataset_fingerprint,
        "analysis_payload": analysis_payload,
    }

    # Save the current dataset analysis to PostgreSQL
    st.subheader("Analysis History")

    if st.button("Save Analysis to Database", key="save_analysis_button"):
        try:
            analysis_id = get_or_create_analysis_id(
                dataset_fingerprint,
                analysis_payload,
                st.session_state,
            )

            st.success(
                f"Analysis saved! Analysis ID: {analysis_id}"
            )

        except Exception as error:
            st.error(f"Could not save analysis: {error}")

    if unified.empty:
        st.success("No findings detected by the supported checks.")
        return analysis_context

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

    return analysis_context


# Data Cleaning helpers
INVALID_ACTION_CHOICES = {
    "Leave unchanged": INVALID_KEEP,
    "Set invalid values to missing": INVALID_MISSING,
    "Clip numbers to the allowed range": INVALID_CLIP
}

MISSING_ACTION_CHOICES = {
    "Leave unchanged": MISSING_KEEP,
    "Fill (median for numbers, most frequent value for text)": MISSING_FILL,
    "Remove rows that contain a missing value": MISSING_DROP
}

# Keeps the change tables fast to draw on large datasets.
MAX_CLEANING_ROWS_SHOWN = 1000


def render_cleaning_operations(data, rules, analysis_context=None):
    """Choose, preview and apply cleaning on a separate copy of the dataset."""

    if not data.columns.is_unique:
        st.warning(
            "Cleaning needs unique column names, but this dataset has "
            "repeated column names."
        )
        return

    st.write(
        "Choose the cleaning operations to use. A preview of the changes is "
        "shown first. Cleaning is applied to a separate copy; your uploaded "
        "dataset is never changed."
    )

    active_rules = {
        column: limits
        for column, limits in rules.items()
        if column in data.columns
    }

    st.subheader("Cleaning Options")

    # Duplicate rows
    duplicate_count = int(get_duplicate_count(data))

    remove_duplicates = st.checkbox(
        "Remove exact duplicate rows",
        value=False,
        help=(
            f"{duplicate_count} duplicate row(s) found. "
            "The first occurrence of each row is kept."
        ),
        key="clean_remove_duplicates"
    )

    # Invalid values
    if active_rules:

        st.caption(
            "Invalid-value rules in use: "
            + ", ".join(
                f"{column} ({limits[0]} to {limits[1]})"
                for column, limits in active_rules.items()
            )
        )

        invalid_label = st.selectbox(
            "Invalid values",
            list(INVALID_ACTION_CHOICES),
            key="clean_invalid_action"
        )

        invalid_action = INVALID_ACTION_CHOICES[invalid_label]

    else:

        st.info(
            "Invalid-value correction is not available: none of the "
            f"columns with rules ({', '.join(rules)}) are in this dataset."
        )

        invalid_action = INVALID_KEEP

    # Missing values
    missing_label = st.selectbox(
        "Missing values",
        list(MISSING_ACTION_CHOICES),
        key="clean_missing_action",
        help="Blank and whitespace-only cells count as missing."
    )

    missing_action = MISSING_ACTION_CHOICES[missing_label]

    if (
        not remove_duplicates
        and invalid_action == INVALID_KEEP
        and missing_action == MISSING_KEEP
    ):
        st.info(
            "No cleaning operations selected. Choose at least one option "
            "to preview the changes. The dataset has not been changed."
        )
        return

    # Preview (computed on a copy; nothing is applied yet)
    try:
        result = clean_dataset(
            data,
            remove_duplicates=remove_duplicates,
            invalid_action=invalid_action,
            invalid_rules=active_rules,
            missing_action=missing_action
        )
    except (TypeError, ValueError) as error:
        st.error(f"The cleaning preview could not be created: {error}")
        return

    st.subheader("Preview of Changes")

    st.caption("Nothing has been applied yet.")

    for note in result.notes:
        st.warning(note)

    if not result.has_changes:
        st.success(
            "The selected operations found nothing to change in this dataset."
        )
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows Before", result.rows_before)

    with col2:
        st.metric(
            "Rows After",
            result.rows_after,
            delta=result.rows_after - result.rows_before,
            delta_color="off"
        )

    with col3:
        st.metric("Cells Changed", result.cells_changed)

    st.dataframe(
        result.summary,
        use_container_width=True,
        hide_index=True
    )

    if not result.removed_rows.empty:

        with st.expander(
            f"Rows that would be removed ({len(result.removed_rows)})"
        ):

            st.dataframe(
                result.removed_rows.head(MAX_CLEANING_ROWS_SHOWN),
                use_container_width=True
            )

    if not result.changes.empty:

        with st.expander(
            f"Cell changes ({len(result.changes)})"
        ):

            st.dataframe(
                result.changes.head(MAX_CLEANING_ROWS_SHOWN),
                use_container_width=True,
                hide_index=True
            )

    if (
        len(result.removed_rows) > MAX_CLEANING_ROWS_SHOWN
        or len(result.changes) > MAX_CLEANING_ROWS_SHOWN
    ):
        st.caption(
            f"Only the first {MAX_CLEANING_ROWS_SHOWN} rows of each change "
            "table are shown. All changes are applied."
        )

    # Apply. The applied result is kept in session state so it stays visible
    # when other widgets rerun the page, and is only shown while the dataset
    # and the selected options are the same as when it was applied.
    # Stored as text so the comparison below never depends on the type that
    # get_dataset_fingerprint returns.
    signature = repr((
        get_dataset_fingerprint(data),
        (remove_duplicates, invalid_action, missing_action),
        tuple(
            sorted(
                (column, tuple(limits))
                for column, limits in active_rules.items()
            )
        )
    ))

    if st.button("Apply Cleaning", key="clean_apply"):

        st.session_state["cleaning_applied"] = {
            "signature": signature,
            "result": result
        }

        if analysis_context is None:
            st.warning(
                "Cleaning was applied, but it could not be saved because "
                "there is no analysis record for this dataset."
            )
        else:
            try:
                analysis_id = persist_applied_cleaning(
                    analysis_context["fingerprint"],
                    analysis_context["analysis_payload"],
                    st.session_state,
                    result,
                    settings={
                        "remove_duplicates": remove_duplicates,
                        "invalid_action": invalid_action,
                        "missing_action": missing_action,
                    },
                )
                st.success(
                    f"Cleaning action saved to analysis {analysis_id}."
                )
            except Exception as error:
                st.warning(
                    "Cleaning was applied, but its action could not be "
                    f"saved to PostgreSQL ({type(error).__name__})."
                )

    applied = st.session_state.get("cleaning_applied")

    if applied is None:
        return

    if applied["signature"] != signature:
        st.info(
            "The dataset or the selected options changed since cleaning was "
            "last applied. Click Apply Cleaning to apply the current "
            "selection."
        )
        return

    cleaned = applied["result"].cleaned

    st.subheader("Cleaned Dataset")

    st.success(
        "Cleaning was applied to a separate copy. "
        "Your uploaded dataset was not changed."
    )

    if cleaned.empty:
        st.warning(
            "All rows were removed by the selected options, so there is no "
            "cleaned dataset to show or download."
        )
        return

    st.dataframe(
        cleaned,
        use_container_width=True
    )

    st.download_button(
        label="Download Cleaned Dataset",
        data=cleaned.to_csv(index=False),
        file_name="cleaned_dataset.csv",
        mime="text/csv",
        key="download_cleaned_dataset"
    )


def render_review_suggestions(data, iqr_outliers):
    """Show IQR outliers and ML anomalies as review suggestions only."""

    st.subheader("Anomaly Review Suggestions")

    st.write(
        "IQR outliers and ML anomalies are unusual records to review, not "
        "errors. They are never removed or changed by cleaning. Row numbers "
        "below refer to the uploaded dataset."
    )

    if isinstance(iqr_outliers, pd.DataFrame) and not iqr_outliers.empty:

        st.info(
            f"{len(iqr_outliers)} IQR outlier value(s) are suggested "
            "for review."
        )

        with st.expander("IQR outliers to review"):
            st.dataframe(
                iqr_outliers,
                use_container_width=True
            )

    else:

        st.success("No IQR outliers to review.")

    include_ml = st.checkbox(
        "Also show ML anomaly suggestions (Isolation Forest + LOF)",
        value=False,
        help=(
            "Uses the same cached results as the Unified Results section "
            "when they are available."
        ),
        key="clean_include_ml"
    )

    if not include_ml:
        return

    try:

        with st.spinner("Running anomaly detection..."):
            anomaly_sources, anomaly_notes = get_anomaly_sources(
                data,
                get_dataset_fingerprint(data)
            )

    except (TypeError, ValueError) as error:
        st.warning(f"ML anomaly suggestions could not be created: {error}")
        return

    for note in anomaly_notes:
        st.info(note)

    shown = False

    if isinstance(anomaly_sources, dict):

        for name, frame in anomaly_sources.items():

            if isinstance(frame, pd.DataFrame) and not frame.empty:

                shown = True

                with st.expander(f"{name}: {len(frame)} record(s) to review"):
                    st.dataframe(
                        frame,
                        use_container_width=True
                    )

    if not shown:
        st.success("No ML anomalies to review.")


def render_data_cleaning(data, rules, iqr_outliers, analysis_context=None):
    """Data Cleaning section: operations, preview, apply and review hints."""

    if data.empty:
        st.info("The uploaded dataset is empty, so there is nothing to clean.")
        return

    render_cleaning_operations(data, rules, analysis_context)

    render_review_suggestions(data, iqr_outliers)


# ML Evaluation helpers
def render_ml_evaluation():
    """Compare IQR, Z-score, Isolation Forest and LOF on labelled synthetic data.

    The uploaded dataset is not used. The result is kept in session state so
    it stays visible when other widgets rerun the page.
    """

    st.write(
        "Compare IQR, Z-score, Isolation Forest and LOF on a synthetic "
        "dataset with known anomalies. This does not use your uploaded "
        "dataset."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        n_samples = st.number_input(
            "Number of records",
            min_value=MIN_SAMPLES,
            max_value=5000,
            value=DEFAULT_N_SAMPLES,
            step=100,
            key="ml_eval_records"
        )

    with col2:
        anomaly_percent = st.slider(
            "Known anomalies (%)",
            min_value=1,
            max_value=20,
            value=int(DEFAULT_ANOMALY_FRACTION * 100),
            key="ml_eval_anomaly_percent"
        )

    with col3:
        random_state = st.number_input(
            "Random seed",
            min_value=0,
            value=DEFAULT_RANDOM_STATE,
            step=1,
            key="ml_eval_seed"
        )

    contamination_choice = st.radio(
        "Isolation Forest and LOF contamination",
        ["Automatic (model default)", "True anomaly rate"],
        horizontal=True,
        key="ml_eval_contamination",
        help=(
            "IQR and Z-score are fixed rules and cannot use this setting. "
            "The true anomaly rate gives Isolation Forest and LOF "
            "information the two rules do not have."
        )
    )

    if st.button("Run ML Evaluation", key="run_ml_evaluation"):

        try:

            with st.spinner("Evaluating the four methods..."):

                st.session_state["ml_evaluation_result"] = run_ml_evaluation(
                    n_samples=int(n_samples),
                    anomaly_fraction=anomaly_percent / 100,
                    random_state=int(random_state),
                    use_true_rate=(contamination_choice == "True anomaly rate")
                )

        except (TypeError, ValueError) as error:

            st.session_state.pop("ml_evaluation_result", None)
            st.error(f"ML evaluation failed: {error}")

    result = st.session_state.get("ml_evaluation_result")

    if result is None:
        st.info("Choose the settings and click Run ML Evaluation.")
        return

    settings = result.settings

    st.caption(
        f"{settings['records']} records, {settings['known_anomalies']} known "
        f"anomalies ({settings['anomaly_rate']:.1%}), seed "
        f"{settings['random_state']}, Isolation Forest / LOF contamination: "
        f"{settings['contamination']}."
    )

    for note in result.warnings:
        st.warning(note)

    if result.metrics.empty:
        st.error("No method could be evaluated.")
        return

    st.subheader("Method Comparison")

    st.dataframe(
        result.metrics,
        use_container_width=True,
        hide_index=True
    )

    best = result.metrics.loc[result.metrics["F1-score"].idxmax()]

    if best["F1-score"] > 0:
        st.success(
            f"Highest F1-score: {best['Method']} ({best['F1-score']:.4f})"
        )

    st.subheader("Precision, Recall and F1-score")

    st.bar_chart(
        result.metrics.set_index("Method")[["Precision", "Recall", "F1-score"]]
    )

    st.subheader("Execution Time")

    st.bar_chart(
        result.metrics.set_index("Method")["Execution Time (s)"]
    )

    st.caption(
        "Mean seconds of the detection step. Preprocessing shared by "
        "Isolation Forest and LOF took "
        f"{settings['preprocessing_time_s']} s and is not included."
    )

    st.subheader("Confusion Matrices")

    matrix_columns = st.columns(len(result.confusion_matrices))

    for column, (method, matrix) in zip(
        matrix_columns,
        result.confusion_matrices.items()
    ):

        with column:
            st.markdown(f"**{method}**")
            st.dataframe(matrix)

    if not result.type_recall.empty:

        st.subheader("Share of Each Anomaly Type Detected")

        st.dataframe(
            result.type_recall,
            use_container_width=True
        )

        st.caption(
            "Extreme: values far outside the normal range. Contextual: "
            "every value is normal but the combination is not."
        )

    with st.expander("Synthetic dataset"):

        st.write(result.data["anomaly_type"].value_counts())

        st.dataframe(
            result.data.head(20),
            use_container_width=True
        )

    st.caption(
        "Results describe this synthetic dataset and seed only; they do "
        "not guarantee the same ranking on other data."
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

        analysis_context = render_unified_results(
            df,
            {
                SRC_MISSING: missing_values,
                SRC_DUPLICATES: duplicate_rows,
                SRC_INVALID: invalid_values,
                SRC_DATA_TYPE: data_type_problems,
                SRC_FORMAT: format_problems,
                SRC_CONSISTENCY: inconsistencies,
                SRC_IQR: outliers
            },
            filename=uploaded_file.name
        )

        # PHASE 12 — DATA CLEANING
        st.header("🧹 Data Cleaning")

        render_data_cleaning(df, rules, outliers, analysis_context)

    # Error Handling
    except Exception as e:
        st.error(f"Error while processing dataset: {e}")
        st.exception(e)
    # PHASE 16 — DATABASE: ANALYSIS HISTORY
render_analysis_history()

# PHASE 18 — ML EVALUATION
st.header("🧪 ML Evaluation")

render_ml_evaluation()
