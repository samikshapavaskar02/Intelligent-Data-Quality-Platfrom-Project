"""Anomaly pipeline: raw DataFrame -> dashboard-ready anomaly data.

Responsibility
--------------
Chain the existing, already tested modules in the order given by the project
PDF (Phases 6-10 and Step 27), without repeating any of their logic:

    df
     -> modules.feature_selection.select_features
     -> modules.preprocessing.preprocess_for_ml        (columns=selected)
     -> models.isolation_forest.run_isolation_forest   (prep.X)
     -> models.lof.run_lof                             (prep.X)
     -> modules.anomaly_analysis.combine_anomaly_results
     -> modules.anomaly_dashboard.get_anomaly_summary / build_anomaly_table

Row identity
------------
The ORIGINAL DataFrame index is carried through every stage. Row ``i`` of
the input is row ``i`` of every result, and ``Record ID`` in the dashboard
table is the original index label. The index must be unique (the combine
and dashboard modules require it); a duplicate index raises a clear error
instead of being reset silently.

What this module does NOT do
----------------------------
* It does not clean data, and it NEVER deletes or filters out anomalies.
  Every input record appears in ``analysis`` and ``table``.
* It contains no Streamlit / UI code and does not modify its input.
* It does not run data-quality checks.

Typical use
-----------
    from modules.anomaly_pipeline import run_anomaly_pipeline

    result = run_anomaly_pipeline(df)
    result.summary   # total_records, if_anomalies, lof_anomalies, common_anomalies
    result.table     # Record ID | IF Result | LOF Result | Status
"""

import warnings
from dataclasses import dataclass, field

import pandas as pd

from models.isolation_forest import (
    COL_SCORE as IF_SCORE_COL,
    DEFAULT_CONTAMINATION as DEFAULT_IF_CONTAMINATION,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_RANDOM_STATE,
    IsolationForestResult,
    run_isolation_forest,
)
from models.lof import (
    COL_SCORE as LOF_SCORE_COL,
    DEFAULT_CONTAMINATION as DEFAULT_LOF_CONTAMINATION,
    LOFResult,
    run_lof,
)
from modules.anomaly_analysis import NORMAL, STATUS_COL, combine_anomaly_results
from modules.anomaly_dashboard import build_anomaly_table, get_anomaly_summary
from modules.feature_selection import (
    DEFAULT_MAX_CATEGORIES,
    FeatureSelectionResult,
    select_features,
)
from modules.preprocessing import PreprocessingResult, preprocess_for_ml


@dataclass
class AnomalyPipelineResult:
    """Everything produced by ``run_anomaly_pipeline``.

    Attributes
    ----------
    feature_selection : FeatureSelectionResult
        Which columns were used and why the others were not.
    preprocessing : PreprocessingResult
        The ML matrix (``.X``), the original missing-value report, and the
        columns preprocessing skipped.
    isolation_forest : IsolationForestResult
        Per-record ``if_prediction``, ``if_status``, ``if_score``.
    lof : LOFResult
        Per-record ``lof_prediction``, ``lof_status``, ``lof_score``.
    analysis : pd.DataFrame
        Combined result (``IF Result``, ``LOF Result``, ``Status``), indexed
        by the ORIGINAL index, one row per input record.
    table : pd.DataFrame
        Dashboard table: ``Record ID | IF Result | LOF Result | Status``.
    summary : dict
        ``total_records``, ``if_anomalies``, ``lof_anomalies``,
        ``common_anomalies``.
    warning_messages : list[str]
        Notes the UI may show (for example scikit-learn warnings from LOF or
        columns preprocessing had to skip). Empty list = nothing to report.
    """

    feature_selection: FeatureSelectionResult
    preprocessing: PreprocessingResult
    isolation_forest: IsolationForestResult
    lof: LOFResult
    analysis: pd.DataFrame
    table: pd.DataFrame
    summary: dict
    warning_messages: list = field(default_factory=list)

    @property
    def n_records(self):
        return len(self.analysis)

    @property
    def flagged(self):
        """Records that are NOT plain 'Normal' (high-confidence or needs review).

        This is only a filtered VIEW for review. Nothing is removed.
        """
        return self.analysis[self.analysis[STATUS_COL] != NORMAL]

    @property
    def scores(self):
        """``if_score`` and ``lof_score`` side by side (higher = more unusual).

        Both come from different scales and must not be averaged without
        normalisation.
        """
        return pd.concat(
            [self.isolation_forest.results[[IF_SCORE_COL]],
             self.lof.results[[LOF_SCORE_COL]]],
            axis=1,
        )


def run_anomaly_pipeline(df, columns=None, exclude_columns=None, force_include=None,
                         scaler="robust", encoding="onehot",
                         max_categories=DEFAULT_MAX_CATEGORIES,
                         n_estimators=DEFAULT_N_ESTIMATORS,
                         if_contamination=DEFAULT_IF_CONTAMINATION,
                         random_state=DEFAULT_RANDOM_STATE,
                         n_neighbors=None,
                         lof_contamination=DEFAULT_LOF_CONTAMINATION):
    """Run the full anomaly pipeline on an uploaded DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset (for example the output of ``load_dataset``). Not
        modified. Its index must be unique and is preserved everywhere.
    columns, exclude_columns, force_include, max_categories :
        Passed to ``select_features`` (``max_categories`` is also passed to
        ``preprocess_for_ml``).
    scaler, encoding :
        Passed to ``preprocess_for_ml``.
    n_estimators, if_contamination, random_state :
        Passed to ``run_isolation_forest`` (``if_contamination`` is its
        ``contamination``).
    n_neighbors, lof_contamination :
        Passed to ``run_lof`` (``lof_contamination`` is its
        ``contamination``).

    Returns
    -------
    AnomalyPipelineResult

    Raises
    ------
    TypeError / ValueError
        For an invalid DataFrame, a duplicate index, no usable features, or
        invalid parameters (raised by the stage that checks them, with its
        own message).
    RuntimeError
        If an internal consistency check on the index fails (should not
        happen; it would mean a stage changed the row identity).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError("The DataFrame is empty (no rows or no columns).")
    if df.index.has_duplicates:
        raise ValueError(
            "The DataFrame index has duplicate labels, so records cannot be "
            "identified uniquely. Make the index unique before running the "
            "anomaly pipeline."
        )

    # 1. Feature selection (PDF Step 15)
    selection = select_features(
        df,
        columns=columns,
        exclude_columns=exclude_columns,
        force_include=force_include,
        max_categories=max_categories,
    )
    if not selection.selected_columns:
        reasons = "; ".join(
            f"{row.column}: {row.reason}" for row in selection.excluded_columns.itertuples()
        )
        raise ValueError(
            "No features left for anomaly detection after feature selection. "
            f"Excluded columns -> {reasons or 'none'}."
        )

    # 2. ML preprocessing (PDF Steps 11-14)
    prep = preprocess_for_ml(
        df,
        columns=selection.selected_columns,
        scaler=scaler,
        encoding=encoding,
        max_categories=max_categories,
    )
    if not prep.X.index.equals(df.index):
        raise RuntimeError("Preprocessing changed the row index; row identity was lost.")

    # 3. Isolation Forest and LOF (PDF Steps 16-17)
    if_out = run_isolation_forest(
        prep.X,
        n_estimators=n_estimators,
        contamination=if_contamination,
        random_state=random_state,
    )
    with warnings.catch_warnings():
        # run_lof stores scikit-learn's warnings in ``warning_messages`` and
        # also re-issues them as "LOF: ..." UserWarnings. We keep them in the
        # result for the UI, so the duplicate console warning is not needed.
        warnings.filterwarnings("ignore", message="LOF: ", category=UserWarning)
        lof_out = run_lof(
            prep.X,
            n_neighbors=n_neighbors,
            contamination=lof_contamination,
        )

    # 4. Anomaly analysis (PDF Step 18): if_status + lof_status -> IF/LOF Result + Status
    analysis = combine_anomaly_results(if_out.results, lof_out.results)

    # 5. Dashboard data preparation (PDF Step 27)
    summary = get_anomaly_summary(analysis)
    table = build_anomaly_table(analysis)

    if not (analysis.index.equals(df.index) and table.index.equals(df.index)):
        raise RuntimeError("The anomaly results do not line up with the original rows.")

    warning_messages = [f"LOF: {message}" for message in lof_out.warning_messages]
    if len(prep.dropped_columns) > 0:
        skipped = ", ".join(
            f"{row.column} ({row.reason})" for row in prep.dropped_columns.itertuples()
        )
        warning_messages.append(f"Preprocessing skipped selected columns: {skipped}")

    return AnomalyPipelineResult(
        feature_selection=selection,
        preprocessing=prep,
        isolation_forest=if_out,
        lof=lof_out,
        analysis=analysis,
        table=table,
        summary=summary,
        warning_messages=warning_messages,
    )
