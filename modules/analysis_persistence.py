"""Small persistence helpers for one uploaded analysis and its cleaning log."""

from modules import database


SAVED_ANALYSIS_IDS_KEY = "saved_analysis_ids_by_identity"


def get_or_create_analysis_id(fingerprint, analysis_payload, session_state):
    """Reuse an exact filename/fingerprint match, creating it only if needed."""
    filename = analysis_payload.get("filename")
    metadata = analysis_payload.get("metadata") or {}
    if not filename:
        raise ValueError("An analysis filename is required for identity lookup.")
    if metadata.get("dataset_fingerprint") != fingerprint:
        raise ValueError(
            "The analysis metadata fingerprint must match the supplied fingerprint."
        )

    identity = (filename, fingerprint)
    saved_ids = session_state.setdefault(SAVED_ANALYSIS_IDS_KEY, {})
    if identity in saved_ids:
        return saved_ids[identity]

    analysis_id = database.get_or_create_analysis_bundle(**analysis_payload)

    saved_ids[identity] = analysis_id
    return analysis_id


def persist_cleaning_action(analysis_id, result, settings):
    """Persist a summary of an applied cleaning result without cell values."""
    changed_rows = set()
    changed_columns = set()

    if not result.changes.empty:
        changed_rows.update(
            int(row) for row in result.changes["Original Row"].dropna()
        )
        changed_columns.update(
            str(column) for column in result.changes["Column"].dropna()
        )

    removed_rows = []
    if not result.removed_rows.empty:
        for row in result.removed_rows.to_dict(orient="records"):
            original_row = row.get("Original Row")
            if original_row is not None:
                changed_rows.add(int(original_row))
            removed_rows.append({
                "original_row": (
                    int(original_row) if original_row is not None else None
                ),
                "reason": (
                    str(row.get("Reason"))
                    if row.get("Reason") is not None else None
                ),
            })

    details = {
        "settings": dict(settings),
        "rows_before": int(result.rows_before),
        "rows_after": int(result.rows_after),
        "cells_changed": int(result.cells_changed),
        "changed_rows": sorted(changed_rows),
        "changed_columns": sorted(changed_columns),
        "removed_rows": removed_rows,
        "summary": result.summary.to_dict(orient="records"),
    }

    database.save_cleaning_action(
        analysis_id=analysis_id,
        action_type="clean_dataset",
        affected_count=len(changed_rows),
        details=details,
    )


def persist_applied_cleaning(
    fingerprint, analysis_payload, session_state, result, settings
):
    """Ensure an analysis exists, then attach one applied cleaning action."""
    analysis_id = get_or_create_analysis_id(
        fingerprint, analysis_payload, session_state
    )
    persist_cleaning_action(analysis_id, result, settings)
    return analysis_id
