
import hashlib
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg.types.json import Jsonb

# Load environment variables from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def get_connection():
    """Connect to the configured PostgreSQL database."""
    password = os.getenv("DB_PASSWORD")

    if not password:
        raise RuntimeError(
            "DB_PASSWORD is missing. Check the project-root .env file."
        )

    return psycopg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "intelligent_data_quality"),
        user=os.getenv("DB_USER", "postgres"),
        password=password,
        connect_timeout=5,
    )


def initialize_database():
    """Create missing tables without deleting existing data."""
    schema = """
    CREATE TABLE IF NOT EXISTS analyses (
        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        filename TEXT NOT NULL,
        row_count BIGINT NOT NULL CHECK (row_count >= 0),
        column_count BIGINT NOT NULL CHECK (column_count >= 0),
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb
    );

    CREATE TABLE IF NOT EXISTS quality_scores (
        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        analysis_id BIGINT NOT NULL UNIQUE
            REFERENCES analyses(id) ON DELETE CASCADE,
        overall_score NUMERIC(5, 1),
        dimensions JSONB NOT NULL DEFAULT '{}'::jsonb,
        severity_summary JSONB NOT NULL DEFAULT '{}'::jsonb
    );

    CREATE TABLE IF NOT EXISTS quality_issues (
        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        analysis_id BIGINT NOT NULL
            REFERENCES analyses(id) ON DELETE CASCADE,
        issue_type TEXT NOT NULL,
        column_name TEXT,
        issue_count BIGINT NOT NULL DEFAULT 0 CHECK (issue_count >= 0),
        severity TEXT,
        details JSONB NOT NULL DEFAULT '{}'::jsonb
    );

    CREATE TABLE IF NOT EXISTS anomaly_results (
        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        analysis_id BIGINT NOT NULL
            REFERENCES analyses(id) ON DELETE CASCADE,
        record_id TEXT NOT NULL,
        isolation_forest_result TEXT,
        lof_result TEXT,
        isolation_forest_score DOUBLE PRECISION,
        lof_score DOUBLE PRECISION,
        status TEXT,
        details JSONB NOT NULL DEFAULT '{}'::jsonb
    );

    CREATE TABLE IF NOT EXISTS cleaning_actions (
        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        analysis_id BIGINT NOT NULL
            REFERENCES analyses(id) ON DELETE CASCADE,
        action_type TEXT NOT NULL,
        affected_count BIGINT NOT NULL DEFAULT 0
            CHECK (affected_count >= 0),
        details JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_analyses_created_at
        ON analyses(created_at DESC);

    CREATE INDEX IF NOT EXISTS idx_quality_issues_analysis
        ON quality_issues(analysis_id);

    CREATE INDEX IF NOT EXISTS idx_anomaly_results_analysis
        ON anomaly_results(analysis_id);

    CREATE INDEX IF NOT EXISTS idx_cleaning_actions_analysis
        ON cleaning_actions(analysis_id);
    """

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(schema)


def _analysis_identity_lock_key(filename, fingerprint):
    """Derive a stable PostgreSQL bigint advisory-lock key for an identity."""
    filename_bytes = filename.encode("utf-8")
    fingerprint_bytes = fingerprint.encode("utf-8")
    identity_bytes = (
        len(filename_bytes).to_bytes(8, "big")
        + filename_bytes
        + len(fingerprint_bytes).to_bytes(8, "big")
        + fingerprint_bytes
    )
    return int.from_bytes(hashlib.sha256(identity_bytes).digest()[:8], "big", signed=True)


def _validate_analysis_identity(filename, fingerprint):
    if not isinstance(filename, str) or not filename:
        raise ValueError("An analysis filename is required for identity lookup.")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("An analysis fingerprint is required for identity lookup.")


def _find_analysis_id(cursor, filename, fingerprint):
    cursor.execute(
        """
        SELECT id
        FROM analyses
        WHERE filename = %s
          AND metadata->>'dataset_fingerprint' = %s
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """,
        (filename, fingerprint),
    )
    row = cursor.fetchone()
    return row[0] if row is not None else None


def _find_analysis_bundle_id(cursor, filename, fingerprint):
    """Find an exact identity that has the bundle's required score row."""
    cursor.execute(
        """
        SELECT a.id
        FROM analyses AS a
        WHERE a.filename = %s
          AND a.metadata->>'dataset_fingerprint' = %s
          AND EXISTS (
              SELECT 1
              FROM quality_scores AS qs
              WHERE qs.analysis_id = a.id
          )
        ORDER BY a.created_at DESC, a.id DESC
        LIMIT 1
        """,
        (filename, fingerprint),
    )
    row = cursor.fetchone()
    return row[0] if row is not None else None


def _get_or_create_analysis_with_identity_lock(
    filename, fingerprint, create_with_cursor, *, require_complete_bundle=False
):
    """Run identity lock, lookup, and insert in one READ COMMITTED transaction."""
    _validate_analysis_identity(filename, fingerprint)
    lock_key = _analysis_identity_lock_key(filename, fingerprint)

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
            cursor.execute("SELECT pg_advisory_xact_lock(%s)", (lock_key,))

            if require_complete_bundle:
                analysis_id = _find_analysis_bundle_id(
                    cursor, filename, fingerprint
                )
            else:
                analysis_id = _find_analysis_id(cursor, filename, fingerprint)
            if analysis_id is None:
                analysis_id = create_with_cursor(cursor)

    return analysis_id


def _insert_analysis(cursor, filename, row_count, column_count, metadata=None):
    cursor.execute(
        """
        INSERT INTO analyses
            (filename, row_count, column_count, metadata)
        VALUES (%s, %s, %s, %s)
        RETURNING id
        """,
        (
            filename,
            int(row_count),
            int(column_count),
            Jsonb(_make_json_safe(metadata or {})),
        ),
    )
    return cursor.fetchone()[0]


def create_analysis(filename, row_count, column_count, metadata=None):
    """Save an analysis; identity-tagged calls reuse under an advisory lock.

    Calls without a dataset_fingerprint retain the historical always-create
    behavior because they do not provide enough information for identity-safe
    reuse.
    """
    metadata = metadata or {}
    fingerprint = metadata.get("dataset_fingerprint")
    if fingerprint is not None:
        return _get_or_create_analysis_with_identity_lock(
            filename,
            fingerprint,
            lambda cursor: _insert_analysis(
                cursor, filename, row_count, column_count, metadata
            ),
        )

    with get_connection() as conn:
        with conn.cursor() as cursor:
            return _insert_analysis(
                cursor, filename, row_count, column_count, metadata
            )


def save_quality_score(
    analysis_id,
    overall_score=None,
    dimensions=None,
    severity_summary=None,
):
    """Save the quality score and its available dimensions."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO quality_scores
                    (analysis_id, overall_score, dimensions, severity_summary)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (analysis_id) DO UPDATE SET
                    overall_score = EXCLUDED.overall_score,
                    dimensions = EXCLUDED.dimensions,
                    severity_summary = EXCLUDED.severity_summary
                """,
                (
                    analysis_id,
                    overall_score,
                    Jsonb(dimensions or {}),
                    Jsonb(severity_summary or {}),
                ),
            )


def save_quality_issues(analysis_id, issues):
    """Save a list of quality issue dictionaries."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            for issue in issues:
                cursor.execute(
                    """
                    INSERT INTO quality_issues
                        (analysis_id, issue_type, column_name,
                         issue_count, severity, details)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        analysis_id,
                        issue["issue_type"],
                        issue.get("column_name"),
                        issue.get("issue_count", 0),
                        issue.get("severity"),
                        Jsonb(issue.get("details") or {}),
                    ),
                )


def save_anomaly_results(analysis_id, results):
    """Save anomaly result dictionaries for an analysis."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            for result in results:
                cursor.execute(
                    """
                    INSERT INTO anomaly_results
                        (analysis_id, record_id, isolation_forest_result,
                         lof_result, isolation_forest_score, lof_score,
                         status, details)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        analysis_id,
                        str(result["record_id"]),
                        result.get("isolation_forest_result"),
                        result.get("lof_result"),
                        result.get("isolation_forest_score"),
                        result.get("lof_score"),
                        result.get("status"),
                        Jsonb(result.get("details") or {}),
                    ),
                )


def save_cleaning_action(
    analysis_id,
    action_type,
    affected_count=0,
    details=None,
):
    """Record a cleaning action without storing the dataset itself."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO cleaning_actions
                    (analysis_id, action_type, affected_count, details)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    analysis_id,
                    action_type,
                    affected_count,
                    Jsonb(details or {}),
                ),
            )


def test_connection():
    """Return the connected database name and user."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT current_database(), current_user")
            return cursor.fetchone()


if __name__ == "__main__":
    try:
        database, user = test_connection()
        print("PostgreSQL connection successful!")
        print(f"Database: {database}")
        print(f"User: {user}")

        initialize_database()
        print("Database tables initialized successfully.")

    except Exception as exc:
        # Do not print connection details or credentials.
        print(f"Database setup failed: {type(exc).__name__}: {exc}")



def get_analysis_history(limit=50):
    """Return the most recent saved analyses with their quality scores."""
    limit = max(1, min(int(limit), 500))

    query = """
        SELECT
            a.id,
            a.created_at,
            a.filename,
            a.row_count,
            a.column_count,
            a.metadata,
            qs.overall_score
        FROM analyses AS a
        LEFT JOIN quality_scores AS qs
            ON qs.analysis_id = a.id
        ORDER BY created_at DESC, id DESC
        LIMIT %s
    """

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()

    columns = [
        "id",
        "created_at",
        "filename",
        "row_count",
        "column_count",
        "metadata",
        "overall_score",
    ]

    return [dict(zip(columns, row)) for row in rows]


def get_analysis_id_by_identity(filename, fingerprint):
    """Find a saved analysis only when its filename and fingerprint match."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id
                FROM analyses
                WHERE filename = %s
                  AND metadata->>'dataset_fingerprint' = %s
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (filename, fingerprint),
            )
            row = cursor.fetchone()

    return row[0] if row is not None else None


def get_analysis_details(analysis_id):
    """Return one analysis and its related database records."""
    analysis_id = int(analysis_id)

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, created_at, filename, row_count,
                       column_count, metadata
                FROM analyses
                WHERE id = %s
                """,
                (analysis_id,),
            )
            row = cursor.fetchone()

            if row is None:
                return None

            analysis = dict(
                zip(
                    [
                        "id",
                        "created_at",
                        "filename",
                        "row_count",
                        "column_count",
                        "metadata",
                    ],
                    row,
                )
            )

            cursor.execute(
                """
                SELECT overall_score, dimensions, severity_summary
                FROM quality_scores
                WHERE analysis_id = %s
                """,
                (analysis_id,),
            )
            score = cursor.fetchone()

            cursor.execute(
                """
                SELECT issue_type, column_name, issue_count,
                       severity, details
                FROM quality_issues
                WHERE analysis_id = %s
                ORDER BY id
                """,
                (analysis_id,),
            )
            issues = cursor.fetchall()

            cursor.execute(
                """
                SELECT record_id, isolation_forest_result,
                       lof_result, isolation_forest_score,
                       lof_score, status, details
                FROM anomaly_results
                WHERE analysis_id = %s
                ORDER BY id
                """,
                (analysis_id,),
            )
            anomalies = cursor.fetchall()

            cursor.execute(
                """
                SELECT action_type, affected_count, details, created_at
                FROM cleaning_actions
                WHERE analysis_id = %s
                ORDER BY id
                """,
                (analysis_id,),
            )
            cleaning_actions = cursor.fetchall()

    analysis["quality_score"] = (
        dict(
            zip(
                ["overall_score", "dimensions", "severity_summary"],
                score,
            )
        )
        if score
        else None
    )

    analysis["quality_issues"] = [
        dict(
            zip(
                [
                    "issue_type",
                    "column_name",
                    "issue_count",
                    "severity",
                    "details",
                ],
                item,
            )
        )
        for item in issues
    ]

    analysis["anomaly_results"] = [
        dict(
            zip(
                [
                    "record_id",
                    "isolation_forest_result",
                    "lof_result",
                    "isolation_forest_score",
                    "lof_score",
                    "status",
                    "details",
                ],
                item,
            )
        )
        for item in anomalies
    ]

    analysis["cleaning_actions"] = [
        dict(
            zip(
                ["action_type", "affected_count", "details", "created_at"],
                item,
            )
        )
        for item in cleaning_actions
    ]

    return analysis


import math
from datetime import date, datetime
from decimal import Decimal

import numpy as np
import pandas as pd


def _make_json_safe(value):
    """Convert Pandas/NumPy values into JSON-compatible values."""
    if isinstance(value, dict):
        return {
            str(key): _make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_make_json_safe(item) for item in value]

    if value is None or value is pd.NA or value is pd.NaT:
        return None

    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()

    if isinstance(value, np.generic):
        return _make_json_safe(value.item())

    if isinstance(value, Decimal):
        return float(value) if value.is_finite() else None

    if isinstance(value, float) and not math.isfinite(value):
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def _insert_analysis_bundle(
    cursor,
    filename,
    row_count,
    column_count,
    metadata=None,
    quality_score=None,
    quality_issues=None,
    anomaly_results=None,
):
    """Insert analysis bundle rows using the caller's transaction cursor."""
    quality_score = quality_score or {}
    quality_issues = quality_issues or []
    anomaly_results = anomaly_results or []

    overall_score = quality_score.get("overall_score")

    if overall_score is not None:
        overall_score = _make_json_safe(overall_score)

    analysis_id = _insert_analysis(
        cursor, filename, row_count, column_count, metadata
    )

    cursor.execute(
        """
        INSERT INTO quality_scores
            (analysis_id, overall_score, dimensions, severity_summary)
        VALUES (%s, %s, %s, %s)
        """,
        (
            analysis_id,
            overall_score,
            Jsonb(_make_json_safe(quality_score.get("dimensions") or {})),
            Jsonb(_make_json_safe(
                quality_score.get("severity_summary") or {}
            )),
        ),
    )

    for issue in quality_issues:
        cursor.execute(
            """
            INSERT INTO quality_issues
                (analysis_id, issue_type, column_name,
                 issue_count, severity, details)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                analysis_id,
                str(issue["issue_type"]),
                issue.get("column_name"),
                int(issue.get("issue_count", 0)),
                issue.get("severity"),
                Jsonb(_make_json_safe(issue.get("details") or {})),
            ),
        )

    for anomaly in anomaly_results:
        cursor.execute(
            """
            INSERT INTO anomaly_results
                (analysis_id, record_id, isolation_forest_result,
                 lof_result, isolation_forest_score, lof_score,
                 status, details)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                analysis_id,
                str(anomaly["record_id"]),
                anomaly.get("isolation_forest_result"),
                anomaly.get("lof_result"),
                _make_json_safe(anomaly.get("isolation_forest_score")),
                _make_json_safe(anomaly.get("lof_score")),
                anomaly.get("status"),
                Jsonb(_make_json_safe(anomaly.get("details") or {})),
            ),
        )

    return analysis_id


def get_or_create_analysis_bundle(
    filename,
    row_count,
    column_count,
    metadata=None,
    quality_score=None,
    quality_issues=None,
    anomaly_results=None,
):
    """Return the exact saved analysis or atomically insert its full bundle."""
    metadata = metadata or {}
    fingerprint = metadata.get("dataset_fingerprint")
    _validate_analysis_identity(filename, fingerprint)

    return _get_or_create_analysis_with_identity_lock(
        filename,
        fingerprint,
        lambda cursor: _insert_analysis_bundle(
            cursor,
            filename,
            row_count,
            column_count,
            metadata,
            quality_score,
            quality_issues,
            anomaly_results,
        ),
        require_complete_bundle=True,
    )


def save_analysis_bundle(
    filename,
    row_count,
    column_count,
    metadata=None,
    quality_score=None,
    quality_issues=None,
    anomaly_results=None,
):
    """Save an analysis bundle; fingerprinted calls use the identity lock.

    Fingerprint-tagged bundles are idempotent by filename and fingerprint.
    An analysis-only row without a quality score is not considered a bundle.
    Calls without that metadata retain the historical always-create behavior.
    """
    metadata = metadata or {}
    if metadata.get("dataset_fingerprint") is not None:
        return get_or_create_analysis_bundle(
            filename,
            row_count,
            column_count,
            metadata,
            quality_score,
            quality_issues,
            anomaly_results,
        )

    with get_connection() as conn:
        with conn.cursor() as cursor:
            return _insert_analysis_bundle(
                cursor,
                filename,
                row_count,
                column_count,
                metadata,
                quality_score,
                quality_issues,
                anomaly_results,
            )

