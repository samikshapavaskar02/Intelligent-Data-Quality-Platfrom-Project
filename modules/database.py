
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


def create_analysis(filename, row_count, column_count, metadata=None):
    """Save an analysis and return its generated ID."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO analyses
                    (filename, row_count, column_count, metadata)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (
                    filename,
                    row_count,
                    column_count,
                    Jsonb(metadata or {}),
                ),
            )
            return cursor.fetchone()[0]


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
    """Return the most recent saved analyses."""
    limit = max(1, min(int(limit), 500))

    query = """
        SELECT
            id,
            created_at,
            filename,
            row_count,
            column_count,
            metadata
        FROM analyses
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
    ]

    return [dict(zip(columns, row)) for row in rows]


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


def save_analysis_bundle(
    filename,
    row_count,
    column_count,
    metadata=None,
    quality_score=None,
    quality_issues=None,
    anomaly_results=None,
):
    """Save an analysis and its results in one transaction."""

    quality_score = quality_score or {}
    quality_issues = quality_issues or []
    anomaly_results = anomaly_results or []

    overall_score = quality_score.get("overall_score")

    if overall_score is not None:
        overall_score = _make_json_safe(overall_score)

    with get_connection() as conn:
        with conn.cursor() as cursor:
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

            analysis_id = cursor.fetchone()[0]

            cursor.execute(
                """
                INSERT INTO quality_scores
                    (analysis_id, overall_score, dimensions, severity_summary)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    analysis_id,
                    overall_score,
                    Jsonb(_make_json_safe(
                        quality_score.get("dimensions") or {}
                    )),
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
                        _make_json_safe(
                            anomaly.get("isolation_forest_score")
                        ),
                        _make_json_safe(anomaly.get("lof_score")),
                        anomaly.get("status"),
                        Jsonb(_make_json_safe(anomaly.get("details") or {})),
                    ),
                )

    return analysis_id

