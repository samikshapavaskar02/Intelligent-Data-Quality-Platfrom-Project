import json

import pytest

from modules import database


def test_missing_password_raises_error(monkeypatch):
    monkeypatch.delenv("DB_PASSWORD", raising=False)

    with pytest.raises(RuntimeError, match="DB_PASSWORD is missing"):
        database.get_connection()


def test_database_configuration_defaults(monkeypatch):
    monkeypatch.setenv("DB_PASSWORD", "test_password")
    monkeypatch.delenv("DB_HOST", raising=False)
    monkeypatch.delenv("DB_PORT", raising=False)
    monkeypatch.delenv("DB_NAME", raising=False)
    monkeypatch.delenv("DB_USER", raising=False)

    captured = {}

    def fake_connect(**kwargs):
        captured.update(kwargs)
        return "fake_connection"

    monkeypatch.setattr(database.psycopg, "connect", fake_connect)

    result = database.get_connection()

    assert result == "fake_connection"
    assert captured["host"] == "localhost"
    assert captured["port"] == 5432
    assert captured["dbname"] == "intelligent_data_quality"
    assert captured["user"] == "postgres"
    assert captured["password"] == "test_password"
    assert captured["connect_timeout"] == 5


def test_database_configuration_uses_environment_variables(monkeypatch):
    monkeypatch.setenv("DB_HOST", "db.example.com")
    monkeypatch.setenv("DB_PORT", "5433")
    monkeypatch.setenv("DB_NAME", "test_database")
    monkeypatch.setenv("DB_USER", "test_user")
    monkeypatch.setenv("DB_PASSWORD", "test_password")

    captured = {}

    def fake_connect(**kwargs):
        captured.update(kwargs)
        return "fake_connection"

    monkeypatch.setattr(database.psycopg, "connect", fake_connect)

    database.get_connection()

    assert captured["host"] == "db.example.com"
    assert captured["port"] == 5433
    assert captured["dbname"] == "test_database"
    assert captured["user"] == "test_user"


def test_initialize_database_executes_schema(monkeypatch):
    executed = []

    class FakeCursor:
        def execute(self, query, params=None):
            executed.append(query)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        database, "get_connection", lambda: FakeConnection()
    )

    database.initialize_database()

    schema = "\n".join(executed)

    for table in (
        "analyses",
        "quality_scores",
        "quality_issues",
        "anomaly_results",
        "cleaning_actions",
    ):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in schema

    assert "CREATE INDEX IF NOT EXISTS" in schema


def test_create_analysis_returns_generated_id(monkeypatch):
    class FakeCursor:
        def execute(self, query, params):
            self.query = query
            self.params = params

        def fetchone(self):
            return (42,)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        database, "get_connection", lambda: FakeConnection()
    )

    result = database.create_analysis(
        "sample.csv",
        100,
        5,
        metadata={"source": "test"},
    )

    assert result == 42


def test_connection_check_returns_database_and_user(monkeypatch):
    class FakeCursor:
        def execute(self, query):
            assert "current_database()" in query

        def fetchone(self):
            return ("intelligent_data_quality", "postgres")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        database, "get_connection", lambda: FakeConnection()
    )

    assert database.test_connection() == (
        "intelligent_data_quality",
        "postgres",
    )



def test_get_analysis_history_returns_records(monkeypatch):
    from modules.database import get_analysis_history

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, query, params):
            assert "ORDER BY created_at DESC" in query
            assert params == (10,)

        def fetchall(self):
            return [
                (1, "2026-10-12", "data.csv", 100, 5, {})
            ]

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def cursor(self):
            return FakeCursor()

    monkeypatch.setattr(
        "modules.database.get_connection",
        lambda: FakeConnection(),
    )

    history = get_analysis_history(limit=10)

    assert len(history) == 1
    assert history[0]["filename"] == "data.csv"
    assert history[0]["row_count"] == 100


def test_get_analysis_details_returns_none_for_missing_record(monkeypatch):
    from modules.database import get_analysis_details

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, query, params):
            assert params == (999,)

        def fetchone(self):
            return None

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def cursor(self):
            return FakeCursor()

    monkeypatch.setattr(
        "modules.database.get_connection",
        lambda: FakeConnection(),
    )

    assert get_analysis_details(999) is None


def test_get_analysis_details_rejects_invalid_id():
    import pytest
    from modules.database import get_analysis_details

    with pytest.raises(ValueError):
        get_analysis_details("invalid")


def test_save_analysis_bundle_saves_all_results_in_one_transaction(monkeypatch):
    from modules.database import save_analysis_bundle

    executed_queries = []

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, query, params=None):
            executed_queries.append((query, params))

        def fetchone(self):
            return (42,)

    class FakeConnection:
        def __init__(self):
            self.cursor_instance = FakeCursor()
            self.committed = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.committed = exc_type is None
            return False

        def cursor(self):
            return self.cursor_instance

    fake_connection = FakeConnection()

    monkeypatch.setattr(
        "modules.database.get_connection",
        lambda: fake_connection,
    )

    analysis_id = save_analysis_bundle(
        filename="sample.csv",
        row_count=100,
        column_count=5,
        metadata={"source": "test"},
        quality_score={
            "overall_score": 90.0,
            "dimensions": {"completeness": 95.0},
            "severity_summary": {"Low": 2},
        },
        quality_issues=[
            {
                "issue_type": "Missing Value",
                "column_name": "Age",
                "issue_count": 2,
                "severity": "Low",
                "details": {"source": "test"},
            }
        ],
        anomaly_results=[
            {
                "record_id": "row-1",
                "isolation_forest_result": "Normal",
                "lof_result": "Anomaly",
                "status": "Needs Review",
                "details": {"source": "test"},
            }
        ],
    )

    assert analysis_id == 42
    assert fake_connection.committed is True
    assert len(executed_queries) == 4

    assert "INSERT INTO analyses" in executed_queries[0][0]
    assert "INSERT INTO quality_scores" in executed_queries[1][0]
    assert "INSERT INTO quality_issues" in executed_queries[2][0]
    assert "INSERT INTO anomaly_results" in executed_queries[3][0]
