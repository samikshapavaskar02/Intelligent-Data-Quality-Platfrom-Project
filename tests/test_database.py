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


def test_locked_bundle_sets_read_committed_locks_then_exactly_looks_up(monkeypatch):
    executed = []
    connections = []

    class FakeCursor:
        query = ""

        def execute(self, query, params=None):
            self.query = query
            executed.append((query, params))

        def fetchone(self):
            if "FROM analyses" in self.query:
                return None
            if "INSERT INTO analyses" in self.query:
                return (64,)
            return None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FakeConnection:
        def __init__(self):
            self.cursor_instance = FakeCursor()
            self.exit_args = None
            connections.append(self)

        def cursor(self):
            return self.cursor_instance

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.exit_args = args
            return False

    monkeypatch.setattr(
        database, "get_connection", lambda: FakeConnection()
    )

    result = database.get_or_create_analysis_bundle(
        filename="nested.json",
        row_count=2,
        column_count=1,
        metadata={"dataset_fingerprint": "fp-1"},
        quality_score={"overall_score": 80},
        quality_issues=[{"issue_type": "Missing Value"}],
        anomaly_results=[{"record_id": "row-1"}],
    )

    queries = [query for query, _ in executed]
    lock_position = next(i for i, query in enumerate(queries)
                         if "pg_advisory_xact_lock" in query)
    lookup_position = next(i for i, query in enumerate(queries)
                            if "FROM analyses" in query)
    insert_position = next(i for i, query in enumerate(queries)
                           if "INSERT INTO analyses" in query)
    assert result == 64
    assert len(connections) == 1
    assert "SET TRANSACTION ISOLATION LEVEL READ COMMITTED" in queries[0]
    assert lock_position < lookup_position < insert_position
    assert "filename = %s" in queries[lookup_position]
    assert "metadata->>'dataset_fingerprint' = %s" in queries[lookup_position]
    assert executed[lookup_position][1] == ("nested.json", "fp-1")
    assert any("INSERT INTO quality_scores" in query for query in queries)
    assert any("INSERT INTO quality_issues" in query for query in queries)
    assert any("INSERT INTO anomaly_results" in query for query in queries)
    assert connections[0].exit_args[0] is None


def test_locked_bundle_reuses_existing_match_without_inserting(monkeypatch):
    executed = []
    connections = []

    class FakeCursor:
        query = ""

        def execute(self, query, params=None):
            self.query = query
            executed.append(query)

        def fetchone(self):
            if "FROM analyses" in self.query:
                return (902,)
            return None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

        def __enter__(self):
            connections.append(self)
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(database, "get_connection", FakeConnection)

    result = database.get_or_create_analysis_bundle(
        "nested.json", 2, 1,
        metadata={"dataset_fingerprint": "fp-1"},
    )

    assert result == 902
    assert len(connections) == 1
    assert any("pg_advisory_xact_lock" in query for query in executed)
    assert any("FROM analyses" in query for query in executed)
    assert not any("INSERT INTO" in query for query in executed)


def test_locked_bundle_rolls_back_and_releases_lock_after_insert_error(monkeypatch):
    class FakeCursor:
        query = ""

        def execute(self, query, params=None):
            self.query = query
            if "INSERT INTO analyses" in query:
                raise RuntimeError("simulated insert failure")

        def fetchone(self):
            if "FROM analyses" in self.query:
                return None
            return (64,)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FakeConnection:
        exit_args = None

        def cursor(self):
            return FakeCursor()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.exit_args = args
            return False

    connection = FakeConnection()
    monkeypatch.setattr(database, "get_connection", lambda: connection)

    with pytest.raises(RuntimeError, match="simulated insert failure"):
        database.get_or_create_analysis_bundle(
            "nested.json", 2, 1,
            metadata={"dataset_fingerprint": "fp-1"},
        )

    assert connection.exit_args[0] is RuntimeError


def test_identity_lock_key_is_stable_signed_and_unambiguously_encoded():
    first = database._analysis_identity_lock_key("ab", "c")
    same_identity = database._analysis_identity_lock_key("ab", "c")
    ambiguous_pair = database._analysis_identity_lock_key("a", "bc")

    assert first == same_identity
    assert -(2**63) <= first < 2**63
    assert first != ambiguous_pair


def test_create_analysis_with_fingerprint_uses_locked_path(monkeypatch):
    calls = []

    def fake_locked(filename, fingerprint, create_with_cursor):
        calls.append((filename, fingerprint))
        return 321

    monkeypatch.setattr(
        database, "_get_or_create_analysis_with_identity_lock", fake_locked
    )

    result = database.create_analysis(
        "nested.json", 2, 1,
        metadata={"dataset_fingerprint": "fp-1"},
    )

    assert result == 321
    assert calls == [("nested.json", "fp-1")]


def test_save_analysis_bundle_with_fingerprint_uses_locked_path(monkeypatch):
    calls = []

    def fake_get_or_create(*args):
        calls.append(args)
        return 654

    monkeypatch.setattr(database, "get_or_create_analysis_bundle", fake_get_or_create)

    result = database.save_analysis_bundle(
        "nested.json", 2, 1,
        metadata={"dataset_fingerprint": "fp-1"},
    )

    assert result == 654
    assert calls == [(
        "nested.json", 2, 1, {"dataset_fingerprint": "fp-1"}, None, None, None
    )]


def test_create_analysis_then_bundle_save_creates_complete_bundle_once(monkeypatch):
    state = {
        "analyses": [],
        "quality_scores": [],
        "quality_issues": [],
        "anomaly_results": [],
    }

    class FakeCursor:
        query = ""

        def execute(self, query, params=None):
            self.query = query
            if "INSERT INTO analyses" in query:
                state["analyses"].append({
                    "id": len(state["analyses"]) + 1,
                    "filename": params[0],
                    "metadata": params[3].obj,
                    "created_at": len(state["analyses"]) + 1,
                })
            elif "INSERT INTO quality_scores" in query:
                state["quality_scores"].append(params[0])
            elif "INSERT INTO quality_issues" in query:
                state["quality_issues"].append(params[0])
            elif "INSERT INTO anomaly_results" in query:
                state["anomaly_results"].append(params[0])

        def fetchone(self):
            if "RETURNING id" in self.query:
                return (state["analyses"][-1]["id"],)

            if "FROM analyses" in self.query:
                filename, fingerprint = self._lookup_params
                matches = [
                    record for record in state["analyses"]
                    if record["filename"] == filename
                    and record["metadata"].get("dataset_fingerprint") == fingerprint
                    and (
                        "FROM quality_scores AS qs" not in self.query
                        or record["id"] in state["quality_scores"]
                    )
                ]
                return (matches[-1]["id"],) if matches else None
            return None

        @property
        def _lookup_params(self):
            return self.params

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FakeConnection:
        def cursor(self):
            cursor = FakeCursor()
            original_execute = cursor.execute

            def execute(query, params=None):
                cursor.params = params
                original_execute(query, params)

            cursor.execute = execute
            return cursor

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(database, "get_connection", FakeConnection)
    identity = {"dataset_fingerprint": "fp-sequence"}

    standalone_id = database.create_analysis(
        "nested.json", 2, 1, metadata=identity
    )
    bundle_id = database.save_analysis_bundle(
        "nested.json", 2, 1, metadata=identity,
        quality_score={"overall_score": 90},
        quality_issues=[{"issue_type": "Missing Value"}],
        anomaly_results=[{"record_id": "row-1"}],
    )
    repeated_bundle_id = database.save_analysis_bundle(
        "nested.json", 2, 1, metadata=identity,
        quality_score={"overall_score": 10},
        quality_issues=[{"issue_type": "Different Value"}],
        anomaly_results=[{"record_id": "row-2"}],
    )

    assert standalone_id == 1
    assert bundle_id == repeated_bundle_id == 2
    assert len(state["analyses"]) == 2
    assert state["quality_scores"] == [2]
    assert state["quality_issues"] == [2]
    assert state["anomaly_results"] == [2]
