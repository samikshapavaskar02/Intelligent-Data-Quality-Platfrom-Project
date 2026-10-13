import ast
from pathlib import Path

import pandas as pd

from modules import analysis_persistence
from modules.cleaning import INVALID_CLIP, clean_dataset
from modules.unified_dashboard import get_dataset_fingerprint


def test_same_dataset_reuses_saved_analysis_id(monkeypatch):
    saved = []

    def fake_get_or_create_analysis_bundle(**payload):
        saved.append(payload)
        return 41

    monkeypatch.setattr(
        analysis_persistence.database,
        "get_or_create_analysis_bundle",
        fake_get_or_create_analysis_bundle,
    )
    state = {}
    payload = {
        "filename": "customers.csv",
        "row_count": 3,
        "column_count": 1,
        "metadata": {"dataset_fingerprint": "dataset-a"},
    }

    first_id = analysis_persistence.get_or_create_analysis_id(
        "dataset-a", payload, state
    )
    second_id = analysis_persistence.get_or_create_analysis_id(
        "dataset-a", payload, state
    )

    assert first_id == second_id == 41
    assert saved == [payload]


def test_identical_typed_dataset_with_same_filename_reuses_analysis(monkeypatch):
    saves = []
    monkeypatch.setattr(
        analysis_persistence.database,
        "get_or_create_analysis_bundle",
        lambda **payload: saves.append(payload) or 19,
    )
    first = pd.DataFrame({
        "payload": pd.Series([[1], "1", {"value": 1}], dtype=object)
    })
    second = pd.DataFrame({
        "payload": pd.Series([[1], "1", {"value": 1}], dtype=object)
    })
    state = {}

    ids = []
    for dataset in (first, second):
        fingerprint = get_dataset_fingerprint(dataset)
        ids.append(analysis_persistence.get_or_create_analysis_id(
            fingerprint,
            {
                "filename": "nested.json",
                "metadata": {"dataset_fingerprint": fingerprint},
            },
            state,
        ))

    assert ids == [19, 19]
    assert len(saves) == 1


def test_different_filenames_with_identical_content_get_separate_ids(monkeypatch):
    saved = []

    def fake_get_or_create_analysis_bundle(**payload):
        saved.append(payload)
        return len(saved) + 40

    monkeypatch.setattr(
        analysis_persistence.database,
        "get_or_create_analysis_bundle",
        fake_get_or_create_analysis_bundle,
    )
    state = {}
    first_payload = {
        "filename": "customers.csv",
        "row_count": 3,
        "column_count": 1,
        "metadata": {"dataset_fingerprint": "same-content"},
    }
    second_payload = {
        **first_payload,
        "filename": "customers-copy.csv",
    }

    first_id = analysis_persistence.get_or_create_analysis_id(
        "same-content", first_payload, state
    )
    second_id = analysis_persistence.get_or_create_analysis_id(
        "same-content", second_payload, state
    )

    assert (first_id, second_id) == (41, 42)
    assert len(saved) == 2
    assert [payload["filename"] for payload in saved] == [
        "customers.csv", "customers-copy.csv"
    ]


def test_same_filename_with_changed_content_gets_separate_id(monkeypatch):
    saved = []

    monkeypatch.setattr(
        analysis_persistence.database,
        "get_or_create_analysis_bundle",
        lambda **payload: saved.append(payload) or len(saved) + 50,
    )
    state = {}

    first_id = analysis_persistence.get_or_create_analysis_id(
        "content-a",
        {
            "filename": "customers.csv",
            "metadata": {"dataset_fingerprint": "content-a"},
        },
        state,
    )
    second_id = analysis_persistence.get_or_create_analysis_id(
        "content-b",
        {
            "filename": "customers.csv",
            "metadata": {"dataset_fingerprint": "content-b"},
        },
        state,
    )

    assert (first_id, second_id) == (51, 52)
    assert len(saved) == 2


def test_new_session_reuses_matching_saved_analysis(monkeypatch):
    requested = []
    monkeypatch.setattr(
        analysis_persistence.database,
        "get_or_create_analysis_bundle",
        lambda **payload: requested.append(payload) or 88,
    )
    payload = {
        "filename": "customers.csv",
        "row_count": 3,
        "column_count": 1,
        "metadata": {"dataset_fingerprint": "dataset-a"},
    }

    analysis_id = analysis_persistence.get_or_create_analysis_id(
        "dataset-a", payload, {}
    )

    assert analysis_id == 88
    assert requested == [payload]


def test_database_lookup_requires_filename_and_fingerprint(monkeypatch):
    captured = {}

    class FakeCursor:
        def execute(self, query, params):
            captured["query"] = query
            captured["params"] = params

        def fetchone(self):
            return (88,)

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
        analysis_persistence.database,
        "get_connection",
        lambda: FakeConnection(),
    )

    result = analysis_persistence.database.get_analysis_id_by_identity(
        "customers.csv", "fingerprint-1"
    )

    assert result == 88
    assert "filename = %s" in captured["query"]
    assert "metadata->>'dataset_fingerprint' = %s" in captured["query"]
    assert captured["params"] == ("customers.csv", "fingerprint-1")


def test_saved_cleaning_action_details_still_load_from_analysis(monkeypatch):
    class FakeCursor:
        def __init__(self):
            self.query = ""

        def execute(self, query, params):
            self.query = query

        def fetchone(self):
            if "FROM analyses" in self.query:
                return (88, "created", "customers.csv", 3, 1, {})
            if "FROM quality_scores" in self.query:
                return (90.0, {}, {})
            return None

        def fetchall(self):
            if "FROM cleaning_actions" in self.query:
                return [("clean_dataset", 2, {"rows_before": 3}, "saved-at")]
            return []

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
        analysis_persistence.database,
        "get_connection",
        lambda: FakeConnection(),
    )

    details = analysis_persistence.database.get_analysis_details(88)

    assert details["id"] == 88
    assert details["cleaning_actions"] == [{
        "action_type": "clean_dataset",
        "affected_count": 2,
        "details": {"rows_before": 3},
        "created_at": "saved-at",
    }]


def test_applied_cleaning_persists_one_action_on_reused_analysis(monkeypatch):
    saved_analyses = []
    saved_actions = []

    def fake_get_or_create_analysis_bundle(**payload):
        saved_analyses.append(payload)
        return 73

    def fake_save_cleaning_action(**payload):
        saved_actions.append(payload)

    monkeypatch.setattr(
        analysis_persistence.database,
        "get_or_create_analysis_bundle",
        fake_get_or_create_analysis_bundle,
    )
    monkeypatch.setattr(
        analysis_persistence.database,
        "save_cleaning_action",
        fake_save_cleaning_action,
    )

    data = pd.DataFrame({"Age": [150, 150, 30]})
    result = clean_dataset(
        data,
        remove_duplicates=True,
        invalid_action=INVALID_CLIP,
        invalid_rules={"Age": (0, 120)},
    )
    state = {}
    analysis_payload = {
        "filename": "customers.csv",
        "row_count": len(data),
        "column_count": len(data.columns),
        "metadata": {"dataset_fingerprint": "dataset-a"},
    }
    settings = {
        "remove_duplicates": True,
        "invalid_action": INVALID_CLIP,
        "missing_action": "keep",
    }

    first_id = analysis_persistence.persist_applied_cleaning(
        "dataset-a", analysis_payload, state, result, settings
    )
    second_id = analysis_persistence.persist_applied_cleaning(
        "dataset-a", analysis_payload, state, result, settings
    )

    assert first_id == second_id == 73
    assert len(saved_analyses) == 1
    assert len(saved_actions) == 2
    for action in saved_actions:
        assert action["analysis_id"] == 73
        assert action["action_type"] == "clean_dataset"
        assert action["affected_count"] == 2
        assert action["details"]["rows_before"] == 3
        assert action["details"]["rows_after"] == 2
        assert action["details"]["cells_changed"] == 1
        assert action["details"]["changed_rows"] == [1, 2]
        assert action["details"]["changed_columns"] == ["Age"]
        assert "Before" not in str(action["details"])
        assert "After" not in str(action["details"])


def _function_node(tree, name):
    return next(
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


def test_app_persists_actions_only_inside_apply_button_handler():
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    tree = ast.parse(app_path.read_text(encoding="utf-8"))
    function = _function_node(tree, "render_cleaning_operations")

    apply_handler = None
    for node in ast.walk(function):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (
            isinstance(test, ast.Call)
            and isinstance(test.func, ast.Attribute)
            and isinstance(test.func.value, ast.Name)
            and test.func.value.id == "st"
            and test.func.attr == "button"
            and test.args
            and isinstance(test.args[0], ast.Constant)
            and test.args[0].value == "Apply Cleaning"
        ):
            continue
        apply_handler = node
        break

    assert apply_handler is not None
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "persist_applied_cleaning"
        for node in ast.walk(apply_handler)
    )

    preview_calls = [
        node for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "clean_dataset"
    ]
    assert preview_calls
    assert all(
        not any(node is call for node in ast.walk(apply_handler))
        for call in preview_calls
    )


def test_manual_save_and_apply_cleaning_share_analysis_identity_helper():
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    app_tree = ast.parse(app_path.read_text(encoding="utf-8"))
    results = _function_node(app_tree, "render_unified_results")
    cleaning = _function_node(app_tree, "render_cleaning_operations")
    persistence = _function_node(
        ast.parse(Path(analysis_persistence.__file__).read_text(encoding="utf-8")),
        "persist_applied_cleaning",
    )

    def calls_named(node, name):
        return any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == name
            for child in ast.walk(node)
        )

    assert calls_named(results, "get_or_create_analysis_id")
    assert calls_named(cleaning, "persist_applied_cleaning")
    assert calls_named(persistence, "get_or_create_analysis_id")


def test_app_keeps_applied_result_when_database_persistence_raises():
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    tree = ast.parse(app_path.read_text(encoding="utf-8"))
    function = _function_node(tree, "render_cleaning_operations")
    apply_handler = next(
        node for node in ast.walk(function)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Call)
        and isinstance(node.test.func, ast.Attribute)
        and node.test.func.attr == "button"
        and node.test.args
        and isinstance(node.test.args[0], ast.Constant)
        and node.test.args[0].value == "Apply Cleaning"
    )

    stored_result_line = next(
        node.lineno for node in ast.walk(apply_handler)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Dict)
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "cleaning_applied"
            for target in node.targets
        )
    )
    persistence_try = next(
        node for node in ast.walk(apply_handler)
        if isinstance(node, ast.Try)
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "persist_applied_cleaning"
            for child in ast.walk(node)
        )
    )

    assert stored_result_line < persistence_try.lineno
    assert any(
        isinstance(handler.type, ast.Name)
        and handler.type.id == "Exception"
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "warning"
            for child in ast.walk(handler)
        )
        for node in ast.walk(apply_handler) if isinstance(node, ast.Try)
        for handler in node.handlers
    )
    assert any(
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "applied"
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == "result"
        for node in ast.walk(function)
    )


def test_analysis_history_renders_saved_cleaning_actions():
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    tree = ast.parse(app_path.read_text(encoding="utf-8"))
    function = _function_node(tree, "render_analysis_history")

    assert any(
        isinstance(node, ast.Constant) and node.value == "cleaning_actions"
        for node in ast.walk(function)
    )
