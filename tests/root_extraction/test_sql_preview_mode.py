from __future__ import annotations

import pathlib
import sqlite3
import sys
import types

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if "enochian_lm.root_extraction.pipeline.run_root_extraction" not in sys.modules:
    root_stub = types.ModuleType("enochian_lm.root_extraction.pipeline.run_root_extraction")

    class _DummyRootCrew:
        def __init__(self, *args, **kwargs):
            pass

    root_stub.RootExtractionCrew = _DummyRootCrew
    sys.modules["enochian_lm.root_extraction.pipeline.run_root_extraction"] = root_stub

if "enochian_lm.root_extraction.pipeline.run_residual_semantic_extraction" not in sys.modules:
    rem_stub = types.ModuleType(
        "enochian_lm.root_extraction.pipeline.run_residual_semantic_extraction"
    )

    class _DummyRemainderCrew:
        def __init__(self, *args, **kwargs):
            pass

    rem_stub.RemainderExtractionCrew = _DummyRemainderCrew
    sys.modules[
        "enochian_lm.root_extraction.pipeline.run_residual_semantic_extraction"
    ] = rem_stub

from enochian_lm.root_extraction.main import parse_args
from enochian_lm.root_extraction.utils.sql_preview import connect_preview_db

# Cleanup stubs so other tests can import the real pipeline modules.
sys.modules.pop("enochian_lm.root_extraction.pipeline.run_root_extraction", None)
sys.modules.pop("enochian_lm.root_extraction.pipeline.run_residual_semantic_extraction", None)


def test_parse_args_accepts_no_db_only_logs(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["enochian-analysis", "--remainders", "--no-db-only-logs"],
    )
    args = parse_args()
    assert args.remainders == ""
    assert args.no_db_only_logs is True


def test_connect_preview_db_captures_sql(tmp_path):
    db_path = tmp_path / "source.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE demo (id INTEGER PRIMARY KEY, value TEXT)")
    conn.commit()
    conn.close()

    preview_conn, recorder = connect_preview_db(db_path, label="pytest")
    preview_conn.execute("INSERT INTO demo(value) VALUES ('alpha')")
    preview_conn.execute("UPDATE demo SET value = 'beta' WHERE id = 1")
    preview_conn.commit()
    preview_conn.set_trace_callback(None)
    log_path = recorder.flush()
    preview_conn.close()

    log_text = pathlib.Path(log_path).read_text(encoding="utf-8")
    assert "INSERT INTO demo" in log_text
    assert "UPDATE demo" in log_text

    # Source DB remains unchanged when preview mode is used.
    source_conn = sqlite3.connect(db_path)
    count = source_conn.execute("SELECT COUNT(*) FROM demo").fetchone()[0]
    source_conn.close()
    assert count == 0
