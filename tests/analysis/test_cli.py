import argparse
import sys
import types
from pathlib import Path

import pytest

sys.modules.setdefault("gensim", types.SimpleNamespace(models=types.SimpleNamespace(FastText=object)))
dummy_gensim_utils = types.ModuleType("gensim.utils")
dummy_gensim_utils.simple_preprocess = lambda text, deacc=True, min_len=1: []
sys.modules["gensim.utils"] = dummy_gensim_utils
sys.modules.setdefault(
    "gensim.models", types.SimpleNamespace(FastText=object)
)
sys.modules.setdefault(
    "sentence_transformers", types.SimpleNamespace(SentenceTransformer=object)
)
dummy_preanalysis = types.ModuleType("enochian_lm.root_extraction.utils.preanalysis")
dummy_preanalysis.execute_preanalysis = lambda **_: {}
sys.modules[
    "enochian_lm.root_extraction.utils.preanalysis"
] = dummy_preanalysis
dummy_embeddings = types.ModuleType("enochian_lm.root_extraction.utils.embeddings")
dummy_embeddings.get_fasttext_model = lambda: None
sys.modules["enochian_lm.root_extraction.utils.embeddings"] = dummy_embeddings
dummy_refresh = types.ModuleType("enochian_lm.root_extraction.utils.residual_refresh")
dummy_refresh.refresh_residual_details = lambda *_, **__: (0, 0)
sys.modules[
    "enochian_lm.root_extraction.utils.residual_refresh"
] = dummy_refresh
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from enochian_lm.analysis import cli


def test_normalize_run_ids_rejects_empty_strings():
    with pytest.raises(ValueError):
        cli._normalize_run_ids(["   ", ""])


def test_run_preanalyze_handles_multiple_runs(monkeypatch, capsys):
    calls: list[str | None] = []

    def fake_execute_preanalysis(db_path, stage, trusted_path, run_id, refresh):
        calls.append(run_id)
        return {
            "stage": stage,
            "run_id": run_id,
            "preanalysis_id": f"pre-{run_id}",
            "created": True,
            "trusted_count": 0,
            "snapshots": [],
        }

    monkeypatch.setattr(cli, "execute_preanalysis", fake_execute_preanalysis)

    args = argparse.Namespace(
        db_path="/tmp/test.sqlite3",
        stage="initial",
        trusted=None,
        run_id=["runA", "runB"],
        refresh=False,
    )

    cli._run_preanalyze(args)

    assert calls == ["runA", "runB"]

    output = capsys.readouterr().out
    assert "runA" in output
    assert "runB" in output


def test_run_residual_refresh_aggregates(monkeypatch, capsys):
    calls: list[str | None] = []

    def fake_refresh(db_path, run_id=None):
        calls.append(run_id)
        return (1, 2)

    monkeypatch.setattr(cli, "refresh_residual_details", fake_refresh)

    args = argparse.Namespace(db_path="/tmp/test.sqlite3", run_id=["a", "b"])

    total_clusters, total_rows = cli._run_residual_refresh(args)

    assert calls == ["a", "b"]
    assert (total_clusters, total_rows) == (2, 4)

    output = capsys.readouterr().out
    assert "[a]" in output
    assert "[b]" in output


def test_build_parser_keeps_refresh_alias():
    parser = cli._build_parser()
    args = parser.parse_args(["refresh", "--run-id", "demo"])

    assert args.command == "refresh"
    assert args.run_id == ["demo"]
    assert args.handler == cli._run_residual_refresh


def test_run_composite_backfill_processes_each_run(monkeypatch, capsys):
    calls: list[str | None] = []

    class DummyConn:
        def close(self):
            pass

    dummy_conn = DummyConn()

    def fake_connect(_):
        return dummy_conn

    def fake_ensure(_):
        return None

    def fake_backfill(conn, run_id=None, **_):
        assert conn is dummy_conn
        calls.append(run_id)
        return (2, run_id or "latest")

    monkeypatch.setattr(cli, "connect_sqlite", fake_connect)
    monkeypatch.setattr(cli, "ensure_analysis_tables", fake_ensure)
    monkeypatch.setattr(cli, "_backfill_composite_reconstruction", fake_backfill)

    args = argparse.Namespace(db_path="/tmp/test.sqlite3", run_id=["x", "y"])

    cli._run_composite_backfill(args)

    assert calls == ["x", "y"]

    output = capsys.readouterr().out
    assert "run x" in output
    assert "run y" in output


def test_run_residual_semantic_pass(monkeypatch, capsys):
    class DummyConn:
        def close(self):
            pass

    dummy_conn = DummyConn()
    monkeypatch.setattr(cli, "connect_sqlite", lambda path: dummy_conn)
    monkeypatch.setattr(cli, "ensure_analysis_tables", lambda conn: None)
    monkeypatch.setattr(cli, "_resolve_run_ids", lambda conn, run_id: ["run-a", "run-b"])

    calls: list[tuple] = []

    class DummyEngine:
        def __init__(self, conn, use_remote=True, llm_responder=None):
            calls.append(("init", conn, use_remote))

        def process_run(self, run_id, limit=None):
            calls.append(("run", run_id, limit))
            return {"processed": 2, "accepted": 1, "rejected": 1}

    monkeypatch.setattr(cli, "SubtractiveSemanticsEngine", DummyEngine)

    args = argparse.Namespace(db_path="/tmp/demo.sqlite3", run_id=None, limit=5, local=True)
    total = cli._run_residual_semantic_pass(args)

    assert total == 4
    assert calls[0] == ("init", dummy_conn, False)  # use_remote = not local
    assert ("run", "run-a", 5) in calls
    assert ("run", "run-b", 5) in calls

    output = capsys.readouterr().out
    assert "accepted=1" in output
