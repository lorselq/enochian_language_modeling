from __future__ import annotations

import argparse
import importlib
import sys
import types
from pathlib import Path

import pytest


def _load_analysis_cli_module():
    """Load the analysis CLI with private dependency shims for this file only.

    What: import the production CLI module directly from disk under a private
    test-only name.
    Why: these tests only exercise small parser/helper behaviors, so we stub the
    heavyweight root-extraction and translation imports without polluting the
    shared module state used by the rest of the suite.
    Big picture: keeps this CLI regression file isolated so translation and
    residual tests can still import the real modules later in collection.
    """

    module_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "enochian_lm"
        / "analysis"
        / "cli.py"
    )
    src_root = module_path.parents[3]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    stub_names = [
        "enochian_lm.root_extraction.utils.embeddings",
        "enochian_lm.root_extraction.utils.candidate_finder",
        "enochian_lm.root_extraction.utils.dictionary_loader",
        "enochian_lm.root_extraction.utils.preanalysis",
        "enochian_lm.root_extraction.utils.residual_refresh",
        "translation.cli",
    ]
    backups = {name: sys.modules.get(name) for name in stub_names}

    embeddings_stub = types.ModuleType("enochian_lm.root_extraction.utils.embeddings")
    embeddings_stub.get_fasttext_model = lambda *_, **__: None
    sys.modules["enochian_lm.root_extraction.utils.embeddings"] = embeddings_stub

    candidate_finder_stub = types.ModuleType(
        "enochian_lm.root_extraction.utils.candidate_finder"
    )

    class _DummyFinder:
        def __init__(self, *args, **kwargs) -> None:
            pass

    candidate_finder_stub.MorphemeCandidateFinder = _DummyFinder
    sys.modules[
        "enochian_lm.root_extraction.utils.candidate_finder"
    ] = candidate_finder_stub

    dictionary_loader_stub = types.ModuleType(
        "enochian_lm.root_extraction.utils.dictionary_loader"
    )
    dictionary_loader_stub.load_dictionary = lambda *_, **__: []
    sys.modules[
        "enochian_lm.root_extraction.utils.dictionary_loader"
    ] = dictionary_loader_stub

    preanalysis_stub = types.ModuleType(
        "enochian_lm.root_extraction.utils.preanalysis"
    )
    preanalysis_stub.execute_preanalysis = lambda **_: {}
    sys.modules["enochian_lm.root_extraction.utils.preanalysis"] = preanalysis_stub

    residual_refresh_stub = types.ModuleType(
        "enochian_lm.root_extraction.utils.residual_refresh"
    )
    residual_refresh_stub.refresh_residual_details = lambda *_, **__: (0, 0)
    sys.modules[
        "enochian_lm.root_extraction.utils.residual_refresh"
    ] = residual_refresh_stub

    translation_cli_stub = types.ModuleType("translation.cli")
    translation_cli_stub.configure_translate_phrase_parser = lambda *_args, **_kwargs: None
    translation_cli_stub.configure_translate_word_parser = lambda *_args, **_kwargs: None
    translation_cli_stub.translate_phrase_from_args = lambda *_args, **_kwargs: 0
    translation_cli_stub.translate_word_from_args = lambda *_args, **_kwargs: 0
    sys.modules["translation.cli"] = translation_cli_stub

    sys.modules.pop("enochian_lm.analysis.cli", None)
    try:
        return importlib.import_module("enochian_lm.analysis.cli")
    finally:
        for name, original in backups.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


cli = _load_analysis_cli_module()


def test_normalize_run_ids_rejects_empty_strings() -> None:
    with pytest.raises(ValueError):
        cli._normalize_run_ids(["   ", ""])


def test_run_preanalyze_handles_multiple_runs(monkeypatch, capsys) -> None:
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


def test_run_residual_refresh_aggregates(monkeypatch, capsys) -> None:
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


def test_build_parser_keeps_refresh_alias() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(["refresh", "--run-id", "demo"])

    assert args.command == "refresh"
    assert args.run_id == ["demo"]
    assert args.handler == cli._run_residual_refresh


def test_run_composite_backfill_processes_each_run(monkeypatch, capsys) -> None:
    calls: list[str | None] = []

    class DummyConn:
        def close(self) -> None:
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
