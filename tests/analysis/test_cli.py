from __future__ import annotations

import argparse
import importlib
import json
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


def test_build_parser_accepts_root_groups_report_command() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(
        [
            "report",
            "root-groups",
            "--root",
            "IO",
            "--variant",
            "debate",
            "--detail",
            "compact",
            "--format",
            "json",
            "--pretty",
        ]
    )

    assert args.command == "report"
    assert args.report_command == "root-groups"
    assert args.root == "IO"
    assert args.variant == "debate"
    assert args.detail == "compact"
    assert args.format == "json"
    assert args.pretty is True
    assert args.handler == cli._run_report_root_groups


def test_build_parser_accepts_find_ngram_subcommand() -> None:
    """Verify `enlm find-ngram` exposes the lookup options.

    Why:
    the ngram lookup must be available under the existing analysis command
    surface as well as the top-level Poetry script.

    How:
    parse a representative subcommand invocation and inspect the resulting
    namespace.

    Responsibility:
    guard the public `enlm find-ngram` CLI contract.
    """

    parser = cli._build_parser()
    args = parser.parse_args(
        [
            "find-ngram",
            "A-B",
            "--canon-only",
            "--citations",
            "--include-alternates",
            "--format",
            "json",
        ]
    )

    assert args.command == "find-ngram"
    assert args.ngram == "A-B"
    assert args.canon_only is True
    assert args.citations is True
    assert args.include_alternates is True
    assert args.format == "json"
    assert args.handler == cli._run_find_ngram


def test_build_find_ngram_parser_accepts_top_level_options() -> None:
    """Verify the top-level `find-ngram` parser mirrors lookup options.

    Why:
    the Poetry script entry point has its own parser and can drift from the
    `enlm` subcommand if it is not tested directly.

    How:
    parse a standalone invocation with verbosity and output options.

    Responsibility:
    protect `poetry run find-ngram "NGRAM"` usability.
    """

    parser = cli._build_find_ngram_parser()
    args = parser.parse_args(["AB", "--verbose", "--output", "matches.txt"])

    assert args.ngram == "AB"
    assert args.verbose is True
    assert args.output == "matches.txt"


def test_find_ngram_skips_top_level_db_bootstrap() -> None:
    """Ensure dictionary lookup remains read-only with respect to the DB.

    Why:
    this command reads dictionary JSON and should not initialize or mutate the
    default analysis SQLite database.

    How:
    parse the subcommand and ask the bootstrap guard for its decision.

    Responsibility:
    prevent accidental database writes from a read-only lookup command.
    """

    parser = cli._build_parser()
    args = parser.parse_args(["find-ngram", "AB"])

    assert cli._should_bootstrap_command_db(args) is False


def test_find_ngram_fast_path_extracts_subcommand_args() -> None:
    """Verify `enlm find-ngram` can bypass the full parser.

    Why:
    the full parser imports translation command wiring, while dictionary lookup
    only needs the lightweight lookup parser.

    How:
    pass a representative `enlm` argv list with global options before the
    command and inspect the extracted standalone arguments.

    Responsibility:
    keep the `enlm find-ngram` path fast without losing global verbosity.
    """

    assert cli._extract_find_ngram_fast_args(
        ["--db", "ignored.sqlite3", "--verbose", "find-ngram", "AB"]
    ) == ["AB", "--verbose"]


def test_root_groups_report_skips_top_level_db_bootstrap() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(["report", "root-groups", "--root", "D"])

    assert cli._should_bootstrap_command_db(args) is False


def test_pipeline_report_still_bootstraps_top_level_db() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(["report", "pipeline"])

    assert cli._should_bootstrap_command_db(args) is True


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


def test_find_ngram_main_writes_output_file(tmp_path, capsys) -> None:
    """Exercise the top-level command's `--output` path.

    Why:
    users asked to save lookup output, and file output should not also echo the
    same payload to stdout.

    How:
    run `find_ngram_main` against a tiny temporary dictionary and inspect the
    written text file.

    Responsibility:
    verify CLI argument handling, dictionary loading, and output routing work
    together.
    """

    dictionary_path = tmp_path / "dictionary.json"
    output_path = tmp_path / "matches.txt"
    dictionary_path.write_text(
        json.dumps(
            [
                {
                    "word": "ABBA",
                    "normalized": "abba",
                    "canon_word": True,
                    "senses": [{"definition": "father"}],
                }
            ]
        ),
        encoding="utf-8",
    )

    exit_code = cli.find_ngram_main(
        ["bb", "--dictionary", str(dictionary_path), "--output", str(output_path)]
    )

    assert exit_code == 0
    assert "ABBA [canon]" in output_path.read_text(encoding="utf-8")
    assert capsys.readouterr().out == ""


def test_find_ngram_main_returns_one_when_no_matches(tmp_path, capsys) -> None:
    """Confirm no-match searches produce message text and status 1.

    Why:
    scripts need a non-zero status for empty searches while humans still need a
    concise explanation of what happened.

    How:
    run the standalone command against a dictionary with no matching word.

    Responsibility:
    lock the no-match behavior requested for the CLI.
    """

    dictionary_path = tmp_path / "dictionary.json"
    dictionary_path.write_text(
        json.dumps(
            [
                {
                    "word": "ABBA",
                    "normalized": "abba",
                    "canon_word": True,
                    "senses": [{"definition": "father"}],
                }
            ]
        ),
        encoding="utf-8",
    )

    exit_code = cli.find_ngram_main(["zz", "--dictionary", str(dictionary_path)])

    assert exit_code == 1
    assert 'Matches for "zz" (normalized: "zz"): 0' in capsys.readouterr().out
