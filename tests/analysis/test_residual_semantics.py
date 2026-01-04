from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from enochian_lm.analysis.analysis.residual_semantics import (  # noqa: E402
    SubtractiveSemanticsEngine,
)
from enochian_lm.analysis.utils.sql import ensure_analysis_tables  # noqa: E402
from enochian_lm.common.sqlite_bootstrap import sqlite3  # noqa: E402


def _build_test_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE runs (run_id TEXT PRIMARY KEY)")
    conn.execute(
        "CREATE TABLE clusters (cluster_id INTEGER PRIMARY KEY, run_id TEXT, ngram TEXT, glossator_def TEXT, verdict TEXT)"
    )
    conn.execute(
        "CREATE TABLE raw_defs (def_id INTEGER PRIMARY KEY, cluster_id INTEGER, source_word TEXT, definition TEXT, variant TEXT)"
    )
    conn.execute(
        "CREATE TABLE residual_details (residual_id INTEGER PRIMARY KEY, cluster_id INTEGER, normalized TEXT, definition TEXT, coverage_ratio REAL, residual_ratio REAL, uncovered_json TEXT)"
    )
    ensure_analysis_tables(conn)
    return conn


def test_subtractive_semantics_inserts_results():
    conn = _build_test_db()
    conn.execute("INSERT INTO runs(run_id) VALUES ('run1')")
    conn.execute(
        "INSERT INTO clusters(cluster_id, run_id, ngram, glossator_def, verdict) VALUES (1, 'run1', 'NAZ', ?, 'accepted')",
        ('{"DEFINITION":"pillar"}',),
    )
    conn.execute(
        "INSERT INTO raw_defs(cluster_id, source_word, definition, variant) VALUES (1, 'NAZPSAD', 'pillar of protection', 'NAZPSAD')"
    )
    conn.execute(
        "INSERT INTO residual_details(cluster_id, normalized, definition, coverage_ratio, residual_ratio, uncovered_json) VALUES (1, 'NAZPSAD', 'pillar of protection', 0.4, 0.6, ?)",
        (json.dumps([{"text": "psad"}]),),
    )
    # Add a residual that should be ignored because it is already a dictionary entry
    conn.execute(
        "INSERT INTO raw_defs(cluster_id, source_word, definition, variant) VALUES (1, 'OMIT', 'known entry', 'OMIT')"
    )
    conn.execute(
        "INSERT INTO residual_details(cluster_id, normalized, definition, coverage_ratio, residual_ratio, uncovered_json) VALUES (1, 'NAZOMIT', 'known entry combo', 0.2, 0.8, ?)",
        (json.dumps([{"text": "OMIT"}]),),
    )

    payload = {
        "evaluation": "accepted",
        "definition": "A protective shard",
        "semantic_core": ["ward", "shield"],
        "example_usage": "The PSAD guarded the gate.",
        "confidence": 0.6,
        "reason": "Leftover after removing NAZ",
    }

    def fake_llm(prompt: str, run_id: str):
        return {"response_text": json.dumps(payload)}

    engine = SubtractiveSemanticsEngine(
        conn, use_remote=False, llm_responder=fake_llm
    )
    stats = engine.process_run("run1")

    assert stats == {"processed": 1, "accepted": 1, "rejected": 0}

    row = conn.execute(
        "SELECT evaluation, definition, semantic_core FROM root_residual_semantics WHERE residual = 'psad'"
    ).fetchone()
    assert row["evaluation"] == "accepted"
    assert "protective" in row["definition"].lower()
    assert "ward" in row["semantic_core"]

