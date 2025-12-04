"""SQLite helpers for Enochian language modeling analysis."""
from __future__ import annotations

import logging
from enochian_lm.common.sqlite_bootstrap import sqlite3
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

ANALYSIS_TABLE_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS attribution_marginals (
      id INTEGER PRIMARY KEY,
      morph_a TEXT NOT NULL,
      morph_b TEXT NOT NULL,
      delta_a_given_b REAL NOT NULL,
      delta_b_given_a REAL NOT NULL,
      n_tokens INTEGER NOT NULL,
      updated_at TEXT NOT NULL
    );
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS uniq_attr_pair
    ON attribution_marginals(morph_a, morph_b);
    """,
    """
    CREATE TABLE IF NOT EXISTS collocation_stats (
      id INTEGER PRIMARY KEY,
      morph_left TEXT NOT NULL,
      morph_right TEXT NOT NULL,
      count_ab INTEGER NOT NULL,
      count_a INTEGER NOT NULL,
      count_b INTEGER NOT NULL,
      pmi REAL,
      llr REAL,
      asym_dep REAL,
      updated_at TEXT NOT NULL
    );
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS uniq_colloc_pair
    ON collocation_stats(morph_left, morph_right);
    """,
    """
    CREATE TABLE IF NOT EXISTS residual_clusters (
      id INTEGER PRIMARY KEY,
      cluster_id INTEGER NOT NULL,
      centroid_json TEXT NOT NULL,
      size INTEGER NOT NULL,
      updated_at TEXT NOT NULL
    );
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS uniq_residual_cluster_id
    ON residual_clusters(cluster_id);
    """,
    """
    CREATE TABLE IF NOT EXISTS residual_cluster_membership (
      id INTEGER PRIMARY KEY,
      residual_span TEXT NOT NULL,
      cluster_id INTEGER NOT NULL,
      sim_to_centroid REAL NOT NULL,
      updated_at TEXT NOT NULL
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_residual_member_cluster
    ON residual_cluster_membership(cluster_id);
    """,
    """
    CREATE TABLE IF NOT EXISTS morph_semantic_vectors (
      id INTEGER PRIMARY KEY,
      morph TEXT NOT NULL UNIQUE,
      vector_json TEXT NOT NULL,
      l2_norm REAL NOT NULL,
      updated_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS composite_reconstruction (
      id INTEGER PRIMARY KEY,
      token TEXT NOT NULL,
      gold_gloss TEXT,
      pred_vector_json TEXT NOT NULL,
      recon_error REAL NOT NULL,
      used_morphs_json TEXT NOT NULL,
      vector_source TEXT,
      updated_at TEXT NOT NULL
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_comp_recon_token
    ON composite_reconstruction(token);
    """,
    """
    CREATE TABLE IF NOT EXISTS token_morph_decomp (
      run_id      TEXT NOT NULL,
      token       TEXT NOT NULL,
      seg_index   INTEGER NOT NULL,
      morph       TEXT NOT NULL,
      span_start  INTEGER NOT NULL,
      span_end    INTEGER NOT NULL,
      score       REAL,
      source      TEXT,
      PRIMARY KEY (run_id, token, seg_index)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS root_remainders (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      run_id TEXT NOT NULL,
      root TEXT NOT NULL,
      word TEXT NOT NULL,
      normalized TEXT NOT NULL,
      remainder TEXT NOT NULL,
      kind TEXT NOT NULL,
      span_start INTEGER NOT NULL,
      span_end INTEGER NOT NULL,
      created_at TEXT NOT NULL
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_root_remainders_root
    ON root_remainders(root);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_root_remainders_root_remainder
    ON root_remainders(root, remainder);
    """,
    """
    CREATE TABLE IF NOT EXISTS root_residual_semantics (
      run_id         TEXT NOT NULL,
      root           TEXT NOT NULL,
      parent_word    TEXT NOT NULL,
      residual       TEXT NOT NULL,
      evaluation     TEXT NOT NULL,
      definition     TEXT,
      semantic_core  TEXT,
      example_usage  TEXT,
      confidence     REAL,
      reason         TEXT,
      raw_json       TEXT,
      created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
      PRIMARY KEY (run_id, root, parent_word, residual)
    );
    """,
)


def connect_sqlite(path: str) -> sqlite3.Connection:
    """Connect to a SQLite database ensuring directories and pragmas."""

    db_path = Path(path)
    if db_path.parent and not db_path.parent.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Opening SQLite database", extra={"path": str(db_path)})
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


def ensure_analysis_tables(conn: sqlite3.Connection) -> None:
    """Ensure the analysis tables exist in *conn*."""

    logger.info("Ensuring analysis tables")
    for statement in ANALYSIS_TABLE_STATEMENTS:
        conn.execute(statement)

    _ensure_column(
        conn,
        "composite_reconstruction",
        "vector_source",
        "TEXT",
    )


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table});")}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl};")


def _prepare_named_parameters(rows: Sequence[dict[str, object]]) -> tuple[str, tuple[str, ...]]:
    columns = tuple(rows[0].keys())
    placeholders = ", ".join(f":{col}" for col in columns)
    column_list = ", ".join(columns)
    return f"({column_list}) VALUES ({placeholders})", columns


def upsert_rows(conn: sqlite3.Connection, table: str, rows: list[dict[str, object]]) -> None:
    """Insert or upsert *rows* into *table*."""

    if not rows:
        return

    values_sql, columns = _prepare_named_parameters(rows)
    base_sql = f"INSERT INTO {table} {values_sql}"

    if table == "morph_semantic_vectors":
        update_assignments = ", ".join(
            f"{col} = excluded.{col}" for col in columns if col not in {"morph", "id"}
        )
        sql = f"{base_sql} ON CONFLICT(morph) DO UPDATE SET {update_assignments}"
    elif table == "attribution_marginals":
        update_assignments = ", ".join(
            f"{col} = excluded.{col}"
            for col in columns
            if col not in {"morph_a", "morph_b", "id"}
        )
        sql = (
            f"{base_sql} ON CONFLICT(morph_a, morph_b) DO UPDATE SET {update_assignments}"
        )
    elif table == "collocation_stats":
        update_assignments = ", ".join(
            f"{col} = excluded.{col}"
            for col in columns
            if col not in {"morph_left", "morph_right", "id"}
        )
        sql = (
            f"{base_sql} ON CONFLICT(morph_left, morph_right) DO UPDATE SET {update_assignments}"
        )
    else:
        sql = base_sql

    conn.executemany(sql, rows)
    conn.commit()


def fetch_all(conn: sqlite3.Connection, query: str, params: tuple[object, ...] = ()) -> list[sqlite3.Row]:
    """Execute *query* with *params* and return all rows."""

    cursor = conn.execute(query, params)
    return cursor.fetchall()


__all__ = [
    "connect_sqlite",
    "ensure_analysis_tables",
    "upsert_rows",
    "fetch_all",
    "ANALYSIS_TABLE_STATEMENTS",
]
