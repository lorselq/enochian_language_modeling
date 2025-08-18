"""
init_insights_db.py

Initializes (or gently migrates) the insights database used by the ELM debate/solo pipelines.
This version implements the schema upgrades we discussed:

- Separate `runs` for batch metadata
- Richer `clusters` with adjudication fields & cohesion metrics
- Keep raw definitions in `raw_defs` (no judgments mixed in)
- Normalize `citations` (one row per citation) + `sources`
- Track synthesized definitions in `synth_defs`
- Record gating/skips in `skips`
- Optional `decisions` table for future per-target adjudications
- Useful indexes + FTS5 over definitions for fast search

The module is safe to re-run: it will create missing tables and indexes, and
attempt to add missing columns on existing tables.

SQLite notes:
- Adding constraints to existing columns is not supported during migration.
- We therefore add only missing columns (no retroactive CHECK constraints).
- New databases created from scratch will have full constraints.

Author: ELM project
"""

from __future__ import annotations
import os
import sqlite3
from typing import Dict, Iterable, Tuple

# -------------------------
# Paths
# -------------------------

def DB_PATH(file_name: str = "debate_derived_definitions.sqlite3") -> str:
    """
    Build an absolute path into your repo's data/ directory.

    Default file name picks the debate DB; pass a different name for alternates, e.g.:
        DB_PATH("solo_analysis_derived_definitions.sqlite3")
    """
    return os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),  # Go up from scripts directory
            "src",
            "enochian_translation_team",
            "data",
            file_name,
        )
    )

# -------------------------
# Helpers
# -------------------------

def _open(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn

def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ? LIMIT 1;",
        (table,),
    )
    return cur.fetchone() is not None

def _index_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ? LIMIT 1;",
        (name,),
    )
    return cur.fetchone() is not None

def _columns(conn: sqlite3.Connection, table: str) -> Dict[str, Tuple[int,str,int,int,str]]:
    """
    Return PRAGMA table_info columns mapping: {name: (cid, name, type, notnull, dflt_value, pk)}
    """
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return {r[1]: r for r in rows}

def _add_column_if_missing(conn: sqlite3.Connection, table: str, col: str, decl: str) -> None:
    cols = _columns(conn, table)
    if col not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl};")

def _create_index_if_missing(conn: sqlite3.Connection, name: str, ddl: str) -> None:
    if not _index_exists(conn, name):
        conn.execute(ddl)

def _fts5_available(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS __fts5_smoketest USING fts5(x);")
        conn.execute("DROP TABLE __fts5_smoketest;")
        return True
    except sqlite3.DatabaseError:
        return False

# -------------------------
# Schema DDL (fresh create)
# -------------------------

SCHEMA_CREATE = """
-- Ensure foreign keys
PRAGMA foreign_keys = ON;

-- 0) Runs: batch-level metadata
CREATE TABLE IF NOT EXISTS runs (
  run_id        TEXT PRIMARY KEY,
  -- free-form name or UUID for human traceability
  run_name      TEXT,
  engine        TEXT CHECK (engine IN ('debate','solo')) NOT NULL DEFAULT 'debate',
  embedder      TEXT,              -- e.g., all-MiniLM-L6-v2
  env_json      TEXT,              -- library versions, seeds, device info
  created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

-- 1) Clusters: one row per evaluated cluster of an n-gram
CREATE TABLE IF NOT EXISTS clusters (
  cluster_id            INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id                TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  ngram                 TEXT NOT NULL,
  cluster_index         INTEGER NOT NULL,       -- ordinal within ngram
  -- evidence size metrics (for gate logging & later filtering)
  sem_count             INTEGER,                -- semantic cluster size
  idx_count             INTEGER,                -- index cluster size
  overlap_count         INTEGER,                -- intersection size
  -- prompt/model echoes (optional but handy)
  glossator_prompt      TEXT,
  glossator_model       TEXT,
  adjudicator_prompt    TEXT,
  adjudicator_model     TEXT,
  -- glossator / adjudicator outcomes
  glossator_def         TEXT,                   -- short synthesized gloss
  adjudicator_output    TEXT,                   -- just output of the adjudicator
  adjudicator_verdict   TEXT,                   -- extracted verdict
  semantic_cohesion     REAL,                   -- adjudicator subscore
  derivational_validity REAL,                   -- adjudicator subscore
  rebuttal_resilience   REAL,                   -- adjudicator subscore
  -- your own pre-adjudication signals
  cohesion              REAL,                   -- mean cosine (upper triangle)
  semantic_coverage     REAL,                   -- fraction of members covered
  best_config           TEXT,                   -- JSON blob or string repr
  created_at            TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  UNIQUE (run_id, ngram, cluster_index)
);

-- 2) Raw definitions: never mix judgments here
CREATE TABLE IF NOT EXISTS raw_defs (
  def_id        INTEGER PRIMARY KEY AUTOINCREMENT,
  cluster_id    INTEGER NOT NULL REFERENCES clusters(cluster_id) ON DELETE CASCADE,
  source_word   TEXT    NOT NULL,               -- member word/variant
  variant       TEXT    NOT NULL,               -- variant (as relevant)
  definition    TEXT    NOT NULL,               -- human-compiled base gloss
  enhanced_def  TEXT,                           -- model-enhanced gloss
  fasttext      REAL,                           -- FastText score (optional)
  similarity    REAL,                           -- embedding sim score
  tier          TEXT,                           -- provenance tier/label
  created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  UNIQUE (cluster_id, source_word, definition)
);

-- 2b) Citations normalized (split from JSON)
CREATE TABLE IF NOT EXISTS citations (
  citation_id   INTEGER PRIMARY KEY AUTOINCREMENT,
  def_id        INTEGER NOT NULL REFERENCES raw_defs(def_id) ON DELETE CASCADE,
  location      TEXT,                           -- folio/page/line
  context       TEXT                            -- short snippet
);

-- 3) Synthesized definitions per ngram/cluster
CREATE TABLE IF NOT EXISTS synth_defs (
  synth_id      INTEGER PRIMARY KEY AUTOINCREMENT,
  ngram         TEXT    NOT NULL,
  cluster_id    INTEGER REFERENCES clusters(cluster_id) ON DELETE SET NULL,
  synth_def     TEXT    NOT NULL,               -- short gloss used for reconstruction
  notes         TEXT,                           -- any kind of commentary necessary
  members       TEXT    NOT NULL,               -- JSON: [def_id, ...]
  method_meta   TEXT,                           -- JSON of best_config/method commentary
  created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

-- 4) Skips (gate rejections before debate)
CREATE TABLE IF NOT EXISTS skips (
  skip_id       INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id        TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  ngram         TEXT NOT NULL,
  cluster_index INTEGER,
  reason_code   TEXT,                           -- e.g., LOW_OVERLAP, SMALL_CLUSTER, LOW_COHESION
  sem_count     INTEGER,
  idx_count     INTEGER,
  overlap_count INTEGER,
  created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

-- 5) Generic decisions table (optional, flexible)
CREATE TABLE IF NOT EXISTS decisions (
  decision_id   INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id        TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  target_kind   TEXT CHECK (target_kind IN ('lexeme','sense','attestation','cluster','synth')) NOT NULL,
  target_id     TEXT NOT NULL,
  verdict       TEXT CHECK (verdict IN ('accept','reject','hold')),
  reason_code   TEXT,
  note          TEXT,
  timestamp     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_clusters_ngram       ON clusters(ngram);
CREATE INDEX IF NOT EXISTS idx_clusters_run_ngram   ON clusters(run_id, ngram);
CREATE INDEX IF NOT EXISTS idx_raw_defs_cluster     ON raw_defs(cluster_id);
CREATE INDEX IF NOT EXISTS idx_raw_defs_source      ON raw_defs(source_word);
CREATE INDEX IF NOT EXISTS idx_citations_def        ON citations(def_id);
CREATE INDEX IF NOT EXISTS idx_skips_run_ngram      ON skips(run_id, ngram);
CREATE INDEX IF NOT EXISTS idx_clusters_ngram_has_gloss
ON clusters(ngram, cluster_id)
WHERE TRIM(COALESCE(glossator_def, '')) <> '';

-- Convenience view because why not
CREATE VIEW IF NOT EXISTS accepted_clusters AS
SELECT cluster_id, ngram, glossator_def
FROM clusters
WHERE TRIM(COALESCE(glossator_def, '')) <> '';
"""

# -------------------------
# FTS5 (created only if available)
# -------------------------

SCHEMA_FTS5 = """
CREATE VIRTUAL TABLE IF NOT EXISTS raw_defs_fts USING fts5(
  definition,
  enhanced_def,
  content='raw_defs',
  content_rowid='def_id'
);

-- Keep FTS in sync
CREATE TRIGGER IF NOT EXISTS raw_defs_ai AFTER INSERT ON raw_defs BEGIN
  INSERT INTO raw_defs_fts(rowid, definition, enhanced_def)
  VALUES (new.def_id, new.definition, new.enhanced_def);
END;

CREATE TRIGGER IF NOT EXISTS raw_defs_ad AFTER DELETE ON raw_defs BEGIN
  DELETE FROM raw_defs_fts WHERE rowid = old.def_id;
END;

CREATE TRIGGER IF NOT EXISTS raw_defs_au AFTER UPDATE ON raw_defs BEGIN
  UPDATE raw_defs_fts
  SET definition = new.definition,
      enhanced_def = new.enhanced_def
  WHERE rowid = new.def_id;
END;
"""

# -------------------------
# Migration helpers
# -------------------------

def _migrate_add_missing_columns(conn: sqlite3.Connection) -> None:
    """
    For existing databases, add new columns that did not exist previously.
    We deliberately avoid retrofitting CHECK constraints.
    """
    # clusters new columns
    cluster_new_cols = {
        "sem_count": "INTEGER",
        "idx_count": "INTEGER",
        "overlap_count": "INTEGER",
        "glossator_prompt": "TEXT",
        "glossator_model": "TEXT",
        "adjudicator_prompt": "TEXT",
        "adjudicator_model": "TEXT",
        "glossator_def": "TEXT",
        "adjudicator_verdict": "TEXT",
        "semantic_cohesion": "REAL",
        "derivational_validity": "REAL",
        "rebuttal_resilience": "REAL",
        "cohesion": "REAL",
        "semantic_coverage": "REAL",
        "best_config": "TEXT",
    }
    if _table_exists(conn, "clusters"):
        for col, decl in cluster_new_cols.items():
            _add_column_if_missing(conn, "clusters", col, decl)

    # raw_defs new columns
    raw_defs_new_cols = {
        "enhanced_def": "TEXT",
        "fasttext": "REAL",
        "similarity": "REAL",
        "tier": "TEXT",
    }
    if _table_exists(conn, "raw_defs"):
        for col, decl in raw_defs_new_cols.items():
            _add_column_if_missing(conn, "raw_defs", col, decl)

def _ensure_fts5(conn: sqlite3.Connection) -> None:
    if _fts5_available(conn):
        conn.executescript(SCHEMA_FTS5)
    else:
        # FTS5 not available; we just skip without failing.
        pass

# -------------------------
# Public API
# -------------------------

def init_db(path: str) -> None:
    """
    Initialize (or migrate) a database at `path`.
    """
    with _open(path) as conn:
        try:
            conn.executescript(SCHEMA_CREATE)
            _migrate_add_missing_columns(conn)
            _ensure_fts5(conn)
        except sqlite3.Error as e:
            raise RuntimeError(f"Database initialization failed: {e}") from e

    print(f"Initialized insights DB at {path}")

# -------------------------
# CLI
# -------------------------

if __name__ == "__main__":
    dbs = [
        "debate_derived_definitions.sqlite3",
        "solo_analysis_derived_definitions.sqlite3",
    ]
    for name in dbs:
        init_db(DB_PATH(name))
