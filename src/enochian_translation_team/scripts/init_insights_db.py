#!/usr/bin/env python3
"""
init_insights_db.py

Initializes (or gently migrates) the insights database used by the ELM debate/solo pipelines.

Design choices per request:
- **No FTS** anywhere (SQLite 3.37.2; json1 may or may not be present).
- Keep the column name **cohesion** (do not rename to semantic_cohesion).
- Provide a **clusters_processed VIEW** that mirrors `clusters` and filters to rows
  whose `glossator_def` is non-empty and (ideally) valid JSON. If JSON1 is available,
  we require `json_valid(glossator_def)=1`; otherwise we fall back to a conservative
  heuristic (starts with '{' and not an error prefix).
- Create schemas for both **solo** and **debate** variants; we infer the variant from
  the DB pathname (contains "solo" → solo; otherwise debate).

The script is idempotent and safe to re-run.
"""
from __future__ import annotations
from os import PathLike
import os
from enochian_common.sqlite_bootstrap import sqlite3
from typing import Dict, Tuple

# -------------------------
# Paths
# -------------------------


def DB_PATH(file_name: str = "debate_derived_definitions.sqlite3") -> str:
    """Build an absolute path into your repo's data/ directory.

    Example overrides:
        DB_PATH("raw_solo_analysis_derived_definitions.sqlite3")
        DB_PATH("processed_debate_derived_definitions.sqlite3")
    """
    return os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),  # up from scripts/
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


def _table_or_view_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ? LIMIT 1;",
        (name,),
    )
    return cur.fetchone() is not None


def _index_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ? LIMIT 1;",
        (name,),
    )
    return cur.fetchone() is not None


def _columns(
    conn: sqlite3.Connection, table: str
) -> Dict[str, Tuple[int, str, int, int, str]]:
    """Return PRAGMA table_info columns mapping: {name: (cid, name, type, notnull, dflt_value, pk)}"""
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return {r[1]: r for r in rows}


def _add_column_if_missing(
    conn: sqlite3.Connection, table: str, col: str, decl: str
) -> None:
    cols = _columns(conn, table)
    if col not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl};")


def _create_index_if_missing(conn: sqlite3.Connection, name: str, ddl: str) -> None:
    if not _index_exists(conn, name):
        conn.execute(ddl)


def _infer_variant_from_path(path: str) -> str:
    path_l = path.lower()
    return "solo" if "solo" in path_l else "debate"


def _has_json1(conn: sqlite3.Connection) -> bool:
    try:
        row = conn.execute("SELECT json_valid('{\"a\":1}')").fetchone()
        return bool(row and row[0] == 1)
    except sqlite3.Error:
        return False


# -------------------------
# Schema DDL (fresh create)
# -------------------------

CLUSTERS_DEBATE = """
CREATE TABLE IF NOT EXISTS clusters (
  cluster_id            INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id                TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  ngram                 TEXT NOT NULL,
  cluster_index         INTEGER NOT NULL,
  cluster_size          INTEGER,
  sem_count             INTEGER,
  idx_count             INTEGER,
  overlap_count         INTEGER,
  action                TEXT,
  reason                TEXT,
  model                 TEXT,
  proposal              TEXT,
  critique              TEXT,
  defense               TEXT,
  adjudicator_rounds    TEXT,
  skeptic_rounds        TEXT,
  linguist_rounds       TEXT,
  glossator_prompt      TEXT,
  glossator_def         TEXT,
  verdict               TEXT,
  derivational_validity REAL,
  rebuttal_resilience   REAL,
  semantic_cohesion     REAL,
  cohesion              REAL,
  semantic_coverage     REAL,
  best_config           TEXT,
  residual_explained    REAL,
  residual_ratio        REAL,
  residual_headline     TEXT,
  residual_focus_prompt TEXT,
  created_at            TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  UNIQUE (run_id, ngram, cluster_index)
);
"""

CLUSTERS_SOLO = """
CREATE TABLE IF NOT EXISTS clusters (
  cluster_id            INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id                TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  ngram                 TEXT NOT NULL,
  cluster_index         INTEGER NOT NULL,
  cluster_size          INTEGER,
  sem_count             INTEGER,
  idx_count             INTEGER,
  overlap_count         INTEGER,
  action                TEXT,
  reason                TEXT,
  model                 TEXT,
  glossator_prompt      TEXT,
  glossator_def         TEXT,
  verdict               TEXT,
  cohesion              REAL,
  semantic_coverage     REAL,
  best_config           TEXT,
  residual_explained    REAL,
  residual_ratio        REAL,
  residual_headline     TEXT,
  residual_focus_prompt TEXT,
  created_at            TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  UNIQUE (run_id, ngram, cluster_index)
);
"""

SCHEMA_CREATE = f"""
PRAGMA foreign_keys = ON;

-- 0) Runs: batch-level metadata
CREATE TABLE IF NOT EXISTS runs (
  run_id        TEXT PRIMARY KEY,
  run_name      TEXT,
  engine        TEXT CHECK (engine IN ('debate','solo')) NOT NULL DEFAULT 'debate',
  embedder      TEXT,
  env_json      TEXT,
  phase         TEXT NOT NULL DEFAULT 'translation',
  created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

-- 1) Clusters per variant
[[[CLUSTERS_BLOCK]]]

-- 2) Raw definitions (optional, kept for provenance)
CREATE TABLE IF NOT EXISTS raw_defs (
  def_id        INTEGER PRIMARY KEY AUTOINCREMENT,
  cluster_id    INTEGER NOT NULL REFERENCES clusters(cluster_id) ON DELETE CASCADE,
  source_word   TEXT    NOT NULL,
  variant       TEXT    NOT NULL,
  definition    TEXT    NOT NULL,
  enhanced_def  TEXT,
  fasttext      REAL,
  similarity    REAL,
  tier          TEXT,
  created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  UNIQUE (cluster_id, source_word, definition)
);

CREATE TABLE IF NOT EXISTS residual_details (
  residual_id        INTEGER PRIMARY KEY AUTOINCREMENT,
  cluster_id         INTEGER NOT NULL REFERENCES clusters(cluster_id) ON DELETE CASCADE,
  normalized         TEXT    NOT NULL,
  definition         TEXT,
  coverage_ratio     REAL,
  residual_ratio     REAL,
  avg_confidence     REAL,
  uncovered_json     TEXT,
  low_conf_json      TEXT,
  created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS citations (
  citation_id   INTEGER PRIMARY KEY AUTOINCREMENT,
  def_id        INTEGER NOT NULL REFERENCES raw_defs(def_id) ON DELETE CASCADE,
  location      TEXT,
  context       TEXT
);

CREATE TABLE IF NOT EXISTS synth_defs (
  synth_id      INTEGER PRIMARY KEY AUTOINCREMENT,
  ngram         TEXT    NOT NULL,
  cluster_id    INTEGER REFERENCES clusters(cluster_id) ON DELETE SET NULL,
  synth_def     TEXT    NOT NULL,
  notes         TEXT,
  members       TEXT    NOT NULL,
  method_meta   TEXT,
  created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS skips (
  skip_id       INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id        TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  ngram         TEXT NOT NULL,
  cluster_index INTEGER,
  reason_code   TEXT,
  sem_count     INTEGER,
  idx_count     INTEGER,
  overlap_count INTEGER,
  created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS preanalysis_runs (
  preanalysis_id   INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id           TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  stage            TEXT NOT NULL CHECK(stage IN ('initial','subsequent')),
  trusted_count    INTEGER NOT NULL DEFAULT 0,
  notes            TEXT,
  created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS preanalysis_seeds (
  seed_id          INTEGER PRIMARY KEY AUTOINCREMENT,
  preanalysis_id   INTEGER NOT NULL REFERENCES preanalysis_runs(preanalysis_id) ON DELETE CASCADE,
  ngram            TEXT NOT NULL,
  status           TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','completed','skipped')),
  analytics_payload TEXT,
  last_run_id      TEXT,
  created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  updated_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  UNIQUE(preanalysis_id, ngram)
);

-- raw
CREATE TABLE IF NOT EXISTS agent_raw (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  cluster_id INTEGER NOT NULL REFERENCES clusters(cluster_id) ON DELETE CASCADE,
  role TEXT,                     -- e.g., 'raw_output'
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_agent_raw_cluster ON agent_raw(cluster_id);

-- llm jobs
CREATE TABLE IF NOT EXISTS llm_job (
  job_id       INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id       TEXT REFERENCES runs(run_id) ON DELETE CASCADE,
  prompt_hash  TEXT UNIQUE,
  role         TEXT,               -- Glossator | Adjudicator | Skeptic | etc
  model        TEXT,
  base_url     TEXT,
  temperature  REAL,
  system_prompt TEXT,
  user_prompt  TEXT,
  request_json TEXT,               -- full request payload (safe to store)
  response_text TEXT,              -- final text (non-streamed we’d use usage JSON too)
  tokens_in    INTEGER,
  tokens_out   INTEGER,
  cost_usd     REAL,
  status       TEXT,               -- queued|ok|cached|failed
  error        TEXT,
  created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  finished_at  TEXT
);

CREATE TABLE IF NOT EXISTS morph_hypotheses (
  hyp_id           INTEGER PRIMARY KEY AUTOINCREMENT,
  morph            TEXT NOT NULL,          -- the residual (e.g., PSAD)
  source_word      TEXT NOT NULL,          -- the composite word that revealed it (e.g., NAZPSAD)
  anchor           TEXT,                   -- chosen anchor morph inside source_word (e.g., NAZ)
  seed_glosses     TEXT,                   -- JSON array of NN seed gloss labels (optional)
  proposed_gloss   TEXT,                   -- model-proposed 1-line gloss
  rationale        TEXT,                   -- model rationale (optional)
  delta_cosine     REAL,                   -- similarity quality metric for Δ→seed (optional)
  residual_before  REAL,                   -- from residual_details
  residual_after   REAL,                   -- recomputed after injecting morph temporarily
  accepted         INTEGER,                -- 0/1 gate
  created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  UNIQUE(morph, source_word)
);

-- Helpful indexes (no FTS)
CREATE INDEX IF NOT EXISTS idx_clusters_ngram       ON clusters(ngram);
CREATE INDEX IF NOT EXISTS idx_clusters_run_ngram   ON clusters(run_id, ngram);
CREATE INDEX IF NOT EXISTS idx_raw_defs_cluster     ON raw_defs(cluster_id);
CREATE INDEX IF NOT EXISTS idx_raw_defs_source      ON raw_defs(source_word);
CREATE INDEX IF NOT EXISTS idx_residual_details_cluster ON residual_details(cluster_id);
CREATE INDEX IF NOT EXISTS idx_citations_def        ON citations(def_id);
CREATE INDEX IF NOT EXISTS idx_skips_run_ngram      ON skips(run_id, ngram);
CREATE INDEX IF NOT EXISTS idx_llm_job_run          ON llm_job(run_id);

-- Convenience view of rows that have a non-empty definition (regardless of verdict)
CREATE VIEW IF NOT EXISTS accepted_clusters AS
SELECT cluster_id, ngram, glossator_def
FROM clusters
WHERE TRIM(COALESCE(glossator_def, '')) <> '';
"""

# -------------------------
# VIEW builders (JSON-valid filter if available)
# -------------------------

_DEF_VIEW_SOLO_JSON = """
CREATE VIEW IF NOT EXISTS clusters_processed AS
SELECT
  cluster_id,
  run_id,
  ngram,
  cluster_index,
  cluster_size,
  sem_count,
  idx_count,
  overlap_count,
  prevaluation,
  action,
  reason,
  model,
  glossator_prompt,
  glossator_def,
  LOWER(COALESCE(json_extract(glossator_def, '$.EVALUATION'), '')) AS verdict,
  cohesion,
  semantic_coverage,
  best_config,
  residual_explained,
  residual_ratio,
  residual_headline,
  residual_focus_prompt,
  created_at
FROM clusters
WHERE TRIM(COALESCE(glossator_def, '')) <> ''
  AND json_valid(glossator_def) = 1;
"""

_DEF_VIEW_DEBATE_JSON = """
CREATE VIEW IF NOT EXISTS clusters_processed AS
SELECT
  cluster_id,
  run_id,
  ngram,
  cluster_index,
  cluster_size,
  sem_count,
  idx_count,
  overlap_count,
  prevaluation,
  action,
  reason,
  model,
  proposal,
  critique,
  defense,
  adjudicator_rounds,
  skeptic_rounds,
  linguist_rounds,
  glossator_prompt,
  glossator_def,
  LOWER(COALESCE(json_extract(glossator_def, '$.EVALUATION'), '')) AS verdict,
  derivational_validity,
  rebuttal_resilience,
  semantic_cohesion,
  cohesion,
  semantic_coverage,
  best_config,
  residual_explained,
  residual_ratio,
  residual_headline,
  residual_focus_prompt,
  created_at
FROM clusters
WHERE TRIM(COALESCE(glossator_def, '')) <> ''
  AND json_valid(glossator_def) = 1;
"""

# Fallback when JSON1 is not available.
# Heuristic: leading '{' and not an error prefix.
_DEF_VIEW_SOLO_HEUR = """
CREATE VIEW IF NOT EXISTS clusters_processed AS
SELECT
  cluster_id,
  run_id,
  ngram,
  cluster_index,
  cluster_size,
  sem_count,
  idx_count,
  overlap_count,
  prevaluation,
  action,
  reason,
  model,
  glossator_prompt,
  glossator_def,
  LOWER(COALESCE(json_extract(glossator_def, '$.EVALUATION'), '')) AS verdict,
  cohesion,
  semantic_coverage,
  best_config,
  residual_explained,
  residual_ratio,
  residual_headline,
  residual_focus_prompt,
  created_at
FROM clusters
WHERE TRIM(COALESCE(glossator_def, '')) <> ''
  AND TRIM(glossator_def) GLOB '{*}'
  AND glossator_def NOT LIKE '[ERROR]%'
  AND glossator_def NOT LIKE 'ERROR%';
"""

_DEF_VIEW_DEBATE_HEUR = """
CREATE VIEW IF NOT EXISTS clusters_processed AS
SELECT
  cluster_id,
  run_id,
  ngram,
  cluster_index,
  cluster_size,
  sem_count,
  idx_count,
  overlap_count,
  prevaluation,
  action,
  reason,
  model,
  proposal,
  critique,
  defense,
  adjudicator_rounds,
  skeptic_rounds,
  linguist_rounds,
  glossator_prompt,
  glossator_def,
  LOWER(COALESCE(json_extract(glossator_def, '$.EVALUATION'), '')) AS verdict,
  derivational_validity,
  rebuttal_resilience,
  semantic_cohesion,
  cohesion,
  semantic_coverage,
  best_config,
  residual_explained,
  residual_ratio,
  residual_headline,
  residual_focus_prompt,
  created_at
FROM clusters
WHERE TRIM(COALESCE(glossator_def, '')) <> ''
  AND TRIM(glossator_def) GLOB '{*}'
  AND glossator_def NOT LIKE '[ERROR]%'
  AND glossator_def NOT LIKE 'ERROR%';
"""

_DEF_VIEW_ANCHORS = """
SELECT ngram, COUNT(*) AS uses, AVG(cohesion) AS avg_coh
FROM clusters
WHERE TRIM(glossator_def) <> ''
  AND COALESCE(cohesion, 0.0) >= 0.40
GROUP BY ngram
HAVING COUNT(*) >= 3
ORDER BY uses DESC, avg_coh DESC;
"""

ANALYSIS_TABLE_STATEMENTS = (
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
    CREATE TABLE IF NOT EXISTS residual_clusters (
      id INTEGER PRIMARY KEY,
      cluster_id INTEGER NOT NULL,
      centroid_json TEXT NOT NULL,
      size INTEGER NOT NULL,
      updated_at TEXT NOT NULL
    );
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
      updated_at TEXT NOT NULL
    );
    """,
)

# -------------------------
# Public API
# -------------------------


def _infer_variant_from_path(path: str | PathLike[str]) -> str:
    path_l = os.fspath(path).lower()
    return "solo" if "solo" in path_l else "debate"

def _open(db_path: str | PathLike[str]) -> sqlite3.Connection:
    db_path = os.fspath(db_path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn

def init_db(path: str | PathLike[str]) -> None:
    """Initialize (or migrate) a database at `path`."""
    variant = _infer_variant_from_path(path)
    with _open(path) as conn:
        try:
            schema = SCHEMA_CREATE.replace(
                "[[[CLUSTERS_BLOCK]]]",
                CLUSTERS_SOLO if variant == "solo" else CLUSTERS_DEBATE,
            )
            conn.executescript(schema)

            # Rebuild the processed VIEW to ensure the latest filter logic
            if _table_or_view_exists(conn, "clusters_processed"):
                conn.execute("DROP VIEW IF EXISTS clusters_processed;")

            if _has_json1(conn):
                ddl = (
                    _DEF_VIEW_SOLO_JSON if variant == "solo" else _DEF_VIEW_DEBATE_JSON
                )
            else:
                ddl = (
                    _DEF_VIEW_SOLO_HEUR if variant == "solo" else _DEF_VIEW_DEBATE_HEUR
                )
            conn.executescript(ddl)

            # Add anchor view
            conn.executescript(_DEF_VIEW_ANCHORS)

            # Ensure newly introduced columns exist for older databases
            shared_columns = {
                "action": "TEXT",
                "verdict": "TEXT",
                "residual_explained": "REAL",
                "residual_ratio": "REAL",
                "residual_headline": "TEXT",
                "residual_focus_prompt": "TEXT",
            }
            for column, decl in shared_columns.items():
                _add_column_if_missing(conn, "clusters", column, decl)

            if variant == "debate":
                debate_columns = {
                    "semantic_cohesion": "REAL",
                }
                for column, decl in debate_columns.items():
                    _add_column_if_missing(conn, "clusters", column, decl)

            _add_column_if_missing(conn, "runs", "phase", "TEXT NOT NULL DEFAULT 'translation'")

            for ddl in ANALYSIS_TABLE_STATEMENTS:
                conn.execute(ddl)

        except sqlite3.Error as e:
            raise RuntimeError(f"Database initialization failed: {e}") from e

    print(
        f"Initialized insights DB at {path} (variant={variant}, no FTS; json1={'yes' if _has_json1(_open(path)) else 'no'})"
    )


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
