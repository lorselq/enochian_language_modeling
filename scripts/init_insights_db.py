import sqlite3
import os

DB_PATH = os.path.abspath(os.path.join(
    os.path.dirname(os.path.dirname(__file__)),  # Go up from scripts directory
    "src",
    "enochian_translation_team",
    "data",
    "new-definitions.sqlite3"
))

SCHEMA = """
PRAGMA foreign_keys = ON;

-- 1️⃣ One row per evaluated cluster
CREATE TABLE IF NOT EXISTS clusters (
  cluster_id       INTEGER PRIMARY KEY AUTOINCREMENT,
  ngram            TEXT      NOT NULL,
  cluster_index    INTEGER   NOT NULL,
  sem_count        INTEGER   NOT NULL,
  idx_count        INTEGER   NOT NULL,
  overlap_count    INTEGER   NOT NULL,                -- NEW: how many in both sets
  clustering_meta  JSON      NOT NULL,
  model_name       TEXT,
  model_params     JSON,
  request_start    TEXT,
  request_end      TEXT,
  latency_s        REAL,
  api_retries      INTEGER   NOT NULL DEFAULT 0,      -- default 0
  did_fallback     INTEGER   NOT NULL DEFAULT 0,
  accepted         INTEGER   NOT NULL DEFAULT 0,      -- 0 = rejected, 1 = accepted
  glossator_def    TEXT,                               -- final gloss (or NULL)
  recorded_at      TEXT    NOT NULL DEFAULT (
    strftime('%Y-%m-%dT%H:%M:%fZ','now')
  )
);

-- 2️⃣ All the raw definitions that went into each cluster
CREATE TABLE IF NOT EXISTS raw_defs (
  def_id         INTEGER PRIMARY KEY AUTOINCREMENT,
  cluster_id     INTEGER  NOT NULL 
                     REFERENCES clusters(cluster_id) ON DELETE CASCADE,
  source_word    TEXT     NOT NULL,
  definition     TEXT     NOT NULL,
  enhanced_def   TEXT,
  citations      JSON,
  fasttext       REAL,                                  -- NEW: fasttext score
  similarity     REAL,                                  -- semantic sim score
  tier           TEXT,
  created_at     TEXT     NOT NULL DEFAULT (
    strftime('%Y-%m-%dT%H:%M:%fZ','now')
  )
);

-- 3️⃣ Synthesized definitions per ngram
CREATE TABLE IF NOT EXISTS synth_defs (
  synth_id       INTEGER PRIMARY KEY AUTOINCREMENT,
  ngram          TEXT    NOT NULL,
  synth_def      TEXT    NOT NULL,
  members        JSON    NOT NULL,   -- list of def_id's that fed this
  method_meta    JSON    NOT NULL,   -- e.g. {"alg":"hdbscan","min_cluster_size":2}
  created_at     TEXT    NOT NULL DEFAULT (
    strftime('%Y-%m-%dT%H:%M:%fZ','now')
  )
);
"""

def init_db(path=DB_PATH):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"Failed to create directory for database: {e}") from e

    try:
        with sqlite3.connect(path) as conn:
            try:
                conn.execute("PRAGMA journal_mode = WAL;")
                conn.execute("PRAGMA foreign_keys = ON;")
                conn.executescript(SCHEMA)
                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                raise RuntimeError(f"Database initialization failed: {e}") from e
    except sqlite3.Error as e:
        raise RuntimeError(f"Failed to connect to database: {e}") from e

    print(f"Initialized insights DB at {path}")

if __name__ == "__main__":
    init_db()