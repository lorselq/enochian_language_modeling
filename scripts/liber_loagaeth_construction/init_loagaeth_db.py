#!/usr/bin/env python3
# init_loagaeth_db.py
# Create an empty SQLite database for Liber Loagaeth research with DLL-ready schema.
# Usage:
#   python init_loagaeth_db.py loagaeth.sqlite --overwrite
#
# The script only creates schema; you can insert rows later with your own loaders.
from enochian_translation_team.utils import sqlite_bootstrap  # noqa: F401
import os
import sys
import argparse
import sqlite3
from pathlib import Path

DDL = r"""
PRAGMA foreign_keys=ON;

-- 1) Source objects (leaf pages, tables/squares, prose blocks, etc.)
CREATE TABLE IF NOT EXISTS artifact (
  id         INTEGER PRIMARY KEY,
  kind       TEXT NOT NULL,               -- 'leaf','square','diamond','triangle','prose_block', etc.
  name       TEXT UNIQUE,                 -- e.g., 'Leaf 1a', '37b'
  title      TEXT,                        -- on the off-chance the leaf has a title, this is where it goes
  meta_json  TEXT                         -- JSON: width/height, transcription flags, comments
);

-- 2) Letter/glyph token (what traversals will link).
--    Allow multi-character cells (no length=1 check) and classify when helpful.
CREATE TABLE IF NOT EXISTS letter_node (
  id            INTEGER PRIMARY KEY,
  char          TEXT NOT NULL,               -- exactly as seen, including diacritics; may be >1 char
  artifact_id   INTEGER,                     -- where this token lives (may be NULL if abstract)
  char_type     TEXT DEFAULT 'letter' CHECK (char_type IN ('letter','digit','punct','space','multi','other')),
  grapheme_len  INTEGER DEFAULT 1,           -- optional: number of characters in grapheme (e.g., "ll" = 2, "é" = 1)
  word_token_id INTEGER,                     -- for use with word token table
  pos_in_word   INTEGER,                     -- for use with word token table
  attrs_json    TEXT,                        -- JSON: start_letter, end_letter, trailing_punctuation, etc.
  notes         TEXT,
  FOREIGN KEY(artifact_id) REFERENCES artifact(id) ON DELETE SET NULL
  FOREIGN KEY(word_token_id) REFERENCES word_token(id) ON DELETE CASCADE
);

-- 2.5) Word tokens (primarily for use with Leaf 1a and Leaf 1b)
CREATE TABLE IF NOT EXISTS word_token (
  id             INTEGER PRIMARY KEY,
  artifact_id    INTEGER NOT NULL,        -- which leaf/prose artifact
  block_idx      INTEGER NOT NULL,        -- paragraph number (0,1,2,…)
  pos_in_block   INTEGER NOT NULL,        -- word order within that paragraph
  form           TEXT NOT NULL,           -- the word, without trailing punctuation/flags
  normalized     TEXT,                    -- optional (e.g., lowercased)
  attrs_json     TEXT,                    -- {"trail_punct":".", "sic":true, "uncertain":true, "bold":true}
  FOREIGN KEY(artifact_id) REFERENCES artifact(id) ON DELETE CASCADE,
  UNIQUE(artifact_id, block_idx, pos_in_block)          -- easy reassembly
);


-- 3) Positions for squares/diamonds/prose (row/col, orientation degrees).
CREATE TABLE IF NOT EXISTS grid_cell (
  id              INTEGER PRIMARY KEY,
  artifact_id     INTEGER NOT NULL,            -- the particular table/square
  row_idx         INTEGER NOT NULL,
  col_idx         INTEGER NOT NULL,
  node_id         INTEGER NOT NULL,            -- which letter_node sits here
  orientation     INTEGER DEFAULT 0,           -- e.g., 0, 45, 90, 135
  meta_json       TEXT,                        -- additional information, e.g., inside-circle, inside-square, bolded
  FOREIGN KEY(artifact_id) REFERENCES artifact(id)    ON DELETE CASCADE,
  FOREIGN KEY(node_id)     REFERENCES letter_node(id) ON DELETE CASCADE,
  UNIQUE(artifact_id, row_idx, col_idx)
);

-- 4) Positions for prose (line -> token -> char). Use levels you actually have.
CREATE TABLE IF NOT EXISTS prose_cell (
  id            INTEGER PRIMARY KEY,
  artifact_id   INTEGER NOT NULL,            -- prose block or leaf
  line_idx      INTEGER NOT NULL,
  token_idx     INTEGER,                     -- optional
  char_idx      INTEGER,                     -- optional (index within token)
  node_id       INTEGER NOT NULL,
  FOREIGN KEY(artifact_id) REFERENCES artifact(id)    ON DELETE CASCADE,
  FOREIGN KEY(node_id)     REFERENCES letter_node(id) ON DELETE CASCADE,
  UNIQUE(artifact_id, line_idx, token_idx, char_idx)
);

-- 5) Traversals = saved reading paths (human-literate).
CREATE TABLE IF NOT EXISTS traversal (
  id           INTEGER PRIMARY KEY,
  name         TEXT NOT NULL UNIQUE,        -- e.g., '37a:row_major', 'Leaf1a:line_12', 'corpus:by_word'
  scope        TEXT NOT NULL,               -- 'prose','square','corpus','alphabet','custom'
  instruction  TEXT,                        -- plain English: "Down-right diagonals of 37a; each diag top->bottom"
  recipe_json  TEXT,                        -- JSON parameters if you want code to materialize edges later
  notes        TEXT
);

-- 6) Doubly-linked edges for a traversal (the materialized path).
CREATE TABLE IF NOT EXISTS traversal_edge (
  traversal_id    INTEGER NOT NULL,
  node_id  INTEGER NOT NULL,
  prev_node_id    INTEGER,
  next_node_id    INTEGER,
  pos             INTEGER NOT NULL,           -- 0..N-1 sequence order
  PRIMARY KEY(traversal_id, node_id),
  UNIQUE(traversal_id, pos),
  UNIQUE(traversal_id, next_node_id),
  UNIQUE(traversal_id, prev_node_id),
  FOREIGN KEY(traversal_id)   REFERENCES traversal(id)   ON DELETE CASCADE,
  FOREIGN KEY(node_id) REFERENCES letter_node(id) ON DELETE CASCADE,
  FOREIGN KEY(prev_node_id)   REFERENCES letter_node(id) ON DELETE SET NULL,
  FOREIGN KEY(next_node_id)   REFERENCES letter_node(id) ON DELETE SET NULL
);

-- =========================
-- JSON export views
-- =========================

-- Traversal as ordered array of objects (node_id, char, prev, next, pos, artifact).
CREATE VIEW IF NOT EXISTS traversal_json AS
SELECT
  t.id          AS traversal_id,
  t.name        AS traversal_name,
  t.scope       AS traversal_scope,
  t.instruction AS instruction,
  (
    SELECT json_group_array(
             json_object(
               'node_id',  e.node_id,
               'char',     n.char,
               'prev',     e.prev_node_id,
               'next',     e.next_node_id,
               'pos',      e.pos,
               'artifact', n.artifact_id
             )
           )
    FROM (
      SELECT node_id, prev_node_id, next_node_id, pos
      FROM traversal_edge
      WHERE traversal_id = t.id
      ORDER BY pos
    ) AS e
    JOIN letter_node n ON n.id = e.node_id
  ) AS as_json
FROM traversal t;

CREATE VIEW IF NOT EXISTS traversal_letters_json AS
SELECT
  t.id    AS traversal_id,
  t.name  AS traversal_name,
  t.scope AS traversal_scope,
  t.instruction AS instruction,
  (
    SELECT json_group_array(x.char)
    FROM (
      SELECT n.char
      FROM traversal_edge e
      JOIN letter_node n ON n.id = e.node_id
      WHERE e.traversal_id = t.id
      ORDER BY e.pos
    ) AS x
  ) AS letters_json
FROM traversal t;

CREATE VIEW IF NOT EXISTS artifact_grid_cells_json AS
SELECT
  a.id   AS artifact_id,
  a.name AS artifact_name,
  a.kind AS artifact_kind,
  (
    SELECT json_group_array(cell_obj)
    FROM (
      SELECT json_object(
               'row', gc.row_idx,
               'col', gc.col_idx,
               'node_id', gc.node_id,
               'char', ln.char,
               'orientation', gc.orientation
             ) AS cell_obj
      FROM grid_cell gc
      JOIN letter_node ln ON ln.id = gc.node_id
      WHERE gc.artifact_id = a.id
      ORDER BY gc.row_idx, gc.col_idx
    )
  ) AS cells_json
FROM artifact a
WHERE a.kind IN ('square','diamond','slants','prose');

CREATE VIEW IF NOT EXISTS artifact_grid_rows_json AS
SELECT
  a.id   AS artifact_id,
  a.name AS artifact_name,
  a.kind AS artifact_kind,
  (
    SELECT json_group_array(row_block)
    FROM (
      SELECT (
        SELECT json_group_array(cell_obj)
        FROM (
          SELECT json_object(
                   'row', gc.row_idx,
                   'col', gc.col_idx,
                   'node_id', gc.node_id,
                   'char', ln.char,
                   'orientation', gc.orientation
                 ) AS cell_obj
          FROM grid_cell gc
          JOIN letter_node ln ON ln.id = gc.node_id
          WHERE gc.artifact_id = a.id
            AND gc.row_idx = r.row_idx
          ORDER BY gc.col_idx
        )
      ) AS row_block
      FROM (
        SELECT DISTINCT row_idx
        FROM grid_cell
        WHERE artifact_id = a.id
        ORDER BY row_idx
      ) AS r
    )
  ) AS rows_json
FROM artifact a
WHERE a.kind IN ('square','diamond','slants','prose');

CREATE VIEW IF NOT EXISTS artifact_diag_dr_json AS
SELECT
  a.id   AS artifact_id,
  a.name AS artifact_name,
  (
    SELECT json_group_array(diag_block)
    FROM (
      SELECT json_object(
               'diag', d.d,
               'cells',
                 (SELECT json_group_array(cell_obj)
                  FROM (
                    SELECT json_object(
                             'row', gc.row_idx,
                             'col', gc.col_idx,
                             'node_id', gc.node_id,
                             'char', ln.char,
                             'orientation', gc.orientation
                           ) AS cell_obj
                    FROM grid_cell gc
                    JOIN letter_node ln ON ln.id = gc.node_id
                    WHERE gc.artifact_id = a.id
                      AND (gc.col_idx - gc.row_idx) = d.d
                    ORDER BY gc.row_idx
                  ))
             ) AS diag_block
      FROM (
        SELECT DISTINCT (col_idx - row_idx) AS d
        FROM grid_cell
        WHERE artifact_id = a.id
        ORDER BY (col_idx - row_idx)
      ) AS d
    )
  ) AS dr_diagonals_json
FROM artifact a
WHERE a.kind IN ('square','diamond','slants','prose');

CREATE VIEW IF NOT EXISTS artifact_diag_dl_json AS
SELECT
  a.id   AS artifact_id,
  a.name AS artifact_name,
  (
    SELECT json_group_array(diag_block)
    FROM (
      SELECT json_object(
               'diag', s.s,
               'cells',
                 (SELECT json_group_array(cell_obj)
                  FROM (
                    SELECT json_object(
                             'row', gc.row_idx,
                             'col', gc.col_idx,
                             'node_id', gc.node_id,
                             'char', ln.char,
                             'orientation', gc.orientation
                           ) AS cell_obj
                    FROM grid_cell gc
                    JOIN letter_node ln ON ln.id = gc.node_id
                    WHERE gc.artifact_id = a.id
                      AND (gc.col_idx + gc.row_idx) = s.s
                    ORDER BY gc.row_idx
                  ))
             ) AS diag_block
      FROM (
        SELECT DISTINCT (col_idx + row_idx) AS s
        FROM grid_cell
        WHERE artifact_id = a.id
        ORDER BY (col_idx + row_idx)
      ) AS s
    )
  ) AS dl_diagonals_json
FROM artifact a
WHERE a.kind IN ('square','diamond','slants','prose');

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_letter_artifact ON letter_node(artifact_id);
CREATE INDEX IF NOT EXISTS idx_grid_by_art    ON grid_cell(artifact_id, row_idx, col_idx);
CREATE INDEX IF NOT EXISTS idx_trav_pos       ON traversal_edge(traversal_id, pos);
CREATE UNIQUE INDEX IF NOT EXISTS ux_artifact_name ON artifact(name);
"""


def ensure_db_file(db_path: str | Path) -> Path:
    """
    Ensure the SQLite file exists. If missing, create parent dirs and an empty DB.
    Returns the resolved Path. Exits on failure.
    """
    path = Path(db_path).expanduser().resolve()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"[FATAL] Cannot create directory {path.parent}: {e}", file=sys.stderr)
        sys.exit(1)

    if not path.exists():
        try:
            # Using sqlite3 to create the file is safest/atomic.
            sqlite3.connect(str(path)).close()
            # Optional: lock down perms on POSIX systems.
            try:
                os.chmod(path, 0o600)
            except Exception:
                pass
            print(f"[init] Created new SQLite file: {path}")
        except sqlite3.Error as e:
            print(f"[FATAL] Could not create database at {path}: {e}", file=sys.stderr)
            sys.exit(1)

    return path


def main():
    file_name = "liber_loagaeth.sqlite3"
    ap = argparse.ArgumentParser(
        description="Initialize a human-literate SQLite DB for Liber Loagaeth."
    )
    # ap.add_argument("db_path", help="Path to the .sqlite file to create")
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),  # Go up from scripts directory
            "src",
            "enochian_translation_team",
            "data",
            file_name,
        )
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing DB file if present"
    )
    args = ap.parse_args()

    ensure_db_file(path)
    db = Path(path)
    if db.exists() and not args.overwrite:
        print(f"Refusing to overwrite existing file: {db} (use --overwrite)")
        raise SystemExit(2)
    if db.exists():
        db.unlink()

    con = sqlite3.connect(str(db))
    try:
        con.executescript(DDL)
        con.commit()
    finally:
        con.close()

    print(f"Initialized human-readable schema in: {db}")


if __name__ == "__main__":
    main()
