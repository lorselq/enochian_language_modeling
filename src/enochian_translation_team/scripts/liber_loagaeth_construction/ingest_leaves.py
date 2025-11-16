#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ingest_loagaeth.py

A single loader for BOTH:
  • Prose artifacts: {"artifact": {"kind":"prose", ...}, "blocks": ["para 1", "para 2", ...]}
  • Grid artifacts (squares/diamonds/triangles):
        {"artifact": {"kind":"square", ...}, "grid_lines": ["a|b|...","..."]}
     or {"artifact": {...}, "grid_psv": "a|b|...\n..."}

What it does:
  - Ensures the SQLite schema (adds missing columns/tables/indexes safely).
  - Inserts/updates the artifact (unique by name; merges meta; stores title).
  - PROSE: splits into words, peels flags (trailing punctuation, [sic], bold '*', uncertainty),
           stores word tokens; inserts one letter_node per character (no spaces).
  - GRID:  parses each pipe-separated cell; supports blanks, [sic], bold, uncertainty, and
           orientation markers (^135 or ^[135]); creates grid_cell rows.
  - Creates letter-level traversals in reading order (prose: by words; grid: row-major).
  - Optional exports for quick verification.

Flags/markers supported:
  - Trailing punctuation on tokens: , . ; : ! ?    → stored in attrs_json.trail_punct
  - [sic] at token end                           → attrs_json.sic = true
  - Trailing '*' (one or more)                   → attrs_json.bold = true
  - Uncertainty marker inside token (default '?', override via meta.uncertain_marker)
                                                → attrs_json.uncertain = true (marker removed)
  - Orientation for grid cells: ^[deg] or ^deg   → orientation column in grid_cell

Usage:
  python ingest_loagaeth.py DB.sqlite INPUT.json [--include-blanks-in-traversal] [--pad-to-meta-width]
  # Quick check (prose): print reconstructed paragraphs
  python ingest_loagaeth.py DB.sqlite INPUT.json --export-prose "Leaf 1a"
"""
from __future__ import annotations
import os
import json
import re
import json
from enochian_common.sqlite_bootstrap import sqlite3
import unicodedata
import sys
from pathlib import Path
from typing import (
    Optional,
    List,
    Tuple,
    Dict,
    Any,
    Mapping,
    Union,
    cast,
    NoReturn,
    Sequence,
)

# -----------------------------
# Regexes / token peelers
# -----------------------------
RE_SIC = re.compile(r"(?i)\[sic\]\s*$")
RE_BOLD = re.compile(r"\*+\s*$")
RE_TRAILPUNCT = re.compile(
    r"([,.;:!?])\s*$"
)  # peel right-edge punctuation (one at a time)
RE_UNCERTAIN = re.compile(r'\?\s*$')
# Grid-only helpers
RE_ORIENT    = re.compile(r'\^\[(-?\d+)\]\s*$') # ^[135]
RE_ORIENT_BR = re.compile(r"\^\[(-?\d+)\]\s*$")  # ^[135]; alternative name
RE_ORIENT_LE = re.compile(r"\^(-?\d+)\s*$")  # legacy ^135
RE_NULL_WORD = re.compile(r"(?i)^\s*NULL\s*$")  # legacy textual NULL
RE_TRAIL_PUNCT = re.compile(r'[.,;:!]+$')  # only at END of token
RE_NULL     = re.compile(r'(?i)^\s*NULL\s*$')

GRID_KINDS = {"square", "rectangle", "diamond", "slants"}

Params = Union[Sequence[Any], Mapping[str, Any]]


# -----------------------------
# Small utilities
# -----------------------------
def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def has_capital(s: str) -> bool:
    return any(ch.isupper() and ch.lower() != ch for ch in s)


def peel_word_flags(raw: str, uncertain_marker: str = "?") -> Tuple[str, Dict]:
    """
    Peel markers from a prose token, returning (clean_form, attrs).
    Order:
      1) [sic]
      2) trailing punctuation (repeat): , . ; : ! ?
      3) trailing * (bold)
      4) uncertainty marker anywhere (default '?')
    """
    t = raw
    attrs: Dict = {}

    m = RE_SIC.search(t)
    if m:
        attrs["sic"] = True
        t = t[: m.start()]

    parts = []
    while True:
        m = RE_TRAILPUNCT.search(t)
        if not m:
            break
        parts.append(m.group(1))
        t = t[: m.start()]
    if parts:
        attrs["trail_punct"] = "".join(reversed(parts))

    m = RE_BOLD.search(t)
    if m:
        attrs["bold"] = True
        t = t[: m.start()]

    if uncertain_marker and uncertain_marker in t:
        attrs["uncertain"] = True
        t = t.replace(uncertain_marker, "")
    elif uncertain_marker != "?" and "?" in t:
        attrs["uncertain"] = True
        t = t.replace("?", "")

    t = nfc(t.strip())
    return t, attrs


def peel_grid_token(raw: str, meta: Dict) -> Tuple[bool, str, Dict, Optional[int]]:
    """
    Peel markers from a grid cell token.
    Returns (is_blank, char, attrs, orientation).
    - Blanks recognized via meta.lacuna_token / meta.intentionally_empty / legacy 'NULL'
    - Orientation: ^[deg] or ^deg
    - [sic], trailing punctuation (repeat), trailing '*', uncertainty marker
    """
    t = raw
    attrs: Dict = {}

    # blanks via meta tokens or legacy word 'NULL'
    lac = (meta.get("lacuna_token") or "").strip()
    emp = (meta.get("intentionally_empty") or "").strip()
    if lac and t.strip().lower() == lac.lower():
        attrs["lacuna"] = True
        attrs["blank"] = True
        return True, "", attrs, None
    if emp and t.strip().lower() == emp.lower():
        attrs["intentionally_empty"] = True
        attrs["blank"] = True
        return True, "", attrs, None
    if RE_NULL_WORD.match(t):
        attrs["blank"] = True
        return True, "", attrs, None

    # orientation (strip it off token)
    m = RE_ORIENT_BR.search(t) or RE_ORIENT_LE.search(t)
    orient = int(m.group(1)) if m else None
    if m:
        t = t[: m.start()]

    # [sic]
    m = RE_SIC.search(t)
    if m:
        attrs["sic"] = True
        t = t[: m.start()]

    # trailing punctuation (collect as a run)
    parts = []
    while True:
        m = RE_TRAILPUNCT.search(t)
        if not m:
            break
        parts.append(m.group(1))
        t = t[: m.start()]
    if parts:
        attrs["trail_punct"] = "".join(reversed(parts))

    # bold via trailing *
    m = RE_BOLD.search(t)
    if m:
        attrs["bold"] = True
        t = t[: m.start()]

    # uncertainty marker (default '?')
    um = meta.get("uncertain_marker") or "?"
    if um and um in t:
        attrs["uncertain"] = True
        t = t.replace(um, "")
    elif um != "?" and "?" in t:
        attrs["uncertain"] = True
        t = t.replace("?", "")

    t = nfc(t.strip())
    if not t:  # became empty after peeling → treat as blank cell
        attrs["blank"] = True
        return True, "", attrs, orient

    if has_capital(t):
        attrs["isCapital"] = True

    return False, t, attrs, orient


def _die(
    msg: str, *, sql: Optional[str] = None, params: Optional[Params] = None
) -> NoReturn:
    print(f"[FATAL] {msg}", file=sys.stderr)
    if sql is not None:
        print("  SQL:", sql, file=sys.stderr)
    if params is not None:
        try:
            pretty = json.dumps(
                params if isinstance(params, Mapping) else list(params),
                ensure_ascii=False,
            )
        except Exception:
            pretty = str(params)
        print("  params:", pretty, file=sys.stderr)
    sys.exit(1)


def _as_json_text(value) -> str | None:
    """Return a JSON string for dict/list, pass through str for other types, or None."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    # already a string? keep it; otherwise coerce to str
    return value if isinstance(value, str) else str(value)


def insert_returning_id(
    con: sqlite3.Connection, sql_base: str, params: Params, *, table: str
) -> int:
    """
    Execute exactly one INSERT and guarantee an integer primary key back.
    Uses INSERT ... RETURNING id when available; otherwise falls back to lastrowid.
    Fails fast with a clear message on any anomaly.
    """
    sql_ret = sql_base.strip() + " RETURNING id"

    def _exec(sql: str) -> Tuple[int]:
        cur = con.execute(sql, params)  # params: Sequence|Mapping → satisfies Pylance
        row = cast(Optional[Tuple[Any]], cur.fetchone())
        if row is None or row[0] is None:
            _die(
                f"`{table}` insert did not return an id (no row returned).",
                sql=sql,
                params=params,
            )
        return cast(Tuple[int], row)

    # Prefer RETURNING on modern SQLite
    try:
        row = _exec(sql_ret)
        return int(row[0])
    except sqlite3.OperationalError as e:
        # Probably no RETURNING support; fall back
        if "RETURNING" not in str(e).upper():
            _die(f"`{table}` insert failed: {e}", sql=sql_ret, params=params)

    # Fallback path (older SQLite)
    try:
        cur = con.execute(sql_base, params)
        rid = cur.lastrowid
        if rid is None:
            _die(
                f"`{table}` insert executed but lastrowid is None "
                "(possible triggers, OR IGNORE, or driver quirk).",
                sql=sql_base,
                params=params,
            )
        return int(rid)
    except sqlite3.IntegrityError as ie:
        _die(
            f"`{table}` insert violated a constraint: {ie}", sql=sql_base, params=params
        )
    except sqlite3.DatabaseError as de:
        _die(f"`{table}` insert failed: {de}", sql=sql_base, params=params)


def _row_exists(con, sql: str, params=()) -> bool:
    return con.execute(sql, params).fetchone() is not None


def insert_word_token(
    con,
    *,
    artifact_id: int,
    block_idx: int,
    pos_in_block: int,
    form: str,
    normalized: str | None,
    attrs: dict | None,
) -> int:
    # Preflight FK: artifact must exist
    if not _row_exists(con, "SELECT 1 FROM artifact WHERE id = ?", (artifact_id,)):
        raise RuntimeError(f"[word_token] artifact_id {artifact_id} does not exist")

    sql = (
        "INSERT INTO word_token(artifact_id, block_idx, pos_in_block, form, normalized, attrs_json) "
        "VALUES (?,?,?,?,?,?) RETURNING id"
    )
    return insert_returning_id(
        con,
        sql,
        (
            artifact_id,
            block_idx,
            pos_in_block,
            form,
            normalized,
            _as_json_text(attrs or {}),
        ),
        table="word_token",
    )


# Split on whitespace only; we keep hyphens and internal punctuation WITH the token.
def tokenize_block_to_words(block: str) -> List[str]:
    # block may contain embedded newlines; treat them as whitespace
    return [t for t in re.split(r"\s+", block.strip()) if t]


def extract_word_attrs(token: str) -> Tuple[dict, str]:
    """
    Pulls off trailing flags from a single token and returns (attrs, core_token).
    We remove, in this order (repeating where applicable):
      1) [sic]
      2) ^[degrees] orientation
      3) ? (uncertain)
      4) * (bold marker, any number of trailing asterisks)
      5) trailing punctuation . , ; : !
    Everything left is the 'core' token (e.g., 'ubrăh-ax', 'IAN').
    """
    attrs: dict = {}
    core = token

    # [sic]
    if RE_SIC.search(core):
        attrs["sic"] = True
        core = RE_SIC.sub("", core).rstrip()

    # ^[deg]
    m = RE_ORIENT.search(core)
    if m:
        try:
            attrs["orientation"] = int(m.group(1))
        except ValueError:
            attrs["orientation_raw"] = m.group(1)
        core = RE_ORIENT.sub("", core).rstrip()

    # uncertain '?'
    if RE_UNCERTAIN.search(core):
        attrs["uncertain"] = True
        core = RE_UNCERTAIN.sub("", core).rstrip()

    # bold *
    if RE_BOLD.search(core):
        attrs["bold"] = True
        core = RE_BOLD.sub("", core).rstrip()

    # trailing punctuation
    m = RE_TRAIL_PUNCT.search(core)
    if m:
        attrs["trail_punct"] = m.group(0)
        core = RE_TRAIL_PUNCT.sub("", core).rstrip()

    return attrs, core


def strip_punct_and_normalize(token: str) -> Tuple[str, str | None]:
    """
    Returns (base_form, normalized). We lower-case for normalized; we do NOT
    remove internal hyphens. Trailing flags are handled by extract_word_attrs().
    """
    attrs, core = extract_word_attrs(token)
    # We return ONLY the stripped base here; normalization = lowercase of base.
    # (ingester will call extract_word_attrs separately and merge attrs at word level)
    return core, core.lower() if core else core


def _grapheme_iter(s: str) -> List[str]:
    """
    Very simple grapheme splitter: base char + following combining marks stay together.
    (Good enough for these texts; no 3rd-party modules required.)
    """
    clusters: List[str] = []
    current = ""
    for ch in s:
        if not current:
            current = ch
            continue
        if unicodedata.combining(ch) != 0:
            current += ch
        else:
            clusters.append(current)
            current = ch
    if current:
        clusters.append(current)
    return clusters


def _classify_char(ch: str) -> str:
    """
    Rough Unicode class → one of: letter, digit, punct, space, multi, other.
    """
    if len(ch) > 1:
        # e.g., 'v́' as single grapheme cluster counts as 'multi' when len>1 at code-point level
        # but if it's base+combining, we still consider it a letter-ish char_type.
        # We'll call it 'letter' if base is a letter; else 'multi'.
        base = ch[0]
        if unicodedata.category(base).startswith("L"):
            return "letter"
        return "multi"
    cat = unicodedata.category(ch)
    if cat.startswith("L"):
        return "letter"
    if cat.startswith("N"):
        return "digit"
    if cat.startswith("P"):
        return "punct"
    if cat.startswith("Z"):
        return "space"
    return "other"


def explode_word_to_chars(base_form: str) -> List[Tuple[str, str, int, dict]]:
    """
    Turn a base word into a list of letter records:
      [(char, char_type, grapheme_len, letter_attrs), ...]
    We PRESERVE internal hyphens and punctuation as their own letter_nodes.
    """
    letters: List[Tuple[str, str, int, dict]] = []
    for g in _grapheme_iter(base_form):
        ctype = _classify_char(g)
        lattrs = {}
        # "isCapital" per character (works for Latin). We check the first codepoint.
        # If the base is hyphen or punctuation, this will be False.
        if g[:1].isupper():
            lattrs["isCapital"] = True
        letters.append((g, ctype, len(g), lattrs))
    return letters


# -----------------------------
# DB schema (safe ensure)
# -----------------------------
def open_db(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA foreign_keys=ON;")
    ensure_schema(con)
    return con


def ensure_schema(con: sqlite3.Connection) -> None:
    # artifact
    con.execute(
        """
    CREATE TABLE IF NOT EXISTS artifact (
      id        INTEGER PRIMARY KEY,
      kind      TEXT NOT NULL,
      name      TEXT,            -- UNIQUE index added below
      title     TEXT,
      meta_json TEXT
    )"""
    )
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_artifact_name ON artifact(name)")

    # word_token (no dictionary; just observed tokens)
    con.execute(
        """
    CREATE TABLE IF NOT EXISTS word_token (
      id            INTEGER PRIMARY KEY,
      artifact_id   INTEGER NOT NULL,
      block_idx     INTEGER NOT NULL,
      pos_in_block  INTEGER NOT NULL,
      form          TEXT NOT NULL,
      normalized    TEXT,
      attrs_json    TEXT,
      FOREIGN KEY(artifact_id) REFERENCES artifact(id) ON DELETE CASCADE,
      UNIQUE(artifact_id, block_idx, pos_in_block)
    )"""
    )

    # letter_node
    con.execute(
        """
    CREATE TABLE IF NOT EXISTS letter_node (
      id            INTEGER PRIMARY KEY,
      char          TEXT NOT NULL,
      artifact_id   INTEGER,
      word_token_id INTEGER,
      pos_in_word   INTEGER,
      char_type     TEXT,        -- 'letter','multi','digit','blank'
      grapheme_len  INTEGER,
      attrs_json    TEXT,
      notes         TEXT,
      FOREIGN KEY(artifact_id)   REFERENCES artifact(id)   ON DELETE SET NULL,
      FOREIGN KEY(word_token_id) REFERENCES word_token(id) ON DELETE SET NULL
    )"""
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS ix_letter_word ON letter_node(word_token_id, pos_in_word)"
    )

    # grid geometry
    con.execute(
        """
    CREATE TABLE IF NOT EXISTS grid_cell (
      id           INTEGER PRIMARY KEY,
      artifact_id  INTEGER NOT NULL,
      row_idx      INTEGER NOT NULL,
      col_idx      INTEGER NOT NULL,
      node_id      INTEGER NOT NULL,
      orientation  INTEGER DEFAULT 0,
      FOREIGN KEY(artifact_id) REFERENCES artifact(id) ON DELETE CASCADE,
      FOREIGN KEY(node_id)     REFERENCES letter_node(id) ON DELETE CASCADE,
      UNIQUE(artifact_id, row_idx, col_idx)
    )"""
    )

    # traversals (doubly-linked lists over letter nodes)
    con.execute(
        """
    CREATE TABLE IF NOT EXISTS traversal (
      id          INTEGER PRIMARY KEY,
      name        TEXT NOT NULL UNIQUE,
      scope       TEXT NOT NULL,     -- 'prose','square','diamond','triangle','corpus','custom'
      instruction TEXT,
      recipe_json TEXT,
      notes       TEXT
    )"""
    )
    con.execute(
        """
    CREATE TABLE IF NOT EXISTS traversal_edge (
      traversal_id INTEGER NOT NULL,
      node_id      INTEGER NOT NULL,
      prev_node_id INTEGER,
      next_node_id INTEGER,
      pos          INTEGER NOT NULL,
      PRIMARY KEY(traversal_id, node_id),
      FOREIGN KEY(traversal_id) REFERENCES traversal(id) ON DELETE CASCADE,
      FOREIGN KEY(node_id)      REFERENCES letter_node(id) ON DELETE CASCADE
    )"""
    )


def ensure_artifact(
    con: sqlite3.Connection,
    *,
    kind: str,
    name: str,
    title: Optional[str],
    meta: Optional[Dict],
) -> int:
    row = con.execute(
        "SELECT id, meta_json, title FROM artifact WHERE name=?", (name,)
    ).fetchone()
    if row:
        art_id, meta_json, old_title = row
        # merge meta
        try:
            base = json.loads(meta_json) if meta_json else {}
        except Exception:
            base = {}
        base.update(meta or {})
        merged = _as_json_text(base)
        if (title and title != old_title) or merged != (meta_json or "{}"):
            con.execute(
                "UPDATE artifact SET title=?, meta_json=? WHERE id=?",
                (title or old_title, merged, art_id),
            )
        return art_id
    sql = "INSERT INTO artifact(kind,name,title,meta_json) VALUES (?,?,?,?)"
    meta_text = _as_json_text(meta)
    return insert_returning_id(
        con, sql, (kind, name, title, meta_text), table="artifact"
    )


# -----------------------------
# Insert helpers
# -----------------------------
def insert_letter_node(
    con,
    *,
    artifact_id: int,
    char: str,
    word_token_id: int | None,
    pos_in_word: int | None,
    ctype: str,
    glen: int,
    attrs: dict | None,
    notes: str = "",
) -> int:
    # Preflight FK: artifact must exist
    if not _row_exists(con, "SELECT 1 FROM artifact WHERE id = ?", (artifact_id,)):
        raise RuntimeError(f"[letter_node] artifact_id {artifact_id} does not exist")

    # If you have a FK on word_token_id in your DB, preflight it too
    if word_token_id is not None:
        if not _row_exists(
            con, "SELECT 1 FROM word_token WHERE id = ?", (word_token_id,)
        ):
            raise RuntimeError(
                f"[letter_node] word_token_id {word_token_id} does not exist (char='{char}', pos_in_word={pos_in_word})"
            )

    sql = (
        "INSERT INTO letter_node(artifact_id, char, word_token_id, pos_in_word, char_type, grapheme_len, attrs_json, notes) "
        "VALUES (?,?,?,?,?,?,?,?) RETURNING id"
    )
    try:
        return insert_returning_id(
            con,
            sql,
            (
                artifact_id,
                char,
                word_token_id,
                pos_in_word,
                ctype,
                glen,
                _as_json_text(attrs or {}),
                notes,
            ),
            table="letter_node",
        )
    except sqlite3.IntegrityError as e:
        # Re-raise with context that points to the bad token
        raise RuntimeError(
            f"[letter_node] FK failed for char='{char}' (artifact_id={artifact_id}, word_token_id={word_token_id}, pos_in_word={pos_in_word}): {e}"
        ) from e


def insert_grid_cell(
    con: sqlite3.Connection,
    *,
    artifact_id: int,
    row: int,
    col: int,
    node_id: int,
    orientation: Optional[int],
) -> int:
    sql = "INSERT INTO grid_cell(artifact_id, row_idx, col_idx, node_id, orientation) VALUES (?, ?, ?, ?, ?)"
    return insert_returning_id(
        con, sql, (artifact_id, row, col, node_id, orientation), table="grid_cell"
    )


def insert_prose_cell(con, artifact_id: int, line_idx: int, token_idx: int, char_idx: int, node_id: int) -> int:
            sql = """
            INSERT INTO prose_cell(artifact_id, line_idx, token_idx, char_idx, node_id)
            VALUES (?, ?, ?, ?, ?)
            """
            cur = con.execute(sql, (artifact_id, line_idx, token_idx, char_idx, node_id))
            return int(cur.lastrowid)
# def create_letter_traversal(
#     con: sqlite3.Connection,
#     *,
#     name: str,
#     scope: str,
#     node_ids: List[int],
#     instruction: str,
# ) -> int:
#     cur = con.cursor()
#     cur.execute(
#         """
#         INSERT INTO traversal(name,scope,instruction,recipe_json,notes)
#         VALUES (?,?,?,?,?)
#     """,
#         (name, scope, instruction, "{}", ""),
#     )
#     trav_id = cur.lastrowid
#     for pos, nid in enumerate(node_ids):
#         prev_id = node_ids[pos - 1] if pos > 0 else None
#         next_id = node_ids[pos + 1] if pos < len(node_ids) - 1 else None
#         con.execute(
#             """
#             INSERT INTO traversal_edge(traversal_id,node_id,prev_node_id,next_node_id,pos)
#             VALUES (?,?,?,?,?)
#         """,
#             (trav_id, nid, prev_id, next_id, pos),
#         )
#     return trav_id


# =========================
# Ingest: PROSE (Leaf 1a/1b style)
# =========================
from typing import Dict, Any, List, Tuple, Optional
import json

def ingest_prose(con, doc: Dict[str, Any]) -> int:
    """
    Load a prose artifact (Leaf 1a/1b style) from JSON:
      {
        "artifact": {"kind":"prose","name":"Leaf 1a","title":"...","meta": {...}},
        "blocks": ["paragraph text ...", "next paragraph ...", ...]
      }

    - Creates/ensures a single artifact row (unique by artifact.name).
    - For each block (paragraph), tokenizes into words (keeps internal hyphens),
      strips only trailing flags/punct into attrs_json, and inserts:
        * word_token row (form, normalized, attrs_json)
        * letter_node rows for each grapheme in the word (no spaces)
        * prose_cell row locating each letter (line_idx=block_idx, token_idx, char_idx)
    - Returns the artifact_id.
    """
    art = doc["artifact"]
    kind  = art.get("kind", "prose")
    name  = art["name"]
    title = art.get("title") or None
    meta  = art.get("meta") or {}

    # Ensure artifact (ensure_artifact should JSON-dump meta internally)
    artifact_id = ensure_artifact(con, kind=kind, name=name, title=title, meta=meta)

    # Extract paragraphs/blocks
    blocks: List[str] = doc["blocks"]
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("prose ingest: 'blocks' must be a non-empty list of strings")

    # Ingest each paragraph
    for block_idx, raw_block in enumerate(blocks):
        tokens = tokenize_block_to_words(raw_block)        # whitespace split; keeps hyphens inside
        for pos_in_block, token in enumerate(tokens):
            # Word-level attrs (trail_punct, sic, ?, *, ^[deg] …)
            word_attrs, _ = extract_word_attrs(token)
            base, norm = strip_punct_and_normalize(token)

            # Skip if base word vanished (e.g., token was only flags); rare but safe.
            if not base:
                continue

            # Insert word_token (FK enforces artifact existence)
            word_id = insert_word_token(
                con,
                artifact_id=artifact_id,
                block_idx=block_idx,
                pos_in_block=pos_in_block,
                form=base,
                normalized=norm,
                attrs=word_attrs,
            )

            # Insert each grapheme as a letter_node; then map into prose_cell
            for pos_in_word, (ch, ctype, glen, lattrs) in enumerate(explode_word_to_chars(base)):
                node_id = insert_letter_node(
                    con,
                    artifact_id=artifact_id,
                    char=ch,
                    word_token_id=word_id,
                    pos_in_word=pos_in_word,
                    ctype=ctype,
                    glen=glen,
                    attrs=lattrs,
                )
                insert_prose_cell(
                    con,
                    artifact_id=artifact_id,
                    line_idx=block_idx,
                    token_idx=pos_in_block,
                    char_idx=pos_in_word,
                    node_id=node_id,
                )

    con.commit()
    return artifact_id


# =========================
# Ingest: GRID (Leaf 2a–48b style tables)
# =========================

def ingest_grid(con, doc: Dict[str, Any]) -> int:
    """
    Load a grid/square/diamond artifact from JSON:
      {
        "artifact": {
          "kind": "square",
          "name": "Leaf 2a",
          "title": "alla opnay qviemmah",
          "meta": {
            "width": 49, "height": 49,
            "delimiter": "|",                    # optional; default '|'
            "intentionally_empty": "[NULL]",     # token for true blanks
            "lacuna_token": "[LACUNA]"           # token for textual gaps
            ...
          }
        },
        "grid_lines": [
          "a|b|...|h",
          "s|e|...|l",
          ...
        ]
      }

    Notes:
      - Each cell becomes ONE letter_node (we do NOT explode into per-grapheme here),
        because manuscript cells can contain multi-letters (e.g., 'll') or a letter with a
        trailing dot 'a.' (the dot is captured in attrs_json.trail_punct).
      - True blank cells are represented by a special node whose char is the
        meta['intentionally_empty'] token (default "[NULL]") and char_type='space'.
      - Orientation markers ^[deg] attached to a cell are stored on grid_cell.orientation.
      - Other trailing flags (sic, ?, *) go into the node attrs_json.
    """
    art = doc["artifact"]
    kind  = art.get("kind", "square")
    name  = art["name"]
    title = art.get("title") or None
    meta  = art.get("meta") or {}

    artifact_id = ensure_artifact(con, kind=kind, name=name, title=title, meta=meta)

    lines: List[str] = doc["grid_lines"]
    if not isinstance(lines, list) or not lines:
        raise ValueError("grid ingest: 'grid_lines' must be a non-empty list of pipe-delimited strings")

    delim: str = meta.get("delimiter", "|")
    null_token: str = meta.get("intentionally_empty", "[NULL]")
    lacuna_token: str = meta.get("lacuna_token", "[LACUNA]")

    # Helper to classify a whole cell's "char"
    def classify_cell_char(text: str) -> str:
        # Multi-codepoint cell (e.g., 'll' or 'v́') → try to infer letter-ish vs other.
        if len(text) > 1:
            base = text[0]
            if unicodedata.category(base).startswith('L'):
                return 'letter'
            return 'multi'
        if not text:
            return 'space'
        cat = unicodedata.category(text)
        if cat.startswith('L'):
            return 'letter'
        if cat.startswith('N'):
            return 'digit'
        if cat.startswith('P'):
            return 'punct'
        if cat.startswith('Z'):
            return 'space'
        return 'other'

    for r, line in enumerate(lines):
        cols = line.split(delim)  # keeps empty fields, including trailing empties
        for c, raw_cell in enumerate(cols):
            cell = raw_cell.strip()

            # Detect "true blank" (empty between pipes or explicit NULL token)
            is_blank = (cell == "" or RE_NULL.match(cell) or cell == null_token)
            is_lacuna = (cell == lacuna_token)

            orientation_val: Optional[int] = None

            # Extract trailing flags/punct/orientation if not blank/lacuna
            cell_attrs: Dict[str, Any] = {}
            cell_core = ""
            if not (is_blank or is_lacuna):
                cell_attrs, cell_core = extract_word_attrs(cell)  # reuses same tail rules
            else:
                # Use representative core text for placeholder nodes
                if is_blank:
                    cell_core = null_token
                    cell_attrs["blank"] = True
                if is_lacuna:
                    cell_core = lacuna_token
                    cell_attrs["lacuna"] = True

            # Pull orientation from attrs into grid_cell column
            if "orientation" in cell_attrs:
                try:
                    orientation_val = int(cell_attrs.pop("orientation"))
                except Exception:
                    orientation_val = 0

            # Safety: ensure non-empty 'char' for the node
            if not cell_core:
                # A cell like "." alone would have been stripped to empty by extract_word_attrs;
                # represent it as punctuation node with literal text "."
                cell_core = raw_cell.strip() or null_token
                if cell_core == null_token:
                    cell_attrs["blank"] = True

            # Build node properties
            node_char = cell_core
            node_ctype = classify_cell_char(node_char)
            # Capital flag (first codepoint check; if punctuation, this will be False)
            if node_char[:1].isupper():
                cell_attrs.setdefault("isCapital", True)

            # grapheme_len: count of code points in this cell's char
            glen = len(node_char)

            # Insert the node
            node_id = insert_letter_node(
                con,
                artifact_id=artifact_id,
                char=node_char,
                word_token_id=None,
                pos_in_word=None,
                ctype=node_ctype,
                glen=glen,
                attrs=cell_attrs,
            )

            # Insert grid position
            insert_grid_cell(
                con,
                artifact_id=artifact_id,
                row=r,
                col=c,
                node_id=node_id,
                orientation=(orientation_val if orientation_val is not None else 0)
            )

    con.commit()
    return artifact_id


# -----------------------------
# Simple prose export (sanity check)
# -----------------------------
def export_prose_blocks(con: sqlite3.Connection, artifact_name: str) -> List[str]:
    row = con.execute(
        "SELECT id FROM artifact WHERE name=?", (artifact_name,)
    ).fetchone()
    if not row:
        raise SystemExit(f"No artifact named {artifact_name!r}")
    art_id = row[0]
    out: List[str] = []
    rows = con.execute(
        """
        SELECT block_idx, pos_in_block, form,
               COALESCE(json_extract(attrs_json,'$.trail_punct'), '') AS trail
          FROM word_token
         WHERE artifact_id=?
         ORDER BY block_idx, pos_in_block
    """,
        (art_id,),
    )
    cur_block = -1
    words: List[str] = []
    for bidx, pos, form, trail in rows:
        if bidx != cur_block:
            if words:
                out.append(" ".join(words))
            cur_block = bidx
            words = []
        words.append(form + trail)
    if words:
        out.append(" ".join(words))
    return out


# -----------------------------
# CLI
# -----------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Ingest Liber Loagaeth JSON (prose or grid) into SQLite; creates letter traversals."
    )
    # ap.add_argument("db", help="Path to SQLite database")
    ap.add_argument("json", help="Path to input JSON (prose or grid)")
    ap.add_argument(
        "--include-blanks-in-traversal",
        action="store_true",
        help="Grid only: include blank cells in the row-major traversal",
    )
    ap.add_argument(
        "--pad-to-meta-width",
        action="store_true",
        help="Grid only: right-pad rows with blanks to meta.width",
    )
    ap.add_argument(
        "--export-prose",
        metavar="ARTIFACT_NAME",
        help="After ingest, print reconstructed prose blocks for the named artifact",
    )
    args = ap.parse_args()

    # db_path = Path(args.db)
    file_name = "liber_loagaeth.sqlite3"
    db_path = os.path.abspath(
        os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(__file__))
            ),  # Go up from scripts directory
            "src",
            "enochian_translation_team",
            "data",
            file_name,
        )
    )
    js_path = Path(args.json)
    if not js_path.exists():
        raise SystemExit(f"JSON not found: {js_path}")

    # open DB & ensure schema
    con = open_db(Path(db_path))
    try:
        doc = json.loads(js_path.read_text(encoding="utf-8"))
        art = doc.get("artifact") or {}
        kind = (art.get("kind") or "").strip().lower()
        if not kind:
            raise SystemExit(
                "artifact.kind is required ('prose','square','diamond','slants')."
            )

        if kind == "prose":
            ingest_prose(con, doc)
        elif kind in GRID_KINDS:
            ingest_grid(
                con,
                doc
            )
        else:
            raise SystemExit(f"Unsupported artifact.kind: {kind!r}")

        con.commit()
        print(f"OK: ingested {art.get('name','(unnamed)')} [{kind}]")

        if args.export_prose:
            print("\n--- Reconstructed prose ---\n")
            for para in export_prose_blocks(con, args.export_prose):
                print(para)
                print()
    finally:
        con.close()


if __name__ == "__main__":
    main()
