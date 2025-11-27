"""Helpers for extracting and persisting root remainders.

Root remainders capture the *rest* of a word once a candidate root is
removed. They complement uncovered residual fragments by explicitly
recording larger suffix/prefix pieces such as ``PSAD`` in ``NAZPSAD``.
The data are purely string-driven (no LLM calls) and can be computed
online during clustering or backfilled after the fact.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Sequence

from enochian_lm.common.sqlite_bootstrap import sqlite3


@dataclass
class RootRemainder:
    """Lightweight record representing a single remainder occurrence."""

    run_id: str
    root: str
    word: str
    normalized: str
    remainder: str
    kind: str
    span_start: int
    span_end: int
    created_at: str


def extract_root_remainders(root_norm: str, normalized: str) -> list[dict]:
    """
    Identify prefix/suffix/infix remainders for each occurrence of ``root_norm``.

    Case-insensitive matching is used to locate the root within ``normalized``;
    the returned remainder values preserve the lowercased ``normalized`` text.

    The returned dictionaries are shaped for direct SQLite insertion and
    include span indices of the remainder within ``normalized``.
    """

    if not root_norm or not normalized:
        return []

    root = str(root_norm).lower()
    target = str(normalized).lower()
    if not root or not target or len(root) > len(target):
        return []

    results: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    start = 0
    while True:
        idx = target.find(root, start)
        if idx == -1:
            break

        prefix = target[:idx]
        suffix = target[idx + len(root) :]

        if prefix:
            key = (prefix, "infix_prefix" if suffix else "prefix", f"{idx}:{idx+len(prefix)}")
            if key not in seen:
                seen.add(key)
                results.append(
                    {
                        "root": root,
                        "normalized": target,
                        "remainder": prefix,
                        "kind": "infix_prefix" if suffix else "prefix",
                        "span_start": 0,
                        "span_end": len(prefix),
                    }
                )

        if suffix:
            span_start = idx + len(root)
            key = (suffix, "infix_suffix" if prefix else "suffix", f"{span_start}:{span_start+len(suffix)}")
            if key not in seen:
                seen.add(key)
                results.append(
                    {
                        "root": root,
                        "normalized": target,
                        "remainder": suffix,
                        "kind": "infix_suffix" if prefix else "suffix",
                        "span_start": span_start,
                        "span_end": span_start + len(suffix),
                    }
                )

        start = idx + 1

    return results


def persist_root_remainders(
    conn: sqlite3.Connection, *, rows: Sequence[RootRemainder]
) -> None:
    """
    Insert ``rows`` into ``root_remainders`` after clearing prior rows for each
    (run_id, root, normalized) triple. This keeps the table idempotent when
    online evaluation or post-hoc backfills are rerun.
    """

    if not rows:
        return

    targets = {(row.run_id, row.root, row.normalized) for row in rows}
    with conn:
        for run_id, root, norm in targets:
            conn.execute(
                "DELETE FROM root_remainders WHERE run_id = ? AND root = ? AND normalized = ?",
                (run_id, root, norm),
            )
        conn.executemany(
            """
            INSERT INTO root_remainders (
                run_id, root, word, normalized, remainder, kind, span_start, span_end, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row.run_id,
                    row.root,
                    row.word,
                    row.normalized,
                    row.remainder,
                    row.kind,
                    row.span_start,
                    row.span_end,
                    row.created_at,
                )
                for row in rows
            ],
        )


def summarize_root_remainders(
    conn: sqlite3.Connection, *, run_id: str, root: str
) -> dict:
    """
    Aggregate stored remainders for ``root`` into a prompt-friendly summary.

    Returns a dictionary of the shape::

        {
            "top": [{"remainder": "psad", "freq": 2, "kinds": ["suffix"]}],
            "by_kind": {
                "suffix": [...],
                ...
            },
        }
    """

    cursor = conn.execute(
        """
        SELECT remainder, kind, COUNT(*) as freq
        FROM root_remainders
        WHERE run_id = ? AND root = ?
        GROUP BY remainder, kind
        """,
        (run_id, root.lower()),
    )

    by_kind: dict[str, list[dict]] = defaultdict(list)
    tally: dict[str, dict] = {}
    for remainder, kind, freq in cursor.fetchall():
        remainder = str(remainder or "").lower()
        kind = str(kind or "").strip()
        freq = int(freq or 0)
        if not remainder or not kind:
            continue
        by_kind[kind].append({"remainder": remainder, "freq": freq})
        if remainder not in tally:
            tally[remainder] = {"remainder": remainder, "freq": 0, "kinds": set()}
        tally[remainder]["freq"] += freq
        tally[remainder]["kinds"].add(kind)

    for entries in by_kind.values():
        entries.sort(key=lambda item: (item["freq"], item["remainder"]), reverse=True)

    top = sorted(
        tally.values(),
        key=lambda item: (item["freq"], item["remainder"]),
        reverse=True,
    )
    top_payload = [
        {"remainder": row["remainder"], "freq": row["freq"], "kinds": sorted(row["kinds"])}
        for row in top[:8]
    ]

    by_kind_payload = {kind: entries for kind, entries in by_kind.items()}

    return {"top": top_payload, "by_kind": by_kind_payload}


def backfill_root_remainders(
    conn: sqlite3.Connection, *, run_ids: Iterable[str]
) -> tuple[int, int]:
    """
    Compute and persist root remainders for completed runs without rerunning LLMs.

    Returns a tuple ``(roots_seen, remainders_inserted)``.
    """

    now = datetime.now(timezone.utc).isoformat()
    processed_roots = 0
    total_rows = 0

    for run_id in run_ids:
        cursor = conn.execute(
            """
            SELECT DISTINCT c.run_id, c.ngram, d.source_word, d.variant
            FROM clusters c
            JOIN raw_defs d ON d.cluster_id = c.cluster_id
            WHERE c.run_id = ?
            ORDER BY c.ngram
            """,
            (run_id,),
        )
        rows = cursor.fetchall()
        if not rows:
            continue

        batch: list[RootRemainder] = []
        seen_pairs: set[tuple[str, str]] = set()
        for row in rows:
            root = str(row["ngram"] or "").strip().lower()
            source_word = str(row["source_word"] or "").strip()
            display = str(row["variant"] or "").strip() or source_word
            normalized = source_word.lower()
            if not root or not normalized:
                continue
            seen_pairs.add((root, normalized))
            for remainder in extract_root_remainders(root, normalized):
                batch.append(
                    RootRemainder(
                        run_id=run_id,
                        root=root,
                        word=display,
                        normalized=normalized,
                        remainder=remainder["remainder"],
                        kind=remainder["kind"],
                        span_start=int(remainder["span_start"]),
                        span_end=int(remainder["span_end"]),
                        created_at=now,
                    )
                )

        if batch:
            persist_root_remainders(conn, rows=batch)
            total_rows += len(batch)
        processed_roots += len(seen_pairs)

    return processed_roots, total_rows

