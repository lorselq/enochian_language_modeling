from __future__ import annotations

import sqlite3


def fetch_token_morphs(
    conn: sqlite3.Connection, token: str, *, run_id: str | None = None
) -> list[str]:
    """Return stored morph sequence for ``token`` ordered by seg_index."""

    if not token:
        return []

    query = "SELECT morph FROM token_morph_decomp WHERE token = ?"
    params: list[object] = [token]
    if run_id is not None:
        query += " AND run_id = ?"
        params.append(run_id)
    query += " ORDER BY seg_index"

    rows = conn.execute(query, tuple(params)).fetchall()
    return [str(row[0]) for row in rows if str(row[0]).strip()]


def fetch_token_segments(
    conn: sqlite3.Connection, token: str, *, run_id: str | None = None
) -> list[dict]:
    """Return morph segments (with spans and scores) for a token."""

    if not token:
        return []

    query = (
        "SELECT morph, span_start, span_end, score, source "
        "FROM token_morph_decomp WHERE token = ?"
    )
    params: list[object] = [token]
    if run_id is not None:
        query += " AND run_id = ?"
        params.append(run_id)
    query += " ORDER BY seg_index"

    rows = conn.execute(query, tuple(params)).fetchall()
    results = []
    for row in rows:
        results.append(
            {
                "morph": str(row[0]),
                "span": [int(row[1]), int(row[2])],
                "score": None if row[3] is None else float(row[3]),
                "source": row[4],
            }
        )
    return results


def tokens_with_morph(
    conn: sqlite3.Connection, morph: str, *, run_id: str | None = None
) -> list[str]:
    """List tokens containing ``morph`` in their stored decomposition."""

    if not morph:
        return []

    query = "SELECT DISTINCT token FROM token_morph_decomp WHERE morph = ?"
    params: list[object] = [morph]
    if run_id is not None:
        query += " AND run_id = ?"
        params.append(run_id)

    rows = conn.execute(query, tuple(params)).fetchall()
    return [str(row[0]) for row in rows if str(row[0]).strip()]


def summarize_token_decomp(
    conn: sqlite3.Connection, *, run_id: str | None = None
) -> dict[str, int]:
    """Return simple distribution counts for token decomposition rows."""

    base_query = "SELECT token, COUNT(*) as segments FROM token_morph_decomp"
    params: list[object] = []
    if run_id is not None:
        base_query += " WHERE run_id = ?"
        params.append(run_id)
    base_query += " GROUP BY token"

    rows = conn.execute(base_query, tuple(params)).fetchall()
    zero = single = multi = 0
    for row in rows:
        segs = int(row[1] or 0)
        if segs == 0:
            zero += 1
        elif segs == 1:
            single += 1
        else:
            multi += 1
    return {"zero": zero, "single": single, "multi": multi, "tokens": len(rows)}
