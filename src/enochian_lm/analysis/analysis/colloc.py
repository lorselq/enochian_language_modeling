"""Collocation statistics derived from attribution marginals."""
from __future__ import annotations

import argparse
import logging
import math
from enochian_lm.common.sqlite_bootstrap import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator

from ..utils.sql import ensure_analysis_tables, upsert_rows
from ..utils.stats import llr, pmi
from ..utils.text import utcnow_iso

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _PairRecord:
    """Aggregated attribution statistics for a morph pair."""

    morph_left: str
    morph_right: str
    delta_a_given_b: float
    delta_b_given_a: float
    count_ab: int


def _fetch_pairs(conn: sqlite3.Connection, limit: int | None) -> list[_PairRecord]:
    query = (
        "SELECT morph_a, morph_b, delta_a_given_b, delta_b_given_a, n_tokens "
        "FROM attribution_marginals ORDER BY n_tokens DESC, morph_a, morph_b"
    )
    params: tuple[object, ...] = ()
    if limit is not None:
        query += " LIMIT ?"
        params = (limit,)

    rows = conn.execute(query, params).fetchall()
    records: list[_PairRecord] = []
    for row in rows:
        count = int(row["n_tokens"])
        if count <= 0:
            continue
        records.append(
            _PairRecord(
                morph_left=row["morph_a"],
                morph_right=row["morph_b"],
                delta_a_given_b=float(row["delta_a_given_b"]),
                delta_b_given_a=float(row["delta_b_given_a"]),
                count_ab=count,
            )
        )
    return records


def _batch_iterable(items: list[dict[str, object]], batch_size: int = 500) -> Iterator[list[dict[str, object]]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def compute_collocations(
    conn: sqlite3.Connection, *, min_count: int = 5, limit: int | None = None
) -> dict[str, object]:
    """Compute PMI/LLR collocation statistics from attribution marginals."""

    start = time.perf_counter()
    pairs = _fetch_pairs(conn, limit)
    if not pairs:
        logger.info("No attribution marginals available; skipping collocation stats")
        return {
            "timestamp": utcnow_iso(),
            "pairs": 0,
            "avg_pmi": 0.0,
            "avg_llr": 0.0,
            "avg_asym": 0.0,
        }

    counts_by_morph: dict[str, int] = defaultdict(int)
    for record in pairs:
        counts_by_morph[record.morph_left] += record.count_ab
        counts_by_morph[record.morph_right] += record.count_ab

    total = sum(record.count_ab for record in pairs)
    if total <= 0:
        logger.info("Total attribution counts are zero; skipping collocation stats")
        return {
            "timestamp": utcnow_iso(),
            "pairs": 0,
            "avg_pmi": 0.0,
            "avg_llr": 0.0,
            "avg_asym": 0.0,
        }

    timestamp = utcnow_iso()
    rows_to_write: list[dict[str, object]] = []
    pmi_values: list[float] = []
    llr_values: list[float] = []
    asym_values: list[float] = []
    debug_candidates: list[tuple[float, str, str, int]] = []

    for record in pairs:
        count_ab = record.count_ab
        if count_ab < min_count:
            continue

        count_a = counts_by_morph[record.morph_left]
        count_b = counts_by_morph[record.morph_right]
        if count_a <= 0 or count_b <= 0:
            continue

        k11 = count_ab
        k12 = max(count_a - count_ab, 0)
        k21 = max(count_b - count_ab, 0)
        remaining = total - (k11 + k12 + k21)
        k22 = max(remaining, 0)

        pmi_value = pmi(k11, count_a, count_b, total)
        llr_value = llr(k11, k12, k21, k22)
        asym = record.delta_a_given_b - record.delta_b_given_a

        if not math.isfinite(pmi_value) or not math.isfinite(llr_value):
            continue

        if math.isfinite(asym):
            asym_values.append(abs(asym))
            asym_stored: float | None = round(asym, 4)
        else:
            asym_stored = None

        pmi_values.append(pmi_value)
        llr_values.append(llr_value)
        debug_candidates.append((pmi_value, record.morph_left, record.morph_right, count_ab))

        rows_to_write.append(
            {
                "morph_left": record.morph_left,
                "morph_right": record.morph_right,
                "count_ab": count_ab,
                "count_a": count_a,
                "count_b": count_b,
                "pmi": round(pmi_value, 3),
                "llr": round(llr_value, 3),
                "asym_dep": asym_stored,
                "updated_at": timestamp,
            }
        )

    if not rows_to_write:
        logger.info(
            "No collocation rows met criteria", extra={"min_count": min_count, "total_pairs": len(pairs)}
        )
        return {
            "timestamp": timestamp,
            "pairs": 0,
            "avg_pmi": 0.0,
            "avg_llr": 0.0,
            "avg_asym": 0.0,
        }

    for batch in _batch_iterable(rows_to_write):
        upsert_rows(conn, "collocation_stats", batch)

    elapsed = time.perf_counter() - start
    avg_pmi = sum(pmi_values) / len(pmi_values) if pmi_values else 0.0
    avg_llr = sum(llr_values) / len(llr_values) if llr_values else 0.0
    avg_asym = sum(asym_values) / len(asym_values) if asym_values else 0.0

    logger.info(
        "Computed collocation statistics",
        extra={
            "pairs": len(rows_to_write),
            "min_count": min_count,
            "avg_pmi": round(avg_pmi, 3),
            "avg_llr": round(avg_llr, 3),
            "avg_asym": round(avg_asym, 3),
            "elapsed_s": round(elapsed, 2),
        },
    )

    debug_candidates.sort(key=lambda item: item[0], reverse=True)
    if debug_candidates:
        top_debug = [
            {
                "morph_left": left,
                "morph_right": right,
                "pmi": round(value, 3),
                "count_ab": count,
            }
            for value, left, right, count in debug_candidates[:5]
        ]
        logger.debug("Top PMI pairs: %s", top_debug)

    return {
        "timestamp": timestamp,
        "pairs": len(rows_to_write),
        "avg_pmi": avg_pmi,
        "avg_llr": avg_llr,
        "avg_asym": avg_asym,
    }


__all__ = ["compute_collocations"]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute collocation statistics")
    parser.add_argument("--db", required=True, help="SQLite database path")
    parser.add_argument("--min-count", type=int, default=5, help="Minimum joint count")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of pairs processed")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    try:
        ensure_analysis_tables(conn)
        compute_collocations(conn, min_count=args.min_count, limit=args.limit)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution path
    raise SystemExit(main())
