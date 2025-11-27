"""Leave-one-out attribution analysis for composite tokens."""
from __future__ import annotations

import json
import logging
import math
from enochian_lm.common.sqlite_bootstrap import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, Iterator, List, Tuple

from tqdm import tqdm

try:  # pragma: no cover - optional dependency
    import numpy as _np
except Exception:  # pragma: no cover - fallback when numpy is unavailable
    _np = None

if _np is not None:  # pragma: no cover - executed when numpy present
    from numpy.typing import NDArray

from ..utils.sql import upsert_rows
from ..utils.text import utcnow_iso

logger = logging.getLogger(__name__)

Epsilon = 1e-9


@dataclass(slots=True)
class VectorData:
    """Container for a vector and its L2 norm."""

    values: "VectorLike"
    norm: float


if _np is not None:  # pragma: no cover - executed when numpy present
    VectorLike = NDArray[float]
else:  # pragma: no cover - keep type checkers satisfied when numpy missing
    VectorLike = List[float]


def _parse_vector(json_blob: str) -> VectorLike:
    """Parse a JSON vector string into an array or list of floats."""

    data = json.loads(json_blob)
    if _np is not None:
        return _np.array(data, dtype=float)
    return [float(x) for x in data]


def _compute_dot(a: VectorLike, b: VectorLike) -> float:
    if _np is not None:
        return float(_np.dot(a, b))
    return float(sum(x * y for x, y in zip(a, b)))


def _compute_norm(vec: VectorLike) -> float:
    if _np is not None:
        return float(_np.linalg.norm(vec))
    return math.sqrt(sum(x * x for x in vec))


def _subtract(a: VectorLike, b: VectorLike) -> VectorLike:
    if _np is not None:
        return a - b
    return [x - y for x, y in zip(a, b)]


def compute_cosine(
    vec_a: VectorLike,
    vec_b: VectorLike,
    *,
    norm_a: float | None = None,
    norm_b: float | None = None,
) -> float:
    """Compute cosine similarity between *vec_a* and *vec_b*."""

    norm_a = norm_a if norm_a is not None else _compute_norm(vec_a)
    norm_b = norm_b if norm_b is not None else _compute_norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return _compute_dot(vec_a, vec_b) / (norm_a * norm_b + Epsilon)


def _canonicalize_pair(
    morph_a: str,
    morph_b: str,
    delta_a_given_b: float,
    delta_b_given_a: float,
) -> Tuple[str, str, float, float]:
    if morph_a <= morph_b:
        return morph_a, morph_b, delta_a_given_b, delta_b_given_a
    return morph_b, morph_a, delta_b_given_a, delta_a_given_b


def _unique_order_preserving(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _batch_iterable(rows: list[dict[str, object]], batch_size: int) -> Iterator[list[dict[str, object]]]:
    for index in range(0, len(rows), batch_size):
        yield rows[index : index + batch_size]


@dataclass
class _PairStats:
    sum_delta_a_given_b: float = 0.0
    sum_delta_b_given_a: float = 0.0
    count: int = 0


def _load_morph_vectors(conn: sqlite3.Connection) -> Dict[str, VectorData]:
    rows = conn.execute(
        "SELECT morph, vector_json, l2_norm FROM morph_semantic_vectors"
    ).fetchall()
    morph_vectors: dict[str, VectorData] = {}
    for row in rows:
        vector = _parse_vector(row["vector_json"])
        norm = float(row["l2_norm"]) if row["l2_norm"] is not None else _compute_norm(vector)
        morph_vectors[row["morph"]] = VectorData(values=vector, norm=norm)
    return morph_vectors


def _iter_composites(conn: sqlite3.Connection, limit: int | None) -> Iterable[sqlite3.Row]:
    query = (
        "SELECT token, pred_vector_json, used_morphs_json "
        "FROM composite_reconstruction ORDER BY id"
    )
    params: tuple[object, ...] = ()
    if limit is not None:
        query += " LIMIT ?"
        params = (limit,)
    return conn.execute(query, params)


def run_leave_one_out(conn: sqlite3.Connection, limit: int | None = None) -> dict[str, float | int | str]:
    """Run leave-one-out attribution analysis.

    Returns a summary dictionary with counts and average absolute delta values.
    """

    start = time.perf_counter()
    morph_vectors = _load_morph_vectors(conn)
    if not morph_vectors:
        logger.warning("No morph semantic vectors found; skipping attribution analysis")
        return {
            "timestamp": utcnow_iso(),
            "composites": 0,
            "pairs": 0,
            "unique_pairs": 0,
            "avg_abs_delta": 0.0,
        }

    summary_timestamp = utcnow_iso()
    missing_morphs: set[str] = set()
    pair_stats: dict[Tuple[str, str], _PairStats] = defaultdict(_PairStats)
    total_abs_delta = 0.0
    delta_count = 0
    composites_processed = 0
    pairs_processed = 0

    total_composites = conn.execute(
        "SELECT COUNT(*) FROM composite_reconstruction"
    ).fetchone()[0]
    if limit is not None:
        total_composites = min(int(total_composites), int(limit))

    progress = tqdm(
        _iter_composites(conn, limit),
        desc="Attribution",
        unit="composite",
        total=total_composites or None,
    )

    for row in progress:
        try:
            used_morphs_raw = json.loads(row["used_morphs_json"])
        except json.JSONDecodeError:
            logger.error(
                "Invalid JSON in used_morphs", extra={"token": row["token"]}
            )
            continue

        if not isinstance(used_morphs_raw, list):
            logger.warning(
                "Skipping composite with non-list used_morphs", extra={"token": row["token"]}
            )
            continue

        used_morphs = _unique_order_preserving(str(m) for m in used_morphs_raw)
        if len(used_morphs) < 2:
            continue

        try:
            pred_vector = _parse_vector(row["pred_vector_json"])
        except json.JSONDecodeError:
            logger.error(
                "Invalid JSON in pred_vector", extra={"token": row["token"]}
            )
            continue

        pred_norm = _compute_norm(pred_vector)
        composite_pairs = 0

        for morph_a, morph_b in combinations(used_morphs, 2):
            vector_a = morph_vectors.get(morph_a)
            vector_b = morph_vectors.get(morph_b)
            if vector_a is None:
                if morph_a not in missing_morphs:
                    missing_morphs.add(morph_a)
                    logger.warning("Missing vector for morph", extra={"morph": morph_a})
                continue
            if vector_b is None:
                if morph_b not in missing_morphs:
                    missing_morphs.add(morph_b)
                    logger.warning("Missing vector for morph", extra={"morph": morph_b})
                continue

            sim_full_a = compute_cosine(pred_vector, vector_a.values, norm_a=pred_norm, norm_b=vector_a.norm)
            minus_b = _subtract(pred_vector, vector_b.values)
            sim_minus_b = compute_cosine(minus_b, vector_a.values, norm_b=vector_a.norm)
            delta_a_given_b = sim_full_a - sim_minus_b

            sim_full_b = compute_cosine(pred_vector, vector_b.values, norm_a=pred_norm, norm_b=vector_b.norm)
            minus_a = _subtract(pred_vector, vector_a.values)
            sim_minus_a = compute_cosine(minus_a, vector_b.values, norm_b=vector_b.norm)
            delta_b_given_a = sim_full_b - sim_minus_a

            canon_a, canon_b, delta_ab, delta_ba = _canonicalize_pair(
                morph_a, morph_b, delta_a_given_b, delta_b_given_a
            )
            stats = pair_stats[(canon_a, canon_b)]
            stats.sum_delta_a_given_b += delta_ab
            stats.sum_delta_b_given_a += delta_ba
            stats.count += 1

            total_abs_delta += abs(delta_a_given_b) + abs(delta_b_given_a)
            delta_count += 2
            pairs_processed += 1
            composite_pairs += 1

        if composite_pairs:
            composites_processed += 1

    progress.close()

    if not pair_stats:
        logger.info(
            "No eligible composites found for attribution",
            extra={"processed": composites_processed, "pairs": 0},
        )
        return {
            "timestamp": summary_timestamp,
            "composites": composites_processed,
            "pairs": 0,
            "unique_pairs": 0,
            "avg_abs_delta": 0.0,
        }

    avg_abs_delta = total_abs_delta / delta_count if delta_count else 0.0

    rows_to_write: list[dict[str, object]] = []
    debug_entries: list[tuple[str, str, str, float]] = []
    for (morph_a, morph_b), stats in pair_stats.items():
        mean_a = stats.sum_delta_a_given_b / stats.count if stats.count else 0.0
        mean_b = stats.sum_delta_b_given_a / stats.count if stats.count else 0.0
        rows_to_write.append(
            {
                "morph_a": morph_a,
                "morph_b": morph_b,
                "delta_a_given_b": round(mean_a, 4),
                "delta_b_given_a": round(mean_b, 4),
                "n_tokens": stats.count,
                "updated_at": summary_timestamp,
            }
        )
        debug_entries.append(("delta_a_given_b", morph_a, morph_b, mean_a))
        debug_entries.append(("delta_b_given_a", morph_a, morph_b, mean_b))

    debug_entries.sort(key=lambda item: item[3])
    negative_examples = debug_entries[:3]
    positive_examples = list(reversed(debug_entries[-3:]))

    if negative_examples:
        logger.debug("Top negative Δ examples: %s", negative_examples)
    if positive_examples:
        logger.debug("Top positive Δ examples: %s", positive_examples)

    for batch in _batch_iterable(rows_to_write, batch_size=200):
        upsert_rows(conn, "attribution_marginals", batch)

    elapsed = time.perf_counter() - start
    logger.info(
        "Completed leave-one-out attribution",
        extra={
            "composites": composites_processed,
            "pairs": pairs_processed,
            "unique_pairs": len(pair_stats),
            "avg_abs_delta": round(avg_abs_delta, 4),
            "elapsed_s": round(elapsed, 2),
        },
    )

    return {
        "timestamp": summary_timestamp,
        "composites": composites_processed,
        "pairs": pairs_processed,
        "unique_pairs": len(pair_stats),
        "avg_abs_delta": avg_abs_delta,
    }


__all__ = ["run_leave_one_out", "compute_cosine"]
