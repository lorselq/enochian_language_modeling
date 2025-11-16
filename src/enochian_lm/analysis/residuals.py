"""Residual clustering for morpheme semantic analysis."""
from __future__ import annotations

import argparse
import json
import logging
import math
from enochian_common.sqlite_bootstrap import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import List

try:  # pragma: no cover - optional dependency guard
    import numpy as _np
except Exception:  # pragma: no cover - fallback path when numpy missing
    _np = None

try:  # pragma: no cover - sklearn may be unavailable in minimal installs
    from sklearn.cluster import KMeans  # type: ignore
except Exception:  # pragma: no cover - fallback implementation will be used
    KMeans = None  # type: ignore

from ..utils.sql import ensure_analysis_tables
from ..utils.text import utcnow_iso

logger = logging.getLogger(__name__)

Epsilon = 1e-9


if _np is not None:  # pragma: no cover - simplify type hints when numpy available
    Vector = _np.ndarray
else:  # pragma: no cover - maintain compatibility when numpy missing
    Vector = List[float]


@dataclass(slots=True)
class _ClusterDetail:
    cluster_id: int
    size: int
    mean_sim: float
    examples: list[str]


def _ensure_numpy() -> None:
    if _np is None:
        raise RuntimeError("Residual clustering requires numpy to be installed")


def _table_has_normalized_residuals(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='residual_details'"
    ).fetchone()
    if row is None:
        return False
    columns = conn.execute("PRAGMA table_info(residual_details)").fetchall()
    return any(col[1] == "normalized" for col in columns)


def _load_morph_vectors(conn: sqlite3.Connection) -> dict[str, Vector]:
    _ensure_numpy()
    rows = conn.execute("SELECT morph, vector_json FROM morph_semantic_vectors").fetchall()
    vectors: dict[str, Vector] = {}
    for row in rows:
        try:
            data = json.loads(row["vector_json"])
        except json.JSONDecodeError:
            logger.warning("Skipping morph with invalid vector JSON", extra={"morph": row["morph"]})
            continue
        vectors[row["morph"]] = _np.array(data, dtype=float)
    return vectors


def _fetch_collocation_data(
    conn: sqlite3.Connection,
) -> tuple[dict[str, int], dict[str, list[tuple[str, float]]]]:
    counts: dict[str, int] = defaultdict(int)
    neighbors: dict[str, list[tuple[str, float]]] = defaultdict(list)
    rows = conn.execute(
        "SELECT morph_left, morph_right, pmi FROM collocation_stats"
    ).fetchall()
    for row in rows:
        left = row["morph_left"]
        right = row["morph_right"]
        counts[left] += 1
        counts[right] += 1
        pmi_value = row["pmi"]
        if pmi_value is None:
            continue
        try:
            score = float(pmi_value)
        except (TypeError, ValueError):
            continue
        neighbors[left].append((right, score))
        neighbors[right].append((left, score))
    return counts, neighbors


def _load_residuals_from_table(
    conn: sqlite3.Connection,
) -> dict[str, Vector]:
    _ensure_numpy()
    rows = conn.execute(
        "SELECT residual_span, normalized FROM residual_details WHERE normalized IS NOT NULL"
    ).fetchall()
    residuals: dict[str, Vector] = {}
    for row in rows:
        try:
            data = json.loads(row["normalized"])
        except (TypeError, json.JSONDecodeError):
            logger.debug(
                "Skipping residual with invalid normalized payload", extra={"span": row["residual_span"]}
            )
            continue
        vector = _np.array(data, dtype=float)
        residuals[row["residual_span"]] = vector
    return residuals


def _compute_residual_fallback(
    morph_vectors: dict[str, Vector],
    neighbors: dict[str, list[tuple[str, float]]],
    *,
    pmi_thresh: float,
) -> dict[str, Vector]:
    _ensure_numpy()
    residuals: dict[str, Vector] = {}
    for morph, vector in morph_vectors.items():
        neighbor_entries = [
            morph_vectors[neighbor]
            for neighbor, score in neighbors.get(morph, [])
            if score is not None and score >= pmi_thresh and neighbor in morph_vectors
        ]
        if neighbor_entries:
            mean_vector = _np.mean(_np.stack(neighbor_entries), axis=0)
            mean_vector = _np.nan_to_num(mean_vector, nan=0.0)
            residual = vector - mean_vector
        else:
            residual = vector.copy()
        residuals[morph] = residual
    return residuals


def _normalize_vectors(raw: dict[str, Vector]) -> dict[str, Vector]:
    _ensure_numpy()
    normalized: dict[str, Vector] = {}
    for morph, vector in raw.items():
        norm = float(_np.linalg.norm(vector))
        if norm <= 0 or not math.isfinite(norm):
            logger.debug("Skipping morph with zero residual norm", extra={"morph": morph})
            continue
        normalized[morph] = vector / max(norm, Epsilon)
    return normalized


def _effective_k(desired: int, available: int) -> int:
    if available <= 0:
        return 0
    return max(1, min(desired, available))


def _run_sklearn_kmeans(vectors: _np.ndarray, k: int) -> _np.ndarray:
    if KMeans is None:
        raise RuntimeError("scikit-learn is not available")
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    return model.fit_predict(vectors)


def _fallback_kmeans(vectors: _np.ndarray, k: int, *, max_iter: int = 50) -> _np.ndarray:
    rng = _np.random.default_rng(42)
    if k == 1:
        return _np.zeros(len(vectors), dtype=int)
    indices = rng.choice(len(vectors), size=k, replace=False)
    centroids = vectors[indices].copy()
    labels = _np.zeros(len(vectors), dtype=int)
    for _ in range(max_iter):
        centroids_norm = _np.linalg.norm(centroids, axis=1, keepdims=True)
        centroids_norm[centroids_norm == 0] = 1.0
        normalized_centroids = centroids / centroids_norm
        sims = vectors @ normalized_centroids.T
        labels = sims.argmax(axis=1)
        new_centroids = centroids.copy()
        for idx in range(k):
            mask = labels == idx
            if not mask.any():
                new_centroids[idx] = vectors[rng.integers(len(vectors))]
                continue
            cluster_vectors = vectors[mask]
            centroid = cluster_vectors.mean(axis=0)
            norm = _np.linalg.norm(centroid)
            if norm > 0:
                centroid /= norm
            new_centroids[idx] = centroid
        if _np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return labels


def _assign_clusters(vectors: _np.ndarray, labels: _np.ndarray) -> dict[int, list[int]]:
    clusters: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        clusters[int(label)].append(index)
    return clusters


def _summarize_clusters(
    names: list[str],
    vectors: _np.ndarray,
    clusters: dict[int, list[int]],
) -> tuple[list[dict[str, object]], list[_ClusterDetail], float]:
    details_for_db: list[dict[str, object]] = []
    table_details: list[_ClusterDetail] = []
    total_sim = 0.0
    total_count = 0
    timestamp = utcnow_iso()
    cluster_id = 1
    for label in sorted(clusters):
        members = clusters[label]
        if not members:
            continue
        member_vectors = vectors[members]
        centroid = member_vectors.mean(axis=0)
        centroid = _np.nan_to_num(centroid, nan=0.0)
        norm = float(_np.linalg.norm(centroid))
        if norm > 0:
            centroid /= norm
        else:
            centroid = _np.zeros_like(centroid)
        sims = member_vectors @ centroid
        sims = _np.nan_to_num(sims, nan=0.0).clip(-1.0, 1.0)
        sim_values = [float(x) for x in sims]
        total_sim += sum(sim_values)
        total_count += len(sim_values)
        membership_rows = []
        examples = []
        sorted_members = sorted(
            zip(members, sim_values), key=lambda item: item[1], reverse=True
        )
        for index, sim in sorted_members:
            morph = names[index]
            membership_rows.append(
                {
                    "residual_span": morph,
                    "cluster_id": cluster_id,
                    "sim_to_centroid": round(sim, 4),
                    "updated_at": timestamp,
                }
            )
            if len(examples) < 5:
                examples.append(morph)
        details_for_db.append(
            {
                "cluster_id": cluster_id,
                "centroid": centroid,
                "membership": membership_rows,
                "size": len(members),
                "updated_at": timestamp,
            }
        )
        mean_sim = sum(sim_values) / len(sim_values) if sim_values else 0.0
        table_details.append(
            _ClusterDetail(
                cluster_id=cluster_id,
                size=len(members),
                mean_sim=mean_sim,
                examples=examples,
            )
        )
        cluster_id += 1
    overall_mean = total_sim / total_count if total_count else 0.0
    return details_for_db, table_details, overall_mean


def _write_clusters(
    conn: sqlite3.Connection,
    clusters: list[dict[str, object]],
) -> None:
    timestamp = clusters[0]["updated_at"] if clusters else utcnow_iso()
    cluster_rows = [
        (
            cluster["cluster_id"],
            json.dumps([float(x) for x in cluster["centroid"].tolist()]),
            cluster["size"],
            cluster["updated_at"],
        )
        for cluster in clusters
    ]
    membership_rows: list[tuple[str, int, float, str]] = []
    for cluster in clusters:
        for member in cluster["membership"]:
            membership_rows.append(
                (
                    member["residual_span"],
                    member["cluster_id"],
                    member["sim_to_centroid"],
                    member["updated_at"],
                )
            )
    cursor = conn.cursor()
    try:
        cursor.execute("BEGIN")
        cursor.execute("DELETE FROM residual_cluster_membership")
        cursor.execute("DELETE FROM residual_clusters")
        cursor.executemany(
            "INSERT INTO residual_clusters (cluster_id, centroid_json, size, updated_at) VALUES (?, ?, ?, ?)",
            cluster_rows,
        )
        cursor.executemany(
            "INSERT INTO residual_cluster_membership (residual_span, cluster_id, sim_to_centroid, updated_at) VALUES (?, ?, ?, ?)",
            membership_rows,
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
    logger.info(
        "Persisted residual clusters", extra={"clusters": len(cluster_rows), "timestamp": timestamp}
    )


def cluster_residuals(
    conn: sqlite3.Connection,
    *,
    k: int = 10,
    min_df: int = 2,
    pmi_thresh: float = 0.0,
) -> dict[str, object]:
    """Cluster morph residual vectors using PMI-derived neighborhoods."""

    _ensure_numpy()
    start = time.perf_counter()
    morph_vectors = _load_morph_vectors(conn)
    if not morph_vectors:
        logger.info("No morph semantic vectors available; skipping residual clustering")
        return {
            "timestamp": utcnow_iso(),
            "clusters": 0,
            "mean_sim": 0.0,
            "sizes": [],
            "details": [],
            "morphs": 0,
        }

    counts, neighbors = _fetch_collocation_data(conn)
    candidate_morphs = {
        morph
        for morph, count in counts.items()
        if count >= min_df and morph in morph_vectors
    }
    if not candidate_morphs:
        logger.info("No candidate morphs met minimum document frequency", extra={"min_df": min_df})
        return {
            "timestamp": utcnow_iso(),
            "clusters": 0,
            "mean_sim": 0.0,
            "sizes": [],
            "details": [],
            "morphs": 0,
        }

    if _table_has_normalized_residuals(conn):
        residual_map = _load_residuals_from_table(conn)
        logger.info(
            "Loaded residual vectors from residual_details", extra={"count": len(residual_map)}
        )
    else:
        residual_map = _compute_residual_fallback(morph_vectors, neighbors, pmi_thresh=pmi_thresh)
        logger.info(
            "Computed fallback residual vectors", extra={"count": len(residual_map), "pmi_thresh": pmi_thresh}
        )

    filtered_residuals: dict[str, Vector] = {
        morph: residual_map[morph]
        for morph in candidate_morphs
        if morph in residual_map
    }

    normalized = _normalize_vectors(filtered_residuals)
    if not normalized:
        logger.info("No residual vectors remained after normalization; skipping clustering")
        return {
            "timestamp": utcnow_iso(),
            "clusters": 0,
            "mean_sim": 0.0,
            "sizes": [],
            "details": [],
            "morphs": 0,
        }

    names = sorted(normalized)
    vectors = _np.stack([normalized[name] for name in names])
    k_eff = _effective_k(k, len(names))
    if k_eff == 0:
        logger.info("Insufficient residual vectors for clustering")
        return {
            "timestamp": utcnow_iso(),
            "clusters": 0,
            "mean_sim": 0.0,
            "sizes": [],
            "details": [],
            "morphs": len(names),
        }

    if k_eff == 1:
        labels = _np.zeros(len(names), dtype=int)
    else:
        try:
            labels = _run_sklearn_kmeans(vectors, k_eff)
        except Exception as error:  # pragma: no cover - rare fallback
            logger.warning("Falling back to custom k-means", exc_info=False, extra={"error": str(error)})
            labels = _fallback_kmeans(vectors, k_eff)

    clusters = _assign_clusters(vectors, labels)
    details_for_db, table_details, overall_mean = _summarize_clusters(names, vectors, clusters)

    if details_for_db:
        _write_clusters(conn, details_for_db)

    elapsed = time.perf_counter() - start
    logger.info(
        "Residual clustering complete",
        extra={
            "clusters": len(table_details),
            "mean_sim": round(overall_mean, 4),
            "elapsed_s": round(elapsed, 2),
            "k": k,
            "min_df": min_df,
            "pmi_thresh": pmi_thresh,
            "morphs": len(names),
        },
    )

    centroid_norms = [
        float(_np.linalg.norm(cluster["centroid"])) for cluster in details_for_db
    ]
    logger.debug(
        "Centroid norms", extra={"norms": [round(value, 4) for value in centroid_norms]}
    )
    logger.debug(
        "Cluster examples",
        extra={
            "examples": {detail.cluster_id: detail.examples for detail in table_details}
        },
    )

    return {
        "timestamp": details_for_db[0]["updated_at"] if details_for_db else utcnow_iso(),
        "clusters": len(table_details),
        "mean_sim": overall_mean,
        "sizes": [detail.size for detail in table_details],
        "details": [
            {
                "cluster_id": detail.cluster_id,
                "size": detail.size,
                "mean_sim": detail.mean_sim,
                "examples": detail.examples,
            }
            for detail in table_details
        ],
        "morphs": len(names),
    }


__all__ = ["cluster_residuals"]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cluster residual morpheme vectors")
    parser.add_argument("--db", required=True, help="SQLite database path")
    parser.add_argument("--k", type=int, default=10, help="Number of clusters")
    parser.add_argument("--min-df", type=int, default=2, help="Minimum collocation frequency")
    parser.add_argument("--pmi-thresh", type=float, default=0.0, help="PMI threshold for neighbors")
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
        cluster_residuals(conn, k=args.k, min_df=args.min_df, pmi_thresh=args.pmi_thresh)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":  # pragma: no cover - script invocation guard
    raise SystemExit(main())
