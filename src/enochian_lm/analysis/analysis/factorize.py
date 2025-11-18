"""Morpheme semantic factorization via ridge regression."""
from __future__ import annotations

import csv
import json
import logging
import math
import os
from enochian_lm.common.sqlite_bootstrap import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from tempfile import NamedTemporaryFile
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge

from ..utils.sql import upsert_rows
from ..utils.text import normalize, utcnow_iso

logger = logging.getLogger(__name__)

EPSILON = 1e-9


@dataclass(slots=True)
class TokenRecord:
    """Container for a token participating in factorization."""

    token: str
    gloss: str
    morphs: list[str]


def _atomic_write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8", newline="") as tmp:
        writer = csv.writer(tmp)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp.name, path)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tmp:
        json.dump(payload, tmp, indent=2, ensure_ascii=False)
        tmp.write("\n")
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp.name, path)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    query = "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?"
    return conn.execute(query, (table,)).fetchone() is not None


def _load_residual_definitions(conn: sqlite3.Connection) -> dict[str, str]:
    if not _table_exists(conn, "residual_details"):
        return {}

    try:
        rows = conn.execute(
            "SELECT normalized, definition FROM residual_details WHERE definition IS NOT NULL"
        ).fetchall()
    except sqlite3.OperationalError:  # pragma: no cover - schema variance
        logger.debug("residual_details table present without expected columns; skipping definitions")
        return {}

    definitions: dict[str, str] = {}
    for row in rows:
        normalized_token = row["normalized"]
        definition = row["definition"]
        if isinstance(normalized_token, str) and isinstance(definition, str) and definition.strip():
            definitions[normalized_token] = definition
    return definitions


def _resolve_gloss(
    token: str,
    gold_gloss: str | None,
    definitions: dict[str, str],
) -> str | None:
    if isinstance(gold_gloss, str) and gold_gloss.strip():
        return gold_gloss
    if token in definitions:
        return definitions[token]
    return token if token else None


def _parse_morphs(raw: str | None) -> list[str]:
    if raw is None:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("Invalid used_morphs_json", extra={"raw": raw})
        return []
    if isinstance(data, list):
        return [str(item) for item in data if isinstance(item, (str, int, float))]
    logger.warning("Skipping non-list morph payload", extra={"payload": raw})
    return []


def _collect_tokens(
    conn: sqlite3.Connection,
    *,
    limit: int | None,
    min_token_morphs: int,
    definitions: dict[str, str],
) -> list[TokenRecord]:
    query = (
        "SELECT token, gold_gloss, used_morphs_json FROM composite_reconstruction ORDER BY id"
    )
    rows = conn.execute(query)
    records: list[TokenRecord] = []

    for row in rows:
        morphs = _parse_morphs(row["used_morphs_json"])
        if len(morphs) < min_token_morphs:
            continue
        gloss = _resolve_gloss(row["token"], row["gold_gloss"], definitions)
        if gloss is None or not gloss.strip():
            continue
        records.append(
            TokenRecord(
                token=row["token"],
                gloss=normalize(gloss),
                morphs=morphs,
            )
        )
        if limit is not None and len(records) >= limit:
            break
    return records


def _build_morph_vocab(records: Sequence[TokenRecord], min_morph_count: int) -> list[str]:
    counter: Counter[str] = Counter()
    for record in records:
        counter.update(record.morphs)
    vocab = [morph for morph, count in counter.items() if count >= min_morph_count]
    vocab.sort()
    return vocab


def _filter_records(records: Sequence[TokenRecord], vocab: set[str], min_token_morphs: int) -> list[TokenRecord]:
    filtered: list[TokenRecord] = []
    for record in records:
        kept = [m for m in record.morphs if m in vocab]
        if len(kept) < min_token_morphs:
            continue
        filtered.append(TokenRecord(token=record.token, gloss=record.gloss, morphs=kept))
    return filtered


def _vectorize_glosses(records: Sequence[TokenRecord], embed: str) -> tuple[NDArray[np.float64], int, str]:
    documents = [record.gloss for record in records]
    if not documents:
        return np.empty((0, 0), dtype=np.float64), 0, ""

    if embed == "gloss-chars":
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=1,
            lowercase=True,
            max_features=6000,
        )
        matrix = vectorizer.fit_transform(documents)
    elif embed == "hashing-words":
        vectorizer = HashingVectorizer(
            n_features=4096,
            alternate_sign=False,
            norm="l2",
        )
        matrix = vectorizer.transform(documents)
    else:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            lowercase=True,
            max_features=5000,
        )
        matrix = vectorizer.fit_transform(documents)
    dense = matrix.astype(np.float64).toarray()
    dim = dense.shape[1]
    return dense, dim, vectorizer.__class__.__name__


def _build_design_matrix(
    records: Sequence[TokenRecord],
    vocab: Sequence[str],
    *,
    row_norm: bool,
) -> tuple[sparse.csr_matrix, dict[str, int]]:
    morph_index = {morph: idx for idx, morph in enumerate(vocab)}
    if not morph_index:
        return sparse.csr_matrix((0, 0), dtype=np.float64), morph_index

    row_indices: list[int] = []
    col_indices: list[int] = []
    data: list[float] = []
    for row_idx, record in enumerate(records):
        morphs = record.morphs
        counts: Counter[str] = Counter(morphs)
        active_items = [(morph_index[m], count) for m, count in counts.items() if m in morph_index]
        if not active_items:
            continue
        norm_factor = 1.0
        total = sum(count for _, count in active_items)
        if row_norm and total > 0:
            norm_factor = 1.0 / math.sqrt(total)
        for morph_col, count in active_items:
            row_indices.append(row_idx)
            col_indices.append(morph_col)
            data.append(float(count) * norm_factor)

    matrix = sparse.csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(records), len(vocab)),
        dtype=np.float64,
    )
    return matrix, morph_index


def _compute_error(actual: NDArray[np.float64], predicted: NDArray[np.float64], metric: str) -> float:
    if metric == "cosine":
        norm_actual = float(np.linalg.norm(actual))
        norm_pred = float(np.linalg.norm(predicted))
        if norm_actual == 0.0 or norm_pred == 0.0:
            return 1.0
        cosine = float(np.dot(actual, predicted) / (norm_actual * norm_pred + EPSILON))
        return 1.0 - max(min(cosine, 1.0), -1.0)
    diff = actual - predicted
    return float(np.mean(np.square(diff)))


def _round_vector(values: Sequence[float]) -> list[float]:
    return [round(float(value), 4) for value in values]


def _maybe_align_to_clusters(
    conn: sqlite3.Connection,
    morph_vectors: dict[str, NDArray[np.float64]],
    out_dir: Path,
) -> None:
    if not _table_exists(conn, "residual_clusters") or not _table_exists(
        conn, "residual_cluster_membership"
    ):
        logger.debug("Residual cluster tables not present; skipping alignment CSV")
        return

    rows = conn.execute(
        "SELECT cluster_id, centroid_json FROM residual_clusters ORDER BY cluster_id"
    ).fetchall()
    if not rows:
        logger.debug("Residual clusters empty; skipping alignment CSV")
        return

    centroids: list[tuple[int, NDArray[np.float64]]] = []
    for row in rows:
        try:
            centroid_vec = np.array(json.loads(row["centroid_json"]), dtype=np.float64)
        except json.JSONDecodeError:
            logger.error("Invalid centroid_json", extra={"cluster_id": row["cluster_id"]})
            continue
        centroids.append((int(row["cluster_id"]), centroid_vec))

    if not centroids:
        return

    def best_alignment(vector: NDArray[np.float64]) -> tuple[int, float]:
        best_cluster = -1
        best_sim = -1.0
        vector_norm = float(np.linalg.norm(vector))
        if vector_norm == 0.0:
            return best_cluster, best_sim
        for cluster_id, centroid in centroids:
            centroid_norm = float(np.linalg.norm(centroid))
            if centroid_norm == 0.0:
                continue
            cosine = float(
                np.dot(vector, centroid) / (vector_norm * centroid_norm + EPSILON)
            )
            if cosine > best_sim:
                best_cluster = cluster_id
                best_sim = cosine
        return best_cluster, best_sim

    alignment_rows: list[tuple[object, ...]] = []
    for morph, vector in morph_vectors.items():
        cluster_id, similarity = best_alignment(vector)
        if cluster_id == -1:
            continue
        alignment_rows.append((morph, cluster_id, round(float(similarity), 4)))

    if alignment_rows:
        alignment_path = Path(out_dir) / "alignment.csv"
        _atomic_write_csv(alignment_path, ("morph", "best_cluster_id", "best_sim"), alignment_rows)
        logger.info(
            "Wrote alignment CSV", extra={"rows": len(alignment_rows), "path": str(alignment_path)}
        )


def factorize_morphemes(
    conn: sqlite3.Connection,
    out_dir: str,
    *,
    alpha: float = 1.0,
    embed: str = "gloss-words",
    min_morph_count: int = 3,
    min_token_morphs: int = 2,
    row_norm: bool = False,
    metric: str = "mse",
    limit: int | None = None,
) -> dict[str, object]:
    """Factorize morpheme semantics using ridge regression."""

    supported_metrics = {"mse", "cosine"}
    if metric not in supported_metrics:
        raise ValueError(f"Unsupported metric '{metric}'. Choose from {sorted(supported_metrics)}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    definitions = _load_residual_definitions(conn)
    records = _collect_tokens(
        conn,
        limit=limit,
        min_token_morphs=min_token_morphs,
        definitions=definitions,
    )

    if not records:
        logger.warning("No eligible tokens for factorization; exiting")
        return {
            "tokens": 0,
            "morphs": 0,
            "dim": 0,
            "alpha": alpha,
            "metric": metric,
            "mean_error": 0.0,
            "median_error": 0.0,
            "timestamp": utcnow_iso(),
            "embed": embed,
        }

    vocab = _build_morph_vocab(records, min_morph_count)
    if not vocab:
        logger.warning("No morphs met the minimum count threshold; exiting")
        return {
            "tokens": 0,
            "morphs": 0,
            "dim": 0,
            "alpha": alpha,
            "metric": metric,
            "mean_error": 0.0,
            "median_error": 0.0,
            "timestamp": utcnow_iso(),
            "embed": embed,
        }

    filtered_records = _filter_records(records, set(vocab), min_token_morphs)
    if not filtered_records:
        logger.warning("No tokens retained after morph filtering; exiting")
        return {
            "tokens": 0,
            "morphs": len(vocab),
            "dim": 0,
            "alpha": alpha,
            "metric": metric,
            "mean_error": 0.0,
            "median_error": 0.0,
            "timestamp": utcnow_iso(),
            "embed": embed,
        }

    gloss_matrix, dim, vectorizer_name = _vectorize_glosses(filtered_records, embed)
    if dim == 0 or gloss_matrix.size == 0:
        logger.warning("Gloss embedding produced empty matrix; exiting")
        return {
            "tokens": len(filtered_records),
            "morphs": len(vocab),
            "dim": 0,
            "alpha": alpha,
            "metric": metric,
            "mean_error": 0.0,
            "median_error": 0.0,
            "timestamp": utcnow_iso(),
            "embed": embed,
        }

    design_matrix, morph_index = _build_design_matrix(
        filtered_records,
        vocab,
        row_norm=row_norm,
    )
    if design_matrix.shape[0] == 0 or design_matrix.shape[1] == 0:
        logger.warning("Design matrix empty; exiting")
        return {
            "tokens": len(filtered_records),
            "morphs": len(vocab),
            "dim": dim,
            "alpha": alpha,
            "metric": metric,
            "mean_error": 0.0,
            "median_error": 0.0,
            "timestamp": utcnow_iso(),
            "embed": embed,
        }

    logger.info(
        "Fitting ridge regression",
        extra={
            "tokens": design_matrix.shape[0],
            "morphs": design_matrix.shape[1],
            "dim": dim,
            "alpha": alpha,
            "embed": embed,
            "vectorizer": vectorizer_name,
        },
    )

    model = Ridge(alpha=alpha, fit_intercept=False, solver="auto")
    model.fit(design_matrix, gloss_matrix)

    coefficients = np.asarray(model.coef_, dtype=np.float64).T  # shape (morphs, dim)
    predictions = model.predict(design_matrix)

    timestamp = utcnow_iso()
    morph_vectors: dict[str, NDArray[np.float64]] = {}
    morph_rows: list[dict[str, object]] = []
    for morph, index in morph_index.items():
        vector = coefficients[index]
        morph_vectors[morph] = vector
        morph_rows.append(
            {
                "morph": morph,
                "vector_json": json.dumps(_round_vector(vector)),
                "l2_norm": round(float(np.linalg.norm(vector)), 4),
                "updated_at": timestamp,
            }
        )

    upsert_rows(conn, "morph_semantic_vectors", morph_rows)

    reconstruction_rows: list[tuple[str, str, float, str]] = []
    errors: list[float] = []
    token_errors: list[tuple[str, float]] = []

    for idx, record in enumerate(filtered_records):
        predicted = predictions[idx]
        actual = gloss_matrix[idx]
        error = _compute_error(actual, predicted, metric)
        errors.append(error)
        token_errors.append((record.token, error))
        rounded_vector = json.dumps(_round_vector(predicted))
        reconstruction_rows.append(
            (
                rounded_vector,
                round(error, 4),
                timestamp,
                record.token,
            )
        )

    update_sql = (
        "UPDATE composite_reconstruction "
        "SET pred_vector_json = ?, recon_error = ?, updated_at = ? "
        "WHERE token = ?"
    )
    conn.executemany(update_sql, reconstruction_rows)

    top_norms = sorted(
        ((row["l2_norm"], row["morph"]) for row in morph_rows), reverse=True
    )[:10]
    if top_norms:
        logger.info(
            "Top morph norms",
            extra={"examples": [(morph, norm) for norm, morph in top_norms]},
        )

    _maybe_align_to_clusters(conn, morph_vectors, out_path)

    _atomic_write_csv(
        out_path / "morph_vectors.csv",
        ("morph", "l2_norm", "vector_json", "updated_at"),
        ((row["morph"], row["l2_norm"], row["vector_json"], timestamp) for row in morph_rows),
    )

    reconstruction_csv_rows = (
        (record.token, round(errors[idx], 4), len(record.morphs))
        for idx, record in enumerate(filtered_records)
    )
    _atomic_write_csv(
        out_path / "reconstruction.csv",
        ("token", "recon_error", "n_morphs"),
        reconstruction_csv_rows,
    )

    mean_error = float(mean(errors)) if errors else 0.0
    median_error = float(median(errors)) if errors else 0.0

    summary = {
        "tokens": len(filtered_records),
        "morphs": len(morph_rows),
        "dim": dim,
        "alpha": alpha,
        "metric": metric,
        "mean_error": round(mean_error, 6),
        "median_error": round(median_error, 6),
        "timestamp": timestamp,
        "embed": embed,
    }
    _atomic_write_json(out_path / "summary.json", summary)

    sorted_errors = sorted(token_errors, key=lambda item: item[1])
    if sorted_errors:
        best = sorted_errors[:3]
        worst = sorted_errors[-3:]
        logger.debug("Best reconstructed tokens", extra={"tokens": best})
        logger.debug("Worst reconstructed tokens", extra={"tokens": worst})

    logger.info(
        "Factorization completed",
        extra={
            "tokens": summary["tokens"],
            "morphs": summary["morphs"],
            "dim": summary["dim"],
            "alpha": alpha,
            "metric": metric,
            "mean_error": summary["mean_error"],
            "median_error": summary["median_error"],
            "embed": embed,
        },
    )

    conn.commit()

    return summary


__all__ = ["factorize_morphemes"]


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    import argparse

    parser = argparse.ArgumentParser(description="Factorize morph semantics")
    parser.add_argument("--db", required=True, help="Path to SQLite database")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--embed",
        choices=["gloss-words", "gloss-chars", "hashing-words"],
        default="gloss-words",
    )
    parser.add_argument("--min-morph-count", type=int, default=3)
    parser.add_argument("--min-token-morphs", type=int, default=2)
    parser.add_argument("--row-norm", action="store_true")
    parser.add_argument("--metric", choices=["mse", "cosine"], default="mse")
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    connection = sqlite3.connect(args.db)
    connection.row_factory = sqlite3.Row
    try:
        summary = factorize_morphemes(
            connection,
            args.out,
            alpha=args.alpha,
            embed=args.embed,
            min_morph_count=args.min_morph_count,
            min_token_morphs=args.min_token_morphs,
            row_norm=args.row_norm,
            metric=args.metric,
            limit=args.limit,
        )
        print(json.dumps(summary, indent=2))
    finally:
        connection.close()
