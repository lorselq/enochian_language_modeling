from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from enochian_lm.root_extraction.utils.candidate_finder import MorphemeCandidateFinder
from enochian_lm.root_extraction.utils.config import get_config_paths
from enochian_lm.root_extraction.utils.dictionary_loader import load_dictionary
from enochian_lm.root_extraction.utils.residual_analysis import (
    exclude_root_segments,
    summarize_residuals,
)
from enochian_lm.root_extraction.utils.sql import connect_sqlite

logger = logging.getLogger(__name__)


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_words(conn: sqlite3.Connection, cluster_id: int) -> list[tuple[str, str]]:
    rows = conn.execute(
        "SELECT DISTINCT source_word, definition FROM raw_defs WHERE cluster_id = ?",
        (cluster_id,),
    ).fetchall()
    return [
        (
            str(row["source_word"] or "").strip(),
            str(row["definition"] or "").strip(),
        )
        for row in rows
    ]


def _target_run_id(conn: sqlite3.Connection, run_id: str | None) -> str:
    if run_id:
        return str(run_id)
    row = conn.execute("SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1").fetchone()
    if not row:
        raise ValueError("No runs found in database; cannot refresh residual details")
    return str(row[0])


def refresh_residual_details(db_path: Path, *, run_id: str | None = None) -> tuple[int, int]:
    """Recompute ``residual_details`` rows without rerunning the full pipeline."""

    paths = get_config_paths()
    candidate_finder = MorphemeCandidateFinder(
        ngram_db_path=paths["ngram_index"],
        fasttext_model_path=paths["model_output"],
        dictionary_entries=load_dictionary(paths["dictionary"]),
    )

    conn = connect_sqlite(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        target_run = _target_run_id(conn, run_id)
        clusters = conn.execute(
            """
            SELECT cluster_id, ngram
            FROM clusters
            WHERE run_id = ?
            ORDER BY cluster_id
            """,
            (target_run,),
        ).fetchall()

        if not clusters:
            logger.info("No clusters found for run; nothing to refresh", extra={"run_id": target_run})
            return (0, 0)

        updated_clusters = 0
        detail_rows_total = 0

        for cluster in clusters:
            cluster_id = int(cluster["cluster_id"])
            root = str(cluster["ngram"] or "").strip().lower()
            words = _load_words(conn, cluster_id)
            if not words:
                continue

            residual_inputs: list[dict[str, object]] = []
            for word, definition in words:
                norm = word.lower()
                breakdown = None
                candidates = candidate_finder.find_candidates(norm, top_k=1)
                if candidates:
                    breakdown = candidates[0].get("breakdown")
                if not breakdown:
                    uncovered = (
                        [{"span": [0, len(norm)], "text": norm}] if norm else []
                    )
                    breakdown = {
                        "segments": [],
                        "uncovered": uncovered,
                        "coverage_ratio": 0.0,
                        "residual_ratio": 1.0 if uncovered else 0.0,
                    }
                else:
                    breakdown = exclude_root_segments(breakdown, root_norm=root, target=norm)

                residual_inputs.append(
                    {
                        "word": word,
                        "normalized": norm,
                        "definition": definition,
                        "breakdown": breakdown,
                    }
                )

            if not residual_inputs:
                continue

            residual_report = summarize_residuals(root=root, analyses=residual_inputs)
            details = residual_report.get("word_details") or []

            rows_to_insert = []
            for detail in details:
                normalized = (
                    str(detail.get("normalized") or detail.get("word") or "").strip()
                )
                definition = str(detail.get("definition") or "").strip()
                coverage_ratio = _safe_float(detail.get("coverage_ratio"))
                residual_ratio = _safe_float(detail.get("residual_ratio"))
                avg_confidence = _safe_float(detail.get("avg_confidence"))
                uncovered = detail.get("uncovered") or []
                low_conf = detail.get("low_conf_segments") or []

                rows_to_insert.append(
                    (
                        cluster_id,
                        normalized,
                        definition,
                        coverage_ratio,
                        residual_ratio,
                        avg_confidence,
                        json.dumps(uncovered, ensure_ascii=False),
                        json.dumps(low_conf, ensure_ascii=False),
                    )
                )

            with conn:
                conn.execute(
                    "DELETE FROM residual_details WHERE cluster_id = ?", (cluster_id,)
                )
                if rows_to_insert:
                    conn.executemany(
                        """
                        INSERT INTO residual_details (
                            cluster_id,
                            normalized,
                            definition,
                            coverage_ratio,
                            residual_ratio,
                            avg_confidence,
                            uncovered_json,
                            low_conf_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        rows_to_insert,
                    )

                conn.execute(
                    """
                    UPDATE clusters
                    SET residual_explained = ?,
                        residual_ratio = ?,
                        residual_headline = ?,
                        residual_focus_prompt = ?
                    WHERE cluster_id = ?
                    """,
                    (
                        _safe_float(residual_report.get("explained_ratio")),
                        _safe_float(residual_report.get("residual_ratio")),
                        residual_report.get("headline"),
                        residual_report.get("focus_prompt"),
                        cluster_id,
                    ),
                )

            updated_clusters += 1
            detail_rows_total += len(rows_to_insert)

        logger.info(
            "Refreshed residual_details",
            extra={
                "run_id": target_run,
                "clusters": updated_clusters,
                "detail_rows": detail_rows_total,
            },
        )
        return updated_clusters, detail_rows_total
    finally:
        conn.close()

