from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from tqdm import tqdm

from enochian_lm.analysis.utils.sql import connect_sqlite
from enochian_lm.root_extraction.utils.candidate_finder import MorphemeCandidateFinder
from enochian_lm.root_extraction.utils.config import get_config_paths
from enochian_lm.root_extraction.utils.dictionary_loader import load_dictionary
from enochian_lm.root_extraction.utils.residual_analysis import (
    exclude_root_segments,
    summarize_residuals,
)

logger = logging.getLogger(__name__)

COMPOSITE_TOP_K = 5
MIN_MULTI_SEGMENTS = 2


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _recompute_coverage(
    *,
    segments: list[dict[str, object]],
    target: str,
    root_norm: str,
    base_breakdown: dict[str, object],
    cover_root: bool = False,
) -> dict[str, object]:
    """
    Recalculate coverage/uncovered spans for a pruned segmentation path.

    ``cover_root`` optionally marks the root span as explained so that suffix
    residuals surface even when the only remaining coverage was a full-word
    self-match.
    """

    total_len = len(target)
    mask = [False] * total_len

    if cover_root and root_norm:
        for idx in range(min(len(root_norm), total_len)):
            mask[idx] = True

    normalized_segments: list[dict[str, object]] = []
    for segment in segments:
        span = segment.get("span") or [segment.get("start", 0), segment.get("end", 0)]
        start = int(span[0] or 0)
        end = int(span[1] or start)
        start = max(0, min(total_len, start))
        end = max(start, min(total_len, end))

        for idx in range(start, end):
            mask[idx] = True

        normalized = dict(segment)
        normalized["span"] = [start, end]
        normalized_segments.append(normalized)

    uncovered: list[dict[str, object]] = []
    idx = 0
    while idx < total_len:
        if mask[idx]:
            idx += 1
            continue
        start = idx
        while idx < total_len and not mask[idx]:
            idx += 1
        uncovered.append({"span": [start, idx], "text": target[start:idx]})

    coverage_ratio = sum(1 for flag in mask if flag) / total_len if total_len else 0.0
    residual_ratio = max(0.0, 1.0 - coverage_ratio)

    updated = dict(base_breakdown)
    updated["segments"] = normalized_segments
    updated["uncovered"] = uncovered
    updated["coverage_ratio"] = coverage_ratio
    updated["residual_ratio"] = residual_ratio
    return updated


def _apply_root_prefix_residual_fallback(
    breakdown: dict[str, object], *, root_norm: str, target: str
) -> dict[str, object]:
    """
    Drop self-covering segments when root-prefixed words still show full coverage.

    If, after stripping root segments, the breakdown still reports complete
    coverage and the word begins with the root, remove any full-word self-match
    and recompute coverage treating the root span as explained. This surfaces
    suffix fragments (e.g., ``PSAD`` in ``NAZPSAD``) for residual analysis.
    """

    total_len = len(target)
    if not target or not root_norm or total_len == 0:
        return breakdown

    coverage_ratio = float(breakdown.get("coverage_ratio") or 0.0)
    uncovered = breakdown.get("uncovered") or []
    if uncovered or coverage_ratio < 1.0:
        return breakdown

    target_norm = target.lower()
    root_norm = root_norm.lower()

    if not target_norm.startswith(root_norm):
        return breakdown

    segments = breakdown.get("segments") or []
    filtered_segments = [
        seg
        for seg in segments
        if str(seg.get("canonical", "")).strip().lower() != target_norm
    ]

    return _recompute_coverage(
        segments=filtered_segments,
        target=target,
        root_norm=root_norm,
        base_breakdown=breakdown,
        cover_root=True,
    )


def _is_self_composite(breakdown: dict[str, object]) -> bool:
    segments = breakdown.get("segments") if isinstance(breakdown, dict) else None
    if not segments or len(segments) < MIN_MULTI_SEGMENTS:
        return False
    canonicals = [
        str(seg.get("canonical") or "").lower()
        for seg in segments
        if isinstance(seg, dict)
    ]
    canonicals = [c for c in canonicals if c]
    return len(canonicals) >= MIN_MULTI_SEGMENTS and len(set(canonicals)) == 1


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
        token_stats = {
            "considered": 0,
            "multi_selected": 0,
            "self_pair_dropped": 0,
            "fallback_used": 0,
            "missing_breakdown": 0,
        }

        for cluster in tqdm(clusters, desc="Refreshing clusters", unit="cluster"):
            cluster_id = int(cluster["cluster_id"])
            root = str(cluster["ngram"] or "").strip().lower()
            words = _load_words(conn, cluster_id)
            if not words:
                continue

            residual_inputs: list[dict[str, object]] = []
            for word, definition in tqdm(
                words,
                desc=f"Cluster {cluster_id} words",
                unit="word",
                leave=False,
            ):
                norm = word.lower()
                token_stats["considered"] += 1
                breakdown = None
                candidates = candidate_finder.find_candidates(
                    norm,
                    top_k=COMPOSITE_TOP_K,
                    min_cos_sim=candidate_finder.min_candidate_cos_sim,
                )

                multi_segment_breakdowns: list[dict[str, object]] = []
                single_breakdowns: list[dict[str, object]] = []
                for cand in candidates:
                    bd = cand.get("breakdown") if isinstance(cand, dict) else None
                    if not bd:
                        token_stats["missing_breakdown"] += 1
                        continue
                    if _is_self_composite(bd):
                        token_stats["self_pair_dropped"] += 1
                        continue
                    segments = bd.get("segments") if isinstance(bd, dict) else []
                    if segments and len(segments) >= MIN_MULTI_SEGMENTS:
                        multi_segment_breakdowns.append(bd)
                    else:
                        single_breakdowns.append(bd)

                if multi_segment_breakdowns:
                    breakdown = multi_segment_breakdowns[0]
                    token_stats["multi_selected"] += 1
                elif single_breakdowns:
                    breakdown = single_breakdowns[0]
                if not breakdown:
                    token_stats["fallback_used"] += 1
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
                    breakdown = _apply_root_prefix_residual_fallback(
                        breakdown, root_norm=root, target=norm
                    )

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
                residual_span = str(detail.get("word") or "").strip()
                normalized = str(detail.get("normalized") or residual_span).strip()
                definition = str(detail.get("definition") or "").strip()
                coverage_ratio = _safe_float(detail.get("coverage_ratio"))
                residual_ratio = _safe_float(detail.get("residual_ratio"))
                avg_confidence = _safe_float(detail.get("avg_confidence"))
                uncovered = detail.get("uncovered") or []
                low_conf = detail.get("low_conf_segments") or []

                rows_to_insert.append(
                    (
                        cluster_id,
                        residual_span or normalized,
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
                            residual_span,
                            normalized,
                            definition,
                            coverage_ratio,
                            residual_ratio,
                            avg_confidence,
                            uncovered_json,
                            low_conf_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                "tokens_considered": token_stats["considered"],
                "multi_selected": token_stats["multi_selected"],
                "self_pair_dropped": token_stats["self_pair_dropped"],
                "fallback_used": token_stats["fallback_used"],
                "missing_breakdown": token_stats["missing_breakdown"],
            },
        )
        return updated_clusters, detail_rows_total
    finally:
        conn.close()

