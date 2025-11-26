"""Command line interface for Enochian language modeling utilities."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
from collections import Counter
import unicodedata
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Iterable, Iterator, Sequence

from enochian_lm.root_extraction.scripts import init_insights_db
from enochian_lm.root_extraction.utils.embeddings import get_fasttext_model
from enochian_lm.root_extraction.utils.preanalysis import execute_preanalysis
from enochian_lm.root_extraction.utils.residual_refresh import refresh_residual_details

from .analysis.attribution import run_leave_one_out
from .analysis.colloc import compute_collocations
from .analysis.factorize import factorize_morphemes
from .analysis.residuals import cluster_residuals
from .report.pipeline_summary import generate_pipeline_report
from .utils.sql import connect_sqlite, ensure_analysis_tables
from .utils.text import set_global_seeds, utcnow_iso

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT)


def _validate_input_file(path: Path, description: str) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{description} '{path}' does not exist or is not a file")
    with path.open("r", encoding="utf-8") as handle:
        handle.read(128)


def _atomic_write(path: Path, write_fn: Callable[[NamedTemporaryFile], None], *, mode: str = "w", newline: str | None = "\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode=mode, delete=False, dir=str(path.parent), encoding="utf-8", newline=newline) as tmp:
        write_fn(tmp)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp.name, path)


def _write_csv_header(path: Path, header: Iterable[str]) -> None:
    def writer(tmp: NamedTemporaryFile) -> None:
        csv_writer = csv.writer(tmp)
        csv_writer.writerow(list(header))

    _atomic_write(path, writer, newline="")


def _write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    def writer(tmp: NamedTemporaryFile) -> None:
        csv_writer = csv.writer(tmp)
        csv_writer.writerow(list(header))
        for row in rows:
            csv_writer.writerow(list(row))

    _atomic_write(path, writer, newline="")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    def writer(tmp: NamedTemporaryFile) -> None:
        json.dump(payload, tmp, indent=2, ensure_ascii=False)
        tmp.write("\n")

    _atomic_write(path, writer)


def _normalize_run_ids(raw_run_ids: str | Sequence[str] | None) -> list[str]:
    if raw_run_ids is None:
        return []

    if isinstance(raw_run_ids, str):
        candidates = [raw_run_ids]
    else:
        candidates = list(raw_run_ids)

    normalized = [str(run).strip() for run in candidates if str(run).strip()]
    if not normalized:
        raise ValueError("No run ids provided; specify at least one --run-id value")

    return normalized


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()  # numpy scalar compatibility
        except Exception:
            pass
    if isinstance(value, (int, float, str)) or value is None:
        return value
    return str(value)


def _iter_json_lines(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                logger.warning(
                    "Skipping malformed JSONL line",
                    extra={"path": str(path), "line": line_no},
                )
                continue
            if isinstance(payload, dict):
                yield payload
            else:
                logger.debug(
                    "Skipping non-dict JSON payload",
                    extra={"path": str(path), "line": line_no},
                )


def _coerce_token(payload: dict[str, object]) -> str | None:
    for key in ("token", "word", "surface", "text", "span"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _coerce_optional_str(payload: dict[str, object], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _coerce_float_list(raw: object) -> list[float] | None:
    candidate: object = raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        if text.startswith("[") and text.endswith("]"):
            try:
                candidate = json.loads(text)
            except json.JSONDecodeError:
                return None
        else:
            parts = [segment.strip() for segment in text.split(",") if segment.strip()]
            try:
                return [float(part) for part in parts]
            except ValueError:
                return None
    if not isinstance(candidate, (list, tuple)):
        return None
    values: list[float] = []
    for item in candidate:
        try:
            values.append(float(item))
        except (TypeError, ValueError):
            return None
    return values


def _round_vector(values: Sequence[float], places: int = 6) -> list[float]:
    factor = 10**places
    return [math.floor(val * factor + 0.5) / factor for val in values]


def _coerce_sequence(raw: object) -> list[str]:
    data: object = raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = [segment.strip() for segment in text.split(",") if segment.strip()]
        else:
            data = [segment.strip() for segment in text.replace("|", ",").split(",") if segment.strip()]
    if isinstance(data, (set, tuple)):
        iterable = list(data)
    elif isinstance(data, list):
        iterable = data
    else:
        return []
    results: list[str] = []
    for item in iterable:
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned:
                results.append(cleaned)
        elif isinstance(item, (int, float)):
            results.append(str(item))
    return results


def _vector_norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in values))


def _ingest_composite_parses(conn, path: Path) -> int:
    timestamp = utcnow_iso()
    rows: list[tuple[str, str | None, str, float, str, str, str]] = []
    for payload in _iter_json_lines(path):
        token = _coerce_token(payload)
        vector = _coerce_float_list(
            payload.get("pred_vector")
            or payload.get("vector")
            or payload.get("predicted_vector")
            or payload.get("embedding")
        )
        morphs = _coerce_sequence(
            payload.get("used_morphs")
            or payload.get("morphs")
            or payload.get("morphemes")
            or payload.get("segments")
            or payload.get("components")
        )
        if not token or vector is None:
            logger.debug(
                "Skipping parse entry lacking token/vector",
                extra={"payload_keys": list(payload.keys())},
            )
            continue
        gold_gloss = _coerce_optional_str(payload, ("gold_gloss", "gloss", "definition"))
        error_raw = payload.get("recon_error") or payload.get("residual") or payload.get("error")
        try:
            recon_error = float(error_raw) if error_raw is not None else 0.0
        except (TypeError, ValueError):
            recon_error = 0.0
        rows.append(
            (
                token,
                gold_gloss,
                json.dumps(_round_vector(vector)),
                round(recon_error, 4),
                json.dumps(morphs),
                str(payload.get("vector_source") or "fasttext"),
                timestamp,
            )
        )

    conn.execute("DELETE FROM composite_reconstruction")
    if rows:
        conn.executemany(
            """
            INSERT INTO composite_reconstruction (
              token, gold_gloss, pred_vector_json, recon_error, used_morphs_json, vector_source, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    conn.commit()
    logger.info(
        "Ingested composite parses",
        extra={"rows": len(rows), "path": str(path)},
    )
    return len(rows)


def _round_vector(values: Sequence[float], places: int = 6) -> list[float]:
    factor = 10**places
    return [math.floor(float(val) * factor + 0.5) / factor for val in values]


def _vector_norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in values))


_FASTTEXT_TOKEN_RE = re.compile(r"[^a-z0-9]+")


def _normalize_for_fasttext(token: str) -> str:
    """Mirror FastText training cleanup for short fragments.

    Training lowercases tokens and strips punctuation/diacritics (via
    ``simple_preprocess(..., deacc=True)`` for definitions). We apply the same
    policy to uncovered fragments so that their embeddings are derived from the
    same character set seen during training.
    """

    lowered = unicodedata.normalize("NFKD", str(token or "").strip().lower())
    without_diacritics = "".join(ch for ch in lowered if not unicodedata.combining(ch))
    return _FASTTEXT_TOKEN_RE.sub("", without_diacritics)


def _parse_uncovered_fragments(payload: str | Sequence[object] | None) -> list[str]:
    if payload is None:
        return []

    data: object = payload
    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning(
                "Skipping invalid uncovered_json payload", extra={"payload": payload}
            )
            return []

    if not isinstance(data, list):
        return []

    fragments: list[str] = []
    for item in data:
        if isinstance(item, dict):
            value = item.get("text") or item.get("residual") or item.get("fragment")
            value = _normalize_for_fasttext(value)
            if value:
                fragments.append(value)
        elif isinstance(item, (str, int, float)):
            text = _normalize_for_fasttext(item)
            if text:
                fragments.append(text)
    return fragments


def _backfill_composite_reconstruction(
    conn,
    *,
    run_id: str | None = None,
) -> tuple[int, str]:
    ensure_analysis_tables(conn)

    target_run_id: str | None = run_id
    if target_run_id is None:
        row = conn.execute("SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1").fetchone()
        if row:
            target_run_id = str(row[0])
    if target_run_id is None:
        raise ValueError("No runs found in database; cannot backfill composites")

    query = """
        SELECT
            rd.normalized,
            rd.definition,
            rd.residual_ratio,
            rd.uncovered_json,
            c.residual_ratio AS cluster_residual,
            c.glossator_def,
            rc.centroid_json
        FROM residual_details rd
        JOIN clusters c ON c.cluster_id = rd.cluster_id
        LEFT JOIN residual_clusters rc ON rc.cluster_id = rd.cluster_id
        WHERE c.run_id = ?
        ORDER BY rd.residual_id
    """
    rows = conn.execute(query, (target_run_id,)).fetchall()
    if not rows:
        logger.warning(
            "No residual_details rows found for run; nothing to backfill",
            extra={"run_id": target_run_id},
        )
        return 0, target_run_id

    ft_model = get_fasttext_model()
    timestamp = utcnow_iso()
    composite_rows: list[tuple[str, str | None, str, float, str, str | None, str]] = []
    morph_rows: dict[str, tuple[str, float, str]] = {}
    vector_source_counts: Counter[str] = Counter()

    def _fasttext_vector(token: str) -> list[float]:
        candidates = []
        normalized = _normalize_for_fasttext(token)
        if normalized:
            candidates.append(normalized)
        cleaned = str(token or "").strip()
        if cleaned:
            candidates.append(cleaned)

        last_candidate = ""
        for candidate in candidates:
            last_candidate = candidate
            try:
                return list(ft_model.wv[candidate])
            except KeyError:
                continue

        if last_candidate:
            return list(ft_model.get_word_vector(last_candidate))

        return [0.0] * ft_model.vector_size

    for row in rows:
        token = str(row["normalized"] or "").strip()
        if not token:
            continue
        gloss = str(row["definition"] or row["glossator_def"] or "").strip()
        recon_error = row["residual_ratio"]
        if recon_error is None:
            recon_error = row["cluster_residual"] if row["cluster_residual"] is not None else 0.0
        morph_breakdown = _parse_uncovered_fragments(row["uncovered_json"])
        for key in ("morph_breakdown_json", "morph_breakdown"):
            if morph_breakdown:
                break
            if key in row.keys():
                morph_breakdown = _parse_uncovered_fragments(row[key])

        accepted_root: str | None = None
        for key in ("accepted_root", "root"):
            if key in row.keys():
                candidate_root = str(row[key] or "").strip()
                if candidate_root:
                    accepted_root = candidate_root
                    break

        if morph_breakdown:
            morphs = [m for m in (_normalize_for_fasttext(m) for m in morph_breakdown) if m]
        elif accepted_root:
            normalized_root = _normalize_for_fasttext(accepted_root) or accepted_root
            morphs = [normalized_root]
        else:
            morphs = []

        token_vector = _fasttext_vector(token)
        vector_source = "fasttext"
        vector_source_counts[vector_source] += 1
        composite_rows.append(
            (
                token,
                gloss if gloss else None,
                json.dumps(_round_vector(token_vector)),
                round(float(recon_error or 0.0), 4),
                json.dumps(morphs),
                vector_source,
                timestamp,
            )
        )

        for morph in morphs:
            if morph in morph_rows:
                continue
            morph_vec = _fasttext_vector(morph)
            morph_rows[morph] = (
                json.dumps(_round_vector(morph_vec)),
                round(_vector_norm(morph_vec), 4),
                timestamp,
            )

    tokens = [(row[0],) for row in composite_rows]
    with conn:
        conn.executemany("DELETE FROM composite_reconstruction WHERE token = ?", tokens)
        conn.executemany(
            """
            INSERT INTO composite_reconstruction (
              token, gold_gloss, pred_vector_json, recon_error, used_morphs_json, vector_source, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            composite_rows,
        )

        if morph_rows:
            conn.executemany(
                """
                INSERT INTO morph_semantic_vectors (morph, vector_json, l2_norm, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(morph) DO UPDATE SET
                  vector_json=excluded.vector_json,
                  l2_norm=excluded.l2_norm,
                  updated_at=excluded.updated_at
                """,
                [
                    (morph, vector_json, l2_norm, ts)
                    for morph, (vector_json, l2_norm, ts) in morph_rows.items()
                ],
            )

    logger.info(
        "Backfilled composite_reconstruction from residual_details",
        extra={
            "rows": len(composite_rows),
            "morphs": len(morph_rows),
            "run_id": target_run_id,
            "vector_sources": dict(vector_source_counts),
        },
    )

    try:
        zero_morphs, single_morph, multi_morphs = conn.execute(
            """
            SELECT
                SUM(json_array_length(used_morphs_json) = 0),
                SUM(json_array_length(used_morphs_json) = 1),
                SUM(json_array_length(used_morphs_json) > 1)
            FROM composite_reconstruction
            """
        ).fetchone()
        logger.info(
            "Composite reconstruction morph counts",
            extra={
                "zero_morphs": zero_morphs,
                "single_morph": single_morph,
                "multi_morphs": multi_morphs,
            },
        )
    except Exception:
        logger.warning("Failed to summarize morph counts", exc_info=True)
    return len(composite_rows), target_run_id


def _export_attribution_csv(db_path: Path, out_path: Path) -> None:
    conn = connect_sqlite(str(db_path))
    try:
        query = (
            "SELECT morph_a, morph_b, delta_a_given_b, delta_b_given_a, n_tokens, updated_at "
            "FROM attribution_marginals ORDER BY n_tokens DESC, morph_a, morph_b"
        )
        rows = conn.execute(query).fetchall()
    finally:
        conn.close()

    header = ("morph_a", "morph_b", "delta_a_given_b", "delta_b_given_a", "n_tokens", "updated_at")
    data = (
        (
            row["morph_a"],
            row["morph_b"],
            float(row["delta_a_given_b"]),
            float(row["delta_b_given_a"]),
            int(row["n_tokens"]),
            row["updated_at"],
        )
        for row in rows
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(out_path, header, data)


def _export_collocation_csv(db_path: Path, out_path: Path) -> None:
    conn = connect_sqlite(str(db_path))
    try:
        query = (
            "SELECT morph_left, morph_right, count_ab, count_a, count_b, pmi, llr, asym_dep, updated_at "
            "FROM collocation_stats ORDER BY count_ab DESC, morph_left, morph_right"
        )
        rows = conn.execute(query).fetchall()
    finally:
        conn.close()

    header = (
        "morph_left",
        "morph_right",
        "count_ab",
        "count_a",
        "count_b",
        "pmi",
        "llr",
        "asym_dep",
        "updated_at",
    )
    data = (
        (
            row["morph_left"],
            row["morph_right"],
            int(row["count_ab"]),
            int(row["count_a"]),
            int(row["count_b"]),
            row["pmi"],
            row["llr"],
            row["asym_dep"],
            row["updated_at"],
        )
        for row in rows
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(out_path, header, data)


def _run_composite_backfill(args: argparse.Namespace) -> None:
    run_ids = _normalize_run_ids(args.run_id) if args.run_id is not None else []
    if not run_ids:
        run_ids = [None]

    conn = connect_sqlite(str(args.db_path))
    total_rows = 0
    try:
        ensure_analysis_tables(conn)
        for run_id in run_ids:
            rows, resolved_run_id = _backfill_composite_reconstruction(
                conn, run_id=run_id
            )
            total_rows += rows
            print(
                f"Backfilled {rows} composite reconstructions for run {resolved_run_id}."
            )
    finally:
        conn.close()

    if total_rows == 0:
        raise ValueError("No composite reconstructions were backfilled")


def _run_attrib_loo(args: argparse.Namespace) -> None:
    db_path = Path(args.db_path)
    logger.info(
        "Running leave-one-out attribution",
        extra={"db": str(db_path), "limit": args.limit},
    )
    conn = connect_sqlite(str(db_path))
    try:
        ensure_analysis_tables(conn)
        summary = run_leave_one_out(conn, limit=args.limit)
    finally:
        conn.close()

    logger.info(
        "Leave-one-out summary",
        extra={
            "composites": summary.get("composites", 0),
            "pairs": summary.get("pairs", 0),
            "unique_pairs": summary.get("unique_pairs", 0),
            "avg_abs_delta": round(float(summary.get("avg_abs_delta", 0.0)), 4),
        },
    )

    print(
        "[{}] Processed {} composites → {} pairs ({} unique) (avg |Δ| = {:.3f})".format(
            summary.get("timestamp", utcnow_iso()),
            summary.get("composites", 0),
            summary.get("pairs", 0),
            summary.get("unique_pairs", 0),
            float(summary.get("avg_abs_delta", 0.0)),
        )
    )


def _run_colloc(args: argparse.Namespace) -> None:
    db_path = Path(args.db_path)
    limit = getattr(args, "limit", None)
    min_count = getattr(args, "min_count", 5)

    logger.info(
        "Running collocation statistics",
        extra={"db": str(db_path), "min_count": min_count, "limit": limit},
    )

    conn = connect_sqlite(str(db_path))
    try:
        ensure_analysis_tables(conn)
        summary = compute_collocations(conn, min_count=min_count, limit=limit)
    finally:
        conn.close()

    pairs = int(summary.get("pairs", 0))
    avg_pmi = float(summary.get("avg_pmi", 0.0))
    avg_llr = float(summary.get("avg_llr", 0.0))
    avg_asym = float(summary.get("avg_asym", 0.0))

    logger.info(
        "Collocation summary",
        extra={
            "pairs": pairs,
            "avg_pmi": round(avg_pmi, 3),
            "avg_llr": round(avg_llr, 3),
            "avg_asym": round(avg_asym, 3),
        },
    )

    print(
        "[{}] Computed {} collocation pairs (avg PMI = {:.3f}, avg LLR = {:.3f}, avg |asym| = {:.3f})".format(
            summary.get("timestamp", utcnow_iso()),
            pairs,
            avg_pmi,
            avg_llr,
            avg_asym,
        )
    )


def _print_cluster_summary(summary: dict[str, object]) -> None:
    timestamp = summary.get("timestamp", utcnow_iso())
    clusters = int(summary.get("clusters", 0))
    mean_sim = float(summary.get("mean_sim", 0.0))
    morphs = int(summary.get("morphs", 0))
    print(
        "[{}] Clustered {} morphs into {} clusters (mean sim = {:.3f})".format(
            timestamp,
            morphs,
            clusters,
            mean_sim,
        )
    )

    details = summary.get("details", [])
    if not isinstance(details, list) or not details:
        print("No clusters to display.")
        return

    header = f"{'Cluster':>7} | {'Size':>5} | {'MeanSim':>7} | Example Morphs"
    divider = "-" * len(header)
    print(divider)
    print(header)
    print(divider)
    for detail in details:
        cluster_id = int(detail.get("cluster_id", 0))
        size = int(detail.get("size", 0))
        cluster_mean = float(detail.get("mean_sim", 0.0))
        examples = detail.get("examples", [])
        if isinstance(examples, list):
            example_str = ", ".join(str(item) for item in examples)
        else:
            example_str = str(examples)
        print(
            f"{cluster_id:>7} | {size:>5} | {cluster_mean:>7.3f} | {example_str}"
        )


def _run_residual_cluster(args: argparse.Namespace) -> dict[str, object]:
    db_path = Path(args.db_path)
    logger.info(
        "Running residual clustering",
        extra={
            "db": str(db_path),
            "k": args.k,
            "min_df": args.min_df,
            "pmi_thresh": args.pmi_thresh,
        },
    )

    conn = connect_sqlite(str(db_path))
    try:
        ensure_analysis_tables(conn)
        summary = cluster_residuals(
            conn,
            k=args.k,
            min_df=args.min_df,
            pmi_thresh=args.pmi_thresh,
        )
    finally:
        conn.close()

    logger.info(
        "Residual clustering summary",
        extra={
            "clusters": summary.get("clusters", 0),
            "mean_sim": round(float(summary.get("mean_sim", 0.0)), 4),
            "morphs": summary.get("morphs", 0),
        },
    )
    
    _print_cluster_summary(summary)
    return summary

def _run_residual_refresh(args: argparse.Namespace) -> tuple[int, int]:
    db_path = Path(args.db_path)
    run_ids = _normalize_run_ids(args.run_id) if args.run_id is not None else []
    if not run_ids:
        run_ids = [None]

    total_clusters = 0
    total_rows = 0
    for run_id in run_ids:
        logger.info(
            "Refreshing residual_details without rerunning pipeline",
            extra={"db": str(db_path), "run_id": run_id},
        )

        clusters, detail_rows = refresh_residual_details(db_path, run_id=run_id)
        total_clusters += clusters
        total_rows += detail_rows

        logger.info(
            "Residual refresh completed",
            extra={
                "run_id": run_id,
                "clusters": clusters,
                "detail_rows": detail_rows,
            },
        )
        run_label = run_id if run_id is not None else "latest"
        print(
            f"[{run_label}] Refreshed {int(clusters)} clusters and {int(detail_rows)} detail rows."
        )

    return total_clusters, total_rows


def _run_morph_factorize(args: argparse.Namespace) -> None:
    db_path = Path(args.db_path)
    out_dir = Path(args.out)
    logger.info(
        "Running morph semantic factorization",
        extra={
            "db": str(db_path),
            "out": str(out_dir),
            "alpha": args.alpha,
            "embed": args.embed,
            "max_features": args.max_features,
            "min_morph_count": args.min_morph_count,
            "min_token_morphs": args.min_token_morphs,
            "row_norm": args.row_norm,
            "metric": args.metric,
            "limit": args.limit,
        },
    )

    conn = connect_sqlite(str(db_path))
    try:
        ensure_analysis_tables(conn)
        summary = factorize_morphemes(
            conn,
            str(out_dir),
            alpha=args.alpha,
            embed=args.embed,
            max_features=args.max_features,
            min_morph_count=args.min_morph_count,
            min_token_morphs=args.min_token_morphs,
            row_norm=args.row_norm,
            metric=args.metric,
            limit=args.limit,
        )
    finally:
        conn.close()

    tokens = int(summary.get("tokens", 0))
    morphs = int(summary.get("morphs", 0))
    dim = int(summary.get("dim", 0))
    mean_error = float(summary.get("mean_error", 0.0))
    median_error = float(summary.get("median_error", 0.0))

    logger.info(
        "Morph factorization summary",
        extra={
            "tokens": tokens,
            "morphs": morphs,
            "dim": dim,
            "alpha": summary.get("alpha", args.alpha),
            "metric": summary.get("metric", args.metric),
            "mean_error": mean_error,
            "median_error": median_error,
        },
    )

    written_files = [
        "morph_vectors.csv",
        "reconstruction.csv",
        "summary.json",
    ]
    alignment_path = out_dir / "alignment.csv"
    if alignment_path.exists():
        written_files.append("alignment.csv")

    print(
        "[{}] Factorized {} tokens over {} morphs (D={}) | alpha={} | metric={}".format(
            summary.get("timestamp", utcnow_iso()),
            tokens,
            morphs,
            dim,
            summary.get("alpha", args.alpha),
            summary.get("metric", args.metric),
        )
    )
    print(
        "mean recon_error={:.4f}, median={:.4f} | wrote: {}".format(
            mean_error,
            median_error,
            ", ".join(written_files),
        )
    )


def _run_report_pipeline(args: argparse.Namespace) -> None:
    db_path = Path(args.db_path)
    out_dir = Path(args.out) if args.out else Path("runs") / f"{utcnow_iso().replace(':', '').replace('-', '')}_pipeline"
    baseline = args.baseline

    logger.info(
        "Generating pipeline report",
        extra={
            "db": str(db_path),
            "out": str(out_dir),
            "baseline": baseline,
        },
    )

    conn = connect_sqlite(str(db_path))
    try:
        summary = generate_pipeline_report(
            conn,
            str(out_dir),
            db_path=str(db_path),
            baseline_path=baseline,
        )
    finally:
        conn.close()

    metadata = summary.get("metadata", {})
    timestamp = metadata.get("timestamp", utcnow_iso())

    coverage = summary.get("coverage", {})
    attrib = summary.get("attribution", {})
    clusters = summary.get("residual_clusters", {})
    factor = summary.get("factorization", {})

    coverage_ratio = coverage.get("coverage_ratio_mean")
    residual_ratio = coverage.get("residual_ratio_mean")
    avg_conf = coverage.get("avg_confidence")
    avg_delta = attrib.get("avg_abs_delta")
    cluster_count = clusters.get("clusters")
    mean_sim = clusters.get("mean_sim")
    mean_error = factor.get("mean_error")
    median_error = factor.get("median_error")

    logger.info(
        "Pipeline report summary",
        extra={
            "coverage_ratio_mean": coverage_ratio,
            "residual_ratio_mean": residual_ratio,
            "avg_confidence": avg_conf,
            "avg_delta": avg_delta,
            "clusters": cluster_count,
            "mean_sim": mean_sim,
            "mean_error": mean_error,
            "median_error": median_error,
        },
    )

    def _fmt(value: float | None, digits: int = 4) -> str:
        return "N/A" if value is None else f"{float(value):.{digits}f}"

    print(
        "[{}] Coverage mean = {} | Residual mean = {} | Avg |Δ| = {}".format(
            timestamp,
            _fmt(coverage_ratio),
            _fmt(residual_ratio),
            _fmt(avg_delta),
        )
    )
    print(
        "Residual clusters: {} (mean sim = {}) | Morph factorization mean error = {} (median = {})".format(
            int(cluster_count or 0),
            _fmt(mean_sim, 3),
            _fmt(mean_error),
            _fmt(median_error),
        )
    )
    print(
        "Report written to {}".format(out_dir.joinpath("pipeline_report.html"))
    )


def _run_preanalyze(args: argparse.Namespace) -> None:
    db_path = Path(args.db_path)
    run_ids = _normalize_run_ids(args.run_id) if args.run_id is not None else []
    if not run_ids:
        run_ids = [None]

    for run_id in run_ids:
        result = execute_preanalysis(
            db_path=db_path,
            stage=args.stage,
            trusted_path=args.trusted,
            run_id=run_id,
            refresh=args.refresh,
        )

        stage = result.get("stage")
        resolved_run = result.get("run_id")
        pre_id = result.get("preanalysis_id")
        created = result.get("created")
        trusted_count = result.get("trusted_count")
        snapshots = result.get("snapshots", [])

        status = "created" if created else "reused"
        print(
            f"[{stage}] Pre-analysis {status} for run {resolved_run} (preanalysis_id={pre_id}; trusted={trusted_count}).",
        )

        for snap in snapshots:
            if not isinstance(snap, dict):
                continue
            ngram = snap.get("ngram")
            occurrences = snap.get("occurrences")
            sample = snap.get("sample") or []
            preview_items: list[str] = []
            for item in sample[:3]:
                if not isinstance(item, dict):
                    continue
                canonical = item.get("canonical")
                gloss = item.get("gloss")
                if canonical and gloss:
                    preview_items.append(f"{canonical}:{gloss}")
                elif canonical:
                    preview_items.append(str(canonical))
            preview = f" → {', '.join(preview_items)}" if preview_items else ""
            print(f"- {ngram}: occurrences={occurrences}{preview}")

def _run_analyze_all(args: argparse.Namespace) -> None:
    parses_path = Path(args.parses) if args.parses else None
    attrib_out = Path(args.attrib_out)
    colloc_out = Path(args.colloc_out)
    residual_out = Path(args.residual_out)

    attrib_out.parent.mkdir(parents=True, exist_ok=True)
    colloc_out.parent.mkdir(parents=True, exist_ok=True)
    residual_out.parent.mkdir(parents=True, exist_ok=True)

    conn = connect_sqlite(str(args.db_path))
    try:
        ensure_analysis_tables(conn)
        for table in (
            "attribution_marginals",
            "collocation_stats",
            "residual_clusters",
            "residual_cluster_membership",
        ):
            conn.execute(f"DELETE FROM {table}")
        conn.commit()

        existing_composites = conn.execute(
            "SELECT COUNT(*) FROM composite_reconstruction"
        ).fetchone()[0]
        reuse_parses = bool(args.reuse_db_parses) and existing_composites > 0
        if reuse_parses:
            logger.info(
                "Reusing composite parses from database",
                extra={"rows": existing_composites},
            )
            ingested_tokens = int(existing_composites)
        else:
            if parses_path is None:
                raise ValueError(
                    "--parses is required unless --reuse-db-parses is used with existing composite data"
                )
            _validate_input_file(parses_path, "Parses file")
            ingested_tokens = _ingest_composite_parses(conn, parses_path)
        morph_vector_count = conn.execute(
            "SELECT COUNT(*) FROM morph_semantic_vectors"
        ).fetchone()[0]
    finally:
        conn.close()

    if ingested_tokens == 0:
        raise ValueError(
            "No composite parses were ingested; check the --parses file contents."
        )
    if morph_vector_count == 0:
        raise ValueError(
            "No morph semantic vectors found in the database; populate the morph_semantic_vectors table before running analyze all."
        )

    combined_args = argparse.Namespace(db_path=args.db_path, limit=None)
    _run_attrib_loo(combined_args)
    _export_attribution_csv(args.db_path, attrib_out)

    combined_args = argparse.Namespace(
        db_path=args.db_path,
        min_count=args.min_count,
        limit=None,
    )
    _run_colloc(combined_args)
    _export_collocation_csv(args.db_path, colloc_out)

    combined_args = argparse.Namespace(
        db_path=args.db_path,
        k=args.k,
        min_df=args.min_df,
        pmi_thresh=args.pmi_thresh,
    )
    residual_summary = _run_residual_cluster(combined_args)
    sanitized_summary = _json_safe(residual_summary)
    if not isinstance(sanitized_summary, dict):
        sanitized_summary = {"result": sanitized_summary}
    _write_json(residual_out, sanitized_summary)

    combined_args = argparse.Namespace(
        db_path=args.db_path,
        out=args.morph_out,
        alpha=args.alpha,
        embed=args.embed,
        max_features=args.max_features,
        min_morph_count=args.min_morph_count,
        min_token_morphs=args.min_token_morphs,
        row_norm=args.row_norm,
        metric=args.metric,
        limit=None,
    )
    _run_morph_factorize(combined_args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="enlm",
        description="Enochian language modeling CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db",
        default=(
            "src/enochian_lm/root_extraction/interpretation/"
            "revised_solo_analysis_derived_definitions.sqlite3"
        ),
        help="Database path",
    )
    parser.add_argument("--seed", type=int, default=93, help="Global seed")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", required=True)

    pre_parser = subparsers.add_parser("preanalyze", help="Pre-analysis safeguards")
    pre_parser.add_argument(
        "--stage",
        choices=["initial", "subsequent"],
        default="initial",
        help="Which pre-analysis stage to run",
    )
    pre_parser.add_argument(
        "--run-id",
        nargs="+",
        metavar="RUN_ID",
        help=(
            "One or more existing translation run ids (auto-selects latest when omitted)"
        ),
    )
    pre_parser.add_argument(
        "--trusted",
        help="Optional path to a JSON list of trusted n-grams",
    )
    pre_parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh snapshots even if a record already exists",
    )
    pre_parser.set_defaults(handler=_run_preanalyze)

    attrib_parser = subparsers.add_parser("attrib", help="Attribution tooling")
    attrib_subparsers = attrib_parser.add_subparsers(dest="attrib_command", required=True)
    attrib_loo = attrib_subparsers.add_parser("loo", help="Leave-one-out attribution")
    attrib_loo.add_argument(
        "--limit", type=int, default=None, help="Limit number of composites processed"
    )
    attrib_loo.set_defaults(handler=_run_attrib_loo)

    colloc = subparsers.add_parser("colloc", help="Collocation statistics")
    colloc.add_argument("--min-count", type=int, default=2, help="Minimum joint count")
    colloc.add_argument("--limit", type=int, default=None, help="Limit number of pairs processed")
    colloc.set_defaults(handler=_run_colloc)

    residual = subparsers.add_parser("residual", help="Residual analytics")
    residual_subparsers = residual.add_subparsers(dest="residual_command", required=True)
    residual_cluster = residual_subparsers.add_parser("cluster", help="Cluster residual spans")
    residual_cluster.add_argument("--k", type=int, default=10, help="Number of clusters")
    residual_cluster.add_argument(
        "--min-df", type=int, default=1, help="Minimum document frequency"
    )
    residual_cluster.add_argument(
        "--pmi-thresh",
        type=float,
        default=0.05,
        help="PMI threshold when computing fallback residuals",
    )
    residual_cluster.set_defaults(handler=_run_residual_cluster)

    residual_refresh = residual_subparsers.add_parser(
        "refresh", help="Recompute residual_details for an existing run"
    )
    residual_refresh.add_argument(
        "--run-id",
        nargs="+",
        metavar="RUN_ID",
        help=(
            "Optional run id(s) to refresh (defaults to latest run in the database)"
        ),
    )
    residual_refresh.set_defaults(handler=_run_residual_refresh)

    refresh = subparsers.add_parser(
        "refresh",
        help=(
            "Alias for 'residual refresh' to recompute residual_details for existing runs"
        ),
    )
    refresh.add_argument(
        "--run-id",
        nargs="+",
        metavar="RUN_ID",
        help=(
            "Optional run id(s) to refresh (defaults to latest run in the database)"
        ),
    )
    refresh.set_defaults(handler=_run_residual_refresh)

    composite = subparsers.add_parser("composite", help="Composite reconstruction utilities")
    composite_subparsers = composite.add_subparsers(dest="composite_command", required=True)
    composite_backfill = composite_subparsers.add_parser(
        "backfill", help="Backfill composite_reconstruction from cluster residuals"
    )
    composite_backfill.add_argument(
        "--run-id",
        nargs="+",
        metavar="RUN_ID",
        help=(
            "Optional run id(s) to backfill (defaults to latest run in the database)"
        ),
    )
    composite_backfill.set_defaults(handler=_run_composite_backfill)

    morph = subparsers.add_parser("morph", help="Morph semantic tooling")
    morph_subparsers = morph.add_subparsers(dest="morph_command", required=True)
    morph_factorize = morph_subparsers.add_parser("factorize", help="Factorize morph semantics")
    morph_factorize.add_argument("--alpha", type=float, default=0.05, help="Regularization strength")
    morph_factorize.add_argument("--out", default="src/enochian_lm/root_extraction/interpretation/", help="Output directory for run artifacts")
    morph_factorize.add_argument(
        "--embed",
        choices=["gloss-words", "gloss-chars", "hashing-words"],
        default="gloss-chars",
        help="Gloss embedding strategy targeting ~512-dim vectors",
    )
    morph_factorize.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Optional cap for TF-IDF features when using gloss embeddings",
    )
    morph_factorize.add_argument(
        "--min-morph-count",
        type=int,
        default=1,
        help="Minimum occurrences for a morph to be included",
    )
    morph_factorize.add_argument(
        "--min-token-morphs",
        type=int,
        default=0,
        help="Minimum morph count per token (single-morph backfills are always kept)",
    )
    morph_factorize.add_argument(
        "--row-norm",
        action="store_true",
        help="Normalize design matrix rows by sqrt(#morphs)",
    )
    morph_factorize.add_argument(
        "--metric",
        choices=["mse", "cosine"],
        default="mse",
        help="Reconstruction error metric",
    )
    morph_factorize.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tokens processed",
    )
    morph_factorize.set_defaults(handler=_run_morph_factorize)

    report = subparsers.add_parser("report", help="Reporting utilities")
    report_subparsers = report.add_subparsers(dest="report_command", required=True)
    report_pipeline = report_subparsers.add_parser("pipeline", help="Generate a pipeline report")
    report_pipeline.add_argument("--out", default="src/enochian_lm/root_extraction/interpretation/", help="Output directory for the report")
    report_pipeline.add_argument(
        "--baseline",
        help="Optional baseline residual JSONL for comparisons",
        default=None,
    )
    report_pipeline.set_defaults(handler=_run_report_pipeline)

    analyze = subparsers.add_parser("analyze", help="Run all placeholder analytics")
    analyze_subparsers = analyze.add_subparsers(dest="analyze_command", required=True)
    analyze_all = analyze_subparsers.add_parser("all", help="Run all placeholder tasks")
    
    analyze_all.add_argument(
        "--parses",
        required=False,
        help="Path to parses JSONL (required unless reusing existing composite parses)",
    )
    analyze_all.add_argument("--attrib-out", default="src/enochian_lm/root_extraction/interpretation/attribution.csv", help="Attribution CSV output path")
    analyze_all.add_argument("--colloc-out", default="src/enochian_lm/root_extraction/interpretation/collocations.csv", help="Collocation CSV output path")
    analyze_all.add_argument("--min-count", type=int, default=2, help="Minimum joint count")
    analyze_all.add_argument("--k", type=int, default=10, help="Number of clusters")
    analyze_all.add_argument("--min-df", type=int, default=1, help="Minimum document frequency")
    analyze_all.add_argument(
        "--pmi-thresh", type=float, default=0.05, help="PMI threshold for residual clustering"
    )
    analyze_all.add_argument("--residual-out", default="src/enochian_lm/root_extraction/interpretation/residual_clusters.json", help="Residual clustering JSON output path")
    analyze_all.add_argument("--alpha", type=float, default=0.05, help="Regularization strength")
    analyze_all.add_argument(
        "--embed",
        choices=["gloss-words", "gloss-chars", "hashing-words"],
        default="gloss-chars",
        help="Gloss embedding strategy targeting ~512-dim vectors",
    )
    analyze_all.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Optional cap for TF-IDF features when using gloss embeddings",
    )
    analyze_all.add_argument(
        "--min-morph-count",
        type=int,
        default=1,
        help="Minimum occurrences for a morph to be included",
    )
    analyze_all.add_argument(
        "--min-token-morphs",
        type=int,
        default=0,
        help="Minimum morph count per token (single-morph backfills are always kept)",
    )
    analyze_all.add_argument(
        "--row-norm",
        action="store_true",
        help="Normalize design matrix rows by sqrt(#morphs)",
    )
    analyze_all.add_argument(
        "--metric",
        choices=["mse", "cosine"],
        default="mse",
        help="Reconstruction error metric",
    )
    analyze_all.add_argument(
        "--morph-out", default="src/enochian_lm/root_extraction/interpretation/", help="Morph factorization output directory"
    )
    analyze_all.add_argument(
        "--reuse-db-parses",
        action="store_true",
        help=(
            "Reuse existing composite_reconstruction rows instead of ingesting --parses"
        ),
    )
    analyze_all.set_defaults(handler=_run_analyze_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # pragma: no cover - delegated to argparse
        return exc.code

    _configure_logging(args.verbose)
    set_global_seeds(args.seed)

    db_path = Path(args.db).expanduser().resolve()
    args.db_path = db_path
    init_insights_db.init_db(str(db_path))

    conn = connect_sqlite(str(db_path))
    try:
        ensure_analysis_tables(conn)
        conn.commit()
    finally:
        conn.close()

    try:
        args.handler(args)
    except (FileNotFoundError, OSError, ValueError) as error:
        logger.error(f"Command failed. Reason: {str(error)}", exc_info=False, extra={"error": str(error)})
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
