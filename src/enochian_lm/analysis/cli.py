"""Command line interface for Enochian language modeling utilities."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Iterable, Iterator, Sequence

from enochian_lm.root_extraction.scripts import init_insights_db
from enochian_lm.root_extraction.utils.preanalysis import execute_preanalysis

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
    rows: list[tuple[str, str | None, str, float, str, str]] = []
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
                timestamp,
            )
        )

    conn.execute("DELETE FROM composite_reconstruction")
    if rows:
        conn.executemany(
            """
            INSERT INTO composite_reconstruction (
              token, gold_gloss, pred_vector_json, recon_error, used_morphs_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    conn.commit()
    logger.info(
        "Ingested composite parses",
        extra={"rows": len(rows), "path": str(path)},
    )
    return len(rows)


def _ingest_morph_inventory(conn, path: Path) -> int:
    timestamp = utcnow_iso()
    rows: list[dict[str, object]] = []
    for payload in _iter_json_lines(path):
        morph = _coerce_optional_str(payload, ("morph", "token", "name"))
        if morph is None:
            continue
        vector = _coerce_float_list(payload.get("vector") or payload.get("embedding"))
        if vector is None:
            vector_json = payload.get("vector_json")
            if isinstance(vector_json, str):
                vector = _coerce_float_list(vector_json)
        if vector is None:
            continue
        norm_raw = payload.get("l2_norm") or payload.get("norm")
        try:
            norm = float(norm_raw) if norm_raw is not None else _vector_norm(vector)
        except (TypeError, ValueError):
            norm = _vector_norm(vector)
        rows.append(
            {
                "morph": morph,
                "vector_json": json.dumps(_round_vector(vector)),
                "l2_norm": round(norm, 4),
                "updated_at": timestamp,
            }
        )

    conn.execute("DELETE FROM morph_semantic_vectors")
    if rows:
        conn.executemany(
            """
            INSERT INTO morph_semantic_vectors (morph, vector_json, l2_norm, updated_at)
            VALUES (:morph, :vector_json, :l2_norm, :updated_at)
            ON CONFLICT(morph) DO UPDATE SET
              vector_json=excluded.vector_json,
              l2_norm=excluded.l2_norm,
              updated_at=excluded.updated_at
            """,
            rows,
        )
    conn.commit()
    logger.info(
        "Ingested morph inventory",
        extra={"rows": len(rows), "path": str(path)},
    )
    return len(rows)


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
    result = execute_preanalysis(
        db_path=db_path,
        stage=args.stage,
        trusted_path=args.trusted,
        run_id=args.run_id,
        refresh=args.refresh,
    )

    stage = result.get("stage")
    run_id = result.get("run_id")
    pre_id = result.get("preanalysis_id")
    created = result.get("created")
    trusted_count = result.get("trusted_count")
    snapshots = result.get("snapshots", [])

    status = "created" if created else "reused"
    print(
        f"[{stage}] Pre-analysis {status} for run {run_id} (preanalysis_id={pre_id}; trusted={trusted_count})."
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
    parses_path = Path(args.parses)
    morphs_path = Path(args.morphs)
    attrib_out = Path(args.attrib_out)
    colloc_out = Path(args.colloc_out)
    residual_out = Path(args.residual_out)

    _validate_input_file(parses_path, "Parses file")
    _validate_input_file(morphs_path, "Morph inventory")

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

        ingested_tokens = _ingest_composite_parses(conn, parses_path)
        ingested_morphs = _ingest_morph_inventory(conn, morphs_path)
    finally:
        conn.close()

    if ingested_tokens == 0:
        raise ValueError(
            "No composite parses were ingested; check the --parses file contents."
        )
    if ingested_morphs == 0:
        raise ValueError(
            "No morph semantic vectors were ingested; check the --morphs file contents."
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
    parser.add_argument("--db", default="src/enochian_translation_team/data/solo_analysis_derived_definitions.sqlite3", help="Database path")
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
        help="Existing translation run id (auto-selects latest when omitted)",
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
    colloc.add_argument("--min-count", type=int, default=5, help="Minimum joint count")
    colloc.add_argument("--limit", type=int, default=None, help="Limit number of pairs processed")
    colloc.set_defaults(handler=_run_colloc)

    residual = subparsers.add_parser("residual", help="Residual analytics")
    residual_subparsers = residual.add_subparsers(dest="residual_command", required=True)
    residual_cluster = residual_subparsers.add_parser("cluster", help="Cluster residual spans")
    residual_cluster.add_argument("--k", type=int, default=10, help="Number of clusters")
    residual_cluster.add_argument(
        "--min-df", type=int, default=2, help="Minimum document frequency"
    )
    residual_cluster.add_argument(
        "--pmi-thresh",
        type=float,
        default=0.0,
        help="PMI threshold when computing fallback residuals",
    )
    residual_cluster.set_defaults(handler=_run_residual_cluster)

    morph = subparsers.add_parser("morph", help="Morph semantic tooling")
    morph_subparsers = morph.add_subparsers(dest="morph_command", required=True)
    morph_factorize = morph_subparsers.add_parser("factorize", help="Factorize morph semantics")
    morph_factorize.add_argument("--alpha", type=float, default=1.0, help="Regularization strength")
    morph_factorize.add_argument("--out", required=True, help="Output directory for run artifacts")
    morph_factorize.add_argument(
        "--embed",
        choices=["gloss-words", "gloss-chars", "hashing-words"],
        default="gloss-words",
        help="Gloss embedding strategy",
    )
    morph_factorize.add_argument(
        "--min-morph-count",
        type=int,
        default=3,
        help="Minimum occurrences for a morph to be included",
    )
    morph_factorize.add_argument(
        "--min-token-morphs",
        type=int,
        default=2,
        help="Minimum morph count per token",
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
    report_pipeline.add_argument("--out", help="Output directory for the report")
    report_pipeline.add_argument(
        "--baseline",
        help="Optional baseline residual JSONL for comparisons",
        default=None,
    )
    report_pipeline.set_defaults(handler=_run_report_pipeline)

    analyze = subparsers.add_parser("analyze", help="Run all placeholder analytics")
    analyze_subparsers = analyze.add_subparsers(dest="analyze_command", required=True)
    analyze_all = analyze_subparsers.add_parser("all", help="Run all placeholder tasks")
    analyze_all.add_argument("--parses", required=True, help="Path to parses JSONL")
    analyze_all.add_argument("--attrib-out", required=True, help="Attribution CSV output path")
    analyze_all.add_argument("--morphs", required=True, help="Path to morph inventory")
    analyze_all.add_argument("--colloc-out", required=True, help="Collocation CSV output path")
    analyze_all.add_argument("--min-count", type=int, default=5, help="Minimum joint count")
    analyze_all.add_argument("--k", type=int, default=10, help="Number of clusters")
    analyze_all.add_argument("--min-df", type=int, default=2, help="Minimum document frequency")
    analyze_all.add_argument(
        "--pmi-thresh", type=float, default=0.0, help="PMI threshold for residual clustering"
    )
    analyze_all.add_argument("--residual-out", required=True, help="Residual clustering JSON output path")
    analyze_all.add_argument("--alpha", type=float, default=1.0, help="Regularization strength")
    analyze_all.add_argument(
        "--embed",
        choices=["gloss-words", "gloss-chars", "hashing-words"],
        default="gloss-words",
        help="Gloss embedding strategy",
    )
    analyze_all.add_argument(
        "--min-morph-count",
        type=int,
        default=3,
        help="Minimum occurrences for a morph to be included",
    )
    analyze_all.add_argument(
        "--min-token-morphs",
        type=int,
        default=2,
        help="Minimum morph count per token",
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
        "--morph-out", required=True, help="Morph factorization output directory"
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
        logger.error("Command failed", exc_info=False, extra={"error": str(error)})
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
