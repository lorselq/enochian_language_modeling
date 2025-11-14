"""Command line interface for Enochian language modeling utilities."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Iterable

from enochian_translation_team.scripts import init_insights_db

from .analysis.attribution import run_leave_one_out
from .analysis.colloc import compute_collocations
from .analysis.factorize import factorize_morphemes
from .analysis.residuals import cluster_residuals
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


def _write_json(path: Path, payload: dict[str, object]) -> None:
    def writer(tmp: NamedTemporaryFile) -> None:
        json.dump(payload, tmp, indent=2, ensure_ascii=False)
        tmp.write("\n")

    _atomic_write(path, writer)


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


def _run_residual_cluster(args: argparse.Namespace) -> None:
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


def _run_analyze_all(args: argparse.Namespace) -> None:
    combined_args = argparse.Namespace(
        db_path=args.db_path,
        limit=None,
    )
    _run_attrib_loo(combined_args)

    combined_args = argparse.Namespace(
        db_path=args.db_path,
        min_count=args.min_count,
        limit=None,
    )
    _run_colloc(combined_args)

    combined_args = argparse.Namespace(
        db_path=args.db_path,
        k=args.k,
        min_df=args.min_df,
        pmi_thresh=args.pmi_thresh,
    )
    _run_residual_cluster(combined_args)

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
    parser.add_argument("--db", default="enlm_insights.sqlite3", help="Database path")
    parser.add_argument("--seed", type=int, default=42, help="Global seed")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", required=True)

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
