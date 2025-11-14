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
    morphs_path = Path(args.morphs)
    _validate_input_file(morphs_path, "Morph list")
    _write_csv_header(
        Path(args.out),
        (
            "morph_left",
            "morph_right",
            "count_ab",
            "count_a",
            "count_b",
            "pmi",
            "llr",
            "asym_dep",
            "updated_at",
        ),
    )
    logger.info("Wrote collocation placeholder", extra={"out": args.out, "min_count": args.min_count})


def _run_residual_cluster(args: argparse.Namespace) -> None:
    payload = {
        "k": args.k,
        "min_df": args.min_df,
        "created_at": utcnow_iso(),
    }
    _write_json(Path(args.out), payload)
    logger.info(
        "Wrote residual clustering placeholder",
        extra={"out": args.out, "k": args.k, "min_df": args.min_df},
    )


def _run_morph_factorize(args: argparse.Namespace) -> None:
    _write_csv_header(
        Path(args.out),
        (
            "morph",
            "l2_norm",
            "updated_at",
        ),
    )
    logger.info("Wrote morph factorization placeholder", extra={"out": args.out, "alpha": args.alpha})


def _run_analyze_all(args: argparse.Namespace) -> None:
    combined_args = argparse.Namespace(
        db_path=args.db_path,
        limit=None,
    )
    _run_attrib_loo(combined_args)

    combined_args = argparse.Namespace(
        morphs=args.morphs,
        out=args.colloc_out,
        min_count=args.min_count,
    )
    _run_colloc(combined_args)

    combined_args = argparse.Namespace(
        k=args.k,
        min_df=args.min_df,
        out=args.residual_out,
    )
    _run_residual_cluster(combined_args)

    combined_args = argparse.Namespace(
        alpha=args.alpha,
        out=args.morph_out,
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
    colloc.add_argument("--morphs", required=True, help="Path to morph inventory")
    colloc.add_argument("--out", required=True, help="Output CSV path")
    colloc.add_argument("--min-count", type=int, default=5, help="Minimum joint count")
    colloc.set_defaults(handler=_run_colloc)

    residual = subparsers.add_parser("residual", help="Residual analytics")
    residual_subparsers = residual.add_subparsers(dest="residual_command", required=True)
    residual_cluster = residual_subparsers.add_parser("cluster", help="Cluster residual spans")
    residual_cluster.add_argument("--k", type=int, default=10, help="Number of clusters")
    residual_cluster.add_argument("--min-df", type=int, default=2, help="Minimum document frequency")
    residual_cluster.add_argument("--out", required=True, help="Output JSON path")
    residual_cluster.set_defaults(handler=_run_residual_cluster)

    morph = subparsers.add_parser("morph", help="Morph semantic tooling")
    morph_subparsers = morph.add_subparsers(dest="morph_command", required=True)
    morph_factorize = morph_subparsers.add_parser("factorize", help="Factorize morph semantics")
    morph_factorize.add_argument("--alpha", type=float, default=1.0, help="Regularization strength")
    morph_factorize.add_argument("--out", required=True, help="Output CSV path")
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
    analyze_all.add_argument("--residual-out", required=True, help="Residual clustering JSON output path")
    analyze_all.add_argument("--alpha", type=float, default=1.0, help="Regularization strength")
    analyze_all.add_argument("--morph-out", required=True, help="Morph factorization CSV output path")
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
