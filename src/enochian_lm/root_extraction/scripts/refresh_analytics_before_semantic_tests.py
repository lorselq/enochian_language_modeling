#!/usr/bin/env python3
"""Refresh analytics tables before semantic-subtraction tests.

Workflow:
1) Ensure composite_reconstruction and morph_semantic_vectors are populated.
2) Backfill composites from residual details when missing.
3) Run `enlm morph factorize` then `enlm analyze all --reuse-db-parses`.
4) Validate non-zero downstream tables/artifacts.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
from pathlib import Path

from enochian_lm.analysis.utils.sql import ensure_analysis_tables


def _count(conn: sqlite3.Connection, table: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return int(row[0]) if row else 0


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _resolve_paths(db_path: Path, out_dir: Path) -> dict[str, Path]:
    stem = db_path.stem
    return {
        "morph": out_dir / f"{stem}_morph_factorize",
        "attrib": out_dir / f"{stem}_attribution.csv",
        "colloc": out_dir / f"{stem}_collocations.csv",
        "residual": out_dir / f"{stem}_residual_clusters.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path, help="Insights DB path")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/analytics_refresh"),
        help="Directory for analyze-all exports",
    )
    parser.add_argument(
        "--skip-backfill",
        action="store_true",
        help="Do not auto-run composite backfill when composite_reconstruction is empty",
    )
    args = parser.parse_args()

    db_path: Path = args.db.resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"DB does not exist: {db_path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = _resolve_paths(db_path, args.out_dir)

    conn = sqlite3.connect(db_path)
    try:
        ensure_analysis_tables(conn)
        composite_count = _count(conn, "composite_reconstruction")
        morph_count = _count(conn, "morph_semantic_vectors")
        accepted_count = conn.execute(
            "SELECT COUNT(*) FROM clusters WHERE lower(COALESCE(verdict,''))='accept'"
        ).fetchone()[0]
    finally:
        conn.close()

    print(
        f"[precheck] accepted_clusters={accepted_count} composites={composite_count} morph_vectors={morph_count}"
    )

    if composite_count == 0 and not args.skip_backfill:
        _run(["poetry", "run", "enlm", "composite", "backfill", "--db", str(db_path)])

    _run(
        [
            "poetry",
            "run",
            "enlm",
            "morph",
            "factorize",
            "--db",
            str(db_path),
            "--out",
            str(outputs["morph"]),
        ]
    )

    _run(
        [
            "poetry",
            "run",
            "enlm",
            "analyze",
            "all",
            "--db",
            str(db_path),
            "--reuse-db-parses",
            "--attrib-out",
            str(outputs["attrib"]),
            "--colloc-out",
            str(outputs["colloc"]),
            "--residual-out",
            str(outputs["residual"]),
            "--morph-out",
            str(outputs["morph"]),
        ]
    )

    conn = sqlite3.connect(db_path)
    try:
        ensure_analysis_tables(conn)
        composite_count = _count(conn, "composite_reconstruction")
        morph_count = _count(conn, "morph_semantic_vectors")
        attrib_count = _count(conn, "attribution_marginals")
        colloc_count = _count(conn, "collocation_stats")
        residual_count = _count(conn, "residual_clusters")
    finally:
        conn.close()

    if not outputs["residual"].exists():
        raise RuntimeError("Residual cluster export missing after analyze all")

    payload = json.loads(outputs["residual"].read_text(encoding="utf-8"))
    cluster_total = int(payload.get("clusters", 0)) if isinstance(payload, dict) else 0

    print(
        "[postcheck] "
        f"composites={composite_count} morph_vectors={morph_count} "
        f"attribution={attrib_count} collocations={colloc_count} "
        f"residual_clusters_table={residual_count} residual_clusters_json={cluster_total}"
    )

    if min(composite_count, morph_count, attrib_count, colloc_count, residual_count) <= 0:
        raise RuntimeError("Analytics refresh incomplete: at least one required table is empty")


if __name__ == "__main__":
    main()
