from __future__ import annotations
from enochian_lm.common.sqlite_bootstrap import sqlite3
import sys
from pathlib import Path

TABLES_ORDER = [
    "runs",
    "clusters",
    "raw_defs",
    "citations",
    "skips",
    "decisions",
]


def colset(conn, schema, table):
    return {r[1] for r in conn.execute(f"PRAGMA {schema}.table_info({table});")}


def copy_table(conn, table):
    src_cols = colset(conn, "src", table)
    dst_cols = colset(conn, "main", table)
    common = [c for c in dst_cols if c in src_cols]  # keep destination column order
    if not common:
        print(f"  - skip {table}: no shared columns")
        return 0
    cols_csv = ", ".join(f'"{c}"' for c in common)
    sql = f'INSERT INTO "{table}" ({cols_csv}) SELECT {cols_csv} FROM src."{table}";'
    cur = conn.execute(sql)
    return cur.rowcount if cur.rowcount is not None else conn.total_changes


def fix_sqlite_sequence(conn, table, pk="cluster_id"):
    # If table has AUTOINCREMENT, align sqlite_sequence to current max(pk)
    has_seq = conn.execute(
        "SELECT name FROM sqlite_master WHERE name='sqlite_sequence'"
    ).fetchone()
    if not has_seq:
        return
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    if not row:
        return
    maxid = conn.execute(f'SELECT COALESCE(MAX({pk}),0) FROM "{table}"').fetchone()[0]
    # update only if there is a sequence row
    r = conn.execute("SELECT 1 FROM sqlite_sequence WHERE name=?", (table,)).fetchone()
    if r:
        conn.execute("UPDATE sqlite_sequence SET seq=? WHERE name=?", (maxid, table))


def main(src_path: str, dst_path: str):
    src = Path(src_path).resolve()
    dst = Path(dst_path).resolve()
    if not src.exists():
        sys.exit(f"Source not found: {src}")
    if not dst.exists():
        sys.exit(f"Destination not found: {dst} (run your init script first)")

    with sqlite3.connect(dst.as_posix()) as conn:
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("ATTACH DATABASE ? AS src;", (src.as_posix(),))

        # Sanity: ensure tables exist
        for t in TABLES_ORDER:
            if not conn.execute(
                "SELECT 1 FROM src.sqlite_master WHERE type='table' AND name=?", (t,)
            ).fetchone():
                print(f"Note: source missing table {t}, skipping.")

        conn.execute("BEGIN;")
        try:
            total = 0
            # Copy in dependency order (parents first)
            for t in TABLES_ORDER:
                if conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (t,)
                ).fetchone():
                    n = copy_table(conn, t)
                    total += n
                    print(f"Copied {n} rows into {t}")
            # Align AUTOINCREMENT sequences for known tables
            for t, pk in [
                ("clusters", "cluster_id"),
                ("raw_defs", "def_id"),
                ("citations", "citation_id"),
                ("skips", "skip_id"),
                ("decisions", "decision_id"),
            ]:
                fix_sqlite_sequence(conn, t, pk)
            conn.commit()
            print(f"Done. Total rows inserted: {total}")
        except Exception as e:
            conn.rollback()
            raise
        finally:
            conn.execute("DETACH DATABASE src;")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python migrate_no_fts.py /path/to/old.db /path/to/new_nofts.db")
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
