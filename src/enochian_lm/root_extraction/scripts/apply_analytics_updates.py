"""Apply analytics-driven definition updates to the insights database."""

from __future__ import annotations

import argparse
import json
from enochian_lm.common.sqlite_bootstrap import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from enochian_lm.root_extraction.utils.config import get_config_paths
from enochian_lm.root_extraction.utils.analytics_bridge import gather_morph_evidence
from enochian_lm.analysis.utils.sql import ensure_analysis_tables


@dataclass
class UpdateDecision:
    cluster_id: int
    root: str
    partner: str
    delta_partner_given_root: float
    delta_root_given_partner: float
    n_tokens: int
    partner_only: int
    root_only: int
    colloc_shared: int


def _connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


def _timestamp() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _load_clusters(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    try:
        rows = conn.execute(
            """
            SELECT cluster_id, ngram, glossator_def, verdict
            FROM clusters_processed
            WHERE TRIM(COALESCE(glossator_def, '')) <> ''
            """
        ).fetchall()
    except sqlite3.Error as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to load clusters: {exc}") from exc
    return rows


def _choose_update(
    analytics: dict[str, Any],
    *,
    root: str,
    min_delta_gap: float,
    min_partner_delta: float,
    min_tokens: int,
) -> UpdateDecision | None:
    pairwise = analytics.get("pairwise") or []
    colloc = analytics.get("collocations") or {}
    best: UpdateDecision | None = None

    for entry in pairwise:
        partner = entry.get("partner")
        if not partner:
            continue
        delta_partner = float(entry.get("delta_partner_given_root", 0.0))
        delta_root = float(entry.get("delta_root_given_partner", 0.0))
        n_tokens = int(entry.get("n_tokens", 0))
        dominance = float(entry.get("dominance", 0.0))
        if n_tokens < min_tokens:
            continue
        if dominance < min_delta_gap or delta_partner < min_partner_delta:
            continue

        partner_cf = partner.casefold()
        colloc_entry = colloc.get(partner_cf)
        partner_only = 0
        root_only = 0
        shared = 0
        if colloc_entry:
            shared = int(colloc_entry.get("shared", 0))
            partner_only = max(int(colloc_entry.get("partner_total", 0)) - shared, 0)
            root_only = max(int(colloc_entry.get("root_total", 0)) - shared, 0)

        decision = UpdateDecision(
            cluster_id=-1,
            root=root,
            partner=partner,
            delta_partner_given_root=delta_partner,
            delta_root_given_partner=delta_root,
            n_tokens=n_tokens,
            partner_only=partner_only,
            root_only=root_only,
            colloc_shared=shared,
        )

        if best is None:
            best = decision
        else:
            best_score = best.delta_partner_given_root - best.delta_root_given_partner
            if dominance > best_score:
                best = decision

    return best


def _definition_has_partner(defn: dict[str, Any], partner: str) -> bool:
    notes = defn.get("ANALYTICS_NOTES")
    if not isinstance(notes, list):
        return False
    partner_cf = partner.casefold()
    for note in notes:
        if isinstance(note, dict) and str(note.get("partner", "")).casefold() == partner_cf:
            return True
    return False


def _apply_definition_update(
    conn: sqlite3.Connection,
    *,
    cluster_id: int,
    def_json: str,
    decision: UpdateDecision,
    dry_run: bool,
) -> tuple[bool, str]:
    try:
        definition = json.loads(def_json)
    except json.JSONDecodeError:
        return False, "definition is not valid JSON"

    if not isinstance(definition, dict):
        return False, "definition payload is not an object"

    if _definition_has_partner(definition, decision.partner):
        return False, "partner already noted"

    timestamp = _timestamp()
    update_sentence = (
        f"[Analytics update {timestamp}] In composites with {decision.partner.upper()}, "
        f"Δ({decision.partner.upper()}|{decision.root.upper()})="
        f"{decision.delta_partner_given_root:.2f} vs Δ({decision.root.upper()}|{decision.partner.upper()})="
        f"{decision.delta_root_given_partner:.2f} (n={decision.n_tokens})."
    )
    if decision.partner_only or decision.root_only:
        update_sentence += (
            f" Collocations show {decision.partner.upper()} occurs without {decision.root.upper()} "
            f"{decision.partner_only}× vs {decision.root.upper()} alone {decision.root_only}×."
        )
    update_sentence += (
        f" Prioritize attributing the shared semantics to {decision.partner.upper()} and treat"
        f" {decision.root.upper()} as supportive when they co-occur."
    )

    existing_definition = str(definition.get("DEFINITION", "")).strip()
    if existing_definition:
        definition["DEFINITION"] = existing_definition + " " + update_sentence
    else:
        definition["DEFINITION"] = update_sentence

    notes = definition.setdefault("ANALYTICS_NOTES", [])
    if isinstance(notes, list):
        notes.append(
            {
                "timestamp": timestamp,
                "root": decision.root.upper(),
                "partner": decision.partner.upper(),
                "delta_partner_given_root": round(decision.delta_partner_given_root, 4),
                "delta_root_given_partner": round(decision.delta_root_given_partner, 4),
                "n_tokens": decision.n_tokens,
                "partner_only_occurrences": decision.partner_only,
                "root_only_occurrences": decision.root_only,
                "shared_occurrences": decision.colloc_shared,
            }
        )

    serialized = json.dumps(definition, indent=2, ensure_ascii=False)

    if dry_run:
        return True, serialized

    conn.execute(
        "UPDATE clusters SET glossator_def = ? WHERE cluster_id = ?",
        (serialized, cluster_id),
    )
    return True, serialized


def apply_updates(
    *,
    db_path: str,
    min_delta_gap: float,
    min_partner_delta: float,
    min_tokens: int,
    max_updates: int | None,
    dry_run: bool,
) -> list[UpdateDecision]:
    conn = _connect(db_path)
    ensure_analysis_tables(conn)
    try:
        rows = _load_clusters(conn)
        decisions: list[UpdateDecision] = []
        updates_applied = 0

        for row in rows:
            if max_updates is not None and updates_applied >= max_updates:
                break

            verdict = str(row["verdict"] or "").strip().lower()
            if verdict and verdict not in {"accepted", "accept"}:
                continue

            ngram = str(row["ngram"] or "").strip()
            if not ngram:
                continue

            analytics = gather_morph_evidence(
                conn,
                root=ngram.lower(),
                partner_counts=Counter(),
                residual_counts=Counter(),
            )

            decision = _choose_update(
                analytics,
                root=ngram,
                min_delta_gap=min_delta_gap,
                min_partner_delta=min_partner_delta,
                min_tokens=min_tokens,
            )

            if decision is None:
                continue

            decision.cluster_id = int(row["cluster_id"])
            applied, _ = _apply_definition_update(
                conn,
                cluster_id=decision.cluster_id,
                def_json=row["glossator_def"],
                decision=decision,
                dry_run=dry_run,
            )

            if applied:
                decisions.append(decision)
                updates_applied += 1

        if not dry_run:
            conn.commit()

        return decisions
    finally:
        conn.close()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Update accepted definitions using attribution/collocation analytics",
    )
    parser.add_argument("--style", choices=["debate", "solo"], default="debate", help="Which insights DB to open")
    parser.add_argument("--db", help="Override database path")
    parser.add_argument("--min-delta-gap", type=float, default=0.05, help="Minimum dominance gap Δ(partner|root)-Δ(root|partner)")
    parser.add_argument("--min-partner-delta", type=float, default=0.12, help="Minimum Δ(partner|root) required to update")
    parser.add_argument("--min-tokens", type=int, default=3, help="Minimum number of tokens supporting the pair")
    parser.add_argument("--max-updates", type=int, help="Optional cap on number of updates to apply")
    parser.add_argument("--dry-run", action="store_true", help="Preview updates without writing to the database")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    paths = get_config_paths()
    db_path = args.db or str(paths[args.style])

    decisions = apply_updates(
        db_path=db_path,
        min_delta_gap=args.min_delta_gap,
        min_partner_delta=args.min_partner_delta,
        min_tokens=args.min_tokens,
        max_updates=args.max_updates,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print(f"[dry-run] {len(decisions)} definitions would be updated at {db_path}")
    else:
        print(f"Applied {len(decisions)} analytics updates to {db_path}")

    for decision in decisions:
        print(
            f"- cluster={decision.cluster_id} root={decision.root.upper()} partner={decision.partner.upper()} "
            f"Δ_partner={decision.delta_partner_given_root:.2f} Δ_root={decision.delta_root_given_partner:.2f} "
            f"n={decision.n_tokens} partner_only={decision.partner_only} root_only={decision.root_only}"
        )

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

