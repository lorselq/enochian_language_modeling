"""Subtractive semantics for singleton residual fragments.

This module performs a secondary pass over residual fragments that only ever
appear as uncovered pieces of longer words. Given a root with an accepted
gloss and a composite word that contains it, we ask the LLM to speculate about
the leftover substring once the root's contribution is subtracted.

Example (conceptual)
--------------------
>>> engine.process_run("demo-run")  # doctest: +SKIP
root=NAZ, word=NAZPSAD, residual=PSAD → propose gloss for PSAD
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Callable, Sequence

from enochian_lm.analysis.utils.sql import ensure_analysis_tables
from enochian_lm.common.sqlite_bootstrap import sqlite3

logger = logging.getLogger(__name__)


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(slots=True)
class ResidualCandidate:
    run_id: str
    cluster_id: int
    root: str
    root_gloss: str
    parent_word: str
    parent_definition: str
    residual: str
    coverage_ratio: float | None
    residual_ratio: float | None

    def describe_relation(self) -> str:
        """Return a short description of how the pieces relate morphologically."""

        word = self.parent_word.lower()
        root = self.root.lower()
        residual = self.residual.lower()

        if word.startswith(root) and word.endswith(residual):
            return (
                f"{self.root.upper()} is a prefix of {self.parent_word.upper()}; "
                f"{self.residual.upper()} is the remaining suffix."
            )
        if word.endswith(root) and word.startswith(residual):
            return (
                f"{self.root.upper()} is a suffix of {self.parent_word.upper()}; "
                f"{self.residual.upper()} is the leading segment."
            )
        idx = word.find(root)
        if idx != -1:
            before = word[:idx]
            after = word[idx + len(root) :]
            return (
                f"{self.parent_word.upper()} contains {self.root.upper()} internally; "
                f"residual spans '{before.upper()}' + '{after.upper()}'."
            )
        return (
            f"{self.parent_word.upper()} exposes residual {self.residual.upper()} after accounting for {self.root.upper()}."
        )


class SubtractiveSemanticsEngine:
    """
    Orchestrates the singleton residual semantics pass.

    Parameters
    ----------
    conn:
        Open SQLite connection for reads/writes.
    use_remote:
        Whether to use the remote LLM endpoint. When ``False``, the local
        endpoint is used. Tests may override ``llm_responder`` instead.
    llm_responder:
        Optional callable ``(prompt: str, *, run_id: str) -> dict[str, str]``
        that mimics :class:`QueryModelTool._run` to avoid real LLM calls.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        use_remote: bool = True,
        llm_responder: Callable[[str, str], dict[str, str]] | None = None,
    ) -> None:
        self.conn = conn
        self.use_remote = use_remote
        self.llm_responder = llm_responder
        ensure_analysis_tables(self.conn)

    # ------------------------
    # Candidate discovery
    # ------------------------
    def _existing_keys(self, run_id: str) -> set[tuple[str, str, str]]:
        rows = self.conn.execute(
            """
            SELECT LOWER(root) AS root, LOWER(parent_word) AS parent_word, LOWER(residual) AS residual
            FROM root_residual_semantics
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchall()
        return {(row["root"], row["parent_word"], row["residual"]) for row in rows}

    def _has_dictionary_entry(self, token: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM raw_defs WHERE LOWER(source_word) = LOWER(?) LIMIT 1",
            (token,),
        ).fetchone()
        return bool(row)

    def _has_glossed_cluster(self, token: str) -> bool:
        row = self.conn.execute(
            """
            SELECT 1
            FROM clusters
            WHERE LOWER(ngram) = LOWER(?)
              AND TRIM(COALESCE(glossator_def, '')) <> ''
            LIMIT 1
            """,
            (token,),
        ).fetchone()
        return bool(row)

    def _fallback_definition(self, cluster_id: int, parent_word: str) -> str:
        row = self.conn.execute(
            """
            SELECT definition
            FROM raw_defs
            WHERE cluster_id = ?
              AND LOWER(source_word) = LOWER(?)
              AND TRIM(COALESCE(definition,'')) <> ''
            LIMIT 1
            """,
            (cluster_id, parent_word),
        ).fetchone()
        return str(row[0]).strip() if row else ""

    def _load_candidates(
        self, run_id: str, *, limit: int | None = None
    ) -> list[ResidualCandidate]:
        existing = self._existing_keys(run_id)
        rows = self.conn.execute(
            """
            SELECT c.cluster_id,
                   c.ngram AS root,
                   COALESCE(c.glossator_def, '') AS root_gloss,
                   rd.normalized AS parent_word,
                   COALESCE(NULLIF(rd.definition, ''), '') AS parent_definition,
                   rd.uncovered_json,
                   rd.coverage_ratio,
                   rd.residual_ratio
            FROM residual_details rd
            JOIN clusters c ON c.cluster_id = rd.cluster_id
            WHERE c.run_id = ?
              AND TRIM(COALESCE(c.glossator_def, '')) <> ''
              AND (c.verdict IS NULL OR LOWER(c.verdict) = 'accepted')
              AND rd.uncovered_json IS NOT NULL
            """,
            (run_id,),
        ).fetchall()

        candidates: list[ResidualCandidate] = []
        for row in rows:
            try:
                uncovered = json.loads(row["uncovered_json"] or "[]")
            except json.JSONDecodeError:
                continue

            parent_word = str(row["parent_word"] or "").strip()
            if not parent_word:
                continue

            parent_definition = str(row["parent_definition"] or "").strip() or self._fallback_definition(
                int(row["cluster_id"]), parent_word
            )
            root = str(row["root"] or "").strip()
            root_gloss = str(row["root_gloss"] or "").strip()

            for frag in uncovered:
                if isinstance(frag, dict):
                    residual_text = frag.get("text") or frag.get("span_text")
                else:
                    residual_text = frag

                residual = str(residual_text or "").strip()
                if not residual:
                    continue
                if len(residual) >= len(parent_word):
                    continue

                key = (root.lower(), parent_word.lower(), residual.lower())
                if key in existing:
                    continue
                if self._has_dictionary_entry(residual) or self._has_glossed_cluster(residual):
                    continue

                candidates.append(
                    ResidualCandidate(
                        run_id=run_id,
                        cluster_id=int(row["cluster_id"]),
                        root=root,
                        root_gloss=root_gloss,
                        parent_word=parent_word,
                        parent_definition=parent_definition,
                        residual=residual,
                        coverage_ratio=_safe_float(row["coverage_ratio"]),
                        residual_ratio=_safe_float(row["residual_ratio"]),
                    )
                )

                if limit is not None and len(candidates) >= limit:
                    return candidates

        return candidates

    # ------------------------
    # Prompting
    # ------------------------
    @staticmethod
    def _build_prompt(candidate: ResidualCandidate) -> str:
        coverage = (
            f"coverage≈{candidate.coverage_ratio:.2f}"
            if candidate.coverage_ratio is not None
            else "coverage≈unknown"
        )
        residual_ratio = (
            f"residual≈{candidate.residual_ratio:.2f}"
            if candidate.residual_ratio is not None
            else "residual≈unknown"
        )

        return (
            "You are a careful internal linguist analyzing leftover fragments in Enochian words. "
            "Infer semantics ONLY from the evidence provided."
            "\n\n"
            f"Root: {candidate.root.upper()}\n"
            f"Root gloss: {candidate.root_gloss or '[no gloss recorded]'}\n"
            f"Composite word: {candidate.parent_word.upper()}\n"
            f"Composite gloss: {candidate.parent_definition or '[no definition recorded]'}\n"
            f"Residual fragment: {candidate.residual.upper()} (appears only inside the composite)\n"
            f"String relationship: {candidate.describe_relation()}\n"
            f"Residual diagnostics: {coverage}; {residual_ratio}.\n"
            "Task: Treat the residual as the semantic remainder after subtracting the root's contribution."
            " Propose a speculative gloss grounded in this evidence alone.\n"
            "If evidence is weak or contradictory, REJECT the residual gloss.\n"
            "\nRespond with STRICT JSON (no markdown, no trailing text) of the form:\n"
            "{"
            '"evaluation": "accepted" or "rejected",'
            '"definition": "1-3 sentences (empty string if rejected)",'
            '"semantic_core": ["up to three nouns or gerunds"],'
            '"example_usage": "1-3 short English sentences that explicitly include the residual token",'
            '"confidence": 0.0,'
            '"reason": "why accepted/rejected, rooted in the provided evidence"'
            "}"
            "\nIf rejected, set definition to an empty string, semantic_core to an empty list,"
            " and example_usage to an empty string."
        )

    def _dispatch_llm(self, prompt: str, *, run_id: str) -> dict[str, str]:
        if self.llm_responder is not None:
            return self.llm_responder(prompt, run_id)

        from enochian_lm.root_extraction.tools.query_model_tool import QueryModelTool

        tool = QueryModelTool(
            system_prompt=(
                "You are a disciplined Enochian linguist focusing on residual morphemes. "
                "Always return pure JSON."
            ),
            use_remote=self.use_remote,
            name="Residual Glossator",
        )
        tool.attach_logging(self.conn, run_id)
        return tool._run(prompt, print_chunks=False, role_name="ResidualGlossator")

    @staticmethod
    def _parse_response(raw: dict[str, str]) -> dict[str, object]:
        text = raw.get("response_text", "") if isinstance(raw, dict) else ""
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {
                "evaluation": "rejected",
                "definition": "",
                "semantic_core": [],
                "example_usage": "",
                "confidence": 0.0,
                "reason": "Malformed or non-JSON response",
                "raw_json": text,
            }

        evaluation = str(parsed.get("evaluation") or "").strip().lower()
        if evaluation not in {"accepted", "rejected"}:
            evaluation = "rejected"

        semantic_core = parsed.get("semantic_core")
        if not isinstance(semantic_core, list):
            semantic_core = []

        return {
            "evaluation": evaluation,
            "definition": str(parsed.get("definition") or ""),
            "semantic_core": json.dumps(semantic_core, ensure_ascii=False),
            "example_usage": str(parsed.get("example_usage") or ""),
            "confidence": _safe_float(parsed.get("confidence")) or 0.0,
            "reason": str(parsed.get("reason") or ""),
            "raw_json": text,
        }

    # ------------------------
    # Persistence
    # ------------------------
    def _persist(self, rows: Sequence[dict[str, object]]) -> None:
        if not rows:
            return

        self.conn.executemany(
            """
            INSERT INTO root_residual_semantics (
              run_id, root, parent_word, residual, evaluation, definition,
              semantic_core, example_usage, confidence, reason, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, root, parent_word, residual) DO UPDATE SET
              evaluation = excluded.evaluation,
              definition = excluded.definition,
              semantic_core = excluded.semantic_core,
              example_usage = excluded.example_usage,
              confidence = excluded.confidence,
              reason = excluded.reason,
              raw_json = excluded.raw_json
            """,
            [
                (
                    row["run_id"],
                    row["root"],
                    row["parent_word"],
                    row["residual"],
                    row["evaluation"],
                    row["definition"],
                    row["semantic_core"],
                    row["example_usage"],
                    row["confidence"],
                    row["reason"],
                    row.get("raw_json", ""),
                )
                for row in rows
            ],
        )
        self.conn.commit()

    # ------------------------
    # Public API
    # ------------------------
    def process_run(self, run_id: str, *, limit: int | None = None) -> dict[str, int]:
        candidates = self._load_candidates(run_id, limit=limit)
        if not candidates:
            logger.info("No singleton residuals found", extra={"run_id": run_id})
            return {"processed": 0, "accepted": 0, "rejected": 0}

        results: list[dict[str, object]] = []
        accepted = 0
        rejected = 0

        for candidate in candidates:
            prompt = self._build_prompt(candidate)
            raw = self._dispatch_llm(prompt, run_id=run_id)
            parsed = self._parse_response(raw)
            parsed.update(
                {
                    "run_id": candidate.run_id,
                    "root": candidate.root,
                    "parent_word": candidate.parent_word,
                    "residual": candidate.residual,
                }
            )
            results.append(parsed)
            if parsed["evaluation"] == "accepted":
                accepted += 1
            else:
                rejected += 1

        self._persist(results)

        logger.info(
            "Subtractive semantics pass complete",
            extra={
                "run_id": run_id,
                "candidates": len(candidates),
                "accepted": accepted,
                "rejected": rejected,
            },
        )

        return {"processed": len(candidates), "accepted": accepted, "rejected": rejected}


__all__ = ["SubtractiveSemanticsEngine", "ResidualCandidate"]
