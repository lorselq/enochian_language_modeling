"""Lightweight subtractive semantics pass for uncovered residuals."""
from __future__ import annotations

import json
import logging
from typing import Callable

from enochian_lm.analysis.utils.sql import ensure_analysis_tables
from enochian_lm.common.sqlite_bootstrap import sqlite3

logger = logging.getLogger(__name__)


def _parse_fragments(payload: str | None) -> list[str]:
    if not payload:
        return []
    try:
        data = json.loads(payload)
    except (TypeError, json.JSONDecodeError):
        return []
    results: list[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                val = item.get("text") or item.get("fragment") or item.get("residual")
                if val:
                    results.append(str(val).strip().lower())
            elif isinstance(item, str):
                cleaned = item.strip().lower()
                if cleaned:
                    results.append(cleaned)
    return results


class SubtractiveSemanticsEngine:
    """Select uncovered residuals and request short semantic sketches."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        use_remote: bool = True,
        llm_responder: Callable[[str, str], dict] | None = None,
    ) -> None:
        self.conn = conn
        self.use_remote = use_remote
        self.llm_responder = llm_responder
        ensure_analysis_tables(self.conn)

    def _known_tokens(self) -> set[str]:
        rows = self.conn.execute("SELECT DISTINCT LOWER(TRIM(source_word)) FROM raw_defs").fetchall()
        return {str(row[0]).strip() for row in rows if str(row[0]).strip()}

    def _iter_residuals(self, run_id: str):
        query = """
            SELECT rd.normalized, rd.definition, rd.uncovered_json, c.ngram
            FROM residual_details rd
            JOIN clusters c ON c.cluster_id = rd.cluster_id
            WHERE c.run_id = ?
        """
        return self.conn.execute(query, (run_id,)).fetchall()

    def _render_prompt(self, root: str, residual: str, parent: str, definition: str) -> str:
        return (
            f"Root: {root}\nParent: {parent}\nResidual: {residual}\nDefinition: {definition}\n"
            "Describe the residual fragment's meaning and whether it stands alone."
        )

    def process_run(self, run_id: str, *, limit: int | None = None) -> dict[str, int]:
        known_tokens = self._known_tokens()
        processed = accepted = rejected = 0

        for row in self._iter_residuals(run_id):
            parent = str(row["normalized"] or "").strip()
            definition = str(row["definition"] or "").strip()
            root = str(row["ngram"] or "").strip()
            for residual in _parse_fragments(row["uncovered_json"]):
                if residual in known_tokens:
                    continue
                if limit is not None and processed >= limit:
                    break

                payload: dict = {}
                if callable(self.llm_responder):
                    prompt = self._render_prompt(root, residual, parent, definition)
                    response = self.llm_responder(prompt, run_id)
                    content = response.get("response_text") if isinstance(response, dict) else None
                    if isinstance(content, str):
                        try:
                            payload = json.loads(content)
                        except json.JSONDecodeError:
                            payload = {}

                evaluation = str(payload.get("evaluation") or "unknown").strip().lower()
                definition_text = payload.get("definition") or definition
                semantic_core = payload.get("semantic_core")
                if isinstance(semantic_core, list):
                    semantic_core = ", ".join(str(item) for item in semantic_core)
                example = payload.get("example_usage")
                confidence = payload.get("confidence")
                reason = payload.get("reason")

                self.conn.execute(
                    """
                    INSERT INTO root_residual_semantics (
                        run_id, root, parent_word, residual, evaluation, definition, semantic_core,
                        example_usage, confidence, reason, raw_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, root, parent_word, residual) DO UPDATE SET
                        evaluation=excluded.evaluation,
                        definition=excluded.definition,
                        semantic_core=excluded.semantic_core,
                        example_usage=excluded.example_usage,
                        confidence=excluded.confidence,
                        reason=excluded.reason,
                        raw_json=excluded.raw_json
                    """,
                    (
                        run_id,
                        root,
                        parent,
                        residual,
                        evaluation,
                        definition_text,
                        json.dumps(semantic_core) if isinstance(semantic_core, list) else semantic_core,
                        example,
                        confidence,
                        reason,
                        json.dumps(payload) if payload else None,
                    ),
                )

                processed += 1
                if evaluation == "accepted":
                    accepted += 1
                elif evaluation == "rejected":
                    rejected += 1

        self.conn.commit()
        return {"processed": processed, "accepted": accepted, "rejected": rejected}
