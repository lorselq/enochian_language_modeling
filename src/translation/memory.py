from __future__ import annotations

"""Persistent translation memory for phrase-level unknown-word handling.

This module stores provisional lexical knowledge that emerges while translating
phrases. The source insights databases remain the canonical record of extraction
results; this memory layer exists solely to accumulate cautious, explicitly
provisional evidence about unknown canonical words across many phrase-level
translation runs.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from enochian_lm.common.sqlite_bootstrap import sqlite3


@dataclass(slots=True)
class ProvisionalLexiconEntry:
    """Summarize the best current provisional read for an unknown word."""

    word: str
    best_gloss: str | None
    role_hint: str | None
    confidence: float
    evidence_count: int
    alternates: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)


class TranslationMemoryRepository:
    """Persist provisional phrase-translation knowledge in a separate SQLite DB."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "TranslationMemoryRepository":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def fetch_entry(self, word: str) -> ProvisionalLexiconEntry | None:
        """Return the stored provisional entry for ``word`` when available."""
        row = self.conn.execute(
            """
            SELECT word, best_gloss, role_hint, confidence, evidence_count,
                   alternates_json, examples_json
            FROM provisional_lexicon
            WHERE word = ?;
            """,
            ((word or "").strip().upper(),),
        ).fetchone()
        if row is None:
            return None
        return ProvisionalLexiconEntry(
            word=str(row["word"]),
            best_gloss=row["best_gloss"],
            role_hint=row["role_hint"],
            confidence=float(row["confidence"] or 0.0),
            evidence_count=int(row["evidence_count"] or 0),
            alternates=_load_json_list(row["alternates_json"]),
            examples=_load_json_list(row["examples_json"]),
        )

    def record_observation(
        self,
        *,
        word: str,
        phrase: str,
        role_hint: str | None,
        glosses: list[str],
        confidence: float,
        left_neighbor: str | None,
        right_neighbor: str | None,
    ) -> dict[str, object]:
        """Upsert a provisional lexical observation and return the applied delta.

        Phrase translation should keep improving its guesses about unknown
        canonical words as more contexts are seen. This method records the raw
        observation and updates the summarized lexicon entry using a conservative
        weighted average so later phrase runs can reuse the best current guess.
        """
        normalized = (word or "").strip().upper()
        cleaned_glosses = [
            gloss.strip()
            for gloss in glosses
            if isinstance(gloss, str) and gloss.strip()
        ]
        existing = self.fetch_entry(normalized)

        if existing is None:
            evidence_count = 1
            blended_confidence = max(0.0, min(1.0, float(confidence)))
            alternates = cleaned_glosses[:5]
            best_gloss = alternates[0] if alternates else None
            examples = [phrase] if phrase else []
            resolved_role = role_hint
        else:
            evidence_count = existing.evidence_count + 1
            blended_confidence = (
                (existing.confidence * existing.evidence_count) + float(confidence)
            ) / float(max(1, evidence_count))
            alternates = _merge_unique(existing.alternates, cleaned_glosses)[:5]
            best_gloss = existing.best_gloss
            if cleaned_glosses and (
                best_gloss is None or float(confidence) >= existing.confidence
            ):
                best_gloss = cleaned_glosses[0]
            examples = _merge_unique(existing.examples, [phrase])[:5]
            resolved_role = role_hint or existing.role_hint

        with self.conn:
            self.conn.execute(
                """
                INSERT INTO phrase_observations (
                    word,
                    phrase,
                    role_hint,
                    confidence,
                    glosses_json,
                    left_neighbor,
                    right_neighbor
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    normalized,
                    phrase,
                    resolved_role,
                    blended_confidence,
                    json.dumps(cleaned_glosses, ensure_ascii=True),
                    left_neighbor,
                    right_neighbor,
                ),
            )
            self.conn.execute(
                """
                INSERT INTO provisional_lexicon (
                    word,
                    best_gloss,
                    role_hint,
                    confidence,
                    evidence_count,
                    alternates_json,
                    examples_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(word) DO UPDATE SET
                    best_gloss = excluded.best_gloss,
                    role_hint = excluded.role_hint,
                    confidence = excluded.confidence,
                    evidence_count = excluded.evidence_count,
                    alternates_json = excluded.alternates_json,
                    examples_json = excluded.examples_json,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now');
                """,
                (
                    normalized,
                    best_gloss,
                    resolved_role,
                    blended_confidence,
                    evidence_count,
                    json.dumps(alternates, ensure_ascii=True),
                    json.dumps(examples, ensure_ascii=True),
                ),
            )

        return {
            "word": normalized,
            "best_gloss": best_gloss,
            "role_hint": resolved_role,
            "confidence": round(blended_confidence, 4),
            "evidence_count": evidence_count,
            "alternates": alternates,
            "examples": examples,
        }

    def _ensure_schema(self) -> None:
        """Create the translation-memory schema when it is missing."""
        with self.conn:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS provisional_lexicon (
                    word TEXT PRIMARY KEY,
                    best_gloss TEXT,
                    role_hint TEXT,
                    confidence REAL NOT NULL DEFAULT 0.0,
                    evidence_count INTEGER NOT NULL DEFAULT 0,
                    alternates_json TEXT NOT NULL DEFAULT '[]',
                    examples_json TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                );

                CREATE TABLE IF NOT EXISTS phrase_observations (
                    obs_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    word TEXT NOT NULL,
                    phrase TEXT NOT NULL,
                    role_hint TEXT,
                    confidence REAL NOT NULL DEFAULT 0.0,
                    glosses_json TEXT NOT NULL DEFAULT '[]',
                    left_neighbor TEXT,
                    right_neighbor TEXT,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                );
                """
            )


def _load_json_list(payload: object) -> list[str]:
    if isinstance(payload, list):
        return [str(item) for item in payload if str(item).strip()]
    if not isinstance(payload, str) or not payload.strip():
        return []
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return [payload]
    if isinstance(data, list):
        return [str(item) for item in data if str(item).strip()]
    return [str(data)]


def _merge_unique(existing: list[str], new_items: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for item in [*existing, *new_items]:
        normalized = str(item).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        merged.append(normalized)
    return merged
