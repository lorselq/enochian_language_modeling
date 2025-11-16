"""Pre-analysis safeguards for translation runs."""
from __future__ import annotations

import json
from enochian_common.sqlite_bootstrap import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from .config import get_config_paths
from .dictionary_loader import load_dictionary

Stage = Literal["initial", "subsequent"]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z"


def load_trusted_ngrams(
    path: str | Path | None = None,
    *,
    dictionary_path: str | Path | None = None,
    max_length: int | None = None,
) -> list[str]:
    """Load trusted n-grams from ``path`` or derive them from the dictionary."""

    if path is None:
        path = get_config_paths()["preanalysis_trusted"]
    data_path = Path(path)
    if data_path.exists():
        payload = json.loads(data_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            result = []
            for item in payload:
                text = str(item).strip().upper()
                if text:
                    result.append(text)
            normalized = sorted(set(result))
            if max_length is not None:
                normalized = [token for token in normalized if len(token) <= max_length]
            return normalized
        raise ValueError(f"Trusted n-gram list must be a JSON array: {data_path}")

    dict_path = Path(dictionary_path) if dictionary_path else Path(get_config_paths()["dictionary"])
    if not dict_path.exists():
        raise FileNotFoundError(
            "Trusted n-gram list missing and dictionary source not found: "
            f"{dict_path}"
        )
    return _derive_trusted_from_dictionary(
        dictionary_path=dict_path,
        max_length=max_length,
    )


def _normalize_trusted(values: Iterable[str]) -> list[str]:
    return sorted({str(v).strip().upper() for v in values if str(v).strip()})


def _derive_trusted_from_dictionary(
    *, dictionary_path: Path, max_length: int | None
) -> list[str]:
    """Build trusted n-grams directly from the canonical dictionary entries."""

    entries = load_dictionary(str(dictionary_path))
    tokens: list[str] = []
    for entry in entries:
        canonical = str(entry.get("canonical") or "").strip()
        if canonical:
            tokens.append(canonical)
        for alt in entry.get("alternates") or []:
            value: str = ""
            if isinstance(alt, dict):
                value = str(alt.get("value") or "").strip()
            else:
                value = str(alt).strip()
            if value:
                tokens.append(value)

    normalized = _normalize_trusted(tokens)
    if max_length is not None:
        normalized = [token for token in normalized if len(token) <= max_length]
    return normalized


@dataclass(slots=True)
class _SeedSnapshot:
    ngram: str
    occurrences: int
    sample: list[dict[str, str]]

    @property
    def status(self) -> str:
        return "pending" if self.occurrences > 0 else "skipped"

    def as_payload(self) -> dict[str, Any]:
        return {
            "ngram": self.ngram,
            "occurrences": self.occurrences,
            "sample": self.sample,
        }


class _PreanalysisManager:
    """Internal helper that writes pre-analysis state to SQLite."""

    _STAGE_NOTES: dict[Stage, str] = {
        "initial": "Seeded trusted n-grams before the first analytics pass",
        "subsequent": "Linked trusted n-grams to a translation run",
    }

    def __init__(self, *, db_path: Path, ngram_index: Path, dictionary_path: Path) -> None:
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self.ngram_conn = sqlite3.connect(str(ngram_index))
        self.ngram_conn.row_factory = sqlite3.Row
        self.gloss_map = self._build_gloss_map(dictionary_path)

    def close(self) -> None:
        self.conn.close()
        self.ngram_conn.close()

    def _build_gloss_map(self, dictionary_path: Path) -> dict[str, str]:
        entries = load_dictionary(str(dictionary_path))
        gloss_map: dict[str, str] = {}
        for entry in entries:
            canonical = str(entry.get("canonical", "")).strip().lower()
            if not canonical:
                continue
            gloss = ""
            senses = entry.get("senses") or []
            for sense in senses:
                if not isinstance(sense, dict):
                    continue
                definition = str(sense.get("definition", "")).strip()
                if definition:
                    gloss = definition
                    break
            if not gloss:
                gloss = str(entry.get("enhanced_definition") or "").strip()
            if not gloss:
                gloss = str(entry.get("definition") or "").strip()
            gloss_map[canonical] = gloss
        return gloss_map

    def _snapshot(self, ngram: str) -> _SeedSnapshot:
        norm = ngram.lower()
        occ_row = self.ngram_conn.execute(
            "SELECT total_occurrences FROM ngrams WHERE ngram = ?",
            (norm,),
        ).fetchone()
        occurrences = int(occ_row[0]) if occ_row and occ_row[0] is not None else 0
        sample: list[dict[str, str]] = []
        cursor = self.ngram_conn.execute(
            """
            SELECT canonical
            FROM ngram_membership
            WHERE ngram = ?
            ORDER BY canonical
            LIMIT 8
            """,
            (norm,),
        )
        for row in cursor.fetchall():
            canonical = str(row[0]).strip()
            if not canonical:
                continue
            gloss = self.gloss_map.get(canonical.lower(), "")
            sample.append(
                {
                    "canonical": canonical.upper(),
                    "gloss": gloss,
                }
            )
        return _SeedSnapshot(ngram=ngram.upper(), occurrences=occurrences, sample=sample)

    def _create_run_row(self, *, stage: Stage) -> str:
        run_id = uuid.uuid4().hex
        run_name = f"preanalysis-{stage}-{_utcnow()}"
        env_json = json.dumps({"stage": stage})
        self.conn.execute(
            """
            INSERT INTO runs (run_id, run_name, engine, embedder, env_json, phase)
            VALUES (?, ?, 'solo', NULL, ?, 'preanalysis')
            """,
            (run_id, run_name, env_json),
        )
        return run_id

    def _ensure_run_id(self, *, stage: Stage, run_id: str | None) -> str:
        if stage == "initial":
            return self._create_run_row(stage=stage)
        if run_id:
            row = self.conn.execute(
                "SELECT run_id FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if not row:
                raise ValueError(f"Run id {run_id} not found in runs table")
            return str(row["run_id"])
        row = self.conn.execute(
            """
            SELECT run_id
            FROM runs
            WHERE phase = 'translation'
            ORDER BY datetime(created_at) DESC
            LIMIT 1
            """,
        ).fetchone()
        if not row:
            raise ValueError("No translation runs available to attach pre-analysis")
        return str(row["run_id"])

    def _get_existing(self, *, stage: Stage, run_id: str) -> sqlite3.Row | None:
        if stage == "initial":
            return self.conn.execute(
                """
                SELECT preanalysis_id, run_id
                FROM preanalysis_runs
                WHERE stage = 'initial'
                ORDER BY preanalysis_id
                LIMIT 1
                """,
            ).fetchone()
        return self.conn.execute(
            """
            SELECT preanalysis_id, run_id
            FROM preanalysis_runs
            WHERE stage = 'subsequent' AND run_id = ?
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()

    def _upsert_seeds(
        self,
        *,
        preanalysis_id: int,
        trusted: Sequence[str],
        refresh: bool,
    ) -> list[_SeedSnapshot]:
        now = _utcnow()
        snapshots = [self._snapshot(item) for item in trusted]
        for snap in snapshots:
            payload = json.dumps(snap.as_payload())
            status = snap.status
            self.conn.execute(
                """
                INSERT INTO preanalysis_seeds (
                    preanalysis_id, ngram, status, analytics_payload, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(preanalysis_id, ngram) DO UPDATE SET
                    analytics_payload = excluded.analytics_payload,
                    status = CASE
                        WHEN preanalysis_seeds.status = 'completed' THEN preanalysis_seeds.status
                        ELSE excluded.status
                    END,
                    updated_at = excluded.updated_at
                """,
                (preanalysis_id, snap.ngram, status, payload, now, now),
            )
            if refresh:
                self.conn.execute(
                    """
                    UPDATE preanalysis_seeds
                    SET status = CASE
                        WHEN status = 'skipped' AND ? > 0 THEN 'pending'
                        ELSE status
                    END,
                        updated_at = ?
                    WHERE preanalysis_id = ? AND ngram = ?
                    """,
                    (snap.occurrences, now, preanalysis_id, snap.ngram),
                )
        return snapshots

    def ensure_stage(
        self,
        *,
        stage: Stage,
        trusted: Sequence[str],
        run_id: str | None,
        refresh: bool,
    ) -> dict[str, Any]:
        trusted_norm = _normalize_trusted(trusted)
        if not trusted_norm:
            raise ValueError("Trusted n-gram list is empty")

        existing = None
        created = False
        run_for_stage = run_id

        if stage == "initial":
            existing = self._get_existing(stage=stage, run_id="")
            if existing and not refresh:
                run_for_stage = str(existing["run_id"])
                pre_id = int(existing["preanalysis_id"])
                snapshots = self._upsert_seeds(
                    preanalysis_id=pre_id,
                    trusted=trusted_norm,
                    refresh=False,
                )
                self.conn.commit()
                return {
                    "stage": stage,
                    "created": False,
                    "run_id": run_for_stage,
                    "preanalysis_id": pre_id,
                    "trusted_count": len(trusted_norm),
                    "snapshots": [snap.as_payload() for snap in snapshots],
                }
            run_for_stage = self._create_run_row(stage=stage)
        else:
            run_for_stage = self._ensure_run_id(stage=stage, run_id=run_id)
            existing = self._get_existing(stage=stage, run_id=run_for_stage)

        if existing:
            pre_id = int(existing["preanalysis_id"])
            self.conn.execute(
                "UPDATE preanalysis_runs SET trusted_count = ? WHERE preanalysis_id = ?",
                (len(trusted_norm), pre_id),
            )
            snapshots = self._upsert_seeds(
                preanalysis_id=pre_id,
                trusted=trusted_norm,
                refresh=refresh,
            )
        else:
            created = True
            notes = self._STAGE_NOTES.get(stage, "")
            cur = self.conn.execute(
                """
                INSERT INTO preanalysis_runs (run_id, stage, trusted_count, notes)
                VALUES (?, ?, ?, ?)
                """,
                (run_for_stage, stage, len(trusted_norm), notes),
            )
            pre_id = int(cur.lastrowid)
            snapshots = self._upsert_seeds(
                preanalysis_id=pre_id,
                trusted=trusted_norm,
                refresh=False,
            )

        self.conn.commit()
        return {
            "stage": stage,
            "created": created,
            "run_id": run_for_stage,
            "preanalysis_id": pre_id,
            "trusted_count": len(trusted_norm),
            "snapshots": [snap.as_payload() for snap in snapshots],
        }


def execute_preanalysis(
    *,
    db_path: str | Path,
    stage: Stage = "initial",
    trusted_ngrams: Sequence[str] | None = None,
    trusted_path: str | Path | None = None,
    run_id: str | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    """Apply the pre-analysis safeguards for ``stage`` to ``db_path``."""

    if trusted_ngrams is None:
        trusted_ngrams = load_trusted_ngrams(trusted_path)

    paths = get_config_paths()
    manager = _PreanalysisManager(
        db_path=Path(db_path),
        ngram_index=Path(paths["ngram_index"]),
        dictionary_path=Path(paths["dictionary"]),
    )
    try:
        result = manager.ensure_stage(
            stage=stage,
            trusted=trusted_ngrams,
            run_id=run_id,
            refresh=refresh,
        )
    finally:
        manager.close()
    return result


def fetch_preanalysis_summary(
    conn: sqlite3.Connection,
    *,
    root: str,
) -> dict[str, list[str]]:
    """Return summary/focus lines for ``root`` from stored pre-analysis seeds."""
    token = str(root or "").strip().upper()
    if not token:
        return {}

    rows = conn.execute(
        """
        SELECT ps.analytics_payload, ps.status, pr.stage, pr.created_at
        FROM preanalysis_seeds AS ps
        JOIN preanalysis_runs AS pr
          ON pr.preanalysis_id = ps.preanalysis_id
        WHERE ps.ngram = ?
        ORDER BY datetime(pr.created_at) ASC
        """,
        (token,),
    ).fetchall()

    if not rows:
        return {}

    summary_lines: list[str] = []
    focus_lines: list[str] = []

    for row in rows:
        payload_raw = row["analytics_payload"]
        status = row["status"]
        stage = row["stage"]
        created_at = row["created_at"]
        payload: dict[str, Any] = {}
        if payload_raw:
            try:
                payload = json.loads(payload_raw)
            except json.JSONDecodeError:
                payload = {}
        occurrences = payload.get("occurrences")
        try:
            occ_int = int(occurrences)
        except (TypeError, ValueError):
            occ_int = None
        sample = payload.get("sample") or []
        preview_items: list[str] = []
        for item in sample[:3]:
            if not isinstance(item, dict):
                continue
            canon = str(item.get("canonical", "")).strip()
            gloss = str(item.get("gloss", "")).strip()
            if canon and gloss:
                preview_items.append(f"{canon}: {gloss}")
            elif canon:
                preview_items.append(canon)
        preview = "; ".join(preview_items)
        headline = f"[Preanalysis:{stage}] {token}"
        if occ_int is not None:
            headline += f" occurrences={occ_int}"
        if preview:
            headline += f" → {preview}"
        if status == "completed":
            headline += " [consumed]"
        summary_lines.append(headline)
        if status != "completed" and occ_int:
            focus_lines.append(
                f"Preanalysis snapshot ({created_at}) recommends anchoring {token} with {preview or 'trusted canonicals'} "
                "before escalating prompts."
            )

    return {
        "summary_lines": summary_lines,
        "focus_lines": focus_lines,
    }


def mark_preanalysis_consumed(
    conn: sqlite3.Connection,
    *,
    root: str,
    run_id: str | None,
) -> None:
    """Mark ``root`` as consumed by ``run_id`` in the pre-analysis tables."""
    token = str(root or "").strip().upper()
    if not token:
        return
    now = _utcnow()
    conn.execute(
        """
        UPDATE preanalysis_seeds
        SET status = CASE
                WHEN status <> 'skipped' THEN 'completed'
                ELSE status
            END,
            last_run_id = CASE WHEN ? IS NOT NULL THEN ? ELSE last_run_id END,
            updated_at = ?
        WHERE ngram = ?
        """,
        (run_id, run_id, now, token),
    )
    conn.commit()


__all__ = [
    "execute_preanalysis",
    "fetch_preanalysis_summary",
    "load_trusted_ngrams",
    "mark_preanalysis_consumed",
]
