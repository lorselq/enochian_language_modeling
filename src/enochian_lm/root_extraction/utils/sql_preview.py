from __future__ import annotations

import datetime
from pathlib import Path

from enochian_lm.common.sqlite_bootstrap import sqlite3


class SQLPreviewRecorder:
    """Capture executed SQL statements and persist them to a local log file."""

    def __init__(self, *, label: str) -> None:
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"sql_preview_{label}_{timestamp}.sql"
        self._statements: list[str] = []

    def trace(self, statement: str) -> None:
        cleaned = (statement or "").strip()
        if cleaned:
            self._statements.append(cleaned)

    def flush(self) -> str:
        body = "\n".join(f"{statement};" for statement in self._statements)
        with self.log_path.open("w", encoding="utf-8") as handle:
            handle.write(body + "\n" if body else "-- No SQL statements recorded.\n")
        return str(self.log_path)


def connect_preview_db(db_path: Path, *, label: str) -> tuple[sqlite3.Connection, SQLPreviewRecorder]:
    """Clone a SQLite database into memory and attach SQL trace logging."""

    source = sqlite3.connect(db_path)
    preview = sqlite3.connect(":memory:")
    source.backup(preview)
    source.close()
    recorder = SQLPreviewRecorder(label=label)
    preview.set_trace_callback(recorder.trace)
    return preview, recorder

