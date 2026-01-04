"""Normalize ``sqlite3`` to ``pysqlite3`` when the enhanced runtime is available."""
from __future__ import annotations

import sqlite3 as _sqlite_std
from collections.abc import Iterable

if not getattr(_sqlite_std, "__enochian_bootstrapped__", False):
    try:
        from pysqlite3 import dbapi2 as _sqlite_new
    except Exception:  # pragma: no cover - fallback to stdlib behavior
        _sqlite_new = None
    else:
        def _copy_attr(names: Iterable[str]) -> None:
            for name in names:
                try:
                    value = getattr(_sqlite_new, name)
                except AttributeError:
                    continue
                setattr(_sqlite_std, name, value)

        _copy_attr(
            (
                "connect",
                "Connection",
                "Cursor",
                "Row",
                "complete_statement",
                "enable_callback_tracebacks",
                "register_adapter",
                "register_converter",
            )
        )
        _copy_attr(("sqlite_version", "sqlite_version_info", "version"))
        setattr(_sqlite_std, "__enochian_bootstrapped__", True)

sqlite3 = _sqlite_std

__all__ = ["sqlite3"]
