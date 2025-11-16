"""Compatibility shim that ensures the shared SQLite bootstrap runs."""
from __future__ import annotations

from enochian_common import sqlite_bootstrap as _sqlite_bootstrap  # noqa: F401

__all__ = []
