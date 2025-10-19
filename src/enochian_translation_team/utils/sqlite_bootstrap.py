"""
Swap stdlib sqlite3.connect to pysqlite3.dbapi2.connect, if available.
Keeps stdlib types/annotations working while giving you JSON1 at runtime.
"""
import sqlite3 as _sqlite_std

try:
    # pysqlite3-binary must be in your project deps
    from pysqlite3 import dbapi2 as _sqlite_new

    # Replace connect (and expose version) — silence type checker on assignment
    _sqlite_std.connect = _sqlite_new.connect  # type: ignore[assignment, attr-defined]
    try:
        _sqlite_std.sqlite_version = _sqlite_new.sqlite_version  # type: ignore[attr-defined]
    except Exception:
        pass
except Exception:
    # Fall back to stdlib sqlite3 if pysqlite3 is missing
    pass