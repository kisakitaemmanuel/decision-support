"""db.py - the only place that opens SQLite connections (consistent pragmas + row factory)."""
from __future__ import annotations

import sqlite3
from pathlib import Path


def connect(path: Path | str, readonly: bool = False) -> sqlite3.Connection:
    path = Path(path)
    if readonly:
        if not path.exists():
            raise FileNotFoundError(f"Database not found: {path}  (run: python -m etl.run_etl)")
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(path, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con
