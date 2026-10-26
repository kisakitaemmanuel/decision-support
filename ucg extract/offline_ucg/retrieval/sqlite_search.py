"""retrieval/sqlite_search.py - lexical retrieval over chunks_fts (FTS5 + bm25) and name autocomplete."""
from __future__ import annotations

import logging
import re
import sqlite3

from schemas import ChunkHit

log = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
STOP = set("a an the of and or in on at to for with is are was were be been it its this that as by from into has have "
           "had not no patient presents".split())


def tokens(q: str, keep_stop: bool = False, limit: int = 12) -> list[str]:
    toks = [t.lower() for t in TOKEN_RE.findall(q or "")]
    if not keep_stop:
        toks = [t for t in toks if t not in STOP and len(t) > 1]
    return toks[:limit]


class FTSSearcher:
    def __init__(self, con: sqlite3.Connection):
        self.con = con

    def _run(self, match: str, k: int) -> list[sqlite3.Row]:
        return self.con.execute(
            """SELECT c.chunk_id, c.disease_name, c.section, c.text, c.page, -bm25(chunks_fts, 6.0, 1.0, 1.0) AS score
               FROM chunks_fts JOIN chunks c ON c.chunk_id = chunks_fts.rowid
               WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts, 6.0, 1.0, 1.0) LIMIT ?""", (match, k)).fetchall()

    def search(self, query: str, k: int = 30, prefix_last: bool = False) -> list[ChunkHit]:
        toks = tokens(query)
        if not toks:
            return []
        quoted = [f'"{t}"' for t in toks]
        if prefix_last:
            quoted[-1] = f'"{toks[-1]}"*'
        rows = self._run(" ".join(quoted), k)                       # strict AND first
        if len(rows) < max(3, k // 3) and len(toks) > 1:            # relax to OR when AND is too thin
            seen = {r["chunk_id"] for r in rows}
            for r in self._run(" OR ".join(quoted), k):
                if r["chunk_id"] not in seen:
                    rows.append(r)
                    seen.add(r["chunk_id"])
            rows = rows[:k]
        return [ChunkHit(r["chunk_id"], r["disease_name"], r["section"], r["text"], r["page"], float(r["score"]),
                         i + 1, "fts") for i, r in enumerate(rows)]

    def suggest(self, prefix: str, limit: int = 8) -> list[dict]:
        toks = tokens(prefix, keep_stop=True)
        if not toks:
            return []
        match = " ".join(f'"{t}"*' for t in toks)
        rows = self.con.execute(
            """SELECT d.disease_name, d.icd10_code FROM disease_names_fts f JOIN diseases d ON d.rowid=f.rowid
               WHERE disease_names_fts MATCH ?
               ORDER BY (lower(d.disease_name) LIKE lower(?)||'%') DESC, bm25(disease_names_fts), length(d.disease_name)
               LIMIT ?""", (match, prefix.strip(), limit)).fetchall()
        return [dict(r) for r in rows]
