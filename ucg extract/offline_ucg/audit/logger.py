"""
audit/logger.py - append-only decision log in audit/decisions.db (separate from the clinical database).

Every triage run stores: input, parsed case, red flags, retrieval hits, tool-call trace, output, and the versions
(pipeline, rules fingerprint, embedder, LLM) needed to reproduce it.  `feedback` lets a clinician mark a decision
agreed / disagreed with a comment - that is the raw material for reviewing and signing off rules.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone

import db as dbmod

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    decision_id   TEXT PRIMARY KEY,
    ts            TEXT NOT NULL,
    input_text    TEXT,                 -- NULL when audit_store_raw_text=False
    input_sha256  TEXT NOT NULL,
    case_json     TEXT NOT NULL,
    red_flags_json TEXT NOT NULL,
    retrieval_json TEXT NOT NULL,
    tool_trace_json TEXT NOT NULL,
    output_json   TEXT NOT NULL,
    versions_json TEXT NOT NULL,
    latency_ms    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions(ts);
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id TEXT NOT NULL REFERENCES decisions(decision_id),
    ts          TEXT NOT NULL,
    verdict     TEXT NOT NULL CHECK (verdict IN ('agree','disagree','unsure')),
    comment     TEXT,
    reviewer    TEXT
);
"""


class AuditLog:
    def __init__(self, path, store_raw_text: bool = True):
        self.con = dbmod.connect(path)
        self.con.executescript(SCHEMA)
        self.store_raw = store_raw_text

    def record(self, text: str, case: dict, red_flags: list, retrieval: dict, tool_trace: list, output: dict,
               versions: dict, latency_ms: float) -> str:
        did = uuid.uuid4().hex
        dump = lambda o: json.dumps(o, default=str, ensure_ascii=False)   # noqa: E731
        self.con.execute("INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                         (did, datetime.now(timezone.utc).isoformat(timespec="seconds"),
                          text if self.store_raw else None, hashlib.sha256((text or "").encode()).hexdigest(),
                          dump(case), dump(red_flags), dump(retrieval), dump(tool_trace), dump(output), dump(versions),
                          round(latency_ms, 1)))
        self.con.commit()
        return did

    def get(self, decision_id: str) -> dict | None:
        r = self.con.execute("SELECT * FROM decisions WHERE decision_id=?", (decision_id,)).fetchone()
        if r is None:
            return None
        d = dict(r)
        for k in [k for k in d if k.endswith("_json")]:
            d[k[:-5]] = json.loads(d.pop(k))
        d["feedback"] = [dict(f) for f in self.con.execute("SELECT * FROM feedback WHERE decision_id=?", (decision_id,))]
        return d

    def recent(self, n: int = 20) -> list[dict]:
        return [dict(r) for r in self.con.execute(
            "SELECT decision_id, ts, substr(coalesce(input_text,'(hash only)'),1,120) AS input, latency_ms "
            "FROM decisions ORDER BY ts DESC LIMIT ?", (n,))]

    def add_feedback(self, decision_id: str, verdict: str, comment: str = "", reviewer: str = "") -> int:
        if self.get(decision_id) is None:
            raise KeyError(decision_id)
        cur = self.con.execute("INSERT INTO feedback(decision_id,ts,verdict,comment,reviewer) VALUES (?,?,?,?,?)",
                               (decision_id, datetime.now(timezone.utc).isoformat(timespec="seconds"), verdict, comment, reviewer))
        self.con.commit()
        return cur.lastrowid
