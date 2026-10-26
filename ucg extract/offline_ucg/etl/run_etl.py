#!/usr/bin/env python3
"""
etl/run_etl.py - build database/ucg.db for the offline app.

    python -m etl.run_etl --source ../ucg/ucg_2023.db            # from the DB produced by extract_pdf.py
    python -m etl.run_etl --pdf UCG-2023-Publication-Final-PDF.pdf   # from the PDF (runs extract_pdf.py first)
    python -m etl.run_etl --source X --no-embed                  # skip vectors (FTS-only)
    python -m etl.run_etl --source X --only validate             # re-run a single stage on an existing DB

Stages (each logged, timed, recorded in table etl_runs; any failure aborts with a clear message):
    load_schema  ->  load_base  ->  chunks  ->  drugs  ->  rules  ->  embeddings  ->  finalize  ->  validate
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import db as dbmod                                   # noqa: E402
from config import get_settings, setup_logging       # noqa: E402
from etl.chunker import chunk_disease                # noqa: E402
from etl.drugs import find_drugs                     # noqa: E402

log = logging.getLogger("etl")
DISEASE_COLS = ["disease_name", "entry_id", "section_number", "icd10_code", "chapter_number", "category",
                "parent_condition", "description", "causes", "risk_factors", "clinical_features",
                "differential_diagnosis", "investigations", "management", "prevention", "complications",
                "levels_of_care", "start_pdf_page", "end_pdf_page"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def stage(con: sqlite3.Connection, name: str):
    t0 = time.perf_counter()
    started = now()
    log.info("[%s] start", name)
    ctx = {"rows": None, "detail": ""}
    try:
        yield ctx
    except Exception as e:
        con.execute("INSERT INTO etl_runs(stage,started_at,finished_at,rows,status,detail) VALUES (?,?,?,?,?,?)",
                    (name, started, now(), ctx["rows"], "failed", repr(e)))
        con.commit()
        log.error("[%s] FAILED: %s", name, e)
        raise
    con.execute("INSERT INTO etl_runs(stage,started_at,finished_at,rows,status,detail) VALUES (?,?,?,?,?,?)",
                (name, started, now(), ctx["rows"], "ok", ctx["detail"]))
    con.commit()
    log.info("[%s] ok  rows=%s  %.2fs  %s", name, ctx["rows"], time.perf_counter() - t0, ctx["detail"])


# ---------------------------------------------------------------------------------------------- stages
def stage_load_schema(con, s):
    con.executescript(s.schema_path.read_text(encoding="utf-8"))


def stage_load_base(con, s, source: Path, ctx):
    con.execute("ATTACH DATABASE ? AS src", (str(source),))
    have = {r[0] for r in con.execute("SELECT name FROM src.sqlite_master WHERE type='table'")}
    for t in ("diseases", "chapters"):
        if t not in have:
            raise RuntimeError(f"source database lacks table {t!r}")
    con.execute("INSERT INTO chapters SELECT chapter_number, chapter_title FROM src.chapters")
    con.execute(f"INSERT INTO diseases({','.join(DISEASE_COLS)}) SELECT {','.join(DISEASE_COLS)} FROM src.diseases")
    ctx["rows"] = con.execute("SELECT count(*) FROM diseases").fetchone()[0]
    con.commit()
    con.execute("DETACH DATABASE src")


def stage_chunks(con, s, ctx, max_chars: int):
    rows = con.execute("SELECT * FROM diseases ORDER BY rowid").fetchall()
    batch = [(d["disease_name"], sec, idx, text, page) for d in rows for sec, idx, text, page in chunk_disease(d, max_chars)]
    con.executemany("INSERT INTO chunks(disease_name, section, chunk_idx, text, page) VALUES (?,?,?,?,?)", batch)
    con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    con.execute("INSERT INTO disease_names_fts(disease_names_fts) VALUES('rebuild')")
    ctx["rows"] = len(batch)
    ctx["detail"] = f"max_chars={max_chars}"


def stage_drugs(con, s, ctx):
    rows = []
    for d in con.execute("SELECT disease_name, management FROM diseases WHERE management IS NOT NULL"):
        for drug, snip in find_drugs(d["management"]).items():
            rows.append((d["disease_name"], drug, snip))
    con.executemany("INSERT OR IGNORE INTO disease_drugs VALUES (?,?,?)", rows)
    ctx["rows"] = len(rows)


def stage_rules(con, s, ctx):
    from rules import seed_all
    counts = seed_all(con)
    ctx["rows"] = sum(counts.values())
    ctx["detail"] = str(counts)


def stage_embeddings(con, s, ctx):
    from retrieval.embedders import make_embedder
    from retrieval.semantic_search import build_index
    emb = make_embedder(s)
    ctx["rows"] = build_index(con, emb, s)
    ctx["detail"] = f"embedder={emb.name} dim={emb.dim}"


def stage_finalize(con, s, ctx):
    con.execute("INSERT OR REPLACE INTO meta VALUES ('built_at', ?)", (now(),))
    con.execute("INSERT OR REPLACE INTO meta VALUES ('pipeline_version', ?)", (s.pipeline_version,))
    con.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', '1')")
    con.commit()
    con.execute("ANALYZE")


def stage_validate(con, s, ctx, expect_vectors: bool):
    problems: list[str] = []

    def one(sql):
        return con.execute(sql).fetchone()[0]

    n_dis, n_chunk = one("SELECT count(*) FROM diseases"), one("SELECT count(*) FROM chunks")
    if n_dis < 50:
        problems.append(f"only {n_dis} diseases")
    if one("SELECT count(*) FROM diseases d WHERE NOT EXISTS (SELECT 1 FROM chunks c WHERE c.disease_name=d.disease_name)"):
        problems.append("diseases without any chunk")
    if one("SELECT count(*) FROM chunks_fts") != n_chunk:
        problems.append("chunks_fts row count != chunks")
    if con.execute("PRAGMA foreign_key_check").fetchall():
        problems.append("foreign key violations")
    if con.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        problems.append("integrity_check failed")
    for t in ("red_flag_rules", "contraindication_rules", "dose_rules"):
        if one(f"SELECT count(*) FROM {t}") == 0:
            problems.append(f"{t} is empty")
    # rule JSON must parse and use known keys
    from rules.engine import evaluate, load_condition
    from schemas import PatientCase
    for t, col in (("red_flag_rules", "condition_json"), ("contraindication_rules", "condition_json")):
        for r in con.execute(f"SELECT rule_id, {col} FROM {t}"):
            try:
                evaluate(load_condition(r[1]), PatientCase())
            except Exception as e:                       # noqa: BLE001
                problems.append(f"{t}.{r[0]}: {e}")
    if expect_vectors:
        meta = con.execute("SELECT n_vectors FROM vector_index_meta WHERE id=1").fetchone()
        if meta is None or meta[0] != n_chunk:
            problems.append("vector count != chunk count")
    if problems:
        raise RuntimeError("validation failed: " + "; ".join(problems))
    ctx["rows"] = n_dis
    ctx["detail"] = f"diseases={n_dis} chunks={n_chunk}"


# ---------------------------------------------------------------------------------------------- driver
def run(source: Path | None, pdf: Path | None, out: Path, embed: bool = True, max_chars: int = 900) -> Path:
    s = get_settings()
    setup_logging()
    tmp = None
    if pdf:
        from etl import extract_pdf
        tmp = Path(tempfile.mkdtemp()) / "extracted.db"
        log.info("extracting PDF -> %s", tmp)
        extract_pdf.build(str(pdf), str(tmp), None)
        source = tmp
    if not source or not Path(source).exists():
        raise SystemExit("need --source <ucg_2023.db> or --pdf <file.pdf>")

    out = Path(out)
    for p in (out, s.npz_path, s.faiss_path):
        if Path(p).exists():
            Path(p).unlink()
    con = dbmod.connect(out)
    try:
        with stage(con, "load_schema") as c:
            stage_load_schema(con, s)
        with stage(con, "load_base") as c:
            stage_load_base(con, s, Path(source), c)
        with stage(con, "chunks") as c:
            stage_chunks(con, s, c, max_chars)
        with stage(con, "drugs") as c:
            stage_drugs(con, s, c)
        with stage(con, "rules") as c:
            stage_rules(con, s, c)
        if embed:
            with stage(con, "embeddings") as c:
                stage_embeddings(con, s, c)
        with stage(con, "finalize") as c:
            stage_finalize(con, s, c)
        with stage(con, "validate") as c:
            stage_validate(con, s, c, expect_vectors=embed)
        con.execute("VACUUM")
    finally:
        con.close()
    log.info("done -> %s (%.1f MB)", out, out.stat().st_size / 1e6)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path)
    ap.add_argument("--pdf", type=Path)
    ap.add_argument("--out", type=Path, default=get_settings().db_path)
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--max-chars", type=int, default=900)
    a = ap.parse_args()
    run(a.source, a.pdf, a.out, embed=not a.no_embed, max_chars=a.max_chars)


if __name__ == "__main__":
    main()
