#!/usr/bin/env python3
"""
build_db.py - Uganda Clinical Guidelines 2023 (PDF) -> SQLite knowledge base

Pipeline
  1. Load page text        (pdftotext by default, or a Docling JSON export via --docling-json)
  2. Parse the printed Table of Contents -> numbered entries (section_number, title)
  3. Locate each entry's heading in the body text (sequentially, by section number)
  4. Split each entry into sections (clinical features, investigations, ...)
  5. Write SQLite: raw_pages, raw_entries, chapters, diseases (master),
     indexes, FTS5 tables (+ sync triggers)
  6. Export CSV of the master table (pandas) and print QA stats

Usage
  python build_db.py UCG-2023.pdf -o ucg_2023.db
  python build_db.py UCG-2023.pdf --docling-json ucg.json -o ucg_2023.db
"""
import argparse
import difflib
import json
import re
import sqlite3
import subprocess
import sys
import unicodedata
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# 1. Page loading
# --------------------------------------------------------------------------

def load_pages_pdftotext(pdf_path: str) -> list[str]:
    out = subprocess.run(["pdftotext", "-layout", pdf_path, "-"], capture_output=True, check=True).stdout
    text = out.decode("utf-8", errors="ignore")
    pages = text.split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    return pages


def load_pages_docling(json_path: str) -> list[str]:
    """Rebuild per-page text from a Docling JSON export (DoclingDocument.export_to_dict()).

    Reads the 'texts' array (reading order) and groups by provenance page number.
    Page furniture labels are skipped.
    """
    doc = json.loads(Path(json_path).read_text(encoding="utf-8"))
    per_page: dict[int, list[str]] = {}
    for item in doc.get("texts", []):
        if item.get("label") in {"page_header", "page_footer"}:
            continue
        prov = item.get("prov") or []
        if not prov:
            continue
        per_page.setdefault(prov[0]["page_no"], []).append(item.get("text", ""))
    n = max(per_page) if per_page else 0
    return ["\n".join(per_page.get(i, [])) for i in range(1, n + 1)]


# --------------------------------------------------------------------------
# 2. Text cleaning helpers
# --------------------------------------------------------------------------

FURNITURE = [
    re.compile(r"^Uganda Clinical Guidelines 2023$", re.I),
    re.compile(r"^CHAPTER\s+\d+\s*:.*$", re.I),
    re.compile(r"^\d{1,4}$"),                    # arabic page numbers
    re.compile(r"^[IVXLC]{1,7}$"),               # roman page numbers
    re.compile(r"^(TREATMENT|LOC|TREATMENT\s+LOC)$"),
]
LOC_RE = re.compile(r"^(HC[1-4]|H|RR|NR|NH|N/A|VHT|HC\s?[1-4])(\s*,\s*(HC[1-4]|H|RR|NR))*$")
TRAIL_LOC_RE = re.compile(r"\s{3,}((?:HC\s?[1-4]|H|RR|NR|NH)(?:\s*,\s*(?:HC\s?[1-4]|H|RR|NR))*)\s*$")
BULLET_RE = re.compile(r"^(~|¾|•|●|▪|\u2022|[\x80-\x9f]|[\ue000-\uf8ff])\s*")
INLINE_BULLET_RE = re.compile(r"\s*(?:¾|[\x80-\x9f]|[\ue000-\uf8ff])\s*")


def is_furniture(s: str) -> bool:
    return any(p.match(s) for p in FURNITURE)


def normalise_line(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).replace("\u00ad", "")
    s = s.strip()
    return s


def to_bullets(s: str) -> str:
    return BULLET_RE.sub("• ", s)


def tidy_block(lines: list[str]) -> str:
    """Clean a list of raw lines into readable text (bullets kept, wraps joined)."""
    out: list[str] = []
    expanded: list[str] = []
    for raw in lines:  # bullets glued mid-line by the layout engine -> own lines
        expanded.extend(INLINE_BULLET_RE.sub("\n• ", raw).split("\n"))
    for raw in expanded:
        s = normalise_line(raw)
        if not s or is_furniture(s):
            continue
        s = re.sub(r"\s{2,}", " ", to_bullets(s))
        if out:
            prev = out[-1]
            # de-hyphenate "rel-" + "evant"
            if re.search(r"[a-z]-$", prev) and re.match(r"^[a-z]", s):
                out[-1] = prev[:-1] + s
                continue
            # join wrapped continuation into previous line
            if (not s.startswith("• ")) and re.match(r"^[a-z0-9(\[%<>/]", s) and not prev.endswith((".", ":")):
                out[-1] = prev + " " + s
                continue
        out.append(s)
    return "\n".join(out).strip()


# --------------------------------------------------------------------------
# 3. TOC parsing and heading location
# --------------------------------------------------------------------------

TOC_LINE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s*([A-Za-z(\"“][^\n]*?)\s*(\.{2,}|\.\s\.{2,}).*$")


def parse_toc(pages: list[str]):
    """Return (entries, body_start_page_index). entries: list of dicts in TOC order."""
    dotted_per_page = [sum(1 for l in p.split("\n") if re.search(r"\.{4,}", l)) for p in pages[:120]]
    toc_pages = [i for i, c in enumerate(dotted_per_page) if c >= 8]
    body_start = (max(toc_pages) + 1) if toc_pages else 0

    entries, seen = [], {}
    for i in toc_pages:
        for line in pages[i].split("\n"):
            line = line.strip()
            m = TOC_LINE.match(line)
            if not m:
                continue
            num, title = m.group(1), m.group(2)
            title = re.sub(r"\s*\.+\s*$", "", title).strip()
            toc_page = re.search(r"(\d+|[IVXLC]+)\s*$", line)
            if len(title) < 3:
                continue
            seen[num] = seen.get(num, 0) + 1
            entry_id = num if seen[num] == 1 else f"{num}#{seen[num]}"
            entries.append({"entry_id": entry_id, "section_number": num, "title": title,
                            "toc_page": toc_page.group(1) if toc_page else None})
    return entries, body_start


def sort_key(num: str):
    return tuple(int(x) for x in num.split("."))


def locate_headings(entries, pages, body_start):
    """Find each entry's heading in the body.

    Collect every candidate line per section number (number + fuzzy-matching title), then choose one
    candidate per entry so that positions increase with section order (longest increasing subsequence).
    This tolerates stray look-alike lines and TOC lines that are out of order in the extracted text.
    Headings split across two lines ("4.1.9" / "Rheumatic Fever") are joined.
    """
    flat = []  # (page_idx, line_idx, text)
    for pi in range(body_start, len(pages)):
        for li, line in enumerate(pages[pi].split("\n")):
            flat.append((pi, li, line.strip()))

    def norm(t):
        return re.sub(r"[^a-z0-9]", "", t.lower())

    head_re = re.compile(r"^(\d+(?:\.\d+)*)\.?\s*([A-Za-z(\"\u201c].*)$")
    num_only = re.compile(r"^(\d+(?:\.\d+)+)\.?$")
    by_num = {}
    for k, (_, _, text) in enumerate(flat):
        if len(text) > 160 or "...." in text:
            continue
        skip = 0
        m = head_re.match(text)
        if not m:
            mo = num_only.match(text)
            if not mo:
                continue
            nxt = None
            for j in range(k + 1, min(k + 4, len(flat))):
                if flat[j][2]:
                    nxt = j
                    break
            if nxt is None or not re.match(r"^[A-Za-z(\"\u201c]", flat[nxt][2]):
                continue
            m = head_re.match(mo.group(1) + " " + flat[nxt][2])
            if not m:
                continue
            skip = nxt - k
        by_num.setdefault(m.group(1), []).append((k, skip, m.group(2)))

    cands = []  # (entry_idx, pos, skip)
    for ei, e in enumerate(entries):
        tnorm = norm(e["title"])
        good = []
        for k, skip, rest in by_num.get(e["section_number"], []):
            cand = norm(re.sub(r"ICD\s*-?10.*$", "", rest, flags=re.I))
            if not cand:
                continue
            if difflib.SequenceMatcher(None, cand[:30], tnorm[:30]).ratio() >= 0.6:
                good.append((k, skip))
        # within one entry process positions in descending order so two candidates never chain
        for k, skip in sorted(good, reverse=True):
            cands.append((ei, k, skip))

    # LIS on pos with strictly increasing entry index
    import bisect
    tails, tail_ids, prev = [], [], {}
    for idx, (ei, k, skip) in enumerate(cands):
        j = bisect.bisect_left(tails, k)
        if j == len(tails):
            tails.append(k); tail_ids.append(idx)
        else:
            tails[j] = k; tail_ids[j] = idx
        prev[idx] = tail_ids[j - 1] if j > 0 else None
    chosen = {}
    cur = tail_ids[-1] if tail_ids else None
    while cur is not None:
        ei, k, skip = cands[cur]
        chosen[ei] = (k, skip)
        cur = prev[cur]

    for ei in range(len(entries)):
        if ei in chosen:
            continue
        lo = max([chosen[j][0] for j in chosen if j < ei], default=-1)
        hi = min([chosen[j][0] for j in chosen if j > ei], default=len(flat))
        tn = norm(re.sub(r"^\d+(\.\d+)*\.?\s*", "", entries[ei]["title"]))
        for k in range(lo + 1, hi):
            text = flat[k][2]
            if not text or len(text) > 120 or "...." in text:
                continue
            body = norm(re.sub(r"^\d+(\.\d+)*\.?\s*", "", re.sub(r"ICD\s*-?10.*$", "", text, flags=re.I)))
            if len(body) >= 4 and difflib.SequenceMatcher(None, body[:30], tn[:30]).ratio() >= 0.88:
                chosen[ei] = (k, 0)
                break

    for ei, e in enumerate(entries):
        if ei in chosen:
            e["pos"], e["skip"] = chosen[ei]
        else:
            e["pos"], e["skip"] = None, 0
    return flat


# --------------------------------------------------------------------------
# 4. Section splitting
# --------------------------------------------------------------------------

HEADINGS = {
    "description": "description",
    "causes": "causes", "cause": "causes",
    "risk factors": "risk_factors",
    "clinical features": "clinical_features", "clinical presentation": "clinical_features",
    "clinical features and diagnosis": "clinical_features",
    "differential diagnosis": "differential_diagnosis",
    "investigations": "investigations", "investigation": "investigations",
    "management": "management",
    "prevention": "prevention",
    "complications": "complications",
}
FIELDS = ["description", "causes", "risk_factors", "clinical_features", "differential_diagnosis",
          "investigations", "management", "prevention", "complications"]
ICD_RE = re.compile(r"ICD\s*-?\s*10\s*CODE\s*:?\s*([A-Z]\d{2}(?:\.\d+)?(?:\s*[-,/&]\s*[A-Z]?\d{1,2}(?:\.\d+)?)*)", re.I)


CF_RE = re.compile(r"^(clinical features|clinical presentation|signs and symptoms|symptoms and signs|"
                   r"common symptoms)\b", re.I)
INV_RE = re.compile(r"^(investigations?|laboratory (investigations?|diagnosis|tests?)|diagnostic tests?)\b", re.I)
MGMT_RE = re.compile(r"^(Management|Treatment) of [A-Za-z0-9 /,()\-]+$")
GEN_MGMT_RE = re.compile(r"^General (management|principles of management)$", re.I)


def variant_heading(s: str, buckets: dict):
    """Recognise heading variants such as 'Clinical Features of Malaria' or 'Common symptoms/signs of ...'.

    Only short, standalone, non-sentence lines qualify, so prose that merely starts with these words is ignored.
    The generic 'Management of ...' / 'Treatment of ...' form is accepted only while no management
    section has started (the printed guideline mislabels some, e.g. Rheumatic Fever).
    """
    if not s or len(s) > 70 or s.endswith((".", ",", ";")) or s.startswith(("•", "~")):
        return None
    words = len(s.split())
    if words <= 7:
        if CF_RE.match(s):
            return "clinical_features"
        if INV_RE.match(s):
            return "investigations"
        if s.lower() in ("diagnosis", "diagnosis:", "treatment"):
            return "investigations" if s.lower().startswith("diagnosis") else ("management" if s == "Treatment" else None)
        if GEN_MGMT_RE.match(s) or (not buckets["management"] and MGMT_RE.match(s)):
            return "management"
    return None


def split_sections(lines: list[str]):
    buckets: dict[str, list[str]] = {f: [] for f in FIELDS}
    loc_codes: list[str] = []
    current = "description"
    for raw in lines:
        s = normalise_line(raw)
        key = HEADINGS.get(s.lower().rstrip(":")) if s != "TREATMENT" else None
        if key is None:
            key = variant_heading(s, buckets)
        if key:
            current = key
            continue
        if current == "management":
            mt = TRAIL_LOC_RE.search(raw)
            if mt:
                for c in re.split(r"\s*,\s*", mt.group(1)):
                    c = c.replace(" ", "")
                    if c not in loc_codes:
                        loc_codes.append(c)
                raw = raw[:mt.start()]
                s = normalise_line(raw)
                if not s:
                    continue
        if current == "management" and LOC_RE.match(s):
            for c in re.split(r"\s*,\s*", s):
                if c not in loc_codes:
                    loc_codes.append(c.replace(" ", ""))
            continue
        buckets[current].append(raw)
    cleaned = {f: tidy_block(v) for f, v in buckets.items()}
    return cleaned, ", ".join(loc_codes)


# --------------------------------------------------------------------------
# 5. SQLite schema
# --------------------------------------------------------------------------

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE raw_pages (
    pdf_page     INTEGER PRIMARY KEY,
    text         TEXT NOT NULL
);

CREATE TABLE chapters (
    chapter_number INTEGER PRIMARY KEY,
    chapter_title  TEXT NOT NULL
);

CREATE TABLE raw_entries (
    entry_id        TEXT PRIMARY KEY,
    section_number  TEXT NOT NULL,          -- number as printed (the source repeats a few)
    parent_id       TEXT REFERENCES raw_entries(entry_id),
    level           INTEGER NOT NULL CHECK (level >= 1),
    chapter_number  INTEGER REFERENCES chapters(chapter_number),
    title           TEXT NOT NULL,
    toc_page        TEXT,
    start_pdf_page  INTEGER,
    end_pdf_page    INTEGER,
    raw_text        TEXT
);

CREATE TABLE diseases (
    disease_name          TEXT PRIMARY KEY NOT NULL CHECK (length(trim(disease_name)) > 0),
    entry_id              TEXT NOT NULL UNIQUE REFERENCES raw_entries(entry_id),
    section_number        TEXT NOT NULL,
    icd10_code            TEXT,
    chapter_number        INTEGER REFERENCES chapters(chapter_number),
    category              TEXT,
    parent_condition      TEXT,
    description           TEXT,
    causes                TEXT,
    risk_factors          TEXT,
    clinical_features     TEXT,
    differential_diagnosis TEXT,
    investigations        TEXT,
    management            TEXT,
    prevention            TEXT,
    complications         TEXT,
    levels_of_care        TEXT,
    start_pdf_page        INTEGER,
    end_pdf_page          INTEGER
);

CREATE INDEX idx_raw_entries_parent   ON raw_entries(parent_id);
CREATE INDEX idx_raw_entries_number   ON raw_entries(section_number);
CREATE INDEX idx_raw_entries_chapter  ON raw_entries(chapter_number, level);
CREATE INDEX idx_diseases_chapter     ON diseases(chapter_number);
CREATE INDEX idx_diseases_category    ON diseases(category);
CREATE INDEX idx_diseases_icd10       ON diseases(icd10_code);
CREATE INDEX idx_diseases_name_nocase ON diseases(disease_name COLLATE NOCASE);
CREATE INDEX idx_diseases_pages       ON diseases(start_pdf_page);

-- Full-text search over the master table (external content, rowid-linked)
CREATE VIRTUAL TABLE diseases_fts USING fts5(
    disease_name, icd10_code, clinical_features, differential_diagnosis,
    investigations, management,
    content='diseases', content_rowid='rowid',
    tokenize='porter unicode61 remove_diacritics 2',
    prefix='2 3 4'
);

-- Small name-only index for fast search-as-you-type suggestions
CREATE VIRTUAL TABLE disease_names_fts USING fts5(
    disease_name, icd10_code, category,
    content='diseases', content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2',
    prefix='1 2 3 4'
);

CREATE TRIGGER diseases_ai AFTER INSERT ON diseases BEGIN
    INSERT INTO diseases_fts(rowid, disease_name, icd10_code, clinical_features,
                             differential_diagnosis, investigations, management)
    VALUES (new.rowid, new.disease_name, new.icd10_code, new.clinical_features,
            new.differential_diagnosis, new.investigations, new.management);
    INSERT INTO disease_names_fts(rowid, disease_name, icd10_code, category)
    VALUES (new.rowid, new.disease_name, new.icd10_code, new.category);
END;

CREATE TRIGGER diseases_ad AFTER DELETE ON diseases BEGIN
    INSERT INTO diseases_fts(diseases_fts, rowid, disease_name, icd10_code, clinical_features,
                             differential_diagnosis, investigations, management)
    VALUES ('delete', old.rowid, old.disease_name, old.icd10_code, old.clinical_features,
            old.differential_diagnosis, old.investigations, old.management);
    INSERT INTO disease_names_fts(disease_names_fts, rowid, disease_name, icd10_code, category)
    VALUES ('delete', old.rowid, old.disease_name, old.icd10_code, old.category);
END;

CREATE TRIGGER diseases_au AFTER UPDATE ON diseases BEGIN
    INSERT INTO diseases_fts(diseases_fts, rowid, disease_name, icd10_code, clinical_features,
                             differential_diagnosis, investigations, management)
    VALUES ('delete', old.rowid, old.disease_name, old.icd10_code, old.clinical_features,
            old.differential_diagnosis, old.investigations, old.management);
    INSERT INTO diseases_fts(rowid, disease_name, icd10_code, clinical_features,
                             differential_diagnosis, investigations, management)
    VALUES (new.rowid, new.disease_name, new.icd10_code, new.clinical_features,
            new.differential_diagnosis, new.investigations, new.management);
    INSERT INTO disease_names_fts(disease_names_fts, rowid, disease_name, icd10_code, category)
    VALUES ('delete', old.rowid, old.disease_name, old.icd10_code, old.category);
    INSERT INTO disease_names_fts(rowid, disease_name, icd10_code, category)
    VALUES (new.rowid, new.disease_name, new.icd10_code, new.category);
END;
"""


# --------------------------------------------------------------------------
# 6. Main build
# --------------------------------------------------------------------------

def build(pdf: str, db_path: str, docling_json: str | None):
    pages = load_pages_docling(docling_json) if docling_json else load_pages_pdftotext(pdf)
    print(f"[1] loaded {len(pages)} pages")

    entries, body_start = parse_toc(pages)
    print(f"[2] TOC entries: {len(entries)}  (body starts at pdf page {body_start + 1})")

    flat = locate_headings(entries, pages, body_start)
    located = [e for e in entries if e["pos"] is not None]
    missing = [e for e in entries if e["pos"] is None]
    print(f"[3] headings located: {len(located)}/{len(entries)}  missing: {len(missing)}")

    # entry slices, in body order (TOC order == body order for located entries)
    located.sort(key=lambda e: e["pos"])
    for i, e in enumerate(located):
        end = located[i + 1]["pos"] if i + 1 < len(located) else min(len(flat), e["pos"] + 600)
        e["end_pos"] = end
        e["lines"] = [t for _, _, t in flat[e["pos"] + 1 + e["skip"]:end]]
        e["start_pdf_page"] = flat[e["pos"]][0] + 1
        e["end_pdf_page"] = flat[end - 1][0] + 1 if end - 1 >= e["pos"] else e["start_pdf_page"]
        e["raw_text"] = "\n".join(t for _, _, t in flat[e["pos"]:end])

    chapters = {}
    for e in entries:
        if "." not in e["section_number"]:
            chapters[int(e["section_number"])] = e["title"].title() if e["title"].isupper() else e["title"]

    by_id = {e["entry_id"]: e for e in entries}
    for i, e in enumerate(entries):          # parent = nearest earlier entry with the parent number
        parts = e["section_number"].split(".")
        e["parent_id"] = None
        if len(parts) > 1:
            pnum = ".".join(parts[:-1])
            for j in range(i - 1, -1, -1):
                if entries[j]["section_number"] == pnum:
                    e["parent_id"] = entries[j]["entry_id"]
                    break

    db = Path(db_path)
    if db.exists():
        db.unlink()
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)

    con.executemany("INSERT INTO raw_pages(pdf_page, text) VALUES (?,?)",
                    [(i + 1, p) for i, p in enumerate(pages)])
    con.executemany("INSERT INTO chapters VALUES (?,?)", sorted(chapters.items()))

    # raw_entries: parents first so the self-FK is satisfied
    raw_rows = []
    def depth_order(e):  # parents must be inserted before children for the self-referencing FK
        d, cur = 0, e
        while cur["parent_id"]:
            d += 1
            cur = by_id[cur["parent_id"]]
        return d
    for e in sorted(entries, key=depth_order):
        parts = e["section_number"].split(".")
        chap = int(parts[0]) if int(parts[0]) in chapters else None
        raw_rows.append((e["entry_id"], e["section_number"], e["parent_id"], len(parts), chap, e["title"],
                         e["toc_page"], e.get("start_pdf_page"), e.get("end_pdf_page"), e.get("raw_text")))
    con.executemany("INSERT INTO raw_entries VALUES (?,?,?,?,?,?,?,?,?,?)", raw_rows)

    # master table
    records, used_names = [], set()
    for e in sorted(located, key=lambda x: x["pos"]):
        lines = list(e["lines"])
        icd = None
        for l in [flat[e["pos"]][2]] + lines[:30]:
            m = ICD_RE.search(l)
            if m:
                icd = re.sub(r"\s+", "", m.group(1)).upper()
                break
        lines = [l for l in lines if not ICD_RE.search(l)]
        fields, loc = split_sections(lines)
        if not any(fields[f] for f in ("clinical_features", "differential_diagnosis",
                                       "investigations", "management")):
            continue  # not a disease-style entry (e.g. a chapter intro or table-only section)

        title = re.sub(r"\s+", " ", e["title"]).strip()
        parts = e["section_number"].split(".")
        parent_title = by_id[e["parent_id"]]["title"] if e["parent_id"] else None
        category, cur = None, e
        while cur["parent_id"]:              # level-2 ancestor (e.g. "Common Emergencies")
            cur = by_id[cur["parent_id"]]
            if cur["section_number"].count(".") == 1:
                category = cur["title"]
                break
        if category is None and parent_title:
            category = parent_title
        if category is None:
            category = chapters.get(int(parts[0]))

        name = title
        if name.lower() in used_names:  # keep disease_name globally unique
            name = f"{title} ({parent_title})" if parent_title and parent_title.lower() != title.lower() \
                else f"{title} [{e['entry_id']}]"
            if name.lower() in used_names:
                name = f"{title} [{e['entry_id']}]"
        used_names.add(name.lower())

        records.append({
            "disease_name": name,
            "entry_id": e["entry_id"],
            "section_number": e["section_number"],
            "icd10_code": icd,
            "chapter_number": int(parts[0]) if int(parts[0]) in chapters else None,
            "category": category,
            "parent_condition": parent_title,
            **fields,
            "levels_of_care": loc or None,
            "start_pdf_page": e["start_pdf_page"],
            "end_pdf_page": e["end_pdf_page"],
        })

    df = pd.DataFrame.from_records(records)
    df = df.replace({"": None})
    cols = ["disease_name", "entry_id", "section_number", "icd10_code", "chapter_number", "category", "parent_condition",
            "description", "causes", "risk_factors", "clinical_features", "differential_diagnosis",
            "investigations", "management", "prevention", "complications", "levels_of_care",
            "start_pdf_page", "end_pdf_page"]
    df = df[cols]
    df.to_sql("diseases", con, if_exists="append", index=False)

    con.execute("INSERT INTO diseases_fts(diseases_fts) VALUES('optimize')")
    con.execute("INSERT INTO disease_names_fts(disease_names_fts) VALUES('optimize')")
    con.commit()
    con.execute("ANALYZE")
    con.execute("VACUUM")
    con.close()

    csv_path = db.with_suffix(".master.csv")
    df.to_csv(csv_path, index=False)

    print(f"[4] master rows: {len(df)}")
    for f in ("icd10_code", "clinical_features", "differential_diagnosis", "investigations", "management"):
        print(f"    {f:24s} filled: {df[f].notna().sum():4d}/{len(df)}")
    if missing:
        print("    headings not found:", ", ".join(f"{m['entry_id']} {m['title'][:25]}" for m in missing[:25]))
    print(f"[5] wrote {db}  ({db.stat().st_size/1e6:.1f} MB) and {csv_path.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("-o", "--output", default="ucg_2023.db")
    ap.add_argument("--docling-json", help="Docling JSON export to use instead of pdftotext")
    a = ap.parse_args()
    build(a.pdf, a.output, a.docling_json)
