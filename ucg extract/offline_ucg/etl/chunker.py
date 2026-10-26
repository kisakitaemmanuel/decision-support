"""etl/chunker.py - split each disease section into retrieval chunks on bullet/line boundaries."""
from __future__ import annotations

import re

SECTIONS = ["description", "causes", "risk_factors", "clinical_features", "differential_diagnosis",
            "investigations", "management", "prevention", "complications"]


def split_text(text: str, max_chars: int = 900, min_chars: int = 120) -> list[str]:
    """Greedy packing of lines; never cuts inside a line unless that single line exceeds max_chars."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    chunks: list[str] = []
    cur: list[str] = []
    size = 0
    for line in lines:
        pieces = [line] if len(line) <= max_chars else re.split(r"(?<=[.;])\s+", line)
        for p in pieces:
            if size + len(p) + 1 > max_chars and cur:
                chunks.append("\n".join(cur))
                cur, size = [], 0
            cur.append(p)
            size += len(p) + 1
    if cur:
        chunks.append("\n".join(cur))
    if len(chunks) > 1 and len(chunks[-1]) < min_chars:          # avoid tiny trailing chunk
        tail = chunks.pop()
        chunks[-1] += "\n" + tail
    return chunks


def chunk_disease(row, max_chars: int = 900):
    """Yield (section, chunk_idx, text, page) for one disease row."""
    for sec in SECTIONS:
        body = row[sec]
        if not body or not body.strip():
            continue
        for i, c in enumerate(split_text(body, max_chars)):
            yield sec, i, c, row["start_pdf_page"]
