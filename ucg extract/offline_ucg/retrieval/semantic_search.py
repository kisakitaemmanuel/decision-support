"""
retrieval/semantic_search.py - dense retrieval.

Vectors live in an .npz file (canonical, always written) and, when FAISS is installed, in a FAISS index
(IndexIDMap2 over IndexFlatIP: exact inner product on normalised vectors == cosine).  With ~2-3k chunks a flat
index is exact and fast; switch to HNSW only if the corpus grows by 100x.  Without FAISS we fall back to numpy.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone

import numpy as np

from retrieval.embedders import Embedder, EmbedderUnavailable, embedder_for_index
from schemas import ChunkHit

log = logging.getLogger(__name__)

try:                                    # optional accelerator
    import faiss  # type: ignore
except Exception:                       # noqa: BLE001 - any import problem means "no FAISS"
    faiss = None


def chunk_embed_text(disease: str, section: str, text: str) -> str:
    return f"{disease} | {section.replace('_', ' ')} | {text}"


def build_index(con: sqlite3.Connection, embedder: Embedder, settings, batch_size: int = 64) -> int:
    rows = con.execute("SELECT chunk_id, disease_name, section, text FROM chunks ORDER BY chunk_id").fetchall()
    if not rows:
        raise RuntimeError("no chunks to embed - run the chunking stage first")
    ids = np.array([r["chunk_id"] for r in rows], dtype=np.int64)
    vecs = embedder.encode([chunk_embed_text(r["disease_name"], r["section"], r["text"]) for r in rows], batch_size)
    vecs = np.ascontiguousarray(vecs, dtype=np.float32)
    settings.npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(settings.npz_path, vectors=vecs, ids=ids, embedder=np.array(embedder.name))
    faiss_file = None
    if faiss is not None:
        idx = faiss.IndexIDMap2(faiss.IndexFlatIP(vecs.shape[1]))
        idx.add_with_ids(vecs, ids)
        faiss.write_index(idx, str(settings.faiss_path))
        faiss_file = settings.faiss_path.name
        log.info("FAISS index written: %s (%d vectors)", settings.faiss_path, len(ids))
    else:
        log.warning("faiss not installed: wrote .npz only (numpy search will be used at runtime)")
    con.execute("DELETE FROM vector_index_meta")
    con.execute("INSERT INTO vector_index_meta VALUES (1,?,?,?,?,?,?)",
                (embedder.name, int(vecs.shape[1]), int(len(ids)),
                 datetime.now(timezone.utc).isoformat(timespec="seconds"), settings.npz_path.name, faiss_file))
    con.commit()
    return len(ids)


class SemanticSearcher:
    def __init__(self, con: sqlite3.Connection, settings):
        meta = con.execute("SELECT * FROM vector_index_meta WHERE id=1").fetchone()
        if meta is None:
            raise EmbedderUnavailable("no vector index in database (ETL ran without embeddings)")
        if not settings.npz_path.exists():
            raise EmbedderUnavailable(f"vector file missing: {settings.npz_path}")
        data = np.load(settings.npz_path, allow_pickle=False)
        self.con = con
        self.ids: np.ndarray = data["ids"]
        self.vectors: np.ndarray = np.ascontiguousarray(data["vectors"], dtype=np.float32)
        stored = str(data["embedder"])
        if stored != meta["embedder"]:
            raise EmbedderUnavailable(f"vector file ({stored}) and database ({meta['embedder']}) disagree - rebuild ETL")
        self.embedder = embedder_for_index(stored, settings)          # raises EmbedderUnavailable if model absent
        self.backend = "numpy"
        self._faiss = None
        if faiss is not None:
            self._faiss = faiss.IndexIDMap2(faiss.IndexFlatIP(self.vectors.shape[1]))
            self._faiss.add_with_ids(self.vectors, self.ids)
            self.backend = "faiss"
        log.info("semantic index ready: %d vectors, embedder=%s, backend=%s", len(self.ids), stored, self.backend)

    def search(self, query: str, k: int = 30) -> list[ChunkHit]:
        q = np.ascontiguousarray(self.embedder.encode_query(query), dtype=np.float32).reshape(1, -1)
        if self._faiss is not None:
            scores, found = self._faiss.search(q, k)
            pairs = [(int(i), float(s)) for i, s in zip(found[0], scores[0]) if i != -1]
        else:
            s = (self.vectors @ q[0])
            top = np.argsort(-s)[:k]
            pairs = [(int(self.ids[i]), float(s[i])) for i in top]
        out: list[ChunkHit] = []
        for rank, (cid, score) in enumerate(pairs, 1):
            r = self.con.execute("SELECT disease_name, section, text, page FROM chunks WHERE chunk_id=?", (cid,)).fetchone()
            if r:
                out.append(ChunkHit(cid, r["disease_name"], r["section"], r["text"], r["page"], score, rank, "semantic"))
        return out
