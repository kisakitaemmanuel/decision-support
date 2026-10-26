"""
retrieval/hybrid.py - hybrid retrieval = FTS5/bm25 + dense vectors, fused with Reciprocal Rank Fusion (RRF).

    rrf_score(chunk) = sum over retrievers of  weight / (rrf_k + rank)

RRF needs no score calibration between bm25 and cosine, which is why it is the default.  Chunk scores are then
rolled up to diseases (best chunk + decayed next two).  Every result keeps the per-retriever rank so you can see
*why* something surfaced.  If one retriever fails the other still answers and `degraded` says what happened.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from collections import defaultdict

from retrieval.embedders import EmbedderUnavailable
from retrieval.semantic_search import SemanticSearcher
from retrieval.sqlite_search import FTSSearcher
from schemas import ChunkHit, DiseaseHit, RetrievalResult

log = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self, con: sqlite3.Connection, settings):
        self.con, self.s = con, settings
        self.fts = FTSSearcher(con)
        self.name_bonus = 0.01                                # per *specific* query token found in the disease name
        from collections import Counter
        from retrieval.sqlite_search import tokens as _tok
        df: Counter = Counter()
        for r in con.execute("SELECT disease_name FROM diseases"):
            df.update({t[:6] for t in _tok(r[0])})
        self._name_df = df
        self.semantic: SemanticSearcher | None = None
        self.semantic_error = ""
        try:
            self.semantic = SemanticSearcher(con, settings)
        except (EmbedderUnavailable, Exception) as e:       # noqa: BLE001 - never let retrieval init crash the app
            self.semantic_error = str(e)
            log.warning("semantic retrieval disabled: %s", e)

    # ------------------------------------------------------------------ single query
    def retrieve(self, query: str, mode: str = "hybrid", k_chunks: int | None = None,
                 k_diseases: int | None = None, prefix_last: bool = False) -> RetrievalResult:
        kc, kd = k_chunks or self.s.top_k_chunks, k_diseases or self.s.top_k_diseases
        timings: dict[str, float] = {}
        degraded: list[str] = []
        lists: dict[str, list[ChunkHit]] = {}

        if mode in ("hybrid", "fts"):
            t = time.perf_counter()
            lists["fts"] = self.fts.search(query, kc, prefix_last)
            timings["fts"] = round((time.perf_counter() - t) * 1000, 2)
        if mode in ("hybrid", "semantic"):
            if self.semantic is None:
                degraded.append(f"semantic: {self.semantic_error}")
            else:
                t = time.perf_counter()
                try:
                    lists["semantic"] = self.semantic.search(query, kc)
                except Exception as e:                       # noqa: BLE001
                    degraded.append(f"semantic: {e}")
                    log.exception("semantic search failed")
                timings["semantic"] = round((time.perf_counter() - t) * 1000, 2)

        fused = self._fuse(lists)
        diseases = self._rollup(fused, lists, query)[:kd]
        log.debug("retrieve %r mode=%s -> %s", query, mode, [(d.disease_name, round(d.score, 4)) for d in diseases])
        return RetrievalResult(query, diseases, degraded, timings)

    def _fuse(self, lists: dict[str, list[ChunkHit]]) -> list[ChunkHit]:
        sem_w = self.s.semantic_weight
        if self.semantic is not None and self.semantic.embedder.name.startswith("hashing-"):
            sem_w *= 0.3        # hashing vectors are lexical noise, not semantics: keep them as a weak tie-breaker
        weights = {"fts": self.s.fts_weight, "semantic": sem_w}
        score: dict[int, float] = defaultdict(float)
        proto: dict[int, ChunkHit] = {}
        for src, hits in lists.items():
            for h in hits:
                score[h.chunk_id] += weights[src] / (self.s.rrf_k + h.rank)
                proto.setdefault(h.chunk_id, h)
        fused = []
        for rank, (cid, sc) in enumerate(sorted(score.items(), key=lambda kv: -kv[1]), 1):
            p = proto[cid]
            fused.append(ChunkHit(cid, p.disease_name, p.section, p.text, p.page, sc, rank, "hybrid"))
        return fused

    def _rollup(self, fused: list[ChunkHit], lists: dict[str, list[ChunkHit]], query: str = "") -> list[DiseaseHit]:
        from retrieval.sqlite_search import tokens
        qtok = [t[:6] for t in tokens(query)]
        by_d: dict[str, list[ChunkHit]] = defaultdict(list)
        for h in fused:
            by_d[h.disease_name].append(h)
        best_rank: dict[str, dict[str, int]] = defaultdict(dict)
        for src, hits in lists.items():
            for h in hits:
                cur = best_rank[h.disease_name].get(src)
                if cur is None or h.rank < cur:
                    best_rank[h.disease_name][src] = h.rank
        out = []
        for d, hs in by_d.items():
            hs.sort(key=lambda h: -h.score)
            sc = sum(h.score * w for h, w in zip(hs[:3], (1.0, 0.5, 0.25)))
            nm = d.lower()
            sc += self.name_bonus * sum(1 for t in qtok if t in nm and self._name_df.get(t, 0) <= 3)
            out.append(DiseaseHit(d, sc, hs[:3], dict(best_rank[d])))
        out.sort(key=lambda d: -d.score)
        icd = {r["disease_name"]: r["icd10_code"] for r in self.con.execute(
            f"SELECT disease_name, icd10_code FROM diseases WHERE disease_name IN ({','.join('?' * len(out))})",
            [d.disease_name for d in out])} if out else {}
        for d in out:
            d.icd10_code = icd.get(d.disease_name)
        return out

    # ------------------------------------------------------------------ several queries (symptom + free text)
    def retrieve_multi(self, queries: list[str], k_diseases: int | None = None) -> RetrievalResult:
        kd = k_diseases or self.s.top_k_diseases
        agg: dict[str, float] = defaultdict(float)
        keep: dict[str, DiseaseHit] = {}
        degraded: list[str] = []
        for q in [q for q in queries if q and q.strip()]:
            res = self.retrieve(q, k_diseases=max(kd * 2, 10))
            degraded += [d for d in res.degraded if d not in degraded]
            for rank, d in enumerate(res.diseases, 1):
                agg[d.disease_name] += 1.0 / (self.s.rrf_k + rank)
                keep.setdefault(d.disease_name, d)
        ranked = sorted(agg.items(), key=lambda kv: -kv[1])[:kd]
        out = []
        for name, sc in ranked:
            d = keep[name]
            out.append(DiseaseHit(name, sc, d.chunks, d.ranks, d.icd10_code))
        return RetrievalResult(" || ".join(queries), out, degraded, {})
