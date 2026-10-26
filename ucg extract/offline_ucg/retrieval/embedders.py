"""
retrieval/embedders.py - pluggable text embedders, all returning L2-normalised float32 vectors.

    fastembed              ONNX, small, CPU friendly            (pip install fastembed)
    sentence-transformers  PyTorch                               (pip install sentence-transformers)
    hashing                no model files, deterministic         (always available; lower quality, used as fallback)

OFFLINE RULE: model files must already be on disk (settings.embedder_dir). We never trigger a download.
The index stores the embedder *name*; at query time `embedder_for_index(name)` rebuilds exactly that embedder,
so a vector index can never be searched with a different model by accident.
"""
from __future__ import annotations

import logging
import math
import os
import re
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_STOP = set("a an the of and or in on at to for with is are was were be been it its this that as by from into "
            "has have had not no patient presents patients".split())
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbedderUnavailable(RuntimeError):
    pass


def _normalise(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


class Embedder:
    name = "base"
    dim = 0

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:  # (n, dim)
        raise NotImplementedError

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


# --------------------------------------------------------------------------- hashing (always available)
class HashingEmbedder(Embedder):
    """Signed feature hashing of word unigrams/bigrams + char 4-grams. Not semantic, but deterministic and offline."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.name = f"hashing-{dim}-v1"

    @staticmethod
    def _stem(w: str) -> str:
        for suf in ("ing", "ies", "es", "ed", "s"):
            if len(w) > len(suf) + 3 and w.endswith(suf):
                return w[: -len(suf)]
        return w

    def _features(self, text: str) -> Counter:
        words = [self._stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP]
        feats: Counter = Counter()
        for w in words:
            feats["w:" + w] += 1.0
            if len(w) >= 5:
                for i in range(len(w) - 3):
                    feats["c:" + w[i:i + 4]] += 0.3
        for a, b in zip(words, words[1:]):
            feats[f"b:{a}_{b}"] += 0.6
        return feats

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for f, tf in self._features(t).items():
                h = zlib.crc32(f.encode())
                out[i, h % self.dim] += (1.0 if (h >> 16) & 1 else -1.0) * (1.0 + math.log(tf))
        return _normalise(out)


# --------------------------------------------------------------------------- fastembed
class FastEmbedEmbedder(Embedder):
    def __init__(self, model: str, cache_dir: Path):
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise EmbedderUnavailable("fastembed is not installed") from e
        try:
            self._m = TextEmbedding(model_name=model, cache_dir=str(cache_dir), local_files_only=True)
        except TypeError:                      # older fastembed without local_files_only
            self._m = TextEmbedding(model_name=model, cache_dir=str(cache_dir))
        except Exception as e:
            raise EmbedderUnavailable(f"fastembed model {model!r} not found in {cache_dir}: {e}") from e
        self.name = f"fastembed:{model}"
        self.dim = len(next(iter(self._m.embed(["probe"]))))

    def encode(self, texts, batch_size=64):
        return _normalise(np.array(list(self._m.embed(texts, batch_size=batch_size))))

    def encode_query(self, text):
        if hasattr(self._m, "query_embed"):
            return _normalise(np.array(list(self._m.query_embed([text]))))[0]
        return self.encode([text])[0]


# --------------------------------------------------------------------------- sentence-transformers
class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model: str, cache_dir: Path):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")           # forbid network access
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbedderUnavailable("sentence-transformers is not installed") from e
        local = Path(cache_dir) / model.replace("/", "_")
        try:
            self._m = SentenceTransformer(str(local) if local.exists() else model, cache_folder=str(cache_dir))
        except Exception as e:
            raise EmbedderUnavailable(f"sentence-transformers model {model!r} unavailable offline: {e}") from e
        self._bge = "bge" in model.lower()
        self.name = f"st:{model}"
        self.dim = int(self._m.get_sentence_embedding_dimension())

    def encode(self, texts, batch_size=64):
        return _normalise(self._m.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                         show_progress_bar=False))

    def encode_query(self, text):
        return self.encode([(_BGE_QUERY_PREFIX + text) if self._bge else text])[0]


# --------------------------------------------------------------------------- factories
def _build(backend: str, model: str, cache_dir: Path) -> Embedder:
    if backend == "fastembed":
        return FastEmbedEmbedder(model, cache_dir)
    if backend in ("sentence-transformers", "st"):
        return SentenceTransformerEmbedder(model, cache_dir)
    if backend == "hashing":
        return HashingEmbedder()
    raise EmbedderUnavailable(f"unknown embedder backend {backend!r}")


def make_embedder(settings) -> Embedder:
    """Used at ETL time: first backend in settings.embedder_order that works."""
    errors = []
    for b in [x.strip() for x in settings.embedder_order.split(",") if x.strip()]:
        try:
            emb = _build(b, settings.embedder_model, settings.embedder_dir)
            log.info("embedder selected: %s (dim=%d)", emb.name, emb.dim)
            if b == "hashing" and errors:
                log.warning("using HASHING embedder (not semantic). Earlier backends failed: %s", "; ".join(errors))
            return emb
        except EmbedderUnavailable as e:
            errors.append(f"{b}: {e}")
            log.info("embedder %s unavailable: %s", b, e)
    raise EmbedderUnavailable("no embedder backend available: " + "; ".join(errors))


def embedder_for_index(index_name: str, settings) -> Embedder:
    """Used at query time: rebuild the exact embedder that produced the stored vectors."""
    if index_name.startswith("hashing-"):
        return HashingEmbedder(int(index_name.split("-")[1]))
    if index_name.startswith("fastembed:"):
        return FastEmbedEmbedder(index_name.split(":", 1)[1], settings.embedder_dir)
    if index_name.startswith("st:"):
        return SentenceTransformerEmbedder(index_name.split(":", 1)[1], settings.embedder_dir)
    raise EmbedderUnavailable(f"unrecognised index embedder {index_name!r}")
