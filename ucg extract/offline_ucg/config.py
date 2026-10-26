"""
config.py - one place for every path, switch and threshold.

Every value can be overridden with an environment variable  UCG_<NAME>  (e.g. UCG_FACILITY_LEVEL=HC4),
so the same code runs on a laptop, a clinic server and in tests without edits.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
from dataclasses import dataclass, fields
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _env(name: str, default, cast=str):
    raw = os.environ.get(f"UCG_{name.upper()}")
    if raw is None:
        return default
    if cast is bool:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return cast(raw)


@dataclass
class Settings:
    # ---- paths -------------------------------------------------------------------------------
    base_dir: Path = BASE_DIR
    db_path: Path = BASE_DIR / "database" / "ucg.db"
    schema_path: Path = BASE_DIR / "database" / "schema.sql"
    npz_path: Path = BASE_DIR / "database" / "ucg_vectors.npz"      # canonical vector store (always written)
    faiss_path: Path = BASE_DIR / "database" / "ucg.faiss"          # optional FAISS accelerator file
    audit_db_path: Path = BASE_DIR / "audit" / "decisions.db"
    log_dir: Path = BASE_DIR / "logs"
    # ---- embeddings (retrieval/embedders.py) ---------------------------------------------------
    # tried in order; "hashing" needs no model files and is the last-resort fallback
    embedder_order: str = "fastembed,sentence-transformers,hashing"
    embedder_model: str = "BAAI/bge-small-en-v1.5"                 # fastembed / sentence-transformers model id
    embedder_dir: Path = BASE_DIR / "models" / "embeddings"         # pre-downloaded model files (offline!)
    # ---- local LLM (agents/llm.py) -------------------------------------------------------------
    llm_model_path: Path = BASE_DIR / "models" / "clinical_model.gguf"
    llm_n_ctx: int = 4096
    llm_threads: int = max(1, (os.cpu_count() or 4) - 1)
    llm_gpu_layers: int = 0
    llm_temperature: float = 0.1
    llm_max_tokens: int = 700
    llm_server_url: str = ""                                        # optional: llama.cpp `llama-server` (OpenAI API)
    llm_enabled: bool = True                                        # False -> always deterministic mode
    # ---- retrieval ---------------------------------------------------------------------------------
    rrf_k: int = 60                                                 # reciprocal-rank-fusion constant
    fts_weight: float = 1.0
    semantic_weight: float = 1.0
    top_k_chunks: int = 30
    top_k_diseases: int = 5
    # ---- clinical logic --------------------------------------------------------------------------
    facility_level: str = "HC3"                                     # HC2 | HC3 | HC4 | H | RR | NR
    llm_rank_bonus: float = 0.15                                    # LLM may nudge, never override, the ranking
    require_reviewed_rules: bool = False                            # True: ignore rules with reviewed=0
    # ---- audit / logging ---------------------------------------------------------------------------
    audit_store_raw_text: bool = True                               # False -> only a hash of the input is stored
    log_level: str = "INFO"
    pipeline_version: str = "1.0.0"

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        for f in fields(cls):
            cur = getattr(s, f.name)
            cast = Path if isinstance(cur, Path) else type(cur)
            setattr(s, f.name, _env(f.name, cur, cast))
        return s


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def set_settings(s: Settings) -> None:
    """Tests and the desktop app use this to point the whole stack at a different database."""
    global _settings
    _settings = s


def setup_logging(level: str | None = None) -> None:
    s = get_settings()
    root = logging.getLogger()
    if getattr(root, "_ucg_configured", False):
        return
    s.log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)-24s %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    filep = logging.handlers.RotatingFileHandler(s.log_dir / "ucg.log", maxBytes=2_000_000, backupCount=3,
                                                 encoding="utf-8")
    filep.setFormatter(fmt)
    root.addHandler(console)
    root.addHandler(filep)
    root.setLevel((level or s.log_level).upper())
    root._ucg_configured = True  # type: ignore[attr-defined]
