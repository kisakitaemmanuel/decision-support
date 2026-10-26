"""
agents/llm.py - local LLM backends behind one tiny interface.

    LlamaCppLLM    in-process llama-cpp-python (GGUF file)         pip install llama-cpp-python
    LlamaServerLLM talks to a running `llama-server` (OpenAI API)   no extra pip package, uses urllib
    NullLLM        no model available -> agents fall back to deterministic behaviour

The LLM is never trusted with facts: agents give it retrieved text and verify what comes back.
"""
from __future__ import annotations

import json
import logging
import threading
import urllib.request

log = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    pass


class LLM:
    name = "none"
    available = False

    def chat(self, messages: list[dict], max_tokens: int | None = None, temperature: float | None = None,
             stop: list[str] | None = None, json_mode: bool = False) -> str:
        raise LLMUnavailable("no LLM configured")


class NullLLM(LLM):
    def __init__(self, reason: str = ""):
        self.reason = reason


class LlamaCppLLM(LLM):
    def __init__(self, settings):
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise LLMUnavailable("llama-cpp-python is not installed") from e
        if not settings.llm_model_path.exists():
            raise LLMUnavailable(f"model file not found: {settings.llm_model_path}")
        self.s = settings
        self._lock = threading.Lock()                     # llama.cpp contexts are not thread-safe
        self._llm = Llama(model_path=str(settings.llm_model_path), n_ctx=settings.llm_n_ctx,
                          n_threads=settings.llm_threads, n_gpu_layers=settings.llm_gpu_layers, verbose=False)
        self.name = f"llama.cpp:{settings.llm_model_path.name}"
        self.available = True

    def chat(self, messages, max_tokens=None, temperature=None, stop=None, json_mode=False):
        kw = dict(messages=messages, max_tokens=max_tokens or self.s.llm_max_tokens,
                  temperature=self.s.llm_temperature if temperature is None else temperature, stop=stop)
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        with self._lock:
            out = self._llm.create_chat_completion(**kw)
        return out["choices"][0]["message"]["content"] or ""


class LlamaServerLLM(LLM):
    def __init__(self, settings):
        self.s = settings
        self.url = settings.llm_server_url.rstrip("/") + "/v1/chat/completions"
        try:                                              # fail fast if the server is not up
            urllib.request.urlopen(settings.llm_server_url.rstrip("/") + "/health", timeout=2).read()
        except Exception as e:                            # noqa: BLE001
            raise LLMUnavailable(f"llama-server not reachable at {settings.llm_server_url}: {e}") from e
        self.name = f"llama-server:{settings.llm_server_url}"
        self.available = True

    def chat(self, messages, max_tokens=None, temperature=None, stop=None, json_mode=False):
        body = {"messages": messages, "max_tokens": max_tokens or self.s.llm_max_tokens,
                "temperature": self.s.llm_temperature if temperature is None else temperature}
        if stop:
            body["stop"] = stop
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(self.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"] or ""


def get_llm(settings) -> LLM:
    if not settings.llm_enabled:
        return NullLLM("disabled by UCG_LLM_ENABLED=0")
    try:
        llm = LlamaServerLLM(settings) if settings.llm_server_url else LlamaCppLLM(settings)
        log.info("LLM ready: %s", llm.name)
        return llm
    except LLMUnavailable as e:
        log.warning("LLM unavailable, running deterministic mode: %s", e)
        return NullLLM(str(e))
