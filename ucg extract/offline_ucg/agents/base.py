"""
agents/base.py - tool registry + tool-calling loop (model-agnostic).

Protocol (works with any GGUF chat model, no special tool-calling template needed):
  * the system prompt lists tools with JSON schemas;
  * the model answers EITHER with one or more  <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    OR with a final answer (no tool_call tags);
  * we validate + execute each call, then feed back  <tool_response name="...">json</tool_response>;
  * every step is recorded in `steps` (and later in the audit log) so a run can be replayed.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger(__name__)
_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}
_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict                                   # JSON-schema "properties" + "required"
    fn: Callable[..., Any]

    def spec_text(self) -> str:
        props = self.parameters.get("properties", {})
        req = set(self.parameters.get("required", []))
        args = ", ".join(f"{k}{'' if k in req else '?'}: {v.get('type', 'any')}" for k, v in props.items())
        return f"- {self.name}({args}): {self.description}"


@dataclass
class ToolResult:
    name: str
    args: dict
    ok: bool
    output: Any = None
    error: str = ""
    ms: float = 0.0


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools)

    def describe(self) -> str:
        return "\n".join(t.spec_text() for t in self._tools.values())

    def call(self, name: str, args: dict | None) -> ToolResult:
        args = args or {}
        t0 = time.perf_counter()
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(name, args, False, error=f"unknown tool {name!r}; available: {self.names()}")
        props, req = tool.parameters.get("properties", {}), tool.parameters.get("required", [])
        for r in req:
            if r not in args:
                return ToolResult(name, args, False, error=f"missing required argument {r!r}")
        clean = {}
        for k, v in args.items():
            if k not in props:
                log.warning("tool %s: ignoring unknown argument %r", name, k)
                continue
            want = _TYPES.get(props[k].get("type", ""), object)
            if isinstance(v, bool) and props[k].get("type") in ("integer", "number"):
                return ToolResult(name, args, False, error=f"argument {k!r} must be {props[k]['type']}")
            if not isinstance(v, want):
                return ToolResult(name, args, False, error=f"argument {k!r} must be {props[k].get('type')}")
            clean[k] = v
        try:
            out = tool.fn(**clean)
            return ToolResult(name, clean, True, out, ms=round((time.perf_counter() - t0) * 1000, 2))
        except Exception as e:                          # noqa: BLE001 - tool errors go back to the model, not up the stack
            log.exception("tool %s failed", name)
            return ToolResult(name, clean, False, error=f"{type(e).__name__}: {e}",
                              ms=round((time.perf_counter() - t0) * 1000, 2))


def extract_json(text: str) -> dict | None:
    """First balanced {...} object in text (models often wrap JSON in prose or code fences)."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            depth += text[i] == "{"
            depth -= text[i] == "}"
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    break
        start = text.find("{", start + 1)
    return None


@dataclass
class LoopResult:
    final_text: str
    steps: list[dict] = field(default_factory=list)
    stopped: str = "final"                              # final | max_steps | llm_error


class ToolCallLoop:
    def __init__(self, llm, registry: ToolRegistry, max_steps: int = 5, max_tool_chars: int = 1800):
        self.llm, self.reg, self.max_steps, self.max_chars = llm, registry, max_steps, max_tool_chars

    def run(self, system: str, user: str) -> LoopResult:
        messages = [{"role": "system", "content": system + "\n\nTools:\n" + self.reg.describe() +
                     '\n\nTo use a tool reply ONLY with <tool_call>{"name": "...", "arguments": {...}}</tool_call>.'
                     " When you have enough information, reply with the final answer and no tool_call."},
                    {"role": "user", "content": user}]
        steps: list[dict] = []
        for n in range(1, self.max_steps + 1):
            try:
                reply = self.llm.chat(messages, stop=["</tool_response>"])
            except Exception as e:                      # noqa: BLE001
                log.exception("LLM call failed")
                return LoopResult("", steps, f"llm_error: {e}")
            calls = _CALL_RE.findall(reply)
            if not calls:
                steps.append({"step": n, "type": "final", "assistant": reply[:600]})
                return LoopResult(reply, steps, "final")
            messages.append({"role": "assistant", "content": reply})
            for raw in calls[:3]:                       # cap parallel calls
                try:
                    obj = json.loads(raw)
                    res = self.reg.call(obj.get("name", ""), obj.get("arguments"))
                except json.JSONDecodeError as e:
                    res = ToolResult("?", {}, False, error=f"invalid JSON in tool_call: {e}")
                payload = json.dumps(res.output if res.ok else {"error": res.error}, default=str)[: self.max_chars]
                steps.append({"step": n, "type": "tool", "tool": res.name, "args": res.args, "ok": res.ok,
                              "error": res.error, "ms": res.ms, "result_preview": payload[:300]})
                messages.append({"role": "user", "content": f'<tool_response name="{res.name}">{payload}</tool_response>'})
        messages.append({"role": "user", "content": "Tool budget used up. Give the final answer now, without tool calls."})
        try:
            reply = self.llm.chat(messages)
        except Exception as e:                          # noqa: BLE001
            return LoopResult("", steps, f"llm_error: {e}")
        steps.append({"step": self.max_steps + 1, "type": "forced_final", "assistant": reply[:600]})
        return LoopResult(reply, steps, "max_steps")
