#!/usr/bin/env python3
"""
cli.py - debug the whole pipeline from a terminal, no web server needed.

    python cli.py ask "3 year old, 14kg, fever and convulsions"          # readable report
    python cli.py ask "..." --trace                                       # + per-step timings and tool calls
    python cli.py ask "..." --json                                        # full structured result
    python cli.py search "rheumatic fever" --mode fts|semantic|hybrid     # compare retrievers side by side
    python cli.py health | audit [decision_id]
"""
from __future__ import annotations

import argparse
import json
import logging

from agents.orchestrator import Pipeline, Services
from config import setup_logging


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask"); a.add_argument("text"); a.add_argument("--json", action="store_true"); a.add_argument("--trace", action="store_true")
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("--mode", default="hybrid"); s.add_argument("-k", type=int, default=8)
    sub.add_parser("health")
    au = sub.add_parser("audit"); au.add_argument("decision_id", nargs="?")
    args = ap.parse_args()
    setup_logging("DEBUG" if args.verbose else "WARNING")
    sv = Services.build()
    pipe = Pipeline(sv)

    if args.cmd == "health":
        print(json.dumps(sv.health(), indent=2))
    elif args.cmd == "ask":
        r = pipe.run(args.text)
        if args.json:
            print(json.dumps(r, indent=2, default=str))
        else:
            print(r["explanation"]["text"])
            print(f"\n[decision {r['decision_id']}  {r['latency_ms']} ms  rules={r['versions']['rules']}]")
        if args.trace:
            print("\nTRACE"); [print(" ", t) for t in r["trace"]]
            print("TOOL CALLS"); [print(" ", t) for t in r["assessment"]["tool_trace"]] or print("  (none: deterministic mode)")
    elif args.cmd == "search":
        res = pipe.search.search(args.query, args.mode, args.k)
        print("degraded:", res.degraded, "timings:", res.timings_ms)
        for d in res.diseases:
            print(f"{d.score:.4f}  {d.disease_name:45.45}  ranks={d.ranks}")
    elif args.cmd == "audit":
        print(json.dumps(sv.audit.get(args.decision_id) if args.decision_id else sv.audit.recent(), indent=2, default=str))


if __name__ == "__main__":
    main()
