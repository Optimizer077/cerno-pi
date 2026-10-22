"""``cerno-pi decide``: typed decisions with a Cerno model on pinfer, one JSON line per request.

    cerno-pi decide -m cerno-4b-q4_0.pq --request requests/correctness.json [--timing]
    echo '{"state": "...", "questions": {...}}' | cerno-pi decide -m cerno-4b-q4_0.pq
    cerno-pi decide -m cerno-4b-q4_0.pq --state "rm -rf /tmp/x" --questions q.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pinfer

from . import __version__
from .decide import DEFAULT_MAX_STATE_TOKENS, Decider


def read_requests(a: argparse.Namespace) -> list[dict[str, Any]]:
    """--request FILE (or stdin): one {"state", "questions"} object, a list of them, or JSON
    lines; or --state with --questions (JSON text or a file)."""
    if a.state is not None:
        if not a.questions:
            raise ValueError("--state needs --questions (a JSON object or a file)")
        q = Path(a.questions)
        text = q.read_text(encoding="utf-8") if q.is_file() else a.questions
        return [{"state": a.state, "questions": json.loads(text)}]
    raw = (Path(a.request).read_text(encoding="utf-8") if a.request and a.request != "-"
           else sys.stdin.read()).strip()
    if not raw:
        raise ValueError("no request given (--request FILE, stdin, or --state/--questions)")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in raw.splitlines() if line.strip()]
    reqs = data if isinstance(data, list) else [data]
    for r in reqs:
        if not isinstance(r, dict) or "state" not in r or "questions" not in r:
            raise ValueError("a request is an object with 'state' and 'questions'")
    return reqs


def cmd_decide(a: argparse.Namespace) -> int:
    reqs = read_requests(a)
    with pinfer.Model.load(a.model, tokenizer=a.tokenizer or None, n_threads=a.threads,
                           max_ctx=a.ctx, lib_path=a.lib, kv_dtype=a.kv_dtype) as m:
        d = Decider(m, temperature=a.temperature, max_state_tokens=a.max_state_tokens,
                    tokenization=a.tokenization, reuse_state=not a.no_reuse, top_n=a.top_n)
        for i, r in enumerate(reqs):
            for rep in range(a.repeat):
                res = d.decide(r["state"], r["questions"])
                out: dict[str, Any] = {"name": r.get("name", i), **res.to_json(timing=a.timing)}
                if a.repeat > 1:
                    out["repeat"] = rep
                if a.ids:
                    out["rows"] = res.rows
                print(json.dumps(out, ensure_ascii=False, default=str), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cerno-pi", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"cerno-pi {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="log to stderr (INFO)")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decide", help="probabilities for typed questions about a state")
    d.add_argument("-m", "--model", required=True, help=".pq file (tools/gguf_to_pq.py of pinfer)")
    d.add_argument("--tokenizer", help="tokenizer dir (default: <model>.tokenizer/)")
    d.add_argument("-t", "--threads", type=int, default=4)
    d.add_argument("-c", "--ctx", type=int, default=4096, help="max tokens per row (default 4096)")
    d.add_argument("--kv-dtype", choices=("fp32", "fp16", "int8"), default="fp32")
    d.add_argument("--lib", help="libpinfer.so (default: pinfer's search)")
    d.add_argument("--request", help="JSON file: {state, questions}, a list, or JSON lines "
                   "(default: stdin)")
    d.add_argument("--state", help="the state (text) of a single request")
    d.add_argument("--questions", help="with --state: questions as JSON or a JSON file")
    d.add_argument("--temperature", type=float,
                   help="readout temperature (default: decider_config.json, else 1.0)")
    d.add_argument("--max-state-tokens", type=int, default=DEFAULT_MAX_STATE_TOKENS)
    d.add_argument("--tokenization", choices=("text", "ids"), default="text",
                   help="text: as llama-server sees the row (default); ids: decider's ids")
    d.add_argument("--no-reuse", action="store_true",
                   help="prefill every question row from scratch (no shared-state snapshot)")
    d.add_argument("--top-n", type=int, help="emulate the llama.cpp reference's top-N "
                   "probability list (labels outside it count as 0; reads the full head)")
    d.add_argument("--timing", action="store_true", help="add per-row timing")
    d.add_argument("--ids", action="store_true", help="add every row's token ids")
    d.add_argument("--repeat", type=int, default=1, help="answer every request N times")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        return cmd_decide(a)
    except (pinfer.PinferError, OSError, ValueError) as e:
        print(f"cerno-pi: error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
