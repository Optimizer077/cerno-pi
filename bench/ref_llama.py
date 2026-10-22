"""Reference answers from Cerno's own llama.cpp engine (third-party code, read before use).

Runs `evals/decision_index/engine_llama.py:LlamaGGUFEngine` of github.com/apertixsk/cerno-inference
against a running llama-server, on a request file, and records for every question row: the
prompt text, the tokens llama-server evaluated, the probabilities the engine computed (before
its 4-digit rounding) and the label candidates of the server's top-n list. Nothing of the
engine is modified: its `_post` and `_row_probs` are wrapped to observe their results.

Run it from a separate environment (uv venv with transformers<5, torch CPU, decision-index
@9eb2dbe installed), isolated, from outside the cloned repositories:

    python -I bench/cerno/ref_llama.py --cerno E:/third_party/cerno-inference \
        --model models/cerno-4b/hf --gguf models/cerno-4b/gguf/cerno-4b-v1.1-q4_0.gguf \
        --base-url http://127.0.0.1:8090 --requests bench/cerno/requests_correctness.json \
        --out bench/cerno/results/ref_llama_pc.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path


def tokenize(base_url: str, text: str) -> list[int]:
    body = json.dumps({"content": text, "add_special": True, "parse_special": True}).encode()
    req = urllib.request.Request(base_url + "/tokenize", data=body,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return [t if isinstance(t, int) else t["id"] for t in json.loads(r.read())["tokens"]]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--cerno", type=Path, required=True, help="clone of cerno-inference")
    p.add_argument("--model", required=True, help="HF dir with tokenizer + decider_config.json")
    p.add_argument("--gguf", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8080")
    p.add_argument("--requests", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repeat", type=int, default=1, help="run every request this many times")
    a = p.parse_args()
    root = a.cerno.resolve()
    sys.path[:0] = [str(root), str(root / "third_party" / "decider")]
    from evals.decision_index.engine_llama import LlamaGGUFEngine

    eng = LlamaGGUFEngine(model=a.model, gguf=a.gguf, base_url=a.base_url)
    rows: list[dict] = []
    orig_post, orig_row = eng._post, eng._row_probs

    def post(payload):            # observe the server's answer
        t0 = time.perf_counter()
        out = orig_post(payload)
        cands = (out.get("completion_probabilities") or [{}])[0]
        cands = cands.get("top_logprobs") or cands.get("top_probs") or cands.get("probs") or []
        rows.append({"prompt": payload["prompt"], "wall_s": time.perf_counter() - t0,
                     "tokens_evaluated": out.get("tokens_evaluated"),
                     "tokens_cached": out.get("tokens_cached"), "timings": out.get("timings"),
                     "top": {c.get("token", c.get("tok_str")): c.get("logprob", c.get("prob"))
                             for c in cands}})
        return out

    def row_probs(prompt, nopts):  # record the unrounded probabilities
        probs = orig_row(prompt, nopts)
        rows[-1]["nopts"] = nopts
        rows[-1]["labels"] = eng.labels[:nopts]
        rows[-1]["label_top"] = {lab: rows[-1]["top"].get(lab) for lab in eng.labels[:nopts]}
        rows[-1]["probs"] = probs
        return probs

    eng._post, eng._row_probs = post, row_probs
    reqs = json.loads(a.requests.read_text(encoding="utf-8"))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as f:
        for rep in range(a.repeat):
            for r in reqs:
                rows.clear()
                t0 = time.perf_counter()
                response, _ = eng(r["state"], r["questions"])
                wall = time.perf_counter() - t0
                for row, qid in zip(rows, r["questions"]):
                    row["question"] = qid
                    row["server_ids"] = tokenize(a.base_url, row["prompt"])
                    del row["top"]
                rec = {"name": r["name"], "repeat": rep, "wall_s": wall,
                       "answers": response["answers"], "rows": list(rows),
                       "temperature": eng.temperature}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print(f"{r['name']}: {len(rows)} rows, {wall:.2f} s", flush=True)


if __name__ == "__main__":
    main()
