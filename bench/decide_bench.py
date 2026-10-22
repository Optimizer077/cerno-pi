"""Time cerno_pi decisions on the Pi (run through piquant's `harness/pi.py bench`).

Prints one `RESULT {json}` line per request (answers, unrounded probabilities, row token ids,
per-row timing) and a final summary line with the latency distribution and peak RSS, which the
harness stores in bench/results/<name>_<ts>.json. Uses the pinfer package and libpinfer given by
PYTHONPATH / PINFER_LIB.

    python bench/decide_bench.py -m models/cerno-4b-q4_0.pq --request requests/correctness.json \
        [--only a,b] [--no-reuse] [--warmup] [--repeat 1]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # cerno_pi from the checkout

import pinfer

from cerno_pi.decide import Decider


def vm_hwm_mib() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    return float("nan")


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model", required=True)
    p.add_argument("--request", required=True)
    p.add_argument("--only", default="", help="comma-separated request names")
    p.add_argument("-t", "--threads", type=int, default=4)
    p.add_argument("-c", "--ctx", type=int, default=4096)
    p.add_argument("--no-reuse", action="store_true")
    p.add_argument("--top-n", type=int)
    p.add_argument("--warmup", action="store_true", help="one untimed decision first")
    p.add_argument("--repeat", type=int, default=1)
    a = p.parse_args()
    reqs = json.loads(Path(a.request).read_text(encoding="utf-8"))
    if a.only:
        keep = set(a.only.split(","))
        reqs = [r for r in reqs if r["name"] in keep]
    t0 = time.perf_counter()
    m = pinfer.Model.load(a.model, n_threads=a.threads, max_ctx=a.ctx)
    load_s = time.perf_counter() - t0
    d = Decider(m, reuse_state=not a.no_reuse, top_n=a.top_n)
    if a.warmup:
        d.decide("warm-up", {"w": {"type": "noul", "instructions": "Is this a warm-up?"}})
    totals, firsts, laters, tok_s = [], [], [], []
    for rep in range(a.repeat):
        for r in reqs:
            res = d.decide(r["state"], r["questions"])
            out = {"name": r["name"], "repeat": rep, **res.to_json(timing=True), "rows": res.rows}
            print("RESULT " + json.dumps(out, ensure_ascii=False, default=str), flush=True)
            totals.append(res.total_seconds)
            rows = res.row_timing
            # first question = shared prefix + its own suffix; later ones = restore + suffix
            firsts.append(res.prefix_seconds + res.snapshot_seconds + rows[0].seconds)
            laters += [x.seconds for x in rows[1:]]
            n_tok = res.prefix_tokens + sum(x.tokens for x in rows)
            tok_s.append(n_tok / max(res.total_seconds, 1e-9))
    summary = {
        "summary": True, "requests": len(totals), "questions": len(firsts) + len(laters),
        "reuse_state": not a.no_reuse, "threads": a.threads, "load_s": round(load_s, 3),
        "lib": m.info().get("library"), "matvec": (m.info().get("build") or {}).get("matvec"),
        "decision_s": {"median": st.median(totals), "mean": st.mean(totals),
                       "p80": pct(totals, 0.8), "p95": pct(totals, 0.95), "max": max(totals)},
        "first_question_s": {"median": st.median(firsts), "mean": st.mean(firsts)},
        "later_question_s": ({"median": st.median(laters), "mean": st.mean(laters),
                              "n": len(laters)} if laters else None),
        "tokens_per_s": {"median": st.median(tok_s), "mean": st.mean(tok_s)},
        "peak_rss_mib": round(vm_hwm_mib(), 1), "pid": os.getpid(),
    }
    print("RESULT " + json.dumps(summary), flush=True)
    m.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
