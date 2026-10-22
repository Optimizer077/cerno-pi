"""Time llama.cpp's llama-server on the Pi with the reference prompts (same GGUF, same protocol).

Starts llama-server, replays the rows recorded by bench/ref_llama.py (the prompts Cerno's
engine_llama.py sends, in the same order, with the same payload: one token, n_probs 260,
temperature 0, cache_prompt on, so llama-server reuses the shared state prefix as in the
reference), and prints one `RESULT {json}` line per request (per-row wall time, the server's
timings, the probabilities computed as engine_llama.py does) and a summary with the latency
distribution and the server's peak RSS (VmHWM). Standard library only (runs with the Pi's
python3); nibblepi's `harness/pi.py bench` stores the output.

    python3 bench/llama_pi.py --server llama/llama-server --gguf models/cerno-4b-v1.1-q4_0.gguf \
        --rows results/ref_llama_pc.jsonl [--only a,b] [--threads 4] [--ctx 4096]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

TEMPERATURE = 1.534      # decider_config.json of apertixai/cerno-4b


def post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        return json.loads(r.read())


def probs(out: dict, labels: list[str]) -> list[float]:
    """engine_llama.py _row_probs: label entries of the top list, p^(1/T), renormalised."""
    top = (out.get("completion_probabilities") or [{}])[0]
    cands = top.get("top_logprobs") or top.get("top_probs") or top.get("probs") or []
    raw = {}
    for c in cands:
        t = c.get("token", c.get("tok_str"))
        raw[t] = float(c["prob"]) if "prob" in c else math.exp(float(c["logprob"]))
    p = [raw.get(lab, 0.0) for lab in labels]
    p = [x ** (1.0 / TEMPERATURE) if x > 0 else 0.0 for x in p]
    s = sum(p)
    return [x / s for x in p] if s > 0 else [1.0 / len(labels)] * len(labels)


def vm_hwm_mib(pid: int) -> float:
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--rows", required=True, help="bench/ref_llama.py output (prompts, labels)")
    ap.add_argument("--only", default="")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--extra", default="", help="more llama-server arguments")
    a = ap.parse_args()
    reqs = [json.loads(line) for line in Path(a.rows).read_text(encoding="utf-8").splitlines()]
    reqs = [r for r in reqs if not r.get("repeat", 0)]
    if a.only:
        keep = set(a.only.split(","))
        reqs = [r for r in reqs if r["name"] in keep]
    cmd = [a.server, "-m", a.gguf, "--host", "127.0.0.1", "--port", str(a.port), "-c", str(a.ctx),
           "-t", str(a.threads), "--parallel", "1", *a.extra.split()]
    t0 = time.perf_counter()
    srv = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=open("tmp/cerno/llama_server.log", "w"))
    base = f"http://127.0.0.1:{a.port}"
    for _ in range(600):
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as r:
                if b"ok" in r.read():
                    break
        except OSError:
            time.sleep(1)
    load_s = time.perf_counter() - t0
    try:
        # warm-up (untimed), like decide_bench.py --warmup
        post(base + "/completion", dict(prompt="Context:\nwarm-up\n\nQuestion: Is this a warm-up?\n"
                                        "Options:\n(A) no\n(B) yes\nAnswer: (", n_predict=1, n_probs=260,
                                        temperature=0.0, cache_prompt=True))
        totals, firsts, laters = [], [], []
        for r in reqs:
            rows, t_req = [], time.perf_counter()
            for row in r["rows"]:
                payload = dict(prompt=row["prompt"], n_predict=1, n_probs=max(260, row["nopts"] + 2),
                               temperature=0.0, top_k=0, top_p=1.0, min_p=0.0, typical_p=1.0,
                               repeat_penalty=1.0, cache_prompt=True)
                t1 = time.perf_counter()
                out = post(base + "/completion", payload)
                dt = time.perf_counter() - t1
                rows.append({"question": row["question"], "seconds": dt, "timings": out.get("timings"),
                             "probs": probs(out, row["labels"])})
            total = time.perf_counter() - t_req
            totals.append(total)
            firsts.append(rows[0]["seconds"])
            laters += [x["seconds"] for x in rows[1:]]
            print("RESULT " + json.dumps({"name": r["name"], "total_seconds": total, "rows": rows}),
                  flush=True)
        print("RESULT " + json.dumps({
            "summary": True, "engine": "llama-server", "cmd": " ".join(cmd), "load_s": load_s,
            "requests": len(totals), "questions": len(firsts) + len(laters),
            "decision_s": {"median": st.median(totals), "mean": st.mean(totals), "max": max(totals)},
            "first_question_s": {"median": st.median(firsts), "mean": st.mean(firsts)},
            "later_question_s": ({"median": st.median(laters), "mean": st.mean(laters), "n": len(laters)}
                                 if laters else None),
            "server_peak_rss_mib": round(vm_hwm_mib(srv.pid), 1)}), flush=True)
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=30)
        except subprocess.TimeoutExpired:
            srv.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
