"""Latency summary over several decide_bench.py result files (all requests together).

    python bench/summary.py results/cerno-decide-reuse-a2_*.json results/cerno-decide-reuse-b2_*.json
"""
from __future__ import annotations

import json
import statistics as st
import sys


def main() -> None:
    totals, firsts, laters, rss, clocks = [], [], [], [], set()
    for path in sys.argv[1:]:
        rec = json.load(open(path, encoding="utf-8"))
        clocks.add((rec["state_before"].get("freq_khz"), rec["state_after"].get("freq_khz")))
        for r in rec["results"]:
            if r.get("summary"):
                rss.append(r["peak_rss_mib"])
                continue
            t = r["timing"]
            totals.append(t["total_seconds"])
            firsts.append(t["prefix_seconds"] + t["snapshot_seconds"] + t["rows"][0]["seconds"])
            laters += [x["seconds"] for x in t["rows"][1:]]
    q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]
    print(json.dumps({
        "requests": len(totals), "questions": len(firsts) + len(laters),
        "decision_s": {"median": st.median(totals), "mean": st.mean(totals), "p80": q(totals, 0.8),
                       "p95": q(totals, 0.95), "min": min(totals), "max": max(totals)},
        "first_question_s": {"median": st.median(firsts), "mean": st.mean(firsts)},
        "later_question_s": {"median": st.median(laters), "mean": st.mean(laters), "n": len(laters)},
        "peak_rss_mib_max": max(rss), "clocks": sorted(clocks)}, indent=1))


if __name__ == "__main__":
    main()
