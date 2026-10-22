"""Per-request latency table (Markdown) from a harness result of bench/decide_bench.py.

    python bench/latency_table.py results/cerno-decide-reuse-a_<ts>.json
"""
from __future__ import annotations

import json
import sys


def main() -> None:
    rec = json.load(open(sys.argv[1], encoding="utf-8"))
    print(f"clock (freq_khz) before: {rec['state_before'].get('freq_khz')}, "
          f"after: {rec['state_after'].get('freq_khz')}; throttled {rec['state_after'].get('throttled')}; "
          f"valid={rec['valid']}")
    print("| request | questions | shared prefix tok | row tok (sum) | decision s | first q s | later q s (mean) |")
    print("|---|---|---|---|---|---|---|")
    for r in rec["results"]:
        if r.get("summary"):
            s = r
            continue
        t = r["timing"]
        rows = t["rows"]
        first = t["prefix_seconds"] + t["snapshot_seconds"] + rows[0]["seconds"]
        later = [x["seconds"] for x in rows[1:]]
        print(f"| {r['name']} | {len(rows)} | {t['prefix_tokens']} | {sum(x['tokens'] for x in rows)} | "
              f"{t['total_seconds']:.1f} | {first:.1f} | "
              f"{(sum(later) / len(later)) if later else float('nan'):.1f} |")
    print("summary:", json.dumps(s))


if __name__ == "__main__":
    main()
