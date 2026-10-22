"""Compare the probabilities of two cerno_pi runs (harness result files of bench/decide_bench.py),
e.g. with and without the shared-state snapshot, or two engine builds: are they bit-identical?

    python bench/compare_runs.py results/cerno-decide-reuse-b2_<ts>.json results/cerno-decide-noreuse_<ts>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from compare import load_ours


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path, nargs="+")
    ap.add_argument("--vs", type=Path, nargs="+", required=True)
    a = ap.parse_args()
    ra, rb = load_ours(a.a), load_ours(a.vs)
    keys = sorted(set(ra) & set(rb))
    same = sum(ra[k]["p"] == rb[k]["p"] for k in keys)
    dmax = max((max(abs(x - y) for x, y in zip(ra[k]["p"], rb[k]["p"])) for k in keys), default=None)
    ids = sum(ra[k]["ids"] == rb[k]["ids"] for k in keys)
    print(json.dumps({"rows": len(keys), "probabilities_identical": same, "token_ids_identical": ids,
                      "max_abs_dp": dmax}))


if __name__ == "__main__":
    main()
