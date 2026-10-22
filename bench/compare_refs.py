"""Compare two reference runs (bench/ref_llama.py outputs) row by row: llama.cpp's own spread
between settings (KV type, flash attention, platform).

    python bench/compare_refs.py results/ref_llama_pc.jsonl results/ref_llama_pc_kvf32_nofa.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from compare import load_ref


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    a = ap.parse_args()
    ra, rb = load_ref(a.a), load_ref(a.b)
    keys = sorted(set(ra) & set(rb))
    d = [max(abs(x - y) for x, y in zip(ra[k]["p"], rb[k]["p"])) for k in keys]
    am = sum(max(range(len(ra[k]["p"])), key=ra[k]["p"].__getitem__)
             == max(range(len(rb[k]["p"])), key=rb[k]["p"].__getitem__) for k in keys)
    print(json.dumps({"rows": len(keys), "max_abs_dp": max(d), "mean_abs_dp": st.mean(d),
                      "median_abs_dp": st.median(d), "argmax_agree": am,
                      "rows_over_1e-3": sum(x > 1e-3 for x in d)}, indent=1))


if __name__ == "__main__":
    main()
