"""Compare cerno_pi (pinfer) probabilities with the llama.cpp reference engine, field by field.

    python bench/compare.py results/ref_llama_pc.jsonl bench_result.json [more bench results...]
        [--json out.json]

`ref` is bench/ref_llama.py's output (Cerno's engine_llama.py on llama-server); the others are
piquant harness result files of bench/decide_bench.py (RESULT lines) or cerno-pi decide JSON
lines. Reports, over every question row both sides answered: the max and mean absolute
probability difference (unrounded probabilities), argmax agreement, whether the token ids of the
row are identical, and the rows with the largest differences.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path


def load_ref(path: Path) -> dict[tuple[str, str], dict]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("repeat", 0):
            continue
        for row in r["rows"]:
            out[(r["name"], row["question"])] = {"p": row["probs"], "ids": row["server_ids"]}
    return out


def load_ours(paths: list[Path]) -> dict[tuple[str, str], dict]:
    out = {}
    for path in paths:
        text = path.read_text(encoding="utf-8")
        recs = []
        if path.suffix == ".json" and text.lstrip().startswith("{") and '"results"' in text[:2000]:
            recs = [r for r in json.loads(text)["results"] if "probabilities" in r]
        else:
            recs = [json.loads(line) for line in text.splitlines() if line.strip()]
        for r in recs:
            if r.get("repeat", 0):
                continue
            for q, p in r["probabilities"].items():
                out[(r["name"], q)] = {"p": p, "ids": (r.get("rows") or {}).get(q)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("ref", type=Path)
    ap.add_argument("ours", type=Path, nargs="+")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--worst", type=int, default=5)
    a = ap.parse_args()
    ref, ours = load_ref(a.ref), load_ours(a.ours)
    keys = sorted(set(ref) & set(ours))
    rows = []
    for k in keys:
        pr, po = ref[k]["p"], ours[k]["p"]
        d = max(abs(x - y) for x, y in zip(pr, po))
        am = max(range(len(pr)), key=pr.__getitem__) == max(range(len(po)), key=po.__getitem__)
        same_ids = ours[k]["ids"] == ref[k]["ids"] if ours[k]["ids"] is not None else None
        rows.append({"request": k[0], "question": k[1], "nopts": len(pr), "max_abs_dp": d,
                     "argmax_same": am, "ids_same": same_ids, "ref": pr, "ours": po})
    diffs = [r["max_abs_dp"] for r in rows]
    summary = {"rows": len(rows), "ref_rows": len(ref), "our_rows": len(ours),
               "max_abs_dp": max(diffs) if diffs else None,
               "mean_abs_dp": st.mean(diffs) if diffs else None,
               "median_abs_dp": st.median(diffs) if diffs else None,
               "argmax_agree": sum(r["argmax_same"] for r in rows),
               "ids_identical": sum(1 for r in rows if r["ids_same"]),
               "rows_over_1e-3": sum(1 for d in diffs if d > 1e-3)}
    print(json.dumps(summary, indent=1))
    for r in sorted(rows, key=lambda r: -r["max_abs_dp"])[:a.worst]:
        print(f"  {r['request']}.{r['question']}: max|dp| {r['max_abs_dp']:.2e} argmax "
              f"{'same' if r['argmax_same'] else 'DIFFERENT'} ids {'same' if r['ids_same'] else r['ids_same']}")
    if a.json:
        a.json.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
