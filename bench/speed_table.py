"""Per-request decision time on the Pi: pinfer (cerno_pi, state reuse) vs llama.cpp's llama-server
(the reference protocol), plus the probability agreement of the Pi's llama.cpp with the x86
reference and with pinfer.

    python bench/speed_table.py --pinfer results/cerno-decide-reuse-a2_*.json results/cerno-decide-reuse-b2_*.json \
        --llama results/cerno-llama-server-1_*.json results/cerno-llama-server-2_*.json \
        --ref results/ref_llama_pc.jsonl [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from compare import load_ours, load_ref


def harness_results(paths: list[Path]) -> list[dict]:
    out = []
    for p in paths:
        rec = json.loads(p.read_text(encoding="utf-8"))
        out += [r for r in rec["results"] if not r.get("summary")]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinfer", type=Path, nargs="+", required=True)
    ap.add_argument("--llama", type=Path, nargs="+", required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    pin = {r["name"]: r for r in harness_results(a.pinfer)}
    lla = {r["name"]: r for r in harness_results(a.llama)}
    rows, firsts_p, firsts_l, later_p, later_l = [], [], [], [], []
    print("| request | questions | pinfer s | llama.cpp s | speed-up | pinfer first / later q s | llama.cpp first / later q s |")
    print("|---|---|---|---|---|---|---|")
    for name, lr in lla.items():
        pr = pin.get(name)
        if not pr:
            continue
        pt = pr["timing"]
        p_first = pt["prefix_seconds"] + pt["snapshot_seconds"] + pt["rows"][0]["seconds"]
        p_later = [x["seconds"] for x in pt["rows"][1:]]
        l_first = lr["rows"][0]["seconds"]
        l_later = [x["seconds"] for x in lr["rows"][1:]]
        firsts_p.append(p_first); firsts_l.append(l_first); later_p += p_later; later_l += l_later
        rows.append({"request": name, "questions": len(lr["rows"]), "pinfer_s": pt["total_seconds"],
                     "llama_s": lr["total_seconds"], "speedup": lr["total_seconds"] / pt["total_seconds"]})
        pl = f"{st.mean(p_later):.1f}" if p_later else "—"
        ll = f"{st.mean(l_later):.1f}" if l_later else "—"
        print(f"| {name} | {len(lr['rows'])} | {pt['total_seconds']:.1f} | {lr['total_seconds']:.1f} | "
              f"{lr['total_seconds'] / pt['total_seconds']:.1f}x | {p_first:.1f} / {pl} | {l_first:.1f} / {ll} |")
    tp = sum(r["pinfer_s"] for r in rows); tl = sum(r["llama_s"] for r in rows)
    summary = {"requests": len(rows), "pinfer_total_s": tp, "llama_total_s": tl, "speedup_total": tl / tp,
               "pinfer_median_s": st.median(r["pinfer_s"] for r in rows),
               "llama_median_s": st.median(r["llama_s"] for r in rows),
               "first_q_median_s": {"pinfer": st.median(firsts_p), "llama": st.median(firsts_l)},
               "later_q_median_s": {"pinfer": st.median(later_p), "llama": st.median(later_l)}}
    # probabilities: the Pi's llama.cpp vs the x86 reference, and vs pinfer
    ref, ours = load_ref(a.ref), load_ours(a.pinfer)
    d_ref, d_pin, am_ref, am_pin = [], [], 0, 0
    argmax = lambda p: max(range(len(p)), key=p.__getitem__)
    for name, lr in lla.items():
        for row in lr["rows"]:
            k = (name, row["question"])
            if k in ref:
                d_ref.append(max(abs(x - y) for x, y in zip(row["probs"], ref[k]["p"])))
                am_ref += argmax(row["probs"]) == argmax(ref[k]["p"])
            if k in ours:
                d_pin.append(max(abs(x - y) for x, y in zip(row["probs"], ours[k]["p"])))
                am_pin += argmax(row["probs"]) == argmax(ours[k]["p"])
    summary["llama_pi_vs_ref_x86"] = {"rows": len(d_ref), "max_abs_dp": max(d_ref),
                                      "mean_abs_dp": st.mean(d_ref), "argmax_agree": am_ref}
    summary["llama_pi_vs_pinfer"] = {"rows": len(d_pin), "max_abs_dp": max(d_pin),
                                     "mean_abs_dp": st.mean(d_pin), "argmax_agree": am_pin}
    print(json.dumps(summary, indent=1))
    if a.json:
        a.json.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
