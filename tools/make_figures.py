"""README figures of cerno-pi, from the result files in results/ (style: tools/figstyle.py).

    python tools/make_figures.py          # -> docs/figures/{decision_example,agreement}{,-mobile}{,-dark}.png

decision_example  one request of requests/correctness.json (support_triage): the probability of
                  every option of its four questions from pinfer on the Pi next to Cerno's
                  llama.cpp engine on the same GGUF (x86 llama-server).
agreement         every option probability of the 52 questions: pinfer vs the reference, and the
                  differences next to llama.cpp's own (f32 vs f16 KV cache).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (
    DASH,
    GAP,
    LAYOUTS,
    MARKER,
    NAME_GAP,
    THEMES,
    Layout,
    Theme,
    header,
    new_fig,
    panel_title,
    px_to_data,
    ring,
    save,
    setup,
    style,
)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PINFER = [RES / "cerno-decide-reuse-a2_20261022_152103.json", RES / "cerno-decide-reuse-b2_20261022_152644.json"]
REF = RES / "ref_llama_pc.jsonl"
REF_F32 = RES / "ref_llama_pc_kvf32_nofa.jsonl"
REQUESTS = ROOT / "requests" / "correctness.json"
EXAMPLE = "support_triage"


def load_pinfer() -> dict[tuple[str, str], list[float]]:
    out = {}
    for p in PINFER:
        for r in json.loads(p.read_text(encoding="utf-8"))["results"]:
            for q, probs in (r.get("probabilities") or {}).items():
                out[(r["name"], q)] = probs
    return out


def load_ref(path: Path) -> dict[tuple[str, str], list[float]]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        for row in r["rows"]:
            out[(r["name"], row["question"])] = row["probs"]
    return out


def option_names(req: dict) -> dict[str, list[str]]:
    """Short option labels per question, in option order (as the answer reports them)."""
    out = {}
    for qid, q in req["questions"].items():
        if q["type"] == "noul":
            out[qid] = ["no", "yes"]
        elif q["type"] == "score":
            out[qid] = [f"{i}: {c}" for i, c in enumerate(q["criteria"])]
        else:
            crit = q["criteria"]
            out[qid] = list(crit) if isinstance(crit, dict) else [str(c) for c in crit]
    return out


def fig_example(t: Theme, L: Layout) -> None:
    req = next(r for r in json.loads(REQUESTS.read_text(encoding="utf-8")) if r["name"] == EXAMPLE)
    pin, ref = load_pinfer(), load_ref(REF)
    names = option_names(req)
    qids = list(req["questions"])
    title_of = {"kind": "kind (choice)", "urgency": "urgency (score)", "tone": "tone (score)",
                "needs_human": "needs_human (yes/no)"}
    n_opts = [len(names[q]) for q in qids]
    name_h = L.text * 1.35 + NAME_GAP if L.mobile else 0
    group = 2 * L.bar + GAP
    slot = group + (name_h + 14 if L.mobile else 10)  # CSS px per option row
    heights = [n * slot + 40 for n in n_opts]
    if L.mobile:
        fig, axes = new_fig(t, L, sum(heights) + 40, nrows=4, ncols=1,
                            gridspec_kw={"height_ratios": heights})
    else:
        rows = [max(heights[0], heights[1]), max(heights[2], heights[3])]
        fig, axes = new_fig(t, L, sum(rows) + 30, nrows=2, ncols=2, gridspec_kw={"height_ratios": rows})
    axes = list(axes.flat)
    for ax, q in zip(axes, qids):
        style(ax, t, grid="x")
        ax.set_xlim(0, 1.16)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", ".25", ".5", ".75", "1"])
        n = len(names[q])
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_yticks([] if L.mobile else range(n), [] if L.mobile else names[q])
        if L.mobile:
            ax.spines["left"].set_visible(False)
        panel_title(ax, t, title_of[q])
    blue, orange = t.blue, t.orange
    keys = [plt.Rectangle((0, 0), 1, 1, color=blue, label="pinfer on the Pi (this repository)"),
            plt.Rectangle((0, 0), 1, 1, color=orange, label="Cerno's llama.cpp engine, same GGUF (reference)")]
    rect = header(fig, t, L, "What one decision looks like: support triage, four questions",
                  'State: "hey do u have showers at the downtown location". Probability of each option; '
                  "both engines read the same Q4_0 weights.", handles=keys, ncol=1 if L.mobile else 2)
    units = [px_to_data(ax, 1, axis="y", rect=rect) for ax in axes]
    for ax, q, unit in zip(axes, qids, units):
        for i, nm in enumerate(names[q]):
            top = i + (name_h - group) / 2 * unit
            for j, (v, c) in enumerate(((pin[(EXAMPLE, q)][i], blue), (ref[(EXAMPLE, q)][i], orange))):
                y = top + (j * (L.bar + GAP) + L.bar / 2) * unit
                ax.barh(y, v, height=L.bar * unit, color=c)
                ax.annotate(f"{v:.3f}", (v, y), xytext=(4, 0), textcoords="offset points", va="center",
                            color=t.ink, fontsize=L.text)
            if L.mobile:
                ax.annotate(nm, (0, top), xytext=(0, NAME_GAP), textcoords="offset points", ha="left",
                            va="bottom", color=t.ink2, annotation_clip=False)
    save(fig, t, L, "decision_example", rect=rect)


def agreement_points() -> tuple[list[float], list[float], list[float], list[float], dict]:
    pin, ref, ref32 = load_pinfer(), load_ref(REF), load_ref(REF_F32)
    xs, ys, rx, ry = [], [], [], []
    am = 0
    worst = 0.0
    for k in sorted(set(pin) & set(ref)):
        p, r = pin[k], ref[k]
        am += max(range(len(p)), key=p.__getitem__) == max(range(len(r)), key=r.__getitem__)
        for a, b in zip(p, r):
            xs.append(b); ys.append(a)
            worst = max(worst, abs(a - b))
        for a, b in zip(ref32[k], r):
            rx.append(b); ry.append(a - b)
    d = [abs(y - x) for x, y in zip(xs, ys)]
    stats = {"questions": len(set(pin) & set(ref)), "options": len(xs), "max": worst,
             "mean": sum(d) / len(d), "top1": am,
             "llama_max": max(abs(v) for v in ry), "llama_mean": sum(abs(v) for v in ry) / len(ry)}
    return xs, ys, rx, ry, stats


def fig_agreement(t: Theme, L: Layout) -> None:
    xs, ys, rx, ry, s = agreement_points()
    if L.mobile:
        fig, (a1, a2) = new_fig(t, L, 760, nrows=2, ncols=1)
    else:
        fig, (a1, a2) = new_fig(t, L, 400, nrows=1, ncols=2)
    style(a1, t, ylabel="pinfer on the Pi", xlabel="reference (llama.cpp, same GGUF)", grid="y")
    style(a2, t, ylabel="difference to the reference", xlabel="reference probability", grid="y")
    for ax in (a1, a2):
        ax.set_xlim(-0.02, 1.02)
    a1.set_ylim(-0.02, 1.02)
    a2.set_ylim(-0.016, 0.016)
    a2.set_yticks([-0.015, -0.01, -0.005, 0, 0.005, 0.01, 0.015],
                  ["−0.015", "−0.010", "−0.005", "0", "+0.005", "+0.010", "+0.015"])
    panel_title(a1, t, "every option probability")
    panel_title(a2, t, "the differences, magnified")
    a1.plot([0, 1], [0, 1], color=t.axis, linewidth=1.0, linestyle=DASH, zorder=1)
    a2.axhline(0, color=t.axis, linewidth=1.0, linestyle=DASH, zorder=1)
    keys = [plt.Line2D([], [], color=t.blue, marker="o", linestyle="", markersize=MARKER - 2,
                       label="pinfer on the Pi vs the reference"),
            plt.Line2D([], [], color=t.orange, marker="o", linestyle="", markersize=MARKER - 2,
                       label="llama.cpp f32 KV vs llama.cpp f16 KV (its own spread)")]
    rect = header(fig, t, L, "pinfer agrees with Cerno's own llama.cpp engine",
                  f"{s['questions']} questions, {s['options']} option probabilities: max |Δp| "
                  f"{s['max']:.4f}, mean over the options {s['mean']:.4f}; same top answer on {s['top1']} of "
                  f"{s['questions']} (the other is a 0.497 vs 0.503 tie). llama.cpp's own spread: "
                  f"max {s['llama_max']:.4f}, mean {s['llama_mean']:.4f}.", handles=keys,
                  ncol=1 if L.mobile else 2)
    a1.plot(xs, ys, "o", color=t.blue, markersize=MARKER - 2, alpha=0.85, **ring(t), zorder=3)
    a2.plot(rx, ry, "o", color=t.orange, markersize=MARKER - 4, alpha=0.8, **ring(t), zorder=2)
    a2.plot(xs, [y - x for x, y in zip(xs, ys)], "o", color=t.blue, markersize=MARKER - 4, alpha=0.85,
            **ring(t), zorder=3)
    save(fig, t, L, "agreement", rect=rect)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for L in LAYOUTS:
        for t in THEMES:
            setup(t, L)
            fig_example(t, L)
            setup(t, L)
            fig_agreement(t, L)
    print(json.dumps(agreement_points()[4]))


if __name__ == "__main__":
    main()
