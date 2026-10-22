"""README animation: how cerno-pi answers one request on the Pi (state prefilled once, then one
short pass per question that reads only the option-label rows of the head).

    python tools/make_animation.py     # -> docs/figures/decision_flow{,-mobile}{,-dark}.gif

The times are the Pi's (request `contract_clause` of requests/correctness.json in
results/cerno-decide-reuse-b2_20261022_152644.json, 1.5 GHz): the animation plays them SPEED
times faster and pauses after each answer so the bars can be read. Colours, type sizes and
widths follow tools/figstyle.py (desktop: 838 CSS px at 2x, phone: 360 CSS px at 3x; text 13 CSS
px and up).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import DESKTOP, MOBILE, THEMES, Layout, Theme

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
RESULT = ROOT / "results" / "cerno-decide-reuse-b2_20261022_152644.json"
REQUESTS = ROOT / "requests" / "correctness.json"
NAME = "contract_clause"
SPEED = 4.0          # playback speed-up of the compute phases
FPS = 5.0            # frames per second while something runs
HOLD_S = 1.8         # pause after each answer (not Pi time)
END_S = 3.0          # pause on the finished request before the loop restarts
FONT_DIRS = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu")]


def font(size_css: float, scale: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = (["seguisb.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf"] if bold
             else ["segoeui.ttf", "DejaVuSans.ttf"])
    for d in FONT_DIRS:
        for n in names:
            if (d / n).is_file():
                return ImageFont.truetype(str(d / n), int(round(size_css * scale)))
    return ImageFont.load_default(int(round(size_css * scale)))


def wrap(text: str, f: ImageFont.FreeTypeFont, width_px: float) -> list[str]:
    lines, cur = [], ""
    for word in text.split(" "):
        trial = f"{cur} {word}" if cur else word
        if cur and f.getlength(trial) > width_px:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return lines + [cur]


@dataclass
class Q:
    qid: str
    kind: str
    text: str
    options: list[str]
    probs: list[float]
    tokens: int
    seconds: float


def load() -> tuple[str, int, float, float, list[Q]]:
    req = next(r for r in json.loads(REQUESTS.read_text(encoding="utf-8")) if r["name"] == NAME)
    res = next(r for r in json.loads(RESULT.read_text(encoding="utf-8"))["results"]
               if r.get("name") == NAME)
    t = res["timing"]
    qs = []
    for row in t["rows"]:
        spec = req["questions"][row["question"]]
        if spec["type"] == "noul":
            opts = ["no", "yes"]
        else:
            crit = spec["criteria"]
            opts = list(crit)          # choice: option names (dict keys or list)
        kind = {"noul": "yes/no", "choice": "choice", "score": "score"}[spec["type"]]
        qs.append(Q(row["question"], kind, spec["instructions"], opts,
                    res["probabilities"][row["question"]], row["tokens"], row["seconds"]))
    return req["state"], t["prefix_tokens"], t["prefix_seconds"], t["snapshot_seconds"], qs


@dataclass
class Frame:
    t_pi: float          # Pi seconds elapsed
    step: int            # 0 prefill, 1 snapshot, 2 question pass, 3 bars, 4 done
    q: int               # current question index
    fill: float          # 0..1 bar growth of question q
    ms: int              # frame duration


def timeline(prefix_s: float, snap_s: float, qs: list[Q]) -> list[Frame]:
    dt = 1000.0 / FPS
    frames: list[Frame] = []

    def run(t0: float, dur: float, step: int, q: int) -> float:
        n = max(1, round(dur / SPEED * FPS))
        for i in range(1, n + 1):
            frames.append(Frame(t0 + dur * i / n, step, q, 0.0, int(dt)))
        return t0 + dur

    frames.append(Frame(0.0, 0, 0, 0.0, 900))                 # the request arrives
    t = run(0.0, prefix_s, 0, 0)
    t = run(t, snap_s, 1, 0)
    for k, q in enumerate(qs):
        t = run(t, q.seconds, 2, k)
        for i in range(1, 4):                                  # bars grow (not Pi time)
            frames.append(Frame(t, 3, k, i / 3, int(dt)))
        frames[-1].ms = int(HOLD_S * 1000)
    frames.append(Frame(t, 4, len(qs) - 1, 1.0, int(END_S * 1000)))
    return frames


class Painter:
    def __init__(self, t: Theme, L: Layout, data: tuple) -> None:
        self.t, self.L, self.s = t, L, L.scale
        self.state, self.n_prefix, self.prefix_s, self.snap_s, self.qs = data
        self.total = self.prefix_s + self.snap_s + sum(q.seconds for q in self.qs)
        self.f_title = font(L.title, self.s, bold=True)
        self.f_sub = font(L.sub, self.s)
        self.f_text = font(L.text, self.s)
        self.f_bold = font(L.text, self.s, bold=True)
        self.W = L.css_width * self.s
        self.m = int(L.margin * self.s)

    def px(self, css: float) -> int:
        return int(round(css * self.s))

    def text(self, d: ImageDraw.ImageDraw, xy: tuple[int, int], s: str, f, fill: str) -> int:
        d.text(xy, s, font=f, fill=fill)
        return f.getbbox("Ag")[3] + self.px(4)

    def para(self, d, x: int, y: int, s: str, f, fill: str, width: int) -> int:
        for line in wrap(s, f, width):
            y += self.text(d, (x, y), line, f, fill)
        return y

    # -------------------------------------------------------------- blocks
    def header(self, d, y: int) -> int:
        t = self.t
        y = self.para(d, self.m, y, "One decision on the Pi: the state is prefilled once, then each "
                      "question needs one short pass", self.f_title, t.ink, self.W - 2 * self.m)
        y += self.px(2)
        sub = (f"Real Pi 4 times (1.5 GHz) for one request of the test set; played {SPEED:g}× "
               "faster, with pauses to read the bars.")
        return self.para(d, self.m, y, sub, self.f_sub, t.ink2, self.W - 2 * self.m) + self.px(10)

    def track(self, d, y: int, fr: Frame) -> int:
        t, m, W = self.t, self.m, self.W
        label = f"Pi time {fr.t_pi:.1f} s of {self.total:.1f} s"
        y += self.text(d, (m, y), label, self.f_bold, t.ink)
        h = self.px(14)
        d.rectangle([m, y, W - m, y + h], fill=t.grid)
        segs = [(self.prefix_s, t.blue), (self.snap_s, t.yellow)]
        segs += [(q.seconds, t.aqua if k % 2 == 0 else t.magenta) for k, q in enumerate(self.qs)]
        x0, acc = m, 0.0
        span = W - 2 * m
        for dur, col in segs:
            a, b = acc, acc + dur
            acc = b
            done = min(max(fr.t_pi - a, 0.0), dur)
            if done > 0:
                d.rectangle([x0 + span * a / self.total, y, x0 + span * (a + done) / self.total, y + h],
                            fill=col)
        y += h + self.px(6)
        keys = [("state prefill", t.blue), ("snapshot", t.yellow), ("questions 1, 3", t.aqua),
                ("questions 2, 4", t.magenta)]
        x = m
        for name, col in keys:
            sq = self.px(10)
            d.rectangle([x, y + self.px(4), x + sq, y + self.px(4) + sq], fill=col)
            d.text((x + sq + self.px(5), y), name, font=self.f_text, fill=t.ink2)
            x += sq + self.px(5) + int(self.f_text.getlength(name)) + self.px(16)
            if x > W - m - self.px(120) and self.L.mobile:
                x = m
                y += self.px(20)
        return y + self.px(28)

    def state_box(self, d, x: int, y: int, w: int, fr: Frame) -> int:
        t = self.t
        y += self.text(d, (x, y), f"State: {self.n_prefix} tokens, shared by all "
                       f"{len(self.qs)} questions", self.f_bold, t.ink)
        lines = wrap(self.state, self.f_text, w - self.px(30))
        lines = lines[:4 if self.L.mobile else 7]
        lines[-1] = lines[-1].rstrip(".,") + " …"
        lh = self.f_text.getbbox("Ag")[3] + self.px(4)
        box_h = len(lines) * lh + self.px(16)
        done = 1.0 if fr.step > 0 else fr.t_pi / self.prefix_s
        d.rectangle([x, y, x + w, y + box_h], outline=t.axis, width=self.px(1))
        if done > 0:     # the prefilled part of the state, as a band along the box's left edge
            d.rectangle([x, y, x + self.px(5), y + box_h * done], fill=t.blue)
        for i, line in enumerate(lines):
            d.text((x + self.px(14), y + self.px(8) + i * lh), line, font=self.f_text, fill=t.ink2)
        return y + box_h + self.px(14)

    def steps(self, d, x: int, y: int, w: int, fr: Frame) -> int:
        t = self.t
        q = self.qs[fr.q]
        # the first question runs on the live state; every later one restores the snapshot first
        start = "Restore it, run" if fr.q else "Run"
        items = [
            (0, f"1  Prefill the state once: {self.n_prefix} tokens, {self.prefix_s:.1f} s"),
            (1, f"2  Save a snapshot of the model state: {self.snap_s:.2f} s"),
            (2, f"3  {start} the question's {q.tokens} tokens, read only the "
                f"{len(q.options)} option-label rows of the 248,320-row head"),
            (3, "4  Probabilities = softmax(label logits / 1.534)"),
        ]
        for step, s in items:
            active = fr.step == step or (fr.step == 4 and step == 3)
            f = self.f_bold if active else self.f_text
            col = t.ink if active else t.ink2
            y = self.para(d, x, y, s, f, col, w) + self.px(4)
        return y

    def question(self, d, x: int, y: int, w: int, fr: Frame) -> int:
        t = self.t
        k = fr.q
        q = self.qs[k]
        if fr.step < 2:
            y += self.text(d, (x, y), "Questions start after the snapshot", self.f_text, t.ink2)
            return y
        what = "restore + " if k else ""
        y += self.text(d, (x, y), f"Question {k + 1} of {len(self.qs)} · {q.kind} · {q.seconds:.1f} s "
                       f"({what}{q.tokens} tokens)", self.f_bold, t.ink)
        y = self.para(d, x, y, q.text, self.f_text, t.ink2, w) + self.px(6)
        name_w = max(self.f_text.getlength(o) for o in q.options) + self.px(10)
        bar_x0, bar_x1 = x + name_w, x + w - self.px(52)
        grow = fr.fill if fr.step == 3 else (1.0 if fr.step == 4 else 0.0)
        bh = self.px(self.L.bar)
        for o, p in zip(q.options, q.probs):
            d.text((x, y), o, font=self.f_text, fill=t.ink2)
            d.rectangle([bar_x0, y + self.px(2), bar_x1, y + self.px(2) + bh], fill=t.grid)
            if grow > 0:
                d.rectangle([bar_x0, y + self.px(2), bar_x0 + (bar_x1 - bar_x0) * p * grow,
                             y + self.px(2) + bh], fill=t.blue)
            if grow >= 1:
                d.text((bar_x1 + self.px(6), y), f"{p:.2f}", font=self.f_text, fill=t.ink)
            y += bh + self.px(8)
        return y + self.px(6)

    def answered(self, d, x: int, y: int, w: int, fr: Frame) -> int:
        t = self.t
        n = fr.q if fr.step < 3 else fr.q + 1
        if n == 0:
            return y
        y += self.text(d, (x, y), "Answered", self.f_bold, t.ink)
        for q in self.qs[:n]:
            j = max(range(len(q.probs)), key=q.probs.__getitem__)
            y += self.text(d, (x, y), f"{q.qid}: {q.options[j]} ({q.probs[j]:.2f})", self.f_text, t.ink2)
        return y

    # -------------------------------------------------------------- frame
    def frame(self, fr: Frame, height: int) -> Image.Image:
        img = Image.new("RGB", (self.W, height), self.t.surface)
        d = ImageDraw.Draw(img)
        y = self.header(d, self.m)
        y = self.track(d, y, fr)
        if self.L.mobile:
            w = self.W - 2 * self.m
            y = self.state_box(d, self.m, y, w, fr)
            y = self.steps(d, self.m, y, w, fr) + self.px(8)
            y = self.question(d, self.m, y, w, fr)
            self.answered(d, self.m, y, w, fr)
        else:
            lw = self.px(380)
            rx = self.m + lw + self.px(36)
            rw = self.W - self.m - rx
            yl = self.state_box(d, self.m, y, lw, fr)
            self.steps(d, self.m, yl, lw, fr)
            yr = self.question(d, rx, y, rw, fr)
            self.answered(d, rx, yr + self.px(4), rw, fr)
        return img


def render(t: Theme, L: Layout, data: tuple) -> Path:
    p = Painter(t, L, data)
    frames = timeline(data[2], data[3], data[4])
    height = p.px(1100 if L.mobile else 600)  # generous canvas, cropped below
    imgs = [p.frame(fr, height) for fr in frames]
    # crop every frame to the lowest content of any frame (+ the margin)
    blank = Image.new("RGB", imgs[0].size, t.surface)
    bottom = max((ImageChops.difference(im, blank).getbbox() or (0, 0, 0, 0))[3] for im in imgs)
    if bottom + p.m > height:
        raise ValueError(f"{L.name}: content ({bottom} px) does not fit the canvas ({height} px)")
    height = bottom + p.m
    imgs = [im.crop((0, 0, p.W, height)) for im in imgs]
    # one palette for all frames: the last frame (every text and bar) plus a large swatch of each
    # theme colour, so a colour that covers only a few pixels (the snapshot) keeps its own entry
    cols = [t.surface, t.ink, t.ink2, t.muted, t.grid, t.axis, t.blue, t.orange, t.aqua, t.yellow,
            t.magenta]
    sw = p.px(60)
    src = Image.new("RGB", (p.W, height + sw), t.surface)
    src.paste(imgs[-1], (0, 0))
    for i, c in enumerate(cols):
        src.paste(_rgb(c), (i * sw, height, (i + 1) * sw, height + sw))
    palette = src.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    pal = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in imgs]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"decision_flow{L.suffix}{t.suffix}.gif"
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=[fr.ms for fr in frames],
                loop=0, optimize=True, disposal=1)
    print(f"wrote {path.name}: {len(frames)} frames, {p.W} x {height}, "
          f"{path.stat().st_size / 1e6:.2f} MB, {sum(fr.ms for fr in frames) / 1000:.1f} s per loop")
    return path


def _rgb(hexcol: str) -> tuple[int, int, int]:
    return tuple(int(hexcol.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def main() -> None:
    data = load()
    for L in (DESKTOP, MOBILE):
        for t in THEMES:
            render(t, L, data)


if __name__ == "__main__":
    main()
