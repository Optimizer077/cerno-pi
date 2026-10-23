"""Cerno animation primitives, retaining make_animation.py's fonts/palette/GIF settings."""
from __future__ import annotations
import json
from pathlib import Path
from PIL import Image, ImageChops, ImageDraw, ImageFont
from figstyle import Layout, Theme

FONT_DIRS = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu")]
FPS = 5.0  # unchanged from the existing Cerno animation
QA = Path(__file__).resolve().parents[1] / "build" / "animation-qa"


def font(size_css: float, scale: int, bold: bool = False):
    names = (["seguisb.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf"] if bold
             else ["segoeui.ttf", "DejaVuSans.ttf"])
    for d in FONT_DIRS:
        for n in names:
            if (d / n).is_file():
                return ImageFont.truetype(str(d / n), round(size_css * scale))
    return ImageFont.load_default(round(size_css * scale))


def wrap(text, f, width):
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}" if cur else word
        if cur and f.getlength(trial) > width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return lines + [cur]


class Canvas:
    def __init__(self, t: Theme, L: Layout):
        self.t, self.L, self.s = t, L, L.scale
        self.W, self.m = L.width_px, round(L.margin * L.scale)
        self.text_font, self.bold = font(L.text, self.s), font(L.text, self.s, True)
        self.title, self.sub = font(L.title, self.s, True), font(L.sub, self.s)

    def px(self, x):
        return round(x * self.s)

    def line(self, d, x, y, text, f=None, color=None):
        f = f or self.text_font
        assert x + f.getlength(text) <= self.W-self.m+1, f'text clipped: {text}'
        d.text((x, y), text, font=f, fill=color or self.t.ink2)
        return y + f.getbbox("Ag")[3] + self.px(4)

    def para(self, d, x, y, text, width, f=None, color=None):
        f = f or self.text_font
        for s in wrap(text, f, width):
            y = self.line(d, x, y, s, f, color)
        return y

    def new(self, css_height):
        im = Image.new("RGB", (self.W, self.px(css_height)), self.t.surface)
        return im, ImageDraw.Draw(im)


def save_animation(images, durations, path: Path, t: Theme, L: Layout):
    """Existing 96-colour shared palette, no dithering, optimize/disposal=1; verify every frame.

    Add a brief fade back to the first frame: the final decoded frame equals the first.
    This transition and answer holds are playback aids, not measured Pi execution time.
    """
    blank = Image.new("RGB", images[0].size, t.surface)
    bottom = max((ImageChops.difference(im, blank).getbbox() or (0, 0, 0, 0))[3] for im in images)
    height = bottom + round(L.margin * L.scale)
    if height > images[0].height:
        raise ValueError(f"{path.name}: canvas clipped")
    images = [im.crop((0, 0, im.width, height)) for im in images]
    images += [Image.blend(images[-1], images[0], a) for a in (.33, .67, 1.0)]
    durations = list(durations) + [200, 200, 400]
    colors = [t.surface, t.ink, t.ink2, t.muted, t.grid, t.axis, t.blue, t.orange,
              t.aqua, t.yellow, t.magenta]
    sw = round(60 * L.scale)
    src = Image.new("RGB", (images[0].width, height + sw), t.surface)
    # Sample content-rich frames, including all labels and answers, for the existing palette method.
    src.paste(images[-4], (0, 0))
    for i, c in enumerate(colors):
        src.paste(c, (i * sw, height, (i + 1) * sw, height + sw))
    palette = src.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    frames = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=True, disposal=1)
    # Pillow coalesces adjacent equal frames. Compare decoded composites and summed durations.
    runs = []
    for im, ms in zip(frames, durations):
        rgb = im.convert("RGB")
        if runs and rgb.tobytes() == runs[-1][0].tobytes():
            runs[-1][1] += ms
        else:
            runs.append([rgb, ms])
    with Image.open(path) as decoded:
        assert decoded.n_frames == len(runs), (path, decoded.n_frames, len(runs))
        for i, (expected, ms) in enumerate(runs):
            decoded.seek(i)
            assert decoded.convert("RGB").tobytes() == expected.tobytes(), (path, i)
            assert decoded.info['duration'] == ms, (path, i, decoded.info['duration'], ms)
        assert runs[0][0].tobytes() == runs[-1][0].tobytes(), 'loop seam differs'
    assert path.stat().st_size <= 1_500_000, f"{path.name}: over 1.5 MB"
    QA.mkdir(parents=True, exist_ok=True)
    indices = [0, len(runs) // 2, len(runs) - 4, len(runs) - 1]
    labels = ['Start', 'Middle', 'Complete', 'Loop seam']
    tw = L.css_width  # contact sheets show each panel at its intended CSS display width
    th = round(height * tw / images[0].width)
    sheet = Image.new('RGB', (tw * 2, (th + 30) * 2), t.surface)
    sd = ImageDraw.Draw(sheet)
    for k, (i, label) in enumerate(zip(indices, labels)):
        x, y = k % 2 * tw, k // 2 * (th + 30)
        sd.text((x + 8, y + 4), label, font=font(15, 1, True), fill=t.ink)
        sheet.paste(runs[i][0].resize((tw, th), Image.Resampling.LANCZOS), (x, y + 30))
    sheet.save(QA / (path.stem + '-contact.png'))
    meta = dict(file=path.name, bytes=path.stat().st_size, width=images[0].width, height=height,
                frames=len(runs), loop_ms=sum(durations), decoded_frames_match=True,
                seamless=True, min_font_css=13)
    (QA / (path.stem + '.json')).write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta), flush=True)
    return path
