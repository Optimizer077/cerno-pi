"""Cerno's checkpoint decision flow, four themes/layouts, from measured Pi timings.

    python tools/make_animation.py
    python tools/make_animation_gpu.py

Sources and every measured constant: animation_measurements.py. Palette/layout/fonts/FPS
follow the original Cerno animation. Brief events and answer bars pause for readability.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from animkit import Canvas, FPS, save_animation
from animation_measurements import (
    NAME, FLOW, PROBS, COPY_SNAPSHOT_MS, CHECKPOINT_MS, COPY_RESTORE_MS, REWIND_MS,
    CHECKPOINT_BYTES, CLOCK_GHZ,
)
from figstyle import DESKTOP, MOBILE, THEMES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/figures'
SPEED = 4.0  # existing playback acceleration, not a new benchmark


def load():
    req = next(r for r in json.loads((ROOT/'requests/correctness.json').read_text(encoding='utf-8'))
               if r['name'] == NAME)
    qs = []
    for row in FLOW['rows']:
        q = req['questions'][row['qid']]
        opts = ['no', 'yes'] if q['type'] == 'noul' else list(q['criteria'])
        qs.append({**row, 'text': q['instructions'], 'options': opts,
                   'kind': {'noul': 'yes/no', 'choice': 'choice', 'score': 'score'}[q['type']],
                   'probs': PROBS[row['qid']]})
    return req['state'], qs


@dataclass
class Frame:
    t: float
    step: int  # prefix, checkpoint, rewind, question, probabilities, complete
    q: int
    fill: float = 0
    ms: int = 200


def timeline(qs):
    frames = [Frame(0, 0, 0, ms=900)]

    def run(t, dur, step, q):
        n = max(1, round(dur / SPEED * FPS))
        frames.extend(Frame(t + dur * i / n, step, q) for i in range(1, n + 1))
        return t + dur

    t = run(0, FLOW['prefix_s'], 0, 0)
    t = run(t, FLOW['checkpoint_s'], 1, 0)
    frames[-1].ms = 600  # show the brief checkpoint; Pi time still uses the measured 19.5 ms
    for k, q in enumerate(qs):
        if k:
            t = run(t, q['rewind_s'], 2, k)
            frames[-1].ms = 600  # readability hold, not extra Pi execution time
        t = run(t, q['seconds'] - q['rewind_s'], 3, k)
        frames.extend(Frame(t, 4, k, i/3) for i in range(1, 4))
        frames[-1].ms = 1800
    # The recorded total also includes Python/request overhead outside the timed phases.
    frames.append(Frame(FLOW['total_s'], 5, len(qs)-1, 1, 3000))
    return frames


class Painter(Canvas):
    def __init__(self, t, L, data):
        super().__init__(t, L)
        self.state, self.qs = data

    def header(self, d, y):
        w = self.W - 2*self.m
        y = self.para(d, self.m, y, 'Cerno: read the state once, rewind for each question',
                      w, self.title, self.t.ink)
        return self.para(d, self.m, y+self.px(3),
            f'Measured Pi 4 / {CLOCK_GHZ:g} GHz · in-engine checkpoint · 4× playback; brief events held to read.',
            w, self.sub) + self.px(12)

    def track(self, d, y, fr):
        t, m, w = self.t, self.m, self.W-2*self.m
        y = self.line(d, m, y, f'Pi time {fr.t:.2f} / {FLOW["total_s"]:.2f} s', self.bold, t.ink)
        d.rectangle((m, y, m+w, y+self.px(13)), fill=t.grid)
        segments = [(FLOW['prefix_s'], t.blue), (FLOW['checkpoint_s'], t.yellow)]
        segments += [(q['seconds'], t.aqua if k % 2 == 0 else t.magenta)
                     for k, q in enumerate(self.qs)]
        start = 0
        for seconds, color in segments:
            filled = min(seconds, max(0, fr.t-start))
            if filled:
                d.rectangle((m+w*start/FLOW['total_s'], y,
                             m+w*(start+filled)/FLOW['total_s'], y+self.px(13)), fill=color)
            start += seconds
        y += self.px(20)
        keys = [('shared prefix', t.blue), ('checkpoint', t.yellow),
                ('question 1', t.aqua), ('question 2', t.magenta)]
        x = m
        for label, col in keys:
            length = self.px(18) + self.text_font.getlength(label) + self.px(16)
            if x > m and x+length > self.W-m:
                x, y = m, y+self.px(23)
            d.rectangle((x,y+self.px(4),x+self.px(10),y+self.px(14)),fill=col)
            self.line(d,x+self.px(15),y,label)
            x += length
        return y+self.px(30)

    def state_box(self, d, x, y, w, fr):
        t = self.t
        y = self.para(d,x,y,f'Shared state · {FLOW["prefix_tokens"]} tokens · two questions',w,
                      self.bold,t.ink)
        # Full sample text, rather than a clipped or rewritten example.
        y = self.para(d,x,y,self.state,w) + self.px(10)
        q = self.qs[fr.q]
        items = [(0,f'1  Read the shared prefix once: {FLOW["prefix_s"]:.2f} s.'),
                 (1,f'2  Checkpoint inside the engine: {FLOW["checkpoint_s"]*1000:.1f} ms.'),
                 (2 if fr.step == 2 else 3,
                  f'3  {"Rewind, then read" if fr.q else "Read"} question {fr.q+1}: '
                  f'{q["tokens"]} new tokens. The prefix cache stays in place.'),
                 (4,f'4  Score the {len(q["options"])} named options; '
                  'return calibrated probabilities. No generated answer text.')]
        for step, text in items:
            active = fr.step == step or (fr.step == 5 and step == 4)
            y = self.para(d,x,y,text,w,self.bold if active else self.text_font,
                          t.ink if active else t.ink2)+self.px(6)
        return y+self.px(5)

    def checkpoint(self,d,x,y,w,fr):
        t = self.t
        y = self.line(d,x,y,'Checkpoint avoids copying model state',self.bold,t.ink)
        y = self.para(d,x,y,f'Save: {COPY_SNAPSHOT_MS:.0f} → {CHECKPOINT_MS:.0f} ms · '
                      f'restore: {COPY_RESTORE_MS:.0f} → {REWIND_MS:.0f} ms (medians).',w)
        y = self.para(d,x,y,f'{CHECKPOINT_BYTES/1e6:.1f} MB of model state stays in the engine; '
                      'the shared-prefix cache is reused.',w)
        if fr.step == 2:
            y = self.line(d,x,y,f'Rewinding now: {self.qs[fr.q]["rewind_s"]*1000:.1f} ms',
                          self.bold,t.ink)
        return y+self.px(12)

    def question(self,d,x,y,w,fr):
        t = self.t
        if fr.step < 2:
            return self.para(d,x,y,'Questions begin after the checkpoint.',w)
        q = self.qs[fr.q]
        y = self.para(d,x,y,f'Question {fr.q+1}/2 · {q["kind"]} · {q["seconds"]:.2f} s'
                      + (' including rewind' if fr.q else ''),w,self.bold,t.ink)
        y = self.para(d,x,y,q['text'],w)+self.px(6)
        names = max(self.text_font.getlength(s) for s in q['options']) + self.px(10)
        left, right = x+names, x+w-self.px(46)
        assert right-left > self.px(65), 'option labels leave no bar space'
        grow = fr.fill if fr.step == 4 else 1 if fr.step == 5 else 0
        for option, p in zip(q['options'],q['probs']):
            self.line(d,x,y,option)
            d.rectangle((left,y+self.px(3),right,y+self.px(3+self.L.bar)),fill=t.grid)
            if grow:
                d.rectangle((left,y+self.px(3),left+(right-left)*p*grow,
                             y+self.px(3+self.L.bar)),fill=t.blue)
            if grow == 1:
                self.line(d,right+self.px(6),y,f'{p:.2f}',color=t.ink)
            y += self.px(self.L.bar+9)
        y += self.px(12)
        answered = fr.q + (fr.step >= 4)
        if answered:
            y = self.line(d,x,y,'Answered',self.bold,t.ink)
            for a in self.qs[:answered]:
                j = max(range(len(a['probs'])),key=a['probs'].__getitem__)
                y = self.para(d,x,y,f'{a["qid"]}: {a["options"][j]} '
                              f'({a["probs"][j]:.2f})',w)
        return y

    def frame(self,fr):
        im,d = self.new(1100 if self.L.mobile else 720)
        y = self.track(d,self.header(d,self.m),fr)
        if self.L.mobile:
            w = self.W-2*self.m
            y = self.state_box(d,self.m,y,w,fr)
            y = self.checkpoint(d,self.m,y,w,fr)
            y = self.question(d,self.m,y,w,fr)
        else:
            lw = self.px(380)
            rx = self.m+lw+self.px(36)
            rw = self.W-self.m-rx
            yl = self.state_box(d,self.m,y,lw,fr)
            yl = self.checkpoint(d,self.m,yl,lw,fr)
            yr = self.question(d,rx,y,rw,fr)
            y = max(yl,yr)
        self.para(d,self.m,y+self.px(12),'Cerno 4B by Apertix · measured example, model already loaded',
                  self.W-2*self.m)
        return im


def main():
    data = load()
    frames = timeline(data[1])
    for L in (DESKTOP,MOBILE):
        for t in THEMES:
            p = Painter(t,L,data)
            save_animation([p.frame(fr) for fr in frames], [fr.ms for fr in frames],
                           OUT/f'decision_flow{L.suffix}{t.suffix}.gif',t,L)


if __name__ == '__main__':
    main()
