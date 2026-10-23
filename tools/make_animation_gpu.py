"""Cerno CPU+GPU explanation and measured comparisons; four README GIF variants.

Every measured constant and its source is in animation_measurements.py. Matrix-stage
motion is explicitly schematic; prompt counters use the measured whole-prompt latency.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from animkit import Canvas, FPS, save_animation
from animation_measurements import (
    CPU, GPU, PREFILL, GPU_SHARE, GPU_BO_MIB, GPU_CAP_MIB, GPU_MATRICES,
    ELIGIBLE_MATRICES, CPU_CORES, QPUS, DECISION_CPU_S, DECISION_GPU_S, MAX_DP, N_QUESTIONS,
    CLOCK_GHZ, PROBS,
)
from figstyle import DESKTOP, MOBILE, THEMES, on_fill

OUT = Path(__file__).resolve().parents[1]/'docs/figures'
SPEED = 4.0


@dataclass
class Frame:
    n: int
    elapsed: float
    phase: int
    ms: int = 200


def timeline():
    frames = []
    for n, rates in PREFILL.items():
        frames.append(Frame(n,0,0,900))
        nf = round(rates['cpu_s']/SPEED*FPS)
        for i in range(1,nf+1):
            frames.append(Frame(n,rates['cpu_s']*i/nf,(i//2)%4))
        frames[-1].ms = 2200
    return frames


class Painter(Canvas):
    def header(self,d,y):
        w = self.W-2*self.m
        y = self.para(d,self.m,y,'Cerno 4B: the CPU and GPU read the prompt together',
                      w,self.title,self.t.ink)
        y = self.para(d,self.m,y+self.px(3),
            f'Pi 4 / {CLOCK_GHZ:g} GHz · opt-in --gpu {GPU_SHARE:.2f} · three paired rounds · model already loaded.',
            w,self.sub)
        return y+self.px(12)

    def arrow(self,d,a,b,color):
        d.line([a,b],fill=color,width=self.px(1))
        x,y = b
        d.polygon([(x,y),(x-self.px(4),y-self.px(6)),(x+self.px(4),y-self.px(6))],fill=color)

    def matrix(self,d,y,fr):
        t,x,w = self.t,self.m,self.W-2*self.m
        y = self.para(d,x,y,'Inside one selected Q4 matrix · stages are schematic, not to scale',
                      w,self.bold,t.ink)
        y = self.para(d,x,y,f'Upload once: {GPU_BO_MIB:.1f} MiB / {GPU_CAP_MIB} MiB cap · '
                      f'{GPU_MATRICES}/{ELIGIBLE_MATRICES} matrices cached.',w)
        y += self.px(5)
        split = x+w*GPU_SHARE
        h = self.px(29)
        d.rectangle((x,y,split,y+h),fill=t.orange)
        d.rectangle((split,y,x+w,y+h),fill=t.blue)
        self.line(d,x+self.px(6),y+self.px(3),'~1/3 GPU',color=on_fill(t,t.orange))
        self.line(d,split+self.px(6),y+self.px(3),'~2/3 CPU rows',color=on_fill(t,t.blue))
        boxy = y+h+self.px(25)
        gap = self.px(16)
        bw = (w-gap)/2
        bh = self.px(100)
        lanes = [(x,t.orange,'VideoCore VI',f'{QPUS} QPUs',QPUS),
                 (x+bw+gap,t.blue,'Cortex-A72',f'{CPU_CORES} cores · NEON',CPU_CORES)]
        self.arrow(d,(x+w*GPU_SHARE/2,y+h),(x+bw/2,boxy),t.ink2)
        self.arrow(d,(split+(w-w*GPU_SHARE)/2,y+h),(x+bw+gap+bw/2,boxy),t.ink2)
        for k,(bx,col,title,sub,nicons) in enumerate(lanes):
            d.rectangle((bx,boxy,bx+bw,boxy+bh),outline=t.axis,width=self.px(1))
            yy = self.line(d,bx+self.px(8),boxy+self.px(5),title,self.bold,t.ink)
            yy = self.line(d,bx+self.px(8),yy,sub)
            iconw = self.px(12 if nicons==8 else 22)
            igap = self.px(4)
            for i in range(nicons):
                ix = bx+self.px(8)+i*(iconw+igap)
                d.rectangle((ix,yy+self.px(2),ix+iconw,yy+self.px(18)),
                            fill=col if fr.phase in (1,2) or (k==0 and fr.phase==0) else t.grid)
            by = boxy+bh-self.px(17)
            fill = ([.12,.65,1,0] if k==0 else [0,.55,1,0])[fr.phase]
            d.rectangle((bx+self.px(8),by,bx+bw-self.px(8),by+self.px(7)),fill=t.grid)
            if fill:
                d.rectangle((bx+self.px(8),by,bx+self.px(8)+(bw-self.px(16))*fill,
                             by+self.px(7)),fill=col)
        jy = boxy+bh+self.px(20)
        self.arrow(d,(x+bw/2,boxy+bh),(x+w/2,jy),t.ink2)
        self.arrow(d,(x+bw+gap+bw/2,boxy+bh),(x+w/2,jy),t.ink2)
        if fr.phase == 2:
            radius = self.px(3)
            d.ellipse((x+w/2-radius,jy-radius,x+w/2+radius,jy+radius),fill=t.yellow)
        phases = ['GPU submitted first', 'CPU computes meanwhile', 'Wait, join the rows', 'Next matrix']
        y = self.line(d,x,jy+self.px(3),phases[fr.phase],self.bold,t.ink)
        return self.para(d,x,y,'Repeat for prefill batches ≥7 tokens; decode stays on CPU.',w)+self.px(14)

    def meter(self,d,x,y,w,fr):
        t,r = self.t,PREFILL[fr.n]
        y = self.para(d,x,y,f'{fr.n}-token prompt · Pi time {fr.elapsed:.1f} s',w,self.bold,t.ink)
        y = self.para(d,x,y,'Whole-prompt counters, played 4× faster.',w)
        for label,key,col in [('CPU only','cpu',t.blue),('CPU + GPU','gpu',t.orange)]:
            progress = min(1,fr.elapsed/r[key+'_s'])
            tokens = min(fr.n,round(fr.n*progress))
            y = self.para(d,x,y,f'{label}: {tokens}/{fr.n} tokens · {r[key+"_tps"]:.2f} tok/s',w)
            d.rectangle((x,y,x+w,y+self.px(13)),fill=t.grid)
            if progress:
                d.rectangle((x,y,x+w*progress,y+self.px(13)),fill=col)
            y += self.px(23)
        y = self.para(d,x,y,f'P64: {PREFILL[64]["cpu_tps"]:.2f} → {PREFILL[64]["gpu_tps"]:.2f} tok/s',w)
        y = self.para(d,x,y,f'P256: {PREFILL[256]["cpu_tps"]:.2f} → {PREFILL[256]["gpu_tps"]:.2f} tok/s',w)
        return y+self.px(12)

    def request(self,d,x,y,w):
        t = self.t
        y = self.para(d,x,y,'Same request: lights + alarm · mean seconds',w,self.bold,t.ink)
        labelw = w*.46
        cell = (w-labelw)/2
        a,b = x+labelw,x+labelw+cell
        y = self.line(d,a,y,'CPU only')
        self.line(d,b,y-self.text_font.getbbox('Ag')[3]-self.px(4),'CPU+GPU')
        rows = [('Shared prefix',CPU['prefix_s'],GPU['prefix_s']),
                ('Q1 · intent',CPU['questions'][0],GPU['questions'][0]),
                ('Q2 · multiple intents',CPU['questions'][1],GPU['questions'][1]),
                ('Whole request',CPU['total_s'],GPU['total_s'])]
        maxv = CPU['total_s']
        for name,v1,v2 in rows:
            yy = y
            f = self.bold if name=='Whole request' else self.text_font
            self.line(d,x,yy,name,f,t.ink if name=='Whole request' else t.ink2)
            for xx,v,col in [(a,v1,t.blue),(b,v2,t.orange)]:
                self.line(d,xx,yy,f'{v:.2f} s',f,t.ink)
                by = yy+self.px(20)
                d.rectangle((xx,by,xx+cell-self.px(8),by+self.px(7)),fill=t.grid)
                d.rectangle((xx,by,xx+(cell-self.px(8))*v/maxv,by+self.px(7)),fill=col)
            y += self.px(36)
        counts = ' / '.join(str(len(p)) for p in PROBS.values())
        y = self.para(d,x,y,f'Scores only {counts} option labels. Rows include rewind; '
                      'total includes checkpoint and Python overhead.',w)
        return y+self.px(10)

    def frame(self,fr):
        im,d = self.new(1200 if self.L.mobile else 850)
        w = self.W-2*self.m
        y = self.matrix(d,self.header(d,self.m),fr)
        if self.L.mobile:
            y = self.meter(d,self.m,y,w,fr)
            y = self.request(d,self.m,y,w)
        else:
            gap = self.px(36)
            half = (w-gap)/2
            ya = self.meter(d,self.m,y,half,fr)
            yb = self.request(d,self.m+half+gap,y,half)
            y = max(ya,yb)
        gain = 100*(1-DECISION_GPU_S/DECISION_CPU_S)
        y = self.para(d,self.m,y,f'Five-request mean: {DECISION_CPU_S:.2f} → {DECISION_GPU_S:.2f} s '
                      f'per decision · {gain:.1f}% less time.',w,self.bold,self.t.ink)
        y = self.para(d,self.m,y+self.px(8),
            f'Opt-in, off by default. Probabilities within {MAX_DP:.3f} of the reference on the '
            f'{N_QUESTIONS}-question '
            'check; near-50/50 answers may flip.',w)
        self.para(d,self.m,y+self.px(8),'Cerno by Apertix · CPU blue, CPU+GPU orange · load excluded',w)
        return im


def main():
    frames = timeline()
    for L in (DESKTOP,MOBILE):
        for t in THEMES:
            p = Painter(t,L)
            save_animation([p.frame(fr) for fr in frames],[fr.ms for fr in frames],
                           OUT/f'decision_gpu{L.suffix}{t.suffix}.gif',t,L)


if __name__ == '__main__':
    main()
