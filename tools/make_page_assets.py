"""Original cerno-pi branding, hero, request card and scorecard, in four README variants.

    python tools/make_page_assets.py

All measured constants are sourced in page_data.py / animation_measurements.py. Shapes are
original geometric artwork; no Apertix, TypeSafe, Raspberry Pi or model-card images are used.
"""
from __future__ import annotations
import json
from pathlib import Path
from PIL import Image, ImageChops, ImageDraw
import animkit
from animkit import Canvas, FPS, font, save_animation
from figstyle import DESKTOP, MOBILE, THEMES
import page_data as D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'docs/figures'
ASSETS = ROOT/'docs/assets'
QA = ROOT/'build/page-qa'
animkit.QA = QA


def mark(d,x,y,size,t):
    """A decision fork ending in probability bars; original160-unit geometry."""
    s=size/160
    def point(a,b):return x+a*s,y+b*s
    d.rounded_rectangle((*point(8,8),*point(152,152)),radius=36*s,fill=t.blue)
    for a,b in [((38,80),(67,80)),((67,45),(67,115)),((67,45),(103,45)),
                ((67,80),(103,80)),((67,115),(103,115))]:
        d.line([point(*a),point(*b)],fill='#ffffff',width=max(1,round(9*s)))
        for v in (a,b):
            px,py=point(*v);r=4.5*s
            d.ellipse((px-r,py-r,px+r,py+r),fill='#ffffff')
    px,py=point(38,80);r=9*s;d.ellipse((px-r,py-r,px+r,py+r),fill='#ffffff')
    for yy,w in [(39,33),(74,23),(109,13)]:
        d.rounded_rectangle((*point(105,yy),*point(105+w,yy+12)),radius=3*s,fill=t.orange)


def branding():
    ASSETS.mkdir(parents=True,exist_ok=True)
    for t in THEMES:
        for L in (DESKTOP,MOBILE):
            svg=f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 160 160" width="160" height="160" role="img" aria-labelledby="title desc">
<title id="title">cerno-pi</title><desc id="desc">Original decision-fork mark with three probability bars.</desc>
<rect x="8" y="8" width="144" height="144" rx="36" fill="{t.blue}"/>
<path d="M38 80H67M67 45V115M67 45H103M67 80H103M67 115H103" fill="none" stroke="white" stroke-width="9" stroke-linecap="round" stroke-linejoin="round"/>
<circle cx="38" cy="80" r="9" fill="white"/>
<g fill="{t.orange}"><rect x="105" y="39" width="33" height="12" rx="3"/><rect x="105" y="74" width="23" height="12" rx="3"/><rect x="105" y="109" width="13" height="12" rx="3"/></g></svg>'''
            (ASSETS/f'logo{L.suffix}{t.suffix}.svg').write_text(svg,encoding='utf-8')
        im=Image.new('RGB',(1280,640),t.surface);d=ImageDraw.Draw(im)
        mark(d,70,170,250,t)
        for xy,text,size,bold in [((370,135),'cerno-pi',88,True),
                ((375,255),'Decisions on a Raspberry Pi.',38,True),
                ((375,322),'Typed questions. Calibrated probabilities.',30,False),
                ((375,395),f'{D.DECISION_CPU_S:.1f} s CPU / {D.DECISION_GPU_S:.1f} s with GPU',30,True),
                ((375,445),'Mean per decision · 5 requests, 3 rounds',28,False),
                ((375,487),'Model already loaded · GPU is optional',28,False),
                ((375,525),'Cerno 4B by Apertix · powered by pinfer',28,False),
                ((375,565),'GPU: near-50/50 answers may flip',28,False)]:
            f=font(size,1,bold);assert xy[0]+f.getlength(text)<1240,text
            d.text(xy,text,font=f,fill=t.ink if bold else t.ink2)
        im.save(ASSETS/f'social-preview{t.suffix}.png',optimize=True)


def save_png(im,path,t,p):
    blank=Image.new('RGB',im.size,t.surface)
    bottom=ImageChops.difference(im,blank).getbbox()[3]+p.m
    assert bottom<=im.height,'canvas clipped'
    im=im.crop((0,0,im.width,bottom))
    im.quantize(colors=256,method=Image.Quantize.MEDIANCUT,dither=Image.Dither.NONE).save(path,optimize=True)
    print(path.name,im.size,path.stat().st_size,flush=True)


class PagePainter(Canvas):
    def arrow(self,d,a,b):
        d.line((a,b),fill=self.t.ink2,width=self.px(1))
        x,y=b;r=self.px(3)
        if b[0]>a[0]:points=[(x,y),(x-r,y-r),(x-r,y+r)]
        else:points=[(x,y),(x-r,y-r),(x+r,y-r)]
        d.polygon(points,fill=self.t.ink2)

    def box(self,d,x,y,w,h):
        d.rounded_rectangle((x,y,x+w,y+h),radius=self.px(9),outline=self.t.axis,width=self.px(1))

    def input(self,d,x,y,w,full=False):
        p=self.px(12);inner=w-2*p
        top=y;y=self.line(d,x+p,y+p,'STATE',self.bold,self.t.ink)
        y=self.para(d,x+p,y,D.REQUEST['state'],inner)+self.px(9)
        y=self.line(d,x+p,y,'TYPED QUESTIONS' if full else 'QUESTION · choice',self.bold,self.t.ink)
        if full:
            for name,kind in [('kind','choice'),('urgency','score'),('tone','score'),('needs_human','yes/no')]:
                spec=D.REQUEST['questions'][name]
                y=self.para(d,x+p,y,f'{name} ({kind})',inner,self.bold,self.t.ink)
                y=self.para(d,x+p,y,spec['instructions'],inner)+self.px(5)
                if kind=='score':
                    criteria=spec['criteria']
                    y=self.para(d,x+p,y,f'0 = {criteria[0]} · {len(criteria)-1} = {criteria[-1]}',
                                inner)+self.px(5)
        else:
            y=self.para(d,x+p,y,D.REQUEST['questions']['kind']['instructions'],inner)
            y=self.para(d,x+p,y,'6 named options; explicit criteria.',inner)+self.px(7)
        y+=p;self.box(d,x,top,w,y-top)
        return y

    def output(self,d,x,y,w,grow=1,full=False):
        t,p=self.t,self.px(12);inner=w-2*p
        top=y;y=self.line(d,x+p,y+p,'OPTION PROBABILITIES',self.bold,t.ink)
        y=self.line(d,x+p,y,'kind · other is most likely' if grow>0 else 'kind · 6 named options')
        opts=list(D.REQUEST['questions']['kind']['criteria'])
        names=max(self.text_font.getlength(s) for s in opts)+self.px(10)
        left,right=x+p+names,x+w-p-self.px(44)
        for name,v in zip(opts,D.SUPPORT['probabilities']['kind']):
            self.line(d,x+p,y,name)
            d.rectangle((left,y+self.px(4),right,y+self.px(17)),fill=t.grid)
            if grow>0:d.rectangle((left,y+self.px(4),left+(right-left)*v*grow,y+self.px(17)),fill=t.blue)
            if grow>=1:self.line(d,right+self.px(6),y,f'{100*v:.1f}%',color=t.ink)
            y+=self.px(23)
        if full:
            y+=self.px(10)
            for text in [f'urgency (score 0–3): {D.SUPPORT["answers"]["urgency"]["score"]:.2f}',
                         f'tone (score 0–3): {D.SUPPORT["answers"]["tone"]["score"]:.2f}',
                         f'needs_human (yes/no): {100*D.SUPPORT["probabilities"]["needs_human"][1]:.1f}% yes']:
                y=self.para(d,x+p,y,text,inner,self.bold,t.ink)
            y=self.para(d,x+p,y+self.px(5),'Scores are expected levels, not probabilities.',inner)
        y+=p;self.box(d,x,top,w,y-top)
        return y

    def strip(self,d,y,w,elapsed=None):
        t=self.t;x=self.m
        h=self.px(43)
        self.box(d,x,y,w,h)
        label='Raspberry Pi 4 · Cerno 4B · pinfer'
        yy=self.para(d,x+self.px(12),y+self.px(9),label,w-self.px(24),self.bold,t.ink)
        if elapsed is not None:
            progress=min(1,elapsed/D.FIRST_ANSWER_S)
            d.rectangle((x+self.px(12),y+h-self.px(6),x+self.px(12)+(w-self.px(24))*progress,
                         y+h-self.px(3)),fill=t.aqua)
        return max(y+h,yy)+self.px(10)

    def hero(self,elapsed,grow):
        im,d=self.new(800 if self.L.mobile else 500);t=self.t;w=self.W-2*self.m
        y=self.para(d,self.m,self.m,'A request in. Probabilities out.',w,self.title,t.ink)
        y=self.line(d,self.m,y,f'Measured Pi compute time: {elapsed:.2f} / {D.FIRST_ANSWER_S:.2f} s',
                    self.bold,t.ink)+self.px(9)
        if self.L.mobile:
            y=self.input(d,self.m,y,w)
            self.arrow(d,(self.W/2,y+self.px(2)),(self.W/2,y+self.px(8)))
            y+=self.px(10)
            y=self.strip(d,y,w,elapsed)
            self.arrow(d,(self.W/2,y-self.px(8)),(self.W/2,y-self.px(2)))
            y=self.output(d,self.m,y,w,grow)+self.px(10)
        else:
            gap=self.px(24);half=(w-gap)/2
            yi=self.input(d,self.m,y,half)
            yo=self.output(d,self.m+half+gap,y,half,grow)
            self.arrow(d,(self.m+half+self.px(3),y+self.px(80)),
                       (self.m+half+gap-self.px(3),y+self.px(80)))
            y=self.strip(d,max(yi,yo)+self.px(12),w,elapsed)
        y=self.para(d,self.m,y,'Shared state + first question, from a measured four-question request.',w)
        self.para(d,self.m,y,f'Full request: {D.REQUEST_S:.2f} s · CPU / 1.5 GHz · load excluded · playback 6×',w)
        return im

    def request(self):
        im,d=self.new(1200 if self.L.mobile else 800);w=self.W-2*self.m
        y=self.para(d,self.m,self.m,'One real request, three question types',w,self.title,self.t.ink)
        y=self.para(d,self.m,y,'Support triage · default CPU mode · measured option probabilities',w)+self.px(10)
        if self.L.mobile:
            y=self.input(d,self.m,y,w,True)
            self.arrow(d,(self.W/2,y+self.px(2)),(self.W/2,y+self.px(9)))
            y+=self.px(12)
            y=self.strip(d,y,w)
            self.arrow(d,(self.W/2,y-self.px(8)),(self.W/2,y-self.px(2)))
            y=self.output(d,self.m,y,w,1,True)+self.px(12)
        else:
            gap=self.px(24);half=(w-gap)/2
            yi=self.input(d,self.m,y,half,True)
            yo=self.output(d,self.m+half+gap,y,half,1,True)
            self.arrow(d,(self.m+half+self.px(3),y+self.px(100)),
                       (self.m+half+gap-self.px(3),y+self.px(100)))
            y=max(yi,yo)+self.px(12)
        self.para(d,self.m,y,f'4 questions in {D.REQUEST_S:.2f} s, model already loaded. '
                  'Full criteria and all option probabilities are linked below.',w)
        return im

    def scorecard(self):
        im,d=self.new(1050 if self.L.mobile else 650);t=self.t;w=self.W-2*self.m
        y=self.para(d,self.m,self.m,'Cerno 4B on a 4 GB Raspberry Pi 4',w,self.title,t.ink)
        y=self.para(d,self.m,y,'Measured at 1.5 GHz · published Q4_0 weights · model load excluded',w)+self.px(12)
        cards=[('Decision speed',f'{D.SPEEDUP:.1f}× faster',
                'pinfer: 211.9 s vs llama.cpp: 948.0 s, over the same 8-request batch.'),
               ('Prompt reading',f'{D.PREFILL[64]["cpu_tps"]:.1f} vs {D.LLAMA_PREFILL[64]:.1f} tok/s',
                f'pinfer vs llama.cpp, 64-token prompt. pinfer reaches {D.PREFILL[64]["gpu_tps"]:.1f} with the GPU.'),
               ('Peak memory',f'{D.PINFER_RSS_MIB:,.0f} vs {D.LLAMA_RSS_MIB:,.0f} MiB',
                f'pinfer vs llama.cpp on the Pi. GPU mode adds {D.GPU_BO_MIB:.0f} MiB of GPU memory.'),
               ('Max probability difference',f'{D.AGREEMENT["max_abs_dp"]:.3f} vs '
                f'{D.MATCHED["llama_pi_vs_ref_x86"]["max_abs_dp"]:.3f}',
                'pinfer vs llama.cpp against the PC reference. Top matches: 51/52 vs 24/26; check sizes differ.')]
        gap=self.px(16);cw=w if self.L.mobile else (w-gap)/2;ch=self.px(185)
        start=y
        for i,(title,value,note) in enumerate(cards):
            x=self.m+(0 if self.L.mobile else i%2*(cw+gap))
            yy=start+(i if self.L.mobile else i//2)*(ch+gap)
            self.box(d,x,yy,cw,ch)
            yy=self.para(d,x+self.px(12),yy+self.px(12),title,cw-self.px(24),self.bold,t.ink)
            yy=self.line(d,x+self.px(12),yy+self.px(3),value,font(29,self.s,True),t.ink)
            bottom=self.para(d,x+self.px(12),yy+self.px(4),note,cw-self.px(24))
            top=start+(i if self.L.mobile else i//2)*(ch+gap)
            assert bottom<top+ch-self.px(5),'scorecard content overlaps its next card'
        y=start+(4 if self.L.mobile else 2)*(ch+gap)
        y=self.para(d,self.m,y,f'Today’s separate 5-request mean: {D.DECISION_CPU_S:.1f} s CPU / '
                    f'{D.DECISION_GPU_S:.1f} s with GPU (three rounds).',w,self.bold,t.ink)
        self.para(d,self.m,y+self.px(6),'Agreement is with Cerno’s PC engine, not labelled task accuracy. '
                  'Run settings and dataset scopes are documented in CERNO.md.',w)
        return im


def main():
    OUT.mkdir(parents=True,exist_ok=True);QA.mkdir(parents=True,exist_ok=True)
    branding()
    n=round(D.FIRST_ANSWER_S/6*FPS)
    timings=[(0,0,700)]+[(D.FIRST_ANSWER_S*i/n,0,200) for i in range(1,n+1)]
    timings += [(D.FIRST_ANSWER_S,i/3,200) for i in range(1,4)]
    timings[-1]=(D.FIRST_ANSWER_S,1,2400)
    for L in (DESKTOP,MOBILE):
        for t in THEMES:
            p=PagePainter(t,L)
            save_animation([p.hero(el,g) for el,g,ms in timings],[ms for el,g,ms in timings],
                           OUT/f'hero{L.suffix}{t.suffix}.gif',t,L)
            save_png(p.request(),OUT/f'decision_request{L.suffix}{t.suffix}.png',t,p)
            save_png(p.scorecard(),OUT/f'results_scorecard{L.suffix}{t.suffix}.png',t,p)


if __name__=='__main__':
    main()
