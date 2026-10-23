"""Render README with GitHub's API and screenshot its local images in four layouts.

Requires gh authentication and Playwright; use an isolated environment for previewing.
Only the Markdown rendering API is POSTed; no repository content is published.
GitHub's own current CSS is cached under ignored build/page-qa.
"""
from __future__ import annotations
import functools
import hashlib
from html import escape, unescape
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import subprocess
import threading
import urllib.parse
import urllib.request
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
QA=ROOT/'build/page-qa'
REPO='Optimizer077/cerno-pi'


def fetch(url,path):
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        request=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(request,timeout=25) as r:path.write_bytes(r.read())
    return path


class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='link' and a.get('rel')=='stylesheet':
            url=a.get('href') or a.get('data-href')
            if url and url not in self.links:self.links.append(url)


class LocalImages(HTMLParser):
    def __init__(self):super().__init__(convert_charrefs=False);self.parts=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='img':
            canonical=a.get('data-canonical-src','')
            if canonical.startswith('https://img.shields.io/'):
                name=hashlib.sha256(canonical.encode()).hexdigest()[:12]+'.svg'
                fetch(canonical,QA/'badges'/name)
                a['src']='/build/page-qa/badges/'+name
            a['loading']='eager'
        for key in ('src','srcset'):
            if a.get(key,'').startswith('docs/'):a[key]='/'+a[key]
        href=a.get('href','')
        if href and not href.startswith(('#','/','http:','https:','mailto:')):
            a['href']='/'+href
        self.parts.append('<'+tag+''.join(' '+k+('="'+escape(v,quote=True)+'"' if v is not None else '')
                                         for k,v in a.items())+'>')
    def handle_endtag(self,tag):self.parts.append('</'+tag+'>')
    def handle_data(self,s):self.parts.append(s)
    def handle_entityref(self,s):self.parts.append('&'+s+';')
    def handle_charref(self,s):self.parts.append('&#'+s+';')
    def handle_comment(self,s):self.parts.append('<!--'+s+'-->')


def build():
    QA.mkdir(parents=True,exist_ok=True)
    payload={'text':(ROOT/'README.md').read_text(encoding='utf-8'),'mode':'gfm','context':REPO}
    infile=QA/'markdown-payload.json';infile.write_text(json.dumps(payload),encoding='utf-8')
    response=subprocess.run(['gh','api','markdown','--method','POST','--input',str(infile)],
                            cwd=ROOT,capture_output=True,check=True)
    raw=response.stdout.decode('utf-8');(QA/'github-rendered.html').write_text(raw,encoding='utf-8')
    parser=LocalImages();parser.feed(raw);body=''.join(parser.parts)
    # The standalone API fragment omits the heading IDs GitHub's repository UI supplies.
    def anchor(m):
        text=unescape(re.sub(r'<[^>]*>','',m[3])).lower()
        slug=re.sub(r'[^\w\s-]','',text).strip().replace(' ','-')
        return f'<h{m[1]}{m[2]} id="{slug}">{m[3]}</h{m[1]}>'
    body=re.sub(r'<h([1-6])([^>]*)>(.*?)</h\1>',anchor,body,flags=re.S)
    live=fetch('https://github.com/'+REPO,QA/'github-live.html').read_text(encoding='utf-8')
    links=Links();links.feed(live);css=[]
    prefixes=('light-','dark-','primer-primitives-','primer-','global-','github-','repository-','code-')
    for url in links.links:
        name=url.rsplit('/',1)[-1]
        if not name.startswith(prefixes) or '.module.' in name:continue
        path=fetch(url,QA/'css'/name)
        # Keep any font/image URLs relative to GitHub's original asset host.
        text=path.read_text(encoding='utf-8')
        text=re.sub(r'url\(([^)]+)\)',lambda m:'url('+urllib.parse.urljoin(url,m[1].strip('\"\''))+')',text)
        css.append('<style>'+text+'</style>')
    (QA/'css-sources.json').write_text(json.dumps(links.links,indent=2),encoding='utf-8')
    chrome='''<style>
*{box-sizing:border-box}body{margin:0;padding:32px 0;background:var(--bgColor-default);color:var(--fgColor-default)}
.preview-label{max-width:904px;margin:0 auto 12px;font:14px system-ui;color:var(--fgColor-muted)}
article.markdown-body{max-width:904px;margin:0 auto;padding:32px;border:1px solid var(--borderColor-default);border-radius:6px}
.markdown-body img{max-width:100%;height:auto}.markdown-body pre,.markdown-body table{max-width:100%;overflow:auto}
@media(max-width:600px){body{padding:0}.preview-label{padding:12px 16px;margin:0}article.markdown-body{padding:16px;border:0;border-radius:0}}
</style>'''
    for theme in ('light','dark'):
        page=f'''<!doctype html><html lang="en" data-color-mode="{theme}" data-light-theme="light" data-dark-theme="dark">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>cerno-pi README preview</title>{''.join(css)}{chrome}</head>
<body><div class="preview-label">Optimizer077 / cerno-pi · GitHub-rendered README preview</div>
<article class="markdown-body entry-content container-lg">{body}</article></body></html>'''
        (QA/f'preview-{theme}.html').write_text(page,encoding='utf-8')


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self,*args):pass


def screenshot():
    handler=functools.partial(Quiet,directory=str(ROOT))
    server=ThreadingHTTPServer(('127.0.0.1',0),handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    cache=Path.home()/'AppData/Local/ms-playwright'
    candidates=sorted(cache.glob('chromium-*/chrome-win64/chrome.exe'),
                      key=lambda x:int(x.parents[1].name.split('-')[1]))
    report=[]
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True,executable_path=str(candidates[-1]),
                args=['--renderer-process-limit=1','--num-raster-threads=2','--disable-background-networking'])
            for name,width,height in [('desktop',1280,1000),('phone',412,915)]:
                for theme in ('light','dark'):
                    context=browser.new_context(viewport={'width':width,'height':height},
                                                color_scheme=theme,device_scale_factor=1)
                    page=context.new_page()
                    page.goto(f'http://127.0.0.1:{server.server_port}/build/page-qa/preview-{theme}.html',
                              wait_until='networkidle')
                    page.evaluate('''async()=>{await document.fonts.ready;await Promise.all([...document.images].map(i=>i.decode().catch(()=>{})))}''')
                    page.wait_for_timeout(3900)  # hero's completed answer hold
                    state=page.evaluate('''()=>({width:innerWidth,scroll:document.documentElement.scrollWidth,
                       broken:[...document.images].filter(i=>!i.complete||i.naturalWidth===0).map(i=>i.src),
                       pictures:[...document.querySelectorAll('picture img')].map(i=>({src:i.currentSrc,width:i.width,natural:i.naturalWidth})),
                       badges:[...document.querySelectorAll('img[data-canonical-src]')].map(i=>({width:i.width,height:i.height})),
                       headings:[...document.querySelectorAll('h2')].map(i=>i.textContent.trim())})''')
                    assert not state['broken'],state['broken']
                    assert state['scroll']<=width,(name,theme,state['scroll'])
                    stem=f'readme-{name}-{theme}'
                    page.screenshot(path=str(QA/(stem+'-top.png')))
                    page.screenshot(path=str(QA/(stem+'.png')),full_page=True)
                    # Additional views make the lower sections inspectable at full text size.
                    for slug in ('what-a-decision-looks-like','results-on-the-pi','how-it-works','quickstart','credit-and-licence'):
                        anchor=page.locator('#'+slug)
                        assert anchor.count()==1,('missing section anchor',slug)
                        anchor.evaluate("e=>e.closest('h2').scrollIntoView({block:'start'})")
                        # Let Chromium paint after a long scroll before capturing the viewport.
                        page.wait_for_timeout(200)
                        page.screenshot(path=str(QA/(stem+'-'+slug+'.png')))
                    page.locator('details').evaluate_all('xs=>xs.forEach(x=>x.open=true)')
                    expanded=page.evaluate('document.documentElement.scrollWidth')
                    assert expanded<=width,(name,theme,'expanded',expanded)
                    state['expanded_scroll_width']=expanded
                    page.screenshot(path=str(QA/(stem+'-expanded.png')),full_page=True)
                    report.append({'name':name,'theme':theme,**state})
                    context.close()
            browser.close()
    finally:server.shutdown();server.server_close()
    (QA/'browser-checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps([{'name':r['name'],'theme':r['theme'],'width':r['width'],'scroll':r['scroll'],
                       'broken':r['broken']} for r in report],indent=2))


if __name__=='__main__':
    build();screenshot()
