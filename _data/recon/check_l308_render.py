#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L3-08 自检：逐幕检查溢出与 JS 错误（官方 check_render.py 只覆盖第一幕）。

用法：.venv/bin/python courseware/_data/recon/check_l308_render.py [--shots DIR]
"""
import os
import sys

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HTML = os.path.join(ROOT, "L3-backbone", "L3-08", "index.html")

OVERFLOW_JS = r"""() => {
  const out = [], vw = innerWidth, vh = innerHeight;
  const root = document.querySelector('#visual');
  if (!root) return ['#visual MISSING'];
  const vr = root.getBoundingClientRect();
  const name = el => (el.className && String(el.className).split(' ')[0]) || el.tagName;
  const box = b => '[' + Math.round(b.left) + ',' + Math.round(b.top) + ',' +
                   Math.round(b.right) + ',' + Math.round(b.bottom) + ']';
  root.querySelectorAll('*').forEach(el => {
    const b = el.getBoundingClientRect();
    if (b.width <= 0 || b.height <= 0) return;
    if (b.left < -1 || b.top < -1 || b.right > vw + 1 || b.bottom > vh + 1) {
      out.push('VIEWPORT ' + name(el) + ' ' + box(b)); return;
    }
    if (el === root.firstElementChild) return;
    if (b.left < vr.left - 1 || b.right > vr.right + 1 ||
        b.top < vr.top - 1 || b.bottom > vr.bottom + 1) {
      out.push('VISUAL ' + name(el) + ' ' + box(b) + ' vs ' + box(vr));
    }
  });
  return out.slice(0, 10);
}"""


def main(argv):
    shots = argv[argv.index("--shots") + 1] if "--shots" in argv else None
    if shots:
        os.makedirs(shots, exist_ok=True)
    chrome = next((p for p in ("/usr/bin/google-chrome", "/usr/bin/chromium-browser",
                               "/usr/bin/google-chrome-stable") if os.path.isfile(p)), None)
    bad = 0
    with sync_playwright() as p:
        br = p.chromium.launch(executable_path=chrome,
                               args=["--allow-file-access-from-files", "--no-sandbox",
                                     "--disable-dev-shm-usage", "--disable-gpu",
                                     "--force-color-profile=srgb"])
        for (w, h) in ((1280, 720), (1920, 1080)):
            pg = br.new_page(viewport={"width": w, "height": h}, device_scale_factor=1)
            errs, cerrs = [], []
            pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.on("console", lambda m: cerrs.append(m.text) if m.type == "error" else None)
            pg.goto("file://" + os.path.abspath(HTML), wait_until="load")
            pg.wait_for_timeout(700)
            n = pg.evaluate("() => RENDER.scenes.length")
            durs = pg.evaluate("() => RENDER.scenes.map(s => s.duration)")
            print("=== %dx%d：%d 幕 ===" % (w, h, n))
            for i in range(n):
                d = durs[i]
                for t in (1200, int(d * 0.55), max(1500, d - 600)):
                    pg.evaluate("([i,t]) => { RENDER.playing = false; RENDER.mount(i, t); }", [i, t])
                    pg.wait_for_timeout(140)
                    ov = pg.evaluate(OVERFLOW_JS)
                    tag = "scene%02d@%5dms" % (i + 1, t)
                    if ov:
                        bad += 1
                        print("  ✗ %s 溢出 %d: %s" % (tag, len(ov), ov[0][:110]))
                    else:
                        print("  ✓ %s" % tag)
                    if shots:
                        pg.screenshot(path=os.path.join(shots, "L308_%dx%d_%02d_%d.png" % (w, h, i + 1, t)))
            if errs:
                bad += 1
                print("  ✗ pageerror %d: %s" % (len(errs), errs[0][:160]))
            if cerrs:
                bad += 1
                print("  ✗ console.error %d: %s" % (len(cerrs), cerrs[0][:160]))
            pg.close()
        br.close()
    print("\n%s" % ("全部通过：0 错误 / 0 溢出" if not bad else "有 %d 项问题" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
