#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐幕溢出检查（check_render.py 只在第 1 幕采样，这里补全 9 幕 x 3 个时刻 x 2 分辨率）。"""
import sys

from playwright.sync_api import sync_playwright

URL = "file:///data/WORKSPACE/transformer-project/courseware/L3-backbone/L3-01/index.html"

OVERFLOW_JS = """() => {
  const out = [], vw = innerWidth, vh = innerHeight;
  const root = document.querySelector('#visual');
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

bad = 0
with sync_playwright() as p:
    br = p.chromium.launch(
        executable_path="/usr/bin/google-chrome",
        args=["--allow-file-access-from-files", "--no-sandbox", "--disable-dev-shm-usage",
              "--disable-gpu", "--force-color-profile=srgb"])
    for (w, h) in ((1280, 720), (1920, 1080)):
        pg = br.new_page(viewport={"width": w, "height": h}, device_scale_factor=1)
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: errs.append("console:" + m.text) if m.type == "error" else None)
        pg.goto(URL, wait_until="load")
        pg.wait_for_timeout(700)
        n = pg.evaluate("() => RENDER.scenes.length")
        pg.evaluate("() => RENDER.pause()")
        for i in range(n):
            pg.evaluate("(i) => RENDER.go(i, 0)", i)
            pg.wait_for_timeout(360)
            for t in (300, 1200, 3000, 8000, 14000):
                pg.evaluate("(t) => RENDER.seek(t)", t)
                pg.wait_for_timeout(200)
                ov = pg.evaluate(OVERFLOW_JS)
                if ov:
                    bad += 1
                    print(f"  ✗ {w}x{h} 第 {i+1} 幕 @{t}ms 溢出 {len(ov)}: {ov[0][:110]}")
            print(f"  · {w}x{h} 第 {i+1} 幕 检查完毕")
        if errs:
            bad += 1
            print(f"  ✗ {w}x{h} JS 错误 {len(errs)}: {errs[0][:160]}")
        pg.close()
    br.close()
print("逐幕溢出检查:", "全部通过" if bad == 0 else f"{bad} 处问题")
sys.exit(1 if bad else 0)
