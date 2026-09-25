#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L3-07 专用：逐幕溢出巡检（check_render.py 只在第 1 幕取样，这里把 9 幕都扫一遍）

对每一幕：跳到该幕 → 在 3 个时间点（幕内早/中/末）检查 #visual 内所有元素是否越界，
并抓一张截图。判据与 tools/check_render.py 的 OVERFLOW_JS 完全一致。

用法： python3 _data/recon/check_scenes_l307.py [--shots /tmp/l307shots]
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import common as C  # noqa: E402
from common import R  # noqa: E402

OVERFLOW_JS = open(os.path.join(C.TOOLS, "check_render.py"), encoding="utf-8").read()
OVERFLOW_JS = OVERFLOW_JS.split('OVERFLOW_JS = """')[1].split('"""')[0]


def main(argv):
    shots = argv[argv.index("--shots") + 1] if "--shots" in argv else None
    if shots:
        os.makedirs(shots, exist_ok=True)
    lessons = [(i, d) for i, d in C.existing_lessons() if i == "L3-07"]
    assert lessons, "找不到 L3-07"
    lid, d = lessons[0]
    url = "file://" + os.path.abspath(os.path.join(d, "index.html"))

    from playwright.sync_api import sync_playwright
    chrome = next((p for p in ("/usr/bin/google-chrome", "/usr/bin/chromium-browser",
                               "/usr/bin/google-chrome-stable") if os.path.isfile(p)), None)
    fails = 0
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
            pg.goto(url, wait_until="load")
            pg.wait_for_timeout(500)
            total = pg.evaluate("() => RENDER.scenes.length")
            for i in range(total):
                pg.evaluate("(i) => RENDER.go(i, 0)", i)
                pg.wait_for_timeout(260)
                dur = pg.evaluate("() => RENDER.scenes[RENDER.idx].duration")
                marks = [int(dur * 0.15), int(dur * 0.5), int(dur * 0.92)]
                got = []
                for t in marks:
                    pg.evaluate("(t) => RENDER.seek(t)", t)
                    pg.wait_for_timeout(300)
                    ov = pg.evaluate(OVERFLOW_JS)
                    if ov:
                        got.append((t, ov))
                    if shots:
                        pg.screenshot(path=os.path.join(shots, f"{lid}_{w}x{h}_s{i+1}_{t}.png"))
                if got:
                    fails += 1
                    R.bad(f"{w}x{h} 第 {i+1} 幕（duration={dur}）溢出 {len(got)} 处")
                    for t, ov in got[:3]:
                        print(f"      @{t}ms  {ov[0][:110]}")
                else:
                    print(f"  {R.G}✓{R.X} {w}x{h} 第 {i+1} 幕 {dur}ms / 0 溢出")
            if errs:
                fails += 1
                R.bad(f"{w}x{h} JS 异常 {len(errs)}: {errs[0][:150]}")
            if cerrs:
                fails += 1
                R.bad(f"{w}x{h} console.error {len(cerrs)}: {cerrs[0][:150]}")
            pg.close()
        br.close()
    if fails:
        R.bad(f"逐幕巡检失败：{fails} 项")
        return 1
    R.ok("逐幕巡检通过：9 幕 x 2 分辨率 x 3 时间点，0 溢出 / 0 错误")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
