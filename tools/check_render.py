#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""渲染门禁 (C 组)

断言：每课在无头浏览器中
  - pageerror 与 console.error 均为 0
  - 可视区内元素**四边**都不越界（早/中/末三个时间点）
  - 交互可用：播放/暂停/上一幕/下一幕/方向键/圆点跳转

用法:
    python3 tools/check_render.py [--all] [--lesson <dir|id>] [--res 1280x720,1920x1080]
                                  [--shots <dir>] [--quiet]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from common import R

OVERFLOW_JS = """() => {
  const out = [], vw = innerWidth, vh = innerHeight;
  const root = document.querySelector('#visual');
  if (!root) return ['#visual MISSING'];
  root.querySelectorAll('*').forEach(el => {
    const b = el.getBoundingClientRect();
    if (b.width <= 0 || b.height <= 0) return;
    if (b.left < -1 || b.top < -1 || b.right > vw + 1 || b.bottom > vh + 1) {
      out.push((el.className && String(el.className).split(' ')[0] || el.tagName) +
               ' [' + Math.round(b.left) + ',' + Math.round(b.top) + ',' +
               Math.round(b.right) + ',' + Math.round(b.bottom) + ']');
    }
  });
  return out.slice(0, 12);
}"""

STAGE_JS = """() => {
  const v = document.querySelector('#visual');
  const s = document.querySelector('#stage');
  if (!v || !s) return null;
  const r = v.getBoundingClientRect(), sr = s.getBoundingClientRect();
  const scale = sr.width / 1600;
  return { scale: +scale.toFixed(4), vw: +(r.width/scale).toFixed(1), vh: +(r.height/scale).toFixed(1) };
}"""


def check_page(page, url, shots_dir, tag):
    errs, cerrs = [], []
    page.on("pageerror", lambda e: errs.append(str(e)))
    page.on("console", lambda m: cerrs.append(m.text) if m.type == "error" else None)
    page.goto(url, wait_until="load")
    page.wait_for_timeout(600)

    overflow = []
    for t in (1200, 4200, 8000):
        page.wait_for_timeout(t)
        ov = page.evaluate(OVERFLOW_JS)
        if ov:
            overflow.append((t, ov))
        if shots_dir:
            page.screenshot(path=os.path.join(shots_dir, f"{tag}_{t}.png"))

    # 交互自检
    inter = []
    def ok(name, cond):
        inter.append((name, bool(cond)))
    try:
        n0 = page.evaluate("() => RENDER.idx")
        page.keyboard.press("ArrowRight"); page.wait_for_timeout(420)
        ok("ArrowRight 换幕", page.evaluate("() => RENDER.idx") != n0)
        page.keyboard.press("ArrowLeft"); page.wait_for_timeout(420)
        ok("ArrowLeft 换幕", page.evaluate("() => RENDER.idx") == n0)
        p0 = page.evaluate("() => RENDER.playing")
        page.keyboard.press(" "); page.wait_for_timeout(220)
        ok("空格暂停", page.evaluate("() => RENDER.playing") != p0)
        page.keyboard.press(" "); page.wait_for_timeout(220)
        ok("空格恢复", page.evaluate("() => RENDER.playing") == p0)
        page.keyboard.press("Home"); page.wait_for_timeout(460)
        ok("Home 回第一幕", page.evaluate("() => RENDER.idx") == 0)
        page.keyboard.press("End"); page.wait_for_timeout(520)
        tot = page.evaluate("() => RENDER.scenes.length")
        ok("End 到最后一幕", page.evaluate("() => RENDER.idx") == tot - 1)
        tk = page.query_selector_all("#ticks .tk")
        ok("刻度数==幕数", len(tk) == tot)
        if len(tk) > 2:
            tk[1].click(); page.wait_for_timeout(460)
            ok("刻度跳转", page.evaluate("() => RENDER.idx") == 1)
        nxt = page.query_selector("#btns .bt:nth-child(4)")
        if nxt:
            cur = page.evaluate("() => RENDER.idx")
            nxt.click(); page.wait_for_timeout(460)
            ok("下一幕按钮", page.evaluate("() => RENDER.idx") != cur)
        ok("场景已挂载", page.evaluate("() => !!document.querySelector('#visual .scene')"))
    except Exception as e:
        inter.append(("交互自检异常: " + str(e)[:70], False))

    return errs, cerrs, overflow, inter


def main(argv):
    only = argv[argv.index("--lesson") + 1] if "--lesson" in argv else None
    allf = "--all" in argv
    quiet = "--quiet" in argv
    shots = argv[argv.index("--shots") + 1] if "--shots" in argv else None
    res = argv[argv.index("--res") + 1] if "--res" in argv else "1280x720,1920x1080"
    resolutions = [tuple(int(x) for x in r.split("x")) for r in res.split(",")]

    lessons = C.existing_lessons()
    if only:
        lessons = [(i, d) for i, d in lessons
                   if os.path.abspath(d) == os.path.abspath(only) or i == only]
        if not lessons:
            R.bad("找不到课件: " + only)
            return 2
    if not allf and not only:
        R.warn("未指定 --all 或 --lesson：默认只检查最后一个课件")
        lessons = lessons[-1:]

    if not lessons:
        R.warn("尚无任何课件目录 —— 空集上无错误可报")
        return 0

    if shots:
        os.makedirs(shots, exist_ok=True)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        R.bad("playwright 不可用，跳过渲染门禁")
        return 2

    chrome = next((p for p in ("/usr/bin/google-chrome", "/usr/bin/chromium-browser",
                               "/usr/bin/google-chrome-stable") if os.path.isfile(p)), None)
    fails = 0
    total_scenes = 0
    with sync_playwright() as p:
        br = p.chromium.launch(executable_path=chrome,
                               args=["--allow-file-access-from-files", "--no-sandbox",
                                     "--disable-dev-shm-usage", "--disable-gpu",
                                     "--force-color-profile=srgb"])
        for lid, d in lessons:
            html = os.path.join(d, "index.html")
            if not os.path.isfile(html):
                R.bad(f"{lid}: 缺少 index.html")
                fails += 1
                continue
            url = "file://" + os.path.abspath(html)
            problems = []
            for (w, h) in resolutions:
                # 浏览器在并发压力下可能瞬时崩溃（Target crashed）。
                # 这是环境问题而非课件问题，重试几次再判定，避免假失败。
                last_err = None
                for attempt in range(3):
                    pg = br.new_page(viewport={"width": w, "height": h},
                                     device_scale_factor=1)
                    try:
                        errs, cerrs, overflow, inter = check_page(
                            pg, url, shots, f"{lid}_{w}x{h}")
                        geo = pg.evaluate(STAGE_JS)
                        nsc = pg.evaluate("() => (RENDER.scenes||[]).length")
                        total_scenes = max(total_scenes, nsc)
                        pg.close()
                        last_err = None
                        break
                    except Exception as e:
                        last_err = e
                        try:
                            pg.close()
                        except Exception:
                            pass
                        if attempt < 2:
                            time.sleep(3 + attempt * 4)
                if last_err is not None:
                    problems.append(f"{w}x{h} 浏览器连续 3 次崩溃（环境问题，非课件问题）: "
                                    f"{str(last_err)[:90]}")
                    continue
                if errs:
                    problems.append(f"{w}x{h} JS 异常 {len(errs)}: " + errs[0][:120])
                if cerrs:
                    problems.append(f"{w}x{h} console.error {len(cerrs)}: " + cerrs[0][:120])
                for t, ov in overflow:
                    problems.append(f"{w}x{h} @{t}ms 溢出 {len(ov)}: " + ov[0][:90])
                bad_inter = [n for n, c in inter if not c]
                if bad_inter:
                    problems.append(f"{w}x{h} 交互失败: " + ", ".join(bad_inter))
                if geo and (geo["vw"] < 900 or geo["vh"] < 560):
                    problems.append(f"{w}x{h} 可视区过小: {geo}")
            if problems:
                fails += 1
                R.bad(f"{lid} ({len(problems)} 项)")
                for x in problems[:8]:
                    print("      " + x)
            elif not quiet:
                chk = len(resolutions) * 3
                print(f"  {R.G}✓{R.X} {lid}  {nsc} 幕 / 0 错误 / 0 溢出 / 交互通过 "
                      f"@ {'+'.join(f'{w}x{h}' for w, h in resolutions)}")

        br.close()

    if fails:
        R.bad(f"C 组失败：{fails}/{len(lessons)} 课未通过")
        return 1
    R.ok(f"C 组通过：{len(lessons)} 课 × {len(resolutions)} 分辨率，0 JS 错误 / 0 溢出 / 交互可用")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
