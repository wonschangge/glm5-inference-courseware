#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""门户页专项检查 —— 门户没有课件数据结构，不能用 check_render.py 跑。

断言：
  - 卡片数与计划课数一致
  - 每张「已交付」卡片的链接都真实存在
  - 0 pageerror / 0 console.error
  - 层过滤可用、搜索可用
  - 无横向溢出

用法:
    python3 tools/check_portal.py [--quiet] [--shots <dir>]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from common import R

sys.path.insert(0, C.TOOLS)
import plan  # noqa: E402


def main(argv):
    quiet = "--quiet" in argv
    shots = argv[argv.index("--shots") + 1] if "--shots" in argv else None
    idx = os.path.join(C.ROOT, "index.html")
    if not os.path.isfile(idx):
        R.bad("门户页不存在：请先跑 python3 tools/generate_portal.py")
        return 1

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        R.bad("playwright 不可用")
        return 2

    chrome = next((p for p in ("/usr/bin/google-chrome", "/usr/bin/chromium-browser",
                               "/usr/bin/google-chrome-stable") if os.path.isfile(p)), None)
    problems = []
    with sync_playwright() as p:
        br = p.chromium.launch(executable_path=chrome,
                               args=["--allow-file-access-from-files", "--no-sandbox"])
        pg = br.new_page(viewport={"width": 1440, "height": 900})
        errs, cerrs = [], []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: cerrs.append(m.text) if m.type == "error" else None)
        pg.goto("file://" + os.path.abspath(idx), wait_until="load")
        pg.wait_for_timeout(900)

        n_cards = pg.evaluate("() => document.querySelectorAll('.lcard').length")
        n_expect = len(plan.LESSONS)
        if n_cards != n_expect:
            problems.append(f"卡片数 {n_cards} != 计划课数 {n_expect}")

        # 链接有效性
        bad_links = pg.evaluate("""() => {
          const out = [];
          document.querySelectorAll('a[href]').forEach(a => {
            const h = a.getAttribute('href');
            if (/^(https?:|mailto:|#|data:)/.test(h)) return;
            out.push(h);
          });
          return out;
        }""")
        for h in bad_links:
            tgt = os.path.normpath(os.path.join(C.ROOT, h.split("#")[0]))
            if not os.path.isfile(tgt):
                problems.append(f"链接指向不存在的文件: {h}")

        # 过滤交互
        before = pg.evaluate("() => document.querySelectorAll('.lcard:not([hidden])').length")
        pg.click('.lf[data-l="L4"]')
        pg.wait_for_timeout(320)
        after = pg.evaluate("() => document.querySelectorAll('.lcard:not([hidden])').length")
        if not (0 < after < before):
            problems.append(f"层过滤无效：{before} -> {after}")
        pg.click('.lf[data-l="*"]')
        pg.wait_for_timeout(260)
        back = pg.evaluate("() => document.querySelectorAll('.lcard:not([hidden])').length")
        if back != before:
            problems.append(f"重置过滤无效：{before} -> {back}")

        # 搜索交互
        pg.fill("#q", "attention")
        pg.wait_for_timeout(320)
        nsearch = pg.evaluate("() => document.querySelectorAll('.lcard:not([hidden])').length")
        if nsearch == 0 or nsearch >= before:
            problems.append(f"搜索无效：关键词 attention 命中 {nsearch}")
        pg.fill("#q", "zzz_no_such_lesson")
        pg.wait_for_timeout(320)
        if not pg.evaluate("() => !document.getElementById('empty').hidden"):
            problems.append("空结果提示未显示")
        pg.fill("#q", "")
        pg.wait_for_timeout(300)

        # 横向溢出
        ov = pg.evaluate("""() => {
          const de = document.documentElement;
          return { sw: de.scrollWidth, cw: de.clientWidth };
        }""")
        if ov["sw"] > ov["cw"] + 2:
            problems.append(f"横向溢出：scrollWidth {ov['sw']} > clientWidth {ov['cw']}")

        if shots:
            os.makedirs(shots, exist_ok=True)
            pg.screenshot(path=os.path.join(shots, "portal_top.png"))
            pg.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
            pg.wait_for_timeout(500)
            pg.screenshot(path=os.path.join(shots, "portal_bottom.png"))

        if errs:
            problems.append(f"pageerror {len(errs)}: " + errs[0][:140])
        if cerrs:
            problems.append(f"console.error {len(cerrs)}: " + cerrs[0][:140])
        br.close()

    if problems:
        R.bad(f"门户检查失败（{len(problems)} 项）")
        for x in problems[:10]:
            print("    " + x)
        return 1
    R.ok(f"门户通过：{n_cards} 张卡片 / 链接全部有效 / 过滤与搜索可用 / 0 JS 错误 / 无横向溢出")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
