#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修正所有课件的 nav 链接（幂等）。

背景（踩过的坑）：
  作业书生成器早期把跨层邻居的路径一律写成 `../<课号>/index.html`，
  但层边界两侧的课不在同一个目录里 —— 正确写法是
  `../../<层目录>/<课号>/index.html`。于是 L0-05.next、L1-01.prev 等
  层边界链接全部指向不存在的路径。lint_lessons.py 的闭环检查把它抓了出来。

本脚本按 tools/plan.py 重新计算每课的 prev/next 与正确的相对路径并回写。

用法:
    python3 tools/fix_nav.py [--check]
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from common import R

sys.path.insert(0, C.TOOLS)
import plan  # noqa: E402


def expected():
    out = {}
    for i, L in enumerate(plan.LESSONS):
        same = [x["id"] for x in plan.LESSONS if x["layer"] == L["layer"]]
        k = same.index(L["id"])
        prev_id = same[k - 1] if k > 0 else (plan.LESSONS[i - 1]["id"] if i > 0 else None)
        nxt_id = (same[k + 1] if k + 1 < len(same)
                  else (plan.LESSONS[i + 1]["id"] if i + 1 < len(plan.LESSONS) else None))

        def href(other):
            if other is None:
                return None
            ol = other.split("-")[0]
            if ol == L["layer"]:
                return f"../{other}/index.html"
            return f"../../{plan.LAYER_DIR[ol]}/{other}/index.html"

        out[L["id"]] = dict(prev=prev_id, next=nxt_id,
                            prev_href=href(prev_id), next_href=href(nxt_id))
    return out


NAV_RE = re.compile(r"(nav\s*:\s*\{)(.*?)(\n\s*\})", re.S)


def rewrite(html, e):
    m = NAV_RE.search(html)
    if not m:
        return html, False
    prev = ("null" if e["prev"] is None
            else "{ href: '" + e["prev_href"] + "', label: '" + e["prev"] + "' }")
    nxt = ("null" if e["next"] is None
           else "{ href: '" + e["next_href"] + "', label: '" + e["next"] + "' }")
    new_inner = ("\n    prev: " + prev + ",\n    next: " + nxt + ",")
    out = html[:m.start(2)] + new_inner + html[m.end(2):]
    return out, out != html


def main(argv):
    check = "--check" in argv
    exp = expected()
    changed, bad = [], []
    for lid, d in C.discover_lessons():
        p = os.path.join(d, "index.html")
        if not os.path.isfile(p):
            continue
        h = C.read_text(p)
        new, ch = rewrite(h, exp[lid])
        if ch:
            changed.append(lid)
            if not check:
                with open(p, "w", encoding="utf-8") as f:
                    f.write(new)
    # 回读校验：每个 nav 目标都必须真实存在
    for lid, d in C.discover_lessons():
        p = os.path.join(d, "index.html")
        if not os.path.isfile(p):
            continue
        t = C.read_text(p)
        m = NAV_RE.search(t)
        if not m:
            bad.append((lid, "无 nav 块"))
            continue
        for key in ("prev", "next"):
            mm = re.search(key + r":\s*\{[^}]*href:\s*'([^']+)'", m.group(2))
            if not mm:
                continue
            tgt = os.path.normpath(os.path.join(d, mm.group(1)))
            if not os.path.isfile(tgt):
                bad.append((lid, f"{key} -> {mm.group(1)} 不存在"))
    if changed:
        R.warn(f"{len(changed)} 课需要修正 nav：{', '.join(changed[:10])}"
               + (" ..." if len(changed) > 10 else ""))
    if bad:
        R.bad(f"{len(bad)} 处 nav 目标仍不存在：")
        for lid, x in bad[:10]:
            print(f"    {lid}: {x}")
        return 1
    if changed and check:
        return 1
    R.ok(f"nav 全部指向存在的课件"
         + (f"（已修正 {len(changed)} 课）" if changed and not check else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
