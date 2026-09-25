#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""静态语法与结构检查（秒级，放在流程最前面）

- 四文件齐备
- lesson.js 语法可通过 `node --check`
- lesson.js 定义了非空 SCENES，且每幕有 title / duration / build
- index.html 引用的本地资源都存在
- README.md 声明的「前置课」编号严格小于本课

用法:
    python3 tools/lint_lessons.py [--quiet] [--lesson <dir|id>]
"""
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from common import R


def node_check(js_path):
    try:
        p = subprocess.run(["node", "--check", js_path], capture_output=True, text=True, timeout=60)
        return p.returncode == 0, (p.stderr or "").strip()
    except FileNotFoundError:
        return None, "node 不可用"


def _brace_match(html):
    """定位 `nav:` 后的 { 与配对 }（不能用非贪婪正则，会吞掉右花括号）。"""
    m = re.search(r"nav\s*:\s*\{", html)
    if not m:
        return None, None
    o = html.index("{", m.start())
    depth, i = 0, o
    while i < len(html):
        if html[i] == "{":
            depth += 1
        elif html[i] == "}":
            depth -= 1
            if depth == 0:
                return o, i
        i += 1
    return o, None


def lint_one(lid, d):
    errs, warns = [], []
    for f in C.LESSON_FILES:
        if not os.path.isfile(os.path.join(d, f)):
            errs.append(f"缺少 {f}")

    js = os.path.join(d, "lesson.js")
    if os.path.isfile(js):
        ok, err = node_check(js)
        if ok is False:
            errs.append("lesson.js 语法错误: " + err.split("\n")[-3:][0][:160]
                        if err else "lesson.js 语法错误")
        elif ok is None:
            warns.append("跳过语法检查（node 不可用）")
        src = C.read_text(js)
        if not re.search(r"const\s+SCENES\s*=\s*\[", src) and not re.search(r"SCENES\s*=\s*\[", src):
            errs.append("lesson.js 未定义 SCENES 数组")
        # 每幕必备字段：duration 与 build 只出现在幕定义里，可作为幕数基准
        n_scenes = len(re.findall(r"^\s*duration\s*:", src, re.M))
        n_build = len(re.findall(r"^\s*build\s*\(", src, re.M))
        n_kicker = len(re.findall(r"^\s*kicker\s*:", src, re.M))
        n_code = len(re.findall(r"^\s*code\s*:", src, re.M))
        if n_scenes == 0:
            errs.append("SCENES 里没有任何分幕（duration 一个都没找到）")
        if n_scenes != n_build:
            errs.append(f"duration({n_scenes}) 与 build({n_build}) 数量不一致")
        if n_kicker < n_scenes:
            errs.append(f"{n_scenes - n_kicker} 幕缺少 kicker")
        if n_code < n_scenes:
            errs.append(f"{n_scenes - n_code} 幕缺少 code")
        # 常见笔误：对象字面量里把引号放错位置
        for m in re.finditer(r"style\s*:\s*'[^']*'\s*:", src):
            errs.append("疑似引号笔误: " + m.group(0)[:60])
        # 时间轴必须递增（粗查：收集 tl.at 的第一个参数）
        ats = [int(x) for x in re.findall(r"tl\.at\(\s*(\d+)", src)]
        if ats != sorted(ats):
            warns.append("tl.at 的时间点未全局递增（可能是循环生成，请目视确认）")

    html = os.path.join(d, "index.html")
    if os.path.isfile(html):
        h = C.read_text(html)
        for m in re.finditer(r'(?:src|href)="([^"]+)"', h):
            u = m.group(1)
            if u.startswith(("data:", "http://", "https://", "#", "mailto:")):
                continue
            tgt = os.path.normpath(os.path.join(d, u))
            if not os.path.isfile(tgt):
                errs.append(f"index.html 引用了不存在的资源: {u}")

    # ★ index.html 的内联 boot 脚本必须单独查语法。
    #   `node --check lesson.js` 覆盖不到它 —— 一段内联脚本少了右花括号时，
    #   页面白屏但 lint 全绿，只有渲染门禁才会发现。这里补上这一课。
    if os.path.isfile(html):
        h = C.read_text(html)
        m = re.search(r"<script>\s*(SHELL\.boot.*?)\s*</script>", h, re.S)
        if not m:
            errs.append("index.html 找不到 SHELL.boot 内联脚本")
        else:
            import tempfile
            with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                             encoding="utf-8") as f:
                f.write(m.group(1))
                tmp = f.name
            try:
                ok2, err2 = node_check(tmp)
                if ok2 is False:
                    errs.append("index.html 内联脚本语法错误: "
                                + (err2.splitlines()[0] if err2 else ""))
            finally:
                os.unlink(tmp)
        # nav 花括号必须配对
        o, c = _brace_match(h)
        if o is None or c is None:
            errs.append("index.html 的 nav 对象花括号不配对")

    rd = os.path.join(d, "README.md")
    if os.path.isfile(rd):
        r = C.read_text(rd)
        m = re.search(r"前置课[^\n]*", r)
        if not m:
            warns.append("README 未声明前置课")
        else:
            nums = re.findall(r"`?(L\d-\d\d)`?", m.group(0))
            me = re.search(r"(L\d)-(\d\d)", lid)
            for n in nums:
                a = re.match(r"(L\d)-(\d\d)", n)
                if not a:
                    continue
                if (a.group(1), a.group(2)) >= (me.group(1), me.group(2)):
                    errs.append(f"前置课 {n} 不小于本课 {lid}")
    return errs, warns


def nav_closure(lessons):
    """导航链闭环：若 A.next = B，则必须 B.prev = A（反之亦然）。

    这是 skill 明确要求的不变量 —— 新增课插在中间时忘记改前后两课，
    交互自检的「下一课」就会跳错，但单课渲染门禁发现不了。
    """
    href = {}
    for lid, d in lessons:
        h = os.path.join(d, "index.html")
        if not os.path.isfile(h):
            continue
        t = C.read_text(h)
        m = re.search(r"nav\s*:\s*\{(.*?)\n\s*\}\s*\)", t, re.S)
        seg = m.group(1) if m else t
        def grab(key):
            mm = re.search(key + r"\s*:\s*(?:\{\s*href\s*:\s*'([^']+)'|(null))", seg)
            if not mm:
                return None
            return None if mm.group(2) else mm.group(1)
        href[lid] = (d, grab("prev"), grab("next"))

    errs = []
    for lid, (d, prev, nxt) in sorted(href.items()):
        for kind, target in (("prev", prev), ("next", nxt)):
            if not target:
                continue
            tgt_abs = os.path.normpath(os.path.join(d, target))
            if not os.path.isfile(tgt_abs):
                errs.append(f"{lid}: {kind} -> {target} 目标不存在")
                continue
            tgt_dir = os.path.dirname(tgt_abs)
            other = None
            for k, (od, _, _) in href.items():
                if os.path.abspath(od) == os.path.abspath(tgt_dir):
                    other = k
            if other is None:
                continue
            _, oprev, onext = href[other]
            want = "prev" if kind == "next" else "next"
            back = onext if kind == "prev" else oprev
            if back is None:
                errs.append(f"{lid}.{kind} -> {other}，但 {other}.{want} 是 null（链未闭环）")
            else:
                back_abs = os.path.normpath(os.path.join(href[other][0], back))
                if os.path.abspath(back_abs) != os.path.abspath(os.path.join(d, "index.html")):
                    errs.append(f"{lid}.{kind} -> {other}，但 {other}.{want} -> {back}（未指回本课）")
    return errs


def main(argv):
    quiet = "--quiet" in argv
    only = argv[argv.index("--lesson") + 1] if "--lesson" in argv else None
    lessons = C.existing_lessons()
    if only:
        lessons = [(i, d) for i, d in lessons
                   if os.path.abspath(d) == os.path.abspath(only) or i == only]
        if not lessons:
            R.bad("找不到课件: " + only)
            return 2

    if not lessons:
        R.warn("尚无任何课件目录 —— 空集上无错误可报")
        return 0

    # 导航链闭环（全量一起查，因为要跨课比对）
    nav_errs = nav_closure(lessons) if not only else []
    for e in nav_errs:
        R.bad("导航链: " + e)

    nbad = 0
    for lid, d in lessons:
        errs, warns = lint_one(lid, d)
        if errs:
            nbad += 1
            R.bad(f"{lid}")
            for e in errs:
                print("      " + e)
        elif not quiet and warns:
            print(f"  {R.Y}!{R.X} {lid}: " + "; ".join(warns))
    if nbad or nav_errs:
        R.bad(f"A4 语法/结构检查失败：{nbad}/{len(lessons)} 课有问题"
              + (f"，导航链 {len(nav_errs)} 处未闭环" if nav_errs else ""))
        return 1
    R.ok(f"A4 通过：{len(lessons)} 课语法与结构检查无错误，导航链闭环")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
