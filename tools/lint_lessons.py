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
    if nbad:
        R.bad(f"A4 语法/结构检查失败：{nbad}/{len(lessons)} 课有问题")
        return 1
    R.ok(f"A4 通过：{len(lessons)} 课语法与结构检查无错误")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
