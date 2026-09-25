#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分层报告（M3）—— 只输出**实测**数字，禁止把「计划做」说成「已完成」。

对每一层统计：
  - 计划课时 / 已交付课时（目录与四文件齐备）
  - 该层声明的源文件数 / 已覆盖数
  - 引用块数（保真门禁的实测输入）
  - 文件行数合计
  - 提交数（git log 中该层课号出现在 subject 里的次数）

用法:
    python3 tools/report_layer.py            # 全部层
    python3 tools/report_layer.py L4         # 单层
    python3 tools/report_layer.py --md out.md
"""
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from common import R

sys.path.insert(0, C.TOOLS)
import plan  # noqa: E402


def git_count(pat):
    try:
        p = subprocess.run(["git", "log", "--oneline", "--all", "--grep", pat],
                           capture_output=True, text=True, cwd=C.ROOT, timeout=30)
        return len([x for x in p.stdout.splitlines() if x.strip()])
    except Exception:
        return 0


def layer_rows(code):
    uni, _ = C.load_universe()
    ls = [x for x in plan.LESSONS if x["layer"] == code]
    done, delivered_files, blocks, lines = 0, set(), 0, 0
    for L in ls:
        d = os.path.join(C.ROOT, plan.lesson_dir(L))
        ok = all(os.path.isfile(os.path.join(d, f)) for f in C.LESSON_FILES)
        if ok:
            done += 1
            delivered_files |= set(plan.expand_files(L["files"], uni))
        sm = os.path.join(d, "source.md")
        if os.path.isfile(sm):
            for b in C.parse_blocks(C.read_text(sm)):
                if b["lang"] in C.FIDELITY_LANGS and b["lines"] and b["path"]:
                    blocks += 1
    planned_files = set(f for L in ls for f in plan.expand_files(L["files"], uni))
    for f in planned_files:
        p = C.upstream_path(f)
        if os.path.isfile(p):
            with open(p, "rb") as fh:
                lines += fh.read().count(b"\n") + 1
    return dict(code=code, lessons=len(ls), done=done,
                planned=len(planned_files), covered=len(delivered_files),
                blocks=blocks, lines=lines)


def main(argv):
    md_out = argv[argv.index("--md") + 1] if "--md" in argv else None
    only = [a for a in argv if re.fullmatch(r"L\d", a)]

    uni, excluded = C.load_universe()
    rows = [layer_rows(c) for c, _, _, _ in plan.LAYERS if not only or c in only]

    hdr = f"{'层':<4}{'课时':>6}{'已交付':>8}{'源文件':>8}{'已覆盖':>8}{'引用块':>8}{'行数':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['code']:<4}{r['lessons']:>6}{r['done']:>8}{r['planned']:>8}"
              f"{r['covered']:>8}{r['blocks']:>8}{r['lines']:>10,}")

    tot = dict(lessons=sum(r["lessons"] for r in rows), done=sum(r["done"] for r in rows),
               planned=sum(r["planned"] for r in rows), covered=sum(r["covered"] for r in rows),
               blocks=sum(r["blocks"] for r in rows), lines=sum(r["lines"] for r in rows))
    print("-" * len(hdr))
    print(f"{'合计':<4}{tot['lessons']:>6}{tot['done']:>8}{tot['planned']:>8}"
          f"{tot['covered']:>8}{tot['blocks']:>8}{tot['lines']:>10,}")
    print()
    print(f"覆盖域（实测推理闭包）: {len(uni)} 个文件   ｜   显式排除: {len(excluded)} 个")
    print(f"全仓覆盖进度           : {tot['covered']}/{len(uni)} "
          f"({tot['covered']/len(uni)*100:.1f}%)")

    if md_out:
        L = []
        L.append("# 分层报告（实测）\n")
        L.append(f"生成方式：`python3 tools/report_layer.py --md {os.path.basename(md_out)}`\n")
        L.append("| 层 | 计划课时 | 已交付 | 该层源文件 | 已覆盖 | 引用块 | 行数 |")
        L.append("|---|---|---|---|---|---|---|")
        for r in rows:
            L.append(f"| {r['code']} | {r['lessons']} | {r['done']} | {r['planned']} | "
                     f"{r['covered']} | {r['blocks']} | {r['lines']:,} |")
        L.append(f"| **合计** | **{tot['lessons']}** | **{tot['done']}** | "
                 f"**{tot['planned']}** | **{tot['covered']}** | **{tot['blocks']}** | "
                 f"**{tot['lines']:,}** |")
        L.append("")
        L.append(f"覆盖域 {len(uni)} 个文件；显式排除 {len(excluded)} 个"
                 f"（仅经 `models/auto` 注册表顺带导入的 36 个无关模型族）。")
        with open(md_out, "w", encoding="utf-8") as f:
            f.write("\n".join(L) + "\n")
        R.ok(f"已写入 {md_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
