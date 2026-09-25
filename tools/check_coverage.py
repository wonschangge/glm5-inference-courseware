#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""覆盖度门禁 (B 组)

B1 全覆盖 : universe - covered == ∅
B2 无空课 : 每课 source.md 的 coverage 声明至少 1 项
B3 无幻影 : 声明里每一项都真实存在（先查 universe，再查文件系统）

用法:
    python3 tools/check_coverage.py [--quiet] [--lesson <dir>]
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
    only = None
    if "--lesson" in argv:
        only = argv[argv.index("--lesson") + 1]

    universe, excluded = C.load_universe()
    uni = set(universe)

    lessons = C.discover_lessons()
    if only:
        lessons = [(i, d) for i, d in lessons if os.path.abspath(d) == os.path.abspath(only)
                   or i == only]
        if not lessons:
            R.bad("找不到课件目录: " + only)
            return 2

    covered = set()
    empty, phantom, nodir, nocov = [], [], [], []
    per_lesson = {}

    for lid, d in lessons:
        sm = os.path.join(d, "source.md")
        if not os.path.isdir(d):
            nodir.append(lid)
            continue
        if not os.path.isfile(sm):
            nocov.append(lid)
            continue
        declared = C.parse_coverage(C.read_text(sm))
        if declared is None:
            nocov.append(lid)
            continue
        if not declared:
            empty.append(lid)
            continue
        per_lesson[lid] = declared
        for rel in declared:
            if rel in uni:
                covered.add(rel)
            elif os.path.isfile(C.upstream_path(rel)):
                pass                      # 真实存在但不是目标后缀 -> 计入被引用，不计入覆盖率
            else:
                phantom.append((lid, rel))

    missing = sorted(uni - covered)
    total = len(uni)

    if not quiet:
        print(f"覆盖域 (实测推理闭包) : {total} 个文件")
        print(f"闭包外（已声明排除）  : {len(excluded)} 个文件")
        print(f"课件目录              : {len(lessons)}")
        print(f"已有 source.md 的课    : {len(per_lesson)}")
        print(f"已被覆盖              : {len(covered)}  ({len(covered)/total*100:.1f}%)")
        print(f"未被覆盖              : {len(missing)}")
        if missing:
            for m in missing[:40]:
                print("    - " + m)
            if len(missing) > 40:
                print(f"    ... 其余 {len(missing)-40} 个")

    fail = False
    if nodir:
        R.warn(f"{len(nodir)} 课尚无目录（计划内、未开工）：{', '.join(nodir[:8])}"
               + (" ..." if len(nodir) > 8 else ""))
    if nocov:
        R.bad(f"{len(nocov)} 课缺少 source.md 的 coverage 声明：{', '.join(nocov[:8])}"
              + (" ..." if len(nocov) > 8 else ""))
        fail = True
    if empty:
        R.bad(f"{len(empty)} 课的 coverage 声明为空：" + ", ".join(empty[:8]))
        fail = True
    if phantom:
        R.bad(f"{len(phantom)} 处幻影引用（文件不存在）：")
        for lid, rel in phantom[:12]:
            print(f"    {lid}: {rel}")
        fail = True

    # B1 只在"全部课件都开工"后才强制；未开工的课计入 pending
    pending = set(nodir)
    if missing:
        R.warn(f"B1 未达成：还有 {len(missing)}/{total} 个文件未被任何课覆盖")
        fail = True

    if not fail:
        R.ok(f"B 组通过：{len(covered)}/{total} 全覆盖，无空课，无幻影引用")
        return 0
    if quiet:
        print(f"B 组：覆盖 {len(covered)}/{total}；"
              f"缺 source.md {len(nocov)} 课；空声明 {len(empty)} 课；"
              f"幻影 {len(phantom)} 处；未开工 {len(nodir)} 课")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
