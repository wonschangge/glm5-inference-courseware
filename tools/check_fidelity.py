#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""保真门禁 (A1/A3)

断言：source.md 里每一段代码引用都**逐字**来自标注的源文件，且**位置连续**。

约定：引用块前必须有 `<!-- src: relative/path.py -->` 标注。
      lang=text 的块视为示意代码，不参与校验（避免把教学伪代码当真实引用）。

用法:
    python3 tools/check_fidelity.py [--quiet] [--lesson <dir|id>] [--json <out>]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from common import R

_cache = {}


def idx_for(rel):
    if rel not in _cache:
        p = C.upstream_path(rel)
        _cache[rel] = C.src_index(C.read_text(p)) if os.path.isfile(p) else None
    return _cache[rel]


def check_lesson(lid, d):
    """返回 (n_blocks, failures, unmarked)"""
    sm = os.path.join(d, "source.md")
    if not os.path.isfile(sm):
        return 0, [], []
    blocks = C.parse_blocks(C.read_text(sm))
    n, fails, unmarked = 0, [], []
    for b in blocks:
        if b["lang"] not in C.FIDELITY_LANGS:
            continue
        if not b["lines"] or all(not l.strip() for l in b["lines"]):
            continue
        if not b["path"]:
            unmarked.append(b["start_line"])
            continue
        n += 1
        pos = idx_for(b["path"])
        if pos is None:
            fails.append(dict(kind="missing-file", path=b["path"], line=b["start_line"]))
            continue
        ok, k, got = C.contiguous_run(b["lines"], pos)
        if not ok:
            fails.append(dict(kind="discontiguous", path=b["path"], line=b["start_line"],
                              bad_idx=k, bad_line=b["lines"][k], matched=len(got),
                              total=len(b["lines"])))
    return n, fails, unmarked


def main(argv):
    quiet = "--quiet" in argv
    only = argv[argv.index("--lesson") + 1] if "--lesson" in argv else None
    jsonout = argv[argv.index("--json") + 1] if "--json" in argv else None

    lessons = C.existing_lessons()
    if only:
        lessons = [(i, d) for i, d in lessons
                   if os.path.abspath(d) == os.path.abspath(only) or i == only]
        if not lessons:
            R.bad("找不到课件: " + only)
            return 2

    tot_blocks, all_fails, all_unmarked, nlessons = 0, [], [], 0
    for lid, d in lessons:
        n, fails, unmarked = check_lesson(lid, d)
        if n == 0 and not fails and not unmarked:
            continue
        nlessons += 1
        tot_blocks += n
        all_fails += [(lid, f) for f in fails]
        all_unmarked += [(lid, ln) for ln in unmarked]

    if not quiet:
        print(f"课件数        : {nlessons}")
        print(f"引用块总数    : {tot_blocks}")
        print(f"未标注 src 的块: {len(all_unmarked)}")

    if all_unmarked:
        R.bad(f"{len(all_unmarked)} 个代码块缺少 <!-- src: ... --> 标注：")
        for lid, ln in all_unmarked[:12]:
            print(f"    {lid} source.md 第 {ln} 行")
        return 1

    if all_fails:
        R.bad(f"{len(all_fails)} 个引用块未通过逐字/连续校验：")
        for lid, f in all_fails[:20]:
            if f["kind"] == "missing-file":
                print(f"    {lid}: 源文件不存在 {f['path']} (第 {f['line']} 行)")
            else:
                print(f"    {lid}: {f['path']} (第 {f['line']} 行起) "
                      f"第 {f['bad_idx']+1}/{f['total']} 行断裂 -> {f['bad_line'][:70]!r}")
        print(f"\n  {R.D}提示：先跑 python3 tools/split_blocks.py 自动拆分不连续块；{R.X}")
        print(f"  {R.D}若同一块反复失败，问题多半是打错字或凭记忆改写，请核对源文件。{R.X}")
        if jsonout:
            json.dump([{"lesson": l, **f} for l, f in all_fails], open(jsonout, "w"),
                      ensure_ascii=False, indent=1)
        return 1

    R.ok(f"A1/A3 通过：{tot_blocks} 个引用块全部逐字命中且连续（{nlessons} 课）")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
