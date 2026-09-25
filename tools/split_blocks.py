#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自动拆块工具

扫描所有 source.md，把"在源文件中位置不连续"的引用块**自动拆成多个块**：
每个连续段前面补一条 `<!-- src: path -->` 标注并独立成围栏块。

★ 重要：若某个块反复拆不动（拆完仍然报同一个块），
   说明问题不在连续性，而在内容本身（打错字 / 凭记忆改写）。此时应停下来核对源文件。

用法:
    python3 tools/split_blocks.py [--check] [--lesson <dir|id>]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from common import R

_idxcache = {}


def pos_for(rel):
    if rel not in _idxcache:
        p = C.upstream_path(rel)
        _idxcache[rel] = C.src_index(C.read_text(p)) if os.path.isfile(p) else None
    return _idxcache[rel]


def split_file(path, dry=True):
    """返回 (changed, n_before, n_after)"""
    text = C.read_text(path)
    lines = text.split("\n")
    blocks = C.parse_blocks(text)

    # 只处理"有 src 标注 + 代码语言 + 存在源文件"的块
    todo = []
    for b in blocks:
        if b["lang"] not in C.FIDELITY_LANGS:
            continue
        if not b["lines"] or not b["path"]:
            continue
        pos = pos_for(b["path"])
        if pos is None:
            continue
        ok, _, _ = C.contiguous_run(b["lines"], pos)
        if ok:
            continue
        runs = C.runs_of(b["lines"], pos)
        if len(runs) <= 1:
            continue                      # 拆不动：内容问题，交给保真门禁报错
        todo.append((b, runs))

    if not todo:
        return False, len(blocks), len(blocks)

    # 从后往前替换，避免行号漂移
    out = list(lines)
    n_after = len(blocks)
    for b, runs in sorted(todo, key=lambda x: -x[0]["start_line"]):
        # 块体在文件中的行号：start_line 是 1 基的"块体首行"
        body_start = b["start_line"] - 1          # 0 基
        body_end = body_start + len(b["lines"])   # 不含
        # 向上找围栏行（body_start-1）与 src 标注行
        fence_open = body_start - 1
        marker = None
        k = fence_open - 1
        while k >= 0 and k > fence_open - 4:
            if C.SRC_RE.search(out[k]):
                marker = k
                break
            k -= 1
        fence_close = body_end
        assert out[fence_open].startswith("```"), (path, fence_open, out[fence_open])
        assert out[fence_close].startswith("```"), (path, fence_close, out[fence_close])

        repl = []
        for (s, e) in runs:
            repl.append(f"<!-- src: {b['path']} -->")
            repl.append("```" + b["lang"])
            repl.extend(b["lines"][s:e])
            repl.append("```")
        lo = marker if marker is not None else fence_open
        out[lo:fence_close + 1] = repl
        n_after += len(runs) - 1

    if not dry:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(out))
    return True, len(blocks), n_after


def main(argv):
    dry = "--check" in argv
    only = argv[argv.index("--lesson") + 1] if "--lesson" in argv else None

    lessons = C.existing_lessons()
    if only:
        lessons = [(i, d) for i, d in lessons
                   if os.path.abspath(d) == os.path.abspath(only) or i == only]

    nchg = 0
    for lid, d in lessons:
        sm = os.path.join(d, "source.md")
        if not os.path.isfile(sm):
            continue
        changed, nb, na = split_file(sm, dry=dry)
        if changed:
            nchg += 1
            print(f"  {'[需拆分]' if dry else '[已拆分]'} {lid}: {nb} 块 -> {na} 块")

    if nchg == 0:
        R.ok("所有引用块已是连续的，无需拆分")
        return 0
    if dry:
        R.warn(f"{nchg} 个 source.md 需要拆分（去掉 --check 即执行）")
        return 1
    R.ok(f"已拆分 {nchg} 个 source.md")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
