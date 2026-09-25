#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""安全地提交**一课**：先验门禁，再只暂存该课目录，最后按规范生成提交信息。

存在的理由（踩过的坑）：
  - `git add -A` 会把子代理**尚未收尾**的中间快照一起提交，
    也会把别的课扫进来。所以这里只 `git add <该课目录>`。
  - 提交信息里的数字必须是实测的，因此先跑门禁、把结果读出来再写进信息。

用法:
    python3 tools/commit_lesson.py L2-01 --body "补充说明…" [--no-render] [--dry-run]
    python3 tools/commit_lesson.py --status          # 列出可提交的课（四文件齐备且门禁待跑）
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

LAYER_DIR = plan.LAYER_DIR


def sh(args, timeout=1800):
    p = subprocess.run(args, capture_output=True, text=True, cwd=C.ROOT, timeout=timeout)
    return p.returncode, ((p.stdout or "") + (p.stderr or ""))


def strip_ansi(s):
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def spec(lid):
    L = next((x for x in plan.LESSONS if x["id"] == lid), None)
    if L is None:
        raise SystemExit("未知课号: " + lid)
    return L


def gate(cmd, lid, timeout=1800):
    rc, out = sh(cmd + ["--lesson", lid], timeout=timeout)
    return rc == 0, strip_ansi(out).strip()


def main(argv):
    if "--status" in argv:
        for L in plan.LESSONS:
            d = os.path.join(C.ROOT, plan.lesson_dir(L))
            have = [f for f in C.LESSON_FILES if os.path.isfile(os.path.join(d, f))]
            rc, out = sh(["git", "status", "--porcelain", "--", plan.lesson_dir(L)])
            mark = "完整" if len(have) == 4 else f"{len(have)}/4"
            dirty = "有改动" if out.strip() else "-"
            print(f"  {L['id']:<7} {mark:<6} {dirty}")
        return 0

    args = [a for a in argv if not a.startswith("--")]
    if not args:
        R.bad("用法: commit_lesson.py <课号> [--body ...] [--no-render] [--dry-run]")
        return 2
    lid = args[0]
    L = spec(lid)
    d = plan.lesson_dir(L)
    do_render = "--no-render" not in argv
    dry = "--dry-run" in argv
    body = argv[argv.index("--body") + 1] if "--body" in argv else ""
    if "--from-plan" in argv and not body:
        # 用 tools/plan.py 里的「讲解要点」自动组稿，保证提交信息与计划一致，
        # 也避免批量提交时写出空泛的信息。
        pts = L["points"]
        body = "\n".join(("- " + p) for p in pts)
        body += ("\n\n覆盖的源文件：" + ", ".join(plan.expand_files(L["files"], C.load_universe()[0])[:3])
                 + (" 等" if len(L["files"]) > 3 else ""))

    # ---- 四文件齐备 ----
    miss = [f for f in C.LESSON_FILES if not os.path.isfile(os.path.join(C.ROOT, d, f))]
    if miss:
        R.bad(f"{lid} 缺少文件: {', '.join(miss)} —— 子代理可能还没收尾，拒绝提交")
        return 1

    # ---- 门禁 ----
    ok, out = gate(["python3", "tools/lint_lessons.py", "--quiet"], lid, 600)
    if not ok:
        R.bad(f"{lid} lint 未通过:\n{out}")
        return 1
    R.ok(f"{lid} lint 通过")

    ok, out = gate(["python3", "tools/check_fidelity.py", "--quiet"], lid, 900)
    if not ok:
        R.bad(f"{lid} 保真未通过:\n{out}")
        return 1
    m = re.search(r"(\d+) 个引用块", out)
    nblocks = m.group(1) if m else "?"
    R.ok(f"{lid} 保真通过（{nblocks} 个引用块）")

    ok, out = gate(["python3", "tools/check_params.py", "--quiet"], lid, 900)
    if not ok:
        R.bad(f"{lid} 参数门禁未通过:\n{out}")
        return 1
    R.ok(f"{lid} 参数门禁通过")

    nscenes = "?"
    if do_render:
        ok, out = gate(["python3", "tools/check_render.py"], lid, 3600)
        if not ok:
            R.bad(f"{lid} 渲染未通过:\n{out}")
            return 1
        m = re.search(r"(\d+) 幕 / 0 错误 / 0 溢出", out)
        nscenes = m.group(1) if m else "?"
        R.ok(f"{lid} 渲染通过（{nscenes} 幕 / 0 错误 / 0 溢出）")
    else:
        R.warn("--no-render：跳过渲染门禁（提交信息里请勿声称已渲染）")

    idx = plan.LESSONS.index(L)
    prev_id = plan.LESSONS[idx - 1]["id"] if idx > 0 else None

    # ---- 只暂存该课目录 ----
    rc, out = sh(["git", "status", "--porcelain", "--", d])
    if not out.strip():
        R.warn(f"{lid} 没有任何未提交改动（可能已提交）")
        return 0
    if dry:
        R.info(f"[dry-run] 将提交 {d}\n{out}")
        return 0
    rc, out = sh(["git", "add", "--", d])
    if rc != 0:
        R.bad("git add 失败: " + out)
        return 1

    title = re.sub(r"<[^>]+>", "", L["title"])
    files = plan.expand_files(L["files"], C.load_universe()[0])
    if len(files) <= 4:
        cov = ", ".join(files)
    else:
        cov = f"{len(files)} 个文件（见 TODOLIST.md 的 {lid} 条目）"

    msg = [f"feat({lid}): {title}", ""]
    if body:
        msg += [body, ""]
    msg.append(f"覆盖: {cov}")
    acc = [f"验收: fidelity=pass（{nblocks} 个引用块逐字命中）  flags=pass  lint=pass"]
    if do_render:
        acc.append(f"      render={nscenes}幕/0错误/0溢出 @1280x720 + 1920x1080")
    msg += acc
    msg.append(f"前置: {prev_id or 'none'}")
    text = "\n".join(msg) + "\n"

    rc, out = sh(["git", "commit", "-q", "-F", "-"], timeout=60)
    # 用 -F - 需要 stdin；改用临时文件
    tmp = os.path.join(C.DATA, ".commitmsg")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    rc, out = sh(["git", "commit", "-q", "-F", tmp])
    os.remove(tmp)
    if rc != 0:
        R.bad("git commit 失败: " + out)
        return 1
    rc, sha = sh(["git", "rev-parse", "--short", "HEAD"])
    R.ok(f"已提交 {lid} -> {sha.strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
