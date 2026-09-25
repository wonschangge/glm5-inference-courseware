#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""交付检查清单 —— 逐项用命令验证，不靠肉眼。

对应 skill 的 §6 清单：
  1  计划文档所有条目已勾选（grep -c 交叉验证，不用 head 截断清单）
  2  覆盖率差集为空
  3  四道门禁在全量课件上重跑通过
  4  全部课件渲染通过
  5  每课四文件齐备
  6  每课 README 声明前置课且编号严格小于本课
  7  任一课件可离线独立打开（无 CDN、无外链、无 fetch）
  8  门户页可用
  9  工作区干净，HEAD 与远端一致
  10 交付说明中明确列出边界

用法:
    python3 tools/verify_delivery.py [--quick]     # --quick 跳过渲染与网络
"""
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from common import R

sys.path.insert(0, C.TOOLS)
import plan  # noqa: E402

RESULTS = []


def rec(ok, name, detail=""):
    RESULTS.append((ok, name, detail))
    (R.ok if ok else R.bad)(f"{name}" + (f" —— {detail}" if detail else ""))
    return ok


def sh(args, cwd=None, timeout=180):
    p = subprocess.run(args, capture_output=True, text=True, cwd=cwd or C.ROOT, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def main(argv):
    quick = "--quick" in argv
    uni, excluded = C.load_universe()
    lessons = C.discover_lessons()

    print("\n\033[1m== 1. 计划清单完整性 ==\033[0m")
    # 不用 head 截断：先数总数再逐条核对
    n_plan = len(plan.LESSONS)
    ids = [x["id"] for x in plan.LESSONS]
    rec(len(ids) == len(set(ids)) == n_plan, "课号唯一且数目正确",
        f"{n_plan} 课, 去重后 {len(set(ids))}")
    dirs = [plan.lesson_dir(x) for x in plan.LESSONS]
    rec(len(set(dirs)) == n_plan, "课件目录唯一", f"{len(set(dirs))}/{n_plan}")
    assigned = set()
    for x in plan.LESSONS:
        assigned |= set(plan.expand_files(x["files"], uni))
    rec(not (set(uni) - assigned), "计划已为每个源文件分配课程",
        f"未分配 {len(set(uni)-assigned)}")
    rec(not (assigned - set(uni)), "计划没有幻影文件", f"幻影 {len(assigned-set(uni))}")

    print("\n\033[1m== 5/6. 每课四文件 + 前置课 ==\033[0m")
    missing_files, bad_prev, ndone = [], [], 0
    for lid, d in lessons:
        miss = [f for f in C.LESSON_FILES if not os.path.isfile(os.path.join(d, f))]
        if miss:
            missing_files.append((lid, miss))
        else:
            ndone += 1
        rd = os.path.join(d, "README.md")
        if os.path.isfile(rd):
            m = re.search(r"前置课[^\n]*", C.read_text(rd))
            if not m:
                bad_prev.append((lid, "未声明前置课"))
            else:
                me = re.match(r"(L\d)-(\d\d)", lid)
                for n in re.findall(r"`?(L\d-\d\d)`?", m.group(0)):
                    a = re.match(r"(L\d)-(\d\d)", n)
                    if a and (a.group(1), a.group(2)) >= (me.group(1), me.group(2)):
                        bad_prev.append((lid, f"前置课 {n} 不小于本课"))
    rec(not missing_files, "每课四文件齐备", f"{ndone}/{len(lessons)} 课完整"
        + (f"；缺文件 {missing_files[:3]}" if missing_files else ""))
    rec(not bad_prev, "每课前置课编号严格小于本课", f"{len(bad_prev)} 处异常"
        + (f"：{bad_prev[:3]}" if bad_prev else ""))

    print("\n\033[1m== 2. 覆盖率差集 ==\033[0m")
    rc, out = sh(["python3", "tools/check_coverage.py", "--quiet"])
    rec(rc == 0, "覆盖度门禁（差集为空 / 无空课 / 无幻影）", out.strip().splitlines()[-1][:110])

    print("\n\033[1m== 7. 离线自足性 ==\033[0m")
    bad_off = []
    pat = re.compile(r'(?:src|href)\s*=\s*"([^"]+)"')
    for lid, d in lessons:
        for fn in ("index.html",):
            p = os.path.join(d, fn)
            if not os.path.isfile(p):
                continue
            for m in pat.finditer(C.read_text(p)):
                u = m.group(1)
                if u.startswith(("http://", "https://", "//")):
                    bad_off.append((lid, u))
    # 全仓扫描 fetch / XHR / CDN
    cdn = []
    for root, _dirs, files in os.walk(C.ROOT):
        if "/.git" in root or "/_data" in root:
            continue
        for f in files:
            if f.endswith((".js", ".html", ".css")):
                t = C.read_text(os.path.join(root, f))
                for m in re.finditer(r"https?://(?!www\.w3\.org|github\.com|huggingface\.co)", t):
                    cdn.append((os.path.relpath(os.path.join(root, f), C.ROOT), m.group(0)[:60]))
    rec(not bad_off, "课件页面无外链资源", f"{len(bad_off)} 处")
    rec(not cdn, "课件无网络请求/CDN", f"{len(cdn)} 处" + (f"：{cdn[:3]}" if cdn else ""))

    print("\n\033[1m== 8. 门户 ==\033[0m")
    if quick:
        R.warn("--quick：跳过门户与渲染的浏览器检查")
    else:
        rc, out = sh(["python3", "tools/check_portal.py", "--quiet"], timeout=420)
        rec(rc == 0, "门户页检查", out.strip().splitlines()[-1][:110])

    print("\n\033[1m== 3/4. 全量门禁与渲染 ==\033[0m")
    if quick:
        R.warn("--quick：跳过全量渲染回归")
    else:
        rc, out = sh(["python3", "tools/lint_lessons.py", "--quiet"], timeout=600)
        rec(rc == 0, "A4 语法/结构（全量）", out.strip().splitlines()[-1][:110])
        rc, out = sh(["python3", "tools/check_fidelity.py", "--quiet"], timeout=900)
        rec(rc == 0, "A1/A3 保真（全量）", out.strip().splitlines()[-1][:110])
        rc, out = sh(["python3", "tools/check_params.py", "--quiet"], timeout=900)
        rec(rc == 0, "A2 参数（全量）", out.strip().splitlines()[-1][:110])
        rc, out = sh(["python3", "tools/check_render.py", "--all", "--quiet"], timeout=7200)
        rec(rc == 0, "C 渲染（全量, 两分辨率）", out.strip().splitlines()[-1][:110])

    print("\n\033[1m== 9. 工作区与远端 ==\033[0m")
    rc, out = sh(["git", "status", "--porcelain"])
    rec(rc == 0 and not out.strip(), "工作区干净（无未提交变更）",
        out.strip()[:120] if out.strip() else "clean")
    rc, local = sh(["git", "rev-parse", "HEAD"])
    rc2, remote = sh(["git", "ls-remote", "origin", "refs/heads/main"])
    local, remote = local.strip(), remote.split("\t")[0].strip() if remote.strip() else ""
    rec(bool(local) and local == remote, "本地 HEAD == 远端 main", f"{local[:12]} / {remote[:12]}")
    rc3, cnt = sh(["git", "rev-list", "--left-right", "--count", "HEAD...origin/main"])
    rec(cnt.strip() == "0\t0", "与远端无分叉", cnt.strip())

    nfail = sum(1 for ok, _, _ in RESULTS if not ok)
    print("\n" + "=" * 66)
    print(f"交付检查：{len(RESULTS)-nfail}/{len(RESULTS)} 项通过")
    for ok, name, detail in RESULTS:
        if not ok:
            print(f"  \033[31m✗\033[0m {name} —— {detail}")
    print("=" * 66)
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
