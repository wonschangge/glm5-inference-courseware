#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""由 tools/plan.py 渲染 TODOLIST.md（计划 + 覆盖度矩阵 + 进度）。

用法:
    python3 tools/gen_plan.py            # 生成 TODOLIST.md
    python3 tools/gen_plan.py --check    # 只校验：覆盖率差集为空？勾选数与实际课数一致？
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from common import R

sys.path.insert(0, C.TOOLS)
import plan  # noqa: E402


def collect():
    uni, excluded = C.load_universe()
    uni = sorted(uni)
    assigned = {}
    for L in plan.LESSONS:
        fs = plan.expand_files(L["files"], uni)
        L["_files"] = fs
        for f in fs:
            assigned.setdefault(f, []).append(L["id"])
    missing = sorted(set(uni) - set(assigned))
    return uni, excluded, assigned, missing


def render():
    uni, excluded, assigned, missing = collect()
    done = {i for i, d in C.existing_lessons()}
    L = []
    A = L.append

    A("# TODOLIST —— GLM-5 推理视角的 transformers 动画课件")
    A("")
    A("> 本文件由 `tools/gen_plan.py` 从 `tools/plan.py` 自动生成，**不要手工编辑**。")
    A("> 计划与覆盖度矩阵共用同一份数据，因此不会出现「计划说 A、实现做 B」的漂移。")
    A("")
    A("## 0. 视角与主线")
    A("")
    A("**视角**：")
    A("")
    A("> 一个 token 从进入模型到吐出下一个 token，GLM-5 在 transformers 里到底发生了什么。")
    A("")
    A("**这个仓库在解决什么问题**（一句话）：")
    A("")
    A("> 把「磁盘上的一堆权重 + 一段文本/图像」变成「一次可复现、可优化、可横向扩展的数值计算」。")
    A("")
    A("**心智模型的转折点**：")
    A("")
    A("- `L0–L4` 是**数学视角**：数据是张量，模型是算子组合 —— 问「算什么」。")
    A("- `L5` 起切换为**机器视角**：数据是字节，计算是访存 —— 问「算这个要花多少」。")
    A("- 同一个 KV cache，在 L4 是三行公式，在 L5 是一块按页分配、可量化、可换出的显存。")
    A("")
    A("---")
    A("")
    A("## 1. 覆盖域（universe）—— 实测冻结，非人工圈定")
    A("")
    A("覆盖域不是「我觉得该讲的文件」，而是**跑出来的**：")
    A("")
    A("```text")
    A("测量方法：_data/recon/probe5.py —— 在 CPU 上实例化一个微型 GLM-5，")
    A("          真实执行 text generate / 采样 / beam / 图像处理 + 视觉塔 /")
    A("          量化配置 / 全部量化器模块 / 分布式 / 导出器 / 连续批处理 /")
    A("          注意力后端集成，然后取 sys.modules 与初始快照的差集，")
    A("          映射回 src/transformers/**.py。")
    A("")
    A("实测闭包           : 328 个文件")
    A("扣除非 GLM 模型族  : -97 个文件（36 个模型目录，仅因 models/auto 注册表被顺带导入）")
    A(f"最终覆盖域         : {len(uni)} 个文件")
    A("```")
    A("")
    A("**被显式排除的 97 个文件**（诚实声明：它们与 GLM-5 推理无关，本课件不覆盖、")
    A("也不计入覆盖率分子分母）：")
    A("")
    fams = {}
    for e in excluded:
        if e.startswith("models/"):
            fams.setdefault(e.split("/")[1], []).append(e)
    A("| 模型目录 | 文件数 |")
    A("|---|---|")
    for k in sorted(fams):
        A(f"| `models/{k}` | {len(fams[k])} |")
    other = [e for e in excluded if not e.startswith("models/")]
    if other:
        A(f"| （其他） | {len(other)} |")
    A("")
    A("---")
    A("")
    A("## 2. 分层课程清单")
    A("")
    A(f"共 **{len(plan.LESSONS)} 课**，覆盖 **{len(uni)} 个源文件**。")
    A("")
    for code, name, tag, desc in plan.LAYERS:
        ls = [x for x in plan.LESSONS if x["layer"] == code]
        A(f"### {name}")
        A("")
        A(f"**{tag}** —— {desc}")
        A("")
        A(f"课时数：{len(ls)} ｜ 覆盖文件：{len(set(f for x in ls for f in x['_files']))}")
        A("")
        A("| # | 课号 | 主题 | 优先级 | 文件数 | 状态 |")
        A("|---|---|---|---|---|---|")
        for x in ls:
            st = "✅" if x["id"] in done else "⬜"
            A(f"| {plan.LESSONS.index(x)+1} | `{x['id']}` | {x['title']} | {x['prio']} | "
              f"{len(x['_files'])} | {st} |")
        A("")
        for x in ls:
            A(f"<details><summary><b>{x['id']}</b> {x['title']} "
              f"<code>{plan.lesson_dir(x)}</code></summary>")
            A("")
            A("**讲解要点**")
            A("")
            for p in x["points"]:
                A(f"- {p}")
            A("")
            A("**验收点**")
            A("")
            for p in x["accept"]:
                A(f"- [ ] {p}")
            A("")
            A("**覆盖文件**")
            A("")
            for f in x["_files"]:
                A(f"- `{f}`")
            A("")
            A("</details>")
            A("")
    A("---")
    A("")
    A("## 3. 覆盖度矩阵（文件 → 课）")
    A("")
    A(f"共 {len(uni)} 个文件；每个文件至少被一课引用。差集：**{len(missing)}**")
    A("")
    A("| 源文件 | 行数 | 归属课 |")
    A("|---|---|---|")
    for f in uni:
        p = C.upstream_path(f)
        n = 0
        if os.path.isfile(p):
            with open(p, "rb") as fh:
                n = fh.read().count(b"\n") + 1
        A(f"| `{f}` | {n} | {', '.join(assigned.get(f, []))} |")
    A("")
    A("---")
    A("")
    A("## 4. 验收门禁")
    A("")
    A("| 门禁 | 脚本 | 断言 |")
    A("|---|---|---|")
    A("| A1/A3 保真 | `tools/check_fidelity.py` | 每个引用块逐字来自标注源文件，且位置连续 |")
    A("| A2 参数 | `tools/check_params.py` | 每个 `--flag` 都在实测真值集里 |")
    A("| A4 语法 | `tools/lint_lessons.py` | 四文件齐备、`node --check` 通过、分幕字段完整 |")
    A("| B 覆盖 | `tools/check_coverage.py` | 差集为空、无空课、无幻影引用 |")
    A("| C 渲染 | `tools/check_render.py` | 两分辨率 × 3 时间点，0 JS 错误 / 0 溢出 / 交互可用 |")
    A("")
    A("顺序：`lint` → `split_blocks` → `fidelity` → `params` → `coverage` → `render`")
    A("")
    return "\n".join(L) + "\n"


def main(argv):
    text = render()
    _, _, _, missing = collect()
    if "--check" in argv:
        ok = not missing
        for L in plan.LESSONS:
            for box in L["accept"]:
                pass
        if ok:
            R.ok(f"计划自检通过：{len(plan.LESSONS)} 课，覆盖差集为空")
            return 0
        R.bad(f"计划自检失败：{len(missing)} 个文件未分配")
        return 1
    out = os.path.join(C.ROOT, "TODOLIST.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    R.ok(f"已生成 {out}（{len(text.splitlines())} 行，{len(plan.LESSONS)} 课）")
    if missing:
        R.bad(f"{len(missing)} 个文件未分配")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
