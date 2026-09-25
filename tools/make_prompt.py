#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为某一课生成**完整的子代理作业书**（写入 _data/prompts/<课号>.md）。

这样派发子代理时只需一句「读 <路径> 并照做」，不必每次重述全部规范，
也保证 70 课的作业口径完全一致。

用法:
    python3 tools/make_prompt.py L1-01 [L1-02 ...]
    python3 tools/make_prompt.py --layer L4
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from common import R

sys.path.insert(0, C.TOOLS)
import plan  # noqa: E402

OUTDIR = os.path.join(C.DATA, "prompts")

# 每层的额外提示：让子代理知道这一层该看哪些实现细节
LAYER_HINTS = {
 "L0": "预备篇只建立直觉，不要展开实现细节；每个结论都要有 config 默认值或实测输出支撑。",
 "L1": "重点是「字符串如何找到类」与「config 如何决定形状」。AutoModel 的惰性映射要讲清"
       "「用时才 import」省下了什么。",
 "L2": "重点是形状变换：文本 token 与视觉 token 如何在同一条序列上对齐。"
       "务必给出真实的张量形状演算。",
 "L3": "重点是逐算子拆解与「为什么这样设计」。涉及数值（如 swiglu_limit、Sinkhorn 迭代）"
       "要给出手算例子。",
 "L4": "这是全课件的重心。KDA 的循环形式与分块形式要在数学上讲清等价性；"
       "DSA 索引器要讲清 2048 个位置的预算怎么来的；MLA 要给出 KV 压缩比。",
 "L5": "★ 这是心智模型的转折点。前四层把 KV 当公式，这一层要把它当字节："
       "占多少显存、按什么粒度分配、溢出怎么办。所有显存数字都要给出算式与单位。",
 "L6": "重点是「一个 token 是怎么被决定的」。logits 处理器的顺序会改变结果，"
       "要显式讲清串联顺序。",
 "L7": "重点是磁盘字节到显存张量的完整路径。量化部分要给出每种方案的位宽、粒度与"
       "是否需要校准。",
 "L8": "重点是切分与通信。每种并行都要给出「切什么、切完还要不要通信」的结论，"
       "以及通信量量级。",
 "L9": "这一层是周边设施，但每一课都要回答「为什么推理离不开它」。"
       "L9-02 要诚实说明：这些 loss 文件确实在闭包里，但推理前向不执行它们。",
}


def src_lines(rel):
    p = C.upstream_path(rel)
    if not os.path.isfile(p):
        return 0
    with open(p, "rb") as f:
        return f.read().count(b"\n") + 1


def outline(rel, maxn=70):
    """抽取 class/def 大纲，帮助子代理快速定位。"""
    p = C.upstream_path(rel)
    if not os.path.isfile(p):
        return []
    out = []
    try:
        text = C.read_text(p)
    except Exception:
        return []
    for i, ln in enumerate(text.split("\n"), 1):
        s = ln.rstrip()
        if s.startswith("class ") or s.startswith("def ") or s.startswith("    def "):
            out.append(f"  L{i}: {s.strip()}")
            if len(out) >= maxn:
                out.append("  …（更多请自行阅读）")
                break
    return out


def build(lid):
    L = next((x for x in plan.LESSONS if x["id"] == lid), None)
    if L is None:
        raise SystemExit("未知课号: " + lid)
    uni, _ = C.load_universe()
    files = plan.expand_files(L["files"], uni)
    idx = plan.LESSONS.index(L)
    prev_id = plan.LESSONS[idx - 1]["id"] if idx > 0 else None
    next_id = plan.LESSONS[idx + 1]["id"] if idx + 1 < len(plan.LESSONS) else None
    prev_dir = ("../" + prev_id.split("-")[0] + "-" + {
        "L0": "intuition", "L1": "config", "L2": "input", "L3": "backbone",
        "L4": "attention", "L5": "cache", "L6": "generation", "L7": "loading",
        "L8": "scaling", "L9": "periphery"}[prev_id.split("-")[0]] + "/" + prev_id + "/index.html") if prev_id else None
    next_dir = ("../" + next_id.split("-")[0] + "-" + {
        "L0": "intuition", "L1": "config", "L2": "input", "L3": "backbone",
        "L4": "attention", "L5": "cache", "L6": "generation", "L7": "loading",
        "L8": "scaling", "L9": "periphery"}[next_id.split("-")[0]] + "/" + next_id + "/index.html") if next_id else None
    # 同层内的邻居优先作为 nav，更容易一条线读下来
    same = [x["id"] for x in plan.LESSONS if x["layer"] == L["layer"]]
    k = same.index(L["id"])
    nav_prev = same[k - 1] if k > 0 else prev_id
    nav_next = same[k + 1] if k + 1 < len(same) else next_id

    def href(other):
        if other is None:
            return None
        return f"../{other}/index.html"

    d = plan.lesson_dir(L)
    P = []
    A = P.append
    A(f"# 作业书 · {L['id']} {L['title']}")
    A("")
    A("你在为一个中文动画课件项目产出一课。**所有文字用简体中文。** 严格照本作业书执行。")
    A("")
    A("## 工作路径")
    A("")
    A(f"- 课件仓库根目录（你的工作目录）：`{C.ROOT}`")
    A(f"- 上游源码根目录：`{C.UPSTREAM}`")
    A(f"- 你要创建的课件目录：`{d}/`")
    A("")
    A("## 第一步 —— 必须先读（按顺序）")
    A("")
    A("1. `STYLE.md` —— 制作规范，具有约束力。")
    A("2. `L0-intuition/L0-01/` 的四个文件 —— **黄金样例**。你的产出要在深度、语气、")
    A("   视觉丰富度上与之持平。")
    A("3. `shared/engine.js`、`shared/widgets.js`、`shared/theme.css` —— 你可以使用的 API。")
    A("   **不要修改这三个文件。**")
    A("4. 下面列出的上游源文件（**完整读，不要只读片段**）。")
    A("")
    A("## 第二步 —— 本课规格")
    A("")
    A(f"- **课号**：`{L['id']}`")
    A(f"- **标题**：{L['title']}")
    A(f"- **所属层**：{L['layer']} —— "
      + next((f"{n}（{t}）" for c, n, t, _ in plan.LAYERS if c == L["layer"]), ""))
    A(f"- **优先级**：{L['prio']}")
    A(f"- **前置课**：{nav_prev or '无'}")
    A("")
    A("### coverage 声明（`source.md` 开头，逐字照抄这一段）")
    A("")
    A("```markdown")
    A("<!-- glm5-coverage")
    for f in files:
        A(f)
    A("-->")
    A("```")
    A("")
    A("### 讲解要点（必须全部覆盖）")
    A("")
    for p in L["points"]:
        A(f"- {p}")
    A("")
    A("### 验收点（必须出现在 README 的「验收点」小节，且课程内容要真的支撑它）")
    A("")
    for p in L["accept"]:
        A(f"- {p}")
    A("")
    A(f"### 本层特别注意\n\n{LAYER_HINTS.get(L['layer'], '')}")
    A("")
    A("### 覆盖的源文件（含行数与该文件的结构大纲，便于定位）")
    A("")
    for f in files:
        n = src_lines(f)
        A(f"#### `{f}` —— {n} 行")
        A("")
        o = outline(f)
        if o:
            A("```text")
            P.extend(o)
            A("```")
        else:
            A("（无 class/def 大纲，请直接阅读）")
        A("")
    A("## 第三步 —— 产出四个文件")
    A("")
    A(f"全部写在 `{d}/` 下：")
    A("")
    A("### 1. `source.md`")
    A("")
    A("- 第一行就是上面的 coverage 声明块。")
    A("- 然后是若干**引用块**，每个引用块**前面必须有一行** "
      "`<!-- src: <相对路径> -->`（路径就是 coverage 里那样写）。")
    A("- 引用块用 ```` ```python ```` 围栏。**逐字**来自源文件，且在源文件里**位置连续**。")
    A("  绝对不要凭记忆重写 —— 直接复制。需要跳过若干行时，写**两个**引用块，")
    A("  不要在一个块里省略。")
    A("- 每个引用块后面跟一段「**读法**」，解释**为什么**这样写，而不只是描述它做了什么。")
    A("- 教学用的伪代码、汇总表、流程图一律用 ```` ```text ```` —— 不会参与保真校验。")
    A(f"- 引用块数量：**8–14 个**（文件多的话每个文件至少 2 个）。")
    A("- 用 `## 一、★ …` 标出 1–3 个真正的洞察。")
    A("- 显式写出跨课呼应，例如「回顾 L0-01：……」。")
    A("")
    A("### 2. `lesson.js`")
    A("")
    A("- `'use strict';` + `const SCENES = [ ... ];`")
    A("- **9 幕**。每幕字段：`kicker / title / sub / caption / lang / codeStart / code / "
      "codeNote / duration / build(root, tl)`。")
    A("- `code` 必须是 source.md 里引用过的**逐字**代码（可直接从 source.md 拷）。")
    A("- `codeStart` = 该段在源文件里的起始行号（大纲里的 L 编号可参考）。")
    A("- `build(root, tl)` 里：外层用 `<div class=\"col gap14\" style=\"width:100%;height:100%\">`，")
    A("- 主视觉块包在 `<div class=\"vizgrow\">` 里（吃掉剩余高度并居中），")
    A("  否则内容会挤在顶部、底部留大片空白 —— 渲染门禁虽不报错，但观感不合格。")
    A("- 用 `tl.at(ms, fn)` 编排，时间点**递增**、间隔 1800–3500ms，最后一个 **< duration**。")
    A("- `build` 必须**幂等**：跳转 = 重建 + 快进重放。所以动画只用 CSS transition 或直接设值，")
    A("  不要用 `setTimeout` / `requestAnimationFrame` / 外部可变状态。")
    A(f"- **排版基准：`#visual` = 994 × 662.7 舞台像素。** 一行 N 张卡片的总宽 + 间距 ≤ 994。")
    A("- 长标识符不要手工 `<br>` 拆行（会被参数门禁误判），交给 CSS 换行。")
    A("")
    A("可用组件见 `STYLE.md` 第 3.5 节（`W.card` / `W.table` / `W.bars` / `W.stack` / "
      "`W.flow` / `W.tensor` / `W.matrix` / `W.mesh` / `W.exercise`）与 `FLOW` 数据流层。")
    A("")
    A("### 3. `index.html`")
    A("")
    A("照抄 `L0-intuition/L0-01/index.html` 的结构，只改 title / kicker / codeName / nav。")
    A("")
    A(f"- stylesheet 路径：**`../../shared/theme.css`**")
    A(f"- `codeName`：`{os.path.basename(files[0]) if files else 'modeling_glm5_next.py'}`")
    A("- nav（注意相对路径）：")
    A("```javascript")
    if nav_prev:
        A(f"  prev: {{ href: '{href(nav_prev)}', label: '{nav_prev}' }},")
    else:
        A("  prev: null,")
    if nav_next:
        A(f"  next: {{ href: '{href(nav_next)}', label: '{nav_next}' }}")
    else:
        A("  next: null")
    A("```")
    A("- 不要给 `SHELL.boot` 传 `css`（它会自动沿用 index.html 里的相对路径）。")
    A("")
    A("### 4. `README.md`")
    A("")
    A("结构照抄 L0-01 的 README：学习目标 / 覆盖的源文件（表格，含行数）/ 场景（N 幕，"
      "★ 标重点）/ 核心结论（带逐字引用）/ 与后续课的接口（表格）/ 验收点（勾选框）/ "
      "一句话总结。")
    A("")
    A(f"**`前置课` 那一行必须存在，且写 `{nav_prev or '无'}`**（编号必须严格小于本课，"
      "`lint_lessons.py` 会检查）。")
    A("")
    A("## 第四步 —— 反复跑到四道门禁全绿")
    A("")
    A("在课件仓库根目录依次执行，**把每一条报错都修掉**：")
    A("")
    A("```bash")
    A(f"python3 tools/split_blocks.py --lesson {L['id']}")
    A(f"python3 tools/check_fidelity.py --lesson {L['id']}")
    A(f"python3 tools/lint_lessons.py --lesson {L['id']}")
    A(f"python3 tools/check_params.py --lesson {L['id']}")
    A(f"python3 tools/check_render.py --lesson {L['id']}")
    A("```")
    A("")
    A("要点：")
    A("")
    A("- **保真门禁最严**：`python` 块里每一行都必须在源文件中逐字、连续地存在。")
    A("  报「第 k/n 行断裂」= 那一行不是打错字就是凭记忆改写了 —— 回去**复制**原文。")
    A("  报「未标注 src」= 那个代码块缺了 `<!-- src: ... -->` 标注。")
    A("- 如果某个块反复拆不动、门禁一直报同一处，**那不是连续性问题，是内容错了**。")
    A("- `check_render.py` 会开无头浏览器，约 1–2 分钟，检查：0 JS 错误、")
    A("  两种分辨率下 0 布局溢出、播放/暂停/换幕/方向键/刻度跳转都可用。")
    A("  报溢出就调小或重排那一幕。")
    A("- 若 `check_params.py` 报某个 `--flag` 不存在，改用真实存在的写法。")
    A("")
    A("## 第五步 —— 汇报")
    A("")
    A("**不要执行任何 `git` 命令**（提交由主控代理统一做）。四道门禁全绿后，简短汇报：")
    A("")
    A("- 课号与目录")
    A("- `source.md` 的引用块数量，以及保真门禁输出的**原文那一行**")
    A("- 幕数，以及渲染门禁输出的**原文那一行**")
    A("- 任何你无法验证、或不得不绕开的地方")
    A("")
    A("汇报控制在 15 行以内。")
    return d, "\n".join(P) + "\n"


def main(argv):
    os.makedirs(OUTDIR, exist_ok=True)
    ids = []
    if "--layer" in argv:
        lay = argv[argv.index("--layer") + 1]
        ids = [x["id"] for x in plan.LESSONS if x["layer"] == lay]
    else:
        ids = [a for a in argv if not a.startswith("--")]
    if not ids:
        R.bad("用法: make_prompt.py L1-01 … 或 --layer L4")
        return 2
    for lid in ids:
        d, text = build(lid)
        out = os.path.join(OUTDIR, lid + ".md")
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
        R.ok(f"{lid} -> {os.path.relpath(out, C.ROOT)}  ({len(text.splitlines())} 行, 目标 {d})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
