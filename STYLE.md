# 单课制作规范（70 课统一口径）

> 这份文件是 L0-01 之外每一课的制作依据。**先读它，再动手。**

## 1. 目录与文件

```
<LAYER_DIR>/<课号>/
├── index.html    # 外壳，只声明 SHELL.boot(...)，不写业务逻辑
├── lesson.js     # const SCENES = [ ... ]
├── source.md     # 逐字引用 + 逐行注释 + coverage 声明块
└── README.md     # 学习目标 / 覆盖清单 / 核心结论 / 验收点
```

`LAYER_DIR` 见 `tools/plan.py` 的 `LAYER_DIR`（如 `L0-intuition`、`L4-attention`）。
**目录名就是课号**（`L0-01`），不带 slug。

## 2. `source.md`

### 2.1 覆盖声明块（B 组门禁的唯一输入）

放在文件**最开头**：

```markdown
<!-- glm5-coverage
models/glm5_next/modeling_glm5_next.py
transformers/generation/utils.py
-->
```

- 路径**相对 `src/transformers/`**，与 `_data/universe.json` 里的写法一致。
- 必须与 `tools/plan.py` 里该课的 `files` 完全一致。
- 只列**覆盖域内**的文件。引用了脚本/文档等非源文件时，在正文里
  **显式标注「不计入覆盖率」**。

### 2.2 代码引用（A 组门禁的唯一输入）

每个引用块**必须**紧跟在 `<!-- src: path -->` 之后：

```markdown
<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        hidden_states = self.norm(self.hc_head(hidden_states))
```
```

**铁律**：

| 规则 | 说明 |
|---|---|
| **逐字** | 一个字都不能改。变量名、空格、注释全部原样。 |
| **连续** | 引用块必须对应源文件里**一段连续的行**。中间夹注释行时，把注释一起引进来。 |
| **不连续就拆** | 需要跳过若干行时，写**两个**引用块，不要在一个块里省略。 |
| **禁止占位符** | 不要写 `(...)`、`<...>`、`# ...` 之类的省略。 |
| **示意代码用 `text`** | 教学伪代码、汇总表、流程图一律用 ```` ```text ````，否则会被门禁当真实引用。 |

写完 source.md 后立刻跑：

```bash
python3 tools/split_blocks.py --lesson <课号>     # 自动拆不连续块
python3 tools/check_fidelity.py --lesson <课号>   # 逐字 + 连续
```

> 若某个块**反复**拆不动、门禁一直报同一处 —— 那不是连续性问题，
> 是**打错字或凭记忆改写**，回去核对源文件。

### 2.3 正文写法

- **先结论后证据**：每节开头先说这节要讲什么。
- **★ 标重点**：一课只有 1–3 个真正的洞察，用 `## 一、★ ...` 标出。
- **跨课呼应显式写**：`回顾 L3-06：……` / `这与 L5-05 的判据一致`。
- **注释讲"为什么"**，不只是描述行为。

## 3. `lesson.js`

### 3.1 骨架

```javascript
/* ==========================================================================
   <课号> · <主题>
   --------------------------------------------------------------------------
   覆盖：<文件>（N 个文件 / M 行）
   目标：<这一课要让学员获得的能力>
   排版基准：#visual 可视区 = 994 x 662.7 舞台像素（无头浏览器实测）。
   ========================================================================== */
'use strict';

const SCENES = [
{
  kicker: '<层名> · <阶段>',
  title: '<标题，可用 <span class="hl-a">高亮</span>>',
  sub: '<一句话说明本幕讲什么>',
  caption: '<页脚补充，常用来指路到相关课>',
  lang: 'python',
  codeStart: <源码起始行号>,
  code: `<逐字引用的代码>`,
  codeNote: '<代码栏底部说明>',
  duration: 15000,          // 必须 > 最后一个 tl.at 的时间点
  build(root, tl) { ... }
},
];
```

### 3.2 硬性约束

| 约束 | 原因 |
|---|---|
| **经典 `<script>`，不用 ES module** | `file://` 双击可开 |
| **零 CDN、零外链、零构建** | 离线可用 |
| **只引用 `shared/`，不复制引擎代码** | 引擎单一样例来源 |
| **`tl.at(ms, fn)` 时间点递增，间隔 2000–4000ms** | 让人读完文字 |
| **最后一个 `tl.at` < `duration`** | 否则收束帧播不到 |
| **`build` 必须幂等** | 跳转 = 重建 + 重放，不能依赖外部状态 |
| **所有动画用 CSS transition 或直接设值** | 补间帧在 seek 时不会重放，会跳变 |

### 3.3 排版基准

```
#visual 可视区 = 994 x 662.7 舞台像素
一行 N 张卡：N × 卡宽 + (N-1) × 间距 ≤ 994，留 10–20px 余量
```

- 分幕的外层用 `<div class="col gap14" style="width:100%;height:100%">`。
- **主视觉块必须包在 `<div class="vizgrow">` 里**：它吃掉剩余高度并在其中居中，
  否则内容会挤在顶部、底部留出大片空白。
- `.vizgrow.hstart` = 水平拉伸；默认 = 居中。
- 长标识符**不要手工 `<br>` 拆行**（会被参数门禁误判），交给 CSS 换行。
- 字号分层：标题 30px（页头统一）/ 卡片标题 12.5px / 正文 11.5px / 说明 10–11px。

### 3.4 分幕设计

- **首幕给全局**：本幕/本课覆盖什么、为什么重要。
- **一幕只讲一件事**。宁可 9 幕，不要 4 幕塞满。
- **末幕做收束**：压成一张表 + 一句话 + 练习 + 下一课指路。
- 用 `tl.at(...)` 编排"聚焦"：`W.card` 加/去 `ac`、`W.flow().focus(k)`、元素 `opacity`。

### 3.5 可用的组件（`shared/widgets.js`）

| 组件 | 用途 |
|---|---|
| `W.card({title, sub, cc, tint, num, body, kids})` | 卡片；`cc` 设强调色 0–5 |
| `W.cardRow(defs, {flex, numbered})` | 等宽卡片行 |
| `W.table(rows, {head})` | 键值表 |
| `W.bars(items, {max})` | 条形图，`.setAll(t)` / `.setTo(i,v)` |
| `W.stack(items)` | 堆叠条，`.reveal(t)` |
| `W.flow(nodes)` | 流程线，`.focus(k)` |
| `W.tensor(rows, cols, owner, {cell,text})` | 张量网格 |
| `W.matrix(rows, cols, val, {cell})` | 热力图，`.set(r,c,v)` |
| `W.mesh(axes, {cell})` | 设备网格 |
| `W.exercise(q, a)` | 折叠式练习 |
| `U.chip(name, ci)` / `U.codeEl(code)` / `U.hot(lines, idxs)` | 工具 |

### 3.6 数据流层（`FLOW`）

```javascript
FLOW.path('p1', [[x1,y1],[x2,y2]]);                 // 舞台坐标
FLOW.pathBetween('p1', elA, elB);                   // 由元素几何自动取
FLOW.rail('p1', 0, '34,211,238');                   // 静态发光虚线
FLOW.send('p1', { at: 2000, dur: 1200, color: '52,211,153', label: 'KV' });
```

`FLOW` 由引擎统一驱动，**暂停/跳转同样有效**（seek 会 clear 后由重放事件重新登记）。

## 4. `index.html`

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title><课号> <主题> · GLM-5 推理课件</title>
<link rel="icon" href="data:image/svg+xml,...">   <!-- 内联，避免 404 -->
<link rel="stylesheet" href="../../shared/theme.css">
</head>
<body>
<script src="../../shared/engine.js"></script>
<script src="../../shared/widgets.js"></script>
<script src="../../shared/lesson-shell.js"></script>
<script src="lesson.js"></script>
<script>
SHELL.boot(SCENES, {
  pageTitle: '<课号> <主题> · GLM-5 推理课件',
  kicker: 'GLM-5 推理 &nbsp;·&nbsp; <课号>',
  codeName: '<主源文件名>',
  links: [{ href: '../../index.html', label: '门户' }],
  nav: { prev: { href: '../<上一课>/index.html', label: '<上一课>' },
         next: { href: '../<下一课>/index.html', label: '<下一课>' } }
});
</script>
</body>
</html>
```

> **`<link rel="stylesheet">` 的相对路径由页面提供，`SHELL.boot` 会沿用它。**
> 不要给 `SHELL.boot` 传 `css`（除非确实要覆盖）。

**导航链必须闭环**：新增课插在中间时，**同时改前后两课**的 `prev` / `next`。

## 5. `README.md`

必含：学习目标（3 条内，对应该课验收点）、覆盖文件表、分幕清单、
核心结论（带逐字引用）、与后续课的接口表、验收点勾选、一句话总结。

`前置课` 一行**必须**存在，且编号**严格小于**本课（`lint_lessons.py` 会查）。

## 6. 一课一提交

```bash
python3 tools/run_gates.sh --lesson <课号>       # 全绿
git add -A && git commit -F - <<'MSG'
feat(<课号>): <简述>

<这一课讲了什么、最核心的洞察>

覆盖: <本课覆盖的文件，逗号分隔>
验收: coverage=<n>/<N>  fidelity=pass（<k> 个引用块逐字命中）  flags=pass
      lint=pass  render=<m>幕/0错误/0溢出 @1280x720 + 1920x1080
前置: <前置课编号，或 none>
MSG
git push origin main
git fetch -q origin && git rev-list --left-right --count HEAD...origin/main   # 必须 "0  0"
```
