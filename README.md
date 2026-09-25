# GLM-5 推理视角的 transformers 动画课件

> 一套**可离线双击打开**的交互式动画课件：以「一个 token 从进入模型到吐出下一个 token」为视角，
> 逐层拆解 [huggingface/transformers](https://github.com/huggingface/transformers) 里
> **GLM-5.3-Flash（`glm5_next`）** 的推理路径。

## 这是什么

- **70 课 / 10 层**，覆盖 **231 个源文件** —— 覆盖域不是人工圈定的，而是**跑出来的**
  （见 `_data/recon/probe5.py`：在 CPU 上真实执行一次 GLM-5 推理闭包，取 `sys.modules` 差集）。
- 每课三件套：**看源码**（逐字引用）→ **看动画**（9 幕左右）→ **做练习**。
- 每一条代码引用都经过**逐字校验**，每一个参数都对照**实测真值集**。

## 快速开始

直接双击任意课的 `index.html` 即可 —— 无需服务器、无需联网、无 CDN、无构建步骤。

从门户进入：打开 `index.html`。

## 分层

| 层 | 主题 | 核心问题 |
|---|---|---|
| L0 | 预备篇 · 直觉 | 45 层的模型长什么样 |
| L1 | 配置与分派 | config 如何决定形状，AutoModel 如何找到 GLM-5 |
| L2 | 输入流水线 | 文本和像素如何变成同一串位置上的向量 |
| L3 | 文本主干 | 一层之内有哪几个算子 |
| L4 | 混合注意力 ★ | 34 层 KDA + 11 层 MLA/DSA 如何协同 |
| L5 | 缓存与内存 ★★ | **转折点**：KV 从公式变成字节 |
| L6 | 生成循环 | 一个 token 是怎么被「决定」出来的 |
| L7 | 权重加载与量化 | 磁盘上的字节如何变成显存里的张量 |
| L8 | 推理规模化 | 单卡跑不动时模型被怎么切开 |
| L9 | 出口与周边 | 导出器、基础设施、模型族继承链 |

## 门禁

四道门禁是「课件可信」的唯一依据。顺序：`lint → split_blocks → fidelity → params → coverage → render`

```bash
tools/run_gates.sh --static          # 静态门禁（秒级）
tools/run_gates.sh --lesson L0-01    # 单课全量（含渲染）
tools/run_gates.sh                   # 全量回归
```

| 门禁 | 脚本 | 断言 |
|---|---|---|
| A1/A3 保真 | `tools/check_fidelity.py` | 每个引用块逐字来自标注源文件，且**位置连续** |
| A2 参数 | `tools/check_params.py` | 每个 `--flag` 都在实测真值集里 |
| A4 语法 | `tools/lint_lessons.py` | 四文件齐备、`node --check` 通过、分幕字段完整 |
| B 覆盖 | `tools/check_coverage.py` | 差集为空、无空课、无幻影引用 |
| C 渲染 | `tools/check_render.py` | 两分辨率 × 3 时间点，0 JS 错误 / 0 溢出 / 交互可用 |

## 制作规范

新增一课请先读 [`STYLE.md`](STYLE.md)；计划与覆盖度矩阵见 [`TODOLIST.md`](TODOLIST.md)
（由 `tools/plan.py` 生成，是计划的唯一真源）。

## 上游

课件引用的源码来自 `huggingface/transformers`（`src/transformers/`）。
上游文件路径在每课 `source.md` 的 coverage 块里逐条列出。
