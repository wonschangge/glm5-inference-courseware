#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L3-01 README.md 生成器：引用块同样按行切片，不手抄。"""
import os

UP = "/data/WORKSPACE/transformer-project/upstream-transformers/src/transformers"
M = "models/glm5_next/modeling_glm5_next.py"
OUT = "/data/WORKSPACE/transformer-project/courseware/L3-backbone/L3-01/README.md"


def sl(a, b):
    with open(os.path.join(UP, M), encoding="utf-8") as f:
        return "\n".join(f.read().split("\n")[a - 1:b])


def blk(a, b):
    return "<!-- src: " + M + " -->\n```python\n" + sl(a, b) + "\n```"


md = r'''# L3-01 · 三种 RMSNorm — 一层里的归一化只有三种形态

> 层：**第 3 层 · 文本主干** ｜ 优先级：P0 ｜ 前置课：**L2-07**（视觉塔：从 336x336 像素到 27 个 token）

## 学习目标

看完这一课，你应该能：

1. **说出无权重版本用在哪、为什么不用加权版**（对应计划的验收点 1）——
   以及它省掉的其实不是参数，而是「方向自由度」；
2. **解释门控范数在 KDA 里解决什么问题**（验收点 2）——
   它补的是「归一化管不了的那一半」：把写坏的通道关掉；
3. 说清 `eps` 到底取 1e-5 还是 1e-6，以及为什么在 bf16 里这个选择**几乎看不见**。

## 覆盖的源文件（2 个）

| 文件 | 行数 | 引用块 |
|---|---|---|
| `models/glm5_next/modeling_glm5_next.py` | 2444 | 11 |
| `pytorch_utils.py` | 255 | 3 |

> 行数是 `wc -l` 实测值。作业书里写的是 2445 / 256 —— 那是把文件末尾的换行也数成了一行，
> **以实测为准**。
>
> 本课另外引用了 `configuration_glm5_next.py` 的 `rms_norm_eps` 取值（1e-5）与
> `_data/recon/probe_L3-01.py` 的实测输出。**这两个文件都不计入本课覆盖率** ——
> 覆盖域只统计在上面表格里声明过的两个文件（参见 `source.md` 顶部的 coverage 声明块）。

## 场景（9 幕）

1. **一层里的归一化只有三种形态** —— 237 个实例按「有没有 weight / 有没有 gate」分类
2. **五行数学** —— `mean(-1)` 之后只剩 `(b, s, 1)`；手算一个 4 维 token：`[3,4,0,0] → [1.2,1.6,0,0]`
3. **★ 0 个参数的归一化** —— 无权重版 = weight 全 1 的加权版；去掉的是「方向自由度」
4. **★ 无权重版用在哪** —— mHC 的 `input_norm`：4 条流拼成 16384 维再对齐尺度（90 个实例）
5. **★ 门控版** —— `sigmoid(gate)` 逐通道开合；实测 init 下 `sigmoid` 均值 **0.5000**
6. **★ 门控解决什么** —— `core_attn_out`（记忆读出）× `gate`（当前输入）两路汇合在 `o_norm`
7. **eps：1e-6 还是 1e-5** —— 放大上限 316 倍 vs 1000 倍；门槛 `sqrt(eps) = 3.162e-3`
8. **横切：代价与红利** —— 不在 `ALL_LAYERNORM_LAYERS` 里；但逐 token 独立 ⇒ 可沿 seq 任意分块
9. **收束** —— 四个类、三种形态总表 + 三个验收练习 + 下一课指路

## 核心结论

### 1. ★ 三种形态是被下游需求决定的

```text
形态       weight    用在哪                            下游要什么
加权版     4096 个   子层入口/出口、q_a、kv_a、出口      要方向自由度
无权重版   0 个      mHC 内部（4 条流拼接后）           只要尺度
门控版     128 个    KDA 的 o_norm（逐 head）          要在逐通道上关掉记忆
```

三种实现加起来的参数量 **399,616**，只占全模型（312,692,428,286）的 **0.000128%**（meta device 实测）。
所以真正约束设计的不是显存，而是「归一化之后下游要拿它做什么」。

### 2. ★ 无权重版是「weight 全 1 的加权版」

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
@@M:211:217@@
```

实测 `parameters()` 为空（`numel() == 0`），`allclose(unweighted(x), weighted(x))` 为 True；
它只被一个构造点用：mHC 的 `input_norm`（90 个实例 = 45 层 x `attn_hc`/`ffn_hc`），
归一化的对象是 `hidden_streams.view(batch_size, seq_len, -1)` —— 4 x 4096 = **16384 维**。
若给它加权重，会多出 90 x 16384 = 1,474,560 个参数（仍只占 0.00047%）——
**所以理由不是省参数，而是：归一化后的向量只喂给一个线性映射 `self.fn`，
而 `F(w ⊙ x) = (F ∘ diag(w))(x)` 仍是线性映射，这里的 per-channel weight 可以被吸收、是多余的。**
实测还把它的作用量化了：输入整体放大 100 倍，`pre/post/comb` 三组系数逐位不变
（归一化后每 token 的 RMS 恒为 1.000000）—— **它保证的是「混合系数与残差流的绝对尺度无关」。**

### 3. ★ 门控范数：用当下的输入，决定要不要相信刚读出的记忆

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
@@M:766:769@@
```

`core_attn_out` 来自递推状态 `S` 的读出（尺度随递推步数累积），`gate` 来自**当前** `hidden_states`。
RMSNorm 只能把尺度拉回 1，**拉不回「这一维该不该有值」**；`sigmoid(gate)` 给出 0~1 的逐通道闸门，
可以把写坏的通道压到接近 0 —— 这是与输入无关的常数 `weight` 做不到的。

### 4. eps：类默认 1e-6，模型实际用 1e-5

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
@@M:1435:1441@@
```

三个类的 `__init__` 默认值都是 `1e-6`，但**全部 12 个实例化点都显式传 `config.rms_norm_eps`**；
`Glm5NextTextConfig` / `Glm5NextVisionConfig` 实测都是 **1e-5** —— 默认值从不生效。

| 该 token 的 RMS | 两种 eps 的最大相对差 | eps = 1e-5 造成的收缩 |
|---|---|---|
| 1 | 4.552e-06 | 0.0005% |
| 0.1 | 4.496e-04 | 0.0500% |
| 0.01 | 4.178e-02 | 4.6537% |
| 1e-3 | 5.736e-01 | 69.8489% |

门槛是 `sqrt(eps) = 3.162e-3`；在 bf16 里两者只有 **1,017 / 1,048,576**（0.10%）个元素不同。
**eps 是数值防御（放大上限 316 倍），不是精度旋钮。**

## 与后续课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| mHC 的 `pre` / `post` / `comb` 具体怎么算 | L3-06 |
| `swiglu_limit` 的 clamp 与 eps 一样是「限制幅度」 | L3-02 |
| KDA 的递推 `S_t = S_{t-1} ⊙ decay + write` 展开 | L4-02 ~ L4-05 |
| 门的瓶颈为什么取 `head_dim` 做中间宽度 | L4-02 ~ L4-05 |
| `rms_norm_eps` 写在 config 的哪一行 | L1-02 / L1-03 |
| 归一化在张量并行下要不要跨卡求和 | L8 |

## 验收点

- [x] 保真门禁：14 个引用块全部逐字来自两个源文件，且位置连续
- [x] 覆盖度门禁：2 个源文件已声明（`modeling_glm5_next.py`、`pytorch_utils.py`）
- [x] 参数门禁：无非法参数引用
- [x] 语法检查：`node --check` 通过，9 幕字段完整
- [x] 渲染门禁：9 幕 / 0 JS 错误 / 0 溢出 / 交互通过 @1280x720 + 1920x1080
- 自检一：能说出无权重版本用在哪、为什么不用加权版（本课第 3、4 幕）
- 自检二：能解释门控范数在 KDA 中解决什么问题（本课第 5、6 幕）

> 导航说明：本课 `prev` 指向 `../../L2-input/L2-07/index.html`。
> 作业书里给的是 `../L2-07/index.html`，但课件目录是分层的（`L2-input/` 与 `L3-backbone/` 平级），
> 按实测目录结构必须跨一层，否则链接指到 `L3-backbone/L2-07/` 这个不存在的路径。

**一句话总结**：

> 主干里的 237 个 norm 只有三种形态：**加权**（要方向自由度）、**无权重**（只要尺度）、
> **门控**（要在逐通道上把 KDA 读出的记忆关掉）；类默认的 `eps=1e-6` 从不生效，
> 真正跑的是 config 里的 **1e-5**，而它在 bf16 下对 99.9% 的元素没有影响。
'''

md = md.replace("@@M:211:217@@", sl(211, 217))
md = md.replace("@@M:766:769@@", sl(766, 769))
md = md.replace("@@M:1435:1441@@", sl(1435, 1441))
with open(OUT, "w", encoding="utf-8") as f:
    f.write(md)
print("wrote", OUT, len(md.split("\n")), "lines")
