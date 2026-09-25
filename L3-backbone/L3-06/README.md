# L3-06 · ★ mHC：流形约束超连接

> 层：**第 3 层 · 文本主干** ｜ 优先级：P0 ｜ 前置课：`L3-05`

## 学习目标

看完这一课，你应该能：

1. **复述 mHC 的三步**（读取 → 收拢 → 写回），并说出每一步对应哪几行源码；
2. **手算 2 次 Sinkhorn 迭代**，并说出为什么第一步是「列归一」而不是「行归一」（验收点 1）；
3. **解释双随机约束对梯度流的稳定作用**：行随机会漏掉什么、列和补上了什么（验收点 2）。

## 覆盖的源文件（1 个）

| 文件 | 行数 | 引用块 |
|---|---|---|
| `models/glm5_next/modeling_glm5_next.py` | 2444 | 11 |

> **实测订正**：作业书写这个文件 2445 行；实测 `wc -l` = 2444、`str.splitlines()` = 2444，
> 文件以换行结尾、最后一行是 `]`。本课以**实测 2444** 为准。
> 作业书给的行号大纲（L220 / L273 / L287 / L334）与实测一致。

> 本课引用了工装脚本 `_data/recon/probe_l306.py` 的**实测输出**（写在 `source.md` 的 `text` 块里）。
> **脚本不计入覆盖率** —— 覆盖域只统计 `src/transformers/**/*.py`。

## 场景（9 幕）

1. **★ 三步全景** —— 读取 / 收拢 / 写回，加上四个关键数字（4 条流 / 24 个数 / 20 次迭代 / 1e-6）
2. **读取** —— 摊平 16384 维 → 无权重 RMSNorm → `fn (24 × 16384)` → 切成 pre / post / comb
3. **三个出口** —— 三种激活、三种值域；`pre` 不归一化、`post` 中心取 1、`comb` 还要过 Sinkhorn
4. **★ 洞察 1：行随机不够** —— 4 行相同的退化例：行和全 1，`sigma_max` 却是 1.4631
5. **★ 手算 2 次 Sinkhorn 迭代** —— 3×3 矩阵，行和误差 1.667e-01 → 2.175e-02 → 2.819e-03
6. **写回** —— `post ⊙ out + combᵀ · 旧流`，广播与两次实例（attn_hc / ffn_hc）
7. **收拢** —— `pre` 是门控不是平均：实测权重和 2.011953，不是 1
8. **出口：HyperHead** —— `mean(dim=2)`，0 参数；与入口的 `pre` 形成对照
9. **收束** —— 一张表 + 两条不变量 + 手算练习 + 下一课指路

## 核心结论

### 1. ★ mHC 只有一次矩阵乘法，但它同时产出三个出口

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        pre_w, post_w, comb_w = flattened.split([hc, hc, hc * hc], dim=-1)
```

`concatenated_weights_size = (2 + self.hc_mult) * self.hc_mult`，实测 `N=4` 时是 **24**：
`pre(4) + post(4) + comb(16)`。实测参数量：`fn` 393,216 + `base` 24 + `scale` 3 = **393,243**，
每层两个、45 层共 **35,391,870**。

### 2. ★ Sinkhorn 的 20 次迭代 = 1 次列归一 + 19 次「行 + 列」

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
```

顺序反常是有原因的：`comb` 刚从 `softmax(dim=-1)` 出来，**行已经和是 1**，先补列才是最快的一步。
实测（`hidden_size=8`）：`iters=1/2/3` 的行和误差是 `3.4e-3 → 3.8e-5 → 1.4e-6`，
`1e-6` 的平台就是 `hc_eps`。

### 3. ★ 双随机 = 两条不变量，合起来把 45 层的界钉在 1

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        # The comb weight is a bit different: it dictates how the input streams (In) are added to the output streams
        # (Out) in this way: Mixed = In @ Comb + Out. To make sure the norm of "Mixed" does not blow up, we constrain
        # the comb weight to be doubly-stochastic (ie. its rows and columns must sum to 1) with a few iterations of the
        # Sinkhorn-Knopp algorithm, which iteratively normalizes the rows and columns to sum to 1.
```

| 量 | 行随机（只有 softmax） | 双随机（再过 Sinkhorn） |
|---|---|---|
| 单层 `sigma_max`（200 个 4×4 样本的上确界） | 1.3017 | 0.999999 |
| 退化例（4 行相同）的 `sigma_max` | 1.4631 | 0.999999 |
| 45 层的界 | `1.3017^45 ≈ 1.42e5` | `1.000000` |

`comb` 的两条实测不变量：`comb @ 1` 与 1 的偏差 `1.043e-06`、`1^T @ comb` 与 1 的偏差 `1.028e-06`；
`comb` 的最小元素是 `0.2021`（没有流被饿死），`sigma_max(comb) = sigma_max(combᵀ) = 0.999999`。

### 4. `pre` 不是加权平均，`hc_head` 也不是

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
        return post, comb, collapsed
```

实测 `pre = [0.5677, 0.4582, 0.5149, 0.4712]`，**和是 2.011953**：收拢是门控求和，不是凸组合。
出口处的 `hc_head` 则相反 —— `hidden_streams.mean(dim=2)`，参数量实测 **0**。

## 与后续课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| `hidden_states` 为什么一开始就是 4 维 | L0-01（已讲） |
| 每层两个 mHC 与注意力 / MoE 的装配顺序 | L3-08 |
| `mlp` 那一支里共享专家怎么合流 | L3-05 |
| 子层内部（KDA / MLA）怎么消费收拢后的 1 条流 | L3-08、L4 整层 |
| `norm(hc_head(x))` 之后到 logits | L6-01 ~ L6-03 |
| 45 层的残差流在推理时怎么被保存 | L5 整层 |

## 验收点

- [x] 保真门禁：11 个引用块全部逐字来自 `modeling_glm5_next.py`（A1/A3 通过）
- [x] 覆盖度门禁：1 个源文件被声明（`models/glm5_next/modeling_glm5_next.py`）
- [x] 参数门禁：无非法参数引用
- [x] 语法检查：`node --check` 通过，9 幕字段完整
- [x] 渲染门禁：0 JS 错误、0 布局溢出、交互可用（1280×720 + 1920×1080）
- [ ] 自检 1：能手算 2 次 Sinkhorn 迭代（见第 5 幕与第 9 幕练习）—— **能手算 2 次 Sinkhorn 迭代**
- [ ] 自检 2：能解释双随机约束对梯度流的稳定作用 —— **能解释双随机约束对梯度流的稳定作用**

**一句话总结**：

> mHC 把残差从「一条和」改成「4 条流 + 一张混合矩阵」：读取一次投影出 24 个数、
> 收拢用未归一化的 `pre` 门控、写回用 `post ⊙ out + combᵀ · 旧流`；
> 而 `comb` 必须先被 Sinkhorn 压成双随机 —— 只有行列和都为 1，才能把 45 层的放大倍数钉死在 1。
