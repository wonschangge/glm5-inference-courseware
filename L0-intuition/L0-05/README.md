# L0-05 · mHC：4 条残差流

> 层：**预备篇 · 直觉** ｜ 优先级：P1 ｜ 前置课：**L0-04**（残差与归一化的入口）

## 学习目标

看完这一课，你应该能：

1. 说出主干里 `hidden_states` 的形状为什么是 `(batch, seq, 4, 4096)`，以及这 4 条流在**哪三处**被读写；
2. **手算两次 Sinkhorn 迭代**，并解释双随机约束为什么能让混合"不爆炸"（对应验收点 1）；
3. 指出 mHC 在解码层里出现**两次**的位置与各自的作用（对应验收点 2）。

## 覆盖的源文件（1 个）

| 文件 | 行数 | 引用块 |
|---|---|---|
| `models/glm5_next/modeling_glm5_next.py` | 2444 | 12 |

> 本课的实测数字来自一次性的探针脚本（`hc_mult=4` 的真实尺寸模块 + 一个 `hidden=64` 的极小模型），
> 汇总写在 `source.md` 顶部的 `text` 块里。**这些是脚本，不计入覆盖率** ——
> 覆盖域只统计 `src/transformers/**/*.py`。

## 场景（9 幕）

1. **全景：三次出现** —— 一行展开 → 每层两次读写 → 一行收拢，四个要记住的数
2. **形状：一条 → 一束** —— `expand` 是 stride=0 的视图，`.contiguous()` 才铺成 4 份
3. **三步：读取 → 混合 → 写回** —— 前两步在 mHC 内部，第三步在解码层
4. **★ 双随机** —— 4×4 `comb` 的行和 / 列和逐轮收敛到 1，最后一次是列归一化
5. **手算两次迭代** —— 四个矩阵并排，每一步的行和 / 列和都写出来
6. **写回公式** —— `新流 = post ⊙ 子层输出 + combᵀ · 旧流`，两路相加 + 形状推演
7. **层内两次** —— 9 行执行顺序表，`attn_hc` / `ffn_hc` 各一次
8. **出口收拢** —— `hc_head` 是等权平均（0 参数），订正 L0-01 的一处说法
9. **收束** —— 结论表 + 两个验收练习 + 下一课指路

## 核心结论

### 1. ★ 残差不是一条，而是一束

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
```

`(batch, seq, 4096)` → `(batch, seq, **4**, 4096)`，而且**层与层之间传的一直是 4 维**。
所以这个模型里没有"一条残差"这回事：教科书上的 `x = x + attn(norm(x))`，
在这里是"4 条流各自混合后再写回 4 条流"。

### 2. ★ 三步里，`pre` 用完即弃

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
        return post, comb, collapsed
```

**读取**（`view` + `input_norm` + `F.linear` → 24 个数）→ **混合**（`pre` / `post` / `comb` 三组权重）
→ **收拢**（`pre` 加权求和成一条，交给注意力或 MLP）。
返回的是 `post, comb, collapsed` —— **`pre` 不在返回值里**，因为它在收拢这一步就用完了。
写回不在这个函数里，它在解码层。

### 3. ★ 20 次 Sinkhorn = 双随机 = 凸组合

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
```

`softmax` 只保证**行和**是 1；补上**列和**的是这 4 行。双随机之后
`Mixed = In @ Comb + Out` 变成凸组合 —— **范数不增**，45 层叠起来才不会指数爆炸。
注意最后一次操作是列归一化，且分母带 `+ hc_eps`：实测行和 / 列和都停在 `0.999999`
（偏差 9.5e-7），**双随机是 eps 意义下的近似，不是精确的 1**。

### 4. ★ mHC 在解码层里出现两次

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        hidden_states = post.to(dtype).unsqueeze(-1) * hidden_states.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual
        )

        residual = hidden_states
        post, comb, hidden_states = self.ffn_hc(hidden_states)
        # Feed forward
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = post.to(dtype).unsqueeze(-1) * hidden_states.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual
        )
```

`attn_hc`（第 1315 行，**注意力之前**）和 `ffn_hc`（第 1342 行，**FFN 之前**）
各做一次"读取 + 混合"；写回则是同一段三行公式，出现两次（1337-1339 / 1346-1348）。
两个 HyperConnection 是**独立实例、不共享参数**：每层 2 × 393,243 = 786,486 个参数。

### 5. 出口是无权重平均（顺手订正 L0-01）

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        return hidden_streams.mean(dim=2)
```

`hc_head` 只有 `mean(dim=2)`：**0 个参数**、等权。
L0-01 第八幕的画面文字写的是"学习一组混合权重，把 4 条流加权求和" —— 按源码应订正为
**等权平均**（它的结论"4 → 1"不变）。真正学出来的混合在每层的 `attn_hc` / `ffn_hc` 里。

### 实测数字

| 量 | 实测值 | 含义 |
|---|---|---|
| `hc_mult` | 4 | 残差流条数 N |
| `hc_eps` | 1e-6 | Sinkhorn 分母上的数值下限 |
| `hc_sinkhorn_iters` | 20 | 列归一化 20 次 + 行归一化 19 次 |
| `fn` 的形状 | (24, 16384) | `(2+N)·N` × `N·hidden_size` |
| HC 参数量 | 393,243 / 个 | 每层 2 个 → 45 层共 35,391,870 |
| 主干 `hidden_states` | (batch, seq, **4**, 4096) | 4 维，层间一直如此 |
| `attn_hc` 返回 | (b,s,4) / (b,s,4,4) / (b,s,4096) | post / comb / collapsed |
| `hc_head` 参数 | 0 | `mean(dim=2)`，不是加权求和 |
| Sinkhorn 后行/列和 | 0.999999 | 偏差 9.5e-7，来自 `+ hc_eps` |

## 与后续课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| `hc_mult` / `hc_eps` / `hc_sinkhorn_iters` 住在哪个 config、怎么校验 | L1-02 |
| `attn_hc` / `ffn_hc` 的梯度怎么穿过 4 条流回到嵌入 | L3-06 |
| 4 维 `hidden_states` 对显存与 KV cache 的影响 | L5-01 ~ L5-03 |
| 子层的归一化为什么发生在收拢之后 | L3-02 / L3-03 |
| 45 层 × 2 个 mHC 在总参数量里占多少 | L7-03 |
| `hc_head` 的等权平均与 DeepSeek-V4 的差别 | L3-06 |

## 验收点

- [x] **（计划验收点 1）能手算 2 次 Sinkhorn 迭代** —— 初始矩阵与每一步的行和 / 列和：

```text
初始（softmax 之后，行和 = 1）：        列和 = 1.30 / 0.70 / 1.00 / 1.00   ← 要修的是列和
  [[0.70, 0.10, 0.10, 0.10],
   [0.40, 0.40, 0.10, 0.10],
   [0.10, 0.10, 0.70, 0.10],
   [0.10, 0.10, 0.10, 0.70]]

② 第 323 行列归一化（每列 ÷ 列和）：
   矩阵 = [[0.5385,0.1429,0.1000,0.1000],[0.3077,0.5714,0.1000,0.1000],
           [0.0769,0.1429,0.7000,0.1000],[0.0769,0.1429,0.1000,0.7000]]
   列和 = 1.000 / 1.000 / 1.000 / 1.000     ← 列修好了
   行和 = 0.881 / 1.079 / 1.020 / 1.020     ← 行坏了（偏差 0.1187）

③ 第 1 次（÷ 行和，再 ÷ 列和）：
   行和 = 0.979 / 1.004 / 1.008 / 1.008     偏差 0.0206
   列和 = 1.000 / 1.000 / 1.000 / 1.000

④ 第 2 次（÷ 行和，再 ÷ 列和）：
   行和 = 0.995 / 0.999 / 1.003 / 1.003     偏差 0.0046
   走满 20 轮：行和 = 列和 = 0.999999（停在 1/(1+hc_eps)，不是精确的 1）

口诀：除以列和 = dim -2（竖着加）；除以行和 = dim -1（横着加）；最后一次除的是列和。
```

- [x] **（计划验收点 2）能指出 mHC 在解码层的前后各出现一次** ——
  `attn_hc`（1315 行，注意力**前**）读取 4 条流并混合，产出 `post` / `comb` / `collapsed`，
  其中 `collapsed` 是注意力看到的 `(b, s, 4096)`；
  `ffn_hc`（1342 行，FFN **前**）对注意力写回后的流做同样的三步。
  写回**不在 mHC 内部**，而是由 1337-1339 / 1346-1348 的同一段公式完成：
  `新流 = post ⊙ 子层输出 + combᵀ · 旧流`。
- [x] **双随机约束的作用**（计划原话）：把 `comb` 变成凸组合的系数 ——
  列和 = 1 保证"新流是旧流的加权平均、范数不放大"，行和 = 1 保证"每条旧流的总贡献恰好一份"。
  副作用是 4 条流之间的耦合被限制住：它们可以互相搬运信息，但不能互相放大。
- [x] 保真门禁：12 个引用块全部逐字来自 `modeling_glm5_next.py` 且位置连续
- [x] 覆盖度门禁：1 个源文件被声明（`models/glm5_next/modeling_glm5_next.py`，属覆盖域 231 个文件之一）
- [x] 参数门禁：27 处参数 / CSS 变量引用全部真实存在
- [x] 语法检查：`node --check` 通过，9 幕字段完整
- [x] 渲染门禁：无 JS 错误、无布局溢出、交互可用（两种分辨率）
- 自检：能否不看笔记说出"为什么 4 条流会分叉，分叉的唯一入口是什么"？
  （答：第 0 层写回时 `post` 是逐流的门控；若 4 条输入流相同，
  `新流_i = post_i · o + (Σⱼ comb[j,i]) · x`，唯一与 i 有关的量就是 `post_i`。）

**一句话总结**：

> 主干里 `hidden_states` 一直是 `(batch, seq, 4, 4096)`；每个子层前后，
> `Glm5NextTextHyperConnection` 用当前 4 条流算出 `pre/post/comb`，
> 把流收拢成 1 条交给注意力或 MLP，再用 **`新流 = post ⊙ 子层输出 + combᵀ · 旧流`** 写回 4 条；
> `comb` 被 20 次 Sinkhorn 迭代约束成**双随机矩阵**（凸组合，范数不爆炸），
> 出口的 `hc_head` 只是**无权重平均**（0 参数）。
