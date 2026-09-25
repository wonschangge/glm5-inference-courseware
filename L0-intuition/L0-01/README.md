# L0-01 · 一次 `generate()` 里发生了什么 — 前向全景

> 层：**预备篇 · 直觉** ｜ 优先级：P0 ｜ 前置课：无（这是第一课）

## 学习目标

看完这一课，你应该能：

1. **不看笔记复述一次前向的 6 个阶段**（对应计划的验收点 1）；
2. **说出 45 层里有多少层跑线性注意力**，以及这个数字是从 config 的哪一行来的（验收点 2）；
3. 解释为什么 `hidden_states` 从一开始就是 **4 维**的，以及多出来的那一维是什么。

## 覆盖的源文件（1 个）

| 文件 | 行数 | 引用块 |
|---|---|---|
| `models/glm5_next/modeling_glm5_next.py` | 2444 | 9 |

> 本课引用了 `_data/recon/probe1.py`、`probe2.py` 的**实测输出**（写在 `source.md` 的
> `text` 块里）。**这些是脚本，不计入覆盖率** —— 覆盖域只统计 `src/transformers/**/*.py`。

## 场景（9 幕）

1. **一次前向，六个阶段** —— 全景流程线 + 四个关键数字（4096 / 45 / 4 / 154880）
2. **入口二选一** —— `input_ids` 与 `inputs_embeds` 的异或，以及多模态为什么走后者
3. **position_ids 从哪来** —— prefill vs decode 的位置编号推演
4. **★ 掩码按层取** —— 45 层排班表，`idx % 4 == 3` 是 MLA 层
5. **★ hidden 是 4 维的** —— mHC 展开，全文件最容易被略过的一行
6. **45 层循环** —— `topk_indices` 在层间传递，`shared` 索引器省下了什么
7. **层内二选一** —— 注意力轴 × MLP 轴的 2×2 组合矩阵
8. **出口收拢** —— `hc_head` 把 4 条流合成 1 条（**等权平均，0 参数**）
9. **收束** —— 六阶段总表 + 实测数字 + 练习

## 核心结论

### 1. prefill 与 decode 是同一个函数

`DynamicCache` 是在 `forward` **内部**创建的，不是在外面：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
```

所以"第一次前向"和"后续每一次前向"走的是同一条代码路径，区别只在
`past_key_values` 是不是空的、以及 `inputs_embeds.shape[1]` 是 `seq` 还是 `1`。

### 2. ★ 这个模型没有「一条残差」

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
```

`(batch, seq, 4096)` → `(batch, seq, **4**, 4096)`。教科书上的
`x = x + attn(norm(x))` 在这里是"4 条流各自混合后再写回 4 条流"。

### 3. ★ 「按层排班」是混合架构的具体含义

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        self.block_type = config.layer_types[layer_idx]
        self.hidden_size = config.hidden_size
        self.self_attn = (
            Glm5NextTextLinearAttention(config, layer_idx)
            if self.block_type == "linear_attention"
            else Glm5NextTextAttention(config, layer_idx)
        )
```

| 组合 | 层数 | 层号 |
|---|---|---|
| KDA × 稠密 MLP | 3 | 0, 1, 2 |
| KDA × MoE | 31 | 其余 `idx % 4 != 3` 的层 |
| MLA+DSA × MoE | 11 | `idx % 4 == 3` |

四种组合里 `MLA × 稠密` 这一格是空的。

### 实测数字

| 量 | 实测值 | 含义 |
|---|---|---|
| `num_hidden_layers` | 45 | 主干层数 |
| `layer_types` | 34 + 11 | KDA : MLA/DSA |
| `mlp_layer_types` | 3 + 42 | 稠密 : MoE |
| `hc_mult` | 4 | 残差流条数 |
| `n_routed_experts` | 288 | 每层专家数 |
| `num_experts_per_tok` | 8 | 每 token 激活 |
| `vocab_size` | 154880 | 词表 |

## 与后续课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| `layer_types` 的 34:11 怎么算出来 | L0-03 |
| mHC 的 `post` / `comb` 具体怎么来 | L3-06 |
| KDA 层内部做了什么 | L4-02 ~ L4-05 |
| DSA 的 2048 个位置怎么选 | L4-06 |
| `DynamicCache` 到底存了什么 | L5-01 ~ L5-03 |
| `lm_head` 之后发生了什么 | L6-01 ~ L6-03 |

## 验收点

- [x] 保真门禁：9 个引用块全部逐字来自 `modeling_glm5_next.py`
- [x] 覆盖度门禁：1 个源文件被声明（属覆盖域 231 个文件之一）
- [x] 参数门禁：无非法参数
- [x] 语法检查：`node --check` 通过，9 幕字段完整
- [x] 渲染门禁：无 JS 错误、无布局溢出、交互可用（两种分辨率）
- 自检：能对着 `#visual` 里的六阶段图，逐一说出它对应 `forward` 的哪几行

**一句话总结**：

> 一次前向 = 嵌入 →（展开成 4 条流）→ 45 次"按 config 排班"的子层 → 收拢成 1 条流 → 归一化；
> **decode 与 prefill 走的是同一个函数**，区别只在 `past_key_values` 是不是空的。
