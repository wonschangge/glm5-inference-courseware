# L3-08 · 解码层装配：三个子块的顺序

> 层：**第 3 层 · 文本主干** ｜ 优先级：P0 ｜ 前置课：`L3-07`（遗忘门与门控范数）

## 学习目标

看完这一课，你应该能：

1. **默写一层的执行顺序**（6 次子模块调用 + 2 次写回），对应验收点 1；
2. **说出 residual 在 mHC 下与标准 Transformer 的差异**（条数 / 合并方式 / 幅度三处），对应验收点 2；
3. 说清推理时 `GradientCheckpointingLayer` 为什么不产生任何开销，以及
   `MoeModelOutputWithPast` 的三条数据类约定。

## 覆盖的源文件（3 个）

| 文件 | 行数（实测 `wc -l`） | 引用块 |
|---|---|---|
| `models/glm5_next/modeling_glm5_next.py` | 2444 | 9 |
| `modeling_layers.py` | 682 | 2 |
| `modeling_outputs.py` | 1662 | 2 |

> ★ **口径订正**：作业书写的是 2445 / 683 / 1663。实测是 **2444 / 682 / 1662** ——
> 作业书用 `len(text.split("\n"))` 口径，把文件末尾换行后的空串也算了一行。
>
> 本课引用块由 `_data/build_l308.py` 从源文件按行号**切片**生成（不是手抄），
> 共 **14 个**。实测脚本 `_data/recon/probe_l308.py`、`probe_l308b.py`
> 是**脚本，不计入覆盖率**；`PreTrainedConfig.get_mtp_config()` 属于
> `configuration_utils.py`，也不在覆盖域内。

## 场景（9 幕）

1. **一层 = 三个子块，六次调用** —— 全景：`attn_hc → input_layernorm → self_attn → ffn_hc → post_attention_layernorm → mlp`（forward hook 实测）
2. **mHC 在 norm 之前** —— 顺序 ①：`post / comb / collapsed` 一起算出来，只有收拢后的那 1 条进 RMSNorm
3. **★ 写回不是加法** —— 顺序 ②：`post ⊙ 子层输出 + combᵀ · 旧流`，与教科书 residual 的逐条对照
4. **pre / post / comb：一次线性切三段** —— 24 个数怎么切、三种激活各为什么；实测 `post` 的 mean 与范围
5. **★ comb 必须双随机：Sinkhorn 手算** —— 2x2 手算 + 真实 4x4 的实测偏差（**它只是近似双随机**）
6. **FFN 子块：同一套模板再缝一次** —— 差异只有三处名字；层返回 `(hidden_states, topk_indices)`
7. **MoE 子块内部** —— 路由 → 稀疏专家 → 共享专家；`swiglu_limit = 10.0` 的手算与被裁两次
8. **推理时 GradientCheckpointingLayer 什么都不做** —— 分支条件、`_gradient_checkpointing_func` 不存在、实测 0 次调用
9. **收束：把一层默写出来** —— 四拍跑两遍 + `modeling_outputs` 的三条约定 + 两道练习

## 核心结论

### 1. ★ 一层的装配模板只有一套，跑两遍

@@B:glm5:1312:1318@@

第二遍把 `attn_hc / input_layernorm / self_attn` 换成
`ffn_hc / post_attention_layernorm / mlp`，写回那两行一字不改。

### 2. ★ residual 在 mHC 下不是一条加法

@@B:glm5:1337:1339@@

| | 教科书 residual | mHC（本课） |
|---|---|---|
| 残差条数 | 1 条 | **4 条**（`hc_mult`） |
| 合并方式 | `x + f(norm(x))` | `post ⊙ f + combᵀ · x` |
| 写回幅度 | 恒为 1 | `post ∈ (0, 2)`，实测 mean **1.004**、范围 **[0.005, 1.968]** |

### 3. ★ `comb` 只是**近似**双随机（实测订正）

@@B:glm5:319:326@@

`hc_sinkhorn_iters = 20` 实际做 **1 + 19 x 2 = 39 次**归一化。真实 4x4 实测：
**列和偏差 1.07e-06**（最后一步就是列归一，被 `hc_eps` 钉死）、
**行和偏差 1.77e-02**（还差 1.8%）。源码注释的意图是"约束范数不放大"，
实现上是迭代到足够好，不是精确投影。

### 4. 推理时基类直接退化成 `nn.Module`

@@B:layers:76:79@@

分支条件是 `self.gradient_checkpointing and self.training`。实测：类属性为 `False`；
刚构造出的层**连 `_gradient_checkpointing_func` 属性都不存在**；
`eval() + no_grad()` 下调用计数 **0 次**，而 `train() + gradient_checkpointing=True` 时是 **1 次**。

### 5. 返回值：数据类的三条约定

@@B:outputs:345:349@@

① 每个字段都有默认值 `None`；② 字段顺序 = 位置参数顺序（实测 `o[0] is o.last_hidden_state` 为 `True`）；
③ `None` 字段在 `keys()` / `to_tuple()` 里被丢掉 —— 所以返回值的长度会随开关变化。

## 与后续课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| mHC 的 `fn` / `base` / `scale` 怎么初始化、整体怎么推 | L3-06 |
| 注意力是 KDA 还是 MLA+DSA | L4-01 ~ L4-06 |
| 路由的 `n_group` / `topk_group` / `e_score_correction_bias` | L3-05 |
| `swiglu_limit` 在视觉塔里的同款裁剪 | L2-07 |
| 主干循环与 `topk_indices` 的跨层传递 | L0-01 |
| 输出数据类的完整字段表、与张量并行的关系 | L7-03 |
| KV cache 怎么累积（本课确认它不会被 checkpoint 干扰） | L5-01 ~ L5-03 |

## 验收点

- [x] 保真门禁：14 个引用块全部逐字来自三个源文件（由 `_data/build_l308.py` 切片生成）
- [x] 覆盖度门禁：3 个源文件被声明（与本课 `plan.py` 的 `files` 一致）
- [x] 参数门禁：无非法参数
- [x] 语法检查：`node --check` 通过，9 幕字段完整
- [x] 渲染门禁：无 JS 错误、无布局溢出、交互可用（两种分辨率）
- 自检 1：能默写 **attn_hc → input_layernorm → self_attn → ffn_hc → post_attention_layernorm → mlp**
- 自检 2：能说出 residual 的三处差异（4 条流 / `post ⊙ f + combᵀ · x` / `post ∈ (0,2)`）

**一句话总结**：

> 一层 = `mHC → norm → 子层 → 写回` 跑两遍（子层 = 注意力 / MoE，二选一）；
> **residual 在 mHC 下不是一条加法，而是"`post` 门控 + 4 条流的 `comb` 混合"**；
> 推理时 `GradientCheckpointingLayer` 直接退化成普通 `nn.Module.__call__`。
