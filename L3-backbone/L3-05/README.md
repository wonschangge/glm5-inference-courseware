# L3-05 · MoE 装配与共享专家

> 层：**第 3 层 · 文本主干** ｜ 优先级：P0 ｜ 前置课：`L3-04`

## 学习目标

看完这一课，你应该能：

1. **画出 MoE 前向的数据流**：`hidden_states → gate → top-8 → experts → (+ shared_experts) → 输出`，
   以及每一步的形状（验收点 1）；
2. **解释共享专家为什么能稳住训练**：它是无条件的加法旁路，不吃 `topk_weights`，
   给每个 token 一条不依赖离散选择的梯度通路（验收点 2）；
3. 说清 `output_router_logits` 在推理时**改变了什么**（实测答案：**什么数值都不改**，
   只决定要不要收集 `router_logits` 去算辅助损失）。

## 覆盖的源文件（1 个）

| 文件 | 行数 | 引用块 |
|---|---|---|
| `models/glm5_next/modeling_glm5_next.py` | 2444 | 9 |

> 本课引用了 `_recon/probe_moe.py`、`probe_moe2.py`、`probe_moe3.py` 的**实测输出**
> （写在 `source.md` 与 `lesson.js` 的 `text` 块里）。**这些是脚本，不计入覆盖率** ——
> 覆盖域只统计 `src/transformers/**/*.py`。

## 场景（9 幕）

1. **三条支路** —— `Glm5NextTextMoE` 的三个子模块分工 + 四个实测数字
2. **前向 8 行** —— 逐行追踪形状变化，点出 L202 的 `residuals` 是理解共享专家的钥匙
3. **分支条件** —— `mlp_layer_types[i] == "sparse"` 在**构造期**二选一；稠密 3 层 vs 稀疏 42 层
4. **数据流与形状** —— `4096 → 288 → 8 → 4096` 全过程 + 每 token 的专家参数账
5. **★ 共享专家** —— 三个消融实验：路由专家置零 / 共享专家置零 / 门置零
6. **★ 被丢掉的返回值** —— `self.gate()` 的三元组，`_` 与 `_can_record_outputs` 的 `index=0`
7. **★ 开关** —— `output_router_logits` 默认 False；实测两次前向逐元素相同；显存代价
8. **★ 辅助损失** —— 均匀时 = `top_k`，全挤一个时 = `num_experts`，两端都可手推
9. **收束** —— 一页总表 + 三句话 + 练习 + 下一课指路

## 核心结论

### 1. ★ 合流是一行无权加法

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        _, topk_weights, topk_indices = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states
```

`residuals` 在任何 reshape 之前留了一份**原样**的输入，第 7 行把它喂给共享专家。
实测 `torch.allclose(moe(x), experts(x) + shared_experts(x)) == True`，
且**没有任何系数**。

### 2. ★ 共享专家 = 一个完整尺寸的 `Glm5NextTextMLP`

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        self.shared_experts = Glm5NextTextMLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )
```

中间维被显式传成 `2048 × 1 = 2048`（默认 `intermediate_size` 是 `12288`）。
所以它不是「小专家」，而是**与路由专家等宽**（实测比值 1.000）、但**每层无条件执行**的一路。

### 3. ★ `router_logits` 算了却没人要

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        _, topk_weights, topk_indices = self.gate(hidden_states)
```

下划线不是省略：hook 实测 `Glm5NextTextTopkRouter` **每次前向都被调用且只调用 1 次**
（与 `output_router_logits` 无关）。它被丢弃是因为前向不需要，
而反向需要的路径由 `_can_record_outputs` 里 `OutputRecorder(..., index=0)` 单独登记。

### 实测数字

| 量 | 实测值 | 含义 |
|---|---|---|
| `mlp_layer_types` | 3 dense + 42 sparse | MoE 只在后 42 层 |
| 单个路由专家 | 25,165,824 | `2·I·H + H·I` |
| 288 个路由专家 | 7,247,757,312 | 每层驻留 |
| 共享专家（`n_shared_experts=1`） | 25,165,824 | 与单专家 1.000 倍 |
| 每 token 激活专家 | 9 / 289 | 3.11% |
| `topk_weights` 每行和 | 2.5 | `= routed_scaling_factor` |
| `output_router_logits` 默认 | `False` | 打开后 45 层 × 1M token ≈ 48.3 GiB |
| 均匀时 `aux_loss` | 2.0 | `= top_k` |
| 全挤一个时 `aux_loss` | 8.0 | `= num_experts` |

## 与后续课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| `mlp_layer_types` 这个 3 + 42 的序列从哪来 | L0-02 |
| 两条排班轴（注意力 / FFN）的关系 | L0-03 |
| `num_local_experts` 为什么写在 `attribute_map` 里 | L1-04 |
| SwiGLU 与 `swiglu_limit` 截断 | L3-02 |
| `sigmoid` 分数、分组筛选、`e_score_correction_bias` | L3-03 |
| 专家权重为什么按 `(E, 2I, H)` 打包、grouped GEMM | L3-04 |
| mHC 的 `post` / `comb` 与残差写回 | L3-06 |
| `router_logits` 这份显存在推理时值多少字节 | L5 全层 |

## 验收点

- [x] 能画出 MoE 前向的数据流（第 4 幕的流程图 + 第 9 幕的总表）
- [x] 能解释共享专家为什么能稳住训练（第 5 幕：三个消融实验 + 无条件执行）
- [x] 保真门禁：9 个引用块全部逐字来自 `modeling_glm5_next.py`
- [x] 覆盖度门禁：1 个源文件被声明（属覆盖域 231 个文件之一）
- [x] 参数门禁：无非法参数
- [x] 语法检查：`node --check` 通过，9 幕字段完整
- [x] 渲染门禁：无 JS 错误、无布局溢出、交互可用（两种分辨率）
- 自检：对着第 4 幕的数据流图，逐一说出它对应 `Glm5NextTextMoE.forward` 的哪几行

**一句话总结**：

> 稀疏层 = **一路被 8 选 1 选出来的路由专家** + **一路永远在线的共享专家**，
> 在 `hidden_states + self.shared_experts(residuals)` 这一行**无权相加**；
> `router_logits` 算了但前向不要它 —— 它只在 `output_router_logits=True` 时被收集起来，
> 去算一个默认关闭、且**完全不影响前向数值**的负载均衡损失。
