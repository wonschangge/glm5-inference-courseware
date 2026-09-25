# L0-04 · MoE 经济学：288 选 8 — 稀疏层与共享专家

> 层：**预备篇 · 直觉** ｜ 优先级：P0 ｜ 前置课：L0-03

## 学习目标

看完这一课，你应该能：

1. **算出稀疏层与稠密层的参数量比**（48.17×），并解释「约 3% 激活」是怎么来的（对应验收点 1）；
2. **说出共享专家存在的理由**，以及为什么它的宽度恰好等于一个路由专家（验收点 2）；
3. 解释 `routed_scaling_factor = 2.5` 与 `norm_topk_prob` 各自解决了什么问题，
   并说出 `n_group` / `topk_group` 在默认配置下的状态。

## 覆盖的源文件（1 个）

| 文件 | 行数 | 引用块 |
|---|---|---|
| `models/glm5_next/modeling_glm5_next.py` | 2444 | 12 |

> 本课引用了一批**实测输出**（写在 `source.md` 的 `text` 块里）：
> 用 `.venv/bin/python` 跑 `Glm5NextTextConfig()`、在 meta 设备上把层真正建出来数参数、
> 以及用真实的 `Glm5NextTextTopkRouter` 跑小配置。**这些是脚本而不是源文件，
> 不计入覆盖率** —— 覆盖域只统计 `src/transformers/**` 下的源文件。

## 场景（9 幕）

1. **一个稀疏层有什么** —— 路由器 + 288 个专家的池子 + 1 个共享专家，以及 313.79B / 17.84B 两个数
2. **288 个盒子里亮 8 个** —— 24 × 12 的专家网格逐个点亮，看 8 / 288 = 2.78%
3. **参数预算** —— 同一横轴上的 151.0M / 7274.1M / 227.7M，读出 48.17× 与 1.51×
4. **★ 共享专家** —— 两条路径汇到一个加号：竞争的 8 个 + 不竞争的 1 个
5. **先打分** —— `sigmoid` 而不是 softmax，288 个分数互相独立
6. **★ 分组路由** —— 两级筛选（组 → 组内），以及默认 `1 / 1` 时的退化
7. **归一化 + 2.5 倍** —— 一串真实数字：`[0.736, 0.606] → [0.548, 0.452] → [1.371, 1.129]`
8. **训练 vs 推理** —— `load_balancing_loss_func` 的两层开关，以及它静默返回 0 的坑
9. **收束** —— 一张实测表 + 练习 + 下一课指路

## 核心结论

### 1. ★ 参数按 288 份买，账单按 9 份付

一个稀疏层就是三样东西：专家池、路由器、共享专家。

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
    def __init__(self, config: Glm5NextConfig):
        super().__init__()
        self.config = config
        self.experts = Glm5NextTextExperts(config)
        self.gate = Glm5NextTextTopkRouter(config)
        self.shared_experts = Glm5NextTextMLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )
```

参数量（meta 设备实测）：

| 量 | 参数 | 算式 |
|---|---|---|
| 稠密 MLP 层 | 150,994,944（151.0M） | 3 × 4096 × 12288 |
| MoE 层总计 | 7,274,102,784（7274.1M） | 288 × 25.17M + 25.17M + 1.18M |
| 每 token 激活 | 227,672,064（227.7M） | 8 × 25.17M + 25.17M + 1.18M |

- **参数比 7274.1 / 151.0 = 48.17×**；
- **算力比 227.7 / 151.0 = 1.51×**。

### 2. ★ 共享专家：不参与竞争的快车道

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

`residuals` 在函数最前面就被留了一份，喂给 `self.shared_experts` —— 它吃的是
**这一层的原始输入**，最后用**加法**叠在路由专家的加权和上。

理由：路由专家在"竞争上岗"中被逼向专精，但语言里总有一部分变换是每个 token 都要做的
（高频词形、常见句法）。没有共享专家，这部分公共知识要么被 8 个专家各自重复学一遍，
要么谁都学不好。共享专家是一条**不参与竞争**的通道，承接公共知识，让路由专家更专精。
它的宽度是 `moe_intermediate_size * n_shared_experts` = 2048 × 1，恰好**一个专家宽**。

### 3. ★ 分组路由：默认关闭的能力

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        group_scores = (
            scores_for_choice.view(-1, self.num_group, self.num_experts // self.num_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
```

先把 288 个专家切成 `n_group` 组，每组取**分数最高的 2 个之和**当组分数，再选出
`topk_group` 个组，落选组整组被 `-inf` 屏蔽。动机是**分布式下的通信与负载**：
288 选 8 是全对全比较，在专家并行下就是一轮 all-to-all；先收敛到组粒度，
通信模式更可预测。

注意默认值是 **`n_group = 1` / `topk_group = 1`** —— 只有一组，整组必然入选，
这段代码在默认配置下是**空操作**。

### 实测数字

| 量 | 实测值 | 含义 |
|---|---|---|
| `n_routed_experts` | 288 | 每层路由专家数 |
| `num_experts_per_tok` | 8 | 每 token 选中 |
| `n_shared_experts` | 1 | 每 token 都过 |
| `n_group` / `topk_group` | 1 / 1 | 分组路由（默认退化） |
| `norm_topk_prob` | True | 权重和恒等于 1，再乘 2.5 |
| `routed_scaling_factor` | 2.5 | `topk_weights` 之和 = 2.5 |
| 每 token 激活比 | 3.125% | (8 + 1) / 288 |
| 模型总参 / 激活 | 313.79 B / 17.84 B | 对外口径「320B / 18B」 |
| `output_router_logits` | False | 推理默认不收集 router_logits |

## 与后续课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| 为什么前 3 层是稠密 MLP | L0-02 |
| 共享专家的输出加到哪条残差流 | L0-05 |
| `router_logits` 的收集机制（`_can_record_outputs`） | L7 |
| `grouped_gemm` 与专家并行计划怎么落地 | L7-03 |
| 专家的初始化与 fp32 保留名单 | L3-07 |
| 训练循环里怎么用这个辅助损失 | L6 |

## 验收点

- [x] 保真门禁：12 个引用块全部逐字来自 `modeling_glm5_next.py`
- [x] 覆盖度门禁：1 个源文件被声明（属覆盖域 231 个文件之一）
- [x] 参数门禁：无非法参数
- [x] 语法检查：`node --check` 通过，9 幕字段完整
- [x] 渲染门禁：无 JS 错误、无布局溢出、交互可用（两种分辨率）
- 自检 1：**能算出稀疏层与稠密层的参数量比（48.17×），并解释「3% 激活」是怎么来的**
  —— 8 个路由专家 + 1 个共享专家，(8 + 1) / 288 = 3.125%
- 自检 2：**能说出共享专家存在的理由** —— 承接公共知识，让路由专家更专精；
  它不参与竞争，所以每个 token 都过它

**一句话总结**：

> 稀疏层 = 288 个路由专家（竞争上岗，每 token 只叫醒 8 个）+ 1 个共享专家（永远在线）；
> 参数按 288 份买（48.17× 稠密层），账单按 9 份付（1.51× 稠密层），
> 而 `load_balancing_loss_func` 只在训练时给路由「排班」，推理路径上不存在。
