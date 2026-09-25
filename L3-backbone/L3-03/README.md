# L3-03 · 路由器：288 个分数如何变成 8 条路径

> 层：**第 3 层 · 文本主干** ｜ 优先级：P0 ｜ 前置课：`L3-02`

## 学习目标

看完这一课，你应该能：

1. **手算一次 top-8 选择**：给定 8 个专家、2 个组、`topk_group=1`、`top_k=2`，从
   logit 一路算到最终权重（对应验收点 1）；
2. **解释分组路由在分布式下的好处**：为什么「8 个专家落在 ≤2 个组」能把通信扇出
   从 ≤8 张卡压到 ≤2 张卡，以及这个能力的代价是什么（验收点 2）；
3. 说清 `norm_topk_prob` 与 `routed_scaling_factor` 各自做了什么值变换 ——
   为什么 **8 条路径的权重和恒等于 2.5**。

## 覆盖的源文件（1 个）

| 文件 | 行数 | 引用块 |
|---|---|---|
| `models/glm5_next/modeling_glm5_next.py` | 2444 | 11 |

> 本课引用了 `_data/recon/probe_l303.py` 的**实测输出**（写在 `source.md` 的 `text`
> 块里），以及 `configuration_glm5_next.py` 的 `base_model_ep_plan`（同样用 `text`
> 块标出）。**这些不计入本课的覆盖率声明** —— coverage 只声明
> `models/glm5_next/modeling_glm5_next.py`。

## 场景（9 幕）

1. **288 个分数，8 条路径** —— 全景流程线 + 四个关键数字（288 / 8 / 2.5 / **1**）
2. **gate 就是一层 `Linear(4096 → 288)`** —— `self.weight` 的形状与强制 fp32
3. **★ 不是 softmax，是 sigmoid** —— 订正作业书，并用 Δ 值证明「改一个专家，
   sigmoid 只动一个、softmax 动全部」
4. **第一级筛选：按组报分** —— 每组取前二之和，`(N, 288) → (N, n_group)`；
   并给出**默认 `n_group=1` 时这一级是 no-op** 的实测
5. **第二级筛选：置 `-inf`** —— 288 格网格：216 个被置 `-inf`，候选池 288 → 72
6. **★ 选择用含偏置的分，权重用无偏的分** —— `scores_for_choice` vs `scores`
7. **★ 手算 top-8** —— 8 专家 / 2 组的最小例子，六步走完到 `[1.268938, 1.231063]`
8. **`norm_topk_prob` 与 `routed_scaling_factor`** —— 同一组 logits，权重和 2.5 vs 18.699324
9. **收束** —— 七步总表 + EP 网格（通信扇出 ≤8 → ≤2）+ 练习

## 核心结论

### 1. ★ 打分函数是 sigmoid，不是 softmax

作业书把要点写成「gate 线性层 + softmax」。**实测不成立**：整个
`Glm5NextTextTopkRouter` 里没有一次 softmax。

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        scores = router_logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias
```

同一个专家被改动时，sigmoid 下其余分数 Δ 全为 0；softmax 下全部非零
（实测 Δ = +0.461925 / −0.267574 / −0.059704 / −0.098435 / −0.036212）。
选择与权重必须能解耦，这是 sigmoid 的理由。

### 2. ★ 选择与计分取自两个不同的变量

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_indices)
```

实测：给 e164 加 −5、给 e7 加 +5 之后 `topk_indices` 变了
（`[164, …]` → `[215, 26, 114, 147, 7, 287, 128, 284]`），但
「权重 == `sigmoid(logits).gather(idx)` 归一化 ×2.5」为 `True`，
「权重 == `scores_for_choice.gather(idx)` 归一化 ×2.5」为 `False`。
**偏置是选拔委员，不是计分员。**

### 3. ★ 权重和恒等于 `routed_scaling_factor`

```text
norm_topk_prob=True    权重和 = 2.500000    （实测，与输入无关）
norm_topk_prob=False   权重和 = 18.699324   （8 个 sigmoid 值直接相加）
```

`True` 时先在**选中的 8 个**里归一化（和 = 1），再乘 2.5 ——
下游专家输出的尺度因此可预测。

### 实测数字（`probe_l303.py`）

| 量 | 实测值 | 含义 |
|---|---|---|
| `n_routed_experts` | 288 | 候选专家（`num_local_experts` 由 attribute_map 指向它） |
| `num_experts_per_tok` | 8 | 每 token 的路径数 |
| `n_group` | **1** | ★ 默认只有 1 个组 → 第一级筛选空转 |
| `topk_group` | **1** | ★ 默认只选 1 个组 |
| `routed_scaling_factor` | 2.5 | 权重和 |
| `norm_topk_prob` | True | 在选中的 8 个里归一化 |
| 分组实测（`n_group=8, topk_group=2`） | 288 → 72 | 216 个专家被置 `-inf` |

## 与后续课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| 288 个专家的权重按什么形状存 | L3-04 |
| `gate_up_proj` 怎么打包、grouped GEMM 怎么用 | L3-04 |
| 共享专家为什么能稳住训练 | L3-05 |
| `output_router_logits` 在推理时的影响 | L3-05 |
| KDA 层里的 `A_log` / `conv1d` 为什么要留 fp32 | L4-02 ~ L4-05 |
| DSA 的 `topk` 与 MoE 的 `topk` 有何异同 | L4-06 |
| `DynamicCache` 在 MoE 层里怎么处理 | L5-01 ~ L5-03 |

## 验收点

- [x] 保真门禁：11 个引用块全部逐字来自 `modeling_glm5_next.py`，且位置连续
- [x] 覆盖度门禁：1 个源文件被声明（属覆盖域 231 个文件之一）
- [x] 参数门禁：无非法参数
- [x] 语法检查：`node --check` 通过，9 幕字段完整
- [x] 渲染门禁：无 JS 错误、无布局溢出、交互可用（两种分辨率）
- 自检 1：能对着第 7 幕的例子，从 `logit=[1.0,2.0,-1.0,1.5,3.0,2.5,2.0,-2.0]` 手算出
  `topk_indices=[4,5]` 与 `topk_weights=[1.268938, 1.231063]`
- 自检 2：能说出「限定 `topk_group=2` 后 8 个专家最多落在 2 张卡上」，
  以及为什么默认配置下这句话目前不成立

**一句话总结**：

> 288 个分数 = 一次 `Linear(4096→288)` 再 `sigmoid`（**不是 softmax**）；
> 两级筛选先按组淘汰再取 `top-8`，**但默认 `n_group=1` 时第一级是空的**；
> 权重从无偏的 `scores` 里 `gather`，归一化后乘 2.5 ——
> **所以 8 条路径的权重和永远等于 `routed_scaling_factor`**。
