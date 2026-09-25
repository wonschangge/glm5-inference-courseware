<!-- glm5-coverage
models/glm5_next/modeling_glm5_next.py
-->

# L0-04 · MoE 经济学：288 选 8 — 源文件

**这一课只回答一个问题：一个"总参 313.8B"的模型，凭什么每个 token 只花 17.84B 的钱？**

视角：一个 token 走进 GLM-5 的稀疏 MLP 层，从 288 个路由专家里被分到 8 个，再额外过 1 个共享专家。

| 文件 | 行数 | 本课引用块 |
|---|---|---|
| `models/glm5_next/modeling_glm5_next.py` | 2444 | 12 |

**实测的 MoE 规模**（用 `.venv/bin/python` 在本机跑 `Glm5NextTextConfig()`，并在 meta 设备上把层真正建出来数参数得到，不是估算）：

```text
Glm5NextTextConfig() 默认值
    n_routed_experts      = 288       每个稀疏层的路由专家数
    num_experts_per_tok   = 8         每个 token 选中的专家数
    n_shared_experts      = 1         每个 token 都过的共享专家数
    n_group / topk_group  = 1 / 1     分组路由（默认退化，见第七节）
    norm_topk_prob        = True
    routed_scaling_factor = 2.5
    hidden_size           = 4096
    moe_intermediate_size = 2048      单个专家的中间维
    intermediate_size     = 12288     稠密层的中间维
    mlp_layer_types       : 3 层 dense + 42 层 sparse —— 45 层里 42 层是 MoE
    attribute_map         = {"num_local_experts": "n_routed_experts"}

单层参数量（meta 设备实测）
    稠密 MLP 层            150,994,944      151.0M    = 3 * 4096 * 12288
    MoE 层总计           7,274,102,784    7274.1M
      288 个路由专家     7,247,757,312    7247.8M    = 288 * 25,165,824
      1 个共享专家          25,165,824      25.2M    = 3 * 4096 * 2048
      路由 gate              1,179,648       1.2M    = 288 * 4096
    每 token 激活           227,672,064     227.7M    = 8 * 25.17M + 25.17M + 1.18M
    激活比                      3.13%                 (8 + 1) / 288 = 3.125%
    稀疏层 / 稠密层 参数比       48.17x                7274.1M / 151.0M
    稀疏层 / 稠密层 算力比        1.51x                227.7M / 151.0M

整个模型（meta 设备实测，Glm5NextForConditionalGeneration 默认配置）
    总参数              313,786,889,214    313.79B
    每 token 激活         17,842,343,902     17.84B    （占总量 5.68%）
    对外口径「320B 总参 / 18B 激活」说的就是这两个数

路由权重实测（用真实的 Glm5NextTextTopkRouter，8 专家 / top-2 的小配置）
    sigmoid 原始分数  [0.736, 0.606]        和 = 1.342
    norm_topk_prob    [0.548, 0.452]        和 = 1.000
    x 2.5             [1.371, 1.129]        和 = 2.500   <- 每 token 的权重和恒等于 routed_scaling_factor
```

> 上面这段是**示意性的汇总**，用 `text` 块标出 —— 它不是源文件的逐字引用，
> 因此不参与保真校验。下面每一段 `python` 块都是逐字引用。

---

## 一、★ 参数量按 288 份买，账单按 9 份付

先看一个稀疏层是由哪几块拼出来的。这三行是整个 MoE 经济学的账本：

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

**读法**：
- `self.experts` 是**路由专家池**（288 个，参数打包成 3D 张量），`self.gate` 是那个"288 选 8"
  的路由器，`self.shared_experts` 是**唯一一个每个 token 都必须过的专家**。
- 共享专家用的是 `Glm5NextTextMLP`，而不是 `Glm5NextTextExperts` —— 这是设计上的明示：
  **它不属于那 288 个候选，不参与竞争，也不需要被路由**。它的宽度是
  `moe_intermediate_size * n_shared_experts` = 2048 × 1，恰好等于**一个**路由专家的宽度。
- 于是这一层的账是：**288 份容量 + 1 份公共容量**；而每个 token 只为
  **8 + 1 = 9 份**付钱。9 / 288 = 3.125%。
- 为什么值得单独记住：MoE 的"省"从来不是省参数，而是**省每个 token 的算力**。
  权重还是要全部装进显存（313.79B），只是绝大多数在某一次前向里不参与计算。

---

## 二、专家不是"更大的 MLP"，是 288 个并排的小 MLP

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.swiglu_limit = config.swiglu_limit
```

**读法**：
- 注意 `self.num_experts = config.num_local_experts`。`Glm5NextTextConfig` 里
  `attribute_map = {"num_local_experts": "n_routed_experts"}` —— 这是 HF 侧统一字段名
  与 GLM-5 侧命名的桥：**模型代码里永远看不到 `n_routed_experts`，它只在配置里存在**。
- 两个参数是** 3D 张量**，不是 288 个 `nn.Linear` 的列表：
  `gate_up_proj` 形状 `(288, 2×2048, 4096)`，`down_proj` 形状 `(288, 4096, 2048)`。
  把 gate 和 up 合并成一个矩阵（`2 * intermediate_dim` 那一维）是推理效率的决定性细节。
- 单个专家 = `2×2048×4096 + 4096×2048` = **25,165,824** 参数（25.17M）；
  288 个 = 7247.8M。而稠密层的 `intermediate_size` 是 12288（是专家的 6 倍），
  所以**一个专家比稠密层的 MLP 小得多** —— 稀疏换来的容量是"数量"上的，不是"单个更大"。
- `base_model_ep_plan` 里 `layers.*.mlp.experts.gate_up_proj` 与 `down_proj` 都标了
  `grouped_gemm`：这 288 份权重在推理时是用**分组 GEMM** 一次算掉的，
  展开成 288 次小矩阵乘会慢到不可用。展开见 L7-03。

---

## 三、★ 稠密层的 151M 是"每 token 都付"的基准

要谈"省了多少"，先要有基准。前 3 层是稠密 MLP，它就是这个基准：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
class Glm5NextTextMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.swiglu_limit = config.swiglu_limit
```

**读法**：
- 三个投影 `bias=False`，所以参数量可以**直接手算**：`3 × hidden × intermediate`
  = 3 × 4096 × 12288 = **150,994,944**（151.0M）。
- 三个投影的**中间维都是 12288**：`gate_proj` 与 `up_proj` 把 4096 抬到 12288，
  `down_proj` 再压回 4096。SwiGLU 需要 gate 与 up 两份，所以是 3 个矩阵而不是 2 个。
- 稀疏层 vs 稠密层（实测）：
  - **参数比 7274.1M / 151.0M = 48.17×** —— 容量大了近 50 倍；
  - **算力比 227.7M / 151.0M = 1.51×** —— 每 token 实际算的只多了 51%。
- 这两个倍数的**落差**就是本课的标题：MoE 经济学。如果只看显存，MoE 是 48 倍的
  昂贵；如果只看每 token 的 FLOPs，它只比稠密层贵一半。
- 顺带一提：前 3 层之所以是稠密，见 L0-02；这份排班写在
  `config.mlp_layer_types`（3 个 `dense` + 42 个 `sparse`）。

---

## 四、每 token 的路径：8 个路由专家 + 1 个共享专家

稀疏层的 `forward` 只有 8 行，但每一行都值得停一下：

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

**读法**：
- `residuals = hidden_states` 在**最前面**就留了一份。它是给共享专家用的 ——
  共享专家吃的是**这一层的原始输入**，不是路由专家的输出。
- `self.gate(hidden_states)` 一次返回三个东西，但这里用 `_` 丢掉了 `router_logits`：
  **推理不需要它的值**（训练需要，见第九节）。
- `self.experts(...)` 内部只对**命中的专家**做矩阵乘（`mask` + `hit` 那段），
  没被选中的 280 个专家这一次前向里完全不参与计算。
- 最后一行 `hidden_states + self.shared_experts(residuals)`：
  **加法**，不是门控、不是拼接。共享专家的输出直接叠在路由专家的加权和上。
- 为什么共享专家必须存在：路由专家是在**互相竞争**中被训练的，竞争会鼓励它们
  "各管一段、越专精越好"；但语言里总有一部分变换是**每个 token 都要做**的
  （常见的句法、高频的词形）。如果没有共享专家，这部分公共知识要么被 8 个专家
  各自重复学一遍（浪费容量，还学不稳），要么谁都学不好。共享专家是一条
  **不参与竞争的快车道**：它承接公共知识，路由专家因此可以更专精。
  这也是 8 + 1 里那个 "1" 的全部理由。

---

## 五、路由器有哪些旋钮

第一个要看的不是 `forward`，而是 `__init__` —— 路由的全部行为都由这 8 行决定：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
    def __init__(self, config: Glm5NextTextConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))
        self.routed_scaling_factor = config.routed_scaling_factor
        self.num_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.e_score_correction_bias = nn.Buffer(torch.zeros(self.num_experts))
```

**读法**：
- `self.weight` 是 `(288, 4096)` —— 路由器**本身**只有 1.18M 参数，
  相对 7.27B 的专家池是千分之一量级：**路由是廉价的，专家才是贵的**。
- 五个旋钮：`top_k`（选几个）、`routed_scaling_factor`（输出缩放）、`num_group` / `topk_group`
  （分组路由）、`norm_topk_prob`（是否归一化）。本课后面逐条讲。
- `e_score_correction_bias` 是 **Buffer 不是 Parameter**：它跟着模型走、会存进权重文件，
  但**不参与梯度更新**（`_keep_in_fp32_modules_strict` 里也点了名，强制留在 fp32）。
  它只在"选谁"这一步起作用，不进入最终权重 —— 见下一节。
- 路由器的输入是 4096 维的 hidden，输出是 **288 个互相独立的分数**。

---

## 六、先打分：sigmoid 而不是 softmax

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        scores = router_logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias
```

**读法**：
- `hidden_states.view(-1, self.hidden_dim)`：路由是**逐 token** 的，
  batch 与 seq 被压平成一维，路由看不到"邻居"。
- `.type(torch.float32)` 出现两次：路由对数用 fp32 算。
  288 路的 top-k 对数值误差很敏感，bf16 下的抖动会让同一个 token 在两次前向里
  选到不同的专家 —— 这是"推理可复现性"的直接来源（L7 会再遇到）。
- **`sigmoid` 不是 `softmax`**：softmax 会把 288 个分数耦合成一个概率分布，
  某一路涨必然导致其它路跌；sigmoid 让每个专家**独立**回答"我有多想做这个 token"。
  代价是分数不再归一化（和不为 1），所以后面必须补一次归一化与缩放 —— 那是第八节。
- `scores_for_choice = scores + self.e_score_correction_bias`：**"选谁"用带 bias 的分数，
  "权重多大"用不带 bias 的 scores**（第八节的 `scores.gather`）。两者分离是为了
  让 bias 只做负载均衡的微调，不改变这一层输出的数值尺度。

---

## 七、★ 两级筛选：组内先筛，组间再选

`n_group` / `topk_group` 的意义只看这一段：

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

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.num_group, self.num_experts // self.num_group)
            .reshape(-1, self.num_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
```

**读法**：
- 第一段把 288 个专家按 `num_experts // num_group` 切成 `num_group` 组，
  每组取**分数最高的 2 个**相加当"组分数"（`topk(2, dim=-1)[0].sum(dim=-1)`）。
  用 top-2 之和而不是最大值，是为了不让一个特别高的分数单独决定整组的命运。
- 第二段 `masked_fill(~score_mask.bool(), float("-inf"))`：**落选组整组被屏蔽**，
  再用 `torch.topk(..., k=self.top_k)` 在剩下的组里选专家。
- 动机：288 选 8 是一个**全对全**的比较。在专家并行（EP）下，288 个专家分散在多台设备上，
  一次路由就是一轮 all-to-all 通信。只按分数选，会频繁出现"8 个都落在同一台设备"
  或"每个设备都被碰到一点"的抖动 —— 通信量与负载都不稳定。先在**组粒度**上收敛范围，
  等价于给路由加了一层"配额"，让通信模式更可预测。
- **默认 `n_group=1` / `topk_group=1`**：只有一组，组分数是"全体 top-2 之和"，
  整组必然入选 —— 这段代码在默认配置下是**空操作**。能力保留、默认关闭，
  这是"同一份代码要同时服务训练与推理"的典型形态。
- 实测（用真实的 `Glm5NextTextTopkRouter`，16 专家 / top-4）：
  `n_group=1, topk_group=1` 时选中的 4 个专家分布在 3 个不同的组；
  改成 `n_group=4, topk_group=2` 后，每行的 4 个专家**只落在 2 个组内** ——
  约束确实生效了。

---

## 八、归一化，然后乘以 2.5

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return router_logits, topk_weights, topk_indices
```

**读法**：
- `scores.gather(1, topk_indices)`：权重来自**没有加 bias 的** `scores`，
  形状 `(tokens, 8)`。前面 288 个分数到这里只剩 8 个。
- `norm_topk_prob=True` 时除以自己的和：**8 个权重之和恒等于 1**。
  这是一个分布式的"票数归一"：不管这一层给 8 个专家打了多高的分，
  进入加权求和的总是相对比例。
- 然后 `* self.routed_scaling_factor`（2.5）。为什么需要它：
  - sigmoid 的分数**天然偏小**。288 选 8 时，被选中的 8 个分数本来就不是 1.0 量级
    （实测例子里是 0.736 与 0.606），归一化之后每份更小；
  - 而这一路的输出要和**残差流**直接相加。如果它整体偏小，就相当于这 42 层稀疏层
    对残差几乎没有贡献，训练时信号会被淹没；
  - 乘 2.5 把这一路抬回与残差可比的量级。**"top-k 只保留了一部分概率，所以要补回来"**
    就是这个词条的全部含义。
- 实测那组数字很直观：`[0.736, 0.606]`（和 1.342）→ 归一化 `[0.548, 0.452]`（和 1.000）
  → 乘 2.5 `[1.371, 1.129]`（和 **2.500**）。**权重和恒等于 `routed_scaling_factor`**，
  这是判断这一行有没有生效的最快方法。
- 注意这一行是 `topk_weights = topk_weights * self.routed_scaling_factor`，
  **不是原地乘**：`topk_weights /= denominator` 已经改过张量了，这里换了个新张量。

---

## 九、训练用的那一项：`load_balancing_loss_func`

本课最后一个要点：**推理时这个函数根本不会被调用**。先看它的签名：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
def load_balancing_loss_func(
    gate_logits: torch.Tensor | tuple[torch.Tensor] | None,
    num_experts: int | None = None,
    top_k=2,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor | int:
```

它甚至有一条"对不上号就返回 0"的早退路径：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0
```

真正的调用点在 `Glm5NextForConditionalGeneration.forward` 的末尾：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device
```

**读法**：
- 这个损失的目标写在它的 docstring 里：**惩罚"路由太不均衡"**（Switch Transformer 的
  辅助损失）。如果一个 token 批次全挤到少数几个专家上，专家之间就得不到均衡的训练，
  这是训练问题，不是推理问题。
- `if output_router_logits:` 是唯一的开关。实测 `Glm5NextTextConfig().output_router_logits`
  默认是 **False**，于是 `aux_loss` 就是 `None`。
- 就算开关打开，第二层保险是 `if labels is not None`：**没有标签就不往 loss 里加**。
  `generate()` 推理时永远没有 labels，所以这一项在 decode 路径上连张量都不会建。
- 早退路径也值得记住：`gate_logits` 不是 tuple（比如只传了单层的张量）就直接 `return 0`。
  实测 `load_balancing_loss_func(None, ...)` 与 `load_balancing_loss_func(单层张量, ...)`
  都返回 **0**，不是抛错 —— 所以"它在训练里到底有没有生效"不能靠没报错来判断，
  要看 `output_router_logits` 和 `labels` 这两个开关。
- 那 `router_logits` 是怎么被收集起来的？`_can_record_outputs` 里挂了
  `OutputRecorder(Glm5NextTextTopkRouter, index=0)`（第 1377 行）——**要的时候才收**。
  这是它不进热路径的原因。展开见 L7。

---

## 十、与后面课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| 为什么前 3 层是稠密 MLP | L0-02 |
| `router_logits` 的收集与记录机制 | L7 |
| `grouped_gemm` / EP 计划怎么落地到设备 | L7-03 |
| 专家的分组 GEMM 具体怎么算 | L7-03 |
| 共享专家与路由专家的初始化差异 | L3-07 |
| mHC 的 4 条残差流（共享专家输出加到哪一条） | L0-05 |
| 路由采样在训练循环里怎么用 | L6 |

**一句话总结**：

> 稀疏层 = 288 个路由专家（竞争上岗，每个 token 只叫醒 8 个）+ 1 个共享专家（永远在线）；
> **参数按 288 份买（48.17× 稠密层），账单按 9 份付（1.51× 稠密层）**，
> 而 `load_balancing_loss_func` 是训练时给路由"排班"用的，推理路径上它不存在。
