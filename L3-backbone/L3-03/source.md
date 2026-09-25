<!-- glm5-coverage
models/glm5_next/modeling_glm5_next.py
-->

# L3-03 · 路由器：288 个分数如何变成 8 条路径 — 源文件

**这一课只回答一个问题：`hidden_states` 里的一行 4096 维向量，是怎么变成「第 17、42、103…号专家各算一份，再按权重加起来」的？**

视角：`Glm5NextTextTopkRouter` 是整套 MoE 的**分诊台**。它不参与任何专家计算，
只做一件事 —— 把 288 个候选压成 8 个有序的 `(索引, 权重)` 对。

| 文件 | 行数 | 本课引用块 |
|---|---|---|
| `models/glm5_next/modeling_glm5_next.py` | 2444 | 11 |

## 〇、先给结论：实测出来的五个数

下面这段是用 `_data/recon/probe_l303.py` 在 CPU 上**跑出来**的结果，不是估算：

```text
Glm5NextTextConfig() 默认值（.venv/bin/python 实测）
    hidden_size            = 4096
    n_routed_experts       = 288      num_local_experts 通过 attribute_map 指向它
    num_experts_per_tok    = 8        -> 每个 token 走 8 条路径
    n_group                = 1        <- 注意！默认只有 1 个组
    topk_group             = 1        <- 注意！默认只选 1 个组
    norm_topk_prob         = True
    routed_scaling_factor  = 2.5

一次完整前向的中间量（288 个专家，随机权重，1 个 token）
    logits        shape (1, 288)   dtype float32
    scores        = sigmoid(logits) ∈ (0, 1)
    topk_indices  = [164, 26, 287, 215, 114, 147, 128, 284]
    topk_weights  = [0.306771, 0.319676, 0.307620, 0.312870,
                     0.315230, 0.319281, 0.312806, 0.305747]
    ★ sum(topk_weights) = 2.500000        <- 正好等于 routed_scaling_factor
    ★ 归一化后这 8 个权重的和 = 1.000000
```

> 上面这个 `text` 块是**实测输出**，不是源文件的逐字引用，**不参与保真校验**。
> 下面每一个 `python` 块都是从 `modeling_glm5_next.py` **切片**出来的逐字副本。

---

## 一、★ 订正：这里没有 softmax，用的是 sigmoid

作业书把这一课的第一条要点写成「gate 线性层 + **softmax** 的完整流程」。
**实测与源码都不支持这个说法** —— 整个 `Glm5NextTextTopkRouter.forward` 里
一次 `softmax` 都没出现，用的是 `sigmoid`：

```text
$ grep -n "softmax" models/glm5_next/modeling_glm5_next.py
1002:        probabilities = torch.nan_to_num(logits.softmax(dim=2)).to(     # Indexer（DSA）
1094:    attn_weights = nn.functional.softmax(attn_weights, dim=-1, ...)     # 注意力本体
2096:        routing_weights = torch.nn.functional.softmax(layer_gate...)    # Glm5NextModel 里的辅助统计
        ^ 没有一行在 Glm5NextTextTopkRouter 里（该类位于 L146-L184）
```

**为什么必须是 sigmoid，这是个真问题**：

| | softmax 分母 | sigmoid |
|---|---|---|
| 单个分数取决于 | **全部 288 个** logit | 只有自己那一个 logit |
| 换一个专家 | 其余 287 个分数**全部变化** | 其余分数**一个都不动** |
| 意义 | 「占全体的比例」 | 「自己有多够格」 |

MoE 路由要的是后者。理由有三条，都能从下面的源码里指出来：

1. **选择与权重必须解耦。** `e_score_correction_bias` 是加在**选择**上的（L163），
   不能污染权重（L179 从 `scores` 而不是 `scores_for_choice` gather）。
   若用 softmax，一个偏置会同时改掉整行的分母，就没法「只调选择」。
2. **归一化只在选中的 8 个上做。** `norm_topk_prob` 干的就是这件事（L180-182）——
   这相当于一个**只有 8 个元素的 softmax 的分子部分**。真正的归一化发生在筛选**之后**。
3. **288 路 softmax 的数值代价与耦合都不划算。** 全行归一化让每个 token 的路由
   都依赖其余 287 个专家的瞬时状态，训练时梯度互相拉扯。

> 这一条是本课最重要的订正。**凡是"文件里写着"的说法，都要跑一遍确认** ——
> 本项目的规矩是「凡推断必实测」，作业书也不例外。

---

## 二、gate 就是一层 `Linear(4096 → 288)`

先看路由器自己持有什么。注意 `self.weight` 的形状是 `(num_experts, hidden_dim)` ——
这不是 `nn.Linear`，而是一个裸 `nn.Parameter`，前向里用 `F.linear` 手写：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
class Glm5NextTextTopkRouter(nn.Module):
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

- `self.weight` 形状 `(288, 4096)` → 输出 `(N, 288)`。**这就是"288 个分数"的来历**：
  一次矩阵乘法 `(N, 4096) @ (4096, 288)`。
- 初始化是 `torch.zeros`（L152），真正的初始化在 `_init_weights` 里（见第七节）。
- `self.top_k` 是 `num_experts_per_tok = 8`，`self.num_experts` 是 288。
  **两者之比 36 就是"压缩率"**：288 → 8，每个 token 只用 2.78% 的专家。
- 五个配置项被**缓存成实例属性**（`routed_scaling_factor` / `num_group` /
  `topk_group` / `norm_topk_prob`），而不是在 forward 里现查 config ——
  理由是 forward 在热路径上，每条 token 都要过一次。
- `e_score_correction_bias` 注册为 `nn.Buffer` 而不是 `Parameter`：
  它是**被优化器以另一套规则更新**的量（DeepSeek-V3 式的「无梯度偏置」），
  不属于反向传播的参数集。

> 回顾 **L0-01**：那里的 `causal_mask_mapping` 是"同一份 hidden 在两层上看到不同掩码"；
> 这里同理 —— **同一份 hidden 在 288 个专家上看到的是 288 个不同的分数**，
> 路由器就是产生这 288 个分数的那个算子。

---

## 三、两级筛选：先选组，再在组里选专家

前向的第一步只有四行，却完成了「降维 → 打分 → 加偏置」三件事：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        scores = router_logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias
```

**读法**：

- L160 的 `view(-1, self.hidden_dim)` 把 `(batch, seq, 4096)` 拍平成 `(batch*seq, 4096)`。
  路由是**逐 token 独立**的，batch/seq 维度在这里没有任何意义。
- L161 **强制升到 float32**：`hidden_states.type(torch.float32)` 与 `self.weight.type(torch.float32)`。
  bf16 下 288 路的 `topk` 排序容易出现"并列但取舍不同"的抖动，
  而路由的抖动会被 MoE 放大（选错专家 = 整条路径换人）。
  这与 L1379 的 `_keep_in_fp32_modules_strict` 是同一套考虑（见第七节）。
- L162 `sigmoid` —— 见第一节的订正。
- L163 `scores_for_choice = scores + e_score_correction_bias`：
  **从这里开始，"用来选的分数"和"用来算权重的分数"就是两个变量了。**
  默认 `bias` 全零，所以两者数值相同；一旦训练后非零，二者分叉（第八节有实测）。

接下来是第一级筛选 —— 把 288 个专家切成 `n_group` 个组，**每组只留最高的 2 个分数之和**作为该组的代表分：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        group_scores = (
            scores_for_choice.view(-1, self.num_group, self.num_experts // self.num_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
```

**读法**：

- `.view(-1, self.num_group, self.num_experts // self.num_group)`：
  `num_group=8` 时形状是 `(N, 8, 36)` —— 288 个专家被切成 8 组、每组 36 个。
- `.topk(2, dim=-1)[0].sum(dim=-1)`：**每组取前 2 名求和**作为组得分。
  为什么是 2 而不是 1？因为一个组要"有资格"被选中，至少得能提供 `top_k` 需要的专家；
  只取最大值会让"一个巨高分 + 35 个垃圾"的组骗过筛选。
  **这是设计意图层面的解释，源码本身只写了 `topk(2)`。**
- 于是 `group_scores` 形状 `(N, n_group)` —— **288 个数被压成了 8 个数**。
  第一级筛选的全部意义就在这个形状变化里。

★ 但这里有一个**默认配置的陷阱**，必须实测才能发现：

```text
n_group = 1（默认）时：
    group_scores.shape = (1, 1)      <- 288 个专家挤在一个组里
    score_mask 全 1 ?  True
    masked_fill 之后还有 -inf 吗 ?  False
    -> 组阶段是一个彻底的 no-op（候选池仍是全部 288 个）
```

**默认配置下，第一级筛选什么也没筛。** 它是一段"为将来的稀疏路由准备好、
但当前 checkpoint 没有启用"的代码。看到 `n_group` / `topk_group` 这两个名字
就以为"GLM-5 一定在做分组路由"，是**没跑过代码的推断**。

第二级筛选：把没被选中的组的分数**整个置成 `-inf`**：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
```

**读法**：

- L169 `torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]`：
  取 `[1]` 是**只要下标**，不要值 —— 后面只需要知道"选中的是第几组"。
- **`sorted=False` 是一个有意的选择**：排序对结果没有影响（后面只做 mask），
  却要多付一次排序代价。路由在热路径上，每一个省下的操作都乘以 token 数。
- L170-171 用 `scatter_` 把选中组标成 1：`group_mask` 是 `(N, n_group)` 的 0/1 矩阵。
  这里**没有用 `==` 比较，而是 scatter 赋值** —— 因为 `group_idx` 里可能有重复
  （`topk` 不保证唯一），scatter 天然幂等。

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.num_group, self.num_experts // self.num_group)
            .reshape(-1, self.num_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
```

**读法**：

- L172-176 是**一次升维再降维的广播**：`(N, n_group)` → `(N, n_group, 1)`
  → `(N, n_group, per_group)` → `(N, num_experts)`。
  展开后 `score_mask` 的第 `i` 位就是"第 `i` 个专家所在的组是否被选中"。
  这一步是"组级决策"翻译成"专家级掩码"的**唯一桥梁**。
- L177 `masked_fill(~score_mask.bool(), float("-inf"))`：
  置成 `-inf` 而不是 `0`，因为 `0` 仍然可能被 `topk` 选中（如果整组都是负数的话），
  而 `-inf` 在数值上**保证不可能**进入前 8。这是"用数值手段表达布尔约束"的惯用写法。

---

## 四、★ 8 条路径：一次 `topk`，三次取值来源

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return router_logits, topk_weights, topk_indices
```

**读法**（这一段是整课的落点，每一行的取值来源都不一样）：

| 行 | 变量 | 取自 | 为什么 |
|---|---|---|---|
| L178 | `topk_indices` | `scores_for_choice`（**含 bias**） | 选择要考虑偏置 |
| L179 | `topk_weights` | `scores.gather(1, topk_indices)`（**不含 bias**） | 权重必须是无偏的真实亲和度 |
| L181 | `denominator` | `topk_weights.sum(...) + 1e-20` | 只归一化**选中的 8 个** |
| L183 | `topk_weights * routed_scaling_factor` | config | 缩放 |

- **L179 是这一课最容易被写错的一行。** 选是用 `scores_for_choice` 选的，
  权重却是从 `scores` 里 `gather` 出来的。实测证据（第八节）：
  给两个专家加上 ±5 的偏置后，`topk_indices` 变了，但
  「权重 == `sigmoid(logits).gather(idx)` 归一化 ×2.5」成立，
  「权重 == `scores_for_choice.gather(idx)` 归一化 ×2.5」**不成立**。
- L181 的 `+ 1e-20` 是**防零除**。`sigmoid` 恒大于 0，理论上不会零除，
  但这个写法在 fp32 下是免费的保险 —— 注意这里是 `1e-20` 而不是常见的 `1e-9`，
  说明作者预期分母是 O(1) 量级。
- L180 的 `if self.norm_topk_prob:` 是**开关**。实测两者差别：

```text
同一组 logits，只改 norm_topk_prob（实测）
    norm_topk_prob=True    权重和 = 2.500000   （恒等于 routed_scaling_factor）
    norm_topk_prob=False   权重和 = 18.699324  （8 个 sigmoid 值直接相加，不再归一化）
```

  **`norm_topk_prob=True` 的直接后果：8 条路径的权重和是一个常数 2.5**，
  与输入无关。这让下游的专家输出尺度**可预测** —— 无论这个 token 的分数是
  0.1 还是 0.9，门控后的总能量都差不多。对训练是稳定性，对推理是可预期的数值范围。
- L178 与 L169 都用了 `sorted=False`：**返回的 8 个下标是无序的**。
  实测输出 `[164, 26, 287, 215, 114, 147, 128, 284]` 确实不是升序。
  下游 `Glm5NextTextExperts` 不依赖顺序，所以没必要排序 ——
  但**做可视化时不能假设第 0 个是分数最高的那个**。
- 返回值是三元组 `(router_logits, topk_weights, topk_indices)`：
  原始 logits 也被交出来，供辅助损失与监控使用（第七节）。

---

## 五、★ 手算一次 top-8（8 个专家 / 2 组的最小例子）

真机是 288 个专家，手算不现实。下面把规模缩到 **8 个专家 / 2 组（每组 4 个）/ `topk_group=1` / `top_k=2`**，
用**同一份代码**跑出每一个中间量 —— 你可以拿计算器逐行核对：

```text
输入  h = [1, 1, 0, 0]

① 打分（L161-L162）
   logits = W·h = [1.0,  2.0, -1.0,  1.5,  3.0,  2.5,  2.0, -2.0]
   scores = sigmoid(logits)
          = [0.731059, 0.880797, 0.268941, 0.817574,
             0.952574, 0.924142, 0.880797, 0.119203]
   ← 例：sigmoid(2.5) = 1/(1+e^-2.5) = 0.924142

② 加偏置（L163）：bias 全零 -> scores_for_choice = scores

③ 第一级：每组取 top-2 求和（L164-L168）
   组 0 = e0..e3 -> 最大的两个是 0.880797(e1) 与 0.817574(e3)
                  和 = 1.698371
   组 1 = e4..e7 -> 最大的两个是 0.952574(e4) 与 0.924142(e5)
                  和 = 1.876716
   group_scores = [1.698371, 1.876716]

④ 选组（L169-L171）：topk_group=1 -> 选中组 1
   group_mask = [0, 1]

⑤ 展开成专家掩码（L172-L176）
   score_mask = [0, 0, 0, 0, 1, 1, 1, 1]

⑥ 屏蔽（L177）
   scores_for_choice = [-inf, -inf, -inf, -inf,
                        0.952574, 0.924142, 0.880797, 0.119203]

⑦ 第二级：选 top_k=2（L178-L179）
   topk_indices = [4, 5]          <- 注意：无序
   topk_weights = scores.gather = [0.952574, 0.924142]

⑧ 归一化（L180-L182）
   分母 = 0.952574 + 0.924142 = 1.876716
   -> [0.507575, 0.492425]        <- 和 = 1

⑨ 缩放（L183）
   × 2.5 -> [1.268938, 1.231063]  <- 和 = 2.5
```

**手工核对**（实跑输出与此一致，误差 < 1e-6）：

```text
0.952574 / (0.952574 + 0.924142) = 0.507575
0.924142 / (0.952574 + 0.924142) × 2.5 = 1.231062
```

**三个必须看出来的点**：

1. **e1 的分数（0.880797）比 e5（0.924142）低，但因为 e1 所在的组输了，
   它连参选的资格都没有。** 分组筛选是**硬约束**，不是加分项 ——
   这就是"两级筛选"与"直接 top-8"的全部区别。
2. **组的代表分是"前两名之和"**，所以组 1 赢不是因为它的第一名更高，
   而是因为它**前两名都不差**。一个"尖子生 + 三个差生"的组会被这条规则压下去。
3. **权重和恒为 2.5**（第 ⑨ 步），与选中谁无关。

---

## 六、这 8 条路径接下来去哪

路由器本身不算专家。它把一个 4096 维向量变成 8 个 `(专家号, 权重)` 对，
真正的计算发生在 `Glm5NextTextExperts` 里。装配关系在 `Glm5NextTextMoE.__init__`：

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

- `self.gate = Glm5NextTextTopkRouter(config)` —— **`gate` 就是路由器**。
  这个名字来自经典 MoE 论文（门控网络），但在代码里它没有任何"门"的形状，
  只有一次线性投影 + 一次 sigmoid。
- `self.experts` 持有全部 288 个专家的权重（L3-04 展开其存储布局）。
- `self.shared_experts` 是**另一条独立通路**：`n_shared_experts=1`，
  宽度是 `moe_intermediate_size × n_shared_experts = 2048 × 1`。
  它**每 token 都跑**，不经过路由器 —— 这是"288 选 8"之外的一路兜底（L3-05 展开）。
- 注意 `config` 在这里是 `Glm5NextConfig`（多模态总配置），
  而路由器要的是 `Glm5NextTextConfig`。因为 `Glm5NextConfig` 把
  `num_local_experts` 等字段代理到了 `text_config`，两种 config 都能喂给路由器。

消费侧：

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

- L202 `residuals = hidden_states` 先**留底**：共享专家要的是**路由前**的原始输入。
  如果写成 `hidden_states + self.shared_experts(hidden_states)`（用被覆盖过的变量），
  共享专家就会吃到专家们的输出 —— 一个很隐蔽的 bug。
- L204 `_, topk_weights, topk_indices = self.gate(hidden_states)`：
  三个返回值里 **`router_logits` 被丢掉了**（下划线）。
  它只在需要辅助损失或监控时才被 `OutputRecorder` 捕获（第七节）。
  **推理热路径上，288 个 logits 算完就扔** —— 这是有意的：它们不参与计算。
- L205 `hidden_states.view(-1, hidden_states.shape[-1])`：这次拍平是为了
  按 `token_idx` 做 `index_add_`（专家输出要 scatter 回逐 token 的位置）。
  注意 `orig_shape` 在 L203 已经存好，L206 末尾 `.view(*orig_shape)` 复原。
- L207 是 MoE 的最终形式：**`专家们的结果 + 共享专家的结果`**。
  8 条路径在这一行被加成一个张量，`topk_weights` 已经在 L134 乘进每条路径了。

而"8 条路径"在专家侧的真实形态是这样一个循环 —— 每个被命中的专家，
把它名下的**所有 token 一次性**算掉（而不是逐 token 逐个专家）：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            current = self._apply_gate(F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx]))
            current = F.linear(current, self.down_proj[expert_idx]) * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final
```

**读法**：

- L126 `F.one_hot(top_k_index, num_classes=self.num_experts + 1)`：
  **`+1` 是给 padding 留的垃圾桶**（L130-131 遇到就把该专家跳过）。
  多一个类是"用 one-hot 表达稀疏归属"时的常见技巧。
- L127 `hit = torch.greater(mask.sum(dim=(-1,-2)), 0).nonzero()`：
  只遍历**至少被一个 token 选中**的专家。288 个专家里，一个 token 只碰 8 个 ——
  如果外层是"遍历 288 个专家"，99% 的迭代是空转。
  **这就是稀疏 MoE 在实现上真正省下时间的地方。**
- L132 `top_k_pos, token_idx = torch.where(mask[expert_idx])`：
  拿到"哪些 token 选中了它、分别排在第几位"。第 `k` 位很重要 ——
  权重是从 `top_k_weights[token_idx, top_k_pos]` 取的（L134），
  **token 与权重的配对靠这个位置对对齐**。
- L134 是"加权"发生的唯一地方：专家输出先乘 `top_k_weights`，再 `index_add_` 累加。
  所以 8 条路径的合并是**在专家循环内部逐条累加**完成的，不是最后再统一加权。
- L135 `final.index_add_(0, token_idx, current.to(final.dtype))`：
  `index_add_` 是**带重复下标的累加**（`index_put_` 会覆盖）。
  同一个 token 被 8 个专家命中，8 次 `index_add_` 正好把它加满。
- 这条循环是**逐专家的串行循环**。L3-04 会讲为什么权重按 `(E, ...)` 排布之后，
  它可以换成一次 grouped GEMM。

> 回顾 **L0-01**：那里的 `topk_indices` 指的是 DSA 在 2048 个**位置**里选出的下标；
> 这里的 `topk_indices` 是 MoE 在 288 个**专家**里选出的下标。
> **同一个名字，两套完全不同的稀疏机制** —— 读这个仓库时务必看清上下文。

---

## 七、`e_score_correction_bias`：一个"只作用于选择"的门

这个 buffer 有两处旁证，说明它不是普通参数：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
    _can_record_outputs = {
        "attentions": Glm5NextTextAttention,
        "hidden_states": Glm5NextTextDecoderLayer,
        "router_logits": OutputRecorder(Glm5NextTextTopkRouter, index=0),
    }
    _keep_in_fp32_modules_strict = ["e_score_correction_bias", "conv1d", "dt_bias", "A_log"]
```

**读法**：

- L1374-1378 `_can_record_outputs` 把 `OutputRecorder(..., index=0)` 挂在
  `Glm5NextTextTopkRouter` 上 —— **`index=0` 就是 `router_logits`**
  （forward 返回三元组的第 0 个）。`output_router_logits=True` 时，
  这 288 个 logits 会被逐层收集起来算负载均衡损失。
  换句话说：**推理时它是垃圾，训练时它是信号。** 同一个张量，两种命运。
- L1379 `_keep_in_fp32_modules_strict = ["e_score_correction_bias", ...]`：
  这个 buffer **强制留在 fp32**，即使模型整体以 bf16 加载。
  原因与 L161 强制 fp32 打分一致：**它是一个被反复累加的小量**，
  在 bf16 下（尾数 8 位）加着加着就"加不动了"（大数吃小数）。
- 这份清单里另外三项 `conv1d` / `dt_bias` / `A_log` 都属于 KDA 层（L4-02 ~ L4-05）——
  **同一个 fp32 白名单，横跨两种完全不同的机制**。

初始化处：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        elif isinstance(module, Glm5NextTextExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, Glm5NextTextTopkRouter):
            init.zeros_(module.e_score_correction_bias)
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
```

**读法**：

- L1418 `init.zeros_(module.e_score_correction_bias)` —— **显式清零**。
  这解释了第一节的实测：默认配置下 `scores_for_choice == scores`（逐位相等）。
  也就是说，**未训练的模型上"两级筛选"退化成"直接 top-8"**。
- L1419 `init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)`：
  `initializer_range = 0.02`，与 `__init__` 里 `torch.zeros` 的占位形成对照 ——
  **占位用零，真正初始化在 `_init_weights` 里**。这是 transformers 的统一约定：
  `__init__` 只保证"能不报错地建出形状"，权重内容交给 `_init_weights`。
- 注意 `Experts` 的初始化就在它上面（L1414-1416），也是 `normal_(std=0.02)`。
  路由器与专家的初始化尺度**故意保持一致**。

用实测看偏置到底改了什么（给 e164 加 −5、给 e7 加 +5）：

```text
bias 非零的专家数  = 2
未加 bias 的 top-8 = [164, 26, 287, 215, 114, 147, 128, 284]
加 bias 后的 top-8 = [215, 26, 114, 147, 7, 287, 128, 284]
   e164 还在吗 = False      e7 进来了吗 = True

★ 权重来自 scores 而非 scores_for_choice：
   权重 == sigmoid(logits).gather(idx) 归一化 ×2.5       ? True
   权重 == scores_for_choice.gather(idx) 归一化 ×2.5     ? False
```

**一句话**：偏置是**选拔委员会**，不是**计分员**。它决定谁上场，不改上场后的得分。

---

## 八、分组路由在分布式下到底好在哪

先声明边界：**`modeling_glm5_next.py` 只实现了"选择"这一步，一行通信代码都没有。**
下面关于分布式的部分，一部分是可算的算术，一部分是**设计意图**，我会分开标。

**可算的部分**：

```text
288 个专家，每组 36 个
    n_group = 8   -> 8 组 × 36 个
    topk_group = 2 -> 候选池 288 -> 72 个（216 个被置 -inf，实测）

EP（专家并行）= 8 的常见切法：每张卡恰好持有 1 个组 = 36 个专家

不分组（默认 n_group=1）：8 个选中专家理论上可能落在 8 张卡上
                          -> 需要向 8 个目的地做 dispatch / all-to-all
限定 topk_group=2        ：8 个专家必然落在 ≤2 个组 = ≤2 张卡上
                          -> 通信扇出从 ≤8 降到 ≤2
```

**实测验证这个约束真的成立**（`n_group=8, topk_group=2`）：

```text
group_scores shape = (1, 8)
各组 top-2 之和     = [1.7871, 1.7063, 1.7538, 1.8790, 1.8731, 1.7889, 1.7762, 1.8351]
选中的 2 个组       = [3, 4]
候选池             : 288 -> 72 个
topk_indices       = [126, 128, 147, 114, 164, 136, 173, 125]
它们所属的组        = [3, 4]
★ 8 个专家全部落在选中的 2 个组里
```

**设计意图（推断，源码未直接表达）**：

- 分组把"选 8 个"从**全局贪心**改成**两级贪心**。代价是可能选不到全局最优的 8 个
  （第五节的 e1 就是这样被牺牲的），换来的是**通信拓扑上的可预测性**：
  一个 token 的 8 份激活最多只走 2 条链路的距离。
- 这与 DSA（L4-06）是**同一个思路的两次应用**：注意力那边在 100 万个位置里
  先粗筛再精选，MoE 这边在 288 个专家里先粗筛再精选。
  **大模型工程里，"先分组再精挑"出现的频率高得惊人。**

★ 但必须回到实测：**默认 checkpoint 是 `n_group=1 / topk_group=1`，
这段能力当前是"装好了但没通电"。** 谁要说"GLM-5 靠分组路由省通信"，
得先证明 checkpoint 里的 `n_group` 不是 1。

本课的 coverage 只声明 `modeling_glm5_next.py`。作为旁证，
`configuration_glm5_next.py` 的 `base_model_ep_plan` 里逐字写着（**该文件不在本课
coverage 内，用 `text` 块标出，不参与保真校验**）：

```text
    base_model_ep_plan = {
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
    }
```

**`"layers.*.mlp.gate": "ep_router"` 是路由器的另一个身份** —— 它在专家并行下
被当作一个**特殊的、跨卡的分发算子**来对待，而不是普通的线性层。
（该 plan 的完整含义属于 L1 与 L3-04/L3-05 的范围。）

---

## 九、把六步连起来

```text
  hidden_states  (batch, seq, 4096)
        │  view(-1, 4096)                        ← 逐 token 独立
        ▼
  h  (N, 4096)
        │  F.linear(h.float(), W.float())        ← W: (288, 4096)，fp32
        ▼
  router_logits  (N, 288)
        │  .sigmoid()                            ← ★ 不是 softmax
        ▼
  scores  (N, 288)   ∈ (0,1)
        │  + e_score_correction_bias             ← 默认全零
        ▼
  scores_for_choice  (N, 288)
        │  view(N, n_group, 288/n_group).topk(2).sum(-1)
        ▼
  group_scores  (N, n_group)                     ← 288 -> n_group（默认 1）
        │  topk(topk_group) -> scatter -> expand -> reshape
        ▼
  score_mask  (N, 288)  0/1
        │  masked_fill(~mask, -inf)
        ▼
  scores_for_choice  (N, 288)  其中 288-topk_group*per_group 个是 -inf
        │  topk(8, sorted=False)                 ← 第二级筛选
        ▼
  topk_indices  (N, 8)     ← 无序
        │  scores.gather(1, topk_indices)        ← ★ 从无偏的 scores 取
        ▼
  topk_weights  (N, 8)
        │  /= sum(+1e-20)          (norm_topk_prob=True)
        │  *= routed_scaling_factor (2.5)
        ▼
  topk_weights  (N, 8)     和恒为 2.5
        │
        └──► Glm5NextTextMoE.forward
                 experts(h, topk_indices, topk_weights)   ← 每专家一次，index_add_
               + shared_experts(residuals)                ← 每 token 都跑，不过路由
```

---

## 十、与后续课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| 288 个专家的权重按什么形状存 | L3-04 |
| `gate_up_proj` 怎么打包、grouped GEMM 怎么用 | L3-04 |
| 共享专家为什么能稳住训练 | L3-05 |
| `output_router_logits` 在推理时的影响 | L3-05 |
| `router_logits` 之外的负载均衡损失怎么算 | L3-05 |
| KDA 层里的 `A_log` / `conv1d` 为什么要 fp32 | L4-02 ~ L4-05 |
| DSA 的 `topk` 与 MoE 的 `topk` 有何异同 | L4-06 |
| `past_key_values` 在 MoE 层里怎么处理 | L5-01 ~ L5-03 |
| `e_score_correction_bias` 的更新规则 | 训练侧（不在本课件覆盖域） |

**一句话总结**：

> 288 个分数 = 一次 `Linear(4096→288)` 再 `sigmoid`（**不是 softmax**）；
> 两级筛选先按组淘汰再取 `top-8`，但**默认 `n_group=1` 时第一级是空的**；
> 权重从**无偏的** `scores` 里 `gather`，归一化后乘 2.5 ——
> **所以 8 条路径的权重和永远等于 `routed_scaling_factor`**。
