<!-- glm5-coverage
models/glm5_next/modeling_glm5_next.py
-->

# L3-05 · MoE 装配与共享专家 — 源文件

**这一课只回答一个问题：一个 token 进了稀疏层以后，走了几条路、在哪一行合流。**

视角：`Glm5NextTextDecoderLayer` 把 `self.mlp` 指到 `Glm5NextTextMoE` 的那一刻起，
前向里就出现了**两条并行的专家路径** —— 一条是「8 选 1 组」的路由专家，一条是**每层都开**的共享专家。
本课把这两条路拆开、量出来、再合回去。

| 文件 | 行数 | 本课引用块 |
|---|---|---|
| `models/glm5_next/modeling_glm5_next.py` | 2444 | 9 |

**实测的 MoE 配置与参数量**（用 `_recon/probe_moe.py`、`probe_moe2.py`、`probe_moe3.py`
在 CPU 上跑出，不是估算）：

```text
Glm5NextTextConfig() 默认值（只列与本课有关的）
    hidden_size          = 4096        moe_intermediate_size = 2048
    intermediate_size    = 12288       n_routed_experts      = 288
    num_experts_per_tok  = 8           n_shared_experts      = 1
    routed_scaling_factor= 2.5         norm_topk_prob        = True
    n_group              = 1           topk_group            = 1
    swiglu_limit         = 10.0        output_router_logits  = False
    router_aux_loss_coef = 0.001       mlp_layer_types       = 3×dense + 42×sparse

一层的参数量（H=4096, I=2048, E=288）
    单个路由专家      = 2*I*H + H*I        = 25,165,824
    288 个路由专家合计                      = 7,247,757,312   (7.248 B)
    共享专家(n_shared_experts=1, 中间维 I)  =    25,165,824   (25.2 M)
    路由分量 / 共享分量                     = 288.0 倍
    每个 token 激活的专家数 = 8(路由) + 1(共享) = 9  /  289  → 3.11%
    稠密层 MLP: 3*12288*4096                =   150,994,944
```

> 上面这一段是**示意性的汇总**，用 `text` 块标出 —— 它不是源文件的逐字引用，
> 因此不参与保真校验。下面每一段 `python` 块都是逐字引用。

---

## 一、MoE 装配：一层里塞了三块东西

`Glm5NextTextMoE.__init__` 只有 6 行，但它决定了这一层里有三个独立子模块。
注意 `shared_experts` 的 `intermediate_size` 参数 —— 这是本课的第一个关键点，
它把「共享专家」直接**声明成一个完整尺寸的 `Glm5NextTextMLP`**。

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
class Glm5NextTextMoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

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
- 三个子模块分工不同：`gate` 只出**分数**（不碰 hidden），`experts` 出 8 个被选中专家的加权和，
  `shared_experts` 出**与路由完全无关**的一路输出。
- `intermediate_size=config.moe_intermediate_size * config.n_shared_experts` 是**显式传入**的，
  不是默认值 —— 默认值是 `config.intermediate_size = 12288`（稠密层的宽度）。
  这里传 `2048 * 1 = 2048`，正好等于一个路由专家的中间维。**共享专家就是一个路由专家那么大。**
  实测：`n_shared_experts=1` 时，共享专家 25.2 M 参数，单个路由专家也是 25.2 M，比值 1.000。
- `config: Glm5NextConfig` 这个类型标注**看着不对但能跑**：`Glm5NextTextExperts` 里读的是
  `config.num_local_experts`，而 `Glm5NextConfig` 顶层没有这个属性（实测 `attribute_map == {}`）。
  真正被传进来的是 `config.text_config`（`Glm5NextTextModel` 拿到的是子配置），
  而 `Glm5NextTextConfig` 有 `attribute_map = {"num_local_experts": "n_routed_experts"}`（L1-04 讲过）。
  **类型标注是历史遗留，不要据此推断类型。**

---

## 二、前向：8 行，两条专家路径

这是本课的主图。整个 MoE 前向只有 8 行，但**分支与合流都在这 8 行里**：

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
- **第 2 行 `residuals = hidden_states` 是整段的核心**。它在任何 reshape 之前留了一份**原样**的输入，
  第 7 行把它喂给 `shared_experts`。也就是说：**共享专家吃的是未经路由、未经展平的原始 hidden。**
- **第 3 行 `orig_shape`** 与第 5 行 `view(-1, ...)` / `.view(*orig_shape)` 配对：
  专家计算要求 2D `(tokens, hidden)`，算完再掰回 `(batch, seq, hidden)`。
- **合流方式是逐元素相加，没有任何系数**：`routed_output + shared_output`。
  没有 `alpha`、没有 `gate`、没有 `scale` —— 这一点值得单独记，见第三节的消融实验。
- `self.experts(...)` 只需要 `topk_indices` 与 `topk_weights`，**不需要 router_logits**。
  这就是第 4 行用 `_` 接住第一个返回值的原因（第六节展开）。

**这一段的实测数据流**（`probe_moe.py`，H=64/I=32/E=8 的小配置）：

```text
输入                       hidden_states (2, 5, 64)
第 2 行                    residuals    (2, 5, 64)      ← 原样留一份
第 4 行 gate 返回          router_logits (10, 8)         ← 被 `_` 丢掉
                           topk_weights  (10, 2)         每行和 = 2.5
                           topk_indices  (10, 2)         例：[[6, 5], ...]
第 5 行 展平               hidden_states (10, 64)
第 6 行 experts            (10, 64) → 再 view 回 (2, 5, 64)
第 7 行 + shared_experts   (2, 5, 64) + (2, 5, 64)

实测：torch.allclose(moe(x), experts(x) + shared_experts(x)) == True
```

---

## 三、★ 共享专家是「加法旁路」：两个消融实验

「共享专家」这个词很容易被理解成「一个所有 token 都会走、权重较小的专家」。
**实测不是这样。** 下面两个实验各自只改一个子模块的权重：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # Key difference using clamping
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)
```

**读法**：
- 这就是 `Glm5NextTextMLP.forward`。共享专家**执行的就是这段代码**，
  所以它带着完整的 SwiGLU 与 `swiglu_limit=10.0` 双向截断（回顾 L3-02：截断是为了压住离群值）。
- 反过来说：**共享专家没有任何「专家」特有的机制** —— 没有 top-k、没有分组、
  没有 `e_score_correction_bias`。它是一个**每层无条件执行的稠密 FFN**，
  只是宽度被显式设成了 `moe_intermediate_size`。

消融实验（`probe_moe3.py`，权重先放大到 `std=0.5` 才有区分度）：

```text
实验 A：把 288 个路由专家的 gate_up_proj / down_proj 全部置零
    → moe(x) 与 shared_experts(x) 逐元素相等（allclose == True）
    → 说明共享专家的输出里**不含任何路由信息**，它是纯加法项

实验 B：把共享专家 down_proj 置零
    → 输出剩下 routed 部分，范数 1148.19（共享部分范数 826.50）
    → 两项量级相当，不是「小补丁」，是**对等的一路**

实验 C：把路由门 weight 置零（注意：不是置零专家）
    → 输出**不等于** shared_experts(x)，因为 sigmoid(0)=0.5 让所有专家同分，
      top-k 仍然会选出 2 个（实测 [6, 5]）并各拿 1.25 权重
    → 结论：门置零 ≠ 路由关闭。真正让路由消失的是把专家权重置零。
```

**为什么共享专家能稳住训练**（这是设计意图，与上面的实测事实分开陈述）：
- 路由分支是一个**离散选择**：某个专家一旦没被选中，它在这一步就收不到梯度。
  极端情况下路由器退化（少量专家吸走全部 token），路由分支的输出会从「多专家混合」
  塌缩成「单专家」。
- 共享专家给每个 token 提供了一份**不经过选择**的稠密变换。
  于是无论路由怎么抖，这一层都还有一路**参数固定、梯度连续**的通路。
- 数字上：每 token 用到 9 个专家（8 路由 + 1 共享），共享专家占其中 **1/9 ≈ 11.1%** 的
  实际计算量，但它背后只有 25.2 M 参数，而路由分支是 7.248 B —— **用 0.35% 的参数
  买了一条永远在线的兜底通路**。
- 补一个「分工」的说法（同样只是设计意图）：路由专家负责**稀疏记忆**（谁被选中谁才算），
  共享专家负责**公共变换**（每一层每一条 token 都要做的那份工作）。

---

## 四、稠密分支：`Glm5NextTextMoE` 只在 `"sparse"` 层被构造

`Glm5NextTextMoE` 这个类本身**没有任何「我是第几层」的概念**。
选不选它，发生在 decoder 层构造的时候：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        self.block_type = config.layer_types[layer_idx]
        self.hidden_size = config.hidden_size
        self.self_attn = (
            Glm5NextTextLinearAttention(config, layer_idx)
            if self.block_type == "linear_attention"
            else Glm5NextTextAttention(config, layer_idx)
        )

        self.mlp = (
            Glm5NextTextMoE(config) if config.mlp_layer_types[layer_idx] == "sparse" else Glm5NextTextMLP(config)
        )
```

**读法**：
- 这是**实例化期的二选一**，不是前向期的 `if`。一旦构件完成，
  这一层的 `self.mlp` 要么是 `Glm5NextTextMoE`、要么是 `Glm5NextTextMLP`，**永不再变**。
  所以 L0-01 里那张「45 层排班表」不是运行时判断，是构造时就定死的结构。
- 判据是 `config.mlp_layer_types[layer_idx] == "sparse"`。实测默认值：
  `['dense', 'dense', 'dense', 'sparse', ...]`，即 **前 3 层稠密、其余 42 层稀疏**。
- 注意这里给 `Glm5NextTextMoE` 传的是 `config` 而不是 `config.text_config` ——
  与第一节的类型标注疑点呼应。两处都写 `Glm5NextConfig`，但实际拿到的对象类型由
  `Glm5NextTextModel.__init__` 决定。
- 对比一下两套 FFN 的规模（实测 H=64/I=96/I_moe=32 的小配置）：
  `Glm5NextTextMLP` 的中间维是 `config.intermediate_size`，`Glm5NextTextMoE` 里
  路由专家与共享专家的中间维都是 `config.moe_intermediate_size`。
  **在默认配置下，稠密层宽 12288，MoE 的每个专家窄到 2048** —— 稀疏化的第一刀是「把专家做窄」。
- 回顾 L3-03：`mlp_layer_types` 与 `layer_types` 是**两条互相独立的排班轴**，
  decoder 层对它们各做一次二选一，这里看到的是 MLP 那一轴。

---

## 五、路由器的输出：三元组，以及被丢掉的那一个

第三幕的主角是 `Glm5NextTextTopkRouter.forward` 的**返回值**：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        topk_weights = topk_weights * self.routed_scaling_factor
        return router_logits, topk_weights, topk_indices
```

**读法**：
- 返回的是**三元组**，顺序是 `(router_logits, topk_weights, topk_indices)`。
  调用方 `Glm5NextTextMoE.forward` 写的是 `_, topk_weights, topk_indices = self.gate(...)` ——
  **第一个元素被明确丢掉了**。
- 三个元素的角色完全不同：

```text
router_logits  (tokens, 288)  float32   给辅助损失（load balancing）用；前向不需要
topk_weights   (tokens, 8)              归一化并乘过 routed_scaling_factor
topk_indices   (tokens, 8)   int64      被选中的专家号，交给 grouped GEMM 分组
```

- `topk_weights` 的**权重和是常数 2.5**，这是两条设置的联合效果：
  `norm_topk_prob=True` 把它归一化成和为 1，`routed_scaling_factor=2.5` 再乘回 2.5。
  实测（H=64/E=8/top_k=2，只放大了门权重）：
  `topk_weights` 逐行和恒为 `2.5`；门权重置零、所有专家同分时，
  归一化后每路各拿 `2.5/2 = 1.25`。
- **★ 手算例子（默认配置）**：`num_experts_per_tok=8`、`routed_scaling_factor=2.5`。
  设某 token 选中的 8 个专家原始 `sigmoid` 分数分别为
  `0.62, 0.55, 0.51, 0.44, 0.40, 0.33, 0.30, 0.25`（和 = 3.40）。
  1. `norm_topk_prob=True` → 除以 `3.40 + 1e-20`，得到
     `0.1824, 0.1618, 0.1500, 0.1294, 0.1176, 0.0971, 0.0882, 0.0735`（和 = 1.0000）；
  2. 乘 `routed_scaling_factor = 2.5` →
     `0.4559, 0.4044, 0.3750, 0.3235, 0.2941, 0.2426, 0.2206, 0.1838`（和 = **2.5000**）。
  3. 于是**每个 token 从路由专家那里拿到多少，是一个只与「选了谁」有关的比例问题**，
     总量恒定 2.5 —— 换层、换 token 都不变。
- 回顾 L3-03：分数为什么用 `sigmoid` 而不是 `softmax`、分组两级筛选怎么走，
  都在那一课；本课只关心**输出的形状与去向**。

---

## 六、★ router_logits 算了却没人要：`_` 与 OutputRecorder

这是本课最值得记住的一处**反直觉**：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
    _can_record_outputs = {
        "attentions": Glm5NextTextAttention,
        "hidden_states": Glm5NextTextDecoderLayer,
        "router_logits": OutputRecorder(Glm5NextTextTopkRouter, index=0),
    }
```

**读法**：
- 第 5 行 `_, topk_weights, topk_indices = self.gate(hidden_states)` 里那个下划线**不是省略**：
  `router_logits` 确实被算出来了（`probe_moe2.py` 用 forward hook 数过：
  **无论开关怎么设，`Glm5NextTextTopkRouter` 每次前向都被调用且只调用 1 次**），
  但 MoE 这一层**根本不需要它**。
- 它需要它的地方在**别处**：`_can_record_outputs` 这张表登记了
  「`Glm5NextTextTopkRouter` 的第 0 个返回值就是 `router_logits`」。
  前向跑完，由 `@capture_outputs` 机制把每一层的这份张量**收集起来**，
  再交给辅助损失 `load_balancing_loss_func`（第七节）。
- 所以文件里同时存在两件事，**不矛盾**：
  1. `Glm5NextTextMoE.forward` 丢弃 `router_logits` —— 因为前向不需要它；
  2. `_can_record_outputs` 登记 `router_logits` —— 因为反向需要它算负载均衡损失。
- `index=0` 这个参数是**位置约定**：路由器返回三元组，第 0 位才是 logits。
  如果有人调换了 `return` 里的顺序（第五节那段代码），这张表会**静默地**记录错东西 ——
  这类改动不会报错，只会让辅助损失变成一个没有意义的数。这是一个真实的脆弱点。
- 下面 `Glm5NextPreTrainedModel` 里还有一条与本课相关的初始化规则：
  路由器的 `weight` 与 `e_score_correction_bias` 怎么初始化。

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        elif isinstance(module, Glm5NextTextTopkRouter):
            init.zeros_(module.e_score_correction_bias)
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
```

**读法**：
- `weight` 用 `normal_(mean=0, std=0.02)` 初始化，而 `Glm5NextTextTopkRouter.__init__` 里
  它一开始是 `torch.zeros(...)`。**也就是说：不跑 `post_init()` 的话，所有门权重是 0。**
- 权重为 0 意味着 `router_logits` 全为 0、`scores = sigmoid(0) = 0.5` 处处相同，
  于是 top-k **退化成「随便挑 8 个」**（实测恒为同一组下标），
  而权重因为归一化仍然各拿 `2.5/8 = 0.3125`。
  **这是「路由还没学会」的初始状态，不是「没有路由」。**
- `e_score_correction_bias` 初始化为 0。它加在 `scores_for_choice` 上（`scores + bias`），
  只在**「选谁」（`torch.topk(scores_for_choice, ...)`）**这一步起作用；
  而权重取的是 `topk_weights = scores.gather(1, topk_indices)` —— **取的是没加 bias 的 `scores`**。
  也就是说：**bias 不改变任何一个专家的得分，只改变谁的得分被采用。**
- 实测（`probe_moe.py` 的同类操作，H=64/E=8/top_k=2）：给 0 号专家加 `+5`、给 1 号专家加 `-5` 后，
  `topk_indices` 从 `[[6, 4], ...]` 变成 `[[0, 6], ...]`；`topk_weights` 也随之改变
  （因为换上来的专家分数不同），但**每一行仍然归一化到和 = 2.5**。
  这两件事要分清：**变的是「选了谁」，不是「bias 被加进了权重」。**
- 这就是 `e_score_correction_bias` 的用途：**用偏置做负载均衡，而不必去改路由分数本身**
  —— 与第八节的辅助损失是两条独立的均衡手段（L3-03 展开）。

---

## 七、`output_router_logits`：推理时它关掉了什么

讲解要点里最容易被讲错的一条。先把代码摆出来：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.text_config.output_router_logits
        )
```

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

**读法（三条，全部实测过）**：
1. **默认是关的。** `Glm5NextTextConfig.output_router_logits` 默认 `False`。
   所以 `generate()` 这条路**永远不会进**这个 `if`：不算 `aux_loss`、不改 `loss`。
2. **关掉它不改变任何一个数值。** `probe_moe2.py` 用同一份输入跑了
   `output_router_logits=False` 与 `True` 两次，
   `last_hidden_state` 的 `torch.allclose(..., atol=0)` 为 `True`，
   逐元素完全相同。原因在第六节：**路由器的调用次数与这个开关无关**（都是 1 次/层），
   开关只决定「收不收集 `router_logits`、算不算 `aux_loss`」。
3. **它的代价是显存。** 打开后每一层都要留一份 `(tokens, 288)` 的 float32 张量：

```text
router_logits 单个 token  = 288 × 4 B = 1,152 B
1 层 × 4k token           =   4.5 MiB
1 层 × 64k token          =  72.0 MiB
45 层 × 64k token         =   3.16 GiB
45 层 × 1M token          =  48.3 GiB          ← 与 L5 的显存账本直接冲突
```

- 换句话说：它是一个**训练期的观测开关**，在推理路径上唯一的作用是**变慢、变占内存**。
  这也是为什么它默认 `False`，且 `load_balancing_loss_func` 里那句
  `if gate_logits is None or not isinstance(gate_logits, tuple): return 0` 是必要的兜底。
- 参数优先级：**调用参数 > config**（代码第一段那个三目）。
  单次 `forward(..., output_router_logits=True)` 可以临时打开，不影响 config。
- 回顾 L1-04：`output_router_logits` 与 `router_aux_loss_coef`（默认 `0.001`）
  是一对 —— 前者决定「算不算」，后者决定「算出来乘多少再加进 loss」。
  它们只在 `labels is not None`（即训练）时才真正影响结果。

---

## 八、辅助损失：实测三个极端值

`load_balancing_loss_func` 的公式（文件 docstring 里写了出处：Switch Transformer 式 (4)-(6)）：

```text
loss = num_experts × Σ_e ( tokens_per_expert[e] × router_prob_per_expert[e] )
     = E × ⟨专家使用率 , 专家概率质量⟩
```

**实测三个极端值**（`probe_moe2.py`，E=8, top_k=2, 2000 个 token，输入直接构造）：

```text
路由完全均匀（所有 logits = 0）     aux_loss = 2.0     ← 最小值，恰好等于 top_k
全部 token 挤在 1 个专家            aux_loss = 8.0     ← 最大值，恰好等于 num_experts
（E=288, top_k=8：均匀 → 32.0 量级，全挤一个 → 1.152e9 量级）
```

**读法**：
- **均匀时的值 = `top_k`**：因为使用率全是 `top_k/E`、概率质量全是 `1/E`，
  `E × Σ (top_k/E)(1/E) = E × (top_k/E) = top_k`。
  所以「均匀时 = 2.0」不是巧合，是可以手推的基准线。
- **全挤一个时的值 = `E`**：使用率与概率质量都集中在同一个专家上，
  `E × (1 × 1) = E = 8`。注意这个值是**与 token 数无关的**（两个量都做了归一化）。
- 所以 `router_aux_loss_coef = 0.001` 乘出来的惩罚项，
  在 `[0.002, 0.008]` 这个量级 —— 相对主损失很小，是一个**温和的矫正力**，
  不是硬约束。这和 L3-06 的 Sinkhorn（20 次迭代硬压成双随机矩阵）是**两种不同的思路**：
  一个用软惩罚引导，一个用迭代投影强制。
- 文件里还留了一段**显存优化**的注释（第 2084–2085 行）：
  逐层累加、最后再归一化，峰值显存与层数无关。这与第七节的代价正好构成一对：
  **`router_logits` 本身的存储是 O(层数 × token × E)，但把它变成 loss 的过程是 O(token × E)。**
- `attention_mask` 不为 `None` 时，padding token 会被 `flat_mask` 加权排除 ——
  否则一段 padding 会凭空改变专家使用率。这是「同一段代码在训练/推理下意义不同」的又一例。

---

## 九、与前后课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| `config.mlp_layer_types` 这个 3 + 42 的序列从哪来 | L0-02 |
| `layer_types` 与 `mlp_layer_types` 两条轴的关系 | L0-03 |
| `num_local_experts` 为什么写在 `attribute_map` 里 | L1-04 |
| SwiGLU 的 `gate/up` 与 `swiglu_limit` 截断 | L3-02 |
| `sigmoid` 分数、分组两级筛选、`e_score_correction_bias` | L3-03 |
| 专家权重为什么按 `(E, 2I, H)` 打包、grouped GEMM 怎么用 `topk_indices` | L3-04 |
| mHC 的 `post` / `comb` 与残差写回 | L3-06 |
| `router_logits` 这份显存在推理时值多少字节 | L5 全层 |

**一句话总结**：

> 稀疏层 = **一路被 8 选 1 选出来的路由专家** + **一路永远在线的共享专家**，
> 在 `hidden_states + self.shared_experts(residuals)` 这一行**无权相加**；
> `router_logits` 算了但前向不要它 —— 它只在 `output_router_logits=True` 时被收集起来，
> 去算一个默认关闭、且**完全不影响前向数值**的负载均衡损失。
