<!-- glm5-coverage
models/glm5_next/modeling_glm5_next.py
pytorch_utils.py
-->

# L3-01 · 三种 RMSNorm — 源文件

**这一课只回答一个问题：一层里的归一化为什么会有三种形态。**

视角：先把主干里 **237 个 norm 实例**按形态分成三类，再逐个回到调用现场，看每一种形态是被什么需求逼出来的。

| 文件 | 行数 | 本课引用块 |
|---|---|---|
| `models/glm5_next/modeling_glm5_next.py` | 2444 | 11 |
| `pytorch_utils.py` | 255 | 3 |

**实测汇总**（在 `.venv` 里用 torch 2.14.0 / transformers 5.18.0.dev0，CPU + meta device 跑
`_data/recon/probe_L3-01.py` 得到；脚本本身不计入覆盖率）：

```text
主干（Glm5NextTextModel，Glm5NextTextConfig() 默认 45 层）里的 norm 实例：237 个
    Glm5NextTextRMSNorm          x 113  = input_layernorm + post_attention_layernorm (45x2)
                                        + q_a_layernorm + kv_a_layernorm (11x2) + 出口 norm (1)
    Glm5NextTextUnweightedRMSNorm x  90  = HyperConnection.input_norm（45 层 x attn_hc/ffn_hc）
    Glm5NextTextRMSNormGated     x  34  = KDA 层的 o_norm（34 正好等于 linear_attention 层数）

参数量（meta device 实测，不占内存）
    模型总参数                       312,692,428,286
    三种 norm 合计                       399,616   ->  占 0.000128%
        Glm5NextTextRMSNorm              395,264
        Glm5NextTextRMSNormGated           4,352
        Glm5NextTextUnweightedRMSNorm          0   <- 一个参数都没有

源码里还有第四个类 Glm5NextRMSNorm，只出现在视觉塔；实测把它去掉类名与 docstring 后的
17 行与 Glm5NextTextRMSNorm 逐字相同（比较结果为 True）。
```

> 上面这段是**实测汇总**，用 `text` 块标出 —— 它不是源文件的逐字引用，因此不参与保真校验。
> 下面每一段 `python` 块都是逐字引用。本课还引用了 `configuration_glm5_next.py` 里的
> `rms_norm_eps` 取值与 `_data/recon/probe_L3-01.py` 的输出，**这两个文件都不计入本课覆盖率**
> （覆盖域只统计已声明的两个文件）。

---

## 一、★ 三种形态，是被下游需求逼出来的

先给结论，再看证据。

```text
形态           weight   逐通道门控   用在哪                          下游要什么
加权版         4096 个   无         子层入口/出口、q_a/kv_a、出口    要方向自由度（一组可学通道尺度）
无权重版       0 个      无         mHC 内部（4 条流拼接后）         只要尺度，不要方向
门控版         128 个    有(sigmoid) KDA 的输出 o_norm             要在逐通道上把「读出的记忆」关掉
```

一句话：**归一化都要把每个 token 的尺度拉回 1，但"拉回之后还要不要形状自由度"由下游决定。**

- 子层入口要的是"标准化之后的输入"，所以给每个通道一个可学的乘数 —— 这是加权版存在的理由；
- mHC 只是想把 4 条流的尺度对齐后再去算混合系数，多一组权重只会多一份优化负担；
- KDA 的输出是递推记忆的读出，尺度会随递推步数漂移，需要一个**与内容相关**的闸门，
  于是 weight 之外再多一个 gate。

三种形态的代码量分别是 19 行、7 行、22 行 —— 都很短，差别全在"有没有 weight""有没有 gate"。

---

## 二、基准形态：加权 RMSNorm 的两半

### 2.1 状态：只有 weight 和 eps

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
@use_kernel_forward_from_hub("RMSNorm")
class Glm5NextTextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Glm5NextTextRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
```

**读法**：

- 整个模块的状态只有两样：长度等于 `hidden_size` 的 `weight`（初始化为全 1）和一个浮点 `eps`。
  **没有 bias、没有 running statistics、没有 buffer** —— 这是它与 `nn.LayerNorm` 最本质的差别：
  RMSNorm 不做去均值、不估训练期统计量，因此训练与推理的行为完全一样，
  也就没有 `track_running_stats` 那一套要在 `state_dict` 里搬来搬去的状态。
- `eps` 存成普通 Python 属性（`self.variance_epsilon = eps`）而不是 buffer：它不进 `state_dict`，
  也不随 `.to(bf16)` 变精度。这意味着一份 checkpoint 可以用不同的 eps 跑 ——
  第五节会看到模型正是这么做的：类默认 1e-6，实际传 1e-5。
- docstring 那句 `equivalent to T5LayerNorm` 是作者给的定位：**这不是 GLM-5 的发明**，
  是 T5 那一支传下来的标准写法，读懂它等于读懂一大批模型。
- `@use_kernel_forward_from_hub("RMSNorm")` 把这层的 forward 交给 Hub 上的融合 kernel 覆盖。
  为什么 norm 值得单独做 kernel：它是**访存密集、逐元素**的算子，一次融合就能省掉
  fp32 中间张量的来回搬。
- `extra_repr` 不是装饰品：`print(model)` 时会打出 `(4096,), eps=1e-05`，
  这是核对"到底哪个 eps 生效"最省事的抓手（第五节用它）。

### 2.2 前向：五行数学，三处精度约定

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
```

**读法**（本课最该逐行读的五行的第一组）：

- 第一行先记下输入 dtype，**第二行无条件升到 fp32**：均值是在 fp32 里求的。
  bf16 只有 8 位尾数，4096 个数求和的舍入误差会随维度累积；升到 fp32 之后可以忽略。
  这是整层最便宜的精度保险 —— 中间张量只多一份 fp32 的副本。
- `pow(2).mean(-1, keepdim=True)` 产出的是**每个 token 一个标量**：形状从 `(b, s, 4096)`
  变成 `(b, s, 1)`（实测）。所以 RMSNorm 是**逐 token 独立**的算子 ——
  这个性质在第七节会直接决定它能不能分块、能不能融合。
- `rsqrt(variance + eps)` 一次算完"开方 + 取倒数"。`+ eps` 的唯一作用是防止 `variance = 0`
  时除零；它的量级怎么选见第五节。
- 最后一行有个不显眼的细节：`hidden_states.to(input_dtype)` **先降精度、再乘 weight**。
  也就是说加权乘法发生在输入精度里；而门控版（第四节）特意把这一步挪回 fp32。
  两版差多少，第五节给实测数字。

---

## 三、★ 无权重版：一个参数都没有

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
class Glm5NextTextUnweightedRMSNorm(nn.Module):
    def __init__(self, eps: float = 1.0e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)
```

**读法**：

- 类里**没有一个 Parameter**（实测：`sum(p.numel() for p in m.parameters()) == 0`），
  整个 forward 就是一行。
- `x.float()` 先把整块 x 升到 fp32，`rsqrt(...).to(x.dtype)` 再把缩放因子降回来，
  最后与**原始的 x**（注意不是 fp32 的那份）相乘。顺序与加权版不同，但式子是同一个：
  `x * rsqrt(mean(x^2) + eps)`。
- 它与加权版的关系可以一句话说清：**weight 全为 1 的加权版**。
  实测 `allclose(unweighted(x), weighted(x))` 为 True，并且
  `unweighted(x) * weight == weighted(x)` 逐元素成立。
  所以它不是"另一种归一化"，而是**同一种归一化去掉了一个自由度**。
- 名字里的 `Unweighted` 就是字面意思。它回答验收点一：**无权重版本用于"只需要尺度、
  不需要方向"的场合** —— 在 mHC 内部把 4 条流的拼接向量对齐到单位尺度，
  好让下一步的线性投影算出一致的混合系数。

它在主干里只被用了一次：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.input_norm = Glm5NextTextUnweightedRMSNorm(eps=config.rms_norm_eps)
```

**读法**：

- 全模型 90 个无权重 norm 全部来自这一个构造点（45 层 × `attn_hc` / `ffn_hc` 两个
  HyperConnection）。**它是 mHC 的内部零件，不对外暴露。**
- 归一化的对象是 `hidden_streams.view(batch_size, seq_len, -1)`：把 **4 条 4096 维的流拼成
  一条 16384 维**再算 RMS（回顾 L0-01 第五节：`hidden_states` 从嵌入之后就一直是
  `(batch, seq, 4, 4096)`）。流与流的边界在这里被抹平 —— 因为下一步
  `F.linear(flattened, self.fn)` 要的正是这个拼接向量。
- 为什么这里不该有权重 —— 这是本课最值得记住的一处推理：
  `flattened = self.input_norm(flattened)` 之后**只喂给一个线性映射**
  `F.linear(flattened, self.fn.float())`。而对任何线性映射 F 都有
  `F(w ⊙ x) = (F ∘ diag(w))(x)`，右边仍然是一个线性映射 ——
  也就是说**逐通道权重可以被完全吸收进 `self.fn` 自己的权重里，在这一处是数学上多余的**。
  对比主干的 `input_layernorm`：它的输出要同时喂给 q/k/v/门等多个下游，
  想吸收就得同时改每一个消费者 —— 那里保留权重更简单，也更符合训练惯例。
  ★ 判据：**当一次归一化的输出只面对一个线性映射时，weight 可以被吸收，因此不必要。**
- 那这个归一化到底在保证什么？**尺度不变性。** 实测：把同一条输入整体放大 100 倍，
  归一化后每个 token 的 RMS 仍是 `1.000000`，三组系数（pre/post/comb）逐位不变
  （pre/post 均值同为 `1.054601`，comb 行和恒为 1），只有 `collapsed` 跟着放大 100 倍。
  也就是说混合系数与残差流的绝对尺度**无关** —— 这对一条有 45 层的残差主干很重要。
- `pre` 会被用来把 4 条流压成一条（`collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2)`），
  所以归一化**间接**影响下一层的输入；但它本身不作为激活往下传。展开见 L3-06。
- 顺带一个被实测确认的边界：这条尺度不变性只在尺度远大于 `sqrt(eps)` 时成立。
  eps = 1e-5 时，把每条流的 RMS 压到约 1e-2，归一化后只剩 `0.9536`；
  压到 1e-3 只剩 `0.3020` —— **eps 正是这条不变性的失效门槛**（第五节展开）。
- `eps=config.rms_norm_eps` 是显式传入的 —— 默认值 1e-6 在这里被覆盖成 1e-5，第五节展开。

---

## 四、★ 门控版：KDA 的读出闸门

### 4.1 门从哪来：一个瓶颈

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.o_norm = Glm5NextTextRMSNormGated(self.head_dim, eps=self.layer_norm_epsilon)
```

**读法**：

- 门来自一个**瓶颈**：`hidden_size(4096) -> head_dim(128) -> qkv_dim(8192)`（实测
  `qkv_dim = head_dim * num_heads = 128 * 64 = 8192`）。先压到 128 再展开，
  参数量 4096×128 + 128×8192 ≈ 1.57M，比直接 4096→8192 的 33.5M 小 21 倍。
- `head_dim` 不是随手挑的中间宽度：它就是下面 `o_norm` 要归一化的那一维。
  **门控范数是逐头做的** —— 每个头的 128 维输出各自归一化、各自门控，
  64 个头再拼回 4096。普通 RMSNorm 是逐 token 的 4096 维，这里是逐 head 的 128 维。
- `eps` 来自 `self.layer_norm_epsilon`（= `config.rms_norm_eps`），不是类默认值。

### 4.2 门控版的 forward：多出来的两处 fp32

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
# NOTE: the FLA package does not re-cast to `input_dtype` in its implementation, maybe we should do the same
@use_kernel_forward_from_hub("RMSNormGated")
class Glm5NextTextRMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, **kwargs) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.activation = "sigmoid"

    def forward(self, hidden_states, gate=None) -> torch.Tensor:
        input_dtype = hidden_states.dtype

        # Strict FP32 norm (do not downcast on the weights)
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight.to(torch.float32) * hidden_states

        # Apply gating
        hidden_states = hidden_states * ACT2FN[self.activation](gate.to(torch.float32))

        return hidden_states.to(input_dtype)
```

**读法**：

- 状态部分与基准版几乎一样，只多了一行 `self.activation = "sigmoid"`。注意它是
  **硬编码的字符串**，不是 config 项 —— 想换成 silu 之类的门必须改代码。
- forward 与基准版的差别只有两处，但两处都是刻意的：
  1. `self.weight.to(torch.float32) * hidden_states`：权重乘法**留在 fp32**，
     而不是像基准版那样先降精度再乘。注释把理由写在了脸上：
     `Strict FP32 norm (do not downcast on the weights)`。
  2. `ACT2FN[self.activation](gate.to(torch.float32))`：门也升到 fp32 再取 sigmoid，
     最后**整个结果一次性** `.to(input_dtype)`。
- 为什么 KDA 值得多花这份精度：`o_norm` 是 KDA 层**唯一**一次把递推状态读出到主干的地方
  （见下一块），算错了没有第二次机会；而它每层只有 128 个权重，
  多留一份 fp32 权重几乎不花钱。**精度预算花在"回不去"的那一步。**
- 实测代价：把"在 fp32 里乘 weight"换成"先降 bf16 再乘"，
  1,048,576 个元素里有 **199,675 个（19.04%）**结果不同，最大相对差 **6.7e-3**
  （约 1.7 个 bf16 ulp）。这不是"差一点点"，是一次实打实的舍入。
- 上面那行 `# NOTE: the FLA package does not re-cast to input_dtype in its implementation,
  maybe we should do the same` 是作者留下的记号：**参考实现（FLA）的做法与这里不同，
  这里是有意不一致。**
- 还有一个只有实测才看得见的差别：如果 weight 留在 fp32 而激活是 bf16，
  基准版会因为类型提升**返回 fp32**，门控版则始终返回输入的 dtype
  （实测：`Glm5NextTextRMSNorm -> torch.float32`，`Glm5NextTextRMSNormGated -> torch.bfloat16`）。
  在多模态塔那种混合精度的路径上，这种 dtype 泄漏会被下游 matmul 放大成一次隐式升精度。

### 4.3 调用现场：为什么 KDA 的输出需要门控

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        # Final gated norm and proj
        gate = self.g_b_proj(self.g_a_proj(hidden_states)).view(hidden_shape)
        output = self.o_norm(core_attn_out, gate).reshape(batch_size, seq_len, -1)
        output = self.o_proj(output)
```

**读法**（这一段回答验收点二）：

- 三个张量的来源完全不同：`core_attn_out` 来自
  `chunk_kimi_delta_attention` / `recurrent_kimi_delta_attention`（**递推记忆的读出**），
  而 `gate` 来自**当前这一步的 `hidden_states`**。所以门控的含义是：
  **用当下的输入，决定要不要相信刚刚读出的记忆。**
- KDA 的递推形式是 `S_t = S_{t-1} ⊙ decay + write`（展开见 L4-02 ~ L4-05）。
  `core_attn_out` 是 `S` 与 query 的乘积，它的**尺度会随递推步数累积**：
  写得多、衰减得慢，读出就大。而 RMSNorm 只能把尺度拉回 1 ——
  它拉不回来的是"这一维该不该有值"：一个已经写坏或写满的通道，归一化之后照样输出单位尺度。
- 门控补的正是这一块：`sigmoid(gate)` 是 (0,1) 的逐通道系数，可以压到接近 0 ——
  相当于给每个通道一个"读出开关"。**基准版给不了这个能力，因为它的 weight 是与输入无关的常数。**
- 还有一层工程原因：门只依赖 `g_a_proj` / `g_b_proj`，与 KDA 的递推结果无关，
  因此可以和 QKV 投影并行算，不增加串行深度。
- 实测 init 下这个门长什么样：用真实的 `g_a_proj` / `g_b_proj` 形状与
  `initializer_range = 0.02` 初始化、输入 RMS 归一到 1，得到 gate 预激活 std ≈ 0.2997、
  `sigmoid(gate)` 均值 **0.5000**、范围 [0.2186, 0.8027]。
  也就是**初始化时门控范数约等于把输出整体乘 0.5**，这个恒定倍数被紧随其后的 `o_proj`
  吸收（它的权重同样是以 0 为均值初始化的）。门真正开始"挑通道"是训练之后的事 ——
  这也说明门控的价值不在初始化，而在长程训练里能不能学会关掉坏通道。

---

## 五、eps：类默认 1e-6，模型实际用 1e-5

先看两个实例化点（一个用位置参数、一个用关键字参数，但传的是同一个东西）：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        self.input_layernorm = Glm5NextTextRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Glm5NextTextRMSNorm(config.hidden_size, config.rms_norm_eps)
```

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Glm5NextTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Glm5NextTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.hc_head = Glm5NextTextHyperHead()
```

**读法**：

- 两处的 `eps` 都是 `config.rms_norm_eps`。实测把文件里**全部 12 个实例化点**列出来，
  无一例外都显式传了 `config.rms_norm_eps`（`input_layernorm`、`post_attention_layernorm`、
  出口 `norm`、`q_a_layernorm`、`kv_a_layernorm`、`o_norm`、mHC 的 `input_norm`）。
- 而三个类的 `__init__` 默认值都是 **1e-6**（第二、三、四节的三块引用里都能看到），
  `Glm5NextTextConfig.rms_norm_eps` 与 `Glm5NextVisionConfig.rms_norm_eps` 实测都是 **1e-5**
  （回顾 L1-02 / L1-03：这个字段就写在 config 里；`configuration_glm5_next.py`
  **不计入本课覆盖率**，只作为实测证据）。
- **结论：那句 `eps: float = 1e-6` 在这个模型里从来不生效，是防御性的默认值。**
  读源码看到 1e-6、读 config 看到 1e-5，两个都对 —— 真正跑起来的是 1e-5。

那么 1e-5 与 1e-6 的实际影响到底有多大？实测：

```text
eps 的物理含义：out = x * rsqrt(rms(x)^2 + eps)
    放大上限 = 1/sqrt(eps)      1e-5 -> 316.2 倍        1e-6 -> 1000 倍
    token 的 RMS 远大于 sqrt(eps) 时 eps 不起作用
                                sqrt(1e-5) = 3.162e-3    sqrt(1e-6) = 1.000e-3

实测：把 x 逐 token 归一到指定 RMS，比较 eps=1e-5 与 1e-6 的输出
    token RMS = 1           最大相对差 4.552e-06      eps 造成的收缩  0.0005%
    token RMS = 0.1         最大相对差 4.496e-04                     0.0500%
    token RMS = 0.01        最大相对差 4.178e-02                     4.6537%
    token RMS = 0.001       最大相对差 5.736e-01                    69.8489%
    token RMS = 3.2e-4      最大相对差 6.697e-01                    89.9321%
    token RMS = 1e-4        最大相对差 6.824e-01                    96.8393%

极限行为（4096 维常量输入，weight = 1）
    输入恒 = 1e-6  ->  输出 3.16e-4   （正好是 1e-6 x 316.2，即被放大到上限）
    输入恒 = 1e-4  ->  输出 3.16e-2
    输入恒 = 0     ->  输出 0         （0 乘任何数还是 0）

bf16（模型真正跑的精度）下 1,048,576 个元素里有多少不同
    Glm5NextTextRMSNorm      :   1,017 个（0.10%）
    Glm5NextTextRMSNormGated :     899 个（0.086%）
    fp32 下则是 1,048,576 个全不同（最大相对差 4.552e-06）
    参考：bf16 的相对精度是 2^-8 = 3.9e-3 —— 比 4.6e-6 大三个数量级
```

**读法**：

- eps 不是"防除零的魔法数"，它有明确的物理含义：**放大上限**。
  `rsqrt(rms^2 + eps)` 在 `rms -> 0` 时趋于 `1/sqrt(eps)`，所以 1e-5 把最坏情况下的放大倍数
  从 1000 压到 316。这是 1e-5 相对 1e-6 唯一的"安全收益"。
- 门槛是 `sqrt(eps)`：1e-5 → **3.162e-3**。只有当某个 token 的 4096 维 RMS 掉到这个值以下，
  eps 的选择才看得见。训练过的模型里 token RMS 通常在 0.1~1 量级（对应 0.0005% ~ 0.05%
  的差别），所以两种取值在正常情况下是"小数点后第四位"的事。
- 真正需要它兜底的是**近似全零的 token**：padding、被 mask 掉的视觉占位、
  或者某条残差流塌掉。这时 eps 决定它被放大到多少 —— 实测常量 1e-6 的输入在 1e-5 下输出
  3.16e-4、在 1e-6 下输出 1e-3。差三倍，但两者都是"接近 0"，不会爆。
  **真正会爆的是 eps = 0**，而这里没有这个选项。
- 一个 bf16 专属结论：bf16 的相对精度 3.9e-3 比两个候选 eps 的差别（4.6e-6）大三个数量级，
  所以**量化到 bf16 之后 99.9% 的元素完全一样**（实测 1017/1048576）。
  凡是"调 eps 能救回精度"的说法，在 bf16 推理里基本站不住：
  eps 在这里是数值防御，不是精度旋钮。

最后看一眼初始化：模型自己给这个自研 norm 补了一行初始化。

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        elif isinstance(module, Glm5NextTextRMSNormGated):
            init.ones_(module.weight)
```

**读法**：

- 这是 `Glm5NextPreTrainedModel._init_weights` 里唯一一条为 norm 写的分支。为什么需要它？
  因为 `nn.LayerNorm` 自带 `reset_parameters`，而这三个自研 norm 类**没有** ——
  它们的 weight 只在 `__init__` 里被 `torch.ones` 初始化一次，
  此后任何重新初始化（`_init_weights`、`from_pretrained` 之后重跑）都得模型自己照看。
- 只给门控版写、没给基准版写：门的 weight 是"会被重新挂 kernel、重新初始化"的那一个，
  而基准版在 `__init__` 里已经是 ones。
- **这一行本身就是"通用登记表不认识它"的证据** —— 下一节给实测。
  同理 `Glm5NextTextHyperConnection` 也被单独初始化（`fn` 取正态、`base` 置零、`scale` 置一），
  说明这个仓库对非标准模块一律"显式列出、自己负责"。

---

## 六、第四个类：与基准版逐字相同

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
@use_kernel_forward_from_hub("RMSNorm")
class Glm5NextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Glm5NextRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
```

**读法**：

- 实测：把这个类去掉类名与 docstring 之后的 **17 行**与 `Glm5NextTextRMSNorm` 逐字相同
  （`a == b` 为 True）。不是"很像"，是**一字不差**。
- 为什么会有两份：视觉塔不能继承文本的 norm 类（两者的 `rms_norm_eps` 来自不同的 config），
  而跨塔共享一个类又会让两边耦合；这个仓库的建模文件是**由 modular 文件生成的独立文件**，
  所以"复制一份、各自演化"是默认做法。
- 所以"三种 RMSNorm"要说得更精确：**四个类、三种形态**（加权 ×2、无权重、门控）。
  数类名是 4，数数学式子是 3。碰到这类问题，永远以式子为准。
- 这也解释了为什么本课的覆盖率里两个文件就够了：视觉塔的 norm 用法在 L2-07 已经讲过，
  这里只需要指出它与文本版**共享同一份实现**。

---

## 七、`pytorch_utils.py`：归一化在仓库基础设施里的两处痕迹

`pytorch_utils.py` 里没有一行 RMSNorm 代码，但它装着两类与本课直接相关的东西。

### 7.1 一张名存实亡的登记表

<!-- src: pytorch_utils.py -->
```python
ALL_LAYERNORM_LAYERS = [nn.LayerNorm]
```

**读法**：

- 这是 transformers 放"归一化层"这类全局登记表的位置：谁想知道"哪些模块是归一化层"，
  就查这张表（`Trainer` 的文档里就提到过它）。
- 实测（meta device 建一次完整的 `Glm5NextTextModel`）：表里只有 `nn.LayerNorm`，
  而模型里的三个 norm 类型与它的**交集为空**。也就是说
  **GLM-5 的 237 个 norm 在这张表里一个都没登记。**
- 后果不是"跑不起来"，而是"所有通用逻辑都照顾不到它"：权重初始化要自己写（第五节那行
  `init.ones_`）、weight decay 分组要自己列、量化与 kernel 融合要单独登记。
  **自研 norm 换来的是形状自由，付出的是"不在任何登记表里"的成本。**
- 读法层面的提醒：这个名字比它的内容大得多 —— 表里只有一个元素。
  看到"登记表"先数它有几个元素，再决定要不要按它推理。

### 7.2 分块：为什么 norm 可以，注意力不行

<!-- src: pytorch_utils.py -->
```python
def apply_chunking_to_forward(
    forward_fn: Callable[..., torch.Tensor],
    chunk_size: int,
    chunk_dim: int,
    *input_tensors,
) -> torch.Tensor:
```

<!-- src: pytorch_utils.py -->
```python
        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        # chunk input tensor into tuples
        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
        # apply forward fn to every tuple
        output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
        # concatenate output at same dimension
        return torch.cat(output_chunks, dim=chunk_dim)
```

**读法**：

- `apply_chunking_to_forward` 是仓库里"用时间换显存"的标准工具：沿 `chunk_dim` 把输入切成
  `num_chunks` 块，逐块调用 `forward_fn`，最后 `torch.cat` 回原维度。
- 它的**正确性前提**写在 docstring 里：`If the forward_fn is independent across the chunk_dim
  this function will yield the same result as directly applying forward_fn to input_tensors`。
  用本课的两个 norm 去验证这句话，结果一目了然：

```text
实测（batch=2, seq=8, hidden=64，chunk_size 使每维切 4 块）
    Glm5NextTextRMSNorm  沿 seq(dim=1) 分块 与 整块：逐位相同（最大绝对差 0.0）
    Glm5NextTextRMSNorm  沿 hidden(dim=-1) 分块    ：RuntimeError
                          —— weight(64) 与切出来的 16 维对不上，形状检查拦下了它
    无权重版             沿 hidden(dim=-1) 分块    ：不报错，但结果错了
                          —— 最大绝对差 0.896，相对差 27.8%
    沿 seq 做 softmax     再沿 seq 分块             ：结果错了（最大绝对差 0.888）
    Gated 版             沿 seq 分块（gate 一起切） ：逐位相同
```

- 这张表就是"一层里哪些算子可以分块"的判据：**RMSNorm 逐 token 独立（只在最后一维归约），
  所以沿 seq 切块完全等价**；而无权重版因为没有 weight，沿 hidden 切块时**形状检查拦不住**，
  会静默算错 —— "少一个参数"在这里换成了"少一道保险"。
- 反过来，凡是沿 seq 混合的算子（softmax、注意力、卷积），沿 seq 分块就是错的。
  回顾 L0-01：一层里的执行顺序是 mHC 混合 → 归一化 → 注意力 ——
  其中**唯一可以随便切、可以融合、可以逐块流式算的就是归一化这一步。**
- 还有一条接口细节：`forward_fn` 的参数个数必须等于张量个数（函数内部用
  `inspect.signature` 校验）。实测基准版的 `forward` 有 1 个参数、门控版有 2 个
  （`hidden_states, gate`），所以**门控版分块时要把 gate 一起切** —— 切了之后结果同样逐位相同。

---

## 八、与后面课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| 4 条流的 `pre` / `post` / `comb` 具体怎么算出来 | L3-06 |
| `swiglu_limit` 的 clamp 为什么和 eps 一样是"限制幅度" | L3-02 |
| KDA 的递推 `S_t = S_{t-1} ⊙ decay + write` 展开 | L4-02 ~ L4-05 |
| `gate` 的瓶颈投影为什么选 `head_dim` 做中间宽度 | L4-02 ~ L4-05 |
| `rms_norm_eps` 写在 config 的哪一行 | L1-02 / L1-03 |
| 归一化在张量并行下要不要跨卡求和 | L8 |
| `use_kernel_forward_from_hub` 换掉的 forward 长什么样 | L8 |

**一句话总结**：

> 主干里的 237 个 norm 只有三种形态：**加权**（要方向自由度）、**无权重**（只要尺度）、
> **门控**（要在逐通道上把 KDA 读出的记忆关掉）；
> 类默认的 eps=1e-6 从不生效，真正跑的是 config 里的 **1e-5** ——
> 它决定的是"最坏情况下放大多少倍"（316 倍 vs 1000 倍），
> 在 bf16 里两种取值的差别有 99.9% 的元素根本看不见。
