<!-- glm5-coverage
models/glm5_next/modeling_glm5_next.py
-->

# L3-06 · ★ mHC：流形约束超连接 — 源文件

**这一课只回答一个问题：`x = x + f(norm(x))` 里的那条 `x`，为什么会变成 4 条；
以及这 4 条互相混合时，凭什么保证数值不被撑爆。**

视角：把 `Glm5NextTextHyperConnection` 当成一台小机器 —— 它读进 4 条流，
吐出 3 个权重（`pre` / `post` / `comb`），其中 `comb` 必须先在 Sinkhorn 里
先在 Sinkhorn 里迭代 20 次才允许出场。

| 文件 | 行数 | 本课引用块 |
|---|---|---|
| `models/glm5_next/modeling_glm5_next.py` | 2444 | 11 |

> **实测订正（作业书 vs 实测）**：作业书写这个文件 2445 行。实测 `wc -l` = 2444、
> `str.splitlines()` 也是 2444，文件以换行结尾、最后一行是 `]` —— 所以是 **2444 行**。
> 本课一律以实测为准；作业书给的行号大纲（L220 / L273 / L287 / L334 …）与实测一致。

**实测的 mHC 常量**（用 `_data/recon/probe_l306.py` 跑出来，不是从注释抄的）：

```text
Glm5NextTextConfig() 默认值              一个 HyperConnection 的结构
    hc_mult            = 4                   fn     (24, 16384)      393,216 个参数
    hc_sinkhorn_iters  = 20                  base   (24,)                   24
    hc_eps             = 1e-6                scale  (3,)                     3
    hidden_size        = 4096                合计                    393,243 个参数
    rms_norm_eps       = 1e-5
                                             每层两个（attn_hc / ffn_hc）
                                             45 层 -> 35,391,870 个参数

24 = (2 + N) * N = (2 + 4) * 4     <-  pre(4) + post(4) + comb(4*4)
```

> 上面这段是**示意性的汇总**，用 `text` 块标出 —— 它不是源文件的逐字引用，
> 因此不参与保真校验。下面每一段 `python` 块都是逐字引用。
>
> 本课还引用了工装脚本 `_data/recon/probe_l306.py` 的实测输出（写在 `text` 块里）。
> **脚本不计入覆盖率** —— 覆盖域只统计 `src/transformers/**/*.py`。

---

## 一、★ 三步：读取 → 收拢 → 写回

一句话先给结论：mHC 做三件事，而且**每一步都发生在解码层里**：

1. **读取**：4 条流 → 摊平 → 一次线性投影得到 24 个数 → 切成 `pre`(4) / `post`(4) / `comb`(16)；
2. **收拢**：用 `pre` 把 4 条流压成 1 条，喂给注意力或 MLP（子层完全不知道有 4 条流）；
3. **写回**：`post ⊙ 子层输出 + combᵀ · 旧流` —— 4 条流各自收回一份子层输出，再互相混合。

三步的第三段在源码里就是这 4 行（截自解码层 `forward`）：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        residual = hidden_states
        post, comb, hidden_states = self.attn_hc(hidden_states)
        # Self attn
        hidden_states = self.input_layernorm(hidden_states)
```

**读法**：
- 第一行把旧的 4 条流原封不动留了一份 `residual` —— 它**不参与**下面的归一化和注意力，
  而是留到写回时参与混合。教科书里的 `x + f(norm(x))`，在这里是"旧流被混合后才回来"。
- `attn_hc` 一次返回三个东西，顺序是 `(post, comb, collapsed)`：
  第三个是收拢后的单条流，直接顶替了原来 `hidden_states` 的位置。
- 注意 `post` / `comb` 是从**旧流**算出来的（`attn_hc(hidden_states)` 的输入就是旧流），
  而不是从子层输出算出来的 —— 门控看的是"进门时的样子"。
- 下一站是 `self.input_layernorm`，归一化作用在**收拢后的 1 条流**上，不是 4 条上。

---

## 二、读取：摊平 → 无权重范数 → 一次投影

`forward` 的前半段。整台机器只有**一次**矩阵乘法，行数固定 24：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        batch_size, seq_len = hidden_streams.shape[:2]
        hc = self.hc_mult

        # Flatten and norm the hidden streams
        flattened = hidden_streams.view(batch_size, seq_len, -1).float()
        flattened = self.input_norm(flattened)
        # Mix the streams together to infer the weight coefficients
        flattened = F.linear(flattened, self.fn.float())
        # Split the weight coefficients
        pre_w, post_w, comb_w = flattened.split([hc, hc, hc * hc], dim=-1)
```

**读法**：
- `view(batch_size, seq_len, -1)` 把 `(B, S, 4, D)` 摊成 `(B, S, 4D)`：4 条流被**首尾相接**，
  不是相加。所以这条投影能看见"哪条流在哪个位置"，而不是只看它们的和。
- `.float()`：整台机器在 fp32 里算权重。这很关键 —— Sinkhorn 要做 39 次除法，
  在 bf16 下累加误差会直接吃掉双随机性。
- 归一化用的是 `Glm5NextTextUnweightedRMSNorm`，**它没有 weight**：

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
- 对比一下 `Glm5NextTextRMSNorm`：那个类里有一行 `self.weight = nn.Parameter(torch.ones(hidden_size))`，
  这个类里**没有**。全文件里它只被 mHC 用一次。
- 为什么故意去掉权重：这里的范数不是"给模型用的特征归一化"，而是**给权重生成器用的输入整形**。
  如果带可学习权重，训练早期权重缩放会直接改变 Sinkhorn 的输入尺度，双随机的收敛速度跟着变。
  去掉权重，等于把"生成权重"这条支路的数值稳定性钉死。
- `eps` 用的是 `config.rms_norm_eps`（默认 `1e-5`），**不是** `hc_eps`（`1e-6`）——
  两个 eps 各管一段，别混。
- 投影 `F.linear(flattened, self.fn.float())`：`fn` 的形状是 `(24, 4D)`，
  也就是 `(2 + N) * N` 行 —— 行数这个表达式本身就是"三段权重"的声明。

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
    def __init__(self, config: Glm5NextTextConfig):
        super().__init__()
        self.hc_mult = config.hc_mult  # number of streams, referred as N below
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.input_norm = Glm5NextTextUnweightedRMSNorm(eps=config.rms_norm_eps)
        # The mHC projects the N inputs streams into 3 weights: pre (size: N), post (size: N) and comb (size: N*N)
        # Hence the output size of the projection is 2 * N + N * N = (2 + N) * N.
        concatenated_weights_size = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(concatenated_weights_size, self.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.empty(concatenated_weights_size))
        # The mHC produces 3 outputs, each with their own scale parameter (the "pre", "post" and "comb" weights)
        self.scale = nn.Parameter(torch.empty(3))
```

**读法**：
- `hc_mult` / `hc_sinkhorn_iters` / `hc_eps` 三个常量都从 config 抄进来，
  所以"4 条流""20 次迭代"这两个数字是**配置驱动的**，不是写死在代码里。
- `concatenated_weights_size = (2 + self.hc_mult) * self.hc_mult`：
  实测 `N=4` 时为 `6 * 4 = 24`。24 个数刚好够切出 `pre(4) + post(4) + comb(16)`。
- `scale` 是 `(3,)` 而不是 `(24,)`：三个出口**各有一个可学习的整体缩放**，
  这是给"三个出口的量纲本来就不同"留的自由度。
- 实测参数量：`fn` 393,216 + `base` 24 + `scale` 3 = **393,243**；每层两个、45 层共
  **35,391,870** 个参数。mHC 不是"零成本的残差改造"。

---

## 三、三个出口：三种激活，三种值域

`pre` / `post` / `comb` 用的是**三种不同的**激活函数。这不是随手写的：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)

        comb_w = comb_w.view(*comb_w.shape[:-1], hc, hc)  # these are matrix weights, unlike pre or post
        comb_b = comb_b.view(hc, hc)

        # All weights are computed with a one layer perceptron. For pre and post, this is it.
        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)
        comb = torch.softmax(comb_w * comb_scale + comb_b, dim=-1) + self.hc_eps
```

**读法**：
- `pre = sigmoid(...) + hc_eps` → 值域 `(0, 1 + eps)`，**并且没有归一化**。
  实测一次真实前向（`hidden_size=8`）：`pre = [0.5677, 0.4582, 0.5149, 0.4712]`，
  和是 **2.011953**，不是 1。所以收拢**不是加权平均**，而是"每条流最多贡献 1 倍"的门控求和。
  这一点很容易凭直觉写错（"收拢嘛，肯定是加权平均"）—— 见 §六。
- `post = 2 * sigmoid(...)` → 值域 `(0, 2)`，**中心在 1**。
  实测初始化（`fn ~ N(0, 0.02)`、`base = 0`、`scale = 1`）后 `post ∈ [0.9152, 1.1225]`。
  中心取 1 的意义：初始状态下"每条流写回的就是子层原始输出"，mHC 退化成标准残差。
- `comb = softmax(..., dim=-1) + hc_eps` → 行和为 1 的**行随机**矩阵。
  到这里为止 `comb` 只是"每行一个概率分布"，**还不是**双随机 —— 下一节才是重头戏。
- 三个 `+ hc_eps` 的作用是把 `sigmoid` / `softmax` 永远够不到 0 这件事补上：
  Sinkhorn 里每一次除法，分母都可能出现 0。`1e-6` 就是那个保险丝。

---

## 四、★ Sinkhorn：20 次迭代把 `comb` 压成双随机

先看作者自己写的理由（注释是源码的一部分，逐字引用）：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        # The comb weight is a bit different: it dictates how the input streams (In) are added to the output streams
        # (Out) in this way: Mixed = In @ Comb + Out. To make sure the norm of "Mixed" does not blow up, we constrain
        # the comb weight to be doubly-stochastic (ie. its rows and columns must sum to 1) with a few iterations of the
        # Sinkhorn-Knopp algorithm, which iteratively normalizes the rows and columns to sum to 1.
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
```

**读法**：
- 注释里那句话是关键判据：「**To make sure the norm of "Mixed" does not blow up**」。
  约束双随机**不是**为了"让权重总和为 1"好看，而是为了让混合后的**范数不涨**。
- 数一下归一化的次数（实测）：第 323 行是**一次列归一**；循环体跑
  `hc_sinkhorn_iters - 1 = 19` 次，每次"行归一 + 列归一"。
  所以 20 次迭代实际是 **19 次行归一 + 20 次列归一 = 39 次除法**，
  而不是 20 次"行+列"往返。配置名是 `hc_sinkhorn_iters`，别把它当成除法次数。
- **顺序反常**：教科书上的 Sinkhorn 一般从行归一出发，这里第一步偏偏是
  `dim=-2`（列）。原因在上一行 —— `comb` 刚从 `softmax(dim=-1)` 出来，**行已经和是 1**，
  再归一次行是白做；先补列，才是往双随机走得最快的一步。
- 收敛速度是实测的（`hidden_size=8`，`fn ~ N(0,0.02)` 初始化）：
  见下面的表。`1e-6` 那个平台就是 `hc_eps` 造成的下限 —— 它不会收敛到绝对 0。

```text
hc_sinkhorn_iters    max|行和-1|     max|列和-1|
        1             3.393e-03      1.132e-06
        2             3.833e-05      1.073e-06     <- 2 次迭代已经到 1e-5 量级
        3             1.431e-06      1.073e-06
        5             1.013e-06      1.013e-06
       20             1.013e-06      1.013e-06     <- 平台 = hc_eps = 1e-6

注：iters=1 时列和已经是 1e-6，因为第 323 行的列归一总是执行；
    "行"那一列要靠循环体。这也印证了 §四 第二条"顺序反常"。
```

---

## 五、★ 手算：2 次 Sinkhorn 迭代

取一个 3×3 的行随机矩阵（行和为 1），列和故意取 `0.5 / 1.0 / 1.5`，方便手算。
`hc_eps` 在手算里忽略（它带来的偏差是 `1e-6` 量级）。

参与运算的三行代码就是（从上一块里单独切出来）：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
```

**读法**：
- 第一步（L323）除以**列**和：第 0 列除以 0.5、第 2 列除以 1.5，第 1 列本来就是 1 不用动。
  做完之后**列和精确等于 1**，但行和被打乱。
- 之后每一轮：先按行归一（行和回到 1），再按列归一（列和回到 1），列归一永远在最后。
  所以"迭代 k 次后"的状态是：**行和 ≈ 1、列和 = 1（精确）**。
- 手算表（数值由 `probe_l306.py` 用源码同款公式算出，逐位可复现）：

```text
M0（行随机；列和故意取 0.5 / 1.0 / 1.5）
    0.30  0.30  0.40          行和 1.000000 / 1.000000 / 1.000000
    0.15  0.40  0.45          列和 0.500000 / 1.000000 / 1.500000   <- 不双随机
    0.05  0.30  0.65

[L323] 列归一（第 0 列 ÷0.5，第 2 列 ÷1.5，第 1 列不动）
    0.60  0.30  0.266667      行和 1.166667 / 1.000000 / 0.833333   <- 行被打乱
    0.30  0.40  0.300000      列和 1.000000 / 1.000000 / 1.000000   <- 列精确为 1
    0.10  0.30  0.433333

[L325-326] 迭代 1（先行归一，再列归一）
    0.550459  0.252809  0.217984   行和 1.021251 / 1.000463 / 0.978286
    0.321101  0.393258  0.286104   列和 1.000000 / 1.000000 / 1.000000
    0.128440  0.353933  0.495913

[L325-326] 迭代 2
    0.543763  0.246952  0.212103   行和 1.002819 / 1.000086 / 0.997095
    0.323786  0.392130  0.284170   列和 1.000000 / 1.000000 / 1.000000
    0.132450  0.360918  0.503727

同一矩阵跑满 20 次迭代（源码默认值）
    0.542732  0.246059  0.211208
    0.324192  0.391944  0.283864   max|行和-1| = 0    max|列和-1| = 0
    0.133076  0.361996  0.504928   <- 不是均匀的 1/3，而是"最近的双随机矩阵"
```

**读法（续）**：
- 行和误差的下降序列（上表逐位可读）：列归一后 `1.667e-01` → 迭代 1 后 `2.175e-02`
  → 迭代 2 后 `2.819e-03`。每轮大约掉一个数量级；真模块里
  `iters=1/2/3` 对应 `3.4e-3 → 3.8e-5 → 1.4e-6`，同一个量级节奏。
- 注意 20 次迭代后的矩阵**并没有变成均匀的 1/3**：它收敛到"离 M0 最近的双随机矩阵"，
  不是均匀矩阵。Sinkhorn 是在做**投影**，不是在做平均。
  这一点很重要：`comb` 保住的是"哪条流跟哪条流亲"的信息，被抹掉的只有"总量不守恒"。

---

## 六、收拢：`pre` 是门控，不是平均

`pre` 只在这一行被消费，随后**不返回**：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        # Since "pre" is meant to be used with the input streams (available here as `hidden_streams`), we collapse the
        # streams here and return `collapsed` tensor, which will be the input for the next attention or MLP block.
        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
        return post, comb, collapsed
```

**读法**：
- `(pre.unsqueeze(-1) * hidden_streams).sum(dim=2)`：`pre` 形状 `(B, S, 4)`，
  加一维后与 `(B, S, 4, D)` 逐流相乘，再在第 2 维求和 → `(B, S, D)`。
- 因为 `pre ∈ (0, 1+eps)` 且**没有归一化**，这个收拢是"每条流最多按 1 倍加权求和"，
  实测权重和是 2.01。**它不是凸组合** —— 收拢后的幅值可以比任何一条流都大。
  这与 `post` 中心取 1 是同一个思路：mHC 想要的是"可控的缩放"，不是"守恒的平均"。
- 返回 `(post, comb, collapsed)`：文档字符串里那句
  「All weights are returned except "pre", which is consumed here.」说的就是这件事。
- 与 §九 的 `HyperHead` 对照着看：`pre` 是**逐 token 学出来的** 4 个门控值，
  `hc_head` 是**全局固定的等权平均**。两者都"把 4 条流收成 1 条"，但一个是算子、一个是出口。

---

## 七、写回：`post ⊙ 输出 + combᵀ · 旧流`

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        hidden_states = post.to(dtype).unsqueeze(-1) * hidden_states.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual
        )
```

**读法**：
- 两项相加，形状都是 `(B, S, 4, D)`：
  - `post.unsqueeze(-1) * hidden_states.unsqueeze(-2)`：子层输出是**一条** `(B,S,D)`，
    加一维变成 `(B,S,1,D)`，乘上 `post` 的 `(B,S,4,1)`，广播成 4 条 —— 每条流拿到同一份
    子层输出，各自乘自己的门控值。实测 `post ∈ (0, 2)`，中心 1。
  - `torch.matmul(comb.transpose(-1, -2), residual)`：把 4 条旧流按 `comb` 混合。
    注意这里乘的是 `combᵀ`；写成 `In @ Comb` 也一样（`(In @ Comb)[i] = Σ_j In[j]·Comb[j,i]`），
    两者逐元素实测相同（`allclose = True`）。
- 实测：`comb` 双随机时 `sigma_max(comb) = 0.999999`，
  一次真实采样的 `||combᵀ · residual|| / ||residual|| = 0.5875` —— 混合这一步在**收缩**，不在放大。
- 别漏了：这 3 行在解码层里出现**两次**（注意力后一次、MLP 后一次）。
  每层两个 `Glm5NextTextHyperConnection`：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        self.attn_hc = Glm5NextTextHyperConnection(config)
        self.ffn_hc = Glm5NextTextHyperConnection(config)
```

**读法**：
- `attn_hc` 与 `ffn_hc` 是**两个独立实例**、两套独立权重（每层 786,486 个参数）。
  它们不共享参数，也不共享 Sinkhorn 的结果。
- 回顾 L0-01：那里看到 `hidden_states` 从一开始就是 4 维；到了这里才知道多出来的那一维
  在每一层里被读了两次、写了两次。

---

## 八、★ 为什么双随机能稳住梯度流

这一段没有代码，全是实测数字（`probe_l306.py` 的第 ⑤ 节）。结论先说：
**行随机允许"所有流被同一个方向放大"，双随机不允许。**

```text
对 200 个随机 4x4 矩阵测 sigma_max（最大奇异值 = 单层最坏放大倍数）
   全部只做 softmax(dim=-1)（行随机）:  max 1.3017   mean 1.0712
   再做 Sinkhorn 到双随机:             max 0.999999 mean 0.999999

退化例（4 行完全相同，p = [0.70, 0.20, 0.06, 0.04]，行随机完全允许）
   行和 = 1.0000   列和 = [2.8, 0.8, 0.24, 0.16]
   sigma_max = 1.4631   <- 恰好等于 ||p||_2 * sqrt(N) = 0.7348 * 2
   Sinkhorn 20 次后 -> 全 0.25 的均匀矩阵，sigma_max = 0.999999

45 层的界（实测单层最坏值外推）:  1.3017^45 = 1.421e+05
双随机:                          1.0000^45 = 1.000000
45 层随机链连乘的 sigma_max（实测）: 行随机 1.0351 / 双随机 1.0000

双随机的两条不变量（真模块输出，实测）
   comb @ 1  与 1 的最大偏差 = 1.043e-06      （每一"行"贡献守恒）
   1^T @ comb 与 1 的最大偏差 = 1.028e-06     （每一"列"接收守恒）
   sigma_max(comb) = 0.999999，sigma_max(comb^T) = 0.999999
   comb 最小元素 = 2.021e-01（没有流被饿死）
```

**读法**：
- **行和 = 1** 保证"每条输入流的总贡献是 1"：混合是加权平均，不放大总和。
- **列和 = 1** 保证"每条输出流接收的总权重是 1"：没有任何一条输出流靠别人喂饱，
  也没有任何一条输出流被整体放大。上面那个"4 行完全相同"的退化例就是反证 ——
  行随机完全允许它，此时 `sigma_max = 1.46`，即存在一个方向每层被放大 46%。
- **界 vs 实际**：45 层随机链的实测 `sigma_max` 是 1.0351，并没有真的爆掉；
  但"最坏情况界"是 `1.3017^45 ≈ 1.4e5`。训练要的是**界**，不是平均表现 ——
  因为反向传播里的梯度会主动去找那个最坏方向。双随机把这个界钉在 1。
- **`combᵀ` 也是双随机**（实测两者 `sigma_max` 都是 0.999999）：
  前向用 `combᵀ`、反向用 `comb`，两个方向用的是同一个"不放大"的算子。
  这就是"双随机约束对梯度流的稳定作用"的完整表述。
- 顺带一个附赠品：`comb` 的最小元素实测 0.2021，远离 0。
  softmax 之后再加 Sinkhorn，不会产出"某条流永远收不到输入"的稀疏矩阵。

---

## 九、出口：`HyperHead` 是 0 参数的等权平均

主干跑完 45 层后，手上还是 4 条流。收尾这一步出人意料地简单：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
class Glm5NextTextHyperHead(nn.Module):
    """Final GLM-5.3-Flash HC-stream collapse. Unlike DeepSeek-V4, this is an unweighted mean."""

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        return hidden_streams.mean(dim=2)
```

**读法**：
- 整个类**没有任何子模块、没有任何 Parameter**。实测
  `sum(p.numel() for p in h.parameters()) == 0`，且 `torch.allclose(h(x), x.mean(dim=2))` 为 `True`。
- 与 §六 的 `pre` 对照：`pre` 是逐 token 的 sigmoid 门控（和 ≠ 1），
  `hc_head` 是全局固定的等权平均（和恒为 1）。**同一个"收拢"动作，入口处学、出口处不学。**
- 回顾 L0-01 的实测订正：那里已经发现 `hc_head` 不是"学习一组混合权重"。
  本课补上了它的另一半 —— `pre` 才是那个带权重的地方，而且它连归一化都没有。
- docstring 里 `Unlike DeepSeek-V4` 这句是作者留下的对比记号：
  上一代用带权重的收拢，这一代在出口处主动去掉了权重。

---

## 十、把三步连起来

```text
  hidden_streams (B, S, 4, 4096)          ← 由 L0-01 的 unsqueeze(2).expand 造出来
        │
        │  ① 读取：view(B,S,-1) → UnweightedRMSNorm(fp32) → fn (24 x 16384)
        ▼
  24 个数 ──split([4,4,16])──► pre(4)  post(4)  comb(4,4)
        │                        │         │         │
        │                        │         │         └─ 20 次 Sinkhorn（1 列 + 19×(行,列)）
        │                        │         │            → 双随机，sigma_max = 1
        │                        ▼         │
        │   ② 收拢：(pre.unsqueeze(-1) * streams).sum(2)  → (B, S, 4096)
        │                        │
        │                  ┌───────────────┐
        │                  │ ATTN 或 MLP   │   ← 子层只看得到 1 条流
        │                  └───────────────┘
        │                        │  (B, S, 4096)
        ▼                        ▼
   ③ 写回：new = post.unsqueeze(-1) * out.unsqueeze(-2) + combᵀ · residual
        │
        ▼
  hidden_streams (B, S, 4, 4096)   → 下一层；每层做两次（attn_hc / ffn_hc）

  45 层之后：HyperHead = mean(dim=2)  →  (B, S, 4096)  →  RMSNorm  →  lm_head
```

---

## 十一、与后面课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| `hidden_states` 为什么一开始就是 4 维 | L0-01（已讲） |
| 子层内部（KDA / MLA）怎么消费那 1 条流 | L3-08、L4 整层 |
| 每层两个 mHC 与注意力 / MoE 的装配顺序 | L3-08 |
| `mlp` 那一支里共享专家怎么合流（收拢后的流去哪） | L3-05 |
| `norm(hc_head(x))` 之后到 logits | L6-01 ~ L6-03 |
| 45 层的残差流在缓存 / 推理时怎么被保存 | L5 整层 |

**一句话总结**：

> mHC 把残差从"一条和"改成"4 条流 + 一张混合矩阵"：
> **读取**一次投影出 24 个数，**收拢**用未归一化的 `pre` 门控，**写回**用 `post ⊙ out + combᵀ · 旧流`；
> 而 `comb` 必须先被 Sinkhorn 压成双随机 —— 因为只有行列和都为 1，才能把 45 层的放大倍数钉死在 1。
