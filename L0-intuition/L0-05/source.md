<!-- glm5-coverage
models/glm5_next/modeling_glm5_next.py
-->

# L0-05 · mHC：4 条残差流 — 源文件

**这一课只讲一件事：GLM-5 的残差不是一条线，而是一束 4 条；这 4 条怎么混合，被一个双随机矩阵管住。**

视角：`hidden_states` 的形状为什么是 4 维、4 条流在一层里被读写几次、
以及"混合"这一步为什么非要做 20 次 Sinkhorn 迭代。

| 文件 | 行数 | 本课引用块 |
|---|---|---|
| `models/glm5_next/modeling_glm5_next.py` | 2444 | 12 |

**实测数字**（本机 CPU，`hc_mult=4` 的真实尺寸模块 + 一个 `hidden=64` 的极小模型；
这是**脚本输出**，不是源文件引用，不计入覆盖率）：

```text
默认 config：hc_mult = 4     hc_eps = 1e-6     hc_sinkhorn_iters = 20
             hidden_size = 4096               num_hidden_layers = 45

Glm5NextTextHyperConnection 的投影（hidden=4096, hc_mult=4，实测形状）：
    fn     (24, 16384)      = (2 + N) * N 行 × (N * D) 列
    base   (24,)            scale (3,)
    参数量 393,243 / 个  →  每层 2 个 = 786,486  →  45 层共 35,391,870

一次前向里实测到的形状（极小模型 hidden=64, hc_mult=4, batch=1, seq=12）：
    embed_tokens 输出    (1, 12, 64)        ← 3 维
    主干 hidden_states   (1, 12, 4, 64)     ← 4 维，多出来的一维就是 hc_mult
    attn_hc 的返回       [(1,12,4), (1,12,4,4), (1,12,64)]   = post / comb / collapsed
    layer0 的输出        (1, 12, 4, 64)
    hc_head 的输出       (1, 12, 64)
    logits               (1, 12, 256)
真实 config 把每个 64 换成 4096：主干就是 (batch, seq, 4, 4096)。

HyperHead：0 个参数，输出恒等于 hidden_streams.mean(dim=2)（allclose 已验）。
Sinkhorn 之后行和 / 列和都停在 0.999999（偏差 9.5e-7），不是精确的 1 —— 分母带着 + hc_eps。
第 0 层之后 4 条流就不再相同（随机初始化下最大差 2.9e-3，第 7 层涨到 4.6e-3）。
```

---

## 一、★ 三次出现：展开 → 层内两次 → 收拢

先把主干读完。**"mHC 是什么"这个问题，在主干里只有三行代码的答案**：
一行展开、一行循环（每层内部各有两次）、一行收拢。

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()

        topk_indices = None
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states, topk_indices = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                # Key change using NoPE
                position_embeddings=None,
                input_ids=input_ids,
                past_key_values=past_key_values,
                prev_topk_indices=topk_indices,
                **kwargs,
            )

        hidden_states = self.norm(self.hc_head(hidden_states))
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)
```

**读法**：
- **第 1495 行是全文唯一把 `hidden_states` 变成 4 维的地方。** `expand(-1, -1, hc_mult, -1)`
  把 `(batch, seq, 4096)` 变成 `(batch, seq, 4, 4096)`。注意 `expand` **不复制数据**：
  它返回一个在流那一维 stride = 0 的视图（实测 `data_ptr` 与输入相同）；`.contiguous()`
  才真的把它铺成 4 份（实测 stride 从 `(15, 5, 0, 1)` 变成 `(60, 20, 5, 1)`）。
- **第 1499 行的循环里，层与层之间传递的 `hidden_states` 一直是 4 维的。**
  也就是说：这个模型里根本没有"一条残差"这回事。教科书上的 `x = x + attn(norm(x))`
  在这里是"4 条流各自混合后再写回 4 条流"。
- **第 1511 行是出口**：先 `hc_head` 把 4 条收成 1 条，再 `norm`。顺序是有意义的 ——
  先收拢再归一化，等价于"对平均后的向量做一次归一化"；反过来则是"每条流各自归一化后再平均"，
  两者数值不同（RMSNorm 是逐位置缩放，缩放系数不一样）。
- 三者合起来就是本课的地图：**展开（1 次，全模型唯一）→ 层内读写（45 × 2 次）→ 收拢（1 次）**。
  第一幕动画画的就是这条线。

**为什么值得单独开一课**：L0-01 只来得及说"多出来的一维是 mHC"。
但只要你要读 L3-06（残差与流的精读）、L5（KV cache 的形状）或 L7（显存），
就必须先知道 4 条流是怎么被读、怎么被写回去的 —— 否则每一次看到 `post` / `comb`
都要停下来重新推。

---

## 二、形状契约：源码作者自己把它写成了文档

这一段是 `Glm5NextTextHyperConnection` 的类声明加文档字符串的第一段，
外加它内部用的那个"无权重" RMSNorm。**先看这三件事：形状、谁被收拢、谁被展开。**

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
class Glm5NextTextUnweightedRMSNorm(nn.Module):
    def __init__(self, eps: float = 1.0e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)


class Glm5NextTextHyperConnection(nn.Module):
    r"""
    A module to implement manifold-constrained Hyper-Connections (mHC) (Xie et al., 2026) which strengthens the
    conventional residual connections between adjacent Transformer blocks.

    When using mHC, each token is projected onto `hc_mult` streams, so the shape of decoder layer inputs changes from
    [batch_size, sequence_length, hidden_size] to [batch_size, sequence_length, hc_mult, hidden_size].
    To keep the same input shape for attention or MLP blocks, the streams are collapsed into one upon entering a block,
    and expanded back into `hc_mult` streams upon exiting. There is also a weighted residual connection between the
    input and output streams.
    The weights used for collapsing (pre), expanding (post) and mixing (comb) are computed from the `hc_mult` input
    streams through a learned projection (plus Sinkhorn-Knopp algorithm for the comb weight).
```

**读法**：
- 类名里的 `Unweighted` 不是形容词凑数：这个 RMSNorm **没有可学的 gain**。
  对照 `Glm5NextTextRMSNorm.__init__`（本文件第 67 行起）里的
  `self.weight = nn.Parameter(torch.ones(hidden_size))` —— 那个有 gain。
  mHC 内部这条归一化只用来**校准投影的输入尺度**，不需要逐维增益；
  如果给它 gain，它就会和后面的 `fn` 投影争抢"谁来定尺度"。
- 文档字符串第 225-229 行把契约写死了：形状从 `[batch_size, sequence_length, hidden_size]`
  变成 `[batch_size, sequence_length, hc_mult, hidden_size]`；
  **进子层前 collapse 成一条，出子层后 expand 回 `hc_mult` 条**。
  所以"注意力 / MLP 看到的是 3 维张量"这句话是对的 —— 它们看到的是 collapse 之后的那一条。
- **最容易误解的一点**：4 条流在初始时刻**完全相同**（`expand` 出来的是同一份数据）。
  它们从第 0 层写回开始才分叉 —— 分叉的唯一入口是逐流的 `post` 门控（推导见第九节）。
  实测：第 0 层之后 4 条流的最大差 2.9e-3，第 7 层涨到 4.6e-3（随机初始化的极小模型）。
- `hc_mult` 是"每个 token 被投影到几条流"，实测默认值 4。
  这是一个**纯结构性**的数字：它决定了主干里每一个张量的维度，也决定了下面所有投影的形状。

---

## 三、文档里自带的那张图

上一段的文档字符串下面，作者直接画了数据流图。**这张图比任何二手解释都准**，所以原样引进来：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
    The diagram below shows the flow of the mHC streams (B = batch_size, S = seq_length, N = hc_mult, D = hidden_size):

                                                  ┌───────────────────┐
                             N input streams  ────│ FLATTEN + PROJECT |────> (pre, post, comb) weights
                              [B, S, N, D]        └───────────────────┘      ([B, S, N],  [B, S, N],  [B, S, N, N])
                                  │ │ │
               ╭──────────────────┴─┼─┼──────────────────╮
               │ ╭──────────────────┴─┼────────────────╮ │
               │ │ ╭──────────────────┴──────────────╮ │ │
               │ │ │                                 │ │ │
```

**读法**：
- 图里 `FLATTEN + PROJECT` 的箭头指向的是 **`(pre, post, comb)` 三组权重，不是 hidden**。
  这是 mHC 与普通残差最根本的区别：普通残差没有"权重"这一步，
  而 mHC 每走一步都要先用**当前的流**算出"这一步该怎么混"。
- 图上的 `│ │ │` 是 3 条竖线代表 N 条流；真实的 N = 4。看源码注释时不要把 3 当成常量。
- 注意箭头方向：权重是从**输入流**算出来的，不是从输出算的。
  这意味着"怎么混"和"混什么"互相依赖，反向传播时要同时穿过两条路 ——
  这就是第九节里"梯度有 4 条通路"的图面来源。

---

## 四、`__init__`：一个 24 × 16384 的投影

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
- `concatenated_weights_size = (2 + self.hc_mult) * self.hc_mult` 这个式子值得背：
  `pre` N 个 + `post` N 个 + `comb` N² 个 = `(2 + N) * N`。N = 4 → 24。
  实测 `fn` 的形状正是 `(24, 16384)`，其中 16384 = `hc_mult * hidden_size` = 4 × 4096。
- `fn` 是唯一"重"的参数：24 × 16384 = 393,216。加上 `base` 的 24 个、
  `scale` 的 3 个，**每个 HyperConnection 实测 393,243 个参数**。
  每层两个 → 786,486；45 层共 35,391,870（≈ 35.4M）。
  对比 `hidden_size=4096`、`vocab_size=154880` 的嵌入表（6.3 亿），它很小；
  但它**不是零**，这也是"mHC 比单条残差贵"的第一笔账。
- `self.scale` 只有 3 个元素，却用 `unbind(0)` 拆成三个标量：
  作者的意图是让**三种权重的锐度各自可学**（`pre` 的 sigmoid 温度、`post` 的、
  `comb` 的 softmax 温度），而不是挤在一个共享温度里。
- 注意 `fn` 的第一维是 24（三种权重拼在一起），**不是** 3 个独立 Linear。
  一次矩阵乘就把 24 个数全算出来，省掉三次 kernel 启动 —— 推理里这是实打实的收益。

---

## 五、`forward` 的签名与三个权重的语义

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the weights used to mix in the `hc_mult` streams with the input and output of the next layer, which can
        be an attention or a MLP layer. This is done through three weights:

        - pre: used to collapse the `hc_mult` input streams into one, creating an input tensor for the next layer
        - post: used to expand the output of the next layer back into `hc_mult` streams
        - comb: used to mix the `hc_mult` input streams with the `hc_mult` output streams

        All weights are returned except "pre", which is consumed here.
        """
        batch_size, seq_len = hidden_streams.shape[:2]
        hc = self.hc_mult

```

**读法**：
- 返回类型是 `tuple[Tensor, Tensor, Tensor]`，**而且不返回 `pre`**。
  `pre` 在这一步就被消费掉了（变成 `collapsed`，见第八节）。
  所以调用方拿到的第三个东西，其实已经是"混好的一条流"，不是 4 条流。
  这解释 L0-01 第八幕里那句 `post, comb, hidden_states = self.attn_hc(hidden_states)`：
  等号左边的 `hidden_states` 是 3 维的。
- 三个权重的分工，文档字符串自己写清楚了：
  `pre` 收拢（collapse）、`post` 展开（expand）、`comb` 混合（mix）。
  **`pre` 是"读"、`post` 是"写"、`comb` 是"旧流怎么进新流"** —— 三步各管一件事。
- `batch_size, seq_len = hidden_streams.shape[:2]` 后面紧跟
  `hidden_streams.view(batch_size, seq_len, -1)`：**这个模块对 4 维输入是硬要求**。
  喂 3 维不会静默算错，会直接抛形状错误 —— 因为 `fn` 的列数固定是 `hc_mult * hidden_size`。

---

## 六、第一步「读取」：从 4 条流里读出 24 个数

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        # Flatten and norm the hidden streams
        flattened = hidden_streams.view(batch_size, seq_len, -1).float()
        flattened = self.input_norm(flattened)
        # Mix the streams together to infer the weight coefficients
        flattened = F.linear(flattened, self.fn.float())
        # Split the weight coefficients
        pre_w, post_w, comb_w = flattened.split([hc, hc, hc * hc], dim=-1)
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
- 第一件事是 `view(batch_size, seq_len, -1)`：把 4 条流**首尾相接**成一条 4D 长的向量
  （4 × 4096 = 16384），**不是求和**。为什么要拼接而不是先平均：
  要给 4 条流各算一组权重，就必须同时"看到"全部 4 条；先平均会把流间的差异抹掉，
  权重就退化成同一个值。
- `self.input_norm` 作用在这条 16384 长的拼接向量上，做的是**无权重 RMSNorm**：
  它保证投影的输入尺度稳定，而且与 `hidden_size` 无关。
- 三种权重的激活函数**故意不同**，这是本课第二个容易看漏的地方：
  - `pre = sigmoid(...) + hc_eps`：正值，且加了 eps 保证不为 0（它要做加权求和的系数）。
  - `post = 2 * sigmoid(...)`：值域 (0, 2)，**可以放大到 2 倍，也可以压到接近 0**。
  - `comb = softmax(..., dim=-1) + hc_eps`：`dim=-1` 是**逐行**归一化，
    所以 softmax 出来的矩阵**行和恰好是 1**，列和不是 —— 这正好是 Sinkhorn 想要的起点。
- 一行代码里三个 `.split(...)` 用的是同一个尺寸表 `[hc, hc, hc * hc]`：
  `(2 + N) * N` 这个式子在这里第二次出现，也解释了为什么 `fn` 的输出必须按这个顺序拼。

---

## 七、★ Sinkhorn：把 `comb` 逼到双随机流形上

这是全课的核心。**双随机（doubly-stochastic）= 行和也是 1、列和也是 1**，
而 softmax 只保证了行和是 1。补上另一半的，就是下面这 4 行。

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
- 代码注释自己给了动机：`To make sure the norm of "Mixed" does not blow up`。
  `Mixed = In @ Comb + Out`：如果 `comb` 的行和大于 1，混合一次就把范数放大一点，
  45 层叠起来是指数增长。双随机把它压成一个**凸组合** ——
  每条新流都是旧流的加权平均，范数不增。**这就是"为什么不会爆炸"的全部答案。**
- **注意循环的边界，这是最容易被读错的一处**：
  第 323 行先做**一次列归一化**，然后循环 `hc_sinkhorn_iters - 1 = 19` 次，
  每次"行 → 列"。合计 **20 次列归一化、19 次行归一化**，**最后一次操作是列归一化**。
  后果：跑完之后列和落在 `1 / (1 + hc_eps)` 上，行和只是逼近同一个值。
  实测：行和 / 列和都是 0.999999（偏差 9.5e-7）——
  **"双随机"是 eps 意义下的近似，不是精确的 1。**
- `+ self.hc_eps` 出现在每一个分母上，作用是**防零除**：
  如果某一行或某一列被压成全 0（概率极小但存在），没有 eps 会算出 inf/nan，
  而 nan 会顺着反向传播污染整个模型。`hc_eps = 1e-6` 是一个**数值下限**，不是可调超参。
- 20 次够不够？softmax 出来的矩阵本来就行和 = 1、元素全正，Sinkhorn 对这类矩阵收敛很快。
  实测行和偏差的下降：起始 0.1187 → 第 1 次 0.0206 → 第 2 次 0.0046 →
  第 3 次 0.0013 → 第 20 次 1e-6。**前三轮就拿掉了 99% 的误差**，
  剩下的 17 轮是在把这个误差按 eps 的水平磨平 —— 迭代次数给得宽，是因为它便宜。
- 为什么不在 forward 里"一次算准"？因为 Sinkhorn 的迭代次数就是**计算量与精度的旋钮**，
  作者选了 20 这个偏保守的值（配置项 `hc_sinkhorn_iters`，在 L1-02 里精读）。

---

## 八、写回的另一半：`pre` 在这里被消费

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        # Since "pre" is meant to be used with the input streams (available here as `hidden_streams`), we collapse the
        # streams here and return `collapsed` tensor, which will be the input for the next attention or MLP block.
        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
        return post, comb, collapsed
```

**读法**：
- `(pre.unsqueeze(-1) * hidden_streams).sum(dim=2)`：`pre` 形状 `(b, s, 4)` →
  `unsqueeze(-1)` → `(b, s, 4, 1)`，与 `(b, s, 4, 4096)` 逐元素相乘后**在第 2 维求和** →
  `(b, s, 4096)`。这就是"收拢"，也是子层真正看到的输入。
  注意求和维是 `dim=2`（流那一维），不是最后一维。
- `.to(hidden_streams.dtype)`：整个投影是在 **float32** 里算的（第 302 行的 `.float()`），
  回到主干前必须转回去。主干可能是 bf16，不转的话下一步 matmul 会吃到一个 float32 张量，
  既是隐式类型提升，也是显存与带宽的浪费。
- 返回的是 `post, comb, collapsed` 三件套；**`pre` 不在返回值里**。
  "读取 → 混合 → 写回"三步里，前两步在这个函数内部完成，第三步在解码层里完成（第九节）。

---

## 九、写回公式：同一个三行，出现两次

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        hidden_states = post.to(dtype).unsqueeze(-1) * hidden_states.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual
        )

        residual = hidden_states
        post, comb, hidden_states = self.ffn_hc(hidden_states)
        # Feed forward
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = post.to(dtype).unsqueeze(-1) * hidden_states.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual
        )
```

**读法**：
- 公式读作：**`新流 = post ⊙ 子层输出 + combᵀ · 旧流`**。
- 左半边：`post.to(dtype).unsqueeze(-1)` → `(b, s, 4, 1)`，
  `hidden_states.unsqueeze(-2)`（子层输出 `(b, s, 4096)`）→ `(b, s, 1, 4096)`，
  两者广播相乘 → `(b, s, 4, 4096)`：**同一个子层输出被 4 个不同的门控值缩放**。
  这就是 4 条流开始分叉的地方 —— 推导：若 4 条输入流相同（记作 x），
  则收拢出的 `collapsed` 只有一条、子层输出 o 也只有一个，
  于是 `新流_i = post_i · o + (Σ_j comb[j,i]) · x`，**唯一与 i 有关的量就是 `post_i`**。
  实测吻合：第 0 层之后 4 条流的最大差 2.9e-3，正是 `post` 之间的差异带来的。
- 右半边：`comb` 是 `(b, s, 4, 4)`，`transpose(-1, -2)` 后做 `matmul`。
  因为 `comb` 的**列和 ≈ 1**，每个新流都是 4 条旧流的凸组合 —— 这一条就是第七节 Sinkhorn 的回报。
  行和 ≈ 1 则保证"每条旧流的总贡献恰好是 1 份"，混合不改变总能量。
  两个方向各管一件事：**列和管"不放大"，行和管"不丢也不多"。**
- `residual` 在第 1314 行和 1341 行**各留了一份**（`residual = hidden_states`）：
  写回用的是**子层之前**的流。所以 mHC 的残差不是 `x + f(x)`，
  而是"4 条流各自混合后再写回 4 条流"。
- **同一个三行公式在本文件里出现了两次**（1337-1339 给注意力、1346-1348 给 FFN）——
  这就是"mHC 在每个子层前后各出现一次"的字面证据，也是第十幕动画画的东西。

---

## 十、层里的两个 mHC 实例

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        self.input_layernorm = Glm5NextTextRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Glm5NextTextRMSNorm(config.hidden_size, config.rms_norm_eps)

        self.attn_hc = Glm5NextTextHyperConnection(config)
        self.ffn_hc = Glm5NextTextHyperConnection(config)
```

**读法**：
- 两个 `Glm5NextTextHyperConnection` 是**独立实例、不共享参数**：
  实测每层 2 × 393,243 = 786,486 个 HC 参数。注意力有自己的一套 `pre/post/comb`，
  FFN 有另一套。
- 为什么不能共享同一套？因为它们混合的是**不同阶段**的流：
  `attn_hc` 读的是"上一层输出（或嵌入）"，`ffn_hc` 读的是"注意力写回之后的流"。
  两者的分布不同，用同一组投影等于强行让一个函数同时拟合两种统计量。
- 名字的挂载位置也有信息量：`attn_hc` / `ffn_hc` 挂在 `Glm5NextTextDecoderLayer` 上，
  **不在 `self_attn` / `mlp` 内部** —— mHC 是**层**的属性，不是子层的属性。
  所以"给某个子层换实现"（L3 会做的事）不会动到 mHC 的管线。

---

## 十一、读取发生在子层之前（`residual` 为什么要留底）

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        dtype = hidden_states.dtype

        residual = hidden_states
        post, comb, hidden_states = self.attn_hc(hidden_states)
        # Self attn
        hidden_states = self.input_layernorm(hidden_states)
        topk_indices = None
        if self.block_type == "linear_attention":
```

**读法**：
- 第 1312 行的 `dtype = hidden_states.dtype` 是给写回准备的：
  后面要写 `post.to(dtype)` / `comb.to(dtype)`，因为投影永远是 float32
  （`self.fn.float()`，第 305 行），而主干可能是 bf16。**先存 dtype，再动数据。**
- 顺序是 **留底 → 读取 + 混合 → 归一化 → 子层**：
  `residual = hidden_states`（4 维）先留下一份，然后 `attn_hc` 把它压成
  `collapsed`（3 维），归一化作用在 `collapsed` 上。
  **归一化不在 4 条流上做** —— 中间的子层各管各的 RMSNorm，流之间的尺度由 Sinkhorn 管。
- 对照 L0-01 第八幕：那里看到的是同一段代码的头部，本课补上的正是
  "`residual` 为什么是 4 维、`post` / `comb` 从哪来、它们后面怎么用"。
- 一段完整的层内顺序（记住这 6 步，L3 逐层精读时会逐行对上）：

```text
  residual = hidden_states                    # (b, s, 4, D)  留底
  post, comb, collapsed = self.attn_hc(hs)    # 读取 + 混合 → (b,s,4) / (b,s,4,4) / (b,s,D)
  x = self.input_layernorm(collapsed)         # (b, s, D)
  x = self.attn(x)                            # (b, s, D)
  hs = post[..., None] * x[:, :, None] + combᵀ · residual      # 写回 → (b, s, 4, D)
  ── 以上 5 步在 FFN 上原样重复一遍（用 self.ffn_hc）──
```

---

## 十二、★ 出口是无权重平均 —— 顺手订正 L0-01 的一处说法

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
class Glm5NextTextHyperHead(nn.Module):
    """Final GLM-5.3-Flash HC-stream collapse. Unlike DeepSeek-V4, this is an unweighted mean."""

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        return hidden_streams.mean(dim=2)
```

**读法**：
- 整个类只有两行：`hidden_streams.mean(dim=2)`。**没有参数**
  （实测 `sum(p.numel() for p in h.parameters()) == 0`），
  也**不是加权求和**（实测 `allclose(h(s), s.mean(dim=2)) == True`）。
- **订正 L0-01 第八幕的画面文字。** 那里写的是"`hc_head` 学习一组混合权重，
  把 4 条流加权求和"—— 按源码这是**错的**：hc_head 是等权平均，一个参数都没有。
  真正"学出来的混合"发生在每一层的 `attn_hc` / `ffn_hc` 里，不在出口。
  （L0-01 的画面文字不影响它的结论"4 条流收成 1 条"，但说法要按这里订正。）
- 文档字符串留了一句对比：`Unlike DeepSeek-V4, this is an unweighted mean.`
  —— 同为多流残差，DeepSeek-V4 的收拢是带权重的，GLM-5 选择不带。
  为什么可以不带：因为 4 条流已经被 Sinkhorn 约束在同一尺度上（凸组合不放大范数），
  出口不需要再学一次"怎么加权"；少一组权重，就少一处过拟合与数值风险。
- 调用点在主干第 1511 行（本课第一个引用块）：
  `hidden_states = self.norm(self.hc_head(hidden_states))`，
  进 `hc_head` 是 `(b, s, 4, 4096)`，出来是 `(b, s, 4096)`。

---

## 十三、手算两次 Sinkhorn 迭代

验收点之一。**给一个初始矩阵，把 20 轮迭代的前两轮手算出来。**
取一个"softmax 之后"的 4×4（行和已经是 1、列和不齐），eps 取 1e-6（对 4 位小数无影响）：

```text
A 初始（= softmax 输出 + eps）
        c0      c1      c2      c3    | 行和
  r0   0.70    0.10    0.10    0.10   | 1.00
  r1   0.40    0.40    0.10    0.10   | 1.00
  r2   0.10    0.10    0.70    0.10   | 1.00
  r3   0.10    0.10    0.10    0.70   | 1.00
 列和  1.30    0.70    1.00    1.00   ← 列和不是 1：这就是要修的东西

B 第 323 行（列归一化：每列 ÷ 该列列和）
        c0      c1      c2      c3    | 行和
  r0   0.5385  0.1429  0.1000  0.1000 | 0.8813  ← 行和被带坏了
  r1   0.3077  0.5714  0.1000  0.1000 | 1.0791
  r2   0.0769  0.1429  0.7000  0.1000 | 1.0198
  r3   0.0769  0.1429  0.1000  0.7000 | 1.0198
 列和  1.0000  1.0000  1.0000  1.0000 ← 列和修好了，行和坏了（来回拉锯就是 Sinkhorn）

C 第 1 次迭代（第 325 行行归一化 + 第 326 行列归一化）
        c0      c1      c2      c3    | 行和
  r0   0.5836  0.1668  0.1145  0.1145 | 0.9794
  r1   0.2723  0.5449  0.0935  0.0935 | 1.0043
  r2   0.0720  0.1442  0.6929  0.0990 | 1.0081
  r3   0.0720  0.1442  0.0990  0.6929 | 1.0081
 列和  1.0000  1.0000  1.0000  1.0000

D 第 2 次迭代之后                          E 走满 20 轮（= 代码的真实行为）
        c0      c1      c2      c3              c0      c1      c2      c3
  r0   0.5900  0.1705  0.1175  0.1175      r0   0.5915  0.1714  0.1185  0.1185
  r1   0.2685  0.5432  0.0936  0.0936      r1   0.2681  0.5439  0.0940  0.0940
  r2   0.0708  0.1432  0.6904  0.0986      r2   0.0702  0.1424  0.6890  0.0984
  r3   0.0708  0.1432  0.0986  0.6904      r3   0.0702  0.1424  0.0984  0.6890
 行和偏差 0.0046   列和偏差 1e-6          行和 0.999999  列和 0.999999
```

**三次归一化之后读数就固定了**：行和偏差 0.1187 → 0.0206 → 0.0046 → 0.0013 → …… → 1e-6。
注意 **B 列修好了、行坏了；C 行修好了、列又差一点点** —— 这就是 Sinkhorn 的全部机制：
在两个约束之间反复投影，最终收敛到**唯一**能由该矩阵经「行 / 列对角缩放」得到的双随机矩阵（Sinkhorn–Knopp 定理）。

**结论**：手算时只要盯住两件事 —— **除以列和（`dim=-2`）还是除以行和（`dim=-1`）**，
以及**最后一次除的是列和**（所以列和更"准"）。eps 只影响最后一位。

---

## 十四、与后面课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| `hc_mult` / `hc_eps` / `hc_sinkhorn_iters` 在 config 里怎么校验 | L1-02 |
| `attn_hc` / `ffn_hc` 的梯度怎么穿过 4 条流回到嵌入 | L3-06 |
| 4 维 `hidden_states` 对显存与 KV cache 的影响 | L5-01 ~ L5-03 |
| 子层归一化为什么在收拢之后 | L3-02 / L3-03 |
| 45 层 × 2 个 mHC 的参数量在总量里占多少 | L7-03 |
| `hc_head` 的等权平均与 DeepSeek-V4 的差别 | L3-06 |

**一句话总结**：

> 主干里 `hidden_states` 一直是 `(batch, seq, 4, 4096)`；每个子层前后，
> `Glm5NextTextHyperConnection` 用当前 4 条流算出 `pre/post/comb` 三组权重，
> 把流收拢成 1 条交给注意力或 MLP，再用 **`新流 = post ⊙ 子层输出 + combᵀ · 旧流`** 写回 4 条；
> `comb` 被 20 次 Sinkhorn 迭代约束成**双随机矩阵**，所以混合永远是凸组合、范数不爆炸；
> 出口的 `hc_head` 只是**无权重平均**（0 参数），不是加权求和。
