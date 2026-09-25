<!-- glm5-coverage
models/glm5_next/modeling_glm5_next.py
modeling_layers.py
modeling_outputs.py
-->

# L3-08 · 解码层装配：三个子块的顺序 — 源文件

**这一课只做一件事：把「一层」拆成三个子块、六次调用、两次写回，并说清它和教科书上的
`x = x + f(norm(x))` 差在哪里。**

视角：L0-01 把 45 层压成一条主线，L3-01 ~ L3-07 逐个拆开了层里的算子
（RMSNorm / SwiGLU / MoE / mHC / 遗忘门）。本课是 L3 的最后一站 ——
**把这些算子装回一层**，看清装配顺序、mHC 在其中的位置，以及一层往外交什么。

| 文件 | 行数（实测 `wc -l`） | 本课引用块 |
|---|---|---|
| `models/glm5_next/modeling_glm5_next.py` | 2444 | 9 |
| `modeling_layers.py` | 682 | 2 |
| `modeling_outputs.py` | 1662 | 2 |

> ★ **口径订正**：作业书写的是 2445 / 683 / 1663。实测 `wc -l` 是 **2444 / 682 / 1662**。
> 作业书用的是 `len(text.split("\n"))` 的口径，文件末尾换行之后那个空串也被算成一行；
> 本课一律用 `wc -l` 的数字。作业书给的行号（如 L1280、L1301、L312、L345）与实测一致，照用。

**本课的实测结论**（`_data/recon/probe_l308.py` 的真实输出；脚本**不计入覆盖率**）：

```text
一层的子模块调用次序（forward hook 实测，一个都不多）
    attn_hc -> input_layernorm -> self_attn -> ffn_hc -> post_attention_layernorm -> mlp
    ↑ 两次写回不在这个列表里 —— 它们是 forward 里的两行内联张量运算，不是模块

形状（小配置实测：B=1, S=5, hc_mult=4, hidden_size=64）
    层输入 / 层输出   (1, 5, 4, 64)      4 条流进、4 条流出
    attn_hc 返回      post (1, 5, 4) | comb (1, 5, 4, 4) | collapsed (1, 5, 64)
    层返回            (hidden_states, topk_indices)     topk_indices 可以是 None
    子模块清单        ['self_attn', 'mlp', 'input_layernorm',
                       'post_attention_layernorm', 'attn_hc', 'ffn_hc']

mHC 的三个数值约束（真实初始化，实测）
    input_norm 之后流的 std                  = 1.0000（0 参数的 unweighted RMSNorm）
    投影输出（pre/post/comb 的 pre-activation）std = 2.49
    post = 2 * sigmoid(...)                -> 落在 (0, 2)；实测 mean 1.004、范围 [0.005, 1.968]
    comb 经 Sinkhorn 后：列和偏差 1.07e-06（≈ hc_eps），行和偏差 1.77e-02
    ★ 也就是说 20 次迭代后 comb 只是**近似**双随机：列被最后一步归一钉死，行还留着 1.8% 的残差
    每层 mHC 参数 = 2 x 393,243 = 786,486   (fn 形状 (24, 16384) + base(24) + scale(3))

推理时的 GradientCheckpointingLayer（实测）
    gradient_checkpointing = False（类属性）
    刚构造出的层连 _gradient_checkpointing_func 属性都不存在
    eval + no_grad 调用一次：checkpoint 函数被调用 0 次
    对照：train() + gradient_checkpointing=True 时被调用 1 次
```

下面每一段 `python` 块都是**逐字**引用（由 `_data/build_l308.py` 从源文件按行号切片生成，
不是手抄）；`text` 块是示意汇总与实测输出，**不参与保真校验**。

---

## 一、★ 一层只有一种装法：三个子块，两次写回

`Glm5NextTextDecoderLayer` 继承的是 `GradientCheckpointingLayer`（本课第八节展开它）。
先看它的构造：**三个子块**就是 `self_attn`、`mlp`，以及出现两次的 `attn_hc` / `ffn_hc`；
两个 RMSNorm 分别服务两个子块。

回顾 L0-01：当时看到的是「层内二选一」（注意力轴 × MLP 轴）。这里把同一个 `__init__`
读全 —— 除了那两个二选一，剩下的全是**固定的装配**。

@@B:glm5:1280:1299@@

**读法**：

- 三个子块的清单是**固定**的：`self_attn` / `mlp` / `attn_hc` + `ffn_hc`。
  真正随层变化的只有两个字符串判断（`config.layer_types[layer_idx]` 与
  `config.mlp_layer_types[layer_idx]`），默认配置下它们给出 34 层 KDA + 11 层 MLA、
  3 层稠密 + 42 层 MoE（实测 `layer_types[0..3]` = KDA, KDA, KDA, MLA）。
- **两个 mHC 不是同一个对象**。实测 `d0.attn_hc is d0.ffn_hc` 为 `False` ——
  它们各自持有一份 `fn` / `base` / `scale`，所以「注意力前怎么混合」与「FFN 前怎么混合」
  是两套独立学出来的权重；同一层的两个子块不会互相干扰。
- 两个 RMSNorm 是**分开的**：`input_layernorm` 只归一化进注意力的那一份，
  `post_attention_layernorm` 只归一化进 MLP 的那一份。它们不是共享的归一化层 ——
  这与 L3-01 里"三种 RMSNorm"的分工一致。
- `super().__init__()` 之后**没有** `self.gradient_checkpointing = ...`：
  这个开关由类属性给出（`False`），只有显式 `gradient_checkpointing_enable()` 才改。
  这是第八节「推理时它什么都不做」的前提。

### 一层的真实调用次序（示意，实测，非引用）

```text
   hidden_states (B, S, N, D)     N = hc_mult = 4
        │
        ├─(1) attn_hc ──────────► post, comb, collapsed
        ├─(2) input_layernorm ◄── collapsed
        ├─(3) self_attn       ◄── norm 后的 (B, S, D)
        │        └─ 写回：新流 = post ⊙ attn_out + combᵀ · 旧流      ← 不在模块列表里
        ├─(4) ffn_hc ───────────► post, comb, collapsed
        ├─(5) post_attention_layernorm ◄── collapsed
        ├─(6) mlp
        │        └─ 写回：新流 = post ⊙ mlp_out + combᵀ · 旧流       ← 同一段代码第二次出现
        ▼
   (hidden_states, topk_indices)
```

注意 (1) 在 (2) **之前**：mHC 先算权重、先把 4 条流收拢成 1 条，归一化只作用于收拢后的那 1 条。
下一节就讲这个顺序。

---

## 二、顺序①：mHC 在 norm 之前

这是本课要默写的第一个位置。教科书式 pre-norm 是 `sublayer(norm(x))`；
这里 `x` 先被 mHC 处理过一遍，**归一化只吃到"收拢后的一份"**。

@@B:glm5:1312:1318@@

**读法**：

- `residual = hidden_states` 这一行留的是**未归一化的 4 条旧流** ——
  它在写回时要和 `combᵀ` 相乘，所以必须是原样的一份（第三节）。
- `self.attn_hc(hidden_states)` 进去的是 `(B, S, 4, D)`，出来三样东西：
  `post (B, S, 4)`、`comb (B, S, 4, 4)`、`hidden_states (B, S, D)`（实测形状）。
  也就是说 mHC **同时**做了两件事：算权重、收拢流；收拢结果直接顶掉了变量名
  `hidden_states`。从这里往下，层内看到的就是普通的 3 维张量了。
- 权重是在**未归一化的 4 条流**上估计的。为什么可以这样：mHC 内部第一步是
  `self.input_norm(flattened)`（第四节），它自己带一个 0 参数的
  `Glm5NextTextUnweightedRMSNorm`。所以顺序不是"忘了归一化"，而是
  **权重估计有自己的归一化，主干那份归一化只服务子层输入**。
- `topk_indices = None` 出现在 if 之前，是因为两种注意力分支的返回值形状不同：
  KDA 分支只返回一个张量，MLA 分支返回三元组并可能产出 `topk_indices`（L0-01 第六节）。
  放在分支前初始化，是为了两条路都能 `return hidden_states, topk_indices`。

---

## 三、★ 顺序②：写回不是加法

本课最要紧的一处。**验收点 2**（"能说出 residual 在 mHC 下与标准 Transformer 的差异"）
的全部证据就是这 4 行。

@@B:glm5:1337:1339@@

**读法**：

- 公式读作：`新流 = post ⊙ 子层输出 + combᵀ · 旧流`。
  - `post.to(dtype).unsqueeze(-1)`：`(B, S, 4)` → `(B, S, 4, 1)`，与
    `hidden_states.unsqueeze(-2)`（`(B, S, 1, D)`）相乘，广播出 `(B, S, 4, D)`。
    含义：**同一个子层输出被复制到 4 条流上，每条流用自己那个 `post` 逐元素门控写多少。**
  - `torch.matmul(combᵀ, residual)`：`(B, S, 4, 4)ᵀ @ (B, S, 4, D)` → `(B, S, 4, D)`。
    含义：**每条新流拿到的是 4 条旧流的一个线性组合**，权重就是 `comb` 的一列。
- 与标准 Transformer 的差异（这就是验收点的答案）：

  | | 教科书 residual | mHC（本课） |
  |---|---|---|
  | 残差条数 | 1 条 | `hc_mult` = **4** 条 |
  | 合并方式 | `x + f(norm(x))`，加法 | `post ⊙ f + combᵀ · x`，门控 + 矩阵混合 |
  | 子层输出的去向 | 原样加到 `x` 上 | 先被 `post` 逐流缩放，再广播到 4 条流 |
  | 旧流的去向 | 原样保留 | 被 `comb` 混合后才保留 |
  | 每层出现次数 | 2 次（注意力后、FFN 后） | 2 次（**同一段代码出现两次**） |
  | 步长控制 | 无（等于 1） | `post ∈ (0, 2)`，初始化附近 ≈ 1（实测 [0.96, 1.15]） |

- `.to(dtype)` 两处都不能省：`post` / `comb` 是在 **fp32** 里算出来的（第四节），
  写回前要转回 `hidden_states` 的 dtype，否则整个主干会被抬成 fp32。
- 为什么值得这样设计：加法残差只能"加"，`comb` 让 4 条流在写回时**互相交换信息**，
  而 Sinkhorn 约束（第五节）保证这个交换不会放大范数。这是 mHC 相对标准残差的全部新意，
  也是 L3-06 讲的那件事在这一层的落点。

---

## 四、mHC 的三个权重是从哪来的

上面反复用到 `post` 与 `comb`。它们来自 mHC 内部的一次线性投影 + 三种不同的激活。

@@B:glm5:314:317@@

**读法**：

- 形状账：`flattened` 是 `(B, S, hc * D)`；`self.fn` 的形状实测是
  `((2 + N) * N, N * D)`，`N = hc_mult = 4` 时就是 `(24, 16384)`。
  一次 `F.linear` 得到 `(B, S, 24)`，正好切成 `pre(4) + post(4) + comb(16)`。
- 三种激活各不相同，各有用意：
  - `pre = sigmoid(...) + hc_eps`：收拢权重，非负、上限 1+ε。加 `hc_eps` 是为了
    **不让任何一条流被彻底清零**（全零会让梯度断掉）。
  - `post = 2 * sigmoid(...)`：范围 `(0, 2)`。取 2 倍是为了让**平均**写回幅度落在 1 附近 ——
    `_init_weights` 把 `base` 置 0、`scale` 置 1、`fn ~ N(0, 0.02)`，
    于是 `post = 2 * sigmoid(z)`，`z` 的均值是 0。
  - `comb = softmax(..., dim=-1) + hc_eps`：**按行**归一，先保证行和为 1，
    再交给下一节的 Sinkhorn 把列和也拉成 1。
- 实测的尺度账（这一步很关键，它决定了门控"活不活"）：
  `input_norm`（0 参数的 unweighted RMSNorm）先把流归到 **std ≈ 1.0000**；
  投影维度是 `N * D = 16384`，`fn` 的初始化 std 是 0.02，
  所以 pre-activation 的 std ≈ 0.02 × √16384 = 2.56（**实测 2.49**）。
  这个尺度下 `post = 2 * sigmoid(z)` 的**均值是 1.004，但取值范围是 [0.005, 1.968]** ——
  "平均写回 1 倍"成立，"每条流都约等于 1"不成立：门控从第一步就是活的。
  （★ 本条是实测订正：只按 `base=0, scale=1` 推断会写成"初始化时 post ≈ 1"，
  跑一遍才发现逐 token 逐流的差别可以到两个数量级。）
- 参数量：`fn` 393,216 + `base` 24 + `scale` 3 = **393,243** 个参数；
  一层两个 mHC，共 786,486，45 层合计 35,391,870。这就是"每层每个子块前面
  多出来的那点成本"。

---

## 五、★ comb 为什么必须双随机：Sinkhorn 手算

`comb` 不是普通的权重矩阵。源码注释把理由写在代码里了 —— 这是本课第二处值得逐字读的注释。

@@B:glm5:319:326@@

**读法**：

- 目标：`comb` 要**双随机**（doubly-stochastic）—— 每一行、每一列的和都等于 1。
  源码注释给出的理由是：混合结果形如 `Mixed = In @ Comb + Out`，
  只有双随机的 `Comb` 才能保证 `Mixed` 的范数不被放大。
- 迭代次数要算对：代码是「**先**按列归一一次，**再**循环 `hc_sinkhorn_iters - 1` 次
  （每次先按行、后按列）」。所以 `hc_sinkhorn_iters = 20` 时，实际归一化
  **1 + 19 × 2 = 39 次**，不是 20 次。
- **手算例子**（`_data/recon/probe_l308.py` 的真实输出）。取

  ```text
  M0 = [[0.9000, 0.4000],       行和 = [1.3000, 1.1000]
        [0.3000, 0.8000]]       列和 = [1.2000, 1.2000]
  ```

  按源码的顺序迭代（列 → 行、列 → 行 …）：

  ```text
  iters = 1   [[0.749999, 0.333333],   行和 = [1.083332, 0.916666]
               [0.250000, 0.666666]]   列和 = [0.999999, 0.999999]
  iters = 2   [[0.717391, 0.297297],   行和 = [1.014688, 0.985310]
               [0.282608, 0.702702]]   列和 = [0.999999, 0.999999]
  iters = 3   [[0.711396, 0.291196],   行和 = [1.002593, 0.997406]
               [0.288603, 0.708803]]   列和 = [0.999999, 0.999999]
  iters = 20  [[0.710101, 0.289898],   行和 = [0.999999, 0.999999]
               [0.289898, 0.710101]]   列和 = [0.999999, 0.999999]
  ```

  读法：每一次"列归一"都把列和精确压到 1，代价是把行和推歪；
  下一次"行归一"再把行和压回 1、把列和推歪一点点。**误差每轮缩小，收敛到双随机矩阵。**
  `iters = 1` 时列和已经精确是 1、行和还差 8%；20 次之后两边都到 1e-6 量级。
  （上表是**去掉 `hc_eps` 的 2x2 简化版**，用来说明迭代的作用；下面是真身。）
- **实测：真实的 4x4 `comb` 20 次迭代后并不是严格双随机**。这是本课的一处实测订正：

  ```text
  真实初始化 + 真实 4x4（hidden_size=4096, hc_mult=4, hc_sinkhorn_iters=20）
      列和与 1 的最大偏差 = 1.073e-06     <- 最后一步就是列归一，被钉死在 eps 上
      行和与 1 的最大偏差 = 1.765e-02     <- 1.8%，不是 1e-6
  ```

  为什么行和留着 1.8%：循环的**最后一步是列归一**，它把每一列乘上不同的因子，
  于是上一步刚刚归好的行和又被推歪；这个残差正是"上一轮列和离 1 有多远"。
  收敛速度取决于 `comb_w` 的尺度：pre-activation 的 std 实测是 2.49（第四节），
  `softmax` 出来非常尖（4 类的行熵实测只有 0.159，均匀分布是 1.386），
  尖矩阵的 Sinkhorn 收敛更慢。所以 20 次迭代给的是**近似**双随机。
- 源码注释的措辞也说明了这一点：`To make sure the norm of "Mixed" does not blow up,
  we constrain the comb weight to be doubly-stochastic` ——
  意图是约束范数不放大，实现上是"迭代到足够好"，不是精确投影。
  `hc_sinkhorn_iters` 这个配置项（默认 20）就是留给这个精度旋钮的。
- 最后三行是收尾（注意 L329 那句注释里带反引号，保真门禁照收）：

@@B:glm5:329:331@@

  `collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2)` ——
  4 条流按 `pre` 加权求和成 1 条，转回原 dtype，和 `post` / `comb` 一起返回。
  这就是下一个子层（注意力或 MLP）的输入。实测 `(1, 5, 4, 64)` → `(1, 5, 64)`。

---

## 六、顺序③：FFN 子块是同一套模板

把注意力换成 FFN，代码几乎是复制的 —— 只改了三个名字。这种"重复"本身就是设计：
装配模板只有一套。

@@B:glm5:1341:1350@@

**读法**：

- 与第三节逐行对照：`residual = hidden_states` 出现第二次；`self.attn_hc` 换成
  `self.ffn_hc`；`input_layernorm` 换成 `post_attention_layernorm`；
  `self_attn(...)` 换成 `self.mlp(...)`；写回那两个 `torch.matmul` 一字不改。
  **同一段代码在一层里出现两次**，这就是 L0-05 说的"mHC 在每个子层前后各出现一次"。
- 两个子块的边界不是"模块调用"，而是**那两行写回**：注意力分支结束后立刻写回，
  写回结果既是下一段的输入、也是 FFN 分支的 `residual`。
  所以一层的语义边界是「子层 + 写回」，不是「子层」。
- `return hidden_states, topk_indices`：层往外交**两个**东西。
  `hidden_states` 仍是 4 维流（`(B, S, 4, D)`），不是普通 hidden；
  `topk_indices` 是 DSA 的稀疏选择，交给下一层（L0-01 第五节）。实测在小配置下
  它是 `None` —— 因为 `next_skip_topk` 为假时注意力分支直接返回 `None`，
  主干那边用 `prev_topk_indices=topk_indices` 原样接住，`None` 也是合法值。

---

## 七、FFN 子块内部：MoE 的路由、共享专家，和被裁两次的 SwiGLU

`mlp` 在 42 层里是 `Glm5NextTextMoE`。它自己内部还有一次残差：
稀疏专家的输出加上共享专家的输出。

@@B:glm5:201:208@@

**读法**：

- MoE 子块的内部顺序只有四步：先存一份 `residuals`（注意存的是**归一化后**的子层输入，
  不是层入口那份）→ 路由 `gate` → 稀疏专家 → 加共享专家。
- `hidden_states.view(-1, hidden)` 之后又 `.view(*orig_shape)`：专家是按 token 逐个
  `index_add_` 累加的（第三节之外的另一处实现细节，L3-05 展开），
  这里只保证进出形状一致。
- `self.shared_experts` 是一个完整的 `Glm5NextTextMLP`，宽度是
  `moe_intermediate_size * n_shared_experts`；它是**每个 token 都走**的稠密路径。
  所以"MoE 层"里其实有两套前馈：一套稀疏（选中 8 个专家）、一套稠密（1 个共享专家）。
- **`swiglu_limit` 在两处各裁一次**。共享专家走 `Glm5NextTextMLP.forward`：

@@B:glm5:99:105@@

  路由专家走 `Glm5NextTextExperts._apply_gate`：

@@B:glm5:138:143@@

  两处的裁剪规则完全相同：`gate` **只裁上界**、`up` **双边**都裁。
- **手算例子**（`swiglu_limit = 10.0`，实测输出）：

  ```text
  gate 原始 [ 12.0,  -3.0,  0.5]   clamp(min=None, max=10.0)   ->  [ 10.0,  -3.0,  0.5]
  up   原始 [ 15.0, -12.0,  2.0]   clamp(min=-10.0, max=10.0)  ->  [ 10.0, -10.0,  2.0]

  silu(gate) * up   裁剪后 = [99.9955, 1.4228, 0.6225]
                    不裁剪 = [180.0,  1.7073, 0.6225]        ← 第 1 个元素差 1.8 倍
  ```

  为什么只裁 `gate` 的上界：`silu` 在正半轴近似线性、在负半轴趋于 0，
  所以 `gate` 只需要防"正得太大"；`up` 是乘数，两边都要防。
  这条 10.0 的限幅就是 L3-03 讲 SwiGLU 时那个 `Key difference using clamping`。

---

## 八、推理时的 `GradientCheckpointingLayer`：它什么都不做

这一层继承的基类在训练时会把前向包进 `torch.utils.checkpoint`。推理时呢？
先读全整个类 —— 它只有两个类属性和一个 `__call__`。

@@B:layers:76:110@@

**读法**：

- 分支条件是两个条件的**与**：`self.gradient_checkpointing and self.training`。
  推理时 `self.training` 为 `False`（`model.eval()` 之后），所以无论
  `gradient_checkpointing` 是什么，都走最后一行 `return super().__call__(*args, **kwargs)`
  —— 也就是**普通的 `nn.Module.__call__`**，没有任何额外开销、没有额外显存、
  也不需要 `hidden_states` 是位置参数。
- 实测（`_data/recon/probe_l308.py`）：
  - `GradientCheckpointingLayer.gradient_checkpointing` 类属性为 `False`；
  - 刚构造出的层**连 `_gradient_checkpointing_func` 属性都没有** ——
    它是 `gradient_checkpointing_enable()` 才 `setattr` 上去的；
    所以推理路径上"误触发 checkpoint"这件事在属性层面就不成立；
  - `eval() + torch.no_grad()` 下把 `_gradient_checkpointing_func` 换成计数器，
    调用次数是 **0**；同样的层 `train() + gradient_checkpointing=True` 时是 **1**。
- 推理时 `use_cache` 也不会被改：注释里那条 `_can_checkpoint_with_cache = False`
  只在训练分支里起作用（把 `use_cache` 强置 False、把 `past_key_values` 置 None），
  因为"重算一次缓存"等于把 KV cache 写两遍。**推理不进入这个分支，KV cache 照常累积。**
- 对本课的结论：解码层在推理时就是一条**普通的前向链**；所有"checkpoint"的复杂度
  都只在训练侧。这也是为什么后面 L5（KV cache）与 L8（注意力后端）可以完全不考虑它。

---

## 九、`modeling_layers.py` 里还有什么：基类与「装配单元」契约

本课覆盖 `modeling_layers.py`，是因为解码层的**基类**住在这个文件里（第八节）。
除此之外它还是 transformers 的"层零件库"：同一个解码层类可以原样被别的装配用掉 ——
只要它遵守 `(config, layer_idx)` 这个接口。

@@B:layers:312:328@@

**读法**：

- `MtpLayer.__init__` 的第二个参数是 `decoder_layer_cls`：一个**类**，不是实例。
  第 327 行 `self.mtp_block = decoder_layer_cls(config, layer_idx)` 说明
  MTP 块就是"再来一个解码层"，构造签名与主干完全一致 —— 这正是本课标题里的"装配"。
- `norm_cls` 也是传进来的：主干用哪种 RMSNorm，MTP 就用哪种。
  换句话说，**解码层的接口只有 `(config, layer_idx)` 两个参数**，
  子块怎么选全部由 config 的 `layer_types` / `mlp_layer_types` 决定 ——
  换 `layer_idx` 就换出一层不同排班的层。这也是为什么
  `Glm5NextTextDecoderLayer(config, layer_idx)` 在主干里能被一行列表推导批量造出来。
- **实测边界（很重要，别过度外推）**：`grep -rn "MtpModel\|MtpLayer" models/glm5_next/`
  命中 **0 处** —— GLM-5 这条链路**没有**装配 MTP；用到 `MtpModel` 的是
  `models/step3p7` 与 `generation/candidate_generator.py`。
  所以本节引用的 MTP 代码是**零件库的设计契约**，不是在说 GLM-5 里有 MTP 层。
- 与之配套的 `PreTrainedConfig.get_mtp_config()` 写在 `configuration_utils.py` 里
  （**不在本课覆盖域内，不计入覆盖率**），docstring 原文是
  "MTP layers restart the indexing of layers at 0"：因为子块是按 `layer_idx`
  从 config 的排班表里取的，复用同一个类就必须先把索引重置。
- 主干那一侧的对应写法：`self.layers = nn.ModuleList([Glm5NextTextDecoderLayer(config, layer_idx)
  for layer_idx in range(config.num_hidden_layers)])` —— 45 层就是这么来的
  （实测默认 `num_hidden_layers = 45`）。主干循环里的
  `self.layers[: self.config.num_hidden_layers]` 切片在默认配置下是空操作
  （列表本来就只有 45 个元素），它把"只跑主干这些层"写成了一条显式契约。

---

## 十、返回值：`modeling_outputs` 的数据类约定

一层返回的是元组，一站（主干）返回的是数据类。主干最后一行构造的就是它
（那两行逐字引用见 L0-01 第八节，本课不重复占引用块）：

@@B:outputs:345:349@@

**读法**（三条约定，全部实测）：

1. **每个字段都有默认值 `None`**：`last_hidden_state` / `past_key_values` /
   `hidden_states` / `attentions` / `router_logits` 全是 `X | None = None`
   （实测 `all(f.default is None for f in fields(...))` 为 `True`）。
   所以 `MoeModelOutputWithPast(last_hidden_state=..., past_key_values=...)` 只给两个字段
   也能构造 —— 主干那行就是这么写的。
2. **字段顺序就是位置参数顺序**。实测 `o[0] is o.last_hidden_state` 为 `True`；
   下游代码可以写 `out[0]` 而不写字段名。因为是 `@dataclass`，
   顺序一改就是破坏性改动。
3. **`None` 字段在序列化时被丢掉**。实测 `o.keys()` 只给出
   `['last_hidden_state', 'past_key_values']`，另一个只给 `last_hidden_state` 的实例
   `keys()` 只有 `['last_hidden_state']`；`to_tuple()` 的注释原文是
   "Convert self to a tuple containing all the attributes/keys that are not `None`"。
   于是 `output_hidden_states=True` 这类开关打开时，返回值的**长度会变** ——
   按位置解包的下游代码必须改用关键字访问。

`ModelOutput` 本身是 `dict` 的子类（实测 `isinstance(o, dict)` 为 `True`，
`dict(o)` 得到两个键），所以 `**outputs` 这种写法也是合法的。

推理侧真正会用到的是另一个数据类 —— 它把训练用的字段排在最前面：

@@B:outputs:391:397@@

**读法**：`loss` / `aux_loss` 排在最前，是因为训练时它们是"第一个要拿的东西"；
推理时这两个字段是 `None`，按约定 3 从 `keys()` 里消失。
`logits` 才是采样要的。注意这里没有 `last_hidden_state` —— 主干交出来的
`last_hidden_state` 在 LM 头里被换成了 `logits`（L0-01 第八节、L6 展开）。
字段的完整清单与张量并行时的用途见 L7-03。

---

## 十一、把一层默写出来

```text
Glm5NextTextDecoderLayer.forward(hidden_states (B,S,N,D), ...) -> (hidden_states, topk_indices)

  1  residual = hidden_states                       # 未归一化的 4 条旧流
  2  post, comb, hidden_states = self.attn_hc(...)  # 算权重 + 收拢成 (B,S,D)
  3  hidden_states = self.input_layernorm(...)      # 只归一化收拢后的 1 条
  4  hidden_states = self.self_attn(...)            # KDA 或 MLA+DSA（二选一）
  5  hidden_states = post ⊙ hidden_states + combᵀ · residual     # 写回（第一次）
  6  residual = hidden_states
  7  post, comb, hidden_states = self.ffn_hc(...)
  8  hidden_states = self.post_attention_layernorm(...)
  9  hidden_states = self.mlp(...)                  # 稠密 MLP 或 MoE（二选一）
 10  hidden_states = post ⊙ hidden_states + combᵀ · residual     # 写回（第二次，同一段代码）
 11  return hidden_states, topk_indices
```

默写时按「mHC → norm → 子层 → 写回」这个**四拍循环**记：两个子块是同一个循环跑两次，
换的只有 `attn_hc→ffn_hc`、`input_layernorm→post_attention_layernorm`、
`self_attn→mlp` 三处名字。

---

## 十二、与后续课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| mHC 的 `fn` / `base` / `scale` 怎么初始化、mHC 整体怎么推 | L3-06 |
| 注意力是 KDA 还是 MLA+DSA，怎么选 | L4-01 ~ L4-06 |
| 路由的 `n_group` / `topk_group` / `e_score_correction_bias` 细节 | L3-05 |
| `swiglu_limit` 在视觉塔里的同款裁剪 | L2-07 |
| 主干循环与 `topk_indices` 的跨层传递 | L0-01 |
| `MoeModelOutputWithPast` 与张量并行的关系、字段全表 | L7-03 |
| KV cache 在推理时怎么累积（本课确认它不会被 checkpoint 干扰） | L5-01 ~ L5-03 |

**一句话总结**：

> 一层 = `mHC → norm → 子层 → 写回` 跑两遍（子层 = 注意力 / MoE，二选一）；
> **residual 在 mHC 下不是一条加法，而是"`post` 门控 + 4 条流的 `comb` 混合"**；
> 推理时 `GradientCheckpointingLayer` 直接退化成普通 `nn.Module.__call__`。
