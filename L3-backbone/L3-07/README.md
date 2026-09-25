# L3-07 · 遗忘门与门控范数

> 层：**第 3 层 · 文本主干** ｜ 优先级：P0 ｜ 前置课：`L3-06`

## 学习目标

看完这一课，你应该能：

1. **解释下界截断防的是什么**：`g = -5.0 x sigmoid(decay_rate x g)` 把 log-decay 关进
   `(-5, 0)`，于是单步衰减 `decay = exp(g) >= e^-5 = 0.006738`，**永远不会一步清零**
   —— 但它防的**不是**分块形式里 `exp` 的上溢（那一路由掩码方向解决，见下）；
2. **说出遗忘门在 KDA 递推里的位置**：`recurrent_kimi_delta_attention` 循环里的
   `last_recurrent_state = last_recurrent_state * g_i`，即「先衰减 → 再读取 → 最后写入」
   三步里的第一步；
3. 说出 `A_log` / `dt_bias` 的初始化来源，以及**为什么 `A_log` 的初始化必须和 forward
   的分支配套**（有下界 → 置 0；无下界 → log-uniform(1,16)）。

## 覆盖的源文件（1 个）

| 文件 | 行数 | 引用块 |
|---|---|---|
| `models/glm5_next/modeling_glm5_next.py` | 2444 | 13 |

> 行数口径：`wc -l` 实测 **2444**；作业书写的是 2445（那是 `text.split("\n")` 的口径，
> 文件末尾的换行符会多算一个空串）。本课按 `wc -l` 记 2444。
>
> 本课引用了 `_data/recon/probe_l307.py` 的**实测输出**（写在 `source.md` 的 `text` 块里）。
> **它是脚本，不计入覆盖率** —— 覆盖域只统计 `src/transformers/**/*.py`。

## 场景（9 幕）

1. **一层 KDA 里两个门** —— 主链路 + 三个放行系数（遗忘门 / 写入门 / 门控范数）
2. **遗忘门的构造** —— `4096 → 128 → 8192` 低秩投影，参数量 1,581,120（q/k/v 的 1.57%）
3. **★ 前向算账** —— `A_log` 每头一个值、`dt_bias` 每 (头, 通道) 一个值，广播方式不同
4. **★ -5.0 的值域意图** —— 手算表（含 fp32 实测值）+ 饱和点 `raw ≈ +15`
5. **另一支：A 是乘法增益** —— `-decay_rate x softplus(raw)` 能掉到 -320，
   而 `exp(-320)` 在 fp32 里恰好是 `0.0`
6. **★ 递推里的落点** —— `状态 = 状态 x exp(g)`；24 步纯衰减实测 + 分块形式的实测订正
7. **★ 初始化** —— `A_log` 跟着分支走；`dt_bias` 的三步逆 softplus（FLA 惯例）
8. **门控范数** —— 严格 fp32 归一化 + `sigmoid(gate)` 放行；顺序不能反
9. **收束** —— 三个门的对照表 + 练习 + 下一课指路

## 核心结论

### 1. ★ 下界截断：把 log-decay 关进 (-5, 0)

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        # Safe lower bound decay
        if self.safe_gate_lower_bound is not None:
            return self.safe_gate_lower_bound * torch.sigmoid(decay_rate * g)
```

`sigmoid` 的值域是 `(0, 1)`，乘 `-5.0` 之后门恒在 `(-5, 0)`：**单步衰减
`decay = exp(g) >= e^-5 = 0.006737947`，永远不等于 0**。实测（fp32）：
`raw = 0 → g = -2.5 → decay = 0.082085`（忘掉 91.79%）；`raw = +15` 与 `raw = +60` 的
decay 在第 6 位小数上才分得开 —— 打分超过约 +15 就饱和了。

**它不防什么（实测订正）**：直觉上会以为下界是防分块形式 `exp(g[i] - g[j])` 上溢的。
实测 `decay_mask ∈ [0, 1]`（减法方向与掩码方向配套，gate=-5 时非对角最大正好是
`6.737947e-03 = e^-5`），不会上溢；真正归零的是 `exp(cumsum)`：gate=-5 时 64 个位置里
有 **44 个**精确为 0。上溢那一路是靠掩码方向解决的，两条保护各管一边。

### 2. ★ 遗忘门在递推里的位置

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        last_recurrent_state = last_recurrent_state * g_i
        kv_mem = (last_recurrent_state * k_i[..., None]).sum(dim=-2)
        delta = (v_i - kv_mem) * b_i
```

`g_i = g[:, i][..., None].exp()`：传进来的 `g` 是 log-decay，`.exp()` 之后才是衰减率。
顺序是 **先衰减（乘 `g_i`）→ 再读取（`kv_mem`）→ 最后写入（加 `k ⊗ delta`）**；
写入是加法，所以衰减只作用在旧记忆上。实测纯衰减 24 步：`gate=-1 → 3.775137e-11`
（手算 `exp(-24) = 3.775135e-11`）；`gate=-5`（正好是下界）那一行**精确为 0**
（手算 7.667648e-53）—— 不是下界失效，是 fp32 撑不住（`exp(-104)` 实测就是 0.0）。

### 3. ★ 初始化与 forward 的分支严格配套

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
            if module.safe_gate_lower_bound is not None:
                init.zeros_(module.A_log)
            else:
                init.copy_(
                    module.A_log,
                    init.uniform_(module.A_log, a=1.0, b=16.0).log(),
                )
```

有下界时 `A` 是 **sigmoid 的斜率**：A=16 会把 sigmoid 推饱和（实测梯度 `5.38e-32`），
门冻住；所以必须置 0（`decay_rate = exp(0) = 1.0`，实测 4 个头全是 1.0）。
无下界时 `A` 是 **乘法增益**（`-decay_rate x softplus(raw)`），`A ~ U(1, 16)` 才给得出
每头不同的时间尺度初值（实测采样 `[3.606, 15.780]`）。这句
`# NOTE: This is incredibly important so keep it this way at all costs` 就是写给这个
if/else 的。

`dt_bias` 没有 if，两种分支共用同一套三步逆 softplus：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
            init.uniform_(
                module.dt_bias,
                a=math.log(1e-3),
                b=math.log(1e-1),
            )
            dt = module.dt_bias.exp().clamp_min(1e-4)
```

实测：`softplus(dt_bias) ∈ [1.05e-3, 9.64e-2]`、`dt_bias ∈ [-6.857, -2.290]`；
手算 `softplus^-1(1e-3) = -6.907255`、`softplus^-1(1e-1) = -2.252168`。
注意 forward 里 `dt_bias` 是**直接相加**、不过 softplus —— 这套初始化是从 FLA 继承来的
惯例，语义在这个公式里已经变了（初始 decay 均值 0.911）。

### 4. 门控范数是第二个门

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        # Strict FP32 norm (do not downcast on the weights)
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight.to(torch.float32) * hidden_states
```

只算均方、不减均值；权重也升到 fp32，最后才 `to(input_dtype)`（实测 bf16 进、bf16 出）。
门控在归一化**之后**：`hidden_states * ACT2FN[self.activation](gate.to(torch.float32))`，
`self.activation` 硬编码为 `"sigmoid"`（不吃 `config.hidden_act`）。

### 实测数字

| 量 | 实测值 | 含义 |
|---|---|---|
| `linear_lower_bound` | -5.0 | log-decay 的下界 |
| floor `e^-5` | 0.006737947 | 单步衰减的下限 |
| `A_log` 形状 | (64,) | 每头一个时间尺度 |
| `dt_bias` 形状 | (8192,) | 每 (头, 通道) 一个偏置 |
| 遗忘门参数量 | 1,581,120 | q/k/v 投影的 1.57% |
| 门控范数两组投影 | 1,572,864 | `g_a_proj` + `g_b_proj` |
| 初始 decay 均值 | 0.911 | 训开始时每步记住约 91% |
| `exp(-104)` | 0.0 | fp32 里累积 21 步到 floor 就归零 |

## 与后续课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| KDA 递推的完整推导（delta 规则、初始状态） | L4-02 |
| 分块形式怎么变成矩阵乘；`decay_mask` 的完整用法 | L4-03 |
| 短卷积与 `l2norm` 在 KDA 里的作用 | L4-04 |
| `Glm5NextTextLinearAttention` 的整体装配 | L4-05 |
| 另外两种 RMSNorm 与 `rms_norm_eps` 的传递 | L3-01 |
| MLP 的 SwiGLU 截断（截的是另一头） | L3-02 |
| 一层里三个子块的顺序 | L3-08 |
| `recurrent_states` 存在 Cache 的哪里 | L5-01 |

## 验收点

- [x] 保真门禁：13 个引用块全部逐字来自 `modeling_glm5_next.py` 且位置连续
- [x] 覆盖度门禁：1 个源文件被声明（与 `tools/plan.py` 里本课的 `files` 一致）
- [x] 参数门禁：无非法参数
- [x] 语法检查：`node --check` 通过，9 幕字段完整
- [x] 渲染门禁：无 JS 错误、无布局溢出、交互可用（两种分辨率）
- [x] **验收点 1「能解释下界截断防的是什么」**：第 4 幕给出 `(-5, 0)` 与 floor
      `e^-5 = 0.006737947` 的实测值，第 6 幕给出实测订正 —— 防的是**单步衰减归零**，
      不是分块形式的 `exp` 上溢（后者由掩码方向解决）
- [x] **验收点 2「能说出遗忘门在 KDA 递推中的位置」**：第 6 幕把递推拆成五步，
      指出 `last_recurrent_state = last_recurrent_state * g_i` 是遗忘发生的唯一一行，
      位置在「读取记忆」之前

**一句话总结**：

> 遗忘门把「上一步的记忆留多少」参数化为一个 log-decay：`g = -5 x sigmoid(A x score)`；
> **下界 -5.0 保证的是单步衰减 >= `e^-5`（不会一步清零），不是防分块形式的上溢**（实测订正）；
> 而 `A_log` 的初始化（有下界置 0、无下界 log-uniform）**必须和 forward 走哪一支配套**。
