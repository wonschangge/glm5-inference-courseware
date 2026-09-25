#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L3-07 · source.md 生成器

设计要点：**代码块一律从源文件切片**，绝不手打 —— 这样保真门禁不可能因为笔误失败。
散文部分写在下面的模板里，代码用 {Bn} 占位符替换。

用法：
    python3 _data/recon/build_l307.py            # 生成 source.md
    python3 _data/recon/build_l307.py --verify   # 只校验 lesson.js 里每幕的 code 与行号
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))                 # courseware/
UP = os.environ.get(
    "GLM5_UPSTREAM",
    os.path.abspath(os.path.join(ROOT, "..", "upstream-transformers", "src", "transformers")),
)
REL = "models/glm5_next/modeling_glm5_next.py"
LESSON = os.path.join(ROOT, "L3-backbone", "L3-07")

# 引用块：名字 -> (起, 止) 1 基闭区间。行号用 sed -n 复核过。
BLOCKS = {
    "B1": (733, 735),      # 调用点：遗忘门 + 写入门
    "B2": (348, 353),      # ForgetGate.__init__
    "B3": (356, 361),      # forward 前半：低秩投影 / dt_bias / A_log
    "B4": (363, 365),      # 下界截断
    "B5": (367, 371),      # softplus 支
    "B6": (377, 381),      # RMSNormGated.__init__
    "B7": (383, 395),      # RMSNormGated.forward
    "B8": (501, 513),      # 递推循环
    "B9": (766, 769),      # 门控范数在 KDA 里的位置
    "B10": (1379, 1379),   # _keep_in_fp32_modules_strict
    "B11": (1385, 1394),   # A_log 初始化
    "B12": (1396, 1407),   # dt_bias 初始化
    "B13": (570, 570),     # decay_mask（L4-03 的范围，本课借来做数值论证）
}


def src_lines():
    with open(os.path.join(UP, REL), encoding="utf-8") as f:
        return f.read().split("\n")


def block_text(lines, a, b):
    return "\n".join(lines[a - 1:b])


SOURCE_MD = """<!-- glm5-coverage
models/glm5_next/modeling_glm5_next.py
-->

# L3-07 · 遗忘门与门控范数 — 源文件

**一层 KDA（线性注意力）里有两个门：一个决定上一步的记忆留多少，一个决定这一步的输出放多少。**
本课只做两件事：把这两个门拆到参数级，并说清 `linear_lower_bound = -5.0` 这个数字到底防的是什么。

视角：`hidden_states` 进了 KDA 子层以后，哪几次乘法是「门」，这些门的取值范围是谁定的。

| 文件 | 行数 | 本课引用块 |
|---|---|---|
| `models/glm5_next/modeling_glm5_next.py` | 2444 | 13 |

**行数口径**：`wc -l` 实测 **2444** 行；按 `text.split("\\n")` 计是 2445（文件末尾有换行符，
多出一个空串）。作业书写的是 2445 —— 本课按 `wc -l` 记 2444，所有引用行号都用 `sed -n` 复核过。

**本课的实测来源**（`_data/recon/probe_l307.py`，CPU / torch 2.14.0+cpu /
transformers 5.18.0.dev0；`flash-linear-attention` 未安装，`fused_recurrent_kda` 与
`chunk_kda` 都走参考 PyTorch 实现）：

```text
linear_attn 相关配置（Glm5NextTextConfig 默认值，实测）
    hidden_size = 4096    linear_head_dim = 128    linear_num_heads = 64
    qkv_dim     = 128 * 64 = 8192
    linear_lower_bound = -5.0     rms_norm_eps = 1e-05    initializer_range = 0.02

Glm5NextTextForgetGate 参数量（默认 config 实测 1,581,120）
    f_a_proj.weight  (128, 4096)      524,288
    f_b_proj.weight  (8192, 128)    1,048,576
    dt_bias          (8192,)            8,192   每个 (头, 通道) 一个
    A_log            (64,)                 64    每个头一个
    对照：q/k/v 三个投影 = 100,663,296   ->  遗忘门只占它的 1.57%

_init_weights 跑完之后的实测取值
    bound = -5.0 : A_log 全 0 -> decay_rate 全 1.0；dt_bias 落在 [-6.857, -2.290]
    bound = None : A_log = log(U(1,16)) -> decay_rate 落在 [3.606, 15.780]（4 个头采样）
    两者 dt_bias 相同：softplus(dt_bias) 落在 [1.05e-3, 9.64e-2]
```

> 上面这段是**示意性的汇总**，用 `text` 块标出 —— 它不是源文件的逐字引用，
> 因此不参与保真校验。下面每一段 `python` 块都是逐字引用。

---

## 一、定位：一层 KDA 里有两个门

先看调用点。这两行是后面所有内容的锚：`g` 是**遗忘门**，`beta` 是**写入门**。

<!-- src: {REL} -->
```python
{B1}
```

**读法**：
- 注释写的是 `# Forget gate and input gate`，但两者的**参数化方式完全不同**：`g` 是
  **对数衰减率**（后面要过 `exp`），`beta` 是 **0~1 的写入门**（直接乘在 delta 上）。
  一个是 log 参数化，一个是概率参数化 —— 数值保护方式也因此不同。
- 同样是「门」，作者给遗忘门的自由度大得多：它独占一个 `Glm5NextTextForgetGate`
  （低秩投影 + 两个专属参数 + 专属初始化），而写入门只是 `b_proj` 加一个 sigmoid。
  理由在第六节：衰减率是递推里**唯一会累积**的量，一步偏了，后面每一步都在乘它。
- 这一行同时给出了**验收点 2** 的答案：遗忘门在递推**之前**算好，在递推的每一步里
  以 `exp(g)` 的身份乘在上一步的状态上。递推代码见第六节，展开留在 L4-02。
- 回顾 L0-01：那一课说 45 层里有 34 层是 `linear_attention`。本课拆的就是这 34 层里
  最小的两个算子 —— 它们每层各出现一次。

---

## 二、遗忘门的构造：低秩投影 + 两个专属参数

<!-- src: {REL} -->
```python
{B2}
```

**读法**：
- 两个 `nn.Linear` 都带 `bias=False`，中间夹一个 `head_dim = 128` 的瓶颈：
  `4096 -> 128 -> 8192`。这就是「低秩」：门的容量被压到 128 维，参数量 1,581,120，
  是同一层 q/k/v 三个投影（100,663,296）的 **1.57%**（实测）。为什么敢这么省？
  因为门只需要吐出「每头每通道的衰减分数」，不需要承载 q/k/v 那样的信息量。
- `dt_bias` 的长度是 `self.qkv_dim = head_dim * num_heads = 8192`：
  **每个 (头, 通道) 一个偏置**。
- `A_log` 的长度只有 `self.num_heads = 64`：**每个头一个时间尺度**，同一头内 128 个
  通道共享。两者的形状差异决定了下一节 `decay_rate` 的广播方式 —— 这是本课最容易
  看漏的一处。
- 两个参数都用 `torch.empty` 而不是 `zeros`：它们的初值**全部**由 `_init_weights`
  的专门分支负责（第七节）。反过来说，任何绕过 `_init_weights` 的构造都会拿到未初始化
  的内存 —— 这就是那句 `# NOTE: This is incredibly important` 的分量。
- `self.safe_gate_lower_bound = config.linear_lower_bound`：把 config 里的值抄进模块。
  forward 每次都要判断它是不是 `None`，抄一份省得每步翻 config；副作用是**它成了
  模块属性**，`_init_weights` 正是靠读这个属性来决定 A_log 怎么初始化（第七节）。

---

## 三、★ 前向算账：A 是每头的斜率，dt_bias 是每通道的偏置

六行代码里有三次 `.float()`、两次 `view`、一次广播。逐个对形状。

<!-- src: {REL} -->
```python
{B3}
```

**读法**：
- `hidden_shape = (*hidden_states.shape[:2], -1, self.head_dim)`：`(b, s, 8192)` 折成
  `(b, s, 64, 128)` —— 64 行是头，128 列是头内通道。
- `forget_gate = self.f_b_proj(self.f_a_proj(hidden_states))`：形状 `(b, s, 8192)` 的
  原始分数。注意**它还不是门**，只是一个可正可负的打分。
- `+ self.dt_bias.float().view(1, 1, -1)`：`(8192,)` 沿最后一维广播 —— 展开后恰好是
  `(64, 128)`，**每一格都有自己的偏置**。这是「逐通道时间尺度」的来源。
- `A_log.view(1, 1, self.num_heads, 1)`：形状 `(1, 1, 64, 1)` —— 一个头一个值，
  靠广播铺满这一头的 128 个通道。
- 于是 `decay_rate = torch.exp(A_log)` 是**每头一个**的乘性系数（就是 FLA / Mamba 里的
  `A`，这里以 `log` 形式存放，所以名字叫 `A_log`）。
- 三处 `.float()` 不是装饰：bf16 只有 8 位有效位，而 `dt_bias` 的量级是 -2 ~ -7、
  raw 分数在初始状态下只有 1e-2 量级，两者相加会把小数吃掉。这一层加法必须在 fp32 里做。
- 回顾 L3-06：那一课讲 mHC 的 `post` / `comb` 也是逐元素门控；区别在于 mHC 的门形状是
  `(b, s, 4, 4096)`（4 条流 × 隐维），本课的门形状是 `(b, s, 64, 128)`（头 × 头内维）。
  两者都靠**广播**把参数铺到 token 维度上。

---

## 四、★ -5.0：把 log-decay 关进 (-5, 0)

<!-- src: {REL} -->
```python
{B4}
```

**读法**：
- 只要 `safe_gate_lower_bound` 不是 `None`，就走这一支并**直接 return** —— 下面那段
  softplus 代码完全不执行。所以 `linear_lower_bound` 决定的不只是一个数值，而是**走哪条
  公式**。
- `-5.0 * sigmoid(...)` 是一个**值域固定**的函数：`sigmoid` ∈ (0, 1)，乘 -5.0 之后
  `g` ∈ (-5, 0)。无论 `decay_rate`（=A）和 raw 分数怎么变，输出都跑不出这个区间。
- 它同时**改变了 A 的角色**：这里 A 是 sigmoid 内部的**斜率**；在没有下界的那一支里
  A 是**乘法增益**（下一节）。同一个参数，两种量纲。
- 手算表（`probe_l307.py` 第 10 节，fp32 实测；`g = -5 * sigmoid(decay_rate * raw)`，
  A_log=0 时 decay_rate = 1.0）：

```text
  raw 分数    g = -5·sigmoid(raw)     decay = exp(g)     每步遗忘
    -30.0        -0.000000000          1.000000000       0.0000%
    -15.0        -0.000001530          0.999998450       0.0002%
     -4.0        -0.089931048          0.913994193       8.6006%
     +0.0        -2.500000000          0.082084998      91.7915%
     +4.0        -4.910068989          0.007371980      99.2628%
    +15.0        -4.999998093          0.006737960      99.3262%
    +30.0        -5.000000000          0.006737947      99.3262%
    +60.0        -5.000000000          0.006737947      99.3262%

  floor：g = -5 时 decay = e^-5 = 0.006737947（实测值）
```

- 读表：`raw = 0` 时门给出「忘掉 91.79%」；`raw = +15` 与 `raw = +60` 的结果在第 6 位
  小数上才分得开 —— **打分超过约 +15 之后这门就饱和了**，再大也只是 99.33% 的遗忘率。
  这就是「截断（clamp）」在函数形式上的体现：它不是 `clamp()`，而是 sigmoid 的饱和。
- 一个直接推论：单步衰减率**永远 ≥ e^-5 ≈ 0.006738，永远不等于 0**。
  第六节会用实测说明这个推论的实际效力与边界。

---

## 五、没有下界的那一支，以及它为什么需要另一套初始化

把 `config.linear_lower_bound` 设成 `None`（走真实的配置路径：
`linear_attn_config={"safe_gate": False, "gate_lower_bound": None}`，实测这样才会得到
`None`；只写 `gate_lower_bound: None` 会被 `safe_gate` 的默认值 `True` 补回 -5.0），
就落到下面这一段。

<!-- src: {REL} -->
```python
{B5}
```

**读法**：
- `torch.where(g > 20.0, g, torch.log(1.0 + torch.exp(g)))` 是**手写 softplus**：
  `exp(g)` 在 g > 88 时 fp32 上溢，所以 g 超过 20 之后直接用 g（注释里写明：Softplus
  在大值处就等于 x）。这里 `g` 是局部变量名，指的是 raw 分数，**不是**第四节那个门。
- `return -decay_rate * g_softplus`：A 在这里是**乘法增益**，而 A = exp(A_log) 不封顶、
  raw 也不封顶 —— 这一支允许门取到任意负值。实测：A=16、raw=20 时
  `g = -16 * softplus(20) = -320.0`，而 `exp(-320)` 在 fp32 里**恰好是 0.0**
  （probe 第 6b 节）。换句话说，这一支允许模型**一步之内把状态乘成 0**。
- 再看有下界那一支的 A：A=16、raw=-4.5 时 sigmoid 的导数只有 **5.38e-32**
  （probe 第 12 节）—— 门被**冻住**，梯度传不回去。把两件事放在一起就得到本课的
  第三个洞察：**A_log 的初始化必须和 forward 的分支配套**，见第七节。
- 配置层面的证据（probe 第 0b 节，走真实构造路径实测）：

```text
  linear_attn_config 传入值                                    linear_lower_bound 结果
  {}                                                            -5.0
  {"gate_lower_bound": None}                                    -5.0   （safe_gate 默认 True 补回）
  {"safe_gate": False, "gate_lower_bound": None}                None   （唯一能关掉下界的写法）
  {"safe_gate": False}                                          -5.0
  {"gate_lower_bound": -3.0}                                    -3.0
```

---

## 六、遗忘门在 KDA 递推里的位置

这是**验收点 2** 要能指出来的一段。函数是 `recurrent_kimi_delta_attention`，
下面只截循环体。

<!-- src: {REL} -->
```python
{B8}
```

**读法**：
- `g_i = g[:, i][..., None].exp()`：门在递推里的**唯一落点**。注意传进来的 `g` 是
  log-decay（第四节那个负数），这里 `exp` 之后才变成 0~1 的衰减率。
- `last_recurrent_state = last_recurrent_state * g_i`：**遗忘发生的就是这一行** ——
  旧状态整体乘一个 ≤1 的数。
- `kv_mem = (last_recurrent_state * k_i[..., None]).sum(dim=-2)`：用衰减后的状态去读记忆。
- `delta = (v_i - kv_mem) * b_i`：delta 规则；`b_i` 就是第一节那个 `beta`（写入门，
  0~1）。读出来的和想写的差多少，就补多少。
- `last_recurrent_state = last_recurrent_state + k_i.unsqueeze(-1) * delta.unsqueeze(-2)`：
  写入是**加法**。所以衰减只作用在「旧记忆」上，本步刚写进去的内容不会被同一行衰减。
- 顺序是 **先衰减 → 再读取 → 最后写入**。如果改成先写入再衰减，本步写入的内容会在
  同一个位置被多衰减一次，等价于把写入门 `b_i` 也乘上了 decay —— 这是个很容易写错、
  又不容易在 loss 上看出来的地方。
- 实测（probe 第 5 节：k=v=0、beta=0、初始状态全 1，只让那一行乘法起作用）：

```text
  gate     24 步后的 |state|        手算 exp(gate*24)
  -5.000   0.000000e+00             exp(-120) = 7.667648e-53
  -1.000   3.775137e-11             exp(-24)  = 3.775135e-11
  -0.050   3.011944e-01             exp(-1.2) = 3.011942e-01
```

- gate=-1 与 -0.05 两行与手算吻合到 6 位有效数字，说明 `exp(g)` 的语义就是逐字面的
  「每步乘一个衰减率」。而 gate=-5（正好是下界）那一行**精确为 0** —— 不是下界没生效，
  是 fp32 撑不住：`exp(-104)` 实测就是 0.0（`exp(-88)` 还有 6.05e-39）。
  逐步下界保证的是 `decay ≥ e^-5`，累积到 **21 步**仍然会在 fp32 里归零。
- 所以下界买到的是一句**有条件**的承诺：「一步最多忘掉 99.33%，不会一步清零」。
  这句话在上面那张表里是可以被验证的；把「一步」读成「永远」，就把它读大了。

---

## 七、★ 初始化：A_log 与 dt_bias 从哪来

`Glm5NextPreTrainedModel._init_weights` 里有一个专门分支。先看 A_log。

<!-- src: {REL} -->
```python
{B11}
```

**读法**：
- 有下界 → `init.zeros_(module.A_log)` → `decay_rate = exp(0) = 1.0`。实测：4 个头全 1.0。
- 没下界 → 先 `init.uniform_(A_log, a=1.0, b=16.0)` 塞进 [1,16]，再取 `log` 写回 →
  `decay_rate = exp(A_log) ~ U(1, 16)`。实测 4 个头采样到 [3.606, 15.780]。
- 那句 `# NOTE: This is incredibly important so keep it this way at all costs` 就是写给这个
  if/else 的：**初始化与 forward 的分支是一对，不能拆开改**。
- 为什么有下界时必须 A_log = 0？因为那里 A 是 sigmoid 的斜率。A=16 一上来就把 sigmoid
  推饱和（实测梯度 5.38e-32），门失去可学习性；A=1 时门落在 sigmoid 的线性区
  （raw=-4.5 → g=-0.0549，梯度 1.087e-2，probe 第 12 节）。
- 反过来，为什么没下界时用 log-uniform？那里 A 是乘法增益，`A * softplus(raw)` 决定门的
  大小：A ∈ [1,16] 给每个头一个不同的时间尺度初值；若也写 0（=1），64 个头会从同一个
  时间尺度出发，等于放弃了这个参数的自由度。

同一个分支里还有 dt_bias，**它没有 if**（两种分支共用）：

<!-- src: {REL} -->
```python
{B12}
```

**读法**：
- 三步：先抽 `dt ~ U(log(1e-3), log(1e-1))`；再 `dt = module.dt_bias.exp().clamp_min(1e-4)`
  把 [1e-3, 1e-1] 真正取出来；最后用**逆 softplus** `dt + log(-expm1(-dt))` 写回参数。
- 结果：`softplus(dt_bias) ∈ [1e-3, 1e-1]`。实测（默认 config）：`dt_bias ∈ [-6.857, -2.290]`，
  `softplus(dt_bias) ∈ [0.00105, 0.09644]`。手算对照：`softplus⁻¹(1e-3) = -6.907255`，
  `softplus⁻¹(1e-2) = -4.600166`，`softplus⁻¹(1e-1) = -2.252168`。
- 这套写法来自 FLA / Mamba 的 `dt`（离散化步长）惯例：那些实现里 `dt` 要过一层 softplus，
  所以初始化必须写在「softplus 之前」的坐标系里 —— 这也是 `clamp_min(1e-4)` 存在的理由
  （保证 `expm1` 不会退化）。
- **但这个模型的 forward 里，`dt_bias` 是直接相加、不过 softplus**（第三节那一行）。
  所以这个初始化的实际效果是：给 8192 个通道各一个 **-2.3 ~ -6.9 的负偏置**，把初始的
  门整体推向「慢遗忘」一侧。实测初始状态下（真模块 + 真初始化 + 随机输入）：
  `g ∈ [-1.294, -0.00226]`，均值 -0.0999，`decay = exp(g)` 均值 0.911 —— 也就是说，
  训练开始时每一层大约「每步记住 91%」。
- 这条值得单独记一笔：**注释和参数名说的是 FLA 的语义，公式里的用法已经变了**。
  照搬 FLA 的 `dt` 直觉去解释这一行会解释错。

---

## 八、实测订正：下界防的不是分块形式的上溢

这一节记录一个**看起来很顺、但被实测推翻**的推断。它涉及 `chunk_kimi_delta_attention`
里的一行（属于 L4-03 的覆盖范围，本课只借它做数值论证）：

<!-- src: {REL} -->
```python
{B13}
```

**读法（以及订正过程）**：
- 直觉推断：`g` 在块内做过 `cumsum`，64 个 token 的下界累积到 -320；两个位置相减能到
  +315，而 `exp(315)` 在 fp32 里是 `inf` → 「下界就是防这个上溢的」。
- 实测（probe 第 6c 节）：差的**方向**是 `g[i] - g[j]`，而掩码 `strict_mask`（严格上三角）
  恰好把 `g[i] - g[j] > 0` 的那一半置成 `-inf`。所以 `decay_mask` 的取值恒在 [0, 1]：
  gate=-5 时非对角最大值恰好是 `6.737947e-03`，正是 `e^-5`。**它不会上溢。**
- 真正会归零的是另一支：`k_cumdecay = attn @ (k_beta * g.exp())` 里的 `exp(cumsum)`。
  实测：gate 全按到 -5 时，64 个位置里有 **44 个** `exp(cumsum)` 精确为 0；
  gate=-1 时最小 1.6e-28（一个都不为 0）；初始化附近的 gate=-0.05 时最小 4.1e-02。
- 直接跑真实实现（`chunk_kimi_delta_attention`，chunk=64）也不出 nan/inf：gate 全按 -5
  时输出 max|out| = 2.503971；gate=-320 时 max|out| = 2.500376。
- 结论：下界在分块形式里买到的是「**老 token 的权重不会在块内被整片抹平**」
  （gate=-5 时抹平 44/64，gate=-0.05 时一个都不抹），**不是**「防上溢」——
  上溢那一路作者是用**掩码方向**解决的。两条保护各管一边。
- 这也是本项目那条老规矩的又一个例子：**推断 → 实测 → 回改**。只看 `cumsum` 的量级
  会得出错误结论，把方向算清楚（或者直接跑一遍）才知道哪一支才是危险的。

---

## 九、门控范数：第二个门

`Glm5NextTextRMSNormGated` 是 KDA 子层的收尾算子。先看构造。

<!-- src: {REL} -->
```python
{B6}
```

**读法**：
- `self.weight = nn.Parameter(torch.ones(hidden_size))`：RMSNorm 的可学习缩放，初值 1，
  与 `_init_weights` 里的 `init.ones_(module.weight)` 一致（两处口径相同）。
- `self.activation = "sigmoid"`：**硬编码的字符串**，不吃 `config.hidden_act`。
  对比 `Glm5NextTextMLP.__init__` 里的 `self.act_fn = ACT2FN[config.hidden_act]`
  （L3-02 会讲）：同一个文件里，MLP 的激活可配，门控范数的激活不可配 —— 这是刻意的，
  门必须是 0~1 的放行系数。
- `eps` 的默认值是 1e-6，但调用方传的是 `config.rms_norm_eps`（第 660 行，实测 1e-5）。
  同一个文件里 `Glm5NextTextRMSNorm` 的默认也是 1e-6 —— **默认值只是默认值，真正生效的
  是调用点传进来的那个**。L3-01 讲的「三种 RMSNorm」里，这是最容易混淆的一处。

<!-- src: {REL} -->
```python
{B7}
```

**读法**：
- 第一行把 `input_dtype` 存起来，最后一行再转回去：整个归一化 + 门控过程都在 fp32 里做。
- `# Strict FP32 norm (do not downcast on the weights)`：不只 `hidden_states` 转 fp32，
  `self.weight` 也 `.to(torch.float32)`。**对比 L3-01**：`Glm5NextTextRMSNorm.forward`
  写的是 `self.weight * hidden_states.to(input_dtype)` —— 那边先把结果降回 input dtype
  再乘权重。两处的差别就是这句注释说的 "Strict"。
- `variance = hidden_states.pow(2).mean(-1, keepdim=True)`：只算均方、不减均值 ——
  这就是 RMSNorm 与 LayerNorm 的全部区别。
- `hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)`：
  `rsqrt` 一次搞定，`eps` 加在方差里防零。
- `# Apply gating` 之后那一行是门控：`sigmoid(gate)` 逐 (头, 通道) 放行。
  **顺序是先归一化、再门控**，不能反：归一化是逐 token 的统计量（`mean(-1)`），
  门控是逐通道的系数；先门控会把统计量本身改掉，归一化的分母就不再是「这一步激活的
  均方」了。
- `gate=None` 会直接抛 `AttributeError: 'NoneType' object has no attribute 'to'`
  （实测）—— 这个函数**不允许不传门**。
- 实测：bf16 输入 → bf16 输出，中间全 fp32；与手工复算
  `x.float() * rsqrt(mean(x²)+eps) * weight.float() * sigmoid(gate.float())` 的
  `allclose` 为 True。
- 最后一行 `.to(input_dtype)` 值得和它头顶那句注释一起读。第 374 行那句注释写的是
  「FLA 的实现不转回 `input_dtype`，也许我们也该这样」，而函数体**转回去了**。
  注释是待办事项，不是行为说明 —— 读代码时以代码为准。

---

## 十、门控范数站在哪一步

<!-- src: {REL} -->
```python
{B9}
```

**读法**：
- `gate = self.g_b_proj(self.g_a_proj(hidden_states)).view(hidden_shape)`：这里**又**从
  `hidden_states` 算了一个门，走的是另一组低秩投影（`4096 -> 128 -> 8192`，
  参数量 1,572,864，与遗忘门 1,581,120 几乎一样大）。也就是说一层 KDA 里有**两个独立
  的门**：一个管递推状态（`g`，第六节），一个管输出（这里的 `gate`）。
- `output = self.o_norm(core_attn_out, gate)`：门控范数作用在 KDA 的输出上，**不是**
  作用在状态上。
- 然后才 `o_proj`。三行的顺序 = 归一化 → 门控 → 输出投影，一步都不能换。
- 回顾 L0-01：那一课把一层压成「掩码 + 注意力 + MLP」三块。本课补的是其中
  `linear_attention` 那一块内部的最后两步 —— 换句话说，L0-01 的图里 KDA 那一格，
  到这里才算能说出「里面有两个门」。

---

## 十一、这两个参数永远不降精度

<!-- src: {REL} -->
```python
{B10}
```

**读法**：
- `_keep_in_fp32_modules_strict` 四个名字里，`dt_bias` 与 `A_log` 都是本课的参数。
  `strict` 的含义是：即使模型整体以 bf16 加载，这两个参数也保持 fp32，
  在权重转换 / 量化流程里不被改成低精度。
- 这与 forward 里那几处 `.float()` 是**两道保险**：一道在加载层（这个列表），
  一道在计算层（`.float()`）。写自定义门控时值得照抄这个组合 —— 门控参数本身很小
  （8192 + 64 个数），用 fp32 存没有任何成本压力，但一旦降精度就会直接毁掉
  `exp(decay_rate * raw)` 的精度。

---

## 十二、与后面课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| KDA 递推的完整推导（delta 规则、初始状态、输出） | L4-02 |
| 分块形式怎么把递推变成矩阵乘；`decay_mask` 的完整用法 | L4-03 |
| 短卷积与 `l2norm` 在 KDA 里的作用 | L4-04 |
| `Glm5NextTextLinearAttention` 的整体装配（缓存、conv 状态） | L4-05 |
| 另外两种 RMSNorm 与 `rms_norm_eps` 的传递 | L3-01 |
| MLP 的 SwiGLU 截断（同样是「截断」，但截的是另一头） | L3-02 |
| 一层里三个子块的顺序（两个门分别挂在哪） | L3-08 |
| 递推状态 `recurrent_states` 存在 Cache 的哪里 | L5-01 |

**一句话总结**：

> 遗忘门把「上一步的记忆留多少」参数化为一个 log-decay：`g = -5 * sigmoid(A * score)`；
> **下界 -5.0 保证的是单步衰减 ≥ e^-5（不会一步清零），而不是防分块形式的上溢**（实测订正）；
> 而 `A_log` 的初始化（有下界时置 0、无下界时 log-uniform）**必须和 forward 走哪一支配套**。
"""


def build():
    lines = src_lines()
    md = SOURCE_MD
    md = md.replace("{REL}", REL)
    for name, (a, b) in BLOCKS.items():
        md = md.replace("{" + name + "}", block_text(lines, a, b))
    os.makedirs(LESSON, exist_ok=True)
    out = os.path.join(LESSON, "source.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    n = len(re.findall(r"^```python$", md, re.M))
    print(f"写入 {out}：{len(md.splitlines())} 行，{n} 个 python 引用块")
    assert n == len(BLOCKS), (n, len(BLOCKS))
    return out


# --------------------------------------------------------------------------
# --verify：把 lesson.js 里每一幕的 code 与源文件行号对上
# --------------------------------------------------------------------------
SCENE_RE = re.compile(r"codeStart:\s*(\d+),\s*\n\s*code:\s*`([^`]*)`", re.M)


def verify():
    lines = src_lines()
    js = os.path.join(LESSON, "lesson.js")
    with open(js, encoding="utf-8") as f:
        src = f.read()
    ms = SCENE_RE.findall(src)
    print(f"lesson.js 里找到 {len(ms)} 幕带 codeStart/code")
    bad = 0
    for i, (start, code) in enumerate(ms, 1):
        start = int(start)
        got = code.split("\n")
        want = lines[start - 1:start - 1 + len(got)]
        if got == want:
            print(f"  ✓ 第 {i} 幕 codeStart={start} 共 {len(got)} 行，与源文件逐字一致")
        else:
            bad += 1
            print(f"  ✗ 第 {i} 幕 codeStart={start} 不一致：")
            for k in range(max(len(got), len(want))):
                g = got[k] if k < len(got) else "<缺失>"
                w = want[k] if k < len(want) else "<缺失>"
                if g != w:
                    print(f"      行 {k+1}: lesson.js={g!r}")
                    print(f"              源文件  ={w!r}")
                    break
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--verify" in sys.argv:
        sys.exit(verify())
    build()
