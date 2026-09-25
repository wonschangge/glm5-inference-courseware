<!-- glm5-coverage
models/glm5_next/modeling_glm5_next.py
activations.py
-->

# L3-02 · MLP 与 SwiGLU 截断 — 源文件

**这一课只讲一层里最后一个算子：前馈网络（MLP）。** 它由三个投影组成，中间夹着一个
门控乘法，乘法之前各有一把钳子 —— `swiglu_limit = 10.0`。

视角：`hidden_states` 走进 MLP，被投影成两份（gate / up），钳位、相乘、再投回来。
**为什么是"两份"而不是"一份"，为什么还要钳位** —— 这是本课要回答的两个问题。

| 文件 | 行数 | 本课引用块 |
|---|---|---|
| `models/glm5_next/modeling_glm5_next.py` | 2444 | 8 |
| `activations.py` | 369 | 5 |

> 行数是 `wc -l` 实测。作业书大纲里写的是 2445 / 370 —— **实测是 2444 / 369**，
> 差的 1 行在文件末尾的换行符上（`wc -l` 数换行符，大纲多半是按最后一行行号记的）。
> 本课以实测为准，下面所有行号都是实测行号。

**本课用到的实测数字**（用仓库自带的 `.venv/bin/python` 真跑出来的，不是估算）：

```text
Glm5NextTextConfig() 的默认值
    hidden_size            = 4096
    intermediate_size      = 12288      稠密 MLP 的中间维（= 3 × 4096）
    moe_intermediate_size  = 2048       路由专家与共享专家的中间维
    hidden_act             = "silu"     → ACT2FN["silu"] → SiLUActivation 实例
    swiglu_limit           = 10.0
    n_routed_experts       = 288        num_local_experts 是它的别名
    num_experts_per_tok    = 8
    n_shared_experts       = 1

一个稠密 MLP（4096 → 12288 → 4096，3 个无偏置投影）
    参数量 = 3 × 4096 × 12288 = 150,994,944 ≈ 151.0M

一个稀疏层（Glm5NextTextMoE）
    experts.gate_up_proj  288 × 4096 × 4096 = 4,831,838,208
    experts.down_proj     288 × 4096 × 2048 = 2,415,919,104
    shared_experts        3 × 4096 × 2048   =    25,165,824
    gate（路由）          288 × 4096        =     1,179,648
    合计 ≈ 7.27B / 层
    每 token 真正参与计算的：top-8 专家 + 共享专家 = 226,492,416 ≈ 226.5M
    → 存储 / 激活 = 32.0×

截断的实测效果（gate、up 都取 std=18 的随机数，各 4000 个样本）
    max|SiLU(gate)·up|   未截断 2419.451  →  截断后 99.995
    被截断的元素比例     69.9%
    clamp 的导数         区间内为 1；区间外为 0（越界即断梯度）
```

> 上面两段是**示意性的汇总**，用 `text` 块标出 —— 它们不是源文件的逐字引用，
> 因此不参与保真校验（脚本本身也不计入覆盖率，覆盖域只统计 `src/transformers/**/*.py`）。
> 下面每一段 `python` 块都是逐字引用。

---

## 一、MLP 是层里最后一个算子

回顾 L0-01：一层之内的顺序是 **mHC 混合 → 归一化 → 子层 → mHC 写回**，而这个"子层"
出现两次 —— 先注意力，后前馈。本课讲的是第二次：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        residual = hidden_states
        post, comb, hidden_states = self.ffn_hc(hidden_states)
        # Feed forward
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
```

**读法**：
- 顺序不能记成"norm → MLP → 加残差"。这里的 `post_attention_layernorm` 作用在
  **mHC 混合后的那一条流**上，MLP 的输出随后被 `post` / `comb` 写回 4 条流（见 L3-06）。
- `self.mlp` 是个二选一的槽位：`config.mlp_layer_types[layer_idx] == "sparse"` 时是
  `Glm5NextTextMoE`，否则是 `Glm5NextTextMLP`（L0-01 第七节已引过那两行）。
  实测排班是 **前 3 层稠密 + 42 层稀疏**，所以本课讲的 `Glm5NextTextMLP` 有两个身份：
  前 3 层的全部，以及稀疏层里的**共享专家**。
- 记住这个形状：进去是 `(batch, seq, 4096)`，出来还是 `(batch, seq, 4096)`。
  MLP 是这一层里唯一"不改形状、只换内容"的算子。

---

## 二、三个投影，都是 `bias=False`

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
- `gate_proj` / `up_proj` 把 4096 抬到 12288，`down_proj` 再压回 4096。
  **`bias=False` 是参数量能口算的原因**：3 × 4096 × 12288 = 150,994,944（实测
  `sum(p.numel() ...)` 也是这个数），一个偏置都没有。
- 中间维 12288 = 3 × 4096。**SwiGLU 家族的通行做法**（这一条是背景知识，不是本文件的实测）：
  门控多了一个矩阵，为了和"两层 4 倍宽"的老 FFN 参数量持平，中间维取 8/3 倍
  （4096 的话是 10922.67，通常再向上取整）。这里取的是 **3 倍**，
  比"参数量持平"**多 12.5%**：实测本课 MLP 是 150,994,944，而 4 倍宽老 FFN
  （2 × 4096 × 16384）是 134,217,728。也就是说这个模型是**主动买宽**，不是持平。
- `intermediate_size` 是**构造参数**，给了就用给的，不给才用 config 的 ——
  这个钩子是给共享专家留的（见第五节 `Glm5NextTextMoE.__init__`）。
- `self.act_fn = ACT2FN[config.hidden_act]` 这行**在构造时执行一次**，
  存下来的是**实例**（不是函数、不是字符串）。`config.hidden_act = "silu"`
  在这里被解析成 `SiLUActivation()`。这条链的细节在第六、七节。
- `self.swiglu_limit = config.swiglu_limit`：把 10.0 从 config 抄到模块上。
  forward 里于是只碰 `self`，不碰 config —— 这是这份代码里反复出现的写法。

---

## 三、★ 前向：一行 SwiGLU，外加两把钳子

整份文件里最重要的一行就是最后那行 `return`：

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

**读法**：把它写成公式（`L = swiglu_limit = 10.0`）：

```text
SwiGLU(x) = W_down( SiLU(clamp(W_gate·x, max = L)) ⊙ clamp(W_up·x, -L, +L) )

    SiLU(z) = z · σ(z) = z / (1 + e^(-z))
    ⊙       逐元素相乘
```

- 两个投影**分工不同**：`up` 提供"内容"，`gate` 提供"闸门"。闸门经过 `SiLU` 后被压到
  `(-0.2785, +∞)`：实测 `SiLU` 的最小值是 **−0.278465**（在 `x = −1.2785` 处取到），
  正的一侧则近似线性放大。
  所以 `gate` 决定"这一维放多少过去"，`up` 决定"过去的是什么"。
- 这正是 SwiGLU 与老式 `W_down(act(W_up x))` 的区别：**乘法把两条独立的线性变换耦合起来**，
  表达能力比单个逐元素非线性强（`W_down(gate ⊙ up)` 里含二次项）。
- 注意两把钳子的**范围不对称**：
  `gate` 只有上界 `max=10`（没有下界，因为 SiLU 在负侧本来就有下界 −0.2785），
  `up` 是双侧 `[-10, 10]`。
  这个不对称不是笔误 —— 它精确对应两个张量的**有界性**：`SiLU` 负侧自带有界，
  正侧无界；`up` 两侧都无界。
- 手算五组（数字由 `.venv/bin/python` 跑出，不是心算示意）：

```text
  gate    up   → clamp 后        SiLU(gate)      输出         不截断的输出
   3.0    2.0   3.0 /  2.0        2.857722       5.715445       5.715445
  12.0    9.0  10.0 /  9.0        9.999546      89.995914     107.999336
   0.5   15.0   0.5 / 10.0        0.311230       3.112297       4.668445
  -8.0    4.0  -8.0 /  4.0       -0.002683      -0.010731      -0.010731
  12.0   15.0  10.0 / 10.0        9.999546      99.995460     179.998894
```

  第一、四行说明**钳子对正常值完全不干预**；第二、三、五行说明它只在越界处生效。

---

## 四、★ `swiglu_limit = 10.0`：这把钳子为什么能提升稳定性

先看这行注释 —— 作者自己把话说明白了：`# Key difference using clamping`：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        # Key difference using clamping
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
```

**读法**：

- `clamp` 的语义是**把超出区间的值钉在边界上**，而它的导数在区间外**恰好是 0**。
  实测：对 `[-3, 5, 12, -21]` 求 `clamp(-10, 10)` 的梯度，得到 `[1, 1, 0, 0]`
  —— 12 和 −21 这两个越界位置**一个梯度都拿不到**。这就是"钳子"的第二重作用：
  它不只限住数值，还**切断了把数值推向更远处的梯度**。
- **上界为什么必须有**：`SiLU(z)` 在正侧不是饱和函数，而是**近似线性**的
  （实测 `SiLU(20) = 20.000000`）。所以 `SiLU(gate)·up` 在没有任何约束时是
  **两个无界量的乘积**，增长近似二次。实测一组 std=18 的随机 gate/up：
  `max|SiLU(gate)·up|` 达到 **2419.451**。
- **加上钳子之后上界是可以手推的**：`|SiLU(clamp(gate))| < 10`、`|clamp(up)| ≤ 10`，
  于是 `|中间激活| < 100`。同一组随机数实测从 2419.451 掉到 **99.995**
  （`SiLU(10) = 9.999546`，乘 10）。
- **阈值 10.0 的位置是有讲究的**：按 config 的 `initializer_range = 0.02` 初始化，
  4096 维输入下 `gate_proj` 输出的标准差约 1.28；实测 64×12288 个样本里
  `max|gate| = 6.44`。**10.0 大约是这个"出厂健康值"的 1.55 倍** ——
  正常训练状态下它几乎不触发，只在某个特征真的跑飞时兜底。
  （**边界**：这是初始化时的分布，不是训练后期权重的分布；仓库里没有权重可供实测后者。）
- 为什么这对**大**模型尤其重要（推断，不是文件里的原话，但每一步都能在这份代码里对上）：
  1. 中间激活是 `down_proj` 的输入，`down_proj` 要把 12288 维**求和**回 4096 维；
     单项从 2419 降到 100（24 倍），和的动态范围跟着降一个数量级以上。
  2. 主干权重通常以 bf16 发布（8 位尾数，约 3 位十进制有效数字）。
     **本仓库不含权重，这是按通行发布惯例的假设，不是实测。**
     动态范围越小，同样的相对舍入误差吃掉的绝对值越小；
     离群值少了，累加误差也不再被单点主导。
  3. 主干有 45 层，每一层都有一把同样的钳子 —— 单层的越界会顺着后面 44 层继续传播，
     而不是就地被吸收。
  4. 梯度在越界处为 0，等于给"跑飞的特征"装了一个**单向阀门**：它不会继续收到
     "再大一点"的梯度。代价是这一维暂时学不动 —— 这是用少量容量换整体稳定。
- **同一个想法在这份仓库里早有先例**：`activations.py` 里的 `ClippedGELUActivation`
  就是把 GELU 的输出夹在 `[min, max]` 之间（注册表里 `gelu_10` 就配的 `{-10, 10}`）。
  GLM-5 做的是同一件事，只不过夹在**门控乘法之前**、且夹的是两个投影的输出：

<!-- src: activations.py -->
```python
    def __init__(self, min: float, max: float):
        if min > max:
            raise ValueError(f"min should be < max (got min: {min}, max: {max})")

        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x: Tensor) -> Tensor:
        return torch.clip(gelu(x), self.min, self.max)
```

  **实测订正**：这里用的是 `torch.clip`，第四节用的是 `Tensor.clamp`。两者**数值上完全一致**
  （同一组随机输入下 `torch.equal(torch.clip(x, -10, 10), torch.clamp(x, -10, 10))` 为 `True`），
  但**不是同一个对象** —— `torch.clip is torch.clamp` 实测为 `False`。
  所以"同一个机制"这句话对，但"同一个函数"这句话不对。

---

## 五、同一套截断，在这份文件里出现了 4 次

`Glm5NextTextMLP` 不是唯一带钳子的地方。稀疏层的**路由专家**走的是另一条代码路径，
但钳子一模一样：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        # Simple swiglu instead of alpha
        return F.silu(gate) * up
```

**读法**：
- `gate_up_proj` 是**一个打包的张量**（gate 和 up 拼在中间维上），所以这里用
  `chunk(2, dim=-1)` 现场切开，而不是像稠密 MLP 那样有两个独立投影。
  打包的目的在 L3-04 展开（grouped GEMM 要求权重按 `(E, ...)` 排布）。
- 切完之后的四行，和 `Glm5NextTextMLP.forward` 的最后四行**逐字同构**
  （除了 `self.act_fn(gate)` 换成 `F.silu(gate)`，以及多了一句"不用 alpha"的注释）。
  换句话说：**稠密 MLP、路由专家、共享专家，三条路用的是同一套数值约定。**
- 共享专家本身就是 `Glm5NextTextMLP` 的实例，只是把中间维换成了
  `moe_intermediate_size × n_shared_experts = 2048 × 1`：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        self.shared_experts = Glm5NextTextMLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )
```

- 路由专家的权重布局（L3-04 的接口）：

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

  实测：`grep -c "clamp(min=None"` 在这份文件里命中 **4 处** ——
  行 103（稠密 MLP）、行 140（路由专家）、行 1530 / 1553（视觉支路，见第八节）。
  也就是说，"截断"不是某一路的特殊处理，而是这份模型定义的**全局约定**。

---

## 六、`ACT2FN` 注册表：字符串怎么变成算子

`self.act_fn` 不是硬编码的 `silu`，而是查表查出来的。表在这里：

<!-- src: activations.py -->
```python
ACT2CLS = {
    "gelu": GELUActivation,
    "gelu_10": (ClippedGELUActivation, {"min": -10, "max": 10}),
    "gelu_fast": FastGELUActivation,
    "gelu_new": NewGELUActivation,
    "gelu_python": (GELUActivation, {"use_gelu_python": True}),
    "gelu_pytorch_tanh": GELUTanh,
    "gelu_python_tanh": (GELUTanh, {"use_gelu_tanh_python": True}),
    "gelu_accurate": AccurateGELUActivation,
    "hardswish": nn.Hardswish,
    "laplace": LaplaceActivation,
    "leaky_relu": nn.LeakyReLU,
    "linear": LinearActivation,
    "mish": MishActivation,
    "quick_gelu": QuickGELUActivation,
    "relu": nn.ReLU,
    "relu2": ReLUSquaredActivation,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "silu": SiLUActivation,
    "sqrtsoftplus": SqrtSoftplusActivation,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
    "prelu": nn.PReLU,
    "xielu": XIELUActivation,
}
ACT2FN = ClassInstantier(ACT2CLS)
```

**读法**：
- `ACT2CLS` 是**字符串 → 类**的字典；`ACT2FN` 是它的实例化版本。
  实测 `len(ACT2CLS) = 24`，键从 `gelu` 一路排到 `xielu`。
- 键的命名本身就是一份"激活函数演化史"：`gelu` 的三个近似
  （`gelu_new` / `gelu_fast` / `quick_gelu`）、`relu` 的变体（`relu2` / `relu6`）、
  以及新加的 `sqrtsoftplus`（docstring 写明是 DeepSeek V4 的路由打分函数）
  和 `xielu`。
- 两种值形态并存：**裸类**（`"silu": SiLUActivation`）和
  **（类, kwargs）二元组**（`"gelu_10": (ClippedGELUActivation, {"min": -10, "max": 10})`）。
  后者就是"同一个类、不同参数"的注册方式 —— 第四节的钳位先例正是靠它落地的。
  这两种形态由第七节的 `ClassInstantier` 统一消费。
- 被本模型选中的那一项：`"silu": SiLUActivation`，它包的是
  `nn.functional.silu`（也就是 `x·σ(x)`，与第三节公式里的 `SiLU` 一致）：

<!-- src: activations.py -->
```python
class SiLUActivation(nn.Module):
    """
    See Gaussian Error Linear Units (Hendrycks et al., https://arxiv.org/abs/1606.08415) where the SiLU (Sigmoid Linear
    Unit) was originally introduced and coined, and see Sigmoid-Weighted Linear Units for Neural Network Function
    Approximation in Reinforcement Learning (Elfwing et al., https://arxiv.org/abs/1702.03118) and Swish: a Self-Gated
    Activation Function (Ramachandran et al., https://arxiv.org/abs/1710.05941v1) where the SiLU was experimented with
    later.
    """

    def forward(self, input: Tensor) -> Tensor:
        return nn.functional.silu(input)
```

- **跨课呼应**：注册表不只服务 MLP。`Glm5NextTextRMSNormGated` 里写着
  `self.activation = "sigmoid"`，前向里同样是 `ACT2FN[self.activation](...)`
  —— 同一个注册表，被同一层的两个算子共用（L3-01 讲 RMSNorm）。

---

## 七、★ 注册表里存的不是实例，是"类 + 参数"

`ACT2FN` 的类型不是 `dict`，而是 `ClassInstantier`：

<!-- src: activations.py -->
```python
class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)
```

**读法**：
- 关键在最后一行：`return cls(**kwargs)`。**每次下标访问都会新建一个实例。**
  实测 `ACT2FN["silu"] is not ACT2FN["silu"]` 为 `True` —— 两次取出的是两个不同对象。
- 为什么这样设计？因为 `nn.Module` **不能共享**：同一种激活被两层各自持有，
  参数（如有）与 `train/eval` 状态必须独立。若表里存的是实例，
  `xielu` 那种带 `nn.Parameter` 的激活被两个模型同时引用就会串味。
- `OrderedDict` 在这里的实际作用是**让键的顺序稳定**：`get_activation` 报错时会把
  `list(ACT2FN.keys())` 整个打给用户看，顺序稳定才可复现。
  （**说实话**：Python 3.7+ 的普通 `dict` 也保序，所以这一选择更像是显式声明与历史沿用，
  而不是非它不可。）

<!-- src: activations.py -->
```python
def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}")
```

- 这也是"拼错一个激活名"的失败方式：不是静默用默认值，而是 `KeyError` 里带着
  24 个合法键一起抛出。对比一下 `self.act_fn = ACT2FN[config.hidden_act]`
  写在 `__init__` 里 —— **配置错误在建模时就炸，不是跑到第 30 层才炸**。

---

## 八、视觉支路：同一段代码的第二份拷贝

`Glm5NextVisionMLP` 和文本 MLP 几乎一模一样 —— 连注释都一样：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
    def forward(self, hidden_state):
        gate = self.gate_proj(hidden_state)
        up = self.up_proj(hidden_state)
        # Key difference using clamping
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)
```

**读法**：
- 差异只有两处：入参叫 `hidden_state`、`bias` 默认 `True`
  （视觉塔的线性层带偏置，文本主干不带）。
- `Glm5NextVisionPatchMerger.forward`（行 1547–1555）是**第三份**拷贝：
  前半段多出一个 `proj` + `LayerNorm` + `GELU`，
  后半段（行 1550–1555）与上面这段**逐字相同** —— 同样是两把钳子 + 一次相乘 + 一个投影。
- 这说明 `swiglu_limit` 是**跨模态的统一约定**：图像 token 经过 patch merger
  进入文本主干时，用的也是同一把钳子。实测 `Glm5NextConfig()` 里
  `text_config.swiglu_limit = 10.0`、`vision_config.swiglu_limit = 10.0`，
  两侧的 `hidden_act` 也都是 `"silu"`。
- **实测订正**：本课件初稿在这里写过"这是 `# Copied from` 标记的重复"——
  实测 `grep -c "Copied from"` 在这份文件里命中 **0 次**，那句话是错的。
  能核实的只有文件头那三行：`modeling_glm5_next.py` 由 `modular_glm5_next.py`
  **生成**（"Do NOT edit this file manually"）。
  **推论（明确标注为推论）**：这种重复是故意的 —— 两个模态共用一套数值约定，
  比"各自演化"更容易保证一致性。

---

## 九、与后面课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| `_apply_gate` 里的 `gate_up` 为什么要打包成一个张量 | L3-04 |
| 288 个专家怎么被选出 8 个（`gate` 那一路） | L3-03 |
| 共享专家为什么"always on"、代价是什么 | L3-04, L3-05 |
| `post_attention_layernorm` 之后、`mlp` 之前的形状是什么 | L3-01, L3-06 |
| MLP 的输出怎么被写回 4 条残差流 | L3-06 |
| `ACT2FN` 还被这一层的哪些地方用到 | L3-01（`RMSNormGated` 的 `sigmoid`） |

**一句话总结**：

> 一层里最后一个算子 = `W_down( SiLU(clamp(W_gate·x, 10)) ⊙ clamp(W_up·x, ±10) )`；
> 钳子不改正常值，只把"会跑到 2400 的中间激活"按在 100 以内，并顺手切断越界处的梯度。
