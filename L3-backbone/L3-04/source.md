<!-- glm5-coverage
models/glm5_next/modeling_glm5_next.py
-->

# L3-04 · 专家层与 grouped GEMM — 源文件

**这一课只回答三个问题：专家权重为什么打包存？`swiglu_limit` 在专家路径上截了什么？为什么必须按 `(E, ...)` 排布？**

视角：从一张 `(288, 4096, 4096)` 的参数表出发，看它怎么同时喂饱三个下游 ——
eager 逐专家循环、grouped GEMM 的组轴、EP 的切分轴。

| 文件 | 行数（实测 `wc -l`） | 本课引用块 |
|---|---|---|
| `models/glm5_next/modeling_glm5_next.py` | 2444 | 10 |
| `models/glm5_next/configuration_glm5_next.py` | 321 | 1 |

> **与作业书的偏差（以实测为准）**：作业书写该文件 2445 行，实测 2444 行。
> 上表第二行是**借来的一处证据** —— 专家并行的切分计划就写在 config 里。
> 本课的 coverage 声明仍然只有 `modeling_glm5_next.py`；`configuration_glm5_next.py`
> 由 L1-02 / L1-03 覆盖，这里**只引用、不声明**。

**实测数字**（`_data/recon/probe_l304.py` 的输出。它是脚本，**不计入覆盖率**）：

```text
[1] Glm5NextTextConfig() 默认值（实测）
    hidden_size            = 4096
    moe_intermediate_size  = 2048
    n_routed_experts       = 288
    num_local_experts      = 288
    num_experts_per_tok    = 8
    n_shared_experts       = 1
    swiglu_limit           = 10.0
    routed_scaling_factor  = 2.5

[2] 真实 config 下的两个 Parameter（meta 设备，不占内存）
    gate_up_proj.shape = (288, 4096, 4096)
    down_proj.shape    = (288, 4096, 2048)
    单个专家: gate_up 16,777,216 + down 8,388,608 = 25,165,824
    一层 288 个专家 = 7,247,757,312  (7.248 B)
    42 个 MoE 层       = 304,405,807,104  (304.41 B)
    对照 · 稠密 MLP(12288) 一层 = 150,994,944  (151.0 M)
    对照 · 共享专家(2048)    一层 = 25,165,824  (25.2 M)
    共享专家 == 1 个路由专家（25,165,824）? True
```

> 上面这段用 `text` 块 —— 它是脚本输出，不是源文件的逐字引用，不参与保真校验。
> 下面每一个 `python` 块都是**从源文件切片**得到的逐字引用。

---

## 一、专家层长什么样：一个模块只装三样东西

先看装配。`Glm5NextTextMoE` 自己不做任何数学，它只是把三个部件摆好。

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
- 三件套：`experts`（288 个**路由**专家）、`gate`（路由器，288 → 8 的选拔在 L3-03 讲）、
  `shared_experts`（一个稠密 MLP，**每个 token 都走**，与路由无关）。
- `shared_experts` 的中间维是 `moe_intermediate_size * n_shared_experts` = 2048 × 1。
  实测它的参数量 **25,165,824 —— 恰好等于 1 个路由专家**（见上面的 `text` 块）。
  换句话说：共享专家不是"额外的路"，它就是第 289 个、不参与选拔的专家。
- 签名注解写的是 `Glm5NextConfig`，实际传进来的是 `Glm5NextTextConfig` ——
  `Glm5NextTextModel` 拿到的就是 `config.text_config`。所以 `config.num_local_experts`
  这个名字要在 text_config 上解析，靠的是那句
  `attribute_map = {"num_local_experts": "n_routed_experts"}`（config 第 99 行，L1-02 讲过这个机制）。
- 回顾 **L0-01**：45 层里 42 层 `mlp_layer_types == "sparse"`，也就是 42 份这样的装配 ——
  实测 304.41 B 参数全在这 42 层里。

---

## 二、★ 布局：两个 3D 参数，而不是 3E 个 2D 参数

这是整课的支点。教科书里的 MoE 实现通常是 `nn.ModuleList([MLP() for _ in range(E)])`；
这里不是。

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
@use_experts_implementation
class Glm5NextTextExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

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
- 两个参数，形状 `(E, 2I, H)` 与 `(E, H, I)`，E=288、I=2048、H=4096。实测：
  `gate_up_proj.shape = (288, 4096, 4096)`、`down_proj.shape = (288, 4096, 2048)`。
- **第 0 维永远是专家编号。** 于是 `gate_up_proj[e]` 就是一个普通的 `(2I, H)` 权重矩阵，
  和 `nn.Linear` 的 `.weight` 完全同构 —— 下一节的 `F.linear(x, self.gate_up_proj[expert_idx])` 就靠这一点。
- `gate_up_proj` 的第 1 维是 `2 * intermediate_dim`：**gate 与 up 的权重是拼在一起的**（顺序 `[gate; up]`）。
  单专家 16,777,216 + 8,388,608 = 25,165,824 个参数；一层 7.248 B；42 层 304.41 B。
- `self.swiglu_limit = config.swiglu_limit` 存成模块属性（默认 10.0），前向里要用两次 —— 见第五节。
- `nn.Parameter(torch.empty(...))` 只分配、不初始化；真正的初始化在 `_init_weights` 里（第九节引用）。
- 类头上的 `@use_experts_implementation` 把 `forward` 换成一个**调度点**：
  用哪套专家实现由 `config._experts_implementation` 决定（实测默认 `None` →
  回落到类里写的 eager 实现，也就是第三节那段循环）。真正跑 grouped GEMM 的那套实现在 L8-04。

作为对照，稠密 MLP 走的是另一条路：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.swiglu_limit = config.swiglu_limit
```

**读法**：
- 三个独立 `nn.Linear`、三个 Parameter，谁也不认识谁 —— 这就是"打包"二字的反面。
- 两边的 `swiglu_limit` 来自同一个 config 字段（10.0），所以稠密路径与专家路径的截断口径一致。

---

## 三、逐专家前向：先做 mask，再按专家循环，最后 `index_add_`

这是**默认（eager）实现**。它的每一步都在为"按专家分组"做准备。

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
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
- `F.one_hot(top_k_index, num_classes=self.num_experts + 1)`：多出来的第 `num_experts` 类是**哨兵**。
  EP 下 `ep_router` 会把"不属于本卡"的槽位置成 `num_local_experts`，正好落进这一类，
  被后面的 `continue` 跳过（实测：索引给 6、`num_experts=6` 时，该 token 的输出全 0）。
- `mask.permute(2, 1, 0)` 把 mask 变成 `(E, top_k, tokens)`；`hit` 只留下**真的被选中过**的专家，
  所以循环次数 = 命中专家数，而不是 288。
- `torch.where(mask[expert_idx])` 给出这一专家的 `(top_k 位置, token 下标)`。
- 每个专家**恰好两次 GEMM**：`F.linear(x, gate_up_proj[e])` 与 `F.linear(cur, down_proj[e])`。
  注意第一次的权重是打包过的 —— 这就是下一节的主角。
- 最后 `final.index_add_(0, token_idx, current)`：一个 token 选的 8 个专家各自往同一行累加。
  用 `index_add_` 而不是 `+=` 是因为不同专家的输出会撞到同一个 token 上。

---

## 四、★ 打包：gate 和 up 拼在第 1 维，`chunk` 只是一个视图

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
```

**读法**：
- `gate_up.chunk(2, dim=-1)` 把 `(..., 2I)` 切成两个 `(..., I)`。实测两个切片与源张量
  **共享同一块 storage**（`data_ptr` 相同、`stride` 相同）—— 没有拷贝、没有额外显存。
- 打包**省下的第一样东西**：每个专家一次 GEMM 调用。实测计数（探针 [9]）：
  命中 5 个专家时，打包 10 次 GEMM，不打包 15 次（每专家 2 vs 3）；
  一层 288 个专家全命中就是 576 vs 864，**差 288 次 kernel 启动**。
- 打包**省下的第二样东西**：一个 Parameter 而不是两个。这让 EP 切分、量化、checkpoint
  都把 `[gate; up]` 当成一个原子对象处理；grouped GEMM 也只需要一次调用、一份 `offsets`。
- 打包**没有省下**的东西：FLOPs。实测一次打包 GEMM 与两次分开 GEMM 的数值完全一致
  （`allclose` 为 `True`）—— 省的是调用次数与权重张量个数，不是算力。
- 为什么敢拼：因为两半**永远是同时被用到的**，而且共用同一个输入 `x`。
  如果 gate 和 up 的输入不同，这个拼法就不成立。

---

## 五、`swiglu_limit` 的两次截断（手算例子）

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        # Simple swiglu instead of alpha
        return F.silu(gate) * up
```

**读法**：
- 两行 `clamp`，但**不是对称的两次**：
  - `gate` 只截**上界**（`min=None`）。因为 `silu` 有全局下界 ——
    实测 `min silu(x) = -0.278465`（在 x = -1.278 处），负方向再大也压不出更多幅值。
  - `up` 截**上下两界**（`±limit`）。它是纯乘子，没有自带下界，必须两侧都拦住。
- 于是单专家输出的幅值上界被钉在 `silu(10) × 10 = 9.999546 × 10 = 99.995`。
- 手算表（实测，`swiglu_limit = 10.0`）：

```text
    gate=    0.5 up=    0.5 -> gate'=   0.5 up'=   0.5 slu=  0.31123  out=      0.156
    gate=   10.0 up=   10.0 -> gate'=  10.0 up'=  10.0 slu=  9.99955  out=     99.995
    gate=   40.0 up=   40.0 -> gate'=  10.0 up'=  10.0 slu=  9.99955  out=     99.995
    gate=  -40.0 up=  -40.0 -> gate'= -40.0 up'= -10.0 slu= -0.00000  out=      0.000
    gate=    2.0 up=  -13.0 -> gate'=   2.0 up'= -10.0 slu=  1.76159  out=    -17.616
    不做截断时 gate=up=40 的输出 = 1600.0
```

  读第四行：`gate = -40` **没有**被截到 `-10`（因为只截上界），而 `silu(-40) = -1.4e-17 ≈ 0`，
  输出 0 —— 负方向本来就无害。读第三行：`gate = up = 40` 被压成 `(10, 10)`，输出从 1600 掉到 99.995。
- 注释 `# Simple swiglu instead of alpha` 是作者留的记号：上一代的做法带一个学习出来的
  `alpha` 缩放，这一代直接用 `silu(gate) * up`。
- 回顾 **L3-02**：稠密 MLP 里是**同一对 clamp**（下面那段引用），连注释都一样。
  两处唯一的区别是输入从哪来 —— 稠密路径来自两个独立 Linear，专家路径来自一次打包 GEMM 的两半。

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

---

## 六、★ 为什么必须按 `(E, ...)` 排布：grouped GEMM 的组轴就是第 0 维

模型文件里看不到 `grouped_mm` 这几个字 —— 调用在 `integrations/moe.py`（L8-04 讲）。
但**能不能用它，取决于这里的布局**。实测（探针 [6]/[6b]）：

```text
[6] grouped GEMM：权重必须是 (E, out, in)，分组靠 offsets
    权重布局 = (3, 8, 8)  offsets = [2, 2, 5] torch.int32
    W.transpose(-2,-1) 是零拷贝吗 ?      True
    grouped 结果 == 逐专家 mm 拼接 ?     True

[6b] grouped GEMM 的硬性布局要求（实测报错原文）
    专家轴不在第 0 维 (out,in,E)        -> RuntimeError: contraction dimension of mat_a and mat_b must match
    offsets 长度 ≠ 专家数               -> RuntimeError: matrix batch sizes have to match
    offsets 不是 int32                  -> RuntimeError: Offsets have to be int32
```

**读法**：
- grouped GEMM 的接口是"一次调用、E 个分组"：输入 `(S, in)`（S = 所有被选中的
  (token, 专家) 对，按专家排好序），权重 `(E, in, out)`，再加一个 `offsets`（每个专家的累积 token 数）。
  分组边界**不是**靠形状推出来的，而是靠 `offsets` 这个长度恰好为 E 的整型向量。
- 所以"`(E, 2I, H)`"这个排布同时满足三件事：
  1. 第 0 维是组轴 —— 换成 `(2I, E, H)` 之类的排布，`offsets` 就对不上号（见上面的报错原文）；
  2. 每个专家自己的权重是**连续**的二维块，转置只是 stride 变换（实测 `transpose(-2,-1)` 零拷贝）；
  3. eager 实现里 `gate_up_proj[expert_idx]` 这句下标也直接可用 —— 同一份布局，两套实现都能吃。
- 反过来说：如果一开始存成 864 个 `nn.Linear`，想用 grouped GEMM 就得先把它们
  `torch.stack` 成一个 `(E, ...)` 张量 —— 要么每次前向拷一遍，要么在加载权重时多一次全局重排。
  **布局就是接口**，这是这一课最想留下的一句话。
- 注意 `gate_up_proj` 的第 1 维（`2I`）不是组轴而是输出维：打包和分组是**两个正交的维度约定**，
  一个在 dim 1 上省调用，一个在 dim 0 上分组。

---

## 七、EP 切分：切的正是第 0 维

专家并行（EP）的切分计划写在 config 里，实测原样如下：

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
    base_model_ep_plan = {
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
    }
```

**读法**：
- 四个条目只做一件事：把 `experts.gate_up_proj` / `experts.down_proj` 的**参数**交给
  `grouped_gemm` 这一种并行样式，把 `gate` 交给 `ep_router`，把 `experts` 这个模块交给 `moe_tp_experts`。
- `grouped_gemm` 样式的实测实现是 `MoEParamShard(Shard(0), shards_expert_dim=True)` ——
  `Shard(0)` 切的**就是专家那一维**（`placement = S(0)`）。
- 切完之后每张卡仍然持有 `(E_local, 2I, H)` 形状的参数：**形状约定没变，只是 E 变小了**。
  所以同一份 grouped GEMM 代码不需要改，只是 `offsets` 的长度变成 `E_local`。
- 实测的每卡专家数与显存（bf16，只算 `gate_up_proj`）：

```text
        ep_size=  1 → 每卡 288 个专家，gate_up_proj 每卡 9.00 GiB (bf16)
        ep_size=  8 → 每卡  36 个专家，gate_up_proj 每卡 1.12 GiB (bf16)
        ep_size= 16 → 每卡  18 个专家，gate_up_proj 每卡 0.56 GiB (bf16)
        ep_size= 32 → 每卡   9 个专家，gate_up_proj 每卡 0.28 GiB (bf16)
```

- 代价在**路由侧**：`gate` 在每张卡上都是全量的（288 个分数），但只有本卡那 `E_local` 个专家
  能被真正执行。`ep_router` 负责把非本地专家的分数清零、把全局专家号重映射成本地号、
  把被丢弃的槽位写成哨兵 `num_local_experts` —— 那个哨兵正好落进第三节里
  `num_classes=self.num_experts + 1` 多出来的那一类，被 `continue` 跳过。
- 一条约束：`288 % ep_size == 0`。实测 1 / 8 / 16 / 32 都能整除。
- 展开在 **L8-03**（张量并行样式表）与 **L8-04**（专家并行与 MoE 集成）。

---

## 八、出口：路由结果加回共享专家

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
- `hidden_states + self.shared_experts(residuals)`：注意用的是**进来时的 `residuals`**，
  不是路由输出 —— 共享专家和路由专家是**并行**的两条支路，最后逐元素相加。
- 实测 `out == routed + shared` 为 `True`；`orig_shape` 只是为了把展平后的 `(tokens, H)` 还原回 `(B, S, H)`。
- 共享专家那条路没有 `topk_weights`、没有 `routed_scaling_factor` 的 2.5 倍缩放 ——
  它的输出是"1 倍"地直接加上去的。这个不对称是设计的一部分（L3-05 展开）。

---

## 九、这些权重在哪被初始化、在哪被实例化

初始化（`Glm5NextPreTrainedModel._init_weights` 的专家分支）：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        elif isinstance(module, Glm5NextTextExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)
```

**读法**：
- 两个 3D 参数各自 `init.normal_`，标准差 `initializer_range`（0.02）。
  没有逐专家的特殊处理 —— 再次说明它们是**批量**管理的。
- 路由器 `e_score_correction_bias` 与 experts 在同一张 `isinstance` 链上（L3-03）。

实例化点（`Glm5NextTextDecoderLayer.__init__`，与 L0-01 第五幕同一段）：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        self.mlp = (
            Glm5NextTextMoE(config) if config.mlp_layer_types[layer_idx] == "sparse" else Glm5NextTextMLP(config)
        )
```

**读法**：
- 一层用稠密还是 MoE，由 config 里的字符串 `mlp_layer_types[layer_idx]` 决定；
  `Glm5NextTextMoE` 就是这么被 new 出来的（每层一个实例，42 层 42 个）。

---

## 十、把这一课连起来

```text
  hidden_states (tokens, 4096)
        │
        ├─ gate（路由器，288 个分数）──────────────► topk_indices (tokens, 8)
        │                                            topk_weights (tokens, 8)
        ▼
  experts.gate_up_proj : (288, 4096, 4096)   ← dim 0 = 专家轴
  experts.down_proj    : (288, 4096, 2048)   ← dim 0 = 专家轴
        │
        │  eager：for e in hit:  F.linear(x, gate_up_proj[e]) → _apply_gate → F.linear(·, down_proj[e])
        │  grouped：1 次 grouped_mm(x_sorted, gate_up_proj, offs) → _apply_gate → 1 次 grouped_mm
        │            offs 的长度 = 专家数（EP 之后 = 本卡专家数）
        ▼
  routed (tokens, 4096)  ──┐
                           ├─ 逐元素相加 ──► hidden_states (tokens, 4096)
  shared_experts(x) ───────┘   （1 个专家大小，每个 token 都走）
```

**一句话总结**：

> `gate_up_proj` 是 `(E, 2I, H)`：dim 1 上的打包让每个专家少一次 GEMM 调用，
> dim 0 上的专家轴同时充当 grouped GEMM 的组轴与 EP 的切分轴 ——
> **布局不是实现细节，它是接口**；而 `swiglu_limit` 那两次不对称的 clamp，
> 把单个专家的输出幅值钉死在 `silu(10) × 10 ≈ 99.995`。

## 与后面课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| 288 个分数怎么选出 8 条路径 | L3-03（前置课） |
| 稠密 MLP 上那对同样的 clamp 为什么必要 | L3-02 |
| 共享专家的装配细节与缩放口径 | L3-05 |
| `grouped_mm` / `sonicmoe` / `deepgemm` 三套实现怎么选 | L8-04 |
| TP / EP 样式表（`Shard(0)`、`ep_router`）怎么落地 | L8-03 |
| config 里的 `attribute_map` 与 `base_model_ep_plan` 怎么读 | L1-02, L1-03 |

