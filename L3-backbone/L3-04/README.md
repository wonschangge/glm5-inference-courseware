# L3-04 · 专家层与 grouped GEMM — 权重怎么排，决定它能怎么算

> 层：**第 3 层 · 文本主干** ｜ 优先级：P0 ｜ 前置课：L3-03

## 学习目标

看完这一课，你应该能：

1. **说出打包存储省下了什么** —— 每个专家少一次 GEMM 调用、少一个 Parameter 对象；
   但 FLOPs 一点没省（验收点 1）；
2. **解释权重布局与 EP 切分的关系** —— `(E, ...)` 的第 0 维既是 grouped GEMM 的组轴，
   也是 `Shard(0)` 的切分轴（验收点 2）；
3. 手算 `swiglu_limit = 10.0` 下的两次截断，并说出为什么 gate 只截一边、up 截两边。

## 覆盖的源文件（1 个）

| 文件 | 行数（实测） | 引用块 |
|---|---|---|
| `models/glm5_next/modeling_glm5_next.py` | 2444 | 10 |
| `models/glm5_next/configuration_glm5_next.py` | 321 | 1（借用的证据，**不计入本课 coverage**） |

> **与作业书的偏差**：作业书说 `modeling_glm5_next.py` 是 2445 行，`wc -l` 实测 **2444** —— 以实测为准。
> 本课引用了 `_data/recon/probe_l304.py` 的**实测输出**（写在 `source.md` 的 `text` 块里）。
> **它是脚本，不计入覆盖率** —— 覆盖域只统计 `src/transformers/**/*.py`。
> `configuration_glm5_next.py` 的 EP 计划由 L1-02 / L1-03 覆盖，本课只引用、不声明。

## 场景（9 幕）

1. **一个 MoE 模块里的三块积木** —— 路由专家 / 路由器 / 共享专家，以及四个实测数字（288 / 8 / 2048 / 1）
2. **★ 两个 3D 参数，而不是 864 个 `nn.Linear`** —— `(E, 2I, H)` 与 `(E, H, I)`，第 0 维是专家编号
3. **★ 打包：一次 GEMM 顶两次** —— `chunk(2, dim=-1)` 是视图；调用次数 10 vs 15（全命中 576 vs 864）
4. **逐专家前向** —— `one_hot` mask（含哨兵列）、`hit`、`torch.where`、`index_add_`
5. **★ 为什么必须是 `(E, …)`** —— offsets 手算 `[2, 2, 5]`，以及三条实测报错原文
6. **两次截断，而且不对称** —— gate 只截上界、up 两侧都截；手算表 + 99.995 的硬上界
7. **EP 切分：`Shard(0)` 切的就是专家那一维** —— 每卡专家数与显存实测（288/8 → 36 个、1.12 GiB）
8. **出口：`routed + shared`** —— 共享专家用入口处的 `residuals`，逐元素相加
9. **收束** —— 三条结论总表 + 实测数字表 + 练习 + 下一课指路

## 核心结论

### 1. 打包：`gate_up_proj` 是一个 `(E, 2I, H)` 张量，第 1 维把 gate 与 up 拼在一起

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
```

省下的是**每个专家一次 GEMM 调用**与**一个 Parameter 对象**：实测（探针 [9]）命中 5 个专家时
打包 10 次、不打包 15 次；一层 288 个专家全命中就是 576 vs 864。数值完全一致
（`allclose=True`）—— **FLOPs 没省，省的是调用次数**。`chunk(2, dim=-1)` 与源张量共享 storage，
是零拷贝视图。

### 2. `swiglu_limit` 的两次截断是不对称的

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
```

`gate` 只截上界，因为 `silu` 自带全局下界（实测 −0.278465，在 x ≈ −1.278 处）；
`up` 是纯乘子，两侧都要拦。手算：`(gate, up) = (40, 40)` → `(10, 10)` → `silu(10) × 10 = 99.995`
（不截断是 1600）；而 `gate = −40` 不会被截，`silu(−40) ≈ 0`，输出 0 —— 负方向本来就无害。

### 3. ★ `(E, ...)` 的第 0 维同时是 grouped GEMM 的组轴与 EP 的切分轴

`grouped_mm` 的 `offsets` 长度必须等于权重第 0 维（实测报错
`matrix batch sizes have to match`）；EP 的实测实现是
`MoEParamShard(Shard(0), shards_expert_dim=True)`，切的正是同一维。切完每张卡仍是
`(E_local, 2I, H)` —— **形状约定没变，只是 E 变小了**，grouped GEMM 的代码一行都不用改。

### 实测数字

| 量 | 实测值 | 含义 |
|---|---|---|
| `n_routed_experts` | 288 | 每层路由专家 |
| `num_experts_per_tok` | 8 | 每 token 命中 |
| `moe_intermediate_size` | 2048 | 单专家中间维 |
| 单专家参数 | 25,165,824 | gate_up 16,777,216 + down 8,388,608 |
| 一层 288 个专家 | 7,247,757,312 | 7.248 B |
| 42 个 MoE 层 | 304,405,807,104 | 304.41 B |
| 共享专家 | 25,165,824 | **恰好等于 1 个路由专家** |
| `swiglu_limit` | 10.0 | 输出上界 99.995 |

## 与后续课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| 288 个分数怎么选出 8 条路径 | L3-03（前置课） |
| 稠密 MLP 上那对同样的 clamp 为什么必要 | L3-02 |
| 共享专家的装配细节与缩放口径 | L3-05 |
| `grouped_mm` / `sonicmoe` / `deepgemm` 三套实现怎么选 | L8-04 |
| TP / EP 样式表（`Shard(0)`、`ep_router`）怎么落地 | L8-03 |
| config 里的 `attribute_map` 与 `base_model_ep_plan` 怎么读 | L1-02, L1-03 |

## 验收点

- [x] 保真门禁：11 个引用块全部逐字来自源文件（10 个来自 `modeling_glm5_next.py`，1 个来自 config）
- [x] 覆盖度门禁：1 个源文件被声明（属覆盖域 231 个文件之一）
- [x] 参数门禁：无非法参数引用
- [x] 语法检查：`node --check` 通过，9 幕字段完整
- [x] 渲染门禁：无 JS 错误、无布局溢出、交互可用（两种分辨率）
- **能说出打包存储省下了什么**：省的是每个专家一次 GEMM 调用与一个 Parameter 对象
  （实测 576 vs 864；`allclose=True` 说明 FLOPs 没变），外加 EP/量化/加载只需处理一个原子对象
- **能解释权重布局与 EP 切分的关系**：`(E, ...)` 的 dim 0 既是 grouped GEMM 的组轴
  （`offsets` 长度 = E），也是 `Shard(0)` 的切分轴；切完仍是 `(E_local, ...)`，形状约定不变

**一句话总结**：

> `gate_up_proj` 是 `(E, 2I, H)`：dim 1 上的打包让每个专家少一次 GEMM 调用，
> dim 0 上的专家轴同时充当 grouped GEMM 的组轴与 EP 的切分轴 —— **布局不是实现细节，它是接口**；
> 而 `swiglu_limit` 那两次不对称的 clamp，把单个专家的输出幅值钉死在 `silu(10) × 10 ≈ 99.995`。
