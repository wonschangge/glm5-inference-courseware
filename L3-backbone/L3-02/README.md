# L3-02 · MLP 与 SwiGLU 截断

> 层：**第 3 层 · 文本主干** ｜ 优先级：P0 ｜ 前置课：`L3-01`

## 学习目标

看完这一课，你应该能：

1. **写出 SwiGLU 的前向公式**，并说出 `gate` 与 `up` 这两个投影各自负责什么
   （对应验收点 1）；
2. **解释 `swiglu_limit = 10.0` 为什么能提升大模型稳定性** —— 数值上它挡住了什么、
   梯度上它做了什么、代价是什么（验收点 2）；
3. 说出 `ACT2FN` 这张注册表为什么存的是「类」而不是「实例」，
   以及一个拼错的 `hidden_act` 会在什么时刻炸。

## 覆盖的源文件（2 个）

| 文件 | 行数 | 引用块 |
|---|---|---|
| `models/glm5_next/modeling_glm5_next.py` | 2444 | 8 |
| `activations.py` | 369 | 5 |

> 行数是 `wc -l` 实测。作业书大纲里写的是 2445 / 370 —— **实测是 2444 / 369**，
> 差的 1 行在文件末尾的换行符上。本课以实测为准。
>
> 本课引用了若干**脚本实测输出**（写在 `source.md` 的 `text` 块里，用仓库自带的
> `.venv/bin/python` 跑出）。**脚本不计入覆盖率** —— 覆盖域只统计
> `src/transformers/**/*.py`，本课声明的就是上面这两个文件。

## 场景（9 幕）

1. **一层里最后一个算子** —— 层内流程线；MLP 是唯一「形状进出不变、只换内容」的算子
2. **三个投影** —— gate / up / down 的形状与分工；`bias=False` 与 151.0M 的由来
3. **★ 一行 SwiGLU** —— 前向公式 + 五组手算（含「不截断」对照列）
4. **★ 10.0 这把钳子** —— 上界手推、实测 2419 → 100、越界梯度为 0、阈值位置的实测依据
5. **同一套截断出现 4 次** —— 稠密 / 路由专家 / 共享专家 / 视觉支路共用一个约定
6. **ACT2FN 注册表** —— 24 个键、两种登记形态、`hidden_act = "silu"` 怎么被解析
7. **★ ClassInstantier** —— 为什么每次访问都要新建实例，以及 KeyError 为什么是好事
8. **视觉支路** —— 同一段代码的第二、第三份拷贝；差异只有入参名与 bias
9. **收束** —— 实测数字总表 + 练习 + 下一课指路

## 核心结论

### 1. SwiGLU 的全部内容就是最后那一行

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)
```

写成公式（`L = swiglu_limit = 10.0`，`SiLU(z) = z·σ(z)`）：

```text
SwiGLU(x) = W_down( SiLU(clamp(W_gate·x, max = L)) ⊙ clamp(W_up·x, -L, +L) )
```

`gate` 决定「放多少过去」，`up` 决定「过去的是什么」；`⊙` 把两条线性变换耦合起来 ——
这就是它比老式 `W_down(act(W_up x))` 表达力更强的原因。

### 2. ★ 钳子上界可以手推，且与输入无关

`|SiLU(clamp(gate))| < 10`、`|clamp(up)| ≤ 10` → **中间激活恒 < 100**。
实测（std=18 的随机 gate/up，各 4000 个样本）：`max|SiLU(gate)·up|` 从
**2419.451** 降到 **99.995**；同一组数据里 69.9% 的元素被截断。

第二重作用在梯度上：`clamp` 在区间外的导数**恰好是 0**。实测对 `[-3, 5, 12, -21]`
求 `clamp(-10, 10)` 的梯度得到 `[1, 1, 0, 0]` —— 越界的那两维拿不到
「再大一点」的学习信号。**用少量容量换整体稳定**，这就是它对 45 层大模型的意义。

### 3. ★ 阈值 10.0 就在「出厂健康值」的上方一点

按 config 的 `initializer_range = 0.02` 初始化，4096 维输入下 `gate_proj` 输出的
标准差约 1.28；实测 64 × 12288 个样本里 `max|gate| = 6.44`。
**10.0 约为它的 1.55 倍** —— 正常训练状态下几乎不触发，特征跑飞时才兜底。

（**边界**：这是初始化时的分布，不是训练后期权重的分布；本仓库不含权重，
后者无法实测。）

### 4. `ACT2FN` 存的是「类 + 参数」，不是实例

<!-- src: activations.py -->
```python
class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)
```

实测 `ACT2FN["silu"] is not ACT2FN["silu"]` 为 `True` —— 每次下标访问都新建实例。
因为 `nn.Module` 不能共享；`ACT2FN["gelu_10"]` 则会拿到
`ClippedGELUActivation(min=-10, max=10)`，这是注册表里早就存在的「钳位」先例。

### 实测数字

| 量 | 实测值 | 含义 |
|---|---|---|
| `hidden_size` | 4096 | 主干宽度 |
| `intermediate_size` | 12288 | 稠密 MLP 中间维 = 3 × 4096 |
| 稠密 MLP 参数量 | 150,994,944 | ≈ 151.0M，零偏置 |
| 一个稀疏层 MLP 参数量 | 7,274,102,784 | ≈ 7.27B（存储） |
| 每 token 激活 | 226,492,416 | ≈ 226.5M → 存储/激活 = 32.0× |
| `swiglu_limit` | 10.0 | 文本侧与视觉侧相同 |
| `hidden_act` | `"silu"` | → `ACT2FN["silu"]` → `SiLUActivation` |
| `len(ACT2CLS)` | 24 | 注册表键数 |
| 钳位点 | 4 处 | 行 103 / 140 / 1530 / 1553 |

## 与后续课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| `_apply_gate` 里 `gate_up` 为什么要打包成一个张量 | L3-04 |
| 288 个专家怎么被选出 8 个 | L3-03 |
| 共享专家为什么 always on、代价是什么 | L3-04, L3-05 |
| `post_attention_layernorm` 与 `mlp` 之间的形状 | L3-01, L3-06 |
| MLP 的输出怎么被写回 4 条残差流 | L3-06 |
| `ACT2FN` 还被这一层的哪些地方用到 | L3-01（`RMSNormGated` 的 `"sigmoid"`） |

## 验收点

- [x] **能写出 SwiGLU 的前向公式**：`W_down( SiLU(clamp(W_gate·x, 10)) ⊙ clamp(W_up·x, ±10) )`
      —— 第 3 幕给出公式与五组手算，`source.md` 第三节逐字引用 `forward`
- [x] **能解释截断为什么能提升大模型稳定性**：第 4 幕给出上界手推（< 100）、
      实测对比（2419.451 → 99.995）、越界梯度为 0 的实测，以及阈值位置的实测依据
- [x] 能说出 `ACT2FN` 为什么存类而不存实例（第 7 幕 + 实测 `is not` 为 `True`）
- [x] 保真门禁：13 个引用块全部逐字来自两个源文件，且位置连续
- [x] 覆盖度门禁：2 个源文件被声明，与 `tools/plan.py` 的 `files` 一致
- [x] 参数门禁：无非法参数引用
- [x] 语法检查：`node --check` 通过，9 幕字段完整
- [x] 渲染门禁：无 JS 错误、无布局溢出、交互可用（两种分辨率）
- 自检：对着第 3 幕的手算表，说出「哪几行钳子没动手、哪几行动了手、为什么」

**一句话总结**：

> 一层里最后一个算子 = `W_down( SiLU(clamp(W_gate·x, 10)) ⊙ clamp(W_up·x, ±10) )`；
> 钳子不改正常值，只把「会跑到 2400 的中间激活」按在 100 以内，并顺手切断越界处的梯度。
