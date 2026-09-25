# L0-02 · 配置即架构：从 config 读出全部形状

> 层：**预备篇 · 直觉** ｜ 优先级：P0 ｜ 前置课：L0-01

## 学习目标

看完这一课，你应该能：

1. **从 config 默认值手算 MLA 层的 Q/K/V 形状**：`4096 → 1536 → 16384`、`4096 → 512 → 32768`（验收点 1）；
2. **解释为什么 `head_dim` 是 0 而不是 256**，并说出真正的头宽叫什么（验收点 2）；
3. 说出 `__post_init__` 生成的三张表各是什么，以及 `validate_architecture` 的五处 `raise` 各防什么错误。

## 覆盖的源文件（1 个）

| 文件 | 行数 | 引用块 |
|---|---|---|
| `models/glm5_next/configuration_glm5_next.py` | 321 | 12 |

> 本课还引用了 `models/glm5_next/modeling_glm5_next.py:1128-1149` 的**实测核对结果**
> （写在 `source.md` 的 `text` 块里）。**它是本课之外的文件，不计入本课覆盖率** ——
> 该文件的正式覆盖在 L0-01 / L3 / L4 等课。

## 场景（9 幕）

1. **一个 config，钉死 45 层的全部形状** —— 四步流程 + 四个关键数字（4096 / 64 / 0 / 45）
2. **三条宽度 4096 / 12288 / 2048** —— 稠密 FFN 是 3 倍、专家 FFN 是一半；头数相等的含义
3. **★ 手算 MLA 的 Q/K/V** —— 形状链上的数据包（FLOW），压缩 → 展开 → 注意力 → 输出
4. **★ 为什么 head_dim 是 0** —— 头内维度分解成「RoPE 段 0 + NoPE 段 256」
5. **★ 那三行是谁写的** —— `kwargs.pop("head_dim")` → 重新赋值 → 算出 `qk_head_dim`
6. **前 3 层稠密，其余 42 层稀疏** —— 45 格排班条 + `min(3, N)` 的保护与 `N - 3` 的不保护
7. **★ 五处 raise** —— 四道形状契约 + 一道 NoPE 断言，逐条点亮
8. **config 里还写着「怎么切开」** —— `tp_plan` 13 条 / `ep_plan` 4 条 / `attribute_map` 别名
9. **收束** —— 七行汇总表 + 两道练习 + 下一课指路

## 核心结论

### 1. 形状不是抄来的，是算出来的

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
    vocab_size: int = 154880
    hidden_size: int = 4096
    intermediate_size: int = 12288
    moe_intermediate_size: int = 2048

    num_hidden_layers: int = 45
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    n_shared_experts: int = 1
    n_routed_experts: int = 288
    routed_scaling_factor: float = 2.5
```

五个字段定死 MLA：`q_lora_rank=1536`、`kv_lora_rank=512`、`qk_rope_head_dim=0`、`qk_nope_head_dim=256`、`v_head_dim=256`。
于是（下表中 `qk_head_dim = 0 + 256 = 256`）：

| 投影 | 输入 | 输出 | 算式 |
|---|---|---|---|
| `q_a_proj` | 4096 | 1536 | `q_lora_rank` |
| `q_b_proj` | 1536 | 16384 | `64 × qk_head_dim(256)` |
| `kv_a_proj_with_mqa` | 4096 | 512 | `kv_lora_rank + qk_rope_head_dim = 512 + 0` |
| `kv_b_proj` | 512 | 32768 | `64 × (qk_nope_head_dim + v_head_dim) = 64 × 512` |
| `o_proj` | 16384 | 4096 | `64 × v_head_dim` → `hidden_size` |

分数矩阵形状 `(batch, 64, seq, seq)`，缩放系数 `1/√256 = 1/16`。

### 2. ★ `head_dim = 0` 是「这个架构不做 RoPE」的编码

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
        # NOTE: this forces an intentional override as we have the convention of head_dim being the RoPE based dim
        kwargs.pop("head_dim", None)
        self.head_dim = self.qk_rope_head_dim
        self.qk_head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim

        super().__post_init__(**kwargs)
```

`head_dim` 在本仓库的约定里表示**会被 RoPE 旋转的那部分维度**，不是头的总宽度。
DSA 层是 NoPE，所以它是 0；真正的头宽是 `qk_head_dim = 0 + 256 = 256`。

实测：`Glm5NextTextConfig(head_dim=256).head_dim` → **0**（`kwargs.pop` 先丢掉传入值）。
第五节还有另一面：`qk_rope_head_dim > 0` 会被 `validate_architecture` 直接拒绝。

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 32
    head_dim: int = 0
    layer_types: list[str] | None = None
    # `"full"` runs the indexer, `"shared"` reuses the previous full layer's index mask.
    indexer_types: list[str] | None = None
```

### 3. ★ 三张表都由 `__post_init__` 现算

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] * min(3, self.num_hidden_layers) + ["sparse"] * (
                self.num_hidden_layers - 3
            )
```

- `mlp_layer_types`：`["dense"] * min(3, N) + ["sparse"] * (N - 3)` → `3 + 42`（**没有一行硬编码 42**）；
- `layer_types`：`idx % 4 != 3` → `linear_attention`，其余 `indexed_attention` → `34 : 11`（展开见 L0-03）；
- `indexer_types`：默认 45 个 `"full"`。
- 边界（实测）：`num_hidden_layers=2` → `['dense', 'dense']` —— `[...] * (-1)` 得到空列表，不报错。

### 4. ★ `validate_architecture`：四道形状契约 + 一道 NoPE 断言

| # | 检查 | 防的错误 |
|---|---|---|
| 1 | `num_attention_heads != num_key_value_heads` | 按 GQA 少配 KV 头；MLA 的 K/V 头数必须 1:1 |
| 2 | `index_kpool < 1` | 池化窗口为 0/负数 → 分组除零 |
| 3 | `index_topk % index_kpool != 0` | 2048 个稀疏位置切不整 |
| 4 | `q_lora_rank is None` | DSA 缺少压缩 Q 的低秩路径 |
| 5 | `qk_rope_head_dim > 0` | 在 NoPE 架构上打开 RoPE |

### 实测数字

| 量 | 实测值 | 来源 |
|---|---|---|
| `hidden_size` / `num_attention_heads` | 4096 / 64 | `Glm5NextTextConfig()` |
| `qk_head_dim` | 256 | `0 + 256` |
| `head_dim` | **0** | 被强制 `= qk_rope_head_dim` |
| `mlp_layer_types` | 3 dense + 42 sparse | `__post_init__` 生成 |
| `layer_types` | 34 KDA + 11 MLA | `__post_init__` 生成 |
| `base_model_tp_plan` | **13 条** | `len(c.base_model_tp_plan)` |
| `base_model_ep_plan` | 4 条 | `len(c.base_model_ep_plan)` |

## 与后续课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| `layer_types` 的 34 : 11 怎么排出来 | L0-03 |
| `indexer_types` 的 full / shared 怎么算 | L0-03 |
| 288 选 8、`routed_scaling_factor=2.5` | L0-04 |
| `hc_mult=4` 与 Sinkhorn 迭代 | L0-05 |
| 线性注意力那四个字段怎么用 | L4-02 ~ L4-05 |
| `index_topk=2048` 个位置怎么选 | L4-06 |
| `tp_plan` / `ep_plan` 怎么变成真的切分 | L8 |
| 多模态外壳 `Glm5NextConfig` 怎么拼 text + vision | L2-06 |

## 验收点

- [x] 保真门禁：12 个引用块全部逐字来自 `configuration_glm5_next.py`
- [x] 覆盖度门禁：1 个源文件被声明（属覆盖域文件之一）
- [x] 参数门禁：无非法参数
- [x] 语法检查：`node --check` 通过，9 幕字段完整
- [x] 渲染门禁：无 JS 错误、无布局溢出、交互可用（两种分辨率）
- [x] **验收点 1：能从 config 默认值手算出 MLA 层的 Q/K/V 形状** —— 见「核心结论 1」的投影表
- [x] **验收点 2：能解释为什么 `head_dim` 是 0 而不是 256** —— 见「核心结论 2」与第 5 幕
- 自检：对着 `#visual` 的形状链，逐一说出每个数字来自 config 的哪一行

**一句话总结**：

> config 默认值 = 架构：4096 宽的隐藏状态、64 个头，每头的 `qk_head_dim = 0 + 256 = 256` 维；
> **`head_dim = 0` 是「这个架构不做 RoPE」的编码，而不是「头宽为 0」。**
