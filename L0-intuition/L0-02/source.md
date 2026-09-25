<!-- glm5-coverage
models/glm5_next/configuration_glm5_next.py
-->

# L0-02 · 配置即架构：从 config 读出全部形状 — 源文件

**这一课不跑任何前向。它只做一件事：把一个 321 行的 config 读成一张形状表。**

视角：`Glm5NextTextConfig` 的默认值不是「可以随便调的超参数」，而是**架构本身** ——
45 层里每一层的 Q/K/V 投影宽度、MLP 的稠密/稀疏排班、哪些配置组合会被直接拒绝，
全部写在这一个文件里。

| 文件 | 行数 | 本课引用块 |
|---|---|---|
| `models/glm5_next/configuration_glm5_next.py` | 321 | 12 |

**实测的默认值**（`.venv/bin/python` 内省 `Glm5NextTextConfig()`，不是抄文档）：

```text
Glm5NextTextConfig() 实测默认值
  宽度      hidden_size = 4096    intermediate_size = 12288    moe_intermediate_size = 2048
  深度      num_hidden_layers = 45
  注意力    num_attention_heads = 64     num_key_value_heads = 64
  MLA 压缩  q_lora_rank = 1536          kv_lora_rank = 512
  头内维度  qk_rope_head_dim = 0        qk_nope_head_dim = 256      v_head_dim = 256
            head_dim = 0（被 __post_init__ 强制 = qk_rope_head_dim）   qk_head_dim = 256
  专家      n_routed_experts = 288      n_shared_experts = 1        num_experts_per_tok = 8
            routed_scaling_factor = 2.5 n_group = 1                 topk_group = 1
  索引器    index_topk = 2048           index_head_dim = 128        index_n_heads = 32
            index_kpool = 16            index_kpool_always_select_tail = True
  其余      hidden_act = "silu"         max_position_embeddings = 1048576   rms_norm_eps = 1e-5
            swiglu_limit = 10.0         hc_mult = 4                 hc_eps = 1e-6
            hc_sinkhorn_iters = 20      linear_head_dim = 128       linear_num_heads = 64
            linear_conv_kernel_dim = 4  linear_lower_bound = -5.0    pad_token_id = 154820
            tie_word_embeddings = False
  并行计划  base_model_tp_plan 13 条     base_model_ep_plan 4 条
  生成的表  mlp_layer_types = 3 × dense + 42 × sparse
            layer_types      = 34 × linear_attention + 11 × indexed_attention
            indexer_types    = 45 × "full"
```

> 上面这段是**示意性的汇总**，用 `text` 块标出 —— 它不是源文件的逐字引用，
> 因此不参与保真校验。下面每一段 `python` 块都是逐字引用。

**实测的边界行为**（同一个脚本，故意配错看它会不会拦）：

```text
Glm5NextTextConfig(head_dim=256).head_dim                → 0          （传进去也被丢掉）
Glm5NextTextConfig(qk_rope_head_dim=64)                  → ValueError: Expecting NoPE for the DSA attention layers ...
Glm5NextTextConfig(num_key_value_heads=8)                → ValueError: num_attention_heads (64) must be the same as ...
Glm5NextTextConfig(index_kpool=0)                        → ValueError: index_kpool must be positive, got 0.
Glm5NextTextConfig(index_topk=2001)                      → ValueError: index_topk (2001) must be divisible ...
Glm5NextTextConfig(index_topk=2048)                      → OK（2048 % 16 == 0，这正是默认值）
Glm5NextTextConfig(q_lora_rank=None)                     → TypeError（@strict 的字段类型检查先拦下，没走到那条 ValueError）
Glm5NextTextConfig(num_hidden_layers=2).mlp_layer_types  → ['dense', 'dense']      （3 层以下的边界）
Glm5NextTextConfig(num_hidden_layers=5).mlp_layer_types  → 3 × dense + 2 × sparse
```

---

## 一、先看声明：这是一份「严格」的 dataclass

先说结论：**这个类不继承任何模型实现，它只描述形状。** 两个装饰器各有分工：
`@auto_docstring` 负责文档字符串，`@strict` 负责在构造时就按字段类型和校验器把非法组合拦下来 ——
第五节的那五处 `raise` 就是 `@strict` 驱动的。

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
@auto_docstring(checkpoint="zai-org/GLM-5.3-Flash")
@strict
class Glm5NextTextConfig(PreTrainedConfig):
```

**读法**：
- `PreTrainedConfig` 是 transformers v5 的配置基类。`Glm5NextTextConfig` 是**纯文本主干**的配置；
  多模态外壳 `Glm5NextConfig` 把它塞进 `text_config` 字段（在本文件末尾，本课只提不引）。
- `@strict` 是「字段即类型」的开关：`index_head_dim: int = 128` 这类声明同时定义**默认值**和
  **反序列化时的类型约束**。实测 `q_lora_rank=None` 在这一关就被 `TypeError` 拦下了。
- 为什么值得先看这一层：后面每一幕读到的数字，都是「有类型、有默认值、有校验」的**字段**，
  而不是随手写在函数体里的常量 —— 这决定了它们可以被 `from_pretrained` 从 `config.json` 覆盖。

两行「元信息」决定了这个类怎么被索引到：

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
    model_type = "glm5_next_text"
    keys_to_ignore_at_inference = ["past_key_values"]
```

**读法**：
- `model_type` 是 `from_pretrained` 反查配置类的键：checkpoint 的 `config.json` 里写着
  `"model_type": "glm5_next_text"`，transformers 才知道该用这个类来解析。
- `keys_to_ignore_at_inference` 把 `past_key_values` 从「配置对比」里排除掉：缓存对象每次前向都在变，
  它是**运行时状态**，不是配置。少了这一行，加载时会出现一堆假差异告警。
- 这两行合起来说明：**配置文件不只是数值表，它还带着「我是谁」的身份信息。**

---

## 二、★ 形状区块：4096 与 64 头拼出 MLA

这是全文件密度最高的一段：**把这几行记住，MLA 的形状就不需要再查了。**

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

**读法**：
- 前四个数是「宽度」：`hidden_size=4096` 是残差流的宽度；`intermediate_size=12288` 是**稠密** FFN
  的隐藏宽度（正好 3 × 4096）；`moe_intermediate_size=2048` 是**单个专家**的宽度（只有 hidden 的一半）。
  这三个数的比例就是「288 选 8 为什么能省算力」的算术基础（L0-04）。
- `num_key_value_heads = 64` 与 `num_attention_heads = 64` **相等** —— 这意味着**不是 GQA**。
  下一节会看到，`validate_architecture` 把「相等」写成了硬约束。
- `n_routed_experts` / `n_shared_experts` / `num_experts_per_tok` 是 MoE 的三个数。
  这里只需要记住一件事：**它们不影响注意力形状。**

MLA 真正需要的，是下面这五个字段：

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 0
    v_head_dim: int = 256
    qk_nope_head_dim: int = 256
    n_group: int = 1
    topk_group: int = 1
    num_experts_per_tok: int = 8
```

**读法**：
- `q_lora_rank=1536` / `kv_lora_rank=512` 是 MLA 的低秩瓶颈：**先把 Q 和 KV 压到低秩，再展开**。
  512 这个数尤其关键 —— K 和 V **共用同一条 512 维的压缩潜向量**，
  所以 `kv_a_proj_with_mqa` 的输出宽度是 `kv_lora_rank + qk_rope_head_dim = 512 + 0 = 512`。
- `qk_rope_head_dim=0` / `qk_nope_head_dim=256`：头内维度被劈成「要旋转」和「不旋转」两半，
  这里「要旋转」的那一半是 **0**。这是下一节的主角。
- `v_head_dim=256` 与 `qk_nope_head_dim=256` **恰好相等**，这让 `kv_b_proj` 的输出可以一次算成
  `num_attention_heads × (qk_nope_head_dim + v_head_dim)`，再一刀切成 K 和 V。

于是**只靠 config 默认值**，就能把所有投影形状写出来：

```text
① Q 路径
   hidden_size 4096  --q_a_proj-->  q_lora_rank 1536  --q_b_proj-->  64 × 256 = 16384
   其中 qk_head_dim = qk_rope_head_dim + qk_nope_head_dim = 0 + 256 = 256
② K/V 路径（K 与 V 共用一条压缩潜向量）
   hidden_size 4096  --kv_a_proj_with_mqa-->  512 + 0 = 512  --kv_b_proj-->  64 × (256 + 256) = 32768
   再把最后一维切成 k_nope = 256  /  v = 256
③ 注意力与输出
   scores = Q · Kᵀ / √256        → (batch, 64, seq, seq)
   o_proj :  64 × 256 = 16384    →  hidden_size 4096
```

> 上面这段推导是**示意性的算术**，用 `text` 块标出，不参与保真校验。
> 每一层实际的 `nn.Linear` 形状已在 `models/glm5_next/modeling_glm5_next.py:1128-1149` 实测核对 ——
> **那是本课之外的文件，不计入本课覆盖率。**

---

## 三、★ head_dim 被强制等于 0

`head_dim` 在这一课里只有一个问题值得问：**为什么是 0，而不是 256？**
先看它的三个邻居：

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

**读法**：
- `index_head_dim=128` / `index_n_heads=32` 是 **DSA 索引器**自己的头宽和头数，与注意力的 256 无关 ——
  索引器是一个 32 头 × 128 维的小注意力，负责挑出 2048 个位置（L4-06）。
- `head_dim: int = 0` 单独看会以为是「头宽为 0」，那头就没法算了。
  错觉来自命名：**在本仓库的约定里，`head_dim` 表示「会被 RoPE 旋转的那部分维度」**，
  而不是「头的总宽度」；总宽度叫 `qk_head_dim`。
- DSA 层是 NoPE（不做旋转），所以「要旋转的维度」就是 0。真正的头宽是
  `qk_head_dim = qk_rope_head_dim + qk_nope_head_dim = 0 + 256 = 256`。

这个 0 不是「默认值恰好为 0」，而是**被显式改写的**：

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
        # NOTE: this forces an intentional override as we have the convention of head_dim being the RoPE based dim
        kwargs.pop("head_dim", None)
        self.head_dim = self.qk_rope_head_dim
        self.qk_head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim

        super().__post_init__(**kwargs)
```

**读法**：
- 注释是作者自己留的：`this forces an intentional override as we have the convention of head_dim
  being the RoPE based dim`（这是一次**故意**的覆盖，因为我们的约定是 head_dim 表示 RoPE 维度）。
  作者知道这行会让读代码的人困惑，所以专门标注了。
- `kwargs.pop("head_dim", None)` 先**丢掉**传进来的 `head_dim`，再 `self.head_dim = self.qk_rope_head_dim`。
  实测：`Glm5NextTextConfig(head_dim=256).head_dim` → `0`。**用户传什么都会被丢掉。**
- 第三行顺手把 `qk_head_dim` 算出来（`0 + 256 = 256`）—— 后面每一层的注意力真正用的是这个数，
  不是 `head_dim`。
- 这三行写在 `super().__post_init__()` **之前**：先改写字段，再交回父类完成剩下的初始化。
- 收口在第五节：`validate_architecture` 里还有一条「`qk_rope_head_dim > 0` 就抛错」。
  也就是说 **`head_dim = 0` 不只是默认值，而是被两面夹死的约定**。

---

## 四、`__post_init__`：三张表是「算出来」的

`mlp_layer_types`、`layer_types`、`indexer_types` 三个字段的默认值都是 `None` ——
真正的表在构造时现生成。

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

**读法**：
- `num_key_value_heads` 为 `None` 时补成 `num_attention_heads`：这是「不写就等于 MHA」的兜底。
  但注意下一节 —— 一旦**显式**写下两个不相等的数，就直接抛错，不会自动帮你改成 GQA。
- `mlp_layer_types` 的表达式是 `["dense"] * min(3, N) + ["sparse"] * (N - 3)`。
  `min(3, N)` **保护了稠密前缀**（层数不足 3 时不会多生成），
  但后面那个 `N - 3` **没有保护**：`N < 3` 时它是负数，而 Python 里 `[...] * (-1)` 得到**空列表**。
  实测 `num_hidden_layers=2` → `['dense', 'dense']`：两层都稠密，没有稀疏层，**也不报错**。
- 默认 `N = 45` → `3 + 42`。这就是「前 3 层 dense，其余 42 层 sparse」的出处 ——
  **没有任何一行硬编码 42**，它是减法算出来的。

同一个 `__post_init__` 里还生成了注意力轴的那张表：

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
        if self.layer_types is None:
            kda_layers = [idx for idx in range(self.num_hidden_layers) if idx % 4 != 3]
            self.layer_types = [
                "linear_attention" if layer_idx in kda_layers else "indexed_attention"
                for layer_idx in range(self.num_hidden_layers)
            ]
        self.layer_types = [
            "indexed_attention" if layer_type == "full_attention" else layer_type for layer_type in self.layer_types
        ]
```

**读法**：
- `kda_layers` 是「`idx % 4 != 3` 的层号」，其余层标成 `indexed_attention`，
  于是 45 层被排成 **34 : 11**。展开见 L0-03 —— 本课只关心「这张表也是算出来的」。
- 紧接着的一段把 `"full_attention"` **改写成** `"indexed_attention"`：同一个概念在这个文件里有两个名字。
  这是兼容层（老配置里写的是 `full_attention`，构造时统一改名）。具体是给哪一版 checkpoint 兜底，
  **这个文件里没有写，我不做推断。**
- 这段解释了为什么 `layer_types` 声明成 `list[str] | None` 而不是字面量：
  **它是一个函数的结果，不是一个常量。**

---

## 五、★ validate_architecture：四道形状契约 + 一道 NoPE 断言

这是文件里唯一会「拒绝」你的地方。**每一条 `raise` 都对应一种「能构造成功、但前向必错」的配置。**

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.num_attention_heads != self.num_key_value_heads:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be the same as "
                f"num_key_value_heads ({self.num_key_value_heads})."
            )

        if self.index_kpool < 1:
            raise ValueError(f"index_kpool must be positive, got {self.index_kpool}.")

        if self.index_topk % self.index_kpool != 0:
            raise ValueError(f"index_topk ({self.index_topk}) must be divisible by index_kpool ({self.index_kpool}).")

        if self.q_lora_rank is None:
            raise ValueError("For DSA usage in the attention layers, the `q_lora_rank` is strictly required!")

        if self.qk_rope_head_dim > 0:
            raise ValueError(
                f"Expecting NoPE for the DSA attention layers, but got {self.qk_rope_head_dim} as RoPE dim."
            )
```

**读法**：

| # | 源码里的检查 | 它防的是什么错误 |
|---|---|---|
| 1 | `num_attention_heads != num_key_value_heads` | 防止按 GQA 的思路少配 KV 头。MLA 的 K/V 由同一条压缩潜向量展开，头数必须 1:1。实测传 `num_key_value_heads=8` 直接抛错。 |
| 2 | `index_kpool < 1` | 防止池化窗口为 0 或负数：`index_topk` 要按 `index_kpool` 分组，窗口为 0 会除零。实测 `index_kpool=0` 抛错。 |
| 3 | `index_topk % index_kpool != 0` | 防止 2048 个稀疏位置切不整。实测 `index_topk=2001` 抛错；默认 `2048 % 16 == 0` 通过。 |
| 4 | `q_lora_rank is None` | 防止 DSA 层没有压缩 Q —— 少了它，`q_a_proj` / `q_b_proj` 这条低秩路径就断了。实测走不到这条：`@strict` 的字段类型检查先以 `TypeError` 拦下 `q_lora_rank=None`，所以这条 `raise` 更像是给反序列化路径兜底。 |
| 5 | `qk_rope_head_dim > 0` | 防止有人在 NoPE 架构上打开 RoPE。实测 `qk_rope_head_dim=64` → `ValueError: Expecting NoPE for the DSA attention layers`。这一条正是第三节那个 `0` 的另一面。 |

补充两点：

- 五条里没有一条检查 `hidden_size % num_attention_heads`（因为头宽不是从 hidden 除出来的，
  而是由 `qk_nope_head_dim` 直接给定），也没有一条检查 `index_head_dim`。
  **校验覆盖的是「会算错」的组合，不是「所有字段」。**
- 这五条都是 `@strict` 驱动的**类级校验**：构造时就跑，不是在第一次前向时才炸。

---

## 六、config 还描述「怎么切开」

形状讲完了，但这份 config 还多管了一件事：**权重在多卡上如何切分。**

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
    base_model_tp_plan = {
        "layers.*.self_attn.q_b_proj": "colwise",
        "layers.*.self_attn.kv_a_proj_with_mqa": "mla_kv_a_proj",
        "layers.*.self_attn.kv_b_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
```

**读法**：
- `base_model_tp_plan` 一共 **13 条**（实测 `len(c.base_model_tp_plan) == 13`），
  每条把一个权重名映射到一种切分方式：`colwise`（按输出维切）、`rowwise`（按输入维切）、
  `packed_colwise`，以及一个**自定义策略** `mla_kv_a_proj`。
- 最值得看的是 `layers.*.self_attn.kv_a_proj_with_mqa: "mla_kv_a_proj"` ——
  MLA 的压缩 KV 不能简单地按列切，需要一个专门的策略。**形状特殊，切法就得特殊。**
- `layers.*.mlp.gate_proj / up_proj / down_proj` 三条只对**稠密层**有意义；
  稀疏层走 `experts.*` 那几条，展开在 L8。

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
- `base_model_ep_plan` 4 条，是**专家并行**计划：路由器 `mlp.gate` 用 `ep_router`，
  专家的两个大矩阵用 `grouped_gemm`（把多个专家的矩阵乘合并成一次分组矩阵乘）。
- 与 `tp_plan` 的关系：TP 切的是「每个矩阵怎么分」，EP 切的是「专家放哪张卡」。
  两者在同一层上同时生效，所以它们是两张表而不是一张。

最后一行「身份信息」是别名兼容：

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
    attribute_map = {"num_local_experts": "n_routed_experts"}
```

**读法**：
- `attribute_map` 让旧名字 `num_local_experts` 继续可用，实际读写的是 `n_routed_experts`。
- 它和第四节的 `full_attention → indexed_attention` 改名是同一类东西：
  **这个文件里有好几处「同一个概念两个名字」**（还有 `__post_init__` 里接受
  `linear_attn_config` 字典来覆写线性注意力四个维度的写法，见 L4）。
  它们都不改变架构，只让别的写法还能加载。

---

## 七、与后面课的接口 + 一句话总结

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| `layer_types` 的 34 : 11 怎么排出来的 | L0-03 |
| `indexer_types` 的 full / shared 怎么算 | L0-03 |
| 288 选 8、`routed_scaling_factor=2.5` 的作用 | L0-04 |
| `hc_mult=4` 与 20 次 Sinkhorn 迭代 | L0-05 |
| `linear_head_dim` 等四个线性注意力字段怎么用 | L4-02 ~ L4-05 |
| `index_topk=2048` 个位置怎么选出来 | L4-06 |
| `tp_plan` / `ep_plan` 怎么变成真的切分 | L8 |
| 多模态外壳 `Glm5NextConfig` 怎么拼 text + vision | L2-06 |

**一句话总结**：

> config 默认值 = 架构：4096 宽的隐藏状态、64 个头，每头的 `qk_head_dim = 0 + 256 = 256` 维；
> **`head_dim = 0` 是「这个架构不做 RoPE」的编码，而不是「头宽为 0」。**
