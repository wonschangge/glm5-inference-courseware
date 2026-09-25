<!-- glm5-coverage
models/glm5_next/configuration_glm5_next.py
-->

# L1-02 · Glm5NextTextConfig 逐字段精读 — 源文件

**这一课不推导形状，也不跑前向。它只做一件事：把这个 config 的 59 个字段拆成 7 组，逐组说清每个字段属于谁、默认值是谁写的、在哪一道闸门上被检查。**

视角：config 不是参数表，是**架构的可执行定义**。L0-02 已经示范过「从默认值读出形状」；
这一课换一个方向读同一个文件 —— 不看值，看**字段本身**：它属于哪个子系统、有没有别名、
是不是真的能被用户覆盖。

| 文件 | 行数 | 本课引用块 |
|---|---|---|
| `models/glm5_next/configuration_glm5_next.py` | 321 | 12 |

> 作业书写的是 322 行；实测 `wc -l` 是 **321**（文件末尾无换行）。本课按实测的 321 行记。

**实测的字段清单**（`.venv/bin/python` 内省，不是抄文档）：

```text
$ .venv/bin/python -c "from dataclasses import fields; from transformers import Glm5NextTextConfig as C; print(len(fields(C)))"
59

其中 49 个在本类里声明，10 个继承自 PreTrainedConfig：
  transformers_version  architectures  output_hidden_states  return_dict  dtype
  chunk_size_feed_forward  is_encoder_decoder  id2label  label2id  problem_type
```

**实测的分组结果**（分组规则写在 §三：按「改了这个字段，谁的行为跟着变」，不按行号）：

```text
家族                  字段数   代表字段与默认值
注意力（MLA）           10     num_attention_heads=64   num_key_value_heads=64
                              kv_lora_rank=512         q_lora_rank=1536
                              qk_rope_head_dim=0       qk_nope_head_dim=256
                              v_head_dim=256           head_dim=0
                              attention_bias=False     attention_dropout=0.0
MoE                     12     intermediate_size=12288  moe_intermediate_size=2048
                              n_routed_experts=288     n_shared_experts=1
                              num_experts_per_tok=8    n_group=1  topk_group=1
                              routed_scaling_factor=2.5  norm_topk_prob=True
                              mlp_layer_types=3×dense+42×sparse
                              output_router_logits=False  router_aux_loss_coef=0.001
KDA（线性注意力）        5     layer_types=34×linear_attention+11×indexed_attention
                              linear_head_dim=128      linear_num_heads=64
                              linear_conv_kernel_dim=4 linear_lower_bound=-5.0
DSA（稀疏索引器）        6     index_topk=2048          index_head_dim=128
                              index_n_heads=32         indexer_types=45×"full"
                              index_kpool=16           index_kpool_always_select_tail=True
mHC（多流残差）          3     hc_mult=4                hc_eps=1e-6
                              hc_sinkhorn_iters=20
主干（通用）            13     vocab_size=154880        hidden_size=4096
                              num_hidden_layers=45     hidden_act="silu"
                              max_position_embeddings=1048576
                              initializer_range=0.02   rms_norm_eps=1e-5
                              use_cache=True           pad_token_id=154820
                              bos_token_id=None        eos_token_id=None
                              tie_word_embeddings=False  swiglu_limit=10.0
基类继承                10     （见上）
                       ────
                       59
```

> 上面两段是**示意性的汇总**，用 `text` 块标出 —— 不是源文件的逐字引用，
> 因此不参与保真校验。下面每一段 `python` 块都是逐字引用。

**实测的边界行为**（同一个环境里故意配错，看谁先拦）：

```text
$ .venv/bin/python   # 只改一个字段，观察报错
Glm5NextTextConfig(num_key_value_heads=32)   → StrictDataclassClassValidationError / ValueError:
                                               num_attention_heads (64) must be the same as num_key_value_heads (32).
Glm5NextTextConfig(index_kpool=0)            → ValueError: index_kpool must be positive, got 0.
Glm5NextTextConfig(index_topk=2049)          → ValueError: index_topk (2049) must be divisible by index_kpool (16).
Glm5NextTextConfig(q_lora_rank=None)         → StrictDataclassFieldValidationError / TypeError:
                                               Field 'q_lora_rank' expected int, got NoneType   ← 没走到 validate_architecture
Glm5NextTextConfig(qk_rope_head_dim=64)      → ValueError: Expecting NoPE for the DSA attention layers, but got 64 as RoPE dim.
Glm5NextTextConfig(index_kpool=0, q_lora_rank=None)
                                             → 报的是 q_lora_rank 的「字段」错 ← 字段校验先于类级校验

$ .venv/bin/python   # 只认 kwargs 的那一层（§九）
Glm5NextTextConfig(linear_attn_config={"head_dim":64,"num_heads":32,"short_conv_kernel_size":2,"gate_lower_bound":-3.0})
  → linear_head_dim=64  linear_num_heads=32  linear_conv_kernel_dim=2  linear_lower_bound=-3.0
Glm5NextTextConfig(linear_attn_config={"gate_lower_bound":None})
  → linear_lower_bound=-5.0    （safe_gate 默认为 True，自动补回 -5.0）
Glm5NextTextConfig(linear_attn_config={"gate_lower_bound":None,"safe_gate":False})
  → linear_lower_bound=None    （显式关掉 safe_gate 才留得住 None）
Glm5NextTextConfig(index_topk_pattern="FSSF")
  → indexer_types=['full','shared','shared','full']，长度 4，而 num_hidden_layers=45 —— 没有任何长度校验
Glm5NextTextConfig(index_topk_freq=4, index_skip_topk_offset=2)
  → indexer_types 里 'full' 出现 12 次 / 共 45 层
Glm5NextTextConfig(totally_unknown_field=1)
  → 不报错；实例上多出 totally_unknown_field=1，而且它会进 to_dict()

$ .venv/bin/python   # 「不是字段」不等于「不存在」（§九）
c = Glm5NextTextConfig()
len(c.to_dict())                        → 63   （59 个字段 + qk_head_dim + model_type
                                                 + _name_or_path + output_attentions）
c = Glm5NextTextConfig(index_topk_pattern="FSSF", index_topk_freq=4, index_skip_topk_offset=2,
                       linear_attn_config={"head_dim":64}, safe_gate=False)
len(c.to_dict())                        → 68   （63 + 那 5 个 kwargs，全都挂在实例上）
"index_topk_pattern" in c.to_json_string() → True  （会被写进 config.json）
"safe_gate"          in c.to_json_string() → True
"qk_head_dim"        in c.to_json_string() → True  （__post_init__ 算出来的，也不是声明字段）
```

---

## 一、先看声明：字段为什么能当契约

先说结论：**这个类不继承任何模型实现，它只描述形状；而 `@strict` 让「字段声明」同时成为「构造时的类型契约」。**

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
@auto_docstring(checkpoint="zai-org/GLM-5.3-Flash")
@strict
class Glm5NextTextConfig(PreTrainedConfig):
```

**读法**：
- `@strict` 不是装饰性的。实测 `q_lora_rank=None` 在构造时就被
  `StrictDataclassFieldValidationError` 拦下（错误原文见 §十）—— 这一关比 `validate_architecture` 更早。
- 它还会**自动发现**类里名为 `validate_architecture` 的方法，把它当作类级校验器调用。
  §十 的 5 处 `raise` 不需要任何注册代码就能生效，靠的就是这个命名约定。
- `PreTrainedConfig` 提供 10 个公共字段（`return_dict` / `dtype` / `architectures` …），
  所以「本文件写了多少字段」与「这个类实际有多少字段」不是同一个数：**49 vs 59**。
- 为什么值得先看这一层：字段有类型、有默认值、能被 `config.json` 覆盖 ——
  这是它能当「架构定义」而不是「常量表」的前提。回顾 L1-01：那一课讲的就是这套公共契约本身。

---

## 二、三条类级声明：它是谁、忽略什么、别名指向谁

先看两条紧挨着的「身份信息」：

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
    model_type = "glm5_next_text"
    keys_to_ignore_at_inference = ["past_key_values"]
```

**读法**：
- `model_type = "glm5_next_text"` 是**字符串查找的键**，不是注释。实测 `CONFIG_MAPPING` 的类型是
  `_LazyConfigMapping`，`CONFIG_MAPPING["glm5_next_text"]` 才去 import 这个模块并返回 `Glm5NextTextConfig`。
  完整的链条见下面的实测块 —— 这就是本层说的「字符串如何找到类」在 config 这一端的接口。
- `keys_to_ignore_at_inference = ["past_key_values"]`：`past_key_values` 是每一步前向都在变的
  **运行时对象**，不是配置。Trainer 的 `prediction_step` 会读这个列表来决定哪些输出键不参与对比
  （`trainer.py:3130`，**那是本课之外的文件，不计入本课覆盖率**）。少了这一行，
  每次前向都会被当成「配置变了」，日志里全是假差异。
- 两行合起来说明：**config 不只描述数值，它还带着「我是谁」和「哪些输出不是配置」的元信息。**

```text
$ .venv/bin/python
CONFIG_MAPPING 的类型                  → _LazyConfigMapping
CONFIG_MAPPING["glm5_next_text"]       → Glm5NextTextConfig
CONFIG_MAPPING["glm5_next"]            → Glm5NextConfig
AutoConfig.for_model("glm5_next_text") → Glm5NextTextConfig 的实例

名字表的原文（models/auto/auto_mappings.py:250-252，不计入本课覆盖率）：
    ("glm5_next", "Glm5NextConfig"),
    ("glm5_next_text", "Glm5NextTextConfig"),
    ("glm5_next_vision", "Glm5NextVisionConfig"),

惰性 import 实测（同一台机器，/usr/bin/time -f "%e s %M KB"）：
    import transformers                                          → 0.81 s /  67 MB；sys.modules 里没有 torch
    from transformers.models.glm5_next import modeling_glm5_next → 3.07 s / 390 MB（连带 torch）
    仓库里 modeling_*.py 共 505 个（models/ 下 522 个目录）
```

**这段实测回答了「用时才 import 省下了什么」**：`import transformers` 只花 0.81 s / 67 MB，
因为它一个 `modeling_*.py` 都不碰；一旦真的要用某个模型，`modeling_glm5_next` 这一个文件
就要 3.07 s / 390 MB。505 个 modeling 文件如果都在 `import transformers` 时加载，
启动成本就是这个量级的乘积 —— **惰性映射把「框架可用」和「某个模型可用」拆成了两笔账**。
（惰性映射本身的实现是 L1-05 的主题，本课只钉住 config 这一端的键。）

第三条类级声明是别名表：

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
    attribute_map = {"num_local_experts": "n_routed_experts"}
```

**读法**：
- 方向是**旧名 → 新名**：key 是历史写法，value 是现在的正名。写反了就会把正名映射到一个不存在的属性上。
- 实现机制在基类：`PreTrainedConfig.__setattr__` / `__getattribute__` 会先查 `attribute_map`
  再落到真实属性（`configuration_utils.py:517-524`，**不计入本课覆盖率**）。
- 用途是**向后兼容**：老 checkpoint 的 `config.json` 里写 `num_local_experts`，
  而新代码只认 `n_routed_experts`，靠这一行把两者缝上。
- 实测（同一个环境）：

```text
$ .venv/bin/python
c = Glm5NextTextConfig(num_local_experts=99)
c.n_routed_experts                  → 99      （写别名 = 写正名）
c.num_local_experts                 → 99      （读别名 = 读正名）
"num_local_experts" in c.to_dict()  → False   （序列化只写正名）
```

- 同一类「一个概念两个名字」在这个文件里还有两处：`full_attention → indexed_attention`（§八）
  与 `linear_attn_config` 字典（§九）。**它们都不改变架构，只让别的写法还能加载。**

---

## 三、★ 声明顺序不是分组顺序

这是本课最该记住的一条。**12 行连续的声明里，MoE 字段被 MLA 字段劈成了两段。**

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
    n_shared_experts: int = 1
    n_routed_experts: int = 288
    routed_scaling_factor: float = 2.5
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 0
    v_head_dim: int = 256
    qk_nope_head_dim: int = 256
    n_group: int = 1
    topk_group: int = 1
    num_experts_per_tok: int = 8
    norm_topk_prob: bool = True
```

**读法**：
- 拆开看：L109–L111（`n_shared_experts` / `n_routed_experts` / `routed_scaling_factor`）是 MoE；
  L112–L116（`kv_lora_rank` … `qk_nope_head_dim`）是 MLA；L117–L120（`n_group` / `topk_group` /
  `num_experts_per_tok` / `norm_topk_prob`）又回到 MoE。
- 所以「按行号读」会得出错误结论：`kv_lora_rank` 夹在 MoE 字段中间，但它和 MoE 没有任何关系。
- 分组只能按**语义**：改这个字段，谁的行为会跟着变？`n_group` / `topk_group` 改的是路由分组，
  属于 MoE；`kv_lora_rank` 改的是 KV 的压缩宽度，属于注意力。
- 这条方法是本课全部「分组」结论的依据：**不靠位置，靠字段名与它影响的模块。**
  后面 §七 还会再遇到一次（一段里挤了三个家族）。

---

## 四、主干骨架：4 个宽度 + 3 个深度/头数

这 8 个字段不属于任何一个「优化家族」，但它们定死了所有子模块的形状：

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
    vocab_size: int = 154880
    hidden_size: int = 4096
    intermediate_size: int = 12288
    moe_intermediate_size: int = 2048

    num_hidden_layers: int = 45
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
```

**读法**：
- 它们决定**所有**子模块的输入输出宽度，所以本课单列一组（「主干」）。改了 `hidden_size`，
  注意力、MoE、KDA、mHC 会一起变 —— 这是它不能被归进某个家族的理由。
- 实测的算术：`intermediate_size = 12288 = 3 × hidden_size`；
  `moe_intermediate_size = 2048 = hidden_size / 2`。这两个比例是「288 选 8 还能省算力」的算术基础（L0-04）。
- `num_attention_heads == num_key_value_heads == 64`：**不是 GQA**。§十 的第 1 条闸门就是把这件事写成硬约束。
- `num_hidden_layers = 45` 是所有「每层一张表」字段的长度：`layer_types` / `mlp_layer_types` /
  `indexer_types` 实测都是 45 项。**它是形状与排班的共同分母。**

---

## 五、KDA 的 5 个字段（验收点 2 的答案）

KDA（线性注意力）相关的字段一共 **5 个**：4 个 `linear_*` 加 1 张排班表。

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
    linear_head_dim: int = 128
    linear_num_heads: int = 64
    linear_conv_kernel_dim: int = 4
    linear_lower_bound: float | None = -5.0
```

**读法**：
- 这 4 行是全部 `linear_*` 字段；**第 5 个是 `layer_types`（在 L138，不在这一段里）** ——
  又一次「位置不等于分组」（§三）。`layer_types` 负责「哪些层是 KDA」（默认 34 / 45），
  4 个 `linear_*` 负责「KDA 层内部怎么算」。
- 逐个看：
  - `linear_head_dim = 128`：递推状态下每个头的宽度。注意它与注意力层的 `qk_nope_head_dim = 256`
    **不是一回事** —— KDA 不跑 softmax 注意力。
  - `linear_num_heads = 64`：头数与注意力的 64 相同，但走的是递推而不是 `Q·Kᵀ`。
  - `linear_conv_kernel_dim = 4`：线性注意力里那段短卷积的核宽。
  - `linear_lower_bound = -5.0`：遗忘门衰减的下界。类型是 `float | None` —— 允许 `None` 是有意的，
    §九 会看到 `safe_gate` 会在它为 `None` 时把它补回 `-5.0`。
- KDA 层内部展开在 L4-02 ~ L4-05；本课只钉住它的字段边界。

---

## 六、DSA 的 6 个字段：从 2048 个位置到 128 个池

DSA（稀疏索引器）相关字段一共 **6 个**，这里出现 4 个：

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 32
    head_dim: int = 0
    layer_types: list[str] | None = None
    # `"full"` runs the indexer, `"shared"` reuses the previous full layer's index mask.
    indexer_types: list[str] | None = None
    base_config_key = "text_config"
```

**读法**：
- 另外 2 个（`index_kpool` / `index_kpool_always_select_tail`）在 L153–L154（§七）。
  6 个字段被拆到两段，和 §三 是同一个现象。
- 算术：`index_topk(2048) ÷ index_kpool(16) = 128` 个池；`index_head_dim = 128` 与
  `index_n_heads = 32` 是**索引器自己的**头宽与头数，与注意力层的 64 头 × 256 维无关。
  索引器是一个 32 头 × 128 维的轻量打分器（`Glm5NextTextIndexer`，在
  `modeling_glm5_next.py:774` —— **那是本课之外的文件，不计入本课覆盖率**），展开见 L4-06。
- `head_dim = 0` 与 `qk_rope_head_dim = 0` 的关系 L0-02 已经推过：**这个架构不做 RoPE**；
  §十 的第 5 条闸门就是这件事的另一面。
- `base_config_key = "text_config"`：这一行说明**这个 config 在多模态外壳里的挂载点**。
  顶层 `Glm5NextConfig` 的 `sub_configs` 把它登记在 `text_config` 这个键上（§十一）。
- `indexer_types` 的默认值不是字面量，是 `__post_init__` 算出来的：`full` 表示这一层跑索引器，
  `shared` 表示复用上一层选好的位置（排班展开见 L0-03）。

---

## 七、mHC 只有 3 个字段，另外 5 个是「借住」的

mHC（多流残差）相关字段只有 3 个，但它们和另外两个家族的字段挤在同一段：

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
    hc_mult: int = 4
    hc_eps: float = 1e-6
    hc_sinkhorn_iters: int = 20
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001

    index_kpool: int = 16
    index_kpool_always_select_tail: bool = True
```

**读法**：
- mHC 三件套：`hc_mult = 4`（残差流条数）、`hc_eps = 1e-6`（Sinkhorn 归一化的数值下限）、
  `hc_sinkhorn_iters = 20`（迭代次数）。`hc_mult = 4` 就是 L0-01 里
  「`hidden_states` 为什么是 4 维」的那个 4；展开在 L0-05 / L3-06。
- 同一段里还夹着 2 个 **MoE** 字段（`output_router_logits` / `router_aux_loss_coef`）
  和 2 个 **DSA** 字段（`index_kpool` / `index_kpool_always_select_tail`）。
- 这是 §三 那条结论的第二次出现：**一段连续声明里可以有三个家族的字段。**
  逐字段精读必须给每个字段单独判组，不能按段落打包。
- `index_kpool_always_select_tail`：池化会在序列尾部留下不足一池的残尾，
  这个开关决定残尾是否永远入选（默认 `True`）。

---

## 八、默认值是算出来的：`mlp_layer_types` 与 `num_key_value_heads`

三个「每层一张表」的字段声明成 `None`，真正的表在构造时现生成。这一段是其中的两个：

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
- `num_key_value_heads is None` 的兜底是「不写就等于 MHA」。但注意：§十 的第 1 条闸门只在
  **两个值不相等**时抛错 —— 也就是说「显式写 32」会被拒绝，而「什么都不写」会被补成 64。
  **兜底与校验是两件事**：兜底负责让默认可用，校验负责拒绝显式的矛盾。
- `mlp_layer_types` 的表达式是 `["dense"] * min(3, N) + ["sparse"] * (N - 3)`：
  前一段有 `min(3, N)` 保护，后一段没有 —— `N < 3` 时 `[...] * 负数` 得到的是**空列表**。
  实测 `num_hidden_layers = 2` → `['dense', 'dense']`：两层都稠密，不报错。
- 默认 `N = 45` → `3 + 42`。**没有任何一行硬编码 42**，它是减法算出来的。
- 这段也解释了为什么 `mlp_layer_types` 声明成 `list[str] | None` 而不是字面量：
  **它的值是一个函数的结果，不是一个常量。**

---

## 九、★ 字段表有两层：49 个声明字段 + 5 个只认 kwargs 的名字

这一段里的 5 个名字**都不是字段**：

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
        # Per-layer indexer mode: a pattern (e.g. `"FSSF..."`) overrides the freq/offset schedule.
        if self.indexer_types is None:
            pattern = kwargs.get("index_topk_pattern")
            if pattern is not None:
                self.indexer_types = (
                    [{"F": "full", "S": "shared"}[c] for c in pattern] if isinstance(pattern, str) else list(pattern)
                )
            else:
                freq = max(kwargs.get("index_topk_freq", 1), 1)
                offset = kwargs.get("index_skip_topk_offset", 2)
                self.indexer_types = [
                    "full" if (max(i - offset + 1, 0) % freq) == 0 else "shared" for i in range(self.num_hidden_layers)
                ]

        # Convert dict to attributes (if given)
        if (linear_attn_dict := kwargs.get("linear_attn_config")) is not None:
            self.linear_head_dim = linear_attn_dict.get("head_dim", self.linear_head_dim)
            self.linear_num_heads = linear_attn_dict.get("num_heads", self.linear_num_heads)
            self.linear_conv_kernel_dim = linear_attn_dict.get("short_conv_kernel_size", self.linear_conv_kernel_dim)
            self.linear_lower_bound = linear_attn_dict.get("gate_lower_bound", self.linear_lower_bound)

            # Additional lower bound logic as per original dict
            if linear_attn_dict.get("safe_gate", True) and self.linear_lower_bound is None:
                self.linear_lower_bound = -5.0
```

**读法**：
- `index_topk_pattern` / `index_topk_freq` / `index_skip_topk_offset` / `linear_attn_config` /
  `safe_gate` **都不是声明字段**。它们能到达 `__post_init__`，是因为基类 `PreTrainedConfig`
  用的是 `@strict(accept_kwargs=True)`：不属于字段的 kwargs 会被收集起来透传下来
  （`configuration_utils.py:137-172`，**不计入本课覆盖率**）—— 实测 `index_topk_pattern="FSSF"`
  确实生效，说明这条路是通的。
- 所以「config 的字段表」有两层，**但第二层不是「用完就丢」**：
  - **声明的 49 个**：有类型、有默认值、参与 `@strict` 校验；
  - **构造时认识的 5 个 kwargs**：没有声明，于是**没有类型、没有默认值、不参与任何校验**；
    但基类会把它们挂成普通属性 —— 实测它们**会进 `to_dict()`，也会被写进序列化 JSON**。
- 这正是「不是字段」≠「不存在」。实测的键数：默认实例 `to_dict()` 有 **63** 个键
  （59 个字段 + `qk_head_dim` + `model_type` + `_name_or_path` + `output_attentions`），
  传了上面那 5 个 kwargs 之后变成 **68** 个；`to_json_string()` 里能直接搜到
  `"index_topk_pattern": "FSSF"` 与 `"safe_gate": false`。
  **它们跟着 config 一起被保存，却永远不享受字段契约的保护** —— 这就是它们能溜过去的原因。
- 代价是**它们绕过了字段契约**。实测：`index_topk_pattern="FSSF"` 得到长度 **4** 的
  `indexer_types`，而 `num_hidden_layers` 是 45 —— 这里没有任何长度校验，也不会有报错。
  「能构造成功但前向必错」的配置，恰恰是 §十 那四道闸门想拦的东西；kwargs 层没有闸门。
- `linear_attn_config` 是「一个字典覆写 4 个字段」的老写法，**键名与字段名故意不同**：
  `head_dim` → `linear_head_dim`，`num_heads` → `linear_num_heads`，
  `short_conv_kernel_size` → `linear_conv_kernel_dim`，`gate_lower_bound` → `linear_lower_bound`。
  它和 §二 的 `attribute_map` 是同一类东西：**为旧写法留的门**。
- `safe_gate` 默认 `True`：实测 `gate_lower_bound=None` 会被补回 `-5.0`；
  只有显式写 `safe_gate=False` 才留得住 `None`。**一个默认开着的安全阀，关掉它必须显式。**
- `indexer_types` 的默认排班由 `freq` 与 `offset` 决定：实测
  `index_topk_freq=4, index_skip_topk_offset=2` → 45 层里 12 层是 `full`。

---

## 十、★ 四道硬约束，外加一道 NoPE 断言

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

**读法**（验收点 1 的答案）：

| # | 源码里的检查 | 它防的是什么错误 | 实测 |
|---|---|---|---|
| 1 | `num_attention_heads != num_key_value_heads` | 防止按 GQA 的思路少配 KV 头。MLA 的 K/V 由同一条压缩潜向量展开，头数必须 1:1。 | `num_key_value_heads=32` → `ValueError: ... must be the same as ...` |
| 2 | `index_kpool < 1` | 防止池化窗口为 0 或负数。**它必须排在下一行前面**：下一行要对它取模，窗口为 0 会变成 `ZeroDivisionError`。 | `index_kpool=0` → `ValueError: index_kpool must be positive, got 0.` |
| 3 | `index_topk % index_kpool != 0` | 防止 2048 个稀疏位置切不整（池化是整池操作的）。 | `index_topk=2049` → `ValueError: ... must be divisible by index_kpool (16).` |
| 4 | `q_lora_rank is None` | 防止 DSA 层没有压缩 Q —— 少了它，`q_a_proj` / `q_b_proj` 这条低秩路径就断了。 | **走不到**：`@strict` 的字段校验先以 `TypeError` 拦下 `q_lora_rank=None` |
| 5 | `qk_rope_head_dim > 0` | 防止有人在 NoPE 架构上打开 RoPE。**这一条是架构语义断言，不是参数契约。** | `qk_rope_head_dim=64` → `ValueError: Expecting NoPE for the DSA attention layers ...` |

补充四点：

- **「四道硬约束」= 上表的前 4 条参数契约**（作业书的说法）；第 5 条是语义断言。
  L0-02 把它们合称「四道形状契约 + 一道 NoPE 断言」，本课沿用这个划分。
- 第 4 条为什么还留着：它是**深度防御**。当前 `@strict` 让 `q_lora_rank=None` 在字段校验阶段就失败
  （构造与直接赋值都一样，实测两次都是 `StrictDataclassFieldValidationError`），
  但反序列化路径或将来放宽类型时，这一行仍然是一句明确的意图声明。
- 顺序有意义：实测把 `index_kpool=0` 与 `q_lora_rank=None` 同时给，报的是 `q_lora_rank` 的**字段**错 ——
  `@strict` 先跑字段校验，再跑类级校验器。
- 校验的边界：这 5 条里**没有任何一条检查 `layer_types` / `indexer_types` 的长度**
  （§九 的 `"FSSF"` 就是证明），也没有一条检查 `hidden_size % num_attention_heads`。
  **闸门只覆盖「会算错」的组合，不覆盖「所有字段」。**

---

## 十一、与后面课的接口

最后两行把本课的 config 放回多模态外壳里：

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
    model_type = "glm5_next"
    sub_configs = {"vision_config": Glm5NextVisionConfig, "text_config": Glm5NextTextConfig}
```

**读法**：
- 顶层 `Glm5NextConfig` 用 `sub_configs` 把两个子配置登记在键上：`"text_config"` → `Glm5NextTextConfig`，
  `"vision_config"` → `Glm5NextVisionConfig`。§六 的 `base_config_key = "text_config"` 正是这个键。
- 于是「字符串如何找到类」是一条**两级**的链：
  `config.json` 的 `model_type: "glm5_next"` → `Glm5NextConfig` → `sub_configs["text_config"]` →
  `Glm5NextTextConfig`。本课读的是这条链的最后一环。
- 这也解释了纯文本 checkpoint 为什么能加载多模态外壳：顶层 `__post_init__` 在 `text_config`
  为 `None` 时，会把顶层多余的 kwargs 转发给文本配置。本课只提不引，展开见 L1-03。

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| `PreTrainedConfig` 的公共字段与序列化契约 | L1-01 |
| `validate_architecture` 之外的校验器（`validate_token_ids` 等） | L1-01 |
| 视觉配置 `Glm5NextVisionConfig` 与顶层嵌套 | L1-03 |
| 惰性映射 `_LazyConfigMapping` / AutoModel 怎么按字符串找类 | L1-05 |
| `layer_types` 的 34 : 11 与 `indexer_types` 的排班 | L0-03 |
| 288 选 8、`routed_scaling_factor = 2.5` 的作用 | L0-04 |
| `hc_mult = 4` 与 20 次 Sinkhorn 迭代 | L0-05 / L3-06 |
| 4 个 `linear_*` 字段在 KDA 层里怎么用 | L4-02 ~ L4-05 |
| `index_topk = 2048` 个位置怎么选出来 | L4-06 |
| `base_model_tp_plan` / `base_model_ep_plan` 怎么变成真的切分 | L8 |

**一句话总结**：

> 59 个字段 = 7 组：注意力 10 + MoE 12 + KDA 5 + DSA 6 + mHC 3 + 主干 13 + 基类 10；
> **文件里的书写顺序不是分组顺序**，而且字段表其实有两层 —— 声明的 49 个能被 `config.json` 覆盖，
> 另外 5 个名字只在构造那一次有效。
