<!-- glm5-coverage
models/glm5_next/configuration_glm5_next.py
-->

# L0-03 · 层类型排布：34 层 KDA + 11 层 MLA — 源文件

**这一课只回答一个问题：45 层的「排班表」是怎么从 config 里长出来的。**

视角：把 `layer_types` 与 `indexer_types` 当成两张**排班表**（每层一项，长度都是 45），
看谁跑什么注意力、谁负责产出 top-k 索引、谁只是消费别人选好的位置，
以及这套 3:1 的班次背后的成本结构。

| 文件 | 行数 | 本课引用块 |
|---|---|---|
| `models/glm5_next/configuration_glm5_next.py` | 321 | 12 |

**实测的排班结果**（在 `.venv` 里实例化 `Glm5NextTextConfig()` 跑出，不是估算）：

```text
Glm5NextTextConfig() 默认值（num_hidden_layers = 45）
  layer_types     = [linear_attention, linear_attention, linear_attention, indexed_attention] 循环 11 次
                    + [linear_attention]
                    -> linear_attention 34 层（KDA）
                    -> indexed_attention 11 层（MLA + DSA），恰好是 idx % 4 == 3 的层：3, 7, ..., 43
  indexer_types   = 45 个 'full'（index_topk_freq 默认 1 -> 每一层都被判为 full）
  mlp_layer_types = 3 层 'dense' + 42 层 'sparse'
  index_topk = 2048   index_kpool = 16      -> 2048 = 16 × 128（128 个池）
  index_head_dim = 128   index_n_heads = 32

同一份代码换层数：
  num_hidden_layers = 8  -> 6 × linear_attention + 2 × indexed_attention（idx 3、7）
  规律：indexed_attention 的层数 = num_hidden_layers // 4，其余为 linear_attention

索引器排班的两个旋钮（都是 kwargs，不是 dataclass 字段）：
  index_topk_freq = 4（offset 用默认 2）
      -> full 落在 [0, 1, 5, 9, ..., 41]，45 层里 12 层 full、33 层 shared
  index_topk_pattern = "FSSF"
      -> indexer_types = ['full', 'shared', 'shared', 'full']（长度 4，不会补齐到 45）
```

> 上面这段是**实测汇总**，用 `text` 块标出 —— 它不是源文件里的一行，
> 因此不参与保真校验。下面每一段 `python` 块都是逐字引用。

---

## 一、入口：45 层，以及"两张同长的表"

整个文件里唯一决定主干规模的一行：

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
    num_hidden_layers: int = 45
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
```

**读法**：
- `layer_types`、`indexer_types`、`mlp_layer_types` 三张表**长度都等于这一行**。
  45 不是一个随便的数字：它是所有排班表的分母，也是后面写循环时唯一要盯住的量。
- `num_attention_heads == num_key_value_heads == 64` 是这套配置的隐含前提 ——
  `validate_architecture` 的第一条检查就是它（见第八节）。这里没有 GQA 的折扣：
  注意力头的压缩完全交给 MLA 的 LoRA 潜变量去做，而不是靠减少 KV 头。
- 这个类是用 `@strict` 装饰的 dataclass，所以 `__post_init__` 与
  `validate_architecture` 在**每次构造 config 时都会执行**（包括从 checkpoint 反序列化时）。
  下面那两段"生成排班表"的代码不是可选工具，而是必经之路。

---

## 二、排班表的默认值：全是 `None`

三张表在字段声明里的默认值都不是表，而是 `None`：

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
- `None` 是"待生成"的记号：真正的排班表在 `__post_init__` 里现算。
  只有 checkpoint 显式带了 `layer_types` 时，才会跳过生成分支直接沿用 —— 这条分支给了
  自定义层数的模型（比如小规模实验配置）一条不写死 45 的口子。
- 注意第 139 行那条注释与上一节 `text` 块里的实测一致：`"full"` 就是**跑索引器**，
  `"shared"` 就是**复用上一个 full 层选好的位置**。这张表控制的是索引器，不是注意力本身。
- `head_dim: int = 0` 与 `qk_rope_head_dim: int = 0`（第 114 行）是同一件事的两面：
  DSA 层是 NoPE，位置信息不由旋转位置编码提供。这条链的完整形状推理在 L0-02。

---

## 三、★ `idx % 4 != 3`：34 + 11 就是这一行算出来的

整份文件里最重要的一段。它把"哪些层是 KDA、哪些层是 MLA"压缩成了一个模运算：

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
        if self.layer_types is None:
            kda_layers = [idx for idx in range(self.num_hidden_layers) if idx % 4 != 3]
            self.layer_types = [
                "linear_attention" if layer_idx in kda_layers else "indexed_attention"
                for layer_idx in range(self.num_hidden_layers)
            ]
```

**读法**：
- 读法只有一句：**每 4 层一个周期，前 3 层是 KDA，第 4 层（`idx % 4 == 3`）是 MLA + DSA。**
  45 层于是被切成 11 个完整周期（44 层）外加一个尾巴层（第 44 层，`44 % 4 == 0` → KDA）：
  **KDA 34 层、MLA 11 层**。写成公式就是 `MLA = num_hidden_layers // 4`。
- 为什么先列表再判断成员，而不是直接写条件表达式？因为这两行分两步表达了同一件事：
  第一步"算出 KDA 的层号集合"，第二步"给每一层贴标签"。看着绕，但它把一个
  **集合语义**写清楚了 —— 如果一个层号不在 KDA 集合里，那它一定是 MLA。
  也就是说这两种类型**穷尽了所有层**，不存在第三种注意力。
- 层数不是 4 的倍数时，尾巴一定是 KDA：`num_hidden_layers=8` 实测得到
  6 层 KDA + 2 层 MLA（idx 3、7）。这个细节在服务小模型配置时会冒出来 ——
  "最后一层总是线性注意力"并不是设计目标，只是取模的副产品。
- 回顾 L0-01：主干循环里那行 `causal_mask_mapping[self.config.layer_types[i]]`
  消费的就是这张表。**这张表不产生任何计算，但它决定了 45 次循环每次走哪条分支。**

紧接着的一行是给外部输入准备的兼容层：

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
        self.layer_types = [
            "indexed_attention" if layer_type == "full_attention" else layer_type for layer_type in self.layer_types
        ]
```

**读法**：
- 这段没有 `if self.layer_types is None` 保护 —— 它对**生成出来的表和外部传进来的表**都执行。
  旧名字 `"full_attention"` 会被就地翻译成 `"indexed_attention"`，其余值原样保留。
- 为什么需要它：`layer_types` 是 checkpoint 里会出现的字段。历史配置用的是
  "full attention" 这套词汇，而这一代的建模代码只认 `"indexed_attention"`。
  这一行把命名债挡在 config 层，让 `Glm5NextTextDecoderLayer` 只需要做一次字符串比较。
- 副作用也要记住：**它只改名，不校验**。写一个拼错的类型字符串（既不是
  `"linear_attention"` 也不是 `"indexed_attention"`）不会被这里拦住，
  而是在建模代码的 `if self.block_type == "linear_attention"` 里静默地走到 MLA 分支。

---

## 四、两条排班轴互不影响

同一个 `__post_init__` 里，MLP 轴也是现算的：

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
- MLP 轴的规则是"前 3 层稠密，其余稀疏"，周期是 1 段而不是循环 ——
  与注意力轴的 `% 4` 完全无关。两条轴相乘才是每一层的真实形态：
  45 层里实际出现的是 KDA×稠密（3 层）、KDA×稀疏（31 层）、MLA×稀疏（11 层），
  **MLA×稠密这一格是空的**。
- `min(3, self.num_hidden_layers)` 暴露了作者的防御心态：层数少于 3 时列表不会算出负数长度。
  对比注意力轴，那里没有类似的保护 —— 因为 `% 4` 在 `range(0)` 上天然安全。
- 为什么稠密层要放在**最前面**（而不是均匀插花）是 L0-02 的问题；这一课只要记住
  "两条轴各自独立，注意力轴才是 3:1 的那个"。

---

## 五、`indexer_types`：谁跑索引器，谁复用上一层的结果

类文档字符串先把这两个词定义死：

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
    layer_types (`list[str]`, *optional*):
        Per-layer attention cache schedule. Values are `"linear_attention"` for
        KDA layers and `"indexed_attention"` for MLA (DSA) layers.
    indexer_types (`list[str]`, *optional*):
        Per-layer DSA indexer mode. Values are `"full"` (run the indexer) or `"shared"`
        (reuse the previous full layer's top-k selection).
```

**读法**：
- 两张表的长度都是 45，但含义完全不同：`layer_types` 决定**缓存形态**
  （常数状态还是线性增长的 KV），`indexer_types` 决定**每层要不要算那 2048 个位置**。
- `"shared"` 的定义里藏着一条硬约束：**复用的是"上一个 full 层"，不是"上一层"。**
  实现里这份选择只通过一个变量（`prev_topk_indices`）在**相邻层**之间传递，
  所以 shared 层必须紧跟在产出它的 full 层后面。第五节末尾有实测。
- KDA 层没有索引器，它在 `indexer_types` 里的那一项**没有作用** ——
  这两张 45 长的表只在 `indexed_attention` 的层号上真正相交。

默认排班由两个 kwargs 推出：

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
            else:
                freq = max(kwargs.get("index_topk_freq", 1), 1)
                offset = kwargs.get("index_skip_topk_offset", 2)
                self.indexer_types = [
                    "full" if (max(i - offset + 1, 0) % freq) == 0 else "shared" for i in range(self.num_hidden_layers)
                ]
```

**读法**：
- 判据是 `max(i - offset + 1, 0) % freq == 0`。默认 `freq=1` 时**任何整数取模 1 都是 0** ——
  于是 45 层全部判为 `"full"`（实测就是 45 个 `'full'`）。
  所谓"默认全 full"不是什么设计决策，只是 `freq` 默认值 1 的算术后果。
- `max(..., 0)` 让开头几层不会因为减去 offset 变成负数：`i=0` 时括号里是 0，仍然 full。
  这也是为什么"偏移"的语义是**跳过前 offset 层之后，再每隔 freq 层留一个 full**。
- 换成 `freq=4`（offset 默认 2）实测：full 出现在 `[0, 1, 5, 9, ..., 41]`，共 **12 层 full、33 层 shared**。
  注意这个排班**只看层号，不看层类型** —— 12 个 full 全都落在 KDA 层上，
  而 11 个 MLA 层全是 shared。在一个 8 层的孪生配置上实测这种组合：建模时直接抛
  `ValueError: Shared DSA layers require top-k indices from a previous full indexer layer.`
  因为 MLA 层前面紧挨着的是 KDA 层，而 KDA 层会把 `topk_indices` 重置为 `None`。
  **结论：这两个旋钮必须和 `layer_types` 对齐使用**；默认排班（每 4 层一个 MLA）
  永远不会出现相邻的两个 MLA 层，所以默认只能是全 full。
- 顺带记住：`freq`、`offset` 都不在 dataclass 字段里，它们是从 `**kwargs` 里取的。
  传了就会留在 config 对象上，但不写进 `to_dict()` 的字段声明 ——
  这是"可调但非标准"的参数惯用写法。

也支持直接给一段模式串：

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
        # Per-layer indexer mode: a pattern (e.g. `"FSSF..."`) overrides the freq/offset schedule.
        if self.indexer_types is None:
            pattern = kwargs.get("index_topk_pattern")
            if pattern is not None:
                self.indexer_types = (
                    [{"F": "full", "S": "shared"}[c] for c in pattern] if isinstance(pattern, str) else list(pattern)
                )
```

**读法**：
- `"F"` → full、`"S"` → shared，逐字符映射；传 list 则原样收下。
  这是给"我就是要这一层 full、那一层 shared"的场景准备的逃生口。
- **模式串的长度就是表的长度**，代码没有补齐、没有循环、没有 `range(num_hidden_layers)`。
  实测：45 层的 config 传 `"FSSF"` 得到的是长度 4 的 `indexer_types`；
  真拿去建模型会在第 4 层之后 `IndexError: list index out of range`。
  用模式串表达 45 层，就要老老实实写 45 个字符（或者传等长的 list）。
- 三个入口的优先级：外部显式给的 `indexer_types` > `index_topk_pattern` > `freq/offset`。
  读到一份 config 时，先看 `indexer_types` 是不是被显式写死了，再看有没有模式串。

---

## 六、`index_kpool`：top-k 预算的最小单位

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
    index_kpool (`int`, *optional*, defaults to 16):
        Pool size of the compressed token groups selected by the DSA indexer.
    index_kpool_always_select_tail (`bool`, *optional*, defaults to `True`):
        Whether the incomplete KPool tail is always included in sparse attention.
```

**读法**：
- 索引器不是逐个 token 打分的：它先把历史压成**每 16 个 token 一组的池**，
  对池打分，再展开成 token。这就是 `index_topk=2048` 与 `index_kpool=16` 的关系：
  `2048 // 16 = 128` 个池，每池展开 16 个位置，正好 2048。
- 池化带来一个尾巴问题：序列长度不是 16 的倍数时，最后一池是"残缺池"。
  `index_kpool_always_select_tail=True` 保证它一定被带上 ——
  长上下文里最新写入的 token 往往正是最该被看见的，不能被"凑不满一组"丢掉。

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
    index_kpool: int = 16
    index_kpool_always_select_tail: bool = True
```

**读法**：
- 这两个字段是 dataclass 字段（不像 `freq/offset` 躲在 kwargs 里），
  它们会影响形状与掩码宽度，所以必须进 `to_dict()`、必须被 checkpoint 记录。
- 2048 与 16 都是"预算类"参数：它们不改变模型参数，只改变**每个 token 需要看多少个位置**，
  因此在服务端是可以按显存预算在 config 层调的两个旋钮（前提是下一节的整除约束成立）。

---

## 七、★ 为什么是 3:1 而不是 1:1：增长的那部分只剩 11/45

两种层的缓存成本，形状完全不同：

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
    linear_head_dim: int = 128
    linear_num_heads: int = 64
```

**读法**：
- KDA 层持有一个**递推状态**，它的形状是
  `(batch, linear_num_heads, linear_head_dim, linear_head_dim)`（在微型孪生模型上实测为
  `(1, 2, 8, 8)`，与这里的 64、128 按同一公式放大）。
  **它不含序列长度这一维** —— 无论上下文多长，每层就是
  `64 × 128 × 128 = 1,048,576` 个数；实现里还被固定存成 float32。
- MLA 层缓存的是压过的潜变量，每 token 每层 `kv_lora_rank = 512` 个数
  （`qk_rope_head_dim = 0`，没有额外那一份；微型孪生模型实测 KV 形状 `(1, 1, 6, 16)`，
  序列维就是我喂进去的 6 个 token，宽度 16 就是孪生配置的 `kv_lora_rank`）。
  **它随序列线性增长。**

把两个数放在一起，交点非常干净：

```text
每层 KDA 状态（常数）     ： 64 × 128 × 128 = 1,048,576 个数
每层 MLA 缓存（每 token） ： kv_lora_rank   =       512 个数
交点 = 1,048,576 / 512 = 2,048 个 token
      -> 上下文短于 2048 时，KDA 的常数状态反而更贵；长于 2048 后 KDA 开始便宜

按层比较（每层缓存/状态的元素个数）：
  seq           每层 MLA        每层 KDA        倍数
  2 048         1,048,576        1,048,576        1×
  8 192         4,194,304        1,048,576        4×
  32 768       16,777,216        1,048,576       16×
  131 072      67,108,864        1,048,576       64×
  1 048 576   536,870,912        1,048,576      512×

45 层的账（seq = 131 072）：
  假设 45 层全放 MLA ： 45 × 512 × 131072 = 3,019,898,880 个数
  实际 11 层 MLA     ： 11 × 512 × 131072 =   738,197,504 个数
  34 层 KDA 常数状态 ： 34 × 1,048,576     =    35,651,584 个数
  合计 773,849,088 -> 约为"全 MLA"的 1/3.9；而增长的那部分只来自 11/45 = 24.4% 的层
```

**读法**：
- **3:1 的直接收益是"增长的那部分只剩 11/45"。** 全 MLA 时每 token 要存
  `45 × 512 = 23,040` 个数，现在只要 `11 × 512 = 5,632` 个数 —— 差 4.09 倍。
  上下文越长，这一项越主导，因为 KDA 那部分是常数。
- 那为什么不是更激进的 7:1 甚至纯 KDA？**这一步是推理，不是实测**：
  KDA 把整段历史压进一个固定大小的状态，是**有损**的；DSA 的稀疏 top-k 检索
  是"从原始 KV 里精确挑出 2048 个位置"，是无损的精确寻址。
  长上下文任务里总有一些层需要精确找回某个远处的 token，所以保留 11 层 MLA 作为检索锚点。
  **3:1 是在"缓存不涨"和"能精确检索"之间取的折中，不是纯算力最优点。**
- 为什么不是 1:1？1:1 会把每 token 的增长缓存直接翻倍（22 层 × 512 = 11,264 个数），
  换来的却只是把 KDA 的常数开销再省一半（约 12 层 × 1,048,576）。
  在 GLM-5 的目标场景（`max_position_embeddings = 1048576`）里，
  线性增长项在 2048 token 之后就全面压过常数项，所以"少放 MLA"才是主要矛盾。
- 一个小巧合：交点 2048 与 `index_topk = 2048` 数值相同，但两者没有因果关系 ——
  一个来自 `linear_num_heads × linear_head_dim²` 与 `kv_lora_rank` 的比值，
  一个来自 DSA 的检索预算。

---

## 八、★ `index_topk % index_kpool == 0` 为什么必须存在

`validate_architecture` 在这里做的是**预算完整性**校验，不是形状校验：

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
```

**读法**：
- 索引器做的事是"**挑池子，再展开**"：它只挑出 `index_topk // index_kpool` 个池
  （2048 // 16 = 128 个），每个池展开成 `index_kpool` 个 token 位置。
  两处都用**整除**：池数由整除得出，而下游掩码的宽度是**写死的 `index_topk`**。
- 一旦不能整除，这两个数就对不上：`index_topk=2050`、`index_kpool=16` 时，
  池数仍然是 128，展开后只有 2048 个位置，比预算少 2 个；
  而掩码/padding 逻辑按 2050 去补，于是"预算 2050"与"实际可选 2048"之间的差额
  变成一片永远选不到、被填成 `-1` 的空位。更糟的是这个错**不会报错**，
  只会让一部分注意力预算静默失效。
- 所以这条约束的作用是**把一个静默的数值错配提前到建 config 的时候变成异常**。
  实测：`Glm5NextTextConfig(index_topk=2050)` 在构造时就抛
  `ValueError: index_topk (2050) must be divisible by index_kpool (16).`；
  `index_kpool=0` 也抛（`index_kpool must be positive, got 0.`）——
  后者是在防"除以零"和"池长为 0 的退化索引器"。
- 另外两条检查是同一节的兄弟：`num_attention_heads == num_key_value_heads`
  （这套 MLA 不接受 KV 头压缩），以及 `qk_rope_head_dim > 0` 直接报错
  （DSA 层必须是 NoPE）。它们共同的风格是：**能在 config 阶段拒绝的配置，
  绝不拖到前向里去报错或者静默降级。**

---

## 九、与后面课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| `mlp_layer_types` 为什么前 3 层稠密 | L0-02 |
| `head_dim = 0` / NoPE 怎么影响注意力形状 | L0-02 |
| 这 45 个字符串在主干里被谁消费 | L0-01（`causal_mask_mapping[...]`）、L4 混合注意力 |
| KDA 层的递推状态具体怎么更新 | L4-02 ~ L4-05 |
| 索引器怎么选出 2048 个位置 | L4-06 |
| `past_key_values` 里两种层各存了什么 | L5-01 ~ L5-03 |
| 288 选 8 的 MoE 经济学 | L0-04 |

**一句话总结**：

> 45 层 = 34 层 KDA（常数状态）+ 11 层 MLA（线性增长的 KV），
> 由 `idx % 4 != 3` 一条模运算生成；`indexer_types` 决定谁跑索引器、谁复用上一层选好的
> 2048 个位置，而 `index_topk % index_kpool == 0` 保证这份"预算"能被整数个池刚好切完。
