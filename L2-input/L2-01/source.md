<!-- glm5-coverage
tokenization_utils_base.py
-->

# L2-01 · 分词器公共契约：慢与快共享的那一层 — 源文件

**这一课不解释任何模型。它只回答一个问题：你写下的那行 `tokenizer(text, ...)`，
在你拿到结果之前经过了几道手续、每道手续由哪一行代码负责。**

视角：L2 层整层讲"文本和像素如何变成同一串位置上的向量"。这一课先把**所有分词器共享的那层契约**钉死 ——
不管背后是 Rust 的快分词器还是纯 Python 的慢分词器，`__call__` 的签名、`padding` / `truncation` 的归一化、
`attention_mask` 的产生、`return_tensors` 的落点，全都写在同一份文件里。

| 文件 | 行数 | 本课引用块 |
|---|---|---|
| `tokenization_utils_base.py` | 3690 | 14 |

**行数订正**：作业书写 3691 行，实测是 **3690** 行 —— `wc -l` 与 `awk 'END{print NR}'` 都给 3690。
3691 这个数来自 `open(f).read().split("\n")`：文件以换行结尾，切出来会多一个空串。
本课所有引用块的行号都按 3690 行这一版标注（`source.md` 里的 `codeStart` 也是）。

> 本课另有一份**实测探针** `_data/recon/probe_l201.py`（**脚本，不计入覆盖率**）：
> 它用 `tokenizers` 库手工构造一个 `PreTrainedTokenizerFast`，全程零网络、不需要下载任何模型，
> 走的正是本文件这条公共路径。下文每一处「实测」都来自它的真实输出，不是推断。

---

## 零、实测订正：`SpecialTokensMixin` 已经不在了

作业书把第一个讲解要点写成「SpecialTokensMixin 与 token 属性体系」。
**实测订正**：在本仓库当前版本里，`SpecialTokensMixin` 这个类**不存在**：

```text
$ grep -rn "SpecialTokensMixin" upstream-transformers/src/transformers/ | wc -l
2
$ grep -rn "class SpecialTokensMixin" upstream-transformers/src/
（无输出）
```

两处命中都只是注释，没有任何类定义、也没有任何 import 引用它。它的 API 被**并入了
`PreTrainedTokenizerBase`**，源码自己在两个地方留了记号 —— 第一个就在类属性区：

<!-- src: tokenization_utils_base.py -->
```python
    # first name has to correspond to main model input name
    # to make sure `tokenizer.pad(...)` works correctly
    model_input_names: list[str] = ["input_ids", "attention_mask"]
    padding_side: str = "right"
    truncation_side: str = "right"
    slow_tokenizer_class = None

    # Special tokens support (moved from SpecialTokensMixin)
    # V5: Clean separation of named special tokens from extra special tokens
    SPECIAL_TOKENS_ATTRIBUTES = [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
    ]
```
**读法**：
- 第 8 行那句 `# Special tokens support (moved from SpecialTokensMixin)` 是**迁移留下的化石**：
  它自己说明了这批 API 是**从 `SpecialTokensMixin` 搬过来的**，而全仓库已经搜不到那个类。
  所以本课讲的是「token 属性体系」这套 API，而不是「`SpecialTokensMixin` 这个类」。
- `model_input_names` 是**这一课最容易被跳过、但决定一切的一行**：它默认就是
  `["input_ids", "attention_mask"]`。后面 `_pad` 里那句
  `return_attention_mask = "attention_mask" in self.model_input_names`
  直接读它 —— **`attention_mask` 会不会被造出来，是这个类属性说了算，不是调用方说了算。**
- `padding_side` / `truncation_side` 也是**类属性默认值**，`__init__` 里可以被 kwargs 覆盖，
  但合法取值只有 `"right"` / `"left"` 两个（`__init__` 里各有一个 `raise ValueError` 守着）。
- **回顾 L1-06**：`AutoTokenizer` 的那张分派表（`models/auto/tokenization_auto.py`）决定你
  **拿到哪一个** tokenizer 类；从拿到那一刻起，所有类共享的就是本课这一层。
  L1 管"找类"，L2 管"找到之后怎么用" —— 这就是两层的分界线。
- `SPECIAL_TOKENS_ATTRIBUTES` 是**7 个固定名字**。注意它是 list 而不是 set：
  顺序有语义 —— `special_tokens_map` / `all_special_tokens` 都按这个顺序遍历，
  实测 `all_special_tokens` 返回 `['[UNK]', '[SEP]', '[PAD]', '[CLS]']`，
  正是这个列表顺序过滤掉未设置项的结果。

---

## 一、token 属性体系：token 与 id 是同一份数据的两个视图

7 个名字里，每个都同时支持两种读法：`pad_token` 给字符串，`pad_token_id` 给整数。
**它们不是两套存储，是同一份存储的两个投影** —— 证据是这对 `__setattr__` / `__getattr__`：

<!-- src: tokenization_utils_base.py -->
```python
    def __setattr__(self, key, value):
        # Handle _id/_ids suffix (eg. bos_token_id -> bos_token)
        key_without_id = key.removesuffix("_ids").removesuffix("_id") if key.endswith(("_id", "_ids")) else key

        # Named special tokens (bos_token, eos_token, etc.)
        if key_without_id in self.SPECIAL_TOKENS_ATTRIBUTES:
            if key != key_without_id and value is not None:
                value = self.convert_ids_to_tokens(value)
            if value is not None and not isinstance(value, (str, AddedToken)):
                raise ValueError(f"Cannot set a non-string value as the {key_without_id}")
            self._special_tokens_map[key_without_id] = value
            return
```
**读法**：
- 第 1 行先用 `removesuffix` 把 `_id` / `_ids` 剥掉，于是 `pad_token_id` 与 `pad_token`
  在**同一个分支**里被处理。这就是"两个视图"的字面实现。
- `if key != key_without_id`：**只有带 `_id` 后缀的赋值才需要翻译**。
  实测 `tok.pad_token_id = 3` 之后 `tok.pad_token == '[SEP]'` —— 传进去的是整数 3，
  存下来的是字符串 `'[SEP]'`。存的是 token，不是 id。
- `value is None` 被显式放过（`if key != key_without_id and value is not None`）：
  允许 `tokenizer.pad_token = None` 把 token **清空**，而不是抛错。
  实测赋 `None` 之后 `pad_token` / `pad_token_id` 都变回 `None`，
  并且它**从 `special_tokens_map` 里消失**了 —— "未设置"与"设成 None"是同一个状态。
- 最后 `self._special_tokens_map[key_without_id] = value` 落进**一个 dict**，
  而不是 `setattr(self, ...)`。所以这些 token 不会出现在 `self.__dict__` 里 ——
  这正是读的时候必须走 `__getattr__` 的原因。

<!-- src: tokenization_utils_base.py -->
```python
    def __getattr__(self, key):
        # Handle _id/_ids suffix (eg. bos_token_id -> bos_token)
        key_without_id = key.removesuffix("_ids").removesuffix("_id") if key.endswith(("_id", "_ids")) else key

        # Named special tokens (bos_token, eos_token, etc.)
        if key_without_id in self.SPECIAL_TOKENS_ATTRIBUTES:
            # Use __dict__.get to avoid recursive __getattr__ when _special_tokens_map
            # is not yet initialized (e.g. during fast tokenizer __init__)
            token_value = self.__dict__.get("_special_tokens_map", {}).get(key_without_id)
            if token_value is None:
                if self.verbose:
                    logger.error(f"Using {key}, but it is not set yet.")
                return None
            return self.convert_tokens_to_ids(str(token_value)) if key != key_without_id else str(token_value)
```
**读法**：
- `self.__dict__.get("_special_tokens_map", {}).get(...)` 里那个 `__dict__.get` 是**防御性的**：
  注释写明是为了避开"`_special_tokens_map` 还没初始化时递归调用 `__getattr__`"
  （快分词器 `__init__` 早期就会读 token 属性）。这是 `__getattr__` 与 `__init__` 的经典冲突。
- `return None`（而不是 `raise AttributeError`）：**未设置的 token 读出来是 `None`**。
  实测 `bos_token` 与 `bos_token_id` 都是 `None`，不抛错。
- 同一行里 `convert_tokens_to_ids(...) if key != key_without_id else str(token_value)`
  与 `__setattr__` 完全对称：**带 `_id` 的读要翻译，不带的直接给字符串**。
- 代价：`tokenizer.pad_token_id` 每次读都要查一次词表。它是属性，不是缓存字段。

---

## 二、★ 策略枚举：`truncation` 的 4 个值在这里，`padding` 的 3 个值不在这里

作业书要求"padding / truncation 策略的完整枚举"。**实测订正**：两个枚举**不都在这份文件里**。
`TruncationStrategy` 定义在本文件；`PaddingStrategy` 定义在 `utils/generic.py`（**属 L1 的覆盖域，不计入本课引用块**）：

```text
utils/generic.py L579-587  PaddingStrategy(ExplicitEnum)
    LONGEST     = "longest"        # 补齐到本批最长
    MAX_LENGTH  = "max_length"     # 补齐到 max_length / model_max_length
    DO_NOT_PAD  = "do_not_pad"     # 完全不补

tokenization_utils_base.py L154-163  TruncationStrategy(ExplicitEnum)
    ONLY_FIRST / ONLY_SECOND / LONGEST_FIRST / DO_NOT_TRUNCATE
（实测：`[m.value for m in PaddingStrategy]` = ['longest', 'max_length', 'do_not_pad']
       `[m.value for m in TruncationStrategy]` = ['only_first', 'only_second',
                                                 'longest_first', 'do_not_truncate']）
```

<!-- src: tokenization_utils_base.py -->
```python
class TruncationStrategy(ExplicitEnum):
    """
    Possible values for the `truncation` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for tab-completion in
    an IDE.
    """

    ONLY_FIRST = "only_first"
    ONLY_SECOND = "only_second"
    LONGEST_FIRST = "longest_first"
    DO_NOT_TRUNCATE = "do_not_truncate"
```
**读法**：
- `ExplicitEnum` 而不是普通 `Enum`：它的成员**可以与字符串直接比较**，
  所以 `truncation="longest_first"` 和 `truncation=TruncationStrategy.LONGEST_FIRST`
  能走同一条路。下面第三行那个 `TruncationStrategy(truncation)` 就是在做这个翻译。
- 只有 4 个值，而且**`do_not_truncate` 是其中之一** —— "不截断"不是一个默认参数，
  它是一个显式策略值。所以判空必须写 `!= TruncationStrategy.DO_NOT_TRUNCATE`，不能写 `is not None`。
- `ONLY_FIRST` / `ONLY_SECOND` / `LONGEST_FIRST` 只有在**成对输入**（`text` + `text_pair`）时才有区别。
  实测（`max_length=14`，序列为 `[CLS] q [SEP] p [SEP]`）：

```text
不截断        : [2,4,5,6,7,8,9,3, 6,7,8,9,10,11,6,12,13,3]   18 个
only_first    : [2,4,5,3, 6,7,8,9,10,11,6,12,13,3]          第一句被砍到 3 个
only_second   : [2,4,5,6,7,8,9,3, 6,7,8,9,10,3]             第二句被砍到 5 个
longest_first : [2,4,5,6,7,8,3, 6,7,8,9,10,11,3]            两句轮流砍
```

下面这段是**归一化**发生的地方 —— 调用方给的 `True` / `False` / 字符串，
在这里被翻译成枚举。注意"`padding=True` 会变成 `PaddingStrategy.LONGEST`"这件事就写在里面：

<!-- src: tokenization_utils_base.py -->
```python
        # Backward compatibility for previous behavior:
        # If you only set max_length, it activates truncation for max_length
        if max_length is not None and padding is False and truncation is None:
            truncation = "longest_first"

        # Get padding strategy
        if padding is not False:
            if padding is True:
                if verbose:
                    if max_length is not None and (
                        truncation is None or truncation is False or truncation == "do_not_truncate"
                    ):
                        warnings.warn(
                            "`max_length` is ignored when `padding`=`True` and there is no truncation strategy. "
                            "To pad to max length, use `padding='max_length'`."
                        )
                padding_strategy = PaddingStrategy.LONGEST  # Default to pad to the longest sequence in the batch
            elif not isinstance(padding, PaddingStrategy):
                padding_strategy = PaddingStrategy(padding)
            elif isinstance(padding, PaddingStrategy):
                padding_strategy = padding
        else:
            padding_strategy = PaddingStrategy.DO_NOT_PAD
```
**读法**：
- 头两行是**向后兼容的暗门**：只给 `max_length`、既不说 `padding` 也不说 `truncation` 时，
  它会**自作主张打开截断**。实测 `tok(text, max_length=4)` 返回 4 个 id，
  而同样文本不传 `max_length` 时是 8 个。这个行为没有任何显式开关，只能从这 3 行读出来。
- `padding is True` 与 `padding is False` 用的是**同一性比较**（`is`），不是真假值比较。
  实测 `padding=1` 不会走进 `True` 分支，而是掉进下面的 `PaddingStrategy(1)`，
  抛 `ValueError: 1 is not a valid PaddingStrategy, please select one of [...]` —— **值必须精确**。
- `padding_strategy = PaddingStrategy.LONGEST  # Default to pad to the longest sequence in the batch`
  这一行就是别名翻译：**`True` 不是策略值，是别名**。
  实测 `padding=True` 与 `padding="longest"` 的输出逐位相同。
- 末尾那个 `else: padding_strategy = PaddingStrategy.DO_NOT_PAD` 说明 `False` 也**不是**"不设置"，
  它是一个与 `LONGEST` 平级的策略值。`truncation` 侧完全对称（见下一节开头），
  `False` 与 `None` 都落到 `DO_NOT_TRUNCATE`。
  于是这个函数出口的 4 个返回值**永远是枚举**，下游不需要再判 `True`/`False`。

---

## 三、`max_length` 的两种失效方式

同一个参数 `max_length`，在这个函数里会**被静默丢掉两次**：

<!-- src: tokenization_utils_base.py -->
```python
        # Get truncation strategy
        if truncation is not False and truncation is not None:
            if truncation is True:
                truncation_strategy = (
                    TruncationStrategy.LONGEST_FIRST
                )  # Default to truncate the longest sequences in pairs of inputs
            elif not isinstance(truncation, TruncationStrategy):
                truncation_strategy = TruncationStrategy(truncation)
            elif isinstance(truncation, TruncationStrategy):
                truncation_strategy = truncation
        else:
            truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE

        # Set max length if needed
        if max_length is None:
            if padding_strategy == PaddingStrategy.MAX_LENGTH:
                if self.model_max_length > LARGE_INTEGER:
                    padding_strategy = PaddingStrategy.DO_NOT_PAD
                else:
                    max_length = self.model_max_length

            if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE:
                if self.model_max_length > LARGE_INTEGER:
                    truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE
                else:
                    max_length = self.model_max_length
```
**读法**：
- `# Set max length if needed` 往下那 7 行（本块第 14-20 行）是**第一次失效**：`max_length` 没给、而 `model_max_length` 是默认的
  `VERY_LARGE_INTEGER`（`int(1e30)`）时，策略被**降级**成 `DO_NOT_PAD` / `DO_NOT_TRUNCATE`。
  注意它是**赋值覆盖**，不是抛错 —— 调用方看不到任何异常。
  实测：手工构造的分词器不传 `model_max_length` 时，`padding='max_length'` 的输出
  `[[2,4,5,3], [2,6,7,8,9,10,3]]` **根本没补齐**；同一个分词器传 `max_length=8` 才补到 8。
- 那两处门槛写的是 `> LARGE_INTEGER`（`int(1e20)`）而不是 `== VERY_LARGE_INTEGER`：
  只要 `model_max_length` 大得离谱就当"没设过"。实测默认值是
  `1000000000000000019884624838656`（`int(1e30)` 的 float 化结果），确实远大于 `1e20`。
- 紧随其后的 `if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE:` 分支
  处理 `padding` 与 `truncation` 的**互相牵制**：
  `padding='max_length'` 要有 `max_length` 才有意义，而 `max_length` 又只能来自
  `model_max_length`。这条链一旦断在第一步，整条策略就退化成"什么都不做"。
- 这段引用之后紧跟着 5 行 `# Test if we have a padding token`，是**整个函数里唯一会抛错的地方**：
  `if padding_strategy != PaddingStrategy.DO_NOT_PAD and (self.pad_token is None or self.pad_token_id < 0):`
  实测错误信息是 `Asking to pad but the tokenizer does not have a padding token.` ——
  它把修法也写进消息里了（`tokenizer.pad_token = tokenizer.eos_token`）。
  注意判据是**两个**：token 是 `None`，**或** id 小于 0。

回到调用方：`__call__` 把归一化的结果**只当成 4 个普通变量**继续往下传，
所以 `padding` / `truncation` 的字符串在进入编码器之前就已经消失了：

<!-- src: tokenization_utils_base.py -->
```python
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=all_kwargs.pop("padding", False),
            truncation=all_kwargs.pop("truncation", None),
            max_length=all_kwargs.pop("max_length", None),
            pad_to_multiple_of=all_kwargs.get("pad_to_multiple_of"),
            verbose=all_kwargs.get("verbose", True),
            **kwargs,
        )
```
**读法**：
- 注意它 `pop` 的是 `all_kwargs` 里的键（第 2-4 行），而 `pad_to_multiple_of` / `verbose`
  用的是 `.get`（第 5-6 行）—— 因为后两个还要继续传给下游，前三个已经被消费掉了。
- 出口的 `padding_strategy` 被原样塞进 `_encode_plus`（见第六节），
  **`_encode_plus` 的形参类型标注就是 `PaddingStrategy` 而不是 `bool | str`** ——
  类型系统在这里替我们确认了"归一化已经完成"。
- 最后那个 `**kwargs` 是被 `_get_padding_truncation_strategies` 原样返回的
  （它的返回值第 4 项就是 `kwargs`）：这个函数不改调用方的其他参数，只做翻译。

---

## 四、★ `attention_mask` 与 `input_ids` 同生同长

验收点问的是"`input_ids` 与 `attention_mask` 的产生时机"。答案在同一个函数里，
而且**两件事是被同一个变量 `difference` 决定的**：

<!-- src: tokenization_utils_base.py -->
```python
        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        required_input = encoded_inputs[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        # Initialize attention mask if not present.
        if return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * len(required_input)
```
**读法**：
- 第 2-3 行是**因果链的起点**：`return_attention_mask` 默认值来自
  `"attention_mask" in self.model_input_names`。回忆第零节的类属性 ——
  默认就是 `["input_ids", "attention_mask"]`，所以**这个 mask 是默认要造的**。
- `# Initialize attention mask if not present.` 那两行（本块第 16-17 行）：
  **mask 先于 padding 被造出来**，而且是 `[1] * len(required_input)`。
  注意条件是 `"attention_mask" not in encoded_inputs` —— 已存在的 mask 不会被覆盖
  （快分词器自己带了 mask，就是从这里"接管"的）。
- `needs_to_be_padded`（本块第 13 行）是**唯一的分支开关**：
  "策略不是 DO_NOT_PAD" **且** "当前长度不等于目标长度"。长度已经对了就一个字符都不改。
- 顺序值得注意：mask 先被造满 1，**再**由下面的 padding 分支追加 0。
  所以 mask 的长度天然等于 `input_ids` 的长度 —— 它们不是两次独立计算的结果。

<!-- src: tokenization_utils_base.py -->
```python
        if needs_to_be_padded:
            difference = max_length - len(required_input)
            padding_side = padding_side if padding_side is not None else self.padding_side

            if padding_side == "right":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"] + [0] * difference
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * difference
                    )
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
                encoded_inputs[self.model_input_names[0]] = required_input + [self.pad_token_id] * difference
            elif padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = [self.pad_token_type_id] * difference + encoded_inputs[
                        "token_type_ids"
                    ]
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input
```
**读法**：
- `difference = max_length - len(required_input)` 是**同一个数**，同时决定
  ids 补几个 pad、mask 补几个 0。这就是"同生同长"的字面证据。
- `right` 侧：`[1]*n + [0]*difference`，pad 追加在尾部；`left` 侧：`[0]*difference + [1]*n`，
  pad 插在头部。实测（`{"input_ids": [4,5,6]}` → `max_length=5`）：

```text
right : input_ids [4,5,6,1,1]   attention_mask [1,1,1,0,0]
left  : input_ids [1,1,4,5,6]   attention_mask [0,0,1,1,1]
```

- 三个兄弟字段（`token_type_ids` / `special_tokens_mask` / `input_ids`）都跟着同一个
  `difference` 一起补，**补法各不相同**：`token_type_ids` 补 `self.pad_token_type_id`，
  `special_tokens_mask` 补 **1**（pad 也是"特殊 token"），`input_ids` 补 `self.pad_token_id`。
  这一段是"一个 padding 动作，四个字段各自变形"的完整清单。
- 补 `special_tokens_mask` 用 1 而不是 0，是为了让"哪些位置是模型自己加的"这件事
  在 padding 之后仍然成立 —— pad 位确实不是真实内容。

**一个必须说清的边界**：上面的 `_pad` 是**慢分词器与显式 `.pad()` 调用**走的路径。
快分词器的 mask 由 Rust 后端在 `encode_batch` 时一起产出，转换发生在
`tokenization_utils_tokenizers.py` 的 `_convert_encoding` 里（`encoding_dict["attention_mask"].append(e.attention_mask)`，
**该文件属 L2-02 的覆盖，不计入本课引用块**）。实测：用快分词器单条编码、完全不 padding，
`attention_mask` 依然存在且全是 1 —— 它来自后端，而不是来自 `_pad`。
两条路径**语义完全一致**（长度等于 ids、有效位为 1、pad 位为 0），这正是"公共契约"的含义。

**回顾 L0-01**：`Glm5NextTextModel.forward` 的入口形参就是 `input_ids` 与 `attention_mask`
（`input_ids: torch.LongTensor | None = None`、`attention_mask: torch.Tensor | None = None`）——
本课第三段产出的，正是 L0-01 那条前向接收的那两个张量。**形状的规矩在这里定，用法在那边。**

---

## 五、★ `return_tensors` 影响的是哪一步

验收点问的是"`return_tensors` 影响的是哪一步"。答案是：**它一步都不影响编码，
它影响的是字典被构造出来的那一瞬间用哪个容器装数据。**
证据是 `BatchEncoding.__init__` 的最后一行：

<!-- src: tokenization_utils_base.py -->
```python
    def __init__(
        self,
        data: dict[str, Any] | None = None,
        encoding: EncodingFast | Sequence[EncodingFast] | None = None,
        tensor_type: None | str | TensorType = None,
        prepend_batch_axis: bool = False,
        n_sequences: int | None = None,
    ):
        super().__init__(data)

        # If encoding is not None, the fast tokenization is used
        if encoding is not None and isinstance(encoding, EncodingFast):
            encoding = [encoding]

        self._encodings = encoding

        if n_sequences is None and encoding is not None and encoding:
            n_sequences = encoding[0].n_sequences

        self._n_sequences = n_sequences

        self.convert_to_tensors(tensor_type=tensor_type, prepend_batch_axis=prepend_batch_axis)
```
**读法**：
- `tensor_type` 与 `prepend_batch_axis` 是**构造参数**，不是编码参数：
  它们在 `super().__init__(data)` 之后、`__init__` 返回之前被消费掉。
- 最后一行 `self.convert_to_tensors(tensor_type=tensor_type, prepend_batch_axis=prepend_batch_axis)`
  是**全流程里唯一一次类型转换**。`pad()` 的出口也长这样：
  `return BatchEncoding(batch_outputs, tensor_type=return_tensors)`。
- `n_sequences` 与 `_encodings` 也在这一步被固定下来 —— `is_fast` 属性就是
  `self._encodings is not None`。实测快分词器 `is_fast` 为 `True`。
- 也就是说：**`BatchEncoding` 是"数据 + 元信息 + 容器类型"的三合一**，
  而 `return_tensors` 只决定第三项。

<!-- src: tokenization_utils_base.py -->
```python
    def convert_to_tensors(self, tensor_type: str | TensorType | None = None, prepend_batch_axis: bool = False):
        """
        Convert the inner content to tensors.

        Args:
            tensor_type (`str` or [`~utils.TensorType`], *optional*):
                The type of tensors to use. If `str`, should be one of the values of the enum [`~utils.TensorType`]. If
                `None`, no modification is done.
            prepend_batch_axis (`bool`, *optional*, defaults to `False`):
                Whether or not to add the batch dimension during the conversion.
        """
        if tensor_type is None:
            return self

        # Convert to TensorType
        if not isinstance(tensor_type, TensorType):
            tensor_type = TensorType(tensor_type)
```
**读法**：
- 函数体第一句 `if tensor_type is None: return self` —— **不传 `return_tensors` 时这个函数是空操作**。
  所以 list 与 tensor 的差别**只可能**来自这里，不可能来自别处。
- `TensorType(tensor_type)` 与 `TruncationStrategy(truncation)` 是同一个套路：
  **先把字符串翻译成枚举**。实测 `TensorType` 的取值是 `pt` / `np` / `mlx`。
- 三个后端各有一个 `as_tensor` 闭包，`is_tensor` 也随之切换（PyTorch / MLX / NumPy）。
  这是一处"用闭包代替 if-else 链"的写法：分支只在**入口**判一次。

<!-- src: tokenization_utils_base.py -->
```python
        # Do the tensor conversion in batch
        for key, value in self.items():
            try:
                if prepend_batch_axis:
                    value = [value]

                if not is_tensor(value):
                    tensor = as_tensor(value)

                    # Removing this for now in favor of controlling the shape with `prepend_batch_axis`
                    # # at-least2d
                    # if tensor.ndim > 2:
                    #     tensor = tensor.squeeze(0)
                    # elif tensor.ndim < 2:
                    #     tensor = tensor[None, :]

                    self[key] = tensor
```
**读法**：
- `if prepend_batch_axis: value = [value]`（本块第 4-5 行）—— **batch 轴是在这里补的**。
  实测：`BatchEncoding({"input_ids": [4,5,6]}, tensor_type="pt")` 形状 `(3,)`，
  加上 `prepend_batch_axis=True` 变成 `(1, 3)`。
- `if not is_tensor(value)`（本块第 7 行）：已经是张量的值**原样保留**，
  所以 `tokenizer.pad()` 收到 numpy 数组时不会来回折腾。
- 本块只截到赋值那行为止；紧随其后的 `except` 分支里的报错信息本身就是文档：
  **"you should probably activate truncation and/or padding ... to have batched tensors
  with the same length"** —— 不补齐的 ragged 批次根本无法变成矩形张量。
  实测：`tok(["hello world", "the quick brown fox jumps"], return_tensors="pt")`
  直接抛 `ValueError`，因为两条序列长度不同。
- 注意异常分支里特意把 `overflowing_tokens` 单独拎出来报错 ——
  溢出 token 天然是变长的，这是设计上的已知例外。

**把三块连起来看，`return_tensors` 的完整影响面是**：

```text
tokenizer(text, padding=True)                  -> input_ids: list[list[int]]
tokenizer(text, padding=True, return_tensors="pt")
        │
        │  ① 编码阶段完全相同（实测逐位相同）
        ▼
   BatchEncoding(batch_outputs, tensor_type="pt")     ← pad() 出口
        │  __init__ 最后一行
        ▼
   convert_to_tensors(tensor_type="pt")
        │  torch.tensor(value, dtype=int64) 逐 key 转换
        ▼
                                                  -> input_ids: Tensor(2, 7) int64
```

---

## 六、参数如何映射到张量：21 个形参 → 1 个字典

`__call__` 有 21 个具名形参（外加 `**kwargs`）。它们不是各走各的路，
而是**其中 17 个先被收进一个字典、再被分发给下游**：

<!-- src: tokenization_utils_base.py -->
```python
    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        text_pair: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        text_target: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        text_pair_target: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        add_special_tokens: bool = True,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool | str | TruncationStrategy | None = None,
        max_length: int | None = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: int | None = None,
        padding_side: str | None = None,
        return_tensors: str | TensorType | None = None,
        return_token_type_ids: bool | None = None,
        return_attention_mask: bool | None = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        tokenizer_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> BatchEncoding:
```
**读法**：
- 前 4 个参数是**输入**（`text` / `text_pair` / `text_target` / `text_pair_target`），
  它们不进字典，而是决定后面走哪条分支。
- 其余 17 个参数是**行为开关**，全部会被塞进 `all_kwargs`（`tokenizer_kwargs` 例外，
  它是"装参数的参数"，在下面被单独合并）。
- `text` 与 `text_target` **至少给一个**，否则抛 `ValueError`；
  两个都给时，`text_target` 的结果会以 `labels` 为键**并进同一个 BatchEncoding**。
- 返回类型标注是 `BatchEncoding` —— 不是 dict、不是 tuple。
  **一次 `__call__` 的产物永远是一个 BatchEncoding**，这是本课最该记住的一句话。

<!-- src: tokenization_utils_base.py -->
```python
        all_kwargs = {
            "add_special_tokens": add_special_tokens,
            "padding": padding,
            "truncation": truncation,
            "max_length": max_length,
            "stride": stride,
            "is_split_into_words": is_split_into_words,
            "pad_to_multiple_of": pad_to_multiple_of,
            "padding_side": padding_side,
            "return_tensors": return_tensors,
            "return_token_type_ids": return_token_type_ids,
            "return_attention_mask": return_attention_mask,
            "return_overflowing_tokens": return_overflowing_tokens,
            "return_special_tokens_mask": return_special_tokens_mask,
            "return_offsets_mapping": return_offsets_mapping,
            "return_length": return_length,
            "split_special_tokens": kwargs.pop("split_special_tokens", self.split_special_tokens),
            "verbose": verbose,
        }
```
**读法**：
- 这个字典就是"参数 → 下游"的**唯一映射表**。`_encode_plus` 收的
  `return_token_type_ids` / `return_attention_mask` / `return_overflowing_tokens`
  这些开关，全部从这张表里来。
- 第 17 行 `"split_special_tokens": kwargs.pop("split_special_tokens", self.split_special_tokens)`
  是唯一带 `.pop` 的项：它**既可以从调用参数来，也可以从分词器属性来**，
  优先级是"调用参数 > 实例属性"。实测 `split_special_tokens` 默认是 `False`。
- 紧随这个字典的三行是**三段式优先级**：先 `all_kwargs.update(tokenizer_kwargs)`，
  再 `all_kwargs.update(kwargs)`，最后才 `pop` 出 `padding` / `truncation` / `max_length`
  送进策略归一化。所以**显式形参在最底层，谁后写谁赢** ——
  `tokenizer_kwargs={"padding": True}` 会被同一次调用里显式的 `padding=False` 盖掉。
- `max_length` 在字典里（第 5 行），但它**不会活着走到 `_encode_plus`**：
  第三节那个 `all_kwargs.pop("max_length", None)` 把它取走、归一化之后再以
  独立的 `max_length=` 形参传下去。**同一个参数，在两条路上有不同的身份。**

---

## 七、把一次 `__call__` 连起来

```text
tokenizer("hello world the quick brown fox", padding=True, return_tensors="pt")
   │
   │ ①  __call__ 收参  ->  all_kwargs（20 个形参 → 1 个字典）
   ▼
_get_padding_truncation_strategies(padding, truncation, max_length, ...)
   │     True -> PaddingStrategy.LONGEST     （别名翻译）
   │     max_length 单独出现 -> 偷偷打开 truncation
   │     model_max_length 是 1e30 -> 策略降级为 DO_NOT_PAD / DO_NOT_TRUNCATE
   │     要 padding 却没有 pad_token -> ValueError
   ▼
(padding_strategy, truncation_strategy, max_length, kwargs)   ← 出口永远是枚举
   │
   │ ②  _encode_plus(text, ..., padding_strategy=..., ...)
   ▼
input_ids / attention_mask（+ 可选 token_type_ids / special_tokens_mask / ...）
   │
   │ ③  _pad：mask 先补满 1，再按 difference 同时补 0 与 pad_id
   ▼
BatchEncoding(batch_outputs, tensor_type=return_tensors)
   │
   │ ④  convert_to_tensors：唯一一次类型转换（batch 轴在这里补）
   ▼
BatchEncoding  input_ids: Tensor(2, 7) int64   attention_mask: Tensor(2, 7) int64
```

**真实的形状演算**（全部来自 `probe_l201.py` 的实测输出，词表里
`[UNK]=0 [PAD]=1 [CLS]=2 [SEP]=3`，且带 `[CLS] $A [SEP]` 后处理器）：

```text
① 单条、不 padding、不转张量
   tok("hello world the quick brown fox")
     input_ids      = [2,4,5,6,7,8,9,3]          list，长度 8（无 batch 轴）
     attention_mask = [1,1,1,1,1,1,1,1]          list，长度 8

② 同一条文本 + return_tensors="pt"
     input_ids      = Tensor(1, 4)   ← 以 "hello world" 为例：batch 轴是补出来的
     对照：return_tensors=None 时是 list [2,4,5,3]（长度 4，没有外层括号）

③ 两条不等长 + padding=True
   tok(["hello world", "the quick brown fox jumps"], padding=True, return_tensors="pt")
     input_ids      = [[2,4,5,3,1,1,1], [2,6,7,8,9,10,3]]      Tensor(2, 7)
     attention_mask = [[1,1,1,1,0,0,0], [1,1,1,1,1,1,1]]        Tensor(2, 7)
                      └─ 短的那条补了 3 个 pad，mask 同位补了 3 个 0

④ 两条 + padding="max_length", max_length=8, truncation=True
     input_ids      = [[2,4,5,3,1,1,1,1], [2,6,7,8,9,10,3,1]]  Tensor(2, 8)
     attention_mask = [[1,1,1,1,0,0,0,0], [1,1,1,1,1,1,1,0]]    Tensor(2, 8)

⑤ padding="max_length" 但分词器没设 model_max_length（默认 1e30）
     input_ids      = [[2,4,5,3], [2,6,7,8,9,10,3]]            Tensor(2, 4)/(2, 7)
                      ← 策略被静默降级，形状根本没对齐，转张量时就会抛错
```

**这就是本层要的形状对齐**：文本 token 与（后面的课里）视觉 token，
最终都要落成同一条 `(B, L)` 的 `input_ids` 加一条同形状的 `attention_mask`。
L2 层后面每一课都在往这条序列里塞不同的东西，但**出口形状的规矩是这一课定的**。

---

## 八、与后面课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| 快分词器与纯 Python 分词器各自的 `_encode_plus` 怎么实现 | L2-02 |
| 快分词器为什么不需要走 `_pad` 就能有 mask | L2-02 |
| 慢→快转换时这些属性怎么映射 | L2-02 |
| 视觉 token 是怎么被拼进同一条 `input_ids` 的 | L2-06 / L2-07 |
| `BatchEncoding` 的 word↔token↔char 映射怎么用 | L2-02（`word_ids` / `offset_mapping`） |
| 张量进了模型之后 `attention_mask` 变成什么 | L4（掩码构造） |
| `model_max_length` 与 RoPE 的位置上限是不是一回事 | L4 / L8 |

**一句话总结**：

> 一次 `tokenizer(text, ...)` = **参数归一化**（`True` 翻译成枚举）→ **编码**（产出 `input_ids`）
> → **padding**（`attention_mask` 与 pad 位由同一个 `difference` 同时造出）
> → **装箱**（`return_tensors` 只在最后这一步决定容器）；
> **`attention_mask` 是 padding 的孪生兄弟，`return_tensors` 是装箱参数 —— 它们都不改变任何一个 token。**
