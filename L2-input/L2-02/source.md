<!-- glm5-coverage
tokenization_utils_tokenizers.py
tokenization_python.py
convert_slow_tokenizer.py
-->

# L2-02 · 快分词器、纯 Python 分词器与慢→快转换 — 源文件

**这一课只回答一个问题：同一句「你好，世界」，从字符串变成 `(batch, seq)` 的整数张量，中间到底走了哪条路？**

视角：**文本进 → 整数 id 出**。三个文件是这条流水线的三种实现：Rust 后端的 fast 分词器
（`tokenization_utils_tokenizers.py`）、纯 Python 的 slow 分词器（`tokenization_python.py`）、
以及把 SentencePiece 模型「翻译」成 Rust 组件的转换器（`convert_slow_tokenizer.py`）。
下一层（L3 起）拿到的 `input_ids`，就是这三个文件里某一个交出来的。

| 文件 | 行数（实测 `wc -l`） | 本课引用块 |
|---|---|---|
| `tokenization_utils_tokenizers.py` | 1514 | 5 |
| `tokenization_python.py` | 1424 | 4 |
| `convert_slow_tokenizer.py` | 2095 | 5 |

> **实测订正**：作业书给的行数是 1515 / 1425 / 2096，`wc -l` 实测 1514 / 1424 / 2095。
> 三个文件都差 1 行，以 `wc -l` 为准 —— 本课全部行号按实测文件编号。

**引用块 ↔ 动画的对应**：第一、二节 → 第 2 幕；第三节 → 第 3、4 幕；第四节 → 第 5、6 幕；
第五节 → 第 8 幕；第六节 → 第 7 幕。`lesson.js` 的每一幕 `code` 都取自下面这些块（逐字）。

**GLM-5.3-Flash 的 tokenizer 实测底账**（`_data/recon/probe_l202.py` 跑出，不是估算）：

```text
checkpoint 的 tokenizer 目录里只有三个文件：
    tokenizer.json          20,217,442 B   ← 唯一的一份词表，Rust 序列化
    tokenizer_config.json   tokenizer_class = "TokenizersBackend"  backend = "tokenizers"
                            padding_side = "left"  model_max_length = 1048576
    processor_config.json   patch_size = 14  merge_size = 2  temporal_patch_size = 2

TokenizersBackend.from_pretrained() 实测（1.06 s）：
    is_fast   = True          backend model = BPE
    vocab_size = 154820       基础词表（模型侧 Glm5NextTextConfig.vocab_size = 154880，多 24 个 embedding 槽位）
    len(tok)   = 154856       含 36 个 added token
    pad = eos  = "<|endoftext|>" = 154820        bos = None   unk = None
    154854 = "<|image|>"        154855 = "<|video|>"
    154830 = "<|begin_of_image|>"   154831 = "<|end_of_image|>"
    can_save_slow_tokenizer = False        hasattr(tok, "vocab_file") = False
    PythonBackend.from_pretrained(同一个目录)  ->  NotImplementedError
```

> 上面两段都是**实测汇总**，用 `text` 块标出 —— 不是源文件的逐字引用，不参与保真校验。
> 下面每一段 `python` 块都是从上游源文件**逐字切片**的引用，位置连续。

---

## 一、★ 洞察一：GLM-5 的 fast 不是「转换」出来的，是「直接读」出来的

`TokenizersBackend.convert_to_native_format` 是加载期的**唯一分叉点**：它按磁盘上有什么文件，
决定后端从哪里来。第一个分支最短，也最容易被忽略 —— 而 GLM-5 走的正是它。

<!-- src: tokenization_utils_tokenizers.py -->
```python
    @classmethod
    def convert_to_native_format(cls, trust_remote_code=False, **kwargs):
        """
        Build a `tokenizers.Tokenizer` backend from the available serialization files (tokenizer.json, sentencepiece
        models, tekken.json, vocab/merges).
        """
        # Preserve kwargs for possible downstream use
        local_kwargs = dict(kwargs)
        fast_tokenizer_file = local_kwargs.pop("tokenizer_file", None)

        if (
            fast_tokenizer_file is not None
            and os.path.isfile(fast_tokenizer_file)
            and (cls is TokenizersBackend or "__init__" not in cls.__dict__ or trust_remote_code)
        ):
            local_kwargs["tokenizer_object"] = TokenizerFast.from_file(fast_tokenizer_file)
            return local_kwargs
```

**读法**：
- 只要 `tokenizers` 解析出来的三个条件同时成立，后端就是**一行 `from_file`** ——
  不再解析词表、不再重算 merges、不再需要 `sentencepiece` 或 `tiktoken`。这就是实测里
  「1.06 s 加载完 20 MB 的 tokenizer.json」的原因。
- 三个条件缺一不可：`tokenizer_file` 是**路径**且这个文件存在；并且当前类要么就是
  `TokenizersBackend` 本身、要么没有自定义 `__init__`。第三条是为**自定义 `__init__` 的类**留的
  —— 它们要拿到 `vocab`/`merges` 自己重建后端（下一段 `elif` 就是给它们用的）。
- ★ **对 GLM-5 的结论**：checkpoint 里只有 `tokenizer.json`，所以 `_convert_from_sentencepiece`
  根本没被调用过。`convert_slow_tokenizer` 虽然在模块顶层被 import 了一次
  （第 38 行 `from .convert_slow_tokenizer import SpmConverter`），但**真正依赖 sentencepiece /
  protobuf 的那两个名字是在句子函数体里才 import 的**（第 308、316 行）——
  也就是说，走这条 fast 路径时 `SentencePieceExtractor` 与 `SLOW_TO_FAST_CONVERTERS` 都没被加载。
  实测里 `can_save_slow_tokenizer = False`、
  `hasattr(tok, "vocab_file") = False`，也印证了「这份 checkpoint 没有慢分词器的原料」。
- 所以本课第四、五、六节讲的东西，不是 GLM-5 的日常路径，而是**它的退路与对照物** ——
  但正是这条退路定义了「fast / slow」这两个词在 transformers 里的含义。
- 跨课呼应：`tokenizer.json` 这份文件什么时候产生、`save_pretrained` 怎么写它，见 L2-01 的公共契约。

---

## 二、三个身份：一个基类、两个后端、56 个转换器

先认出「fast 分词器」这个类长什么样。它的类属性直接写明了它认哪几种序列化文件。

<!-- src: tokenization_utils_tokenizers.py -->
```python
@add_end_docstrings(INIT_TOKENIZER_DOCSTRING)
class TokenizersBackend(PreTrainedTokenizerBase):
    """
    Base class for all fast tokenizers (wrapping HuggingFace tokenizers library).

    Inherits from [`~tokenization_utils_base.PreTrainedTokenizerBase`].

    Handles all the shared methods for tokenization and special tokens, as well as methods for
    downloading/caching/loading pretrained tokenizers, as well as adding tokens to the vocabulary.

    This class also contains the added tokens in a unified way on top of all tokenizers so we don't have to handle the
    specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece...).
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model = None
    _tokenizer = None
```

**读法**：
- `vocab_files_names` 指向 `VOCAB_FILES_NAMES = {"tokenizer_file": "tokenizer.json", "vocab_file": "tokenizer.model"}`
  —— 一个类**同时声明了两种序列化格式的槽位**：Rust 的 `tokenizer.json` 和 SentencePiece 的 `.model`。
  磁盘上有哪个就用哪个，两个都有则 `tokenizer.json` 优先（上一节的第一个分支）。
- `model = None` / `_tokenizer = None` 是**给子类覆盖的锚点**：子类填 `model = WordPiece`、
  或者预置一个 `_tokenizer`，`__init__` 里的分支就按它们走。
- 「fast 分词器」的公开名字是别名：文件最后一行 `PreTrainedTokenizerFast = TokenizersBackend`；
  对称地，slow 那边最后一行是 `PreTrainedTokenizer = PythonBackend`。**两个词指的是后端实现，
  不是两个功能不同的分词器** —— 这是本课最容易被误解的一点。
- 实测旁证（`grep -rlE '^class .*\(.*TokenizersBackend.*\)' models/*/tokenization*.py`）：
  本仓库有 **50** 个模型分词器文件定义了 `TokenizersBackend` 子类；同时仍有 **32** 个文件定义了
  `PreTrainedTokenizer`（= `PythonBackend` 的别名）子类，多数是 `*_legacy` 或自研规则的分词器；
  而按名字提到 `PythonBackend` 的模型侧文件只有 `models/auto/tokenization_auto.py` 一处决策代码。
  v5 的 `AutoTokenizer` 甚至直接把 `use_fast` 参数丢掉（该文件第 749–750 行），**默认只有 fast 一条路**，
  慢分词器要显式指名某个 legacy 类才会被用到。

---

## 三、★ 洞察二：批量编码的循环在 Rust 里，形状也在那里定

`_encode_plus` 是整个 fast 路径的中枢：单条和批量走同一个函数，区别只在 `is_batched`。
下面是它把活交给 Rust 的那三行 —— 本课最短、也最贵的一段引用。

<!-- src: tokenization_utils_tokenizers.py -->
```python
        # Direct rust backend call
        encodings = self._tokenizer.encode_batch(
            batch_text_or_text_pairs,
            add_special_tokens=add_special_tokens,
            is_pretokenized=is_split_into_words,
        )
```

**读法**：
- 注意函数名：**`encode_batch`，不是 `encode`**。哪怕你只传了一条文本，`_encode_plus`
  也会先把它包成 `[(text, text_pair)]` 这样的单元素 batch，再走批量入口。
  「单条 = 长度为 1 的批量」是这条路上唯一的形状规则。
- `add_special_tokens` 与 `is_pretokenized` 直接透传给 Rust —— Python 侧**没有**任何逐条的分词逻辑，
  也没有逐字符的循环。所以 Python 侧的开销与序列长度无关。
- 实测：3 条不等长文本（1 / 4 / 40 个 token）一次调用返回 `input_ids (3, 40)`；
  256 条一次调用 4 ms。同一个后端上分开调 256 次则要 7 ms —— 差的就是 Python 侧的调用开销。
- 对照下一节：慢路径在这里是**一个 Python `for` 循环**。

然后 `encodings` 被翻译成 `BatchEncoding`，形状在这里定型（也是单条被「降维」的地方）：

<!-- src: tokenization_utils_tokenizers.py -->
```python
        batched_output = BatchEncoding(sanitized_tokens, sanitized_encodings, tensor_type=return_tensors)

        # If single input, remove the batch dimension (unless returning overflowing tokens)
        if not is_batched and return_tensors is None and not return_overflowing_tokens:
            batched_output = BatchEncoding(
                {
                    key: (value[0] if len(value) > 0 and isinstance(value[0], list) else value)
                    for key, value in batched_output.items()
                },
                batched_output.encodings,
            )

        return batched_output
```

**读法**：
- `return_tensors` 不是在编码时生效的，而是在**构造 `BatchEncoding` 时**生效 ——
  它决定这份「dict of list」要不要转成 `pt` / `tf` / `np` 张量。所以「张量化」是编码之后的独立一步。
- 降维条件写得很精确：`not is_batched`（单条）**且** `return_tensors is None`（不张量化）
  **且** 没有 overflow。三条同时成立才把 `(1, L)` 压成 `(L,)`。
  于是**单条 + `return_tensors="pt"` 仍然是 `(1, L)`** —— 这一点在写代码时最容易踩。
- 实测的形状账（GLM-5.3-Flash 真 tokenizer）：

```text
输入                                                            返回                     形状
------------------------------------------------------------------------------------------------
tok("你好，世界！GLM-5 是一个混合架构模型。")                    list[int]                (14,)
tok([同一句])                                                    list[list[int]]          (1, 14) 个 id，但还不是张量
tok(同一句, return_tensors="pt")                                 Tensor                   (1, 14)   ← 不降维
tok(3 条不等长, padding=True, return_tensors="pt")                input_ids / attention_mask  (3, 40)，每行真实长度 [1, 4, 40]
tok(同上, padding="max_length", max_length=20, truncation=True, pt)                         (3, 20)
                                                                 第 0 行 attention_mask 前 19 位是 0  ← 左侧填充
tok(3 条, padding=True, return_offsets_mapping=True, pt)         offset_mapping           (3, 40, 2)
```

> 上表是**实测输出**（`text` 块，不参与保真校验）。`padding_side = "left"` 来自 GLM-5 的
> `tokenizer_config.json`，填充用的 id 就是 154820 `<|endoftext|>`（它同时是 `pad_token` 和 `eos_token`）。

---

## 四、慢路径：纯 Python 的三件套（Trie、逐条循环、统一 pad）

`tokenization_python.py` 是「没有 `tokenizers` 库也要能分词」的那条路。它由三个零件组成。

### 4.1 零件一：Trie —— 先把不该被切碎的 token 摘出来

<!-- src: tokenization_python.py -->
```python
class Trie:
    """
    Trie in Python. Creates a Trie out of a list of words. The trie is used to split on `added_tokens` in one pass
    Loose reference https://en.wikipedia.org/wiki/Trie
    """

    def __init__(self, *args):
        self.data = {}
        self._tokens = set()
        self._termination_char = ""
        self.update(*args)
```

**读法**：
- `data` 是嵌套 dict（`{"H": {"e": {...}}}`），`_termination_char = ""` 是**终止标记**：
  某个节点的 dict 里有 `""` 键，就说明从根到这里是一个完整 token。
- 「一次扫描」是关键：慢路径不允许多次 `replace`（那会破坏已插入的内容），
  所以它用 Trie 在**单遍**里找到所有 added token 的边界。这与引擎里语法高亮的
  「单遍 tokenizer」是同一条工程纪律（见 `shared/engine.js` 的注释）。
- 慢路径为什么要它：`<|image|>`、`[gMASK]` 这类 token 必须**整体**成为一个 id，
  绝不能被后面的 BPE/WordPiece 再切。fast 那边靠 Rust 内部的 added-token 表做同一件事。

它到底怎么切，docstring 里的例子就是规格说明书：

<!-- src: tokenization_python.py -->
```python
    def split(self, text: str) -> list[str]:
        """
        Will look for the words added to the trie within `text`. Output is the original string split along the
        boundaries of the words found.

        This trie will match the longest possible word first !

        Example:

        ```python
        >>> trie = Trie()
        >>> trie.split("[CLS] This is a extra_id_100")
        ["[CLS] This is a extra_id_100"]

        >>> trie.add("[CLS]")
        >>> trie.add("extra_id_1")
        >>> trie.add("extra_id_100")
        >>> trie.split("[CLS] This is a extra_id_100")
        ["[CLS]", " This is a ", "extra_id_100"]
        ```
```

**读法**：
- 「longest possible word first」是这套算法唯一的难点：`extra_id_1` 与 `extra_id_100` 同时在表里时，
  必须吃满更长的那一个。实现里为此保留了 `states`（多个起点）与 `skip`（lookahead）。
- 实测（把 GLM-5 的三个控制 token 装进 Trie）：

```text
Trie.add("[gMASK]"), Trie.add("<sop>"), Trie.add("<|image|>")
Trie.split("前缀[gMASK]<sop>中间<|image|>后缀[gMASK]")
  -> ['前缀', '[gMASK]', '<sop>', '中间', '<|image|>', '后缀', '[gMASK]']

ExtensionsTrie（Trie 的子类）还可以按前缀枚举：
ExtensionsTrie.extensions("app")  ->  ['app', 'apple', 'application']
```

### 4.2 零件二：批量 = Python 里的 for 循环

<!-- src: tokenization_python.py -->
```python
        if is_batched:
            if text_pair is not None:
                if not isinstance(text_pair, (list, tuple)) or len(text_pair) != len(text):
                    raise ValueError("If `text` is a batch, `text_pair` must also be a batch of the same length.")
            pairs = text_pair if text_pair is not None else [None] * len(text)

            batch_outputs = {}
            for current_text, current_pair in zip(text, pairs):
                # Handle tuples/lists as sequence pairs like ("text1", "text2")
                # For is_split_into_words=True, only tuples are treated as pairs; lists are single pretokenized sequences
```

**读法**：
- 慢路径**也支持批量**，但批量的循环体在 Python 里。`batch_outputs` 是
  `dict[key] -> list`（key 先出现哪个就先建哪个），形状是**后拼**出来的，不是一开始就定的。
- 这里也是 fast / slow 校验差异的来源：慢路径必须自己检查 `text_pair` 的长度，
  fast 那条路把这些校验交给 Rust（`encode_batch` 会抛错）。

循环体里最值得注意的是「补 pad 不在这里做」：

<!-- src: tokenization_python.py -->
```python
                current_output = self._encode_plus(
                    text=current_text,
                    text_pair=current_pair,
                    add_special_tokens=add_special_tokens,
                    padding_strategy=PaddingStrategy.DO_NOT_PAD,  # we pad in batch afterward
                    truncation_strategy=truncation_strategy,
                    max_length=max_length,
                    stride=stride,
                    is_split_into_words=is_split_into_words,
                    pad_to_multiple_of=None,  # we pad in batch afterward
                    padding_side=None,  # we pad in batch afterward
                    return_tensors=None,  # We convert the whole batch to tensors at the end
                    return_token_type_ids=return_token_type_ids,
                    return_attention_mask=False,  # we pad in batch afterward
                    return_overflowing_tokens=return_overflowing_tokens,
                    return_special_tokens_mask=return_special_tokens_mask,
                    return_length=return_length,
                    verbose=verbose,
                    **kwargs,
                )
                for key, value in current_output.items():
                    batch_outputs.setdefault(key, []).append(value)
```

**读法**：
- 这是一个**递归调用**：批量分支对每一条又调回 `_encode_plus`，只是这次传的是单条文本。
  三个 `# we pad in batch afterward` 注释把「逐条只负责切词，对齐留到最后」写死了 ——
  这就是为什么慢路径的 `(B, L)` 一定比 fast 晚一步成形。
- `return_tensors=None` + `return_attention_mask=False`：循环里不张量化、不建 mask，
  全部留给循环之后的 `self.pad(...)` 与 `BatchEncoding(..., tensor_type=return_tensors)`。
- 代价账（读代码即可推，不需要 benchmark）：慢路径的 Python 调用次数 ∝ 批量条数；
  fast 路径的 Python 调用次数 = 1。这就是 256 条实测 4 ms vs 7 ms 的差来源，
  而这个差距会随「自定义 `_tokenize` 的复杂度」进一步放大。

### 4.3 零件三：为什么还要保留它

慢路径不是历史包袱，它有三个当下就成立的用途：

| 用途 | 依据 |
|---|---|
| **没有 `tokenizers` 二进制也能跑** | 纯 Python 实现，只依赖 `unicodedata` / `bisect` |
| **可读可改**：`_tokenize` 是子类唯一要实现的方法 | `PythonBackend._tokenize` 直接 `raise NotImplementedError` |
| **fast 的等价性基准** | 源码里明写 `# very important for fast and slow equivalence!` |

---

## 五、★ 洞察三：视觉 token 与文本 token 靠 **id 的位置**对齐

L2 这一层的问题是「文本 token 与视觉 token 如何在同一条序列上对齐」。答案在这一节：
**对齐不靠长度约定，靠同一个 id 出现的次数与位置。**

先看两边怎么变成「一个 id」的。SentencePiece 的 proto 里，type 3/4 的 piece 是控制符与用户自定义符，
`extract()` 把它们**原样**变成 `AddedToken`：

<!-- src: convert_slow_tokenizer.py -->
```python
    def extract(self, model_type, **kwargs) -> tuple[dict[str, int], list[tuple]]:
        """
        By default will return vocab and merges with respect to their order, by sending `vocab_scores` we're going to
        order the merges with respect to the piece scores instead.
        """
        self.proto.trainer_spec.unk_id
        if model_type is None:
            from tokenizers.models import BPE, Unigram

            model_type = Unigram if self.proto.trainer_spec.model_type == 1 else BPE
        vocab = [(piece.piece, piece.score) for piece in self.proto.pieces]

        if model_type.__name__ != "BPE":
            kwargs["unk_id"] = self.proto.trainer_spec.unk_id
            kwargs["vocab"] = vocab
        else:
            from .tokenization_utils_base import generate_merges

            vocab = {word: i for i, (word, score) in enumerate(vocab)}
            merges = generate_merges(vocab)
            kwargs["vocab"] = vocab
            kwargs["merges"] = merges

        # control tokens are special
        # user defined symbols are not
        # both user and control tokens are AddedTokens
        # Add user defined symbols (type == 4) from sentencepiece (https://github.com/google/sentencepiece/blob/6225e08edb2577757163b3f5dbba4c0b670ef445/src/sentencepiece_model.proto#L299C29-L299C33)
        spm_added_tokens = [(id, p.piece, p.type == 3) for id, p in enumerate(self.proto.pieces) if p.type in [3, 4]]
        kwargs["additional_special_tokens"] = [
            AddedToken(token, normalized=False, special=special)
            for id, token, special in sorted(spm_added_tokens, key=lambda x: x[0])
        ]
        kwargs["_spm_precompiled_charsmap"] = getattr(self.proto.normalizer_spec, "precompiled_charsmap", None)
        return kwargs
```

**读法**：
- `normalized=False` 是这一段的灵魂：它告诉后端「不要拿 normalizer 去动这个 token」，
  于是 `<|image|>` 无论出现在文本的哪个位置，都稳定地映射到**同一个 id**。
- `model_type` 为 `None` 时的推断（`trainer_spec.model_type == 1` → Unigram，否则 BPE）说明
  「同一个 `.model` 文件可以装两种算法」，这正是 L2-03 要展开的 unigram / BPE 之分。
- 慢路径里做同一件事的是第四节的 Trie；fast 路径里是 Rust 内部的 added-token 表。
  **三条路都在保证同一个不变量：added token 不被切碎。**

现在把两边接起来。实测（GLM-5 真 tokenizer + 真图像处理器）：

```text
① 图像侧：Glm5NextImageProcessor(patch_size=14, merge_size=2)
     448x448  -> image_grid_thw = [[1, 32, 32]]   pixel_values (1024, 1176)
     视觉 token 数 = 1*32*32 / merge_size^2 = 1024 / 4 = 256
       672x336 -> [[1, 24, 48]]  -> 1152 / 4 = 288
       224x224 -> [[1, 16, 16]]  ->  256 / 4 =  64

② 文本侧：只负责放同样多个占位 id
     "<|begin_of_image|>" + "<|image|>" * 256 + "<|end_of_image|>" + "你好，描述这张图。"
     = [154830] + [154854] * 256 + [154831] + 6 个文本 token
     -> 总长 264；视觉槽位 = 下标 [1..256]，共 256 个
     （换成 224x224 的图：64 个占位 -> 总长 72）

③ 两条样本一起 batch（一条带图、一条纯文本）：
     input_ids (2, 262)      attention_mask (2, 262)      每行长度 [262, 4]
     左侧补 154820（padding_side = "left"）
     inputs_embeds (2, 262, 4096) 里：视觉槽 256 个、文本槽 268 个

④ 对齐规则：模型只看 input_ids == image_token_id(154854) 的位置，
     把这一张图切出来的 256 个视觉向量按顺序填进去
```

**读法**：
- 这就是 L2 这一层的「形状变换」全貌：**图像分辨率的自由度被 `merge_size²` 除掉了**
  （448×448 的 1024 个 patch → 256 个视觉 token），而序列长度 = 视觉占位 + 文本 token + 控制符。
- 视觉 token 数不是 tokenizer 决定的，是**图像处理器**决定的；tokenizer 只保证
  「放 N 个占位 id」这件事是廉价的（一个 id 重复 N 次，不占词表）。
- 因此 shape 的两个数（`L`、`4096`）来源不同：`L` 由②的拼接决定，`4096` 由文本塔的 `hidden_size`
  决定 —— 两者在 `inputs_embeds` 里合流，之后主干再也分不出哪一列是图。展开见 L2-06 / L2-07。
- 反过来说：**如果占位 id 被切碎（比如 `<|image|>` 被 BPE 切成 5 个 piece），整个对齐就崩了。**
  这就是第四节 Trie 与上面 `added_tokens` 存在的真正理由 —— 它们不是优化，是正确性前提。

---

## 六、慢→快转换：把 SentencePiece 模型「翻译」成 Rust 组件

这一节是作业书点名的第三个问题。入口在 fast 类里，但真正的活在 `convert_slow_tokenizer.py`。

<!-- src: tokenization_utils_tokenizers.py -->
```python
    @classmethod
    def _convert_from_sentencepiece(cls, vocab_file: str, local_kwargs: dict[str, Any]) -> dict[str, Any]:
        from .convert_slow_tokenizer import SentencePieceExtractor

        # 1. Extract vocab, merges, and spm_precompiled from the .model proto
        extractor = SentencePieceExtractor(vocab_file)
        local_kwargs = extractor.extract(cls.model, **local_kwargs)

        # 2. If a model-specific converter exists, use it.
        try:
            from .convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS

            converter_class = SLOW_TO_FAST_CONVERTERS.get(cls.__name__)
            if converter_class is not None and hasattr(converter_class, "convert_from_spm"):
                local_kwargs = converter_class.convert_from_spm(**local_kwargs)
        except Exception as e:
            logger.warning(
                f"Could not reorder vocab using converter for {cls.__name__} due to {e}. Falling back to raw SentencePiece extraction."
            )
        if hasattr(cls, "convert_from_spm_model"):
            local_kwargs = cls.convert_from_spm_model(**local_kwargs)
```

**读法**：
- 三步走：① 从 `.model`（protobuf）里**抽**出 vocab / merges / charsmap；
  ② 若该类名在 `SLOW_TO_FAST_CONVERTERS` 里有专属转换器，让它**重排**词表（Gemma、Llama 等
  对前几个 id 的处理各不相同）；③ 没有模型专属类时，直接按 proto 组装一个后端。
- 这里的 `try/except Exception` 是把「重排失败」降级成「用原始抽取结果」，而不是让加载失败 ——
  词表顺序错一点会让 id 漂移，但至少还能跑。这是**刻意的容错**，也让这类 bug 更隐蔽。
- 注意 `SentencePieceExtractor` 是**在函数体里 import** 的：SentencePiece 与 protobuf 都是可选依赖，
  只有真的要用这条路径时才要求它们在场。GLM-5 走不到这里（洞察一），所以它对 GLM-5 的安装环境
  完全没有要求。

抽取器本身只做一件事：把 `.model` 读成 proto，并**立刻**把 vocabulary 摊平：

<!-- src: convert_slow_tokenizer.py -->
```python
class SentencePieceExtractor:
    """
    Extractor implementation for SentencePiece trained models. https://github.com/google/sentencepiece
    """

    def __init__(self, model: str):
        requires_backends(self, "sentencepiece")
        requires_backends(self, "protobuf")

        # from .utils import sentencepiece_model_pb2 as model_pb2
        model_pb2 = import_protobuf()

        m = model_pb2.ModelProto()
        with open(model, "rb") as f:
            m.ParseFromString(f.read())
        self.proto = m
```

**读法**：
- `requires_backends` 在**构造期**就把「缺依赖」变成一条清楚的报错，而不是等到用的时候
  `AttributeError`。这是可选依赖的标准姿势。
- `import_protobuf()` 优先用 `sentencepiece` 自带的 pb2，其次用 transformers 内置的两份之一
  （按 `google.protobuf` 版本分流，见第 95–109 行）—— 目的只有一个：**让 protobuf 成为可选项**。
- `self.proto = m` 把整个 proto 留在实例上：后面的 `vocab()` / `unk_id()` / `normalizer()` 都要读它。

真正「翻译」的组装函数长这样（Unigram 与 BPE 两条支路）：

<!-- src: convert_slow_tokenizer.py -->
```python
        byte_fallback = proto.trainer_spec.byte_fallback
        unk_piece = proto.trainer_spec.unk_piece
        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap

        # model
        if isinstance(vocab, dict):
            tokenizer = Tokenizer(
                BPE(
                    vocab=vocab,
                    merges=merges or [],
                    unk_token=unk_piece,
                    fuse_unk=True,
                    byte_fallback=byte_fallback,
                    dropout=None,
                )
            )
        elif isinstance(vocab, list) and vocab and isinstance(vocab[0], tuple | list):
            tokenizer = Tokenizer(
                Unigram(
                    vocab=vocab,
                    unk_id=proto.trainer_spec.unk_id,
                    byte_fallback=byte_fallback,
                )
            )
        else:
            return None
```

**读法**：
- 分支只看 `vocab` 的**数据结构**：`dict`（word → index）走 BPE，`list[tuple]`（piece, score）走 Unigram。
  「用哪个算法」不是在这里决定的，而是被上游 `extract()` 的产物决定的 —— 类型即契约。
- 三个 proto 字段被直接翻译成 Rust 侧的概念：`byte_fallback` → 字节回退、
  `unk_piece`/`unk_id` → 未知词、`precompiled_charsmap` → 预编译归一化表。
- `fuse_unk=True`：连续的未知字符合并成一个 UNK，这是 SentencePiece 的原始行为，
  「翻译」必须与原文一致，否则 token 序列会变。

组装完模型，还要把**前后处理**接上：

<!-- src: convert_slow_tokenizer.py -->
```python
        # normalizer
        _normalizers = [normalizers.Replace(" ", "▁")]
        if precompiled_charsmap:
            _normalizers.insert(0, normalizers.Precompiled(precompiled_charsmap))
        tokenizer.normalizer = normalizers.Sequence(_normalizers)

        # decoder
        if byte_fallback:
            tokenizer.decoder = decoders.Sequence(
                [decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse()]
            )
        else:
            tokenizer.decoder = decoders.Sequence([decoders.Replace("▁", " ")])

        return tokenizer
```

**读法**：
- **实测订正（很重要）**：这个「组装」函数**只装 `normalizer` 与 `decoder`，不装 `pre_tokenizer`**。
  实测 `SpmConverter.build_tokenizer_from_spm_proto(...)` 的返回对象是
  `normalizer=Sequence, pre_tokenizer=NoneType, decoder=Sequence`。
  「空格 → ▁」在这里是由 **normalizer 的 `Replace`** 完成的，不是 Metaspace 预分词器。
  完整的转换路径上，`pre_tokenizer` 由 `SpmConverter.pre_tokenizer()`（第 785–787 行，
  `pre_tokenizers.Metaspace`）在 `converted()` 里补上 —— 见下一段。
- 「对称」是这里的门道：编码侧 `" " → "▁"`，解码侧 `"▁" → " "`；
  开了 byte fallback 再加 `ByteFallback` 与 `Fuse`。**三个组件必须成对设计**，
  少一个就会出现「编码出来的东西解不回去」。
- 实测这个最小组装体的往返：`encode("▁你好▁世界") -> [1, 2, 3]`（tokens `['▁你', '好', '▁世界']`），
  `decode([1, 2, 3]) -> ' 你好 世界'`。5 个 piece 装进去，`get_vocab_size()` 就是 5。

而模型专属转换器的**收敛点**是 `converted()`：它把上面这些零件拼成一个可用后端。

<!-- src: convert_slow_tokenizer.py -->
```python
    def converted(self) -> Tokenizer:
        tokenizer = self.tokenizer(self.proto)

        # Tokenizer assemble
        normalizer = self.normalizer(self.proto)
        if normalizer is not None:
            tokenizer.normalizer = normalizer

        replacement = "▁"
        add_prefix_space = True
        if hasattr(self.original_tokenizer, "add_prefix_space"):
            add_prefix_space = self.original_tokenizer.add_prefix_space

        pre_tokenizer = self.pre_tokenizer(replacement, add_prefix_space)
        if pre_tokenizer is not None:
            tokenizer.pre_tokenizer = pre_tokenizer

        tokenizer.decoder = self.decoder(replacement, add_prefix_space)
        post_processor = self.post_processor()
        if post_processor:
            tokenizer.post_processor = post_processor

        return tokenizer
```

**读法**：
- 四个钩子 —— `tokenizer` / `normalizer` / `pre_tokenizer` / `decoder`（外加可选的 `post_processor`）
  —— 就是「模型专属转换器」的全部自由度。子类通常只覆写其中一两件：
  `LlamaConverter` 覆写 `vocab`/`normalizer`/`pre_tokenizer`/`decoder` 四件来处理 `legacy` 开关，
  `GemmaConverter` 覆写 `normalizer`/`pre_tokenizer`/`vocab`/`unk_id`。
- `add_prefix_space` 决定 `_get_prepend_scheme()` 返回 `"always"` / `"first"` / `"never"` ——
  这是「首 token 前面有没有 ▁」的唯一开关，L2-03 会用实测展开它。
- 分发只按**类名**查表（第 2071–2074 行）：`SLOW_TO_FAST_CONVERTERS[transformer_tokenizer.__class__.__name__]`。
  实测这张表有 **56 条**，复用最多的是 `BertConverter`（12 个类共用）、`RobertaConverter`（6）、
  `LlamaConverter`（3）。查不到就落到 tiktoken / tekken 分支。
- **两份 `generate_merges`**：`convert_slow_tokenizer` 里那份（第 122 行）要求显式传 `vocab_scores`，
  `tokenization_utils_base` 里那份（第 3669 行）默认按词表序。实测同一份小词表的输出顺序不同：

```text
generate_merges(按词表序)      -> [('a','b'), ('b','c'), ('a','bc'), ('ab','c')]
generate_merges(按 piece 分数) -> [('a','b'), ('b','c'), ('ab','c'), ('a','bc')]
```

> 这份「转换器」对 GLM-5 是**冷路径**：它的 checkpoint 里没有 `.model` 文件。
> 但当你要把一个只有 `tokenizer.model` 的自研/旧模型接进 fast 后端时，这条路径是唯一的门。

---

## 七、实测数字汇总（本课全部数字的来源）

| 量 | 实测值 | 怎么测的 |
|---|---|---|
| fast 加载耗时 | 1.06 s | `TokenizersBackend.from_pretrained(本地目录)` |
| `vocab_size` / `len(tok)` | 154820 / 154856 | 基础词表 / 含 added token |
| 后端模型 | BPE | `type(tok.backend_tokenizer.model).__name__` |
| pipeline | normalizer=None, pre_tokenizer=Sequence, decoder=ByteLevel | 同上 |
| pad = eos | `<|endoftext|>` = 154820 | `tok.pad_token_id` / `eos_token_id` |
| 图像占位 id | 154854 = `<|image|>` | `convert_ids_to_tokens(154854)` |
| 3 条批量 | `(3, 40)`，行实际长度 `[1, 4, 40]` | `padding=True, return_tensors="pt"` |
| 截断到 20 | `(3, 20)`，第 0 行 mask 前 19 位为 0 | `max_length=20, truncation=True` |
| 256 条批量 | 4 ms（分批调 7 ms） | 同一进程 wall clock |
| 448×448 图 | `grid_thw [[1,32,32]]` → 256 个视觉 token | `Glm5NextImageProcessor` |
| 带图序列 | 总长 264，视觉槽位 `[1..256]` | 真 tokenizer 编码占位串 |
| 两条 batch | `(2, 262)`；`inputs_embeds (2, 262, 4096)` | 视觉槽 256 / 文本槽 268 |
| 转换器表 | 56 条 | `len(SLOW_TO_FAST_CONVERTERS)` |
| 慢后端可用性 | `PythonBackend.from_pretrained` → `NotImplementedError` | 同一个 GLM-5 目录 |

---

## 八、与后面课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| `__call__` 的参数怎么变成张量、特殊 token 的属性体系 | L2-01 |
| `.model` 里的 unigram / BPE 之分、`add_dummy_prefix` | L2-03 |
| 图像处理器的尺寸策略与归一化 | L2-04 |
| `image_grid_thw` 与视觉塔 | L2-06 / L2-07 |
| padding / truncation 的完整策略枚举 | L2-01 |
| `attention_mask` 之后怎么变成掩码 | L4-06（掩码构造） |
| `inputs_embeds` 进了主干以后发生什么 | L0-01、L3 |

**一句话总结**：

> GLM-5 的文本→id 只有一条路：`tokenizer.json` 直接喂给 Rust 后端，
> 一次 `encode_batch` 就把 `L` 与 `B` 定死；纯 Python 后端与慢→快转换器是**退路与对照物**，
> 它们存在的意义是「没有 `tokenizers` 二进制也能跑」以及「只有 `.model` 的模型也有门进」。
> 而视觉 token 与文本 token 的对齐，靠的是**同一个占位 id 出现的位置**，不是任何长度约定。
