<!-- glm5-coverage
tokenization_utils_sentencepiece.py
-->

# L2-03 · SentencePiece 分词器 — 源文件

**这一课只回答一个问题：一份 `tokenizer.model` 是怎么变成一串 id 的？**

视角：L2-01 讲的是分词器的**公共契约**（`__call__` 怎么产出 `input_ids` / `attention_mask`），
L2-02 讲的是**快**路径（`tokenizers` 库）。这一课补上第三条腿：**慢**路径里 SentencePiece
模型的共同父类。它只有 315 行，却决定了「同一个字符串，首 token 长什么样」。

| 文件 | 行数 | 本课引用块 |
|---|---|---|
| `tokenization_utils_sentencepiece.py` | 315 | 13 |

> 行数是 `wc -l` 实测。作业书写的是 **316** —— 差 1 是「文件末尾那个换行算不算一行」的
> 口径差异（`splitlines()` 与 `wc -l` 都是 315，`split("\n")` 才会多出一个空尾元素）。
> 作业书给的类/函数大纲行号 **L45 / L60 / L100 / L104 / L110 / L190 / L204 / L223 / L227 /
> L232 / L237 / L266 / L289 / L294 / L301 逐条命中**，只有总行数那一处需要订正。

**本课用到的实测数字**（在仓库自带 venv 里跑 `.tmp/l203/probe_l203.py` 与 `probe2_l203.py`
得到，不是估算；脚本与临时目录**不计入覆盖率**，覆盖域只统计 `src/transformers/**`）：

```text
语料：120 句混合中英文 → 训练两个 220 词表的 sentencepiece 模型（unigram / BPE）
proto 里写死的算法：unigram 的 trainer_spec.model_type = 1，BPE 的 = 2
    —— 同一段 Python 加载两者，分支不在 Python 里
normalizer_spec.add_dummy_prefix：训练产物都是 1；legacy=False 加载后实测变成 0

"Hello world"（同一个 unigram 模型，只换 legacy）
    legacy=True  → ['▁Hello', '▁world']         2 个 token，首 token 带 ▁
    legacy=False → ['H', 'el', 'lo', '▁world']  4 个 token，首 token 是裸 'H'
同一个字符串换 BPE 模型
    legacy=True  → ['▁Hello', '▁world']         2 个 token
    legacy=False → ['H', 'ello', '▁world']      3 个 token

unk_token="<unk>" 在本课模型里被切成 5 个 piece：['<','u','n','k','>'] → unk_token_length = 5
    （docstring 里举的例子是 4 —— 说明这个数必须运行时算，不能写死）
legacy=False 时 sp_model.encode(" Hello")   → ['H','el','lo']   ← 开头的 ▁ 真的会被吃掉
legacy=False 时 encode("<unk> Hello")[5:]   → ['▁Hello']        ← 所以要先拼 unk 再切片

_add_tokens(["▁Hello"])        → 返回 0，len 不变，id 仍是 base vocab 里的 8（命中 in_base_vocab）
_add_tokens(["<|brand_new|>"]) → 返回 1，id = 99；len 99→100，而 vocab_size 仍是 99
piece_to_id("definitely_not_a_piece") = 0 = unk_id，而 IdToPiece(0) = "<unk>"

SentencePieceExtractor.extract() 与 .extract("unigram")：vocab / scores / merges 三者完全相同
    unigram 模型 → vocab 99 条、scores 99 条、merges 21 条
    BPE 模型     → vocab 220 条、merges 161 条

真实 GLM-5 分词器（zai-org/GLM-5.3-Flash 的 tokenizer.json，本地加载实测）
    类名 TokenizersBackend，backend="tokenizers"，is_fast=True，len=154856，vocab_size=154820
    tokenizer.json 的 model.type = "BPE"，merges 321649 条 —— 不是 unigram
    "你好，GLM-5" → 6 个 token → input_ids (1, 6)
一张 448x448 图（该 checkpoint 的 vision_config.image_size = 448）：
    image_grid_thw = [[1, 32, 32]]，pixel_values (1024, 1176)，视觉 token = 1024 / 2^2 = 256
    同一条序列：6 + 258 + 3 = 267 → input_ids (1, 267) → inputs_embeds (1, 267, 4096)
```

> 上面这段是**示意性汇总**，用 `text` 块标出 —— 它不是源文件的逐字引用，不参与保真校验。
> 下面每一个 `python` 块都是从源文件里**按行号直接切出来**的连续片段，逐字未改。

---

## 一、这个文件在 L2 里管什么

先看它对外暴露的两个名字：`SentencePieceBackend` 是**类**，`VOCAB_FILES_NAMES` 是它认的
文件名 —— 默认值 `tokenizer.model`，也就是 sentencepiece 的二进制 proto。

<!-- src: tokenization_utils_sentencepiece.py -->
```python
VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}

SPIECE_UNDERLINE = "▁"


@add_end_docstrings(INIT_TOKENIZER_DOCSTRING)
class SentencePieceBackend(PreTrainedTokenizer):
```

**读法**：
- `VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}`：这个基类只认**一个**文件，
  且必须是 sentencepiece 训练出来的 `.model`（不是 `vocab.txt`，也不是 `tokenizer.json`）。
  子类可以改这个名字 —— `BertGenerationTokenizer` 就把它改成了 `spiece.model`。
- `SPIECE_UNDERLINE = "▁"`（U+2581）是**空格在句内被转义后的样子**。sentencepiece 先把
  空格替换成 ▁ 再做子词切分，所以「词边界」这件事在 token 串里是**看得见**的。
  本课后面所有关于首 token 的讨论，都围绕这个字符展开。
- `@add_end_docstrings(INIT_TOKENIZER_DOCSTRING)` 只是把公共的 `__init__` 参数文档拼上去，
  不改变行为。真正的内容在下面 300 行里。
- 类的父类是 `PreTrainedTokenizer`，而 `tokenization_python.py:1424` 有一行
  `PreTrainedTokenizer = PythonBackend` —— 也就是说这个基类实际上继承的是 **Python 后端**，
  实测 `is_fast = False`、`backend = "sentencepiece"`。
- **同名两实现**：`convert_slow_tokenizer.py:146` 里还有一个**同名**的
  `SentencePieceExtractor`，而 `tokenization_utils_tokenizers.py:308` 导入的是**那一个**，
  不是本文件第 289 行这个。本文件里的这个，给 gemma 系权重转换脚本与 mbart50 / mluke 用。
  这一条在第九节会用实测再确认一次。

---

## 二、加载顺序：先把 `sp_model` 装好，再交给父类

`__init__` 的前半段只做三件事：取出 sentencepiece 专属参数、把 `backend` 补成
`"sentencepiece"`、**在调用父类之前**把模型加载完。

<!-- src: tokenization_utils_sentencepiece.py -->
```python
    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(self, **kwargs):
        # Ensure optional dependency is available before loading
        requires_backends(self, "sentencepiece")

        # Extract sentencepiece-specific parameters
        self.vocab_file = kwargs.get("vocab_file")
        self.legacy = kwargs.get("legacy", True)
        self.sp_model_kwargs = kwargs.pop("sp_model_kwargs", {})
```

**读法**：
- `requires_backends(self, "sentencepiece")` 把「可选依赖没装」变成一个**明确的报错**，
  而不是后面某一行莫名其妙的 `AttributeError` —— `spm` 在文件顶部是
  `try / except ImportError` 导进来的，完全可能是 `None`。
- `self.vocab_file = kwargs.get("vocab_file")` 用的是 `get` 而不是 `pop`：这个键要**留在
  kwargs 里**交给父类，才能进 `init_kwargs`、被 `save_pretrained` 写回
  `tokenizer_config.json`。而 `sp_model_kwargs` 用的是 `pop`，随后（第 91 行）又手动塞回去。
  两个参数处理方式不同，是为了让交给父类的 `kwargs` 既不重复也不丢失。
  实测 `init_kwargs` 的键：`['backend','bos_token','eos_token','legacy','pad_token',
  'sp_model_kwargs','unk_token','vocab_file']`。
- `self.legacy = kwargs.get("legacy", True)`：**默认是 True**。这是第三节的主角。
- `if "backend" not in kwargs: kwargs["backend"] = "sentencepiece"`：v5 用 `backend`
  字段标记「这个 tokenizer 走哪条实现路径」。父类在
  `tokenization_utils_base.py:1094` 把它取出来存成 `self.backend`（实测 `"sentencepiece"`）。
  用户显式传了就不覆盖 —— 这是留给 L2-02 那条快路径的接口。

<!-- src: tokenization_utils_sentencepiece.py -->
```python
        # Load the SentencePiece model before calling parent __init__
        # This is needed because parent __init__ may call methods that depend on sp_model
        tokenizer = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        tokenizer.Load(self.vocab_file)
```

**读法**：
- **顺序不能反**。注释写得很直白：父类 `__init__` 会调用依赖 `sp_model` 的方法，最典型的
  链条是 `_add_tokens` → `len(self)` → `get_vocab()` → `convert_ids_to_tokens` → `sp_model`。
  所以 `tokenizer.Load(...)` 必须在 `super().__init__(**kwargs)` **之前**完成。
- `spm.SentencePieceProcessor(**self.sp_model_kwargs)` 再 `Load(self.vocab_file)`：
  `sp_model_kwargs` 是给 **processor 构造函数**的参数（例如 `enable_sampling`、
  `nbest_size`、`alpha`），不是给训练器的。
- 注意这里没有「按模型类型分支」的代码 —— 见第三节末尾的实测：unigram 与 BPE 是**同一段
  代码**加载的。

<!-- src: tokenization_utils_sentencepiece.py -->
```python
        # Initialize total_vocab_size before parent __init__ (which may call _add_tokens -> len(self))
        self.total_vocab_size = self.sp_model.get_piece_size()

        # Add sp_model_kwargs back to kwargs so it gets stored in init_kwargs
        kwargs["sp_model_kwargs"] = self.sp_model_kwargs
```

**读法**：
- `self.total_vocab_size = self.sp_model.get_piece_size()` 也是「提前」：父类会把它先初始化
  成 `0`，而 `__len__` 的实现是「为 0 就重新算一遍」。这里先填上 base 词表大小，父类在
  初始化过程中调 `len(self)` 时拿到的才是对的数。
- `kwargs["sp_model_kwargs"] = self.sp_model_kwargs`：因为第 67 行把它 `pop` 走了，
  这里补回去，`init_kwargs` 里才有它。
- 之后（第 96-97 行）才是 `super().__init__(**kwargs)` 与 `self._update_trie()`。
  也就是说：**这个类的构造是「自己先备好料，再让父类上桌」**，而不是常见的
  「先 `super()` 再补自己的字段」。

---

## 三、★ legacy 不是运行时分支，是**加载时改写 proto**

这是本课第一个真正的洞察，也是验收点「`add_dummy_prefix` 对首 token 的影响」的入口。

<!-- src: tokenization_utils_sentencepiece.py -->
```python
        if not self.legacy:
            model_pb2 = import_protobuf()
            proto = model_pb2.ModelProto.FromString(tokenizer.serialized_model_proto())
            if proto.normalizer_spec.add_dummy_prefix:
                proto.normalizer_spec.add_dummy_prefix = False
                tokenizer.LoadFromSerializedProto(proto.SerializeToString())

        self.sp_model = tokenizer
```

**读法**：
- `legacy=False` 时做的事只有一件：把 `.model` 反序列化成 `ModelProto`，**把
  `normalizer_spec.add_dummy_prefix` 改成 False**，再 `LoadFromSerializedProto` 装回去。
  换句话说，**它改的不是 Python 代码，是那份模型的规格**。
- 为什么要这么绕？因为 `add_dummy_prefix` 是 sentencepiece **归一化器**的行为，不是分词器
  的行为，它没有运行时开关 —— 只能改 proto 再重新加载。这也解释了为什么必须
  `import_protobuf()`：不解析 proto，就没有第二个地方能碰到这个开关。
- `if proto.normalizer_spec.add_dummy_prefix:` 这个判断保证「本来就是 False 的模型不会
  白白多一次 `LoadFromSerializedProto`」—— 幂等。
- **实测**：本课训练的 unigram / BPE 模型，proto 里 `add_dummy_prefix` 都是 `1`；
  `legacy=False` 加载之后，从 `sp_model.serialized_model_proto()` 里读回来是 `0`。
- `self.sp_model = tokenizer` 之后，这个实例的行为就**永久**变了 —— 同一份 `.model` 文件，
  `legacy=True` 与 `legacy=False` 会加载出两个行为不同的对象。

**它对首 token 到底做了什么**（同一个 unigram 模型，只换 `legacy`）：

```text
输入 "Hello world"
  legacy=True   → ['▁Hello', '▁world']          长度 2，首 token = ▁Hello
  legacy=False  → ['H', 'el', 'lo', '▁world']   长度 4，首 token = H
输入 "Hello"
  legacy=True   → ['▁Hello']        长度 1
  legacy=False  → ['H', 'el', 'lo'] 长度 3
输入 " Hello"（开头真的是空格）
  legacy=True   → ['▁Hello']
  legacy=False  → ['▁Hello']        ← 见第四节：这是被 _tokenize 抢救回来的
```

一句话：**`add_dummy_prefix=True` 时，模型看到的文本前面被偷偷加了一个 ▁**，于是首 token
是一个「带词边界的整词」；关掉之后，首 token 退化成**裸字符** `H`。对上层来说这不是格式
差异，而是 **token 数差异**：`input_ids` 的形状都会跟着变（长度 2 → 4）。

**顺带回答「unigram / BPE 两种模型的加载」**：这个文件里**没有**任何区分两者的分支 ——
算法写在 proto 的 `trainer_spec.model_type` 里（实测 unigram = 1、BPE = 2），
`sentencepiece` 自己在 `Load()` 时读它。所以「两种模型」在这个类里是**同一个代码路径**；
唯一的差别出现在第九节的 `extract()`：unigram 要合成 merges，BPE 本来就有 merges。

---

## 四、`_tokenize`：关掉开关之后，词边界要自己补回来

`legacy=False` 之后，sentencepiece 会把**开头的 ▁ 直接吃掉**。这个类的做法是：先把
`unk_token` 拼在文本前面，切完之后再把 `unk_token` 占掉的那些 piece 切掉。

<!-- src: tokenization_utils_sentencepiece.py -->
```python
    def _tokenize(self, text, **kwargs):
        """
        Returns a tokenized string.

        We de-activated the `add_dummy_prefix` option, thus the sentencepiece internals will always strip any
        SPIECE_UNDERLINE. For example: `self.sp_model.encode(f"{SPIECE_UNDERLINE}Hey", out_type = str)` will give
        `['H', 'e', 'y']` instead of `['▁He', 'y']`. Thus we always encode `f"{unk_token}text"` and strip the
        `unk_token`. Here is an example with `unk_token = "<unk>"` and `unk_token_length = 4`.
        `self.tokenizer.sp_model.encode("<unk> Hey", out_type = str)[4:]`.
        """
        if self.legacy or not text.startswith((SPIECE_UNDERLINE, " ")):
            return self.sp_model.encode(text, out_type=str)

        # 1. Encode string + prefix ex: "<unk> Hey"
        tokens = self.sp_model.encode(self.unk_token + text, out_type=str)
        # 2. Remove self.unk_token from ['<','unk','>', '▁Hey']
        unk_token_length = len(self.sp_model.encode(str(self.unk_token)))
        return tokens[unk_token_length:] if len(tokens) >= unk_token_length else tokens
```

**读法**：
- 第一分支 `if self.legacy or not text.startswith((SPIECE_UNDERLINE, " ")):` —— 只有
  「非 legacy **且** 文本以空格或 ▁ 开头」才走特殊路径。也就是说：**这个技巧只为「首字符
  就是词边界」这一种情况服务**。普通文本（`"Hello"`）在 `legacy=False` 下就是老老实实
  `sp_model.encode(text)`，得到裸 `H`。
- **docstring 里那句断言我实测过**：`legacy=False` 的模型上 `encode(" Hello")` 返回
  `['H','el','lo']` —— 开头的 ▁ 真的没了；`encode("▁Hello")` 同样返回 `['H','el','lo']`。
  所以 docstring 说的「the sentencepiece internals will always strip any SPIECE_UNDERLINE」
  是**真的**，不是想当然。
- 于是第 218 行的 `self.sp_model.encode(self.unk_token + text, out_type=str)` 把 `" Hello"`
  变成 `"<unk> Hello"`：**开头的空格不再是开头**，sentencepiece 就不会去动它。
- 第 220 行的 `unk_token_length` 是**运行时算的**：
  `len(self.sp_model.encode(str(self.unk_token)))`。本课模型实测是 **5**
  （`<unk>` 被切成 `['<','u','n','k','>']`），而 docstring 举的例子是 4 —— 把它写死成 4
  或 1 都会切错位置。这正是它必须现算的原因。
- 第 221 行 `tokens[unk_token_length:] if len(tokens) >= unk_token_length else tokens` 是
  边界保护：万一整个文本被切成比 `unk_token` 还短的序列，宁可原样返回也不越界。
- 实测：`encode("<unk> Hello")` = `['<','u','n','k','>','▁Hello']`，切掉前 5 个正好是
  `['▁Hello']` —— 词边界被救回来了。
- 顺带一提：`_tokenize` **看不到**特殊 token。父类 `PreTrainedTokenizer.tokenize()` 会先用
  trie 把文本按「已加入的 token」切开，只有切剩下的片段才会送进这里（第七节）。

---

## 五、词表的两套口径：`vocab_size` 与 `len()`

<!-- src: tokenization_utils_sentencepiece.py -->
```python
    @property
    def vocab_size(self) -> int:
        """Returns vocab size"""
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        """Returns vocab as a dict"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab
```

**读法**：
- `vocab_size` 是一个 `@property`，直接问 sentencepiece 要 `get_piece_size()` ——
  它只数**模型自带**的 piece，不含任何后加的 token。
- `get_vocab()` 返回**完整**词表：先按 `range(self.vocab_size)` 把 base piece 全部转成
  token，再 `update(self.added_tokens_encoder)` 把后加的 token 并进来。
- 两者差在哪，实测一眼可见：base 99 条；`_add_tokens(["<|brand_new|>"])` 之后
  `len(tok)` 变成 100，而 `tok.vocab_size` **仍然是 99**。
- 父类 `__len__` 返回的是 `total_vocab_size`，也就是 `len(self.get_vocab())`。
  所以 **`len(tokenizer)` 才是 embedding 表该有多大**；`vocab_size` 是「模型原始词表有多大」。
  L2-01 里 `resize_token_embeddings(len(tokenizer))` 用的是前者，原因就在这里。

<!-- src: tokenization_utils_sentencepiece.py -->
```python
        if not new_tokens:
            return 0

        next_index = len(self)  # total size (base + added)
        num_added = 0
```

**读法**：
- `next_index = len(self)` —— 新 token 的编号从「**完整**词表长度」开始排，而不是从
  `vocab_size` 开始。上面那组 `len()` / `vocab_size` 的区分，在这里变成了实际后果：
  如果写成 `self.vocab_size`，第二个新 token 就会盖掉第一个的位置。
- 注释 `# total size (base + added)` 是作者自己留的提醒，和 L2-01 的
  `added_tokens_encoder` 是同一套账。

---

## 六、★ 为什么 `in_base_vocab` 要判两次

这是本课第二个洞察：一行看起来啰嗦的判断，去掉后半句就会**静默出错**。

<!-- src: tokenization_utils_sentencepiece.py -->
```python
            # Check if token already exists in the SentencePiece base vocab
            tok_id = self.sp_model.piece_to_id(token.content)
            in_base_vocab = (
                tok_id < self.sp_model.get_piece_size() and self.sp_model.IdToPiece(tok_id) == token.content
            )

            if in_base_vocab:
                token_index = tok_id
            else:
                token_index = next_index
                next_index += 1
                num_added += 1
```

**读法**：
- `self.sp_model.piece_to_id(token.content)` 对**不存在的 piece** 不会报错，而是返回
  `unk_id`。本课模型实测：`piece_to_id("definitely_not_a_piece")` = `0`，而 `0` 就是
  `<unk>` 的编号，`IdToPiece(0)` = `"<unk>"`。
- 所以 `tok_id < self.sp_model.get_piece_size()` 这一半**几乎永远成立**（0 < 99）。
  真正干活的是后半句 `self.sp_model.IdToPiece(tok_id) == token.content`：把编号换回
  piece 字符串，确认 **round-trip 一致**，才算「这个词真的在 base 词表里」。
- 少了后半句会发生什么：任何新 token 都会被判成「已在词表里」，`token_index = 0` ——
  于是所有新增 token 全部指向 `<unk>`，而 `num_added` 永远是 0。
- 命中时（实测）：`_add_tokens(["▁Hello"])` 返回 **0**，`len` 不变，`▁Hello` 的 id 仍然是
  base 里的 **8** —— 它被登记成 `AddedToken`（为了控制 strip / normalize 行为），但
  **不占新编号**。
- 没命中时：`_add_tokens(["<|brand_new|>"])` 返回 **1**，id = 99（= 当时的 `len(self)`），
  `len` 变成 100。

---

## 七、trie：加进来的 token 怎么才能被「切」出来

<!-- src: tokenization_utils_sentencepiece.py -->
```python
        self._update_trie()
        self._update_total_vocab_size()
        return num_added

    def _update_trie(self, unique_no_split_tokens: list[str] | None = None):
        # Add all added tokens
        for token in self._added_tokens_decoder.values():
            if token.content not in self.tokens_trie._tokens:
                self.tokens_trie.add(token.content)
        # Also add all special tokens (even if they're in base vocab) so they get split during tokenization
        for token in self.all_special_tokens:
            if token not in self.tokens_trie._tokens:
                self.tokens_trie.add(token)
        # Add any additional no-split tokens
        for token in unique_no_split_tokens or []:
            if token not in self.tokens_trie._tokens:
                self.tokens_trie.add(token)
```

**读法**：
- `_add_tokens` 的结尾固定做三件事：重建 trie、更新总词表大小、返回「真正新增了几个」。
  返回值是给调用方做 `resize_token_embeddings` 判据用的。
- `_update_trie` 加三类东西：**所有 added token**、**所有 special token**、以及调用方额外
  指定的 no-split token。
- 中间那个循环旁边的注释是重点：`# Also add all special tokens (even if they're in base
  vocab) so they get split during tokenization` —— 特殊 token 即使本来就在 base 词表里
  （比如 `<unk>`、`</s>`、`<pad>`），也必须进 trie。因为父类 `tokenize()` 的逻辑是
  「**先按 trie 切，再对剩下的片段调 `_tokenize`**」：不进 trie，它们就会被 sentencepiece
  当成普通文本再切一遍，切出别的 piece 来。
- `if token.content not in self.tokens_trie._tokens` 让这个函数**幂等**：`_add_tokens`
  每次都会调它，重复添加不会出问题。
- 实测：加完 `<|brand_new|>` 之后，`tokenize("Hello<|brand_new|>world")` 返回
  `['▁Hello', '<|brand_new|>', '▁world']` —— 新 token 被原样切出来了。
- **一个容易忽略的副作用**：切出来的三段是**分别**送进 `_tokenize` 的，所以第三段
  `"world"` 会重新享受一次 `add_dummy_prefix`，变成 `▁world`（而不是 `world`）。
  换句话说，**在句子里插一个 added token，会改变它后面那一段的词边界**。

---

## 八、取 id、还原字符串，以及 `_decode` 的透传

<!-- src: tokenization_utils_sentencepiece.py -->
```python
    def _convert_token_to_id(self, token):
        """Converts a token (str) to an id using the vocab."""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self.sp_model.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """Converts a sequence of tokens (string) in a single string."""
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string
```

**读法**：
- `_convert_token_to_id` 与 `_convert_id_to_token` 就是 `piece_to_id` / `IdToPiece` 的薄
  封装 —— 分词器内部**只用 id**，字符串只是进出口的表示。
- `convert_tokens_to_string` 是反向拼装：`"".join(tokens)` 之后把 `▁` 换回空格，再
  `.strip()`。注意最后那个 `strip()`：**首尾的空格会被吃掉**，所以
  `decode(encode(x))` 不保证与 `x` 逐字节相同（首尾空白会丢）。
- `_decode` 只做了一件事：**把参数原样转交给父类的通用实现**。

<!-- src: tokenization_utils_sentencepiece.py -->
```python
    def _decode(
        self,
        token_ids: int | list[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None,
        spaces_between_special_tokens: bool = False,
        **kwargs,
    ) -> str:
        """
        Decode token ids to string.

        Uses the generic decode path from PreTrainedTokenizer which works for all vocabularies,
        including custom vocabularies that override _convert_id_to_token.
        """
        # Use parent class's generic decode method - it's simpler and works for all cases
        return super()._decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )
```

**读法**：
- 为什么不直接删掉这个方法？因为它的**签名**多了一个 `spaces_between_special_tokens`，
  这是老版本留下的兼容参数。实测两个签名：
  `SentencePieceBackend._decode(self, token_ids, skip_special_tokens=False,
  clean_up_tokenization_spaces=None, spaces_between_special_tokens=False, **kwargs)`
  对
  `PreTrainedTokenizer._decode(self, token_ids, skip_special_tokens=False,
  clean_up_tokenization_spaces=None, **kwargs)`。
- 关键细节：`spaces_between_special_tokens` **收下了但没有传下去**。实测
  `decode(ids, spaces_between_special_tokens=True)` 与 `=False` 的输出**完全相同**
  （都是 `"Hello world"`）—— 这个参数现在是个摆设。
- docstring 说明了为什么不走 `sp_model.decode()`：「Uses the generic decode path from
  PreTrainedTokenizer which works for all vocabularies, including custom vocabularies that
  override `_convert_id_to_token`」。走通用路径 = **added token 与特殊 token 的跳过逻辑
  由父类统一处理**；直接调 `sp_model.decode()` 会把它们漏掉。
- 这又是「凡文件里写着的，都要跑一遍」的一个例子：签名里有这个参数，不代表它有效。

---

## 九、★ `SentencePieceExtractor`：一个被忽略的参数

文件末尾这个类不是给分词器用的，是给**转换脚本**用的：把 `.model` 里的 piece 与分数读出来，
交给 `generate_merges` 合成 BPE 式的 merges，供 fast tokenizer 使用。

<!-- src: tokenization_utils_sentencepiece.py -->
```python
    def extract(self, vocab_scores=None) -> tuple[dict[str, int], list[tuple[str, float]], list[tuple]]:
        """
        By default will return vocab and merges with respect to their order, by sending `vocab_scores` we're going to
        order the merges with respect to the piece scores instead.
        """
        sp = self.sp
        vocab_ids = {sp.id_to_piece(index): index for index in range(sp.GetPieceSize())}

        vocab_scores_dict = {sp.id_to_piece(i): sp.get_score(i) for i in range(sp.GetPieceSize())}

        merges = generate_merges(vocab_ids, vocab_scores_dict)

        vocab_scores_list = [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.GetPieceSize())]

        return vocab_ids, vocab_scores_list, merges
```

**读法**：
- `__init__` 只做两件事：检查 sentencepiece 后端、`SentencePieceProcessor()` + `Load(model)`。
  注意它**没有** `self.proto` —— 和 `convert_slow_tokenizer.py` 里那个同名类完全不同
  （那个类解析 proto，并且要求一个必填的 `model_type` 参数）。
- `extract()` 的 docstring 承诺：「By default will return vocab and merges with respect to
  their order, by sending `vocab_scores` we're going to order the merges with respect to the
  piece scores instead.」—— 也就是说，传不传 `vocab_scores` 应该有**两种排序**。
- **实测：没有两种。** 函数体里 `vocab_scores` 这个形参从头到尾**没有被引用过一次**；
  第 309 行与第 313 行构造的 `vocab_scores_dict` / `vocab_scores_list` 都是**现算**的
  `sp.get_score(i)`。用同一个模型跑 `extract()` 与 `extract("unigram")`，返回的
  vocab / scores / merges **三者完全相同**（`==` 为 True）。
- 也就是说：**这个参数现在是死的，merges 永远按 piece 分数排序**。docstring 描述的是历史
  行为（旧版实现里确实有 `if vocab_scores is not None` 的分支），v5 删掉了分支却留下了
  文档 —— 这正是 STYLE.md 说的「注释不一定是真的」。
- 返回值三件套：`vocab_ids`（piece → id）、`vocab_scores_list`（(piece, score) 列表）、
  `merges`（(左, 右) 对）。实测规模：unigram 模型 vocab 99 / merges 21；BPE 模型
  vocab 220 / merges 161。
- 第 311 行的 `generate_merges(vocab_ids, vocab_scores_dict)` 在
  `tokenization_utils_base.py:3669`：`reverse = vocab_scores is not None` —— 传了分数表就
  **降序**排。本文件永远传，所以永远是降序。
- 这个类的地位：**它是「sentencepiece 血统 → fast tokenizer」这座桥的一侧**。桥的另一侧
  （`convert_slow_tokenizer.py` 里的同名类）在 L2-02 讲。

---

## 十、和 GLM-5 的关系（实测订正）

作业书的讲解要点里有一条是「为什么 GLM-5 的 tokenizer 往往走这个基类」。**实测结果与这条
相反**，以实测为准：

```text
zai-org/GLM-5.3-Flash/tokenizer_config.json 实测（2026-09 拉取）
    "backend": "tokenizers"
    "tokenizer_class": "TokenizersBackend"
本地加载该 tokenizer 实测
    type(tok).__name__ = TokenizersBackend，is_fast = True，backend = "tokenizers"
    len(tok) = 154856，vocab_size = 154820，added_tokens = 36
    tokenizer.json 的 model.type = "BPE"，merges 321649 条
models/glm5_next/ 下没有任何 tokenization_*.py（实测 ls）
TOKENIZER_MAPPING_NAMES 里 glm 系（glm / glm4 / glm4_moe / glm4v / ...）全部映射到 TokenizersBackend
整个 models/ 目录下直接继承 SentencePieceBackend 的类只有 6 个：
    BertGenerationTokenizer / GPTSw3Tokenizer / SiglipTokenizer /
    SpeechT5Tokenizer / BartphoTokenizer / PLBartTokenizer
    —— LlamaTokenizer 与 GemmaTokenizer 在 v5 里已经改成继承 TokenizersBackend
```

正确的说法是：

1. **GLM-5 自己的 tokenizer 不走这个基类** —— 它是 `TokenizersBackend`（快路径，读
   `tokenizer.json`），那是 L2-02 的内容。
2. 这个基类现在是**少数 sentencepiece 慢分词器的共同父类**（实测只剩 6 个子类）。
3. 两者不是无关：sentencepiece 血统的模型要变成 fast tokenizer，走的正是**第九节那个
   extractor + `generate_merges`** 这条转换链；`LlamaTokenizer` 虽然改了父类，加载
   `.model` 时仍然要靠 `SentencePieceExtractor` 把 proto 翻译成 tokenizers 的 vocab/merges。
   所以准确的说法是：**这个文件是「sentencepiece 血统怎么进入 v5」的那道门，而不是
   GLM-5 运行时会经过的那段代码。**

---

## 十一、与后面课的接口

**回顾 L0-01**：本课产出的那串 `input_ids`，就是那一课 `forward` 的第一行输入 ——
`embed_tokens` 把它变成 `(batch, seq, 4096)`，之后才是 4 条残差流与 45 层主干。
本课负责的是它**上游**那一步：字符串 → id。**回顾 L2-01**：`__call__` 把本课的 token
序列打包成 `input_ids` / `attention_mask`，`resize_token_embeddings` 用的是
`len(tokenizer)` 而不是 `vocab_size`。

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| `__call__` 怎么把 token 变成 `input_ids` / `attention_mask` | L2-01 |
| `backend="tokenizers"` 那条快路径怎么工作 | L2-02 |
| `convert_slow_tokenizer` 里那个**同名**的 extractor | L2-02 |
| 448×448 图怎么变成 256 个视觉 token | L2-05 / L2-07 |
| 视觉占位符怎么被替换成向量、和文本拼在一条序列上 | L2-06 |
| `inputs_embeds` 之后的 45 层主干 | L0-01 / L3 |

**一句话总结**：

> 这个基类只做三件事：**在父类之前把 `sp_model` 装好**（顺序不能反）、**用 `legacy` 决定
> 要不要改写 proto 里的 `add_dummy_prefix`**（首 token 因此差一个 ▁、序列长度因此差 1）、
> **把词表边界（base piece 与 added token）算清楚**。
> 而它末尾那个 `SentencePieceExtractor` 提醒我们：**签名里有的参数，不一定真的有用。**
