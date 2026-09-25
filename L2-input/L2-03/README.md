# L2-03 · SentencePiece 分词器

> 层：**第 2 层 · 输入流水线** ｜ 优先级：P1 ｜ 前置课：L2-02

## 学习目标

看完这一课，你应该能：

1. **解释 `add_dummy_prefix` 对首 token 的影响**：关掉它之后，首 token 从 `▁Hello`
   退化成裸 `H`，序列长度从 2 变成 4（对应本课验收点）；
2. 说清 `legacy` 开关为什么是「加载时改写 proto」而不是「运行时分支」，以及
   `_tokenize` 为什么要用 `unk_token` 前缀把词边界抢回来；
3. 说出 `vocab_size` 与 `len(tokenizer)` 的口径差异，以及 `in_base_vocab` 为什么要判两次。

## 覆盖的源文件（1 个）

| 文件 | 行数 | 引用块 |
|---|---|---|
| `tokenization_utils_sentencepiece.py` | 315 | 13 |

> 行数为 `wc -l` 实测（作业书写 316，差 1 是末尾换行的口径差异；作业书给的
> 类/函数大纲行号 L45 / L60 / L100 / … / L301 **逐条命中**）。
> 本课的实测数字来自仓库自带 venv 里的 `transformers 5.18.0.dev0`，脚本在
> `.tmp/l203/probe_l203.py` 与 `probe2_l203.py`（**脚本与临时目录不计入覆盖率**，
> 覆盖域只统计 `src/transformers/**`）。为跑通 `legacy=False` 那条路，venv 里补装了
> `sentencepiece 0.2.2` 与 `protobuf`（`import_protobuf()` 需要）。

## 场景（9 幕）

1. **全景** —— 主链路 `tokenizer.model → ids → inputs_embeds`，以及实测订正：
   GLM-5 的 tokenizer 是 `TokenizersBackend`，**不走**这个基类
2. **构造顺序** —— 六个动作，`Load` 必须在 `super().__init__` 之前
3. **★ legacy 改的是 proto** —— 反序列化 → 改一个布尔 → 装回去
4. **`_tokenize` 的两个分支** —— `unk_token` 前缀技巧与运行时算出的 `unk_token_length`
5. **★ 首 token 实测** —— `▁Hello` 对 `H`，token 数 2 对 4（验收点）
6. **词表两套口径** —— `vocab_size` 99 对 `len(tokenizer)` 100
7. **★ `in_base_vocab` 判两次** —— 未知 piece 会伪装成 `unk_id = 0`
8. **形状演算** —— 文本 6 + 图像 258 + 文本 3 = 267 个位置，`inputs_embeds (1, 267, 4096)`
9. **收束** —— 六个部件总表 + `extract(vocab_scores)` 这个死参数 + 练习

## 核心结论

### 1. ★ legacy 不是运行时分支，是加载时改写 proto

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

`add_dummy_prefix` 属于 sentencepiece 的**归一化器**，没有运行时开关 —— 只能把模型
反序列化、改掉这个布尔、再 `LoadFromSerializedProto` 装回去。实测：训练产物里它是 `1`，
`legacy=False` 加载后读回来是 `0`；同一个字符串的首 token 因此从 `▁Hello` 变成 `H`。

### 2. ★ 首 token 的差异是「token 数」的差异

| 输入 | `legacy=True` | `legacy=False` | 长度 |
|---|---|---|---|
| `"Hello world"` | `['▁Hello', '▁world']` | `['H', 'el', 'lo', '▁world']` | 2 → 4 |
| `"Hello"` | `['▁Hello']` | `['H', 'el', 'lo']` | 1 → 3 |
| `" Hello"` | `['▁Hello']` | `['▁Hello']` | 1 → 1 |
| BPE 模型 `"Hello world"` | `['▁Hello', '▁world']` | `['H', 'ello', '▁world']` | 2 → 3 |

第三行说明分岔只在**首字符不是词边界**时发生：开头真的是空格时，`_tokenize` 会用
`unk_token` 前缀把词边界抢回来。

<!-- src: tokenization_utils_sentencepiece.py -->
```python
        # 1. Encode string + prefix ex: "<unk> Hey"
        tokens = self.sp_model.encode(self.unk_token + text, out_type=str)
        # 2. Remove self.unk_token from ['<','unk','>', '▁Hey']
        unk_token_length = len(self.sp_model.encode(str(self.unk_token)))
        return tokens[unk_token_length:] if len(tokens) >= unk_token_length else tokens
```

### 3. ★ `in_base_vocab` 的两个条件缺一不可

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

`piece_to_id` 对不存在的 piece 返回 `unk_id`（实测 = `0`），而 `0 < get_piece_size()`
永远成立 —— 所以前半句挡不住任何东西。真正干活的是 `IdToPiece(tok_id) == token.content`
这个 **round-trip 检查**。去掉它，所有新增 token 都会指向 `<unk>`，`num_added` 永远是 0。

### 4. `vocab_size` 与 `len(tokenizer)` 是两套账

`vocab_size` = `sp_model.get_piece_size()`（实测 99，只数模型自带的 piece）；
`len(tokenizer)` = `len(get_vocab())` = base + added（加 1 个 token 后 99 → 100）。
第 143 行的 `next_index = len(self)` 用的是后者，L2-01 的
`resize_token_embeddings(len(tokenizer))` 也是后者。

### 5. ★ `SentencePieceExtractor.extract(vocab_scores)` 的参数是死的

<!-- src: tokenization_utils_sentencepiece.py -->
```python
        sp = self.sp
        vocab_ids = {sp.id_to_piece(index): index for index in range(sp.GetPieceSize())}

        vocab_scores_dict = {sp.id_to_piece(i): sp.get_score(i) for i in range(sp.GetPieceSize())}

        merges = generate_merges(vocab_ids, vocab_scores_dict)

        vocab_scores_list = [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.GetPieceSize())]
```

docstring 承诺「传了 `vocab_scores` 就按 piece 分数排序」，但函数体里**从未引用**这个形参：
`vocab_scores_dict` 是现算的 `sp.get_score(i)`。实测 `extract()` 与 `extract("unigram")`
返回的 vocab / scores / merges **完全相同**。文档描述的是历史行为，v5 删了分支留了文档。

### 6. 实测订正：GLM-5 不走这个基类

`zai-org/GLM-5.3-Flash/tokenizer_config.json` 实测写的是
`"backend": "tokenizers"` / `"tokenizer_class": "TokenizersBackend"`，本地加载得到
`TokenizersBackend`（`is_fast=True`，`len=154856`，`tokenizer.json` 的 `model.type = "BPE"`）。
`models/glm5_next/` 下没有任何 `tokenization_*.py`。整个 `models/` 目录里直接继承
`SentencePieceBackend` 的类只有 **6** 个（BertGeneration / GPTSw3 / Siglip / SpeechT5 /
Bartpho / PLBart）—— Llama 与 Gemma 在 v5 里已经改成继承 `TokenizersBackend`。
准确的说法是：**这个文件是「sentencepiece 血统怎么进入 v5」的那道门，不是 GLM-5 运行时
会经过的代码。**

## 与后续课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| `__call__` 怎么把 token 变成 `input_ids` / `attention_mask` | L2-01 |
| `backend="tokenizers"` 那条快路径怎么工作 | L2-02 |
| `convert_slow_tokenizer` 里那个**同名**的 extractor | L2-02 |
| 448×448 图怎么变成 256 个视觉 token | L2-05 / L2-07 |
| 视觉占位符怎么被替换成向量、和文本拼在一条序列上 | L2-06 |
| `inputs_embeds` 之后的 45 层主干 | L0-01 / L3 |

## 验收点

- [x] 能解释 `add_dummy_prefix` 对首 token 的影响（第 5 幕 + 本文件「核心结论 2」的实测表）
- [x] 保真门禁：13 个引用块全部逐字来自 `tokenization_utils_sentencepiece.py`
- [x] 覆盖度门禁：1 个源文件被声明（属覆盖域 231 个文件之一）
- [x] 参数门禁：无非法参数
- [x] 语法检查：`node --check` 通过，9 幕字段完整
- [x] 渲染门禁：无 JS 错误、无布局溢出、交互可用（两种分辨率）
- 自检：能对着第 8 幕的比例条，说出 6 / 258 / 3 各自对应什么，以及 `256` 是怎么算出来的

**一句话总结**：

> 这个基类只做三件事：**在父类之前把 `sp_model` 装好**（顺序不能反）、**用 `legacy`
> 决定要不要改写 proto 里的 `add_dummy_prefix`**（首 token 因此差一个 ▁、序列长度因此
> 差 1）、**把词表边界算清楚**。而它末尾那个 `SentencePieceExtractor` 提醒我们：
> **签名里有的参数，不一定真的有用。**
