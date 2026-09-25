# L2-02 · 快分词器、纯 Python 分词器与慢→快转换

> 层：**第 2 层 · 输入流水线**（文本和像素如何变成同一串位置上的向量） ｜ 优先级：P0 ｜ 前置课：L2-01

## 学习目标

看完这一课，你应该能：

1. **说出一条文本在 GLM-5 上变成 id 的完整路径**，并解释它为什么落在 fast 那一支（验收点）；
2. **说出 fast / slow 在 GLM-5 上的三处行为差异**：能不能加载、循环在哪、给不给 offsets（验收点）；
3. 解释「视觉 token 与文本 token 在同一条序列上对齐」到底靠什么，并把 `(2, 262)` 这类形状算出来。

## 覆盖的源文件（3 个）

| 文件 | 行数 | 引用块 |
|---|---|---|
| `tokenization_utils_tokenizers.py` | 1514 | 5 |
| `tokenization_python.py` | 1424 | 4 |
| `convert_slow_tokenizer.py` | 2095 | 5 |

> **实测订正**：作业书给的行数是 1515 / 1425 / 2096；`wc -l` 实测 1514 / 1424 / 2095。
> 本课全部行号按实测文件编号。
>
> 本课引用了 `_data/recon/probe_l202.py` 的**实测输出**（写在 `source.md` / `lesson.js` 的
> `text` 块与卡片里）。**它是脚本，不计入覆盖率** —— 覆盖域只统计 `src/transformers/**/*.py`。
> 测的是真实的 `zai-org/GLM-5.3-Flash` tokenizer 与 `Glm5NextImageProcessor`。

## 场景（9 幕）

1. **三个文件，三条路** —— fast / slow / 转换器的分工 + 四个实测数（154820 / 154856 / 154854 / 1048576）
2. **★ 走的是最短那一支** —— `convert_to_native_format` 的四个出口，GLM-5 命中第一个
3. **一次 `encode_batch`，B 和 L 就都定了** —— 三条不等长文本 → `(3, 40)`，左侧补 154820
4. **单条为什么会被降维** —— `(14,)` / `(1, 14)` / `(3, 40)` 三种形状的判据（含练习）
5. **纯 Python 的第一件：Trie 单遍切分** —— 实测把控制 token 逐个切出来
6. **慢路径的批量 = Python 里的 for 循环** —— 逐条递归 + 「pad 留到最后」
7. **★ 把 SentencePiece 翻译成 Rust 组件** —— `normalizer` / `decoder` / ★`pre_tokenizer = None`（实测订正）
8. **★ 视觉 token 与文本 token 靠 id 的位置对齐** —— 448×448 → 256 个视觉 token、总长 264（含练习）
9. **收束** —— 三条路对照表 + 实测数字 + GLM-5 为什么加载不了慢分词器（含练习）

## 核心结论

### 1. ★ GLM-5 的 fast 不是「转换」出来的，是「直接读」出来的

`convert_to_native_format` 的第一个分支只有一行：

<!-- src: tokenization_utils_tokenizers.py -->
```python
        if (
            fast_tokenizer_file is not None
            and os.path.isfile(fast_tokenizer_file)
            and (cls is TokenizersBackend or "__init__" not in cls.__dict__ or trust_remote_code)
        ):
            local_kwargs["tokenizer_object"] = TokenizerFast.from_file(fast_tokenizer_file)
            return local_kwargs
```

GLM-5.3-Flash 的 checkpoint 里**只有 `tokenizer.json`**（没有 `tokenizer.model`、没有 `vocab.json`），
所以：`can_save_slow_tokenizer = False`，`PythonBackend.from_pretrained(同一个目录)` 直接抛
`NotImplementedError`。本课讲的慢→快转换，对 GLM-5 是**冷路径**。

### 2. ★ 形状由一次 Rust 调用决定

<!-- src: tokenization_utils_tokenizers.py -->
```python
        # Direct rust backend call
        encodings = self._tokenizer.encode_batch(
            batch_text_or_text_pairs,
            add_special_tokens=add_special_tokens,
            is_pretokenized=is_split_into_words,
        )
```

单条也是「长度为 1 的批量」。Python 侧没有任何逐条逻辑 —— 慢路径在这里是一个 `for` 循环
（`tokenization_python.py` 的批量分支），这就是两条路最本质的差别。

### 3. ★ 对齐靠的是占位 id 出现的位置

<!-- src: convert_slow_tokenizer.py -->
```python
        spm_added_tokens = [(id, p.piece, p.type == 3) for id, p in enumerate(self.proto.pieces) if p.type in [3, 4]]
        kwargs["additional_special_tokens"] = [
            AddedToken(token, normalized=False, special=special)
            for id, token, special in sorted(spm_added_tokens, key=lambda x: x[0])
        ]
```

`normalized=False` 让 `<|image|>` 这类占位符**永远是一个 id**。于是图像侧算出来的 256 个视觉 token，
在文本侧就体现为 `input_ids` 上 256 个 154854 —— 模型按这个 id 的位置把视觉向量填进去。
**对齐不靠长度约定，靠 id 的位置**；而保证 id 不被切碎的，正是这里的 `AddedToken` 与慢路径的 Trie。

### 实测数字

| 量 | 实测值 | 含义 |
|---|---|---|
| `vocab_size` / `len(tok)` | 154820 / 154856 | 基础词表 / 含 36 个 added token |
| 后端模型 | BPE | `type(tok.backend_tokenizer.model).__name__` |
| pad = eos | 154820 `<|endoftext|>` | `padding_side = "left"` |
| `image_token_id` | 154854 `<|image|>` | 占位 id；video 是 154855 |
| 3 条批量 | `(3, 40)`，行长度 `[1, 4, 40]` | `padding=True, return_tensors="pt"` |
| 448×448 的图 | 256 个视觉 token | `1*32*32 / merge_size² = 1024/4` |
| 带图序列 | 总长 264，视觉槽位 `[1..256]` | `1 + 256 + 1 + 6` |
| 两条样本 batch | `(2, 262)` → `inputs_embeds (2, 262, 4096)` | 视觉槽 256 / 文本槽 268 |
| 转换器表 | 56 条 | `len(SLOW_TO_FAST_CONVERTERS)` |

## fast / slow 在 GLM-5 上的行为差异（验收点）

| 维度 | fast（`TokenizersBackend`） | slow（`PythonBackend`） |
|---|---|---|
| GLM-5 能否加载 | **能**（`tokenizer.json` 直达，1.06 s） | **不能**（`NotImplementedError`：目录里没有词表文件） |
| 循环在哪 | Rust（一次 `encode_batch`） | Python（逐条递归 `_encode_plus`） |
| `is_fast` | `True` | `False` |
| offsets / 逐字符账本 | 有（`return_offsets_mapping`） | 没有（同一个参数返回 `None`） |
| padding 时机 | Rust 内部一次做完 | 循环之后统一 `self.pad(...)` |
| 形状成形时点 | 一次调用后即 `(B, L_max)` | 逐条 dict 之后再拼 |
| 存在意义 | GLM-5 的日常路径 | 没有 `tokenizers` 二进制时的退路 + 可读可改的等价性基准 |

## 与后续课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| `SpecialTokensMixin`、`__call__` 的参数体系 | L2-01 |
| `.model` 里的 unigram / BPE、`add_dummy_prefix` | L2-03 |
| 图像处理器的尺寸策略与归一化 | L2-04 |
| `image_grid_thw` 怎么变成视觉向量 | L2-06 / L2-07 |
| `attention_mask` 之后怎么变成掩码 | L4-06 |
| `inputs_embeds` 进了主干以后发生什么 | L0-01、L3 |

## 验收点

- [x] 保真门禁：14 个引用块全部逐字来自 3 个源文件，且位置连续
- [x] 覆盖度门禁：3 个源文件被声明（与 `tools/plan.py` 的 L2-02 一致）
- [x] 参数门禁：无非法参数引用
- [x] 语法检查：`node --check` 通过，9 幕字段完整
- [x] 渲染门禁：无 JS 错误、无布局溢出、交互可用（1280×720 + 1920×1080）
- 自检：能说出「fast/slow 在 GLM-5 上的行为差异点」（见上表），
  并能把「448×448 的图 → 264 个位置」这一步算给同伴听

**一句话总结**：

> GLM-5 的文本→id 只有一条路：`tokenizer.json` 直接喂给 Rust 后端，一次 `encode_batch` 定下 `B` 与 `L`；
> 纯 Python 后端与慢→快转换器是退路与对照物 —— 前者保证没有 `tokenizers` 二进制也能跑，后者是只有
> `.model` 的模型唯一的门。而视觉 token 与文本 token 的对齐，靠的是**同一个占位 id 出现的位置**。
