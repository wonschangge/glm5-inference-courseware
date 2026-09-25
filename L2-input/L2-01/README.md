# L2-01 · 分词器公共契约：慢与快共享的那一层

> 层：**第 2 层 · 输入流水线** ｜ 优先级：P0 ｜ 前置课：`L1-07`

## 学习目标

看完这一课，你应该能：

1. **说出 `input_ids` 与 `attention_mask` 的产生时机**（对应验收点 1）：两者都在 `__call__` 的第三段产生，
   且 `attention_mask` 由 `_pad` **先补满 1、再按同一个 `difference` 补 0**，长度天然等于 `input_ids`；
2. **解释 `return_tensors` 影响的是哪一步**（验收点 2）：它不影响编码，只决定
   `BatchEncoding` 被构造时用哪个容器 —— 唯一一次类型转换发生在 `convert_to_tensors` 里；
3. 说出 `padding` / `truncation` 的**完整枚举**，以及 `True` / `False` 为什么是别名而不是策略值。

## 覆盖的源文件（1 个）

| 文件 | 行数 | 引用块 |
|---|---|---|
| `tokenization_utils_base.py` | 3690 | 14 |

> **行数订正**：作业书写 3691 行。实测 `wc -l` 与 `awk 'END{print NR}'` 都给 **3690**；
> 3691 是 `open(f).read().split("\n")` 的结果（文件以换行结尾，会多切出一个空串）。
>
> 本课另引用了一份**实测探针** `_data/recon/probe_l201.py`（用 `tokenizers` 库手工构造
> `PreTrainedTokenizerFast`，零网络），以及 `utils/generic.py` 的 `PaddingStrategy` 定义。
> **这些都不计入覆盖率** —— 覆盖域只统计 `src/transformers/**/*.py`，且本课声明只含
> `tokenization_utils_base.py` 一项。

## 场景（9 幕）

1. **四段流水线** —— 收参 → 归一化 → 编码+padding → 装箱，以及决定一切的三个类属性
2. **7 个名字** —— `SPECIAL_TOKENS_ATTRIBUTES` 的全名单与实测值；★ 订正：`SpecialTokensMixin` 已不存在
3. **token 与 id 是同一份数据的两个视图** —— `__getattr__` / `__setattr__` 的对称实现
4. **★ 4 种截断在这里，3 种填充不在这里** —— 两个枚举、两处定义，以及成对输入上的实测差别
5. **`True` 不是策略值，是别名** —— 归一化的完整映射表（含 `padding=1` 抛错）
6. **`max_length` 的两种静默失效** —— 默认 `model_max_length` 会让「补到最大长度」退化成「什么都不做」
7. **★ `attention_mask` 与 `input_ids` 同生同长** —— `_pad` 里同一个 `difference` 决定两行
8. **★ `return_tensors` 影响的是装箱** —— `BatchEncoding.__init__` 的最后一行是唯一转换点
9. **收束** —— 参数 → 段 → 后果的完整映射表 + 两个验收点 + 练习

## 核心结论

### 1. `attention_mask` 是 padding 的孪生兄弟

`_pad` 先用 `[1] * len(required_input)` 造出 mask，再由 padding 分支按同一个 `difference` 追加 0：

<!-- src: tokenization_utils_base.py -->
```python
        # Initialize attention mask if not present.
        if return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * len(required_input)
```

而"要不要造"这件事，由类属性 `model_input_names` 决定
（`return_attention_mask = "attention_mask" in self.model_input_names`）。
实测 `{"input_ids": [4,5,6]}` → `max_length=5` 的结果是
`input_ids [4,5,6,1,1]` / `attention_mask [1,1,1,0,0]`。

### 2. ★ `return_tensors` 只决定"装进什么"

<!-- src: tokenization_utils_base.py -->
```python
        self.convert_to_tensors(tensor_type=tensor_type, prepend_batch_axis=prepend_batch_axis)
```

这是 `BatchEncoding.__init__` 的最后一行，也是全流程里唯一一次类型转换。
实测同一批数据：`return_tensors=None` 给 `list`，`"pt"` 给 `Tensor(2, 7) int64`，
**数值逐位相同**；单条文本的 batch 轴也是在这一步补出来的（`(4,)` → `(1, 4)`）。

### 3. ★ `padding` / `truncation` 的出口永远是枚举

<!-- src: tokenization_utils_base.py -->
```python
                padding_strategy = PaddingStrategy.LONGEST  # Default to pad to the longest sequence in the batch
```

`True` 是 `LONGEST` 的别名，`False` 是与它平级的 `DO_NOT_PAD`；
`truncation=True` 落到 `LONGEST_FIRST`，`False` / `None` 落到 `DO_NOT_TRUNCATE`。
实测 `padding=True` 与 `padding="longest"` 输出逐位相同，而 `padding=1` 直接抛
`ValueError: 1 is not a valid PaddingStrategy, ...` —— 因为它用的是 `is` 比较。

### 实测数字

| 量 | 实测值 | 含义 |
|---|---|---|
| `TruncationStrategy` | 4 个值 | `only_first` / `only_second` / `longest_first` / `do_not_truncate` |
| `PaddingStrategy` | 3 个值 | `longest` / `max_length` / `do_not_pad`（定义在 `utils/generic.py`） |
| `SPECIAL_TOKENS_ATTRIBUTES` | 7 个名字 | 顺序即 `special_tokens_map` 的遍历顺序 |
| 默认 `model_max_length` | `1000000000000000019884624838656` | `int(1e30)`；`> LARGE_INTEGER` 即视为"没设过" |
| 两条 (4, 7) + `padding=True` | `(2, 7)` | 短的那条补 3 个 pad / 3 个 0 |
| 同上 + `return_tensors="pt"` | `Tensor(2, 7) int64` | 数值不变，容器变了 |

## 与后续课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| 快分词器与纯 Python 分词器各自的 `_encode_plus` 怎么实现 | L2-02 |
| 快分词器不走 `_pad` 为什么也有 `attention_mask` | L2-02 |
| 慢→快转换时这些 token 属性怎么映射 | L2-02 |
| 视觉 token 怎么被拼进同一条 `input_ids` | L2-06 / L2-07 |
| `BatchEncoding` 的 `word_ids` / `offset_mapping` 怎么用 | L2-02 |
| 张量进模型后 `attention_mask` 变成什么 | L4（掩码构造） |
| `model_max_length` 与位置上限是不是一回事 | L4 / L8 |

## 验收点

- [x] 保真门禁：14 个引用块全部逐字来自 `tokenization_utils_base.py`，且位置连续
- [x] 覆盖度门禁：声明 1 个源文件（`tokenization_utils_base.py`），与 `tools/plan.py` 一致
- [x] 参数门禁：无非法参数（本课不涉及命令行参数）
- [x] 语法检查：`node --check` 通过，9 幕字段完整
- [x] 渲染门禁：无 JS 错误、无布局溢出、交互可用（两种分辨率）
- [x] 实测复核：9 幕的 `code` 字段与源文件同 `codeStart` 起的切片逐行相同
- 自检 1：能说出 `input_ids` 与 `attention_mask` 的产生时机（第三段，同一个 `difference`）
- 自检 2：能解释 `return_tensors` 影响的是哪一步（第四段装箱，`convert_to_tensors`）

**一句话总结**：

> 一次 `tokenizer(text, ...)` = **参数归一化**（`True` 翻译成枚举）→ **编码**（产出 `input_ids`）
> → **padding**（`attention_mask` 与 pad 位由同一个 `difference` 同时造出）
> → **装箱**（`return_tensors` 只在最后这一步决定容器）；
> **`attention_mask` 是 padding 的孪生兄弟，`return_tensors` 是装箱参数 —— 它们都不改变任何一个 token。**
