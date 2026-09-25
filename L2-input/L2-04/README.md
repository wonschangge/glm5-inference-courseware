# L2-04 · 图像处理基座：尺寸、归一化与后端

> 层：**第 2 层 · 输入流水线** ｜ 优先级：P0 ｜ 前置课：`L2-03`

## 学习目标

看完这一课，你应该能：

1. **按顺序复述 `BaseImageProcessor.preprocess` 的四步编排**，并说出「哪一步之后代码就交给后端了」；
2. **说出默认后端的选择顺序**（本课验收点）：显式 `backend=` → `use_fast` → 强制 PIL 名单 → `torchvision`/`pil` 兜底；
3. **手算一张 `336×336` 的 RGB 图经过 patchify + merge 之后的视觉 token 数**，并说清序列长度由谁决定。

## 覆盖的源文件（3 个）

| 文件 | 行数（实测） | 引用块 |
|---|---|---|
| `image_processing_utils.py` | 697 | 4 |
| `image_processing_base.py` | 510 | 3 |
| `image_processing_backends.py` | 667 | 7 |

> **★ 行数订正**：作业书声明的是 511 / 698 / 668，比实测各多 1 行。
> `wc -l` 与 `read` 工具的口径都是 510 / 697 / 667 —— 三个文件都以换行符结尾，
> 作业书的数字等于 `len(text.split("\n"))`，即"换行符个数 + 1"。
> 本课 `lesson.js` 里所有 `codeStart` 都以实测口径为准。

> **★ 后端数量订正**：作业书写的是「三个后端（PIL / torchvision / numpy）」，
> 但 `image_processing_backends.py` 里只有 **2 个**后端类：
> `TorchvisionBackend`（`backend == "torchvision"`）与 `PilBackend`（`backend == "pil"`）；
> `hasattr(module, "NumpyBackend") == False`。
> numpy 是 `PilBackend` 的**底层实现**（它的 `resize/rescale/normalize/center_crop`
> 全部转调 `image_transforms` 里的 numpy 函数），不是第三个可选后端。
> 详情见 `source.md` 的「实测订正（2）」。

> 本课引用了 `_data/recon/probe_l204.py` 的**实测输出**（写在 `source.md` 的 `text` 块里）。
> **它是脚本，不计入覆盖率。** 第九节引用的 `models/auto/image_processing_auto.py`
> 与 `models/auto/auto_mappings.py` 同样**不计入覆盖率** —— 覆盖域只统计本课声明的 3 个文件。

## 场景（9 幕）

1. **一张图怎么变成一串向量** —— 主链路 + 四个关键数字（14 块引用 / 2 个后端 / 144 token / 1176 维）
2. **★ preprocess 的四步编排** —— 校验 → 补默认值 → 标准化尺寸 → 委派；契约到此结束
3. **size 的四种写法归一** —— `224` / `(336,336)` / 字典 / 非法组合，以及 list→tuple 的原因
4. **★ 后端是类级选择** —— 两条继承链、默认值住在类属性上、`_set_attributes` 故意不上提的代价
5. **rescale 与 normalize 的融合** —— `(x·f−μ)/σ ≡ (x−μ/f)/(σ/f)`，以及三个容易看漏的细节
6. **★ 336×336 → 144 个视觉 token** —— `(576, 1176)`、`image_grid_thw (1,24,24)`、`merge_size²`
7. **从一个 JSON 到一个处理器对象** —— `resolve_revision` 只调一次、kwargs 进得去出得来
8. **同一个契约，两种写法** —— 逐张循环 vs 按形状分组批处理；五步顺序为什么不能换
9. **收束** —— 默认后端选择顺序四步表 + 实测数字表 + 练习

## 核心结论

### 1. ★ 契约在基座，实现交后端

`preprocess` 一共四步，最后一行是分水岭：

<!-- src: image_processing_utils.py -->
```python
        image_like_kwargs = {} if image_like_kwargs is None else image_like_kwargs

        return self._preprocess_image_like_inputs(images, *args, **image_like_kwargs, **kwargs)
```

在这之前是**校验与默认值**（与后端无关），在这之后是 `process_image` / `_preprocess`
（两个后端各写一份）。所以 `BaseImageProcessor` 自己**不能被实例化来用** ——
它的 `process_image` 与 `_preprocess` 都只 `raise NotImplementedError`。

### 2. ★ 默认值住在类属性上，不住在签名里

<!-- src: image_processing_utils.py -->
```python
    valid_kwargs = ImagesKwargs

    default_to_square = True
    rescale_factor = 1 / 255
    model_input_names = ["pixel_values"]
```

`preprocess` 里那句 `kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))`
就是从这四个类属性（以及模型自己的覆盖值）取默认值的。**想改行为，传开关即可，
不用改代码** —— 例如 `do_normalize=False`。

### 3. ★ 视觉 token 数由模型参数决定，不由后端决定

<!-- src: image_processing_backends.py -->
```python
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
```

后端只负责"把图变成形状正确的张量"；**多少个视觉 token 是 `patch_size` / `merge_size`
决定的**：

| 量 | 实测值 | 怎么来的 |
|---|---|---|
| `pixel_values.shape` | `(576, 1176)` | patch 数 × (C×temporal×patch×patch) |
| `image_grid_thw` | `(1, 24, 24)` | 336/14 = 24 |
| 视觉 token | **144** | 24×24 / 2² |
| 另一张 700×1000 | `(1, 50, 72)` → **900** | 3600 / 4 |

**序列长度是文本说了算**：processor 把 prompt 里的 `<|image|>` 展开成 144 个
`image_token_id`，视觉向量只负责填满这 144 个已经预定好的位置。

### 4. 默认后端的选择顺序（验收点）

| 优先级 | 判据 | 结果 |
|---|---|---|
| ① | 显式 `backend="torchvision"` / `backend="pil"` | 原样返回 |
| ② | 旧参数 `use_fast`（已弃用，会 warning） | `True → torchvision`，`False → pil` |
| ③ | `DEFAULT_TO_PIL_BACKEND_IMAGE_PROCESSORS` 名单 | 名单内强制 `pil`（torchvision ≥ 0.27 时该名单为**空**） |
| ④ | 兜底 | `is_torchvision_available()` → `torchvision`，否则 `pil` |

选完之后还有一层 **兜底加载**：目标后端类在当前环境不可用时，换另一个后端再试一次。
所以「选中的后端」与「最终拿到的类」在依赖缺失时可能不是同一个。

## 与后续课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| `resize` / `center_crop` / `normalize` 的 numpy 实现 | L2-05（`image_transforms.py`） |
| `SizeDict` / `ImageInput` / `infer_channel_dimension_format` 的类型定义 | L2-05（`image_utils.py`） |
| `patchify` 的 8 维 reshape 全过程 | L2-06 |
| processor 怎么把 tokenizer 与 image processor 拼起来 | L2-06（`processing_utils.py`） |
| 336×336 → 27 个 token 的那套算法（和 144 的差异从哪来） | L2-07（视觉塔） |
| `AutoImageProcessor` 的完整分派逻辑 | L1-05（`auto` 系列） |
| `inputs_embeds` 与 `input_ids` 二选一的入口 | L0-01（已学） |

## 验收点

- [x] 保真门禁：14 个引用块全部逐字来自 3 个源文件，且位置连续
- [x] 覆盖度门禁：3 个源文件被声明（属覆盖域文件）
- [x] 参数门禁：无非法参数
- [x] 语法检查：`node --check` 通过，9 幕字段完整
- [x] 渲染门禁：无 JS 错误、无布局溢出、交互可用（两种分辨率）
- [x] **能说出默认后端的选择顺序**（见「核心结论 4」，四步表）
- 自检：能对着 `#visual` 里的形状链，说出 `(576, 1176)` 的两个维度各是怎么算出来的

**一句话总结**：

> **契约在基座（四步编排 + 类属性当默认值），实现交后端（torchvision / pil 二选一）；
> `size` 先归一再校验，`rescale` 与 `normalize` 在 torchvision 上会被融合成一步；
> 真正决定「多少个视觉 token」的不是后端，而是模型自己的 `patch_size / merge_size`。**
