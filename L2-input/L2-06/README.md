# L2-06 · 多模态处理器装配 — 从一句话加一张图到 `inputs`

> 层：**第 2 层 · 输入流水线** ｜ 优先级：P0 ｜ 前置课：`L2-05`

## 学习目标

看完这一课，你应该能：

1. **默写一次图像对话的 `inputs` 字典有哪五个 key**，并说出每个 key 由哪个子处理器产生（对应验收点 1）；
2. 解释「模板里 1 个 `<|image|>`」是怎么变成「序列里 144 个」的，并算出任意尺寸的图片会占几个位置；
3. 说出 `image_processing_glm5_next.py` 与 `image_processing_pil_glm5_next.py`
   这两套同样 322 行的实现**真正的分歧**在哪里，以及谁在什么时候被选中。

## 覆盖的源文件（6 个 / 4384 行）

| 文件 | 行数（`wc -l` 实测） | 引用块 |
|---|---|---|
| `processing_utils.py` | 2430 | 4 |
| `models/glm5_next/processing_glm5_next.py` | 193 | 4 |
| `models/glm5_next/image_processing_glm5_next.py` | 322 | 2 |
| `models/glm5_next/image_processing_pil_glm5_next.py` | 322 | 2 |
| `models/glm5_next/video_processing_glm5_next.py` | 429 | 1 |
| `feature_extraction_utils.py` | 688 | 1 |
| **合计** | **4384** | **14** |

> **口径订正（实测）**：作业书给的行数是 2431 / 194 / 323 / 323 / 430 / 689，
> `wc -l` 实测为 2430 / 193 / 322 / 322 / 429 / 688 —— 六份**都恰好差 1**，
> 是「最后一行行号」与「换行符个数」的口径差，不是内容差。本课一律用实测口径，
> 引用块行号已逐个核对（如 `Glm5NextProcessor.__init__` 在第 46 行）。
>
> 本课引用了 `_data/recon/probe_l206.py` 的**实测输出**（写在 `source.md` 的 `text` 块里），
> 以及 `models/auto/auto_mappings.py` 的后端映射表。**这些都不计入覆盖率** ——
> 覆盖域只统计 `source.md` 开头 `glm5-coverage` 块里那六个文件。

## 场景（9 幕）

1. **全景** —— 一次图像对话的装配链 + 五个 key + 四个实测数字（144 / 576 / 1176 / 5）
2. **装配三件套** —— `get_attributes()` 是从签名推出来的，`__init__` 只验收与挂载
3. **★ 验收点：五个 key 在哪一行产生** —— 三个来源 + 合并成字典的那一行
4. **★ 模板写 1 个，序列里有 144 个** —— `replace_image_token` 的三行算式
5. **chat template 只负责渲染** —— 收集内容块 → 渲染文本 → 回到同一个 `__call__`
6. **展开循环与 offset** —— 迭代器按出现顺序消费，`offset` 增量维护
7. **★ 形状演算** —— 336×336 → 24×24 个 patch → 144 个位置，`pixel_values = (576, 1176)`
8. **视频** —— 时间维先补齐再折成 patch，每帧 72 个占位符
9. **收束：序列对齐** —— 图像与视频共用一个 token，靠 `cumsum` 之差区分（`[0,2,2,0,1,0]`）

## 核心结论

### 1. `inputs` 的五个 key 是可推导的，不是"看返回值才知道"

`__call__` 的后半段只有三件事：展开文本、tokenize、拼字典。

<!-- src: processing_utils.py -->
```python
        # Pop unused keys from the inputs, e.g. inputs used only to compute number of image tokens
        data = {**text_inputs, **processed_images, **processed_videos, **processed_audio}
        data = {k: v for k, v in data.items() if k not in self.unused_input_names}

        if not kwargs.get("return_metadata"):
            data.pop("video_metadata", None)

        return BatchFeature(data, tensor_type=return_tensors, skip_tensor_conversion=self.skip_tensor_conversion)
```

实测（336×336 的图，`return_tensors="pt"`）：

| key | 形状 | 谁给的 |
|---|---|---|
| `input_ids` | `(1, 146)` | tokenizer（144 个占位符 + 2 个文本 token） |
| `attention_mask` | `(1, 146)` | tokenizer |
| `mm_token_type_ids` | `(1, 146)` | processor 自己（默认 `return_mm_token_type_ids=True`） |
| `pixel_values` | `(576, 1176)` | image_processor（`1176 = 3×2×14×14`） |
| `image_grid_thw` | `(1, 3)` = `[[1, 24, 24]]` | image_processor |

### 2. ★ 模板写 1 个占位符，处理器展开成 N 个

<!-- src: models/glm5_next/processing_glm5_next.py -->
```python
    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        merge_length = self.image_processor.merge_size**2
        num_image_tokens = image_inputs["image_grid_thw"][image_idx].prod() // merge_length
        return self.image_token * num_image_tokens
```

`N = grid_t × grid_h × grid_w / merge_size²`。336×336 的图：`1×24×24 = 576`，`576 / 4 = 144`。
**视觉 token 与文本 token 共用一张词表、一条序列** —— 展开发生在 tokenize 之前，
所以"视觉位置"在 tokenizer 眼里就是普通文本 token。

### 3. ★ 图像与视频共用一个 token，靠括号区分

<!-- src: models/glm5_next/processing_glm5_next.py -->
```python
            starts = np.cumsum(array_ids == self.video_start_id, axis=0)
            ends = np.cumsum(array_ids == self.video_end_id, axis=0)
            is_video_modality = starts > ends

            mm_token_types[(array_ids == self.image_token_id) & is_video_modality] = 2
            mm_token_types[(array_ids == self.image_token_id) & (~is_video_modality)] = 1
```

实测：输入 `[3, 1, 1, 4, 1, 5]` → 输出 `[0, 2, 2, 0, 1, 0]`。
模型侧用**同一套规则**把占位符切成 image / video 两堆再填向量，规则不一致就会整体错位（展开见 L2-07）。

### 4. 两套图像处理器：同 322 行，真正的分歧只有三处

`Glm5NextImageProcessor`（torchvision）与 `Glm5NextImageProcessorPil`（PIL）是同一份规格的两种执行方式：

| | torchvision（默认） | PIL（回退） |
|---|---|---|
| 张量库 | `torch` + `tvF.pad` | `numpy` + `np.pad` |
| 批处理 | 按形状分组 `group_images_by_shape` / `reorder_images` | 逐张循环 + `np.concatenate` |
| resize 的 `factor` | `patch_size * merge_size * patch_expand_factor` | `patch_size * merge_size`（漏乘，默认 1 所以结果相同） |

类属性完全一致（`patch_size=14` / `temporal_patch_size=2` / `merge_size=2` /
`min_image_tokens=16` / `max_image_tokens=8000`），选谁由环境（torchvision 是否可用）
与 `backend=` 参数决定，**与模型语义无关**。

### 实测数字

| 量 | 实测值 | 含义 |
|---|---|---|
| 五个 key | 5 | `input_ids` / `attention_mask` / `mm_token_type_ids` / `pixel_values` / `image_grid_thw` |
| 336×336 → 位置数 | 144 | `24×24 / 4` |
| 512×512 → 位置数 | 361 | 画布被**放大**到 532×532（向上对齐到 28 的倍数） |
| 1024×768 → 位置数 | 1036 | 画布 1036×784 |
| 2048×1536 → 位置数 | 4070 | 画布 2072×1540 |
| 14×14 → 位置数 | 16 | `min_image_tokens` 的地板 |
| `pixel_values` 每行 | 1176 | `3 × 2 × 14 × 14` |
| 两张不同尺寸的图 | `(2020, 1176)` | `576 + 1444`，沿第 0 维拼接 |
| 16 帧 336×336 | `(8, 24, 24)` → 1152 → 72/帧 | 每帧占位符 = `24×24 / (4×2)` |

## 与后续课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| `smart_resize` 的二分收敛与对齐规则 | L2-05 |
| `pixel_values` 的 `(总 patch 数, 1176)` 怎么被视觉塔吃进去 | L2-07 |
| `image_grid_thw` 里 `t` 维在视觉塔里做什么 | L2-07（轴向 RoPE / temporal patch） |
| `get_placeholder_mask` 怎么把视觉向量填回占位符位置 | L2-07 |
| `AutoProcessor` 怎么找到 `Glm5NextProcessor` | L1-06 |
| `inputs_embeds` 里视觉 token 的替换发生在哪 | L0-01 + L2-07 |

## 验收点

- [x] 保真门禁：14 个引用块全部逐字来自 6 个源文件，且位置连续
- [x] 覆盖度门禁：6 个源文件被声明（与 `tools/plan.py` 中 L2-06 的 `files` 一致）
- [x] 参数门禁：8 处参数引用（全部是 `shared/theme.css` 里声明过的 CSS 变量）均真实存在
- [x] 语法检查：`node --check` 通过，9 幕字段完整
- [x] 渲染门禁：无 JS 错误、无布局溢出、交互可用（1280×720 + 1920×1080）
- 自检：能不看笔记写出五个 key，并说出 `image_grid_thw` 里三个数各是什么

**一句话总结**：

> 一次图像对话 = chat template 写 **1 个** `<|image|>` → `replace_image_token` 按
> `image_grid_thw.prod() // merge_size²` 展开成 **N 个** → 与文本一起 tokenize 成**同一条**序列
> → 再挂上 `pixel_values` 与 `image_grid_thw`，交给 `BatchFeature` 打包；
> **处理器只决定"占几个位置"，视觉塔才决定"每个位置是什么向量"。**
