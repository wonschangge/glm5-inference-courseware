<!-- glm5-coverage
processing_utils.py
models/glm5_next/processing_glm5_next.py
models/glm5_next/image_processing_glm5_next.py
models/glm5_next/image_processing_pil_glm5_next.py
models/glm5_next/video_processing_glm5_next.py
feature_extraction_utils.py
-->

# L2-06 · 多模态处理器装配 — 源文件

**这一课只回答一个问题：一句「描述这张图」加一张 336×336 的 PNG，是怎么变成一个 `inputs` 字典的？**

视角：`processor(...)` 一次调用内部的**装配顺序** —— 谁调谁、图片在哪一步变成占位符、
文本 token 与视觉 token 又是在哪一行被对齐到同一条序列上。
本课不解释视觉塔内部（那是 L2-07），只解释「视觉塔的输入是怎么被摆好的、它的输出要占几个位置」。

| 文件 | 行数（`wc -l` 实测） | 本课引用块 |
|---|---|---|
| `processing_utils.py` | 2430 | 4 |
| `models/glm5_next/processing_glm5_next.py` | 193 | 4 |
| `models/glm5_next/image_processing_glm5_next.py` | 322 | 2 |
| `models/glm5_next/image_processing_pil_glm5_next.py` | 322 | 2 |
| `models/glm5_next/video_processing_glm5_next.py` | 429 | 1 |
| `feature_extraction_utils.py` | 688 | 1 |
| **合计** | **4384** | **14** |

> **口径订正（实测）**：作业书给的六个行数是 2431 / 194 / 323 / 323 / 430 / 689，
> 用 `wc -l` 实测是 2430 / 193 / 322 / 322 / 429 / 688 —— **每一份都恰好差 1**。
> 这是「最后一行的行号」与「换行符个数」两种口径的差，**不是内容差**：
> 六个文件都以换行结尾，按「行号」数会多出末尾那一行。本课一律用实测口径，
> 下面的引用块行号也已逐个核对过（例如 `Glm5NextProcessor.__init__` 在第 46 行）。

**本课全部数字来自一次真实运行**（`_data/recon/probe_l206.py`，工装，**不计入覆盖率**）：
用一份最小词表把 `Glm5NextProcessor` 真的装起来，喂一张真的 336×336 图，然后把
返回字典的 key、形状、占位符个数全部打印出来。

```text
一次图像对话（336×336）的 inputs —— 实测
  keys()            : ['input_ids', 'attention_mask', 'mm_token_type_ids', 'pixel_values', 'image_grid_thw']
  input_ids         : (1, 146)  int64      # 144 个占位符 + 2 个文本 token
  attention_mask    : (1, 146)  int64
  mm_token_type_ids : (1, 146)  int64      # 144 个 1（图像）+ 2 个 0（文本）
  pixel_values      : (576, 1176) float32  # 576 = 24×24 个 patch，1176 = 3×2×14×14
  image_grid_thw    : (1, 3)    int64      # [[1, 24, 24]] = (t, h, w) 三个网格数

同一个 processor 的其它实测值
  get_attributes()   : ['image_processor', 'tokenizer', 'video_processor']
  model_input_names  : ['pixel_values', 'image_grid_thw', 'input_ids', 'attention_mask',
                        'pixel_values_videos', 'video_grid_thw', 'mm_token_type_ids']
  占位符展开          : prompt 里 1 个 <|image|>  ->  序列里 144 个
  create_mm_token_type_ids([3,1,1,4,1,5]) -> [0,2,2,0,1,0]     # 视频段内的 1 变成 2
```

> 上面两个 `text` 块是**实测汇总**，不是源文件引用，不参与保真校验。
> 这一段用的词表是我自己造的最小词表（`<|image|>` 恰好落在 id=1），
> 所以下面的 id 数字**只在"相对关系"上有意义**；真实词表里 `image_token_id` 的默认值是
> 154854（`Glm5NextConfig` 的字段，属 L2-07 的覆盖范围，本课不引用其源码）。

---

## 一、★ 装配：`ProcessorMixin.__init__` 只做三件事

先看结论：**ProcessorMixin 不是"有三个成员变量的类"，而是一次"把位置参数折进 kwargs、
逐个 setattr 挂上去"的装配过程。** 看懂这 20 行，后面所有 `hasattr(self, "image_processor")`
式的写法都顺理成章。

<!-- src: processing_utils.py -->
```python
        # Sanitize args and kwargs
        for key in kwargs:
            if key not in self.get_attributes():
                raise TypeError(f"Unexpected keyword argument {key}.")
        for arg, attribute_name in zip(args, self.get_attributes()):
            if attribute_name in kwargs:
                raise TypeError(f"Got multiple values for argument {attribute_name}.")
            else:
                kwargs[attribute_name] = arg

        if len(kwargs) != len(self.get_attributes()):
            raise ValueError(
                f"This processor requires {len(self.get_attributes())} arguments: {', '.join(self.get_attributes())}. Got "
                f"{len(args)} arguments instead."
            )

        # Check each arg is of the proper class (this will also catch a user initializing in the wrong order)
        for attribute_name, arg in kwargs.items():
            self.check_argument_for_proper_class(attribute_name, arg)
            setattr(self, attribute_name, arg)
```

**读法**：
- 第一段做的是**去重与折叠**：位置参数被塞进 `kwargs[attribute_name]`，于是
  `Glm5NextProcessor(image_processor, tokenizer, video_processor)` 与三个关键字写法
  走的是**同一条**路径 —— 位置顺序错了不会"悄悄错位"，因为下一段要按名字查类型。
- `len(kwargs) != len(self.get_attributes())` 这一行说明**参数个数不是写死的**：
  `get_attributes()` 是从子类 `__init__` 的签名里现推出来的（GLM-5 的签名见下一节），
  所以 GLM-5 要三个、别的模型要两个，都靠同一个 `__init__` 兜住。
- `check_argument_for_proper_class` 在 `setattr` **之前**执行：类型不对就根本挂不上去。
  这是"用户把 tokenizer 传给了 image_processor"这类错误的唯一拦截点。
- 三件事里**没有**任何一件是"创建子处理器"。子处理器是外面造好传进来的 ——
  `__init__` 只负责**验收 + 挂载**。这条边界决定了下一节的问题：谁来造？

---

## 二、GLM-5 的签名：三件套，以及两个"猜出来"的 token

`Glm5NextProcessor.__init__` 的签名就是它的装配清单。注意它对 token 的态度：
**先问 tokenizer 有没有，没有就用字面量兜底。**

<!-- src: models/glm5_next/processing_glm5_next.py -->
```python
    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):
        self.image_token = "<|image|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.video_token = "<|video|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )
        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)
        self.video_start_id = tokenizer.convert_tokens_to_ids("<|begin_of_video|>")
        self.video_end_id = tokenizer.convert_tokens_to_ids("<|end_of_video|>")
```

**读法**：
- 三个子处理器是**并列**的：`image_processor` / `tokenizer` / `video_processor`。
  它继承 `ProcessorMixin`，所以第一节那套"按名字查类型再 setattr"直接生效 ——
  这就是"把 tokenizer + image/video processor 拼起来"的全部机制，没有别的魔法。
- `self.image_token = "<|image|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token`：
  **占位符字符串是处理器自己的属性**，词表里有没有这个 token 由 `convert_tokens_to_ids` 决定。
  处理器不检查词表，所以这个字符串必须与 checkpoint 里的特殊 token 一致，否则
  `_check_special_mm_tokens`（第四节）会在 tokenize 之后报错。
- `video_start_id` / `video_end_id` 是在 `super().__init__` **之后**才取的。
  这两个 id 是本课第十一节的关键：GLM-5 **没有**单独的"视频占位符"，
  `<|begin_of_video|>` / `<|end_of_video|>` 这一对括号才是视频与图像的分界。
- `super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)`：
  位置顺序与签名一致。`chat_template` 只能走关键字 —— 因为它是 kwargs.pop 出来的，不是属性。

---

## 三、默认 kwargs：为什么 `mm_token_type_ids` 一定在返回字典里

`Glm5NextProcessorKwargs._defaults` 是整条链路上最容易被忽略、却决定了验收点的那 9 行。

<!-- src: models/glm5_next/processing_glm5_next.py -->
```python
class Glm5NextProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_token_type_ids": False,
            "return_mm_token_type_ids": True,
        },
        "videos_kwargs": {"return_metadata": True},
    }
```

**读法**：
- `"return_mm_token_type_ids": True` 是**默认开启**的。所以只要调
  `processor(text=..., images=...)`，返回字典里就会多一个 `mm_token_type_ids`
  —— 它不是模型必需，而是给推理引擎（vLLM 那类）判断"哪个位置该换成视觉向量"用的。
- `"padding": False` 意味着**默认不补齐**：一个 batch 里两条长度不同的样本会得到
  ragged 的 list，而不是 (B, L) 张量。这解释了 `create_mm_token_type_ids` 里
  为什么要"逐条转 numpy"（第十节）。
- `"return_token_type_ids": False`：文本侧不产生 `token_type_ids`。
  所以返回字典里的 `mm_token_type_ids` **不是** tokenizer 给的，是处理器自己加的。
- `"videos_kwargs": {"return_metadata": True}` 只影响**视频子处理器**的输出；
  最终字典里留不留 `video_metadata` 还取决于调用方传没传 `return_metadata`（见下节最后一行的 pop）。

---

## 四、一次调用的收尾：五个 key 是在这里拼出来的

`ProcessorMixin.__call__` 前半段做模态分发，后半段把三路结果**合并成一个字典**。
验收点「一次图像对话的 inputs 有哪些 key」的答案就在这 23 行里。

<!-- src: processing_utils.py -->
```python
            text, text_replacement_offsets = self.get_text_with_replacements(
                text,
                images_replacements,
                videos_replacements,
                audio_replacements,
            )
            text_inputs = self.tokenizer(text, **merged_kwargs["text_kwargs"])
            self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video", "audio"])

            if return_text_replacement_offsets:
                text_inputs["text_replacement_offsets"] = text_replacement_offsets

            if return_mm_token_type_ids:
                text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(text_inputs["input_ids"])

        # Pop unused keys from the inputs, e.g. inputs used only to compute number of image tokens
        data = {**text_inputs, **processed_images, **processed_videos, **processed_audio}
        data = {k: v for k, v in data.items() if k not in self.unused_input_names}

        if not kwargs.get("return_metadata"):
            data.pop("video_metadata", None)

        return BatchFeature(data, tensor_type=return_tensors, skip_tensor_conversion=self.skip_tensor_conversion)
```

**读法**：
- 顺序是 **先展开文本 → 再 tokenize**：`get_text_with_replacements` 把占位符扩成 N 个，
  然后才交给 tokenizer。所以"视觉 token"在 tokenizer 眼里**就是普通文本 token**，
  走的是同一张词表、同一条序列 —— 这是本课最重要的一条设计决定。
- `self.tokenizer(text, **merged_kwargs["text_kwargs"])` 返回的是 tokenizer 的字典
  （`input_ids` + `attention_mask`）；`text_inputs["mm_token_type_ids"] = ...` 是**再往里加一个 key**。
- 第 710 行才是"拼字典"：`{**text_inputs, **processed_images, **processed_videos, **processed_audio}`。
  图像那一路贡献的是 `pixel_values` + `image_grid_thw`（`model_input_names` 里写着）。
  于是图像对话的五个 key 全部到齐：**`input_ids` / `attention_mask` /
  `mm_token_type_ids` / `pixel_values` / `image_grid_thw`**。
- `unused_input_names` 是一个可覆写的"黑名单"，默认空 —— 也就是说**子处理器多返回的 key 会漏进最终字典**，
  想拦住必须显式声明。
- 最后一行返回 `BatchFeature` 而不是 dict：这让 `inputs.input_ids` 与 `inputs["input_ids"]` 等价（第十二节）。
- 第 701 行 `_check_special_mm_tokens(...)` 卡在 tokenize 与打包之间：它比较
  **展开后的文本**里占位符出现次数与 `input_ids` 里对应 id 的出现次数，不一致就抛错。
  默认 `padding=False` 且不截断时两者必然相等；一旦 `truncation="max_length"` 把序列截断，
  占位符被截掉而文本计数不变，这个断言就会救你一次 —— 它是整条链路上唯一的"数量守恒"检查。
- **回顾 L0-01**：那次前向为什么在多模态场景走 `inputs_embeds` 而不是 `input_ids`？
  正是因为这里的 144 个位置会在进模型之前被换成向量。本课决定"占几个位置"，
  L0-01 决定"位置上的东西从哪来"。
- `data.pop("video_metadata", None)` 读的是**调用方原始 kwargs**。由于默认值写在
  `videos_kwargs` 里、而这里只看顶层，所以想要 `video_metadata` 得在顶层传 `return_metadata=True`。
  注意 `replace_video_token` 在此之前就已经用掉了 `video_metadata`（第五节），pop 只影响返回值。

---

## 五、★ 占位符：模板里 1 个，序列里 N 个

这是本课第一个真正的洞察。**chat template 只写一个占位符；把 1 变成 N 的是处理器。**

<!-- src: models/glm5_next/processing_glm5_next.py -->
```python
    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        merge_length = self.image_processor.merge_size**2
        num_image_tokens = image_inputs["image_grid_thw"][image_idx].prod() // merge_length
        return self.image_token * num_image_tokens

    def replace_video_token(self, video_inputs: dict, video_idx: int, **kwargs) -> str:
        merge_length = self.video_processor.merge_size**2
        num_frames = video_inputs["video_grid_thw"][video_idx][0]
        num_image_tokens = video_inputs["video_grid_thw"][video_idx].prod() // merge_length // num_frames
```

**读法**：
- `replace_image_token` 只有三行，但它定义了整条对齐关系：
  `N = image_grid_thw[i].prod() // merge_size²`。实测 336×336 的图：
  `1×24×24 = 576`，`576 // 4 = 144` —— 于是 prompt 里那**一个** `<|image|>` 被替换成 **144 个**。
- 这里的 `prod()` 连时间维一起乘了：图像的时间维恒为 1，所以图片的 N = 空间 patch 数 ÷ 合并数。
- `replace_video_token` 多除了一次 `num_frames`：视频的 `video_grid_thw = [grid_t, grid_h, grid_w]`
  已经是"时间 patch 数"，所以 `prod() // merge²` 是**整段视频**的 token 数，
  再除以帧数才是**每帧**的占位符数（第十节实测 16 帧 336×336 → 每帧 72 个）。
- 两个函数都用 `self.image_token` 本身，而**不是**另一个"视频 token"。
  GLM-5 的图像与视频占位符是同一个字符串——这正是第十一节要解决的问题。
- 返回值是**字符串**，不是张量：`self.image_token * 144` 就是 Python 的字符串重复
  （实测返回长度 1296 = 144 × 9，`<|image|>` 正好 9 个字符）。
  "视觉 token 的数量"在这一步就已经定死了，与后续 tokenizer 无关。

---

## 六、展开与记账：`get_text_with_replacements` 的核心循环

上一节算出了"替换成什么"，这一节解决"替换到哪儿、按什么顺序"。

<!-- src: processing_utils.py -->
```python
            for m in re.finditer(regex_special_mm_tokens, text[batch_idx]):
                start, end = m.span()
                expanded_sample.append(text[batch_idx][last:start])

                # adjust spans using running offset if one sample has several MM data associated
                start_with_offset = start + offset

                mm_type = m.lastgroup
                replacement_text = next(replacements_iters[mm_type])
                replacement_offsets.append(
                    {
                        "type": mm_type,
                        "span": (start, end),
                        "new_span": (start_with_offset, start_with_offset + len(replacement_text)),
                        "text": m.group(),
                        "replacement": replacement_text,
                    }
                )
                expanded_sample.append(replacement_text)
                # update the offsets and the last position
                offset += len(replacement_text) - (end - start)
                last = end
```

**读法**：
- `re.finditer` 一次扫描同时做两件事：**定位**（`m.span()`）与**分类**（`m.lastgroup`）。
  命名分组 `(?P<image>...)` / `(?P<video>...)` 让同一个正则能分辨模态，不必按模态扫多遍。
- `next(replacements_iters[mm_type])`：一个模态的替换串是一个**迭代器**，
  按句子里的出现顺序**逐个消费**。所以"第 i 个占位符"与"第 i 张图"的对应关系
  完全由出现顺序决定 —— 顺序错了不会报错，只会错位。
- `offset` 是这段代码里唯一需要动脑的地方：`start_with_offset` 把"原文里的位置"
  换算成"展开后文本里的位置"，`offset += len(replacement) - (end - start)` 是增量维护。
  没有它，`return_text_replacement_offsets` 给出的 offset 会全部作废。
- 产物有两个：**展开后的文本**（喂给 tokenizer）和**每个占位符的账目**
  （`span` 原文位置 / `new_span` 展开后位置 / `replacement` 替换串）。
  账目是给 `return_assistant_tokens_mask` 用的 —— 要在"被撑长的序列"上反推
  哪一段是 assistant 生成的。
- 循环结束后第 921 行把尾巴 `text[batch_idx][last:]` 补上；拼装用的是列表 + `"".join`，
  所以 `+` 拼接的开销不会随占位符个数平方增长。

---

## 七、chat template 做的事只有一件：把图和文凑到**一次**调用里

`apply_chat_template(tokenize=True)` 不是"另一个处理器"，它只是**先渲染文本、再把素材收集起来**。

<!-- src: processing_utils.py -->
```python
        if tokenize:
            batch_images, batch_videos = [], []
            batch_audios = []
            for conversation in conversations:
                images, videos = [], []
                for message in conversation:
                    content = message.get("content") or []
                    if isinstance(content, str):
                        continue
                    visuals = [
                        content_block for content_block in content if content_block["type"] in ["image", "video"]
                    ]
```

**读法**：
- 模板（jinja）负责把 `{"type": "image"}` 这个**内容块**渲染成占位符文本；
  处理器负责把内容块里的**文件本身**收集到 `batch_images`。两者靠"顺序"对齐：
  模板里第 i 个 image 块 ↔ `batch_images` 里的第 i 个文件。
- 所以"GLM-5 的 chat template 如何插入视觉占位符"的答案分两半：
  **模板写 1 个 `<|image|>`**（GLM 系列模板里就是这个字面量），
  **处理器把它展开成 N 个**（第五节）。模板里出现的次数与图片张数一一对应，
  而序列里的长度由 `image_grid_thw` 决定 —— 这两件事分别发生在两个文件里。
- 注意 `visuals` 同时收 `image` 和 `video` 两种内容块；具体谁是谁由后面的
  `image_fnames` / `video_fnames` 两个列表推导式分流。
- 收尾在第 2250 行附近：`out = self(text=prompt, images=batch_images if images_exist else None, ...)`
  —— 绕了一圈，还是回到第四节那个 `__call__`。**`apply_chat_template` 没有第二条代码路径。**

---

## 八、形状演算：336×336 是怎么变成 144 个 token 的

上面说了 N = `grid_h × grid_w / merge²`。这一节把 `grid` 从像素算出来。
**`pixel_values` 不是 (B, C, H, W)，而是"每个 patch 一行"的二维表** —— 这是理解后面所有形状的钥匙。

<!-- src: models/glm5_next/image_processing_glm5_next.py -->
```python
        batch_size, channel, resized_height, resized_width = images.shape
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        patches = images.reshape(
            batch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
        flatten_patches = (
            patches.unsqueeze(6)
            .expand(-1, -1, -1, -1, -1, -1, temporal_patch_size, -1, -1)
            .reshape(
                batch_size,
                grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )
        )
        return flatten_patches, grid_h, grid_w
```

**读法**：
- 输入是 `(batch, channel, resized_height, resized_width)`，实测 336×336 的图 resize 后仍是 336×336，
  于是 `grid_h = grid_w = 336 // 14 = 24`。
- `reshape` 把空间维**同时**拆成 `(grid // merge, merge, patch)`：这不是为了计算，
  而是为了把 `merge_size × merge_size` 个相邻 patch 排到一起，供后面的 merger 使用。
- `permute(0, 2, 5, 3, 6, 1, 4, 7)` 把这 8 维重排成
  `(batch, gh, gw, mh, mw, channel, patch, patch)` —— 于是最后两维 `patch × patch` 相邻，
  可以一次性拉平成"一个 patch 的像素"。
- `unsqueeze(6).expand(..., temporal_patch_size, ...)` 是给静态图补一个**时间维**：
  同一张图在时间上重复 `temporal_patch_size = 2` 次。这就是为什么
  图像的 `pixel_values` 最后一维是 `channel × 2 × 14 × 14 = 1176`，而不是 `3×14×14 = 588`。
  **图像和视频在视觉塔入口处共用同一种布局**，只是图像的时间维是"复制"出来的。
- 输出 `flatten_patches` 的形状是 `(batch, grid_h × grid_w, channel × temporal × patch × patch)`
  = 实测 `(1, 576, 1176)`；`_preprocess` 里再去掉 batch 维、把多张图沿第 0 维 `cat` 起来，
  于是 batch 里有两张不同尺寸的图时得到 `(576 + 1444, 1176) = (2020, 1176)`。
- `grid_h, grid_w` 被原样返回并写进 `image_grid_thw` —— **这才是占位符数量的唯一来源**，
  像素数据本身不携带"我值几个 token"的信息。

<!-- src: models/glm5_next/image_processing_glm5_next.py -->
```python
    def get_number_of_image_patches(self, height: int, width: int, images_kwargs: dict | None = None) -> int:
        """
        A utility that returns number of image patches for a given image size.

        Args:
            height (`int`):
                Height of the input image.
            width (`int`):
                Width of the input image.
            images_kwargs (`dict`, *optional*)
                Any kwargs to override defaults of the image processor.
        Returns:
            `int`: Number of image patches per image.
        """
        images_kwargs = images_kwargs or {}
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)

        # Key difference is the dynamically based resize on min/max image tokens
        min_image_tokens = images_kwargs.get("min_image_tokens", self.min_image_tokens)
        max_image_tokens = images_kwargs.get("max_image_tokens", self.max_image_tokens)

        factor = patch_size * merge_size
        resized_height, resized_width = smart_resize(
            num_frames=self.temporal_patch_size,
            height=height,
            width=width,
            factor=factor,
            min_pixels=min_image_tokens,
            max_pixels=max_image_tokens,
            temporal_factor=self.temporal_patch_size,
        )
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        return grid_h * grid_w
```

**读法**：
- 这个函数是"**不真的处理图片**，只回答要几个 patch"的入口。它存在的理由很实际：
  推理引擎要在分配 KV cache 之前知道总长度，而那时还没解码图片。
- `factor = patch_size * merge_size`（28）比视觉塔的 patch（14）大：对齐的粒度是
  **一个"合并后"的单元**，因为 merger 会把 2×2 个 patch 并成一个 token。
- `num_frames=self.temporal_patch_size`：静态图被当成"2 帧"，与第八节的 `expand` 保持一致 ——
  这是"图像预算"和"视频预算"能用同一个 `smart_resize` 的原因。
- 实测的整条链路（`smart_resize` → 对齐画布 → grid → token）：

```text
   输入尺寸      对齐画布      grid(h×w)    patch 数   token 数
   336×336   ->  336×336       24×24          576        144
   512×512   ->  532×532       38×38         1444        361
   1024×768  ->  1036×784      74×56         4144       1036
   2048×1536 ->  2072×1540    148×110       16280       4070
   100×100   ->  112×112        8×8            64         16
    50×50    ->  112×112        8×8            64         16   <- 被 min_image_tokens=16 托住
    14×14    ->  112×112        8×8            64         16   <- 再小也一样，16 就是地板
   200×100   ->  224×112       16×8           128         32
```

- 注意 512×512 **不是**缩小到 448 或 512 的倍数，而是**放大**到 532：`align` 是向上取整到 28 的倍数。
  所以"输入多大"与"输出多少 token"之间隔着一个**向上对齐**，不要用 `H/14` 直接心算。

---

## 九、两套图像处理器：同 322 行里的三处分歧

`image_processing_glm5_next.py` 与 `image_processing_pil_glm5_next.py` **行数完全相同（各 322 行）**，
因为它们是同一个 modular 源生成的两份后端实现。分工是这样的：

```text
   谁来选后端（下面这张映射表在 models/auto/auto_mappings.py，属本课覆盖域之外的文件，
   此处只作背景说明，不计入覆盖率）
     ("glm5_next", {"pil": "Glm5NextImageProcessorPil", "torchvision": "Glm5NextImageProcessor"})
     backend=None 时：torchvision 可用 -> 用 Glm5NextImageProcessor（默认）；否则回退 PIL

   两边的类属性一模一样：patch_size=14, temporal_patch_size=2, merge_size=2,
                        min_image_tokens=16, max_image_tokens=8000,
                        model_input_names=["pixel_values", "image_grid_thw"]
   两边都有：smart_resize / resize / patchify / _preprocess / get_number_of_image_patches

   真正的分歧只有三处
     ① 张量库：torchvision 用 torch + tvF.pad；PIL 用 numpy + np.pad
     ② 批处理：torchvision 按形状分组（group_images_by_shape / reorder_images）；
                 PIL 逐张循环，最后 np.concatenate
     ③ resize 的 factor：torchvision 传 patch_size*merge_size*patch_expand_factor；
                          PIL 只传 patch_size*merge_size（漏了 patch_expand_factor，
                          默认值为 1 所以结果相同）
```

PIL 版的 `patchify` 与第八节那段是同一个算法，可以直接对照着看：

<!-- src: models/glm5_next/image_processing_pil_glm5_next.py -->
```python
        patches = image.reshape(
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        # (gh, gw, mh, mw, C, ph, pw)
        patches = np.transpose(patches, (1, 4, 2, 5, 0, 3, 6))

        # expand temporal_patch_size as a broadcast (zero-copy)
        patches = np.broadcast_to(
            patches[:, :, :, :, :, None, :, :],
            (*patches.shape[:5], temporal_patch_size, *patches.shape[5:]),
        )

        flatten_patches = patches.reshape(
            grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size,
        )
        return flatten_patches, grid_h, grid_w
```

**读法**：
- 维度顺序不同但语义相同：torchvision 版是 8 维 `reshape` + `permute`，PIL 版是
  7 维 `reshape`（没有 batch 维，因为它是**逐张**处理）+ `np.transpose`。
  注释 `# (gh, gw, mh, mw, C, ph, pw)` 就是重排后的维度名，与第八节的 `permute` 目标一致。
- `np.broadcast_to(..., (*patches.shape[:5], temporal_patch_size, *patches.shape[5:]))`：
  注释写明是 **zero-copy** —— 时间维不复制数据，只改步长。对照第八节的
  `.expand(...)`：torch 与 numpy 在这里做的是同一件事（广播视图），
  真正的复制发生在后面的 `reshape`。
- 最后 `reshape(grid_h * grid_w, channel * temporal * patch * patch)` 直接给出
  二维表（没有 batch 维）—— 这正是 `_preprocess` 里能 `np.concatenate(..., axis=0)`
  把不同尺寸的图拼成一条的原因。

PIL 版 `_preprocess` 的循环头，把"逐张"这个策略写得很直白：

<!-- src: models/glm5_next/image_processing_pil_glm5_next.py -->
```python
        for image in images:
            if do_resize:
                image = self.resize(
                    image,
                    resample=resample,
                    factor=patch_size * merge_size,
                    temporal_factor=temporal_patch_size,
                    min_image_tokens=min_image_tokens,
                    max_image_tokens=max_image_tokens,
                )
```

**读法**：
- `for image in images:` —— PIL 后端**没有** `disable_grouping` 这个开关，
  也没有 `group_images_by_shape`；形状对齐完全交给 `resize` 里的 padding 处理。
- `factor=patch_size * merge_size`（少乘 `patch_expand_factor`）是两份实现里
  唯一一处**算术差异**。默认 `patch_expand_factor = 1`，所以两边的 `smart_resize`
  参数完全相同、输出逐元素一致；但它是一颗埋着的雷：一旦有人把
  `patch_expand_factor` 调成非 1，两条后端会给出不同的画布。
- 这也解释了为什么"默认/PIL 分工"不是功能差异：**它们是同一份规格的两种执行方式**，
  选谁由环境（torchvision 是否可用）和 `backend=` 参数决定，与模型语义无关。

---

## 十、视频：时间维先补齐，再折成 patch

视频比图像多一个坑：`temporal_patch_size = 2` 要求帧数是偶数。修的地方在 `patchify` 里。

<!-- src: models/glm5_next/video_processing_glm5_next.py -->
```python
        # Check that videos have `num_frames` divisible by `temporal_patch_size`
        if pad := -num_frames % temporal_patch_size:
            repeats = videos[:, -1:].expand(-1, pad, -1, -1, -1)
            videos = torch.cat((videos, repeats), dim=1)
            num_frames += pad

        grid_t = num_frames // temporal_patch_size
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
```

**读法**：
- `if pad := -num_frames % temporal_patch_size:` 是海象运算符的经典用法：
  一边算出"要补几帧"一边判断非零。15 帧 → `-15 % 2 = 1` → 补 1 帧。
- 补的方式是 `videos[:, -1:]` **重复最后一帧**，不是补零：视觉塔看到的是"静止画面多停一瞬"，
  而不是一块黑。这条选择直接决定了时间维上的语义。
- `grid_t = num_frames // temporal_patch_size`：时间维被**折叠**进 patch，
  所以 `video_grid_thw[0]` 是 patch 数而不是帧数。实测 16 帧 336×336：
  `(grid_t, grid_h, grid_w) = (8, 24, 24)`，`flatten_patches` 形状 `(1, 4608, 1176)`。
- 占位符怎么算（结合第五节的 `replace_video_token`）：
  `4608 // 4 = 1152` 个 token，再 `// 16 帧 = 72` 个/帧；
  等价地 `24 × 24 / (2² × 2) = 72`。**分母上那个额外的 2 就是 `temporal_patch_size`。**
- 15 帧时补一帧后同样得到 `(8, 24, 24)` —— 补齐让"帧数"这个变量从 grid 里彻底消失，
  这与 `sample_frames` 里 `if len(uniq) & 1: uniq.append(uniq[-1])`（第 234 行）
  是同一个约束在两个层面的两次落实：**采样阶段保证偶数帧，patchify 阶段再兜一次底。**

---

## 十一、★ 图像和视频共用一个 token，靠 span 区分

`Glm5NextProcessor.create_mm_token_type_ids` 是本课最"反直觉"的一段：
它不按 token 的种类分类，而是先用**括号**划出视频段，再在段内分类。

<!-- src: models/glm5_next/processing_glm5_next.py -->
```python
    def create_mm_token_type_ids(self, input_ids: list) -> list[list[int]]:
        # We have to iterate for each list separately because inputs
        # might be non-padded lists and we can't cast numpy on that!
        # Then cast numpy as each input for faster indexing
        mm_token_type_ids = []
        for input in input_ids:
            array_ids = np.array(input)
            mm_token_types = np.zeros_like(input)

            # Replace 0 -> 2 only inside video segments because Glm5Next
            # uses the same special token to denote images and video
            # Otherwise replace 0 -> 1 for image modality
            starts = np.cumsum(array_ids == self.video_start_id, axis=0)
            ends = np.cumsum(array_ids == self.video_end_id, axis=0)
            is_video_modality = starts > ends

            mm_token_types[(array_ids == self.image_token_id) & is_video_modality] = 2
            mm_token_types[(array_ids == self.image_token_id) & (~is_video_modality)] = 1
            mm_token_type_ids.append(mm_token_types.tolist())
        return mm_token_type_ids
```

**读法**：
- 前两行注释就把原因写明了：**GLM-5 用同一个特殊 token 同时表示图像和视频**，
  所以单看 token id 无法区分模态。
- `np.cumsum(array_ids == self.video_start_id)` 与 `... == self.video_end_id` 两个**累计计数**之差，
  就是"当前位置是否落在 `<|begin_of_video|> ... <|end_of_video|>` 之间"。
  用一个 O(n) 的前缀和代替显式的区间循环，是这段代码最值得学的地方。
- 实测：输入 `[3, 1, 1, 4, 1, 5]`（3=begin_of_video，4=end_of_video，1=image_token）
  输出 `[0, 2, 2, 0, 1, 0]` —— **视频段内的占位符是 2，段外的是 1，括号本身是 0**。
- 掩码写法 `mm_token_types[(cond_a) & (cond_b)] = 2` 是 numpy 的布尔索引：
  先算 2 再算 1，顺序不能反（否则视频段会被 1 覆盖）。
  又因为两个条件互斥（`is_video_modality` 与 `~is_video_modality`），
  实际上顺序无关 —— 但写成互斥形式更容易看出"这是二分类"。
- 它为什么必须存在：模型侧要用同一套规则把 `input_ids` 里的占位符切成
  image / video 两堆（`get_placeholder_mask` 里 `cumsum` 的写法与这里**完全同构**，
  源码注释也写着 "Core difference to other VLMs as img token == vid token"），
  然后把视觉塔输出的向量按这两堆填回去。**处理器与模型必须用同一条规则，
  否则填进去的位置会整体错位** —— 展开在 L2-07。
- `mm_token_types` 用 `np.zeros_like(...)` 起手：默认全 0（普通文本），
  再用两个布尔掩码把两类占位符分别改成 1 与 2 —— 三种取值就是三个模态。
- 逐条转 numpy 的原因写在注释里：默认 `padding=False`，序列长度不一，
  不能整批转成 (B, L) 张量，只能一条一条来（`np` 在这里是拿来加速索引的，不是拿来批处理的）。

---

## 十二、`inputs` 为什么既能 `inputs["input_ids"]` 又能 `inputs.input_ids`

`BatchFeature` 是 `UserDict` 的子类，两个 dunder 方法把它变成"字典 + 对象"的双身份。

<!-- src: feature_extraction_utils.py -->
```python
    def __getitem__(self, item: str) -> Any:
        """
        If the key is a string, returns the value of the dict associated to `key` ('input_values', 'attention_mask',
        etc.).
        """
        if isinstance(item, str):
            return self.data[item]
        else:
            raise KeyError("Indexing with integers is not available when using Python based feature extractors")

    def __getattr__(self, item: str):
        try:
            return self.data[item]
        except KeyError:
            raise AttributeError
```

**读法**：
- `__getitem__` 只接受**字符串**键，整数下标会抛 `KeyError` —— 这是刻意的：
  它防止你把 `inputs[0]` 误当成"第一条样本"（在旧的 feature extractor 里
  整数下标曾经有别的含义）。
- `__getattr__` 是"属性式取值"的实现：`inputs.pixel_values` → `self.data["pixel_values"]`。
  注意它在 `KeyError` 时抛的是 `AttributeError` 而不是 `KeyError` ——
  因为 Python 的属性协议只认 `AttributeError`，抛别的会让 `hasattr()` 直接崩。
- `__getattr__` 只在**常规属性查找失败后**才被调用，所以 `inputs.data`、`inputs.items()`
  这些真实属性/方法不会走到这里。这是个容易踩的坑：**字典里有一个 key 叫 `data` 就取不到**。
- 另一半在 `__init__` 与 `convert_to_tensors`：构造时就按 `tensor_type` 把内层列表转成张量，
  所以 `processor(..., return_tensors="pt").input_ids` 拿到的是张量而不是 list。
  还有两个细节值得记：`to()` 只把**浮点**张量搬设备/改 dtype（避免把 `LongTensor`
  的 `input_ids` 变成浮点），`convert_to_tensors` 会跳过非数组值（字符串、list of str）。
  这两条合起来解释了"为什么 `video_metadata` 那种对象能安全地躺在同一个字典里"。
- 与 L1-06 的呼应：`FeatureExtractionMixin`（同一文件的另一个类）是 `BatchFeature`
  的生产者，处理器链路上所有子处理器返回的都是它。

---

## 十三、与后面课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| `smart_resize` 的 `fit_within_budget` 二分是怎么收敛的 | 本课第八节 + L2-05 |
| 同一个 `smart_resize` 为什么在三份文件里各抄了一遍 | L2-05（resize 家族的张量级实现） |
| `pixel_values` 的 `(总 patch 数, 1176)` 怎么被视觉塔吃进去 | L2-07 |
| `image_grid_thw` 里那个 `t` 维在视觉塔里做什么 | L2-07（temporal patch 与轴向 RoPE） |
| `get_placeholder_mask` 怎么把视觉向量填回占位符位置 | L2-07 |
| `AutoProcessor` 怎么找到 `Glm5NextProcessor` | L1-06（三级解析） |
| `inputs_embeds` 里视觉 token 的替换发生在哪 | L0-01（入口二选一）+ L2-07 |

**一句话总结**：

> 一次图像对话 = chat template 写 **1 个** `<|image|>` → `replace_image_token` 按
> `image_grid_thw.prod() // merge_size²` 展开成 **N 个** → 与文本一起 tokenize 成**同一条**序列
> → 再挂上 `pixel_values` 与 `image_grid_thw`，交给 `BatchFeature` 打包；
> **处理器只决定"占几个位置"，视觉塔才决定"每个位置是什么向量"。**
