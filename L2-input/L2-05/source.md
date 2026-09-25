<!-- glm5-coverage
image_transforms.py
image_utils.py
video_processing_utils.py
video_utils.py
vision_utils.py
audio_utils.py
-->

# L2-05 · 图像变换、视频与通用视觉工具 — 源文件

**这一课回答一件事：像素是怎么被压成 token 的。**

第 2 层一路走来，L2-01 ~ L2-04 讲的是"处理器怎么装配、后端怎么选"。
这一课下沉到最下面那一层：**张量级的形状变换**。六个文件、5799 行，
真正需要记住的只有三步 ——

> **重采样（resize）→ 切块（patchify）→ 合并（merge）**。
> 三步之后的输出就是视觉 token，它会和文本 token 被丢进**同一条序列**。

| 文件 | 行数 | 本课引用块 |
|---|---|---|
| `image_transforms.py` | 1102 | 7 |
| `image_utils.py` | 1076 | 2 |
| `video_processing_utils.py` | 795 | 2 |
| `video_utils.py` | 927 | 3 |
| `vision_utils.py` | 379 | 2 |
| `audio_utils.py` | 1520 | 1 |

> **行数订正**：作业书给的是 1103 / 1077 / 796 / 928 / 380 / 1521。
> 实测 `wc -l` 分别是 1102 / 1076 / 795 / 927 / 379 / 1520 —— **每个文件都少 1**。
> 作业书的数字像是"行号最大值 + 1"（把最后一个换行符也算成一行）。
> 本课一律以实测为准；作业书正文里那些 `L46` / `L89` 这样的**定位行号是对的**，
> 逐个核对过 `def` / `class` 都落在标注的那一行上。

**同样实测出来的两个数字**（本课验收点就建立在这上面）：

```text
图像侧（GLM-5 图像处理器默认值，从源码里读出来的）
    image_size          = 336          14 的整数倍，也是 28 的整数倍
    patch_size          = 14
    spatial_merge_size  = 2            → merge_length = 2**2 = 4
    temporal_patch_size = 2            图像路径下这一维固定为 1（单帧）

    336 / 14 = 24                      → grid_h = grid_w = 24
    grid_h * grid_w     = 24 * 24 = 576 个 patch
    576 / 4             = 144          个视觉 token  ← 本课要能自己算出来

视频侧
    temporal_patch_size = 2            grid_t = num_frames / temporal_patch_size
    帧数必须是 temporal_patch_size 的整数倍，否则 patchify 会补齐（见第九节）
```

> 上面这段是**示意性的汇总**，用 `text` 块标出 —— 它不是源文件的逐字引用，
> 因此不参与保真校验。下面每一段 `python` 块都是逐字引用。

**读法里的行号口径**：本节之后所有「块内第 N 行」都从**该引用块的第一行**开始数
（引用块的起始行号就是作业书大纲里的 `L` 编号，可以直接对回去）；
写「源码第 N 行」的才是文件里的绝对行号。两种口径混着看会数错，这里统一声明。

**跨课呼应**：回顾 L0-01 —— 那里 `input_ids (batch, seq)` 里每一个整数都对应
词表里的一行向量；这一课要说明的是，**视觉 token 走了另一条完全不同的路进来**，
最后却落在同一个 `seq` 轴上。L2-06 / L2-07 接着讲这些 token 怎么真正拼到一起。

---

## 一、三个文件为什么要一起看

它们的分工不是按"功能"切的，而是按**数据在什么形态上被处理**切的：

```text
                    PIL / np.ndarray / torch.Tensor
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
  image_transforms.py     image_utils.py          video_processing_utils.py
  「函数式」numpy 版       「类方法」PIL 版         「视频版」同样的三件套
   resize / normalize      resize / normalize       sample_frames
   center_crop / pad       center_crop / rotate     / resize / crop
        │                        │                video_utils.py：解码后端
        └────────────────────────┴────────────────────────┘
                                 │
                        patchify + merge
                                 │
                        vision_utils.py（位置编号 / 分段）
                                 │
                        audio_utils.py（另一条模态，同样的"先变换再拼段"）
```

**读法**：这一层最容易被读糊的地方是**同名函数有两份**。
`image_transforms.resize` 与 `ImageFeatureExtractionMixin.resize` 名字一样、
签名很像，但一份处理 `np.ndarray`、一份处理 `PIL.Image.Image`。
第二节会把两个 resize 并排看，差别正好把"张量级实现"这句话讲清楚。

---

## 二、★ resize：两个后端，两种代价

先看**张量级**那份。它处理 `np.ndarray`，但内部还是要绕一趟 PIL：

<!-- src: image_transforms.py -->
```python
    # For all transformations, we want to keep the same data format as the input image unless otherwise specified.
    # The resized image from PIL will always have channels last, so find the input format first.
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)
    data_format = input_data_format if data_format is None else data_format

    # To maintain backwards compatibility with the resizing done in previous image feature extractors, we use
    # the pillow library to resize the image and then convert back to numpy
    do_rescale = False
    if not isinstance(image, PIL.Image.Image):
        do_rescale = _rescale_for_pil_conversion(image)
        image = to_pil_image(image, do_rescale=do_rescale, input_data_format=input_data_format)
    height, width = size
    # PIL images are in the format (width, height)
    resized_image = image.resize((width, height), resample=resample, reducing_gap=reducing_gap)

    if return_numpy:
        resized_image = np.array(resized_image)
        # If the input image channel dimension was of size 1, then it is dropped when converting to a PIL image
        # so we need to add it back if necessary.
        resized_image = np.expand_dims(resized_image, axis=-1) if resized_image.ndim == 2 else resized_image
        # The image is always in channels last format after converting from a PIL image
        resized_image = to_channel_dimension_format(
            resized_image, data_format, input_channel_dim=ChannelDimension.LAST
        )
        # If an image was rescaled to be in the range [0, 255] before converting to a PIL image, then we need to
        # rescale it back to the original range.
        resized_image = rescale(resized_image, 1 / 255) if do_rescale else resized_image
    return resized_image
```

**读法**：
- 块内第 5 行 `data_format = input_data_format if data_format is None else data_format`
  是本层的**总纲**：变换不改变通道顺序，除非你显式要求。
  后面所有函数（`normalize` / `center_crop` / `pad`）都有这一行。
- 块内第 10~12 行是这个函数的**代价所在**：numpy → PIL 是有损往返
  （PIL 只能存 uint8），所以要先判断要不要 rescale；
  块内倒数第 2 行再把 `[0, 255]` 除回 `[0, 1]`。**这一步的记账不能丢**，
  丢了图像数值量纲就错了，而且不会报错。
- `do_rescale` 这个名字有歧义：它在这里的含义是"为了转 PIL **临时**放大到 0~255"，
  不是处理器的 `do_rescale`（那是 1/255 归一化到 [0,1]）。同名不同义。

再看**PIL 版**。它干脆假设输入已经是 PIL 图像，于是省掉整个往返：

<!-- src: image_utils.py -->
```python
        if not isinstance(image, PIL.Image.Image):
            image = self.to_pil_image(image)

        if isinstance(size, list):
            size = tuple(size)

        if isinstance(size, int) or len(size) == 1:
            if default_to_square:
                size = (size, size) if isinstance(size, int) else (size[0], size[0])
            else:
                width, height = image.size
                # specified size only for the smallest edge
                short, long = (width, height) if width <= height else (height, width)
                requested_new_short = size if isinstance(size, int) else size[0]

                if short == requested_new_short:
                    return image
```

**读法**：
- 没有 `do_rescale`、没有 `to_channel_dimension_format`。
  **代价换了个地方付**：调用者必须先把张量转成 PIL（`to_pil_image`）。
- `size` 是 int 时 `default_to_square=True` 走 `(size, size)`；
  `False` 时按短边对齐、长边等比 —— 这正是 torchvision `Resize` 的语义
  （`default_to_square` 这个名字就是从那里来的）。
- `if short == requested_new_short: return image` 是个提前返回：
  尺寸已经对上了就别做一次无意义的插值。**重采样是纯损失**，能不做就不做。

### 裁剪：`center_crop` 里的整数除法

<!-- src: image_transforms.py -->
```python
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)
    output_data_format = data_format if data_format is not None else input_data_format

    # We perform the crop in (C, H, W) format and then convert to the output format
    image = to_channel_dimension_format(image, ChannelDimension.FIRST, input_data_format)

    orig_height, orig_width = get_image_size(image, ChannelDimension.FIRST)
    crop_height, crop_width = size
    crop_height, crop_width = int(crop_height), int(crop_width)

    # In case size is odd, (image_shape[0] + size[0]) // 2 won't give the proper result.
    top = (orig_height - crop_height) // 2
    bottom = top + crop_height
    # In case size is odd, (image_shape[1] + size[1]) // 2 won't give the proper result.
    left = (orig_width - crop_width) // 2
    right = left + crop_width

    # Check if cropped area is within image boundaries
    if top >= 0 and bottom <= orig_height and left >= 0 and right <= orig_width:
        image = image[..., top:bottom, left:right]
        image = to_channel_dimension_format(image, output_data_format, ChannelDimension.FIRST)
        return image
```

**读法**：
- 块内第 5 行先把通道轴换到最前面（`(C, H, W)`），裁完再换回去。
  **统一在一套坐标系里做几何运算**，是这个文件的另一个总纲。
- `top = (orig_height - crop_height) // 2` —— 用 `//` 而不是 `/`，
  奇数差时左边少切一个像素。原地注释写明了原因：`(H + h) // 2` 写法在奇数时不对。
- 块内第 18 行的边界检查（源码第 500 行）是**快路径**：够大就直接切片，
  不做任何拷贝以外的动作。
- 不够大时（本块之后、源码第 505 行起）会**补零再裁**，所以 `center_crop`
  的返回尺寸永远等于 `size` —— 这是一个"不会失败"的函数，
  这一点在拼 batch 时很重要。

---

## 三、★ normalize：为什么第一步是 `astype(np.float32)`

<!-- src: image_transforms.py -->
```python
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)

    channel_axis = get_channel_dimension_axis(image, input_data_format=input_data_format)
    num_channels = image.shape[channel_axis]

    # We cast to float32 to avoid errors that can occur when subtracting uint8 values.
    # We preserve the original dtype if it is a float type to prevent upcasting float16.
    if not np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32)

    if isinstance(mean, Collection):
        if len(mean) != num_channels:
            raise ValueError(f"mean must have {num_channels} elements if it is an iterable, got {len(mean)}")
    else:
        mean = [mean] * num_channels
    mean = np.array(mean, dtype=image.dtype)
```

**读法**：
- 块内第 7~9 行是本课最便宜也最重要的一处：`uint8` 做 `image - mean` 会**回绕**
  （`3 - 5 = 254`），所以必须先 cast 到 float32。
- 注释第二句同样关键：**已经是浮点类型就不动它**。
  否则 `float16` 会被悄悄提升成 `float32`，显存翻倍而没人发现。
- 块内第 12~19 行把 `mean` / `std` 广播成"每通道一个数"。
  注意 `mean = [mean] * num_channels` —— 标量写法是被**显式**支持的，
  但前提是它和 `num_channels` 对得上（对不上就 raise，不静默）。
- 真正的算式在源码第 436~439 行（见下），它按数据格式分成两支。

<!-- src: image_transforms.py -->
```python
    if input_data_format == ChannelDimension.LAST:
        image = (image - mean) / std
    else:
        image = ((image.T - mean) / std).T

    image = to_channel_dimension_format(image, data_format, input_data_format) if data_format is not None else image
    return image
```

**读法**：
- `channels_last` 时 `mean` 直接对上最后一维，一行搞定。
- `channels_first` 时先 `.T` 把通道轴换到末尾、算完再 `.T` 回来。
  **这不是为了省事，是为了让广播规则简单**：两种格式最终都归约成
  "最后一维是通道"这一种情况，代码只有一处算式。
- 这些函数的输入输出形状**完全一样**（`(C,H,W)` 进 `(C,H,W)` 出），
  变的是数值范围 —— 和下面要讲的 patchify 形成对比，那个是**改形状**的。

---

## 四、切块：`divide_to_patches` 的签名与循环

`image_transforms.py` 提供的是**通用工具**：两个步进循环，返回一个 patch 列表。
可读性优先，代价是**它不带 batch 维**。先看签名与 docstring：

<!-- src: image_transforms.py -->
```python
def divide_to_patches(
    image: Union[np.ndarray, "torch.Tensor"], patch_size: int | tuple[int, int]
) -> list[Union[np.ndarray, "torch.Tensor"]]:
    """
    Divides an image into patches of a specified size.

    Args:
        image (`np.array | "torch.Tensor"`):
            The input image.
        patch_size (`int` or `tuple[int, int]`):
            The size of each patch. If an int, patches are square. If a tuple,
            it is interpreted as `(patch_height, patch_width)`.
    Returns:
        list: A list of `np.array | "torch.Tensor"` representing the patches.
    """
```

**读法**：
- `patch_size` 既接受 `int` 也接受 `(h, w)` 元组 —— 全库统一的"尺寸参数"惯例
  （`SizeDict` / `resize` 的 `size` 都一样宽进严出）。
- docstring 里 `Returns: list` 是**唯一的形状线索**：它不承诺堆叠，
  所以调用方必须自己处理"一批形状不同的 patch"。
- `get_image_size(image, channel_dim=ChannelDimension.FIRST)` 先把
  `(H, W)` 取出来，且**显式指定**了通道在前 —— 不靠推断，
  因为这里马上要用 `image[..., i:i+patch_h, j:j+patch_w]` 做切片。

循环体紧接在 docstring 之后（同一个函数的后半段）：

<!-- src: image_transforms.py -->
```python
    patch_h, patch_w = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
    patches = []
    height, width = get_image_size(image, channel_dim=ChannelDimension.FIRST)
    for i in range(0, height, patch_h):
        for j in range(0, width, patch_w):
            patch = image[..., i : i + patch_h, j : j + patch_w]
            patches.append(patch)

    return patches
```

**读法**：
- 两个 `range(..., patch_h/patch_w)` 的步进循环，**行优先**遍历，
  所以返回顺序是光栅序（第五节的 permute 版会打破这个顺序）。
- 块内第 5 行 `image[..., i : i + patch_h, j : j + patch_w]` ——
  前导省略号让它可以同时吃 3 维 `(C,H,W)` 和 4 维 `(B,C,H,W)` 输入，
  **不需要为 batch 写第二个分支**。
- 断言只有 `grid_h = height // patch_h` 这一种除法，**没有余数一说**：
  除不尽的部分被直接丢掉。这就是为什么前面必须先 resize 到 patch 的整数倍。
- 返回 `list` 而不是堆叠的张量：要用它做 batch 就得自己 `torch.stack`，
  这也是为什么模型处理器不用它。

---

## 五、★ 合并前的重排：`patchify` 的 reshape + permute

视觉 token 的真正来源是这里。它不在 `image_transforms.py` 里 ——
模型的切块逻辑写在处理器里，用的是纯张量操作：

<!-- src: models/glm5_next/image_processing_glm5_next.py -->
```python
    def patchify(
        self,
        images: "torch.Tensor",
        patch_size: int,
        merge_size: int,
        temporal_patch_size: int,
    ) -> tuple["torch.Tensor", int, int]:
        """Patchifies each image into flat layout of shape (`seq_len`, `patch_dim`) so we can concat dynamically shaped pixels."""
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

> **边界声明**：本课一共 18 个引用块，其中 **15 个**来自上面 coverage 声明的六个文件，
> 另外 3 个（`image_processing_glm5_next.py`、`processing_glm5_next.py`、
> `video_processing_glm5_next.py`）属于 **L2-06 的覆盖域，不计入本课覆盖率**。
> 引用它们是因为：本课这六个文件里没有任何一处真正实现过带 `merge_size` 的
> 切块与占位符计数，而那两件事恰好是本课两个验收点的落点。

**读法**（逐行推）：
- 块内第 14 行把 `H × W` 重排成 `(H/m, m, p) × (W/m, m, p)` —— 一共 8 维。
  **`merge_size` 维被放在 `patch_size` 的左边**，所以同一个 2×2 块里的
  两个横向相邻 patch 在展平后是挨着的。
- 块内第 24 行 `patches.permute(0, 2, 5, 3, 6, 1, 4, 7)` 把 8 维重排成
  `(B, H/m, W/m, m, m, C, p, p)`。**先块、后块内** —— 这就是"合并"。
  到此为止，展平出来的第 1 维是 `(H/m)*(W/m)`，恰好等于 token 数。
- 块内第 26~34 行 `unsqueeze(6).expand(temporal_patch_size)` 给每一块
  再插一个长度 `temporal_patch_size` 的维度。**图像路径下这一维是 1**
  （图像处理器调用时传的就是它，单帧），所以对图像来说这是一次无损复制；
  对视频来说它才是真的时间维。**同一段代码服务两种模态 —— 这是刻意的。**
- 和 `divide_to_patches` 对比：那边是"取子块"，这边是"重排内存布局"，
  后者才能在 GPU 上一次做完一批图。

---

## 六、算一遍 336×336 的 token 数

现在把前面几步接起来。假设输入正好是 336×336、patch 14、merge 2：

```text
输入                    (3, 336, 336)
  grid_h = 336 // 14 = 24
  grid_w = 336 // 14 = 24
                        │
patchify（不改变像素个数）
  patches per image     = grid_h * grid_w = 24 * 24 = 576
  patch 展平后的宽度     = C * temporal_patch_size * p * p
                        = 3 * 1 * 14 * 14 = 588     ← 单帧，temporal 维记 1
                        │
merge（spatial_merge_size = 2）
  merge_length          = merge_size ** 2 = 4
  视觉 token 数          = 576 / 4  = 144
  merge 后 LLM 侧的网格   = (24/2, 24/2) = (12, 12) → 12*12 = 144
```

**为什么除 4 而不是别的数**：`merge_size=2` 表示 2×2=4 个 patch 合成 1 个 token。
`permute` 那一步已经把 4 个 patch 排到了连续位置，所以"合并"不需要真的做加法，
只是**换个分组方式**（后续由视觉编码器的 merger 层把 4×588 投到 hidden）。

`576 / 4 = 144` 这个除法在处理器里是显式写着的 —— 它决定了要往 prompt 里
塞多少个占位符，也就是文本序列要为视觉预留多少个位置：

<!-- src: models/glm5_next/processing_glm5_next.py -->
```python
    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        merge_length = self.image_processor.merge_size**2
        num_image_tokens = image_inputs["image_grid_thw"][image_idx].prod() // merge_length
        return self.image_token * num_image_tokens
```

> **边界声明**：同第五节 —— 此块属于 L2-06 覆盖域，**不计入本课覆盖率**。

**读法**：
- `image_grid_thw.prod()` = `t * h * w`。图像时 `t = 1`，所以就是 `576`。
- `// merge_length` 就是上面那个 4。**一个除法决定了序列长度**。
- 返回的是 `self.image_token` 重复 `num_image_tokens` 次 ——
  一个字符串。所以**视觉在文本序列里的"占位"是一串重复的同一个 token**，
  真正的向量替换发生在更后面（L2-06 / L2-07）。
- 记住这条链：`grid_thw → prod() → // merge_length → 占位符个数`。
  它和 L0-01 里 `input_ids (batch, seq)` 的 `seq` 直接相接。
- 紧跟着的 `replace_video_token`（源码第 68 行起）是同一件事的视频版，
  但它多除一次帧数：`prod() // merge_length // num_frames` ——
  得到的是**每帧**的 token 数，再按帧拼时间戳。这就是第九节那条等式的来源。

---

## 七、批量与形状分组：为什么要有 `group_images_by_shape`

一张图算完了，一批图怎么办？`preprocess` 里对图片做的是
"按形状分组 → 组内堆叠 → 一起算 → 再还原顺序"。核心是下面这段：

<!-- src: image_transforms.py -->
```python
    # If disable grouping is not explicitly provided, we favor disabling it if the images are on CPU, and enabling it otherwise.
    if disable_grouping is None:
        device = _get_device_from_images(images, is_nested)
        disable_grouping = device.type == "cpu"

    if disable_grouping:
        grouped_images_index = {key: (key, 0) for key, _ in _iterate_items(images, is_nested)}
        if is_nested:
            grouped_images_index["_num_sublists"] = len(images)

        grouped_images = {key: img.unsqueeze(0) for key, img in _iterate_items(images, is_nested)}
        paired_grouped_values = [
            dict.fromkeys(grouped_images, None)
            if paired_list is None
            else {key: [item] for key, item in _iterate_items(paired_list, is_nested)}
            for paired_list in paired_inputs
        ]

        return grouped_images, *paired_grouped_values, grouped_images_index
```

**读法**：
- 块内第 4~5 行是本函数唯一的**性能决策**：CPU 上不做分组。
  注释写明了依据（PR #38157）—— CPU 上 `torch.stack` 的拷贝比省下的
  批处理收益还贵。GPU 上才分组。**同一个函数在两种设备上走不同分支**，
  这是"实测出来的默认值"，不是理论推导。
- `disable_grouping` 分支里每个图各自 `unsqueeze(0)` 成 batch=1，
  索引表退化成 `{i: (i, 0)}` —— **结构一致，代价为零**。
- `paired_inputs` 用来让"跟着图片一起分组的附属数据"（比如注记）
  走同一条重排路径。这在 `_preprocess` 里就是把 `image_grid_thw` 对齐地重排。
- 关键约定：**分组只影响计算顺序，不影响输出顺序**。

还原那一步没有技巧，全是查表 —— 注意它用 `is_nested` 分成两支：

<!-- src: image_transforms.py -->
```python
    if not is_nested:
        return [
            processed_images[grouped_images_index[i][0]][grouped_images_index[i][1]]
            for i in range(len(grouped_images_index))
        ]

    return _reconstruct_nested_structure(grouped_images_index, processed_images)
```

**读法**：
- 非嵌套时就是一行列表推导：`processed_images[shape][组内第几个]`。
  两个索引都在 `grouped_images_index` 里现成放着。
- 嵌套时交给 `_reconstruct_nested_structure`，因为要保留"有空子列表"
  这种结构（`[[], [image]]`）—— 展平再拼回去会丢信息。
- ★ 这一对函数（`group_images_by_shape` / `reorder_images`）**必须成对使用**，
  单看一个会以为是错的：前者打乱顺序，后者负责还原。

---

## 八、视频 = 同一套变换 + 一个采样决策

视频处理器复用图像后端，只在前面多了一步：**先决定要哪些帧**。

<!-- src: video_utils.py -->
```python
def get_uniform_frame_indices(total_num_frames: int, num_frames: int | None = None):
    """
    Creates a numpy array for uniform sampling of `num_frame` frames from `total_num_frames`
    when loading a video.

    Args:
        total_num_frames (`int`):
            Total number of frames that a video has.
        num_frames (`int`, *optional*):
            Number of frames to sample uniformly. If not specified, all frames are sampled.

    Returns:
        np.ndarray: np array of frame indices that will be sampled.
    """
    if num_frames is not None:
        indices = np.arange(0, total_num_frames, total_num_frames / num_frames).astype(int)
    else:
        indices = np.arange(0, total_num_frames).astype(int)
    return indices
```

**读法**：
- `np.arange(0, total, total / num_frames)` —— 步长**不取整**，最后才 `.astype(int)`。
  这样最后一帧尽量贴近末尾，比 `linspace(0, N-1, n)` 更接近"均匀覆盖整段视频"。
- `num_frames=None` 时退化成"全都要"，**没有采样**。
- 抽到的索引是**原视频里的绝对帧号**（0, 5, 10, …），不是 0..n-1 ——
  这一点在下面算时间戳时会用到。

同一个决策在视频处理器里还有一份 torch 实现：

<!-- src: video_processing_utils.py -->
```python
    def sample_frames(
        self,
        metadata: VideoMetadata,
        num_frames: int | None = None,
        fps: int | float | None = None,
        **kwargs,
    ):
        """
        Default sampling function which uniformly samples the desired number of frames between 0 and total number of frames.
        If `fps` is passed along with metadata, `fps` frames per second are sampled uniformly. Arguments `num_frames`
        and `fps` are mutually exclusive.

        Args:
            metadata (`VideoMetadata`):
                Metadata of the video containing information about total duration, fps and total number of frames.
            num_frames (`int`, *optional*):
                Maximum number of frames to sample. Defaults to `self.num_frames`.
            fps (`int` or `float`, *optional*):
                Target frames to sample per second. Defaults to `self.fps`.

        Returns:
            np.ndarray:
                Indices to sample video frames.
        """
        if fps is not None and num_frames is not None:
            raise ValueError(
                "`num_frames`, `fps`, and `sample_indices_fn` are mutually exclusive arguments, please use only one!"
            )

        num_frames = num_frames if num_frames is not None else self.num_frames
        fps = fps if fps is not None else self.fps
        total_num_frames = metadata.total_num_frames

        # If num_frames is not given but fps is, calculate num_frames from fps
        if num_frames is None and fps is not None:
            if metadata is None or metadata.fps is None:
                raise ValueError(
                    "Asked to sample `fps` frames per second but no video metadata was provided which is required when sampling with `fps`. "
                    "Please pass in `VideoMetadata` object or use a fixed `num_frames` per input video"
                )
            num_frames = int(total_num_frames / metadata.fps * fps)

        if num_frames > total_num_frames:
            raise ValueError(
                f"Video can't be sampled. The `num_frames={num_frames}` exceeds `total_num_frames={total_num_frames}`. "
            )

        if num_frames is not None:
            indices = torch.arange(0, total_num_frames, total_num_frames / num_frames).int()
        else:
            indices = torch.arange(0, total_num_frames).int()
        return indices
```

**读法**：
- 块内第 1~3 行是"三重默认"：显式参数 → 类属性 → metadata。
  `BaseVideoProcessor` 上那一排 `None` 类属性（`num_frames` / `fps` / …）
  就是为了让子类只覆盖需要的那几个。
- 块内第 7~14 行：`fps` 和 `num_frames` **互斥**，但允许"只给 fps"，
  由 `int(total_num_frames / metadata.fps * fps)` 反推出帧数。
  这也是为什么视频采样**必须有 metadata** —— 没有 fps 就算不出目标帧数。
- 块内第 16~19 行是**硬约束检查**：要的帧数超过总帧数直接 raise，
  不做静默截断。宁可在处理器里报错，也不要在模型里变成形状玄学。
- 块内第 20 行 `torch.arange(0, N, N / n).int()` 与 `video_utils` 那一版
  **是同一套逻辑的两份实现**（一份 torch、一份 numpy）—— 这是本课第二次
  看到"同名不同源"，和第二节的两个 resize 是同一类问题。

帧索引留下来干什么？时间戳是后面 mrope 要用的：

<!-- src: video_utils.py -->
```python
    @property
    def timestamps(self) -> list[float]:
        "Timestamps of the sampled frames in seconds."
        if self.fps is None or self.frames_indices is None:
            raise ValueError("Cannot infer video `timestamps` when `fps` or `frames_indices` is None.")
        return [frame_idx / self.fps for frame_idx in self.frames_indices]

    @property
    def sampled_fps(self) -> float:
        "FPS of the sampled video."
        if self.frames_indices is None or self.total_num_frames is None or self.fps is None:
            return self.fps or 24
        return len(self.frames_indices) / self.total_num_frames * self.fps
```

**读法**：
- `VideoMetadata` 带着原始 `fps` 和**实际抽到的** `frames_indices`，
  于是 `timestamps` 可以精确还原每一帧在原视频里的秒数 ——
  这是把"帧"重新变成"时间"的唯一途径。
- `sampled_fps` 是**采样后**的有效帧率：`len(indices) / total * fps`。
  一个 30fps 的视频抽 16 帧，它的 `sampled_fps` 远小于 30，
  **模型看到的时间密度和原始视频不同** —— 这个数字必须显式留档。
- 两个属性都做了 `None` 保护并给出可读的错误信息。
  这不是防御性编程，是**让"没给 metadata"这个错误停在处理器层**。

---

## 九、★ 视频专有的一步：temporal patch

视频切块比图像多一维：`(B, T, C, H, W)`。多出来的 `T` 会被折进
**patch 的宽度**，而不是变成新的 token：

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

> **边界声明**：此块属于 L2-06 覆盖域，**不计入本课覆盖率**。

**读法**：
- `-num_frames % temporal_patch_size` 是**向上取整到倍数**的惯用写法：
  8 % 2 = 0（不用补）；7 % 2 = 1（补 1 帧）。
- 补的方式是**重复最后一帧**（`videos[:, -1:]`），不是补零。
  补零会在时间维上造出一段"黑场"，重复最后一帧则近似"这一瞬间没变化"。
- `grid_t = num_frames // temporal_patch_size`：**时间是一维网格，不是 token 数**。
  2 帧合成 1 个 grid_t，这 2 帧的信息被拼进同一个 patch 的特征维里。
- 所以 `grid_thw` 里的 `t` 是 **grid_t**，不是原始帧数：
  16 帧的视频，`t = 16 / 2 = 8`。

### 这对帧数意味着什么（本课验收点 2）

`temporal_patch_size = 2` 要求 `num_frames` 是 2 的整数倍。
在 GLM-5 的采样函数里这条被**显式**兜住：

```text
        if len(uniq) & 1:
            uniq.append(uniq[-1])
```

上面这两行是 `Glm5NextVideoProcessor.sample_frames` 的第 234–235 行，
**同属不计入覆盖率的 L2-06 文件**，所以写在 `text` 块里（不是逐字引用块）。
它做的事和 `patchify` 的补齐是同一个意思：**奇数帧就复制最后一帧**。

把整条链写出来：

```text
总帧数 total_num_frames = 240，目标 16 帧
  采样：arange(0, 240, 240/16) = 0,15,30,...,225     → 16 个索引（偶数，OK）
  若目标 15 帧：采样得 15 个索引 → 奇数 → 复制最后一帧 → 16
  patchify：num_frames=16，temporal_patch_size=2
           grid_t = 16 / 2 = 8，grid_h = grid_w = 24
  视觉 token 数（视频）= grid_t * grid_h * grid_w // merge_length
                      = 8 * 24 * 24 // 4 = 1152
  每帧折算              = 1152 / 16 = 72
```

> 视频的 token 数算法和图像**不是同一个式子**：图像用
> `thw.prod() // merge_length`（t=1），视频用
> `thw.prod() // merge_length // num_frames` —— 后者多除一次帧数，
> 得到的是**每帧**的 token 数，最后再乘回帧数。

---

## 十、`vision_utils.py`：把"因图而异"的张量预先算出来

切完块之后，视觉编码器还需要一批**依赖 `grid_thw` 的索引张量**：
每个 token 的位置编号、每张图的注意力分段边界。
它们的共同问题是：**形状随输入变化，且用了 `torch.compile` 不认识的算子**
（`repeat_interleave` / `.tolist()` / 循环）。所以这个文件把它们统一成
"算一次、存起来、下次 pop 出来"的形式。

<!-- src: vision_utils.py -->
```python
    if kwargs is not None and (cu_seqlens := kwargs.pop("cu_seqlens", None)) is not None:
        return cu_seqlens
    dtype = grid_thw.dtype if torch.jit.is_tracing() else torch.int32
    if merge_temporal:
        seqlens = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
    else:
        seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    return F.pad(seqlens.cumsum(dim=0, dtype=dtype), (1, 0), value=0)
```

**读法**：
- 块内第 1 行是贯穿整个文件的**模式**：`kwargs.pop("cu_seqlens", None)`。
  有预算好的就直接返回，并且**顺手把它从 kwargs 里删掉**
  （`pop` 而不是 `get`）—— 这个副作用是有意的：调用方不必再清理。
- `merge_temporal` 决定"一张图的 `t` 帧算 1 段还是 `t` 段"：
  `False` → `repeat_interleave(h*w, t)`，每帧独立一段；
  `True` → `t*h*w`，整段视频当一段。**同一份权重、两种注意力范围**。
- `F.pad(cumsum, (1, 0), value=0)` 把前缀和补上开头的 0，
  得到 `num_segments + 1` 个边界（flash-attention 的 `cu_seqlens` 约定）。

位置编号同理，而且能直接看出 merge 分组：

<!-- src: vision_utils.py -->
```python
    device = grid_thw.device
    if isinstance(spatial_merge_size, int):
        spatial_merge_size = torch.tensor([spatial_merge_size], device=device).expand(len(grid_thw))

    position_ids = []
    for (t, h, w), merge_size in zip(grid_thw.tolist(), spatial_merge_size.tolist()):
        hpos_ids, wpos_ids = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        block_shape = (h // merge_size, merge_size, w // merge_size, merge_size)
        hpos_ids = hpos_ids.reshape(block_shape).transpose(1, 2).flatten()
        wpos_ids = wpos_ids.reshape(block_shape).transpose(1, 2).flatten()
        if include_temporal:
            tpos_ids = torch.arange(t, device=device).repeat_interleave(h * w)
            position_ids.append(torch.stack([tpos_ids, hpos_ids.repeat(t), wpos_ids.repeat(t)], dim=-1))
        else:
            position_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

    return torch.cat(position_ids, dim=0)
```

**读法**：
- 块内第 2 行 `torch.meshgrid(arange(h), arange(w), indexing="ij")` 造出
  每个 patch 的 `(行号, 列号)`。
- 块内第 12~13 行是关键：`reshape(block_shape).transpose(1, 2).flatten()` ——
  **和 patchify 的 permute 完全同一个意图**：把 2×2 块内的 4 个位置
  排到连续位置。所以 `position_ids` 的顺序和 token 的顺序是**一致的**。
  这两处必须同序，否则位置会整体错位 —— 而且不会报错。
- `include_temporal=True` 时前置一列时间索引（minimax_m3_vl 这类要转 T 轴）；
  默认 `False` 只给 `(h, w)` 两轴，视频则把这两轴 `repeat(t, 1)`。
- 再次注意块内第 2~3 行：`spatial_merge_size` 允许传 int 或**每图一个值的张量**，
  因为不同图可能有不同 merge 配置（ragged batch）。

---

## 十一、音频走的是同一条路

`audio_utils.py` 离视觉最远，但它做的**是同一件事**：
把原始信号切成固定长度的帧、投到一个"感知上均匀"的基上、
再取对数压缩动态范围。看它的最后一步就明白了：

<!-- src: audio_utils.py -->
```python
    if reference <= 0.0:
        raise ValueError("reference must be greater than zero")
    if min_value <= 0.0:
        raise ValueError("min_value must be greater than zero")

    reference = max(min_value, reference)

    spectrogram = np.clip(spectrogram, a_min=min_value, a_max=None)
    spectrogram = 10.0 * (np.log10(spectrogram) - np.log10(reference))

    if db_range is not None:
        if db_range <= 0.0:
            raise ValueError("db_range must be greater than zero")
        spectrogram = np.clip(spectrogram, a_min=spectrogram.max() - db_range, a_max=None)

    return spectrogram
```

**读法**：
- 对应到视觉侧：`10*log10` ↔ `normalize` 的 `(x-mean)/std`，
  都是**把量纲压到模型好学的范围**。
- `np.clip(a_min=min_value)` 先夹住再取对数 —— 和视觉侧
  `astype(np.float32)` 一样，是"防数值事故"的一步，不是数学的一部分。
- 同一文件里 mel 那一步是
  `np.tensordot(spectrogram, mel_filters.T, axes=([2], [1]))`，
  等价于视觉侧的 merge：**用一个预定义的矩阵把 `N` 个频率格压到 `M` 个**。
- 区别在于时间维：音频不 merge 时间（`length` 保持不变），
  视频却把 2 帧折进 patch。**同样是"压缩"，两条模态选择了不同的轴。**
- 这一节也解释了本课为什么覆盖 `audio_utils.py`：第 2 层讲的是
  "输入流水线"，而这条流水线的骨架在三种模态上是**同一套**。

---

## 十二、形状总表：一张图从文件到 `seq`

```text
步骤            函数/位置                                  形状
--------------------------------------------------------------------------------
0  读入         load_image / load_video                    (3, 336, 336)
1  重采样       image_transforms.resize                    (3, 336, 336)  ← 尺寸对齐
2  归一化       image_transforms.normalize                 (3, 336, 336)  ← 只动数值
3  切块+合并    patchify                                    (576, 588)
4  折时间维     （图像 t=1 / 视频 grid_t = T/temporal）      (t*h*w, C*tp*p*p)
5  视觉编码     视觉塔 + merger                             (144, hidden)
6  占位符       processing.replace_image_token              "<|image|>" * 144
7  拼序列       文本 token + 视觉占位符                      (seq,)
--------------------------------------------------------------------------------
```

**读法**：第 1、2 步**不改形状**（只是重采样与线性变换），
第 3 步**只改形状**（不改数值，是一次纯重排）。
第 5 步的 `144` 就是本课验收点要求你手算出来的那个数。
第 6 步之后，视觉就"变成文本"了 —— 这是第 2 层所有课的落点。

不过一步都不白给：视觉侧还要把块内坐标变成位置编号（第十节），
音频侧要把频率格压成 mel 格（第十一节）。**"压到同一条序列上"是所有模态的共同终点。**

---

## 十三、与前后课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| `pixel_values` 怎么和占位符真正对上 | L2-06 / L2-07 |
| 图像处理器后端的 `resize` / `normalize` 在哪 | L2-04 |
| 视觉塔拿到 `(576, 588)` 之后怎么算 | L3（主干） |
| `position_ids` / `cu_seqlens` 谁在消费 | L4（注意力） |
| 视频的时间戳怎么进 mrope | L4 |
| `merge_size` 和 patch 展平宽度 588 的关系 | L3-05 |

**一句话总结**：

> 视觉 token 数不是"算出来的魔法"，而是
> `T * H * W / (p * p * merge_size²)` —— 336×336 单帧就是 **576 / 4 = 144**；
> 视频则先把 `T` 帧按 `temporal_patch_size` 折成 `grid_t`，
> **所以帧数必须是 temporal_patch_size 的整数倍**，不是就补最后一帧。