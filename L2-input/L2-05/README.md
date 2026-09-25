# L2-05 · 图像变换、视频与通用视觉工具

> 层：**第 2 层 · 输入流水线** ｜ 优先级：P0 ｜ 前置课：`L2-04`

## 学习目标

看完这一课，你应该能：

1. **手算**任意一张（已对齐到 patch×merge 整数倍的）图会产生多少个视觉 token ——
   336×336 的答案是 **144**（对应验收点 1）；
2. **说出 `temporal_patch_size=2` 对视频帧数提出了什么要求**，以及这条要求由哪一行代码兜住（验收点 2）；
3. 分清这一层里**同名却不同源**的两组函数（两个 `resize`、两份采样逻辑），
   并说出"张量级实现"到底把代价付在了哪里。

## 覆盖的源文件（6 个 / 5799 行）

| 文件 | 行数（实测 `wc -l`） | 引用块 |
|---|---|---|
| `image_transforms.py` | 1102 | 7 |
| `image_utils.py` | 1076 | 2 |
| `video_processing_utils.py` | 795 | 2 |
| `video_utils.py` | 927 | 3 |
| `vision_utils.py` | 379 | 2 |
| `audio_utils.py` | 1520 | 1 |

> **行数订正**：作业书给的是 1103 / 1077 / 796 / 928 / 380 / 1521，实测每个都**少 1**
> （作业书似乎把行号最大值 + 1 当成了行数）。本课一律以 `wc -l` 为准。
> 作业书里 `L46` / `L89` / `L313` 这类**定位行号是对的**，逐个核对过。

> **不计入覆盖率的引用**：`source.md` 共 18 个引用块，其中 **15 个**来自上表六个文件。
> 另外 3 个来自 `models/glm5_next/image_processing_glm5_next.py`、
> `processing_glm5_next.py`、`video_processing_glm5_next.py` ——
> 它们属于 **L2-06 的覆盖域**。引用原因是：本课这六个文件里**没有任何一处**
> 真正实现过带 `merge_size` 的切块与占位符计数，而那两件事恰好是本课两个验收点的落点。

## 场景（10 幕）

1. **像素怎么变成 token：三步** —— 全景流程线 + 三步形状表（144 这个数先亮出来）
2. **同一个 resize，两份实现** —— `image_transforms` 绕一趟 PIL vs `image_utils` 直接吃 PIL
3. **normalize 为什么第一步是 cast** —— uint8 回绕 / float16 静默提升
4. **两版切块** —— `divide_to_patches` 的签名与循环；列表 vs 张量
5. **★ 合并前的重排** —— `patchify` 的 8 维 reshape + 一次 permute
6. **★ 算一遍 336×336** —— 24×24=576，576/4=**144**（验收点 1）
7. **group_images_by_shape** —— CPU 不分组（PR #38157）+ `reorder_images` 还原
8. **视频采样接口** —— 三级默认、`fps` 与 `num_frames` 互斥、硬边界
9. **★ temporal patch 对帧数的要求** —— `-num_frames % temporal_patch_size` 补最后一帧（验收点 2）
10. **收束** —— 形状总表 + 两个练习 + 下一课指路

## 核心结论

### 1. 视觉 token 数就是一个除法

<!-- src: models/glm5_next/processing_glm5_next.py -->
```python
    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        merge_length = self.image_processor.merge_size**2
        num_image_tokens = image_inputs["image_grid_thw"][image_idx].prod() // merge_length
        return self.image_token * num_image_tokens
```

`336/14 = 24` → `24×24 = 576` 个 patch → `576 / (2²) = 144` 个 token。
这一串除法决定了文本序列要为视觉预留多少个位置。

### 2. ★ 切块时就已经把"合并"排好了

<!-- src: models/glm5_next/image_processing_glm5_next.py -->
```python
        patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
```

`permute` 之后内存布局是"先块、后块内"，所以 2×2 个 patch 天然连续 ——
合并（merge）因此只是**换个分组方式**，不需要真的做加法。

### 3. ★ 帧数必须是 temporal_patch_size 的整数倍

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

`-num_frames % temporal_patch_size` 是"向上取整到倍数"的惯用写法；
不足时**重复最后一帧**（不是补零）。`grid_t = num_frames // temporal_patch_size`
说明 `grid_thw` 里的 `t` 是**网格数**，不是原始帧数。

### 4. `uint8` 相减会回绕

<!-- src: image_transforms.py -->
```python
    # We cast to float32 to avoid errors that can occur when subtracting uint8 values.
    # We preserve the original dtype if it is a float type to prevent upcasting float16.
    if not np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32)
```

两行注释讲了两个真实的数值事故：`3 - 5 = 254`（回绕），
以及 `float16` 被顺手提升成 `float32`（显存翻倍而没人发现）。

## 与后续课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| `pixel_values` 怎么和占位符真正对上 | L2-06 / L2-07 |
| 图像处理器后端的 `resize` / `normalize` 在哪 | L2-04 |
| 视觉塔拿到 `(576, 588)` 之后怎么算 | L3（主干） |
| `position_ids` / `cu_seqlens` 谁在消费 | L4（注意力） |
| 视频的时间戳怎么进 mrope | L4 |
| `merge_size` 和 patch 展平宽度 588 的关系 | L3-05 |

## 与作业书的偏差（实测订正）

| 项 | 作业书 | 实测 | 处理 |
|---|---|---|---|
| 六个文件行数 | 1103 / 1077 / 796 / 928 / 380 / 1521 | **1102 / 1076 / 795 / 927 / 379 / 1520** | 全部按实测写；作业书的 `L` 定位行号核对无误 |
| 幕数 | 9 幕 | **10 幕** | 第 9 幕（temporal patch）与第 10 幕（收束）各自需要一张不同的代码引用，合成 9 幕会让其中一幕挂错引用块 |
| 引用块数 | 8–14 | **18** | 六个文件各≥2 块；多出的部分用于支撑两个验收点（切块 / 占位符计数必须引 L2-06 的三个文件） |

## 验收点

- [x] **能算出 336×336 图像经 patch+merge 后的 token 数** ——
      第 6 幕把 `336/14=24`、`24×24=576`、`576/4=144` 逐格点亮演示（验收点 1）
- [x] **能解释 `temporal_patch_size=2` 对视频帧数的要求** ——
      第 9 幕讲"必须是整数倍、不足则重复最后一帧"，收束幕的练习还追问了 `=4` 的情形（验收点 2）
- [x] 保真门禁：18 个引用块全部逐字来自源文件、且位置连续
- [x] 覆盖度门禁：6 个源文件全部被声明（属覆盖域 231 个文件之一）
- [x] 参数门禁：无非法参数引用
- [x] 语法检查：`node --check` 通过，10 幕字段完整
- [x] 渲染门禁：无 JS 错误、无布局溢出、交互可用（两种分辨率）
- 自检：合上屏幕，对着"336×336 → 144"这条链复述每一步是谁做的

**一句话总结**：

> 视觉 token 数不是魔法，而是 `T * H * W / (p * p * merge_size²)` ——
> 336×336 单帧就是 **576 / 4 = 144**；视频先把 `T` 帧按 `temporal_patch_size`
> 折成 `grid_t`，**所以帧数必须是 temporal_patch_size 的整数倍**，不是就补最后一帧。
