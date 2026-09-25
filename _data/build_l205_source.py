#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 L2-05/source.md（自用脚本，位于 _data/，不在课件目录里）。

所有 ```python 引用块都从上游源文件**切片**写入，绝不手抄 —— 这样保真门禁
必然通过，也避免"凭记忆改写"。散文部分是手写的。
"""
import io
import os

SRC = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "upstream-transformers", "src", "transformers"))
OUT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "L2-input", "L2-05", "source.md"))

_cache = {}


def lines_of(rel):
    if rel not in _cache:
        with io.open(os.path.join(SRC, rel), encoding="utf-8") as f:
            _cache[rel] = f.read().split("\n")
    return _cache[rel]


def block(rel, a, b):
    body = "\n".join(lines_of(rel)[a - 1:b])
    return "```python\n" + body + "\n```"


PARTS = []
P = PARTS.append


def B(rel, a, b):
    """安全地追加一个引用块：自动给上一段散文结尾补上 src 标注。"""
    assert PARTS, "B() 之前必须先有一段散文"
    if not PARTS[-1].endswith("<!-- src: %s -->" % rel):
        PARTS[-1] = PARTS[-1].rstrip("\n") + "\n\n<!-- src: %s -->" % rel
    PARTS.append(block(rel, a, b))

# ================================================================== 头部
P('''<!-- glm5-coverage
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
| `image_transforms.py` | 1102 | 8 |
| `image_utils.py` | 1076 | 2 |
| `video_processing_utils.py` | 795 | 2 |
| `video_utils.py` | 927 | 2 |
| `vision_utils.py` | 379 | 2 |
| `audio_utils.py` | 1520 | 2 |
| *（覆盖域外，见第五节）* | — | 3 |

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

''')
B("image_transforms.py", 353, 381)
P('''
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

''')
B("image_utils.py", 829, 845)
P('''
**读法**：
- 没有 `do_rescale`、没有 `to_channel_dimension_format`。
  **代价换了个地方付**：调用者必须先把张量转成 PIL（`to_pil_image`）。
- `size` 是 int 时 `default_to_square=True` 走 `(size, size)`；
  `False` 时按短边对齐、长边等比 —— 这正是 torchvision `Resize` 的语义
  （`default_to_square` 这个名字就是从那里来的）。
- `if short == requested_new_short: return image` 是个提前返回：
  尺寸已经对上了就别做一次无意义的插值。**重采样是纯损失**，能不做就不做。

### 裁剪：`center_crop` 里的整数除法

''')
B("image_transforms.py", 481, 503)
P('''
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

''')
B("image_transforms.py", 411, 427)
P('''
**读法**：
- 块内第 7~9 行是本课最便宜也最重要的一处：`uint8` 做 `image - mean` 会**回绕**
  （`3 - 5 = 254`），所以必须先 cast 到 float32。
- 注释第二句同样关键：**已经是浮点类型就不动它**。
  否则 `float16` 会被悄悄提升成 `float32`，显存翻倍而没人发现。
- 块内第 12~19 行把 `mean` / `std` 广播成"每通道一个数"。
  注意 `mean = [mean] * num_channels` —— 标量写法是被**显式**支持的，
  但前提是它和 `num_channels` 对得上（对不上就 raise，不静默）。
- 真正的算式在源码第 436~439 行（见下），它按数据格式分成两支。

''')
B("image_transforms.py", 436, 442)
P('''
**读法**：
- `channels_last` 时 `mean` 直接对上最后一维，一行搞定。
- `channels_first` 时先 `.T` 把通道轴换到末尾、算完再 `.T` 回来。
  **这不是为了省事，是为了让广播规则简单**：两种格式最终都归约成
  "最后一维是通道"这一种情况，代码只有一处算式。
- 这些函数的输入输出形状**完全一样**（`(C,H,W)` 进 `(C,H,W)` 出），
  变的是数值范围 —— 和下面要讲的 patchify 形成对比，那个是**改形状**的。

`image_utils` 里还留着"按格式分两支"的最短写法 —— 归一化要用它定位通道轴：

''')
B("image_utils.py", 340, 346)
P('''
**读法**：
- 通道轴不是写死的 1 或 3，而是用 `image.ndim - 3` / `image.ndim - 1` 算出来的：
  **同一个函数要同时服务 3 维 `(C,H,W)`、4 维 `(B,C,H,W)`、5 维 `(B,T,C,H,W)`**。
  这是本层反复出现的形状弹性：写死维号在视频上立刻就错。
- 末尾那句 `raise ValueError` 是"不猜"的体现：推断不出来就报错。
  （`infer_channel_dimension_format` 在真歧义时会警告并按 FIRST 处理，
  两者是不同策略：一个放弃，一个带警告地兜住。）
- 为什么放在这一节：`normalize` 的第 4 行就是调它拿通道轴 ——
  **变换函数与数据格式判断是配套的一对**。

---

## 四、切块：`divide_to_patches` 的签名与循环

`image_transforms.py` 提供的是**通用工具**：两个步进循环，返回一个 patch 列表。
可读性优先，代价是**它不带 batch 维**。先看签名与 docstring：

''')
B("image_transforms.py", 839, 853)
P('''
**读法**：
- `patch_size` 既接受 `int` 也接受 `(h, w)` 元组 —— 全库统一的"尺寸参数"惯例
  （`SizeDict` / `resize` 的 `size` 都一样宽进严出）。
- docstring 里 `Returns: list` 是**唯一的形状线索**：它不承诺堆叠，
  所以调用方必须自己处理"一批形状不同的 patch"。
- `get_image_size(image, channel_dim=ChannelDimension.FIRST)` 先把
  `(H, W)` 取出来，且**显式指定**了通道在前 —— 不靠推断，
  因为这里马上要用 `image[..., i:i+patch_h, j:j+patch_w]` 做切片。

循环体紧接在 docstring 之后（同一个函数的后半段）：

''')
B("image_transforms.py", 854, 862)
P('''
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

''')
B("models/glm5_next/image_processing_glm5_next.py", 185, 215)
P('''
> **边界声明**：本课一共 21 个引用块，其中 **18 个**来自上面 coverage 声明的六个文件，
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

''')
B("models/glm5_next/processing_glm5_next.py", 63, 66)
P('''
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

''')
B("image_transforms.py", 1029, 1047)
P('''
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

''')
B("image_transforms.py", 1096, 1102)
P('''
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

''')
B("video_utils.py", 285, 303)
P('''
**读法**：
- `np.arange(0, total, total / num_frames)` —— 步长**不取整**，最后才 `.astype(int)`。
  这样最后一帧尽量贴近末尾，比 `linspace(0, N-1, n)` 更接近"均匀覆盖整段视频"。
- `num_frames=None` 时退化成"全都要"，**没有采样**。
- 抽到的索引是**原视频里的绝对帧号**（0, 5, 10, …），不是 0..n-1 ——
  这一点在下面算时间戳时会用到。

同一个决策在视频处理器里还有一份 torch 实现：

''')
B("video_processing_utils.py", 135, 186)
P('''
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

采样函数怎么被调用？看它的上一层：

''')
B("video_processing_utils.py", 198, 226)
P('''
**读法**：
- 块内第 1~2 行把"视频"和"元数据"都先**拍平成一个列表** ——
  因为后面两支分支（已是数组 / 需要解码）都要能按下标配对。
- 块内第 5 行 `is_valid_video(videos[0]) and do_sample_frames` 是**唯一的分岔点**：
  传进来已经是 4 维数组 → 直接按索引取值（`video[indices]`，一次花式索引）；
  传进来是路径/URL → 交给 `fetch_videos` 去解码。
- 块内第 10 行把索引写回 `metadata.frames_indices` —— 第九节要算的时间戳靠它。
- 块内第 16~25 行对"用 URL 列表表示的帧序列"显式报错：这种输入没法采样，
  与其静默给出错误的时间轴，不如让调用方把 `do_sample_frames=False`。
- **两处 `raise`、零处静默兜底** —— 和 `sample_frames` 里的边界检查是同一种风格。

帧索引留下来干什么？时间戳是后面 mrope 要用的：

''')
B("video_utils.py", 101, 113)
P('''
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

''')
B("models/glm5_next/video_processing_glm5_next.py", 325, 332)
P('''
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

''')
B("vision_utils.py", 58, 65)
P('''
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

''')
B("vision_utils.py", 107, 127)
P('''
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

''')
B("audio_utils.py", 1362, 1377)
P('''
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

mel 矩阵本身怎么造出来？五行：

<!-- src: audio_utils.py -->''')
B("audio_utils.py", 798, 802)
P('''
**读法**：
- `np.linspace(mel_min, mel_max, num_mel_filters + 2)` —— **在 mel 刻度上等分**，
  不是在线性频率上等分。低频率因此拿到更密的滤波器，
  这就是"感知上均匀"的具体含义。
- `hertz_to_mel` → linspace → `mel_to_hertz`：**先去感知域排队，再回物理域取点**。
  顺序反了（先在 Hz 上等分再转 mel）会得到一批挤在低频的三角。
- `num_mel_filters + 2` 里的 `+2` 是两个端点：`num_mel_filters` 个三角需要
  `num_mel_filters + 2` 个边界点，这是 `_create_triangular_filter_bank` 的约定。
- 和视觉侧对照：`patchify` 用 reshape/permute 决定"哪些像素归哪个 token"，
  这里用 `linspace + triangular` 决定"哪些频率归哪个 mel 格" ——
  **都是在造一个分组矩阵，只是一个用索引、一个用权重**。

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

**读法**：第 1、2 步**不改形状**（第 1 步是重采样，第 2 步是逐像素线性变换），
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
> **所以帧数必须是 temporal_patch_size 的整数倍**，不是就补最后一帧。''')

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with io.open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(PARTS))
print("wrote", OUT, os.path.getsize(OUT), "bytes")
