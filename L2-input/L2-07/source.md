<!-- glm5-coverage
models/glm5_next/modeling_glm5_next.py
backbone_utils.py
utils/backbone_utils.py
-->

# L2-07 · 视觉塔：从 336x336 像素到 27 个 token — 源文件

> ★ **实测订正（本课最先要说清的一件事）**：课程标题里的「27 个 token」**与源码不符**。
> 用 `Glm5NextVisionConfig()` 的实测默认值推：`336/14 = 24`，`24 x 24 = 576` 个 patch，
> 每个 2x2 块合成 1 个 token，得到 **144** 个视觉 token。
> 本课正文、动画与练习一律用 **144**，证据链在第三节。
> 实测脚本：`_data/recon/probe_l207.py`（脚本，**不计入覆盖率**）。

**这一课只做一件事：把一张 336x336 的 RGB 图变成 144 个 1536 维向量，并说清它们怎样与文本 token 排在同一条序列上。**

视角：像素 → patch → 视觉 token → 拼进 `inputs_embeds`。这是 L2（输入流水线）的最后一站：
文本一侧在 L2-01 ~ L2-03，图像一侧在 L2-04 ~ L2-05，装配在 L2-06，**塔本身在这里**。

| 文件 | 行数 | 本课引用块 |
|---|---|---|
| `models/glm5_next/modeling_glm5_next.py` | 2444 | 10 |
| `backbone_utils.py` | 387 | 2 |
| `utils/backbone_utils.py` | 19 | 2 |

> 上面的行数是**实测**（`wc -l`）。作业书里写的是 2445 / 388 / 20 —— 那是按
> `len(text.split("\n"))` 数的，会把文件末尾换行之后那个空串也算成一行。大纲里的
> `L1515`、`L1718` 这类行号则与实测一致，可以照用。

**实测的一张图走完全程**（`_data/recon/probe_l207.py` 的输出，**实测数字**）：

```text
一张 336x336 的 RGB 图
  │ 图处理器：smart_resize 到 28 的整数倍（28 = patch_size 14 x merge_size 2）
  ▼  pixel_values (576, 1176)      1176 = 3 通道 x 2 帧 x 14 x 14
  │ Glm5NextVisionPatchEmbed：Conv3d(kernel = stride = (2, 14, 14))
  ▼  (576, 1024)                   576 = 24 x 24 个 patch 位置
  │ 24 x Glm5NextVisionBlock（packed 变长注意力 + 轴向 RoPE）
  ▼  (576, 1024)
  │ view(-1, 2, 2, 1024).permute(0,3,1,2) -> Conv2d(kernel = stride = 2)
  ▼  (144, 1536)                   <- last_hidden_state 已经是 144 个
  │ Glm5NextVisionPatchMerger：proj -> LayerNorm -> GELU -> SwiGLU
  ▼  (144, 1536)                   <- pooler_output，真正拼进文本的那一串
  │ masked_scatter：填进 <|image|> 占位符（宽度必须等于文本 hidden_size）
  ▼  inputs_embeds (text_len + 144, 4096)
```

下面每一段 `python` 块都是**逐字**引用；`text` 块是示意汇总与实测输出，不参与保真校验。

---

## 一、★ 一张图进塔、一串 token 出塔

先看主干函数的前 5 行。它把「整座塔」压成了四件事：算位置编号、算变长注意力边界、
patch 嵌入、算 RoPE 的 cos/sin。

回顾 L0-01：文本主干的第一件事是"二选一"（`input_ids` 还是 `inputs_embeds`）；
视觉塔这边没有这个分叉 —— 它**只吃已经 patch 化好的像素**，`input_ids` 这个概念在塔里不存在。

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size, kwargs=kwargs)
        cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid_thw, self.config, kwargs=kwargs)

        hidden_states = self.patch_embed(hidden_states)
        position_embeddings = self.rotary_pos_emb(hidden_states, position_ids)
```


**读法**：

- 输入 `hidden_states` 的形状是 `(576, 1176)`，**不是** `(B, C, H, W)`。
  图像在进塔之前就被"patch 化"成了一条 `(seq, patch_dim)` 的序列 ——
  这是所有 packed 视觉塔的共同约定：一张图的 patch 不是四维张量，而是序列里的一段。
  这个 `(seq, patch_dim)` 布局是 L2-04 / L2-05 的图处理器负责产出的。
- `grid_thw` 是 `(num_images, 3)` 的 `[t, h, w]`；本课的单图情形实测是 `[[1, 24, 24]]`。
  注意 `h`/`w` 是 **patch 网格**的大小，不是像素数：`24 = 336 / 14`。
- `cu_seqlens` 取代了 `attention_mask`：多张图拼在一条序列上时，它就是"每张图占哪一段"的边界数组。
  n 张图就是 n 段互不可见的自注意力；单图时实测 `cu_seqlens = [0, 576]`，长度是 `N+1`。
- `get_vision_position_ids(...)` 返回的是 `(total_tokens, 2)`（h、w 两列），**不是** `(2, N)` ——
  源码里 `forward` 的第一行注释写的是 `(2, N)`，那是过时注释，实测证据在第四节。

### 形状演算（示意，非引用）

```text
grid_thw = [[1, 24, 24]]
position_ids   (576, 2)     每行 (h, w)，h, w 属于 [0, 24)
cu_seqlens     (2,)         [0, 576]
hidden_states  (576, 1176)  --patch_embed-->  (576, 1024)
```

---

## 二、patch embed 的卷积等价形式

`patch_embed` 是整座塔里唯一碰原始像素的地方。整个类只有 18 行，前向只有 6 行，但每一行都在做形状约定。

回顾 L2-05：`(576, 1176)` 这条序列是图处理器 `patchify` 出来的；
本节的代码要回答的是"为什么是 1176，以及它怎样变回 1024"。

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
class Glm5NextVisionPatchEmbed(nn.Module):
    def __init__(self, config: Glm5NextVisionConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states
```


**读法**：

- 先看权重形状（**实测**）：`proj.weight = (1024, 3, 2, 14, 14)`、`stride = (2, 14, 14)`、`kernel = (2, 14, 14)`。
  `Conv3d` 的权重是 `(out_channels, in_channels, T, P, P)`，所以这里一次卷积就把
  `3 x 2 x 14 x 14 = 1176` 个输入数映射成 `1024` 维 —— `1176` 这个数不是拍脑袋来的，
  它就是 `in_channels x temporal_patch_size x patch_size x patch_size`。
- **卷积等价形式**：`kernel_size == stride` 且没有 padding 时，滑窗互不重叠，
  这一层等价于"把输入切成 576 个不重叠的 `3 x 2 x 14 x 14` 管子，每管做一次
  `Linear(1176 -> 1024)`"。写成 `Conv3d` 而不是 `Linear` 的好处是：
  切窗由框架完成，不必手写 `unfold`/`reshape`；而且权重形状自带 `(T, P, P)` 语义，
  在检查点里一眼能看出它卷的是什么。
- `hidden_states.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)`：
  把 `(576, 1176)` 还原成 `(576, 3, 2, 14, 14)` 再卷积。`-1` 是 patch 位置数，
  由元素总数反推：`576 x 1176 / (3 x 2 x 14 x 14) = 576`。**这里没有"猜"形状**：
  除了这一个由总数定出来的维度，其余每个维度的语义都由 config 直接提供
  （`in_channels` / `temporal_patch_size` / `patch_size` 三个字段）。
- `.to(dtype=target_dtype)`：输入常常是 fp32（图处理器给的），权重可能是 bf16。
  统一到**权重的** dtype 而不是反过来 —— 视觉塔的数值精度由权重决定。
  漏掉这一步会直接 dtype mismatch 报错，所以它在源码里是显式写出来的。
- 出口 `.view(-1, self.embed_dim)` 把空间维摊平回序列：`(576, 1024)`。
  从这里开始，"图像"就只是一个序列了 —— 后面 24 层 block 完全不知道自己在处理图片。

---

## 三、★ 实测订正：是 144 个 token，不是 27

这一节给出"一张图几个 token"的**运行时定义**。它不在视觉塔里，而在多模态模型的外层：

回顾 L0-01：`inputs_embeds` 与 `input_ids` 二选一；视觉 token 就是在这一层被"替换"进去的。

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_outputs = self.visual(pixel_values, grid_thw=image_grid_thw, **kwargs)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(vision_outputs.pooler_output, split_sizes)
        vision_outputs.pooler_output = image_embeds

        return vision_outputs
```


**读法**：

- `split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size ** 2).tolist()`：
  `prod(-1)` 是 `t x h x w`（本课 `1 x 24 x 24 = 576`），除以 `2 ** 2 = 4` 得到 **144**。
  这一行就是"一张图几个 token"的运行时定义：**不是配置里写死的常数，而是每次前向现算的**。
- `torch.split(...)` 把一个 batch 的合并特征按每张图的 token 数切开 —— 所以支持同一批里
  每张图尺寸不同。这也是为什么 `cu_seqlens` 必须现算。
- 这段代码与处理器的 `replace_image_token` 用的是**同一个公式**
  （`num_image_tokens = image_grid_thw.prod() // merge_length`）。
  两边必须一致，否则 `get_placeholder_mask` 会抛
  `Image features and image tokens do not match`。**一处改了另一处没改，模型会在运行时炸**，
  这是 L2-06 与本课的接缝。

### 实测：一次真实的 336x336 前向

下面这段是 `_data/recon/probe_l207.py` 的**原始输出**（脚本，不计入覆盖率）：

```text
[1] 配置（Glm5NextVisionConfig() 默认值）
    image_size=336 patch_size=14 spatial_merge_size=2 temporal_patch_size=2
    hidden_size=1024 depth=24 num_heads=16 out_hidden_size=1536
    网格 24x24 -> patch 数 576 -> 合并后 token 数 144
    head_dim=64 -> spatial_dim=32 -> 频率个数=16
[6] 真实图像处理器：一张 336x336 的图产出多少 token
    pixel_values (576, 1176)  image_grid_thw = [[1, 24, 24]]
    num_image_tokens = 576 // 2**2 = 144
```

**为什么"27"不可能是对的**（三种独立算法都给 144）：

```text
算法一（先算 patch 再合并）：(336/14)^2 / 2^2 = 24^2 / 4 = 576 / 4 = 144
算法二（先合并再数格）    ：(336/14/2)^2   = 12^2       = 144
算法三（跑代码）          ：image_grid_thw.prod() // merge_size**2 = 144
```

`27` 连"能被 4 整除"都不满足（`576 / 27 = 21.33`），也不等于任何 `576 / 2^k`。
所以本课一律用 **144**；`27` 只作为"标题里的错误"出现一次。

---

## 四、★ 轴向 RoPE：16 个频率，height / width 各一套

视觉塔的位置编码叫 **axial（轴向）RoPE**。作者在类注释里把设计意图写得很直白 ——
这是本课最值得逐字读的一段注释：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
class Glm5NextVisionRotaryEmbedding(nn.Module):
    """
    Simple axial 2D rope with same freqs used for H and W grids. The freqs are
    pre-computed using `head-dim//4` which is later used to concat H and W positions.
    The final angles rotate over the whole head dim, no partial rotation involved.
    """
```


**读法**：

- "same freqs used for H and W grids"：H 与 W **共用同一张频率表**。
  这不是省事，而是轴向 RoPE 的定义：两个轴各自有坐标，但"转速"集合相同。
- "pre-computed using `head-dim//4`"：频率个数是 `head_dim // 4`。
  实测 `head_dim = 1024 / 16 = 64`，所以是 `64 // 4 = 16` 个频率。
- "The final angles rotate over the whole head dim, no partial rotation involved"：
  旋转发生在**整个 head_dim** 上，不存在"只转一半维度"的部分旋转
  （有些 VLM 会只对一半维度做 2D RoPE，这里不是）。

频率就是这么算出来的：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        spatial_dim = dim // 2

        attention_factor = 1.0  # Unused in this type of RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim))
```


**读法**：

- `base = config.rope_parameters["rope_theta"]`：实测 `10000.0`。
  `dim` 先看 `config.head_dim`，没有就用 `hidden_size // num_attention_heads` ——
  这个 "getattr 兜底" 是 transformers 里的通用写法（回顾 L1-03：配置字段可能来自不同世代）。
- `spatial_dim = dim // 2`：`64 // 2 = 32`。
- 关键是 `torch.arange(0, spatial_dim, 2)`：**步长是 2**，所以数量是 `32 / 2 = 16`。
  一维 RoPE 用的是 `arange(0, dim, 2) / dim` → `32` 个频率，覆盖整个 `head_dim`；
  这里**只算了一半**，因为另一半要留给 W 轴。**"频率减半"就是轴向化付出的代价与得到的自由**：
  用 16 个频率换来两个独立坐标轴。
- 实测频率表（前 4 个与最后 1 个）：

```text
inv_freq.shape = (16,)
前 4 个 = [1.0, 0.562341, 0.316228, 0.177828]
最后 1 个 = 0.00017783        # 1 / 10000^(30/32)
```

- `attention_factor = 1.0  # Unused in this type of RoPE`：缩放位留着但恒为 1。
  这行注释本身就是证据 —— 标准 RoPE 不需要对 cos/sin 做后处理，
  只有 YaRN / NTK 这类改频率的变体才会用到它。
- `__init__` 里那句 `if self.rope_type != "axial": raise ValueError(...)` 把
  "配置声明"和"实现能力"绑在一起：视觉配置的 `rope_type` 只能是 `axial`（实测默认值就是它）。
  这正是 L1-03 讲的配置契约在运行时被强制执行的样子。

---

## 五、★ 重组：H/W 两套频率怎样拼成 64 维

上一步算出 `(N, 2, 16)` 的 `cos/sin`（第 0 列是 h 的频率，第 1 列是 w 的频率），
但注意力要的是每个 token 一份、长度等于 `head_dim` 的旋转角。这两者之间的桥只有 4 行：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
    def recomposition_frequencies(self, freq):
        """
        Recompose the frequencies into the final spatial layout used per each grid.
        """
        freq_h, freq_w = freq[:, 0], freq[:, 1]
        freq_hw = torch.cat([freq_h, freq_w], dim=-1)
        return torch.cat([freq_hw, freq_hw], dim=-1)
```


**读法**（这一步的形状变化是本课的第二个★）：

```text
freq        (N, 2, 16)     # [:, 0] = h 频率, [:, 1] = w 频率
freq_h      (N, 16)        # 只随 h 变
freq_w      (N, 16)        # 只随 w 变
freq_hw     (N, 32)        # cat([h, w]) -> 前 16 维归 h，后 16 维归 w
返回        (N, 64)        # cat([freq_hw, freq_hw]) -> 复制一份，铺满 head_dim
```

- **为什么要复制**：下游 `apply_rotary_pos_emb_vision` 用的是 `rotate_half`，
  它把向量**前一半与后一半配对**（`rotate_half(x) = cat([-x2, x1])`）。
  当 `cos = [freq_hw, freq_hw]` 时，配对恰好是"第 i 维与第 i+32 维"，两边的角度相同 ——
  于是每个频率都得到一个标准的二维旋转。这就是注释里"整个 head_dim 都在转"的实现方式：
  **先把频率拼成 32 维，再复制成 64 维，让 rotate_half 的配对落在同一个频率上**。
- **与一维 RoPE 的区别**（本课验收点 2）：

| | 一维 RoPE（文本） | 轴向 RoPE（本课） |
|---|---|---|
| 位置输入 | 一个标量 `p` | 两个坐标 `(h, w)` |
| 频率个数 | `head_dim / 2 = 32` | `head_dim / 4 = 16`，H/W **共用** |
| 角度来源 | `p * inv_freq[i]` | 前 16 维 `h * inv_freq[i]`，后 16 维 `w * inv_freq[i]` |
| 语义 | "第几个 token" | "第几行、第几列" |

- 实测验证（`probe_l207.py` 第 [4] 节，改动一个坐标看 cos 哪一段跟着变）：

```text
只改 w（h 不变）: cos 前 16 维相同 = True,  16..32 维不同 = True
只改 h（w 不变）: cos 16..32 维相同 = True, 前 16 维不同 = True
cos 的前 32 维与后 32 维完全相同: True
```

  也就是说：**h 坐标只影响前 16 个旋转角，w 坐标只影响接下来 16 个**。
  一维 RoPE 里"距离"是一条线上的距离；这里上下与左右各自编码在不同的维度块里，
  一个 patch 与另一个 patch 的相对角度同时依赖 `dh` 和 `dw`。图像的"上下"和"左右"
  本来就不是同一个方向，这就是轴向 RoPE 存在的理由。
- 顺带：**没有第三根轴**。时间维不参与 RoPE，而是被 `cu_seqlens` 切成独立的段
  （每帧一段）；`get_vision_position_ids` 里那个 `include_temporal` 开关默认是关的。

---

## 六、patch merger 的空间 2x2 合并与投影

24 层 block 跑完，手上还是 576 个 token、1024 维。文本主干要的是"少量、宽"的视觉 token，
所以最后一步是**空间 2x2 合并 + 投影**。这是主干函数的尾巴：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        hidden_states = self.post_layernorm(hidden_states)

        hidden_states = hidden_states.view(
            -1, self.spatial_merge_size, self.spatial_merge_size, hidden_states.shape[-1]
        )
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.downsample(hidden_states).view(-1, self.config.out_hidden_size)

        merged_hidden_states = self.merger(hidden_states)
        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
        )
```


**读法**：

- `hidden_states.view(-1, self.spatial_merge_size, self.spatial_merge_size, hidden_states.shape[-1])`：
  `(576, 1024)` → `(144, 2, 2, 1024)`。**之所以敢这样直接 view，是因为序列顺序已经排好了**：
  `get_vision_position_ids` 把位置编号按块主序铺开（实测，`grid_thw=[[1,4,4]]`）：

```text
前 8 行 (h, w) = [[0,0], [0,1], [1,0], [1,1], [0,2], [0,3], [1,2], [1,3]]
每 4 个连续 token 恰好构成一个 2x2 块: True
```

  也就是说 token 顺序是"块 → 块内 4 个位置"，与 `view(-1, 2, 2, H)` 完全对齐；
  图处理器的 `patchify` 用的是同一个块主序。**这个约定一旦破了，view 不会报错，只会算错** ——
  这正是"形状对但语义错"的典型陷阱。
- `permute(0, 3, 1, 2)` → `(144, 1024, 2, 2)`，再 `Conv2d(kernel=2, stride=2)`：
  又是"kernel == stride 的卷积等价形式"，但这一次它是 `1024 -> 1536` 的**线性投影**，
  作用在每个 2x2 块上，输出 `(144, 1536, 1, 1)` → `view(-1, out_hidden_size)` → `(144, 1536)`。
- **顺序要点：先投影到 1536，再过 merger。** `Glm5NextVisionModel.__init__` 里写的是
  `merger = Glm5NextVisionPatchMerger(dim=config.out_hidden_size, ...)` ——
  merger 的 `dim` 是 **1536**（`out_hidden_size`），不是 1024（`hidden_size`）。
  看 merger 的代码时如果不注意这一点，会把 `proj` 误读成"1024 → 1024"。
- `BaseModelOutputWithPooling(last_hidden_state=hidden_states, pooler_output=merged_hidden_states)`：
  **名字有坑，实测两者形状相同但语义不同**：

```text
last_hidden_state = (144, 1536)   # 已合并、已投影，但没过 merger 的 MLP
pooler_output     = (144, 1536)   # 过完 merger，真正被拼进文本的那一串
pooler_output == merger(last_hidden_state): True
```

  `last_hidden_state` **不是** 576 个 patch（`hidden_states` 这个名字在函数里被复用了两次，
  出口时已经是 downsample 之后的值）。要拿"送进文本的视觉 token"，看 `pooler_output`
  与 `get_image_features` 里的 `.pooler_output`。

merger 内部做了什么：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.proj(hidden_state)
        hidden_state = self.act1(self.post_projection_norm(hidden_state))
        gate = self.gate_proj(hidden_state)
        up = self.up_proj(hidden_state)
        # Key difference using clamping
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)
```


**读法**：

- 四步：`proj`（1536 → 1536 的线性）→ `LayerNorm` → `GELU` → SwiGLU。
  `proj` 与 `post_projection_norm` 的作用是"先在自己的宽度上混一次、归一化"，再做通道数
  10240 的扩张 —— 这个 10240 是 `projection_intermediate_size`，比 `intermediate_size`
  （4096，视觉 MLP 用的那个）大得多，因为 merger 只跑一次，可以贵一点。
- SwiGLU 的写法与文本侧同源（回顾 L3-02）：`down(act(gate) * up)`。
  区别在注释标出的那三行 `# Key difference using clamping`：
  `gate.clamp(min=None, max=self.swiglu_limit)` 只截上界，
  `up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)` 双边截断，`swiglu_limit = 10.0`。
  **为什么视觉塔也要截断**：视觉特征的动态范围比文本大（亮暗、边缘），
  SwiGLU 的 `gate * up` 是乘法，任何一边爆掉都会直接污染残差流；截断把上界钉死，
  这是 L3-02 讲的稳定性动机在视觉侧的复现。
- `act_fn = ACT2FN[hidden_act]`：`silu`（实测默认值）。`ACT2FN` 是 L3-02 的注册表，
  同一个名字在文本和视觉两侧查到同一个函数 —— 这是"配置即接口"的另一个例子。

---

## 七、24 层 block 内部：packed 变长注意力

`Glm5NextVisionBlock` 是标准 pre-norm 残差：
`hidden_states = hidden_states + self.attn(self.norm1(hidden_states), ...)`，
然后 `hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))`
（源文件 L1708 ~ L1715，本课不单独引用）。

**注意它没有 mHC 的 4 条流**（回顾 L0-01：文本主干的 `hidden_states` 从一开始就是 4 维的）。
视觉塔的残差是教科书式的加性残差，维度始终 `(576, 1024)`。

注意力本身只有 13 行有效代码，但信息量很大：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)
```


**读法**：

- `self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)`：
  一次线性把 `(576, 1024)` 变成 `(576, 3072)`，再 reshape 成 `(576, 3, 16, 64)` ——
  **把"3"当成 batch 维借用**，`permute(1, 0, 2, 3)` 之后 `unbind(0)` 正好切出 q/k/v 三份
  `(576, 16, 64)`。一次矩阵乘、一次 reshape，没有第二次投影。
- `query_states = self.q_norm(query_states)` / `key_states = self.k_norm(key_states)`：
  q/k 各做一次**逐 head 的 RMSNorm**（`Glm5NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)`，
  实测 `eps = 1e-5`）。它归一化的是**最后一维 64**，不是 1024 —— 头部之间互不影响。
  动机是视觉特征的尺度随内容波动很大，逐头归一化让注意力的 logits 不至于被少数大分量主导
  （这是设计意图，源码里没有注释；**事实部分是「有这一步、维度是 head_dim」，动机部分是解释**）。
- `cos, sin = position_embeddings` 然后 `apply_rotary_pos_emb_vision(...)`：
  cos/sin 形状是 `(576, 64)`，`unsqueeze(-2)` 成 `(576, 1, 64)` 后广播到 16 个头。
  函数内部还做了一次 `q.float()` / `k.float()` 再转回原 dtype：**旋转在 fp32 里算**，
  避免 bf16 下的角度精度损失 —— 这也是所有 RoPE 实现的共同做法。
- 之后 `transpose(0, 1).unsqueeze(0)`：把 `(576, 16, 64)` 变成 `(1, 16, 576, 64)`，
  即 `(batch, heads, seq, head_dim)` —— 标准注意力内核的入参布局。
- **`self.is_causal = False`**：视觉塔不做因果掩码。一张图内部所有 patch 互相可见。
  这是它与文本塔最本质的行为差异，也是它敢用 packed 变长布局的前提。
- packed 的两条实现路径：非 flash 路径用
  `lengths = cu_seqlens[1:] - cu_seqlens[:-1]` 把 q/k/v 按图切开、逐段算完再 `cat`；
  flash 路径直接把 `cu_seqlens` 交给内核（`cu_seq_lens_q = cu_seq_lens_k = cu_seqlens`）。
  **两条路径语义相同：段内全可见、段间不可见。** 后端分派本身是 L8-01 / L8-02 的内容。
- `attn_output.reshape(seq_length, -1)` 回到 `(576, 1024)` 再 `self.proj`（源文件 L1683 ~ L1684）。
  整座塔里没有一处 `attention_mask` 参与计算 —— 位置与可见性全部由 `cu_seqlens` + RoPE 表达。

---

## 八、回填文本序列：视觉 token 在这里"落到位置上"

塔的输出是 `(144, 1536)`。它要变成 `inputs_embeds` 里的一段，动作只有 4 行：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw, **kwargs).pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
```


**读法**：

- `image_embeds = torch.cat(image_embeds, dim=0)`：`get_image_features` 返回的是
  **每张图一个张量**的列表（按 `split_sizes` 切开的），这里按 batch 维拼回一条
  `(144, 1536)` 的长序列。多图时长度是各图 token 数之和（尺寸相同才是 `144 x n`）。
- `.to(inputs_embeds.device, inputs_embeds.dtype)`：视觉塔与文本主干的 dtype/设备可能不同
  （塔常年在 bf16，主干可能 fp32），**在这里对齐** —— 也是显式写出来的，不靠隐式转换。
- `get_placeholder_mask(...)` 产出一个布尔掩码：`input_ids == image_token_id`（实测 `154854`）
  **并且不在视频区间内**。这里有个 GLM-5 特有的设计：
  `# Core difference to other VLMs as img token == vid token so we differentiate by start/end spans instead`
  —— 图和视频**共用同一个占位符 id**，靠 `<|begin_of_image|>` / `<|end_of_image|>` 的
  `cumsum` 区间来区分（源码 L1973 ~ L1975）。所以"是不是图片 token"这个问题只有配合区间才答得出来。
- `inputs_embeds.masked_scatter(image_mask, image_embeds)`：把占位符位置的向量**原地替换**成
  视觉向量。`masked_scatter` 要求 `mask.sum() * hidden_size == image_embeds.numel()`，
  这就是第三节那句 `torch_compilable_check` 的由来 ——
  **占位符数量与视觉 token 数量必须严丝合缝**（144 对 144）。但"数量对"只是半个条件，
  另一半是宽度：见下一条。
- **★ 实测约束：视觉塔的出口宽度必须等于文本的 `hidden_size`。**
  `get_placeholder_mask` 里那句 `torch_compilable_check(n_image_tokens * inputs_embeds.shape[-1] == image_features.numel(), ...)`
  与 `masked_scatter` 的语义是同一件事：填进去的元素个数必须正好等于"占位符数 x 文本宽度"。
  实测（`probe_l207.py` 第 [7] 节，小配置）：

```text
文本 hidden=64 视觉 out_hidden= 64 -> 通过了 masked_scatter 这一步（随后在玩具配置的 ValueError 上报错，与本步无关）
文本 hidden=64 视觉 out_hidden= 32 -> 就卡在这一步：Image features and image tokens do not match
类默认值：hidden_size=4096 vs out_hidden_size=1536；144 x 4096 = 589824，144 x 1536 = 221184
```

  而两个**类默认值**是 `hidden_size = 4096`、`out_hidden_size = 1536` —— **它们不能放在一起用**：
  `144 x 4096 = 589824` 而 `144 x 1536 = 221184`。所以本课表格里的 1536 只是**类默认值**，
  真实检查点必然把它覆盖成文本宽度，否则多模态前向根本走不过这一步。
  这也解释了 `out_hidden_size` 这个配置项存在的意义：它不是"随便定一个交付宽度"，
  而是**必须与语言模型对齐**的接口宽度 —— 又一个 L1-03 讲的"配置即契约"。
- 替换完成后，`self.language_model(input_ids=None, inputs_embeds=inputs_embeds, ...)`
  接手。回顾 L0-01：主干只看见"一条长度为 `text_len + 144` 的向量序列"，
  它**完全不知道**哪一段是图。文本 token 与视觉 token 的对齐，就是在这一行完成的：
  序列位置上先摆 144 个 `<|image|>` 占位符，再把向量填进去。

### 一条序列上的两种 token（示意，非引用）

```text
input_ids      : [BOS] ... <|image|> x 144 ... "描述这张图" [EOS]
                            |
                            | get_image_features -> (144, 1536)
                            | masked_scatter
                            v
inputs_embeds  : [emb] ... 视觉向量 x 144（宽度 = 文本 hidden_size） ... [emb]
                            ^
                  从这里开始，主干看到的只有 (text_len + 144, 4096) 的一条序列
```

---

## 九、为什么本课要读 `backbone_utils.py`

coverage 声明里有 `backbone_utils.py` 与 `utils/backbone_utils.py` 两个文件。它们**不在**视觉塔的
调用链上 —— 本节的任务是**划清边界**：读 transformers 时，"backbone"与"vision tower"
是两套不同的契约。

`backbone_utils.py` 里那套东西的形状是这样的：

<!-- src: backbone_utils.py -->
```python
class BackboneType(enum.Enum):
    TIMM = "timm"
    TRANSFORMERS = "transformers"
```


**读法**：

- `BackboneType` 只有两个值：`timm` 与 `transformers`。这说明 `BackboneMixin` 服务的场景是
  "**主干网络可以来自两个生态**"——CV 侧的经典用法，与 GLM-5 的视觉塔无关。
- `BackboneConfigMixin` 管的是 `out_features` / `out_indices`：一个 backbone 有
  `stage_names`（如 `stem / stage1 / stage2 / stage3 / stage4`），可以只导出其中几级，
  给检测、分割头做**多尺度特征金字塔**。契约的入口在这里：

<!-- src: backbone_utils.py -->
```python
    def _init_transformers_backbone(self) -> None:
        self.stage_names = self.config.stage_names
        self.config.verify_out_features_out_indices()
        # Number of channels for each stage. This is set in the transformer backbone model init
        self.num_features = None
```


**读法**：

- `self.stage_names = self.config.stage_names` 然后立刻 `verify_out_features_out_indices()`；
  如果 config 没设 `stage_names`，那句校验会抛
  `ValueError("Stage_names must be set for transformers backbones")`。
  **"必须有 stage"** 就是 backbone 契约的核心。
- 现在对照 GLM-5 的视觉塔（两条**实测**证据）：

```text
grep -c "backbone" models/glm5_next/modeling_glm5_next.py          -> 0
grep -c "stage_names|out_features|out_indices" configuration_glm5_next.py -> 0
```

  即：GLM-5 的 `Glm5NextVisionModel` 不是 `BackboneMixin`，`Glm5NextVisionConfig` 也不是
  `BackboneConfigMixin`，它没有 stage、没有 `out_features`、没有多尺度输出。
- 差别在**出口的形状**：backbone 出口是"若干张不同分辨率的特征图"（`channels` 是每个 stage 的
  通道数列表）；GLM-5 视觉塔出口是**一条 token 序列** `(144, 1536)`，直接给语言模型当输入。
  一个是 CV 任务的中间表示，一个是 VLM 的输入表示。
- 所以这一节的作用是记住这条判据：**看到 `stage_names` / `out_features` 就往 backbone 想；
  看到 `grid_thw` / `spatial_merge_size` / `pooler_output` 就往 vision tower 想。**

---

## 十、`utils/backbone_utils.py`：import 路径的弃用垫片

第三个文件只有 19 行，是两个同名类的"再继承"：

<!-- src: utils/backbone_utils.py -->
```python
import warnings

from ..backbone_utils import BackboneConfigMixin, BackboneMixin


class BackboneConfigMixin(BackboneConfigMixin):
    warnings.warn(
        "Importing `BackboneConfigMixin` from `utils/backbone_utils.py` is deprecated and will be removed in "
        "Transformers v5.10. Import as `from transformers.backbone_utils import BackboneConfigMixin` instead.",
        FutureWarning,
    )
```



<!-- src: utils/backbone_utils.py -->
```python
class BackboneMixin(BackboneMixin):
    warnings.warn(
        "Importing `BackboneMixin` from `utils/backbone_utils.py` is deprecated and will be removed in "
        "Transformers v5.10. Import as `from transformers.backbone_utils import BackboneMixin` instead.",
        FutureWarning,
    )
```


**读法**：

- `class BackboneConfigMixin(BackboneConfigMixin)`：名字相同，父类是 `..backbone_utils`
  里的真身。子类唯一的动作是 `warnings.warn(...)`。
- 警告写在**类体里**，不是 `__init__` 里 —— 所以**一 import 就触发**，不需要实例化。
  这是"把 import 路径本身当成弃用信号"的写法：
  `from transformers.utils.backbone_utils import BackboneConfigMixin` 一旦执行就报警，
  并且提示新路径 `from transformers.backbone_utils import BackboneConfigMixin`。
- 为什么要留着它：`Transformers v5.10` 之前旧路径还有用户，直接删文件会让这些人拿到
  `ImportError`（一句"找不到模块"），而不是一条**可执行的迁移提示**。
  兼容层的价值就在于把"崩溃"换成"警告 + 照常工作"。
- 对本课的意义要说清楚：**视觉塔完全绕开了这个文件**。L2-07 声明覆盖它，是因为覆盖域要求
  每个源文件都被至少一课读到；它本身只值这 6 行说明。这句话也是本课唯一一处
  "为覆盖率而读"的坦白。

---

## 十一、把形状串起来

```text
阶段                             形状                     参数（实测）
------------------------------  -----------------------  ---------------------------
pixel_values                    (576, 1176)              1176 = 3 x 2 x 14 x 14
patch_embed (Conv3d 14/2)       (576, 1024)              weight (1024,3,2,14,14)
24 x VisionBlock（packed attn） (576, 1024)              heads=16 head_dim=64 非因果
axial RoPE                      (576, 64) x 2            16 个频率，H/W 各一套
view -> permute -> downsample   (144, 1536)              Conv2d(kernel=stride=2)
merger（proj/norm/GELU/SwiGLU） (144, 1536)              dim=1536 ctx=10240（1536 是类默认值）
pooler_output -> masked_scatter (text_len + 144, 4096)   out_hidden_size 必须等于文本 hidden_size
```

**与后续课的接口**：

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| 任意尺寸的图怎么变成 28 的整数倍、怎么 patchify | L2-04, L2-05 |
| 144 个 `<|image|>` 占位符是怎么写进 prompt 的 | L2-06 |
| 视觉侧的 `Glm5NextRMSNorm` 与文本三种 RMSNorm 的关系 | L3-01 |
| SwiGLU 截断 `swiglu_limit` 的完整动机 | L3-02 |
| 注意力的后端分派（flash / sdpa / eager） | L8-01, L8-02 |
| `inputs_embeds` 进主干之后的 45 层 | L0-01, L3-08 |
| 视觉 token 要不要进 KV cache | L5-01（答案是：塔里没有 cache，主干侧照常） |

**一句话总结**：

> 视觉塔 = `Conv3d` 切 patch（576）→ 24 层非因果 packed 注意力（轴向 RoPE，16 个频率给 H/W 共用）
> → `Conv2d` 把 2x2 块合成 1 个并投影到 1536（144）→ `masked_scatter` 填进占位符；
> **标题里的「27 个 token」是错的，实测是 144。**
> 另一条容易踩的线：视觉出口宽度 `out_hidden_size` 必须等于文本 `hidden_size`，
> 否则 `masked_scatter` 那一步直接报 `Image features and image tokens do not match`。
