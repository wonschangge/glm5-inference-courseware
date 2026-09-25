# L2-07 · 视觉塔：从 336x336 像素到 27 个 token — 一张图怎样变成 144 个视觉 token

> 层：**L2 · 输入流水线** ｜ 优先级：P0 ｜ 前置课：`L2-06`（多模态处理器装配）

> ★ **实测订正（先读这一段）**：标题里的「27 个 token」**与源码不符**。
> `(336/14)^2 / 2^2 = 576 / 4 = 144`；实测（`_data/recon/probe_l207.py`）一张 336x336 的图
> 给出 `image_grid_thw = [[1, 24, 24]]`，视觉塔 `pooler_output` 的形状是 `(144, 1536)`。
> 本课的正文、动画、练习与验收点一律用 **144**；「27」只作为标题里的错误出现一次。

## 学习目标

看完这一课，你应该能：

1. **手算一张 336x336 的图产出多少视觉 token**，并用实测形状核对（验收点 1）；
2. **解释轴向 RoPE 与一维 RoPE 的区别** —— 频率个数、坐标个数、以及 64 维是怎么拼出来的（验收点 2）；
3. 说出 patch embed 的「卷积等价形式」，以及 patch merger 的 2x2 合并发生在序列的哪一步。

## 覆盖的源文件（3 个）

| 文件 | 行数 | 引用块 |
|---|---|---|
| `models/glm5_next/modeling_glm5_next.py` | 2444 | 10 |
| `backbone_utils.py` | 387 | 2 |
| `utils/backbone_utils.py` | 19 | 2 |

> 行数是 `wc -l` 的**实测**值。作业书里写的 2445 / 388 / 20 是按 `len(text.split("\n"))` 数的，
> 会把文件末尾换行之后那个空串也算成一行。
>
> 本课还引用了 `_data/recon/probe_l207.py` 的**实测输出**（写在 `source.md` 的 `text` 块里）。
> **它是脚本，不计入覆盖率** —— 覆盖域只统计 `src/transformers/**/*.py`。
>
> `backbone_utils.py` 与 `utils/backbone_utils.py` **不在视觉塔的调用链上**：实测
> `modeling_glm5_next.py` 里 `backbone` 出现 0 次。本课读它们是为了**划清"backbone 契约"与
> "vision tower 契约"的边界**（source.md 第九、十节）。

## 场景（9 幕）

1. **全景** —— 336x336 → 24x24 个 patch 位置（576）→ 每 2x2 合成 1 个 → **144** 个 1536 维向量
2. **patch embed 的卷积等价形式** —— `kernel == stride` ⇒ 每个位置一次 `Linear(1176 -> 1024)`
3. **★ 实测订正** —— 三种独立算法都给 144；`576 / 27` 连整除都不成立
4. **★ 轴向 RoPE（上）** —— `head_dim // 4 = 16` 个频率，H/W 各一套（一维 RoPE 是 32 个）
5. **★ 轴向 RoPE（下）** —— `cat([h, w])` 拼成 32 维、复制成 64 维；`rotate_half` 把 i 与 i+32 配对
6. **patch merger** —— 块主序 4 个 token → 1 个；先投影到 1536，再过 merger
7. **塔内** —— 一次 qkv、逐头 RMSNorm、`is_causal = False`、`cu_seqlens` 的 packed 变长
8. **拼回文本** —— `masked_scatter` 把 144 个向量填进占位符；主干从此只看见一条序列
9. **收束** —— 形状总表 + backbone / tower 的判据 + 两道练习

## 核心结论

### 1. patch embed 是「kernel == stride」的卷积，等价于逐位置的线性层

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
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


实测权重 `(1024, 3, 2, 14, 14)`、`stride = kernel = (2, 14, 14)`：滑窗互不重叠，
所以 `(576, 1176)` 进、`(576, 1024)` 出，**位置一一对应，不混合邻域**。
写成 `Conv3d` 而不是 `Linear`，是为了让框架代劳切窗、并让权重形状自带 `(T, P, P)` 语义。

### 2. ★ 「一张图几个 token」是运行时现算的：144，不是 27

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_outputs = self.visual(pixel_values, grid_thw=image_grid_thw, **kwargs)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(vision_outputs.pooler_output, split_sizes)
        vision_outputs.pooler_output = image_embeds

        return vision_outputs
```


`grid_thw.prod(-1)` = `1 x 24 x 24 = 576`，除以 `spatial_merge_size ** 2 = 4` 得 **144**。
三种算法一致：`(336/14)^2 / 4 = 144`、`(336/14/2)^2 = 144`、代码实跑 `= 144`。
处理器端的 `replace_image_token` 用的是同一个公式 —— 两边不一致会在 `masked_scatter` 处报
`Image features and image tokens do not match`。

### 3. ★ 轴向 RoPE：16 个频率给 H/W 共用，拼成 32 维后复制成 64 维

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


`freq` 的形状是 `(N, 2, 16)`：`[:, 0]` 是 h 的频率、`[:, 1]` 是 w 的频率。
`cat([freq_h, freq_w])` → `(N, 32)`（前 16 维归 h、后 16 维归 w），
再 `cat([freq_hw, freq_hw])` → `(N, 64)` 铺满 `head_dim`。
**复制是为了配对**：`rotate_half` 把第 i 维与第 i+32 维配成一次二维旋转，两边角度相同。

一维 RoPE（文本）用 `head_dim / 2 = 32` 个频率、一个位置标量；
轴向 RoPE 用 `head_dim / 4 = 16` 个频率、两个坐标 `(h, w)`。实测判据：

| 改动 | cos[0:16] | cos[16:32] |
|---|---|---|
| 只改 w | 不变 | 变 |
| 只改 h | 变 | 不变 |

### 4. patch merger：块主序的 4 个 token 合成 1 个，先投影再 merger

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        hidden_states = self.post_layernorm(hidden_states)

        hidden_states = hidden_states.view(
            -1, self.spatial_merge_size, self.spatial_merge_size, hidden_states.shape[-1]
        )
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.downsample(hidden_states).view(-1, self.config.out_hidden_size)
```


`view(-1, 2, 2, 1024)` 敢直接写，是因为 `get_vision_position_ids` 已经把位置编号按块主序排好
（实测前 8 个 `(h, w)`：`[0,0] [0,1] [1,0] [1,1] [0,2] [0,3] [1,2] [1,3]`）。
`downsample` 是 `Conv2d(kernel=2, stride=2)`、`1024 -> 1536`；**merger 的 `dim` 拿到的是
`out_hidden_size = 1536`，不是 1024**。出口两个名字要分清：

| 字段 | 形状（实测） | 含义 |
|---|---|---|
| `last_hidden_state` | `(144, 1536)` | 已合并、已投影，但没过 merger 的 MLP |
| `pooler_output` | `(144, 1536)` | 过完 merger，**真正拼进文本的那一串** |

**★ 宽度约束（实测）**：`out_hidden_size` 不是自由的 —— `masked_scatter` 与
`get_placeholder_mask` 里的 `torch_compilable_check` 共同要求它**等于文本 `hidden_size`**。
小配置实测（`probe_l207.py` 第 [7] 节）：文本 `hidden=64` 配视觉 `out_hidden=32` 时，
前向就卡在这一步报 `Image features and image tokens do not match`；配成 64 则通过。
而类默认值是 `hidden_size = 4096` / `out_hidden_size = 1536`，`144 x 4096 = 589824`
而 `144 x 1536 = 221184` —— **两个默认值不能放在一起用**，真实检查点必然覆盖 `out_hidden_size`。

### 5. 视觉塔不是 backbone

`backbone_utils.py` 管的是 `stage_names` / `out_features` / `out_indices` ——
多尺度特征图那一套；GLM-5 的视觉塔是**单出口**的（一串 token）。
实测：`modeling_glm5_next.py` 里 `backbone` 出现 **0** 次，
`configuration_glm5_next.py` 里 `stage_names / out_features / out_indices` 也是 **0** 次。
`utils/backbone_utils.py` 只是 import 路径迁移的弃用垫片（警告写在类体里，一 import 就触发）。

### 实测数字（`_data/recon/probe_l207.py`）

| 量 | 实测值 | 含义 |
|---|---|---|
| `image_size` / `patch_size` | 336 / 14 | 网格 24x24 |
| `spatial_merge_size` | 2 | 576 → 144 |
| patch 数 → token 数 | 576 → **144** | `(336/14)^2 / 2^2` |
| `hidden_size` / `depth` | 1024 / 24 | 塔内宽度与层数 |
| `num_heads` / `head_dim` | 16 / 64 | 频率个数 `64/4 = 16` |
| `out_hidden_size` | 1536（**类默认值**） | 必须等于文本 `hidden_size`；文本默认 4096，实测不等就报错 |
| `projection_intermediate_size` | 10240 | merger 的 SwiGLU 中间维 |
| `image_token_id` | 154854 | 图与视频**共用**，靠 span 区分 |

## 与后续课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| 任意尺寸的图怎么变成 28 的整数倍、怎么 patchify | L2-04, L2-05 |
| 144 个 `<|image|>` 占位符怎么写进 prompt | L2-06 |
| 视觉侧的 `Glm5NextRMSNorm` 与文本三种 RMSNorm 的关系 | L3-01 |
| SwiGLU 截断 `swiglu_limit = 10.0` 的完整动机 | L3-02 |
| 注意力后端分派（flash / sdpa / eager） | L8-01, L8-02 |
| `inputs_embeds` 进主干之后的 45 层 | L0-01, L3-08 |
| 视觉 token 要不要进 KV cache | L5-01（塔里没有 cache，主干侧照常） |

## 验收点

- [x] 保真门禁：14 个引用块全部逐字来自 3 个源文件（10 + 2 + 2）
- [x] 覆盖度门禁：3 个源文件被声明
- [x] 参数门禁：无非法参数
- [x] 语法检查：`node --check` 通过，9 幕字段完整
- [x] 渲染门禁：无 JS 错误、无布局溢出、交互可用（两种分辨率）
- [x] 实测核对：`pooler_output = (144, 1536)` 与手算 `(336/14)^2 / 2^2 = 144` 一致
- [ ] 自检（验收点 1）：能手算 `(336/14)^2 / 2^2 = 144`，并说出实际 token 数从哪一行代码来
- [ ] 自检（验收点 2）：能解释轴向 RoPE 与一维 RoPE 的区别（频率个数 / 坐标个数 / 64 维的拼法）

**一句话总结**：

> 视觉塔 = `Conv3d` 切 patch（576）→ 24 层非因果 packed 注意力（轴向 RoPE，16 个频率给 H/W 共用）
> → `Conv2d` 把 2x2 块合成 1 个并投影到 1536（**144**）→ `masked_scatter` 填进占位符；
> **标题里的「27 个 token」是错的，实测是 144。**
> 另一条容易踩的线：视觉出口宽度 `out_hidden_size` 必须等于文本 `hidden_size`。
