# L1-03 · 视觉配置与顶层嵌套

> 层：**第 1 层 · 配置与分派** ｜ 优先级：P1 ｜ 前置课：**L1-02**（`Glm5NextTextConfig` 逐字段精读）

## 学习目标

看完这一课，你应该能：

1. **手算任意分辨率产生多少个视觉 token**：说出 `patch_size` 与 `spatial_merge_size` 各自管什么，
   并算出 336×336 的图产生 **144** 个视觉 token（对应验收点 1）；
2. **解释 `image_token_id` 与 `image_start/end_token_id` 的分工**：谁决定"填几个"、谁决定"有几张"（验收点 2）；
3. 说出 `Glm5NextConfig` 怎么用 `sub_configs` 把 text / vision 两个子配置缝起来，以及
   扁平 checkpoint 的向后兼容分支为什么是"转交"而不是"搬走"。

## 覆盖的源文件（1 个）

| 文件 | 行数 | 引用块 |
|---|---|---|
| `models/glm5_next/configuration_glm5_next.py` | 321 | 11 |

> 本课的实测数字来自 `.venv` 里的探针：`Glm5NextImageProcessor.get_number_of_image_patches`、
> 一次真实 `preprocess`、以及 `Glm5NextConfig(...)` 的构造与 `from_pretrained`。
> 汇总写在 `source.md` 顶部的 `text` 块里。**这些是探针脚本，不计入覆盖率** ——
> 覆盖域只统计 `src/transformers/**/*.py`；引用到的 `image_processing_glm5_next.py`
> 与 `modeling_glm5_next.py` 同样**不计入本课覆盖率**，只作为实测证据。

## 场景（9 幕）

1. **一个文件，三个 config 类** —— 三个类的分工与嵌套关系，`sub_configs` 是唯一那根缝线
2. **视觉塔的规模** —— 24 层 / 1024 宽 / 16 头，以及四个身份字符串（`model_type`、`base_config_key`、`default_rope_type`、`attribute_map`）
3. **★ 336×336 → 144** —— `patch_size = 14` 切出 24×24 个 patch、`spatial_merge_size = 2` 合并成 12×12 个 token
4. **四个宽度** —— `hidden_size` / `intermediate_size` / `out_hidden_size` / `projection_intermediate_size` 各在哪一层
5. **★ 顶层是一个容器** —— 两个子配置 + 六个 token 编号；往下建用 `sub_configs`，往上找用 `base_config_key`
6. **★ 扁平 checkpoint 的兼容分支** —— `elif self.text_config is None` 那条 `**kwargs` 转交，以及它的副作用
7. **视觉那一半少了什么** —— `cls()` 而不是 `cls(**kwargs)`：视觉配置必须显式嵌套
8. **六个编号的分工** —— 占位符 vs 跨度标记；实测 `image_end_token_id` / `video_token_id` 引用数为 0
9. **收束** —— 五要点总表 + 两个验收练习 + 下一课指路

## 核心结论

### 1. ★ 视觉塔是一台换算器：两个数字定一个 token

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
    image_size: int | list[int] | tuple[int, int] = 336
    patch_size: int | list[int] | tuple[int, int] = 14
    rms_norm_eps: float = 1e-05
    spatial_merge_size: int = 2
```

`patch_size = 14` 把一个 14×14 的像素块切成一个 patch；`spatial_merge_size = 2` 把 2×2 个相邻 patch
合成一个视觉 token。于是：

```text
336 / 14 = 24           -> 24 x 24 = 576 个 patch
576 / 2^2 = 144         -> 144 个视觉 token
每个 token 覆盖 14 x 2 = 28 像素一条边  ->  28 x 28 像素
实测：get_number_of_image_patches(336, 336) = 576
      preprocess 后 image_grid_thw = [[1, 24, 24]]、pixel_values.shape = (576, 1176)
```

`image_size = 336` 只是**参考分辨率**：实测 100×100 的输入被放大到 112×112（16 token，撞下限），
4000×4000 的输入被压到 2492×2492（7921 token，贴着 `max_image_tokens = 8000`）。

### 2. ★ 顶层只做三件事

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
    model_type = "glm5_next"
    sub_configs = {"vision_config": Glm5NextVisionConfig, "text_config": Glm5NextTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]
```

- `sub_configs` 的**键既是字段名，也是 config.json 里的键名**，值是类；
- 子配置那边写着 `base_config_key = "text_config"` / `"vision_config"`，让子类能从同一个文件里
  **反向找到自己那一层**（实测三个类都能独立读取同一份嵌套 config.json）；
- 剩下的字段全是六个特殊 token 的编号，全部落在词表尾部 154830–154855（`vocab_size = 154880`）。

### 3. ★ 扁平 checkpoint 的兼容分支是"转交"，不是"搬走"

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
        elif self.text_config is None:
            # Flat (text-only) GLM-5.3-Flash checkpoints store the text fields at the
            # top level; forward them so `text_config` is populated for BC.
            self.text_config = self.sub_configs["text_config"](**kwargs)
```

实测：从扁平 config.json 读进来之后，`c.num_hidden_layers` 与 `c.text_config.num_hidden_layers`
**同时是 45**。副作用是 kwargs **整包**转交给 text_config、不按前缀派发 ——
`Glm5NextConfig(depth=48, patch_size=16)` 建出来的对象里 `vision_config.depth` 仍是 24，
视觉字段被 text_config 收下且**不报错**。要给视觉塔换尺寸必须显式嵌套。

### 4. 六个编号的分工（验收点 2）

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
    image_token_id: int = 154854
    video_token_id: int = 154855
    image_start_token_id: int = 154830
    image_end_token_id: int = 154831
    video_start_token_id: int = 154832
    video_end_token_id: int = 154833
```

| 角色 | 字段 | 管什么 |
|---|---|---|
| 占位符 | `image_token_id`（154854） | 一个占位符对应一个视觉 token：336×336 的图要写 **144 个** |
| 跨度标记 | `image_start_token_id`（154830） | 数出**有几张图**；与视频跨度配合区分模态 |
| 跨度标记 | `image_end_token_id`（154831） | 理论上闭合图片跨度；**实测建模代码引用 0 处**（预留席位） |

实测引用次数（`modeling_glm5_next.py`，不计入覆盖率）：`image_token_id` 2 处、
`image_start_token_id` 2 处、`video_start_token_id` 4 处、`video_end_token_id` 4 处、
`image_end_token_id` **0 处**、`video_token_id` **0 处**。
另外，图片与视频**共用同一个占位符**，所以占位符个数必须精确等于视觉特征个数，
否则前向直接抛 `Image features and image tokens do not match`。

## 与后续课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| `@strict` 校验什么、`to_dict()` 怎么序列化 | L1-01、L1-04 |
| 文本侧 45 层 / 4096 宽 / 64 头怎么来的 | L1-02、L0-02 |
| `model_type` 字符串怎么变成类、惰性映射省下了什么 | L1-05 |
| 轴向 RoPE（`default_rope_type = "axial"`）怎么排频率 | L2-07 |
| 图像处理器怎么把任意尺寸对齐到 28 的倍数 | L2-04 ~ L2-06 |
| 144 个视觉向量怎么被塞进文本流 | L2-06、L2-07 |

## 验收点

- [x] 保真门禁：11 个引用块全部逐字来自 `configuration_glm5_next.py` 且位置连续
- [x] 覆盖度门禁：1 个源文件被声明（`models/glm5_next/configuration_glm5_next.py`）
- [x] 参数门禁：无非法参数
- [x] 语法检查：`node --check` 通过，9 幕字段完整
- [x] 渲染门禁：无 JS 错误、无布局溢出、交互可用（两种分辨率）
- [ ] 自检 1：能算出 336×336 的图像产生 **144** 个视觉 token（`(336/14)² / 2²`），并说出每 token 覆盖 28×28 像素
- [ ] 自检 2：能解释 `image_token_id` 是占位符（填几个）、`image_start/end_token_id` 是跨度标记（有几张）

**一句话总结**：

> 视觉塔把 `(336 / 14)² = 576` 个 patch 按 2×2 合并成 **144** 个视觉 token，每个覆盖 28×28 像素；
> 顶层 `Glm5NextConfig` 只做三件事 —— 用 `sub_configs` 把两个子配置缝起来、用 `base_config_key` 让
> 子配置能反向找到自己、用一条"把 kwargs 转交给 text_config"的分支兼容扁平的纯文本 checkpoint。
