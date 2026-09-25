<!-- glm5-coverage
models/glm5_next/configuration_glm5_next.py
-->

# L1-03 · 视觉配置与顶层嵌套 — 源文件

**这一课只回答两个问题：多模态的那一半配置长什么样；两个子配置是怎么被缝进同一个顶层的。**

视角：`Glm5NextVisionConfig` 是一台**「像素 → token」的换算器**（两个数字决定一个 token 覆盖多大一块图）；
`Glm5NextConfig` 是一个**容器**（它自己不持有任何形状，只持有两个子配置和六个特殊 token 的编号）。

| 文件 | 行数 | 本课引用块 |
|---|---|---|
| `models/glm5_next/configuration_glm5_next.py` | 321 | 11 |

**实测的换算结果**（在 `.venv` 里跑 `Glm5NextImageProcessor.get_number_of_image_patches`、
一次真实 `preprocess`、以及 `Glm5NextConfig(...)`，不是估算）：

```text
Glm5NextVisionConfig() 默认值
    depth = 24        hidden_size = 1024        num_heads = 16
    patch_size = 14   spatial_merge_size = 2    temporal_patch_size = 2
    image_size = 336  out_hidden_size = 1536    projection_intermediate_size = 10240

一次 336x336 的图像预处理
    pixel_values.shape = (576, 1176)     # 576 个 patch；每个 patch 3 x 2 x 14 x 14 = 1176 个数
    image_grid_thw     = [[1, 24, 24]]   # t = 1, h = 24, w = 24
    patch 数           = 24 x 24         = 576
    合并后的视觉 token = 576 / 2^2       = 144
    每个视觉 token 覆盖 = 14 x 2 = 28 像素一条边  ->  28 x 28 像素

同一套换算器在别的分辨率上（边界由 min_image_tokens = 16 / max_image_tokens = 8000 卡住）
    112x112    ->    64 patch ->    16 token    （下限）
    336x336    ->   576 patch ->   144 token    （image_size 的默认值）
    504x504    ->  1296 patch ->   324 token
    100x100    ->    64 patch ->    16 token    （被放大到 112x112：对齐到 28 的倍数）
    4000x4000  -> 31684 patch ->  7921 token    （被压到 2492x2492：贴着上限）

扁平 checkpoint（文本字段摊在顶层）读进来之后
    Glm5NextConfig.from_pretrained(...)  ->  text_config 已填充，num_hidden_layers = 45
    同一个对象上 c.num_hidden_layers 与 c.text_config.num_hidden_layers 同时是 45
    顶层传 depth=48 / patch_size=16      ->  vision_config 仍是 24 / 14（被 text_config 收下）
```

> 上面这段是**实测汇总**，用 `text` 块标出 —— 它不是源文件里的一行，因此不参与保真校验。
> 其中 `image_processing_glm5_next.py` 与本课的覆盖域无关，**不计入覆盖率**；
> 它只作为"这两个数字到底换算出什么"的证据出现在这里。
> 下面每一段 `python` 块都是 `configuration_glm5_next.py` 的逐字引用。

---

## 一、一个文件，三个 config 类

三个类共用同一套装饰器与同一个 checkpoint 锚点：

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
@auto_docstring(checkpoint="zai-org/GLM-5.3-Flash")
@strict
class Glm5NextTextConfig(PreTrainedConfig):
```

**读法**：
- `@auto_docstring(checkpoint=...)` 是"这份文档是从某个真实 checkpoint 的 config 反推的"标记（L1-04 展开）；
  `@strict` 让 dataclass 的字段在**赋值那一刻**就做类型校验 —— 实测 `Glm5NextVisionConfig(depth="24")`
  在构造时就抛 `Validation error for field 'depth': Field 'depth' expected int, got str`，
  而不是拖到建模型时才炸。
- 三个类分别是 `Glm5NextTextConfig`（L29）、`Glm5NextVisionConfig`（L232）、`Glm5NextConfig`（L269）。
  **L1-02 已经把第一个讲透了**，本课只加后两个 —— 也就是"多模态那一半"和"顶层那一个"。

每个 config 类都靠一个字符串在注册表里挂号：

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
    model_type = "glm5_next_text"
    keys_to_ignore_at_inference = ["past_key_values"]
```

**读法**：
- `model_type` 是 `AutoConfig` 那张"字符串 → 类"映射表的**键**：checkpoint 的 `config.json` 里写着
  `"model_type": "glm5_next"`，`AutoConfig` 就把它换成 `Glm5NextConfig`。本文件里三个类各有一个键：
  `glm5_next` / `glm5_next_text` / `glm5_next_vision`。**惰性映射怎么做到"用时才 import"，是 L1-05 的正题。**
- `keys_to_ignore_at_inference` 是另一张清单，管的是"哪些字段不该被当成推理返回值"。
  三个类写的都是 `past_key_values`（L70、L293）。

---

## 二、★ 视觉塔是一台换算器：两个数字定一个 token

视觉配置类头，以及它的四个"身份字符串"：

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
@auto_docstring(checkpoint="zai-org/GLM-5.3-Flash")
@strict
class Glm5NextVisionConfig(PreTrainedConfig):
    r"""
    out_hidden_size (`int`, *optional*, defaults to 1536):
        The output hidden size of the vision model.
    projection_intermediate_size (`int`, *optional*, defaults to 10240):
        The projection_intermediate_size size for the vision patch merger.
    swiglu_limit (`float`, *optional*, defaults to 10.0):
        Clamp limit applied to the vision SwiGLU gate/up projections.
    """

    model_type = "glm5_next_vision"
    base_config_key = "vision_config"
    default_rope_type = "axial"
    attribute_map = {"num_attention_heads": "num_heads"}
```

**读法**：
- 文档字符串只列了 3 个字段 —— 那是这个类**真的想让用户调**的东西（交付宽度、投影中间宽度、SwiGLU 夹紧值）。
  其余字段（深度、宽度、头数、patch 大小）都是"照着参考模型抄下来的规模"，不是旋钮。
- 四个身份字符串各管一件事：`model_type` 是注册表的键；`base_config_key = "vision_config"` 说明
  **这份子配置在顶层文件里住在哪个键下面**（第三节末尾展开）；`default_rope_type = "axial"` 是本文件里
  唯一一句"视觉塔的位置编码与文本主干不同"的声明 —— 轴向 RoPE（高/宽两套频率）在 L2-07 展开；
  `attribute_map = {"num_attention_heads": "num_heads"}` 是一条**别名**：实测
  `getattr(v, "num_attention_heads")` 能读到 16，而 `to_dict()` 里写的键是 `num_heads`。
- 顺带记住一个不对称：视觉塔的 `attention_bias` 是 `True`，文本主干是 `False`（L132）。
  这不是笔误，两半塔的注意力实现本来就不一样（L2-07）。

几何字段 —— 本课最该记住的一段：

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
    depth: int = 24
    hidden_size: int = 1024
    hidden_act: str = "silu"
    attention_bias: bool = True
    attention_dropout: float | int = 0.0
    num_heads: int = 16
    in_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 336
    patch_size: int | list[int] | tuple[int, int] = 14
    rms_norm_eps: float = 1e-05
    spatial_merge_size: int = 2
    temporal_patch_size: int | list[int] | tuple[int, int] = 2
```

**读法**：
- 三个数决定一次换算：`patch_size = 14` 把 14×14 的像素块切成一个 patch；
  `spatial_merge_size = 2` 把 2×2 个相邻 patch 合成**一个视觉 token**；
  于是每个 token 覆盖 `14 × 2 = 28` 像素一条边，也就是 **28 × 28 像素**。
- 336 × 336 的代数是干净的：`336 / 14 = 24` → 24 × 24 = **576** 个 patch → `576 / 2² =` **144** 个视觉 token。
  实测（`get_number_of_image_patches(336, 336)`）返回 576；真实 `preprocess` 之后
  `image_grid_thw = [[1, 24, 24]]`、`pixel_values.shape = (576, 1176)`。
- `image_size = 336` 是**参考分辨率，不是硬约束**：真正吃进去的尺寸由图像处理器对齐到
  `patch_size × spatial_merge_size = 28` 的整数倍。实测 100×100 的输入被放大到 112×112
  （64 patch → 16 token，正好撞上下限），4000×4000 的输入被压到 2492×2492（7921 token）。
  **"多少 token"是算出来的，"多大图"是让出来的。**
- `temporal_patch_size = 2` 是时间维的同类参数：patch 张量在时间方向也按 2 打包，
  所以每个 patch 展平后是 `3 × 2 × 14 × 14 = 1176` 个数（实测）。图片这一支 t = 1，
  它只影响打包宽度，不影响 token 数。
- `depth = 24` / `hidden_size = 1024` / `num_heads = 16` 是塔本身的规模。对比文本主干的
  45 层 / 4096 宽 / 64 头（L0-01、L1-02）：**视觉塔是个小得多的模型，它的贵不在参数，而在 token 数。**

输出与投影侧的四个宽度：

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
    out_hidden_size: int = 1536
    intermediate_size: int = 4096
    initializer_range: float = 0.02
    rope_parameters: dict | None = None
    projection_intermediate_size: int = 10240
    swiglu_limit: float = 10.0
```

**读法**：
- 四个宽度属于四个不同的位置，别混：`hidden_size = 1024` 是塔内每个 patch 的宽度；
  `intermediate_size = 4096` 是塔内 FFN 的宽度；`out_hidden_size = 1536` 是**合并块交付出去的宽度**，
  也就是"一个视觉 token 交付时是多少维"；`projection_intermediate_size = 10240` 是合并块里那层
  SwiGLU 的中间宽度 —— 它比塔内 FFN 的 4096 宽得多（1536 → 10240 → 1536），因为合并四宫格
  要在一次投影里做完空间压缩。
- `swiglu_limit = 10.0` 与文本侧的 `swiglu_limit`（L142）**同名同值但是两份独立字段**：
  一个夹视觉塔的 gate/up，一个夹文本 FFN 的 gate/up。文档字符串里写得很明确：
  `Clamp limit applied to the vision SwiGLU gate/up projections`。
- `out_hidden_size = 1536` 与文本的 `hidden_size = 4096` **不是同一个宽度**。1536 是视觉塔的交付宽度；
  把它接进文本流是另一层的事（L2-06 / L2-07 的装配与拼接）。本文件只负责把这两个数写清楚。
- `initializer_range` / `rope_parameters` 是标准字段：后者是 `None`，说明轴向 RoPE 的频率走
  `default_rope_type = "axial"` 这条默认路径，而不是在这里显式给参数。

---

## 三、★ 顶层 config 是一个容器

顶层这一段的每个字段都只有两种：**两个子配置**，和**六个特殊 token 的编号**。

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
    model_type = "glm5_next"
    sub_configs = {"vision_config": Glm5NextVisionConfig, "text_config": Glm5NextTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 154854
    video_token_id: int = 154855
    image_start_token_id: int = 154830
    image_end_token_id: int = 154831
    video_start_token_id: int = 154832
    video_end_token_id: int = 154833
    tie_word_embeddings: bool = False
```

**读法**：
- `sub_configs` 是**唯一**把两个子配置缝到顶层的声明：字典的键（`"vision_config"` / `"text_config"`）
  既是字段名，也是**它在 config.json 里的键名**；值是类本身。顶层类不继承任何东西 ——
  它靠这张表知道"这两个字段该用哪个类来实例化"。
- 字段类型写成 `dict | PreTrainedConfig | None`，是为了让**三条入口**都能走：传字典（配置文件的形态）、
  传已经建好的对象（代码里手搓的形态）、什么都不传（用默认值）。这三种情况正是
  `__post_init__` 里的三个分支（第四节）。
- 六个编号全落在词表尾部：`154830` 起、`154855` 止，而 `vocab_size = 154880`、`pad_token_id = 154820`
  （L101、L126）。**这不是随手挑的数：它们是词表里预留的一段特殊 token 区。**
- `tie_word_embeddings = False`（顶层与文本侧都是 False）意味着输入词嵌入与 `lm_head` 是两份权重。

子配置自己那边，也写着"我住在哪"：

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
    # `"full"` runs the indexer, `"shared"` reuses the previous full layer's index mask.
    indexer_types: list[str] | None = None
    base_config_key = "text_config"
    swiglu_limit: float = 10.0
```

**读法**：
- `base_config_key = "text_config"` 这一行的作用是**让子配置类能直接打开顶层文件**：
  `from_pretrained` 时先看这个键，如果顶层 JSON 里有 `text_config`，就自动钻进去取自己的字段。
  实测：把同一份嵌套的 `config.json` 分别喂给 `Glm5NextConfig` / `Glm5NextTextConfig` /
  `Glm5NextVisionConfig`，三个类都能读出来（两个子类读到的分别是 45 层、24 层深的那一份）。
- 这就是"嵌套"与"字符串"的交汇点：**顶层靠 `sub_configs` 往下建，子配置靠 `base_config_key` 往上找。**
  两边的键名必须**逐字一致**（`"text_config"` / `"vision_config"`），否则顶层建得出来、子类却找不到自己。

---

## 四、★ 扁平 checkpoint 的兼容分支

顶层 `__post_init__` 的前半段，就是"text_config 从哪来"的全部答案：

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
    def __post_init__(self, **kwargs):
        if isinstance(self.text_config, dict):
            self.text_config = self.sub_configs["text_config"](**self.text_config)
        elif self.text_config is None:
            # Flat (text-only) GLM-5.3-Flash checkpoints store the text fields at the
            # top level; forward them so `text_config` is populated for BC.
            self.text_config = self.sub_configs["text_config"](**kwargs)
```

**读法**：
- 三个分支：字典 → 建对象；`None` → **把顶层剩下的 kwargs 原样转交**；已经是对象 → 什么都不做。
  第二条分支是为**扁平 checkpoint** 准备的：GLM-5.3-Flash 早期只有纯文本权重，config.json 里
  没有 `text_config` 这一层，文本字段直接摊在顶层。
- 注释里那句 `forward them so text_config is populated for BC` 的关键词是**转交**，不是搬走。
  实测：从扁平文件读进来之后，同一个对象上 `c.num_hidden_layers` 与
  `c.text_config.num_hidden_layers` **同时是 45**，再 `to_dict()` 时两处都在。
- 这条分支还有个会咬人的地方：它把 kwargs **整包喂给 text_config，而不是按前缀派发**。
  实测：`Glm5NextConfig(depth=48, patch_size=16, num_hidden_layers=4)` 建出来的对象里，
  `vision_config.depth` 仍然是 **24**、`patch_size` 仍然是 **14**，而 `text_config.depth` 变成了 **48**
  —— 视觉字段被文本配置收下了（`@strict` 的 dataclass 接受额外 kwargs，所以**不报错**）。
  要给视觉塔换尺寸，必须走嵌套那条路：`Glm5NextConfig(vision_config={"depth": 48})` 实测生效。

视觉那一半紧跟其后，形状相似但**少一条回退**：

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
        if isinstance(self.vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        super().__post_init__(**kwargs)
```

**读法**：
- 字典分支与文本侧完全对称；`None` 分支**没有 `**kwargs`** —— 它就是一对空括号：直接建一份全默认的视觉配置。
  这是有意为之：扁平 checkpoint 是纯文本的，里面根本不存在视觉字段，"回退"无从谈起；
  而一个多模态模型**必须有**视觉塔，所以这里给的是"默认塔"而不是"空"。
- 两段的不对称本身就是一份文档：**文本配置是"可从顶层反向填充的"，视觉配置是"必须显式嵌套的"。**
  读到一份 checkpoint 时如果发现视觉塔的分辨率/深度不对，先检查是不是走了扁平分支。
- 最后一行 `super().__post_init__(**kwargs)` 把 kwargs 继续上交 —— `@strict` 的校验、dtype 归一化等
  都发生在父类里（L1-01）。这也解释了上一条实测里"顶层也留下了 `depth=48`"：kwargs 一路上交，
  沿途被记成了对象属性。

---

## 五、六个 token 编号的分工

类文档字符串先把六个编号各自的身份写死：

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
@auto_docstring(checkpoint="zai-org/GLM-5.3-Flash")
@strict
class Glm5NextConfig(PreTrainedConfig):
    r"""
    image_token_id (`int`, *optional*, defaults to 154854):
        The image token index to encode the image prompt.
    video_token_id (`int`, *optional*, defaults to 154855):
        The video token index to encode the video prompt.
    image_start_token_id (`int`, *optional*, defaults to 154830):
        The image start token index to encode the start of image.
    image_end_token_id (`int`, *optional*, defaults to 154831):
        The image end token index to encode the end of image.
    video_start_token_id (`int`, *optional*, defaults to 154832):
        The video start token index to encode the start of video.
    video_end_token_id (`int`, *optional*, defaults to 154833):
        The video end token index to encode the end of video.
```

**读法**：
- **两类角色的分工**：`image_token_id` 是**占位符** —— 它出现在 `input_ids` 里，一个占位符对应
  一个视觉 token（336×336 的图就是 **144 个占位符**），前向时被整体替换成 144 个向量；
  `image_start_token_id` / `image_end_token_id` 是**跨度标记** —— 它们标记"这张图从哪开始、到哪结束"，
  用来**数出有几张图**，而不是用来填向量。
- 占位符是图片与视频**共用**的：建模代码里那句 `img token == vid token` 说的就是这件事 ——
  光看占位符分不出模态，必须靠 `video_start` / `video_end` 划出的跨度来区分。代价是：
  **占位符个数必须精确等于视觉特征个数**，否则前向里直接抛
  `Image features and image tokens do not match`。
- 实测这六个编号在建模代码（`modeling_glm5_next.py`，**不计入本课覆盖率**）里的引用次数：

```text
image_token_id        2 处（占位符的识别与替换）
image_start_token_id  2 处（数图片张数：is_image 的判据之一）
image_end_token_id    0 处   <- 声明了，但当前实现里没有消费者
video_start_token_id  4 处（视频跨度的开）
video_end_token_id    4 处（视频跨度的闭，以及"当前是否在视频跨度内"的累积和）
video_token_id        0 处   <- 同样没有消费者
```

- 结论要比注释说得更准：**`image_end_token_id` 与 `video_token_id` 目前是"预留席位"。**
  分工的实质是三重：`image_token_id` 决定**填几个**，`image_start_token_id` 决定**有几张**，
  `video_start` / `video_end` 决定**这段占位符属于哪种模态**。注释是按设计意图写的，
  引用次数是按当前实现数的 —— 两者不一致时，以实现为准（这正是 L0-05 订正过的那类问题）。

---

## 六、与后面课的接口

文件最后一行把对外名字收成三个：

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
__all__ = ["Glm5NextConfig", "Glm5NextTextConfig", "Glm5NextVisionConfig"]
```

**读法**：
- `__all__` 是"这个模块对外的三个类"，也是**唯一的对外清单**：本文件里没有别的公开类
  —— 所有形状、所有排班表、所有 token 编号都挂在这三个类上。
- 到这一课为止，"配置"这条线讲完了三件事：公共契约（L1-01）、文本配置逐字段（L1-02）、
  视觉配置与顶层嵌套（本课）。接下来是配置怎么被序列化与继承（L1-04），
  以及**字符串如何找到这三个类**（L1-05 的惰性映射）。

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| `@strict` 到底校验什么、`to_dict()` 怎么序列化 | L1-01、L1-04 |
| 文本侧的 45 层 / 4096 宽 / 64 头怎么来的 | L1-02、L0-02 |
| `model_type` 字符串怎么变成类、惰性映射省下了什么 | L1-05 |
| 轴向 RoPE（`default_rope_type = "axial"`）怎么排频率 | L2-07 |
| 图像处理器怎么把任意尺寸对齐到 28 的倍数 | L2-04 ~ L2-06 |
| 144 个视觉向量怎么被塞进文本流 | L2-06、L2-07 |
| `attention_bias` 在视觉塔与文本主干为什么不同 | L2-07 |

**一句话总结**：

> 视觉塔把 `(336 / 14)² = 576` 个 patch 按 2×2 合并成 **144** 个视觉 token，每个覆盖 28×28 像素；
> 顶层 `Glm5NextConfig` 只做三件事 —— 用 `sub_configs` 把两个子配置缝起来、用 `base_config_key` 让
> 子配置能反向找到自己、用一条"把 kwargs 转交给 text_config"的分支兼容扁平的纯文本 checkpoint。
