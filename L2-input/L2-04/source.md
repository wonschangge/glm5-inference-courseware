<!-- glm5-coverage
image_processing_base.py
image_processing_utils.py
image_processing_backends.py
-->

# L2-04 · 图像处理基座：尺寸、归一化与后端 — 源文件

**这一课只回答一个问题：一张 `H×W×3` 的图，是怎么变成模型能吃的一串向量的？**
以及这条路上**哪一步是可换的**（后端），**哪一步是不能换的**（契约）。

视角：把 `BaseImageProcessor.preprocess` 当成一条流水线，看它怎么把「校验 / 默认值 /
尺寸标准化」做完之后，把真正的像素运算**整体委派**给一个后端类。

| 文件 | 行数（实测） | 本课引用块 |
|---|---|---|
| `image_processing_utils.py` | 697 | 4 |
| `image_processing_base.py` | 510 | 3 |
| `image_processing_backends.py` | 667 | 7 |

> **★ 实测订正（1）：本课三个源文件的行数比作业书写各少 1 行。**
> `wc -l` 与 `read` 工具的数法都是 510 / 697 / 667；作业书写的是 511 / 698 / 668。
> 差 1 的原因见下面的实测输出 —— 不是文件被改过，是**计数口径**不同。

```text
$ wc -l image_processing_base.py image_processing_utils.py image_processing_backends.py
  510 image_processing_base.py
  697 image_processing_utils.py
  667 image_processing_backends.py

# 三个文件都以换行符结尾，所以 str.split("\n") 会比真实行数多出一个 ""：
#   image_processing_base.py   真实 510 行   split("\n") = 511
#   image_processing_utils.py  真实 697 行   split("\n") = 698
#   image_processing_backends.py 真实 667 行 split("\n") = 668
# 作业书的三个数字正好等于 split("\n") 的结果，即"换行符个数 + 1"。
# 本课采用 read 工具的口径（最后一行的行号 = 真实行数），
# 因此 lesson.js 里所有 codeStart 都以 510 / 697 / 667 为界。
```

> **★ 实测订正（2）：不是「三个后端（PIL / torchvision / numpy）」。**
> `image_processing_backends.py` 里只有**两个**后端类，`backend` 属性也只有两个取值：

```text
$ python -c "import transformers.image_processing_backends as B; print([n for n in dir(B) if n.endswith('Backend')])"
['PilBackend', 'TorchvisionBackend']
   TorchvisionBackend.backend = 'torchvision'      （矩阵/tensor 运算，可上 GPU）
   PilBackend.backend         = 'pil'              （np.ndarray 运算，只走 CPU）
   hasattr(B, 'NumpyBackend') = False

# numpy 是**底层实现库**，不是第三个后端：
# PilBackend 的 resize / rescale / normalize / center_crop 全部转调
# image_transforms 里的 numpy 实现（本课第 12 个引用块就是这段转调）。
# AutoImageProcessor 的 backend 参数也只认 'pil' / 'torchvision' / 自定义注册名。
```

**实测环境**（`_data/recon/probe_l204.py`，用仓库自带 venv 跑出）：

```text
transformers = 5.18.0.dev0
torch        = 2.14.0+cpu
torchvision  = 0.29.0+cpu
is_vision_available()      = True
is_torchvision_available() = True
```

> 上面这段是**实测输出**，用 `text` 块标出 —— 它不是源文件的逐字引用，
> 不参与保真校验。下面每一段 `python` 块都是逐字引用。

---

## 一、★ 契约：所有模型处理器共用的那个 `preprocess`

这是整个基座的中心。`BaseImageProcessor.preprocess` 只做四件事：**校验类型 → 补默认值 →
标准化尺寸 → 委派给 `_preprocess_image_like_inputs`**。它自己一行像素运算都不写。

<!-- src: image_processing_utils.py -->
```python
        # Perform type validation on received kwargs
        validate_typed_dict(self.valid_kwargs, kwargs)

        # Set default kwargs from self
        for kwarg_name in self._valid_kwargs_names:
            kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))

        # Update kwargs that need further processing before being validated
        kwargs = self._standardize_kwargs(**kwargs)

        # Validate kwargs
        self._validate_preprocess_kwargs(**kwargs)

        image_like_kwargs = {} if image_like_kwargs is None else image_like_kwargs

        return self._preprocess_image_like_inputs(images, *args, **image_like_kwargs, **kwargs)
```

**读法**（讲"为什么"）：

- **`validate_typed_dict(self.valid_kwargs, kwargs)` 是第一道门，而且它是"按类的"** ——
  `valid_kwargs` 是类属性，默认 `ImagesKwargs`，模型可以换成自己的子类
  （`Glm5NextImageProcessorKwargs` 就是这么加 `patch_size` / `merge_size` 的）。
  **契约不是"能传什么"，而是"这个类声明了能传什么"。**
- **`getattr(self, kwarg_name, None)` 是"默认值只有一份"的实现方式**：
  默认值住在**类属性**上（`rescale_factor = 1 / 255`、`do_resize = True`……），
  调用时如果不传，就把类属性抄进 kwargs。所以读一个处理器的默认行为，
  永远要去读它的类属性，而不是读 `preprocess` 的签名。
- **`_standardize_kwargs` 在"校验"之前**。这一步不是锦上添花：
  `size=224` 和 `size={"height":224,"width":224}` 是两种合法输入，
  必须先在 `_standardize_kwargs` 里统一成 `SizeDict`，后面的 `_validate_preprocess_kwargs`
  才有一条唯一的判据。
- **最后一行是分水岭**：`preprocess` 到此结束，后面全是后端的事。
  这就是为什么 `BaseImageProcessor` 自己**不能**被实例化来用 —— 它的
  `process_image` 和 `_preprocess` 都是 `raise NotImplementedError`。

**回顾 L0-01**：`Glm5NextTextModel.forward` 也是"先做校验和默认值、再交给层"的同一手法。
basement 类负责**编排**，具体干活的方法留给子类 —— 这是这套代码库反复出现的分层习惯。

---

## 二、`_standardize_kwargs`：尺寸的四种写法归一

尺寸是图像处理里歧义最多的参数：`224` 到底是 `224×224` 还是"短边 224"？

<!-- src: image_processing_utils.py -->
```python
        if kwargs is None:
            kwargs = {}
        if size is not None and not isinstance(size, SizeDict):
            size = SizeDict(**get_size_dict(size=size, default_to_square=default_to_square))
        if crop_size is not None and not isinstance(crop_size, SizeDict):
            crop_size = SizeDict(**get_size_dict(crop_size, param_name="crop_size"))
        if pad_size is not None and not isinstance(pad_size, SizeDict):
            pad_size = SizeDict(**get_size_dict(size=pad_size, param_name="pad_size"))
        if isinstance(image_mean, list):
            image_mean = tuple(image_mean)
        if isinstance(image_std, list):
            image_std = tuple(image_std)

        kwargs["size"] = size
        kwargs["crop_size"] = crop_size
        kwargs["pad_size"] = pad_size
        kwargs["image_mean"] = image_mean
        kwargs["image_std"] = image_std
```

**读法**：

- **注意参数名的不一致**：`size` 走 `get_size_dict(size=size, ...)`，
  而 `crop_size` / `pad_size` 走的是**第一个位置参数**。这不是笔误 ——
  `get_size_dict` 的第一个形参就叫 `size`，`param_name` 只影响日志里的措辞。
  把 `param_name` 传进去是为了让报错信息说"crop_size must have one of ..."，
  而不是笼统地说 size。**读这段要看的是"谁被写进了 kwargs"，而不是参数怎么传。**
- **`image_mean` 从 list 转 tuple** 是为了可哈希：`ImageProcessingMixin.to_dict()`
  和后面的归一化都要用它，list 不能做 dict key，tuple 可以。
- **四个 key 无条件写回 `kwargs`**，哪怕值是 `None`。这样可以保证
  下游 `_validate_preprocess_kwargs` 的形参**一定**拿得到值 ——
  校验函数签名里的 `= None` 只是形式，真正的默认值在类属性上。

实测这四种写法各自落到哪个 size 字典：

```text
get_size_dict(224, {})                                  -> {'height': 224, 'width': 224}
get_size_dict(224, {'default_to_square': False})         -> {'shortest_edge': 224}
get_size_dict((336, 336), {})                            -> {'height': 336, 'width': 336}
get_size_dict((336, 336), {'height_width_order': False}) -> {'height': 336, 'width': 336}
get_size_dict({'shortest_edge': 224, 'longest_edge': 1333}, {}) -> 原样保留
get_size_dict(224, max_size=1333)                        -> ValueError:
     "Cannot specify both size as an int, with default_to_square=True and max_size"

# 合法的键集合（源文件里的 VALID_SIZE_DICT_KEYS，共 6 组）：
#   {height, width} / {shortest_edge} / {shortest_edge, longest_edge}
#   {longest_edge}  / {max_height, max_width} / {min_pixels, max_pixels}
```

---

## 三、★ 后端是"类级"的选择，不是"调用级"的选择

第二道门比契约更值得记：**一个处理器用哪个后端，在写类的时候就定死了** ——
`resize` / `normalize` / `pad` 这些方法**重写在哪一层，就用哪一层的实现**。

<!-- src: image_processing_utils.py -->
```python
    valid_kwargs = ImagesKwargs

    default_to_square = True
    rescale_factor = 1 / 255
    model_input_names = ["pixel_values"]

    def __init__(self, **kwargs: Unpack[ImagesKwargs]):
        super().__init__(**kwargs)
        # We don't call self._set_attributes in BaseImageProcessor for backward compatibility with remote code
        # We call it instead in the backend subclasses' __init__ methods.
```

**读法**：

- **`valid_kwargs` / `default_to_square` / `rescale_factor` / `model_input_names`
  这四个类属性，就是"基座给模型留下的默认值"**。任何一个图像处理器只要不改它们，
  行为就是"缩放 1/255、方图、输出叫 `pixel_values`"。
- **`__init__` 里那句注释是本课最值钱的一行**：`_set_attributes` **故意不在基座调用**，
  原因写明了是 **backward compatibility with remote code** ——
  社区仓库里有大量老的自定义处理器，它们的 `__init__` 不认 `valid_kwargs` 那一套，
  一旦基座抢先 `_set_attributes`，这些类就会炸。所以契约被拆成两半：
  基座只留空壳，**每个后端在自己的 `__init__` 里补上这一句**。
- 后果：**`BaseImageProcessor` 不能被当成可用的处理器实例化**。
  你继承它、但不继承任何后端、又不自己调 `_set_attributes`，
  那么 `self.size` 之类的属性根本不存在 —— 而 `preprocess` 里
  `getattr(self, kwarg_name, None)` 会安静地拿到 `None`，不报错。
  这是一个**静默失败**的设计，值得单独记住。

`_set_attributes` 就是"把类属性抄成实例属性"的那段：

<!-- src: image_processing_utils.py -->
```python
    def _set_attributes(self, **kwargs):
        """Resolve and set instance attributes from kwargs and class-level defaults for all valid kwargs."""
        attributes = {}
        for key in self.valid_kwargs.__annotations__:
            kwarg = kwargs.pop(key, None)
            if kwarg is not None:
                attributes[key] = kwarg
            else:
                attributes[key] = deepcopy(getattr(self, key, None))
        attributes = self._standardize_kwargs(**attributes)
        for key, value in attributes.items():
            setattr(self, key, value)

        self._valid_kwargs_names = list(self.valid_kwargs.__annotations__.keys())
```

**读法**：

- 循环的**遍历源是 `self.valid_kwargs.__annotations__`，不是 `kwargs`**。
  也就是说：**只有在这个 TypedDict 里声明过的键才会变成实例属性**，
  用户多传的键会被静默丢掉（`from_dict` 里有另一条兼容路径，见下）。
- `deepcopy(getattr(self, key, None))` 里的 `deepcopy` 不是多余的：
  `image_mean` 是 tuple、`size` 是 `SizeDict`，共享引用会让 `setattr` 写坏类属性。
- 最后一行存的 `_valid_kwargs_names` 正是第一节 `preprocess` 里
  `for kwarg_name in self._valid_kwargs_names` 用的那个列表 —— **两段代码是同一个契约的两半**。

---

## 四、`_fuse_mean_std_and_rescale_factor`：三个容易看漏的细节

torchvision 后端把 `rescale` 和 `normalize` **合成一步**。合成公式本身不难，
难的是它身上挂着的三个细节：

<!-- src: image_processing_backends.py -->
```python
    @lru_cache(maxsize=10)
    def _fuse_mean_std_and_rescale_factor(
        self,
        do_normalize: bool | None = None,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        do_rescale: bool | None = None,
        rescale_factor: float | None = None,
        device: Optional["torch.device"] = None,
    ) -> tuple:
        if do_rescale and do_normalize:
            # Fused rescale and normalize
            image_mean = torch.tensor(image_mean, device=device) * (1.0 / rescale_factor)
            image_std = torch.tensor(image_std, device=device) * (1.0 / rescale_factor)
            do_rescale = False
        return image_mean, image_std, do_rescale
```

**读法**：

- **细节一：`do_rescale` 被改成 `False` 返回出去。** 融合的数学是
  `(x*f - μ) / σ  ==  (x - μ/f) / (σ/f)`，所以把 `μ`、`σ` 各除以 `f`，
  再让 `normalize` 在**原图**上跑一次，就等价于"先缩放再归一化"。
  代价是**必须**告诉调用方"别再单独 rescale 了"，否则会缩放两次。
- **细节二：`image_mean` 被 `torch.tensor(...)` 变成了张量，而且 `.to(device)` 了。**
  这条路径允许在 GPU 上做归一化 —— 图像处理不一定在 CPU 上跑。
  返回值的类型因此**取决于分支**：融合时是 `Tensor`，不融合时还是原来的 `list`。
- **细节三：`@lru_cache(maxsize=10)`。** 装饰器把 `self` 当成第一个缓存键，
  所以缓存是**按实例**的，最多 10 组 (mean, std, factor, device) 组合。
  这在稳态推理里很划算（每次调用都省几毫秒的张量构造），
  但也意味着**改类属性不会让缓存失效** —— 运行时改 `image_std` 后行为可能不自洽。

实测：融合与分开做，结果在 `atol=1e-5` 下一致，但**最后一位有效数字不同** ——
融合路径多了一次 `float32` 的乘除，不是数学上完全相同的两条路径。

```text
先 rescale 再 normalize : [-1.179128646850586,  0.33910778164863586, -1.558687686920166]
rescale_and_normalize   : [-1.179128646850586,  0.33910769224166876, -1.558687686920166]
allclose(atol=1e-5)     = True        ← 等价，但不是逐位相同
dtype                   = torch.float32
```

---

## 五、★ 真实的张量形状演算：336×336 → 多少个视觉 token

这一节回答「L2 层」的核心问题：**视觉 token 怎么和文本 token 排在同一条序列上。**
先看后端的输出形状，再看它和文本 token 的拼装关系。

<!-- src: image_processing_backends.py -->
```python
    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        pad_size: SizeDict | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        """Preprocess using Torchvision backend (fast, GPU-accelerated)."""
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
```

**读法**：

- **这个签名就是"契约的落地清单"**：把第一节那四步做完之后，真正需要后端知道的东西
  全在这里 —— 尺寸、重采样、裁剪、缩放、归一化、padding、返回类型。
  **一眼就能看出哪些操作是"标配"**（`do_resize` / `do_center_crop` / `do_rescale` /
  `do_normalize` / `do_pad` 五个开关），哪些是"可选的编排参数"
  （`disable_grouping` 控制要不要按形状分组批量处理）。
- `disable_grouping` 是个**显存 / 正确性的取舍开关**：按形状分组可以把同尺寸的图
  叠成一个 batch 一次算完（快），但分组本身要拷贝数据；不分组就逐张算。
- **注意 `size` 和 `resample` 在这个后端里是"被忽略/被转译"的**：
  `Glm5NextImageProcessor` 重写了 `resize`，用自己的 `smart_resize` 覆盖了 `size` 的含义
  （`size = {"longest_edge": 1}` 那句 TODO 就是这个意思）。
  **基座给的是"标准插槽"，模型可以换掉插槽里的东西** —— 这正是分层的价值。

实测（`Glm5NextImageProcessor`，默认 `patch_size=14`、`merge_size=2`、`temporal_patch_size=2`）：

```text
输入 (H, W, C)      = (336, 336, 3)
pixel_values.shape  = (576, 1176)      dtype=torch.float32
image_grid_thw      = [[1, 24, 24]]    (t, h, w)

patch 数 = grid_h × grid_w = 24 × 24 = 576            ← pixel_values 的第 0 维
patch 展平维 = C × temporal × patch × patch = 3×2×14×14 = 1176   ← 第 1 维
视觉 token 数 = t×h×w / merge_size²  = 1×24×24/4 = 144

再试一张 700×1000 的非方图：
输入 (H, W, C)      = (700, 1000, 3)
pixel_values.shape  = (3600, 1176)
image_grid_thw      = [[1, 50, 72]]   → t×h×w/merge² = 900 个视觉 token

（`temporal_patch_size=2` 对单张图意味着：同一帧被复制成 2 帧再展平，
  所以第 1 维里那个因子 2 来自"时间轴补帧"。视频才真正用得上它。）
```

**文本 token 与视觉 token 怎么对齐**（`processing_glm5_next.py`，本课不覆盖该文件）：

```text
文本侧：input_ids / attention_mask  形状 (batch, seq)          每个位置 1 个 id
视觉侧：pixel_values                形状 (patch 总数, 1176)    不是序列，是"备料"
        经 vision tower + merge 2×2 后，每张 336×336 图产出 144 个向量
拼装  ：processor 把 prompt 里的 <|image|> 占位符展开成 144 个 image_token_id
        num_image_tokens = image_grid_thw.prod() // merge_length
        → (batch, seq) 里这 144 个位置，在 forward 里被 inputs_embeds 的视觉向量替换

一句话：**序列长度是文本说了算，视觉只负责"填满被占位符预定的那 144 个位置"。**
```

**回顾 L0-01**：那次 `forward` 里 `input_ids` 与 `inputs_embeds` 是**二选一**的，
多模态走的是 `inputs_embeds`。**这一课产出的 `pixel_values` 就是那两个分支合流之前，
视觉侧交出来的原始物料。**

---

## 六、`from_pretrained` 怎么把磁盘上的 JSON 变成一个处理器

读完 `preprocess` 容易忘掉：这个对象本身是从**一个 JSON 文件**来的。

<!-- src: image_processing_base.py -->
```python
        kwargs["cache_dir"] = cache_dir
        kwargs["force_download"] = force_download
        kwargs["local_files_only"] = local_files_only
        # Resolve the revision once, so that all the files of this load come from the same repository state.
        kwargs["revision"] = resolve_revision(
            pretrained_model_name_or_path,
            revision,
            token=token,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )

        if token is not None:
            kwargs["token"] = token

        image_processor_dict, kwargs = cls.get_image_processor_dict(pretrained_model_name_or_path, **kwargs)

        return cls.from_dict(image_processor_dict, **kwargs)
```

**读法**：

- **`resolve_revision` 只调一次，然后把结果塞回 `kwargs`。** 注释写得很直白：
  要保证这次加载涉及的**所有文件来自同一个仓库快照**。如果让
  `get_image_processor_dict` 内部各自解析一次，两个文件可能来自不同 commit ——
  这是分布式仓库上一个真实存在的一致性坑。
- **`kwargs` 是"进得去出得来"的**：`get_image_processor_dict` 返回的是
  `(字典, 剩下的 kwargs)`，`**kwargs` 里被消费掉的键（`cache_dir` 等）会被 pop 掉，
  剩下的继续传给 `from_dict`。所以"用户显式传的覆盖值"能一路活到实例化那一刻。
- `from_pretrained` 到这里只有 15 行真代码，**它把"找文件"和"造对象"彻底分开了**：
  `get_image_processor_dict` 只管找和读，`from_dict` 只管造。

`from_dict` 与 `to_dict` 是这条路的另一端：

<!-- src: image_processing_base.py -->
```python
        image_processor_dict = image_processor_dict.copy()
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        image_processor_dict.update({k: v for k, v in kwargs.items() if k in cls.valid_kwargs.__annotations__})
        image_processor = cls(**image_processor_dict)

        # Apply extra kwargs to instance (BC for remote code, e.g. phi4_multimodal)
        extra_keys = []
        for key in reversed(list(kwargs.keys())):
            if hasattr(image_processor, key) and key not in cls.valid_kwargs.__annotations__:
                setattr(image_processor, key, kwargs.pop(key, None))
                extra_keys.append(key)
```

**读法**：

- **`.copy()` 是防御性的**：不复制的话，`update` 会改到调用方（可能来自缓存的）字典。
- **两段式赋值**：先按 `valid_kwargs` 声明正式更新一次，再把**声明之外**、
  但实例上**已经存在**的键当作兼容补丁写回去（注释点名了 `phi4_multimodal`）。
  这段就是第三节说的"多传的键被静默丢掉"的**补偿路径** ——
  只对老代码生效，新代码会被 `validate_typed_dict` 拦下。
- `reversed(list(kwargs.keys()))` 的顺序是为了让 `extra_keys` 的日志顺序和用户传参顺序一致。

再看它怎么被存回去 —— 落盘的是**一个 JSON**，不是 pickle：

<!-- src: image_processing_base.py -->
```python
        output = copy.deepcopy(self.__dict__)
        output["image_processor_type"] = self.__class__.__name__

        return output
```

**读法**：

- `self.__dict__` 是**实例属性全量**，所以"能被存下来"取决于"有没有被 `setattr`"——
  又绕回 `_set_attributes`。**类属性本身不会进 JSON**，
  这正是第四节 `to_dict` 要把 `SizeDict` 转成普通 `dict`、并保留显式 `None` 的原因。
- `image_processor_type` 存的是**类名**，`AutoImageProcessor` 靠它反查类。
  **注意 `PilBackend.to_dict` 把它改写了**（本课不单独引用，只描述）：
  它在 `super().to_dict()` 之后判断 `image_processor_type.endswith("Pil")`，
  成立就把**结尾三个字符**砍掉（`[:-3]`，不是 `replace`）。
  于是 `Glm5NextImageProcessorPil` 存成 `Glm5NextImageProcessor` ——
  **存盘的名字是"模型处理器的名字"，不是"后端变体的名字"**。
  下次加载时走的是"模型名 → 后端映射表"，由 `AutoImageProcessor` 再决定用哪个后端。
  **存盘时不固化后端是一个刻意的选择**：否则在没装 torchvision 的机器上，
  这份 config 会直接加载失败。这条线索在第九节闭环。

---

## 七、`process_image`：两个后端唯一的"入口差异"

后端之间最大的分歧不在 resize，而在**第一步把输入变成什么**。
先看 torchvision 后端：

<!-- src: image_processing_backends.py -->
```python
@requires(backends=("torch", "torchvision"))
class TorchvisionBackend(BaseImageProcessor):
    """Torchvision backend for GPU-accelerated batched image processing."""

    def __init__(self, **kwargs: Unpack[ImagesKwargs]):
        super().__init__(**kwargs)
        self._set_attributes(**kwargs)
```

**读法**：

- **`@requires(backends=("torch", "torchvision"))` 是"类的可导入性"开关**：
  装饰器给类挂一个 `__backends` 元组，让惰性导入层知道"这个类需要什么才能用"。
  在没装 torchvision 的环境里，这个类会变成 `DummyObject` 子类，
  于是 `AutoImageProcessor` 的兜底逻辑会找得到"它不可用"这件事（见第九节）。
- **`self._set_attributes(**kwargs)` 就在这里**（基座故意不调，见第三节）。
  两个后端的 `__init__` 都是这两行 —— 这也是"后端可换、契约不变"的最小证据。

然后看它怎么把 PIL / numpy / tensor 三种输入统一成 tensor：

<!-- src: image_processing_backends.py -->
```python
        if image_type == ImageType.PIL:
            image = tvF.pil_to_tensor(image)
        elif image_type == ImageType.NUMPY:
            image = torch.from_numpy(image).contiguous()

        if image.ndim == 2:
            image = image.unsqueeze(0)

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)

        if input_data_format == ChannelDimension.LAST:
            image = image.permute(2, 0, 1).contiguous()

        if device is not None:
            image = image.to(device)

        return image
```

**读法**：

- **`.contiguous()` 出现两次，各有原因**：`torch.from_numpy` 与 numpy 共享内存，
  后续 `permute` 会产生非连续视图；不 `.contiguous()` 的话，某些 kernel 会隐式拷贝甚至报错。
  第二次（`permute` 之后）才是关键的那次。
- **`ndim == 2` 是灰度图**：`(H, W)` → `(1, H, W)`。**通道维必须存在**，
  否则后面 `permute(2,0,1)` 会直接越界。
- **`infer_channel_dimension_format` 只在用户没指定时推断**：HWC 还是 CHW 是有歧义的
  （3×3 的图两面都说得通），所以这里把"显式指定"优先于"推断"。
- **`device` 是 torchvision 后端独有的参数**（PIL 后端签名里根本没有它）——
  这就是"换后端会换掉哪些能力"的具体例子。

对照 PIL 后端：同样的入口，目标是 `np.ndarray`：

<!-- src: image_processing_backends.py -->
```python
        if image_type == ImageType.PIL:
            image = np.array(image)
            # Set LAST only for multi-channel PIL images (H, W, C); for grayscale (H, W), leave as is to avoid shape errors after expand_dims.
            if image.ndim >= 3:
                input_data_format = ChannelDimension.LAST if input_data_format is None else input_data_format
        elif image_type == ImageType.TORCH:
            image = image.numpy()

        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)
```

**读法**：

- **注释是这里唯一的信息源**：灰度 PIL 图 `np.array` 出来是 `(H, W)`，
  这时**不能**把 `input_data_format` 设成 `LAST` —— 因为下一步 `expand_dims(axis=0)`
  会把它变成 `(1, H, W)`，也就是 **FIRST**。先设 LAST 再 expand，通道维就跑到了最后，
  后面按 FIRST 解释会得到完全错误的形状。**这是两个后端唯一一处"逻辑不同而非调用不同"的地方。**
- 注意 torchvision 分支写成 `image.ndim == 2` 就 `unsqueeze(0)`，
  而 PIL 分支还要额外把 `input_data_format` 保护起来 —— **同一个语义，
  两个后端的实现代价不一样**。这就是"多后端"真正的维护成本。

---

## 八、两个后端的 `_preprocess`：批处理 vs 逐张

契约相同，实现风格完全不同。PIL 后端是最朴素的写法，正好当对照：

<!-- src: image_processing_backends.py -->
```python
        processed_images = []
        for image in images:
            if do_resize:
                image = self.resize(image=image, size=size, resample=resample)
            if do_center_crop:
                image = self.center_crop(image, crop_size)
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)
            processed_images.append(image)

        if do_pad:
            processed_images = self.pad(processed_images, pad_size=pad_size)

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)
```

**读法**：

- **顺序就是答案**：`resize → center_crop → rescale → normalize → pad`。
  这个顺序不能随便换：crop 要在 resize 之后（否则裁的位置不对），
  rescale/normalize 要在几何变换之后（否则插值会在归一化后的数值上做，结果不同），
  pad 放最后（padding 值 0 是"归一化后的 0"，和黑边含义一致）。
- **`if do_*` 四个开关是串联的**，每个默认值都来自类属性。
  想把一个处理器改成"只缩放不归一化"，只需要在 `from_pretrained` 时传
  `do_normalize=False`，不用改任何代码。
- **每个分支都是调用 `self.resize` / `self.normalize`**，不是调 numpy 函数 ——
  所以**子类只要重写这些方法就能换掉单步实现**（`Glm5NextImageProcessor.resize` 就是这么做的）。
- `BatchFeature(data={"pixel_values": ...}, tensor_type=return_tensors)`
  是这条流水线唯一的出口形状：**一个 key 叫 `pixel_values` 的字典**。

PIL 后端的这些方法自己不做运算，而是转调 numpy 实现：

<!-- src: image_processing_backends.py -->
```python
        """Rescale an image by a scale factor using NumPy."""
        return np_rescale(
            image,
            scale=scale,
            data_format=ChannelDimension.FIRST,
            input_data_format=ChannelDimension.FIRST,
        )
```

**读法**：

- **注意 `np_rescale` 这个名字**：它来自 `from .image_transforms import rescale as np_rescale`——
  模块里的函数**原本就叫 `rescale`**，改名只是为了和 `self.rescale` 区分。
  **所以"PIL 后端"其实是"numpy 实现在 CPU 上的门面"**，这也解释了为什么没有第三个
  `NumpyBackend`：numpy 根本不是"可选后端"，它是 PIL 后端的**底座**。
- `data_format` 和 `input_data_format` 都被钉死成 `FIRST`：
  走到这一步的图已经被 `process_image` 转成通道在前了，
  显式写明可以避免 numpy 实现再去猜一次。

---

## 九、★ 「默认后端的选择顺序」——验收点

这一节是 README 验收点「能说出默认后端的选择顺序」的正文答案。
决策逻辑不在本课覆盖的三个文件里，而在 `models/auto/image_processing_auto.py`。
**下面两段是"不计入覆盖率"的旁证引用**（该文件不属于本课覆盖域），用 `text` 块给出关键行：

```text
# models/auto/image_processing_auto.py  L333-357（旁证，不计入覆盖率）
def _resolve_backend(backend, use_fast, base_class_name):
    if use_fast is not None:                        # ① 旧参数（已弃用）
        if backend is None:
            backend = "torchvision" if use_fast else "pil"
    if backend is None:                             # ② 没显式给后端
        if base_class_name in DEFAULT_TO_PIL_BACKEND_IMAGE_PROCESSORS:
            return "pil"                            #    ②a Lanczos 系处理器强制 PIL
        return "torchvision" if is_torchvision_available() else "pil"   # ②b
    return backend                                  # ③ 显式 backend= 原样返回
```

```text
# 实测（probe_l204.py，torchvision 0.29.0）：
DEFAULT_TO_PIL_BACKEND_IMAGE_PROCESSORS = []        # torchvision >= 0.27 → 空列表
_resolve_backend(None,  None,  'Glm5NextImageProcessor') = 'torchvision'
_resolve_backend('pil', None,  'Glm5NextImageProcessor') = 'pil'
_resolve_backend('torchvision', None, 'Glm5NextImageProcessor') = 'torchvision'
_resolve_backend(None,  True,  'Glm5NextImageProcessor') = 'torchvision'   # use_fast=True
_resolve_backend(None,  False, 'Glm5NextImageProcessor') = 'pil'           # use_fast=False
```

**读法 —— 默认后端的选择顺序（四步，从高到低）**：

1. **显式 `backend="torchvision"` / `backend="pil"`** —— 最高优先级，原样返回（③）。
2. **旧参数 `use_fast`**（已弃用，会打 warning）—— `True → torchvision`、`False → pil`，
   且**只在 `backend` 为 `None` 时生效**（①）。
3. **`DEFAULT_TO_PIL_BACKEND_IMAGE_PROCESSORS` 名单** —— 名单里的处理器**强制 PIL**（②a）。
   名单内容取决于 torchvision 版本：`torchvision >= 0.27` 时是**空列表**
   （Lanczos 原生支持了），否则是 `Chameleon / Flava / Idefics3 / SmolVLM` 四个
   Lanczos 系处理器。**同一个 `backend=None`，在不同环境里会得到不同后端 —— 这就是"默认"的全部含义。**
4. **兜底：`torchvision` 可用就用 `torchvision`，否则 `pil`**（②b）。

选完之后还有一层**兜底加载**（`_load_class_with_fallback`）：
先试目标后端；如果那个类在当前环境里是 `DummyObject`（依赖缺失），
就按映射表里的顺序**换一个后端再试**，并打一句 warning。
所以"选中的后端"和"最终拿到的类"在依赖缺失时**可能不是同一个**。

GLM-5 自己的映射表长这样（旁证，不计入覆盖率）：

```text
# models/auto/auto_mappings.py  L1268
("glm5_next", {"pil": "Glm5NextImageProcessorPil", "torchvision": "Glm5NextImageProcessor"})
```

**这就是为什么同一份 `preprocessor_config.json` 能被两个后端读** ——
存盘的 `image_processor_type` 是去掉 `Pil` 后缀的基名，
后端由**当前环境**在加载时决定，而不是由磁盘上的文件决定。

---

## 十、把一条流水线连起来

```text
  一张 PIL.Image / np.ndarray / torch.Tensor     (H, W, 3)
        │  BaseImageProcessor.__call__
        │  → preprocess()   ← ★ 契约在这里，四步编排
        │      validate_typed_dict(valid_kwargs)      ① 类型契约
        │      setdefault(getattr(self, kwarg_name))  ② 类属性补默认值
        │      _standardize_kwargs()                  ③ size/crop_size/pad_size → SizeDict
        │      _validate_preprocess_kwargs()          ④ 组合合法性（do_resize 必须有 size …）
        ▼
   _preprocess_image_like_inputs()
        │  _prepare_images_structure → fetch_images（URL 先下载）
        │  _prepare_image_like_inputs → 逐张 process_image()   ← ★ 后端差异入口
        │      torchvision: pil_to_tensor / from_numpy + permute(2,0,1) + .to(device)
        │      pil        : np.array   / .numpy()     + transpose(2,0,1)
        ▼
   _preprocess()                                  ← ★ 后端差异主体
        │  torchvision: group_images_by_shape → 批量 resize → 批量 crop
        │               → rescale_and_normalize（μ/σ 先除以 f，融合成一步）
        │  pil        : for image in images: resize → crop → rescale → normalize
        ▼
   BatchFeature({"pixel_values": [...]}, tensor_type=...)   ← 两个后端共同的出口
        │  Glm5NextImageProcessor 在这一步之前插了自己的 patchify()
        ▼
   pixel_values (576, 1176)  +  image_grid_thw (1, 24, 24)
        │  processor 把 <|image|> 展开成 144 个 image_token_id
        ▼
   input_ids (batch, seq) 里的 144 个位置 ←→ 与文本 token 同一条序列
```

---

## 十一、与后面课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| `resize` / `center_crop` / `normalize` 的 numpy 实现细节 | L2-05（`image_transforms.py`） |
| `SizeDict` / `ImageInput` / `infer_channel_dimension_format` 的定义 | L2-05（`image_utils.py`） |
| `Glm5NextImageProcessor.patchify` 的 8 维 reshape 全过程 | L2-06 |
| processor 怎么把 tokenizer 和 image processor 拼起来 | L2-06（`processing_utils.py`） |
| 336×336 → 27 个 token 的那套算法（与 144 的差异从哪来） | L2-07（视觉塔） |
| 视觉向量是在哪一步替换掉占位符的 | L2-06 / L2-07 |
| `inputs_embeds` 与 `input_ids` 二选一的入口 | L0-01（已学） |
| `AutoImageProcessor` 的完整分派逻辑 | L1-05（`auto` 系列） |

**一句话总结**：

> **契约在基座（四步编排 + 类属性当默认值），实现交后端（torchvision / pil 二选一）；
> `size` 先归一再校验，`rescale` 与 `normalize` 在 torchvision 上会被融合成一步；
> 真正决定"多少个视觉 token"的不是后端，而是模型自己的 `patch_size / merge_size`。**
