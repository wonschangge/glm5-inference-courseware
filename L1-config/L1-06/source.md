<!-- glm5-coverage
models/auto/modeling_auto.py
models/auto/tokenization_auto.py
models/auto/image_processing_auto.py
models/auto/video_processing_auto.py
models/auto/feature_extraction_auto.py
models/auto/processing_auto.py
-->

# L1-06 · 其余四张自动注册表 — 源文件

**这一课只回答一个问题：屏幕上的一个字符串，是怎么变成一个真正被 import 的类的？**

视角：L1-05 拆开了 `AutoConfig` 与 `AutoModel` 的分派链（`configuration_auto.py` 的 718 条底表、
`auto_factory.py` 的 `_LazyAutoMapping`、`auto_mappings.py` 的生成表）。本课接着看 `models/auto/`
**剩下**的 6 个文件：它们把同一套「表 + 惰性映射 + `register`」的写法，复制到了分词器、图像处理器、
视频处理器、特征提取器、处理器，以及 `modeling_auto.py` 里 48 个 `AutoModel*` 任务类上。

| 文件 | 行数 | 引用块 |
|---|---|---|
| `models/auto/modeling_auto.py` | 2689 | 4 |
| `models/auto/tokenization_auto.py` | 1039 | 2 |
| `models/auto/image_processing_auto.py` | 780 | 2 |
| `models/auto/video_processing_auto.py` | 434 | 2 |
| `models/auto/feature_extraction_auto.py` | 393 | 2 |
| `models/auto/processing_auto.py` | 383 | 2 |

> 行数按作业书口径（`len(text.split("\n"))`，含文件末尾换行）；`wc -l` 各少 1。
> 六个文件合计 5718 行。

**实测数字**（`_data/recon/probe_l106.py`，transformers 5.18.0.dev0，本机 CPU，torchvision 可用）：

```text
底表（import transformers.models.auto.auto_mappings 之后立刻读，before）
    CONFIG_MAPPING_NAMES             718 条
    IMAGE_PROCESSOR_MAPPING_NAMES    120 条
    VIDEO_PROCESSOR_MAPPING_NAMES     31 条
    FEATURE_EXTRACTOR_MAPPING_NAMES   29 条
    PROCESSOR_MAPPING_NAMES          147 条
    TOKENIZER_MAPPING_NAMES          —— 不在 auto_mappings.py 里，自成一家

补丁是打在**同一个 dict 对象**上的（import 对应模块后，读同一个对象，after）
    IMAGE      120 -> 211   (+91)
    VIDEO       31 ->  45   (+14)
    FEATURE     29 ->  69   (+40)
    PROCESSOR  147 -> 180   (+33)
    TOKENIZER          309 条，其中 87 条的值就是兜底的 "TokenizersBackend"
    MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS   46 个 model_type

modeling_auto.py
    MODEL_MAPPING_NAMES                          556 条
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES            178 条
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES    88 条
    _LazyAutoMapping 实例                          54 个
    导出的 Auto* 类                                49 个（48 个 AutoModel* + AutoBackbone）

惰性 import 的账单
    import transformers 之后                    243 个 transformers.* 模块
    读 MODEL_MAPPING_NAMES["glm5_next"]          新增 0 个模块（值还是字符串）
    AutoModel._model_mapping[Glm5NextConfig]     新增 3 个模块，12.1 ms
    再取 Glm5NextTextConfig                      新增 0 个模块，0.23 ms（同一个包已缓存）
    一路走到底                                   423 个模块

GLM-5 的落点
    Glm5NextConfig in MODEL_MAPPING                        True  -> Glm5NextModel
    Glm5NextConfig in MODEL_FOR_CAUSAL_LM_MAPPING          False
    Glm5NextConfig in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING True  -> Glm5NextForConditionalGeneration
    PROCESSOR_MAPPING[Glm5NextConfig]                      Glm5NextProcessor
    Glm5NextProcessor.get_attributes()                     ['image_processor', 'tokenizer', 'video_processor']
```

> 上面这段是**实测汇总**（探针输出），用 `text` 块标出 —— 它不是源文件的逐字引用，
> 因此不参与保真校验。探针 `_data/recon/probe_l106.py` 是工装脚本，**不计入覆盖率**
> （覆盖域只统计 `src/transformers/**/*.py`）。下面每一个 `python` 块都是逐字引用。

---

## 一、六个文件的对称骨架：一个类 = 一行

先看 `modeling_auto.py` 的尾部。49 个 `Auto*` 类里，绝大多数都长成同一个形状：
**一个空壳类，唯一的内容是一行 `_model_mapping = 某张表`**，然后被 `auto_class_update` 补上文档。

<!-- src: models/auto/modeling_auto.py -->
```python
class AutoModel(_BaseAutoModelClass):
    _model_mapping = MODEL_MAPPING


AutoModel = auto_class_update(AutoModel)


class AutoModelForPreTraining(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_PRETRAINING_MAPPING


AutoModelForPreTraining = auto_class_update(AutoModelForPreTraining, head_doc="pretraining")


class AutoModelForCausalLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_CAUSAL_LM_MAPPING
```

**读法**：
- `_BaseAutoModelClass` 里没有任何模型知识 —— 它只认 `cls._model_mapping` 这一个约定。
  所以「新增一种任务」在这套设计里等于**新增一张表 + 新增一行类体**，不碰任何逻辑。
- `auto_class_update(...)` 负责把 `head_doc` 之类的文档字符串贴回去。注释里那句
  `# override to give better return typehint` 值得注意：**这个类存在的意义之一只是类型提示**。
- 这就是为什么 49 个类的文件能有 2689 行却几乎没有分支 —— 2688 行里绝大多数是表，不是代码。

再看 `AutoModelForImageTextToText`，它是 GLM-5 实际会命中的那一个（下面第三节给出实测）。
它的 `from_pretrained` 被重写，只为了把返回类型收窄：

<!-- src: models/auto/modeling_auto.py -->
```python
class AutoModelForImageTextToText(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING

    # override to give better return typehint
    @classmethod
    def from_pretrained(
        cls: type["AutoModelForImageTextToText"],
        pretrained_model_name_or_path: str | os.PathLike[str],
        *model_args,
        **kwargs,
    ) -> "_BaseModelWithGenerate":
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


AutoModelForImageTextToText = auto_class_update(AutoModelForImageTextToText, head_doc="image-text-to-text modeling")
```

**读法**：
- 重写后的方法体只有一行 `return super().from_pretrained(...)`。**这不是逻辑改动，是类型改动** ——
  `_BaseModelWithGenerate` 在 `TYPE_CHECKING` 里被定义成 `PreTrainedModel + GenerationMixin`，
  于是编辑器知道它 `.generate()` 可用，运行时则完全没有这层包装。
- 回顾 L0-01：那一课里的 `Glm5NextForConditionalGeneration.generate()` 能成立，
  前提就是这里返回的是 `_BaseModelWithGenerate` 语义下的对象。

---

## 二、★ 底表是生成的，补丁是手写的 —— 而且补丁是「原地」打上去的

这是本课第一个真正的洞察：**六张表不是六份手写清单，而是「机器生成的底表 + 人手写的补丁表」，
并且补丁不是拷贝一份新表，而是在同一个 dict 对象上 `update`。**
所以「这张表有多少条」取决于**哪些模块已经被 import 过**。

先看补丁的写法（`feature_extraction_auto.py`，`_auto_mappings` 底表 29 条，补丁 40 条）：

<!-- src: models/auto/feature_extraction_auto.py -->
```python
FEATURE_EXTRACTOR_MAPPING_NAMES.update(MISSING_FEATURE_EXTRACTOR_MAPPING_NAMES)
FEATURE_EXTRACTOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, FEATURE_EXTRACTOR_MAPPING_NAMES)


def feature_extractor_class_from_name(class_name: str):
    for module_name, extractors in FEATURE_EXTRACTOR_MAPPING_NAMES.items():
```

**读法**：
- `FEATURE_EXTRACTOR_MAPPING_NAMES` 是从 `auto_mappings.py` **import 进来的同一个对象**，
  这一行 `update` 直接改写了它。实测：import 该模块前 29 条，import 后 69 条，`+40` 条全部来自
  这个文件里的 `MISSING_FEATURE_EXTRACTOR_MAPPING_NAMES`。
- 为什么需要补丁表？因为底表是由 `utils/check_auto.py` 根据每个 config 的 `cls.model_type`
  **自动推导**的（`auto_mappings.py` 头部就写着 "automatically generated ... Do NOT edit"）。
  推导不出来的是「一个 config 对应多个处理器」「类名不符合推导规则」这些例外，只能手写。
- `feature_extractor_class_from_name` 的第一行就是**遍历这张已经被打上补丁的表**去反查类 ——
  它不关心条目是生成的还是手写的，只看最终那张表。

视频处理器把这套写法用得更狠：**整张表会被环境判一次死刑**。

<!-- src: models/auto/video_processing_auto.py -->
```python
    VIDEO_PROCESSOR_MAPPING_NAMES.update(MISSING_VIDEO_PROCESSOR_MAPPING_NAMES)

for model_type, video_processors in VIDEO_PROCESSOR_MAPPING_NAMES.items():
    fast_video_processor_class = video_processors

    # If the torchvision is not available, we set it to None
    if not is_torchvision_available():
        fast_video_processor_class = None

    VIDEO_PROCESSOR_MAPPING_NAMES[model_type] = fast_video_processor_class

VIDEO_PROCESSOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, VIDEO_PROCESSOR_MAPPING_NAMES)
```

**读法**：
- 注意这个 `for` 循环没有 `break`、没有条件分支里的 `continue` —— 它把**每一条**都重新赋值一遍，
  不可用时统统写成 `None`。也就是说「torchvision 在不在」这件事不是运行时判断，
  而是**被烤进了表的数据里**。
- 后果：`VIDEO_PROCESSOR_MAPPING[config]` 在无 torchvision 的环境里会拿到 `None`，
  `from_pretrained` 于是走不到类，只能抛 `requires torchvision` 的错误（见本文件 L404~L413）。
- 本机 torchvision 可用，实测 45 条里 `None` 的条数是 **0**；把 `is_torchvision_available()` 换成
  `lambda: False` 再 `reload` 该模块，**45 条全部**变 `None`，`VIDEO_PROCESSOR_MAPPING.get(Glm5NextConfig)`
  随之返回 `None`（同一探针第 ⑨ 步的实测）。同一次 `reload` 之后，
  `auto_mappings.VIDEO_PROCESSOR_MAPPING_NAMES` 也是 **45/45 个 `None`** —— 这就是「同一个对象」的直接证据。
- 对比第二节第一段：`feature_extraction_auto.py` 的 `update` 是**加键**，
  这里的循环是**改值**。同样是「表」，一处是索引，一处是策略。

---

## 三、★ 惰性映射：读表零 import，取格子才 import 那一个包

`modeling_auto.py` 里 54 个 `_LazyAutoMapping` 全都是同一行写法：

<!-- src: models/auto/modeling_auto.py -->
```python
MODEL_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_MAPPING_NAMES)
MODEL_FOR_PRETRAINING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_PRETRAINING_MAPPING_NAMES)
MODEL_FOR_CAUSAL_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)
```

**读法**：
- 左边的 `CONFIG_MAPPING_NAMES` 是 `str -> str`，右边的 `MODEL_MAPPING_NAMES` 也是 `str -> str`。
  **两张表里没有任何一个类对象。** `_LazyAutoMapping` 只是把两张表按 `model_type` 对齐，
  真正的 `importlib.import_module("transformers.models.glm5_next")` 发生在 `__getitem__` 里（L1-05 已拆过）。
- 省下了什么？实测：`import transformers` 之后是 243 个 `transformers.*` 模块；
  读 `MODEL_MAPPING_NAMES["glm5_next"]` 新增 **0** 个模块（拿到的还是字符串 `'Glm5NextModel'`）；
  直到 `AutoModel._model_mapping[Glm5NextConfig]` 才新增 **3** 个模块、耗时 12.1 ms。
- 而第二次访问同一个包是免费的：`_LazyAutoMapping` 自己缓存了已 import 的模块，
  所以紧接着取 `Glm5NextTextConfig` 只新增 **0** 个模块、0.23 ms ——
  因为 `glm5_next_text` 经 `model_type_to_module_name()` 归一化后指向同一个包（L1-05 讲过的归一化）。
- 代价是**不能用 `dict` 的方式遍历**：`keys()` 给的是字符串，而 `values()` 会把整张表的类一次性 import
  （L1-05 的探针第 ⑥ 步在 `CONFIG_MAPPING` 上实测过这个反面：718 个 config 被一起拉了进来）。

GLM-5 具体落在哪张表？表项本身就是答案：

<!-- src: models/auto/modeling_auto.py -->
```python
        ("git", "GitForCausalLM"),
        ("glm46v", "Glm46VForConditionalGeneration"),
        ("glm4v", "Glm4vForConditionalGeneration"),
        ("glm4v_moe", "Glm4vMoeForConditionalGeneration"),
        ("glm5_next", "Glm5NextForConditionalGeneration"),
        ("glm_ocr", "GlmOcrForConditionalGeneration"),
        ("glmga", "Glm46VForConditionalGeneration"),
```

**读法**：
- 这一段的表是 `MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES`（88 条）。GLM-5 就在第 1127 行，
  值是 `Glm5NextForConditionalGeneration`。
- 实测三连：`Glm5NextConfig in MODEL_MAPPING` → `True`（→ `Glm5NextModel`）；
  `in MODEL_FOR_CAUSAL_LM_MAPPING` → `False`；`in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING` → `True`。
- **「config 如何决定形状」在这里落地**：同一个 `Glm5NextConfig`，进 `AutoModel` 拿到裸主干，
  进 `AutoModelForImageTextToText` 拿到带 `lm_head` 的生成模型；而它不是 `AutoModelForCausalLM`，
  所以用 `AutoModelForCausalLM.from_pretrained("zai-org/GLM-5")` 是**选错门**。这不是猜的，是查表查出来的。
- 上下文：`("glmga", "Glm46VForConditionalGeneration")` 这类「一个类服务多个 model_type」的行也在同一张表里，
  说明表的值**允许重复** —— 表描述的是「哪个 config 走哪个类」，不是「哪个类属于哪个模型」。

---

## 四、★ 命名冲突：注册表 > Hub 配置 > 内置表 > 去掉 Fast 再试

分词器的 `tokenizer_class_from_name()` 是全仓库里**优先级层次最多**的一个反查函数：

<!-- src: models/auto/tokenization_auto.py -->
```python
def tokenizer_class_from_name(class_name: str) -> type[Any] | None:
    # Bloom tokenizer classes were removed but should map to the fast backend for BC
    if class_name in {"BloomTokenizer", "BloomTokenizerFast"}:
        return TokenizersBackend

    if class_name in REGISTERED_FAST_ALIASES:
        return REGISTERED_FAST_ALIASES[class_name]

    if class_name in REGISTERED_TOKENIZER_CLASSES:
        return REGISTERED_TOKENIZER_CLASSES[class_name]

    if class_name == "TokenizersBackend":
        return TokenizersBackend
```

**读法**：
- 顺序就是答案：① 历史别名（`BloomTokenizer` → `TokenizersBackend`）；② `REGISTERED_FAST_ALIASES`；
  ③ `REGISTERED_TOKENIZER_CLASSES`；④ `TokenizersBackend` 这个「万能兜底」；⑤ 最后才是遍历
  `TOKENIZER_MAPPING_NAMES` 这张 309 条的大表。
- **注册表排在表前面**，这是本课要记住的裁决顺序：用户用 `AutoTokenizer.register()` 塞进来的类，
  优先于库自己维护的表。实测初始 `REGISTERED_TOKENIZER_CLASSES` 规模为 **0**，它是纯运行时状态。
- 这个函数末尾还有一手：找不到 `XxxFast` 就 `class_name[:-4]` 再递归一次（v5 里 fast/slow 合并后的兼容层）。

但优先级不只是「注册表 > 表」，还有一处**反转**：有些 Hub 配置里写的分词器类名，官方认定是错的。

<!-- src: models/auto/tokenization_auto.py -->
```python
for model_type in MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS:
    if model_type not in TOKENIZER_MAPPING_NAMES:
        TOKENIZER_MAPPING_NAMES[model_type] = "TokenizersBackend" if is_tokenizers_available() else None

TOKENIZER_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, TOKENIZER_MAPPING_NAMES)

CONFIG_TO_TYPE = {v: k for k, v in CONFIG_MAPPING_NAMES.items()}
```

**读法**：
- `MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS` 有 **46** 个 `model_type`（实测）。对这个名单里的模型，
  `from_pretrained` 会**假装 Hub 上的 `tokenizer_class` 不存在**，改用注册表里的类（见本文件 L854~L869）。
- 那 `CONFIG_TO_TYPE` 做什么？它是 `CONFIG_MAPPING_NAMES` 的**反向表**（类名 → `model_type`），
  让「已经拿到 config 类」的调用方也能回到字符串世界里查表。
- 一句话：**「谁权威」不是固定的**。默认相信 Hub 的 `tokenizer_config.json`，
  但对这 46 个例外，内置表说了算。

---

## 五、image：表的「值」不是字符串，而是 backend 字典

`IMAGE_PROCESSOR_MAPPING_NAMES` 的值长这样（实测）：
`{"pil": "Glm5NextImageProcessorPil", "torchvision": "Glm5NextImageProcessor"}`。
于是取类之前必须先回答「用哪个 backend」：

<!-- src: models/auto/image_processing_auto.py -->
```python
def _resolve_backend(backend: str | None, use_fast: bool | None, base_class_name: str | None) -> str:
    """Resolve raw backend inputs to a concrete backend name ('torchvision' or 'pil').

    Handles, in order:
    - Deprecated ``use_fast`` flag: warns and converts to an explicit backend string when no
      explicit backend is given.
    - Explicit backend string: returned as-is.
    - None resolution: forces 'pil' for processors in DEFAULT_TO_PIL_BACKEND_IMAGE_PROCESSORS
      (Lanczos interpolation, unsupported by torchvision < 0.27); otherwise picks 'torchvision'
      when available, falling back to 'pil'.
    """
    if use_fast is not None:
        logger.warning_once(
            "The `use_fast` parameter is deprecated and will be removed in a future version. "
            'Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.'
        )
        if backend is None:
            backend = "torchvision" if use_fast else "pil"

    if backend is None:
        if base_class_name in DEFAULT_TO_PIL_BACKEND_IMAGE_PROCESSORS:
            return "pil"
        return "torchvision" if is_torchvision_available() else "pil"

    return backend
```

**读法**：
- 判定顺序写得很清楚：① 传了弃用的 `use_fast` 就换算成 backend 并告警；② 显式 `backend` 原样返回；
  ③ 都没给时，先看是不是 Lanczos 名单里的处理器（它们默认强制 `pil`），再按 `torchvision` 可用性选。
- 实测：`_resolve_backend(None, None, "Glm5NextImageProcessor")` → `'torchvision'`；
  `_resolve_backend("pil", None, ...)` → `'pil'`。GLM-5 不在 Lanczos 名单里，所以走默认分支。
- 为什么要有这一层？因为**同一个模型的两套 backend 会给出数值上不同的结果**，
  这不是性能开关，是**输出契约**的一部分 —— 所以它被显式命名，而不是藏在布尔值里。

拿到 backend 之后，还要能容忍「这个 backend 装不上」：

<!-- src: models/auto/image_processing_auto.py -->
```python
    backends_to_try = [backend] + [k for k in mapping if k != backend]

    for b in backends_to_try:
        value = mapping.get(b)
        if value is None:
            continue

        # Value can be a class object (from resolved mapping) or a string class name
        if isinstance(value, type):
            processor_class = value
        else:
            processor_class = get_image_processor_class_from_name(value)

        if processor_class is None or getattr(processor_class, "is_dummy", False):
            continue

        if b != backend:
            logger.warning_once(f"Requested {backend} backend is not available. Falling back to {b} backend.")
        return processor_class
```

**读法**：
- `backends_to_try = [backend] + [k for k in mapping if k != backend]`：先试你要的，
  再试另一个，最后试任何注册过的自定义 backend。**降级是静默的**（只在真的换 backend 时 `warning_once`）。
- 跳过条件是 `getattr(processor_class, "is_dummy", False)`：依赖缺失时，延迟加载模块给出的是
  一个 Dummy 类（占位符），它在这里被识别并跳过，而不是等到实例化时才炸。
- 这就是「四张表」里唯一一张**值有内部结构**的表：它必须同时回答「是哪个类」和「哪个类在当前环境可用」。

---

## 六、video：类型没写全时，从 image 反推

视频处理器的配置文件常常只有 image processor 的字段（历史包袱）。`from_pretrained` 里专门有一段兜底：

<!-- src: models/auto/video_processing_auto.py -->
```python
        # If we still don't have the video processor class, check if we're loading from a previous image processor config
        # and if so, infer the video processor class from there.
        if video_processor_class is None and video_processor_auto_map is None:
            image_processor_class = config_dict.pop("image_processor_type", None)
            if image_processor_class is not None:
                video_processor_class_inferred = image_processor_class.replace("ImageProcessor", "VideoProcessor")

                # Some models have different image processors, e.g. InternVL uses GotOCRImageProcessor
                # We cannot use GotOCRVideoProcessor when falling back for BC and should try to infer from config later on
                if video_processor_class_from_name(video_processor_class_inferred) is not None:
                    video_processor_class = video_processor_class_inferred
            if "AutoImageProcessor" in config_dict.get("auto_map", {}):
                image_processor_auto_map = config_dict["auto_map"]["AutoImageProcessor"]
                video_processor_auto_map = image_processor_auto_map.replace("ImageProcessor", "VideoProcessor")
```

**读法**：
- 手法是**字符串替换**：`"Glm5NextImageProcessor"` → `"Glm5NextVideoProcessor"`。
  但注意它替换完还要 `video_processor_class_from_name(...) is not None` 验一次 ——
  因为 `InternVL` 那类模型用的是 `GotOCRImageProcessor`，硬替换出来的 `GotOCRVideoProcessor` 并不存在
  （源码注释原话：`We cannot use GotOCRVideoProcessor when falling back for BC`）。
- `auto_map` 也要跟着换：`AutoImageProcessor` → `AutoVideoProcessor`。**远程代码的类引用同样遵守命名约定**，
  这正是「config 决定形状」在自定义模型上的投影。
- 顺带说明为什么第五节那张 backend 字典表没有波及这里：video 表的值是纯字符串，
  它的「环境依赖」是用第二节那个循环整体置 `None` 来表达的 —— 两处解决的是同一个问题，手段不同。

---

## 七、feature extractor：类名反查的最后一段兜底链

`AutoFeatureExtractor.from_pretrained` 的尾部，是「内置表 + 远程代码 + 映射表」三段的收口：

<!-- src: models/auto/feature_extraction_auto.py -->
```python
        has_remote_code = feature_extractor_auto_map is not None
        has_local_code = feature_extractor_class is not None or type(config) in FEATURE_EXTRACTOR_MAPPING
        explicit_local_code = has_local_code and not (
            feature_extractor_class or FEATURE_EXTRACTOR_MAPPING[type(config)]
        ).__module__.startswith("transformers.")
        if has_remote_code:
            if "--" in feature_extractor_auto_map:
                upstream_repo = feature_extractor_auto_map.split("--")[0]
            else:
                upstream_repo = None
            trust_remote_code = resolve_trust_remote_code(
                trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code, upstream_repo
            )

        if has_remote_code and trust_remote_code and not explicit_local_code:
            feature_extractor_class = get_class_from_dynamic_module(
                feature_extractor_auto_map, pretrained_model_name_or_path, **kwargs
            )
            _ = kwargs.pop("code_revision", None)
            feature_extractor_class.register_for_auto_class()
            return feature_extractor_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif feature_extractor_class is not None:
            return feature_extractor_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        # Last try: we use the FEATURE_EXTRACTOR_MAPPING.
        elif type(config) in FEATURE_EXTRACTOR_MAPPING:
            feature_extractor_class = FEATURE_EXTRACTOR_MAPPING[type(config)]
            return feature_extractor_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
```

**读法**：
- `has_local_code` / `explicit_local_code` 这一对变量的含义是：**本地到底有没有能用的类**、
  以及这个类是不是「库外」的。`explicit_local_code` 为真时，即使仓库里声明了 `auto_map`，
  也不会去加载远程代码 —— 本地明明白白有实现，就不该执行网上下来的代码。
- 三个出口顺序固定：远程代码 → 配置里写死的类 → `FEATURE_EXTRACTOR_MAPPING[type(config)]`。
  最后一步就是第一节那张「一行一个类」的表在真正干活。
- 与第四节对照：tokenizer 把「注册表」放在最前，feature extractor 把「配置里写的类名」放在最前。
  **没有全局统一的优先级，每个门面按自己的信任模型排** —— 这是本课最容易被忽略的一点。

---

## 八、AutoProcessor：一个门面背后是三个子处理器

`AutoProcessor.from_pretrained` 的第一段，是「先找哪个文件说话」：

<!-- src: models/auto/processing_auto.py -->
```python
        processor_class = None
        processor_auto_map = None

        # First, let's see if we have a processor or preprocessor config.
        # Filter the kwargs for `cached_file`.
        _hub_valid_kwargs = (
            "cache_dir",
            "force_download",
            "proxies",
            "token",
            "revision",
            "local_files_only",
            "subfolder",
            "repo_type",
            "user_agent",
        )
        cached_file_kwargs = {key: kwargs[key] for key in _hub_valid_kwargs if key in kwargs}
        # We don't want to raise
        cached_file_kwargs.update(
            {
                "_raise_exceptions_for_gated_repo": False,
                "_raise_exceptions_for_missing_entries": False,
                "_raise_exceptions_for_connection_errors": False,
            }
        )

        # Let's start by checking whether the processor class is saved in a processor config
        processor_config_file = cached_file(pretrained_model_name_or_path, PROCESSOR_NAME, **cached_file_kwargs)
        if processor_config_file is not None:
            config_dict, _ = ProcessorMixin.get_processor_dict(pretrained_model_name_or_path, **kwargs)
            processor_class = config_dict.get("processor_class")
            if "AutoProcessor" in config_dict.get("auto_map", {}):
                processor_auto_map = config_dict["auto_map"]["AutoProcessor"]
```

**读法**：
- 先把 Hub 相关的 kwargs 过滤成 `cached_file_kwargs`，并且把三个「不要抛异常」开关关掉 ——
  **探测配置文件存在与否，不允许失败**，否则「没有 processor 的模型」会直接报错。
- 然后是**文件优先级**：`processor_config.json`（`PROCESSOR_NAME`）→ 找不到才看
  `preprocessor_config.json` / `video_processor.json` → 再看 `tokenizer_config.json` → 最后才去读
  `config.json` 的 `processor_class` 属性。**越靠近「专门描述处理器」的文件越优先。**
- 每一层都同时取 `processor_class` 和 `auto_map["AutoProcessor"]`：前者是本地类名，后者是远程代码引用。
- 这一段解释了为什么 `AutoProcessor` 能同时当「多模态处理器工厂」和「纯 tokenizer 工厂」用：
  它先认文件，再认类。

三级都落空时，它不会立刻报错，而是**挨个降级**：

<!-- src: models/auto/processing_auto.py -->
```python
        # At this stage, there doesn't seem to be a `Processor` class available for this model.
        # Let's try the commonly available classes
        for klass in (AutoTokenizer, AutoImageProcessor, AutoVideoProcessor, AutoFeatureExtractor):
            try:
                return klass.from_pretrained(
                    pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
                )
            except Exception:
                continue

        raise ValueError(
            f"Unrecognized processing class in {pretrained_model_name_or_path}. Can't instantiate a processor, a "
            "tokenizer, an image processor, a video processor or a feature extractor for this model. "
            "Make sure the repository contains the files of at least one of those processing classes."
        )
```

**读法**：
- `for klass in (AutoTokenizer, AutoImageProcessor, AutoVideoProcessor, AutoFeatureExtractor)` ——
  四连兜底，谁先成功用谁，异常被 `except Exception: continue` 吞掉。
  所以一个只有 `tokenizer_config.json` 的纯文本模型，`AutoProcessor.from_pretrained` 也能用。
- 四个都失败才抛 `ValueError`，并且错误信息把四种文件都列了一遍 —— **报错本身就是文档**。
- 子处理器不是在这里组合的：`PROCESSOR_MAPPING[type(config)]` 拿到 `Glm5NextProcessor` 之后，
  由 `ProcessorMixin` 按该类的 `__init__` 形参名去逐个加载。实测
  `Glm5NextProcessor.get_attributes()` → `['image_processor', 'tokenizer', 'video_processor']`，
  所以 GLM-5 的 AutoProcessor 会实例化**三个**子处理器（验收点的答案，展开在 `lesson.js` 第 8 幕）。

---

## 九、与后面课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| `_LazyAutoMapping` 内部怎么按 config 类反查 | L1-05（已讲） |
| `auto_mappings.py` 是怎么生成的 | L1-05 的表 + L1-07 的 `check_auto` 视角 |
| `Glm5NextProcessor.__init__` 内部怎么拼 prompt | L2-06 / L2-07 |
| `Glm5NextImageProcessorPil` 与 torchvision 版的数值差异 | L2 图像预处理 |
| 远程代码（`auto_map` / `trust_remote_code`）的信任模型 | L1-07 |
| 权重名与类名的映射表（`conversion_mapping.py`） | L1-07 |

**一句话总结**：

> 六个文件、六张表、六套 `register` —— 写法完全对称，只有**裁决顺序**各不相同：
> 读表永远零 import，取格子才 import 那一个包；表的内容则取决于**生成、补丁、环境**三件事。
