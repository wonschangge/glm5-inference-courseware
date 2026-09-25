# L1-06 · 其余四张自动注册表 — 六张表的裁决顺序

> 层：**第 1 层 · 配置与分派** ｜ 优先级：P1 ｜ 前置课：`L1-05`

## 学习目标

看完这一课，你应该能：

1. **说出「字符串 → 类」的两步**：查表（零 import）→ 取格子（才 import 那一个包），
   并解释 `_LazyAutoMapping` 到底省下了什么（对应验收点 1）；
2. **说出六张表各自的「谁说了算」**：tokenizer 是注册表优先、processor 是文件优先、
   image 是 backend 字典、video 是整表随环境重写（对应验收点 2）；
3. **回答验收点 3**：AutoProcessor 对 GLM-5 会实例化哪几个子处理器，依据是哪一行实测。

## 覆盖的源文件（6 个）

| 文件 | 行数 | 引用块 |
|---|---|---|
| `models/auto/modeling_auto.py` | 2689 | 4 |
| `models/auto/tokenization_auto.py` | 1039 | 2 |
| `models/auto/image_processing_auto.py` | 780 | 2 |
| `models/auto/video_processing_auto.py` | 434 | 2 |
| `models/auto/feature_extraction_auto.py` | 393 | 2 |
| `models/auto/processing_auto.py` | 383 | 2 |
| **合计** | **5718** | **14** |

> 行数按作业书口径（`len(text.split("\n"))`，含文件末尾换行）。
> 本课引用的实测数字来自 `_data/recon/probe_l106.py` —— 它是**工装脚本，不计入覆盖率**
> （覆盖域只统计 `src/transformers/**/*.py`）。

## 场景（9 幕）

1. **六个文件，六张表，一个动作** —— 主链路 `model_type → 表 → 惰性映射 → 类`，外加 49 / 54 / 5718 三个数
2. **config 决定形状** —— GLM-5 在 `MODEL_MAPPING` / `…CAUSAL_LM…` / `…IMAGE_TEXT_TO_TEXT…` 三张表上的三个答案（实测）
3. **★ 读表零 import，取格子才 import** —— 243 → 423 个模块的账单，以及 12.1 ms / 0.23 ms
4. **★ 底表是生成的，补丁是手写的** —— `update` 打在同一个 dict 对象上：120→211 / 147→180 / 29→69 / 31→45
5. **★ 同一个类名，五级裁决顺序** —— 注册表排在 309 条大表之前；46 个 model_type 上 Hub 说的不算数
6. **image：值是 backend 字典** —— `torchvision` / `pil` 两个键，与 `_resolve_backend` 的判定顺序
7. **video：可用性被烤进表里** —— 45 条一起塌（含把开关换成 `False` 再 reload 的反面实测）
8. **AutoProcessor：先认文件，再认类** —— 五层文件优先级 + 一个类带三个子处理器
9. **收束** —— 六张表对照表 + 验收点答案 + 练习 + 下一课指路

## 核心结论

### 1. 表里装的是名字，不是类

<!-- src: models/auto/modeling_auto.py -->
```python
MODEL_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_MAPPING_NAMES)
```

左右两张表都是 `str → str`。实测：读 `MODEL_MAPPING_NAMES["glm5_next"]` 新增 **0** 个模块；
直到 `AutoModel._model_mapping[Glm5NextConfig]` 才新增 **3** 个模块。**惰性省下的是「把 556 个模型全 import 一遍」。**

### 2. ★ 补丁打在同一个 dict 对象上

<!-- src: models/auto/feature_extraction_auto.py -->
```python
FEATURE_EXTRACTOR_MAPPING_NAMES.update(MISSING_FEATURE_EXTRACTOR_MAPPING_NAMES)
FEATURE_EXTRACTOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, FEATURE_EXTRACTOR_MAPPING_NAMES)
```

底表由 `auto_mappings.py` 自动生成，例外由各 auto 文件手写补丁。**因为是同一个对象，**
「表里有多少条」取决于哪些模块已经被 import 过：IMAGE 120→211、PROCESSOR 147→180、
FEATURE 29→69、VIDEO 31→45。

### 3. ★ video 表按环境整表重写

<!-- src: models/auto/video_processing_auto.py -->
```python
    # If the torchvision is not available, we set it to None
    if not is_torchvision_available():
        fast_video_processor_class = None

    VIDEO_PROCESSOR_MAPPING_NAMES[model_type] = fast_video_processor_class
```

「torchvision 在不在」不是运行时判断，而是 import 时**写进表的数据**。本机实测 0 条 `None`；
把开关换成 `False` 再 reload，**45/45 条全变 `None`**，`VIDEO_PROCESSOR_MAPPING.get(Glm5NextConfig)` 直接返回 `None`。

### 4. tokenizer 的注册表排在大表之前

<!-- src: models/auto/tokenization_auto.py -->
```python
    if class_name in REGISTERED_FAST_ALIASES:
        return REGISTERED_FAST_ALIASES[class_name]

    if class_name in REGISTERED_TOKENIZER_CLASSES:
        return REGISTERED_TOKENIZER_CLASSES[class_name]
```

用户用 `AutoTokenizer.register()` 塞进来的类，优先于库自己维护的 309 条表。
而另有 **46** 个 `model_type` 反过来：Hub 上写的 `tokenizer_class` 被认定有错，**内置表说了算**。

### 5. AutoProcessor 先认文件，再认类

<!-- src: models/auto/processing_auto.py -->
```python
        processor_config_file = cached_file(pretrained_model_name_or_path, PROCESSOR_NAME, **cached_file_kwargs)
        if processor_config_file is not None:
            config_dict, _ = ProcessorMixin.get_processor_dict(pretrained_model_name_or_path, **kwargs)
            processor_class = config_dict.get("processor_class")
```

文件优先级：`processor_config.json` → `preprocessor_config.json` / `video_processor.json` →
`tokenizer_config.json` → `config.json` 的 `processor_class`。三级都落空才走四连兜底。

### 实测数字

| 量 | 实测值 | 说明 |
|---|---|---|
| `modeling_auto` 的 Auto* 类 | 49 | 48 个 `AutoModel*` + `AutoBackbone` |
| `_LazyAutoMapping` 实例 | 54 | 同一个类被 new 54 次 |
| `MODEL_MAPPING_NAMES` | 556 | GLM-5 命中 → `Glm5NextModel` |
| `MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES` | 88 | GLM-5 命中 → `Glm5NextForConditionalGeneration` |
| `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES` | 178 | GLM-5 **不在**其中 |
| `TOKENIZER_MAPPING_NAMES` | 309 | 其中 87 条的值是 `TokenizersBackend` |
| `MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS` | 46 | 这些 model_type 不信 Hub |
| 模块数 `import transformers` → 全部取一遍 | 243 → 423 | 惰性 import 的差值 |

## 与后续课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| `_LazyAutoMapping` 内部怎么按 config 类反查 | L1-05（已讲） |
| `auto_mappings.py` 是怎么生成、怎么校验的 | L1-07 |
| `Glm5NextProcessor.__init__` 内部怎么拼 prompt | L2-06 / L2-07 |
| `Glm5NextImageProcessorPil` 与 torchvision 版的差异 | L2 图像预处理 |
| 远程代码（`auto_map` / `trust_remote_code`）的信任模型 | L1-07 |
| 权重名与类名的映射表 | L1-07 |

## 验收点

- [x] 保真门禁：14 个引用块全部逐字来自 6 个 `models/auto/*.py`
- [x] 覆盖度门禁：6 个源文件被声明（均属覆盖域 231 个文件）
- [x] 参数门禁：无非法参数
- [x] 语法检查：`node --check` 通过，9 幕字段完整
- [x] 渲染门禁：无 JS 错误、无布局溢出、交互可用（两种分辨率）
- [x] **能说出 AutoProcessor 对 GLM-5 会实例化哪几个子处理器**：**三个** ——
      `image_processor`（`Glm5NextImageProcessor` 或 `Glm5NextImageProcessorPil`，取决于 backend）、
      `tokenizer`、`video_processor`（`Glm5NextVideoProcessor`，需要 torchvision）。
      依据是实测的 `Glm5NextProcessor.get_attributes()` → `['image_processor', 'tokenizer', 'video_processor']`
- 自检：能对着第 3 幕的账单，解释「为什么读表不用 import、取格子才 import」

**一句话总结**：

> 六个文件、六张表、六套 `register` —— 写法完全对称，只有**裁决顺序**各不相同：
> 读表永远零 import，取格子才 import 那一个包；表的内容则取决于**生成、补丁、环境**三件事。
