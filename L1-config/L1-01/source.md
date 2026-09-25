<!-- glm5-coverage
configuration_utils.py
-->

# L1-01 · PreTrainedConfig：配置对象的公共契约 — 源文件

**这一课只回答三个问题：一份 `config.json` 怎么变成对象、`@strict` 在什么时刻检查它、嵌套配置怎么被寻址。**

视角：不看任何一个模型的具体字段，只看**所有 config 共用的那台机器** ——
文件怎么读进来、字典怎么变成对象、对象怎么再写回文件，以及这台机器凭什么敢说"你给的配置是合法的"。

| 文件 | 行数 | 本课引用块 |
|---|---|---|
| `configuration_utils.py` | 1566 | 14 |

> 行数用 `wc -l` 实测；计划文档记作 1567，差的 1 行在文件末尾的换行符上。

**实测数字**（在本机 venv 里用 `transformers 5.18.0.dev0` + `huggingface_hub 1.33.0` 跑出来。
这是**脚本输出**，不是源文件引用，所以用 `text` 块标出，**不计入覆盖率**）：

```text
configuration_utils.py               1566 行（wc -l）
本文件里定义的 validate_* 方法        4 个：validate_output_attentions / validate_architecture
                                          validate_token_ids / validate_layer_type
实例上能看到的 validate_*             5 个（多出来的 validate_rope 来自 RotaryEmbeddingConfigMixin）
Glm5NextConfig.sub_configs            {'vision_config': Glm5NextVisionConfig,
                                       'text_config':  Glm5NextTextConfig}
Glm5NextConfig.base_config_key        ''              ← 顶层：不下钻
Glm5NextTextConfig.base_config_key    'text_config'   ← 子配置：去顶层文件里下钻

Glm5NextConfig()  to_dict() 22 个顶层键      to_diff_dict() 11 个顶层键
                  嵌套 text_config 62 键             嵌套 text_config 51 键
to_json_string(use_diff=False) 6024 字节 ；use_diff=True 4894 字节
save_pretrained 落盘的 config.json = 4894 字节 / 226 行（即 diff 形态）

Glm5NextTextConfig(num_hidden_layers=45, layer_types=[...44 个...])
    → StrictDataclassClassValidationError：validator 'validate_layer_type'
      ValueError: `num_hidden_layers` (45) must be equal to the number of `layer_types` (44)
Glm5NextTextConfig(num_hidden_layers="四十五")
    → StrictDataclassFieldValidationError：field 'num_hidden_layers'
      TypeError: Field 'num_hidden_layers' expected int, got str
Glm5NextTextConfig(pad_token_id=-1)  → 不抛错，只 warning_once 一条日志

attn_implementation={"": "sdpa", "text_config": "eager", "vision_config": "flash_attention_2"}
    → 顶层 sdpa / text_config eager / vision_config flash_attention_2（三处各不相同）
attn_implementation="sdpa" → 三处都是 sdpa
```

---

## 一、入口：`from_pretrained` 只做四件事

先看这个类方法的**尾部**。前面 80 行全是文档字符串与形参声明，真正干活的只有这 8 行：
打包参数 → 拿字典 → 下钻 → 造对象。

<!-- src: configuration_utils.py -->
```python
        kwargs["cache_dir"] = cache_dir
        kwargs["force_download"] = force_download
        kwargs["local_files_only"] = local_files_only
        kwargs["revision"] = revision

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if cls.base_config_key and cls.base_config_key in config_dict:
            config_dict = config_dict[cls.base_config_key]
```

**读法**：
- 前四行是**把具名形参塞回 `kwargs`**。所以 `from_pretrained` 之后所有下游函数只认一个 `**kwargs`，
  不需要一路传 5 个形参。这是"薄签名 + 厚 kwargs"的写法，代价是下游必须自己 `pop`（见第三节）。
- `get_config_dict` 返回的是 `(dict, kwargs)` **一对** —— 第二个返回值是"还没被消费的 kwargs"，
  它会一路传到 `from_dict`，最后用来覆盖配置值。
- `cls.base_config_key` 是**类属性**，不是实例属性。空字符串表示"我是顶层配置，不用下钻"；
  非空（实测 `Glm5NextTextConfig` 是 `"text_config"`）表示"我是别人的一部分，请先把我那一格取出来"。
  下一行 `config_dict = config_dict[cls.base_config_key]` 就是这条规则的全部实现。
- 回顾 L0-01：那一课里 `Glm5NextTextModel.forward` 收的是 `self.config`；而 `self.config`
  正是这一课造出来的对象。**config 不是参数表，是模型形状的可执行定义** —— 这句话在 L1-02 会被逐字段验证。

---

## 二、`get_config_dict`：revision 只解析一次，配置文件可能换一个

`from_pretrained` 把活交给了 `get_config_dict`。它做的第一件事不是读文件，而是**解析版本号**。

<!-- src: configuration_utils.py -->
```python
        # Resolve the revision once, so that both config files below are read from the exact same repository state.
        kwargs["revision"] = resolve_revision(
            pretrained_model_name_or_path,
            kwargs.get("revision"),
            token=kwargs.get("token"),
            local_files_only=kwargs.get("local_files_only", False),
            cache_dir=kwargs.get("cache_dir"),
        )

        original_kwargs = copy.deepcopy(kwargs)
        # Get config dict associated with the base config file
        config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
```

**读法**：
- `"main"` 这种分支名在 Hub 上是**会动的**。如果第一次读 `config.json` 时 `main` 指向 A，
  第二次读 `config.5.0.0.json` 时 `main` 指向 B，两个文件就来自不同的仓库状态 —— 拼出来的 config
  谁也没见过。所以这里先把 revision 解析成一个**确定的 commit**，再往下传。
- `original_kwargs = copy.deepcopy(kwargs)` 是为**第二次读取**准备的：下面那次读取会 `pop` 掉一堆键
  （`cache_dir`、`token`、`subfolder`…），所以必须留一份未被污染的副本。
- 这是本课第一个"路径上的隐藏动作"：**`from_pretrained` 可能读两个文件，而不是一个。**

那第二个文件从哪来？就藏在第一次读到的字典里：

<!-- src: configuration_utils.py -->
```python
        # That config file may point us toward another config file to use.
        if "configuration_files" in config_dict:
            configuration_file = get_configuration_file(config_dict["configuration_files"])
            config_dict, kwargs = cls._get_config_dict(
                pretrained_model_name_or_path, _configuration_file=configuration_file, **original_kwargs
            )

        return config_dict, kwargs
```

**读法**：
- 有些仓库同时存了 `config.json`、`config.4.30.0.json`、`config.5.0.0.json`……
  老版本 transformers 读 `config.json`，新版本应该读"不高于自己版本的最新那份"。
  选择逻辑在模块级函数 `get_configuration_file`（第 1470 行）里，本课不展开。
- 实测：候选 `["config.json", "config.4.0.json", "config.4.30.0.json", "config.5.0.0.json", "config.9.9.9.json"]`
  在 `5.18.0.dev0` 下选中 `config.5.0.0.json` —— 9.9.9 比自己新，不选。
- 注意 `**original_kwargs`：第二次读取用的是**干净的**那份 kwargs，只有 `_configuration_file` 是新增的。
  如果这里传的是被 `pop` 过的 `kwargs`，`cache_dir` 和 `token` 就丢了，会变成匿名下载。

---

## 三、JSON → dict → 对象：两个隐藏动作

真正碰文件的函数只有两个。`from_json_file` 是给"我手上就是一个 json 文件路径"用的，
它和 `from_pretrained` 在最后一步汇合：都要把 dict 变成对象。

<!-- src: configuration_utils.py -->
```python
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file: str | os.PathLike):
        with open(json_file, encoding="utf-8") as reader:
            text = reader.read()
        config_dict = json.loads(text)

        return cls._decode_special_floats(config_dict)
```

**读法**：
- 隐藏动作一：`json.loads` 之后**不是直接返回**，而是先过一遍 `_decode_special_floats`。
- 隐藏动作二：`cls(**config_dict)` —— 注意是 `cls(...)`，不是 `cls.from_dict(...)`。
  **`from_json_file` 不经过 `from_dict`**，所以 `from_dict` 里那套 "kwargs 覆盖 config_dict" 的逻辑
  在这条路径上不存在。两条路径的差别，正是下面这个函数要解决的问题：

<!-- src: configuration_utils.py -->
```python
        if isinstance(obj, dict):
            if set(obj.keys()) == {_FLOAT_TAG_KEY} and isinstance(obj[_FLOAT_TAG_KEY], str):
                tag = obj[_FLOAT_TAG_KEY]
                if tag in _FLOAT_TAG_VALUES:
                    return _FLOAT_TAG_VALUES[tag]
                return obj

            return {k: cls._decode_special_floats(v) for k, v in obj.items()}
```

**读法**：
- `Infinity` / `-Infinity` / `NaN` 不是合法 JSON。写出去时被编码成 `{"__float__": "Infinity"}`
  （编码在 `_encode_special_floats`，第 996 行），读回来时在这里还原。
- **实测的一个坑**：解码只发生在 `_dict_from_json_file` 这一层。用
  `Mini.from_dict(json.loads(open(path).read()))` 读同一个文件，`rope_theta` 会停在
  `{'__float__': '-Infinity'}` 这个字典上；而 `Mini.from_json_file(path)` 得到的是真正的 `-inf`。
  换句话说：**"从文件读"和"从字典读"不是同一条路，特殊浮点只在文件那条路上被还原。**
- 这也解释了为什么 `_encode_special_floats` 要递归处理 `dict` 与 `list`：config 里到处是嵌套字典。

---

## 四、★ `@strict`：把校验挂在「构造完成」那一刻

现在回答第二个问题。三个装饰器叠在类上，顺序不是随便写的：

<!-- src: configuration_utils.py -->
```python
@dataclass_transform(kw_only_default=True)
@strict(accept_kwargs=True)
@dataclass(repr=False)
class PreTrainedConfig(PushToHubMixin, RotaryEmbeddingConfigMixin, HeterogeneousConfigMixin):
```

**读法**：
- 从下往上读：`@dataclass(repr=False)` 先生成 `__init__` 与 `__eq__`；
  然后 `@strict(accept_kwargs=True)` 在**已经是 dataclass** 的类上做加工（它开头就会检查
  `__dataclass_fields__`，不是 dataclass 直接报错）；最上面的 `@dataclass_transform` 只影响类型检查器，
  运行时不做任何事。
- `@strict` 来自 `huggingface_hub.dataclasses`（不是 transformers 自己的代码）。实测它装了三样东西：
  1. 重写 `__setattr__`，按字段类型逐次校验 —— 类型不对时抛 `StrictDataclassFieldValidationError`；
  2. 扫描类里**所有** `validate_*` 方法（只收 `self` 一个参数的），收集成一张表；
  3. 生成一个 `validate()`，把这张表里的方法挨个跑一遍，并把 `ValueError`/`TypeError` 包成
     `StrictDataclassClassValidationError`；**再包一层 `__init__`，在构造结束时调用 `cls.validate(self)`**。
- 第 3 条就是验收点的答案：**strict 校验的触发时机是"构造完成的那一刻"，不是字段被赋值的那一刻。**
  所以一个非法组合（比如 `layer_types` 长度对不上 `num_hidden_layers`）在 `Config(...)` 这一行就炸，
  而不是等到模型跑到一半才炸。
- `accept_kwargs=True` 是给向后兼容用的：老 checkpoint 里可能带着现在已经不存在的键，
  `__init__` 不能因为不认识就拒绝。

基类装好了，子类怎么办？答案在 `__init_subclass__` 里 —— 每一个子类被定义时都会再走一遍：

<!-- src: configuration_utils.py -->
```python
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls_has_custom_init = "__init__" in cls.__dict__
        # kw_only=True ensures fields without defaults in subclasses can follow
        # parent fields that have defaults (Python dataclass ordering rule).
        # Config fields are always passed as keyword arguments, so this is safe.
        cls = dataclass(cls, repr=False, kw_only=True)

        if not cls_has_custom_init:
            # Wrap all subclasses to accept arbitrary kwargs for BC
            # only if the subclass has no custom `__init__`. Most
            # remote code has an init defined, but some model are not
            # See https://huggingface.co/hmellor/Ilama-3.2-1B/blob/main/configuration_ilama.py
            cls = wrap_init_to_accept_kwargs(cls)
```

**读法**：
- 这段代码在**类定义执行完、装饰器还没跑**的时候被 Python 自动调用。所以顺序是：
  子类字段声明 → `__init_subclass__` 把它 dataclass 化并包一层 `__init__` → 子类自己的 `@strict` 再包一层
  （这次包的是刚生成的 `__init__`），校验就此生效。
- `kw_only=True` 不是风格偏好，是**语法必需**：Python 的 dataclass 规则不允许"无默认值的字段"
  排在"有默认值的字段"后面，而 `PreTrainedConfig` 已经有一堆带默认值的字段。既然 config 永远用关键字
  传参，强制 kw_only 就没有任何代价。
- `cls_has_custom_init` 那一支是给远程代码留的后门：有自定义 `__init__` 的类不包装，
  避免把作者自己的初始化逻辑盖掉。
- 注意：GLM-5 的三个 config 类在自己的文件里**又写了一次 `@strict`**（实测
  `configuration_glm5_next.py` 里有 3 处）。基类那份管不到子类新增的 `validate_*`，
  所以子类要重新收集一遍。

---

## 五、★ `validate_` 家族：谁抛错，谁只警告

`@strict` 只负责"收集并调用"，具体查什么由这四个方法决定。先看最硬的那个：

<!-- src: configuration_utils.py -->
```python
    def validate_layer_type(self):
        """Check that `mlp_layer_types` and `layer_types` is correctly defined."""
        for allowed_types, layer_types in zip(
            [ALLOWED_ATTN_LAYER_TYPES, ALLOWED_MLP_LAYER_TYPES], ["layer_types", "mlp_layer_types"]
        ):
            layers = getattr(self, layer_types, None)
            if not (layers is not None and hasattr(self, "num_hidden_layers")):
                return

            if not all(layer_type in allowed_types for layer_type in layers):
                raise ValueError(f"The `{layer_types}` entries must be in {allowed_types} but got {layers}")
            elif self.num_hidden_layers is not None and self.num_hidden_layers != len(layers):
                raise ValueError(
                    f"`num_hidden_layers` ({self.num_hidden_layers}) must be equal to the number of `{layer_types}` "
                    f"({len(layers)})"
                )
```

**读法**：
- 两条硬约束：**每个层类型名必须在白名单里**，且**列表长度必须等于 `num_hidden_layers`**。
  第二条正是 L0-01 里那句 `causal_mask_mapping[self.config.layer_types[i]]` 的前提 ——
  如果长度对不上，主干循环会 `IndexError`。**校验在这里拦住的，是后面一定会炸的形状错误。**
- 白名单是模块级常量 `ALLOWED_ATTN_LAYER_TYPES` / `ALLOWED_MLP_LAYER_TYPES`（第 65、82 行），
  两者相加得到 `ALLOWED_LAYER_TYPES`。GLM-5 用到的 `linear_attention` 与 `indexed_attention` 都在里面。
- 实测失败时的异常是 `StrictDataclassClassValidationError`，消息里点名了 validator：
  `Class validation error for validator 'validate_layer_type': ValueError: num_hidden_layers (45) must be
  equal to the number of layer_types (44)`。

但同一个家族里，有一个方法**故意不抛错**：

<!-- src: configuration_utils.py -->
```python
    def validate_token_ids(self):
        """Part of `@strict`-powered validation. Validates the contents of the special tokens."""
        text_config = self.get_text_config(decoder=True)
        vocab_size = getattr(text_config, "vocab_size", None)
        if vocab_size is not None:
            # Check for all special tokens, e..g. pad_token_id, image_token_id, audio_token_id
            for name in text_config:
                value = getattr(text_config, name)
                if name.endswith("_token_id") and isinstance(value, int) and not 0 <= value < vocab_size:
                    # Can't be an exception until we can load configs that fail validation: several configs on the Hub
                    # store invalid special tokens, e.g. `pad_token_id=-1`
                    logger.warning_once(
                        f"Model config: {name} must be `None` or an integer within the vocabulary (between 0 "
                        f"and {vocab_size - 1}), got {value}. This may result in unexpected behavior."
                    )
```

**读法**：
- 判据是"名字以 `_token_id` 结尾且值是 int" → 必须在 `[0, vocab_size)` 内。
  实测 `Glm5NextTextConfig(pad_token_id=-1)` 构造**成功**，只在日志里留下一条 warning。
- 注释把原因写得很清楚：**Hub 上真的有 `pad_token_id=-1` 的 config**。如果这里抛错，
  这些模型就再也加载不起来了 —— 所以选择"降级为警告"。
- 注意 `for name in text_config`：`PreTrainedConfig.__iter__`（第 1048 行）让 config 对象**可以直接被遍历**，
  产出的是配置键名。没有这个 `__iter__`，这一行会报 `TypeError`。
- 一句话总结这个家族的分工：**"形状错了"抛错，"取值可疑"警告。**
  同类里 `validate_architecture` 查 `head_dim * num_heads == embed_dim`，`validate_output_attentions` 查
  "要 attentions 就必须用 eager 实现"，两者都抛错；`validate_rope`（继承自 `RotaryEmbeddingConfigMixin`，
  不在本文件里）也走同一条收集通道。

---

## 六、★ 嵌套配置的寻址：`base_config_key` 与 `sub_configs`

第三个问题。两个类属性就是全部规则：

<!-- src: configuration_utils.py -->
```python
    base_config_key: ClassVar[str] = ""
    sub_configs: ClassVar[dict[str, type[PreTrainedConfig]]] = {}
```

**读法**：
- `base_config_key` 回答"**我在别人的字典里叫什么**"。实测：顶层 `Glm5NextConfig` 是 `""`，
  子配置 `Glm5NextTextConfig` 是 `"text_config"`。第一节里那句
  `if cls.base_config_key and cls.base_config_key in config_dict` 就是这条规则的唯一消费者。
- `sub_configs` 回答"**我下面有谁**"。实测 `Glm5NextConfig.sub_configs` 是
  `{'vision_config': Glm5NextVisionConfig, 'text_config': Glm5NextTextConfig}` —— 键名与子类的
  `base_config_key` 必须一致，否则下钻会落空。
- 两个属性都用 `ClassVar` 标注，因此**不会进 `__dict__`、不会被序列化**。这也意味着它们只能在类定义里
  写死，不能从 json 里读出来。展开见 L1-03。

`sub_configs` 不只用在下钻上，还被用来做**递归下发**：

<!-- src: configuration_utils.py -->
```python
        for subconfig_key in self.sub_configs:
            subconfig = getattr(self, subconfig_key, None)
            if subconfig is not None:
                current_subconfig_attn = getattr(subconfig, "_attn_implementation", None)
                sub_implementation = (
                    value if not isinstance(value, dict) else value.get(subconfig_key, current_subconfig_attn)
                )
                subconfig._attn_implementation = sub_implementation
```

**读法**：
- 这是 `_attn_implementation` 的 setter。传字符串时，同一个值被写到顶层和每一个子配置上；
  传字典时，**按 `sub_configs` 的键分别取值** —— 实测
  `{"": "sdpa", "text_config": "eager", "vision_config": "flash_attention_2"}` 得到三处各不相同，
  空字符串键 `""` 是顶层的槽位。
- 所以 `sub_configs` 既是"寻址表"（往下找），也是"广播名单"（往下发）。改一个键名，
  这两件事会同时失效。
- 运行时还有第三种寻址方式：`get_text_config()`（第 1346 行）按名字表
  `("text_encoder", "decoder", "generator", "text_config")` 去找文本配置；实测在 `Glm5NextConfig` 上
  返回 `text_config`，在 `Glm5NextTextConfig` 上返回它自己。**找到多个就抛错**（ambiguous），
  这就是它不做模糊匹配的原因。
- 回顾 L0-05：那一课的 `hc_mult = 4` 决定残差流条数，而 `hc_mult` 就住在 `text_config` 这一格里。
  **顶层 config 的形状由子配置决定，子配置的形状由字段决定** —— 这是 L1-02 与 L1-03 的分工。

---

## 七、出口：`to_dict` → `to_diff_dict` → `to_json_string`

回程比去程多一层：写文件时默认**只写差异**。先看 `to_dict` 的开头三件事：

<!-- src: configuration_utils.py -->
```python
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type

        # Transformers version when serializing the model
        output["transformers_version"] = __version__

        # Pop "kwargs" since they are unpacked and set in the post init
        output.pop("kwargs", None)
```

**读法**：
- `copy.deepcopy(self.__dict__)`：**序列化的输入是实例字典**，不是 dataclass 字段表。所以
  任何被塞进 `__dict__` 的东西都会出现在 json 里 —— 这也解释了为什么 `_remove_keys_not_serialized`
  必须存在（它负责把 `_attn_implementation_internal` 这类内部键删掉，并把 `_output_attentions`
  改名成 `output_attentions` 再写出去）。
- `model_type` 来自**类**（`self.__class__.model_type`），不是实例。它是 L1-05 里"字符串如何找到类"
  要用的那个字符串，必须落盘。
- `transformers_version` 被**无条件覆写**成当前版本。实测：一个新建的 `Glm5NextConfig()` 的
  `transformers_version` 字段是 `None`，保存再加载后变成 `"5.18.0.dev0"`；因为它是 dataclass 字段，
  这会参与 `__eq__`，于是"存盘再读回来"的对象和原对象**不相等**。
- `output.pop("kwargs", None)` 是历史遗留：早期实现把未识别的 kwargs 塞进 `self.kwargs`。
  现在它们在 `__post_init__` 里就被展开了，这个 `pop` 只是保险。

最后一步是字符串化：

<!-- src: configuration_utils.py -->
```python
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()

        # Handle +/-Infinity and NaNs
        config_dict = self._encode_special_floats(config_dict)

        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"
```

**读法**：
- `use_diff` 默认是 `True`：**默认写差异**。`to_diff_dict`（第 1053 行）会同时和
  `PreTrainedConfig()`（基类默认）与 `self.__class__()`（同类默认）比一遍，只留不同的键，
  嵌套配置再递归地比（`recursive_diff_dict`，第 1500 行），最后删掉 `_name_or_path`。
- 实测差异有多大：`Glm5NextConfig()` 的 `to_dict()` 有 **22** 个顶层键，
  `to_diff_dict()` 只剩 **11** 个；序列化出来 **6024 字节 → 4894 字节**。
  `save_pretrained` 落盘的 `config.json` 就是后者。
- `sort_keys=True` 让 diff 稳定可比较；结尾补一个 `"\n"`，避免 git 报 "No newline at end of file"。
- 顺序很重要：**先 diff，再编码特殊浮点**。反过来的话，被编码成字典的 `{"__float__": ...}` 会被
  diff 当成普通字典递归比较，行为就不确定了。

---

## 八、保存前的第二次校验

`@strict` 已经保证构造时校验过一次。为什么保存时还要再来一次？

<!-- src: configuration_utils.py -->
```python
        if hasattr(self, "validate"):
            self.validate()
        self.to_json_file(output_config_file, use_diff=True)
```

**读法**：
- 因为**构造之后对象还可以被改**。`config.num_hidden_layers = 7` 这种赋值只过 `__setattr__` 的
  逐字段类型校验，不会重跑 `validate_layer_type` —— 改完可能就和 `layer_types` 长度对不上了。
- 于是 `save_pretrained` 在写盘前显式调一次 `self.validate()`：**不让非法配置流出这台机器**。
  这是"校验的两个时机"里的第二个（第一个是构造完成）。
- `hasattr(self, "validate")` 这个判断是给没用 `@strict` 的老 config 类留的活口。
- 顺带一提，同一个函数上面还有一道不同的闸：`_get_generation_parameters()` 非空就抛
  `ValueError`。实测在构造时传 `top_k=50` 会被 `__post_init__` 静默丢掉（config 上根本没有这个属性），
  只有**构造之后再赋值**才会被这道闸拦住。

---

## 九、与后面课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| `Glm5NextTextConfig` 的每个字段是什么意思 | L1-02 |
| 视觉子配置与顶层嵌套怎么拼起来 | L1-03 |
| `attribute_map` / 文档生成 / 继承体系 | L1-04 |
| `model_type` 这个字符串怎么变成类 | L1-05 |
| 另外四张自动注册表 | L1-06 |
| 远程代码与权重名映射 | L1-07 |
| `layer_types` 为什么是 34 : 11 | L0-03 |
| `hc_mult` 怎么变成 4 条残差流 | L0-05 |

**一句话总结**：

> 一份 `config.json` 的旅程是 **`cached_file` → `json.loads` → `_decode_special_floats` →
> （必要时 `base_config_key` 下钻）→ `cls(**config_dict)` → `@strict` 在构造末尾跑一遍 `validate_*`**；
> 回程是 **`__dict__` → `to_dict` → `to_diff_dict` → `_encode_special_floats` → `json.dumps`**，
> 而 `save_pretrained` 会在写盘前把 `validate_*` **再跑一遍**。
