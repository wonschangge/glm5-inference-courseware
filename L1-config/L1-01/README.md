# L1-01 · PreTrainedConfig：配置对象的公共契约

> 层：**第 1 层 · 配置与分派** ｜ 优先级：P0 ｜ 前置课：`L0-05`

## 学习目标

看完这一课，你应该能：

1. **画出 config 从 json 到对象再到模型的完整路径**，并说出中间有两个容易被略过的动作
   （`_decode_special_floats` 与 `base_config_key` 下钻）（对应验收点 1）；
2. **解释 `@strict` 的校验在什么时机触发** —— 为什么是"构造完成那一刻"而不是"字段被赋值那一刻"，
   以及 `save_pretrained` 为什么还要再查一遍（验收点 2）；
3. 说出 `base_config_key` 与 `sub_configs` 各自回答什么问题，以及它们怎样同时充当
   "寻址表"与"广播名单"。

## 覆盖的源文件（1 个）

| 文件 | 行数 | 引用块 |
|---|---|---|
| `configuration_utils.py` | 1566 | 14 |

> 行数是 `wc -l` 实测；计划文档记作 1567，差的 1 行在文件末尾的换行符上。
>
> 本课引用了 `huggingface_hub.dataclasses.strict` 的行为与一批**脚本实测输出**（写在 `source.md`
> 的 `text` 块里）。**`strict` 不在覆盖域内，脚本也不计入覆盖率** —— 覆盖域只统计
> `src/transformers/**/*.py`，本课声明的是 `configuration_utils.py` 一个文件。

## 场景（9 幕）

1. **两条路** —— 进：文件 → 字典 → 对象 → 模型；出：对象 → 差异字典 → 文件
2. **入口 `from_pretrained`** —— 四个形参塞回 kwargs，然后只做三件事
3. **版本解析** —— revision 只解析一次，配置文件可能读两份（实测版本选择表）
4. **读文件** —— 从文件读与从字典读不是同一条路：特殊浮点只在文件那条路上被还原
5. **★ `@strict`** —— 它装了三样东西，第三样决定校验时机（实测两种异常）
6. **★ `validate_` 家族** —— 谁抛错、谁只警告；白名单与长度两条硬约束
7. **★ 嵌套寻址** —— `base_config_key` 往下找、`sub_configs` 往下发（实测三个类）
8. **出口** —— 写出去的是差异：22 键 → 11 键，6024 字节 → 4894 字节
9. **收束** —— 两个校验时机 + 一条故意不抛错的检查 + 练习

## 核心结论

### 1. 完整路径上只有两个函数碰文件

<!-- src: configuration_utils.py -->
```python
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)
```

`_dict_from_json_file` 里 `json.loads` 之后还要过一遍 `_decode_special_floats`，
而 `from_json_file` 的最后一步是 `cls(**config_dict)` —— **它不经过 `from_dict`**。
`from_pretrained` 那条路才会经过 `from_dict`（那里有 "kwargs 覆盖字典" 的逻辑）。

### 2. ★ `@strict` 的校验时机是「构造完成」

<!-- src: configuration_utils.py -->
```python
@dataclass_transform(kw_only_default=True)
@strict(accept_kwargs=True)
@dataclass(repr=False)
class PreTrainedConfig(PushToHubMixin, RotaryEmbeddingConfigMixin, HeterogeneousConfigMixin):
```

`@strict`（来自 `huggingface_hub`）会收集类里所有 `validate_*` 方法，包一层 `__init__`
在**构造结束时**调用 `cls.validate(self)`。实测：

| 输入 | 结果 |
|---|---|
| `num_hidden_layers=45, layer_types=[...44 个...]` | `StrictDataclassClassValidationError`（validator `validate_layer_type`） |
| `num_hidden_layers="四十五"` | `StrictDataclassFieldValidationError`（field `num_hidden_layers`） |
| `pad_token_id=-1` | **不抛错**，只 `warning_once` |

`save_pretrained` 写盘前会**再跑一遍**：

<!-- src: configuration_utils.py -->
```python
        if hasattr(self, "validate"):
            self.validate()
        self.to_json_file(output_config_file, use_diff=True)
```

### 3. ★ 嵌套配置的寻址只有两个类属性

<!-- src: configuration_utils.py -->
```python
    base_config_key: ClassVar[str] = ""
    sub_configs: ClassVar[dict[str, type[PreTrainedConfig]]] = {}
```

| 类 | `base_config_key` | `sub_configs` 的键 |
|---|---|---|
| `Glm5NextConfig` | `''` | `vision_config`, `text_config` |
| `Glm5NextTextConfig` | `'text_config'` | （空） |
| `Glm5NextVisionConfig` | `'vision_config'` | （空） |

`base_config_key` 是**往下找**（`from_pretrained` 里下钻那一格），`sub_configs` 既是
**往下找**的表，也是**往下发**的名单（`_attn_implementation` 递归下发）。

### 实测数字

| 量 | 实测值 | 含义 |
|---|---|---|
| `configuration_utils.py` | 1566 行 | `wc -l` |
| 本文件定义的 `validate_*` | 4 个 | 实例上能看到 5 个（多的来自 mixin） |
| `to_dict()` 顶层键 | 22 | 全量 |
| `to_diff_dict()` 顶层键 | 11 | 只留差异 |
| `to_json_string` 字节 | 6024 → 4894 | `use_diff=False → True` |
| 落盘的 `config.json` | 4894 字节 / 226 行 | 即 diff 形态 |

## 与后续课的接口

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

## 验收点

- [x] 保真门禁：14 个引用块全部逐字来自 `configuration_utils.py`，且位置连续
- [x] 覆盖度门禁：1 个源文件被声明（`configuration_utils.py`，属覆盖域文件）
- [x] 参数门禁：无非法参数
- [x] 语法检查：`node --check` 通过，9 幕字段完整
- [x] 渲染门禁：无 JS 错误、无布局溢出、交互可用（两种分辨率）
- [ ] 自检 ①**能画出 config 从 json 到对象再到模型的路径**：对着第 1 幕的路径图，
      写出 `cached_file → json.loads → _decode_special_floats → base_config_key 下钻 →
      cls(**config_dict) → @strict → PreTrainedModel.__init__(config)`，并说出哪两步最容易被略过
- [ ] 自检 ②**能解释 strict 校验在什么时机触发**：构造完成那一刻（`init_with_validate` 调
      `cls.validate(self)`）与 `save_pretrained` 写盘前各一次；字段赋值走的是另一条
      `__setattr__` 通道，不会重跑 `validate_*`

**一句话总结**：

> 一份 `config.json` 的旅程是 **`cached_file` → `json.loads` → `_decode_special_floats` →
> （必要时 `base_config_key` 下钻）→ `cls(**config_dict)` → `@strict` 在构造末尾跑一遍 `validate_*`**；
> 回程是 **`__dict__` → `to_dict` → `to_diff_dict` → `_encode_special_floats` → `json.dumps`**，
> 而 `save_pretrained` 会在写盘前把 `validate_*` **再跑一遍**。
