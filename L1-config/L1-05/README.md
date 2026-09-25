# L1-05 · AutoModel 分派：字符串如何找到类

> 层：**第 1 层 · 配置与分派** ｜ 优先级：P0 ｜ 前置课：L1-04

## 学习目标

看完这一课，你应该能：

1. **画出从 `config.json` 的 `model_type` 字符串到 config 类的完整调用链**（对应验收点 1）；
2. **说清惰性映射省下的是哪一笔钱**，以及它不省哪一笔（验收点 2）；
3. 指出这条链上**唯一一次真正的 `import`** 发生在哪一行，以及 `keys()` 与 `values()` 为什么价格不同。

## 覆盖的源文件（4 个）

| 文件 | 行数 | 引用块 |
|---|---|---|
| `models/auto/__init__.py` | 34 | 2 |
| `models/auto/configuration_auto.py` | 466 | 5 |
| `models/auto/auto_factory.py` | 707 | 4 |
| `models/auto/auto_mappings.py` | 1359 | 2 |

> 行数取自 `_data/universe.json`（比 `wc -l` 各多 1：文件末行没有换行符）。
> 本课还引用了 `_data/recon/probe_l105.py` 的**实测输出**（写在 `source.md` 的 `text` 块里）。
> **它是脚本，不计入覆盖率** —— 覆盖域只统计 `src/transformers/**/*.py`。

## 场景（9 幕）

1. **一个字符串，怎么变成一个类** —— 五跳路线图 + 三个关键数字（718 / 240 / 556）
2. **连 auto 包自己都是惰性的** —— `if TYPE_CHECKING` 的 8 行运行时从不执行，`_LazyModule` 换掉模块对象
3. **表里全是字符串** —— `CONFIG_MAPPING_NAMES` 的 718 条 `str → str`，以及 glm5_next 那三条
4. **类名 → 包名** —— `-` 换 `_`，`SPECIAL_MODEL_TYPE_TO_MODULE_NAME` 把三个 glm5 名字归一
5. **★ 「用时才导入」就在这 4 行里** —— `_LazyConfigMapping.__getitem__` 与它的三个字典
6. **字符串不是参数，是 config.json 里的字段** —— `from_pretrained` vs `for_model`，以及 mistral→ministral 的改写
7. **★ config 类 → 模型类：反查回字符串再走一遍** —— `_LazyAutoMapping` 与那条反向引用
8. **惰性省下了什么** —— 冷启动实测：+2 / +101 / +1000 个模块
9. **收束** —— `724 = 718 + 5 + 1`，表的边界与用户的注册口子 + 练习

## 核心结论

### 1. ★ 表里存的是名字，不是类

分派链上没有任何一个"注册表里的类"。`auto_mappings.py` 里那张表，键和值实测都是 `str`：

<!-- src: models/auto/auto_mappings.py -->
```python
        ("glm5_next", "Glm5NextConfig"),
        ("glm5_next_text", "Glm5NextTextConfig"),
        ("glm5_next_vision", "Glm5NextVisionConfig"),
```

所以"找到类"这件事必须由一段代码来完成 —— 而它只有 3 行：

<!-- src: models/auto/configuration_auto.py -->
```python
        module_name = model_type_to_module_name(key)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(f".{module_name}", "transformers.models")
```

`self._modules` 是包级缓存：`glm5_next_text` 与 `glm5_next_vision` 算出来的包名都是 `glm5_next`，
第二次访问直接命中缓存。

### 2. ★ 同一套手法用了第二次

`AutoModel` 的键是 **config 类**，所以 `_LazyAutoMapping` 先把类名反查成 `model_type`，
再复用同一套 `importlib` 手法：

<!-- src: models/auto/auto_factory.py -->
```python
        self._config_mapping = config_mapping
        self._reverse_config_mapping = {v: k for k, v in config_mapping.items()}
        self._model_mapping = model_mapping
        self._model_mapping._model_mapping = self
```

第 4 行那条反向引用是给自动生成 docstring 用的（`auto_class_update`）。
实测：`AutoModel._model_mapping[Glm5NextConfig]` → `Glm5NextModel`（单个类，不是 tuple）。

### 3. ★ 惰性省下的是启动成本（实测）

`_data/recon/probe_l105.py` 冷启动一次跑完：

| 操作 | 新增 `transformers.*` 模块 | 耗时 |
|---|---|---|
| `import transformers` | —（基线 240 个） | — |
| `len(AutoModel._model_mapping)` = 556 | **0** | 0.1 ms |
| `CONFIG_MAPPING.keys()` = 724 | **0** | 0.2 ms |
| `CONFIG_MAPPING['glm5_next']` | +2 | 6.8 ms |
| `AutoModel._model_mapping[Glm5NextConfig]` | +101 | 172.4 ms |
| `CONFIG_MAPPING.values()`（全都要） | +1000 | 1553 ms |

省下的不是"一次 import"，而是**把 500 多个模型包的导入成本从"任何一次 `import transformers`"
推迟到"用户真的点名某个 `model_type`"**。反过来，`values()` / `items()` / 自动生成 docstring
都是全量遍历，一样得把 700 多条全部导入 —— **惰性只推迟成本，不消除成本**。

### 4. `from_pretrained` 的第一个参数不是 `model_type`

`model_type` 是 `PreTrainedConfig.get_config_dict()` 从 `config.json` 里读出来的字段，
不是函数参数。想直接按 `model_type` 造配置，用 `AutoConfig.for_model("glm5_next")`
（实测返回 `Glm5NextConfig`，`text_config.hidden_size = 4096`、45 层）。

### 实测数字

| 量 | 实测值 | 含义 |
|---|---|---|
| `CONFIG_MAPPING_NAMES` | 718 → 723 条 | 自动生成，被 `configuration_auto.py` 就地改大 |
| `CONFIG_MAPPING.keys()` | 724 | 再加 `gpt-sw3`（历史包袱） |
| `len(AutoModel._model_mapping)` | 556 | config 类 → 模型类，零导入可得 |
| 冷访问一个 config 类 | +2 个模块 | 6.8 ms |
| 首次取模型类 | +101 个模块 | 172.4 ms（`modeling_glm5_next.py` 的真实体积） |
| 全量 `values()` | +1000 个模块 | 1553 ms，总数 1415 |

## 与后续课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| `PreTrainedConfig.from_dict` 拿到 dict 之后做了什么校验 | L1-01 |
| `Glm5NextConfig` 的每个字段（形状从哪来） | L1-02, L1-03 |
| `save_pretrained` / `to_dict` 怎么把 config 写回去 | L1-04 |
| 其余四张自动注册表（modeling / tokenization / image_processing / processor） | L1-06 |
| `auto_map` 与 `trust_remote_code`：远端代码怎么被拉进来 | L1-07 |
| 模型类拿到 config 之后怎么建权重 | L7-01, L7-03 |

## 验收点

- [x] **能画出 `AutoConfig.from_pretrained('glm5_next')` 的调用链** —— 第一幕给出五跳路线图，
      第六幕补上"字符串来自 `config.json` 而非参数"这一处修正，第九幕练习要求逐步复述
- [x] **能解释惰性映射省下了什么** —— 第八幕给出冷启动实测账（+2 / +101 / +1000 个模块），
      并指出 `keys()` 与 `values()` 的价格差
- [x] 保真门禁：13 个引用块全部逐字来自 4 个源文件，且位置连续
- [x] 覆盖度门禁：4 个源文件被声明（均属覆盖域 231 个文件）
- [x] 参数门禁：无非法参数
- [x] 语法检查：`node --check` 通过，9 幕字段完整
- [x] 渲染门禁：无 JS 错误、无布局溢出、交互可用（两种分辨率）
- 自检：能说出这条链上**唯一一次 import** 发生在哪一行

**一句话总结**：

> 分派链上没有任何一个"注册表里的类"：表里存的是 `str → str`，类只在被点名的那一刻，
> 由一处 `importlib.import_module` + `getattr` 从包里取出来。
> **惰性映射不消除导入成本，只把它从 `import transformers` 推迟到第一次点名。**
