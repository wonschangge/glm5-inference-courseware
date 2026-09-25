# L1-04 · 配置的序列化、继承与文档生成

> 层：**第 1 层 · 配置与分派** ｜ 优先级：P1 ｜ 前置课：`L1-03`

## 学习目标

看完这一课，你应该能：

1. **说出 `@auto_docstring` 在类定义时改了什么** —— 它把 `cls.__doc__` 整份换成了
   一段新字符串，类对象本身没有变（对应验收点 1）；
2. **解释为什么需要 `hp_naming` 这层间接** —— 短名字里丢掉的类型信息，
   必须靠 `DEFAULTS` 补回来，而双向映射表是补回来的唯一依据（验收点 2）；
3. 说出 `type_validators` 的校验发生在**哪个时刻**，以及为什么不能在 forward 里。

## 覆盖的源文件（4 个）

| 文件 | 行数 | 引用块 |
|---|---|---|
| `utils/hp_naming.py` | 163 | 5 |
| `utils/doc.py` | 1092 | 2 |
| `utils/auto_docstring.py` | 4800 | 8 |
| `utils/type_validators.py` | 271 | 3 |

> 合计 6326 行，18 个引用块。行数取 `_data/universe.json` 的口径（含文件末尾换行），
> 比 `wc -l` 多 1。
>
> 本课引用的**实测输出**（写在 `source.md` 的 `text` 块里）来自
> `/data/WORKSPACE/transformer-project/.venv` 里的 `transformers` 与
> `tests/trainer/test_trainer_hyperparameter.py` 的用法复现。
> **这些是脚本输出，不计入覆盖率** —— 覆盖域只统计 `src/transformers/**/*.py`。

## 场景（9 幕）

1. **四个文件，一个共同动作** —— 全景：谁在定义期改元数据、谁在调用期改值
2. **★ 名字是一张双向映射表** —— `PREFIX` / `DEFAULTS` / `NAMING_INFO` 三分工
3. **短名生成：两级回退** —— 词内缩前缀 + 词间换分隔符
4. **值的字符串化与类型丢失** —— `hp_b-3` 里的减号到底是分隔符还是负号
5. **doc.py：改的是 `__doc__`** —— 装饰器返回同一个 `fn`，两个装饰器容错态度相反
6. **缩进也被算术化** —— `equalize_indent`（做加法）与 `set_min_indent`（做减法）
7. **参数区解析** —— 一条正则四个捕获组；`*optional*` 是数据结构，不是注释
8. **类名后缀 → 模板 → `{占位符}`** —— 字符串查类的正式答案
9. **收束** —— 一个判据贯穿全课：名字决定分支；＋ 练习与下一课指路

## 核心结论

### 1. ★ `@auto_docstring` 改的是 `cls.__doc__`，而且只改这一个属性

<!-- src: utils/auto_docstring.py -->
```python
    # 5) Assign the dynamically generated docstring to the wrapper class
    cls.__doc__ = docstring
    return cls
```

`return cls` 说明它**返回同一个类对象**：不产生子类、不包装方法，
`__mro__` 与 `isinstance` 全部不受影响。实测 `Glm5NextConfig.__doc__` 是
**1901 个字符**，而源码里只有一小段自定义参数说明 —— 其余全部是从
`__init__` 签名、类型注解和类名后缀推出来的。

### 2. ★ 名字决定分支，这是全课同一个判据

<!-- src: utils/auto_docstring.py -->
```python
    def auto_docstring_decorator(obj):
        if len(obj.__qualname__.split(".")) > 1:
            return auto_method_docstring(
                obj, custom_args=custom_args, custom_intro=custom_intro, checkpoint=checkpoint
            )
        else:
            return auto_class_docstring(obj, custom_args=custom_args, custom_intro=custom_intro, checkpoint=checkpoint)
```

同一个「数 `__qualname__` 里的点」的判据，在本课出现三次：
决定装饰器走类还是方法（上面）、决定缩进层数（`get_indent_level`）、
以及决定取值域（`get_model_name` 用文件路径推断模型名）。

### 3. ★ 名字会丢信息，所以校验必须放在构造期

<!-- src: utils/hp_naming.py -->
```python
            sep = "" if isinstance(v, (int, float)) else "-"
            e = f"{key}{sep}{v}"
            name.append(e)

        return "_".join(name)
```

`sep` 只有两种取值，于是名字变成一套只认「数字 / 非数字」的语法。
实测：`hp_a4` 反解回来是 `4.0`（int 变 float），`hp_b-3` 反解回来是 `'3'`
（字符串，负号被当成分隔符吃掉），而单独解析 `hp_b-3` 直接
`ValueError: could not convert string to float: ''`。

**这不是缺陷报告，是边界声明**：它服务的是「人类可读的 trial 目录名」，
不是通用序列化。真正的类型保障在另一条路上：

<!-- src: utils/type_validators.py -->
```python
    @as_validated_field
    def _inner(value: int | float):
        min_valid = min <= value if not exclude_min else min < value
        max_valid = value <= max if not exclude_max else value < max
        if not (min_valid and max_valid):
            raise ValueError(error_message.format(value=value))

    return _inner
```

`interval(min=0.0, max=1.0)(default=0.02)` 把校验器挂进 dataclass 字段的
`metadata["validator"]`。实测 `LlamaConfig(initializer_range=2.0)` →
`StrictDataclassFieldValidationError`，**在构造对象时就炸**，
而不是等到 forward。

### 本课用到的实测数字

| 量 | 实测值 | 来源 |
|---|---|---|
| `Glm5NextConfig.__doc__` 长度 | 1901 字符 | `.venv` 里 import 后读 `__doc__` |
| `import MODEL_MAPPING` 新增模块 | 0 | `sys.modules` 前后差集 |
| `len(MODEL_MAPPING)` 新增模块 | 2536（共 556 个映射） | 同上 |
| `shortname({'a': 0, 'b': 0})` | `'hp'` | 复现 `test_trainer_hyperparameter.py` 的用法 |
| `learning_rate → lr` / `learning_runtime → lru` | 两级回退生效 | 同上 |
| `parse_repr('hp_a4_b-3')` | `ValueError` | 同上 |

## 与后续课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| `MODEL_MAPPING` / `MODEL_MAPPING_NAMES` 怎么注册 | L1-05 |
| 其余四张自动注册表（tokenizer / processor / image / video） | L1-06 |
| `to_dict()` / `to_json_string()` 的保类型序列化 | L1-01 |
| `strict` dataclass 的完整校验时机 | L1-01 |
| `Glm5NextTextConfig` 每个字段的取值范围 | L1-02 |
| 顶层嵌套配置的寻址（`sub_configs`） | L1-03 |
| `dynamic_module_utils` 与权重名映射表 | L1-07 |

## 验收点

- [x] 保真门禁：18 个引用块全部逐字来自 4 个源文件，且位置连续
- [x] 覆盖度门禁：4 个源文件被声明（均属覆盖域 231 个文件）
- [x] 参数门禁：无非法参数
- [x] 语法检查：`node --check` 通过，9 幕字段完整
- [x] 渲染门禁：无 JS 错误、无布局溢出、交互可用（两种分辨率）
- [x] **能说出 `auto_docstring` 在类定义时改了什么** —— 见「核心结论 1」：
      只改 `cls.__doc__`，返回同一个 `cls`
- [x] **能解释为什么需要 `hp_naming` 这层间接** —— 见「核心结论 3」：
      名字是给人看的压缩形式，反解必须依赖 `DEFAULTS` 与双向映射表；
      它换来的是**可读的目录名**，代价是类型信息丢失

**一句话总结**：

> 这三套机制的共同点是**名字即契约**：`__qualname__` 决定缩进与分支、
> 类名后缀决定文档模板、参数名决定短名与反查键；
> 而字符串一旦被写进名字（`hp_b-3`）或文档（`{model_class}`），
> 类型与「这个类当时是否存在」这两条信息就可能**永久丢失** ——
> 所以 `type_validators` 才必须把校验放在**构造期**，
> 用值本身兜住注解兜不住的东西。
