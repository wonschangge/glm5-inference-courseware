<!-- glm5-coverage
utils/hp_naming.py
utils/doc.py
utils/auto_docstring.py
utils/type_validators.py
-->

# L1-04 · 配置的序列化、继承与文档生成 — 源文件

**这一课回答一个问题：一个字符串（类名、参数名、占位符）是怎么变成可执行事实的？**

视角：L1 前几课读的是「config 里有什么字段」。这一课翻到背面，看三套**元机制**——
名字怎么来、文档怎么来、校验怎么来。

| 文件 | 行数 | 本课引用块 |
|---|---|---|
| `utils/hp_naming.py` | 163 | 5 |
| `utils/doc.py` | 1092 | 2 |
| `utils/auto_docstring.py` | 4800 | 8 |
| `utils/type_validators.py` | 271 | 3 |

> 行数取 `_data/universe.json` 的口径（含文件末尾换行），比 `wc -l` 多 1。
> 下面每一个 `python` 块都是**逐字、连续**引用；教学用的汇总表与推演一律用
> `text` 块，不参与保真校验。

**实测的证据**（在 `/data/WORKSPACE/transformer-project/.venv` 里跑出来的，
不是读代码猜的）：

```text
$ python -c "from transformers import Glm5NextConfig; print(len(Glm5NextConfig.__doc__))"
1901
  原文里只有 r""" ... """ 那段「自定义参数」；其余 1901 个字符是 @auto_docstring 加的

$ python -c "... from transformers.models.auto.modeling_auto import MODEL_MAPPING"
import MODEL_MAPPING 后新增模块数 : 0
len(MODEL_MAPPING) 之后新增模块数   : 2536（共 556 个映射）
  「用时才 import」= 注册表本身只是两张字符串表；只有取某个 key 才 import 它的模块

$ python -c "class MyTrialShortNamer(TrialShortNamer): DEFAULTS = {'a':0,'b':0}; ..."
shortname({'a': 4})            -> 'hp_a4'
shortname({'a': 4, 'b': -3})   -> 'hp_a4_b-3'
shortname({'a': 0, 'b': 0})    -> 'hp'          ← 全默认值 = 空名字
parse_repr('hp_a4_b-3')        -> 抛 ValueError: could not convert string to float: ''
parse_repr('hp_a4_b3')         -> {'a': 4.0, 'b': '3'}   ← 3 回来变成了字符串
```

后面每讲一个机制，都会用它自己那一组实测输出作证。

---

## 一、★ 名字是一张**双向映射表**，不是一个字符串

先看 `TrialShortNamer` 的类属性。它只有四个名字，但已经决定了整个机制的形状：

<!-- src: utils/hp_naming.py -->
```python
class TrialShortNamer:
    PREFIX = "hp"
    DEFAULTS = {}
    NAMING_INFO = None

    @classmethod
    def set_defaults(cls, prefix, defaults):
        cls.PREFIX = prefix
        cls.DEFAULTS = defaults
        cls.build_naming_info()
```

**读法**：
- 三个类属性各管一件事：`PREFIX` 是名字的开头、`DEFAULTS` 是**参照系**、
  `NAMING_INFO` 是**算好的映射表**。
- 为什么需要 `DEFAULTS`？因为短名字里**只出现与默认值不同的参数**。
  实测：`shortname({'a': 0, 'b': 0})` → `'hp'` —— 一个参数都没写进去。
  所以没有默认值就**无法反解**：看到 `hp_lr0.5`，你不知道 `num_train_epochs` 是多少，
  只能回落到 `DEFAULTS` 里查。这解释了 `parse_repr` 最后那个补全循环。
- `NAMING_INFO = None` 是缓存哨兵。`build_naming_info` 开头就是
  `if cls.NAMING_INFO is not None: return` —— 所以**必须**由 `set_defaults` 触发一次，
  而 `set_defaults` 会**同时**写 `PREFIX` / `DEFAULTS` / 映射表，三者永远同代。

回顾 L1-02：`Glm5NextTextConfig` 的 `to_dict()` 输出里，字段名就是这里的
`param_name`；`hp_naming` 把它压成 `lr` 这样的短键，`parse_repr` 再翻回来。

---

## 二、短名的生成：先试「无分隔」，撞了才退到「有分隔」

`param_name` 按 `_` 拆成词，每个词各自缩到最短前缀，再拼起来：

<!-- src: utils/hp_naming.py -->
```python
    def shortname_for_key(info, param_name):
        words = param_name.split("_")

        shortname_parts = [TrialShortNamer.shortname_for_word(info, word) for word in words]

        # We try to create a separatorless short name, but if there is a collision we have to fallback
        # to a separated short name
        separators = ["", "_"]

        for separator in separators:
            shortname = separator.join(shortname_parts)
            if shortname not in info["reverse_short_param"]:
                info["short_param"][param_name] = shortname
                info["reverse_short_param"][shortname] = param_name
                return shortname

        return param_name
```

**读法**：
- 拆词是**必需的**，因为短前缀只在「词」这一级唯一：`learning_rate` → `learning`+`rate`。
- `separators = ["", "_"]` 是一次**两级回退**：先试 `lr`，被别人占了才试 `l_r`。
  这解释了实测里 `learning_rate → lr`、`learning_runtime → lru`：
  `rate` 先占到 `r`，`runtime` 只能退到 `ru`。
- 两个字典是**同时写**的（`short_param` 正查、`reverse_short_param` 反查）。
  少了任何一个，`shortname` 或 `parse_repr` 就会断链。
- 兜底的 `return param_name` 返回**原名**：宁可名字长，也不能返回一个没登记的短名 ——
  否则反查表里就多了一个永远查不到的键。

**这两级回退的实测结果**：

```text
DEFAULTS = {'learning_rate': 0.1, 'learning_runtime': 1, 'num_train_epochs': 3,
            'seed': 42, 'warmup_ratio': 0.0}

short_param        : {'learning_rate': 'lr', 'learning_runtime': 'lru',
                      'num_train_epochs': 'nte', 'seed': 's', 'warmup_ratio': 'wra'}
short_word         : {'learning': 'l', 'rate': 'r', 'runtime': 'ru', 'num': 'n',
                      'train': 't', 'epochs': 'e', 'seed': 's', 'warmup': 'w', 'ratio': 'ra'}
reverse_short_word : {'l': 'learning', 'r': 'rate', 'ru': 'runtime', 'n': 'num', ...}
  ← rate 抢走了 'r'，runtime 只能退到 'ru'：前缀冲突在这一层就被挡掉
```

---

## 三、★ 序列化时值退化成字符串，反序列化时类型回不来

正向的入口是 `build_naming_info`，它把 `DEFAULTS` 的每一个键登记进映射表：

<!-- src: utils/hp_naming.py -->
```python
        field_keys = list(cls.DEFAULTS.keys())

        for k in field_keys:
            cls.add_new_param_name(info, k)

        cls.NAMING_INFO = info
```

**读法**：
- 遍历的是 `DEFAULTS` 的键 —— 映射表里**只有默认值里出现过的参数**。
  新参数想被命名，必须先进 `DEFAULTS`；否则在 `shortname` 里会直接抛
  `You should provide a default value for the param name ...`。
- `add_new_param_name` 内部是「先算短名，再补登记」：
  它调用 `shortname_for_key`（后者已经写了正反两个字典），然后再写一遍，是**幂等**的。
- `cls.NAMING_INFO = info` 写在**最后**：表没建完就不发布。这就是为什么它必须是
  「一次建好、整代复用」的不可变对象。

值是怎么写进名字里的：

<!-- src: utils/hp_naming.py -->
```python
            key = cls.NAMING_INFO["short_param"][k]

            if isinstance(v, bool):
                v = 1 if v else 0

            sep = "" if isinstance(v, (int, float)) else "-"
            e = f"{key}{sep}{v}"
            name.append(e)

        return "_".join(name)
```

**读法**：
- `bool` 先被**降级成 int**。这是必须的：`isinstance(True, int)` 在 Python 里为真，
  不先处理，`sep` 的判断就会把布尔当成数字、生成 `hp_flagTrue`。
- `sep` 是**类型判别器**：数字直接贴着键（`a4`），其余用 `-` 隔开（`b-3`）。
  所以名字**不只是一个字符串**，它是一个手写的、只认这两种语法的格式。
- `"_".join(name)` → `hp_a4_b-3`。分隔符与「负数里的减号」是同一个字符族，
  这就是下一节那个 bug 的根源。

反解时会发生什么（**实测**）：

<!-- src: utils/hp_naming.py -->
```python
        parameters = {}

        for value in values:
            if "-" in value:
                p_k, p_v = value.split("-")
            else:
                p_k = re.sub("[0-9.]", "", value)
                p_v = float(re.sub("[^0-9.]", "", value))

            key = cls.NAMING_INFO["reverse_short_param"][p_k]

            parameters[key] = p_v
```

**读法**：
- 两条分支看着对称，其实**不对称**：带 `-` 的走「原样字符串」，不带的走
  `re.sub` + `float`。于是同一个 `3`，写成 `hp_nte3` 回来是 `3.0`，
  写成 `hp_b-3` 回来是 `'3'`（字符串）—— **类型信息在名字里丢了**。
- 负数是真会炸的：`hp_a4_b-3` 先按 `_` 拆成 `['a4', 'b-3']`，
  `'b-3'.split('-')` → `['b', '3']`，`float('3')` = 3.0，但 `p_v = '3'`（字符串）。
  而 `'hp_b-3'` 单独出现时更糟：实测直接
  `ValueError: could not convert string to float: ''`。
- 结论：这套机制**只保证「自己写出去的名字自己能读回来」**，不保证数学性质。
  它服务的是「给一份 trial 起一个人类可读的目录名」，不是通用序列化格式。
- 回顾 L1-01：`to_dict()` / `to_json_string()` 是**有类型**的序列化（JSON 保类型）；
  这里是**无类型**的命名压缩。两者不要混为一谈。

---

## 四、`doc.py`：一次纯文本搬运，函数体一行都没被解释

`doc.py` 的四个装饰器是同一个动作：算出一段字符串，赋给 `fn.__doc__`。
`add_end_docstrings` 是最短的那一个：

<!-- src: utils/doc.py -->
```python
def add_end_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = (fn.__doc__ if fn.__doc__ is not None else "") + "".join(docstr)
        return fn

    return docstring_decorator
```

**读法**：
- 装饰器**返回的是同一个 `fn`**，只改了一个属性。函数体、签名、闭包全都没动 ——
  这是「机械替换」四个字的准确含义：它改的是**元数据**，不是行为。
- `fn.__doc__` 可能是 `None`（没写 docstring 的函数），所以要先兜底成 `""`。
  这个 `if ... is not None` 在 `doc.py` 里出现两次（`add_start_docstrings` 也有）。
- 为什么值得单独讲：这一层是**纯字符串层**，出错了不会有任何异常 ——
  只会让文档静默变形。所以 `doc.py` 里的三处 `raise ValueError` 都极具体，
  下面那段就是其中之一。

`replace_return_docstrings` 是「机械替换」里唯一**会拒绝执行**的：

<!-- src: utils/doc.py -->
```python
        lines = func_doc.split("\n")
        i = 0
        while i < len(lines) and re.search(r"^\s*Returns?:\s*$", lines[i]) is None:
            i += 1
        if i < len(lines):
            indent = len(_get_indent(lines[i]))
            lines[i] = _prepare_output_docstrings(output_type, config_class, min_indent=indent)
            func_doc = "\n".join(lines)
        else:
            raise ValueError(
                f"The function {fn} should have an empty 'Return:' or 'Returns:' in its docstring as placeholder, "
                f"current docstring is:\n{func_doc}"
            )
        fn.__doc__ = func_doc
        return fn
```

**读法**：
- 它先用正则找一行**只有** `Returns:` / `Return:` 的占位行。
  `^\s*Returns?:\s*$` 要求这一行被空行包围 —— 写在一句话中间不算占位。
- 找不到就 `raise`。**这是设计选择**：它宁可让 import 失败，也不肯静默产出一份
  没有返回值说明的文档。对比 `add_end_docstrings` 的「没有就兜底成空串」，
  两个装饰器对「缺输入」的态度是相反的。
- 找到占位行之后，`lines[i]` 被整行**替换**成 `_prepare_output_docstrings(...)` 的结果，
  `min_indent=indent` 保证新内容与原来那一行对齐 —— 缩进必须**算出来**，
  不能写死，因为同一个装饰器会用在类方法（8 空格）和普通函数（4 空格）上。

---

## 五、`auto_docstring.py`：缩进也被算术化了

`auto_docstring.py` 有 4800 行，其中大半是**参数字典**。
真正干活的是下面这组小函数。先看缩进是怎么被算出来的：

<!-- src: utils/auto_docstring.py -->
```python
def get_indent_level(func):
    # Use this instead of `inspect.getsource(func)` as getsource can be very slow
    return (len(func.__qualname__.split(".")) - 1) * 4


def equalize_indent(docstring: str, indent_level: int) -> str:
    """
    Adjust the indentation of a docstring to match the specified indent level.
    """
    prefix = " " * indent_level
    # Uses splitlines() (no keepends) to match previous behaviour that dropped
    # any trailing newline via the old splitlines() + "\n".join() + textwrap.indent path.
    return "\n".join(prefix + line.lstrip() if line.strip() else "" for line in docstring.splitlines())


def set_min_indent(docstring: str, indent_level: int) -> str:
    """
    Adjust the indentation of a docstring to match the specified indent level.
    """
    # Equivalent to textwrap.dedent + textwrap.indent but avoids the two regex
    # passes that textwrap uses internally (one per call in dedent, one in indent).
    lines = docstring.split("\n")
    min_indent = min(
        (len(line) - len(line.lstrip()) for line in lines if line.strip()),
        default=0,
    )
    prefix = " " * indent_level
    return "\n".join(prefix + line[min_indent:] if line.strip() else "" for line in lines)
```

**读法**：
- `get_indent_level` 不用 `inspect.getsource`，因为 `getsource` 慢（注释里写了原因）。
  它数的是 `__qualname__` 里的点：`Glm5NextConfig.__init__` 有两个点，减一乘四 = 4。
  **命名即位置**：这是「名字决定一切」在本课的第二个例子。
- 对比 `doc.py` 的 `get_docstring_indentation_level`：那个用 `inspect.isclass`
  加 `inspect.getsource` 逐字符数缩进；`auto_docstring.py` 直接算。
  两代实现并存，是这份代码库演进的化石。
- `equalize_indent` 与 `set_min_indent` **方向相反、结果相同**：
  前者把每行**砍到零再统一加**（`lstrip` 然后 `prefix +`），适合「完全自己生成的文本」；
  后者先量出全局最小缩进再整体平移（`line[min_indent:]`），适合「用户写的、
  但不知道他缩了几格」的文本。用错一个，多行 description 就会歪。
- 两个函数都用 `if line.strip() else ""` 处理空行 —— **空行保持空**，
  否则文档里会多出一堆只有空格的「假空行」。

缩进是「加法」，解析是「减法」。参数区的解析靠一条正则：

<!-- src: utils/auto_docstring.py -->
```python
        if max_indent_level == 0:
            param_pattern = _re_param
        else:
            param_pattern = re.compile(
                # |--- Group 1 ---|| Group 2 ||- Group 3 -||---------- Group 4 ----------|
                rf"^\s{{0,{max_indent_level}}}(\w+)\s*\(\s*([^, \)]*)(\s*.*?)\s*\)\s*:\s*((?:(?!\n^\s{{0,{max_indent_level}}}\w+\s*\().)*)",
                re.DOTALL | re.MULTILINE,
            )
        for match in param_pattern.finditer(args_section):
            param_name = match.group(1)
            param_type = match.group(2)
            additional_info = match.group(3)
            optional = "optional" in additional_info
            shape = parse_shape(additional_info)
            default = parse_default(additional_info)
            param_description = match.group(4).strip()
            # indent the first line of param_description to 4 spaces:
```

**读法**：
- 四个捕获组各管一件事：名字、类型、附加信息（`*optional*, defaults to ...`）、描述。
  类型用 `([^, \)]*)` —— **不允许**逗号和空格，所以 `Union[dict, PreTrainedConfig]`
  会在这里被截断。这是真实存在的局限，不是笔误。
- 第四组用了一个**负向先行断言** `(?!\n^\s{0,N}\w+\s*\()`：
  「一直吃，直到下一行又是一个参数名」。这是「用正则切段落」的经典写法，
  也是 `DOTALL` 必须开的原因 —— 描述里可以有空行。
- `optional` / `shape` / `default` 都不是独立数据，而是**从描述文字里再抠出来的**：
  `optional = "optional" in additional_info`，`parse_shape` / `parse_default` 各查一次正则。
  也就是说：**文档的语法就是数据结构**。写错一个 `*optional*`，下游的
  `return_tensors` 判断就会变。
- `max_indent_level == 0` 时走预编译的 `_re_param`。这是热路径优化：
  `parse_docstring` 会被每个类调用几十次。

然后是「名字 → 类」这一步中最关键的一次跳转：

<!-- src: utils/auto_docstring.py -->
```python
class ClassDocstring:
    Config = r"""
    This is the configuration class to store the configuration of a {model_base_class}. It is used to instantiate a {model_name}
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the [{model_checkpoint}](https://huggingface.co/{model_checkpoint})

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.
    """
```

**读法**：
- `ClassDocstring` 的每个属性名（`Config` / `PreTrainedModel` / `ForCausalLM` / ...）
  都是**类名的后缀**。`auto_class_docstring` 里是这么选的：
  `re.findall(rf"({'|'.join(ClassDocstring.__dict__.keys())})$", cls.__name__)` ——
  取出 `cls.__name__` 的**结尾**去匹配属性名。
- 这就是「字符串如何找到类」的正式答案：不是查一张手写的表，
  而是**用类名自己的后缀去命中一个文档模板**。`Glm5NextConfig` 以 `Config` 结尾，
  于是拿到配置类的开场白；`Glm5NextForCausalLM` 以 `ForCausalLM` 结尾，拿到另一段。
- 模板里的 `{model_name}` / `{model_base_class}` / `{model_checkpoint}` 还是**字符串占位符**，
  下一节才会被替换。

占位符是怎么被填上的：

<!-- src: utils/auto_docstring.py -->
```python
            except ImportError:
                # In case a library is not installed, we don't want to fail the docstring generation
                place_holder_value = None
            if place_holder_value is not None:
                if isinstance(place_holder_value, list | tuple):
                    place_holder_value = (
                        place_holder_value[-1] if place_holder_value[-1] is not None else place_holder_value[0]
                    )
                placeholders_dict[placeholder] = place_holder_value if place_holder_value is not None else placeholder
            else:
                placeholders_dict[placeholder] = placeholder
```

**读法**：
- `PLACEHOLDER_TO_AUTO_MODULE` 把 `{model_class}` 这样的占位符映射到
  `("modeling_auto", "MODEL_MAPPING_NAMES")`。**取值的动作就是查注册表** ——
  这正好接上 L1-05。
- 三条兜底路径，一条比一条宽：库没装 → `None`；查不到 → **保留占位符原文**
  （`placeholders_dict[placeholder] = placeholder`）。所以一份文档里如果出现
  字面量 `{model_class}`，那不是 bug，是「这个名字当时真的不存在」的诚实记录。
- 列表取值 `place_holder_value[-1] if ... is not None else place_holder_value[0]`
  是「多实现取最后一个」的约定：新架构注册在后面。
- 全程 try/except `ImportError`：**文档生成永远不能成为 import 失败的原因**。
  这与 `replace_return_docstrings` 的「找不到占位行就 raise」形成对照 ——
  一个容错、一个严格，因为前者发生在 import 期（一炸全炸），后者发生在开发期（早炸早改）。

替换动作本身只有四行，`{...}` 的两种命运都写在这里：

<!-- src: utils/auto_docstring.py -->
```python
def format_args_docstring(docstring: str, model_name: str) -> str:
    """
    Replaces placeholders such as {image_processor_class} in the docstring with the actual values,
    deducted from the model name and the auto modules.
    """
    # first check if there are any placeholders in the docstring, if not return it as is
    placeholders = set(_re_placeholders.findall(docstring))
    if not placeholders:
        return docstring
```

**读法**：
- 先用一条正则把**整份文档里出现过的**占位符名字收集成集合，再一次性查表。
  这是「先收集、后替换」而不是「边扫边换」—— 因为查表（`get_placeholders_dict`）
  要 import 各个 `auto` 模块，做一次比做 N 次便宜。
- `if not placeholders: return docstring` 是最重要的一行：**没有占位符就原样返回**。
  绝大多数文档经过这里时不做任何事 —— 这也解释了为什么这个函数的开销可以忽略。
- 替换用的是 `docstring.replace(f"{{{placeholder}}}", value)`（`{{` 是格式化转义）。
  如果 `value` 是 `None`（占位符没查到时就是这个），`str.replace` 会**抛 TypeError** ——
  所以上一段那个「查不到就赋成占位符自己」的兜底不是保守，是**必需**。

最后是入口本身。`@auto_docstring` 可以是 `@auto_docstring`，
也可以是 `@auto_docstring(...)` —— 这两种用法靠「第一个参数是不是对象」区分：

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

**读法**：
- 分支判据与 `get_indent_level` **完全同一个**：数 `__qualname__` 里的点。
  有类名在前 → 是方法，走 `auto_method_docstring`；只有裸名字 → 是类，
  走 `auto_class_docstring`。**位置再一次由名字决定。**
- 所以「`@auto_docstring` 在类定义时改了什么」的答案是：
  它把 `cls.__doc__` **整个换成了一段新字符串**。实测 `Glm5NextConfig.__doc__`
  是 1901 个字符，而源码里只有那一小段自定义参数说明。
  由于这段赋值发生在**类体执行完之后、名字绑定之前**，所以它对
  `help()`、Sphinx、IDE 提示全部生效，且**不需要任何运行期开销**。
- 回顾 L1-03：`Glm5NextConfig` 的 `image_token_id` 等字段，源码里带默认值，
  文档里的 `` `int` `` 和 `` defaults to 154854 `` 都是**从签名和注解推出来的**，
  不是人手写的第二遍。**一处定义，两处消费** —— 这就是 auto_docstring 存在的理由。

把零件装成成品的那三行：

<!-- src: utils/auto_docstring.py -->
```python
    # 3) Set the correct indentation
    docstring = set_min_indent(f"{pre_block}", indent_level) if len(pre_block) else ""
    if name != "PreTrainedModel" and "PreTrainedModel" in (x.__name__ for x in cls.__mro__):
        docstring += set_min_indent(f"{ClassDocstring.PreTrainedModel}", indent_level)

    # 4) Add the __init__ docstring if it's found, (e.g. for processing or modeling classes). If not add
    # args docstring for dataclass fields (e.g. config or model output classes)
    if docstring_init:
        docstring += set_min_indent(f"\n{docstring_init}", indent_level)
```

**读法**：
- 整份文档是**字符串拼出来的**，顺序就是阅读顺序：开场白（`pre_block`）→
  `PreTrainedModel` 通用说明 → `__init__` 的参数表（或 dataclass 字段表）。
  每一段都用 `set_min_indent(..., indent_level)` **重新对齐**，而不是预先写死缩进。
- 第二步是**条件追加**：只有当这个类真的继承自 `PreTrainedModel`、且它不是
  `PreTrainedModel` 本身时，才加那段通用说明。`(x.__name__ for x in cls.__mro__)`
  是一个生成器表达式，`in` 会短路 —— 找到就停。
- `name != "PreTrainedModel"` 这个额外条件防的是「基线类不要把说明加给自己」。
  这类「差一个就重复」的边界，在文档生成里到处都是。
- 注意 `elif is_dataclass:` 分支（下一行开始）：**配置类走的是另一条路** ——
  没有 `__init__` 文档时，改为遍历 `cls.__annotations__` 生成字段表。
  这就是 `Glm5NextConfig` 的 1901 字符里那段参数表的来源。

最后一步只有两行，但它回答了本课的验收点：

<!-- src: utils/auto_docstring.py -->
```python
    # 5) Assign the dynamically generated docstring to the wrapper class
    cls.__doc__ = docstring
    return cls
```

**读法**：
- `cls.__doc__ = docstring` 之后紧跟 `return cls` —— **同一个类对象**被返回。
  `@auto_docstring` 从不产生子类、也不包装任何方法，所以
  `Glm5NextConfig.__mro__` 与没有装饰器时**完全一致**，`isinstance` 判断不受影响。
- 赋值发生在**类体已经执行完**的时刻：`__init__`、注解、字段默认值都已就位，
  所以生成文档时能读到最完整的信息；又因为它早于「名字绑定到模块」，
  后续所有的 `help()` / Sphinx / IDE 看到的都是新文档。
- 反过来说：这份文档**只生成一次**，之后不再随代码变化。改了签名却忘了重新
  import，看到的还是旧文档 —— 这也是它被放在 import 期而不是懒加载的原因。

---

## 六、`type_validators.py`：类型注解 → 运行期校验

前五节的机制都发生在**定义期**（改名字、改文档）。这一节发生在**调用期**。

<!-- src: utils/type_validators.py -->
```python
def tensor_type_validator(value: str | TensorType | None = None):
    possible_names = ["pt", "np", "mlx"]
    if value is None:
        pass
    elif not isinstance(value, str) or value not in possible_names:
        raise ValueError(f"The tensor type should be one of {possible_names} but got tensor_type={value}")
```

**读法**：
- 它长得像一个普通函数，但**不是**靠被调用起作用的。它的身份由使用点决定 ——
  `processing_utils.py` 里写的是
  `return_tensors: Annotated[str | TensorType | None, tensor_type_validator()]`。
  `Annotated` 的第二个参数是**元数据**，`huggingface_hub` 的 `strict` dataclass
  会把它当成校验器取出来执行。
- 契约只有两条：**通过时返回 `None`，不通过时 `raise ValueError`**。
  注意它**不改值、不做转换** —— 校验和规范化是两件事。
- `possible_names = ["pt", "np", "mlx"]` 是**手写的真值集**，与
  `TensorType` 这个 `Literal` 类型**重复**。这就埋着一个漂移风险：
  两边不同步时，类型检查放行、运行期拒绝。
- 实测：`tensor_type_validator('pt')` → `None`；
  `tensor_type_validator('tf')` → `ValueError: The tensor type should be one of ['pt', 'np', 'mlx'] ...`。

那 `@as_validated_field` 这个装饰器到底做了什么？答案是**换掉了它的返回值**：

<!-- src: utils/type_validators.py -->
```python
@as_validated_field
def positive_int_field(value: int | None = None):
    """
    Same as `positive_int` but functions as a dataclass field rather than type annotation metadata.
    """
    if value is not None and (not isinstance(value, int) or not value > 0):
        raise ValueError(f"Value must be a positive integer, got {value}")
```

**读法**：
- 有装饰器时，`positive_int_field(default=1024)` 返回的**不是** `None`，
  而是一个 `dataclasses.Field`，其 `metadata` 里挂着原函数。
  实测：`positive_int_field(1)` → `Field(... default=1, metadata={'validator': [<function positive_int_field ...>]})`。
  没有装饰器时，调用它只会**当场校验一次那个默认值**，然后返回 `None` ——
  对参数完全没有约束力。
- 所以「`positive_int_field` 既有装饰器版本、又有 `positive_int` 无装饰器版本」
  不是重复：前者用于 `vocab_size: int = positive_int_field(default=131072)`，
  后者用于 `Annotated[int | None, positive_int()]`。两条路通向同一个校验函数。
- 实测的触发点是**构造配置对象**：

  ```text
  LlamaConfig(initializer_range=2.0)
    → StrictDataclassFieldValidationError: Validation error for field 'initializer_range'
      ValueError: Value must be greater or equal to 0.0 and smaller or equal to 1.0, got 2.0.
  ```

带参数的校验器只能写成**工厂**。`interval` 就是范本：

<!-- src: utils/type_validators.py -->
```python
    min = min if min is not None else float("-inf")
    max = max if max is not None else float("inf")

    @as_validated_field
    def _inner(value: int | float):
        min_valid = min <= value if not exclude_min else min < value
        max_valid = value <= max if not exclude_max else value < max
        if not (min_valid and max_valid):
            raise ValueError(error_message.format(value=value))

    return _inner
```

**读法**：
- `interval` 本身**不是校验器**：`interval(min=0.0, max=1.0)` 返回的是 `_inner`
  这个已经被 `@as_validated_field` 包过的函数（实测类型：`function`）。
  真正用在 config 里的写法是 `interval(min=0.0, max=1.0)(default=0.02)` ——
  **两次调用**：先绑定区间，再绑定默认值。
- 边界语义是**闭区间**，除非显式 `exclude_min` / `exclude_max`；
  缺省侧补 `float("-inf")` / `float("inf")`，所以 `interval(min=0)` 也能用。
  这四行是「开闭区间 + 缺省无穷」的最小实现。
- 错误消息在**闭包外**就拼好了（`error_message`），只留一个 `{value}` 到最后
  `.format()`。这样校验失败时不用再拼字符串 —— 在 `strict` dataclass 的
  构造路径上，这是有意义的。
- 实测：`LlamaConfig(initializer_range=-0.5)` 与 `=2.0` 都被拒，
  默认值 `0.02` 通过。**校验发生在构造期，不在 forward 里** ——
  所以错误在「模型还没建起来」的时候就暴露了。

---

## 七、把四个文件连起来

```text
  定义期 ──────────────────────────────────────────────────────────────
   类名 (__qualname__ / __name__)
        │  ① 后缀匹配 ClassDocstring.<后缀>      → 开场白模板
        │  ② 数点算缩进 get_indent_level()       → 对齐层数
        │  ③ 解析 __init__ 签名 + 参数区正则     → 参数表
        │  ④ 替换 {占位符}（查 auto 注册表）      → 真实类名
        ▼
   cls.__doc__ = <1901 字符>        ← @auto_docstring 唯一做的事

   字段名
        │  shortname_for_key: 拆词 → 缩前缀 → 试 ""/"_" → 登记双向表
        ▼
   'hp_lr0.5_nte5'  ←→  {'learning_rate': 0.5, 'num_train_epochs': 5.0, ...}

  调用期 ──────────────────────────────────────────────────────────────
   构造配置 / 调用 processor
        │  Annotated[T, validator()]  或  T = validator_field(default=...)
        ▼
   校验通过 → None ；不通过 → ValueError（在构造点炸，不在 forward 里炸）
```

---

## 八、与后面课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| `MODEL_MAPPING` 这张字符串表长什么样、怎么注册 | L1-05 |
| `{model_class}` 查的 `MODEL_MAPPING_NAMES` 是谁写的 | L1-05 |
| 其余四张自动注册表（tokenizer / processor / image / video） | L1-06 |
| `to_dict()` / `to_json_string()` 的类型保真序列化 | L1-01 |
| `Glm5NextTextConfig` 每个字段的取值范围 | L1-02 |
| `dynamic_module_utils` 与权重名映射表 | L1-07 |
| `strict` dataclass 的完整校验时机 | L1-01 |

**一句话总结**：

> 这三套机制的共同点是**名字即契约**：`__qualname__` 决定缩进与分支、
> 类名后缀决定文档模板、参数名决定短名与反查键；
> 而字符串一旦被写进名字（`hp_b-3`）或文档（`{model_class}`），
> 类型与「这个类当时是否存在」这两条信息就可能**永久丢失** ——
> 所以 `type_validators` 才必须把校验放在**构造期**，用值本身兜住注解兜不住的东西。
