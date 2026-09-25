<!-- glm5-coverage
dynamic_module_utils.py
_typing.py
conversion_mapping.py
-->

# L1-07 · 动态模块、类型契约与权重名映射表 — 源文件

**这一课只回答三个"字符串问题"**：
一个仓库名字符串怎么变成可导入的类；一个类型别名字符串怎么在几十个文件之间统一注解；
一个权重名字符串怎么对应到本实现的模块名。

视角：L0 与 L1 前六课都默认「代码已经在库里」。这一课补上这个前提 ——
当代码**不在**库里（`trust_remote_code` 那条路），transformers 如何把它搬进一个
**可缓存、可校验、有信任边界**的地方；顺带看清类型契约与权重名映射这两张"名字的语法"。

| 文件 | 行数 | 本课引用块 |
|---|---|---|
| `dynamic_module_utils.py` | 849 | 8 |
| `_typing.py` | 185 | 2 |
| `conversion_mapping.py` | 2127 | 4 |

> 行数是 `wc -l` 实测。作业书写的是 850 / 186 / 2128，差 1 是"文件末尾换行是否计入"的口径差异。

**本课用到的实测数字**（在仓库自带 venv 里跑 `transformers 5.18.0.dev0` 得到，不是估算）：

```text
HF_MODULES_CACHE                       = ~/.cache/huggingface/modules
TRANSFORMERS_DYNAMIC_MODULE_NAME       = "transformers_modules"
_sanitize_module_name("THUDM/glm-5")   = "THUDM/glm_hyphen_5"   并打一条 warning（不是合法标识符）
_sanitize_module_name("9lives")        = "_9lives"
modeling_glm5_next.py 的相对导入目标数   = 18
_compute_local_source_files_hash 的产物  = 16 位十六进制（本课实测目录名 a32f4f65faa60c4e）
__transformers_module_hash__            = 16 位十六进制（本课实测 52a224fd204265ae）
TIME_OUT_REMOTE_CODE                    = 15（秒）
_build_checkpoint_conversion_mapping() 手写键 = 118（101 条字典字面量 + 17 条 mapping["X"] = [...]）
_MODEL_TO_CONVERSION_PATTERN 别名条数    = 97（其中 5 条与手写键重名被跳过，补齐全新键 92 条）
_build_checkpoint_conversion_mapping() 最终键数 = 210  =  118 + 92
mapping["glm5_next"] 的变换条数          = 13；"glm5_next_text" 走别名，与它同一份 .copy()
```

> 上面这段是**示意性汇总**，用 `text` 块标出 —— 不是源文件的逐字引用，不参与保真校验。
> 下面每一个 `python` 块都是从源文件里**按行号直接抽出来**的连续片段，逐字未改。

---

## 一、★ 缓存与校验：决定"要不要重新执行"的是一次哈希

验收点要的那个东西（动态模块的缓存与校验策略）全部实现就在下面两个块里。
先看第一块：**哈希算的是什么**。

<!-- src: dynamic_module_utils.py -->
```python
    name = os.path.normpath(module_path)
    name = name.removesuffix(".py")
    name = name.replace(os.path.sep, ".")
    module_file: Path = Path(HF_MODULES_CACHE) / module_path
    with _HF_REMOTE_CODE_LOCK:
        if force_reload:
            sys.modules.pop(name, None)
            importlib.invalidate_caches()
        cached_module: ModuleType | None = sys.modules.get(name)
        module_spec = importlib.util.spec_from_file_location(name, location=module_file)

        # Hash the module file and all its relative imports to check if we need to reload it
        module_files: list[Path] = [module_file] + sorted(map(Path, get_relative_import_files(module_file)))
        module_hash: str = hashlib.sha256(b"".join(bytes(f) + f.read_bytes() for f in module_files)).hexdigest()
```
**读法**：

- 前三行把 `module_path`（缓存目录里的相对路径）压成一个**模块全名**：`normpath` 归一化分隔符、
  去掉 `.py` 后缀、把路径分隔符换成 `.`。名字决定 `sys.modules` 里的键，所以必须稳定且唯一。
- `_HF_REMOTE_CODE_LOCK` 是模块级全局锁。动态导入会写 `sys.modules`，两个线程同时 exec 同名模块会互相踩；
  锁的粒度是"一次 `get_class_in_module`"。
- 关键在最后两行：`module_files` = 模块文件 + **它所有相对导入的文件**（`get_relative_import_files`），
  然后把这堆文件的**字节**拼起来算一个 sha256。注意是"把所有字节喂给同一个哈希"，不是"逐个文件哈希再拼"——
  文件名不参与，**只有内容参与**。
- `force_reload=True` 那条分支是给 `force_download` 用的：先把旧模块从 `sys.modules` 里踢掉，
  再 `invalidate_caches()`，等于"这一轮一定重新 exec"。

再看第二块：**哈希怎么被用掉**。

<!-- src: dynamic_module_utils.py -->
```python
        module: ModuleType
        if cached_module is None:
            module = importlib.util.module_from_spec(module_spec)
            # insert it into sys.modules before any loading begins
            sys.modules[name] = module
        else:
            module = cached_module
        # reload in both cases, unless the module is already imported and the hash hits
        if getattr(module, "__transformers_module_hash__", "") != module_hash:
            module_spec.loader.exec_module(module)
            module.__transformers_module_hash__ = module_hash
        return getattr(module, class_name)
```
**读法**：

- 判定只有一行：`getattr(module, "__transformers_module_hash__", "") != module_hash`。
  相等就不 exec，直接 `getattr(module, class_name)` 把**内存里已有的类对象**交出去；不等才重新执行，并把新哈希记回模块。
- `__transformers_module_hash__` 挂在**模块对象**上，既不是磁盘标记也不是文件属性 —— 进程一退就没了。
  所以"缓存"其实是两层：`sys.modules` 管**进程内**复用，磁盘上的 `transformers_modules/...` 管**跨进程**复用。
- 第 303 行那句"insert it into sys.modules before any loading begins"是有意为之：
  模块里若有循环导入，靠的正是 `sys.modules` 里那个**半成品**对象，否则会无限递归。
- ★ 校验的粒度是"**整个依赖闭包的字节**"。任何一个被相对导入的文件改了内容，哈希就变，主模块跟着重新执行。
  这也解释了为什么下一节那个递归函数必须存在。

---

## 二、缓存落在哪里：一个合法的模块名，一个进 `sys.path` 的目录

动态模块最终住在 `HF_MODULES_CACHE` 下。先看仓库名怎么被加工成模块名。

<!-- src: dynamic_module_utils.py -->
```python
    new_name = name.replace(".", "_dot_").replace("-", "_hyphen_")
    if new_name and new_name[0].isdigit():
        new_name = f"_{new_name}"
    if keyword.iskeyword(new_name):
        logger.warning(
            f"The module name {new_name} (originally {name}) is a reserved keyword in Python. "
            "Please rename the original module to avoid import issues."
        )
    elif not new_name.isidentifier():
        logger.warning(
            f"The module name {new_name} (originally {name}) is not a valid Python identifier. "
            "Please rename the original module to avoid import issues."
        )
    return new_name
```
**读法**：

- `.` → `_dot_`、`-` → `_hyphen_`，而不是统一替换成 `_`。函数开头那三行注释把理由写死了：
  `_` 是模块名里极常见的分隔符，统一替换会**撞名**；而 `sys.modules` 是"先到先得"，
  撞名的第二个模块会被**静默地**复用成第一个 —— 这种 bug 极难查。
- 数字开头补一个 `_`：Python 标识符不能以数字开头。
- 两条 warning 而不是 raise：名字不合法只是**可能**出问题，不该直接拒绝加载。
  实测：`_sanitize_module_name("THUDM/glm-5")` 返回 `THUDM/glm_hyphen_5`，并打一条
  "is not a valid Python identifier" 的 warning（因为里面还有 `/`）。
- 一个细节：`get_cached_module_file` 里是**逐段**调用它的（对仓库名 `split("/")` 后 `map`），
  所以 `zai-org/GLM-5` 得到的是两段 `zai_hyphen_org/GLM_hyphen_5` —— 仍然是一条包路径，不是一个文件名。

名字有了，接下来要让这个目录能被 `import` 看见。

<!-- src: dynamic_module_utils.py -->
```python
    # This function has already been executed if HF_MODULES_CACHE already is in the Python path.
    if HF_MODULES_CACHE in sys.path:
        return

    sys.path.append(HF_MODULES_CACHE)
    os.makedirs(HF_MODULES_CACHE, exist_ok=True)
    init_path = Path(HF_MODULES_CACHE) / "__init__.py"
    if not init_path.exists():
        init_path.touch()
        importlib.invalidate_caches()
```
**读法**：

- `init_hf_modules` 只做三件事：把缓存目录 append 进 `sys.path`、确保目录存在、确保目录里有 `__init__.py`。
- 第一行的早退是关键："已经在 `sys.path` 里"就当做过。这个函数在一次加载里可能被调用很多次，早退是它的常态路径。
- 为什么必须有 `__init__.py`：`transformers_modules.<repo>.<hash>.modeling_x` 要能作为**常规包**被导入，
  而 `sys.path` 上的一格要成为包，就得有它。
- `importlib.invalidate_caches()` 只在**真的新建了文件**时才调。Python 的路径查找器会缓存目录内容，
  不通知它就可能出现"文件明明在，import 却说没有"。

---

## 三、依赖闭包：两个正则换来一个哈希的输入集合

上一节说了哈希要喂"整个依赖闭包"。闭包怎么算出来的？就是这两个正则。

<!-- src: dynamic_module_utils.py -->
```python
    with open(module_file, encoding="utf-8") as f:
        content = f.read()

    # Imports of the form `import .xxx`
    relative_imports = re.findall(r"^\s*import\s+\.(\S+)\s*$", content, flags=re.MULTILINE)
    # Imports of the form `from .xxx import yyy`
    relative_imports += re.findall(r"^\s*from\s+\.(\S+)\s+import", content, flags=re.MULTILINE)
    # Unique-ify
    return list(set(relative_imports))
```
**读法**：

- 只抓两种形态：`import .xxx` 与 `from .xxx import yyy`。`\S+` 会把 `..activations` 这样的多级相对导入
  也整串吞进名字里，交给后面拼路径。
- `list(set(...))`：既去重，也**放弃了顺序**。顺序在 `get_class_in_module` 里被 `sorted(...)` 补回来 ——
  哈希必须稳定，同一份代码两次要算出同样的值。
- 这是**正则**，不是 AST：`import` 字样出现在字符串或注释里也会被算进去。这里的取舍是"宁可多带一个不存在的路径",
  真正读字节的时候才会暴露。
- 实测：`get_relative_imports(".../models/glm5_next/modeling_glm5_next.py")` 返回 **18** 个目标。
- 实测的反面（很重要）：把**库内**文件喂给递归版 `get_relative_import_files` 会抛
  `FileNotFoundError: .../models/glm5_next/..activations.py`。因为 `..activations` 是**包内相对导入**，
  不是"同目录的兄弟文件"。这说明这条代码路径**只为自包含的远端仓库设计**：
  远端仓库里 `from .configuration_x import X` 对应的确实是同目录文件。
  这是本课实测出来的边界，不是从注释里读来的。

---

## 四、本地路径：目录名就是"这份代码的版本号"

远端仓库用 commit hash 做目录名；本地目录没有 commit，于是 transformers 自己算一个哈希。

<!-- src: dynamic_module_utils.py -->
```python
    # Check we have all the requirements in our environment
    modules_needed = check_imports(resolved_module_file)
    if is_local:
        local_model_name = _sanitize_module_name(os.path.basename(os.path.normpath(pretrained_model_name_or_path)))
        local_source_files_hash = _compute_local_source_files_hash(pretrained_model_name_or_path, resolved_module_file)
        if local_model_name:
            submodule = os.path.sep.join([local_model_name, local_source_files_hash])
        else:
            submodule = local_source_files_hash

    # Now we move the module inside our cached dynamic modules.
    full_submodule = TRANSFORMERS_DYNAMIC_MODULE_NAME + os.path.sep + submodule
    create_dynamic_module(full_submodule)
    submodule_path = Path(HF_MODULES_CACHE) / full_submodule
```
**读法**：

- `check_imports` 先跑：它用 AST 收集**绝对**导入的顶层包名，逐个 `importlib.import_module` 试，
  缺包直接抛 `ImportError`。注意它返回的却是 `get_relative_imports(filename)` —— 函数名和返回值看着不搭，
  这是历史包袱，调用方要的是"相对导入清单"。
- 本地走 `local_model_name + local_source_files_hash` **两级目录**：一级是人类可读的模型目录名，
  一级是"源码内容哈希"（`_compute_local_source_files_hash`，16 位十六进制）。远端分支用的是 **commit hash** 那一级 ——
  两者语义完全一致：**拿内容的身份当目录名**。
- 所以 `transformers_modules/<name>/<hash16>/modeling_x.py` 这条路径本身就是一句断言："这份代码，就是这个版本。"
- 实测：把两个文件放进临时目录当成本地模型，`get_cached_module_file` 落在
  `transformers_modules/demo_repo/a32f4f65faa60c4e/modeling_demo.py`，
  拿到的类是 `transformers_modules.demo_repo.a32f4f65faa60c4e.modeling_demo.DemoModel`。
- 后半段（本课没引）是**文件级**判重：`filecmp.cmp` 比"缓存里的副本"和"源文件"，一样就不复制。
  第 453 行的注释已经把理由写了 —— 只在"文件是新的、或自上次拷贝以来改过"时才拷。
  目录级哈希决定"进不进这个版本"，文件级判重决定"这个文件还要不要重写"。

---

## 五、★ `trust_remote_code` 不是一个布尔，是三个出口

上面整条链都在"用户已经同意执行远端代码"之后。同意这件事本身，也有一段代码。

<!-- src: dynamic_module_utils.py -->
```python
    if trust_remote_code is None:
        if has_local_code:
            trust_remote_code = False
        elif has_remote_code and TIME_OUT_REMOTE_CODE > 0:
            prev_sig_handler = None
            try:
                prev_sig_handler = signal.signal(signal.SIGALRM, _raise_timeout_error)
                signal.alarm(TIME_OUT_REMOTE_CODE)
                while trust_remote_code is None:
                    answer = input(
                        f"{error_message} You can inspect the repository content at https://hf.co/{model_name}.\n"
                        f"You can avoid this prompt in future by passing the argument `trust_remote_code=True`.\n\n"
                        f"Do you wish to run the custom code? [y/N] "
                    )
                    if answer.lower() in ["yes", "y", "1"]:
                        trust_remote_code = True
                    elif answer.lower() in ["no", "n", "0", ""]:
                        trust_remote_code = False
                signal.alarm(0)
```
**读法**：

- 三态判定就是前三行：**本地有实现 → 直接 `False`（根本不问）**；远端有代码且超时值大于 0 → 走提问；
  两边都不是 → 交给下一块的 `elif` 抛错。
- `signal.alarm(TIME_OUT_REMOTE_CODE)` 是"提问的硬上限"：用户不回答，15 秒后 `_raise_timeout_error`
  把局面变成一句明确的报错，而不是永久挂住。实测 `TIME_OUT_REMOTE_CODE = 15`。
- `while trust_remote_code is None` 配 `input()`：**空回答**（直接回车）落进下面第二个 `elif` 的 `""`，
  等于拒绝。默认值是 **No**，不是 Yes。
- 提问文本里带了仓库地址和"下次怎么免打扰"（`trust_remote_code=True`）。这不是客套：
  `error_message` 的设计意图是让用户**先去看代码再决定**。
- 为什么用 `signal.SIGALRM` 而不是线程或超时库：提问是阻塞的 `input()`，只有信号能打断它。
  代价是 Windows 上没有这个信号 —— 下面 `except Exception` 就是为这类平台留的出口。

失败的两个出口：

<!-- src: dynamic_module_utils.py -->
```python
        elif has_remote_code:
            # For the CI which puts the timeout at 0
            _raise_timeout_error(None, None)

    if has_remote_code and not has_local_code and not trust_remote_code:
        raise ValueError(
            f"{error_message} You can inspect the repository content at https://hf.co/{model_name}.\n"
            f"Please pass the argument `trust_remote_code=True` to allow custom code to be run."
        )

    return trust_remote_code
```
**读法**：

- 第 783-785 行是 **CI 出口**：超时值被设成 0 时不允许"边跑边问"，直接报错。
- 第 787-791 行是**最后一道闸**："有远端代码、本地没实现、用户没同意"三者同时成立就抛。
  它和前面的关系是"或"不是"与"—— 任何一条路径没把值变成真，最后都会撞在这里。
- 返回的是**解析后的值**，不是原样返回：`False` 和 `None` 对调用方的含义完全不同。
- `has_local_code` 这个参数不是猜的，它来自 AutoModel 的注册表：`models/auto/auto_factory.py` 里写着
  `has_local_code = type(config) in cls._model_mapping`。
  也就是说：**"这个字符串能不能在本库里找到类"，决定了要不要走整套远端代码流程。**
  这正是 L1-05 / L1-06 与本课的接缝。
- 实测（非交互环境）：`resolve_trust_remote_code(None, "some/model", False, True)` 会先把提问打印出来，
  然后 `input()` 抛异常，被兜底成 `ValueError: ... Please pass the argument trust_remote_code=True ...`。
  CI 里"卡在提问"的真实形态就是这个。

---

## 六、`_typing.py`：别名统一注解，Protocol 只给类型检查器看

换一个文件。这个文件 185 行，没有一个运行期对象 —— 它全部的价值都在"写法"上。

<!-- src: _typing.py -->
```python
# A few helpful type aliases
Level: TypeAlias = int
ExcInfo: TypeAlias = (
    None
    | bool
    | BaseException
    | tuple[type[BaseException], BaseException, object]  # traceback is `types.TracebackType`, but keep generic here
)
DeviceMeshLike: TypeAlias = Any  # PyTorch stubs do not model torch.distributed.device_mesh consistently yet.
```
**读法**：

- 三个别名，三种"绕开类型噪音"的姿势：
  - `Level: TypeAlias = int` —— `logging` 的等级本来就是 int，但裸写 `int` 会丢掉语义，起个名字让签名自解释；
  - `ExcInfo` —— 这个参数合法的形态确实有四种（`None` / `bool` / 异常实例 / 三元组），联合类型把它写全。
    第 34 行注释解释了为什么第三项写成 `object`：`types.TracebackType` 在类型检查器里不好用，**宁可宽一点**；
  - `DeviceMeshLike: TypeAlias = Any` —— 第 36 行注释给了理由：PyTorch 的 stub 对
    `torch.distributed.device_mesh` 的建模还不稳定。与其跟着上游抖动，不如显式退化成 `Any`，**并用注释留下为什么**。
- 共同点：三个别名在运行期都不产生对象（`TypeAlias` 只是注解），它们的价值是让几百个文件里的签名写法一致。
- **实测纠偏**：作业书的要点写的是"`_typing` 里的 `TensorType` 等别名"，但本版本的 `_typing.py` 里**没有**
  `TensorType` —— 它定义在 `utils/generic.py`（`class TensorType(ExplicitEnum)`），由 `tokenization_utils_base.py`
  等文件使用。"等别名"在这个文件里指的就是上面这三个。计划文档也会过时，以上游文件为准。

同文件里体量最大的是这个 Protocol。

<!-- src: _typing.py -->
```python
class GenerativePreTrainedModel(Protocol):
    """Protocol for the model interface that GenerationMixin expects.

    GenerationMixin is designed to be mixed into PreTrainedModel subclasses. This Protocol documents the
    attributes and methods the mixin relies on from its host class. It is *not* used at runtime — its
    purpose is to help the ``ty`` type checker resolve ``self.<attr>`` accesses inside the mixin.
    """

    config: Any  # PretrainedConfig — kept as Any to avoid circular imports
    device: torch.device
    dtype: torch.dtype
    main_input_name: str
    base_model_prefix: str
    _is_stateful: bool
    hf_quantizer: Any
    encoder: Any
    hf_device_map: dict[str, Any]
    _previous_max_cache_length: int

    generation_config: Any  # GenerationConfig

    def __getattr__(self, name: str) -> Any: ...
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def can_generate(self) -> bool: ...
    def get_encoder(self) -> Any: ...
    def get_output_embeddings(self) -> Any: ...
    def get_input_embeddings(self) -> Any: ...
    def set_output_embeddings(self, value: Any) -> None: ...
    def set_input_embeddings(self, value: Any) -> None: ...
    def get_compiled_call(self, compile_config: Any) -> Any: ...
    def set_experts_implementation(self, *args: Any, **kwargs: Any) -> Any: ...
    def _supports_logits_to_keep(self) -> bool: ...
```
**读法**：

- `GenerativePreTrainedModel` **不是基类，是 `Protocol`**：它只描述"`GenerationMixin` 眼里的宿主类长什么样"。
  文档字符串把意图写明了 —— **运行期完全不用，只为类型检查器解析 `self.<attr>`**。
- 属性区先列"它会读什么"：`config`、`device`、`dtype`、`main_input_name`、`base_model_prefix`、
  `hf_quantizer`、`hf_device_map`、`generation_config`。`config: Any` 后面那句注释解释了为什么是 `Any`：
  循环导入。
- 方法区列"它会调什么"：`forward` / `__call__` / `can_generate` / `get_input_embeddings` /
  `set_output_embeddings` / `_supports_logits_to_keep` …… 这是混入类对宿主的**最小接口集**。
- 为什么值得单独讲：这是"鸭子类型"被写成合同的样子。运行时一行都不执行，
  但一旦 `GenerationMixin` 里访问了合同之外的属性，类型检查器就会报错。
- 同文件另外三个 Protocol 分工相同：`StringValuedEnumLike`（只要求 `.value: str`）、
  `PeftConfigLike`（PEFT 适配器要读的字段 + `save_pretrained`）、
  `WhisperGenerationConfigLike`（Whisper 特有字段）。四个字：**结构化契约**。

---

## 七、★ 权重名映射：97 条别名 + 118 条手写 = 210 个 model_type

第三个文件回答"磁盘上的名字怎么变成内存里的名字"。它的组织方式是**两张表**。

第一张表在文件最上面，只有 15 行。

<!-- src: conversion_mapping.py -->
```python
_MODEL_TO_CONVERSION_PATTERN = {
    # Mixtral-style MoE
    "minimax": "mixtral",
    "minimax_m2": "mixtral",
    # Qwen2-style MoE
    "afmoe": "qwen2_moe",
    "deepseek_v2": "qwen2_moe",
    "deepseek_v3": "qwen2_moe",
    "deepseek_v32": "qwen2_moe",
    "dots1": "qwen2_moe",
    "ernie4_5_moe": "qwen2_moe",
    "glm4_moe": "qwen2_moe",
    "glm4_moe_lite": "qwen2_moe",
    "glm_moe_dsa": "qwen2_moe",
    "glm4v_moe": "qwen2_moe",
```
**读法**：

- 这是一张**别名表**：`model_type → 另一条已经写好的映射`。左边是"长得像谁"，右边是"抄谁的"。
- 为什么需要它：同一类权重名差异（比如 `w1/w2/w3` 的专家布局）被几十个模型共享。
  没有这张表，下面那个几千行的构造函数里要写几十份一模一样的列表。
- 实测：这张表 **97** 条。第 105 行那条 `"glm5_next_text": "glm5_next"` 说明 GLM-5 的文本主干
  在权重名上与 `glm5_next` 同源。
- 注意 key 混着两套命名空间：小写的 `model_type` 字符串，和大写开头的**类名**（`"AriaModel"`、`"ViTModel"`）。
  记住这一点，本节最后一段的查找顺序会用到。

第二张表是手写的真身。GLM-5 自己的那份在这里。

<!-- src: conversion_mapping.py -->
```python
        "glm5_next": [
            WeightRenaming(
                source_patterns=r"self_attn\.f_a_proj\.",
                target_patterns=r"self_attn.forget_gate.f_a_proj.",
            ),
            WeightRenaming(
                source_patterns=r"self_attn\.f_b_proj\.",
                target_patterns=r"self_attn.forget_gate.f_b_proj.",
            ),
            WeightRenaming(
                source_patterns=r"self_attn\.dt_bias",
                target_patterns=r"self_attn.forget_gate.dt_bias",
            ),
            WeightRenaming(
                source_patterns=r"self_attn\.A_log",
                target_patterns=r"self_attn.forget_gate.A_log",
            ),
            WeightRenaming(source_patterns="hc_attn_fn", target_patterns="attn_hc.fn"),
            WeightRenaming(source_patterns="hc_attn_base", target_patterns="attn_hc.base"),
            WeightRenaming(source_patterns="hc_attn_scale", target_patterns="attn_hc.scale"),
            WeightRenaming(source_patterns="hc_ffn_fn", target_patterns="ffn_hc.fn"),
            WeightRenaming(source_patterns="hc_ffn_base", target_patterns="ffn_hc.base"),
            WeightRenaming(source_patterns="hc_ffn_scale", target_patterns="ffn_hc.scale"),
```
**读法**：这是 `mapping["glm5_next"]` 的开头，一共 **13** 条变换，逐条看它说了三件事：

1. **改名**：`self_attn.f_a_proj.` → `self_attn.forget_gate.f_a_proj.`。磁盘上 KDA 的"遗忘门"参数
   是直挂在 attention 上的，本实现把它收进 `forget_gate` 子模块。`hc_attn_fn` → `attn_hc.fn` 同理
   —— 那正是 L0-05 讲的 mHC。
2. **合并**：`mlp.experts.*.gate_proj.weight` + `up_proj.weight` → 一个 `gate_up_proj`，
   `operations=[MergeModulelist(dim=0), Concatenate(dim=1)]`：先把每层专家列表叠成一个张量，
   再在 dim=1 上拼 gate/up。**顺序不能反**。
3. **堆叠**：`q_conv1d/k_conv1d/v_conv1d` 合成一个 `conv1d`（`Concatenate(dim=0)`）——
   KDA 的短卷积三个权重在磁盘上分开、在内存里连续，一次 kernel 调用就够。

- `WeightRenaming` 是"只改名"，`WeightConverter` 是"改名 + 张量操作"。前者不需要读数据，后者必须把张量搬进内存再动手。
- `source_patterns` 是**正则**：`\.` 是转义过的路径分隔符，`.*` 匹配专家序号。
  所以这张表同时也是"名字空间的正则语法"—— 写错一个反斜杠就会静默匹配不上，不报错。

第二张表怎么和第一张表合流：

<!-- src: conversion_mapping.py -->
```python
    for model_type, base_pattern in _MODEL_TO_CONVERSION_PATTERN.items():
        if model_type in mapping:
            continue
        mapping[model_type] = mapping[base_pattern].copy()

    return mapping


_checkpoint_conversion_mapping_cache = None


def get_checkpoint_conversion_mapping(model_type):
    global _checkpoint_conversion_mapping_cache
    if _checkpoint_conversion_mapping_cache is None:
        _checkpoint_conversion_mapping_cache = _build_checkpoint_conversion_mapping()
    return deepcopy(_checkpoint_conversion_mapping_cache.get(model_type))
```
**读法**：

- 循环只有 4 行：别名表里的每个 `model_type`，只要没被手写过，就 `mapping[base_pattern].copy()` 一份。
- `.copy()` 是**浅拷贝**。这里够用：值是一个 list，copy 之后两个 `model_type` 拿到的是**两个不同的 list 对象**，
  往里 append 不会互相污染。但**元素是共享的** —— 所以里面的变换对象必须当成只读。
  （这一点在第九幕的练习里会再问一次。）
- 实测：手写键 **118**（101 条字典字面量 + 17 条 `mapping["X"] = [...]`）、别名表 **97** 条
  （其中 5 条与手写键重名，被 `continue` 跳过）、最终 **210** 个键；118 + 92 = 210。
- `get_checkpoint_conversion_mapping` 是"读"的入口：用模块级 `None` 当哨兵，第一次调用时才构建，之后走缓存。
  返回前 `deepcopy` —— 因为调用方会给每个变换设置 `scope_prefix`（见下一块），共享一份可变对象会让子模块的作用域互相覆盖。
  **缓存的代价，是可变对象必须深拷贝出去。**
- 实测：未知 `model_type` 返回 `None` 而不是空列表。这个区别有意义 —— 下一块正是用 `is None` 判断要不要回退。

最后一块：拿到一个模型对象时，两张表怎么被查。

<!-- src: conversion_mapping.py -->
```python
    class_name = type(model).__name__
    model_type = model.config.model_type

    # Class name takes priority — allows ForXxx-specific overrides
    conversions = get_checkpoint_conversion_mapping(class_name)
    if conversions is None and model_type:
        conversions = get_checkpoint_conversion_mapping(model_type)
    return conversions
```
**读法**：

- 查找只有两条路：先按**类名**（`type(model).__name__`）查，查到就用；查不到再按 `model.config.model_type` 查。
- 为什么要两条：`model_type` 是"架构族"的粒度（`glm5_next`），类名是"任务头"的粒度
  （`Glm5NextForConditionalGeneration`）。同一架构的不同任务头可能有不同的权重名布局，类名优先让它们各自覆盖。
- 返回 `None` 的语义是"我这儿没有专门规则"，调用方据此决定"还要不要再按 model_type 试一次"。
  要是返回 `[]`，就会被当成"有规则、只是空的"，第二条路直接断掉。
- 本课只讲**查表**。真正拿这些变换去改键名（以及量化器如何在链上插一脚）在 L7 那一层 ——
  这一课的三张表是那一层的输入。

---

## 八、与前后课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| `auto_map` 里的字符串怎么被 AutoModel 用起来 | L1-05、L1-06 |
| `has_local_code` 与 `_model_mapping` 的关系 | L1-05 |
| config 如何决定形状（本课只碰了"名字"） | L0-02、L1-02 |
| `WeightConverter` 的 `operations` 具体怎么做张量运算 | L7 |
| 量化器如何改写这条变换链 | L7 |
| mHC 的 `attn_hc` / `ffn_hc` 是什么 | L0-05、L3-06 |

**一句话总结**：

> 动态模块的"缓存与校验"只有一句话：**用依赖闭包的字节算哈希，哈希相同就复用 `sys.modules` 里的模块对象**；
> 而"字符串找到类"在本课里分成两段 —— 动态模块那段靠文件系统 + 哈希，权重名那段靠"两张表 + 类名优先"。
