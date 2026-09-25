# L1-07 · 动态模块、类型契约与权重名映射表

> 层：**第 1 层 · 配置与分派** ｜ 优先级：P1 ｜ 前置课：L1-06

## 学习目标

看完这一课，你应该能：

1. **说出动态模块的缓存与校验判据**：哈希算什么、存在哪、什么时候跳过重新执行（对应本课验收点）；
2. 解释 `trust_remote_code` 为什么不是一个布尔，以及 15 秒超时到底在保护什么；
3. 说清 `conversion_mapping` 的两张表怎么合流成 210 个 `model_type` 的映射，以及查找为什么"类名优先"。

## 覆盖的源文件（3 个）

| 文件 | 行数 | 引用块 |
|---|---|---|
| `dynamic_module_utils.py` | 849 | 8 |
| `_typing.py` | 185 | 2 |
| `conversion_mapping.py` | 2127 | 4 |

> 行数为 `wc -l` 实测（与作业书的 850 / 186 / 2128 差 1，是末尾换行的口径差异）。
> 本课的实测数字来自仓库自带 venv 里的 `transformers 5.18.0.dev0`，产出一个临时仓库
> `.tmp/dynmod/demo_repo`（**脚本与临时目录不计入覆盖率**，覆盖域只统计 `src/transformers/**`）。

## 场景（9 幕）

1. **三个字符串问题，三个文件** —— 全景流程线：`repo id` → 模块 → 类
2. **缓存目录** —— `init_hf_modules` 的三件事 + 一个只触发一次的 `invalidate_caches`
3. **依赖闭包** —— 两个正则 + 递归，算出哈希要喂哪些文件
4. **★ 校验的粒度** —— 哈希相等就复用 `sys.modules`，不等才 `exec`
5. **本地代码：目录名就是版本号** —— 本地源码哈希 vs 远端 commit hash
6. **★ 信任三态** —— 真值表 + 15 秒超时；`has_local_code` 从哪来
7. **类型契约** —— `Level` / `ExcInfo` / `DeviceMeshLike` 与 `Protocol`
8. **★ 权重名映射：97 + 118 = 210** —— 别名表与手写表合流，`glm5_next` 的 13 条变换
9. **收束** —— 三文件总表 + 查找顺序 + 练习

## 核心结论

### 1. ★ 校验的粒度是"依赖闭包的字节"

<!-- src: dynamic_module_utils.py -->
```python
        if getattr(module, "__transformers_module_hash__", "") != module_hash:
            module_spec.loader.exec_module(module)
            module.__transformers_module_hash__ = module_hash
```

哈希把**模块文件 + 它递归依赖的每个文件**的字节依次喂进同一个 sha256；
哈希挂在**模块对象**上（`__transformers_module_hash__`），所以进程退出就没了 ——
`sys.modules` 管进程内复用，磁盘上的 `transformers_modules/<name>/<16 位哈希>/` 管跨进程复用。

### 2. 本地代码的目录名 = 内容身份

<!-- src: dynamic_module_utils.py -->
```python
        if local_model_name:
            submodule = os.path.sep.join([local_model_name, local_source_files_hash])
        else:
            submodule = local_source_files_hash
```

远端走 commit hash，本地走 `_compute_local_source_files_hash`（16 位十六进制）——
两者语义一致：**内容变了，目录就变，旧目录不会串味**。实测路径：
`transformers_modules/demo_repo/a32f4f65faa60c4e/modeling_demo.py`。

### 3. ★ `trust_remote_code` 是三态，默认是 No

| `has_local_code` | 传入值 | 解析结果 |
|---|---|---|
| `True` | 任意 | `False`（**根本不问**） |
| `False` | `True` | `True` |
| `False` | `False` | `ValueError` |
| `False` | `None` | 提问，`signal.alarm(15)` 兜底 |

实测 `TIME_OUT_REMOTE_CODE = 15`。而 `has_local_code` 来自 AutoModel 的注册表
（`models/auto/auto_factory.py` 里 `has_local_code = type(config) in cls._model_mapping`）——
这就是 L1-05 / L1-06 与本课的接缝。

### 4. ★ 权重名映射：两张表 + 类名优先

<!-- src: conversion_mapping.py -->
```python
    conversions = get_checkpoint_conversion_mapping(class_name)
    if conversions is None and model_type:
        conversions = get_checkpoint_conversion_mapping(model_type)
```

实测：手写键 **118** + 别名表补齐 **92** = **210** 个 `model_type`；
`mapping["glm5_next"]` 有 **13** 条变换（10 条 `WeightRenaming` + 3 条 `WeightConverter`）；
未知 `model_type` 返回 `None`（不是空列表），调用方靠这个 `None` 决定要不要回退。

### 5. `_typing.py` 的"实测纠偏"

作业书的要点写的是"`_typing` 里的 `TensorType` 等别名"，但本版本的 `_typing.py` 里**没有**
`TensorType` —— 它定义在 `utils/generic.py`（`class TensorType(ExplicitEnum)`）。
这个文件的别名族是 `Level` / `ExcInfo` / `DeviceMeshLike`，
外加 5 个 `Protocol`（`TransformersLogger`、`GenerativePreTrainedModel`、`StringValuedEnumLike`、
`PeftConfigLike`、`WhisperGenerationConfigLike`）。**计划文档也会过时，以上游文件为准。**

## 与后续课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| `auto_map` 里的字符串怎么被 AutoModel 用起来 | L1-05、L1-06 |
| `has_local_code` 与 `_model_mapping` 的关系 | L1-05 |
| config 如何决定形状（本课只碰了"名字"） | L0-02、L1-02 |
| `WeightConverter` 的 `operations` 具体怎么做张量运算 | L7 |
| 量化器如何改写这条变换链 | L7 |
| mHC 的 `attn_hc` / `ffn_hc` 是什么 | L0-05、L3-06 |

## 验收点

- [x] **能解释动态模块的缓存与校验策略**：哈希 = 依赖闭包的字节（模块文件 + 递归相对导入），
      存在模块对象的 `__transformers_module_hash__` 上；相等就复用 `sys.modules` 里的类、不重新执行
- [x] 保真门禁：14 个引用块全部逐字来自 3 个源文件，且各自连续
- [x] 覆盖度：3 个文件都在覆盖域（231 个文件）内
- [x] 参数门禁：无非法参数引用
- [x] 语法检查：`node --check` 通过，9 幕字段完整
- [x] 渲染门禁：0 JS 错误、0 布局溢出、交互可用（1280x720 + 1920x1080）

**一句话总结**：

> 动态模块的"缓存与校验"只有一句话：**用依赖闭包的字节算哈希，哈希相同就复用 `sys.modules` 里的模块对象**；
> 而"字符串找到类"在本课里分成两段 —— 动态模块那段靠文件系统 + 哈希，权重名那段靠"两张表 + 类名优先"。
