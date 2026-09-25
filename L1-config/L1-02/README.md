# L1-02 · Glm5NextTextConfig 逐字段精读

> 层：**L1 · 配置与分派** ｜ 优先级：P0 ｜ 前置课：`L1-01`
>
> 前置课 L1-01 讲的是 `PreTrainedConfig` 的公共契约；这一课回到 GLM-5 自己的 config 上，
> 把它的字段一个个判组。

## 学习目标

看完这一课，你应该能：

1. **把 59 个字段按 7 组分完类**（注意力 10 / MoE 12 / KDA 5 / DSA 6 / mHC 3 / 主干 13 / 基类 10），
   并说清分组判据是「改了这个字段，谁的行为跟着变」而不是行号；
2. **列出 KDA 相关的 5 个字段**：`layer_types` + 4 个 `linear_*`（验收点 2）；
3. **说出 `validate_architecture` 四道硬约束各自防的是什么错误**，
   并知道第 5 条 `qk_rope_head_dim > 0` 是语义断言、而第 4 条在当前 `@strict` 下走不到（验收点 1）。

## 覆盖的源文件（1 个）

| 文件 | 行数 | 引用块 |
|---|---|---|
| `models/glm5_next/configuration_glm5_next.py` | 321 | 12 |

> 本课的实测数字来自现场命令（`.venv/bin/python` 内省 `Glm5NextTextConfig()`、`/usr/bin/time`
> 量 import 成本、`dataclasses.fields()` 数列），命令与输出都写在 `source.md` 的 `text` 块里 ——
> **它们是输出而不是源文件，不计入覆盖率**。覆盖域只统计 `src/transformers/**/*.py`；
> 本课声明的就是上表这一个文件。
>
> 作业书写的是 322 行，实测 `wc -l` 是 **321**（文件末尾无换行），本课按 321 记。

## 场景（9 幕）

1. **59 个字段，7 个组** —— 59 = 49 本类声明 + 10 基类继承；七组逐行点亮
2. **三行类级声明** —— `model_type` / `keys_to_ignore_at_inference` / `attribute_map`，
   以及「字符串如何找到类」的链与惰性 import 的实测成本
3. **★ 声明顺序不是分组顺序** —— L109–L120 的 12 行里，MoE 被 MLA 劈成 3 + 4
4. **主干骨架** —— 4 个宽度 + 3 个深度/头数：`12288 = 3 × 4096`、`64 == 64` → 不是 GQA
5. **KDA 的 5 个字段** —— `layer_types` + 4 个 `linear_*`，而它们并不挨着（验收点 2）
6. **DSA 的 6 个字段** —— 2048 个位置 ÷ 16 = 128 个池；索引器自己是 32 头 × 128 维
7. **mHC 只有 3 个字段** —— 同一段里还借住了 2 个 MoE + 2 个 DSA 字段
8. **★ 四道硬约束 + 一道 NoPE 断言** —— 每条防什么错、实测报什么、哪一条走不到（验收点 1）
9. **收束** —— 七组总表 + 两道练习 + 下一课指路

## 核心结论

### 1. ★ 字段的书写顺序不是分组顺序

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
    n_shared_experts: int = 1
    n_routed_experts: int = 288
    routed_scaling_factor: float = 2.5
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 0
    v_head_dim: int = 256
    qk_nope_head_dim: int = 256
    n_group: int = 1
    topk_group: int = 1
    num_experts_per_tok: int = 8
    norm_topk_prob: bool = True
```

MoE(3) → MLA(5) → MoE(4)。`kv_lora_rank` 夹在 MoE 字段中间，但它与 MoE 无关。
判据只有一条：**改这个字段，谁的行为跟着变。**

### 2. ★ 字段表其实有两层

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
        # Convert dict to attributes (if given)
        if (linear_attn_dict := kwargs.get("linear_attn_config")) is not None:
            self.linear_head_dim = linear_attn_dict.get("head_dim", self.linear_head_dim)
```

`linear_attn_config` / `index_topk_pattern` / `index_topk_freq` / `index_skip_topk_offset` / `safe_gate`
**都不是声明字段**：没有类型、没有默认值、不参与任何校验。但「不是字段」≠「不存在」——
实测它们会被挂成普通属性，`to_dict()` 的键数从 **63** 涨到 **68**，
`to_json_string()` 里能直接搜到 `"index_topk_pattern": "FSSF"`。
所以它们**跟着 config 一起被保存，却永远不享受字段契约的保护** ——
实测 `index_topk_pattern="FSSF"` 会得到一个长度 4 的 `indexer_types`，而层数是 45，不报错。

### 3. 四道硬约束防的是「能构造成功、但前向必错」

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
        if self.index_kpool < 1:
            raise ValueError(f"index_kpool must be positive, got {self.index_kpool}.")

        if self.index_topk % self.index_kpool != 0:
            raise ValueError(f"index_topk ({self.index_topk}) must be divisible by index_kpool ({self.index_kpool}).")
```

这两条**顺序不能换**：第 2 条要对 `index_kpool` 取模，窗口为 0 时先炸的是 `ZeroDivisionError`，
用户看到的会是一句栈而不是「index_kpool must be positive」。

| # | 检查 | 防的是什么错 | 实测 |
|---|---|---|---|
| 1 | `num_attention_heads != num_key_value_heads` | 按 GQA 思路少配 KV 头（MLA 头数必须 1:1） | `num_key_value_heads=32` → `ValueError` |
| 2 | `index_kpool < 1` | 池化窗口为 0；必须排在取模之前 | `index_kpool=0` → `ValueError` |
| 3 | `index_topk % index_kpool != 0` | 2048 个位置切不整（池化是整池操作） | `index_topk=2049` → `ValueError` |
| 4 | `q_lora_rank is None` | DSA 层没有压缩 Q，低秩路径断掉 | **走不到**：`@strict` 字段校验先拦下 |
| 5 | `qk_rope_head_dim > 0` | 在 NoPE 架构上打开 RoPE（语义断言，非参数契约） | `qk_rope_head_dim=64` → `ValueError` |

### 4. 三条类级声明：身份 / 忽略键 / 别名

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
    model_type = "glm5_next_text"
    keys_to_ignore_at_inference = ["past_key_values"]
```

`model_type` 是字符串查找的键（`CONFIG_MAPPING` 实测是 `_LazyConfigMapping`，
取值时才 import 这个模块）；`keys_to_ignore_at_inference` 把每步都在变的 `past_key_values`
排除在配置对比之外。别名表 `attribute_map = {"num_local_experts": "n_routed_experts"}`
让旧 checkpoint 继续可加载 —— 实测写别名等于写正名，而 `to_dict()` 只写正名。

### 实测数字

| 量 | 实测值 | 含义 |
|---|---|---|
| `dataclasses.fields()` | 59 | 字段总数 |
| 本类声明 / 基类继承 | 49 / 10 | L101–L154 / `PreTrainedConfig` |
| KDA 字段 | 5 | `layer_types` + 4 个 `linear_*` |
| DSA 字段 | 6 | `index_topk` / `index_kpool` / `index_head_dim` / `index_n_heads` / `indexer_types` / `index_kpool_always_select_tail` |
| 池化算术 | 2048 ÷ 16 = 128 | `index_topk ÷ index_kpool` |
| `import transformers` | 0.81 s / 67 MB | 不含 torch；505 个 `modeling_*.py` 一个都不碰 |
| `import modeling_glm5_next` | 3.07 s / 390 MB | 连带 torch —— 惰性映射省下的就是这笔 |

## 与后续课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| `PreTrainedConfig` 的公共字段与序列化契约 | L1-01 |
| `validate_architecture` 之外的校验器 | L1-01 |
| `Glm5NextVisionConfig` 与顶层嵌套 | L1-03 |
| `_LazyConfigMapping` / AutoModel 怎么按字符串找类 | L1-05 |
| `layer_types` 的 34 : 11 与 `indexer_types` 的排班 | L0-03 |
| 288 选 8、`routed_scaling_factor = 2.5` | L0-04 |
| `hc_mult = 4` 与 20 次 Sinkhorn 迭代 | L0-05 / L3-06 |
| 4 个 `linear_*` 字段在 KDA 层里怎么用 | L4-02 ~ L4-05 |
| `index_topk = 2048` 个位置怎么选出来 | L4-06 |
| `base_model_tp_plan` / `base_model_ep_plan` 怎么变成真的切分 | L8 |

## 验收点

- [x] 保真门禁：12 个引用块全部逐字来自 `configuration_glm5_next.py`，且位置连续
- [x] 覆盖度门禁：1 个源文件被声明（属覆盖域 231 个文件之一）
- [x] 参数门禁：无非法参数
- [x] 语法检查：`node --check` 通过，9 幕字段完整
- [x] 渲染门禁：无 JS 错误、无布局溢出、交互可用（两种分辨率）
- [ ] **验收点 1**：能说出四条校验各自防的是什么错误 → 第 8 幕的表 + 第 9 幕练习 2
- [ ] **验收点 2**：能列出 KDA 相关的 5 个字段 → 第 5 幕 + 第 9 幕练习 1

**一句话总结**：

> 59 个字段 = 7 组：注意力 10 + MoE 12 + KDA 5 + DSA 6 + mHC 3 + 主干 13 + 基类 10；
> **书写顺序不是分组顺序**，字段表还有第二层（5 个不是字段的名字：没有类型与校验，
> 却会被序列化），而真正会拦住你的只有 4 道参数契约 + 1 道 NoPE 断言。
