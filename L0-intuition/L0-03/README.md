# L0-03 · 层类型排布：34 层 KDA + 11 层 MLA

> 层：**预备篇 · 直觉** ｜ 优先级：P0 ｜ 前置课：L0-01

## 学习目标

看完这一课，你应该能：

1. **给定 `num_hidden_layers` 写出 `layer_types`**（对应验收点 1）：
   规则是 `idx % 4 != 3 → "linear_attention"`，其余 `"indexed_attention"`；
   45 层 → 34 + 11，8 层 → 6 + 2，一般地 `MLA = n // 4`。
2. **解释 `shared` 索引器省掉了什么**（验收点 2）：
   索引器子模块不被实例化、索引器前向不跑、索引器自己的压缩键缓存不写入，
   直接复用"上一个 full 层"选好的 2048 个位置。
3. 说出 **3:1 而不是 1:1** 的成本结构：KDA 状态是常数大小，
   MLA 的 KV 随序列线性增长 —— 3:1 让"增长的那部分"只剩 11/45 的层。

## 覆盖的源文件（1 个）

| 文件 | 行数 | 引用块 |
|---|---|---|
| `models/glm5_next/configuration_glm5_next.py` | 321 | 12 |

> 本课还引用了两处**实测**：默认 config 的值（`.venv` 里实例化 `Glm5NextTextConfig()`）、
> 以及在一个缩小的孪生模型上验证 `shared` 的相邻性约束。这些实测写在 `source.md`
> 与 `lesson.js` 的 `text` 块/说明文字里，**不计入覆盖率** —— 覆盖域只统计 `src/transformers/**/*.py`。
> 另外为了解释"为什么必须整除"，本课以**散文**方式引用了 `modeling_glm5_next.py`
> 里索引器的池化实现（`select_k = index_topk // index_kpool`），**同样不计入覆盖率**。

## 场景（9 幕）

1. **45 格排班表** —— 一眼看出 3 层 KDA + 1 层 MLA 的周期，数出 34 : 11
2. **生成规律** —— 逐层推演 `idx % 4 != 3`，并看紧跟着的旧名兼容层
3. **★ 为什么是 3:1** —— 常数状态 vs 线性增长的两条曲线，交点 2048 token
4. **索引器排班的默认值** —— 45 层全 `full`，以及 `full` / `shared` 各做什么
5. **★ `shared` 省掉了什么** —— 生产者/消费者对照 + 实测的相邻性约束
6. **freq / offset 判据** —— `max(i − offset + 1, 0) % freq == 0` 与它的错配陷阱
7. **`index_topk_pattern`** —— F/S 逐字符映射，以及"长度就是表长"的坑
8. **★ 整除约束** —— 池化预算 2048 = 16 × 128，为什么不能整除就要在 config 阶段报错
9. **收束** —— 两张表 + 两个数字 + 练习 + 下一课指路

## 核心结论

### 1. 排班表的规则只有一行

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
        if self.layer_types is None:
            kda_layers = [idx for idx in range(self.num_hidden_layers) if idx % 4 != 3]
            self.layer_types = [
                "linear_attention" if layer_idx in kda_layers else "indexed_attention"
                for layer_idx in range(self.num_hidden_layers)
            ]
```

45 层 → `KDA 34 + MLA 11`；两类的并集恰好是全部 45 层（不存在第三种注意力）。

### 2. ★ 增长的那部分只剩 11/45

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
    linear_head_dim: int = 128
    linear_num_heads: int = 64
```

| 量 | 值 | 是否随 seq 增长 |
|---|---|---|
| KDA 每层递推状态 | `64 × 128 × 128 = 1,048,576` 个数 | 否（常数） |
| MLA 每 token 每层缓存 | `kv_lora_rank = 512` 个数 | 是（线性） |
| 交点 | `1,048,576 / 512 = 2,048` token | —— |
| 128K 时全 MLA 的账（假设） | `3,019,898,880` 个数 | 增长项 100% |
| 128K 时的实际账 | `738,197,504` + `35,651,584` 个数 | 增长项 11/45 ≈ 24.4% |

### 3. `shared` 的边界：必须紧跟在 full 层之后

<!-- src: models/glm5_next/configuration_glm5_next.py -->
```python
    indexer_types: list[str] | None = None
```

文档字符串写的是 `reuse the previous full layer's top-k selection` —— 实现里这份选择
只通过 `prev_topk_indices` 在**相邻层**之间传递，而 KDA 层会把它重置为 `None`。
默认 3:1 排班里两个 MLA 层相隔 4 层，所以默认 `indexer_types` 只能是 45 个 `full`
（`index_topk_freq` 默认 1，取模恒为 0）。

### 4. 可除性约束防的是"静默错配"

索引器挑池再展开：池数 = `index_topk // index_kpool`，而掩码宽度写死 `index_topk`。
实测 `index_topk=2050`（与 16 不能整除）在**构造 config 时**就抛
`ValueError: index_topk (2050) must be divisible by index_kpool (16).`

## 与后续课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| `mlp_layer_types` 为什么前 3 层稠密 | L0-02 |
| `head_dim = 0` / NoPE 怎么影响注意力形状 | L0-02 |
| 这 45 个字符串在主干里被谁消费 | L0-01（`causal_mask_mapping[...]`）、L4 |
| KDA 的递推状态怎么更新 | L4-02 ~ L4-05 |
| 索引器怎么选出 2048 个位置 | L4-06 |
| 两种层在 `past_key_values` 里各存什么 | L5-01 ~ L5-03 |
| 288 选 8 的 MoE 经济学 | L0-04 |

## 验收点

- [x] **给定 `num_hidden_layers` 能写出 `layer_types`** —— 规则 `idx % 4 != 3`；
      本课实测 45 → 34 + 11、8 → 6 + 2，第 2 幕逐层推演、第 9 幕练习要求写 12 层
- [x] **能解释 `shared` 索引器省掉了什么** —— 模块不实例化（实测 `indexer = None`）、
      前向不跑、索引器缓存不写入（实测 `indexer_keys = None`），
      直接复用上一个 full 层的 2048 个位置；并给出"必须相邻"的实测约束
- [x] 保真门禁：12 个引用块全部逐字来自 `configuration_glm5_next.py`
- [x] 覆盖度门禁：1 个源文件被声明（属覆盖域 231 个文件之一）
- [x] 参数门禁：无非法参数
- [x] 语法检查：`node --check` 通过，9 幕字段完整
- [x] 渲染门禁：无 JS 错误、无布局溢出、交互可用（两种分辨率）
- 自检：能对着第 1 幕的 45 格条带，报出任意一层是 KDA 还是 MLA，并说出它的
  `indexer_types` 取值有没有作用

**一句话总结**：

> 45 层 = 34 层 KDA（常数状态）+ 11 层 MLA（线性增长的 KV），由 `idx % 4 != 3` 一条模运算生成；
> `indexer_types` 决定谁跑索引器、谁复用上一层选好的 2048 个位置，
> 而 `index_topk % index_kpool == 0` 保证这份预算能被整数个池刚好切完。
