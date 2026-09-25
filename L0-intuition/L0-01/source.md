<!-- glm5-coverage
models/glm5_next/modeling_glm5_next.py
-->

# L0-01 · 一次 `generate()` 里发生了什么 — 源文件

**这一课不解释任何优化。它只做一件事：把 45 层的模型压缩成一张能记住的图。**

视角：一个 token 从进入模型到吐出下一个 token，GLM-5 在 transformers 里到底发生了什么。

| 文件 | 行数 | 本课引用块 |
|---|---|---|
| `models/glm5_next/modeling_glm5_next.py` | 2444 | 6 |

**实测的模型规模**（用 `_data/recon/probe1.py` 与 `probe2.py` 在一台 CPU 机器上跑出来的）：

```text
Glm5NextTextConfig() 默认值
    num_hidden_layers  = 45
    layer_types        : 34 层 linear_attention (KDA) + 11 层 indexed_attention (MLA+DSA)
    mlp_layer_types    :  3 层 dense            + 42 层 sparse (MoE)
    hidden_size        = 4096,  num_attention_heads = 64
    n_routed_experts   = 288,   num_experts_per_tok = 8,  n_shared_experts = 1
    hc_mult            = 4      (mHC 残差流条数)
```

> 上面这段是**示意性的汇总**，用 `text` 块标出 —— 它不是源文件的逐字引用，
> 因此不参与保真校验。下面每一段 `python` 块都是逐字引用。

---

## 一、入口：`Glm5NextTextModel.forward` 的签名

先看函数收什么。注意 `input_ids` 与 `inputs_embeds` 是二选一的：
多模态场景下，视觉 token 已经在外面被替换成了向量，所以走 `inputs_embeds` 这条路。

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
```

**读法**：
- `past_key_values: Cache` 不是 tuple —— 这是 transformers v5 的 Cache 对象（L5 整层讲它）。
- 返回类型 `MoeModelOutputWithPast` 里带 `past_key_values`，说明**这一层函数自己会更新缓存**。
- 没有 `use_rope` 之类的开关；位置信息在这套架构里换了一条路（见下）。

---

## 二、六阶段的第一步到第三步

这一段是整条前向里信息密度最高的 8 行：校验 → 建缓存 → 嵌入。

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
```

**读法**：
- 第一行的异或 `^` 是"恰好一个"的惯用写法：两个都给或两个都不给都会抛错。
- 缓存是**在这里**创建的，不是在外面。所以第一次前向（prefill）和后续前向（decode）
  走的是同一个入口，区别只在传不传 `past_key_values`。
- `self.embed_tokens` 是 `nn.Embedding(vocab_size, hidden_size, padding_idx)`，
  形状 `(batch, seq)` → `(batch, seq, 4096)`。

---

## 三、位置编号：`position_ids` 从哪来

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        if position_ids is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
            position_ids = position_ids.unsqueeze(0)
```

**读法**：
- `past_seen` 就是"已经缓存了多少个 token"。decode 时 `inputs_embeds.shape[1] == 1`，
  于是 `position_ids` 就是 `[[past_seen]]` —— 一个数。**这就是"第几个 token"的唯一来源。**
- 注意这里算出来的位置编号**只用于索引**（哪些位置可见），
  它不会变成 RoPE 的旋转角 —— 因为 DSA 层是 NoPE。

---

## 四、★ 掩码要按层类型分别取

这是本课最值得记住的一处：**同一个隐藏状态，在 34 层和 11 层上看到的掩码不是同一个对象。**

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
            causal_mask_mapping = {
                "indexed_attention": attention_mask,
                "linear_attention": attention_mask,
            }
```

**读法**：
- 表面上两个 key 指向同一个张量。**但真到用的时候，两种层会用完全不同的方式消费它** ——
  KDA 层只需要知道"哪些位置是 padding"，MLA+DSA 层还要在它之上叠加一个"只允许看我选中的
  2048 个位置"的稀疏掩码。
- 这个字典的存在本身就是架构的化石证据：**如果只有一种注意力，就不需要这张表。**
- 详细展开在 L4（混合注意力）与 L5-05（掩码构造）。

---

## 五、★ 隐藏状态从一开始就是 4 维的

整份文件里最容易被略过、但影响最大的一行：

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
```

**读法**：
- `(batch, seq, 4096)` → `(batch, seq, 4, 4096)`。
- 多出来的 `4` 就是 **mHC 的 4 条残差流**（`hc_mult=4`）。
- 这意味着：**这个模型里根本没有"一条残差"这回事。** 你在教科书上看到的
  `x = x + attn(norm(x))` 在这里是"4 条流各自混合后再写回 4 条流"。
- 为什么值得单独拎出来讲：因为它是后面所有形状推理的前提。看到 `hidden_states` 的
  形状是 4 维而不是 3 维，你才不会在后面某一课里被 mHC 的 `post` / `comb` 搞晕。
- 展开在 L3-06。

---

## 六、45 层的循环与出口

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        topk_indices = None
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states, topk_indices = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                # Key change using NoPE
                position_embeddings=None,
                input_ids=input_ids,
                past_key_values=past_key_values,
                prev_topk_indices=topk_indices,
                **kwargs,
            )

        hidden_states = self.norm(self.hc_head(hidden_states))
```

**读法**：
- `causal_mask_mapping[self.config.layer_types[i]]`：**第 i 层用哪种掩码，是由 config 里的
  字符串决定的**。这一行是 L0-02 / L0-03 与主干的连接点。
- `topk_indices` 在两个相邻的 MLA 层之间传递 —— 这是 DSA 的 `shared` 模式：
  上一层选好的 2048 个位置，这一层直接复用，省掉一次索引器前向。
- `position_embeddings=None` 旁边那句注释 `# Key change using NoPE` 是架构作者自己留的记号：
  **这是相对上一代模型最关键的改动**。
- 出口是 `norm(hc_head(hidden_states))`：`hc_head` 先把 4 条流收成 1 条，再归一化。
  顺序不能反 —— 归一化是逐流的，收拢必须在前。

---

## 七、一层之内：注意力与 MLP 都是"二选一"

`Glm5NextTextDecoderLayer.__init__` 的两处三元表达式，是"混合架构"这个词的具体含义。

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        self.block_type = config.layer_types[layer_idx]
        self.hidden_size = config.hidden_size
        self.self_attn = (
            Glm5NextTextLinearAttention(config, layer_idx)
            if self.block_type == "linear_attention"
            else Glm5NextTextAttention(config, layer_idx)
        )

        self.mlp = (
            Glm5NextTextMoE(config) if config.mlp_layer_types[layer_idx] == "sparse" else Glm5NextTextMLP(config)
        )
```

**读法**：
- 注意力二选一：`linear_attention` → KDA；否则 → MLA+DSA。
- MLP 二选一：`sparse` → MoE；否则 → 稠密 MLP。
- 两个选择**互相独立**：前 3 层是"KDA 或 MLA" × "稠密"，第 4 层是"MLA × 稀疏"。
- 所以"GLM-5 是 MoE 模型"和"GLM-5 是线性注意力模型"这两句话都不完整 ——
  它是**按层排班**的。

---

## 八、一层之内的执行顺序

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        residual = hidden_states
        post, comb, hidden_states = self.attn_hc(hidden_states)
        # Self attn
        hidden_states = self.input_layernorm(hidden_states)
        topk_indices = None
        if self.block_type == "linear_attention":
```

**读法**：
- 顺序是 **mHC 混合 → 归一化 → 注意力**，而不是教科书里的 归一化 → 注意力 → 加残差。
- mHC 返回三个东西：`post`（写回权重）、`comb`（流间组合矩阵）、`hidden_states`（混合后的流）。
- `residual` 被单独留了一份 —— 它在后面和 `comb` 一起参与写回，见下一段。

---

## 九、写回：mHC 的残差合并

<!-- src: models/glm5_next/modeling_glm5_next.py -->
```python
        hidden_states = post.to(dtype).unsqueeze(-1) * hidden_states.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual
        )

        residual = hidden_states
        post, comb, hidden_states = self.ffn_hc(hidden_states)
        # Feed forward
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
```

**读法**：
- 合并公式读作：`新流 = post ⊙ 子层输出 + combᵀ · 旧流`。
- `post` 逐元素门控"这一层的输出写多少"，`comb` 决定"4 条旧流怎么混合"。
- **同一段代码出现两次**：一次在注意力后，一次在 FFN 后。这就是 L0-05 要展开的
  "mHC 在每个子层前后各出现一次"。

---

## 十、把 6 个阶段连起来

```text
  input_ids (batch, seq)
        │  embed_tokens
        ▼
  inputs_embeds (batch, seq, 4096)
        │  unsqueeze(2).expand  ← mHC 展开成 4 条流
        ▼
  hidden_states (batch, seq, 4, 4096)
        │
        │  for i in range(45):
        │      attn  = KDA(i)  if layer_types[i]=="linear_attention"
        │              MLA(i)  otherwise
        │      mlp   = MoE(i)  if mlp_layer_types[i]=="sparse"
        │              MLP(i)  otherwise
        │      hidden_states = mHC_residual(sub_layer(norm(hidden_states)))
        ▼
  hidden_states (batch, seq, 4, 4096)
        │  hc_head   ← 4 条流收成 1 条
        │  norm
        ▼
  last_hidden_state (batch, seq, 4096)
        │  lm_head   (在 Glm5NextForConditionalGeneration 里)
        ▼
  logits (batch, seq, 154880)
        │  采样
        ▼
  next_token ──┐
               └──► 追加到 input_ids，回到最上面（这一次 past_key_values 不为空）
```

---

## 十一、与后面课的接口

| 本课留下的问题 | 在哪一课回答 |
|---|---|
| `layer_types` 的 34:11 是怎么算出来的 | L0-03 |
| `mlp_layer_types` 为什么前 3 层是稠密 | L0-02 |
| mHC 的 `post` / `comb` 具体怎么算出来 | L3-06 |
| KDA 层内部做了什么 | L4-02 ~ L4-05 |
| DSA 的 2048 个位置怎么选出来 | L4-06 |
| `DynamicCache` 到底存了什么 | L5-01 ~ L5-03 |
| `lm_head` 之后到 `next_token` 之间发生了什么 | L6-01 ~ L6-03 |
| 多模态时 `inputs_embeds` 是怎么拼出来的 | L2-06, L2-07 |

**一句话总结**：

> 一次前向 = 嵌入 →（展开成 4 条流）→ 45 次"按 config 排班"的子层 → 收拢成 1 条流 → 归一化；
> **decode 与 prefill 走的是同一个函数**，区别只在 `past_key_values` 是不是空的。
