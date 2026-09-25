# TODOLIST —— GLM-5 推理视角的 transformers 动画课件

> 本文件由 `tools/gen_plan.py` 从 `tools/plan.py` 自动生成，**不要手工编辑**。
> 计划与覆盖度矩阵共用同一份数据，因此不会出现「计划说 A、实现做 B」的漂移。

## 0. 视角与主线

**视角**：

> 一个 token 从进入模型到吐出下一个 token，GLM-5 在 transformers 里到底发生了什么。

**这个仓库在解决什么问题**（一句话）：

> 把「磁盘上的一堆权重 + 一段文本/图像」变成「一次可复现、可优化、可横向扩展的数值计算」。

**心智模型的转折点**：

- `L0–L4` 是**数学视角**：数据是张量，模型是算子组合 —— 问「算什么」。
- `L5` 起切换为**机器视角**：数据是字节，计算是访存 —— 问「算这个要花多少」。
- 同一个 KV cache，在 L4 是三行公式，在 L5 是一块按页分配、可量化、可换出的显存。

---

## 1. 覆盖域（universe）—— 实测冻结，非人工圈定

覆盖域不是「我觉得该讲的文件」，而是**跑出来的**：

```text
测量方法：_data/recon/probe5.py —— 在 CPU 上实例化一个微型 GLM-5，
          真实执行 text generate / 采样 / beam / 图像处理 + 视觉塔 /
          量化配置 / 全部量化器模块 / 分布式 / 导出器 / 连续批处理 /
          注意力后端集成，然后取 sys.modules 与初始快照的差集，
          映射回 src/transformers/**.py。

实测闭包           : 328 个文件
扣除非 GLM 模型族  : -97 个文件（36 个模型目录，仅因 models/auto 注册表被顺带导入）
最终覆盖域         : 231 个文件
```

**被显式排除的 97 个文件**（诚实声明：它们与 GLM-5 推理无关，本课件不覆盖、
也不计入覆盖率分子分母）：

| 模型目录 | 文件数 |
|---|---|
| `models/__init__.py` | 1 |
| `models/blt` | 3 |
| `models/clip` | 3 |
| `models/cohere_compass` | 3 |
| `models/data2vec` | 3 |
| `models/encoder_decoder` | 2 |
| `models/ernie4_5_vl_moe` | 3 |
| `models/exaone4_5` | 3 |
| `models/gemma` | 2 |
| `models/gpt2` | 2 |
| `models/llama` | 2 |
| `models/llama4` | 3 |
| `models/minicpmv4_6` | 3 |
| `models/minicpmv4_7` | 3 |
| `models/mra` | 3 |
| `models/muse_glimmer` | 3 |
| `models/nllb_moe` | 3 |
| `models/paddleocr_vl` | 3 |
| `models/phi3` | 3 |
| `models/phimoe` | 3 |
| `models/qwen2` | 2 |
| `models/qwen2_5_omni` | 3 |
| `models/qwen2_5_vl` | 3 |
| `models/qwen2_vl` | 3 |
| `models/qwen3_asr` | 3 |
| `models/qwen3_omni_moe` | 3 |
| `models/qwen3_vl` | 3 |
| `models/qwen3_vl_moe` | 3 |
| `models/rwkv` | 3 |
| `models/sam3` | 3 |
| `models/sam3_video` | 3 |
| `models/t5` | 2 |
| `models/video_llama_3` | 3 |
| `models/wavlm` | 3 |
| `models/yoso` | 3 |

---

## 2. 分层课程清单

共 **70 课**，覆盖 **231 个源文件**。

### 预备篇 · 直觉

**把 45 层的模型压缩成一张能记住的图** —— 从一次 generate() 出发，先建立「这个模型长什么样」的整体直觉，不碰任何优化。

课时数：5 ｜ 覆盖文件：2

| # | 课号 | 主题 | 优先级 | 文件数 | 状态 |
|---|---|---|---|---|---|
| 1 | `L0-01` | 一次 generate() 里发生了什么 | P0 | 1 | ⬜ |
| 2 | `L0-02` | 配置即架构：从 config 读出全部形状 | P0 | 1 | ⬜ |
| 3 | `L0-03` | 层类型排布：34 层 KDA + 11 层 MLA | P0 | 1 | ⬜ |
| 4 | `L0-04` | MoE 经济学：288 选 8 | P0 | 1 | ⬜ |
| 5 | `L0-05` | mHC：4 条残差流 | P1 | 1 | ⬜ |

<details><summary><b>L0-01</b> 一次 generate() 里发生了什么 <code>L0-intuition/L0-01</code></summary>

**讲解要点**

- 把 generate() 拆成 6 个阶段：取 token → 嵌入 → 45 层 → 归一 → 取 logits → 采样
- 实测参数量与 45 层排布，建立「这模型多大」的量感
- 标出后面每一层课程对应这张全景图的哪一块

**验收点**

- [ ] 能不看笔记复述一次前向的 6 个阶段
- [ ] 能说出 45 层里有多少层跑线性注意力

**覆盖文件**

- `models/glm5_next/modeling_glm5_next.py`

</details>

<details><summary><b>L0-02</b> 配置即架构：从 config 读出全部形状 <code>L0-intuition/L0-02</code></summary>

**讲解要点**

- hidden=4096 / 64 头 / qk_nope=256 / v=256 如何拼出注意力形状
- head_dim 被强制等于 qk_rope_head_dim(=0)，这意味着 DSA 层完全不用 RoPE
- mlp_layer_types：前 3 层 dense，其余 42 层 sparse

**验收点**

- [ ] 能从 config 默认值手算出 MLA 层的 Q/K/V 形状
- [ ] 能解释为什么 head_dim 是 0 而不是 256

**覆盖文件**

- `models/glm5_next/configuration_glm5_next.py`

</details>

<details><summary><b>L0-03</b> 层类型排布：34 层 KDA + 11 层 MLA <code>L0-intuition/L0-03</code></summary>

**讲解要点**

- layer_types 的生成规律 idx % 4 != 3 → linear_attention
- indexer_types：哪些层跑索引器（full），哪些复用上一层（shared）
- 为什么是 3:1 而不是 1:1 —— 长上下文服务的成本结构

**验收点**

- [ ] 给定 num_hidden_layers 能写出 layer_types
- [ ] 能解释 shared 索引器省掉了什么

**覆盖文件**

- `models/glm5_next/configuration_glm5_next.py`

</details>

<details><summary><b>L0-04</b> MoE 经济学：288 选 8 <code>L0-intuition/L0-04</code></summary>

**讲解要点**

- n_routed_experts=288, top-k=8, +1 共享专家 → 激活比约 3%
- routed_scaling_factor=2.5 的作用
- n_group / topk_group 分组路由的动机

**验收点**

- [ ] 能算出稀疏层与稠密层的 FLOPs 比
- [ ] 能说出共享专家存在的理由

**覆盖文件**

- `models/glm5_next/modeling_glm5_next.py`

</details>

<details><summary><b>L0-05</b> mHC：4 条残差流 <code>L0-intuition/L0-05</code></summary>

**讲解要点**

- hc_mult=4 —— 残差不是一条而是一束
- Sinkhorn 迭代 20 次把路由矩阵约束到双随机流形上
- 为什么这比单条残差更难训、但扩展性更好

**验收点**

- [ ] 能用自己的话说出双随机约束的作用
- [ ] 能指出 mHC 在解码层的前后各出现一次

**覆盖文件**

- `models/glm5_next/modeling_glm5_next.py`

</details>

### 第 1 层 · 配置与分派

**config 不是参数表，是架构的可执行定义** —— 看懂一个 config 如何决定张量形状，以及 AutoModel 如何找到 GLM-5。

课时数：7 ｜ 覆盖文件：19

| # | 课号 | 主题 | 优先级 | 文件数 | 状态 |
|---|---|---|---|---|---|
| 6 | `L1-01` | PreTrainedConfig：配置对象的公共契约 | P0 | 1 | ⬜ |
| 7 | `L1-02` | Glm5NextTextConfig 逐字段精读 | P0 | 1 | ⬜ |
| 8 | `L1-03` | 视觉配置与顶层嵌套 | P1 | 1 | ⬜ |
| 9 | `L1-04` | 配置的序列化、继承与文档生成 | P1 | 4 | ⬜ |
| 10 | `L1-05` | AutoModel 分派：字符串如何找到类 | P0 | 4 | ⬜ |
| 11 | `L1-06` | 其余四张自动注册表 | P1 | 6 | ⬜ |
| 12 | `L1-07` | 动态模块、类型契约与权重名映射表 | P1 | 3 | ⬜ |

<details><summary><b>L1-01</b> PreTrainedConfig：配置对象的公共契约 <code>L1-config/L1-01</code></summary>

**讲解要点**

- from_pretrained / to_dict / to_json_string 的完整路径
- @strict 装饰器做了什么校验
- sub_configs 与 base_config_key：嵌套配置的寻址规则

**验收点**

- [ ] 能画出 config 从 json 到对象再到模型的路径
- [ ] 能解释 strict 校验在什么时机触发

**覆盖文件**

- `configuration_utils.py`

</details>

<details><summary><b>L1-02</b> Glm5NextTextConfig 逐字段精读 <code>L1-config/L1-02</code></summary>

**讲解要点**

- 每个字段属于哪一组：注意力 / MoE / KDA / DSA / mHC
- validate_architecture 的四条硬约束
- attribute_map 与 keys_to_ignore_at_inference 的用途

**验收点**

- [ ] 能说出四条校验各自防的是什么错误
- [ ] 能列出 KDA 相关的 5 个字段

**覆盖文件**

- `models/glm5_next/configuration_glm5_next.py`

</details>

<details><summary><b>L1-03</b> 视觉配置与顶层嵌套 <code>L1-config/L1-03</code></summary>

**讲解要点**

- Glm5NextVisionConfig：depth=24, patch=14, merge=2 → 每 token 覆盖 28 像素
- Glm5NextConfig 如何把 text/vision 两个子配置拼起来
- 扁平 checkpoint 的向后兼容分支

**验收点**

- [ ] 能算出 336x336 图像产生多少个视觉 token
- [ ] 能解释 image_token_id 与 image_start/end_token_id 的分工

**覆盖文件**

- `models/glm5_next/configuration_glm5_next.py`

</details>

<details><summary><b>L1-04</b> 配置的序列化、继承与文档生成 <code>L1-config/L1-04</code></summary>

**讲解要点**

- hp_naming 如何把自定义类名映射回上游名字
- doc.py 的文档字符串机械替换
- type_validators 如何把类型注解变成运行期校验

**验收点**

- [ ] 能说出 auto_docstring 在类定义时改了什么
- [ ] 能解释为什么需要 hp_naming 这层间接

**覆盖文件**

- `utils/hp_naming.py`
- `utils/doc.py`
- `utils/auto_docstring.py`
- `utils/type_validators.py`

</details>

<details><summary><b>L1-05</b> AutoModel 分派：字符串如何找到类 <code>L1-config/L1-05</code></summary>

**讲解要点**

- CONFIG_MAPPING 的惰性注册表
- auto_factory 的 _LazyAutoMapping 如何做到「用时才导入」
- 从 model_type 字符串到 Glm5NextConfig 的完整链路

**验收点**

- [ ] 能画出 AutoConfig.from_pretrained('glm5_next') 的调用链
- [ ] 能解释惰性映射省下了什么

**覆盖文件**

- `models/auto/__init__.py`
- `models/auto/configuration_auto.py`
- `models/auto/auto_factory.py`
- `models/auto/auto_mappings.py`

</details>

<details><summary><b>L1-06</b> 其余四张自动注册表 <code>L1-config/L1-06</code></summary>

**讲解要点**

- 模型/分词器/图像/视频/处理器五种 Auto 类的对称结构
- AutoProcessor 如何把多个子处理器组合起来
- 命名冲突时的解析优先级

**验收点**

- [ ] 能说出 AutoProcessor 对 GLM-5 会实例化哪几个子处理器

**覆盖文件**

- `models/auto/modeling_auto.py`
- `models/auto/tokenization_auto.py`
- `models/auto/image_processing_auto.py`
- `models/auto/video_processing_auto.py`
- `models/auto/feature_extraction_auto.py`
- `models/auto/processing_auto.py`

</details>

<details><summary><b>L1-07</b> 动态模块、类型契约与权重名映射表 <code>L1-config/L1-07</code></summary>

**讲解要点**

- trust_remote_code 路径下的动态导入
- _typing 里的 TensorType 等别名如何统一注解
- conversion_mapping 如何描述跨实现权重名对应关系

**验收点**

- [ ] 能解释动态模块的缓存与校验策略

**覆盖文件**

- `dynamic_module_utils.py`
- `_typing.py`
- `conversion_mapping.py`

</details>

### 第 2 层 · 输入流水线

**文本和像素如何变成同一串位置上的向量** —— 分词 → 图像/视频处理 → 视觉塔 → 占位符展开，多模态输入如何在序列里对齐。

课时数：7 ｜ 覆盖文件：23

| # | 课号 | 主题 | 优先级 | 文件数 | 状态 |
|---|---|---|---|---|---|
| 13 | `L2-01` | 分词器公共契约：慢与快共享的那一层 | P0 | 1 | ⬜ |
| 14 | `L2-02` | 快分词器、纯 Python 分词器与慢→快转换 | P0 | 3 | ⬜ |
| 15 | `L2-03` | SentencePiece 分词器 | P1 | 1 | ⬜ |
| 16 | `L2-04` | 图像处理基座：尺寸、归一化与后端 | P0 | 3 | ⬜ |
| 17 | `L2-05` | 图像变换、视频与通用视觉工具 | P0 | 6 | ⬜ |
| 18 | `L2-06` | 多模态处理器装配 | P0 | 6 | ⬜ |
| 19 | `L2-07` | 视觉塔：从 336x336 像素到 27 个 token | P0 | 3 | ⬜ |

<details><summary><b>L2-01</b> 分词器公共契约：慢与快共享的那一层 <code>L2-input/L2-01</code></summary>

**讲解要点**

- SpecialTokensMixin 与 token 属性体系
- __call__ 的参数如何映射到张量
- padding / truncation 策略的完整枚举

**验收点**

- [ ] 能说出 input_ids 与 attention_mask 的产生时机
- [ ] 能解释 return_tensors 影响的是哪一步

**覆盖文件**

- `tokenization_utils_base.py`

</details>

<details><summary><b>L2-02</b> 快分词器、纯 Python 分词器与慢→快转换 <code>L2-input/L2-02</code></summary>

**讲解要点**

- tokenizers 库后端的批量编码
- 纯 Python 回退路径的存在意义
- convert_slow_tokenizer 如何把 SentencePiece 模型翻译成 fast tokenizer

**验收点**

- [ ] 能说出 fast/slow 在 GLM-5 上的行为差异点

**覆盖文件**

- `tokenization_utils_tokenizers.py`
- `tokenization_python.py`
- `convert_slow_tokenizer.py`

</details>

<details><summary><b>L2-03</b> SentencePiece 分词器 <code>L2-input/L2-03</code></summary>

**讲解要点**

- unigram / BPE 两种模型的加载
- legacy 行为开关
- 为什么 GLM-5 的 tokenizer 往往走这个基类

**验收点**

- [ ] 能解释 add_dummy_prefix 对首 token 的影响

**覆盖文件**

- `tokenization_utils_sentencepiece.py`

</details>

<details><summary><b>L2-04</b> 图像处理基座：尺寸、归一化与后端 <code>L2-input/L2-04</code></summary>

**讲解要点**

- BaseImageProcessor 的 preprocess 契约
- 三个后端（PIL / torchvision / numpy）的取舍
- size / resample / rescale 的组合语义

**验收点**

- [ ] 能说出默认后端的选择顺序

**覆盖文件**

- `image_processing_base.py`
- `image_processing_utils.py`
- `image_processing_backends.py`

</details>

<details><summary><b>L2-05</b> 图像变换、视频与通用视觉工具 <code>L2-input/L2-05</code></summary>

**讲解要点**

- resize / center_crop / normalize 的张量级实现
- 视频按帧采样与 temporal patch 的关系
- vision_utils 的通用工具集

**验收点**

- [ ] 能算出 336x336 图像经 patch+merge 后的 token 数
- [ ] 能解释 temporal_patch_size=2 对视频帧数的要求

**覆盖文件**

- `image_transforms.py`
- `image_utils.py`
- `video_processing_utils.py`
- `video_utils.py`
- `vision_utils.py`
- `audio_utils.py`

</details>

<details><summary><b>L2-06</b> 多模态处理器装配 <code>L2-input/L2-06</code></summary>

**讲解要点**

- ProcessorMixin 如何把 tokenizer + image/video processor 拼起来
- GLM-5 的 chat template 如何插入视觉占位符
- 两套图像处理器实现（默认 / PIL）的分工

**验收点**

- [ ] 能写出一次图像对话的 inputs 字典有哪些 key

**覆盖文件**

- `processing_utils.py`
- `models/glm5_next/processing_glm5_next.py`
- `models/glm5_next/image_processing_glm5_next.py`
- `models/glm5_next/image_processing_pil_glm5_next.py`
- `models/glm5_next/video_processing_glm5_next.py`
- `feature_extraction_utils.py`

</details>

<details><summary><b>L2-07</b> 视觉塔：从 336x336 像素到 27 个 token <code>L2-input/L2-07</code></summary>

**讲解要点**

- patch embed 的卷积等价形式
- 轴向 RoPE：height/width 两套频率的重组
- patch merger 的空间 2x2 合并与投影

**验收点**

- [ ] 能手算 (336/14)^2 / 2^2 = 144... 并核对实际 token 数
- [ ] 能解释轴向 RoPE 与一维 RoPE 的区别

**覆盖文件**

- `models/glm5_next/modeling_glm5_next.py`
- `backbone_utils.py`
- `utils/backbone_utils.py`

</details>

### 第 3 层 · 文本主干

**一层之内到底有哪几个算子** —— RMSNorm / MoE / mHC / 门控范数的逐个拆解，重点是「为什么这样设计」。

课时数：8 ｜ 覆盖文件：5

| # | 课号 | 主题 | 优先级 | 文件数 | 状态 |
|---|---|---|---|---|---|
| 20 | `L3-01` | 三种 RMSNorm | P0 | 2 | ⬜ |
| 21 | `L3-02` | MLP 与 SwiGLU 截断 | P0 | 2 | ⬜ |
| 22 | `L3-03` | 路由器：288 个分数如何变成 8 条路径 | P0 | 1 | ⬜ |
| 23 | `L3-04` | 专家层与 grouped GEMM | P0 | 1 | ⬜ |
| 24 | `L3-05` | MoE 装配与共享专家 | P0 | 1 | ⬜ |
| 25 | `L3-06` | ★ mHC：流形约束超连接 | P0 | 1 | ⬜ |
| 26 | `L3-07` | 遗忘门与门控范数 | P0 | 1 | ⬜ |
| 27 | `L3-08` | 解码层装配：三个子块的顺序 | P0 | 3 | ⬜ |

<details><summary><b>L3-01</b> 三种 RMSNorm <code>L3-backbone/L3-01</code></summary>

**讲解要点**

- Glm5NextTextRMSNorm 与 Unweighted 的差别
- Glm5NextTextRMSNormGated：KDA 输出为什么需要门控
- eps 取 1e-5 还是 1e-6 的实际影响

**验收点**

- [ ] 能说出无权重版本的用途
- [ ] 能解释门控范数在 KDA 中解决什么问题

**覆盖文件**

- `models/glm5_next/modeling_glm5_next.py`
- `pytorch_utils.py`

</details>

<details><summary><b>L3-02</b> MLP 与 SwiGLU 截断 <code>L3-backbone/L3-02</code></summary>

**讲解要点**

- gate/up/down 三投影的 SwiGLU 结构
- swiglu_limit=10.0 的 clamp 作用与数值动机
- activations 里的 ACT2FN 注册表

**验收点**

- [ ] 能写出 SwiGLU 的前向公式
- [ ] 能解释截断为什么能提升大模型稳定性

**覆盖文件**

- `models/glm5_next/modeling_glm5_next.py`
- `activations.py`

</details>

<details><summary><b>L3-03</b> 路由器：288 个分数如何变成 8 条路径 <code>L3-backbone/L3-03</code></summary>

**讲解要点**

- gate 线性层 + softmax 的完整流程
- n_group / topk_group 的两级筛选
- norm_topk_prob 与 routed_scaling_factor 的数值效果

**验收点**

- [ ] 能手算一次 top-8 选择
- [ ] 能解释分组路由在分布式下的好处

**覆盖文件**

- `models/glm5_next/modeling_glm5_next.py`

</details>

<details><summary><b>L3-04</b> 专家层与 grouped GEMM <code>L3-backbone/L3-04</code></summary>

**讲解要点**

- gate_up_proj 合并打包的存储布局
- _apply_gate 里 swiglu_limit 的二次截断
- 为什么专家权重按 (E, ...) 排布才能用 grouped GEMM

**验收点**

- [ ] 能说出打包存储省下了什么
- [ ] 能解释权重布局与 EP 切分的关系

**覆盖文件**

- `models/glm5_next/modeling_glm5_next.py`

</details>

<details><summary><b>L3-05</b> MoE 装配与共享专家 <code>L3-backbone/L3-05</code></summary>

**讲解要点**

- Glm5NextTextMoE 的前向：共享专家与路由专家的合并方式
- 稠密层走 Glm5NextTextMLP 的分支条件
- output_router_logits 在推理时的影响

**验收点**

- [ ] 能画出 MoE 前向的数据流
- [ ] 能解释共享专家为什么能稳住训练

**覆盖文件**

- `models/glm5_next/modeling_glm5_next.py`

</details>

<details><summary><b>L3-06</b> ★ mHC：流形约束超连接 <code>L3-backbone/L3-06</code></summary>

**讲解要点**

- 4 条残差流的读取、混合、写回三步
- Sinkhorn 迭代 20 次得到双随机矩阵的过程
- HyperHead 如何把多条流收回一条

**验收点**

- [ ] 能手算 2 次 Sinkhorn 迭代
- [ ] 能解释双随机约束对梯度流的稳定作用

**覆盖文件**

- `models/glm5_next/modeling_glm5_next.py`

</details>

<details><summary><b>L3-07</b> 遗忘门与门控范数 <code>L3-backbone/L3-07</code></summary>

**讲解要点**

- ForgetGate 的线性投影 + 下界截断
- linear_lower_bound=-5.0 的数值意图
- A_log / dt_bias 这类参数的初始化来源

**验收点**

- [ ] 能解释下界截断防的是什么
- [ ] 能说出遗忘门在 KDA 递推中的位置

**覆盖文件**

- `models/glm5_next/modeling_glm5_next.py`

</details>

<details><summary><b>L3-08</b> 解码层装配：三个子块的顺序 <code>L3-backbone/L3-08</code></summary>

**讲解要点**

- mHC → 注意力 → mHC → MoE 的完整顺序
- GradientCheckpointingLayer 在推理时的实际行为
- modeling_outputs 的数据类约定

**验收点**

- [ ] 能默写解码层的执行顺序
- [ ] 能说出 residual 在 mHC 下与标准 Transformer 的差异

**覆盖文件**

- `models/glm5_next/modeling_glm5_next.py`
- `modeling_layers.py`
- `modeling_outputs.py`

</details>

### 第 4 层 · 混合注意力 ★

**34 层线性注意力 + 11 层稀疏注意力如何协同** —— KDA 的循环与分块两种算法、DSA 索引器如何选 2048 个位置、MLA 如何压缩 KV。

课时数：8 ｜ 覆盖文件：3

| # | 课号 | 主题 | 优先级 | 文件数 | 状态 |
|---|---|---|---|---|---|
| 28 | `L4-01` | ★ 为什么 GLM-5 要混合两种注意力 | P0 | 1 | ⬜ |
| 29 | `L4-02` | KDA 循环形式：逐 token 的递推 | P0 | 1 | ⬜ |
| 30 | `L4-03` | KDA 分块形式：把递推变成矩阵乘 | P0 | 1 | ⬜ |
| 31 | `L4-04` | 短卷积与 L2 归一 | P0 | 1 | ⬜ |
| 32 | `L4-05` | 线性注意力层装配 | P0 | 1 | ⬜ |
| 33 | `L4-06` | ★ DSA 索引器：2048 个位置是怎么选出来的 | P0 | 1 | ⬜ |
| 34 | `L4-07` | MLA：把 KV 压进 512 维隐向量 | P0 | 2 | ⬜ |
| 35 | `L4-08` | 掩码构造与注意力计算 | P0 | 2 | ⬜ |

<details><summary><b>L4-01</b> ★ 为什么 GLM-5 要混合两种注意力 <code>L4-attention/L4-01</code></summary>

**讲解要点**

- 全注意力的 KV 线性增长 vs 线性注意力的常数状态
- 3:1 排布如何兼顾精确检索与长文本成本
- NoPE 的取舍：位置信息交给 KDA 与索引器

**验收点**

- [ ] 能画出两种层的成本曲线
- [ ] 能解释为什么 DSA 层不需要 RoPE

**覆盖文件**

- `models/glm5_next/modeling_glm5_next.py`

</details>

<details><summary><b>L4-02</b> KDA 循环形式：逐 token 的递推 <code>L4-attention/L4-02</code></summary>

**讲解要点**

- recurrent_kimi_delta_attention 的状态更新公式
- delta 规则与普通线性注意力的区别
- 状态矩阵的形状 (head_dim, v_head_dim)

**验收点**

- [ ] 能手推两步状态更新
- [ ] 能说出 delta 规则修正了什么

**覆盖文件**

- `models/glm5_next/modeling_glm5_next.py`

</details>

<details><summary><b>L4-03</b> KDA 分块形式：把递推变成矩阵乘 <code>L4-attention/L4-03</code></summary>

**讲解要点**

- chunk_kimi_delta_attention 的分块矩阵化
- 块内并行、块间递推的两级结构
- 为什么训练用分块、解码用循环

**验收点**

- [ ] 能解释分块如何提升 GPU 利用率
- [ ] 能说出两种形式在数学上等价的条件

**覆盖文件**

- `models/glm5_next/modeling_glm5_next.py`

</details>

<details><summary><b>L4-04</b> 短卷积与 L2 归一 <code>L4-attention/L4-04</code></summary>

**讲解要点**

- causal_conv1d_fn 与 update 的分工（prefill / decode）
- kernel=4 的深度可分离卷积在 KDA 中的作用
- l2norm 对 q/k 的归一化

**验收点**

- [ ] 能说出 prefill 与 decode 走不同实现的理由
- [ ] 能解释短卷积补齐了什么信息

**覆盖文件**

- `models/glm5_next/modeling_glm5_next.py`

</details>

<details><summary><b>L4-05</b> 线性注意力层装配 <code>L4-attention/L4-05</code></summary>

**讲解要点**

- q/k/v 投影 + 短卷积 + 门控 + 范数的完整前向
- apply_mask_to_padding_states 的用途
- 缓存状态如何读写

**验收点**

- [ ] 能画出 KDA 层的完整数据流
- [ ] 能说出每步的张量形状

**覆盖文件**

- `models/glm5_next/modeling_glm5_next.py`

</details>

<details><summary><b>L4-06</b> ★ DSA 索引器：2048 个位置是怎么选出来的 <code>L4-attention/L4-06</code></summary>

**讲解要点**

- index_head_dim=128, index_n_heads=32 的小型注意力
- index_kpool=16：先池化再选择的粗筛
- get_visible_tokens / get_pooled_states / append_visible_tail 三步
- full 与 shared 两种 indexer_types

**验收点**

- [ ] 能算出索引器的选择预算
- [ ] 能解释池化粗筛省下了多少计算

**覆盖文件**

- `models/glm5_next/modeling_glm5_next.py`

</details>

<details><summary><b>L4-07</b> MLA：把 KV 压进 512 维隐向量 <code>L4-attention/L4-07</code></summary>

**讲解要点**

- q_lora_rank=1536 / kv_lora_rank=512 的压缩路径
- expand_kv 如何从隐向量还原出多头 K/V
- NoPE 下 head_dim=0 导致的实现细节

**验收点**

- [ ] 能算出 MLA 相对 MHA 的 KV 压缩比
- [ ] 能解释 expand_kv 为什么不能省

**覆盖文件**

- `models/glm5_next/modeling_glm5_next.py`
- `modeling_rope_utils.py`

</details>

<details><summary><b>L4-08</b> 掩码构造与注意力计算 <code>L4-attention/L4-08</code></summary>

**讲解要点**

- build_attention_mask_from_topk 如何把索引变成掩码
- eager_attention_forward 与 repeat_kv 的参考实现
- GQA 的 n_rep 展开

**验收点**

- [ ] 能写出掩码的布尔表达式
- [ ] 能说出参考实现与融合后端的差异

**覆盖文件**

- `models/glm5_next/modeling_glm5_next.py`
- `modeling_flash_attention_utils.py`

</details>

### 第 5 层 · 缓存与内存 ★★

**CPU 视角的 KV 记的是公式，GPU 视角记的是字节** —— 转折点。同样的 KV，这里要问：占多少显存、按什么粒度分配、溢出时怎么办。

课时数：6 ｜ 覆盖文件：9

| # | 课号 | 主题 | 优先级 | 文件数 | 状态 |
|---|---|---|---|---|---|
| 36 | `L5-01` | ★ 转折点：Cache 抽象层 | P0 | 1 | ⬜ |
| 37 | `L5-02` | ★ 线性注意力层的缓存：常数状态 | P0 | 1 | ⬜ |
| 38 | `L5-03` | ★ 索引注意力层的缓存：KV + 索引键 | P0 | 1 | ⬜ |
| 39 | `L5-04` | 分页缓存：按块分配与复用 | P0 | 4 | ⬜ |
| 40 | `L5-05` | 掩码生成：从 padding 到因果 | P0 | 2 | ⬜ |
| 41 | `L5-06` | 连续批处理的缓存分配策略 | P1 | 3 | ⬜ |

<details><summary><b>L5-01</b> ★ 转折点：Cache 抽象层 <code>L5-cache/L5-01</code></summary>

**讲解要点**

- DynamicCache 的 layers 列表契约
- 为什么每种层类型需要自己的 Cache 类
- record_past / 预分配 两种策略

**验收点**

- [ ] 能说出 Cache 抽象的三条不变量
- [ ] 能解释为什么不能用统一结构装两种层

**覆盖文件**

- `cache_utils.py`

</details>

<details><summary><b>L5-02</b> ★ 线性注意力层的缓存：常数状态 <code>L5-cache/L5-02</code></summary>

**讲解要点**

- conv_states 与 recurrent_states 的形状
- 为什么 KDA 层的显存与序列长度无关
- has_previous_state 标志的作用

**验收点**

- [ ] 能手算 KDA 层每 token 的额外显存
- [ ] 能解释 conv 状态为什么必须单独存

**覆盖文件**

- `cache_utils.py`

</details>

<details><summary><b>L5-03</b> ★ 索引注意力层的缓存：KV + 索引键 <code>L5-cache/L5-03</code></summary>

**讲解要点**

- keys/values 的追加式增长
- indexer_keys 的独立存储与 dtype
- 每 token 显存的完整算式

**验收点**

- [ ] 能手算 32K 上下文下 MLA 层的 KV 显存
- [ ] 能说出 indexer_keys 为什么可以低精度

**覆盖文件**

- `cache_utils.py`

</details>

<details><summary><b>L5-04</b> 分页缓存：按块分配与复用 <code>L5-cache/L5-04</code></summary>

**讲解要点**

- page/block 的粒度与碎片
- 池化分配器的空闲链表
- 引用计数与共享前缀

**验收点**

- [ ] 能解释分页相比连续分配省下什么
- [ ] 能说出块大小对碎片率的影响

**覆盖文件**

- `generation/continuous_batching/cache.py`
- `generation/continuous_batching/cache_allocators/cache_pool.py`
- `generation/continuous_batching/cache_allocators/cache_allocator.py`
- `generation/continuous_batching/cache_allocators/__init__.py`

</details>

<details><summary><b>L5-05</b> 掩码生成：从 padding 到因果 <code>L5-cache/L5-05</code></summary>

**讲解要点**

- 4D 因果掩码与 2D padding 掩码的合成
- 滑动窗口掩码的构造
- sdpa / flash 各自需要的掩码形态

**验收点**

- [ ] 能手写一个 (1,1,q,k) 因果掩码
- [ ] 能解释不同后端对掩码形状的要求

**覆盖文件**

- `masking_utils.py`
- `modeling_attn_mask_utils.py`

</details>

<details><summary><b>L5-06</b> 连续批处理的缓存分配策略 <code>L5-cache/L5-06</code></summary>

**讲解要点**

- 全注意力与滑窗注意力的块生命周期差异
- 滑窗导致的块回收时机
- 前缀共享在批内的实现

**验收点**

- [ ] 能画出两种策略的块状态机

**覆盖文件**

- `generation/continuous_batching/cache_allocators/full_attention.py`
- `generation/continuous_batching/cache_allocators/sliding_attention.py`
- `generation/continuous_batching/cache.py`

</details>

### 第 6 层 · 生成循环

**一个 token 是怎么被「决定」出来的** —— generate 主循环、logits 处理、停止条件、投机解码、连续批处理调度。

课时数：8 ｜ 覆盖文件：19

| # | 课号 | 主题 | 优先级 | 文件数 | 状态 |
|---|---|---|---|---|---|
| 42 | `L6-01` | generate 主循环 | P0 | 2 | ⬜ |
| 43 | `L6-02` | GenerationConfig：采样参数的容器 | P0 | 1 | ⬜ |
| 44 | `L6-03` | logits 处理链 | P0 | 1 | ⬜ |
| 45 | `L6-04` | 停止条件与流式输出 | P0 | 2 | ⬜ |
| 46 | `L6-05` | 投机解码 | P1 | 1 | ⬜ |
| 47 | `L6-06` | 水印 | P1 | 1 | ⬜ |
| 48 | `L6-07` | 连续批处理：调度器与模型运行器 | P0 | 6 | ⬜ |
| 49 | `L6-08` | 连续批处理：处理器、分发与卸载 | P1 | 5 | ⬜ |

<details><summary><b>L6-01</b> generate 主循环 <code>L6-generation/L6-01</code></summary>

**讲解要点**

- prepare_inputs → prefill → decode 循环 → finalize
- cache_position 的计算与更新
- stateful 模型（GLM-5）走的分支

**验收点**

- [ ] 能画出 generate 的主循环流程图
- [ ] 能解释 cache_position 的作用

**覆盖文件**

- `generation/utils.py`
- `generation/__init__.py`

</details>

<details><summary><b>L6-02</b> GenerationConfig：采样参数的容器 <code>L6-generation/L6-02</code></summary>

**讲解要点**

- 采样相关字段的完整清单与互斥规则
- validate 阶段的参数校验
- generation_config 与 model.config 的关系

**验收点**

- [ ] 能说出 temperature=0 时实际走了哪条分支

**覆盖文件**

- `generation/configuration_utils.py`

</details>

<details><summary><b>L6-03</b> logits 处理链 <code>L6-generation/L6-03</code></summary>

**讲解要点**

- 处理器按优先级串联的顺序（这决定了结果）
- top-k / top-p / min-p 的累积概率实现
- 重复惩罚与存在惩罚的差别

**验收点**

- [ ] 能说出 top-p 与 top-k 同时开启时的实际顺序
- [ ] 能手算一次重复惩罚

**覆盖文件**

- `generation/logits_process.py`

</details>

<details><summary><b>L6-04</b> 停止条件与流式输出 <code>L6-generation/L6-04</code></summary>

**讲解要点**

- MaxLength / EOS / StopString 三类条件的组合
- 流式输出的增量解码
- stopping criteria 在批处理下的向量化

**验收点**

- [ ] 能说出多个停止条件是或还是与
- [ ] 能解释流式解码为什么要处理多字节字符

**覆盖文件**

- `generation/stopping_criteria.py`
- `generation/streamers.py`

</details>

<details><summary><b>L6-05</b> 投机解码 <code>L6-generation/L6-05</code></summary>

**讲解要点**

- 草稿模型与主模型的对齐
- 为什么 GLM-5 不支持 assisted generation
- 候选生成的接受/拒绝逻辑

**验收点**

- [ ] 能解释接受率与加速比的关系

**覆盖文件**

- `generation/candidate_generator.py`

</details>

<details><summary><b>L6-06</b> 水印 <code>L6-generation/L6-06</code></summary>

**讲解要点**

- logits 偏置式水印的原理
- 水印强度与文本质量的权衡

**验收点**

- [ ] 能说出水印对困惑度的影响方向

**覆盖文件**

- `generation/watermarking.py`

</details>

<details><summary><b>L6-07</b> 连续批处理：调度器与模型运行器 <code>L6-generation/L6-07</code></summary>

**讲解要点**

- 请求队列 → 调度 → 前向 → 回写 的闭环
- continuous batching 相比静态批处理省下什么
- 请求的优先级与抢占

**验收点**

- [ ] 能画出调度器的主循环
- [ ] 能解释静态批处理下 GPU 空转的原因

**覆盖文件**

- `generation/continuous_batching/__init__.py`
- `generation/continuous_batching/scheduler.py`
- `generation/continuous_batching/model_runner.py`
- `generation/continuous_batching/continuous_api.py`
- `generation/continuous_batching/requests.py`
- `generation/continuous_batching/input_outputs.py`

</details>

<details><summary><b>L6-08</b> 连续批处理：处理器、分发与卸载 <code>L6-generation/L6-08</code></summary>

**讲解要点**

- 批内 logits 处理器如何在调度器里执行
- 多进程/多卡下的请求分发
- 权重卸载到 CPU 的时机

**验收点**

- [ ] 能说出卸载管理器在显存不足时的决策顺序

**覆盖文件**

- `generation/continuous_batching/cb_logits_processors.py`
- `generation/continuous_batching/initialization.py`
- `generation/continuous_batching/distributed.py`
- `generation/continuous_batching/utils.py`
- `generation/continuous_batching/offloading_manager.py`

</details>

### 第 7 层 · 权重加载与量化

**磁盘上的字节如何变成显存里的张量** —— from_pretrained 的完整路径、权重名映射、以及 30 个量化器各自的取舍。

课时数：7 ｜ 覆盖文件：46

| # | 课号 | 主题 | 优先级 | 文件数 | 状态 |
|---|---|---|---|---|---|
| 50 | `L7-01` | from_pretrained 全景 | P0 | 2 | ⬜ |
| 51 | `L7-02` | 核心加载器：权重名映射 | P0 | 1 | ⬜ |
| 52 | `L7-03` | 张量并行计划与模型输出 | P0 | 3 | ⬜ |
| 53 | `L7-04` | 量化基座：Quantizer 契约 | P0 | 5 | ⬜ |
| 54 | `L7-05` | 量化器家族（一）：GPU 主流方案 | P1 | 10 | ⬜ |
| 55 | `L7-06` | 量化器家族（二）：新格式与实验方案 | P1 | 15 | ⬜ |
| 56 | `L7-07` | GGUF：另一种权重容器 | P1 | 10 | ⬜ |

<details><summary><b>L7-01</b> from_pretrained 全景 <code>L7-loading/L7-01</code></summary>

**讲解要点**

- config → 模型骨架 → 权重下载 → 加载 → 后处理 的完整阶段
- device_map 与 dtype 的解析顺序
- 低内存加载（meta device）路径

**验收点**

- [ ] 能画出 from_pretrained 的 6 个阶段
- [ ] 能解释 meta device 省下了什么

**覆盖文件**

- `modeling_utils.py`
- `initialization.py`

</details>

<details><summary><b>L7-02</b> 核心加载器：权重名映射 <code>L7-loading/L7-02</code></summary>

**讲解要点**

- checkpoint 名 → 模型参数名 的映射规则
- 量化权重的反量化与拼接
- 加载报告中的 missing / unexpected 分类

**验收点**

- [ ] 能解释一个权重名是如何被解析到目标模块的

**覆盖文件**

- `core_model_loading.py`

</details>

<details><summary><b>L7-03</b> 张量并行计划与模型输出 <code>L7-loading/L7-03</code></summary>

**讲解要点**

- base_model_tp_plan 的 colwise/rowwise 标注
- base_model_ep_plan 的 grouped_gemm 标注
- 模型输出数据类的字段

**验收点**

- [ ] 能把 GLM-5 的 tp_plan 逐条对应到模块
- [ ] 能解释 colwise 与 rowwise 的配对规则

**覆盖文件**

- `modeling_layers.py`
- `modeling_outputs.py`
- `models/glm5_next/modeling_glm5_next.py`

</details>

<details><summary><b>L7-04</b> 量化基座：Quantizer 契约 <code>L7-loading/L7-04</code></summary>

**讲解要点**

- HfQuantizer 的 6 个钩子
- auto 如何从 config 选出量化器
- 量化配置的校验与默认值

**验收点**

- [ ] 能说出量化器在加载流程中的介入时机
- [ ] 能解释 pre_quantized 与 on-the-fly 的差别

**覆盖文件**

- `quantizers/__init__.py`
- `quantizers/base.py`
- `quantizers/auto.py`
- `quantizers/quantizers_utils.py`
- `utils/quantization_config.py`

</details>

<details><summary><b>L7-05</b> 量化器家族（一）：GPU 主流方案 <code>L7-loading/L7-05</code></summary>

**讲解要点**

- 每种的位宽、粒度（per-tensor/channel/group）与是否需要校准
- 推理时反量化发生在哪一步
- 哪种适合 GLM-5 这类 MoE

**验收点**

- [ ] 能说出 4bit 分组量化的分组大小对精度的影响

**覆盖文件**

- `quantizers/quantizer_bnb_4bit.py`
- `quantizers/quantizer_bnb_8bit.py`
- `quantizers/quantizer_gptq.py`
- `quantizers/quantizer_awq.py`
- `quantizers/quantizer_eetq.py`
- `quantizers/quantizer_hqq.py`
- `quantizers/quantizer_aqlm.py`
- `quantizers/quantizer_vptq.py`
- `quantizers/quantizer_spqr.py`
- `quantizers/quantizer_quark.py`

</details>

<details><summary><b>L7-06</b> 量化器家族（二）：新格式与实验方案 <code>L7-loading/L7-06</code></summary>

**讲解要点**

- FP8 / MXFP4 / NVFP4 的浮点量化与整数量化的差别
- 压缩张量格式如何描述任意量化方案
- 逐块缩放因子的开销

**验收点**

- [ ] 能解释 MXFP4 的块共享指数

**覆盖文件**

- `quantizers/quantizer_quanto.py`
- `quantizers/quantizer_torchao.py`
- `quantizers/quantizer_finegrained_fp8.py`
- `quantizers/quantizer_fbgemm_fp8.py`
- `quantizers/quantizer_fp_quant.py`
- `quantizers/quantizer_mxfp4.py`
- `quantizers/quantizer_nvfp4.py`
- `quantizers/quantizer_compressed_tensors.py`
- `quantizers/quantizer_auto_round.py`
- `quantizers/quantizer_bitnet.py`
- `quantizers/quantizer_higgs.py`
- `quantizers/quantizer_sinq.py`
- `quantizers/quantizer_metal.py`
- `quantizers/quantizer_gemma.py`
- `quantizers/quantizer_fouroversix.py`

</details>

<details><summary><b>L7-07</b> GGUF：另一种权重容器 <code>L7-loading/L7-07</code></summary>

**讲解要点**

- GGUF 文件结构与量化块的反量化
- safetensors 自动转换的存在意义
- GGUF 分词器到 HF 分词器的映射

**验收点**

- [ ] 能说出 GGUF 相比 safetensors 的取舍

**覆盖文件**

- `quantizers/quantizer_gguf.py`
- `modeling_gguf_pytorch_utils.py`
- `safetensors_conversion.py`
- `integrations/gguf/__init__.py`
- `integrations/gguf/dequant.py`
- `integrations/gguf/gguf_conversion_mapping.py`
- `integrations/gguf/gguf_tokenizer_mapping.py`
- `integrations/gguf/kernels.py`
- `integrations/gguf/reader.py`
- `integrations/gguf/utils.py`

</details>

### 第 8 层 · 推理规模化

**单卡跑不动时，模型被怎么切开** —— 张量/专家/流水线并行计划，注意力后端分派，异构设备与内核选择。

课时数：6 ｜ 覆盖文件：32

| # | 课号 | 主题 | 优先级 | 文件数 | 状态 |
|---|---|---|---|---|---|
| 57 | `L8-01` | 注意力后端分派 | P0 | 7 | ⬜ |
| 58 | `L8-02` | 分页注意力后端 | P0 | 4 | ⬜ |
| 59 | `L8-03` | 张量并行 | P0 | 4 | ⬜ |
| 60 | `L8-04` | 专家并行与 MoE 集成 | P0 | 4 | ⬜ |
| 61 | `L8-05` | 流水线并行、分片与配置 | P1 | 6 | ⬜ |
| 62 | `L8-06` | 异构设备与加速器集成 | P1 | 7 | ⬜ |

<details><summary><b>L8-01</b> 注意力后端分派 <code>L8-scaling/L8-01</code></summary>

**讲解要点**

- attn_implementation 的取值与回退顺序
- SDPA 后端的 GQA 处理
- flex_attention 的可编程掩码

**验收点**

- [ ] 能说出指定 flash_attention_2 但不可用时的实际行为

**覆盖文件**

- `integrations/__init__.py`
- `integrations/sdpa_attention.py`
- `integrations/flash_attention.py`
- `integrations/flex_attention.py`
- `integrations/msa_attention.py`
- `integrations/npu_flash_attention.py`
- `monkey_patching.py`

</details>

<details><summary><b>L8-02</b> 分页注意力后端 <code>L8-scaling/L8-02</code></summary>

**讲解要点**

- paged attention 与连续 KV 的接口差异
- 块表（block table）如何参与注意力计算
- 三种分页后端的适用场景

**验收点**

- [ ] 能解释块表在注意力核里怎么用

**覆盖文件**

- `integrations/eager_paged.py`
- `integrations/flash_paged.py`
- `integrations/sdpa_paged.py`
- `modeling_attn_mask_utils.py`

</details>

<details><summary><b>L8-03</b> 张量并行 <code>L8-scaling/L8-03</code></summary>

**讲解要点**

- colwise / rowwise / packed_colwise 的切分方式
- mla_kv_a_proj 这类自定义切分策略
- 分配后如何保证数值等价

**验收点**

- [ ] 能说出 colwise 后接 rowwise 为什么不需要 all-reduce

**覆盖文件**

- `distributed/__init__.py`
- `distributed/tensor_parallel.py`
- `distributed/mixin.py`
- `integrations/tensor_parallel.py`

</details>

<details><summary><b>L8-04</b> 专家并行与 MoE 集成 <code>L8-scaling/L8-04</code></summary>

**讲解要点**

- ep_router / grouped_gemm / moe_tp_experts 三种标注
- 288 个专家如何切到多卡
- all-to-all 通信量的估算

**验收点**

- [ ] 能算出 EP=8 时每卡负责多少专家
- [ ] 能解释路由 all-to-all 的通信复杂度

**覆盖文件**

- `integrations/moe.py`
- `integrations/sonicmoe.py`
- `integrations/deepgemm.py`
- `distributed/utils.py`

</details>

<details><summary><b>L8-05</b> 流水线并行、分片与配置 <code>L8-scaling/L8-05</code></summary>

**讲解要点**

- base_model_pp_plan 的输入输出约定
- 流水线气泡与微批
- 分片策略与推理的关系

**验收点**

- [ ] 能画出 GLM-5 的 pp_plan 切分点

**覆盖文件**

- `distributed/pipeline_parallel.py`
- `distributed/sharding_utils.py`
- `distributed/configuration_utils.py`
- `distributed/fsdp.py`
- `integrations/fsdp.py`
- `integrations/deepspeed.py`

</details>

<details><summary><b>L8-06</b> 异构设备与加速器集成 <code>L8-scaling/L8-06</code></summary>

**讲解要点**

- 异构配置如何在 config 上挂载额外字段
- TPU 与 GPU 的路径差异
- hub_kernels 的动态内核下载

**验收点**

- [ ] 能说出异构配置生效的时机

**覆盖文件**

- `integrations/heterogeneity/__init__.py`
- `integrations/heterogeneity/configuration_utils.py`
- `integrations/mistral/__init__.py`
- `integrations/mistral/constants.py`
- `integrations/accelerate.py`
- `integrations/tpu.py`
- `integrations/hub_kernels.py`

</details>

### 第 9 层 · 出口与周边

**推理之外，但推理离不开的那些文件** —— 导出器、运行期基础设施、对话模板、量化集成、模型族继承链，以及全量收束。

课时数：8 ｜ 覆盖文件：81

| # | 课号 | 主题 | 优先级 | 文件数 | 状态 |
|---|---|---|---|---|---|
| 63 | `L9-01` | 导出器：把模型交给别的运行时 | P1 | 9 | ⬜ |
| 64 | `L9-02` | 推理时为什么还 import 了训练损失 | P2 | 12 | ⬜ |
| 65 | `L9-03` | 运行期基础设施 | P1 | 11 | ⬜ |
| 66 | `L9-04` | 加载报告、内核配置与调试工具 | P2 | 8 | ⬜ |
| 67 | `L9-05` | 对话模板与工具调用解析 | P1 | 5 | ⬜ |
| 68 | `L9-06` | 量化集成层 | P1 | 13 | ⬜ |
| 69 | `L9-07` | 模型族继承链：GLM-5 站在谁的肩膀上 | P1 | 23 | ⬜ |
| 70 | `L9-08` | 收束：一次 GLM-5 推理的完整文件清单 | P0 | 0 | ⬜ |

<details><summary><b>L9-01</b> 导出器：把模型交给别的运行时 <code>L9-periphery/L9-01</code></summary>

**讲解要点**

- 统一导出接口与各后端的差异
- 动态轴的声明
- 为什么要导出 —— 推理部署的最后一步

**验收点**

- [ ] 能说出 ONNX 导出的动态轴怎么声明

**覆盖文件**

- `exporters/__init__.py`
- `exporters/auto.py`
- `exporters/base.py`
- `exporters/configs.py`
- `exporters/exporter_dynamo.py`
- `exporters/exporter_executorch.py`
- `exporters/exporter_onnx.py`
- `exporters/exporter_openvino.py`
- `exporters/utils.py`

</details>

<details><summary><b>L9-02</b> 推理时为什么还 import 了训练损失 <code>L9-periphery/L9-02</code></summary>

**讲解要点**

- 实测：这些文件确实在推理闭包里，但前向不执行
- 导入副作用与配置时间开销
- Router 的 load balancing loss 在哪

**验收点**

- [ ] 能说出 loss_for_object_detection 被导入的直接原因

**覆盖文件**

- `loss/__init__.py`
- `loss/loss_utils.py`
- `loss/loss_for_object_detection.py`
- `loss/loss_d_fine.py`
- `loss/loss_deformable_detr.py`
- `loss/loss_deimv2.py`
- `loss/loss_grounding_dino.py`
- `loss/loss_lw_detr.py`
- `loss/loss_rf_detr.py`
- `loss/loss_rnnt.py`
- `loss/loss_rt_detr.py`
- `loss/loss_tdt.py`

</details>

<details><summary><b>L9-03</b> 运行期基础设施 <code>L9-periphery/L9-03</code></summary>

**讲解要点**

- 惰性模块 __getattr__ 的实现机制
- import_utils 的能力探测与缓存
- 版本守卫与依赖检查表

**验收点**

- [ ] 能解释 `from transformers import X` 为什么不会一次性导入全部模型

**覆盖文件**

- `utils/__init__.py`
- `utils/logging.py`
- `utils/hub.py`
- `utils/import_utils.py`
- `utils/versions.py`
- `utils/deprecation.py`
- `utils/constants.py`
- `dependency_versions_check.py`
- `dependency_versions_table.py`
- `utils/generic.py`
- `__init__.py`

</details>

<details><summary><b>L9-04</b> 加载报告、内核配置与调试工具 <code>L9-periphery/L9-04</code></summary>

**讲解要点**

- 加载报告如何把 missing/unexpected 变成可读结论
- kernel_config 描述可替换内核
- dummy objects 如何让未安装依赖也能被 import

**验收点**

- [ ] 能解释 dummy objects 的运作方式

**覆盖文件**

- `utils/loading_report.py`
- `utils/kernel_config.py`
- `utils/output_capturing.py`
- `utils/attention_visualizer.py`
- `utils/peft_utils.py`
- `utils/dummy_mistral_common_objects.py`
- `utils/dummy_sentencepiece_and_tokenizers_objects.py`
- `trainer_utils.py`

</details>

<details><summary><b>L9-05</b> 对话模板与工具调用解析 <code>L9-periphery/L9-05</code></summary>

**讲解要点**

- Jinja 模板的渲染与校验
- 工具调用的流式解析状态机
- 响应模板如何适配不同模型的输出格式

**验收点**

- [ ] 能说出 apply_chat_template 插入视觉占位符的位置

**覆盖文件**

- `utils/chat_template_utils.py`
- `utils/chat_parsing/__init__.py`
- `utils/chat_parsing/content_parsers.py`
- `utils/chat_parsing/response_parser.py`
- `utils/chat_parsing/response_templates.py`

</details>

<details><summary><b>L9-06</b> 量化集成层 <code>L9-periphery/L9-06</code></summary>

**讲解要点**

- 每个集成提供的算子替换点
- PEFT 与量化叠加时的顺序
- 为什么有了 quantizer 还需要 integration

**验收点**

- [ ] 能说出 quantizer 与 integration 的职责边界

**覆盖文件**

- `integrations/bitsandbytes.py`
- `integrations/quanto.py`
- `integrations/torchao.py`
- `integrations/fp_quant.py`
- `integrations/finegrained_fp8.py`
- `integrations/ggml.py`
- `integrations/compressed_tensors.py`
- `integrations/aqlm.py`
- `integrations/hqq.py`
- `integrations/spqr.py`
- `integrations/peft.py`
- `integrations/tiktoken.py`
- `integrations/liger.py`

</details>

<details><summary><b>L9-07</b> 模型族继承链：GLM-5 站在谁的肩膀上 <code>L9-periphery/L9-07</code></summary>

**讲解要点**

- 实测：为什么这些模型会被 GLM-5 的推理闭包带进来
- KDA 与 Kimi 的渊源、DSA 与 DeepSeek 的渊源
- 共享组件与复制粘贴的边界

**验收点**

- [ ] 能说出 GLM-5 的 KDA 与 kimi_k25 的关系

**覆盖文件**

- `models/glm5_next/__init__.py`
- `models/glm4v/__init__.py`
- `models/glm4v/configuration_glm4v.py`
- `models/glm4v/modeling_glm4v.py`
- `models/glm4v_moe/__init__.py`
- `models/glm4v_moe/configuration_glm4v_moe.py`
- `models/glm4v_moe/modeling_glm4v_moe.py`
- `models/glm_ocr/__init__.py`
- `models/glm_ocr/configuration_glm_ocr.py`
- `models/glm_ocr/modeling_glm_ocr.py`
- `models/kimi_k25/__init__.py`
- `models/kimi_k25/configuration_kimi_k25.py`
- `models/kimi_k25/modeling_kimi_k25.py`
- `models/qwen3_5/__init__.py`
- `models/qwen3_5/configuration_qwen3_5.py`
- `models/qwen3_5/modeling_qwen3_5.py`
- `models/qwen3_5/tokenization_qwen3_5.py`
- `models/qwen3_5_moe/__init__.py`
- `models/qwen3_5_moe/configuration_qwen3_5_moe.py`
- `models/qwen3_5_moe/modeling_qwen3_5_moe.py`
- `models/deepseek_v2/__init__.py`
- `models/deepseek_v2/configuration_deepseek_v2.py`
- `models/deepseek_v2/modeling_deepseek_v2.py`

</details>

<details><summary><b>L9-08</b> 收束：一次 GLM-5 推理的完整文件清单 <code>L9-periphery/L9-08</code></summary>

**讲解要点**

- 把 9 层课压缩成一张「文件 → 阶段」的总表
- 重新回答 L0-01 的全景图，这次每一块都能说出文件名
- 实测统计：231 个文件、多少行、多少课

**验收点**

- [ ] 能默画全景图并标注每块的源文件
- [ ] 能说出四道门禁各自断言了什么

**覆盖文件**


</details>

---

## 3. 覆盖度矩阵（文件 → 课）

共 231 个文件；每个文件至少被一课引用。差集：**0**

| 源文件 | 行数 | 归属课 |
|---|---|---|
| `__init__.py` | 877 | L9-03 |
| `_typing.py` | 186 | L1-07 |
| `activations.py` | 370 | L3-02 |
| `audio_utils.py` | 1521 | L2-05 |
| `backbone_utils.py` | 388 | L2-07 |
| `cache_utils.py` | 2163 | L5-01, L5-02, L5-03 |
| `configuration_utils.py` | 1567 | L1-01 |
| `conversion_mapping.py` | 2128 | L1-07 |
| `convert_slow_tokenizer.py` | 2096 | L2-02 |
| `core_model_loading.py` | 1859 | L7-02 |
| `dependency_versions_check.py` | 63 | L9-03 |
| `dependency_versions_table.py` | 93 | L9-03 |
| `distributed/__init__.py` | 54 | L8-03 |
| `distributed/configuration_utils.py` | 100 | L8-05 |
| `distributed/fsdp.py` | 277 | L8-05 |
| `distributed/mixin.py` | 279 | L8-03 |
| `distributed/pipeline_parallel.py` | 284 | L8-05 |
| `distributed/sharding_utils.py` | 332 | L8-05 |
| `distributed/tensor_parallel.py` | 851 | L8-03 |
| `distributed/utils.py` | 353 | L8-04 |
| `dynamic_module_utils.py` | 850 | L1-07 |
| `exporters/__init__.py` | 22 | L9-01 |
| `exporters/auto.py` | 189 | L9-01 |
| `exporters/base.py` | 206 | L9-01 |
| `exporters/configs.py` | 217 | L9-01 |
| `exporters/exporter_dynamo.py` | 722 | L9-01 |
| `exporters/exporter_executorch.py` | 1362 | L9-01 |
| `exporters/exporter_onnx.py` | 1178 | L9-01 |
| `exporters/exporter_openvino.py` | 2244 | L9-01 |
| `exporters/utils.py` | 1236 | L9-01 |
| `feature_extraction_utils.py` | 689 | L2-06 |
| `generation/__init__.py` | 222 | L6-01 |
| `generation/candidate_generator.py` | 1769 | L6-05 |
| `generation/configuration_utils.py` | 1915 | L6-02 |
| `generation/continuous_batching/__init__.py` | 30 | L6-07 |
| `generation/continuous_batching/cache.py` | 682 | L5-04, L5-06 |
| `generation/continuous_batching/cache_allocators/__init__.py` | 18 | L5-04 |
| `generation/continuous_batching/cache_allocators/cache_allocator.py` | 369 | L5-04 |
| `generation/continuous_batching/cache_allocators/cache_pool.py` | 137 | L5-04 |
| `generation/continuous_batching/cache_allocators/full_attention.py` | 234 | L5-06 |
| `generation/continuous_batching/cache_allocators/sliding_attention.py` | 182 | L5-06 |
| `generation/continuous_batching/cb_logits_processors.py` | 329 | L6-08 |
| `generation/continuous_batching/continuous_api.py` | 1470 | L6-07 |
| `generation/continuous_batching/distributed.py` | 208 | L6-08 |
| `generation/continuous_batching/initialization.py` | 314 | L6-08 |
| `generation/continuous_batching/input_outputs.py` | 832 | L6-07 |
| `generation/continuous_batching/model_runner.py` | 332 | L6-07 |
| `generation/continuous_batching/offloading_manager.py` | 394 | L6-08 |
| `generation/continuous_batching/requests.py` | 380 | L6-07 |
| `generation/continuous_batching/scheduler.py` | 409 | L6-07 |
| `generation/continuous_batching/utils.py` | 254 | L6-08 |
| `generation/logits_process.py` | 3239 | L6-03 |
| `generation/stopping_criteria.py` | 644 | L6-04 |
| `generation/streamers.py` | 407 | L6-04 |
| `generation/utils.py` | 4265 | L6-01 |
| `generation/watermarking.py` | 551 | L6-06 |
| `image_processing_backends.py` | 668 | L2-04 |
| `image_processing_base.py` | 511 | L2-04 |
| `image_processing_utils.py` | 698 | L2-04 |
| `image_transforms.py` | 1103 | L2-05 |
| `image_utils.py` | 1077 | L2-05 |
| `initialization.py` | 341 | L7-01 |
| `integrations/__init__.py` | 317 | L8-01 |
| `integrations/accelerate.py` | 1024 | L8-06 |
| `integrations/aqlm.py` | 71 | L9-06 |
| `integrations/bitsandbytes.py` | 350 | L9-06 |
| `integrations/compressed_tensors.py` | 383 | L9-06 |
| `integrations/deepgemm.py` | 940 | L8-04 |
| `integrations/deepspeed.py` | 747 | L8-05 |
| `integrations/eager_paged.py` | 84 | L8-02 |
| `integrations/finegrained_fp8.py` | 1144 | L9-06 |
| `integrations/flash_attention.py` | 103 | L8-01 |
| `integrations/flash_paged.py` | 151 | L8-02 |
| `integrations/flex_attention.py` | 369 | L8-01 |
| `integrations/fp_quant.py` | 146 | L9-06 |
| `integrations/fsdp.py` | 25 | L8-05 |
| `integrations/ggml.py` | 373 | L9-06 |
| `integrations/gguf/__init__.py` | 47 | L7-07 |
| `integrations/gguf/dequant.py` | 691 | L7-07 |
| `integrations/gguf/gguf_conversion_mapping.py` | 440 | L7-07 |
| `integrations/gguf/gguf_tokenizer_mapping.py` | 285 | L7-07 |
| `integrations/gguf/kernels.py` | 174 | L7-07 |
| `integrations/gguf/reader.py` | 229 | L7-07 |
| `integrations/gguf/utils.py` | 300 | L7-07 |
| `integrations/heterogeneity/__init__.py` | 23 | L8-06 |
| `integrations/heterogeneity/configuration_utils.py` | 400 | L8-06 |
| `integrations/hqq.py` | 130 | L9-06 |
| `integrations/hub_kernels.py` | 1252 | L8-06 |
| `integrations/liger.py` | 54 | L9-06 |
| `integrations/mistral/__init__.py` | 40 | L8-06 |
| `integrations/mistral/constants.py` | 39 | L8-06 |
| `integrations/moe.py` | 598 | L8-04 |
| `integrations/msa_attention.py` | 263 | L8-01 |
| `integrations/npu_flash_attention.py` | 144 | L8-01 |
| `integrations/peft.py` | 702 | L9-06 |
| `integrations/quanto.py` | 120 | L9-06 |
| `integrations/sdpa_attention.py` | 171 | L8-01 |
| `integrations/sdpa_paged.py` | 75 | L8-02 |
| `integrations/sonicmoe.py` | 220 | L8-04 |
| `integrations/spqr.py` | 75 | L9-06 |
| `integrations/tensor_parallel.py` | 94 | L8-03 |
| `integrations/tiktoken.py` | 54 | L9-06 |
| `integrations/torchao.py` | 234 | L9-06 |
| `integrations/tpu.py` | 240 | L8-06 |
| `loss/__init__.py` | 14 | L9-02 |
| `loss/loss_d_fine.py` | 374 | L9-02 |
| `loss/loss_deformable_detr.py` | 202 | L9-02 |
| `loss/loss_deimv2.py` | 262 | L9-02 |
| `loss/loss_for_object_detection.py` | 571 | L9-02 |
| `loss/loss_grounding_dino.py` | 295 | L9-02 |
| `loss/loss_lw_detr.py` | 360 | L9-02 |
| `loss/loss_rf_detr.py` | 473 | L9-02 |
| `loss/loss_rnnt.py` | 107 | L9-02 |
| `loss/loss_rt_detr.py` | 475 | L9-02 |
| `loss/loss_tdt.py` | 194 | L9-02 |
| `loss/loss_utils.py` | 202 | L9-02 |
| `masking_utils.py` | 1602 | L5-05 |
| `modeling_attn_mask_utils.py` | 504 | L5-05, L8-02 |
| `modeling_flash_attention_utils.py` | 841 | L4-08 |
| `modeling_gguf_pytorch_utils.py` | 817 | L7-07 |
| `modeling_layers.py` | 683 | L3-08, L7-03 |
| `modeling_outputs.py` | 1663 | L3-08, L7-03 |
| `modeling_rope_utils.py` | 1136 | L4-07 |
| `modeling_utils.py` | 5160 | L7-01 |
| `models/auto/__init__.py` | 34 | L1-05 |
| `models/auto/auto_factory.py` | 707 | L1-05 |
| `models/auto/auto_mappings.py` | 1359 | L1-05 |
| `models/auto/configuration_auto.py` | 466 | L1-05 |
| `models/auto/feature_extraction_auto.py` | 393 | L1-06 |
| `models/auto/image_processing_auto.py` | 780 | L1-06 |
| `models/auto/modeling_auto.py` | 2689 | L1-06 |
| `models/auto/processing_auto.py` | 383 | L1-06 |
| `models/auto/tokenization_auto.py` | 1039 | L1-06 |
| `models/auto/video_processing_auto.py` | 434 | L1-06 |
| `models/deepseek_v2/__init__.py` | 30 | L9-07 |
| `models/deepseek_v2/configuration_deepseek_v2.py` | 143 | L9-07 |
| `models/deepseek_v2/modeling_deepseek_v2.py` | 660 | L9-07 |
| `models/glm4v/__init__.py` | 32 | L9-07 |
| `models/glm4v/configuration_glm4v.py` | 193 | L9-07 |
| `models/glm4v/modeling_glm4v.py` | 1587 | L9-07 |
| `models/glm4v_moe/__init__.py` | 28 | L9-07 |
| `models/glm4v_moe/configuration_glm4v_moe.py` | 223 | L9-07 |
| `models/glm4v_moe/modeling_glm4v_moe.py` | 1828 | L9-07 |
| `models/glm5_next/__init__.py` | 33 | L9-07 |
| `models/glm5_next/configuration_glm5_next.py` | 322 | L0-02, L0-03, L1-02, L1-03 |
| `models/glm5_next/image_processing_glm5_next.py` | 323 | L2-06 |
| `models/glm5_next/image_processing_pil_glm5_next.py` | 323 | L2-06 |
| `models/glm5_next/modeling_glm5_next.py` | 2445 | L0-01, L0-04, L0-05, L2-07, L3-01, L3-02, L3-03, L3-04, L3-05, L3-06, L3-07, L3-08, L4-01, L4-02, L4-03, L4-04, L4-05, L4-06, L4-07, L4-08, L7-03 |
| `models/glm5_next/processing_glm5_next.py` | 194 | L2-06 |
| `models/glm5_next/video_processing_glm5_next.py` | 430 | L2-06 |
| `models/glm_ocr/__init__.py` | 29 | L9-07 |
| `models/glm_ocr/configuration_glm_ocr.py` | 195 | L9-07 |
| `models/glm_ocr/modeling_glm_ocr.py` | 1513 | L9-07 |
| `models/kimi_k25/__init__.py` | 32 | L9-07 |
| `models/kimi_k25/configuration_kimi_k25.py` | 102 | L9-07 |
| `models/kimi_k25/modeling_kimi_k25.py` | 850 | L9-07 |
| `models/qwen3_5/__init__.py` | 29 | L9-07 |
| `models/qwen3_5/configuration_qwen3_5.py` | 202 | L9-07 |
| `models/qwen3_5/modeling_qwen3_5.py` | 2117 | L9-07 |
| `models/qwen3_5/tokenization_qwen3_5.py` | 95 | L9-07 |
| `models/qwen3_5_moe/__init__.py` | 28 | L9-07 |
| `models/qwen3_5_moe/configuration_qwen3_5_moe.py` | 215 | L9-07 |
| `models/qwen3_5_moe/modeling_qwen3_5_moe.py` | 2292 | L9-07 |
| `monkey_patching.py` | 372 | L8-01 |
| `processing_utils.py` | 2431 | L2-06 |
| `pytorch_utils.py` | 256 | L3-01 |
| `quantizers/__init__.py` | 17 | L7-04 |
| `quantizers/auto.py` | 392 | L7-04 |
| `quantizers/base.py` | 347 | L7-04 |
| `quantizers/quantizer_aqlm.py` | 76 | L7-05 |
| `quantizers/quantizer_auto_round.py` | 72 | L7-06 |
| `quantizers/quantizer_awq.py` | 98 | L7-05 |
| `quantizers/quantizer_bitnet.py` | 125 | L7-06 |
| `quantizers/quantizer_bnb_4bit.py` | 188 | L7-05 |
| `quantizers/quantizer_bnb_8bit.py` | 179 | L7-05 |
| `quantizers/quantizer_compressed_tensors.py` | 227 | L7-06 |
| `quantizers/quantizer_eetq.py` | 111 | L7-05 |
| `quantizers/quantizer_fbgemm_fp8.py` | 207 | L7-06 |
| `quantizers/quantizer_finegrained_fp8.py` | 307 | L7-06 |
| `quantizers/quantizer_fouroversix.py` | 124 | L7-06 |
| `quantizers/quantizer_fp_quant.py` | 165 | L7-06 |
| `quantizers/quantizer_gemma.py` | 76 | L7-06 |
| `quantizers/quantizer_gguf.py` | 198 | L7-07 |
| `quantizers/quantizer_gptq.py` | 112 | L7-05 |
| `quantizers/quantizer_higgs.py` | 179 | L7-06 |
| `quantizers/quantizer_hqq.py` | 265 | L7-05 |
| `quantizers/quantizer_metal.py` | 131 | L7-06 |
| `quantizers/quantizer_mxfp4.py` | 318 | L7-06 |
| `quantizers/quantizer_nvfp4.py` | 123 | L7-06 |
| `quantizers/quantizer_quanto.py` | 123 | L7-06 |
| `quantizers/quantizer_quark.py` | 105 | L7-05 |
| `quantizers/quantizer_sinq.py` | 263 | L7-06 |
| `quantizers/quantizer_spqr.py` | 82 | L7-05 |
| `quantizers/quantizer_torchao.py` | 229 | L7-06 |
| `quantizers/quantizer_vptq.py` | 76 | L7-05 |
| `quantizers/quantizers_utils.py` | 69 | L7-04 |
| `safetensors_conversion.py` | 117 | L7-07 |
| `tokenization_python.py` | 1425 | L2-02 |
| `tokenization_utils_base.py` | 3691 | L2-01 |
| `tokenization_utils_sentencepiece.py` | 316 | L2-03 |
| `tokenization_utils_tokenizers.py` | 1515 | L2-02 |
| `trainer_utils.py` | 1250 | L9-04 |
| `utils/__init__.py` | 352 | L9-03 |
| `utils/attention_visualizer.py` | 256 | L9-04 |
| `utils/auto_docstring.py` | 4800 | L1-04 |
| `utils/backbone_utils.py` | 20 | L2-07 |
| `utils/chat_parsing/__init__.py` | 21 | L9-05 |
| `utils/chat_parsing/content_parsers.py` | 250 | L9-05 |
| `utils/chat_parsing/response_parser.py` | 403 | L9-05 |
| `utils/chat_parsing/response_templates.py` | 242 | L9-05 |
| `utils/chat_template_utils.py` | 632 | L9-05 |
| `utils/constants.py` | 7 | L9-03 |
| `utils/deprecation.py` | 176 | L9-03 |
| `utils/doc.py` | 1092 | L1-04 |
| `utils/dummy_mistral_common_objects.py` | 10 | L9-04 |
| `utils/dummy_sentencepiece_and_tokenizers_objects.py` | 10 | L9-04 |
| `utils/generic.py` | 1211 | L9-03 |
| `utils/hp_naming.py` | 163 | L1-04 |
| `utils/hub.py` | 1050 | L9-03 |
| `utils/import_utils.py` | 3377 | L9-03 |
| `utils/kernel_config.py` | 350 | L9-04 |
| `utils/loading_report.py` | 319 | L9-04 |
| `utils/logging.py` | 442 | L9-03 |
| `utils/output_capturing.py` | 319 | L9-04 |
| `utils/peft_utils.py` | 115 | L9-04 |
| `utils/quantization_config.py` | 2106 | L7-04 |
| `utils/type_validators.py` | 271 | L1-04 |
| `utils/versions.py` | 117 | L9-03 |
| `video_processing_utils.py` | 796 | L2-05 |
| `video_utils.py` | 928 | L2-05 |
| `vision_utils.py` | 380 | L2-05 |

---

## 4. 验收门禁

| 门禁 | 脚本 | 断言 |
|---|---|---|
| A1/A3 保真 | `tools/check_fidelity.py` | 每个引用块逐字来自标注源文件，且位置连续 |
| A2 参数 | `tools/check_params.py` | 每个 `--flag` 都在实测真值集里 |
| A4 语法 | `tools/lint_lessons.py` | 四文件齐备、`node --check` 通过、分幕字段完整 |
| B 覆盖 | `tools/check_coverage.py` | 差集为空、无空课、无幻影引用 |
| C 渲染 | `tools/check_render.py` | 两分辨率 × 3 时间点，0 JS 错误 / 0 溢出 / 交互可用 |

顺序：`lint` → `split_blocks` → `fidelity` → `params` → `coverage` → `render`

