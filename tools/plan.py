#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GLM-5 推理课件 —— 计划的唯一真源 (single source of truth)。

本文件定义"分层课程清单 + 课↔源文件分配"，并由 gen_plan.py 渲染成 TODOLIST.md。
覆盖度门禁 (check_coverage.py) 也用同一份数据交叉校验，避免计划与实现漂移。

视角 (perspective)
------------------
    一个 token 从进入模型到吐出下一个 token，GLM-5 在 transformers 里到底发生了什么。

一句话概括仓库在解决的问题
--------------------------
    把"磁盘上的一堆权重 + 一段文本/图像"变成"一次可复现、可优化、可横向扩展的数值计算"。

心智模型的转折点 (turning point)
--------------------------------
    L0–L4 是**数学视角**：数据是张量，模型是算子组合 —— 问"算什么"。
    L5 起切换为**机器视角**：数据是字节，计算是访存 —— 问"算这个要花多少"。
    同一个 KV cache，在 L4 是三行公式，在 L5 是一块按页分配、可量化、可换出的显存。
"""

# ===========================================================================
# 覆盖域 (universe) —— 实测冻结，见 _data/universe.json
# ===========================================================================
UNIVERSE_JSON = "_data/universe.json"
UPSTREAM = "../upstream-transformers/src/transformers"

LAYERS = [
    ("L0", "预备篇 · 直觉", "把 45 层的模型压缩成一张能记住的图",
     "从一次 generate() 出发，先建立「这个模型长什么样」的整体直觉，不碰任何优化。"),
    ("L1", "第 1 层 · 配置与分派", "config 不是参数表，是架构的可执行定义",
     "看懂一个 config 如何决定张量形状，以及 AutoModel 如何找到 GLM-5。"),
    ("L2", "第 2 层 · 输入流水线", "文本和像素如何变成同一串位置上的向量",
     "分词 → 图像/视频处理 → 视觉塔 → 占位符展开，多模态输入如何在序列里对齐。"),
    ("L3", "第 3 层 · 文本主干", "一层之内到底有哪几个算子",
     "RMSNorm / MoE / mHC / 门控范数的逐个拆解，重点是「为什么这样设计」。"),
    ("L4", "第 4 层 · 混合注意力 ★", "34 层线性注意力 + 11 层稀疏注意力如何协同",
     "KDA 的循环与分块两种算法、DSA 索引器如何选 2048 个位置、MLA 如何压缩 KV。"),
    ("L5", "第 5 层 · 缓存与内存 ★★", "CPU 视角的 KV 记的是公式，GPU 视角记的是字节",
     "转折点。同样的 KV，这里要问：占多少显存、按什么粒度分配、溢出时怎么办。"),
    ("L6", "第 6 层 · 生成循环", "一个 token 是怎么被「决定」出来的",
     "generate 主循环、logits 处理、停止条件、投机解码、连续批处理调度。"),
    ("L7", "第 7 层 · 权重加载与量化", "磁盘上的字节如何变成显存里的张量",
     "from_pretrained 的完整路径、权重名映射、以及 30 个量化器各自的取舍。"),
    ("L8", "第 8 层 · 推理规模化", "单卡跑不动时，模型被怎么切开",
     "张量/专家/流水线并行计划，注意力后端分派，异构设备与内核选择。"),
    ("L9", "第 9 层 · 出口与周边", "推理之外，但推理离不开的那些文件",
     "导出器、运行期基础设施、对话模板、量化集成、模型族继承链，以及全量收束。"),
]

LAYER_DIR = {
    "L0": "L0-intuition", "L1": "L1-config", "L2": "L2-input", "L3": "L3-backbone",
    "L4": "L4-attention", "L5": "L5-cache", "L6": "L6-generation", "L7": "L7-loading",
    "L8": "L8-scaling", "L9": "L9-periphery",
}

def lesson_dir(L):
    return LAYER_DIR[L["layer"]] + "/" + L["id"]

# ===========================================================================
# 课程表
#   每课: id, layer, title, priority, files(声明覆盖的源文件), points(讲解要点),
#         accept(验收点)
#   files 里以 "~" 结尾表示"该目录下全部文件"(由 gen_plan 展开)
# ===========================================================================
LESSONS = [

# ---------------------------------------------------------------- L0 预备篇
dict(id="L0-01", layer="L0", prio="P0",
     title="一次 generate() 里发生了什么",
     files=["models/glm5_next/modeling_glm5_next.py"],
     points=["把 generate() 拆成 6 个阶段：取 token → 嵌入 → 45 层 → 归一 → 取 logits → 采样",
             "实测参数量与 45 层排布，建立「这模型多大」的量感",
             "标出后面每一层课程对应这张全景图的哪一块"],
     accept=["能不看笔记复述一次前向的 6 个阶段",
             "能说出 45 层里有多少层跑线性注意力"]),

dict(id="L0-02", layer="L0", prio="P0",
     title="配置即架构：从 config 读出全部形状",
     files=["models/glm5_next/configuration_glm5_next.py"],
     points=["hidden=4096 / 64 头 / qk_nope=256 / v=256 如何拼出注意力形状",
             "head_dim 被强制等于 qk_rope_head_dim(=0)，这意味着 DSA 层完全不用 RoPE",
             "mlp_layer_types：前 3 层 dense，其余 42 层 sparse"],
     accept=["能从 config 默认值手算出 MLA 层的 Q/K/V 形状",
             "能解释为什么 head_dim 是 0 而不是 256"]),

dict(id="L0-03", layer="L0", prio="P0",
     title="层类型排布：34 层 KDA + 11 层 MLA",
     files=["models/glm5_next/configuration_glm5_next.py"],
     points=["layer_types 的生成规律 idx % 4 != 3 → linear_attention",
             "indexer_types：哪些层跑索引器（full），哪些复用上一层（shared）",
             "为什么是 3:1 而不是 1:1 —— 长上下文服务的成本结构"],
     accept=["给定 num_hidden_layers 能写出 layer_types",
             "能解释 shared 索引器省掉了什么"]),

dict(id="L0-04", layer="L0", prio="P0",
     title="MoE 经济学：288 选 8",
     files=["models/glm5_next/modeling_glm5_next.py"],
     points=["n_routed_experts=288, top-k=8, +1 共享专家 → 激活比约 3%",
             "routed_scaling_factor=2.5 的作用",
             "n_group / topk_group 分组路由的动机"],
     accept=["能算出稀疏层与稠密层的 FLOPs 比",
             "能说出共享专家存在的理由"]),

dict(id="L0-05", layer="L0", prio="P1",
     title="mHC：4 条残差流",
     files=["models/glm5_next/modeling_glm5_next.py"],
     points=["hc_mult=4 —— 残差不是一条而是一束",
             "Sinkhorn 迭代 20 次把路由矩阵约束到双随机流形上",
             "为什么这比单条残差更难训、但扩展性更好"],
     accept=["能用自己的话说出双随机约束的作用",
             "能指出 mHC 在解码层的前后各出现一次"]),

# ------------------------------------------------------- L1 配置与分派
dict(id="L1-01", layer="L1", prio="P0",
     title="PreTrainedConfig：配置对象的公共契约",
     files=["configuration_utils.py"],
     points=["from_pretrained / to_dict / to_json_string 的完整路径",
             "@strict 装饰器做了什么校验",
             "sub_configs 与 base_config_key：嵌套配置的寻址规则"],
     accept=["能画出 config 从 json 到对象再到模型的路径",
             "能解释 strict 校验在什么时机触发"]),

dict(id="L1-02", layer="L1", prio="P0",
     title="Glm5NextTextConfig 逐字段精读",
     files=["models/glm5_next/configuration_glm5_next.py"],
     points=["每个字段属于哪一组：注意力 / MoE / KDA / DSA / mHC",
             "validate_architecture 的四条硬约束",
             "attribute_map 与 keys_to_ignore_at_inference 的用途"],
     accept=["能说出四条校验各自防的是什么错误",
             "能列出 KDA 相关的 5 个字段"]),

dict(id="L1-03", layer="L1", prio="P1",
     title="视觉配置与顶层嵌套",
     files=["models/glm5_next/configuration_glm5_next.py"],
     points=["Glm5NextVisionConfig：depth=24, patch=14, merge=2 → 每 token 覆盖 28 像素",
             "Glm5NextConfig 如何把 text/vision 两个子配置拼起来",
             "扁平 checkpoint 的向后兼容分支"],
     accept=["能算出 336x336 图像产生多少个视觉 token",
             "能解释 image_token_id 与 image_start/end_token_id 的分工"]),

dict(id="L1-04", layer="L1", prio="P1",
     title="配置的序列化、继承与文档生成",
     files=["utils/hp_naming.py", "utils/doc.py", "utils/auto_docstring.py",
            "utils/type_validators.py"],
     points=["hp_naming 如何把自定义类名映射回上游名字",
             "doc.py 的文档字符串机械替换",
             "type_validators 如何把类型注解变成运行期校验"],
     accept=["能说出 auto_docstring 在类定义时改了什么",
             "能解释为什么需要 hp_naming 这层间接"]),

dict(id="L1-05", layer="L1", prio="P0",
     title="AutoModel 分派：字符串如何找到类",
     files=["models/auto/__init__.py", "models/auto/configuration_auto.py",
            "models/auto/auto_factory.py", "models/auto/auto_mappings.py"],
     points=["CONFIG_MAPPING 的惰性注册表",
             "auto_factory 的 _LazyAutoMapping 如何做到「用时才导入」",
             "从 model_type 字符串到 Glm5NextConfig 的完整链路"],
     accept=["能画出 AutoConfig.from_pretrained('glm5_next') 的调用链",
             "能解释惰性映射省下了什么"]),

dict(id="L1-06", layer="L1", prio="P1",
     title="其余四张自动注册表",
     files=["models/auto/modeling_auto.py", "models/auto/tokenization_auto.py",
            "models/auto/image_processing_auto.py", "models/auto/video_processing_auto.py",
            "models/auto/feature_extraction_auto.py", "models/auto/processing_auto.py"],
     points=["模型/分词器/图像/视频/处理器五种 Auto 类的对称结构",
             "AutoProcessor 如何把多个子处理器组合起来",
             "命名冲突时的解析优先级"],
     accept=["能说出 AutoProcessor 对 GLM-5 会实例化哪几个子处理器"]),

dict(id="L1-07", layer="L1", prio="P1",
     title="动态模块、类型契约与权重名映射表",
     files=["dynamic_module_utils.py", "_typing.py", "conversion_mapping.py"],
     points=["trust_remote_code 路径下的动态导入",
             "_typing 里的 TensorType 等别名如何统一注解",
             "conversion_mapping 如何描述跨实现权重名对应关系"],
     accept=["能解释动态模块的缓存与校验策略"]),

# ------------------------------------------------------- L2 输入流水线
dict(id="L2-01", layer="L2", prio="P0",
     title="分词器公共契约：慢与快共享的那一层",
     files=["tokenization_utils_base.py"],
     points=["SpecialTokensMixin 与 token 属性体系",
             "__call__ 的参数如何映射到张量",
             "padding / truncation 策略的完整枚举"],
     accept=["能说出 input_ids 与 attention_mask 的产生时机",
             "能解释 return_tensors 影响的是哪一步"]),

dict(id="L2-02", layer="L2", prio="P0",
     title="快分词器、纯 Python 分词器与慢→快转换",
     files=["tokenization_utils_tokenizers.py", "tokenization_python.py",
            "convert_slow_tokenizer.py"],
     points=["tokenizers 库后端的批量编码",
             "纯 Python 回退路径的存在意义",
             "convert_slow_tokenizer 如何把 SentencePiece 模型翻译成 fast tokenizer"],
     accept=["能说出 fast/slow 在 GLM-5 上的行为差异点"]),

dict(id="L2-03", layer="L2", prio="P1",
     title="SentencePiece 分词器",
     files=["tokenization_utils_sentencepiece.py"],
     points=["unigram / BPE 两种模型的加载",
             "legacy 行为开关",
             "为什么 GLM-5 的 tokenizer 往往走这个基类"],
     accept=["能解释 add_dummy_prefix 对首 token 的影响"]),

dict(id="L2-04", layer="L2", prio="P0",
     title="图像处理基座：尺寸、归一化与后端",
     files=["image_processing_base.py", "image_processing_utils.py",
            "image_processing_backends.py"],
     points=["BaseImageProcessor 的 preprocess 契约",
             "三个后端（PIL / torchvision / numpy）的取舍",
             "size / resample / rescale 的组合语义"],
     accept=["能说出默认后端的选择顺序"]),

dict(id="L2-05", layer="L2", prio="P0",
     title="图像变换、视频与通用视觉工具",
     files=["image_transforms.py", "image_utils.py", "video_processing_utils.py",
            "video_utils.py", "vision_utils.py", "audio_utils.py"],
     points=["resize / center_crop / normalize 的张量级实现",
             "视频按帧采样与 temporal patch 的关系",
             "vision_utils 的通用工具集"],
     accept=["能算出 336x336 图像经 patch+merge 后的 token 数",
             "能解释 temporal_patch_size=2 对视频帧数的要求"]),

dict(id="L2-06", layer="L2", prio="P0",
     title="多模态处理器装配",
     files=["processing_utils.py", "models/glm5_next/processing_glm5_next.py",
            "models/glm5_next/image_processing_glm5_next.py",
            "models/glm5_next/image_processing_pil_glm5_next.py",
            "models/glm5_next/video_processing_glm5_next.py",
            "feature_extraction_utils.py"],
     points=["ProcessorMixin 如何把 tokenizer + image/video processor 拼起来",
             "GLM-5 的 chat template 如何插入视觉占位符",
             "两套图像处理器实现（默认 / PIL）的分工"],
     accept=["能写出一次图像对话的 inputs 字典有哪些 key"]),

dict(id="L2-07", layer="L2", prio="P0",
     title="视觉塔：从 336x336 像素到 27 个 token",
     files=["models/glm5_next/modeling_glm5_next.py", "backbone_utils.py",
            "utils/backbone_utils.py"],
     points=["patch embed 的卷积等价形式",
             "轴向 RoPE：height/width 两套频率的重组",
             "patch merger 的空间 2x2 合并与投影"],
     accept=["能手算 (336/14)^2 / 2^2 = 144... 并核对实际 token 数",
             "能解释轴向 RoPE 与一维 RoPE 的区别"]),

# ------------------------------------------------------- L3 文本主干
dict(id="L3-01", layer="L3", prio="P0",
     title="三种 RMSNorm",
     files=["models/glm5_next/modeling_glm5_next.py", "pytorch_utils.py"],
     points=["Glm5NextTextRMSNorm 与 Unweighted 的差别",
             "Glm5NextTextRMSNormGated：KDA 输出为什么需要门控",
             "eps 取 1e-5 还是 1e-6 的实际影响"],
     accept=["能说出无权重版本的用途",
             "能解释门控范数在 KDA 中解决什么问题"]),

dict(id="L3-02", layer="L3", prio="P0",
     title="MLP 与 SwiGLU 截断",
     files=["models/glm5_next/modeling_glm5_next.py", "activations.py"],
     points=["gate/up/down 三投影的 SwiGLU 结构",
             "swiglu_limit=10.0 的 clamp 作用与数值动机",
             "activations 里的 ACT2FN 注册表"],
     accept=["能写出 SwiGLU 的前向公式",
             "能解释截断为什么能提升大模型稳定性"]),

dict(id="L3-03", layer="L3", prio="P0",
     title="路由器：288 个分数如何变成 8 条路径",
     files=["models/glm5_next/modeling_glm5_next.py"],
     points=["gate 线性层 + softmax 的完整流程",
             "n_group / topk_group 的两级筛选",
             "norm_topk_prob 与 routed_scaling_factor 的数值效果"],
     accept=["能手算一次 top-8 选择",
             "能解释分组路由在分布式下的好处"]),

dict(id="L3-04", layer="L3", prio="P0",
     title="专家层与 grouped GEMM",
     files=["models/glm5_next/modeling_glm5_next.py"],
     points=["gate_up_proj 合并打包的存储布局",
             "_apply_gate 里 swiglu_limit 的二次截断",
             "为什么专家权重按 (E, ...) 排布才能用 grouped GEMM"],
     accept=["能说出打包存储省下了什么",
             "能解释权重布局与 EP 切分的关系"]),

dict(id="L3-05", layer="L3", prio="P0",
     title="MoE 装配与共享专家",
     files=["models/glm5_next/modeling_glm5_next.py"],
     points=["Glm5NextTextMoE 的前向：共享专家与路由专家的合并方式",
             "稠密层走 Glm5NextTextMLP 的分支条件",
             "output_router_logits 在推理时的影响"],
     accept=["能画出 MoE 前向的数据流",
             "能解释共享专家为什么能稳住训练"]),

dict(id="L3-06", layer="L3", prio="P0",
     title="★ mHC：流形约束超连接",
     files=["models/glm5_next/modeling_glm5_next.py"],
     points=["4 条残差流的读取、混合、写回三步",
             "Sinkhorn 迭代 20 次得到双随机矩阵的过程",
             "HyperHead 如何把多条流收回一条"],
     accept=["能手算 2 次 Sinkhorn 迭代",
             "能解释双随机约束对梯度流的稳定作用"]),

dict(id="L3-07", layer="L3", prio="P0",
     title="遗忘门与门控范数",
     files=["models/glm5_next/modeling_glm5_next.py"],
     points=["ForgetGate 的线性投影 + 下界截断",
             "linear_lower_bound=-5.0 的数值意图",
             "A_log / dt_bias 这类参数的初始化来源"],
     accept=["能解释下界截断防的是什么",
             "能说出遗忘门在 KDA 递推中的位置"]),

dict(id="L3-08", layer="L3", prio="P0",
     title="解码层装配：三个子块的顺序",
     files=["models/glm5_next/modeling_glm5_next.py", "modeling_layers.py",
            "modeling_outputs.py"],
     points=["mHC → 注意力 → mHC → MoE 的完整顺序",
             "GradientCheckpointingLayer 在推理时的实际行为",
             "modeling_outputs 的数据类约定"],
     accept=["能默写解码层的执行顺序",
             "能说出 residual 在 mHC 下与标准 Transformer 的差异"]),

# ------------------------------------------------------- L4 混合注意力
dict(id="L4-01", layer="L4", prio="P0",
     title="★ 为什么 GLM-5 要混合两种注意力",
     files=["models/glm5_next/modeling_glm5_next.py"],
     points=["全注意力的 KV 线性增长 vs 线性注意力的常数状态",
             "3:1 排布如何兼顾精确检索与长文本成本",
             "NoPE 的取舍：位置信息交给 KDA 与索引器"],
     accept=["能画出两种层的成本曲线",
             "能解释为什么 DSA 层不需要 RoPE"]),

dict(id="L4-02", layer="L4", prio="P0",
     title="KDA 循环形式：逐 token 的递推",
     files=["models/glm5_next/modeling_glm5_next.py"],
     points=["recurrent_kimi_delta_attention 的状态更新公式",
             "delta 规则与普通线性注意力的区别",
             "状态矩阵的形状 (head_dim, v_head_dim)"],
     accept=["能手推两步状态更新",
             "能说出 delta 规则修正了什么"]),

dict(id="L4-03", layer="L4", prio="P0",
     title="KDA 分块形式：把递推变成矩阵乘",
     files=["models/glm5_next/modeling_glm5_next.py"],
     points=["chunk_kimi_delta_attention 的分块矩阵化",
             "块内并行、块间递推的两级结构",
             "为什么训练用分块、解码用循环"],
     accept=["能解释分块如何提升 GPU 利用率",
             "能说出两种形式在数学上等价的条件"]),

dict(id="L4-04", layer="L4", prio="P0",
     title="短卷积与 L2 归一",
     files=["models/glm5_next/modeling_glm5_next.py"],
     points=["causal_conv1d_fn 与 update 的分工（prefill / decode）",
             "kernel=4 的深度可分离卷积在 KDA 中的作用",
             "l2norm 对 q/k 的归一化"],
     accept=["能说出 prefill 与 decode 走不同实现的理由",
             "能解释短卷积补齐了什么信息"]),

dict(id="L4-05", layer="L4", prio="P0",
     title="线性注意力层装配",
     files=["models/glm5_next/modeling_glm5_next.py"],
     points=["q/k/v 投影 + 短卷积 + 门控 + 范数的完整前向",
             "apply_mask_to_padding_states 的用途",
             "缓存状态如何读写"],
     accept=["能画出 KDA 层的完整数据流",
             "能说出每步的张量形状"]),

dict(id="L4-06", layer="L4", prio="P0",
     title="★ DSA 索引器：2048 个位置是怎么选出来的",
     files=["models/glm5_next/modeling_glm5_next.py"],
     points=["index_head_dim=128, index_n_heads=32 的小型注意力",
             "index_kpool=16：先池化再选择的粗筛",
             "get_visible_tokens / get_pooled_states / append_visible_tail 三步",
             "full 与 shared 两种 indexer_types"],
     accept=["能算出索引器的选择预算",
             "能解释池化粗筛省下了多少计算"]),

dict(id="L4-07", layer="L4", prio="P0",
     title="MLA：把 KV 压进 512 维隐向量",
     files=["models/glm5_next/modeling_glm5_next.py", "modeling_rope_utils.py"],
     points=["q_lora_rank=1536 / kv_lora_rank=512 的压缩路径",
             "expand_kv 如何从隐向量还原出多头 K/V",
             "NoPE 下 head_dim=0 导致的实现细节"],
     accept=["能算出 MLA 相对 MHA 的 KV 压缩比",
             "能解释 expand_kv 为什么不能省"]),

dict(id="L4-08", layer="L4", prio="P0",
     title="掩码构造与注意力计算",
     files=["models/glm5_next/modeling_glm5_next.py",
            "modeling_flash_attention_utils.py"],
     points=["build_attention_mask_from_topk 如何把索引变成掩码",
             "eager_attention_forward 与 repeat_kv 的参考实现",
             "GQA 的 n_rep 展开"],
     accept=["能写出掩码的布尔表达式",
             "能说出参考实现与融合后端的差异"]),

# ------------------------------------------------------- L5 缓存与内存 ★★
dict(id="L5-01", layer="L5", prio="P0",
     title="★ 转折点：Cache 抽象层",
     files=["cache_utils.py"],
     points=["DynamicCache 的 layers 列表契约",
             "为什么每种层类型需要自己的 Cache 类",
             "record_past / 预分配 两种策略"],
     accept=["能说出 Cache 抽象的三条不变量",
             "能解释为什么不能用统一结构装两种层"]),

dict(id="L5-02", layer="L5", prio="P0",
     title="★ 线性注意力层的缓存：常数状态",
     files=["cache_utils.py"],
     points=["conv_states 与 recurrent_states 的形状",
             "为什么 KDA 层的显存与序列长度无关",
             "has_previous_state 标志的作用"],
     accept=["能手算 KDA 层每 token 的额外显存",
             "能解释 conv 状态为什么必须单独存"]),

dict(id="L5-03", layer="L5", prio="P0",
     title="★ 索引注意力层的缓存：KV + 索引键",
     files=["cache_utils.py"],
     points=["keys/values 的追加式增长",
             "indexer_keys 的独立存储与 dtype",
             "每 token 显存的完整算式"],
     accept=["能手算 32K 上下文下 MLA 层的 KV 显存",
             "能说出 indexer_keys 为什么可以低精度"]),

dict(id="L5-04", layer="L5", prio="P0",
     title="分页缓存：按块分配与复用",
     files=["generation/continuous_batching/cache.py",
            "generation/continuous_batching/cache_allocators/cache_pool.py",
            "generation/continuous_batching/cache_allocators/cache_allocator.py",
            "generation/continuous_batching/cache_allocators/__init__.py"],
     points=["page/block 的粒度与碎片",
             "池化分配器的空闲链表",
             "引用计数与共享前缀"],
     accept=["能解释分页相比连续分配省下什么",
             "能说出块大小对碎片率的影响"]),

dict(id="L5-05", layer="L5", prio="P0",
     title="掩码生成：从 padding 到因果",
     files=["masking_utils.py", "modeling_attn_mask_utils.py"],
     points=["4D 因果掩码与 2D padding 掩码的合成",
             "滑动窗口掩码的构造",
             "sdpa / flash 各自需要的掩码形态"],
     accept=["能手写一个 (1,1,q,k) 因果掩码",
             "能解释不同后端对掩码形状的要求"]),

dict(id="L5-06", layer="L5", prio="P1",
     title="连续批处理的缓存分配策略",
     files=["generation/continuous_batching/cache_allocators/full_attention.py",
            "generation/continuous_batching/cache_allocators/sliding_attention.py",
            "generation/continuous_batching/cache.py"],
     points=["全注意力与滑窗注意力的块生命周期差异",
             "滑窗导致的块回收时机",
             "前缀共享在批内的实现"],
     accept=["能画出两种策略的块状态机"]),

# ------------------------------------------------------- L6 生成循环
dict(id="L6-01", layer="L6", prio="P0",
     title="generate 主循环",
     files=["generation/utils.py", "generation/__init__.py"],
     points=["prepare_inputs → prefill → decode 循环 → finalize",
             "cache_position 的计算与更新",
             "stateful 模型（GLM-5）走的分支"],
     accept=["能画出 generate 的主循环流程图",
             "能解释 cache_position 的作用"]),

dict(id="L6-02", layer="L6", prio="P0",
     title="GenerationConfig：采样参数的容器",
     files=["generation/configuration_utils.py"],
     points=["采样相关字段的完整清单与互斥规则",
             "validate 阶段的参数校验",
             "generation_config 与 model.config 的关系"],
     accept=["能说出 temperature=0 时实际走了哪条分支"]),

dict(id="L6-03", layer="L6", prio="P0",
     title="logits 处理链",
     files=["generation/logits_process.py"],
     points=["处理器按优先级串联的顺序（这决定了结果）",
             "top-k / top-p / min-p 的累积概率实现",
             "重复惩罚与存在惩罚的差别"],
     accept=["能说出 top-p 与 top-k 同时开启时的实际顺序",
             "能手算一次重复惩罚"]),

dict(id="L6-04", layer="L6", prio="P0",
     title="停止条件与流式输出",
     files=["generation/stopping_criteria.py", "generation/streamers.py"],
     points=["MaxLength / EOS / StopString 三类条件的组合",
             "流式输出的增量解码",
             "stopping criteria 在批处理下的向量化"],
     accept=["能说出多个停止条件是或还是与",
             "能解释流式解码为什么要处理多字节字符"]),

dict(id="L6-05", layer="L6", prio="P1",
     title="投机解码",
     files=["generation/candidate_generator.py"],
     points=["草稿模型与主模型的对齐",
             "为什么 GLM-5 不支持 assisted generation",
             "候选生成的接受/拒绝逻辑"],
     accept=["能解释接受率与加速比的关系"]),

dict(id="L6-06", layer="L6", prio="P1",
     title="水印",
     files=["generation/watermarking.py"],
     points=["logits 偏置式水印的原理",
             "水印强度与文本质量的权衡"],
     accept=["能说出水印对困惑度的影响方向"]),

dict(id="L6-07", layer="L6", prio="P0",
     title="连续批处理：调度器与模型运行器",
     files=["generation/continuous_batching/__init__.py",
            "generation/continuous_batching/scheduler.py",
            "generation/continuous_batching/model_runner.py",
            "generation/continuous_batching/continuous_api.py",
            "generation/continuous_batching/requests.py",
            "generation/continuous_batching/input_outputs.py"],
     points=["请求队列 → 调度 → 前向 → 回写 的闭环",
             "continuous batching 相比静态批处理省下什么",
             "请求的优先级与抢占"],
     accept=["能画出调度器的主循环",
             "能解释静态批处理下 GPU 空转的原因"]),

dict(id="L6-08", layer="L6", prio="P1",
     title="连续批处理：处理器、分发与卸载",
     files=["generation/continuous_batching/cb_logits_processors.py",
            "generation/continuous_batching/initialization.py",
            "generation/continuous_batching/distributed.py",
            "generation/continuous_batching/utils.py",
            "generation/continuous_batching/offloading_manager.py"],
     points=["批内 logits 处理器如何在调度器里执行",
             "多进程/多卡下的请求分发",
             "权重卸载到 CPU 的时机"],
     accept=["能说出卸载管理器在显存不足时的决策顺序"]),

# ------------------------------------------------------- L7 加载与量化
dict(id="L7-01", layer="L7", prio="P0",
     title="from_pretrained 全景",
     files=["modeling_utils.py", "initialization.py"],
     points=["config → 模型骨架 → 权重下载 → 加载 → 后处理 的完整阶段",
             "device_map 与 dtype 的解析顺序",
             "低内存加载（meta device）路径"],
     accept=["能画出 from_pretrained 的 6 个阶段",
             "能解释 meta device 省下了什么"]),

dict(id="L7-02", layer="L7", prio="P0",
     title="核心加载器：权重名映射",
     files=["core_model_loading.py"],
     points=["checkpoint 名 → 模型参数名 的映射规则",
             "量化权重的反量化与拼接",
             "加载报告中的 missing / unexpected 分类"],
     accept=["能解释一个权重名是如何被解析到目标模块的"]),

dict(id="L7-03", layer="L7", prio="P0",
     title="张量并行计划与模型输出",
     files=["modeling_layers.py", "modeling_outputs.py",
            "models/glm5_next/modeling_glm5_next.py"],
     points=["base_model_tp_plan 的 colwise/rowwise 标注",
             "base_model_ep_plan 的 grouped_gemm 标注",
             "模型输出数据类的字段"],
     accept=["能把 GLM-5 的 tp_plan 逐条对应到模块",
             "能解释 colwise 与 rowwise 的配对规则"]),

dict(id="L7-04", layer="L7", prio="P0",
     title="量化基座：Quantizer 契约",
     files=["quantizers/__init__.py", "quantizers/base.py", "quantizers/auto.py",
            "quantizers/quantizers_utils.py",
            "utils/quantization_config.py"],
     points=["HfQuantizer 的 6 个钩子",
             "auto 如何从 config 选出量化器",
             "量化配置的校验与默认值"],
     accept=["能说出量化器在加载流程中的介入时机",
             "能解释 pre_quantized 与 on-the-fly 的差别"]),

dict(id="L7-05", layer="L7", prio="P1",
     title="量化器家族（一）：GPU 主流方案",
     files=["quantizers/quantizer_bnb_4bit.py", "quantizers/quantizer_bnb_8bit.py",
            "quantizers/quantizer_gptq.py", "quantizers/quantizer_awq.py",
            "quantizers/quantizer_eetq.py", "quantizers/quantizer_hqq.py",
            "quantizers/quantizer_aqlm.py", "quantizers/quantizer_vptq.py",
            "quantizers/quantizer_spqr.py", "quantizers/quantizer_quark.py"],
     points=["每种的位宽、粒度（per-tensor/channel/group）与是否需要校准",
             "推理时反量化发生在哪一步",
             "哪种适合 GLM-5 这类 MoE"],
     accept=["能说出 4bit 分组量化的分组大小对精度的影响"]),

dict(id="L7-06", layer="L7", prio="P1",
     title="量化器家族（二）：新格式与实验方案",
     files=["quantizers/quantizer_quanto.py", "quantizers/quantizer_torchao.py",
            "quantizers/quantizer_finegrained_fp8.py", "quantizers/quantizer_fbgemm_fp8.py",
            "quantizers/quantizer_fp_quant.py", "quantizers/quantizer_mxfp4.py",
            "quantizers/quantizer_nvfp4.py", "quantizers/quantizer_compressed_tensors.py",
            "quantizers/quantizer_auto_round.py", "quantizers/quantizer_bitnet.py",
            "quantizers/quantizer_higgs.py", "quantizers/quantizer_sinq.py",
            "quantizers/quantizer_metal.py", "quantizers/quantizer_gemma.py",
            "quantizers/quantizer_fouroversix.py"],
     points=["FP8 / MXFP4 / NVFP4 的浮点量化与整数量化的差别",
             "压缩张量格式如何描述任意量化方案",
             "逐块缩放因子的开销"],
     accept=["能解释 MXFP4 的块共享指数"]),

dict(id="L7-07", layer="L7", prio="P1",
     title="GGUF：另一种权重容器",
     files=["quantizers/quantizer_gguf.py", "modeling_gguf_pytorch_utils.py",
            "safetensors_conversion.py",
            "integrations/gguf/__init__.py", "integrations/gguf/dequant.py",
            "integrations/gguf/gguf_conversion_mapping.py",
            "integrations/gguf/gguf_tokenizer_mapping.py",
            "integrations/gguf/kernels.py", "integrations/gguf/reader.py",
            "integrations/gguf/utils.py"],
     points=["GGUF 文件结构与量化块的反量化",
             "safetensors 自动转换的存在意义",
             "GGUF 分词器到 HF 分词器的映射"],
     accept=["能说出 GGUF 相比 safetensors 的取舍"]),

# ------------------------------------------------------- L8 推理规模化
dict(id="L8-01", layer="L8", prio="P0",
     title="注意力后端分派",
     files=["integrations/__init__.py", "integrations/sdpa_attention.py",
            "integrations/flash_attention.py", "integrations/flex_attention.py",
            "integrations/msa_attention.py", "integrations/npu_flash_attention.py",
            "monkey_patching.py"],
     points=["attn_implementation 的取值与回退顺序",
             "SDPA 后端的 GQA 处理",
             "flex_attention 的可编程掩码"],
     accept=["能说出指定 flash_attention_2 但不可用时的实际行为"]),

dict(id="L8-02", layer="L8", prio="P0",
     title="分页注意力后端",
     files=["integrations/eager_paged.py", "integrations/flash_paged.py",
            "integrations/sdpa_paged.py", "modeling_attn_mask_utils.py"],
     points=["paged attention 与连续 KV 的接口差异",
             "块表（block table）如何参与注意力计算",
             "三种分页后端的适用场景"],
     accept=["能解释块表在注意力核里怎么用"]),

dict(id="L8-03", layer="L8", prio="P0",
     title="张量并行",
     files=["distributed/__init__.py", "distributed/tensor_parallel.py",
            "distributed/mixin.py", "integrations/tensor_parallel.py"],
     points=["colwise / rowwise / packed_colwise 的切分方式",
             "mla_kv_a_proj 这类自定义切分策略",
             "分配后如何保证数值等价"],
     accept=["能说出 colwise 后接 rowwise 为什么不需要 all-reduce"]),

dict(id="L8-04", layer="L8", prio="P0",
     title="专家并行与 MoE 集成",
     files=["integrations/moe.py", "integrations/sonicmoe.py",
            "integrations/deepgemm.py", "distributed/utils.py"],
     points=["ep_router / grouped_gemm / moe_tp_experts 三种标注",
             "288 个专家如何切到多卡",
             "all-to-all 通信量的估算"],
     accept=["能算出 EP=8 时每卡负责多少专家",
             "能解释路由 all-to-all 的通信复杂度"]),

dict(id="L8-05", layer="L8", prio="P1",
     title="流水线并行、分片与配置",
     files=["distributed/pipeline_parallel.py", "distributed/sharding_utils.py",
            "distributed/configuration_utils.py", "distributed/fsdp.py",
            "integrations/fsdp.py", "integrations/deepspeed.py"],
     points=["base_model_pp_plan 的输入输出约定",
             "流水线气泡与微批",
             "分片策略与推理的关系"],
     accept=["能画出 GLM-5 的 pp_plan 切分点"]),

dict(id="L8-06", layer="L8", prio="P1",
     title="异构设备与加速器集成",
     files=["integrations/heterogeneity/__init__.py",
            "integrations/heterogeneity/configuration_utils.py",
            "integrations/mistral/__init__.py", "integrations/mistral/constants.py",
            "integrations/accelerate.py", "integrations/tpu.py",
            "integrations/hub_kernels.py"],
     points=["异构配置如何在 config 上挂载额外字段",
             "TPU 与 GPU 的路径差异",
             "hub_kernels 的动态内核下载"],
     accept=["能说出异构配置生效的时机"]),

# ------------------------------------------------------- L9 出口与周边
dict(id="L9-01", layer="L9", prio="P1",
     title="导出器：把模型交给别的运行时",
     files=["exporters/__init__.py", "exporters/auto.py", "exporters/base.py",
            "exporters/configs.py", "exporters/exporter_dynamo.py",
            "exporters/exporter_executorch.py", "exporters/exporter_onnx.py",
            "exporters/exporter_openvino.py", "exporters/utils.py"],
     points=["统一导出接口与各后端的差异",
             "动态轴的声明",
             "为什么要导出 —— 推理部署的最后一步"],
     accept=["能说出 ONNX 导出的动态轴怎么声明"]),

dict(id="L9-02", layer="L9", prio="P2",
     title="推理时为什么还 import 了训练损失",
     files=["loss/__init__.py", "loss/loss_utils.py", "loss/loss_for_object_detection.py",
            "loss/loss_d_fine.py", "loss/loss_deformable_detr.py", "loss/loss_deimv2.py",
            "loss/loss_grounding_dino.py", "loss/loss_lw_detr.py", "loss/loss_rf_detr.py",
            "loss/loss_rnnt.py", "loss/loss_rt_detr.py", "loss/loss_tdt.py"],
     points=["实测：这些文件确实在推理闭包里，但前向不执行",
             "导入副作用与配置时间开销",
             "Router 的 load balancing loss 在哪"],
     accept=["能说出 loss_for_object_detection 被导入的直接原因"]),

dict(id="L9-03", layer="L9", prio="P1",
     title="运行期基础设施",
     files=["utils/__init__.py", "utils/logging.py", "utils/hub.py",
            "utils/import_utils.py", "utils/versions.py", "utils/deprecation.py",
            "utils/constants.py", "dependency_versions_check.py",
            "dependency_versions_table.py", "utils/generic.py", "__init__.py"],
     points=["惰性模块 __getattr__ 的实现机制",
             "import_utils 的能力探测与缓存",
             "版本守卫与依赖检查表"],
     accept=["能解释 `from transformers import X` 为什么不会一次性导入全部模型"]),

dict(id="L9-04", layer="L9", prio="P2",
     title="加载报告、内核配置与调试工具",
     files=["utils/loading_report.py", "utils/kernel_config.py",
            "utils/output_capturing.py", "utils/attention_visualizer.py",
            "utils/peft_utils.py", "utils/dummy_mistral_common_objects.py",
            "utils/dummy_sentencepiece_and_tokenizers_objects.py",
            "trainer_utils.py"],
     points=["加载报告如何把 missing/unexpected 变成可读结论",
             "kernel_config 描述可替换内核",
             "dummy objects 如何让未安装依赖也能被 import"],
     accept=["能解释 dummy objects 的运作方式"]),

dict(id="L9-05", layer="L9", prio="P1",
     title="对话模板与工具调用解析",
     files=["utils/chat_template_utils.py", "utils/chat_parsing/__init__.py",
            "utils/chat_parsing/content_parsers.py",
            "utils/chat_parsing/response_parser.py",
            "utils/chat_parsing/response_templates.py"],
     points=["Jinja 模板的渲染与校验",
             "工具调用的流式解析状态机",
             "响应模板如何适配不同模型的输出格式"],
     accept=["能说出 apply_chat_template 插入视觉占位符的位置"]),

dict(id="L9-06", layer="L9", prio="P1",
     title="量化集成层",
     files=["integrations/bitsandbytes.py", "integrations/quanto.py",
            "integrations/torchao.py", "integrations/fp_quant.py",
            "integrations/finegrained_fp8.py", "integrations/ggml.py",
            "integrations/compressed_tensors.py", "integrations/aqlm.py",
            "integrations/hqq.py", "integrations/spqr.py", "integrations/peft.py",
            "integrations/tiktoken.py", "integrations/liger.py"],
     points=["每个集成提供的算子替换点",
             "PEFT 与量化叠加时的顺序",
             "为什么有了 quantizer 还需要 integration"],
     accept=["能说出 quantizer 与 integration 的职责边界"]),

dict(id="L9-07", layer="L9", prio="P1",
     title="模型族继承链：GLM-5 站在谁的肩膀上",
     files=["models/glm5_next/__init__.py", "models/glm4v/__init__.py",
            "models/glm4v/configuration_glm4v.py", "models/glm4v/modeling_glm4v.py",
            "models/glm4v_moe/__init__.py", "models/glm4v_moe/configuration_glm4v_moe.py",
            "models/glm4v_moe/modeling_glm4v_moe.py", "models/glm_ocr/__init__.py",
            "models/glm_ocr/configuration_glm_ocr.py", "models/glm_ocr/modeling_glm_ocr.py",
            "models/kimi_k25/__init__.py", "models/kimi_k25/configuration_kimi_k25.py",
            "models/kimi_k25/modeling_kimi_k25.py", "models/qwen3_5/__init__.py",
            "models/qwen3_5/configuration_qwen3_5.py", "models/qwen3_5/modeling_qwen3_5.py",
            "models/qwen3_5/tokenization_qwen3_5.py", "models/qwen3_5_moe/__init__.py",
            "models/qwen3_5_moe/configuration_qwen3_5_moe.py",
            "models/qwen3_5_moe/modeling_qwen3_5_moe.py", "models/deepseek_v2/__init__.py",
            "models/deepseek_v2/configuration_deepseek_v2.py",
            "models/deepseek_v2/modeling_deepseek_v2.py"],
     points=["实测：为什么这些模型会被 GLM-5 的推理闭包带进来",
             "KDA 与 Kimi 的渊源、DSA 与 DeepSeek 的渊源",
             "共享组件与复制粘贴的边界"],
     accept=["能说出 GLM-5 的 KDA 与 kimi_k25 的关系"]),

dict(id="L9-08", layer="L9", prio="P0",
     title="收束：一次 GLM-5 推理的完整文件清单",
     files=["models/glm5_next/modeling_glm5_next.py",
            "models/glm5_next/configuration_glm5_next.py"],
     points=["把 9 层课压缩成一张「文件 → 阶段」的总表",
             "重新回答 L0-01 的全景图，这次每一块都能说出文件名",
             "实测统计：231 个文件、多少行、多少课"],
     accept=["能默画全景图并标注每块的源文件",
             "能说出四道门禁各自断言了什么"]),
]


def expand_files(entries, universe):
    """展开 "~" 结尾的目录通配，返回去重后的文件列表。"""
    out = []
    for e in entries:
        if e.endswith("~"):
            pref = e[:-1]
            hits = [u for u in universe if u.startswith(pref)]
            if not hits:
                raise SystemExit("通配无命中: " + e)
            out.extend(sorted(hits))
        else:
            out.append(e)
    seen, res = set(), []
    for f in out:
        if f not in seen:
            seen.add(f); res.append(f)
    return res


if __name__ == "__main__":
    import json, os, sys
    here = os.path.dirname(os.path.abspath(__file__))
    uni = json.load(open(os.path.join(here, "..", "_data", "universe.json")))["universe"]
    covered = set()
    for L in LESSONS:
        fs = expand_files(L["files"], uni)
        covered |= set(fs)
    missing = sorted(set(uni) - covered)
    phantom = sorted(covered - set(uni))
    print("课时数 :", len(LESSONS))
    print("覆盖域 :", len(uni))
    print("已覆盖 :", len(covered))
    print("未覆盖 :", len(missing))
    if missing: print("  ", missing)
    if phantom: print("   幻影:", phantom)
    sys.exit(1 if missing or phantom else 0)
