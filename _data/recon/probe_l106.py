"""L1-06 实测探针：六张自动注册表的规模、原地打补丁的顺序效应、惰性 import 的代价、
以及 GLM-5 的三级解析（processor -> 三个子处理器）。

用法（仓库根目录 /data/WORKSPACE/transformer-project）：
    ./.venv/bin/python courseware/_data/recon/probe_l106.py

本脚本是**工装**，不属于覆盖域（覆盖域只统计 src/transformers/**/*.py），
不计入覆盖率。
"""
import importlib
import sys
import time
import warnings

warnings.filterwarnings("ignore")


def line(k, v):
    print(f"   {k:<40s}: {v}")


def step(t):
    print("=" * 76)
    print(t)


def nmods():
    return len([m for m in sys.modules if m.startswith("transformers")])


step("① 基座：只 import transformers + auto_mappings，底表各有多少条")
import transformers  # noqa: E402

AM = importlib.import_module("transformers.models.auto.auto_mappings")
line("transformers 版本", transformers.__version__)
line("import transformers 后 transformers.* 模块数", nmods())
line("底表 CONFIG_MAPPING_NAMES", len(AM.CONFIG_MAPPING_NAMES))
line("底表 IMAGE_PROCESSOR_MAPPING_NAMES", len(AM.IMAGE_PROCESSOR_MAPPING_NAMES))
line("底表 VIDEO_PROCESSOR_MAPPING_NAMES", len(AM.VIDEO_PROCESSOR_MAPPING_NAMES))
line("底表 FEATURE_EXTRACTOR_MAPPING_NAMES", len(AM.FEATURE_EXTRACTOR_MAPPING_NAMES))
line("底表 PROCESSOR_MAPPING_NAMES", len(AM.PROCESSOR_MAPPING_NAMES))
line("auto_mappings 里有 TOKENIZER 表吗", hasattr(AM, "TOKENIZER_MAPPING_NAMES"))

step("② 每个 auto 模块 import 时，对**同一个 dict 对象**原地 update 补丁")
_ = importlib.import_module("transformers.models.auto.modeling_auto")
line("modeling_auto 有自己的表，不动 auto_mappings",
     len(AM.IMAGE_PROCESSOR_MAPPING_NAMES) == 120 and len(AM.VIDEO_PROCESSOR_MAPPING_NAMES) == 31)

n_img = len(AM.IMAGE_PROCESSOR_MAPPING_NAMES)
_ = importlib.import_module("transformers.models.auto.image_processing_auto")
line(f"IMAGE 表 {n_img} -> import image_processing_auto 后", len(AM.IMAGE_PROCESSOR_MAPPING_NAMES))

n_vid = len(AM.VIDEO_PROCESSOR_MAPPING_NAMES)
_ = importlib.import_module("transformers.models.auto.video_processing_auto")
line(f"VIDEO 表 {n_vid} -> import video_processing_auto 后", len(AM.VIDEO_PROCESSOR_MAPPING_NAMES))

n_feat = len(AM.FEATURE_EXTRACTOR_MAPPING_NAMES)
_ = importlib.import_module("transformers.models.auto.feature_extraction_auto")
line(f"FEAT 表 {n_feat} -> import feature_extraction_auto 后", len(AM.FEATURE_EXTRACTOR_MAPPING_NAMES))

n_proc = len(AM.PROCESSOR_MAPPING_NAMES)
_ = importlib.import_module("transformers.models.auto.processing_auto")
line(f"PROC 表 {n_proc} -> import processing_auto 后", len(AM.PROCESSOR_MAPPING_NAMES))

TOK = importlib.import_module("transformers.models.auto.tokenization_auto").TOKENIZER_MAPPING_NAMES
line("TOKENIZER 表（自成一家，不在 auto_mappings）", len(TOK))
line("其中 'TokenizersBackend' 兜底条数",
     sum(1 for v in TOK.values() if v == "TokenizersBackend"))
from transformers.models.auto.tokenization_auto import MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS as BADHUB  # noqa: E402
line("MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS", len(BADHUB))

step("③ 五张任务表（modeling_auto.py 内部，字符串 -> 字符串）")
MA = importlib.import_module("transformers.models.auto.modeling_auto")
line("MODEL_MAPPING_NAMES", len(MA.MODEL_MAPPING_NAMES))
line("MODEL_FOR_CAUSAL_LM_MAPPING_NAMES", len(MA.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES))
line("MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES", len(MA.MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES))
line("modeling_auto 里 _LazyAutoMapping 实例数",
     len([n for n in dir(MA) if n.endswith("_MAPPING") and type(getattr(MA, n)).__name__ == "_LazyAutoMapping"]))
line("modeling_auto 导出的 Auto* 类数",
     len([n for n in dir(MA) if n.startswith("Auto") and isinstance(getattr(MA, n), type)]))

step("④ 惰性映射：读表零导入，取格子只 import 那一个包")
from transformers import AutoModel, Glm5NextConfig, Glm5NextTextConfig  # noqa: E402
mm = AutoModel._model_mapping
line("AutoModel._model_mapping 类型 / __len__()", f"{type(mm).__name__} / {len(mm)}")
n0 = nmods()
_ = MA.MODEL_MAPPING_NAMES["glm5_next"]
line("读 MODEL_MAPPING_NAMES['glm5_next'] 新增模块数", nmods() - n0)
line("读到的值（还是字符串）", repr(MA.MODEL_MAPPING_NAMES["glm5_next"]))
n0, t0 = nmods(), time.perf_counter()
val = mm[Glm5NextConfig]
line("mm[Glm5NextConfig]", repr(val))
line("新导入 transformers.* 模块数", nmods() - n0)
line("耗时(ms)", "%.1f" % ((time.perf_counter() - t0) * 1000))
n0, t0 = nmods(), time.perf_counter()
_ = mm[Glm5NextTextConfig]
line("再取 Glm5NextTextConfig 新增模块数 / 耗时(ms)",
     f"{nmods() - n0} / %.2f" % ((time.perf_counter() - t0) * 1000))

step("⑤ GLM-5 落在哪张任务表里（config 决定形状）")
line("Glm5NextConfig in MODEL_MAPPING", Glm5NextConfig in MA.MODEL_MAPPING)
line("Glm5NextConfig in MODEL_FOR_CAUSAL_LM_MAPPING", Glm5NextConfig in MA.MODEL_FOR_CAUSAL_LM_MAPPING)
line("Glm5NextConfig in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING",
     Glm5NextConfig in MA.MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING)
line("MODEL_MAPPING[Glm5NextConfig]", repr(MA.MODEL_MAPPING[Glm5NextConfig]))
line("MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING[Glm5NextConfig]",
     repr(MA.MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING[Glm5NextConfig]))

step("⑥ AutoProcessor 对 GLM-5：一个处理器 + 三个子处理器")
from transformers import (  # noqa: E402
    IMAGE_PROCESSOR_MAPPING,
    PROCESSOR_MAPPING,
    TOKENIZER_MAPPING,
    VIDEO_PROCESSOR_MAPPING,
    Glm5NextProcessor,
)
line("PROCESSOR_MAPPING[Glm5NextConfig]", repr(PROCESSOR_MAPPING[Glm5NextConfig]))
line("Glm5NextProcessor.get_attributes()", Glm5NextProcessor.get_attributes())
line("IMAGE_PROCESSOR_MAPPING[Glm5NextConfig]",
     "{" + ", ".join(f"{k}: {v.__name__}" for k, v in IMAGE_PROCESSOR_MAPPING[Glm5NextConfig].items()) + "}")
line("VIDEO_PROCESSOR_MAPPING[Glm5NextConfig]", repr(VIDEO_PROCESSOR_MAPPING[Glm5NextConfig]))
line("TOKENIZER_MAPPING.get(Glm5NextConfig, 'TokenizersBackend')",
     repr(TOKENIZER_MAPPING.get(Glm5NextConfig, "TokenizersBackend")))
from transformers.models.auto.image_processing_auto import _resolve_backend  # noqa: E402
line("_resolve_backend(None, None, 'Glm5NextImageProcessor')", _resolve_backend(None, None, "Glm5NextImageProcessor"))
line("_resolve_backend('pil', None, 'Glm5NextImageProcessor')", _resolve_backend("pil", None, "Glm5NextImageProcessor"))
from transformers.utils import is_torchvision_available, is_vision_available  # noqa: E402
line("is_torchvision_available() / is_vision_available()", f"{is_torchvision_available()} / {is_vision_available()}")

step("⑦ 命名冲突：注册表 > 表 > 主模块 > 去掉 Fast 再试")
from transformers.models.auto.processing_auto import processor_class_from_name  # noqa: E402
from transformers.models.auto.image_processing_auto import get_image_processor_class_from_name  # noqa: E402
from transformers.models.auto.video_processing_auto import video_processor_class_from_name  # noqa: E402
from transformers.models.auto.tokenization_auto import (  # noqa: E402
    REGISTERED_TOKENIZER_CLASSES,
    tokenizer_class_from_name,
)
line("processor_class_from_name('Glm5NextProcessor')", repr(processor_class_from_name("Glm5NextProcessor")))
line("get_image_processor_class_from_name('Glm5NextImageProcessorPil')",
     repr(get_image_processor_class_from_name("Glm5NextImageProcessorPil")))
line("video_processor_class_from_name('Glm5NextVideoProcessor')",
     repr(video_processor_class_from_name("Glm5NextVideoProcessor")))
line("processor_class_from_name('NoSuchProcessor')", repr(processor_class_from_name("NoSuchProcessor")))
line("tokenizer_class_from_name('BertTokenizer')", repr(tokenizer_class_from_name("BertTokenizer")))
line("REGISTERED_TOKENIZER_CLASSES 初始规模", len(REGISTERED_TOKENIZER_CLASSES))

step("⑧ 环境相关的表：video 表整表随 torchvision 塌陷（本机 torchvision 可用）")
line("VIDEO 表里值为 None 的条数（本机）", sum(1 for v in AM.VIDEO_PROCESSOR_MAPPING_NAMES.values() if v is None))
line("VIDEO 表条目总数（补丁后）", len(AM.VIDEO_PROCESSOR_MAPPING_NAMES))
line("import transformers 之后 transformers.* 模块总数（终点）", nmods())

step("⑨ 反面实测：把 is_torchvision_available 换成 False 后 reload video 模块")
import transformers.utils as TU  # noqa: E402
import transformers.utils.import_utils as TIU  # noqa: E402
V = importlib.import_module("transformers.models.auto.video_processing_auto")
line("reload 前，值为 None 的条数", sum(1 for v in V.VIDEO_PROCESSOR_MAPPING_NAMES.values() if v is None))
_real = TIU.is_torchvision_available
TIU.is_torchvision_available = lambda: False
TU.is_torchvision_available = lambda: False        # `from ...utils import` 会重读这个属性
V = importlib.reload(V)                            # 让模块自己那段循环重跑一遍
line("reload 后，值为 None 的条数", sum(1 for v in V.VIDEO_PROCESSOR_MAPPING_NAMES.values() if v is None))
line("reload 后，条目总数", len(V.VIDEO_PROCESSOR_MAPPING_NAMES))
line("此时 VIDEO_PROCESSOR_MAPPING.get(Glm5NextConfig)",
     repr(V.VIDEO_PROCESSOR_MAPPING.get(Glm5NextConfig, "<无此键>")))
line("auto_mappings 里的同名对象也被一起改了（None 条数）",
     f"{sum(1 for v in AM.VIDEO_PROCESSOR_MAPPING_NAMES.values() if v is None)} / {len(AM.VIDEO_PROCESSOR_MAPPING_NAMES)}")
line("（恢复）is_torchvision_available()", _real())
