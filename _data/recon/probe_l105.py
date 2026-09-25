"""L1-05 实测探针：AutoModel 分派链的每一步都跑一遍，不靠读代码推断。

用法（仓库根目录）：
    ./.venv/bin/python courseware/_data/recon/probe_l105.py

顺序敏感：0 与 1 必须在 configuration_auto 被导入之前跑（借以观察表的就地修改），
2 必须在 glm5_next 被点名之前跑（借以观察冷访问成本）。
"""
import sys, time, warnings
warnings.filterwarnings("ignore")


def nmods():
    """已加载的 transformers.* 模块数。"""
    return len([m for m in sys.modules if m.startswith("transformers")])


def line(k, v):
    print(f"   {k:<40s}: {v}")


def step(title):
    print("=" * 78)
    print(title)


# ===========================================================================
# 0. 起点
# ===========================================================================
step("0 起点：只 import transformers")
import transformers
line("版本", transformers.__version__)
line("transformers.* 模块数", nmods())
line("glm5_next.configuration 已加载", "transformers.models.glm5_next.configuration_glm5_next" in sys.modules)

# ===========================================================================
# 1. 表是纯数据，而且是「就地修改」的
# ===========================================================================
step("1 表的形状：718 条 str -> str，然后被 configuration_auto 就地改大")
from transformers.models.auto.auto_mappings import CONFIG_MAPPING_NAMES as T
line("刚 import auto_mappings 时 len(T)", len(T))
line("T['glm5_next']", repr(T["glm5_next"]) + "   （值是字符串）")
n_before = nmods()
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
line("导入 configuration_auto 之后 len(T)", f"{len(T)}（新增 {len(T) - 718} 条，就地 update）")
line("len(CONFIG_MAPPING.keys())", f"{len(CONFIG_MAPPING.keys())}（= {len(T)} + gpt-sw3）")
line("导入 configuration_auto 新增模块数", nmods() - n_before)

# ===========================================================================
# 2. 成本账：点名 vs 全都要
# ===========================================================================
step("2 成本账（此时 glm5_next 仍是冷的）")
from transformers import AutoConfig, AutoModel
line("拿到 AutoConfig/AutoModel 后模块数", nmods())

t0 = time.perf_counter()
L = len(AutoModel._model_mapping)
line("len(AutoModel._model_mapping)", f"{L}（新增 0 个模块，{(time.perf_counter() - t0) * 1000:.1f} ms）")

n0, t0 = nmods(), time.perf_counter()
ks = CONFIG_MAPPING.keys()
line("CONFIG_MAPPING.keys()", f"{len(ks)} 项（新增 {nmods() - n0} 个模块，{(time.perf_counter() - t0) * 1000:.1f} ms）")

n0, t0 = nmods(), time.perf_counter()
cls = CONFIG_MAPPING["glm5_next"]
line("CONFIG_MAPPING['glm5_next']", f"{cls.__name__}（新增 {nmods() - n0} 个模块，{(time.perf_counter() - t0) * 1000:.1f} ms）")

n0, t0 = nmods(), time.perf_counter()
val = AutoModel._model_mapping[cls]
line("AutoModel._model_mapping[Glm5NextConfig]", f"{val.__name__}（新增 {nmods() - n0} 个模块，{(time.perf_counter() - t0) * 1000:.1f} ms）")

n0, t0 = nmods(), time.perf_counter()
vals = CONFIG_MAPPING.values()
line("CONFIG_MAPPING.values()（全都要）", f"{len(vals)} 项（新增 {nmods() - n0} 个模块，{(time.perf_counter() - t0) * 1000:.0f} ms）")
line("transformers.* 模块总数", nmods())

# ===========================================================================
# 3. 分派链逐跳核对
# ===========================================================================
step("3 第二跳：model_type -> 包名")
from transformers.models.auto.configuration_auto import model_type_to_module_name, DEPRECATED_MODELS
for k in ("glm5_next", "glm5_next_text", "glm5_next_vision",
          "audio-spectrogram-transformer", "bert-generation", "transfo-xl"):
    line(k, "-> " + model_type_to_module_name(k))
line("DEPRECATED_MODELS", repr(DEPRECATED_MODELS))

step("4 反向：类 -> model_type")
from transformers.models.auto.configuration_auto import config_class_to_model_type
from transformers import Glm5NextConfig, Glm5NextTextConfig, Glm5NextVisionConfig
line("config_class_to_model_type('Glm5NextConfig')", config_class_to_model_type("Glm5NextConfig"))
line("Glm5NextConfig.model_type", Glm5NextConfig.model_type)
line("Glm5NextTextConfig.model_type", Glm5NextTextConfig.model_type)
line("CONFIG_MAPPING['glm5_next'] is Glm5NextConfig", CONFIG_MAPPING["glm5_next"] is Glm5NextConfig)

step("5 AutoConfig.for_model('glm5_next') 真的造得出 config")
c = AutoConfig.for_model("glm5_next")
line("type(config)", type(c).__name__)
line("config.model_type", c.model_type)
line("text_config.hidden_size / num_hidden_layers", f"{c.text_config.hidden_size} / {c.text_config.num_hidden_layers}")
try:
    AutoConfig.for_model("definitely_not_a_model")
except ValueError as e:
    line("未知 model_type", type(e).__name__ + ": " + str(e)[:40] + " ...")

step("6 contains 不导入，getitem 才导入")
line("'glm5_next_vision' in CONFIG_MAPPING", "glm5_next_vision" in CONFIG_MAPPING)
line("'definitely_not_a_model' in CONFIG_MAPPING", "definitely_not_a_model" in CONFIG_MAPPING)

step("7 _LazyAutoMapping：config 类 -> 模型类")
mm = AutoModel._model_mapping
line("类型", type(mm).__name__)
line("__len__()", len(mm))
line("mm[Glm5NextConfig]", repr(mm[Glm5NextConfig]))
line("是 tuple（多候选才是）", isinstance(mm[Glm5NextConfig], tuple))
line("Glm5NextTextConfig in mm", Glm5NextTextConfig in mm)
line("Glm5NextVisionConfig in mm", Glm5NextVisionConfig in mm)
line("mm[Glm5NextTextConfig]", repr(mm[Glm5NextTextConfig]))
