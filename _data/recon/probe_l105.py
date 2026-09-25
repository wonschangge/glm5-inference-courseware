"""L1-05 实测探针：AutoModel 分派链的每一步都跑一遍，不靠读代码推断。

用法（仓库根目录）：
    ../.venv/bin/python _data/recon/probe_l105.py
"""
import sys, time, warnings
warnings.filterwarnings("ignore")

MOD_PREFIX = "transformers.models.glm5_next"
GLM_MOD = MOD_PREFIX + ".configuration_glm5_next"


def n_mods():
    return len([m for m in sys.modules if m.startswith("transformers")])


print("=" * 72)
print("① 导入 transformers 之后，glm5_next 的 config 模块进来了吗？")
import transformers
print("   transformers", transformers.__version__)
print("   Glm5NextConfig 已在 sys.modules :", GLM_MOD in sys.modules)
print("   transformers.models.glm5_next   :", MOD_PREFIX in sys.modules)
print("   transformers.* 模块数            :", n_mods())

print("=" * 72)
print("② CONFIG_MAPPING_NAMES 是一张纯字符串表 —— 不导入任何模型代码")
from transformers.models.auto.auto_mappings import CONFIG_MAPPING_NAMES
print("   条目数                           :", len(CONFIG_MAPPING_NAMES))
print("   glm5_next ->                     :", CONFIG_MAPPING_NAMES["glm5_next"])
kv = list(CONFIG_MAPPING_NAMES.items())[0]
print("   第一项（类型）                    :", type(kv[0]).__name__, type(kv[1]).__name__)

print("=" * 72)
print("③ model_type_to_module_name：字符串 -> 模块名")
from transformers.models.auto.configuration_auto import model_type_to_module_name
for k in ("glm5_next", "glm5_next_text", "glm5_next_vision",
          "audio-spectrogram-transformer", "bert-generation", "transfo-xl"):
    print(f"   {k:32s} -> {model_type_to_module_name(k)}")

print("=" * 72)
print("④ 逐条访问 CONFIG_MAPPING：只看被点到的那个模块被拉进来")
from transformers import CONFIG_MAPPING, AutoConfig, Glm5NextConfig
before = n_mods()
t0 = time.perf_counter()
cls = CONFIG_MAPPING["glm5_next"]
dt = (time.perf_counter() - t0) * 1000
print("   CONFIG_MAPPING['glm5_next']      ->", cls.__name__)
print("   与 Glm5NextConfig 同一个对象      :", cls is Glm5NextConfig)
print("   返回类的 __module__              :", cls.__module__)
print("   新导入 transformers.* 模块        :", n_mods() - before)
print("   该次访问耗时                      : %.1f ms" % dt)
print("   glm5_next.configuration 已加载    :", GLM_MOD in sys.modules)

print("=" * 72)
print("⑤ 再点一个别的模型：只多拉它自己那一个模块")
before = n_mods()
_ = CONFIG_MAPPING["bert"]
print("   新导入 transformers.* 模块        :", n_mods() - before)
print("   bert.configuration 已加载         :",
      "transformers.models.bert.configuration_bert" in sys.modules)

print("=" * 72)
print("⑥ 反向查：类 -> model_type")
from transformers.models.auto.configuration_auto import config_class_to_model_type
print("   config_class_to_model_type('Glm5NextConfig')     ->",
      config_class_to_model_type("Glm5NextConfig"))
print("   Glm5NextConfig.model_type（类属性，实测）          ->", Glm5NextConfig.model_type)

print("=" * 72)
print("⑦ AutoConfig.for_model('glm5_next') 真的能造出 config")
c = AutoConfig.for_model("glm5_next")
print("   type(config)                     :", type(c).__name__)
print("   config.model_type                :", c.model_type)
print("   隐藏维 / 层数                     :", c.text_config.hidden_size, "/", c.text_config.num_hidden_layers)
try:
    AutoConfig.for_model("definitely_not_a_model")
except ValueError as e:
    print("   未知 model_type ->", type(e).__name__, str(e)[:60], "...")

print("=" * 72)
print("⑧ _LazyAutoMapping：config 类 -> 模型类，同样是「用时才导入」")
from transformers import AutoModel
mm = AutoModel._model_mapping
print("   类型                              :", type(mm).__name__)
print("   模块内 mem 前缀（导入前）           :", GLM_MOD in sys.modules)
t0 = time.perf_counter()
val = mm[Glm5NextConfig]
dt = (time.perf_counter() - t0) * 1000
print("   mm[Glm5NextConfig]               ->", val)
print("   是 tuple                          :", isinstance(val, tuple), "长度", len(val))
print("   耗时                              : %.1f ms" % dt)
print("   模型类 __module__                 :", val[0].__module__)

print("=" * 72)
print("⑨ 惰性映射省下了什么：load-all 的代价")
from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES
before = n_mods()
t0 = time.perf_counter()
_ = [(k, mm.__getitem__(v)) for k, v in [] ] or None
# 触发表里所有条目：这也是 keys()/items() 会做的事
keys = mm.keys()
dt_all = (time.perf_counter() - t0) * 1000
print("   MODEL_MAPPING_NAMES 条目数         :", len(MODEL_MAPPING_NAMES))
print("   全量 keys() 新导入模块数            :", n_mods() - before)
print("   全量 keys() 耗时                   : %.0f ms" % dt_all)
print("   transformers.* 模块总数（全量后）    :", n_mods())
