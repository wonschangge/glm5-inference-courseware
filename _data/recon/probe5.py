import sys, os, json, warnings, importlib
warnings.filterwarnings("ignore")
ROOT = os.path.abspath("upstream-transformers/src/transformers")
base = set(sys.modules)
import torch, numpy as np
from PIL import Image
def step(name, fn):
    try: fn(); print("  ok   ", name)
    except Exception as e: print("  SKIP ", name, "->", type(e).__name__, str(e)[:80])

from transformers import Glm5NextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextForConditionalGeneration
txt = dict(vocab_size=256, hidden_size=64, intermediate_size=64, moe_intermediate_size=32,
    num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=4,
    q_lora_rank=32, kv_lora_rank=16, qk_nope_head_dim=16, v_head_dim=16, qk_rope_head_dim=0,
    n_routed_experts=8, n_shared_experts=1, num_experts_per_tok=2,
    linear_head_dim=16, linear_num_heads=4, hc_mult=2,
    index_head_dim=16, index_n_heads=2, index_topk=8, index_kpool=4,
    max_position_embeddings=1024, pad_token_id=0, eos_token_id=1)
cfg = Glm5NextConfig(text_config=txt)
v = cfg.vision_config
v.depth=2; v.hidden_size=32; v.num_heads=2; v.intermediate_size=32
v.image_size=28; v.patch_size=14; v.out_hidden_size=64; v.projection_intermediate_size=64
m = Glm5NextForConditionalGeneration(cfg).eval()

step("text generate", lambda: [m.generate(torch.randint(0,256,(1,10)), max_new_tokens=3, do_sample=False)])
img = Image.fromarray((np.random.rand(28,28,3)*255).astype('uint8'))
def vision():
    from transformers.models.glm5_next.image_processing_glm5_next import Glm5NextImageProcessor
    ip = Glm5NextImageProcessor()
    out = ip(images=img, return_tensors='pt')
    print("       ip keys:", list(out.keys()), {k: tuple(x.shape) for k,x in out.items()})
    with torch.no_grad(): m.get_image_features(**out)
step("image processor + vision tower", vision)
step("image processor pil", lambda: importlib.import_module("transformers.models.glm5_next.image_processing_pil_glm5_next"))
step("video processor", lambda: importlib.import_module("transformers.models.glm5_next.video_processing_glm5_next"))
step("processor", lambda: importlib.import_module("transformers.models.glm5_next.processing_glm5_next"))
step("tokenizer tokenizers", lambda: importlib.import_module("transformers.tokenization_utils_tokenizers"))
step("tokenizer python", lambda: importlib.import_module("transformers.tokenization_python"))
step("tokenizer base", lambda: importlib.import_module("transformers.tokenization_utils_base"))
step("tokenization_utils_sentencepiece", lambda: importlib.import_module("transformers.tokenization_utils_sentencepiece"))
step("sampling", lambda: [m.generate(torch.randint(0,256,(1,8)), max_new_tokens=3, do_sample=True,
        temperature=.7, top_p=.9, top_k=5, repetition_penalty=1.1)])
step("beam", lambda: [m.generate(torch.randint(0,256,(1,8)), max_new_tokens=3, num_beams=2)])
step("assisted?", lambda: [m.generate(torch.randint(0,256,(1,8)), max_new_tokens=3,
        assistant_model=m, do_sample=False)])
step("quant configs", lambda: [__import__("transformers", fromlist=["BitsAndBytesConfig"]).BitsAndBytesConfig(load_in_4bit=True),
        __import__("transformers", fromlist=["QuantoConfig"]).QuantoConfig(weights='int8')])
for qn in ["auto","base","quantizer_bnb_4bit","quantizer_bnb_8bit","quantizer_awq","quantizer_gptq",
           "quantizer_quanto","quantizer_torchao","quantizer_finegrained_fp8","quantizer_gguf",
           "quantizer_mxfp4","quantizer_hqq","quantizer_eetq","quantizer_aqlm","quantizer_vptq",
           "quantizer_compressed_tensors","quantizer_fbgemm_fp8","quantizer_fp_quant",
           "quantizer_metal","quantizer_bitnet","quantizer_higgs","quantizer_sinq","quantizer_spqr",
           "quantizer_quark","quantizer_nvfp4","quantizer_fouroversix","quantizer_gemma"]:
    step("quantizers."+qn, (lambda n: (lambda: importlib.import_module("transformers.quantizers."+n)))(qn))
for n in ["distributed.tensor_parallel","distributed.utils","distributed.configuration_utils",
          "distributed.pipeline_parallel","distributed.sharding_utils","distributed.fsdp","distributed.mixin"]:
    step(n, (lambda x: (lambda: importlib.import_module("transformers."+x)))(n))
for n in ["exporters","integrations.accelerate","integrations.bitsandbytes","integrations.deepgemm",
          "integrations.eager_paged","integrations.finegrained_fp8","integrations.flash_attention",
          "integrations.flash_paged","integrations.flex_attention","integrations.fp_quant",
          "integrations.ggml","integrations.hub_kernels","integrations.liger","integrations.moe",
          "integrations.msa_attention","integrations.peft","integrations.quanto",
          "integrations.sdpa_attention","integrations.sdpa_paged","integrations.sonicmoe",
          "integrations.tensor_parallel","integrations.tiktoken","integrations.torchao",
          "integrations.tpu","integrations.npu_flash_attention","integrations.heterogeneity.configuration_utils",
          "integrations.mistral","integrations.gguf","generation.continuous_batching",
          "generation.utils","generation.configuration_utils","generation.logits_process",
          "generation.stopping_criteria","generation.candidate_generator","generation.streamers",
          "generation.watermarking","modeling_utils","configuration_utils","core_model_loading",
          "cache_utils","masking_utils","modeling_rope_utils","modeling_layers",
          "modeling_flash_attention_utils","modeling_attn_mask_utils","activations","pytorch_utils",
          "initialization","quantizers.auto","loss.loss_utils","loss.loss_utils","utils.generic",
          "utils.import_utils","utils.auto_docstring","utils.logging","utils.hub","utils.versions",
          "utils.chat_template_utils","utils.quantization_config","utils.deprecation","utils.constants",
          "utils.type_validators","utils.kernel_config","utils.loading_report","utils.peft_utils",
          "utils.attention_visualizer","utils.backbone_utils","utils.chat_parsing","utils.hp_naming"]:
    step(n, (lambda x: (lambda: importlib.import_module("transformers."+x)))(n))

mods = set(sys.modules) - base
files = set()
for mm in mods:
    mod = sys.modules.get(mm); f = getattr(mod, "__file__", None)
    if not f: continue
    f = os.path.abspath(f)
    if f.startswith(ROOT) and f.endswith(".py"):
        files.add(os.path.relpath(f, os.path.dirname(ROOT)))
files = sorted(files)
print("CLOSURE_FILES", len(files))
json.dump(files, open("_recon/closure_full.json","w"), indent=1)
