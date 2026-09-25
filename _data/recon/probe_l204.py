#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L2-04 实测探针：后端选择顺序 / 形状演算 / rescale+normalize 的融合。

全部结论来自本脚本的真实运行输出，不靠阅读推断。
运行：/data/WORKSPACE/transformer-project/.venv/bin/python _data/recon/probe_l204.py
"""
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch

print("=" * 74)
print("[1] 环境")
import torchvision

import transformers
from transformers.utils import is_torchvision_available, is_vision_available

print("  transformers =", transformers.__version__)
print("  torch        =", torch.__version__)
print("  torchvision  =", torchvision.__version__)
print("  is_vision_available()      =", is_vision_available())
print("  is_torchvision_available() =", is_torchvision_available())

print("=" * 74)
print("[2] _resolve_backend 的真实分支（不靠读源码，直接调用）")
from transformers.models.auto.image_processing_auto import (
    DEFAULT_TO_PIL_BACKEND_IMAGE_PROCESSORS,
    _resolve_backend,
)

print("  DEFAULT_TO_PIL_BACKEND_IMAGE_PROCESSORS =", DEFAULT_TO_PIL_BACKEND_IMAGE_PROCESSORS)
for name in ["Glm5NextImageProcessor", "CLIPImageProcessor", "SmolVLMImageProcessor"]:
    print(f"  _resolve_backend(None, None, {name!r}) = {_resolve_backend(None, None, name)!r}")
print("  _resolve_backend('pil',        None, 'Glm5NextImageProcessor') =",
      repr(_resolve_backend("pil", None, "Glm5NextImageProcessor")))
print("  _resolve_backend('torchvision',None, 'Glm5NextImageProcessor') =",
      repr(_resolve_backend("torchvision", None, "Glm5NextImageProcessor")))
print("  _resolve_backend(None, True,  'Glm5NextImageProcessor') =",
      repr(_resolve_backend(None, True, "Glm5NextImageProcessor")))
print("  _resolve_backend(None, False, 'Glm5NextImageProcessor') =",
      repr(_resolve_backend(None, False, "Glm5NextImageProcessor")))

print("=" * 74)
print("[3] 两个后端类真的存在吗")
import transformers.image_processing_backends as B

print("  image_processing_backends 里的类 =", [n for n in dir(B) if n.endswith("Backend")])
print("  TorchvisionBackend.backend =", B.TorchvisionBackend.backend.fget(None))
print("  PilBackend.backend        =", B.PilBackend.backend.fget(None))
print("  有 NumpyBackend 吗        =", hasattr(B, "NumpyBackend"))

print("=" * 74)
print("[4] 后端选择：同一个模型，两种 backend 参数")
from transformers import AutoImageProcessor

for be in [None, "torchvision", "pil"]:
    try:
        ip = AutoImageProcessor.from_pretrained("hf-internal-testing/tiny-random-glm5", backend=be)
        print(f"  backend={be!r:14} -> {type(ip).__name__:34} .backend={ip.backend!r}")
    except Exception as e:
        print(f"  backend={be!r:14} -> 需要联网/权重不可用: {type(e).__name__}: {str(e)[:90]}")

print("=" * 74)
print("[5] 形状演算：336x336 -> patchify -> pixel_values  (真实跑一遍)")
from transformers.models.glm5_next.image_processing_glm5_next import Glm5NextImageProcessor

ip = Glm5NextImageProcessor()
img = np.random.randint(0, 256, (336, 336, 3), dtype=np.uint8)
out = ip(images=img, return_tensors="pt")
px = out["pixel_values"]
thw = out["image_grid_thw"]
print("  输入 (H, W, C)      =", img.shape)
print("  pixel_values.shape  =", tuple(px.shape), px.dtype)
print("  image_grid_thw      =", thw.tolist(), " (t, h, w) 网格")
print("  patch_size=14 merge_size=2 temporal_patch_size=2")
g_t, g_h, g_w = thw[0].tolist()
print(f"  grid = ({g_t}, {g_h}, {g_w}) -> 视觉 token 数 = t*h*w/merge^2 ="
      f" {g_t}*{g_h}*{g_w}/{2 ** 2} = {g_t * g_h * g_w // 4}")
print("  patch 展平维度 = C*temporal*patch*patch =",
      3 * ip.temporal_patch_size * ip.patch_size * ip.patch_size)
print("  实际 pixel_values.shape[1] =", px.shape[1])

print("  --- 再来一张 1000x700 的非方形图 ---")
img2 = np.random.randint(0, 256, (700, 1000, 3), dtype=np.uint8)
out2 = ip(images=img2, return_tensors="pt")
print("  输入 (H, W, C)      =", img2.shape)
print("  pixel_values.shape  =", tuple(out2["pixel_values"].shape))
print("  image_grid_thw      =", out2["image_grid_thw"].tolist())

print("=" * 74)
print("[6] rescale + normalize 的语义（Torchvision 后端会融合）")
from transformers.image_utils import SizeDict

tvip = Glm5NextImageProcessor()
t = torch.randint(0, 256, (1, 3, 4, 4)).to(torch.uint8)

# 分开做
x_sep = tvip.rescale(t, 1 / 255)
x_sep = tvip.normalize(x_sep.to(torch.float32), ip.image_mean, ip.image_std)
# 融合做
x_fused = tvip.rescale_and_normalize(
    t, do_rescale=True, rescale_factor=1 / 255, do_normalize=True,
    image_mean=ip.image_mean, image_std=ip.image_std,
)
print("  先 rescale 再 normalize :", x_sep.flatten()[:3].tolist())
print("  rescale_and_normalize   :", x_fused.flatten()[:3].tolist())
print("  allclose                :", bool(torch.allclose(x_sep, x_fused, atol=1e-5)))
print("  融合后的 dtype          :", x_fused.dtype, " 分开做的 dtype:", x_sep.dtype)
print("  image_mean =", ip.image_mean, " image_std =", ip.image_std)
print("  0.5/0.5/0.5 归一化后    :", ((0.5 - np.array(ip.image_mean)) / np.array(ip.image_std)).round(4).tolist())
print("  0.0 归一化后            :", ((0.0 - np.array(ip.image_mean)) / np.array(ip.image_std)).round(4).tolist())
print("  1.0 归一化后            :", ((1.0 - np.array(ip.image_mean)) / np.array(ip.image_std)).round(4).tolist())
del SizeDict

print("=" * 74)
print("[7] size 的四种形态 -> size_dict（get_size_dict 实测）")
from transformers.image_processing_utils import get_size_dict

for s, kw in [
    (224, {}),
    (224, {"default_to_square": False}),
    ((336, 336), {}),
    ((336, 336), {"height_width_order": False}),
    ({"height": 336, "width": 336}, {}),
    ({"shortest_edge": 224, "longest_edge": 1333}, {}),
]:
    try:
        print(f"  get_size_dict({s!r:36}, {kw}) -> {get_size_dict(s, **kw)}")
    except Exception as e:
        print(f"  get_size_dict({s!r:36}, {kw}) -> {type(e).__name__}: {e}")
try:
    get_size_dict(224, max_size=1333)
except Exception as e:
    print("  get_size_dict(224, max_size=1333) ->", type(e).__name__, str(e)[:80])

print("=" * 74)
print("[8] 文本 token 与视觉 token 在同一条序列上对齐（真实处理器）")
try:
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained("hf-internal-testing/tiny-random-glm5")
    print("  处理器 =", type(proc).__name__)
except Exception as e:
    print("  需要联网，跳过:", type(e).__name__, str(e)[:90])
