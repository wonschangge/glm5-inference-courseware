#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L2-07 实测脚本（视觉塔形状链路）。

跑法：  .venv/bin/python _data/recon/probe_l207.py
说明：  这是脚本，不是 src/transformers 下的源文件，**不计入覆盖率**。
        它的输出被逐字抄进 L2-07 的 source.md（用 text 块，不参与保真校验）。
"""
import warnings

warnings.filterwarnings("ignore")

import torch

from transformers import Glm5NextVisionConfig
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextVisionModel,
    Glm5NextVisionPatchEmbed,
    Glm5NextVisionRotaryEmbedding,
)
from transformers.vision_utils import get_vision_position_ids

cfg = Glm5NextVisionConfig()
side = cfg.image_size // cfg.patch_size
patches = side * side
merged = patches // cfg.spatial_merge_size**2
head_dim = cfg.hidden_size // cfg.num_heads

print("[1] 配置（Glm5NextVisionConfig() 默认值）")
print(f"    image_size={cfg.image_size} patch_size={cfg.patch_size} "
      f"spatial_merge_size={cfg.spatial_merge_size} temporal_patch_size={cfg.temporal_patch_size}")
print(f"    hidden_size={cfg.hidden_size} depth={cfg.depth} num_heads={cfg.num_heads} "
      f"out_hidden_size={cfg.out_hidden_size}")
print(f"    网格 {side}x{side} -> patch 数 {patches} -> 合并后 token 数 {merged}")
print(f"    head_dim={head_dim} -> spatial_dim={head_dim // 2} -> 频率个数={len(range(0, head_dim // 2, 2))}")

print("[2] patch embed：Conv3d 的权重形状与前向形状")
pe = Glm5NextVisionPatchEmbed(cfg)
print(f"    proj.weight = {tuple(pe.proj.weight.shape)}  stride = {pe.proj.stride}  "
      f"kernel = {pe.proj.kernel_size}")
px = torch.randn(patches, cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size**2)
with torch.no_grad():
    emb = pe(px)
print(f"    输入 {tuple(px.shape)} -> 输出 {tuple(emb.shape)}")

print("[3] position_ids：get_vision_position_ids 返回的形状与排布")
pos = get_vision_position_ids(torch.tensor([[1, 4, 4]]), cfg.spatial_merge_size)
print(f"    grid_thw=[[1,4,4]] merge=2 -> shape {tuple(pos.shape)}  (注释写的是 (2, N))")
print(f"    前 8 行 (h, w) = {pos[:8].tolist()}")
ok_block = all(
    {int(v) for v in pos[4 * k : 4 * k + 4, 0]} == {2 * (k // 2), 2 * (k // 2) + 1}
    and {int(v) for v in pos[4 * k : 4 * k + 4, 1]} == {2 * (k % 2), 2 * (k % 2) + 1}
    for k in range(4)
)
print(f"    每 4 个连续 token 恰好构成一个 2x2 块: {ok_block}")

print("[4] 轴向 RoPE：cos/sin 的形状与配对关系")
rope = Glm5NextVisionRotaryEmbedding(cfg)
print(f"    inv_freq.shape = {tuple(rope.inv_freq.shape)}  rope_type = {rope.rope_type}")
with torch.no_grad():
    cos, sin = rope(emb, pos)
print(f"    cos/sin shape = {tuple(cos.shape)}  (= (token, head_dim))")
a = pos[0].tolist()          # [0, 0]
rows = torch.tensor([
    [a[0], a[1]],            # 原样
    [a[0], a[1] + 5],        # 只改 w
    [a[0] + 5, a[1]],        # 只改 h
])
cos3, sin3 = rope(emb, rows)
print("    维度分块: [0:16]=h 频率, [16:32]=w 频率, [32:64]=前 32 维的副本")
print(f"    只改 w（行 0 vs 行 1）: cos 前 16 维相同 = "
      f"{bool(torch.allclose(cos3[0, :16], cos3[1, :16]))}, 16..32 维不同 = "
      f"{not bool(torch.allclose(cos3[0, 16:32], cos3[1, 16:32]))}")
print(f"    只改 h（行 0 vs 行 2）: cos 16..32 维相同 = "
      f"{bool(torch.allclose(cos3[0, 16:32], cos3[2, 16:32]))}, 前 16 维不同 = "
      f"{not bool(torch.allclose(cos3[0, :16], cos3[2, :16]))}")
print(f"    cos 的前 32 维与后 32 维完全相同（所以 rotate_half 把 i 与 i+32 配对）: "
      f"{bool(torch.allclose(cos3[:, :32], cos3[:, 32:]))}")

print("[5] 整塔前向（depth=1 便于跑）：三个出口的形状")
m = Glm5NextVisionModel(Glm5NextVisionConfig(depth=1)).eval()
grid = torch.tensor([[1, side, side]])
with torch.no_grad():
    out = m(px, grid_thw=grid)
print(f"    pixel_values {tuple(px.shape)} + grid_thw {grid.tolist()}")
print(f"    last_hidden_state = {tuple(out.last_hidden_state.shape)}")
print(f"    pooler_output     = {tuple(out.pooler_output.shape)}")
with torch.no_grad():
    same = torch.allclose(out.pooler_output, m.merger(out.last_hidden_state), atol=1e-5)
print(f"    pooler_output == merger(last_hidden_state): {same}")
print(f"    last_hidden_state 已经是合并后的 {merged} 个 token（不是 {patches} 个 patch）")

print("[6] 真实图像处理器：一张 336x336 的图产出多少 token")
from PIL import Image

from transformers import Glm5NextImageProcessor

ip = Glm5NextImageProcessor()
feat = ip(images=Image.new("RGB", (336, 336)), return_tensors="pt")
thw = feat["image_grid_thw"].tolist()
print(f"    pixel_values {tuple(feat['pixel_values'].shape)}  image_grid_thw = {thw}")
print(f"    num_image_tokens = {thw[0][0] * thw[0][1] * thw[0][2]} // {ip.merge_size}**2 = "
      f"{thw[0][0] * thw[0][1] * thw[0][2] // ip.merge_size ** 2}")

print("[7] 宽度约束：视觉 out_hidden_size 必须等于文本 hidden_size")
from transformers import Glm5NextConfig, Glm5NextTextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextModel

TC = dict(vocab_size=256, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
          num_key_value_heads=4, n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=1,
          intermediate_size=128, moe_intermediate_size=128, kv_lora_rank=32, q_lora_rank=32,
          v_head_dim=16, qk_nope_head_dim=16, head_dim=16, index_head_dim=32,
          index_n_heads=2, index_topk=32, pad_token_id=0, tie_word_embeddings=False)
toy_pv = torch.randn(4, 3 * 2 * 14 * 14)
toy_ids = torch.tensor([[1, 2, 5, 3, 4]])


def toy(out_hidden):
    vc = dict(depth=1, hidden_size=64, num_heads=4, out_hidden_size=out_hidden,
              projection_intermediate_size=64, intermediate_size=64, image_size=28,
              patch_size=14, spatial_merge_size=2)
    return Glm5NextModel(Glm5NextConfig(text_config=TC, vision_config=vc, image_token_id=5)).eval()


for oh in (64, 32):
    try:
        with torch.no_grad():
            toy(oh)(input_ids=toy_ids, pixel_values=toy_pv, image_grid_thw=torch.tensor([[1, 2, 2]]))
        print(f"    文本 hidden=64 视觉 out_hidden={oh:3d} -> 整条前向通过")
    except Exception as e:                                    # noqa: BLE001
        if "Image features and image tokens do not match" in str(e):
            print(f"    文本 hidden=64 视觉 out_hidden={oh:3d} -> 就卡在这一步："
                  f"Image features and image tokens do not match")
        else:
            print(f"    文本 hidden=64 视觉 out_hidden={oh:3d} -> 通过了 masked_scatter 这一步"
                  f"（随后在玩具配置的 {type(e).__name__} 上报错，与本步无关）")
print(f"    类默认值：hidden_size={Glm5NextTextConfig().hidden_size} vs "
      f"out_hidden_size={cfg.out_hidden_size}；"
      f"144 x 4096 = {144 * 4096}，144 x 1536 = {144 * 1536}")
