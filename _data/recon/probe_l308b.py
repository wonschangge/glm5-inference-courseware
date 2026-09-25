#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L3-08 探针之二：把 mHC 的**真实初始化**补上再测。

探针一里直接构造了 DecoderLayer，没有走 post_init()，所以 HyperConnection 的
fn / base / scale 还是 torch.empty 的未初始化内存 —— 那个 post 区间不能用。
这里按 modeling_glm5_next.py 的 _init_weights 逐字复现初始化，再测一遍。
"""
import json
import sys

import torch
from torch import nn

sys.path.insert(0, "/data/WORKSPACE/transformer-project/upstream-transformers/src")

from transformers import Glm5NextTextConfig  # noqa: E402
from transformers.models.glm5_next import modeling_glm5_next as G  # noqa: E402

OUT = {}
torch.manual_seed(0)

# _init_weights 里对 Glm5NextTextHyperConnection 的三行（L1410-1413 逐字复现）
def real_init(hc):
    with torch.no_grad():
        hc.fn.normal_(mean=0.0, std=0.02)
        hc.base.zero_()
        hc.scale.fill_(1.0)
    return hc


print("=" * 78)
print("1. 真实配置（hidden_size=4096, hc_mult=4）下 mHC 的形状与参数量")
print("=" * 78)
cfg = Glm5NextTextConfig()            # 默认：45 层 / 4096 / hc_mult=4
print("num_hidden_layers=%d hidden_size=%d hc_mult=%d swiglu_limit=%s rms_norm_eps=%s"
      % (cfg.num_hidden_layers, cfg.hidden_size, cfg.hc_mult, cfg.swiglu_limit, cfg.rms_norm_eps))
print("hc_sinkhorn_iters=%d hc_eps=%s" % (cfg.hc_sinkhorn_iters, cfg.hc_eps))
hc = real_init(G.Glm5NextTextHyperConnection(cfg))
print("fn.shape   :", tuple(hc.fn.shape), " = ((2+N)*N, N*D) = ((2+4)*4, 4*4096)")
print("base.shape :", tuple(hc.base.shape), " scale.shape:", tuple(hc.scale.shape))
n = sum(p.numel() for p in hc.parameters())
print("单个 mHC 参数量:", n, " 一层两个:", 2 * n, " 45 层:", 45 * 2 * n)
OUT["hc_params"] = n
OUT["hc_params_layer"] = 2 * n

print()
print("=" * 78)
print("2. 真实初始化下 post / comb 的取值分布")
print("=" * 78)
B, S = 2, 8
streams = torch.randn(B, S, cfg.hc_mult, cfg.hidden_size) * 0.3   # 模拟主干里的流
# 4096 维 x 16384 的 linear 在 CPU 上也不慢
with torch.no_grad():
    post, comb, collapsed = hc(streams)
print("post  min %.6f  max %.6f  mean %.6f  (2*sigmoid -> (0,2))"
      % (post.min(), post.max(), post.mean()))
print("post 与 1 的最大偏差: %.6f" % (post - 1).abs().max())
print("comb  row-sum 与 1 的最大偏差: %.3e" % (comb.float().sum(-1) - 1).abs().max())
print("comb  col-sum 与 1 的最大偏差: %.3e" % (comb.float().sum(-2) - 1).abs().max())
print("collapsed.shape:", tuple(collapsed.shape), " post.shape:", tuple(post.shape),
      " comb.shape:", tuple(comb.shape))
OUT["post_min"] = post.min().item()
OUT["post_max"] = post.max().item()
OUT["post_mean"] = post.mean().item()
OUT["comb_row_dev"] = (comb.float().sum(-1) - 1).abs().max().item()

print()
print("=" * 78)
print("3. 写回公式的逐流展开（手算核对：新流[i] = post[i]*out + sum_j comb[j,i]*res[j]）")
print("=" * 78)
D = 8
res = torch.randn(1, 1, cfg.hc_mult, D)
out = torch.randn(1, 1, D)
p = torch.sigmoid(torch.randn(1, 1, cfg.hc_mult)) * 2
c = torch.softmax(torch.randn(1, 1, cfg.hc_mult, cfg.hc_mult), dim=-1)
c = c / (c.sum(dim=-2, keepdim=True) + 1e-6)
mixed = p.unsqueeze(-1) * out.unsqueeze(-2) + torch.matmul(c.transpose(-1, -2), res)
manual = torch.stack([p[0, 0, i] * out[0, 0] + sum(c[0, 0, j, i] * res[0, 0, j] for j in range(4))
                      for i in range(4)])
print("mixed.shape:", tuple(mixed.shape), " 与手算一致:",
      torch.allclose(mixed[0, 0], manual, atol=1e-6))
print("第 0 条流 = post[0]*out + Σ_j comb[j,0]*res[j]  ->", manual[0][:4].tolist())
OUT["writeback_ok"] = bool(torch.allclose(mixed[0, 0], manual, atol=1e-6))

print()
print("=" * 78)
print("4. hc_mult = 1 时 mHC 退化成什么")
print("=" * 78)
cfg1 = Glm5NextTextConfig(hc_mult=1, hidden_size=64, intermediate_size=128, num_hidden_layers=4)
hc1 = real_init(G.Glm5NextTextHyperConnection(cfg1))
s1 = torch.randn(1, 3, 1, 64) * 0.3
with torch.no_grad():
    p1, c1, col1 = hc1(s1)
print("comb.shape (N=1):", tuple(c1.shape), " 值:", float(c1[0, 0, 0, 0]))
print("post 范围: [%.6f, %.6f]" % (p1.min(), p1.max()))
print("→ N=1 时 comb 只有一个元素且被 Sinkhorn 逼成 1；写回 = post ⊙ out + 旧流")
OUT["n1_comb"] = float(c1[0, 0, 0, 0])

print()
print("=" * 78)
print("5. 数据类：三个约定的实测证据")
print("=" * 78)
from transformers.modeling_outputs import MoeModelOutputWithPast  # noqa: E402
import dataclasses  # noqa: E402

o = MoeModelOutputWithPast(last_hidden_state=torch.zeros(2, 3, 4), past_key_values=None)
o.past_key_values = "C"
print("字段顺序:", [f.name for f in dataclasses.fields(o)])
print("全部字段默认值 None:", all(f.default is None for f in dataclasses.fields(o)))
print("o[0] is o.last_hidden_state:", o[0] is o.last_hidden_state)
print("keys()  =", list(o.keys()))
print("to_tuple 元素类型 =", [type(x).__name__ for x in o.to_tuple()])
print("isinstance(o, dict) =", isinstance(o, dict), " dict(o) keys =", list(dict(o).keys()))
o2 = MoeModelOutputWithPast(last_hidden_state=torch.zeros(1))
print("只给 last_hidden_state 时 keys() =", list(o2.keys()), "（past_key_values 为 None 被丢掉）")
OUT["fields_all_default_none"] = all(f.default is None for f in dataclasses.fields(o))

print("\n@@JSON@@" + json.dumps(OUT, ensure_ascii=False))
