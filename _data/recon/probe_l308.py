#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L3-08 实测探针：解码层装配顺序 / GradientCheckpointingLayer 推理行为 / ModelOutput 约定。

全部结论必须来自本脚本的输出，不得凭记忆。
运行：.venv/bin/python courseware/_data/recon/probe_l308.py
"""
import inspect
import json
import sys

import torch

sys.path.insert(0, "/data/WORKSPACE/transformer-project/upstream-transformers/src")

from transformers import Glm5NextTextConfig  # noqa: E402
from transformers.modeling_layers import GradientCheckpointingLayer  # noqa: E402
from transformers.modeling_outputs import (  # noqa: E402
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)
from transformers.models.glm5_next import modeling_glm5_next as G  # noqa: E402

OUT = {}


def hdr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# --------------------------------------------------------------------------
hdr("1. DecoderLayer 的两个 mHC 子模块与执行顺序（构造层面）")
cfg = Glm5NextTextConfig(
    hidden_size=64,
    intermediate_size=128,
    moe_intermediate_size=32,
    num_hidden_layers=8,
    num_attention_heads=4,
    num_key_value_heads=4,
    head_dim=16,
    linear_num_heads=4,
    linear_head_dim=16,
    n_routed_experts=8,
    num_experts_per_tok=2,
    n_shared_experts=1,
    hc_mult=4,
)
cfg._attn_implementation = "eager"
print("layer_types      :", cfg.layer_types)
print("mlp_layer_types  :", cfg.mlp_layer_types)
print("hc_mult / hc_sinkhorn_iters / hc_eps:", cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps)
print("swiglu_limit     :", cfg.swiglu_limit)

d0 = G.Glm5NextTextDecoderLayer(cfg, 0)
d3 = G.Glm5NextTextDecoderLayer(cfg, 3)
print("d0.block_type    :", d0.block_type, "| attn:", type(d0.self_attn).__name__,
      "| mlp:", type(d0.mlp).__name__)
print("d3.block_type    :", d3.block_type, "| attn:", type(d3.self_attn).__name__,
      "| mlp:", type(d3.mlp).__name__)
print("d0.attn_hc is d0.ffn_hc ?", d0.attn_hc is d0.ffn_hc)
print("d0 submodules    :", [n for n, _ in d0.named_children()])
print("MRO              :", [c.__name__ for c in type(d0).__mro__[:4]])
OUT["d0_block"] = d0.block_type
OUT["d3_block"] = d3.block_type
OUT["d0_children"] = [n for n, _ in d0.named_children()]

# 前向顺序：用 hook 记录调用次序
hdr("2. 一次 forward 里子模块的真实调用次序（forward hook 实测）")
order = []


def mk(tag):
    def hook(mod, args, kwargs, out):
        order.append(tag)
    return hook


d0.attn_hc.register_forward_hook(mk("attn_hc"), with_kwargs=True)
d0.input_layernorm.register_forward_hook(mk("input_layernorm"), with_kwargs=True)
d0.self_attn.register_forward_hook(mk("self_attn"), with_kwargs=True)
d0.ffn_hc.register_forward_hook(mk("ffn_hc"), with_kwargs=True)
d0.post_attention_layernorm.register_forward_hook(mk("post_attention_layernorm"), with_kwargs=True)
d0.mlp.register_forward_hook(mk("mlp"), with_kwargs=True)

torch.manual_seed(0)
B, S = 1, 5
hs = torch.randn(B, S, cfg.hc_mult, cfg.hidden_size) * 0.3
out = d0(hs, attention_mask=None, position_ids=None, past_key_values=None, use_cache=False)
print("call order       :", " -> ".join(order))
print("return type      :", type(out).__name__, "| len:", len(out))
print("out[0].shape     :", tuple(out[0].shape), "| out[1]:", out[1])
OUT["call_order"] = order
OUT["out0_shape"] = list(out[0].shape)

# --------------------------------------------------------------------------
hdr("3. mHC 的写回：post ⊙ 子层输出 + combᵀ · 旧流")
hc = G.Glm5NextTextHyperConnection(cfg)
streams = torch.randn(B, S, cfg.hc_mult, cfg.hidden_size) * 0.3
post, comb, collapsed = hc(streams)
print("post.shape       :", tuple(post.shape), " comb.shape:", tuple(comb.shape),
      " collapsed.shape:", tuple(collapsed.shape))
print("post range       : [%.4f, %.4f]  (2*sigmoid -> (0,2))" % (post.min().item(), post.max().item()))
print("collapsed == (pre.unsqueeze(-1)*streams).sum(2)  ->  (B,S,D)")
sub = torch.randn(B, S, cfg.hidden_size) * 0.3
mixed = post.to(torch.float32).unsqueeze(-1) * sub.unsqueeze(-2) + torch.matmul(
    comb.to(torch.float32).transpose(-1, -2), streams)
print("mixed.shape      :", tuple(mixed.shape), "(= 4 条新流)")
OUT["post_shape"] = list(post.shape)
OUT["comb_shape"] = list(comb.shape)

# Sinkhorn：行/列和
c = comb.float()
rs = c.sum(-1)
cs = c.sum(-2)
print("Sinkhorn 后 row-sum: min %.6f max %.6f | col-sum: min %.6f max %.6f"
      % (rs.min(), rs.max(), cs.min(), cs.max()))
print("最大偏差         : %.3e (hc_eps=%.1e, hc_sinkhorn_iters=%d)"
      % (max((rs - 1).abs().max().item(), (cs - 1).abs().max().item()), cfg.hc_eps, cfg.hc_sinkhorn_iters))
OUT["sinkhorn_row_dev"] = (rs - 1).abs().max().item()
OUT["sinkhorn_col_dev"] = (cs - 1).abs().max().item()

# --------------------------------------------------------------------------
hdr("4. Sinkhorn 手算例子：2x2 矩阵，迭代 1 次 vs 2 次")
M0 = torch.tensor([[0.9, 0.4], [0.3, 0.8]])
print("M0 =\n", M0, "  行和:", M0.sum(-1).tolist(), " 列和:", M0.sum(-2).tolist())


def sinkhorn(m, iters, eps=1e-6):
    m = m / (m.sum(dim=-2, keepdim=True) + eps)      # 源码第 1 步：先按列归一
    for _ in range(iters - 1):
        m = m / (m.sum(dim=-1, keepdim=True) + eps)  # 行
        m = m / (m.sum(dim=-2, keepdim=True) + eps)  # 列
    return m


for it in (1, 2, 3, 20):
    mm = sinkhorn(M0.clone(), it)
    print("iters=%2d ->\n%s  行和: %s  列和: %s"
          % (it, mm.numpy().round(6), mm.sum(-1).numpy().round(6), mm.sum(-2).numpy().round(6)))
OUT["sinkhorn_demo_20"] = sinkhorn(M0.clone(), 20).tolist()

# --------------------------------------------------------------------------
hdr("5. GradientCheckpointingLayer 在推理时的实际行为")
print("gradient_checkpointing 类属性 :", GradientCheckpointingLayer.gradient_checkpointing)
print("_can_checkpoint_with_cache   :", GradientCheckpointingLayer._can_checkpoint_with_cache)
print("d0.gradient_checkpointing    :", d0.gradient_checkpointing)
print("d0.training                  :", d0.training)
print("Glm5NextTextModel.gradient_checkpointing 默认:", G.Glm5NextTextModel.__init__.__doc__ is None)

print("刚构造出的层有没有 _gradient_checkpointing_func:",
      hasattr(d0, "_gradient_checkpointing_func"))

called = {"ckpt": 0}
d0._gradient_checkpointing_func = lambda *a, **k: called.__setitem__("ckpt", called["ckpt"] + 1)
d0.eval()
with torch.no_grad():
    o_eval = d0(hs, attention_mask=None, position_ids=None, past_key_values=None, use_cache=False)
print("eval + no_grad：checkpoint 函数被调用次数:", called["ckpt"])
print("eval + no_grad：梯度是否被记录:", o_eval[0].requires_grad)

# torch.no_grad 推理（课件面向推理）
am = torch.ones(B, S, dtype=torch.bool)
with torch.no_grad():
    d3.eval()
    o_infer = d3(hs, attention_mask=am, position_ids=torch.arange(S).unsqueeze(0),
                 past_key_values=None, use_cache=False, prev_topk_indices=None)
print("d3 (indexed_attention) 输出:", tuple(o_infer[0].shape),
      "| topk_indices:", None if o_infer[1] is None else tuple(o_infer[1].shape))
print("d0 (linear_attention) 的 topk_indices 是否为 None:", out[1] is None)
print("eval 下 __call__ 直接走 nn.Module.__call__（无 checkpoint 分支）:",
      called["ckpt"] == 0)
OUT["ckpt_calls_eval"] = called["ckpt"]
OUT["d3_topk_shape"] = None if o_infer[1] is None else list(o_infer[1].shape)
OUT["d0_topk_is_none"] = out[1] is None
print("layer_topk 形状:", None if o_infer[1] is None else tuple(o_infer[1].shape),
      "| cfg.num_topk_indices/ index_topk:", getattr(cfg, "index_topk", None),
      getattr(cfg, "num_topk_indices", None))

# 训练态对照：打开 gradient_checkpointing 才会走 checkpoint 分支
d0.train()
d0.gradient_checkpointing = True
msgs = []


def fake_ckpt2(fn, *a, **k):
    called["ckpt"] += 1
    return fn(*a, **k)


d0._gradient_checkpointing_func = fake_ckpt2
before = called["ckpt"]
try:
    d0(hs, attention_mask=None, position_ids=None, past_key_values=None, use_cache=False)
except Exception as e:  # noqa: BLE001
    msgs.append(repr(e)[:120])
print("train()+gradient_checkpointing=True 时 checkpoint 调用次数增量:",
      called["ckpt"] - before, msgs)
OUT["ckpt_calls_train"] = called["ckpt"] - before

# --------------------------------------------------------------------------
hdr("6. modeling_outputs：数据类约定")
o = MoeModelOutputWithPast(last_hidden_state=torch.zeros(1, 3, 8), past_key_values="CACHE")
print("type(o)          :", type(o).__name__)
print("o.keys()         :", list(o.keys()))
print("o.to_tuple()     :", tuple(type(x).__name__ for x in o.to_tuple()))
print("o[0] is last_hidden_state:", o[0] is o.last_hidden_state)
print("o['past_key_values']      :", o["past_key_values"])
print("缺失字段的值      :", o.hidden_states, o.attentions, o.router_logits)
print("dataclass fields :", [f.name for f in __import__("dataclasses").fields(MoeModelOutputWithPast)])
print("是 dict 吗        :", isinstance(o, dict), "| 可解包:", list(dict(o).keys()))
print("MoeCausalLMOutputWithPast fields:",
      [f.name for f in __import__("dataclasses").fields(MoeCausalLMOutputWithPast)])
print("ModelOutput.to_tuple 源码:\n",
      inspect.getsource(type(o).to_tuple).strip()[:600])
OUT["moe_keys"] = list(o.keys())
OUT["moe_tuple"] = [type(x).__name__ for x in o.to_tuple()]

# --------------------------------------------------------------------------
hdr("7. swiglu_limit 手算例子（Glm5NextTextMLP.forward 的 clamp）")
print("cfg.swiglu_limit =", cfg.swiglu_limit)
gate_raw = [12.0, -3.0, 0.5]
up_raw = [15.0, -12.0, 2.0]
g = torch.tensor(gate_raw).clamp(min=None, max=cfg.swiglu_limit)
u = torch.tensor(up_raw).clamp(min=-cfg.swiglu_limit, max=cfg.swiglu_limit)
print("gate raw %s -> clamp(max=%.1f) -> %s" % (gate_raw, cfg.swiglu_limit, g.tolist()))
print("up   raw %s -> clamp(±%.1f)   -> %s" % (up_raw, cfg.swiglu_limit, u.tolist()))
print("silu(gate)*up ->", (torch.nn.functional.silu(g) * u).tolist())
print("未裁剪的 silu*up ->", (torch.nn.functional.silu(torch.tensor(gate_raw)) * torch.tensor(up_raw)).tolist())
OUT["swiglu"] = dict(gate=g.tolist(), up=u.tolist(),
                     out=(torch.nn.functional.silu(g) * u).tolist())

print("\n@@JSON@@" + json.dumps(OUT, ensure_ascii=False))
