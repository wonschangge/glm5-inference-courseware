#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L3-07 实测探针：遗忘门（Glm5NextTextForgetGate）与门控范数（Glm5NextTextRMSNormGated）

本脚本只做测量、不写课件。产出的每个数字都直接抄进 L3-07 的 source.md / lesson.js。
用法： /data/WORKSPACE/transformer-project/.venv/bin/python _data/recon/probe_l307.py
"""
import math
import sys

import torch

from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextTextForgetGate,
    Glm5NextTextRMSNormGated,
    Glm5NextPreTrainedModel,
    recurrent_kimi_delta_attention,
)

torch.manual_seed(0)
OUT = []


def P(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s)


def hr(t):
    P("")
    P("=" * 78)
    P(t)
    P("=" * 78)


# ---------------------------------------------------------------- 0 配置
hr("0. 默认 config 里与本科相关的字段")
cfg0 = Glm5NextTextConfig()
for k in ("hidden_size", "linear_head_dim", "linear_num_heads", "linear_conv_kernel_dim",
          "linear_lower_bound", "rms_norm_eps", "initializer_range"):
    P(f"  {k:24s} = {getattr(cfg0, k)!r}")

# safe_gate 开关：configuration_glm5_next.py 里的逃生门
cfg_safe_off = Glm5NextTextConfig()
cfg_safe_off.linear_attn_config = {"safe_gate": False}
cfg_safe_off.linear_lower_bound = None
P("  把 linear_attn_config={'safe_gate': False} 且 gate_lower_bound=None 传入 ->",
  "linear_lower_bound =", cfg_safe_off.linear_lower_bound)

hr("0b. 逃生门走真实构造路径（configuration_glm5_next.py:190-198）")
for kw in ({}, {"gate_lower_bound": None}, {"safe_gate": False, "gate_lower_bound": None},
           {"safe_gate": False}, {"gate_lower_bound": -3.0}):
    c = Glm5NextTextConfig(**({"linear_attn_config": kw} if kw else {}))
    P(f"    linear_attn_config={kw!s:52s} -> linear_lower_bound={c.linear_lower_bound!r}")

# ---------------------------------------------------------------- 1 形状
hr("1. 参数形状：A_log 每头一个，dt_bias 每 (头, 通道) 一个")
cfg = Glm5NextTextConfig(hidden_size=64, linear_head_dim=8, linear_num_heads=4)
fg = Glm5NextTextForgetGate(cfg)
for n, p in fg.named_parameters():
    P(f"  {n:12s} {tuple(p.shape)!s:14s} numel={p.numel():6d}")
P(f"  qkv_dim = head_dim * num_heads = {fg.head_dim} * {fg.num_heads} = {fg.qkv_dim}")
P(f"  A_log.view(1,1,num_heads,1) 广播到 (b,s,{fg.num_heads},{fg.head_dim}) -> 每头一个衰减率")
P(f"  参数量合计 = {sum(p.numel() for p in fg.parameters())}")

# ---------------------------------------------------------------- 2 初始化
hr("2. _init_weights：A_log / dt_bias 的实际取值（跑真代码，不靠读）")


def fake_model(config):
    """借 Glm5NextPreTrainedModel._init_weights，但不构造 45 层主干。"""
    m = Glm5NextPreTrainedModel.__new__(Glm5NextPreTrainedModel)
    m.config = config
    m.name_or_path = ""
    return m


for bound in (-5.0, None):
    c = Glm5NextTextConfig(hidden_size=64, linear_head_dim=8, linear_num_heads=4,
                           linear_lower_bound=bound)
    m = Glm5NextTextForgetGate(c)
    fake_model(c)._init_weights(m)
    with torch.no_grad():
        al = m.A_log
        dr = torch.exp(al.float())
        dt = m.dt_bias.float()
        sp = torch.nn.functional.softplus(dt)
        P(f"  bound={bound!r}")
        P(f"    A_log      min={al.min():+.6f} max={al.max():+.6f}  (numel={al.numel()})")
        P(f"    decay_rate exp(A_log)  min={dr.min():.6f} max={dr.max():.6f}")
        P(f"    dt_bias    min={dt.min():+.6f} max={dt.max():+.6f}")
        P(f"    softplus(dt_bias)      min={sp.min():.6f} max={sp.max():.6f}")

# FLA 那套 init 的解析值（供手算对照）
P("  手算对照：")
P(f"    log(1e-3) = {math.log(1e-3):+.6f}   log(1e-1) = {math.log(1e-1):+.6f}")
for u in (1e-3, 1e-2, 1e-1):
    P(f"    softplus^-1({u}) = log(exp({u})-1) = {math.log(math.exp(u) - 1):+.6f}")
P(f"    log(1)= {math.log(1.0):.6f}  log(16)={math.log(16.0):.6f}"
  f"  -> A_log=log(uniform(1,16)) 即 decay_rate~U(1,16)")

# ---------------------------------------------------------------- 3 值域
hr("3. 前向值域：下界截断把 g 关进 (-5, 0)")
cfg = Glm5NextTextConfig(hidden_size=8, linear_head_dim=1, linear_num_heads=1)
raw = torch.linspace(-30, 60, 7).view(1, 7, 1)
P("  令 f_b_proj(f_a_proj(x)) 的原始打分 = 下面这一列（用它绕过随机权重）：")
P("    raw     safe: -5*sigmoid(exp(0)*raw)    unsafe: -1*softplus(raw)   exp(safe)      exp(unsafe)")
for i in range(raw.shape[1]):
    x = raw[0, i, 0]
    g_safe = -5.0 * torch.sigmoid(torch.exp(torch.tensor(0.0)) * x)
    g_uns = -torch.nn.functional.softplus(x)
    e_s = torch.exp(g_safe)
    e_u = torch.exp(g_uns)
    P(f"    {x.item():+6.1f}   {g_safe.item():+18.6f}   {g_uns.item():+18.4f}"
      f"   {e_s.item():.6e}   {e_u.item():.6e}")

P("")
P("  A_log（=decay_rate）变大后，safe 分支的 sigmoid 会不会饱和？（raw=-4.5 时）")
for A in (1.0, 4.0, 16.0):
    v = torch.sigmoid(torch.tensor(A) * torch.tensor(-4.5))
    P(f"    decay_rate={A:5.1f} -> gate = -5*sigmoid({A:5.1f}*(-4.5)) = {-5*v.item():+.6f}"
      f"   sigmoid梯度 = {(v*(1-v)).item():.3e}")

hr("3b. 真模块 + 真初始化 + 随机输入：初始状态下的遗忘门到底多大")
for bound in (-5.0, None):
    c = Glm5NextTextConfig(hidden_size=256, linear_head_dim=16, linear_num_heads=4,
                           linear_lower_bound=bound)
    m = Glm5NextTextForgetGate(c)
    fake_model(c)._init_weights(m)
    m.eval()
    with torch.no_grad():
        x = torch.randn(1, 32, 256)
        g = m(x)
        dr = torch.exp(m.A_log.float()).view(1, 1, 4, 1)
        P(f"  bound={bound!r}  gate: min={g.min():+.6f} max={g.max():+.6f} "
          f"mean={g.mean():+.6f}")
        P(f"      decay=exp(gate): min={g.exp().min():.6f} max={g.exp().max():.6f} "
          f"mean={g.exp().mean():.6f}")
        P(f"      每头 decay_rate（bound=None 时是 U(1,16) 采样）: "
          f"{[round(v, 3) for v in dr.flatten().tolist()]}")

# ---------------------------------------------------------------- 4 fp32 下溢
hr("4. fp32 里 exp(g) 什么时候恰好等于 0")
for v in (-5.0, -20.0, -44.0, -87.0, -88.0, -90.0, -104.0, -320.0):
    e = torch.exp(torch.tensor(v, dtype=torch.float32))
    P(f"    exp({v:8.1f}) = {e.item():.6e}   is_zero={bool(e.item()==0.0)}")
P(f"    e^-5 = {math.exp(-5):.9f}")
P("    衰减率为 floor 时，累积到 fp32 归零需要多少步："
  f" ceil(103.28/5) = {math.ceil(103.28/5)} 步（exp(-103.28) 已是 0）")

# ---------------------------------------------------------------- 5 递推实测
hr("5. 递推实测：recurrent_kimi_delta_attention 里的 decay = exp(g)")
b, s, h, d = 1, 24, 1, 4
q = torch.randn(b, s, h, d)
k = torch.randn(b, s, h, d)
v = torch.randn(b, s, h, d)
beta = torch.full((b, s, h), 0.5)
for gv in (-5.0, -1.0, -0.05):
    g = torch.full((b, s, h), gv)
    out, st = recurrent_kimi_delta_attention(q, k, v, g=g, beta=beta,
                                            initial_state=None, output_final_state=True)
    P(f"    gate={gv:+.3f}  decay=exp(g)={math.exp(gv):.6f}   |final_state|_max="
      f"{st.abs().max().item():.6e}")
# 纯衰减：k=v=0, beta=0，初始状态全 1，只有 last_state = last_state * exp(g) 在起作用
P("    纯衰减实测（k=v=0、beta=0、初始状态全 1，只跑乘法）：")
for gv in (-5.0, -1.0, -0.05):
    z = torch.zeros(b, s, h, d)
    st0 = torch.ones(b, h, d, d)
    _, st = recurrent_kimi_delta_attention(z, z, z, g=torch.full((b, s, h), gv),
                                           beta=torch.zeros(b, s, h), initial_state=st0,
                                           output_final_state=True)
    P(f"      gate={gv:+.3f} 24 步后 |state| = {st.abs().max().item():.6e}"
      f"   hand: exp({gv*24:+.1f}) = {math.exp(gv*24):.6e}")
P("    -> 逐步下界只保证「不瞬间清零」，累积 21 步仍会在 fp32 里归零")

# ---------------------------------------------------------------- 6 分块形式的 cumsum 差
hr("6. 分块形式：逐 step 有界 != 块内有界（cuomsum 之后才做差）")
cs = 64
gc = torch.full((cs,), -5.0).cumsum(0)
# 源码写法：g.unsqueeze(-2) - g.unsqueeze(-3) -> [i,j] = g[j] - g[i]
diff = gc.view(1, -1) - gc.view(-1, 1)
strict = torch.triu(torch.ones(cs, cs, dtype=torch.bool), diagonal=1)
dm = torch.exp(diff.masked_fill(strict, float("-inf")))
P(f"    g_cum 首/末 = {gc[0].item():.1f} / {gc[-1].item():.1f}  (chunk_size={cs})")
P(f"    decay_mask 最大值 = {dm.max().item()!r}  （inf 表示 fp32 上溢）")
P(f"    decay_mask 最小值 = {dm.min().item():.6e}   对角线上 = {dm.diagonal().max().item():.6f}")
gc2 = torch.full((cs,), -1.0).cumsum(0)
dm2 = torch.exp((gc2.view(1, -1) - gc2.view(-1, 1)).masked_fill(strict, float("-inf")))
P(f"    gate=-1 时 decay_mask 最大 = {dm2.max().item():.6e}")
gc3 = torch.full((cs,), -0.05).cumsum(0)
dm3 = torch.exp((gc3.view(1, -1) - gc3.view(-1, 1)).masked_fill(strict, float("-inf")))
P(f"    gate=-0.05（初始化附近）decay_mask 最大 = {dm3.max().item():.6e}")
P(f"    exp(315) fp32 = {torch.exp(torch.tensor(315.0)).item()!r}"
  f"   exp(88) = {torch.exp(torch.tensor(88.0)).item():.6e}"
  f"   exp(89) = {torch.exp(torch.tensor(89.0)).item()!r}")

hr("6b. 直接跑真实的分块实现：gate 全按到下界 -5 会发生什么")
from transformers.models.glm5_next.modeling_glm5_next import chunk_kimi_delta_attention  # noqa: E402
# 真实调用约定（LinearAttention.forward 733-761 行）：
#   query/key/value: (b, s, heads, head_dim)   g: (b, s, heads, head_dim)   beta: (b, s, heads)
b2, s2, h2, d2 = 1, 64, 1, 4
q2 = torch.randn(b2, s2, h2, d2)
k2 = torch.randn(b2, s2, h2, d2)
v2 = torch.randn(b2, s2, h2, d2)
for gv, tag in ((-5.0, "全按下界"), (-1.0, "gate=-1"), (-0.05, "初始化附近")):
    o2, _ = chunk_kimi_delta_attention(q2, k2, v2, g=torch.full((b2, s2, h2, d2), gv),
                                       beta=torch.full((b2, s2, h2), 0.5))
    P(f"    {tag:8s} gate={gv:+.2f} -> out 含 inf: {bool(torch.isinf(o2).any())}"
      f"  含 nan: {bool(torch.isnan(o2).any())}  max|out|={o2.abs().max().item():.6e}")
# 无下界那一支能造出的极端：A=16、raw=+20
g_uns = -16 * math.log1p(math.exp(20))
P(f"    无下界支的极端：gate = -16*softplus(20) = {g_uns:.1f}"
  f"  -> exp(gate) fp32 = {torch.exp(torch.tensor(g_uns)).item()!r}")
o3, _ = chunk_kimi_delta_attention(q2, k2, v2, g=torch.full((b2, s2, h2, d2), g_uns),
                                   beta=torch.full((b2, s2, h2), 0.5))
P(f"    gate={g_uns:.0f}（无下界可造出的值）-> 含 inf: {bool(torch.isinf(o3).any())}"
  f"  含 nan: {bool(torch.isnan(o3).any())}  max|out|={o3.abs().max().item()!r}")

# ---------------------------------------------------------------- 7 门控范数
hr("7. 门控范数 Glm5NextTextRMSNormGated")
cfg = Glm5NextTextConfig(hidden_size=16, rms_norm_eps=1e-6)
on = Glm5NextTextRMSNormGated(8, eps=cfg.rms_norm_eps)
fake_model(cfg)._init_weights(on)
P(f"  weight 初值 全 1 = {bool((on.weight == 1).all())}   activation = {on.activation!r}"
  f"   variance_epsilon = {on.variance_epsilon!r}  (== config.rms_norm_eps)")
x = torch.randn(1, 3, 8)
gate = torch.randn(1, 3, 8)
o = on(x, gate)
P(f"  输入 dtype={x.dtype} -> 输出 dtype={o.dtype}  shape={tuple(o.shape)}")
# 手工复算
xf = x.float()
vn = xf.pow(2).mean(-1, keepdim=True)
man = xf * torch.rsqrt(vn + 1e-6) * on.weight.float() * torch.sigmoid(gate.float())
P(f"  与手工 fp32 复算 allclose = {bool(torch.allclose(o, man.to(o.dtype), atol=1e-6))}")
# gate=None 会怎样？
try:
    on(x, None)
    P("  gate=None: 没报错")
except Exception as e:
    P(f"  gate=None: {type(e).__name__}: {str(e)[:80]}")
# 半精度下的行为（权重要不要降精度）
xb = x.to(torch.bfloat16)
ob = on(xb, gate.to(torch.bfloat16))
P(f"  bf16 输入 -> {ob.dtype}；内部按 fp32 归一（源码注释 Strict FP32 norm）")

hr("6c. 分块形式内部逐步复算（照抄 560-579 行的形状约定）")
# 与真实实现一致：g 的形状是 (b, h, chunks, chunk, dim)
b3, h3, nch, cs3, d3 = 1, 1, 3, 64, 4
for gv in (-5.0, -1.0, -0.05):
    g3 = torch.full((b3, h3, nch, cs3, d3), gv).cumsum(dim=-2)
    smask = torch.triu(torch.ones(cs3, cs3, dtype=torch.bool), diagonal=1)[..., None]
    decay_mask = (g3.unsqueeze(-2) - g3.unsqueeze(-3)).masked_fill(smask, float("-inf")).exp()
    P(f"    gate={gv:+.2f}: decay_mask max={decay_mask.max():.6e} min={decay_mask.min():.1e}"
      f"  非对角最大值={decay_mask.masked_fill(torch.eye(cs3, dtype=torch.bool)[..., None], 0).max():.6e}")
    P(f"             差方向检查：[i,j] = g[i]-g[j]（下三角差<=0）"
      f"  max(g[i]-g[j]) 全矩阵 = {(g3.unsqueeze(-2) - g3.unsqueeze(-3)).max():.1f}"
      f"  下三角最大 = {(g3.unsqueeze(-2) - g3.unsqueeze(-3)).masked_fill(smask, float('-inf')).max():.1f}")
# exp(cumsum) 那一支：k_cumdecay = attn @ (k_beta * g.exp())
for gv in (-5.0, -1.0, -0.05):
    gc4 = torch.full((cs,), gv).cumsum(0)
    e = gc4.exp()
    P(f"    gate={gv:+.2f}: exp(cumsum) 最小={e.min():.1e}  恰好为 0 的步数="
      f"{int((e == 0).sum())}/{cs}"
      f"  跨块项 (g_last-g).exp() 最小={torch.exp(gc4[-1] - gc4).min():.1e}")
P("    => 分块形式里会归零的是 exp(cumsum)（老 token 的写入权重），不是 decay_mask")

hr("9. 默认 config（hidden=4096 / head_dim=128 / heads=64）下的参数量")
cfgD = Glm5NextTextConfig()
fgD = Glm5NextTextForgetGate(cfgD)
tot = 0
for n, p in fgD.named_parameters():
    P(f"    {n:16s} {tuple(p.shape)!s:16s} numel={p.numel():9,d}")
    tot += p.numel()
P(f"    遗忘门合计 = {tot:,d}")
qkv = 3 * cfgD.hidden_size * fgD.qkv_dim
ga = cfgD.hidden_size * cfgD.linear_head_dim + cfgD.linear_head_dim * fgD.qkv_dim
P(f"    对照：q/k/v 三个投影 = 3*{cfgD.hidden_size}*{fgD.qkv_dim} = {qkv:,d}")
P(f"          g_a/g_b 输出门 = {ga:,d}；o_proj = {fgD.qkv_dim*cfgD.hidden_size:,d}；"
  f"conv1d = 3*{fgD.qkv_dim}*{cfgD.linear_conv_kernel_dim:,d}")
P(f"    遗忘门占 qkv 投影的 {tot/qkv*100:.2f}%")

hr("10. S4 手算表（高精度）")
P("    raw_score | gate=-5*sigmoid(raw) | decay=exp(gate) | 1-decay")
for x in (-30, -15, -8, -4, 0, 4, 8, 15, 30, 60):
    t = torch.tensor(float(x), dtype=torch.float64)
    g64 = -5.0 * torch.sigmoid(t)
    d64 = torch.exp(g64)
    P(f"    {x:+9.1f} | {g64.item():+.12f} | {d64.item():.9f} | {1-d64.item():.6f}")
P("    用 fp32 算同一个表（课件里显示的就是 fp32 结果）：")
for x in (-30, -15, -4, 0, 4, 15, 30, 60):
    t = torch.tensor(float(x), dtype=torch.float32)
    g32 = -5.0 * torch.sigmoid(t)
    P(f"    {x:+9.1f} | gate={g32.item():+.9f} | decay={torch.exp(g32).item():.9f}")
P(f"    floor: -5 -> decay = {math.exp(-5):.9f}"
  f"（= e^-5，实测 {torch.exp(torch.tensor(-5.0)).item():.9f}）")
P(f"    A_log=0 时 decay_rate = {math.exp(0.0):.1f}")

hr("11. dt_bias 手算对照表")
for u in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
    v = math.log(math.exp(u) - 1)
    P(f"    dt={u:<8g} -> dt_bias=softplus^-1(dt)={v:+.6f}"
      f"   反算 softplus(dt_bias)={math.log1p(math.exp(v)):.6g}")

hr("12. A_log 变大时 sigmoid 饱和（实测：门被冻住）")
for A in (1.0, 2.0, 4.0, 8.0, 16.0):
    for raw in (-4.5, 0.0):
        v = torch.sigmoid(torch.tensor(A * raw))
        P(f"    decay_rate={A:5.1f} raw={raw:+.1f} -> gate={-5*v.item():+.9f}"
          f"  sigmoid'={float(v*(1-v)):.3e}")

hr("8. 门控范数在 KDA 里的位置（源码 767-769 行）")
P("  gate = g_b_proj(g_a_proj(hidden_states))  -> view (b,s,num_heads,head_dim)")
P(f"  g_a_proj: hidden {cfg0.hidden_size} -> {cfg0.linear_head_dim}；"
  f"g_b_proj: {cfg0.linear_head_dim} -> qkv_dim {cfg0.linear_head_dim*cfg0.linear_num_heads}")
P("  o_norm(core_attn_out, gate) 然后才 o_proj —— 门控在归一化之后、输出投影之前")

open("/tmp/probe_l307.out", "w", encoding="utf-8").write("\n".join(OUT) + "\n")
