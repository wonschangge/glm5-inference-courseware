#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L3-01「三种 RMSNorm」的实测脚本（CPU / meta device，不靠记忆）。

跑法：
    /data/WORKSPACE/transformer-project/.venv/bin/python \
        /data/WORKSPACE/transformer-project/_data/recon/probe_L3-01.py

输出全部数字都直接进 source.md 的 text 块。
"""
import inspect
import torch
import torch.nn as nn

from transformers.models.glm5_next import modeling_glm5_next as M
from transformers.models.glm5_next.configuration_glm5_next import (
    Glm5NextTextConfig,
    Glm5NextVisionConfig,
)

torch.manual_seed(0)
LINE = "=" * 72


def p(*a):
    print(*a)


p(LINE, "\n[1] 三个类的源码文本对比（只比 class 名与 docstring 之外的内容）")
src = inspect.getsource(M)
text_cls = inspect.getsource(M.Glm5NextTextRMSNorm)
vis_cls = inspect.getsource(M.Glm5NextRMSNorm)
p("Glm5NextTextRMSNorm 源码行数 :", len(text_cls.splitlines()))
p("Glm5NextRMSNorm     源码行数 :", len(vis_cls.splitlines()))
a = [l for l in text_cls.splitlines() if "Glm5NextTextRMSNorm" not in l]
b = [l for l in vis_cls.splitlines() if "Glm5NextRMSNorm" not in l]
p("去掉类名后的行是否逐字相同   :", a == b, f"({len(a)} 行)")

p(LINE, "\n[2] 参数量：三种 norm 各自的 Parameter 数")
for name in ("Glm5NextTextRMSNorm", "Glm5NextTextUnweightedRMSNorm", "Glm5NextTextRMSNormGated"):
    cls = getattr(M, name)
    arg = (4096,) if name != "Glm5NextTextUnweightedRMSNorm" else ()
    mod = cls(*arg)
    n = sum(x.numel() for x in mod.parameters())
    p(f"{name}{arg}: parameters={n}  buffers={sum(x.numel() for x in mod.buffers())}")
p("Glm5NextTextRMSNormGated(128):",
  sum(x.numel() for x in M.Glm5NextTextRMSNormGated(128).parameters()))
p("Glm5NextRMSNorm(256)          :",
  sum(x.numel() for x in M.Glm5NextRMSNorm(256).parameters()))

p(LINE, "\n[3] 类默认 eps 与 config 实际 eps（谁在真正生效）")
p("Glm5NextTextRMSNorm.__init__ 默认   :", inspect.signature(M.Glm5NextTextRMSNorm.__init__).parameters["eps"].default)
p("Glm5NextTextUnweightedRMSNorm 默认  :", inspect.signature(M.Glm5NextTextUnweightedRMSNorm.__init__).parameters["eps"].default)
p("Glm5NextTextRMSNormGated 默认       :", inspect.signature(M.Glm5NextTextRMSNormGated.__init__).parameters["eps"].default)
p("Glm5NextTextConfig().rms_norm_eps   :", Glm5NextTextConfig().rms_norm_eps)
p("Glm5NextVisionConfig().rms_norm_eps :", Glm5NextVisionConfig().rms_norm_eps)

p(LINE, "\n[4] 全部实例化点（谁传了 eps、传的是什么）")
import re
lines = inspect.getsource(M).splitlines()
for i, ln in enumerate(lines, 1):
    if re.search(r"= Glm5Next(Text)?(Unweighted)?RMSNorm(Gated)?\(", ln) or "RMSNorm(eps=" in ln:
        p(f"  L{i}: {ln.strip()}")

p(LINE, "\n[5] eps=1e-5 与 1e-6 的差异：按 token 的 RMS 分档实测")
x = torch.randn(4, 64, 4096, dtype=torch.float32)
def norm_eps(t, eps):
    v = t.float().pow(2).mean(-1, keepdim=True)
    return t.float() * torch.rsqrt(v + eps)
for rms_target in (1.0, 1e-1, 1e-2, 1e-3, 3.2e-4, 1e-4):
    t = x / x.pow(2).mean(-1, keepdim=True).sqrt() * rms_target
    o5, o6 = norm_eps(t, 1e-5), norm_eps(t, 1e-6)
    rel = ((o5 - o6).abs().max() / o6.abs().max()).item()
    shrink = (1 - (rms_target ** 2 / (rms_target ** 2 + 1e-5)) ** 0.5)
    p(f"  token RMS={rms_target:<8g}  两 eps 输出的最大相对差={rel:.3e}   "
      f"eps=1e-5 造成的收缩={shrink*100:.4f}%")

p(LINE, "\n[6] bf16 下还有区别吗（模型实际跑的精度）")
w = 1 + 0.01 * torch.randn(4096)
g = torch.randn(4, 64, 4096)
for dt in (torch.float32, torch.bfloat16):
    t = x.to(dt)
    outs = []
    for eps in (1e-5, 1e-6):
        mm = M.Glm5NextTextRMSNorm(4096, eps=eps).to(dt)
        with torch.no_grad():
            mm.weight.copy_(w.to(dt))
        outs.append(mm(t))
    p(f"  TextRMSNorm  {str(dt).split('.')[-1]:<8}: eps 1e-5 vs 1e-6 逐元素不同个数 = "
      f"{(outs[0] != outs[1]).sum().item()} / {outs[0].numel()}   dtype={outs[0].dtype}")
    outsg = []
    for eps in (1e-5, 1e-6):
        mg = M.Glm5NextTextRMSNormGated(4096, eps=eps).to(dt)
        with torch.no_grad():
            mg.weight.copy_(w.to(dt))
        outsg.append(mg(t, g.to(dt)))
    p(f"  RMSNormGated {str(dt).split('.')[-1]:<8}: eps 1e-5 vs 1e-6 逐元素不同个数 = "
      f"{(outsg[0] != outsg[1]).sum().item()} / {outsg[0].numel()}   dtype={outsg[0].dtype}")
p(LINE, "\n[6b] 参数留在 fp32、激活是 bf16 时，返回的 dtype 是谁")
for name in ("Glm5NextTextRMSNorm", "Glm5NextTextRMSNormGated"):
    cls = getattr(M, name)
    m = cls(4096, eps=1e-5)                       # 模块留在 fp32
    o1 = m(x.to(torch.bfloat16), gate=g.to(torch.bfloat16)) if "Gated" in name else m(x.to(torch.bfloat16))
    m2 = cls(4096, eps=1e-5).to(torch.bfloat16)   # 整个模块 bf16（真实加载路径）
    o2 = m2(x.to(torch.bfloat16), gate=g.to(torch.bfloat16)) if "Gated" in name else m2(x.to(torch.bfloat16))
    p(f"  {name:<26} weight={str(m.weight.dtype):<14} → 返回 {str(o1.dtype):<14}"
      f"| weight=bf16 → 返回 {o2.dtype}")

p(LINE, "\n[6c] 真实的 g_a_proj/g_b_proj 在 init 下给出的 gate 分布")
cfg2 = Glm5NextTextConfig(hidden_size=4096, num_hidden_layers=4)
cfg2.rms_norm_eps = 1e-5
attn = M.Glm5NextTextLinearAttention(cfg2, 0)
nn.init.normal_(attn.g_a_proj.weight, mean=0.0, std=cfg2.initializer_range)
nn.init.normal_(attn.g_b_proj.weight, mean=0.0, std=cfg2.initializer_range)
with torch.no_grad():
    h = torch.randn(2, 8, cfg2.hidden_size)
    h = h / h.pow(2).mean(-1, keepdim=True).sqrt()      # 每个 token 的 RMS 归一到 1
    gate_pre = attn.g_b_proj(attn.g_a_proj(h))
    s = torch.sigmoid(gate_pre)
p("  head_dim=%d, qkv_dim=%d, g_a_proj=%s, g_b_proj=%s"
  % (attn.head_dim, attn.qkv_dim, tuple(attn.g_a_proj.weight.shape), tuple(attn.g_b_proj.weight.shape)))
p(f"  gate 预激活: 均值={gate_pre.mean():+.4f} 标准差={gate_pre.std():.4f} "
  f"min={gate_pre.min():+.4f} max={gate_pre.max():+.4f}")
p(f"  sigmoid(gate): 均值={s.mean():.4f} min={s.min():.4f} max={s.max():.4f} "
  f"→ 输出被整体乘上约 {s.mean():.3f}")


p(LINE, "\n[7] 零向量 / 极小 RMS 的极限行为（eps 存在的理由）")
def rms_out(const, eps):
    t = torch.full((1, 1, 4096), const)
    m = M.Glm5NextTextRMSNorm(4096, eps=eps)
    with torch.no_grad():
        m.weight.copy_(torch.ones(4096))
    return m(t).abs().max().item(), 1.0 / (eps ** 0.5)
for eps in (1e-5, 1e-6):
    p(f"  eps={eps:g}: 放大上限 1/sqrt(eps) = {rms_out(0.0, eps)[1]:.1f}")
    for const in (1e-6, 1e-4, 1e-3):
        p(f"    常量输入 {const:g} → 输出绝对值 = {rms_out(const, eps)[0]:.6g}"
          f"   （输入自身 {const:g}，被放大了 {rms_out(const, eps)[0]/const:.1f} 倍）")
p("  eps 的物理含义：token 的 RMS 低于 sqrt(eps) 时，norm 不再把它拉到 1，")
p("  而是只放大到 rms/sqrt(eps)。sqrt(1e-5)=%.3e, sqrt(1e-6)=%.3e"
  % (1e-5 ** 0.5, 1e-6 ** 0.5))


p(LINE, "\n[8] Gated 版里权重乘法的精度位置（fp32 里乘 vs bf16 里乘）")
t = x.to(torch.bfloat16)
wq = (1 + 0.01 * torch.randn(4096)).to(torch.bfloat16)
raw = t.float()
v = raw.pow(2).mean(-1, keepdim=True)
h = raw * torch.rsqrt(v + 1e-5)
late = (h * wq.float()).to(torch.bfloat16)          # Gated 的做法：fp32 里乘，最后再降
early = h.to(torch.bfloat16) * wq                   # Text 的做法：先降再乘
diff = (late != early).sum().item()
rel = ((late.float() - early.float()).abs().max() / late.float().abs().max()).item()
p(f"  逐元素不同个数 = {diff} / {late.numel()}  ({100*diff/late.numel():.2f}%)")
p(f"  最大相对差     = {rel:.3e}   （bf16 的 1 ulp ≈ 2^-8 = 3.9e-3）")

p(LINE, "\n[9] gate 的实际分布：sigmoid(gate) 落在哪里")
for std in (0.02, 0.1):
    gg = torch.randn(200000) * std
    s = torch.sigmoid(gg)
    p(f"  gate ~ N(0, {std}): sigmoid 均值={s.mean():.4f}  "
      f"min={s.min():.4f} max={s.max():.4f}  占 [0.05,0.95] 的比例={((s>0.05)&(s<0.95)).float().mean():.4f}")

p(LINE, "\n[10] 也从源码里数：o_norm 的 gate 是几层、瓶颈多宽")
src_txt = inspect.getsource(M.Glm5NextTextLinearAttention)
for ln in src_txt.splitlines():
    if "g_a_proj" in ln or "g_b_proj" in ln or "o_norm" in ln or "gate" in ln:
        p("   " + ln.strip())

p(LINE, "\n[11] 整个模型的 norm 参数占比（meta device，不占内存）")
cfg = Glm5NextTextConfig()
p("  Glm5NextTextConfig(): hidden_size=%d layers=%d rms_norm_eps=%g"
  % (cfg.hidden_size, cfg.num_hidden_layers, cfg.rms_norm_eps))
with torch.device("meta"):
    model = M.Glm5NextTextModel(cfg)
tot = sum(x.numel() for x in model.parameters())
per = {}
for name, mod in model.named_modules():
    k = type(mod).__name__
    if "RMSNorm" in k:
        per[k] = per.get(k, 0) + sum(x.numel() for x in mod.parameters())
n_norm = sum(per.values())
p("  模型总参数              : %,d".replace("%,d", f"{tot:,}"))
for k, v in sorted(per.items()):
    p(f"  {k:<32}: {v:>12,}  ({100*v/tot:.6f}% of total)")
p(f"  三种 norm 合计           : {n_norm:>12,}  ({100*n_norm/tot:.6f}% of total)")
p("  UnweightedRMSNorm 参数   :",
  sum(x.numel() for x in M.Glm5NextTextUnweightedRMSNorm().parameters()), "（0 参数）")
n_inst = sum(1 for _, mod in model.named_modules() if "RMSNorm" in type(mod).__name__)
p("  norm 实例个数            :", n_inst)

p(LINE, "\n[12] 加权 vs 无权重的差别：只是差一个乘数")
u = M.Glm5NextTextUnweightedRMSNorm(eps=cfg.rms_norm_eps)
wmod = M.Glm5NextTextRMSNorm(64, eps=cfg.rms_norm_eps)
t = torch.randn(2, 5, 64)
p("  UnweightedRMSNorm(x) 与 RMSNorm(x) 在 weight=1 时相等 :",
  torch.allclose(u(t), wmod(t), atol=1e-6))
with torch.no_grad():
    wmod.weight.copy_(torch.randn(64))
p("  与 weight 的比值是否恒等于 weight（逐元素）           :",
  torch.allclose(u(t) * wmod.weight, wmod(t), atol=1e-5))
p("  rsqrt(mean(x^2)+eps) 是逐 token 的标量：形状           :",
  tuple(torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-5).shape), "vs x 的形状", tuple(t.shape))

p(LINE, "\n[13] 分块为什么对 norm 成立、对注意力不成立（用仓库自带的 apply_chunking_to_forward）")
from transformers.pytorch_utils import apply_chunking_to_forward

xn = torch.randn(2, 8, 64)
nm = M.Glm5NextTextRMSNorm(64, eps=1e-5)
with torch.no_grad():
    nm.weight.copy_(1 + 0.05 * torch.randn(64))
full = nm(xn)
c_seq = apply_chunking_to_forward(nm.forward, 2, 1, xn)      # 沿 seq（dim=1）分 4 块
p("  沿 seq 分块 与 整块: 逐位相同 =", bool((full == c_seq).all().item()),
  " 最大绝对差 =", (full - c_seq).abs().max().item())
try:
    apply_chunking_to_forward(nm.forward, 16, -1, xn)        # 沿 hidden（dim=-1）分 4 块
    p("  加权版沿 hidden 分块: 没有报错（不应该）")
except RuntimeError as e:
    p("  加权版沿 hidden 分块: RuntimeError ->", str(e)[:90])
unw = M.Glm5NextTextUnweightedRMSNorm(eps=1e-5)
full_u = unw(xn)
c_hid_u = apply_chunking_to_forward(unw.forward, 16, -1, xn)
p("  无权重版沿 hidden 分块: 不报错，但结果错了 -> 最大绝对差 =",
  (full_u - c_hid_u).abs().max().item(), " 相对差 =",
  ((full_u - c_hid_u).abs().max() / full_u.abs().max()).item())
def _sm_seq(t):
    return torch.softmax(t, dim=1)          # 沿 seq 做 softmax —— 注意力/卷积那一类
sm = _sm_seq(xn)
sm_c = apply_chunking_to_forward(_sm_seq, 2, 1, xn)
p("  反例：沿 seq 的 softmax 再沿 seq 分块: 逐位相同 =", bool((sm == sm_c).all().item()),
  " 最大绝对差 =", (sm - sm_c).abs().max().item())
gm = M.Glm5NextTextRMSNormGated(64, eps=1e-5)
gt = torch.randn(2, 8, 64)
p("  Gated 版的 forward 形参个数 =", len(inspect.signature(gm.forward).parameters),
  "（hidden_states, gate）→ 分块时要同时切两个张量")
p("  Gated 沿 seq 分块 逐位相同 =",
  bool((gm(xn, gt) == apply_chunking_to_forward(gm.forward, 2, 1, xn, gt)).all().item()))

p(LINE, "\n[14] ALL_LAYERNORM_LAYERS 登记表里有谁、谁在用它")
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
p("  登记表内容 =", [c.__name__ for c in ALL_LAYERNORM_LAYERS])
with torch.device("meta"):
    model2 = M.Glm5NextTextModel(Glm5NextTextConfig())
norm_types = {type(m) for m in model2.modules() if "RMSNorm" in type(m).__name__}
p("  本模型里的 norm 类型 =", sorted(t.__name__ for t in norm_types))
p("  与登记表的交集 =", [t.__name__ for t in norm_types if t in ALL_LAYERNORM_LAYERS])

p(LINE, "\n[15] mHC 的 input_norm 到底在保证什么（尺度不变性）")
cfg3 = Glm5NextTextConfig(hidden_size=512, num_hidden_layers=4)
hc = M.Glm5NextTextHyperConnection(cfg3)
nn.init.normal_(hc.fn, mean=0.0, std=cfg3.initializer_range)
nn.init.zeros_(hc.base); nn.init.ones_(hc.scale)
with torch.no_grad():
    base = torch.randn(2, 6, cfg3.hc_mult, cfg3.hidden_size)
    for scale in (1.0, 100.0, 0.01, 0.001):
        hs = base * scale
        flat = hs.view(2, 6, -1).float()
        normed = hc.input_norm(flat)
        post, comb, collapsed = hc(hs)
        p(f"  输入 x{scale:<7g} 归一化后每个 token 的 RMS = {normed.pow(2).mean(-1).sqrt().mean():.6f}"
          f" | pre/post 的均值 = {post.mean():.6f} | comb 的行和均值 = {comb.sum(-1).mean():.6f}"
          f" | collapsed 的 RMS = {collapsed.pow(2).mean(-1).sqrt().mean():.4f}")
p("  -> 归一化把输入尺度抹掉：x 整体放大 100 倍，三组系数完全不变，只有 collapsed 跟着放大。")
p("  -> input_norm 的输出只喂给一个线性映射 F.linear(flattened, self.fn)，")
p("     而 F(w⊙x) = (F∘diag(w))(x) 仍然是线性映射 —— 这里加 per-channel 权重是数学上多余的。")
