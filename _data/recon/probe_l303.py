#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L3-03 实测脚本（路由器：288 个分数如何变成 8 条路径）。

跑法：  .venv/bin/python _data/recon/probe_l303.py
说明：  这是脚本，不是 src/transformers 下的源文件，**不计入覆盖率**。
        它的输出被逐字抄进 L3-03 的 source.md（用 text 块，不参与保真校验）。
"""
import warnings

warnings.filterwarnings("ignore")

import torch

from transformers import Glm5NextTextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextTopkRouter


def mk(**kw):
    cfg = Glm5NextTextConfig()
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def line(t):
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


# ---------------------------------------------------------------- [1] 默认值
line("[1] Glm5NextTextConfig() 默认值（实测，不是估算）")
cfg = Glm5NextTextConfig()
for k in ("hidden_size", "n_routed_experts", "num_local_experts", "num_experts_per_tok",
          "n_group", "topk_group", "norm_topk_prob", "routed_scaling_factor",
          "n_shared_experts", "moe_intermediate_size"):
    print(f"    {k:24s} = {getattr(cfg, k)}")

# ------------------------------------------------- [2] 默认 n_group=1 的组阶段
line("[2] n_group=1 / topk_group=1 时，两级筛选实际发生了什么")
cfg = mk(n_routed_experts=288, n_group=1, topk_group=1)
r = Glm5NextTextTopkRouter(cfg)
print("    e_score_correction_bias 全零 ? ", bool((r.e_score_correction_bias == 0).all()))

torch.manual_seed(0)
x = torch.randn(1, cfg.hidden_size)
with torch.no_grad():
    r.weight.copy_(torch.randn(288, cfg.hidden_size) * 0.02)
    logits, w, idx = r(x)
sc = logits.sigmoid()
sfc = sc + r.e_score_correction_bias
groups = sfc.view(-1, 1, 288).topk(2, dim=-1)[0].sum(dim=-1)
print("    group_scores.shape        =", tuple(groups.shape), "-> 只有一个组")
gidx = torch.topk(groups, k=1, dim=-1, sorted=False)[1]
mask = torch.zeros_like(groups)
mask.scatter_(1, gidx, 1)
score_mask = mask.unsqueeze(-1).expand(-1, 1, 288).reshape(-1, 288)
print("    score_mask 全 1 ?          ", bool(score_mask.bool().all()))
print("    masked_fill 后还有 -inf ?  ", bool(torch.isinf(sfc.masked_fill(~score_mask.bool(), float('-inf'))).any()))
print("    -> 默认配置下组阶段是 no-op（每个专家都在候选池里）")

# ------------------------------------------------------- [3] 288 -> 8 全过程
line("[3] 288 个分数 -> 8 条路径：一次完整前向的中间量")
print("    top_k =", r.top_k, " num_experts =", r.num_experts, " hidden =", r.hidden_dim)
print("    logits       shape", tuple(logits.shape), " dtype", logits.dtype)
print("    scores       = sigmoid(logits)  ∈ (0,1)")
print("    scores  min/max      = %.6f / %.6f" % (sc.min().item(), sc.max().item()))
print("    scores_for_choice    = scores + bias（bias 全零 -> 与 scores 相同:",
      bool(torch.equal(sfc, sc)), "）")
print("    topk_indices (8 个)  =", idx[0].tolist())
print("    topk_weights         =", [round(v, 6) for v in w[0].tolist()])
print("    ★ sum(topk_weights)  = %.6f   <-- routed_scaling_factor = %s"
      % (w.sum().item(), r.routed_scaling_factor))
print("    ★ 归一化后的 8 个权重之和 = %.6f" % (w[0].sum().item() / r.routed_scaling_factor))
_g = sc[0].gather(0, idx[0])
print("    ★ 权重的三个来源核对：")
print("       gather 自 scores（不是 scores_for_choice） renormalize 后乘因子 :",
      bool(torch.allclose(w[0], _g / (_g.sum() + 1e-20) * r.routed_scaling_factor, atol=1e-6)))
print("       仅 gather 后乘因子（不归一化）                        :",
      bool(torch.allclose(w[0], _g * r.routed_scaling_factor, atol=1e-6)))

# ------------------------------------------- [4] norm_topk_prob 的数值效果
line("[4] norm_topk_prob 的数值效果（同一组 logits）")
for flag in (True, False):
    rr = Glm5NextTextTopkRouter(mk(n_routed_experts=288, norm_topk_prob=flag))
    with torch.no_grad():
        rr.weight.copy_(r.weight)
        _, ww, ii = rr(x)
    print(f"    norm_topk_prob={str(flag):5s}  权重={[round(v,6) for v in ww[0].tolist()]}")
    print(f"                        和={ww.sum().item():.6f}  最大={ww.max().item():.6f}")

# ---------------------------------------------------- [5] 两级筛选真正生效
line("[5] 把 n_group 调到 8、topk_group 调到 2 之后，两级筛选才真的在筛")
cfg8 = mk(n_routed_experts=288, n_group=8, topk_group=2)
r8 = Glm5NextTextTopkRouter(cfg8)
with torch.no_grad():
    r8.weight.copy_(r.weight)
    logits8, w8, idx8 = r8(x)
sfc8 = logits8.sigmoid() + r8.e_score_correction_bias
gs8 = sfc8.view(-1, 8, 36).topk(2, dim=-1)[0].sum(dim=-1)
print("    group_scores shape =", tuple(gs8.shape), "（8 组 × 每组 36 个专家）")
print("    各组 top-2 之和     =", [round(v, 4) for v in gs8[0].tolist()])
gi8 = torch.topk(gs8, k=2, dim=-1, sorted=False)[1]
print("    选中的 2 个组       =", sorted(gi8[0].tolist()))
m8 = torch.zeros_like(gs8)
m8.scatter_(1, gi8, 1)
sm8 = m8.unsqueeze(-1).expand(-1, 8, 36).reshape(-1, 288)
masked = sfc8.masked_fill(~sm8.bool(), float("-inf"))
print("    候选池             : %d -> %d 个（被置 -inf 的 %d 个）"
      % (288, int(sm8.sum().item()), int((~sm8.bool()).sum().item())))
print("    topk_indices       =", idx8[0].tolist())
print("    它们所属的组        =", sorted(set((i // 36) for i in idx8[0].tolist())))
print("    ★ 8 个专家全部落在选中的 2 个组里 —— 这就是「分组」的约束力")

# --------------------------------------------------------- [6] 手算小例子
line("[6] 可手算的最小例子：8 个专家 / 2 组（每组 4 个）/ topk_group=1 / top_k=2")
cfgS = mk(n_routed_experts=8, n_group=2, topk_group=1, num_experts_per_tok=2,
          hidden_size=4, norm_topk_prob=True, routed_scaling_factor=2.5)
rs = Glm5NextTextTopkRouter(cfgS)
with torch.no_grad():
    rs.weight.copy_(torch.tensor([
        [1.0, 0.0, 0.0, 0.0],   # e0
        [2.0, 0.0, 0.0, 0.0],   # e1
        [-1.0, 0.0, 0.0, 0.0],  # e2
        [0.5, 1.0, 0.0, 0.0],   # e3
        [3.0, 0.0, 0.0, 0.0],   # e4
        [2.5, 0.0, 0.0, 0.0],   # e5
        [2.0, 0.0, 0.0, 0.0],   # e6
        [-2.0, 0.0, 0.0, 0.0],  # e7
    ]))
    xs = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    lg, ws, ix = rs(xs)
scs = lg.sigmoid()
sfcs = scs + rs.e_score_correction_bias
gss = sfcs.view(-1, 2, 4).topk(2, dim=-1)[0].sum(dim=-1)
print("    h = [1, 1, 0, 0]")
print("    logits            =", [round(v, 4) for v in lg[0].tolist()])
print("    scores=sigmoid    =", [round(v, 6) for v in scs[0].tolist()])
print("    每组取 top-2 之和  =", [round(v, 6) for v in gss[0].tolist()])
gis = torch.topk(gss, k=1, dim=-1, sorted=False)[1]
print("    选中的组           =", gis[0].tolist(), "（组 0 = e0..e3，组 1 = e4..e7）")
ms = torch.zeros_like(gss)
ms.scatter_(1, gis, 1)
sms = ms.unsqueeze(-1).expand(-1, 2, 4).reshape(-1, 8)
print("    score_mask        =", sms[0].tolist())
print("    masked 后候选      =", [None if v == float('-inf') else round(v, 6)
                                  for v in sfcs.masked_fill(~sms.bool(), float('-inf'))[0].tolist()])
print("    topk_indices      =", ix[0].tolist())
print("    topk_weights(原始) =", [round(v, 6) for v in scs[0].gather(0, ix[0]).tolist()])
print("    归一化(和=1)       =", [round(v, 6) for v in (scs[0].gather(0, ix[0])
                                                    / (scs[0].gather(0, ix[0]).sum() + 1e-20)).tolist()])
print("    乘 2.5             =", [round(v, 6) for v in ws[0].tolist()])
_raw = scs[0].gather(0, ix[0])
print("    手工核对           : 0.952574/(0.952574+0.924142) = %.6f"
      % (0.9525741268 / (0.9525741268 + 0.9241418199)))
print("                         0.924142/(0.952574+0.924142)*2.5 = %.6f"
      % (0.9241418199 / (0.9525741268 + 0.9241418199) * 2.5))
print("                         实跑归一化值 =", [round(v, 6) for v in (_raw / (_raw.sum() + 1e-20)).tolist()])

# --------------------------------------------------- [7] 分布式：通信量line("[7] 分组在分布式（EP）下的意义 —— 可算的部分")
print("    288 个专家按 EP=8 切分 -> 每卡 36 个专家 = 恰好一个组")
print("    1088? 不：每组 36 个专家，正好等于 288/8")
print("    不分组时理论最坏情况：8 个选中专家可能落在 8 张卡上 -> 8 次 all-to-all 目标")
print("    限定 topk_group=2 时：最多落在 2 个组 = 最多 2 张卡上")
print("    -> 通信扇出从 ≤8 降到 ≤2（设计意图；本文件只实现选择逻辑，不实现通信）")

# ------------------------------------------ [8] bias 只改「选谁」，不改「权重」
line("[8] e_score_correction_bias 的效果：只改选择，不改权重")
rb = Glm5NextTextTopkRouter(mk(n_routed_experts=288, norm_topk_prob=True))
with torch.no_grad():
    rb.weight.copy_(r.weight)
    b = torch.zeros(288)
    b[164] = -5.0          # 把 [3] 里排第一的 e164 压下去
    b[7] = +5.0            # 把 e7 抬上来
    rb.e_score_correction_bias.copy_(b)
    lgb, wb, ib = rb(x)
scb = lgb.sigmoid()
sfcb = scb + rb.e_score_correction_bias
print("    bias 非零的专家数        =", int((b != 0).sum().item()))
print("    未加 bias 的 top-8       =", idx[0].tolist())
print("    加 bias 后的 top-8       =", ib[0].tolist())
print("    e164 还在吗              =", 164 in ib[0].tolist(), " e7 进来了吗 =", 7 in ib[0].tolist())
print("    ★ 权重来自 scores 而非 scores_for_choice：")
print("       weights == sigmoid(logits).gather(idx) 归一化 *2.5 ?",
      bool(torch.allclose(wb[0], scb[0].gather(0, ib[0]) / (scb[0].gather(0, ib[0]).sum() + 1e-20) * 2.5, atol=1e-6)))
print("       weights == scores_for_choice.gather(idx) 归一化 *2.5 ?",
      bool(torch.allclose(wb[0], sfcb[0].gather(0, ib[0]) / (sfcb[0].gather(0, ib[0]).sum() + 1e-20) * 2.5, atol=1e-6)))
print("    -> bias 是一个「只作用于选择」的门，不进权重")
