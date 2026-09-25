#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L3-04 实测脚本（专家层：打包存储、swiglu_limit 截断、grouped GEMM 的布局要求）。

跑法：  .venv/bin/python _data/recon/probe_l304.py
说明：  这是脚本，不是 src/transformers 下的源文件，**不计入覆盖率**。
        它的输出被逐字抄进 L3-04 的 source.md（用 text 块，不参与保真校验）。
"""
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F

from transformers import Glm5NextTextConfig
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextTextExperts,
    Glm5NextTextMLP,
    Glm5NextTextMoE,
)


def line(t):
    print("\n" + "=" * 76)
    print(t)
    print("=" * 76)


# ------------------------------------------------------------ [1] 默认值
line("[1] Glm5NextTextConfig() 默认值（实测）")
cfg = Glm5NextTextConfig()
for k in ("hidden_size", "moe_intermediate_size", "n_routed_experts", "num_local_experts",
          "num_experts_per_tok", "n_shared_experts", "swiglu_limit", "hidden_act",
          "routed_scaling_factor", "num_hidden_layers"):
    print(f"    {k:22s} = {getattr(cfg, k)}")

# ------------------------------------------- [2] 真实 config 下的参数形状/规模
line("[2] 真实 config 下的两个 Parameter（meta 设备，不占内存）")
with torch.device("meta"):
    ex = Glm5NextTextExperts(cfg)
print("    gate_up_proj.shape =", tuple(ex.gate_up_proj.shape))
print("    down_proj.shape    =", tuple(ex.down_proj.shape))
n_experts = ex.num_experts
gu = 2 * ex.intermediate_dim * ex.hidden_dim
dp = ex.hidden_dim * ex.intermediate_dim
per_layer = n_experts * (gu + dp)
print(f"    单个专家: gate_up {gu:,} + down {dp:,} = {gu + dp:,}")
print(f"    一层 {n_experts} 个专家 = {per_layer:,}  ({per_layer/1e9:.3f} B)")
print(f"    42 个 MoE 层       = {per_layer * 42:,}  ({(per_layer*42)/1e9:.2f} B)")
print(f"    bf16 权重显存      = {per_layer * 42 * 2 / 2**30:.1f} GiB")

# 对照：稠密 MLP（intermediate_size=12288）
with torch.device("meta"):
    mlp = Glm5NextTextMLP(cfg)
dense = sum(p.numel() for p in mlp.parameters())
print(f"    对照 · 稠密 MLP(12288) 一层 = {dense:,}  ({dense/1e6:.1f} M)")
with torch.device("meta"):
    shared = Glm5NextTextMLP(cfg, intermediate_size=cfg.moe_intermediate_size * cfg.n_shared_experts)
n_shared = sum(p.numel() for p in shared.parameters())
print(f"    对照 · 共享专家(2048)    一层 = {n_shared:,}  ({n_shared/1e6:.1f} M)")
print(f"    共享专家 == 1 个路由专家（{gu + dp:,}）? {n_shared == gu + dp}")

# ----------------------------------------------- [3] 打包布局 vs 分开两次 GEMM
line("[3] gate_up_proj 打包：一次 GEMM 顶两次，且 chunk 是视图")
torch.manual_seed(0)
E, I, H = 4, 6, 5
Wg = torch.randn(E, I, H)
Wu = torch.randn(E, I, H)
Wp = torch.cat([Wg, Wu], dim=1)              # 打包：(E, 2I, H)
x = torch.randn(3, H)
packed = F.linear(x, Wp[0])                  # 一次 GEMM
g1, u1 = packed.chunk(2, dim=-1)
g2, u2 = F.linear(x, Wg[0]), F.linear(x, Wu[0])
print("    packed.shape                     =", tuple(packed.shape))
print("    一次 GEMM 与两次 GEMM 数值一致 ?   ",
      bool(torch.allclose(g1, g2) and torch.allclose(u1, u2)))
print("    chunk 后 gate 与 packed 同存储 ?   ",
      g1.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr())
print("    gate.stride() =", g1.stride(), " ours =", packed.stride())
Gu = torch.cat([Wg[0], Wu[0]], dim=0)
print("    行拼接口径 chunk(2, dim=0) 也可复原 ? ",
      bool(torch.allclose(Gu[:I], Wg[0]) and torch.allclose(Gu[I:], Wu[0])))

# ------------------------------------------------ [4] swiglu_limit 手算例子
line("[4] _apply_gate 的两次截断（swiglu_limit = 10.0）")
lim = cfg.swiglu_limit
for g, u in ((0.5, 0.5), (10.0, 10.0), (40.0, 40.0), (-40.0, -40.0), (2.0, -13.0)):
    gc = min(g, lim)                       # gate：只截上界
    uc = max(-lim, min(u, lim))            # up：上下都截
    print(f"    gate={g:7.1f} up={u:7.1f} -> gate'={gc:6.1f} up'={uc:6.1f} "
          f"slu={F.silu(torch.tensor(gc)).item():9.5f}  out={F.silu(torch.tensor(gc)).item()*uc:11.3f}")
print(f"    silu({lim}) = {F.silu(torch.tensor(lim)).item():.6f}   "
      f"上界乘出来最大 {F.silu(torch.tensor(lim)).item()*lim:.3f}")
print("    不做截断时 gate=up=40 的输出 =", F.silu(torch.tensor(40.0)).item() * 40.0)
print("    silu(-40) =", F.silu(torch.tensor(-40.0)).item(), "（≈ 0，所以 gate 的负方向不需要截断）")
xs_ = torch.linspace(-30, 30, 200001)
mn = F.silu(xs_)
print(f"    silu 的全局下界 = {mn.min().item():.6f}（在 x={xs_[mn.argmin()].item():.3f} 处）"
      f" → gate 那一侧不需要下截断")

line("[4b] _apply_gate 与手写 swiglu 完全一致")
with torch.device("meta"):
    ex_s = Glm5NextTextExperts(Glm5NextTextConfig(
        hidden_size=8, moe_intermediate_size=4, n_routed_experts=2, swiglu_limit=lim))
ex_s = Glm5NextTextExperts(Glm5NextTextConfig(
    hidden_size=8, moe_intermediate_size=4, n_routed_experts=2, swiglu_limit=lim))
torch.manual_seed(1)
with torch.no_grad():
    for p in ex_s.parameters():
        p.normal_(0, 0.5)
gu_in = torch.randn(7, 8) * 5
out = ex_s._apply_gate(gu_in)
g, u = gu_in.chunk(2, dim=-1)
manual = F.silu(g.clamp(max=lim)) * u.clamp(min=-lim, max=lim)
print("    allclose ?", bool(torch.allclose(out, manual)))

# ------------------------------------- [5] 专家 forward（eager 逐专家 + index_add_）
line("[5] eager 专家前向：one_hot mask → 逐专家 → index_add_")
with torch.device("meta"):
    ex_t = Glm5NextTextExperts(Glm5NextTextConfig(
        hidden_size=8, moe_intermediate_size=4, n_routed_experts=6, swiglu_limit=lim))
ex_t = Glm5NextTextExperts(Glm5NextTextConfig(
    hidden_size=8, moe_intermediate_size=4, n_routed_experts=6, swiglu_limit=lim))
torch.manual_seed(2)
with torch.no_grad():
    for p in ex_t.parameters():
        p.normal_(0, 0.3)
h = torch.randn(5, 8)
idx = torch.tensor([[0, 3], [1, 3], [5, 0], [2, 2], [4, 1]])
wts = torch.rand(5, 2)
out = ex_t(h, idx, wts)
mask = F.one_hot(idx, num_classes=6 + 1).permute(2, 1, 0)
hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
print("    hit 专家数 =", len(hit), "（含哨兵列则多 1）")
print("    num_experts+1 =", ex_t.num_experts + 1, "→ 最后一行是哨兵，被 continue 跳过")
ref = torch.zeros_like(h)
for e in hit:
    e = e[0]
    if e == ex_t.num_experts:
        continue
    pos, tok = torch.where(mask[e])
    cur = ex_t._apply_gate(F.linear(h[tok], ex_t.gate_up_proj[e]))
    cur = F.linear(cur, ex_t.down_proj[e]) * wts[tok, pos, None]
    ref.index_add_(0, tok, cur)
print("    与逐专家手算一致 ?", bool(torch.allclose(out, ref)))

# 哨兵槽位：EP 下 ep_router 会把「不属于本卡」的槽位置成 num_local_experts
sent_idx = torch.tensor([[0], [6]])            # 6 == num_experts → 哨兵
sent_out = ex_t(h[:2], sent_idx, torch.ones(2, 1))
print("    哨兵槽位(索引=num_experts)的输出全 0 ?", bool((sent_out[1] == 0).all()),
      " 第 0 个 token 非 0 ?", bool((sent_out[0] != 0).any()))

# --------------------------------------------- [6] grouped GEMM 的布局要求
line("[6] grouped GEMM：权重必须是 (E, out, in)，分组靠 offsets")
print("    torch.nn.functional.grouped_mm 存在 ?", hasattr(torch.nn.functional, "grouped_mm"))
print("    torch._grouped_mm 存在 ?            ", hasattr(torch, "_grouped_mm"))
from transformers.integrations.moe import _grouped_linear, _can_use_grouped_mm

E, I, H = 3, 4, 8
W = torch.randn(E, 2 * I, H)                       # (E, out, in) —— 就是 gate_up_proj 的排布
xs = torch.randn(5, H)                             # 5 个 (token, expert) 对，已按专家排好序
counts = [2, 0, 3]                                 # 专家 0 吃 2 行，专家 1 空，专家 2 吃 3 行
offs = torch.tensor(counts, dtype=torch.int32).cumsum(0, dtype=torch.int32)
print("    权重布局 =", tuple(W.shape), " offsets =", offs.tolist(), offs.dtype)
print("    W.transpose(-2,-1) 是零拷贝吗 ?     ",
      W.transpose(-2, -1).untyped_storage().data_ptr() == W.untyped_storage().data_ptr())
print("    _can_use_grouped_mm(...) 本机实测    =", _can_use_grouped_mm(xs, W.transpose(-2, -1), offs))
out = _grouped_linear(xs, W, offs, is_transposed=False)
ref = torch.cat([F.linear(xs[:2], W[0]), F.linear(xs[2:5], W[2])], dim=0)
print("    grouped 结果 == 逐专家 mm 拼接 ?    ", bool(torch.allclose(out, ref)))

line("[6b] grouped GEMM 的硬性布局要求（实测报错原文）")
for tag, fn in (
    ("专家轴不在第 0 维 (out,in,E)", lambda: torch.nn.functional.grouped_mm(
        xs, W.permute(1, 2, 0).transpose(-2, -1), offs=offs)),
    ("offsets 长度 ≠ 专家数", lambda: torch.nn.functional.grouped_mm(
        xs, W.transpose(-2, -1), offs=torch.tensor([5], dtype=torch.int32))),
    ("offsets 不是 int32", lambda: torch.nn.functional.grouped_mm(
        xs, W.transpose(-2, -1), offs=offs.long())),
):
    try:
        fn()
        print(f"    {tag:28s} -> 没报错")
    except Exception as e:
        print(f"    {tag:28s} -> {type(e).__name__}: {str(e)[:60]}")

# ------------------------------------------------- [7] EP 切分与权重的第 0 维
line("[7] EP 切分：Shard(0) 切的正是「专家」这一维")
print("    config.base_model_ep_plan（实测值）:")
for k, v in Glm5NextTextConfig().base_model_ep_plan.items():
    print(f"        {k:42s} -> {v}")
try:
    from transformers.distributed.tensor_parallel import ParallelInterface, MoEParamShard
    st = ParallelInterface._global_mapping.get("grouped_gemm")
    print("    ParallelInterface['grouped_gemm'] =", type(st).__name__,
          "placement =", st.placement, " shards_expert_dim =", st.shards_expert_dim)
    print("    → 每卡专家数 = n_routed_experts / ep_size（要整除，否则抛 ValueError）")
    for ep in (1, 8, 16, 32):
        print(f"        ep_size={ep:3d} → 每卡 {288 // ep:3d} 个专家，"
              f"gate_up_proj 每卡 {288 // ep * 2 * 2048 * 4096 * 2 / 2**30:.2f} GiB (bf16)")
except Exception as e:                                       # pragma: no cover
    print("    (跳过 EP 样式检查:", type(e).__name__, str(e)[:60], ")")

# ------------------------------------------- [8] MoE：路由专家 + 共享专家
line("[8] Glm5NextTextMoE.forward：路由输出与共享专家逐 token 相加")
with torch.device("meta"):
    moe = Glm5NextTextMoE(Glm5NextTextConfig(
        hidden_size=8, moe_intermediate_size=4, n_routed_experts=6,
        n_shared_experts=1, num_experts_per_tok=2, swiglu_limit=lim))
moe = Glm5NextTextMoE(Glm5NextTextConfig(
    hidden_size=8, moe_intermediate_size=4, n_routed_experts=6,
    n_shared_experts=1, num_experts_per_tok=2, swiglu_limit=lim))
torch.manual_seed(3)
with torch.no_grad():
    for p in moe.parameters():
        p.normal_(0, 0.2)
h = torch.randn(2, 4, 8)
out = moe(h)
_, tw, ti = moe.gate(h)
routed = moe.experts(h.view(-1, 8), ti, tw).view(2, 4, 8)
shared = moe.shared_experts(h)
print("    out.shape =", tuple(out.shape))
print("    out == routed + shared ?", bool(torch.allclose(out, routed + shared)))
print("    shared_experts 的 intermediate =", moe.shared_experts.intermediate_size,
      "= moe_intermediate_size * n_shared_experts")
print("    shared 对每个 token 都算（无路由），routed 只算 top-2")

# ------------------------------------- [9] 打包到底省下了什么（数 GEMM 调用）
line("[9] 打包 vs 不打包：一层前向里的 GEMM 调用次数（命中 5 个专家）")
_orig_linear = F.linear
calls = {"n": 0}


def counting_linear(*a, **k):
    calls["n"] += 1
    return _orig_linear(*a, **k)


E9 = 8
torch.manual_seed(4)
Wg9 = torch.randn(E9, 4, 8)
Wu9 = torch.randn(E9, 4, 8)
Wd9 = torch.randn(E9, 8, 4)
h9 = torch.randn(6, 8)
idx9 = torch.tensor([[0], [1], [2], [3], [4], [0]])   # 6 个 token 各选 1 个专家，命中 5 个
w9 = torch.ones(6, 1)

F.linear = counting_linear
try:
    calls["n"] = 0
    packed_w = torch.cat([Wg9, Wu9], dim=1)            # (E, 2I, H)
    for e in (0, 1, 2, 3, 4):
        t = torch.where(idx9[:, 0] == e)[0]
        gu = F.linear(h9[t], packed_w[e])              # 一次，两半都在里面
        g9, u9 = gu.chunk(2, dim=-1)
        cur = F.silu(g9.clamp(max=lim)) * u9.clamp(min=-lim, max=lim)
        F.linear(cur, Wd9[e])
    n_packed = calls["n"]

    calls["n"] = 0
    for e in (0, 1, 2, 3, 4):                          # 不打包：gate / up 各一次
        t = torch.where(idx9[:, 0] == e)[0]
        cur = F.silu(F.linear(h9[t], Wg9[e]).clamp(max=lim)) * F.linear(h9[t], Wu9[e]).clamp(min=-lim, max=lim)
        F.linear(cur, Wd9[e])
    n_split = calls["n"]
finally:
    F.linear = _orig_linear
print(f"    命中 5 个专家：打包 {n_packed} 次 GEMM，不打包 {n_split} 次（每个专家 2 vs 3）")
print(f"    一层 288 个专家全命中：调用次数 {288*2} vs {288*3}，差 {288} 次 kernel 启动")
print("    注意：FLOPs 完全一样 —— 省的是调用次数与权重张量个数，不是算力")
