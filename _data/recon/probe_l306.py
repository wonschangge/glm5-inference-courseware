"""L3-06 实测探针：mHC（manifold-constrained Hyper-Connections）的真实行为。

用法（仓库根目录 /data/WORKSPACE/transformer-project）：
    ./.venv/bin/python courseware/_data/recon/probe_l306.py

本脚本是**工装**，不属于覆盖域（覆盖域只统计 src/transformers/**/*.py），
不计入覆盖率。它的作用：把 L3-06 里每一句断言变成一次真实运行 ——
包括「Sinkhorn 20 次迭代把 comb 压成双随机矩阵」与「双随机约束稳住梯度流」。
"""
import os
import warnings

warnings.filterwarnings("ignore")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.nn import init  # noqa: E402

from transformers import Glm5NextTextConfig  # noqa: E402
from transformers.models.glm5_next.modeling_glm5_next import (  # noqa: E402
    Glm5NextTextHyperConnection,
    Glm5NextTextHyperHead,
)


def line(k, v):
    print(f"   {k:<46s}: {v}")


def step(t):
    print("=" * 78)
    print(t)


def make_hc(iters=20, hidden=8, mult=4, seed=0):
    """按 _init_weights 的真实初始化造一个 HyperConnection（源码 L1411-1414）。"""
    cfg = Glm5NextTextConfig(hidden_size=hidden, hc_mult=mult, hc_sinkhorn_iters=iters,
                             num_hidden_layers=2,
                             layer_types=["linear_attention"] * 2,
                             mlp_layer_types=["dense"] * 2)
    m = Glm5NextTextHyperConnection(cfg).eval()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        m.fn.copy_(torch.randn(m.fn.shape, generator=g) * 0.02)
        m.base.zero_()
        m.scale.fill_(1.0)
    return m


# ---------------------------------------------------------------------------
# ① config 里与 mHC 有关的三个数字
# ---------------------------------------------------------------------------
step("① Glm5NextTextConfig 里与 mHC 有关的字段")
c = Glm5NextTextConfig()
line("hc_mult", c.hc_mult)
line("hc_sinkhorn_iters", c.hc_sinkhorn_iters)
line("hc_eps", c.hc_eps)
line("hidden_size", c.hidden_size)
line("rms_norm_eps", c.rms_norm_eps)

# ---------------------------------------------------------------------------
# ② 真模块前向：形状、取值范围、双随机性
# ---------------------------------------------------------------------------
step("② Glm5NextTextHyperConnection 真实前向（hidden_size=8, hc_mult=4）")
hc = make_hc(20)
B, S, N, D = 1, 3, 4, 8
x = torch.randn(B, S, N, D, generator=torch.Generator().manual_seed(1))
# pre 不返回，这里按源码同款公式单独算一遍，用于观察取值范围
with torch.no_grad():
    post, comb, collapsed = hc(x)
    flat = hc.input_norm(x.view(B, S, -1).float())
    allw = torch.nn.functional.linear(flat, hc.fn.float())
    pre_w, post_w, comb_w = allw.split([N, N, N * N], dim=-1)
    pre_b, post_b, comb_b = hc.base.split([N, N, N * N])
    pre_s, post_s, comb_s = hc.scale.unbind(0)
    pre = torch.sigmoid(pre_w * pre_s + pre_b) + hc.hc_eps

line("hidden_streams", tuple(x.shape))
line("post", f"{tuple(post.shape)}  范围 [{post.min():.4f}, {post.max():.4f}]  (0, 2)")
line("comb", f"{tuple(comb.shape)}  范围 [{comb.min():.4f}, {comb.max():.4f}]")
line("collapsed（下一层的输入）", tuple(collapsed.shape))
line("pre（源码内部消费，不返回）", f"{tuple(pre.shape)}  范围 [{pre.min():.4f}, {pre.max():.4f}]")
line("pre 的 4 条流权重和（不归一化！）",
     f"{[round(v, 6) for v in pre[0, 0].tolist()]}  sum={float(pre[0, 0].sum()):.6f}")
line("comb 行和 max|row-1|", f"{(comb.sum(-1) - 1).abs().max():.3e}")
line("comb 列和 max|col-1|", f"{(comb.sum(-2) - 1).abs().max():.3e}")
line("comb 最小元素", f"{comb.min():.3e}  （>0：没有流被饿死）")

# ---------------------------------------------------------------------------
# ③ 迭代次数的实测效应
# ---------------------------------------------------------------------------
step("③ hc_sinkhorn_iters 取不同值时，comb 离双随机有多远（同一种子初始化）")
for it in (1, 2, 3, 5, 20):
    h = make_hc(it)
    with torch.no_grad():
        _, cb, _ = h(x)
    line(f"iters={it:<3d} max|row-1| / max|col-1|",
         f"{(cb.sum(-1) - 1).abs().max():.3e} / {(cb.sum(-2) - 1).abs().max():.3e}")

# ---------------------------------------------------------------------------
# ④ 手算例：3x3 行随机矩阵 + 2 次 Sinkhorn 迭代（源码同款公式）
# ---------------------------------------------------------------------------
step("④ 手算例：3x3，列和为 0.5 / 1.0 / 1.5（便于手算）")
M0 = torch.tensor([[0.30, 0.30, 0.40],
                   [0.15, 0.40, 0.45],
                   [0.05, 0.30, 0.65]], dtype=torch.float64)
line("M0 行和", [f"{v:.6f}" for v in M0.sum(-1).tolist()])
line("M0 列和", [f"{v:.6f}" for v in M0.sum(-2).tolist()])

eps = 0.0     # 手算时忽略 1e-6；下面单独量它带来多少偏差
comb_a = M0 / (M0.sum(dim=-2, keepdim=True) + eps)
print("   [源码 L323] 列归一 -> ")
for r in comb_a.tolist():
    print("        " + "  ".join(f"{v:.6f}" for v in r))
line("  行和", [f"{v:.6f}" for v in comb_a.sum(-1).tolist()])
for k in range(2):
    comb_a = comb_a / (comb_a.sum(dim=-1, keepdim=True) + eps)
    comb_a = comb_a / (comb_a.sum(dim=-2, keepdim=True) + eps)
    print(f"   [L324-326] 迭代 {k + 1} 后 -> ")
    for r in comb_a.tolist():
        print("        " + "  ".join(f"{v:.6f}" for v in r))
    line("  行和", [f"{v:.6f}" for v in comb_a.sum(-1).tolist()])
    line("  列和", [f"{v:.6f}" for v in comb_a.sum(-2).tolist()])

comb20 = M0.clone()
comb20 = comb20 / (comb20.sum(dim=-2, keepdim=True) + eps)
for _ in range(19):
    comb20 = comb20 / (comb20.sum(dim=-1, keepdim=True) + eps)
    comb20 = comb20 / (comb20.sum(dim=-2, keepdim=True) + eps)
print("   同一矩阵跑满 20 次迭代 -> ")
for r in comb20.tolist():
    print("        " + "  ".join(f"{v:.6f}" for v in r))
line("20 次后 max|行和-1| / max|列和-1|",
     f"{(comb20.sum(-1) - 1).abs().max():.3e} / {(comb20.sum(-2) - 1).abs().max():.3e}")

# eps 的真实影响：真模块 forward 输出
with torch.no_grad():
    h20 = make_hc(20)
    _, cbb, _ = h20(x)
line("真模块（eps=1e-6, 20 次）max|行和-1| / max|列和-1|",
     f"{(cbb.sum(-1) - 1).abs().max():.3e} / {(cbb.sum(-2) - 1).abs().max():.3e}")

# ---------------------------------------------------------------------------
# ⑤ 双随机约束稳住什么
# ---------------------------------------------------------------------------
step("⑤ 双随机 vs 行随机：谱范数、退化例、45 层的界")
torch.manual_seed(7)


def sigma_max(a):
    return float(torch.linalg.svdvals(a)[0])


def sinkhorn(m, iters=20):
    m = m / (m.sum(dim=-1, keepdim=True) + 1e-6)
    for _ in range(iters - 1):
        m = m / (m.sum(dim=-1, keepdim=True) + 1e-6)
        m = m / (m.sum(dim=-2, keepdim=True) + 1e-6)
    return m


rows, dbl = [], []
for _ in range(200):
    r = torch.softmax(torch.randn(4, 4, dtype=torch.float64), dim=-1)
    rows.append(r)
    dbl.append(sinkhorn(r))
line("单层 sigma_max 行随机（200 个样本）",
     f"max {max(map(sigma_max, rows)):.4f} / mean {sum(map(sigma_max, rows)) / 200:.4f}")
line("单层 sigma_max 双随机（同 200 个）",
     f"max {max(map(sigma_max, dbl)):.6f} / mean {sum(map(sigma_max, dbl)) / 200:.6f}")

# 退化例：所有行一模一样（行随机允许，双随机不允许）
p = torch.tensor([0.70, 0.20, 0.06, 0.04], dtype=torch.float64)
Mrow = p.repeat(4, 1)
Mdbl = sinkhorn(Mrow)
line("退化例（4 行相同）行和 / 列和",
     f"{float(Mrow.sum(-1)[0]):.4f} / {[round(v, 4) for v in Mrow.sum(-2).tolist()]}")
line("退化例 sigma_max 行随机", f"{sigma_max(Mrow):.4f}   (= ||p||_2 * sqrt(N) = "
     f"{float(p.norm()) * 2:.4f})")
line("同一矩阵 Sinkhorn 20 次后 sigma_max", f"{sigma_max(Mdbl):.6f}")
line("Sinkhorn 后矩阵", [[round(v, 6) for v in r] for r in Mdbl.tolist()])

lam = max(map(sigma_max, rows))
line("最坏单层界 lambda", f"{lam:.4f}   -> 45 层的界 lambda^45 = {lam ** 45:.3e}")
line("双随机 45 层的界", f"1.0000^45 = 1.000000")

# 双随机的不变量：全 1 向量是特征向量，且转置也是双随机
with torch.no_grad():
    hh = make_hc(20)
    _, cb, _ = hh(x)
cb64 = cb.double()
line("comb @ 1 与 1 的最大偏差", f"{(cb64.sum(-1) - 1).abs().max():.3e}")
line("1^T @ comb 与 1 的最大偏差", f"{(cb64.sum(-2) - 1).abs().max():.3e}")
line("sigma_max(comb) / sigma_max(comb^T)",
     f"{sigma_max(cb64[0, 0]):.6f} / {sigma_max(cb64[0, 0].t()):.6f}")

# 45 层连乘
v = torch.randn(4, dtype=torch.float64)


def prod_max(mats):
    P = torch.eye(4, dtype=torch.float64)
    for m in mats:
        P = m.t() @ P
    return sigma_max(P), float((P @ v).norm() / v.norm())


rm, dv = [], []
for _ in range(45):
    r = torch.softmax(torch.randn(4, 4, dtype=torch.float64), dim=-1)
    rm.append(r)
    dv.append(sinkhorn(r))
line("45 层连乘 sigma_max 行随机 / 双随机", f"{prod_max(rm)[0]:.4f} / {prod_max(dv)[0]:.4f}")

# ---------------------------------------------------------------------------
# ⑥ HyperHead
# ---------------------------------------------------------------------------
step("⑥ Glm5NextTextHyperHead：参数量与等价性")
hd = Glm5NextTextHyperHead()
line("参数量", sum(p.numel() for p in hd.parameters()))
hs = torch.randn(2, 5, 4, 8)
with torch.no_grad():
    out = hd(hs)
line("输入 / 输出形状", f"{tuple(hs.shape)} -> {tuple(out.shape)}")
line("等价于 mean(dim=2)", bool(torch.allclose(out, hs.mean(dim=2))))

# ---------------------------------------------------------------------------
# ⑦ 写回公式
# ---------------------------------------------------------------------------
step("⑦ 写回：post ⊙ 子层输出 + combᵀ · 旧流")
resid = torch.randn(B, S, N, D, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(2))
sub = torch.randn(B, S, D, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
post64, comb64 = post.double(), cb64
mixed = post64.unsqueeze(-1) * sub.unsqueeze(-2) + torch.matmul(
    comb64.transpose(-1, -2), resid)
line("混合结果形状", tuple(mixed.shape))
man = torch.stack([post64[0, 0, i] * sub[0, 0]
                   + sum(comb64[0, 0, j, i] * resid[0, 0, j] for j in range(N))
                   for i in range(N)])
line("与逐元素手写公式一致", bool(torch.allclose(mixed[0, 0], man)))
mix_only = torch.matmul(comb64.transpose(-1, -2), resid)
line("||combᵀ·residual|| / ||residual||", f"{float(mix_only.norm() / resid.norm()):.4f}")

# ---------------------------------------------------------------------------
# ⑧ 行数实测
# ---------------------------------------------------------------------------
step("⑧ 覆盖文件实测行数（作业书写 2445）")
p = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                 "..", "upstream-transformers", "src", "transformers",
                                 "models", "glm5_next", "modeling_glm5_next.py"))
txt = open(p, encoding="utf-8").read()
line("换行符个数 / splitlines()", f"{txt.count(chr(10))} / {len(txt.splitlines())}")
line("文件以换行结尾", txt.endswith("\n"))
line("第 2444 行内容", repr(txt.splitlines()[-1]))
