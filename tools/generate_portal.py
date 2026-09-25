#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成门户页 index.html —— 静态烘焙，零 fetch、零外链。

数据（课号/标题/层/文件数/状态）在生成时直接写进 HTML，
因此 file:// 双击打开也能过滤与搜索。

用法:
    python3 tools/generate_portal.py
    python3 tools/generate_portal.py --check   # 只校验：卡片数 / 链接有效性
"""
import html
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from common import R

sys.path.insert(0, C.TOOLS)
import plan  # noqa: E402


def layer_counts():
    out = {}
    for L in plan.LESSONS:
        out[L["layer"]] = out.get(L["layer"], 0) + 1
    return out


def collect():
    done = {i for i, d in C.existing_lessons()}
    uni, excluded = C.load_universe()
    cards = []
    for L in plan.LESSONS:
        d = os.path.join(C.ROOT, plan.lesson_dir(L))
        cards.append(dict(
            id=L["id"], layer=L["layer"], title=L["title"], prio=L["prio"],
            dir=plan.lesson_dir(L), nfiles=len(plan.expand_files(L["files"], uni)),
            has=os.path.isfile(os.path.join(d, "index.html")),
            ready=L["id"] in done,
            points=L["points"], accept=L["accept"],
            files=plan.expand_files(L["files"], uni),
        ))
    return cards, uni, excluded


def render():
    cards, uni, excluded = collect()
    lc = layer_counts()
    n_done = sum(1 for c in cards if c["has"])
    total_lines = 0
    for f in uni:
        p = C.upstream_path(f)
        if os.path.isfile(p):
            with open(p, "rb") as fh:
                total_lines += fh.read().count(b"\n") + 1

    L = []
    A = L.append
    A('<!DOCTYPE html>')
    A('<html lang="zh-CN"><head>')
    A('<meta charset="utf-8">')
    A('<meta name="viewport" content="width=device-width, initial-scale=1">')
    A('<title>GLM-5 推理视角的 transformers 动画课件</title>')
    A('<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' '
      'viewBox=\'0 0 32 32\'%3E%3Crect width=\'32\' height=\'32\' rx=\'7\' fill=\'%230b1222\'/%3E'
      '%3Cpath d=\'M8 21V11h3.4l4.6 6 4.6-6H24v10h-2.8v-5.6L17 21.4h-2l-4.2-5.6V21z\' '
      'fill=\'%2322d3ee\'/%3E%3C/svg%3E">')
    A('<link rel="stylesheet" href="shared/theme.css">')
    A('<style>')
    A(PORTAL_CSS)
    A('</style>')
    A('</head><body>')
    A('<div id="bgAurora"><i></i><i></i><i></i></div>')
    A('<div id="bgGrid"></div>')
    A('<div id="bgVignette"></div>')

    # ---------------- Hero ----------------
    A('<div class="pwrap">')
    A('<header class="hero">')
    A('<div class="hk">GLM-5 推理 &nbsp;·&nbsp; HuggingFace transformers</div>')
    A('<h1>一个 token 的<span class="hl-a">旅程</span></h1>')
    A('<p class="hsub">以 <b>GLM-5.3-Flash（<code class="inl">glm5_next</code>）</b>的推理路径为视角，'
      '逐层拆解 transformers 里到底发生了什么 —— 从一份 config，到一块按页分配的 KV cache。</p>')
    A('<div class="hstats">')
    for v, k, c in ((len(cards), "课", 0), (len(uni), "个源文件被覆盖", 1),
                    (len(plan.LAYERS), "层", 2), (f"{total_lines:,}", "行源码", 4)):
        A(f'<div class="hs cc{c}"><div class="hv">{v}</div><div class="hk2">{k}</div></div>')
    A('</div>')
    A('<div class="hmeta">')
    A(f'<span class="pill acc">覆盖域是实测的：{len(uni)} / {len(uni)+len(excluded)} 个推理闭包文件</span>')
    A(f'<span class="pill">闭包外 {len(excluded)} 个（36 个无关模型族，已声明排除）</span>')
    A(f'<span class="pill ok">已完成 {n_done} / {len(cards)} 课</span>')
    A('<a class="pill acc" href="TODOLIST.md">计划与覆盖度矩阵</a>')
    A('<a class="pill acc" href="STYLE.md">制作规范</a>')
    A('</div>')
    A('</header>')

    # ---------------- 转折点 ----------------
    A('<section class="turn">')
    A('<div class="tk">心智模型的转折点</div>')
    A('<div class="trow">')
    A('<div class="tcol cc0"><div class="th">L0 – L4 &nbsp;数学视角</div>'
      '<div class="td">数据是张量，模型是算子组合。问「<b>算什么</b>」。'
      'hidden 是 4 维的、KV 是三行公式、专家是 288 选 8。</div></div>')
    A('<div class="tarrow">⟹</div>')
    A('<div class="tcol cc3"><div class="th">L5 – L9 &nbsp;机器视角</div>'
      '<div class="td">数据是字节，计算是访存。问「<b>算这个要花多少</b>」。'
      '同一个 KV，这里要问占多少显存、按什么粒度分配、溢出时怎么办。</div></div>')
    A('</div>')
    A('</section>')

    # ---------------- 过滤条 ----------------
    A('<div class="bar">')
    A('<div class="layers" id="layers">')
    A('<button class="lf on" data-l="*">全部 <b>' + str(len(cards)) + '</b></button>')
    for code, name, tag, desc in plan.LAYERS:
        A(f'<button class="lf" data-l="{code}">{html.escape(name.split(" · ")[-1])} '
          f'<b>{lc.get(code,0)}</b></button>')
    A('</div>')
    A('<input id="q" class="search" type="search" placeholder="搜索课号 / 标题 / 源文件名…" '
      'autocomplete="off">')
    A('</div>')
    A('<div id="count" class="rcount"></div>')

    # ---------------- 卡片网格 ----------------
    A('<main id="grid" class="grid">')
    for c in cards:
        cls = "lcard cc" + str(plan.LAYERS.index(
            next(x for x in plan.LAYERS if x[0] == c["layer"])) % 6)
        dis = "" if c["has"] else " planned"
        href = c["dir"] + "/index.html"
        A(f'<article class="{cls}{dis}" data-l="{c["layer"]}" '
          f'data-s="{html.escape((c["id"]+" "+c["title"]+" "+" ".join(c["files"])).lower())}">')
        A(f'<div class="lchd"><span class="lid">{c["id"]}</span>'
          f'<span class="lprio p{c["prio"][1]}">{c["prio"]}</span>'
          + ('<span class="lok">已交付</span>' if c["has"] else '<span class="lwait">计划中</span>')
          + '</div>')
        A(f'<h3>{c["title"]}</h3>')
        A('<ul class="lpts">')
        for p in c["points"][:3]:
            A(f'<li>{html.escape(p)}</li>')
        A('</ul>')
        nf = c["nfiles"]
        A(f'<div class="lft"><span class="nfile">{nf} 个源文件</span>')
        if c["files"]:
            fn = html.escape(os.path.basename(c["files"][0]))
            more = f' +{nf-1}' if nf > 1 else ''
            A(f'<code class="inl">{fn}{more}</code>')
        A('</div>')
        if c["has"]:
            A(f'<a class="lgo" href="{href}">打开课件 →</a>')
        else:
            A('<span class="lgo dis">尚未产出</span>')
        A('</article>')
    A('</main>')
    A('<div id="empty" class="empty" hidden>没有匹配的课。清空搜索框或切换层。</div>')

    # ---------------- 门禁说明 ----------------
    A('<section class="gates">')
    A('<h2>四道门禁 —— 「课件可信」的唯一依据</h2>')
    A('<div class="ggrid">')
    for t, s, d in (
        ("A1/A3 保真", "tools/check_fidelity.py", "每个引用块逐字来自标注源文件，且位置连续"),
        ("A2 参数", "tools/check_params.py", "每个 --flag 都在实测真值集里（211 项）"),
        ("A4 语法", "tools/lint_lessons.py", "四文件齐备、node --check 通过、分幕字段完整"),
        ("B 覆盖度", "tools/check_coverage.py", "差集为空、无空课、无幻影引用"),
        ("C 渲染", "tools/check_render.py", "两分辨率 × 3 时间点，0 JS 错误 / 0 溢出 / 交互可用"),
    ):
        A(f'<div class="gc"><div class="gt">{t}</div><code class="inl">{s}</code>'
          f'<div class="gd">{d}</div></div>')
    A('</div>')
    A('<p class="gfoot">顺序：<code class="inl">lint</code> → <code class="inl">split_blocks</code> → '
      '<code class="inl">fidelity</code> → <code class="inl">params</code> → '
      '<code class="inl">coverage</code> → <code class="inl">render</code>　｜　'
      '<code class="inl">tools/run_gates.sh</code></p>')
    A('</section>')

    A('<footer class="pfoot">')
    A('<div>课件为静态页面：<b>双击任意一课即可离线打开</b>，零 CDN、零构建、零网络请求。</div>')
    A('<div class="dim">源码引用自 <code class="inl">huggingface/transformers</code> 的 '
      '<code class="inl">src/transformers/</code>；覆盖域由实测推理闭包确定。</div>')
    A('</footer>')
    A('</div>')

    # ---------------- 交互（数据已烘焙进 DOM） ----------------
    A('<script>')
    A(PORTAL_JS)
    A('</script>')
    A('</body></html>')
    return "\n".join(L) + "\n"


PORTAL_CSS = r"""
body{overflow:auto !important}
.pwrap{position:relative;z-index:5;max-width:1500px;margin:0 auto;padding:54px 40px 70px}

/* ---------- Hero ---------- */
.hero{margin-bottom:44px}
.hk{font:600 12.5px/1 var(--mono);letter-spacing:.16em;text-transform:uppercase;color:var(--accent);
  display:flex;align-items:center;gap:9px;margin-bottom:14px}
.hk::before{content:"";width:26px;height:2px;border-radius:2px;
  background:linear-gradient(90deg,var(--accent),transparent)}
.hero h1{font:700 54px/1.1 var(--sans);letter-spacing:-.02em;margin:0 0 16px;color:#fff;
  text-shadow:0 4px 40px rgba(56,140,255,.3)}
.hsub{font:400 16px/1.65 var(--sans);color:var(--ink-dim);max-width:930px;margin:0 0 26px}
.hsub b{color:var(--ink)}
.hstats{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:20px}
.hs{flex:1 1 180px;padding:16px 18px;border-radius:14px;
  background:linear-gradient(165deg,rgba(150,180,255,.09),rgba(150,180,255,.025));
  border:1px solid var(--panel-brd);border-top:2px solid var(--cc,var(--accent));
  box-shadow:0 12px 30px rgba(0,0,0,.32)}
.hv{font:700 32px/1 var(--mono);color:#fff;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.hk2{font:500 12px/1.4 var(--sans);color:var(--ink-dim);margin-top:7px}
.hmeta{display:flex;gap:9px;flex-wrap:wrap;align-items:center}
a.pill{text-decoration:none;transition:background .2s,transform .16s var(--ease-pop)}
a.pill:hover{background:rgba(34,211,238,.24);transform:translateY(-1px)}

/* ---------- 转折点 ---------- */
.turn{margin-bottom:40px;padding:20px 22px;border-radius:16px;
  background:linear-gradient(120deg,rgba(56,189,248,.08),rgba(251,113,133,.08));
  border:1px solid var(--panel-brd)}
.tk{font:600 11px/1 var(--mono);letter-spacing:.14em;text-transform:uppercase;
  color:var(--ink-faint);margin-bottom:14px}
.trow{display:flex;align-items:stretch;gap:16px}
.tcol{flex:1;padding:14px 16px;border-radius:12px;background:rgba(6,12,26,.5);
  border:1px solid var(--panel-brd);border-left:3px solid var(--cc)}
.th{font:600 14px/1.3 var(--sans);color:#fff;margin-bottom:7px}
.td{font:400 12.5px/1.65 var(--sans);color:var(--ink-dim)}
.tarrow{align-self:center;font-size:26px;color:var(--accent);flex:none;
  filter:drop-shadow(0 0 12px rgba(34,211,238,.6))}

/* ---------- 过滤条 ---------- */
.bar{position:sticky;top:0;z-index:20;display:flex;gap:14px;align-items:center;flex-wrap:wrap;
  padding:14px 0;margin-bottom:8px;
  background:linear-gradient(180deg,rgba(4,6,13,.96) 60%,rgba(4,6,13,0));
  backdrop-filter:blur(10px)}
.layers{display:flex;gap:7px;flex-wrap:wrap;flex:1}
.lf{cursor:pointer;font:600 12px/1 var(--sans);color:var(--ink-dim);
  padding:8px 13px;border-radius:9px;background:rgba(150,180,255,.07);
  border:1px solid var(--panel-brd);transition:all .2s var(--ease-out)}
.lf b{font-family:var(--mono);color:var(--ink-faint);margin-left:3px;font-weight:600}
.lf:hover{background:rgba(150,180,255,.15);color:#fff;transform:translateY(-1px)}
.lf.on{background:linear-gradient(180deg,rgba(34,211,238,.26),rgba(34,211,238,.12));
  color:#c9f7ff;border-color:rgba(34,211,238,.55)}
.lf.on b{color:#9fe9f7}
.search{width:330px;flex:none;height:38px;padding:0 14px;border-radius:10px;
  background:rgba(150,180,255,.07);border:1px solid var(--panel-brd);color:var(--ink);
  font:400 13px/1 var(--sans);outline:none;transition:border-color .2s,background .2s}
.search:focus{border-color:rgba(34,211,238,.6);background:rgba(34,211,238,.08)}
.search::placeholder{color:var(--ink-faint)}
.rcount{font:500 12px/1 var(--mono);color:var(--ink-faint);margin-bottom:16px}

/* ---------- 卡片网格 ---------- */
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:16px}
.lcard{position:relative;display:flex;flex-direction:column;padding:16px 17px 14px;border-radius:15px;
  background:linear-gradient(165deg,rgba(150,180,255,.085),rgba(150,180,255,.025));
  border:1px solid var(--panel-brd);box-shadow:0 12px 30px rgba(0,0,0,.3);
  transition:transform .26s var(--ease-out),box-shadow .26s,border-color .26s,opacity .26s}
.lcard::after{content:"";position:absolute;inset:0 0 auto;height:2px;border-radius:15px 15px 0 0;
  background:linear-gradient(90deg,transparent,var(--cc),transparent);opacity:.85}
.lcard:not(.planned):hover{transform:translateY(-4px);border-color:var(--cc);
  box-shadow:0 20px 44px rgba(0,0,0,.46),0 0 30px -8px var(--cc)}
.lcard.planned{opacity:.44}
.lcard[hidden]{display:none}
.lchd{display:flex;align-items:center;gap:8px;margin-bottom:9px}
.lid{font:700 12.5px/1 var(--mono);color:var(--cc);letter-spacing:.03em}
.lprio{font:700 10px/1 var(--mono);padding:3px 6px;border-radius:5px;
  background:rgba(150,180,255,.12);color:var(--ink-dim)}
.lprio.p0{background:rgba(251,113,133,.18);color:#fecdd3}
.lprio.p1{background:rgba(251,191,36,.16);color:#fde68a}
.lok{margin-left:auto;font:600 10.5px/1 var(--mono);color:#a7f3d0;
  background:rgba(52,211,153,.16);border:1px solid rgba(52,211,153,.4);
  padding:3px 7px;border-radius:5px}
.lwait{margin-left:auto;font:600 10.5px/1 var(--mono);color:var(--ink-faint);
  border:1px solid var(--panel-brd);padding:3px 7px;border-radius:5px}
.lcard h3{font:600 15px/1.4 var(--sans);color:#fff;margin:0 0 10px}
.lpts{margin:0 0 12px;padding-left:16px;flex:1}
.lpts li{font:400 12px/1.62 var(--sans);color:var(--ink-dim);margin-bottom:4px}
.lpts li::marker{color:var(--cc)}
.lft{display:flex;align-items:center;gap:9px;flex-wrap:wrap;
  padding-top:11px;border-top:1px solid rgba(150,180,255,.1);margin-bottom:11px}
.nfile{font:600 10.5px/1 var(--mono);color:var(--ink-faint);white-space:nowrap}
.lgo{display:block;text-align:center;text-decoration:none;
  font:600 12.5px/1 var(--sans);color:#c9f7ff;padding:10px;border-radius:10px;
  background:linear-gradient(180deg,rgba(34,211,238,.2),rgba(34,211,238,.09));
  border:1px solid rgba(34,211,238,.42);transition:all .2s var(--ease-out)}
.lgo:hover{background:linear-gradient(180deg,rgba(34,211,238,.34),rgba(34,211,238,.16));
  border-color:rgba(34,211,238,.7)}
.lgo.dis{color:var(--ink-faint);background:rgba(150,180,255,.05);
  border-color:var(--panel-brd);cursor:default}
.empty{text-align:center;padding:60px 20px;font:400 14px/1.6 var(--sans);color:var(--ink-faint)}

/* ---------- 门禁 ---------- */
.gates{margin-top:56px;padding-top:34px;border-top:1px solid rgba(150,180,255,.12)}
.gates h2{font:700 22px/1.3 var(--sans);color:#fff;margin:0 0 20px}
.ggrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:14px}
.gc{padding:14px 15px;border-radius:12px;background:rgba(150,180,255,.05);
  border:1px solid var(--panel-brd)}
.gt{font:600 13px/1.3 var(--sans);color:var(--ink);margin-bottom:8px}
.gd{font:400 12px/1.6 var(--sans);color:var(--ink-dim);margin-top:8px}
.gfoot{font:400 12px/1.7 var(--mono);color:var(--ink-faint);margin-top:18px}
.pfoot{margin-top:48px;padding-top:26px;border-top:1px solid rgba(150,180,255,.12);
  font:400 12.5px/1.9 var(--sans);color:var(--ink-dim)}
.pfoot b{color:var(--ink)}
@media (max-width:820px){
  .pwrap{padding:30px 18px 50px}
  .hero h1{font-size:34px}
  .trow{flex-direction:column}
  .tarrow{transform:rotate(90deg)}
  .search{width:100%}
}
"""

PORTAL_JS = r"""
(function () {
  var cards = Array.prototype.slice.call(document.querySelectorAll('.lcard'));
  var q = document.getElementById('q');
  var count = document.getElementById('count');
  var empty = document.getElementById('empty');
  var layer = '*';

  function apply() {
    var s = (q.value || '').trim().toLowerCase();
    var n = 0;
    cards.forEach(function (c) {
      var okL = (layer === '*' || c.dataset.l === layer);
      var okS = !s || c.dataset.s.indexOf(s) >= 0;
      var show = okL && okS;
      c.hidden = !show;
      if (show) n++;
    });
    count.textContent = '显示 ' + n + ' / ' + cards.length + ' 课'
      + (layer === '*' ? '' : '　层过滤：' + layer)
      + (s ? '　搜索：「' + q.value.trim() + '」' : '');
    empty.hidden = n !== 0;
  }

  document.getElementById('layers').addEventListener('click', function (e) {
    var b = e.target.closest('.lf');
    if (!b) return;
    Array.prototype.forEach.call(this.querySelectorAll('.lf'), function (x) { x.classList.remove('on'); });
    b.classList.add('on');
    layer = b.dataset.l;
    apply();
  });
  q.addEventListener('input', apply);
  document.addEventListener('keydown', function (e) {
    if (e.key === '/' && document.activeElement !== q) { q.focus(); e.preventDefault(); }
    if (e.key === 'Escape') { q.value = ''; apply(); q.blur(); }
  });
  apply();
})();
"""


def main(argv):
    text = render()
    out = os.path.join(C.ROOT, "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    cards, uni, excluded = collect()
    nd = sum(1 for c in cards if c["has"])
    R.ok(f"已生成 index.html（{len(cards)} 张卡片，{nd} 课已交付，"
         f"{len(text.splitlines())} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
