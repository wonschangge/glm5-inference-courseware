/* ==========================================================================
   L3-06 · ★ mHC：流形约束超连接
   --------------------------------------------------------------------------
   覆盖：models/glm5_next/modeling_glm5_next.py（1 个文件 / 2444 行）
   目标：看完能手算 2 次 Sinkhorn 迭代，并说清双随机约束为什么能稳住梯度流。
   排版基准：#visual 可视区 = 994 x 662.7 舞台像素（无头浏览器实测）。
   实测来源：_data/recon/probe_l306.py（工装脚本，不计入覆盖率）。
   ========================================================================== */
'use strict';

/* 实测常量（probe_l306.py 跑出来的，不是估算） */
const HC = {
  N: 4,               // hc_mult：残差流条数
  iters: 20,          // hc_sinkhorn_iters
  eps: 1e-6,          // hc_eps
  D: 4096,            // hidden_size
  wsize: 24,          // (2 + N) * N = pre(4) + post(4) + comb(16)
  fnParams: 393216,   // fn (24, 16384)
  baseParams: 24,
  scaleParams: 3,
  perHC: 393243,
  perLayer: 786486,
  allLayers: 35391870,
};

/* --------------------------------------------------------------------------
   纯函数小工具（build 幂等：只用参数，不碰外部状态）
   -------------------------------------------------------------------------- */
function gridOf(rows, cols, val, o) {
  o = o || {};
  const cs = o.cell || 30, gap = o.gap == null ? 2 : o.gap;
  const rgb = o.rgb || '56,189,248';
  const rgbAt = (r, c) => (typeof rgb === 'function' ? rgb(r, c) : rgb);
  const g = U.el('div', { class: 'tgrid' });
  g.style.gridTemplateColumns = 'repeat(' + cols + ',' + cs + 'px)';
  g.style.gridTemplateRows = 'repeat(' + rows + ',' + cs + 'px)';
  g.style.gap = gap + 'px';
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = val(r, c), rg = rgbAt(r, c);
      const e = U.el('div', { class: 'tcell' });
      e.style.fontSize = (o.fs || 9.5) + 'px';
      e.style.background = 'rgba(' + rg + ',' + (0.05 + 0.70 * v).toFixed(3) + ')';
      e.style.borderColor = 'rgba(' + rg + ',' + (0.16 + 0.50 * v).toFixed(3) + ')';
      e.style.color = v > 0.52 ? '#04121c' : '#dce9ff';
      e.style.transition = 'background .45s cubic-bezier(.16,1,.3,1),'
                         + 'transform .45s cubic-bezier(.16,1,.3,1),opacity .45s';
      e.textContent = o.text ? o.text(r, c) : v.toFixed(2);
      g.appendChild(e);
    }
  }
  g.cellList = Array.prototype.slice.call(g.children);
  return g;
}

/* 一个「带标题 + 行和/列和注脚」的小矩阵块 */
function mtxBlock(label, rows, cols, val, o) {
  o = o || {};
  const col = U.el('div', { class: 'col gap4 center' });
  col.appendChild(U.el('div', {
    class: 'klabel', style: 'text-align:center;margin-bottom:0;white-space:nowrap', text: label,
  }));
  const g = gridOf(rows, cols, val, o);
  col.appendChild(g);
  if (o.note) {
    col.appendChild(U.el('div', {
      class: 'mono',
      style: 'font-size:9.5px;line-height:1.45;text-align:center;white-space:pre;color:'
           + (o.noteColor || 'var(--ink-faint)'),
      text: o.note,
    }));
  }
  col._g = g;
  return col;
}

/* 一条「流」的横条 */
function streamBar(label, cc, w, h, txt) {
  const e = U.el('div', {
    style: 'width:' + w + 'px;height:' + (h || 26) + 'px;border-radius:8px;flex:none;'
         + 'border:1px solid var(--c' + cc + ');background:rgba(150,180,255,.10);'
         + 'display:flex;align-items:center;gap:8px;padding:0 10px;'
         + 'font:600 10.5px var(--mono);color:var(--ink-dim);'
         + 'transition:opacity .45s cubic-bezier(.16,1,.3,1),transform .45s cubic-bezier(.16,1,.3,1)',
    html: '<span style="color:var(--c' + cc + ')">' + label + '</span>'
        + (txt ? '<span style="margin-left:auto;color:var(--ink-faint)">' + txt + '</span>' : ''),
  });
  return e;
}

/* 等宽「小卡」：数字 + 标签 */
function statCard(v, k, cc, sub) {
  return W.card({
    cc: cc, tint: cc,
    body: U.el('div', { class: 'col gap4' }, [
      U.el('div', { class: 'n big', text: v }),
      U.el('div', { class: 'mono', style: 'font-size:10px;color:var(--ink-dim)', text: k }),
      sub ? U.el('div', { class: 'mono', style: 'font-size:9.5px;color:var(--ink-faint)', text: sub }) : null,
    ]),
    style: 'flex:1;padding:9px 12px',
  });
}

const SCENES = [

/* ------------------------------------------------- 1 全景：三步 */
{
  kicker: '第 3 层 · 文本主干 · 算子拆解',
  title: '★ mHC：残差从 <span class="hl-a">1 条</span>变成 <span class="hl-c">4 条</span>',
  sub: 'manifold-constrained Hyper-Connections 把教科书里的 x + f(norm(x)) 拆成三步：读取 → 收拢 → 写回。',
  caption: '回顾 L0-01：hidden_states 从一开始就是 (B, S, 4, 4096)；本课拆开多出来的那个 4 到底怎么用。',
  lang: 'python',
  codeStart: 1314,
  code: `        residual = hidden_states
        post, comb, hidden_states = self.attn_hc(hidden_states)
        # Self attn
        hidden_states = self.input_layernorm(hidden_states)`,
  codeNote: '解码层 forward 的头 4 行 —— 旧流先留一份，再由 attn_hc 产出 post / comb / 收拢后的流。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: 'mHC 的三步（每一步都在解码层里发生，每层做两次）' }));
    const fl = W.flow([
      { t: 'hidden_streams', s: '(B, S, 4, 4096)', cc: 0 },
      { t: '① 读取', s: '一次投影 → 24 个数', cc: 1 },
      { t: '② 收拢', s: 'pre 门控 → (B, S, 4096)', cc: 2 },
      { t: '③ 写回', s: 'post ⊙ out + combᵀ · 旧流', cc: 4 },
      { t: '下一层', s: '(B, S, 4, 4096)', cc: 5 },
    ], { style: 'width:100%' });
    viz.appendChild(fl);

    const steps = [
      { cc: 1, t: '① 读取', s: '4 条流摊平 → <b>UnweightedRMSNorm</b> → 一次线性投影',
        b: 'fn (24 × 16384)：<b>pre</b>(4) + <b>post</b>(4) + <b>comb</b>(4×4)' },
      { cc: 2, t: '② 收拢', s: 'pre 把 4 条流压成 1 条，喂给 ATTN 或 MLP —— 子层完全不知道有 4 条流',
        b: '(pre.unsqueeze(-1) * streams).sum(dim=2)' },
      { cc: 4, t: '③ 写回', s: '4 条流各收回一份子层输出，再按 comb 互相混合',
        b: 'post ⊙ out + combᵀ · 旧流' },
    ];
    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cards = steps.map(st => {
      const c = W.card({
        cc: st.cc, tint: st.cc, title: st.t, sub: st.s,
        body: U.el('div', { class: 'mono', style: 'font-size:10.5px;color:var(--c' + st.cc + ');margin-top:5px', html: st.b }),
        style: 'flex:1 1 0',
      });
      row.appendChild(c); return c;
    });
    viz.appendChild(row);

    const stats = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const stEls = [
      statCard(String(HC.N), 'hc_mult · 残差流条数', 0),
      statCard(String(HC.wsize), '(2 + N) × N · 投影输出', 1),
      statCard(String(HC.iters), 'hc_sinkhorn_iters', 2),
      statCard('1e-6', 'hc_eps', 3),
      statCard(U.fmt(HC.allLayers), '45 层的 mHC 参数', 4, '每层 2 个 × 393,243'),
    ].map(e => { stats.appendChild(e); return e; });
    viz.appendChild(stats);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    fl.focus(0);
    msg.innerHTML = '<span class="cm">// 进门时手上是 4 条流，不是 1 条</span>';
    tl.at(1800, () => { fl.focus(1); cards[0].classList.add('ac');
      msg.innerHTML = '① <em>读取</em>：24 个数里，<span class="hl2">pre</span> 管收拢、<span class="hl4">post</span> 管写回、<span class="hl">comb</span> 管混合'; });
    tl.at(5200, () => { fl.focus(2); cards[0].classList.remove('ac'); cards[1].classList.add('ac');
      msg.innerHTML = '② <em>收拢</em>：<span class="hl3">pre 没有归一化</span> —— 实测权重和 2.01，不是 1'; });
    tl.at(8500, () => { fl.focus(3); cards[1].classList.remove('ac'); cards[2].classList.add('ac');
      msg.innerHTML = '③ <em>写回</em>：<span class="hl4">post</span> 中心取 1（初始退化成标准残差），<span class="hl">comb</span> 必须是双随机'; });
    tl.at(11800, () => {
      fl.focus(4); cards[2].classList.remove('ac');
      stEls.forEach((e, i) => { e.classList.toggle('ac', i >= 2); });
      msg.innerHTML = '<span class="cm">// 记住这三个数：N=4 / 24 个数 / 20 次 Sinkhorn 迭代</span>';
    });
    tl.at(15200, () => {
      fl.focus(-1); stEls.forEach(e => e.classList.remove('ac'));
      cards.forEach(c => c.classList.add('ac'));
      msg.innerHTML = 'mHC 不是「加一条残差」，而是<b>把残差变成 4 条流 + 一张混合矩阵</b>';
    });
  },
},

/* ------------------------------------------------- 2 读取 */
{
  kicker: '第 1 步 · 读取',
  title: '读取：<span class="hl-a">摊平 → 无权重范数 → 一次投影</span>',
  sub: '整台机器只有一次矩阵乘法：把 4 条流首尾相接成 16384 维，投影出 24 个数。',
  caption: '注意 view(B,S,-1) 是首尾相接而不是相加 —— 投影要看得见「哪条流在哪个位置」。',
  lang: 'python',
  codeStart: 298,
  code: `        batch_size, seq_len = hidden_streams.shape[:2]
        hc = self.hc_mult

        # Flatten and norm the hidden streams
        flattened = hidden_streams.view(batch_size, seq_len, -1).float()
        flattened = self.input_norm(flattened)
        # Mix the streams together to infer the weight coefficients
        flattened = F.linear(flattened, self.fn.float())
        # Split the weight coefficients
        pre_w, post_w, comb_w = flattened.split([hc, hc, hc * hc], dim=-1)`,
  codeNote: 'Glm5NextTextHyperConnection.forward 的前 10 行：摊平 → fp32 → 无权重 RMSNorm → fn → 切三段。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const stage = U.el('div', { class: 'row gap14 center', style: 'width:100%' });
    viz.appendChild(stage);

    /* 左：4 条流 */
    const left = U.el('div', { class: 'col gap5 center' });
    left.appendChild(U.el('div', { class: 'klabel', style: 'white-space:nowrap', text: 'hidden_streams (B, S, 4, 4096)' }));
    const bars = [];
    for (let s = 0; s < HC.N; s++) { const b = streamBar('流 ' + s, s, 210, 24); left.appendChild(b); bars.push(b); }
    stage.appendChild(left);

    const arrow = (top, bottom) => U.el('div', { class: 'col center gap4', style: 'flex:none;width:150px' }, [
      U.el('div', { class: 'mono', style: 'font-size:10.5px;color:var(--accent);text-align:center', html: top }),
      U.el('div', { style: 'font-size:20px;color:var(--accent);line-height:1', text: '⟹' }),
      U.el('div', { class: 'mono', style: 'font-size:9.5px;color:var(--ink-faint);text-align:center', html: bottom }),
    ]);

    stage.appendChild(arrow('view(b, s, -1)<br>.float()', '(B, S, <b>16384</b>)'));

    /* 中：无权重范数 + 投影 */
    const mid = U.el('div', { class: 'col gap6', style: 'flex:none;width:246px' });
    const c1 = W.card({ cc: 2, tint: 2, title: 'UnweightedRMSNorm',
      sub: '没有 weight 的 RMSNorm：只做尺度整形，不做特征缩放', style: 'width:100%' });
    const c2 = W.card({ cc: 1, tint: 1, title: 'fn (24 × 16384)',
      sub: '整台机器唯一的一次矩阵乘法；fp32 计算', style: 'width:100%' });
    mid.appendChild(c1); mid.appendChild(c2);
    stage.appendChild(mid);

    stage.appendChild(arrow('F.linear<br>split([4, 4, 16])', '24 个数'));

    /* 右：24 个数 */
    const right = U.el('div', { class: 'col gap6' });
    right.appendChild(U.el('div', { class: 'klabel', text: '投影输出：24 = pre(4) + post(4) + comb(4×4)' }));
    const g = gridOf(4, 6, (r, c) => (c < 1 ? 0.30 : (c < 2 ? 0.62 : 0.86)), {
      cell: 30, fs: 8.5,
      rgb: (r, c) => (c === 0 ? '251,191,36' : (c === 1 ? '52,211,153' : '167,139,250')),
      text: (r, c) => (c === 0 ? 'pre' + r : (c === 1 ? 'post' + r : 'c' + r + (c - 2))),
    });
    right.appendChild(g);
    const legend = U.el('div', { class: 'row gap8', style: 'width:100%' });
    legend.innerHTML = U.chip('pre · 4', 2) + U.chip('post · 4', 4) + U.chip('comb · 16', 1);
    right.appendChild(legend);
    stage.appendChild(right);

    const stats = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const sf = statCard(U.fmt(HC.fnParams), 'fn 参数量', 1, '(24, 16384)');
    const sl = statCard(U.fmt(HC.perLayer), '每层两个 HC', 4, '393,243 × 2');
    const sa = statCard(U.fmt(HC.allLayers), '45 层合计', 0, '45 × 786,486');
    [sf, sl, sa].forEach(e => stats.appendChild(e));
    viz.appendChild(stats);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    bars.forEach(b => { b.style.opacity = '.25'; b.style.transform = 'translateX(-14px)'; });
    c1.style.opacity = '.3'; c2.style.opacity = '.3';
    msg.innerHTML = '<span class="cm">// 4 条流首尾相接，不是相加</span>';

    tl.at(2000, () => {
      bars.forEach((b, k) => { b.style.opacity = '1'; b.style.transform = 'none'; b.style.transitionDelay = (k * 90) + 'ms'; });
      msg.innerHTML = '(B, S, 4, 4096) <span class="op">→</span> <em>(B, S, 16384)</em> &nbsp;<span class="cm">// 摊平后一次投影就能看见「哪条流在哪个位置」</span>';
    });
    tl.at(5300, () => {
      c1.style.opacity = '1';
      msg.innerHTML = '<em>UnweightedRMSNorm</em>：<span class="hl3">故意不带 weight</span> —— 它的活是给权重生成器整形，不是给模型做特征归一化';
    });
    tl.at(8600, () => {
      c2.style.opacity = '1'; c1.classList.add('ac');
      msg.innerHTML = 'fn 的形状 <em>(24, 16384)</em> = <span class="hl2">(2 + N) × N</span> 行 —— 行数这个表达式本身就是「三段权重」的声明';
    });
    tl.at(11900, () => {
      c1.classList.remove('ac'); c2.classList.add('ac');
      g.cellList.forEach((e, k) => {
        e.style.transitionDelay = (k * 18) + 'ms';
        e.style.transform = 'translateY(-4px) scale(1.1)';
      });
      msg.innerHTML = '24 个数切成 <span class="hl3">pre(4)</span> + <span class="hl4">post(4)</span> + <span class="hl2">comb(16)</span> —— 三个出口，三种激活';
    });
    tl.at(15200, () => {
      [sf, sl].forEach(e => e.classList.add('ac'));
      g.cellList.forEach(e => { e.style.transform = 'none'; });
      msg.innerHTML = '<span class="cm">// 代价：每层 786,486 个参数，45 层共 35,391,870 —— mHC 不是零成本的改造</span>';
    });
  },
},

/* ------------------------------------------------- 3 三个出口 */
{
  kicker: '第 1 步 · 三个出口',
  title: '三种激活，三种值域：<span class="hl-c">pre</span> / <span class="hl-b">post</span> / <span class="hl-a">comb</span>',
  sub: 'sigmoid、2×sigmoid、softmax —— 三个出口的激活函数是分开选的，因为它们的活完全不同。',
  caption: '三个 + hc_eps 都是保险丝：Sinkhorn 里的每一次除法都可能碰上 0 分母。',
  lang: 'python',
  codeStart: 308,
  code: `        pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)

        comb_w = comb_w.view(*comb_w.shape[:-1], hc, hc)  # these are matrix weights, unlike pre or post
        comb_b = comb_b.view(hc, hc)

        # All weights are computed with a one layer perceptron. For pre and post, this is it.
        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)
        comb = torch.softmax(comb_w * comb_scale + comb_b, dim=-1) + self.hc_eps`,
  codeNote: '三个出口的全部算式：base 加偏置、scale 是各自的可学习缩放。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '三个出口：同一个投影出来的 24 个数，走三条不同的激活' }));
    const defs = [
      { cc: 2, t: 'pre · 收拢门控', f: 'sigmoid(w·s + b) + hc_eps', r: '(0, 1 + eps]',
        m: '实测 [0.4458, 0.5677]', n: '权重和 2.01（≠ 1）', act: 'sigmoid' },
      { cc: 4, t: 'post · 写回门控', f: '2 × sigmoid(w·s + b)', r: '(0, 2)',
        m: '实测 [0.9152, 1.1225]', n: '中心在 1 → 初始退化成标准残差', act: '2×sigmoid' },
      { cc: 1, t: 'comb · 流间混合', f: 'softmax(w·s + b, dim=-1) + hc_eps', r: '行和为 1',
        m: '实测 [0.2021, 0.3118]', n: '行随机 → 还要过 Sinkhorn', act: 'softmax' },
    ];
    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cards = defs.map(d => {
      const c = W.card({
        cc: d.cc, tint: d.cc, title: d.t,
        kids: [
          U.el('div', { class: 'mono', style: 'font-size:10.5px;color:var(--c' + d.cc + ');margin:2px 0 4px', text: d.f }),
          U.el('div', { class: 'n sm', style: 'color:#fff', text: '值域 ' + d.r }),
          U.el('div', { class: 'mono', style: 'font-size:10px;color:var(--ink-dim);margin-top:4px', text: d.m }),
          U.el('div', { class: 'mono', style: 'font-size:10px;color:var(--ink-faint);margin-top:3px', text: d.n }),
        ],
        style: 'flex:1 1 0',
      });
      row.appendChild(c); return c;
    });
    viz.appendChild(row);

    const tb = W.table([
      ['pre', 'sigmoid', '(0, 1+eps]', '<span class="hlbad">不归一化</span>', '收拢时每条流最多贡献 1 倍'],
      ['post', '2 × sigmoid', '(0, 2)', '中心 1', '写回时每条流自带一个门控'],
      ['comb', 'softmax(dim=-1)', '行和 = 1', '再过 Sinkhorn', '流与流之间怎么混合'],
    ], { head: ['出口', '激活', '值域', '归一化', '它管什么'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 0, style: 'padding:9px 12px;width:100%' }, [
      U.el('div', { class: 'klabel', text: '三个出口的对照表（值域为实测 / 推导）' }), tb,
    ]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// base 给偏置，scale 给三个出口各自的可学习缩放</span>';
    tl.at(2200, () => {
      cards[0].classList.add('ac');
      msg.innerHTML = '<span class="hl3">pre</span> 值域 (0, 1+eps] 且<b>没有归一化</b>：实测 4 条流权重和 = <em>2.01</em>，不是 1 —— 收拢不是加权平均';
    });
    tl.at(5400, () => {
      cards[0].classList.remove('ac'); cards[1].classList.add('ac');
      msg.innerHTML = '<span class="hl4">post</span> 用 <em>2 × sigmoid</em> 把中心挪到 1：初始化时 post ≈ 1，mHC 退化成教科书残差 x + f(norm(x))';
    });
    tl.at(8600, () => {
      cards[1].classList.remove('ac'); cards[2].classList.add('ac');
      msg.innerHTML = '<span class="hl2">comb</span> 先 softmax 成<b>行随机</b>矩阵 —— 这只是半成品，需要 Sinkhorn 补上「列和也是 1」';
    });
    tl.at(11800, () => {
      cards.forEach(c => c.classList.add('ac'));
      msg.innerHTML = '三个 <em>+ hc_eps</em> 是保险丝：<span class="cm">softmax / sigmoid 永远够不到 0，而 Sinkhorn 要拿它们做分母</span>';
    });
    tl.at(15000, () => {
      msg.innerHTML = '<span class="cm">// 下一幕：为什么 comb 必须是双随机，而不只是行随机</span>';
    });
  },
},

/* ------------------------------------------------- 4 ★ 为什么双随机 */
{
  kicker: '第 1 步 · ★ 洞察 1',
  title: '★ 行随机允许「<span class="hlbad">所有流被同一方向放大</span>」',
  sub: '只看行和，一张退化的矩阵仍能合法存在：4 行完全相同。它的最大奇异值是 1.46，不是 1。',
  caption: 'sigma_max（最大奇异值）= 这一层对最坏方向上的放大倍数。双随机的价值就是把 45 层的界钉在 1。',
  lang: 'python',
  codeStart: 319,
  code: `        # The comb weight is a bit different: it dictates how the input streams (In) are added to the output streams
        # (Out) in this way: Mixed = In @ Comb + Out. To make sure the norm of "Mixed" does not blow up, we constrain
        # the comb weight to be doubly-stochastic (ie. its rows and columns must sum to 1) with a few iterations of the
        # Sinkhorn-Knopp algorithm, which iteratively normalizes the rows and columns to sum to 1.
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)`,
  codeNote: '作者把理由写在注释里：不是为了让权重「好看」，而是为了不让混合后的范数涨上去。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '退化例：4 行完全相同（行随机完全允许，Sinkhorn 不允许）' }));

    const p = [0.70, 0.20, 0.06, 0.04];
    const stage = U.el('div', { class: 'row gap16 center', style: 'width:100%' });
    viz.appendChild(stage);

    const A = mtxBlock('softmax 之后：4 行都是 p', 4, 4, (r, c) => p[c], {
      cell: 30, rgb: '251,113,133', fs: 8.5,
      text: (r, c) => p[c].toFixed(2),
      note: '行和 1.00 1.00 1.00 1.00\n列和 2.80 0.80 0.24 0.16\nsigma_max = 1.4631',
      noteColor: 'var(--c3)',
    });
    stage.appendChild(A);
    stage.appendChild(U.el('div', { class: 'col center gap4', style: 'flex:none;width:120px' }, [
      U.el('div', { class: 'mono', style: 'font-size:10.5px;color:var(--accent);text-align:center', html: 'Sinkhorn<br>20 次' }),
      U.el('div', { style: 'font-size:22px;color:var(--accent);line-height:1', text: '⟹' }),
    ]));
    const B = mtxBlock('Sinkhorn 之后：均匀矩阵', 4, 4, () => 0.25, {
      cell: 30, rgb: '52,211,153', fs: 8.5,
      text: () => '0.25',
      note: '行和 1.00 1.00 1.00 1.00\n列和 1.00 1.00 1.00 1.00\nsigma_max = 0.999999',
      noteColor: 'var(--c4)',
    });
    stage.appendChild(B);

    const right = U.el('div', { class: 'col gap6', style: 'flex:1' });
    right.appendChild(W.card({ cc: 3, tint: 3, title: '行随机漏掉了什么',
      sub: '行和 = 1 只保证「每条输入流的总贡献守恒」；4 行相同意味着<b>所有流都偏向同一列</b>，'
         + '此时存在一个方向每层被放大 46%。', style: 'width:100%' }));
    right.appendChild(W.card({ cc: 4, tint: 4, title: '双随机补上了什么',
      sub: '列和 = 1 保证「每条输出流接收的总权重也是 1」：没有流被整体放大，也没有流被饿死'
         + '（实测 comb 最小元素 0.2021）。', style: 'width:100%' }));
    stage.appendChild(right);

    const stats = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const s1 = statCard('1.3017', '200 个行随机样本的 sigma_max 最大值', 3, '均值 1.0712');
    const s2 = statCard('0.999999', '同 200 个样本 Sinkhorn 之后', 4, '均值 0.999999');
    const s3 = statCard('1.42e+05', '行随机 45 层的界 λ⁴⁵', 2, '双随机恒为 1');
    [s1, s2, s3].forEach(e => stats.appendChild(e));
    viz.appendChild(stats);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 左边这张矩阵：行和全是 1，看起来「很合法」</span>';
    tl.at(2000, () => {
      s1.classList.add('ac');
      msg.innerHTML = '但它 4 行相同 → 列和是 <em>2.80 / 0.80 / 0.24 / 0.16</em>：第 0 列被所有流一起喂，<span class="hlbad">sigma_max = 1.4631</span>';
    });
    tl.at(5400, () => {
      s1.classList.remove('ac'); s2.classList.add('ac');
      msg.innerHTML = 'Sinkhorn 把它推成 <em>0.25 均匀矩阵</em>：行和、列和都精确为 1，<span class="hl4">sigma_max = 0.999999</span>';
    });
    tl.at(8800, () => {
      s2.classList.remove('ac'); s3.classList.add('ac');
      msg.innerHTML = '训练要的是<b>界</b>不是平均：45 层的界从 <span class="hl3">1.3017⁴⁵ ≈ 1.4e5</span> 压到 <em>1</em> —— 反向传播的梯度会主动去找那个最坏方向';
    });
    tl.at(12200, () => {
      B._g.cellList.forEach((e, k) => {
        e.style.transitionDelay = (k * 22) + 'ms';
        e.style.transform = 'translateY(-3px) scale(1.08)';
      });
      msg.innerHTML = '双随机还保证 <em>combᵀ 也是双随机</em>（实测两者 sigma_max 都是 0.999999）：前向与反向用的是同一个「不放大」的算子';
    });
    tl.at(15200, () => {
      s3.classList.remove('ac');
      B._g.cellList.forEach(e => { e.style.transform = 'none'; });
      msg.innerHTML = '<span class="cm">// 回顾 §一：注释里那句 "does not blow up"，说的就是这一幕</span>';
    });
  },
},

/* ------------------------------------------------- 5 ★ 手算 */
{
  kicker: '第 1 步 · ★ 手算',
  title: '★ 手算 2 次 Sinkhorn 迭代：<span class="hl-a">先归列，再行、列交替</span>',
  sub: '取列和 = 0.5 / 1.0 / 1.5 的 3×3 行随机矩阵，两次迭代看行和误差怎么掉一个数量级。',
  caption: '顺序反常是有原因的：comb 刚从 softmax(dim=-1) 出来，行已经和是 1 —— 先补列才是最快的一步。',
  lang: 'python',
  codeStart: 323,
  code: `        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)`,
  codeNote: '就这 4 行。dim=-2 是列（沿行方向求和），dim=-1 是行。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '一个 3×3 行随机矩阵的两次迭代（数值由源码同款公式算出）' }));

    const M0 = [[0.30, 0.30, 0.40], [0.15, 0.40, 0.45], [0.05, 0.30, 0.65]];
    const M1 = [[0.60, 0.30, 0.266667], [0.30, 0.40, 0.30], [0.10, 0.30, 0.433333]];
    const M2 = [[0.550459, 0.252809, 0.217984], [0.321101, 0.393258, 0.286104], [0.128440, 0.353933, 0.495913]];
    const M3 = [[0.543763, 0.246952, 0.212103], [0.323786, 0.392130, 0.284170], [0.132450, 0.360918, 0.503727]];

    const defs = [
      { lab: 'M0 · softmax 输出', m: M0, c: 3, note: '行和 1.000 1.000 1.000\n列和 0.500 1.000 1.500' },
      { lab: 'L323 · 列归一', m: M1, c: 2, note: '行和 1.167 1.000 0.833\n列和 1.000 1.000 1.000' },
      { lab: '迭代 1 · 行→列', m: M2, c: 1, note: '行和 1.0213 1.0005 0.9783\n列和 1.000 1.000 1.000' },
      { lab: '迭代 2 · 行→列', m: M3, c: 4, note: '行和 1.0028 1.0001 0.9971\n列和 1.000 1.000 1.000' },
    ];
    const stage = U.el('div', { class: 'row gap10 center wrap', style: 'width:100%;justify-content:center' });
    const blocks = defs.map((d, i) => {
      if (i > 0) {
        stage.appendChild(U.el('div', { class: 'mono', style: 'flex:none;font-size:16px;color:var(--accent)', text: '⟹' }));
      }
      const b = mtxBlock(d.lab, 3, 3, (r, c) => d.m[r][c], {
        cell: 34, rgb: '56,189,248', fs: 8.5,
        text: (r, c) => (d.m[r][c] < 0.1 ? d.m[r][c].toFixed(2) : d.m[r][c].toFixed(3)),
        note: d.note, noteColor: 'var(--c' + d.c + ')',
      });
      b.style.transition = 'opacity .4s,transform .4s';
      stage.appendChild(b);
      return b;
    });
    viz.appendChild(stage);

    const low = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const c1 = W.card({ cc: 0, tint: 0, title: '行和误差：1.667e-01 → 2.175e-02 → 2.819e-03',
      sub: '每轮大约掉一个数量级。真模块（iters=1/2/3）实测 3.4e-3 → 3.8e-5 → 1.4e-6，同一个节奏。',
      style: 'flex:1 1 0' });
    const c2 = W.card({ cc: 5, tint: 5, title: '列归一永远在最后一步',
      sub: '迭代 k 次后的状态是「行和 ≈ 1、列和 = 1（精确）」。这也是为什么第一步先归列 —— '
         + 'softmax 已经把行归过了，再归一次行是白做。', style: 'flex:1 1 0' });
    low.appendChild(c1); low.appendChild(c2);
    viz.appendChild(low);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    blocks.forEach((b, i) => { if (i > 0) { b.style.opacity = '.28'; b.style.transform = 'translateX(-10px)'; } });
    msg.innerHTML = '<span class="cm">// 起点：行都是 1，列是 0.5 / 1.0 / 1.5</span>';
    tl.at(2200, () => {
      blocks[1].style.opacity = '1'; blocks[1].style.transform = 'none';
      msg.innerHTML = '第一步（L323）除以<b>列</b>和：第 0 列 ÷0.5、第 2 列 ÷1.5 → <em>列和精确等于 1</em>，但行被打乱成 1.167 / 1.000 / 0.833';
    });
    tl.at(5600, () => {
      blocks[2].style.opacity = '1'; blocks[2].style.transform = 'none'; blocks[0].style.opacity = '.5';
      msg.innerHTML = '迭代 1：先按行归一（行和回到 1），再按列归一 —— 行和误差掉到 <em>2.175e-02</em>';
    });
    tl.at(9000, () => {
      blocks[3].style.opacity = '1'; blocks[3].style.transform = 'none';
      msg.innerHTML = '迭代 2：误差 <em>2.819e-03</em>。20 次迭代后是 <span class="hl4">0.5427 / 0.2461 / 0.2112</span>…，'
        + '<span class="cm">// 不是均匀的 1/3</span>';
    });
    tl.at(12400, () => {
      c1.classList.add('ac');
      msg.innerHTML = 'Sinkhorn 是<b>投影</b>到双随机矩阵的集合上，不是做平均：它保住「哪条流跟哪条流亲」，只抹掉「总量不守恒」';
    });
    tl.at(15600, () => {
      c1.classList.remove('ac'); c2.classList.add('ac');
      msg.innerHTML = '20 次迭代实际是 <em>19 次行归一 + 20 次列归一 = 39 次除法</em> —— 配置名不等于除法次数';
    });
  },
},

/* ------------------------------------------------- 6 写回 */
{
  kicker: '第 3 步 · 写回',
  title: '写回：<span class="hl-c">post ⊙ 子层输出</span> + <span class="hl-a">combᵀ · 旧流</span>',
  sub: '两项都在 (B, S, 4, 4096) 上做加法：一项是「各自的子层输出」，一项是「旧流之间的混合」。',
  caption: '回顾 L3-05：MoE 的共享专家在 mlp 内部合流；mHC 在 mlp 外面，先收拢再写回。',
  lang: 'python',
  codeStart: 1337,
  code: `        hidden_states = post.to(dtype).unsqueeze(-1) * hidden_states.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual
        )`,
  codeNote: '广播是关键：子层输出 (B,S,D) 加一维成 (B,S,1,D)，post 加一维成 (B,S,4,1)。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const stage = U.el('div', { class: 'row gap12 center hstart', style: 'width:100%;align-items:stretch' });
    viz.appendChild(stage);

    const mkTerm = (cc, title, formula, shapeRows, sub) => {
      const box = W.card({ cc: cc, tint: cc, title: title, style: 'flex:1 1 0' });
      box.appendChild(U.el('div', { class: 'mono', style: 'font-size:10.5px;color:var(--c' + cc + ');margin:6px 0 8px;line-height:1.6', html: formula }));
      shapeRows.forEach(r => box.appendChild(U.el('div', {
        class: 'row gap8', style: 'width:100%',
      }, [
        U.el('div', { class: 'mono', style: 'font-size:10px;color:var(--ink-dim);width:150px;flex:none', text: r[0] }),
        U.el('div', { class: 'mono', style: 'font-size:10px;color:var(--ink-faint)', text: r[1] }),
      ])));
      if (sub) box.appendChild(U.el('div', { class: 'mono', style: 'font-size:10px;color:var(--ink-faint);margin-top:7px;line-height:1.5', text: sub }));
      return box;
    };

    const A = mkTerm(4, '第一项 · 各自的子层输出',
      'post.to(dtype).unsqueeze(-1)<br>&nbsp;&nbsp;* hidden_states.unsqueeze(-2)',
      [['post', '(B, S, 4, 1)'], ['子层输出', '(B, S, 1, 4096)'], ['广播结果', '(B, S, 4, 4096)']],
      '每条流拿到同一份子层输出，各乘自己的门控值');
    const plus = U.el('div', { class: 'col center', style: 'flex:none;width:44px' }, [
      U.el('div', { class: 'mono', style: 'font-size:20px;color:var(--accent)', text: '+' }),
    ]);
    const B = mkTerm(1, '第二项 · 旧流之间的混合',
      'torch.matmul(comb.transpose(-1, -2),<br>&nbsp;&nbsp;residual)',
      [['combᵀ', '(B, S, 4, 4)'], ['residual（旧流）', '(B, S, 4, 4096)'], ['混合结果', '(B, S, 4, 4096)']],
      '写成 In @ Comb 也一样：两者逐元素实测相同');
    const eq = U.el('div', { class: 'col center', style: 'flex:none;width:44px' }, [
      U.el('div', { class: 'mono', style: 'font-size:20px;color:var(--accent)', text: '=' }),
    ]);
    const C = W.card({ cc: 0, tint: 0, title: '4 条新流 (B, S, 4, 4096)', style: 'flex:0 0 190px' });
    const outBars = [];
    for (let s = 0; s < HC.N; s++) { const b = streamBar('新流 ' + s, (s + 2) % 6, 150, 22); C.appendChild(b); outBars.push(b); }

    [A, plus, B, eq, C].forEach(e => stage.appendChild(e));

    const stats = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const s1 = statCard('0.999999', 'sigma_max(comb)', 4, '双随机 → 不放大');
    const s2 = statCard('0.5875', '一次采样的 ‖combᵀ·r‖ / ‖r‖', 1, '这一步在收缩');
    const s3 = statCard('×2', '每层出现次数', 2, 'attn_hc / ffn_hc 各一套权重');
    [s1, s2, s3].forEach(e => stats.appendChild(e));
    viz.appendChild(stats);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    outBars.forEach(b => { b.style.opacity = '.2'; });
    msg.innerHTML = '<span class="cm">// 两项形状一样，直接相加</span>';
    tl.at(2000, () => {
      A.classList.add('ac');
      msg.innerHTML = '第一项：子层输出只有 <em>1 条</em>，靠广播复制成 4 条 —— 每条流自己决定「我写回多少」';
    });
    tl.at(5400, () => {
      A.classList.remove('ac'); B.classList.add('ac');
      msg.innerHTML = '第二项：旧流按 <em>combᵀ</em> 混合。双随机保证这一步<b>不放大</b>：实测 sigma_max = 0.999999';
    });
    tl.at(8800, () => {
      B.classList.remove('ac'); C.classList.add('ac');
      outBars.forEach((b, k) => { b.style.opacity = '1'; b.style.transitionDelay = (k * 100) + 'ms'; });
      msg.innerHTML = '4 条新流 = <span class="hl4">post ⊙ out</span> <span class="op">+</span> <span class="hl">combᵀ · 旧流</span>';
    });
    tl.at(12000, () => {
      [s1, s2].forEach(e => e.classList.add('ac'));
      msg.innerHTML = '一次真实采样：‖combᵀ · residual‖ / ‖residual‖ = <em>0.5875</em> —— 混合这一步在收缩，不在放大';
    });
    tl.at(15000, () => {
      s3.classList.add('ac');
      msg.innerHTML = '<span class="cm">// 这 3 行在解码层里出现两次（注意力后、MLP 后），attn_hc 与 ffn_hc 是两套独立权重</span>';
    });
  },
},

/* ------------------------------------------------- 7 收拢 */
{
  kicker: '第 2 步 · 收拢',
  title: '收拢：<span class="hl-c">pre 是门控</span>，不是加权平均',
  sub: '4 条流权重实测加起来是 2.01，不是 1 —— 收拢后的幅值可以比任何一条流都大。',
  caption: '这与 post 中心取 1 是同一个思路：mHC 要的是「可控的缩放」，不是「守恒的平均」。',
  lang: 'python',
  codeStart: 328,
  code: `        # Since "pre" is meant to be used with the input streams (available here as \`hidden_streams\`), we collapse the
        # streams here and return \`collapsed\` tensor, which will be the input for the next attention or MLP block.
        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
        return post, comb, collapsed`,
  codeNote: 'pre 在这里被消费掉，不随返回值出去 —— 所以外面永远看不到它。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const stage = U.el('div', { class: 'row gap16 hstart', style: 'width:100%;align-items:stretch' });
    viz.appendChild(stage);

    /* 左：pre 的实测权重（条形图） */
    const left = U.el('div', { class: 'col gap6', style: 'flex:1 1 0' });
    left.appendChild(U.el('div', { class: 'klabel', text: 'pre · 一次真实前向的 4 个门控值（hidden_size=8）' }));
    const bars = W.bars([
      { label: '流 0', value: 0.567682, cc: 0, valueText: '0.5677' },
      { label: '流 1', value: 0.458231, cc: 1, valueText: '0.4582' },
      { label: '流 2', value: 0.514862, cc: 2, valueText: '0.5149' },
      { label: '流 3', value: 0.471178, cc: 3, valueText: '0.4712' },
    ], { max: 1.0 });
    left.appendChild(bars);
    const sumLine = U.el('div', { class: 'mono', style: 'font-size:11px;color:var(--c3);margin-top:2px' });
    sumLine.innerHTML = '权重和 = <b>2.011953</b> &nbsp;<span class="cm">// 若做过归一化，这里必须是 1</span>';
    left.appendChild(sumLine);
    stage.appendChild(left);

    /* 右：两张对照卡 */
    const right = U.el('div', { class: 'col gap8', style: 'flex:1 1 0' });
    right.appendChild(W.card({ cc: 3, tint: 3, title: '如果是加权平均（softmax / 归一化）',
      sub: '权重和恒为 1 → 收拢后的幅值一定不超过最大的那条流。<b>守恒，但不可调</b>。',
      style: 'width:100%' }));
    right.appendChild(W.card({ cc: 2, tint: 2, title: '实测的 pre 门控',
      sub: '每条流最多贡献 1 倍（sigmoid + eps），和可以是 2.01 —— 收拢后的幅值<b>可以比任何一条流都大</b>。'
         + '它不是凸组合。', style: 'width:100%' }));
    right.appendChild(W.card({ cc: 0, tint: 0, title: '返回 (post, comb, collapsed)',
      sub: '文档字符串写着：All weights are returned except "pre", which is consumed here.',
      style: 'width:100%' }));
    stage.appendChild(right);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    bars.setAll(0);
    msg.innerHTML = '<span class="cm">// 注意力 / MLP 只看得到收拢后的那 1 条流</span>';
    tl.at(2000, () => {
      bars.setAll(1);
      msg.innerHTML = '(pre.unsqueeze(-1) * hidden_streams).sum(dim=2) &nbsp;<span class="op">→</span>&nbsp; <em>(B, S, 4096)</em>';
    });
    tl.at(5300, () => {
      right.children[0].classList.add('ac');
      msg.innerHTML = '如果这里是 softmax / 归一化：<span class="hlbad">权重和恒为 1</span>，收拢只能在这 4 条流之间插值';
    });
    tl.at(8600, () => {
      right.children[0].classList.remove('ac'); right.children[1].classList.add('ac');
      msg.innerHTML = '<span class="hl3">实测门控和 2.011953</span> —— mHC 故意不给它归一化：收拢要能整体放大或缩小';
    });
    tl.at(11900, () => {
      right.children[1].classList.remove('ac'); right.children[2].classList.add('ac');
      msg.innerHTML = 'pre 在这里被消费掉、<b>不返回</b>：外面拿到的只有 post / comb / collapsed';
    });
    tl.at(14800, () => {
      msg.innerHTML = '<span class="cm">// 对照下一幕的 hc_head：出口处的收拢是 0 参数的等权平均，入口处的收拢是学出来的门控</span>';
    });
  },
},

/* ------------------------------------------------- 8 出口 */
{
  kicker: '出口 · HyperHead',
  title: '收尾：<span class="hl-a">hc_head</span> 是 0 参数的等权平均',
  sub: '45 层跑完还是 4 条流，主干出口用一行 mean(dim=2) 收成 1 条 —— 连一个参数都没有。',
  caption: '回顾 L0-01 的实测订正：hc_head 不是「学习一组混合权重」；本课补上另一半 —— 带权重的是入口的 pre。',
  lang: 'python',
  codeStart: 334,
  code: `class Glm5NextTextHyperHead(nn.Module):
    """Final GLM-5.3-Flash HC-stream collapse. Unlike DeepSeek-V4, this is an unweighted mean."""

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        return hidden_streams.mean(dim=2)`,
  codeNote: '整个类的全部内容。实例化后 parameters() 是空的。',
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const stage = U.el('div', { class: 'row gap20 center', style: 'width:100%' });
    viz.appendChild(stage);

    const inCol = U.el('div', { class: 'col gap5 center' });
    inCol.appendChild(U.el('div', { class: 'klabel', text: '(B, S, 4, 4096)' }));
    const inBars = [];
    for (let s = 0; s < HC.N; s++) { const b = streamBar('流 ' + s, s, 230, 30); inCol.appendChild(b); inBars.push(b); }
    stage.appendChild(inCol);

    stage.appendChild(U.el('div', { class: 'col center gap6', style: 'flex:none' }, [
      U.el('div', { style: 'font-size:24px;color:var(--accent);line-height:1', text: '⟹' }),
      U.el('div', { class: 'mono', style: 'font-size:10.5px;color:var(--c2);text-align:center', html: 'hc_head<br>mean(dim=2)' }),
    ]));

    const outCol = U.el('div', { class: 'col gap5 center' });
    outCol.appendChild(U.el('div', { class: 'klabel', text: '(B, S, 4096)' }));
    const outBar = U.el('div', {
      style: 'width:230px;height:140px;border-radius:10px;border:1px solid var(--accent);'
           + 'background:linear-gradient(180deg,rgba(34,211,238,.30),rgba(34,211,238,.10));'
           + 'display:flex;align-items:center;justify-content:center;'
           + 'font:600 12px var(--mono);color:#c9f7ff;'
           + 'transition:opacity .5s,transform .5s',
      text: 'last_hidden_state',
    });
    outCol.appendChild(outBar);
    stage.appendChild(outCol);

    const right = U.el('div', { class: 'col gap8', style: 'flex:1 1 0;min-width:280px' });
    const r1 = W.card({ cc: 4, tint: 4, title: '实测：参数量 = 0',
      sub: 'sum(p.numel() for p in h.parameters()) == 0，且 torch.allclose(h(x), x.mean(dim=2)) 为 True。',
      style: 'width:100%' });
    const r2 = W.card({ cc: 2, tint: 2, title: '与入口的 pre 对照',
      sub: 'pre 是逐 token 学出来的 sigmoid 门控（和 = 2.01）；hc_head 是全局固定的等权平均（和恒为 1）。'
         + '同一个「收拢」动作，入口处学、出口处不学。', style: 'width:100%' });
    const r3 = W.card({ cc: 1, tint: 1, title: 'docstring 里的记号',
      sub: 'Unlike DeepSeek-V4, this is an unweighted mean —— 上一代用带权重的收拢，这一代主动去掉了权重。',
      style: 'width:100%' });
    [r1, r2, r3].forEach(e => right.appendChild(e));
    stage.appendChild(right);

    viz.appendChild(U.el('div', { class: 'row gap10 center', style: 'width:100%' }, [
      U.el('div', { class: 'mono', style: 'font-size:11px;color:var(--ink-dim)', html: '主干出口：' }),
      U.el('div', { class: 'mono', style: 'font-size:11px;color:var(--accent)', html: 'hidden_states = self.norm(self.hc_head(hidden_states))' }),
      U.el('div', { class: 'mono', style: 'font-size:11px;color:var(--ink-faint)', html: '&nbsp;// 先收拢、再归一化，顺序不能反' }),
    ]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    outBar.style.opacity = '0'; outBar.style.transform = 'scale(.92)';
    msg.innerHTML = '<span class="cm">// 45 层之后，手上还是 4 条流</span>';
    tl.at(2200, () => {
      r1.classList.add('ac');
      msg.innerHTML = '<em>hc_head</em> 的全部实现是 <span class="hl4">hidden_streams.mean(dim=2)</span> —— 0 个参数、0 个可学习门控';
    });
    tl.at(5600, () => {
      inBars.forEach(b => { b.style.opacity = '.3'; b.style.transform = 'translateX(12px)'; });
      outBar.style.opacity = '1'; outBar.style.transform = 'none';
      msg.innerHTML = '4 条 → <em>1</em> 条：<span class="cm">等权平均，不需要 Sinkhorn，也没有 comb 矩阵</span>';
    });
    tl.at(8900, () => {
      inBars.forEach(b => { b.style.opacity = '1'; b.style.transform = 'none'; });
      r1.classList.remove('ac'); r2.classList.add('ac');
      msg.innerHTML = '入口的 <span class="hl3">pre</span>（逐 token 门控，和 2.01）与出口的 <span class="hl4">hc_head</span>（全局等权，和 1）是同一个动作的两种做法';
    });
    tl.at(12100, () => {
      r2.classList.remove('ac'); r3.classList.add('ac');
      msg.innerHTML = '<span class="cm">// 顺序不能反：norm 是逐流的，收拢必须在前 —— 与 L0-01 的结论一致</span>';
    });
  },
},

/* ------------------------------------------------- 9 收束 */
{
  kicker: '收束',
  title: '把三步连起来：<span class="hl-a">一张表 + 两条不变量</span>',
  sub: 'mHC 的全部机制就在这 24 个数、4 行 Sinkhorn 代码和一次广播加法里。',
  caption: '下一课 L3-07：遗忘门与门控范数 —— 收拢后的那条流在 KDA 里还会再被门控一次。',
  lang: 'python',
  codeStart: 273,
  code: `    def __init__(self, config: Glm5NextTextConfig):
        super().__init__()
        self.hc_mult = config.hc_mult  # number of streams, referred as N below
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.input_norm = Glm5NextTextUnweightedRMSNorm(eps=config.rms_norm_eps)
        # The mHC projects the N inputs streams into 3 weights: pre (size: N), post (size: N) and comb (size: N*N)
        # Hence the output size of the projection is 2 * N + N * N = (2 + N) * N.
        concatenated_weights_size = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(concatenated_weights_size, self.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.empty(concatenated_weights_size))
        # The mHC produces 3 outputs, each with their own scale parameter (the "pre", "post" and "comb" weights)
        self.scale = nn.Parameter(torch.empty(3))`,
  codeNote: '回到 __init__：本课的每个数字都写在这几行里（N=4、iters=20、eps=1e-6、fn 的形状）。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const tb = W.table([
      ['① 读取', 'L298-317', '24 = (2+4)×4 个数', '一次投影 → pre(4) / post(4) / comb(16)；fp32'],
      ['② 收拢', 'L328-331', '权重和 2.01', '(pre ⊙ streams).sum(2) → (B,S,D)，未归一化'],
      ['③ 写回', 'L1337-1339', 'σ_max(comb) = 1', 'post ⊙ out + combᵀ · 旧流；每层两次'],
      ['Sinkhorn', 'L323-326', '1 列 + 19×(行,列)', '39 次除法 → 双随机；iters 是配置名'],
      ['出口', 'L334-338', '参数量 0', 'hc_head = mean(dim=2)'],
    ], { head: ['步骤', '源码位置', '关键数字（实测）', '一句话'] });
    const tblCard = U.el('div', { class: 'card', cc: 0, style: 'padding:10px 12px;width:100%' }, [
      U.el('div', { class: 'klabel', text: '本课全部内容压成一张表' }), tb,
    ]);
    viz.appendChild(tblCard);

    const inv = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const i1 = W.card({ cc: 4, tint: 4, title: '不变量 1 · 每一行和为 1',
      sub: '每条输入流的总贡献守恒：混合是加权平均，不放大总和。', style: 'flex:1 1 0' });
    const i2 = W.card({ cc: 1, tint: 1, title: '不变量 2 · 每一列和为 1',
      sub: '每条输出流接收的总权重守恒：没有流被整体放大，也没有流被饿死。'
         + '两条合起来 → σ_max = 1 → 45 层的放大倍数被钉死。', style: 'flex:1 1 0' });
    inv.appendChild(i1); inv.appendChild(i2);
    viz.appendChild(inv);

    viz.appendChild(W.exercise(
      '取 M0 = [[0.30, 0.30, 0.40], [0.15, 0.40, 0.45], [0.05, 0.30, 0.65]]（行和为 1，列和为 0.5 / 1.0 / 1.5）。'
      + '按源码顺序手算第一次迭代：先归列，再归行、归列。第 0 行最后三个数各是多少？',
      '① <b>列归一</b>（除以 0.5 / 1.0 / 1.5）：第 0 行 → 0.60 / 0.30 / 0.266667，新的行和 = <b>1.166667</b>。'
      + '<br>② <b>行归一</b>（除以 1.166667）：→ 0.514286 / 0.257143 / 0.228571。'
      + '<br>③ <b>列归一</b>：列和 = 0.934286 / 1.017143 / 1.048571，除以它们后 → '
      + '<b>0.550459 / 0.252809 / 0.217984</b>（与上一幕表格一致，误差 2.175e-02 就是这么来的）。'
      + '<br>注意第 ① 步的列和一定是精确的 1：源码里那次除法用的是 <code class="inl">dim=-2</code>。'));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    [tblCard, inv].forEach(e => { e.style.opacity = '.35'; });
    msg.innerHTML = '<span class="cm">// 三步 + 一次 Sinkhorn，就是 mHC 的全部</span>';
    tl.at(2000, () => {
      tblCard.style.opacity = '1';
      msg.innerHTML = '读取 24 个数 → 收拢用 <span class="hl3">未归一化的 pre</span> → 写回用 <span class="hl4">post</span> 与 <span class="hl">combᵀ</span>';
    });
    tl.at(5400, () => {
      i1.classList.add('ac'); inv.style.opacity = '1';
      msg.innerHTML = '不变量 1：<em>行和 = 1</em> —— combᵀ·旧流 是加权平均，不放大总和（实测偏差 1.04e-06）';
    });
    tl.at(8700, () => {
      i1.classList.remove('ac'); i2.classList.add('ac');
      msg.innerHTML = '不变量 2：<em>列和 = 1</em> —— 每一条输出流接收的总权重也是 1，sigma_max 恰好为 0.999999';
    });
    tl.at(12000, () => {
      i2.classList.remove('ac');
      msg.innerHTML = '两条合起来：<b>双随机 ⇒ σ_max = 1 ⇒ 45 层的界是 1 而不是 1.4e5</b> —— 这就是它稳住梯度流的方式';
    });
    tl.at(15200, () => {
      msg.innerHTML = '<span class="cm">// 下一课 L3-07：收拢后的那条流进入 KDA，还会遇到遗忘门与门控范数</span>';
    });
  },
},

];
