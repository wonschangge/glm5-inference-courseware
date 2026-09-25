/* ==========================================================================
   L0-05 · mHC：4 条残差流
   --------------------------------------------------------------------------
   覆盖：models/glm5_next/modeling_glm5_next.py（1 个文件 / 2444 行）
   目标：看完能说清「4 条流怎么被读、怎么被混合、怎么被写回」，
         并能手算两次 Sinkhorn 迭代、指出 mHC 在解码层的前后各出现一次。
   排版基准：#visual 可视区 = 994 x 662.7 舞台像素（无头浏览器实测）。
   ========================================================================== */
'use strict';

/* ---- 实测常量（hc_mult=4 的真实尺寸模块 + hidden=64 的极小模型，见 source.md 顶部） ---- */
const HC = {
  mult: 4, eps: '1e-6', iters: 20,
  hidden: 4096, layers: 45,
  fnRows: 24, fnCols: 16384,
  params: 393243, perLayer: 786486, allLayers: 35391870,
};

/* ---- Sinkhorn 手算的五个状态（source.md 第十三节，全部来自实测脚本） ---- */
const SK = {
  A: { t: '① 初始（softmax 之后）',
       m: [[0.7000, 0.1000, 0.1000, 0.1000], [0.4000, 0.4000, 0.1000, 0.1000],
           [0.1000, 0.1000, 0.7000, 0.1000], [0.1000, 0.1000, 0.1000, 0.7000]],
       r: [1.0000, 1.0000, 1.0000, 1.0000], c: [1.3000, 0.7000, 1.0000, 1.0000],
       n: '行和 = 1，列和 1.30 / 0.70 / 1.00 / 1.00' },
  B: { t: '② 第 323 行：列归一化',
       m: [[0.5385, 0.1429, 0.1000, 0.1000], [0.3077, 0.5714, 0.1000, 0.1000],
           [0.0769, 0.1429, 0.7000, 0.1000], [0.0769, 0.1429, 0.1000, 0.7000]],
       r: [0.8813, 1.0791, 1.0198, 1.0198], c: [1.0000, 1.0000, 1.0000, 1.0000],
       n: '列和修好了，行和坏了' },
  C: { t: '③ 第 1 次（行 → 列）',
       m: [[0.5836, 0.1668, 0.1145, 0.1145], [0.2723, 0.5449, 0.0935, 0.0935],
           [0.0720, 0.1442, 0.6929, 0.0990], [0.0720, 0.1442, 0.0990, 0.6929]],
       r: [0.9794, 1.0043, 1.0081, 1.0081], c: [1.0000, 1.0000, 1.0000, 1.0000],
       n: '行和偏差 0.1187 → 0.0206' },
  D: { t: '④ 第 2 次（行 → 列）',
       m: [[0.5900, 0.1705, 0.1175, 0.1175], [0.2685, 0.5432, 0.0936, 0.0936],
           [0.0708, 0.1432, 0.6904, 0.0986], [0.0708, 0.1432, 0.0986, 0.6904]],
       r: [0.9954, 0.9988, 1.0029, 1.0029], c: [1.0000, 1.0000, 1.0000, 1.0000],
       n: '行和偏差降到 0.0046' },
  E: { t: '⑤ 走满 20 轮（代码的真实行为）',
       m: [[0.5915, 0.1714, 0.1185, 0.1185], [0.2681, 0.5439, 0.0940, 0.0940],
           [0.0702, 0.1424, 0.6890, 0.0984], [0.0702, 0.1424, 0.0984, 0.6890]],
       r: [0.999999, 0.999999, 0.999999, 0.999999], c: [0.999999, 0.999999, 0.999999, 0.999999],
       n: '行和 = 列和 = 0.999999（差 1e-6）' },
};

/* ---- 4x4 数字矩阵：格子底色随数值深浅，格子里写数字 ---- */
function numGrid(cell, font) {
  const g = U.el('div', { style: 'display:grid;grid-template-columns:repeat(4,' + cell + 'px);'
    + 'grid-template-rows:repeat(4,' + cell + 'px);gap:3px' });
  const cs = [];
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      const e = U.el('div', { style: 'border-radius:5px;display:flex;align-items:center;'
        + 'justify-content:center;font:600 ' + font + 'px var(--mono);'
        + 'border:1px solid rgba(150,180,255,.18);color:#cfe6ff;'
        + 'transition:background .45s var(--ease-out),box-shadow .45s,transform .45s' });
      g.appendChild(e); cs.push(e);
    }
  }
  g.cells = cs;
  g.paint = (m) => {
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        const v = m[r][c], e = cs[r * 4 + c];
        e.textContent = (cell >= 40 ? v.toFixed(4) : v.toFixed(2));
        e.style.background = 'rgba(34,211,238,' + (0.06 + 0.86 * Math.min(1, v / 0.75)).toFixed(3) + ')';
        e.style.color = v > 0.5 ? '#04121c' : '#cfe6ff';
      }
    }
  };
  /* 描边某一行(axis='r')或某一列(axis='c') —— 用来指出「谁不等于 1」 */
  g.hi = (idx, axis, on) => {
    for (let k = 0; k < 4; k++) {
      const e = cs[axis === 'r' ? idx * 4 + k : k * 4 + idx];
      e.style.boxShadow = on ? '0 0 0 2px var(--warn)' : 'none';
      e.style.transform = on ? 'scale(1.06)' : 'none';
    }
  };
  g.paint(SK.A.m);
  return g;
}

/* ---- 行和 / 列和 的读数格：≈1 变绿，≠1 变红 ---- */
function sumBox(label, vals) {
  const box = U.el('div', { class: 'col gap4', style: 'width:100%' });
  box.appendChild(U.el('div', { class: 'klabel', text: label }));
  const row = U.el('div', { class: 'row gap5', style: 'width:100%' });
  const els = [];
  for (let i = 0; i < 4; i++) {
    const e = U.el('div', { class: 'n sm', style: 'flex:1;text-align:center;padding:5px 0;'
      + 'border-radius:7px;border:1px solid rgba(150,180,255,.16);'
      + 'background:rgba(150,180,255,.07);color:var(--ink-dim);'
      + 'transition:background .4s,border-color .4s,color .4s' });
    row.appendChild(e); els.push(e);
  }
  box.appendChild(row);
  box.els = els;
  box.set = (v) => v.forEach((x, i) => {
    const good = Math.abs(x - 1) < 2e-3;
    els[i].textContent = x.toFixed(3);
    els[i].style.borderColor = good ? 'var(--ok)' : 'var(--bad)';
    els[i].style.background = good ? 'rgba(52,211,153,.15)' : 'rgba(251,113,133,.15)';
    els[i].style.color = good ? '#a7f3d0' : '#fecdd3';
  });
  box.set(vals);
  return box;
}

/* ---- 「一束流」的小条：画 N 条并行流 ---- */
function streamBars(n, w, h) {
  const col = U.el('div', { class: 'col gap5' });
  const bars = [];
  for (let s = 0; s < n; s++) {
    const b = U.el('div', {
      style: 'width:' + w + 'px;height:' + h + 'px;border-radius:8px;'
        + 'border:1px solid var(--c' + (s % 6) + ');background:rgba(150,180,255,.10);'
        + 'display:flex;align-items:center;justify-content:center;'
        + 'font:600 10px var(--mono);color:var(--ink-dim);'
        + 'transition:opacity .5s var(--ease-out),transform .5s var(--ease-out),background .5s',
      text: '流 ' + s });
    col.appendChild(b); bars.push(b);
  }
  col.bars = bars;
  return col;
}

const SCENES = [

/* ------------------------------------------------- 1 全景：三次出现 */
{
  kicker: '预备篇 · mHC',
  title: 'mHC：残差不是<span class="hl-a">一条</span>，而是<span class="hl-c">一束</span>',
  sub: '主干里 hidden_states 从第一步就是 4 维的。这一课只回答一个问题：这 4 条流怎么被读、怎么被混合、怎么被写回。',
  caption: '本课覆盖 modeling_glm5_next.py 的 12 个引用块，全部逐字来自源码。',
  lang: 'python',
  codeStart: 1495,
  code: `        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()

        topk_indices = None
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states, topk_indices = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                # Key change using NoPE
                position_embeddings=None,
                input_ids=input_ids,
                past_key_values=past_key_values,
                prev_topk_indices=topk_indices,
                **kwargs,
            )

        hidden_states = self.norm(self.hc_head(hidden_states))
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)`,
  codeNote: '主干里 mHC 的三个落点：展开（1495）→ 层内两次（1499）→ 收拢（1511）。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);
    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '主干：一行展开 → 每层两次读写 → 一行收拢' }));
    const fl = W.flow([
      { t: 'inputs_embeds', s: '(b, s, 4096)', cc: 0 },
      { t: '× 4 展开', s: 'unsqueeze(2).expand', cc: 2 },
      { t: '45 层', s: '每层 attn_hc + ffn_hc', cc: 1 },
      { t: 'hc_head', s: '4 → 1（等权平均）', cc: 5 },
      { t: 'last_hidden_state', s: '(b, s, 4096)', cc: 4 },
    ]);
    viz.appendChild(fl);

    const stats = [
      { k: 'hc_mult', v: '4', s: '残差流条数 N', cc: 2 },
      { k: 'hc_sinkhorn_iters', v: '20', s: '双随机迭代次数', cc: 1 },
      { k: 'hc_eps', v: '1e-6', s: '数值下限（防零除）', cc: 3 },
      { k: 'HC 参数 / 层', v: '786,486', s: '2 × 393,243', cc: 4 },
    ];
    const row = U.el('div', { class: 'row gap10', style: 'width:100%;align-items:stretch' });
    const statEls = stats.map(st => {
      const c = W.card({ cc: st.cc, tint: st.cc, style: 'flex:1',
        title: '<span class="mono" style="font-size:11px">' + st.k + '</span>',
        body: U.el('div', { class: 'col gap3' }, [
          U.el('div', { class: 'n big', text: st.v }),
          U.el('div', { class: 'mono faint', style: 'font-size:10px', text: st.s }),
        ]) });
      row.appendChild(c); return c;
    });
    viz.appendChild(row);

    const tb = W.table([
      ['展开（全模型 1 次）', '<span class="hl">1495</span>', '(b, s, 4096) → (b, s, <b>4</b>, 4096)'],
      ['读取 + 写回（每层 2 次）', '<span class="hl">1315 / 1342</span>', 'self.attn_hc / self.ffn_hc'],
      ['收拢（全模型 1 次）', '<span class="hl">1511</span>', 'hc_head → norm，回到 (b, s, 4096)'],
    ], { head: ['出现的位置', '源码行号', '做什么'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 5, style: 'padding:9px 11px' }, [
      U.el('div', { class: 'klabel', text: '三次出现的坐标（行号为本文件实测）' }), tb,
    ]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    fl.focus(0);
    msg.innerHTML = '<span class="cm">// 一个 token 在 GLM-5 里是「一束」，不是「一条」</span>';
    tl.at(1200, () => { fl.focus(1); msg.innerHTML = '第 1495 行：<em>(b, s, 4096)</em> <span class="op">→</span> <em>(b, s, 4, 4096)</em>，一条变一束'; });
    tl.at(4200, () => { fl.focus(2); msg.innerHTML = '层与层之间传的一直是 <em>4 维</em>张量；<span class="fn">attn_hc</span> / <span class="fn">ffn_hc</span> 各读写一次'; });
    tl.at(7400, () => { fl.focus(3); msg.innerHTML = '出口 <em>hc_head</em> 把 4 条收成 1 条 —— 它是<b>等权平均</b>，0 个参数'; });
    tl.at(10400, () => { fl.focus(4); msg.innerHTML = '收拢之后才做 <span class="fn">norm</span>：顺序反了数值就不一样'; });
    tl.at(13400, () => {
      fl.focus(-1); statEls.forEach(e => e.classList.add('ac'));
      msg.innerHTML = '<span class="cm">// 四个要记住的数：4 / 20 / 1e-6 / 786,486</span>';
    });
    tl.at(15800, () => {
      statEls.forEach(e => e.classList.remove('ac'));
      msg.innerHTML = '本课主线：<em>形状</em> → <em>读取</em> → <em>双随机</em> → <em>写回</em> → <em>两次</em> → <em>收拢</em>';
    });
  },
},

/* ------------------------------------------------- 2 形状：一条 → 一束 */
{
  kicker: '形状 · 第一步',
  title: '一行代码，<span class="hl-a">(b, s, 4096)</span> 变成 <span class="hl-c">(b, s, 4, 4096)</span>',
  sub: 'expand 不复制数据，它只是把 stride 置 0 的视图铺开；.contiguous() 才真的把它变成 4 份。',
  caption: '实测：第 0 层之后 4 条流就不再相同（最大差 2.9e-3）—— 分叉的唯一入口是 post 门控。',
  lang: 'python',
  codeStart: 1495,
  code: `@@L1495@@`,
  codeNote: '全文唯一把 hidden_states 变成 4 维的地方；这一行之后，所有张量都多一维。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);
    const viz = U.el('div', { class: 'vizgrow hstart' });
    wrap.appendChild(viz);
    FLOW.attach(U.q('#visual'));
    const lane = U.el('div', { class: 'row gap16 center', style: 'width:100%' });
    viz.appendChild(lane);

    /* 左：3 维张量 */
    const left = U.el('div', { class: 'col gap6 center' });
    left.appendChild(U.el('div', { class: 'klabel', text: 'inputs_embeds' }));
    const t1 = W.tensor(6, 8, () => -1, { cell: 26 });
    t1.querySelectorAll('.tcell').forEach(c => {
      c.style.background = 'rgba(52,211,153,.26)';
      c.style.borderColor = 'rgba(52,211,153,.42)';
    });
    left.appendChild(t1);
    left.appendChild(U.el('div', { class: 'mono dim', style: 'font-size:11.5px', text: '(batch, seq, 4096)' }));
    lane.appendChild(left);

    /* 中：箭头 */
    const mid = U.el('div', { class: 'col center gap6' });
    mid.innerHTML = '<div class="mono" style="font-size:22px;color:var(--accent)">⟹</div>'
      + '<div class="mono" style="font-size:10.5px;color:var(--c2);text-align:center">'
      + 'unsqueeze(2)<br>.expand(-1,-1,4,-1)<br>.contiguous()</div>';
    lane.appendChild(mid);

    /* 右：4 条流 */
    const right = U.el('div', { class: 'col gap5' });
    right.appendChild(U.el('div', { class: 'klabel', text: 'hidden_states —— 4 条流' }));
    const planes = [];
    for (let s = 0; s < 4; s++) {
      const prow = U.el('div', { class: 'row gap8 center' });
      prow.appendChild(U.el('div', { class: 'n sm', style: 'width:30px;color:var(--c' + (s % 6) + ')', text: '流' + s }));
      const t = W.tensor(2, 8, () => -1, { cell: 26 });
      t.querySelectorAll('.tcell').forEach(c => {
        c.style.background = 'rgba(251,191,36,.20)';
        c.style.borderColor = 'rgba(251,191,36,.34)';
        c.style.opacity = '0';
        c.style.transform = 'translateX(-16px)';
      });
      prow.appendChild(t);
      right.appendChild(prow);
      planes.push(t);
    }
    right.appendChild(U.el('div', { class: 'mono dim', style: 'font-size:11.5px', text: '(batch, seq, 4, 4096)' }));
    lane.appendChild(right);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 嵌入之后，就是这一行</span>';
    planes.forEach((p, s) => {
      const a = U.rectIn(t1), r = U.rectIn(p);
      FLOW.path('p' + s, [[a.x + a.w, a.cy], [(a.x + a.w + r.x) / 2, a.cy],
                          [(a.x + a.w + r.x) / 2, r.cy], [r.x, r.cy]]);
      FLOW.send('p' + s, { at: 3250 + s * 1500, dur: 900,
        color: ['56,189,248', '167,139,250', '251,191,36', '52,211,153'][s], label: '流' + s });
    });
    planes.forEach((p, s) => {
      tl.at(1600 + s * 1500, () => {
        p.querySelectorAll('.tcell').forEach((c, k) => {
          c.style.transition = 'opacity .4s var(--ease-out), transform .4s var(--ease-out)';
          c.style.transitionDelay = (k * 10) + 'ms';
          c.style.opacity = '1'; c.style.transform = 'none';
        });
        msg.innerHTML = '第 <em>' + (s + 1) + '</em> 条流展开 &nbsp;<span class="cm">// 初始时刻 4 条完全相同</span>';
      });
    });
    tl.at(8000, () => {
      msg.innerHTML = 'expand 返回的是 <em>stride = 0 的视图</em>（实测 data_ptr 与输入相同）；'
        + '<span class="fn">.contiguous()</span> 才真的铺成 4 份';
    });
    tl.at(11200, () => {
      msg.innerHTML = '实测：stride 从 <em>(15, 5, 0, 1)</em> 变成 <em>(60, 20, 5, 1)</em>，'
        + '共享存储 <span class="hlbad">False</span>';
    });
    tl.at(14200, () => {
      msg.innerHTML = '★ 4 条流从第 0 层写回开始分叉 —— 唯一入口是逐流的 <em>post</em> 门控（下一幕）';
    });
  },
},

/* ------------------------------------------------- 3 三步：读取 → 混合 → 写回 */
{
  kicker: '机制 · 三步',
  title: 'mHC 的三步：<span class="hl-a">读取</span> → <span class="hl-b">混合</span> → <span class="hl-c">写回</span>',
  sub: '函数内部只做完前两步（读取、混合），第三步在解码层里完成 —— 这就是它返回 post / comb 而不是最终结果的原因。',
  caption: 'pre 不在返回值里：它在第 330 行就被用掉了，变成 collapsed。',
  lang: 'python',
  codeStart: 301,
  code: `        # Flatten and norm the hidden streams
        flattened = hidden_streams.view(batch_size, seq_len, -1).float()
        flattened = self.input_norm(flattened)
        # Mix the streams together to infer the weight coefficients
        flattened = F.linear(flattened, self.fn.float())
        # Split the weight coefficients
        pre_w, post_w, comb_w = flattened.split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)

        comb_w = comb_w.view(*comb_w.shape[:-1], hc, hc)  # these are matrix weights, unlike pre or post
        comb_b = comb_b.view(hc, hc)

        # All weights are computed with a one layer perceptron. For pre and post, this is it.
        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)
        comb = torch.softmax(comb_w * comb_scale + comb_b, dim=-1) + self.hc_eps`,
  codeNote: '读取（view + input_norm + F.linear）与混合（split + 三种激活）的全部代码。',
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);
    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '一个 HyperConnection 内部（实测形状，D = 4096）' }));
    const mkStage = (n, title, cc, lines) => W.card({ cc: cc, tint: cc, num: n, title: title, style: 'flex:1',
      body: U.el('div', { class: 'col gap4', style: 'margin-top:4px' },
        lines.map(l => U.el('div', { class: 'mono', style: 'font-size:10.5px;line-height:1.55;'
          + 'color:var(--ink-dim);overflow-wrap:anywhere', html: l }))) });

    const row = U.el('div', { class: 'row gap10', style: 'width:100%;align-items:stretch' });
    const s1 = mkStage(1, '读取：4 条流 → 24 个数', 0, [
      'hidden_streams <span class="dim">(b, s, 4, D)</span>',
      '↓ <span class="hl">view(b, s, -1)</span> <span class="dim">→ (b, s, 16384)</span>',
      '↓ input_norm <span class="dim">（无权重 RMSNorm）</span>',
      '↓ F.linear(fn) <span class="dim">→ (b, s, 24)</span>',
    ]);
    const s2 = mkStage(2, '混合：24 → 三组权重', 1, [
      'split [4, 4, 16]',
      'pre <span class="dim">= sigmoid(·) + 1e-6</span> <span class="dim">(b,s,4)</span>',
      'post <span class="dim">= 2 · sigmoid(·)</span> <span class="dim">(b,s,4)</span>',
      'comb <span class="dim">= softmax(·, -1) + 1e-6</span>',
      '↓ <span class="hl3">Sinkhorn × 20</span> <span class="dim">(b,s,4,4)</span>',
    ]);
    const s3 = mkStage(3, '收拢（写回的前半）', 4, [
      'collapsed =',
      '<span class="dim">(pre.unsqueeze(-1) * streams)</span>',
      '<span class="dim">  .sum(dim=2)</span>',
      '→ <span class="hl4">(b, s, D)</span> 交给子层',
    ]);
    [s1, s2, s3].forEach((c, i) => {
      row.appendChild(c);
      if (i < 2) row.appendChild(U.el('div', { class: 'mono', style: 'flex:none;color:var(--accent);font-size:15px', text: '→' }));
    });
    viz.appendChild(row);

    viz.appendChild(W.card({ cc: 3, title: '写回：不在这个函数里',
      sub: '返回的是 post / comb / collapsed 三件套。<b>写回公式在解码层</b>（第 1337-1339 行）：'
        + '<code class="inl">新流 = post ⊙ 子层输出 + combᵀ · 旧流</code>。'
        + '因为要写回，就必须先把 post 与 comb <b>带出去</b>，而 pre 用完即弃。' }));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    s2.style.opacity = '.35'; s3.style.opacity = '.35';
    msg.innerHTML = '<span class="cm">// 三步里，前两步在 mHC 内部，第三步在解码层</span>';
    tl.at(1200, () => { s1.classList.add('ac'); msg.innerHTML = '① 读取：把 4 条流<b>首尾相接</b>成 (b, s, 16384) —— 是拼接，不是求和'; });
    tl.at(4200, () => {
      s1.classList.remove('ac'); s2.style.opacity = '1'; s2.classList.add('ac');
      msg.innerHTML = '② 混合：一次矩阵乘算出 <em>24</em> 个数 = pre(4) + post(4) + comb(4×4)';
    });
    tl.at(7400, () => {
      msg.innerHTML = '三种权重的激活<b>故意不同</b>：softmax 只保证行和为 1，列和交给 <span class="hl3">Sinkhorn</span>';
    });
    tl.at(10400, () => {
      s2.classList.remove('ac'); s3.style.opacity = '1'; s3.classList.add('ac');
      msg.innerHTML = '③ 收拢：<em>pre</em> 加权求和成 (b, s, D) —— <span class="cm">pre 用完即弃，不进返回值</span>';
    });
    tl.at(13200, () => {
      s3.classList.remove('ac');
      msg.innerHTML = '所以调用方拿到的是 <em>post</em> / <em>comb</em> / <em>collapsed</em>，写回要自己动手做';
    });
  },
},

/* ------------------------------------------------- 4 ★ Sinkhorn */
{
  kicker: '机制 · ★ 核心',
  title: '★ 双随机：<span class="hl3">20 次 Sinkhorn</span> 把 comb 压成凸组合',
  sub: 'softmax 只保证行和 = 1。列和也 = 1 之后，混合就变成「旧流的加权平均」—— 范数不增，45 层叠起来才不会爆。',
  caption: '注意最后一次操作是列归一化：所以列和更准（精确落在 1/(1+hc_eps)），行和只是逼近。',
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
  codeNote: '第 323 行先做一次列归一化，然后循环 19 次「行 → 列」；分母上的 + hc_eps 是防零除。',
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);
    const viz = U.el('div', { class: 'vizgrow hstart' });
    wrap.appendChild(viz);
    const lane = U.el('div', { class: 'row gap20 center', style: 'width:100%' });
    viz.appendChild(lane);

    const left = U.el('div', { class: 'col gap6 center' });
    left.appendChild(U.el('div', { class: 'klabel', text: 'comb —— 一个 token 的 4×4 混合矩阵' }));
    const g = numGrid(56, 10);
    left.appendChild(g);
    const cap = U.el('div', { class: 'mono dim', style: 'font-size:10.5px;text-align:center' });
    left.appendChild(cap);
    lane.appendChild(left);

    const right = U.el('div', { class: 'col gap10', style: 'flex:1' });
    const rowSum = sumBox('行和（dim = -1）—— softmax 保证它是 1', SK.A.r);
    const colSum = sumBox('列和（dim = -2）—— Sinkhorn 要修的就是它', SK.A.c);
    right.appendChild(rowSum);
    right.appendChild(colSum);
    right.appendChild(U.el('div', { class: 'card', cc: 2, style: 'padding:9px 11px' },
      [U.el('div', { class: 'mono', style: 'font-size:10.5px;line-height:1.6;color:var(--ink-dim)' },
        [U.el('div', { html: 'Mixed = In @ Comb + Out' }),
         U.el('div', { html: '<span class="cm">// 行和 &gt; 1 → 每层放大一点 → 45 层指数爆炸</span>' }),
         U.el('div', { html: '<span class="cm">// 行和 = 列和 = 1 → 凸组合，范数不增</span>' })])]));
    lane.appendChild(right);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    const paint = (st) => { g.paint(st.m); rowSum.set(st.r); colSum.set(st.c); cap.textContent = st.n; };
    paint(SK.A);
    msg.innerHTML = '<span class="cm">// 起点：softmax 出来的矩阵，行和 = 1，列和不是</span>';
    tl.at(1400, () => {
      [0, 1].forEach(i => g.hi(i, 'c', true));
      msg.innerHTML = '列和 <em>1.30 / 0.70</em> <span class="op">≠</span> 1 —— <span class="hlbad">这两列就是问题</span>';
    });
    tl.at(4600, () => {
      [0, 1].forEach(i => g.hi(i, 'c', false));
      paint(SK.B);
      msg.innerHTML = '第 323 行：每列 ÷ 列和 <span class="op">→</span> 列和变 1，<span class="hlbad">行和被带坏了</span>';
    });
    tl.at(8200, () => {
      paint(SK.C);
      msg.innerHTML = '第 1 次「行 → 列」：两边都靠近 1 —— <span class="cm">行和偏差 0.1187 → 0.0206</span>';
    });
    tl.at(11400, () => {
      paint(SK.D);
      msg.innerHTML = '第 2 次：偏差 <em>0.0046</em>；第 3 次 <em>0.0013</em> —— 前三轮就拿掉 99% 的误差';
    });
    tl.at(14400, () => {
      paint(SK.E);
      msg.innerHTML = '走满 20 轮：行和 = 列和 = <em>0.999999</em> —— 停在 1/(1 + hc_eps)，不是精确的 1';
    });
    tl.at(16800, () => {
      msg.innerHTML = '★ 双随机 = <b>列和管「不放大」</b>、<b>行和管「不丢也不多」</b>；eps 是数值下限，不是超参';
    });
  },
},

/* ------------------------------------------------- 5 手算两次迭代 */
{
  kicker: '动手 · 验收点 1',
  title: '手算两次 Sinkhorn：<span class="hl-a">盯住"除以谁"</span>',
  sub: '除以行和是 dim=-1（第 325 行），除以列和是 dim=-2（第 323 / 326 行）。来回投影，就是 Sinkhorn 的全部。',
  caption: '读数来自实测脚本；eps = 1e-6 对 4 位小数没有影响，手算时可以忽略。',
  lang: 'python',
  codeStart: 323,
  code: `        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)`,
  codeNote: '四行代码：一次列归一化 + 循环里的「行 → 列」。20 轮 = 20 次列 + 19 次行。',
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);
    const viz = U.el('div', { class: 'col gap10 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '同一个初始矩阵，两次迭代的每一步（数字为实测值）' }));
    const row = U.el('div', { class: 'row gap10 center', style: 'width:100%' });
    const steps = ['A', 'B', 'C', 'D'];
    const cells = [];
    steps.forEach((k, i) => {
      if (i > 0) row.appendChild(U.el('div', { class: 'mono', style: 'flex:none;color:var(--accent);font-size:14px', text: '→' }));
      const col = U.el('div', { class: 'col gap5 center' });
      col.appendChild(U.el('div', { class: 'klabel', style: 'text-align:center;white-space:normal', text: SK[k].t }));
      const g = numGrid(30, 8);
      g.paint(SK[k].m);
      col.appendChild(g);
      const s = U.el('div', { class: 'mono', style: 'font-size:9.5px;line-height:1.5;text-align:center;color:var(--ink-faint)' });
      s.innerHTML = '行和 ' + SK[k].r.map(v => v.toFixed(3)).join(' / ')
        + '<br>列和 ' + SK[k].c.map(v => v.toFixed(3)).join(' / ');
      col.appendChild(s);
      row.appendChild(col); cells.push(col);
    });
    viz.appendChild(row);

    const edge = U.el('div', { class: 'row gap10', style: 'width:100%;align-items:stretch' });
    edge.appendChild(W.card({ cc: 2, title: '② 为什么行和会坏',
      sub: '列归一化把每一列都除了一次，行和自然就不再是 1。<b>两个约束不能一次满足</b> —— 只能交替投影。',
      style: 'flex:1' }));
    edge.appendChild(W.card({ cc: 4, title: '③④ 收敛有多快',
      sub: '行和偏差：0.1187 → <b>0.0206</b> → <b>0.0046</b> → 0.0013 → …… → 1e-6（第 20 轮）。',
      style: 'flex:1' }));
    viz.appendChild(edge);
    viz.appendChild(U.el('div', { class: 'card', cc: 1, style: 'padding:9px 11px' }, [
      U.el('div', { class: 'mono', style: 'font-size:11px;line-height:1.6;color:var(--ink-dim)' },
        [U.el('div', { html: '手算口诀：<span class="hl">除以列和 = dim -2</span>（竖着加）&nbsp;·&nbsp;'
          + '<span class="hl2">除以行和 = dim -1</span>（横着加）' }),
         U.el('div', { html: '<span class="cm">// 循环的最后一次是「列」，所以列和更准：精确落在 1/(1+1e-6)</span>' })]),
    ]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// ① 起点：行和已经是 1（softmax 的功劳），列和是 1.30 / 0.70 / 1.00 / 1.00</span>';
    tl.at(1400, () => { msg.innerHTML = '① 列和不等于 1 → 这一步混合会把不同的旧流<b>不等权</b>地搬进新流，范数就会被放大'; });
    tl.at(4200, () => { msg.innerHTML = '② 第 323 行：每列 ÷ 自己的列和 → 列和 = 1.000，行和变成 0.881 / 1.079 / 1.020 / 1.020'; });
    tl.at(7600, () => { msg.innerHTML = '③ 第 325 + 326 行：先 ÷ 行和再 ÷ 列和 —— 行和偏差 <em>0.0206</em>，列和回到 1'; });
    tl.at(11000, () => { msg.innerHTML = '④ 再来一次：行和偏差 <em>0.0046</em> —— 每迭代一次大约砍到 <em>1/3.5</em>'; });
    tl.at(14200, () => { msg.innerHTML = '所以 20 轮不是"刚好够"，而是<b>给得宽</b>：第 4 轮之后就只剩 1e-4 量级了'; });
    tl.at(16600, () => { msg.innerHTML = '★ 手算验收：给出 ① 的矩阵，你要能算出 ② 的列和 = 1、③ 的行和偏差 ≈ 0.02'; });
  },
},

/* ------------------------------------------------- 6 写回公式 */
{
  kicker: '机制 · 写回',
  title: '写回：<span class="hl-a">新流 = post ⊙ 子层输出 + combᵀ · 旧流</span>',
  sub: '同一个子层输出被 4 个不同的 post 门控缩放 —— 这就是 4 条流分叉的唯一入口。',
  caption: '两路相加：一路是「这一层新算出来的东西」，一路是「旧的 4 条流混合」。',
  lang: 'python',
  codeStart: 1337,
  code: `        hidden_states = post.to(dtype).unsqueeze(-1) * hidden_states.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual
        )`,
  codeNote: '注意力之后的写回。同一段代码在 1346-1348 行对 FFN 原样重复一遍。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);
    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);
    FLOW.attach(U.q('#visual'));

    viz.appendChild(U.el('div', { class: 'klabel', text: '两路相加：广播相乘 + 矩阵乘' }));
    const lane = U.el('div', { class: 'row gap16 center', style: 'width:100%' });
    const leftCol = U.el('div', { class: 'col gap10', style: 'flex:1' });
    const mkBox = (cc, title, lines) => W.card({ cc: cc, tint: cc, title: title, style: 'width:100%',
      body: U.el('div', { class: 'mono', style: 'font-size:11px;line-height:1.6;color:var(--ink-dim);margin-top:3px' },
        lines.map(l => U.el('div', { html: l }))) });
    const bA = mkBox(4, '左路：post ⊙ 子层输出', [
      'post <span class="dim">(b, s, 4)</span> → unsqueeze(-1) <span class="dim">(b, s, 4, 1)</span>',
      '× 子层输出 <span class="dim">(b, s, 1, D)</span>',
      '= <span class="hl4">(b, s, 4, D)</span> <span class="cm">// 每个流一个门控</span>',
    ]);
    const bB = mkBox(1, '右路：combᵀ · 旧流', [
      'comb <span class="dim">(b, s, 4, 4)</span> → transpose(-1, -2)',
      '× residual <span class="dim">(b, s, 4, D)</span>',
      '= <span class="hl2">(b, s, 4, D)</span> <span class="cm">// 列和 ≈ 1 → 凸组合</span>',
    ]);
    leftCol.appendChild(bA); leftCol.appendChild(bB);
    lane.appendChild(leftCol);

    lane.appendChild(U.el('div', { class: 'mono', style: 'flex:none;color:var(--accent);font-size:20px', text: '⊕' }));

    const outBox = W.card({ cc: 0, tint: 0, title: '新流 hidden_states', style: 'flex:none;width:300px;align-self:center',
      body: U.el('div', { class: 'col gap6 center' }, [
        U.el('div', { class: 'n big', text: '(b, s, 4, D)' }),
        U.el('div', { class: 'mono faint', style: 'font-size:10.5px;text-align:center', text: '残差留底用的是「子层之前」的 4 条流' }),
      ]) });
    lane.appendChild(outBox);
    viz.appendChild(lane);

    const tb = W.table([
      ['post', '(b, s, 4)', '逐流门控：值域 (0, 2)，可放大可压小'],
      ['子层输出', '(b, s, D)', '注意力 / MLP 的结果，只有一条（收拢之后）'],
      ['comb', '(b, s, 4, 4)', '旧流的混合矩阵，双随机（Sinkhorn 的产物）'],
      ['residual', '(b, s, 4, D)', '子层<b>之前</b>留的底，第 1314 行的那一份'],
    ], { head: ['名字', '形状', '在这里的角色'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 3, style: 'padding:9px 11px' }, [
      U.el('div', { class: 'klabel', text: '广播的形状推演（b = batch, s = seq, D = 4096）' }), tb,
    ]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    const RA = U.rectIn(bA), RB = U.rectIn(bB), RO = U.rectIn(outBox);
    const xm = (RA.x + RA.w + RO.x) / 2;
    FLOW.path('a', [[RA.x + RA.w, RA.cy], [xm, RA.cy], [xm, RO.cy], [RO.x, RO.cy]]);
    FLOW.path('b', [[RB.x + RB.w, RB.cy], [xm, RB.cy], [xm, RO.cy], [RO.x, RO.cy]]);
    FLOW.send('a', { at: 1550, dur: 850, color: '52,211,153', label: 'post ⊙ out' });
    FLOW.send('b', { at: 4750, dur: 850, color: '167,139,250', label: 'combᵀ · old' });

    bA.style.opacity = '.35'; bB.style.opacity = '.35'; outBox.style.opacity = '.35';
    msg.innerHTML = '<span class="cm">// 写回发生在解码层，不在 mHC 内部</span>';
    tl.at(1200, () => {
      bA.style.opacity = '1'; bA.classList.add('ac');
      msg.innerHTML = '左路：<em>post</em> 逐流门控 —— 4 条流在这里分叉（同一个输出 × 4 个门控值）';
    });
    tl.at(4400, () => {
      bB.style.opacity = '1'; bB.classList.add('ac');
      msg.innerHTML = '右路：<em>combᵀ · 旧流</em> —— 列和 ≈ 1，所以是<b>凸组合</b>，范数不放大';
    });
    tl.at(7600, () => {
      outBox.style.opacity = '1'; outBox.classList.add('ac');
      msg.innerHTML = '两路相加得到新的 4 条流：<em>(b, s, 4, D)</em> —— 形状与进来时完全一致';
    });
    tl.at(10600, () => {
      bA.classList.remove('ac'); bB.classList.remove('ac');
      msg.innerHTML = '★ 若 4 条输入流相同（记作 x），则 <em>新流_i = post_i · o + (Σⱼ comb[j,i]) · x</em>'
        + ' —— 唯一与 i 有关的量就是 <span class="hl4">post_i</span>';
    });
    tl.at(13400, () => {
      msg.innerHTML = '实测吻合：第 0 层之后 4 条流最大差 <em>2.9e-3</em>，第 7 层 <em>4.6e-3</em>';
    });
    tl.at(15300, () => {
      msg.innerHTML = '<span class="cm">// 同一段三行代码在文件里出现两次 —— 下一幕看它们的位置</span>';
    });
  },
},

/* ------------------------------------------------- 7 层内两次 */
{
  kicker: '机制 · 一层之内',
  title: '一层里 mHC 出现<span class="hl-a">两次</span>：<span class="hl-a">attn_hc</span> 与 <span class="hl-b">ffn_hc</span>',
  sub: '两个独立实例、不共享参数（每层 2 × 393,243）。读取永远发生在子层之前，写回永远发生在子层之后。',
  caption: 'residual = hidden_states 出现两次（1314 / 1341）—— 每次写回都要拿「子层之前」的流。',
  lang: 'python',
  codeStart: 1295,
  code: `        self.input_layernorm = Glm5NextTextRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Glm5NextTextRMSNorm(config.hidden_size, config.rms_norm_eps)

        self.attn_hc = Glm5NextTextHyperConnection(config)
        self.ffn_hc = Glm5NextTextHyperConnection(config)`,
  codeNote: '两个实例在 __init__ 里创建；它们在 forward 里的调用点见下表。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);
    const viz = U.el('div', { class: 'col gap10 vizgrow hstart' });
    wrap.appendChild(viz);

    const chips = U.el('div', { class: 'row gap10 wrap', style: 'width:100%' });
    chips.innerHTML =
      U.chip('self.attn_hc → Glm5NextTextHyperConnection', 0) +
      U.chip('self.ffn_hc → Glm5NextTextHyperConnection', 1) +
      '<span class="chip c2">2 × 393,243 = 786,486 参数 / 层</span>' +
      '<span class="dim mono" style="margin-left:auto;font-size:10.5px">两个实例不共享参数</span>';
    viz.appendChild(chips);

    const rows = [
      ['1314', 'residual = hidden_states', '(b, s, 4, D)', '留底'],
      ['1315', 'post, comb, hidden_states = self.attn_hc(…)', '(b,s,4) / (b,s,4,4) / (b,s,D)', '<b>★ mHC 第 1 次</b>'],
      ['1317', 'self.input_layernorm(hidden_states)', '(b, s, D)', '归一化在收拢之后'],
      ['1320', 'self.self_attn(…) / indexed 分支', '(b, s, D)', '子层看到的是 3 维'],
      ['1337', 'post ⊙ out + combᵀ · residual', '(b, s, 4, D)', '写回：<b>回到 4 维</b>'],
      ['1341', 'residual = hidden_states', '(b, s, 4, D)', '再留一次底'],
      ['1342', 'post, comb, hidden_states = self.ffn_hc(…)', '(b,s,4) / (b,s,4,4) / (b,s,D)', '<b>★ mHC 第 2 次</b>'],
      ['1344', 'self.post_attention_layernorm(…) + mlp(…)', '(b, s, D)', 'FFN 子层'],
      ['1346', 'post ⊙ out + combᵀ · residual', '(b, s, 4, D)', '写回：<b>回到 4 维</b>'],
    ];
    const tb = W.table(rows, { head: ['行号', '这一行做什么', '形状', '备注'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 1, style: 'padding:9px 11px' }, [
      U.el('div', { class: 'klabel', text: '一层之内的执行顺序（Glm5NextTextDecoderLayer.forward）' }), tb,
    ]));
    const trs = Array.from(tb.querySelectorAll('tbody tr'));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    const light = (idxs) => trs.forEach((tr, i) => tr.classList.toggle('ac', idxs.indexOf(i) >= 0));
    light([]);
    msg.innerHTML = '<span class="cm">// 一层 9 行，mHC 占了 4 行（两次读 + 两次写）</span>';
    tl.at(1400, () => { light([0, 1]); msg.innerHTML = '注意力之前：留底 → <em>attn_hc</em> 读取并混合（第 1 次 mHC）'; });
    tl.at(4400, () => { light([2, 3]); msg.innerHTML = '归一化 + 注意力：<b>都只看到 3 维的那一条</b>（collapsed）'; });
    tl.at(7400, () => { light([4]); msg.innerHTML = '写回：<em>(b, s, D)</em> 的注意力输出被写成 4 条流'; });
    tl.at(10400, () => { light([5, 6]); msg.innerHTML = 'FFN 之前：再留一次底 → <em>ffn_hc</em> 读取并混合（第 2 次 mHC）'; });
    tl.at(13200, () => { light([7, 8]); msg.innerHTML = '归一化 + MLP + 写回：又是一次 3 维 ↔ 4 维的往返'; });
    tl.at(15600, () => {
      light([1, 6]);
      msg.innerHTML = '★ 验收点 2：<em>attn_hc</em> 在注意力<b>前</b>（读+混），<em>ffn_hc</em> 在 FFN <b>前</b>（读+混）；'
        + '两次写回都由同一段公式完成';
    });
  },
},

/* ------------------------------------------------- 8 出口收拢 */
{
  kicker: '机制 · 出口',
  title: '出口：<span class="hl-a">hc_head</span> 只是<b>等权平均</b>（0 参数）',
  sub: 'mean(dim=2) —— 没有权重、没有参数。真正学出来的混合在每一层的 attn_hc / ffn_hc 里。',
  caption: '订正 L0-01 第八幕的画面文字：hc_head 不是「学习一组混合权重」，源码里它是无权重平均。',
  lang: 'python',
  codeStart: 334,
  code: `class Glm5NextTextHyperHead(nn.Module):
    """Final GLM-5.3-Flash HC-stream collapse. Unlike DeepSeek-V4, this is an unweighted mean."""

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        return hidden_streams.mean(dim=2)`,
  codeNote: '整个类只有两行；文档字符串还专门写了「Unlike DeepSeek-V4, this is an unweighted mean」。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);
    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const stage = U.el('div', { class: 'row gap20 center', style: 'width:100%' });
    const inCol = streamBars(4, 220, 36);
    const inWrap = U.el('div', { class: 'col gap5 center' }, [
      U.el('div', { class: 'klabel', text: '(batch, seq, 4, 4096)' }), inCol]);
    stage.appendChild(inWrap);

    const mid = U.el('div', { class: 'col center gap6' });
    mid.innerHTML = '<div style="font-size:24px;color:var(--accent)">⟹</div>'
      + '<div class="mono" style="font-size:10.5px;color:var(--c2);text-align:center">'
      + 'hc_head<br>mean(dim=2)</div>';
    stage.appendChild(mid);

    const outBar = U.el('div', {
      style: 'width:210px;height:164px;border-radius:10px;border:1px solid var(--accent);'
        + 'background:linear-gradient(180deg,rgba(34,211,238,.30),rgba(34,211,238,.10));'
        + 'display:flex;align-items:center;justify-content:center;'
        + 'font:600 12px var(--mono);color:#c9f7ff;transition:opacity .5s var(--ease-out),transform .5s var(--ease-out)',
      text: 'last_hidden_state' });
    stage.appendChild(U.el('div', { class: 'col gap5 center' }, [
      U.el('div', { class: 'klabel', text: '(batch, seq, 4096)' }), outBar]));
    viz.appendChild(stage);

    const ev = U.el('div', { class: 'row gap10', style: 'width:100%;align-items:stretch' });
    ev.appendChild(W.card({ cc: 4, title: '实测：0 个参数', style: 'flex:1',
      sub: 'sum(p.numel() for p in h.parameters()) == <b>0</b>；'
        + 'allclose(h(s), s.mean(dim=2)) == <b>True</b>。' }));
    ev.appendChild(W.card({ cc: 3, title: '订正 L0-01', style: 'flex:1',
      sub: 'L0-01 第八幕写的是「学习一组混合权重」—— 按源码应订正为<b>等权平均</b>；'
        + '它的结论（4 → 1）不变。' }));
    ev.appendChild(W.card({ cc: 2, title: '为什么可以不学', style: 'flex:1',
      sub: '4 条流已被 Sinkhorn 约束在同一尺度上（凸组合不放大范数），出口不必再学一次加权。' }));
    viz.appendChild(ev);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    outBar.style.opacity = '0'; outBar.style.transform = 'scale(.9)';
    msg.innerHTML = '<span class="cm">// 45 层跑完，手上还是 4 条流</span>';
    tl.at(1400, () => { msg.innerHTML = '出口第一件事：<em>hidden_streams.mean(dim=2)</em> —— 沿流那一维取平均'; });
    tl.at(4400, () => {
      inCol.bars.forEach(b => { b.style.opacity = '.35'; b.style.transform = 'translateX(10px)'; });
      outBar.style.opacity = '1'; outBar.style.transform = 'none';
      msg.innerHTML = '4 条 <span class="op">→</span> <em>1</em> 条：<b>没有权重，也没有参数</b>';
    });
    tl.at(7400, () => {
      inCol.bars.forEach(b => { b.style.opacity = '1'; b.style.transform = 'none'; });
      msg.innerHTML = '然后才是 <span class="fn">norm</span>：<code class="inl">hidden_states = self.norm(self.hc_head(hidden_states))</code>';
    });
    tl.at(10400, () => {
      msg.innerHTML = '顺序不能反：先收拢再归一化 = 对平均后的向量归一化；反过来是「每条流各自归一化再平均」，数值不同';
    });
    tl.at(13400, () => {
      msg.innerHTML = '★ 真正「学出来的混合」在每层的 <em>attn_hc</em> / <em>ffn_hc</em>（35.4M 参数），不在出口';
    });
    tl.at(15400, () => {
      msg.innerHTML = '<span class="cm">// 回到主干第 1511 行 —— 出口只有这一行</span>';
    });
  },
},

/* ------------------------------------------------- 9 收束 */
{
  kicker: '收束',
  title: '把 mHC 压成一张表',
  sub: '三次出现、三步机制、一个公式、两个验收点。下一课 L1-01 回到 config，看这些数字从哪里来。',
  caption: '下一课 L1-01：PreTrainedConfig —— 配置对象的公共契约（以及 hc_mult 住在哪个嵌套 config 里）。',
  lang: 'python',
  codeStart: 1511,
  code: `        hidden_states = self.norm(self.hc_head(hidden_states))
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)`,
  codeNote: '回到主干出口：现在这一行里的每个名字你都能说出它属于哪一步。',
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);
    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const cards = [
      { n: '①', t: '展开', c: 2, d: '(b,s,4096) → (b,s,4,4096)', l: '1495' },
      { n: '②', t: '层内读 + 写', c: 1, d: 'attn_hc / ffn_hc 各一次', l: '1315 / 1342' },
      { n: '③', t: '收拢', c: 4, d: 'hc_head 等权平均 → norm', l: '1511' },
    ];
    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const els = cards.map(c => {
      const e = W.card({ cc: c.c, tint: c.c, style: 'flex:1',
        title: '<span class="mono" style="font-size:11px">' + c.n + ' ' + c.t + '</span>',
        body: U.el('div', { class: 'col gap4' }, [
          U.el('div', { class: 'mono', style: 'font-size:11px;color:var(--ink-dim)', text: c.d }),
          U.el('div', { class: 'mono faint', style: 'font-size:10px', text: 'L' + c.l }),
        ]) });
      row.appendChild(e); return e;
    });
    viz.appendChild(row);

    const tb = W.table([
      ['残差结构', '<span class="hl">4 条流</span>', 'hc_mult = 4，主干里 hidden_states 恒为 4 维'],
      ['写回公式', '<span class="hl">post ⊙ out + combᵀ · 旧流</span>', '同一个三行，在层内出现两次'],
      ['双随机', '<span class="hl">20 次 Sinkhorn</span>', '行和 = 列和 ≈ 0.999999（差 1e-6，来自 hc_eps）'],
      ['为什么不会爆', '<span class="hl">凸组合</span>', '列和 = 1 → 新流是旧流的加权平均，范数不增'],
      ['代价', '<span class="hl">35,391,870</span>', '45 层 × 2 个 HC × 393,243 参数'],
      ['出口', '<span class="hl">mean(dim=2)</span>', '0 参数；不是加权求和'],
    ], { head: ['结论', '数字 / 公式', '说明'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 0, style: 'padding:9px 11px' }, [
      U.el('div', { class: 'klabel', text: '本课全部结论（数字均为实测）' }), tb,
    ]));

    viz.appendChild(W.exercise(
      '给定初始 comb（softmax 之后）：<code class="inl">[[0.70,0.10,0.10,0.10],'
      + '[0.40,0.40,0.10,0.10],[0.10,0.10,0.70,0.10],[0.10,0.10,0.10,0.70]]</code>。'
      + '手算 <b>两次</b> Sinkhorn 迭代，写出每一步的行和与列和。',
      '① 初始：行和 = 1.000（softmax 保证）；列和 = <b>1.30 / 0.70 / 1.00 / 1.00</b>。<br>'
      + '② 第 323 行列归一化（每列 ÷ 列和）：列和 = <b>1.000</b>；行和 = 0.881 / 1.079 / 1.020 / 1.020。<br>'
      + '③ 第 1 次（÷ 行和，再 ÷ 列和）：行和 = 0.979 / 1.004 / 1.008 / 1.008，偏差 <b>0.0206</b>。<br>'
      + '④ 第 2 次：行和偏差 <b>0.0046</b>，矩阵收敛到 '
      + '<code class="inl">[[0.5900,0.1705,0.1175,0.1175],…]</code>。<br>'
      + '走满 20 轮：行和 = 列和 = <b>0.999999</b> —— 停在 1/(1+hc_eps) 而不是精确 1。'
      + '<br>关键：<b>最后一次除的是列和</b>，所以列和比行和更准。'));

    viz.appendChild(W.exercise(
      'mHC 在解码层里出现在哪两个位置？各自做什么？写回公式在谁手里？',
      '<b>attn_hc</b>（第 1315 行）在<b>注意力之前</b>：读取 4 条流 → 算出 pre/post/comb → '
      + '用 pre 收拢成 (b,s,D) 交给注意力。'
      + '<b>ffn_hc</b>（第 1342 行）在 <b>FFN 之前</b>：同样的三步，作用在注意力写回后的流上。<br>'
      + '<b>写回不在 mHC 内部</b>：它由解码层第 1337-1339 / 1346-1348 行的公式完成 —— '
      + '<code class="inl">新流 = post ⊙ 子层输出 + combᵀ · 旧流</code>，'
      + '两处用的是同一段代码（这就是「前后各一次」的字面证据）。'));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    els.forEach(e => { e.style.opacity = '.3'; });
    msg.innerHTML = '<span class="cm">// 三次出现，逐个点亮</span>';
    cards.forEach((c, i) => {
      tl.at(1400 + i * 2200, () => {
        els.forEach((e, k) => { e.style.opacity = (k === i) ? '1' : '.3'; });
        msg.innerHTML = '<em>' + c.n + '</em> ' + c.t + ' &nbsp;<span class="cm">// 源码第 ' + c.l + ' 行</span>';
      });
    });
    tl.at(9000, () => {
      els.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = '4 条流 + 双随机混合 + 两次读写 = mHC 的全部；写回公式是唯一需要背下来的东西';
    });
    tl.at(12000, () => {
      msg.innerHTML = '★ 验收点：能手算 2 次 Sinkhorn（列和 → 行和 → 列和），能说出两次 mHC 的位置与作用';
    });
    tl.at(15000, () => {
      msg.innerHTML = '为什么难训但扩展性好：梯度有 4 条通路（更难优化），流间耦合被双随机限制住（不会爆炸）';
    });
    tl.at(17800, () => {
      msg.innerHTML = '<span class="cm">// 下一课 L1-01：这些数字全部写在一个 config 里 —— hc_mult / hc_eps / hc_sinkhorn_iters</span>';
    });
  },
},

];
