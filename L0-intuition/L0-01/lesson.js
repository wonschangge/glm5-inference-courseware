/* ==========================================================================
   L0-01 · 一次 generate() 里发生了什么
   --------------------------------------------------------------------------
   覆盖：models/glm5_next/modeling_glm5_next.py
   目标：看完能把 45 层的 GLM-5 压缩成一张图，并说出每个阶段对应哪段代码。
   排版基准：#visual 可视区 = 994 x 662.7 舞台像素（无头浏览器实测）。
   ========================================================================== */
'use strict';

/* 实测常量（由 _data/recon/probe1.py 与 probe2.py 跑出，不是估算） */
const M = {
  layers: 45, kda: 34, mla: 11,
  dense: 3, sparse: 42,
  hidden: 4096, heads: 64, vocab: 154880,
  experts: 288, topk: 8, shared: 1, streams: 4,
};

const SCENES = [

/* ------------------------------------------------- 1 全景：六个阶段 */
{
  kicker: '预备篇 · 直觉',
  title: '一次前向，<span class="hl-a">六个阶段</span>',
  sub: '不看任何优化。先建立"这个模型长什么样"的整体直觉：数据从哪进、经过什么、从哪出。',
  caption: '这张图是后面 69 课的目录 —— 每一块都会有一课专门拆开。',
  lang: 'python',
  codeStart: 1449,
  code: `    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:`,
  codeNote: 'Glm5NextTextModel.forward —— 整条前向的唯一入口。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    /* ---- 主流程线 ---- */
    const nodes = [
      { t: 'input_ids', s: '(batch, seq)', cc: 0 },
      { t: 'embed_tokens', s: '词表 → 4096', cc: 4 },
      { t: '×4 流', s: 'mHC 展开', cc: 2 },
      { t: '45 层', s: 'KDA 34 + MLA 11', cc: 1 },
      { t: 'logits', s: ' (batch, seq, 154880)', cc: 3 },
    ];
    const fl = W.flow(nodes, { style: 'width:100%' });
    wrap.appendChild(U.el('div', { class: 'col gap8', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: '前向主链路' }), fl,
    ]));

    /* ---- 关键数字 ---- */
    const stats = [
      { k: '隐藏维', v: M.hidden, cc: 4, s: 'hidden_size' },
      { k: '层数', v: M.layers, cc: 1, s: 'num_hidden_layers' },
      { k: '残差流', v: M.streams, cc: 2, s: 'hc_mult' },
      { k: '词表', v: M.vocab, cc: 3, s: 'vocab_size' },
    ];
    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const statEls = stats.map(st => {
      const c = W.card({ cc: st.cc, tint: st.cc,
        title: `<span class="mono" style="font-size:11px">${st.s}</span>`,
        body: U.el('div', { class: 'n big', text: U.fmt(st.v) }),
        style: 'flex:1' });
      row.appendChild(c); return c;
    });
    wrap.appendChild(row);

    /* ---- prefill / decode：同一个函数 ---- */
    const cmp = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cA = W.card({ cc: 0, title: '① prefill',
      sub: '<code class="inl">past_key_values = None</code> → 在 forward 内部 new 一个 DynamicCache',
      body: U.el('div', { class: 'mono', style: 'font-size:11.5px;color:var(--c0)', text: 'seq = 你的整个 prompt' }) });
    const cB = W.card({ cc: 5, title: '② decode（第 2 次起）',
      sub: '<code class="inl">past_key_values</code> 已有内容 → 只算新来的那 1 个 token',
      body: U.el('div', { class: 'mono', style: 'font-size:11.5px;color:var(--c5)', text: 'seq = 1，代价与轮次无关' }) });
    cA.style.flex = '1'; cB.style.flex = '1';
    cmp.appendChild(cA); cmp.appendChild(cB);
    wrap.appendChild(cmp);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    /* ---- 时间轴 ---- */
    fl.focus(0);
    msg.innerHTML = '<span class="cm">// 一个 token 的旅程从这里开始</span>';
    tl.at(1800, () => { fl.focus(1); msg.innerHTML = '嵌入：<em>(batch, seq)</em> <span class="op">→</span> <em>(batch, seq, 4096)</em>'; });
    tl.at(4600, () => { fl.focus(2); msg.innerHTML = '★ 展开成 <em>4 条残差流</em> <span class="op">→</span> <em>(batch, seq, 4, 4096)</em>'; });
    tl.at(7800, () => { fl.focus(3); msg.innerHTML = '<em>45</em> 层：<span class="fn">34</span> 层走 KDA、<span class="fn">11</span> 层走 MLA + DSA'; });
    tl.at(10800, () => { fl.focus(4); msg.innerHTML = '<span class="fn">lm_head</span> 投到词表 <span class="op">→</span> <em>(batch, seq, 154880)</em> 的 logits'; });
    tl.at(13200, () => {
      fl.focus(-1); statEls.forEach(e => e.classList.add('ac'));
      msg.innerHTML = '<span class="cm">// 记住这四个数：4096 / 45 / 4 / 154880</span>';
    });
    tl.at(15200, () => {
      statEls.forEach(e => e.classList.remove('ac'));
      cA.classList.add('ac'); cB.classList.remove('ac');
      msg.innerHTML = 'prefill 与 decode <em>走同一个 forward</em>，区别只在有没有传 past_key_values';
    });
  },
},

/* ------------------------------------------------- 2 入口：二选一 */
{
  kicker: '阶段 ① · 入口',
  title: '为什么 <span class="hl-a">input_ids</span> 和 <span class="hl-b">inputs_embeds</span> 只能给一个',
  sub: '纯文本走 input_ids；多模态时视觉 token 已经在外面被换成了向量，所以走 inputs_embeds。',
  caption: '异或 ^ 是「恰好一个」的惯用写法：两个都给、或两个都不给，都会抛错。',
  lang: 'python',
  codeStart: 1459,
  code: `        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)`,
  codeNote: '校验 → 建缓存 → 嵌入，三件事在 6 行里做完。',
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    wrap.appendChild(U.el('div', { class: 'klabel', text: '异或真值表 —— 只有中间两行能通过' }));
    const tb = W.table([
      ['✓', '<code class="inl">ids=None</code>', '<code class="inl">embeds=None</code>', '<span class="hlbad">两个都没给</span>', 'raise'],
      ['✓', '<code class="inl">ids=None</code>', '<code class="inl">embeds=给定</code>', '<span class="hl4">多模态路径</span>', '直接用'],
      ['✓', '<code class="inl">ids=给定</code>', '<code class="inl">embeds=None</code>', '<span class="hl4">纯文本路径</span>', 'embed_tokens'],
      ['✓', '<code class="inl">ids=给定</code>', '<code class="inl">embeds=给定</code>', '<span class="hlbad">两个都给了</span>', 'raise'],
    ], { head: ['', 'input_ids', 'inputs_embeds', '含义', '结果'] });
    wrap.appendChild(U.el('div', { class: 'card', cc: 0, style: 'padding:10px 12px' }, [tb]));

    /* 两条路径图 */
    const lr = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const p1 = W.card({ cc: 4, tint: 4, title: '纯文本',
      body: U.el('div', { class: 'mono', style: 'font-size:11.5px;line-height:1.7' },
        [U.el('div', { html: 'input_ids &nbsp;<span class="dim">(1, 7)</span>' }),
         U.el('div', { class: 'dim', html: '↓ embed_tokens' }),
         U.el('div', { html: 'inputs_embeds &nbsp;<span class="dim">(1, 7, 4096)</span>' })]) });
    const p2 = W.card({ cc: 1, tint: 1, title: '多模态',
      body: U.el('div', { class: 'mono', style: 'font-size:11.5px;line-height:1.7' },
        [U.el('div', { html: 'input_ids &nbsp;<span class="dim">含 image_token_id</span>' }),
         U.el('div', { class: 'dim', html: '↓ 外部替换视觉占位符' }),
         U.el('div', { html: 'inputs_embeds &nbsp;<span class="dim">已拼好</span>' })]) });
    p1.style.flex = '1'; p2.style.flex = '1';
    lr.appendChild(p1); lr.appendChild(p2);
    wrap.appendChild(lr);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    p1.style.opacity = '.3'; p2.style.opacity = '.3';
    msg.innerHTML = '<span class="cm">// 两条路在 forward 里合流，后面完全一样</span>';
    tl.at(2600, () => { p1.style.opacity = '1'; msg.innerHTML = '纯文本：<em>embed_tokens</em> 把整数 id 查表成向量'; });
    tl.at(6200, () => { p2.style.opacity = '1'; msg.innerHTML = '多模态：<em>inputs_embeds 已经是向量</em>，直接跳过 embedding 层'; });
    tl.at(10200, () => {
      p1.style.opacity = '1'; p2.style.opacity = '1';
      msg.innerHTML = '<span class="cm">// 缓存也是在这里建的，不是在外面 —— prefill 与 decode 共用一个入口</span>';
    });
  },
},

/* ------------------------------------------------- 3 位置编号 */
{
  kicker: '阶段 ② · 位置',
  title: 'position_ids：<span class="hl-a">第几个 token</span> 的唯一来源',
  sub: 'decode 时 inputs_embeds 只有 1 列，但位置编号必须接着 prefill 往下数 —— past_seen 就是那个偏移量。',
  caption: '注意：位置编号在这里只用于「哪些位置可见」，不会变成 RoPE 的旋转角 —— DSA 层是 NoPE。',
  lang: 'python',
  codeStart: 1468,
  code: `        if position_ids is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
            position_ids = position_ids.unsqueeze(0)`,
  codeNote: '第 2 行的 get_seq_length() 是「已经缓存了多少 token」。',
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const mk = (name, cc, n, showPast) => {
      const box = W.card({ cc: cc, title: name, style: 'flex:1' });
      const strip = U.el('div', { class: 'row gap4', style: 'width:100%;margin:8px 0' });
      const cells = [];
      for (let i = 0; i < 8; i++) {
        const on = i < n;
        const e = U.el('div', {
          class: 'n sm',
          style: 'flex:1;height:46px;border-radius:7px;display:flex;align-items:center;'
               + 'justify-content:center;border:1px solid ' + (on ? 'var(--c' + cc + ')' : 'rgba(150,180,255,.14)')
               + ';background:' + (on ? 'rgba(150,180,255,.13)' : 'transparent')
               + ';color:' + (on ? '#fff' : 'var(--ink-faint)'),
          text: on ? String(i) : '·' });
        strip.appendChild(e); cells.push(e);
      }
      box.appendChild(strip);
      const out = U.el('div', { class: 'mono', style: 'font-size:11.5px;color:var(--ink-dim)' });
      box.appendChild(out);
      box._cells = cells; box._out = out; box._showPast = showPast;
      return box;
    };

    const row = U.el('div', { class: 'row gap12 vizgrow', style: 'width:100%;align-items:stretch' });
    const A = mk('prefill：past_seen = 0', 0, 7, false);
    const B = mk('decode：past_seen = 7', 5, 8, true);
    A.style.display = 'flex'; A.style.flexDirection = 'column'; A.style.justifyContent = 'center';
    B.style.display = 'flex'; B.style.flexDirection = 'column'; B.style.justifyContent = 'center';
    row.appendChild(A); row.appendChild(B);
    wrap.appendChild(row);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    const setTxt = (box, seq, past) => {
      box._out.innerHTML = 'shape[1] = <em>' + seq + '</em> &nbsp; past_seen = <em>' + past + '</em>'
        + ' &nbsp;→&nbsp; position_ids = <span class="hl">' + (seq === 1 ? '[[' + past + ']]' : '[' + Array.from({length: seq}, (_, i) => i + past).join(', ') + ']') + '</span>';
    };
    setTxt(A, 7, 0); setTxt(B, 0, 0);
    B._out.innerHTML = '<span class="cm">// 还没开始 decode</span>';

    msg.innerHTML = '<span class="cm">// prefill：一次算完整个 prompt</span>';
    tl.at(3000, () => { msg.innerHTML = '输入 <em>7</em> 个 token，past_seen=<em>0</em> → 位置编号 <em>0..6</em>'; });
    tl.at(6500, () => {
      B._cells[7].style.borderColor = 'var(--c5)';
      B._cells[7].style.background = 'rgba(244,114,182,.18)';
      B._cells[7].style.color = '#fff'; B._cells[7].textContent = '7';
      setTxt(B, 1, 7);
      msg.innerHTML = 'decode：来 <em>1</em> 个 token，past_seen=<em>7</em> → 位置编号 <em>7</em>';
    });
    tl.at(10500, () => {
      msg.innerHTML = '<span class="cm">// past_seen 就是 KV cache 的长度 —— 这是它与 L5 的接口</span>';
      A.classList.add('ac');
    });
    tl.at(13000, () => { A.classList.remove('ac'); B.classList.add('ac'); });
  },
},

/* ------------------------------------------------- 4 ★ 掩码按层取 */
{
  kicker: '阶段 ③ · 排班',
  title: '★ 同一份 hidden，在 <span class="hl-a">两层上看到的掩码不同</span>',
  sub: 'causal_mask_mapping 只有两个键，但 45 层按 layer_types[i] 各取各的 —— 这张表本身就是「混合架构」的化石证据。',
  caption: '如果只有一种注意力，就不需要这张表。展开见 L0-03 与 L4。',
  lang: 'python',
  codeStart: 1490,
  code: `            causal_mask_mapping = {
                "indexed_attention": attention_mask,
                "linear_attention": attention_mask,
            }`,
  codeNote: '表面上两个键指向同一个张量；消费方式完全不同。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '两条互相独立的排班轴（都是实测值）' }));
    /* 两个条带共用列宽 —— 一眼看出两条轴的周期不同、互不对齐 */
    const mkStrip = (label, classify, sink) => {
      const col = U.el('div', { class: 'col gap6', style: 'width:100%' });
      col.appendChild(U.el('div', { class: 'klabel', text: label }));
      const row = U.el('div', { class: 'row gap4', style: 'width:100%' });
      for (let i = 0; i < 45; i++) {
        const c = classify(i);
        const e = U.el('div', {
          class: 'n sm',
          style: 'flex:1 1 0;min-width:14px;height:54px;border-radius:8px;display:flex;'
               + 'align-items:center;justify-content:center;font-size:9.5px;'
               + 'border:1px solid ' + c.brd + ';background:' + c.bg + ';color:' + c.fg,
          title: 'layer ' + i + ' → ' + c.name,
          text: String(i) });
        row.appendChild(e); sink.push(e);
      }
      col.appendChild(row);
      return col;
    };
    const cellsA = [], cellsB = [];
    viz.appendChild(mkStrip('注意力轴 · config.layer_types', i => (i % 4 === 3)
      ? { name: 'indexed_attention (MLA+DSA)', brd: 'var(--c1)', bg: 'rgba(167,139,250,.20)', fg: '#ddd6fe' }
      : { name: 'linear_attention (KDA)', brd: 'var(--c0)', bg: 'rgba(56,189,248,.13)', fg: '#bae6fd' }, cellsA));
    viz.appendChild(mkStrip('MLP 轴 · config.mlp_layer_types', i => (i < 3)
      ? { name: 'dense (Glm5NextTextMLP)', brd: 'var(--c4)', bg: 'rgba(52,211,153,.18)', fg: '#a7f3d0' }
      : { name: 'sparse (Glm5NextTextMoE)', brd: 'var(--c2)', bg: 'rgba(251,191,36,.15)', fg: '#fde68a' }, cellsB));
    const cells = cellsA;

    const legend = U.el('div', { class: 'row gap12 wrap', style: 'width:100%' });
    legend.innerHTML =
      '<span class="chip c0"><i class="sw sw-c0"></i>KDA · 34 层</span>' +
      '<span class="chip c1"><i class="sw sw-c1"></i>MLA+DSA · 11 层</span>' +
      '<span class="chip c4"><i class="sw sw-c4"></i>稠密 MLP · 3 层</span>' +
      '<span class="chip c2"><i class="sw sw-c2"></i>MoE · 42 层</span>' +
      '<span class="dim mono" style="margin-left:auto;font-size:11px">周期 4 : 1 → 互不对齐</span>';
    viz.appendChild(legend);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 主干循环里这一行决定每层用哪张掩码</span>';
    const codeLine = U.el('div', { class: 'mono', style: 'font-size:12px;color:var(--ink-dim);padding:2px 0' });
    codeLine.innerHTML = 'attention_mask=<span class="hl">causal_mask_mapping</span>[<span class="hl2">self.config.layer_types[i]</span>]';
    wrap.appendChild(codeLine);

    const pick = (i) => {
      /* 同时点亮两条轴上的第 i 列 —— 强调「同一层，两个独立的排班决定」 */
      [cellsA, cellsB].forEach(arr => arr.forEach((c, k) => {
        c.style.opacity = (k === i) ? '1' : '.34';
        c.style.transform = (k === i) ? 'translateY(-4px) scale(1.22)' : 'none';
        c.style.zIndex = (k === i) ? '3' : '1';
      }));
      const mla = (i % 4 === 3);
      msg.innerHTML = 'layer <em>' + i + '</em> → <span class="' + (mla ? 'hl2' : 'hl') + '">'
        + (mla ? 'indexed_attention' : 'linear_attention') + '</span>'
        + (mla ? ' &nbsp;<span class="cm">// 再叠一层「只看选中的 2048 个位置」</span>'
               : ' &nbsp;<span class="cm">// 只关心哪些位置是 padding</span>');
    };
    tl.at(1200, () => pick(0));
    tl.at(4400, () => pick(3));
    tl.at(7600, () => pick(7));
    tl.at(10800, () => {
      [cellsA, cellsB].forEach(arr => arr.forEach(c => { c.style.opacity = '1'; c.style.transform = 'none'; }));
      msg.innerHTML = '<em>34 : 11</em> &nbsp;—— 长上下文服务成本结构决定的排布（L0-03 展开）';
    });
    tl.at(14000, () => {
      msg.innerHTML = '<span class="cm">// 两个键指向同一个张量，但消费方式不同 —— 这张表就是混合架构的证据</span>';
    });
  },
},

/* ------------------------------------------------- 5 ★ 4 条残差流 */
{
  kicker: '阶段 ④ · 形状',
  title: '★ hidden_states 从一开始就是 <span class="hl-c">4 维</span>的',
  sub: '全文件里最容易被略过、影响最大的一行。多出来的那一维就是 mHC 的 4 条残差流。',
  caption: '这意味着这个模型里根本没有「一条残差」这回事。展开见 L3-06。',
  lang: 'python',
  codeStart: 1495,
  code: `        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()`,
  codeNote: 'hc_mult = 4。这一行之后，所有张量都多一维。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const stage = U.el('div', { class: 'row gap20 center', style: 'width:100%;flex:1' });
    wrap.appendChild(stage);

    /* 左：3D 张量 (seq, hidden) */
    const left = U.el('div', { class: 'col gap8 center', style: 'flex:none' });
    left.appendChild(U.el('div', { class: 'klabel', text: 'inputs_embeds' }));
    const t1 = W.tensor(6, 10, () => -1, { cell: 27 });
    t1.querySelectorAll('.tcell').forEach(c => { c.style.background = 'rgba(52,211,153,.28)'; c.style.borderColor = 'rgba(52,211,153,.4)'; });
    left.appendChild(t1);
    left.appendChild(U.el('div', { class: 'mono dim', style: 'font-size:11.5px', text: '(batch, seq, 4096)' }));
    stage.appendChild(left);

    const arrow = U.el('div', { class: 'col center gap6', style: 'flex:none' });
    arrow.innerHTML = '<div class="mono" style="font-size:22px;color:var(--accent)">⟹</div>'
      + '<div class="mono" style="font-size:10.5px;color:var(--c2)">unsqueeze(2)<br>.expand(hc_mult)</div>';
    stage.appendChild(arrow);

    /* 右：4 层叠放的流 */
    const right = U.el('div', { class: 'col gap6', style: 'flex:none' });
    right.appendChild(U.el('div', { class: 'klabel', text: 'hidden_states —— 4 条流' }));
    const planes = [];
    for (let s = 0; s < 4; s++) {
      const prow = U.el('div', { class: 'row gap8 center' });
      prow.appendChild(U.el('div', { class: 'n sm', style: 'width:26px;color:var(--c' + (s % 6) + ')', text: '流' + s }));
      const t = W.tensor(2, 10, () => -1, { cell: 27 });
      t.querySelectorAll('.tcell').forEach(c => {
        c.style.background = 'rgba(251,191,36,.20)';
        c.style.borderColor = 'rgba(251,191,36,.34)';
        c.style.transition = 'opacity .45s cubic-bezier(.16,1,.3,1), transform .45s cubic-bezier(.16,1,.3,1)';
        c.style.opacity = '0'; c.style.transform = 'translateX(-18px)';
      });
      prow.appendChild(t);
      right.appendChild(prow);
      planes.push(t);
    }
    right.appendChild(U.el('div', { class: 'mono dim', style: 'font-size:11.5px', text: '(batch, seq, 4, 4096)' }));
    stage.appendChild(right);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 嵌入之后，就是这一行</span>';

    planes.forEach((p, s) => {
      tl.at(2200 + s * 1500, () => {
        p.querySelectorAll('.tcell').forEach((c, k) => {
          c.style.transitionDelay = (k * 12) + 'ms';
          c.style.opacity = '1'; c.style.transform = 'none';
        });
        msg.innerHTML = '第 <em>' + (s + 1) + '</em> 条残差流展开 &nbsp;<span class="cm">// 完全相同的初始拷贝</span>';
      });
    });

    tl.at(9000, () => {
      msg.innerHTML = '公式读作：<span class="fn">新流</span> = <em>post</em> ⊙ 子层输出 + <em>combᵀ</em> · 旧流';
    });
    tl.at(12500, () => {
      msg.innerHTML = '<span class="cm">// post 门控「写多少」，comb 决定「4 条旧流怎么混合」</span>';
    });
    tl.at(15000, () => {
      msg.innerHTML = '★ 看到 hidden 是 4 维，你才不会被 L3-06 的 post / comb 搞晕';
    });
  },
},

/* ------------------------------------------------- 6 45 层循环 */
{
  kicker: '阶段 ⑤ · 循环',
  title: '45 层循环：<span class="hl-a">topk_indices</span> 在层间传递',
  sub: 'DSA 的 shared 模式：上一层选好的 2048 个位置，这一层直接复用，省掉一次索引器前向。',
  caption: 'position_embeddings=None 旁边那句注释，是架构作者自己留的记号。',
  lang: 'python',
  codeStart: 1497,
  code: `        topk_indices = None
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
            )`,
  codeNote: '注意 prev_topk_indices=topk_indices —— 上一层的输出就是这一层的输入。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    /* 层带：把 45 层压成一段，标出 MLA 层 */
    const strip = U.el('div', { class: 'row gap3', style: 'width:100%;height:40px' });
    const cells = [];
    for (let i = 0; i < 45; i++) {
      const mla = (i % 4 === 3);
      const e = U.el('div', {
        style: 'flex:1 1 0;border-radius:5px;border:1px solid '
             + (mla ? 'var(--c1)' : 'rgba(56,189,248,.55)') + ';background:'
             + (mla ? 'rgba(167,139,250,.22)' : 'rgba(56,189,248,.10)')
             + ';transition:box-shadow .3s,transform .3s',
        title: 'layer ' + i });
      strip.appendChild(e); cells.push(e);
    }
    wrap.appendChild(strip);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    /* topk 传递示意 */
    const tkRow = U.el('div', { class: 'row gap10 center', style: 'width:100%' });
    tkRow.innerHTML =
      '<span class="chip c0">L2 KDA</span><span class="arrow" style="max-width:34px"></span>' +
      '<span class="chip c1">L3 MLA · indexer=full</span><span class="arrow" style="max-width:34px"></span>' +
      '<span class="chip c0">L4 KDA</span><span class="arrow" style="max-width:34px"></span>' +
      '<span class="chip c0">L5 KDA</span><span class="arrow" style="max-width:34px"></span>' +
      '<span class="chip c1">L6 KDA</span><span class="arrow" style="max-width:34px"></span>' +
      '<span class="chip c1">L7 MLA · <span style="color:var(--c2)">shared</span></span>';
    wrap.appendChild(U.el('div', { class: 'col gap6', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: 'topk_indices 的传递路径（idx % 4 == 3 的层才产生它）' }),
      U.el('div', { class: 'card', cc: 1, style: 'padding:9px 11px' }, [tkRow]),
    ]));

    const note = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    note.appendChild(W.card({ cc: 4, title: 'position_embeddings=None',
      sub: 'NoPE。位置信息交给 KDA 的递推与 DSA 的索引器，注意力本身不做旋转。', style: 'flex:1' }));
    note.appendChild(W.card({ cc: 2, title: '共享索引器省下了什么',
      sub: '索引器本身是一次小注意力（32 头 × 128 维）；复用上一层的选择，等于白拿一层。', style: 'flex:1' }));
    wrap.appendChild(note);

    msg.innerHTML = '<span class="cm">// 45 次迭代，每次换一层</span>';
    const light = (i) => {
      cells.forEach((c, k) => {
        c.style.boxShadow = (k === i) ? '0 0 0 2px var(--accent), 0 0 14px -2px var(--accent)' : 'none';
        c.style.transform = (k === i) ? 'scaleY(1.28)' : 'none';
      });
    };
    const marks = [0, 3, 7, 11, 23, 44];
    marks.forEach((li, k) => {
      tl.at(1800 + k * 2100, () => {
        light(li);
        const mla = (li % 4 === 3);
        msg.innerHTML = 'layer <em>' + li + '</em> &nbsp;→&nbsp; '
          + (mla ? '<span class="hl2">MLA + DSA</span>，产出 topk_indices <span class="cm">// indexer=full</span>'
                 : '<span class="hl">KDA</span>，直接消费上一层的 topk_indices');
      });
    });
    tl.at(15200, () => {
      cells.forEach(c => { c.style.boxShadow = 'none'; c.style.transform = 'none'; });
      msg.innerHTML = '<span class="cm">// 循环结束后还剩两步，见下一幕</span>';
    });
  },
},

/* ------------------------------------------------- 7 层内二选一 */
{
  kicker: '阶段 ⑥ · 层内',
  title: '一层之内：注意力和 MLP 都是<span class="hl-a">二选一</span>',
  sub: '「GLM-5 是 MoE 模型」和「GLM-5 是线性注意力模型」这两句话都不完整 —— 它是按层排班的。',
  caption: '两个选择互相独立：前 3 层是「KDA 或 MLA」×「稠密」，第 4 层是「MLA × 稀疏」。',
  lang: 'python',
  codeStart: 1283,
  code: `        self.block_type = config.layer_types[layer_idx]
        self.hidden_size = config.hidden_size
        self.self_attn = (
            Glm5NextTextLinearAttention(config, layer_idx)
            if self.block_type == "linear_attention"
            else Glm5NextTextAttention(config, layer_idx)
        )

        self.mlp = (
            Glm5NextTextMoE(config) if config.mlp_layer_types[layer_idx] == "sparse" else Glm5NextTextMLP(config)
        )`,
  codeNote: 'Glm5NextTextDecoderLayer.__init__ —— 混合架构在这里落地。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    /* 2x2 组合矩阵 */
    wrap.appendChild(U.el('div', { class: 'klabel', text: '层配置的四种组合（45 层里出现了哪些）' }));
    const grid = U.el('div', { style: 'display:grid;grid-template-columns:120px 1fr 1fr;gap:8px;width:100%' });
    const hdr = (t) => U.el('div', { class: 'n sm dim', style: 'text-align:center;padding:4px', html: t });
    grid.appendChild(U.el('div', {})); grid.appendChild(hdr('MLP 稠密')); grid.appendChild(hdr('MoE 稀疏'));
    const cells = [];
    const addRow = (label, cc, a, b) => {
      grid.appendChild(U.el('div', { class: 'n sm', style: 'color:var(--c' + cc + ')', text: label }));
      [a, b].forEach((v, k) => {
        const e = U.el('div', {
          class: 'card cc' + cc,
          style: 'padding:9px 11px;text-align:center;' + (v ? '' : 'opacity:.24') });
        e.innerHTML = v
          ? '<div class="mono" style="font-size:11.5px;color:#fff">' + v.n + '</div>'
            + '<div class="mono" style="font-size:10px;color:var(--ink-faint);margin-top:3px">' + v.l + '</div>'
          : '<div class="mono" style="font-size:11px;color:var(--ink-faint)">不出现</div>';
        grid.appendChild(e); cells.push(e);
      });
    };
    addRow('KDA 线性', 0, { n: '3 层', l: 'layer 0,1,2' }, { n: '31 层', l: 'layer 4,5,6,8…' });
    addRow('MLA + DSA', 1, { n: '0 层', l: '' }, { n: '11 层', l: 'layer 3,7,11…' });
    wrap.appendChild(grid);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 两条独立的选择轴</span>';
    tl.at(2400, () => { msg.innerHTML = '注意力轴：<em>layer_types[i]</em> 决定 KDA 还是 MLA'; });
    tl.at(5600, () => { msg.innerHTML = 'MLP 轴：<em>mlp_layer_types[i]</em> 决定稠密还是 MoE'; });
    tl.at(9000, () => {
      msg.innerHTML = '两轴相乘 → <em>KDA×稠密</em>(3) + <em>KDA×稀疏</em>(31) + <em>MLA×稀疏</em>(11) = 45';
    });
    tl.at(12800, () => {
      msg.innerHTML = '<span class="cm">// MLA×稠密 这一格是空的 —— 2^2 种组合里只用 3 种</span>';
    });
  },
},

/* ------------------------------------------------- 8 出口 */
{
  kicker: '阶段 ⑥ · 出口',
  title: '收拢：<span class="hl-a">hc_head</span> 把 4 条流合成 1 条',
  sub: 'norm(hc_head(hidden_states)) —— 顺序不能反：归一化是逐流的，收拢必须在前。'
     + '而这一步本身出乎意料地简单：等权平均，0 个参数。',
  caption: '到这里主干就结束了。lm_head 与采样在 Glm5NextForConditionalGeneration 里。',
  lang: 'python',
  codeStart: 1511,
  code: `        hidden_states = self.norm(self.hc_head(hidden_states))
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)`,
  codeNote: '主干最后一行的全部内容。',
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const stage = U.el('div', { class: 'row gap20 center', style: 'width:100%;flex:1' });
    wrap.appendChild(stage);

    /* 4 流 -> 1 流 */
    const inCol = U.el('div', { class: 'col gap5 center' });
    inCol.appendChild(U.el('div', { class: 'klabel', text: '(batch, seq, 4, 4096)' }));
    const inBars = [];
    for (let s = 0; s < 4; s++) {
      const b = U.el('div', {
        style: 'width:230px;height:38px;border-radius:8px;border:1px solid var(--c' + (s % 6) + ');'
             + 'background:rgba(251,191,36,.16);transition:opacity .5s,transform .5s;'
             + 'display:flex;align-items:center;justify-content:center;'
             + 'font:600 10px var(--mono);color:var(--ink-dim)',
        text: '流 ' + s });
      inCol.appendChild(b); inBars.push(b);
    }
    stage.appendChild(inCol);

    const mid = U.el('div', { class: 'col center gap6' });
    mid.innerHTML = '<div style="font-size:24px;color:var(--accent)">⟹</div>'
      + '<div class="mono" style="font-size:10.5px;color:var(--c2)">hc_head</div>';
    stage.appendChild(mid);

    const outCol = U.el('div', { class: 'col gap5 center' });
    outCol.appendChild(U.el('div', { class: 'klabel', text: '(batch, seq, 4096)' }));
    const outBar = U.el('div', {
      style: 'width:230px;height:184px;border-radius:10px;border:1px solid var(--accent);'
           + 'background:linear-gradient(180deg,rgba(34,211,238,.30),rgba(34,211,238,.10));'
           + 'display:flex;align-items:center;justify-content:center;'
           + 'font:600 12px var(--mono);color:#c9f7ff;transition:opacity .5s,transform .5s',
      text: 'last_hidden_state' });
    outCol.appendChild(outBar);
    stage.appendChild(outCol);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    outBar.style.opacity = '0'; outBar.style.transform = 'scale(.9)';
    msg.innerHTML = '<span class="cm">// 45 层跑完，手上还是 4 条流</span>';
    tl.at(3000, () => {
      msg.innerHTML = '<em>hc_head</em> 做的是 <span class="hl3">等权平均</span> '
        + '<span class="cm">// hidden_streams.mean(dim=2)，0 个参数</span>';
    });
    tl.at(6200, () => {
      inBars.forEach((b, s) => { b.style.opacity = '.35'; b.style.transform = 'translateX(10px)'; });
      outBar.style.opacity = '1'; outBar.style.transform = 'none';
      msg.innerHTML = '4 条 → <em>1</em> 条 &nbsp;<span class="cm">// 这一步之后才做 RMSNorm</span>';
    });
    tl.at(10000, () => {
      inBars.forEach(b => { b.style.opacity = '1'; b.style.transform = 'none'; });
      msg.innerHTML = '<span class="cm">// 顺序不能反：norm 是逐流的，先收拢就没得归了</span>';
    });
    tl.at(12600, () => {
      msg.innerHTML = '返回值带 <em>past_key_values</em> —— 缓存由主干函数自己交出来';
    });
  },
},

/* ------------------------------------------------- 9 收束 */
{
  kicker: '收束',
  title: '把六个阶段连起来',
  sub: '这张图是后面 69 课的目录。每一块都能说出对应的源文件与课号。',
  caption: '下一课 L0-02：从一个 config 读出上面全部形状。',
  lang: 'python',
  codeStart: 1459,
  code: `        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)`,
  codeNote: '回到开头：现在这 6 行里的每一个名字你都能说出它属于哪个阶段。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const steps = [
      { n: '①', t: '取 token', c: 0, d: 'input_ids 或 inputs_embeds（二选一）' },
      { n: '②', t: '嵌入', c: 4, d: 'embed_tokens → (b, s, 4096)' },
      { n: '③', t: '展开', c: 2, d: 'hc_mult=4 → (b, s, 4, 4096)' },
      { n: '④', t: '45 层', c: 1, d: '按 layer_types 排班' },
      { n: '⑤', t: '收拢', c: 5, d: 'hc_head 等权平均 + norm → (b, s, 4096)' },
      { n: '⑥', t: '采样', c: 3, d: 'lm_head → logits → next_token' },
    ];
    const row = U.el('div', { class: 'row gap8', style: 'width:100%;align-items:stretch' });
    const els = steps.map(s => {
      const e = W.card({ cc: s.c, tint: s.c, num: null,
        title: '<span class="mono" style="font-size:11px">' + s.n + ' ' + s.t + '</span>',
        sub: s.d, style: 'flex:1;text-align:center' });
      row.appendChild(e); return e;
    });
    wrap.appendChild(row);

    /* 实测数字表 */
    const tb = W.table([
      ['层数', '<span class="hl">45</span>', '34 KDA + 11 MLA/DSA'],
      ['MLP 排班', '<span class="hl">3 : 42</span>', '前 3 层稠密，其余 MoE'],
      ['路由', '<span class="hl">288 → 8</span>', '+1 个共享专家'],
      ['残差流', '<span class="hl">4</span>', 'hc_mult'],
      ['词表', '<span class="hl">154880</span>', 'pad_token_id = 154820'],
    ], { head: ['量', '实测值', '说明'] });
    wrap.appendChild(U.el('div', { class: 'card', cc: 0, style: 'padding:10px 12px' }, [
      U.el('div', { class: 'klabel', text: '本课用到的实测数字（probe1.py / probe2.py）' }), tb,
    ]));

    /* 练习 */
    wrap.appendChild(W.exercise(
      'decode 第 5 个 token 时，<code class="inl">position_ids</code> 的值是多少？'
      + '<code class="inl">inputs_embeds</code> 的形状呢？',
      '若 prompt 长 <b>7</b>：prefill 之后缓存里有 7 个 token。'
      + '第 5 次 decode 时 <code class="inl">past_seen</code> = 7+4 = <b>11</b>，'
      + '所以 <code class="inl">position_ids = [[11]]</code>，'
      + '<code class="inl">inputs_embeds</code> 形状 <code class="inl">(1, 1, 4096)</code>。'
      + '<br>关键：<b>形状永远是 seq=1</b>，代价与轮次无关 —— 这正是 KV cache 存在的理由。'));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    els.forEach(e => e.style.opacity = '.28');
    msg.innerHTML = '<span class="cm">// 六个阶段，逐个点亮</span>';
    steps.forEach((s, i) => {
      tl.at(1500 + i * 1900, () => {
        els.forEach((e, k) => { e.style.opacity = (k === i) ? '1' : '.28'; });
        msg.innerHTML = '<em>' + s.n + '</em> ' + s.t + ' &nbsp;<span class="cm">// ' + s.d + '</span>';
      });
    });
    tl.at(13500, () => {
      els.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = '一次前向 = 嵌入 → 展开成 4 条流 → 45 次排班子层 → 收拢 → 归一化';
    });
    tl.at(16200, () => {
      msg.innerHTML = '<span class="cm">// 下一课 L0-02：这些形状全部写在一个 config 里</span>';
    });
  },
},

];
