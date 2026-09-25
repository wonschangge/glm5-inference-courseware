/* ==========================================================================
   L3-07 · 遗忘门与门控范数
   --------------------------------------------------------------------------
   覆盖：models/glm5_next/modeling_glm5_next.py（1 个文件 / 2444 行）
   目标：看完能说出 (1) linear_lower_bound=-5.0 在下界截断里防的是什么，
        (2) 遗忘门落在 KDA 递推的哪一步，(3) A_log / dt_bias 的初始化从哪来、
        以及为什么它必须和 forward 的分支配套。
   实测：_data/recon/probe_l307.py（CPU / torch 2.14.0+cpu / transformers 5.18.0.dev0）
   排版基准：#visual 可视区 = 994 x 662.7 舞台像素（无头浏览器实测）。
   ========================================================================== */
'use strict';

/* 实测常量（probe_l307.py 的输出，不是估算） */
const M307 = {
  hidden: 4096, headDim: 128, heads: 64, qkv: 8192,
  lower: -5.0,
  floor: 0.006737947,           /* exp(-5)，实测 */
  fgParams: 1581120,            /* 遗忘门参数量 */
  qkvParams: 100663296,         /* q/k/v 三投影参数量，遗忘门占 1.57% */
  dtLo: -6.857, dtHi: -2.290,   /* 实测 dt_bias 区间 */
  spLo: 0.00105, spHi: 0.09644, /* 实测 softplus(dt_bias) 区间 */
  decay1: 0.911,                /* 初始状态下 decay 的均值 */
};

/* ---------- 小工具：一段带标题的连接符（不依赖 .arrow 的 flex:1） ---------- */
function ar307(label, w) {
  return U.el('div', { class: 'col center gap4', style: 'flex:none;width:' + (w || 130) + 'px' }, [
    U.el('div', { class: 'mono', style: 'font-size:10px;color:var(--ink-faint);text-align:center', text: label }),
    U.el('div', { class: 'mono', style: 'font-size:15px;color:var(--accent)', text: '⟶' }),
  ]);
}

/* ---------- 小工具：一个方框（张量形状用） ---------- */
function bx307(t, s, cc, w) {
  const e = U.el('div', {
    class: 'card cc' + cc,
    style: 'flex:none;width:' + w + 'px;text-align:center;padding:9px 10px',
  });
  e.innerHTML = '<div class="mono" style="font-size:12px;color:#fff">' + t + '</div>'
    + '<div class="mono" style="font-size:10px;color:var(--ink-faint);margin-top:3px">' + s + '</div>';
  return e;
}

const SCENES = [

/* ------------------------------------------------- 1 全景：两个门在哪 */
{
  kicker: 'L3 主干 · 开场',
  title: '一层 KDA 里，<span class="hl-a">两个门</span>各管一件事',
  sub: '遗忘门管「上一步的记忆留多少」，门控范数管「这一步的输出放多少」。本课把这两个门拆到参数级。',
  caption: '回顾 L0-01：45 层里 34 层是 linear_attention。本课拆的就是这 34 层里最小的两个算子。',
  lang: 'python',
  codeStart: 733,
  code: `        # Forget gate and input gate
        g = self.forget_gate(hidden_states)
        beta = torch.sigmoid(self.b_proj(hidden_states))`,
  codeNote: 'g 是对数衰减率（后面要过 exp），beta 是 0~1 的写入门 —— 两种完全不同的参数化。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap14 vizgrow', style: 'width:100%' });
    wrap.appendChild(viz);

    const fl = W.flow([
      { t: 'hidden_states', s: '(b, s, 4096)', cc: 0 },
      { t: 'q/k/v + 短卷积', s: '8192 x 3', cc: 4 },
      { t: 'KDA 递推', s: '状态 64x128x128', cc: 1 },
      { t: '门控范数', s: 'RMSNormGated', cc: 2 },
      { t: 'o_proj', s: '(b, s, 4096)', cc: 3 },
    ], { style: 'width:100%' });
    viz.appendChild(U.el('div', { class: 'col gap6', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: '一层 linear_attention（KDA）内部：本课只讲被圈出来的两步' }),
      fl,
    ]));

    const defs = [
      { n: '门 1', t: '遗忘门 g', cc: 1, d: 'log-decay，值域 (-5, 0)。每一步乘在递推状态上 —— 本课主角' },
      { n: '门 2', t: '写入门 beta', cc: 4, d: 'sigmoid，值域 (0, 1)。乘在 delta 上，决定这一步写多少' },
      { n: '门 3', t: '门控范数 gate', cc: 2, d: 'sigmoid，逐通道放行 KDA 的输出 —— 本课主角' },
    ];
    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cards = defs.map(d => {
      const e = W.card({
        cc: d.cc, tint: d.cc, style: 'flex:1',
        title: '<span class="mono" style="font-size:10.5px">' + d.n + '</span> ' + d.t,
        sub: d.d,
      });
      row.appendChild(e); return e;
    });
    viz.appendChild(row);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    fl.focus(0);
    msg.innerHTML = '<span class="cm">// hidden_states 进来，先做投影与短卷积</span>';
    tl.at(1200, () => { fl.focus(1); msg.innerHTML = 'q / k / v 三路投影：<em>4096</em> <span class="op">→</span> <em>8192</em> 各一份'; });
    tl.at(4000, () => { fl.focus(2); msg.innerHTML = '<span class="fn">KDA</span> 递推：状态 <em>(b, 64, 128, 128)</em> 逐步更新 —— 遗忘门在这里面'; });
    tl.at(6800, () => { fl.focus(3); msg.innerHTML = '门控范数：先 <em>RMSNorm</em> 再逐通道放行 —— 第二个门在这里'; });
    tl.at(9600, () => { fl.focus(4); msg.innerHTML = '<span class="fn">o_proj</span> 回到 <em>4096</em>，这一层结束'; });
    tl.at(12400, () => {
      fl.focus(-1);
      cards.forEach((c, i) => { c.classList.toggle('ac', i !== 1); c.classList.toggle('dimmed', i === 1); });
      msg.innerHTML = '三个放行系数，<em>门 1</em> 与 <em>门 3</em> 是本课主角';
    });
    tl.at(14800, () => {
      cards.forEach(c => { c.classList.remove('ac'); c.classList.remove('dimmed'); });
      msg.innerHTML = '<span class="cm">// 门 1 是 log 参数化（要过 exp），门 2/3 是概率参数化（sigmoid）</span>';
    });
  },
},

/* ------------------------------------------------- 2 遗忘门的构造 */
{
  kicker: '阶段 1 · 构造',
  title: '遗忘门 = <span class="hl-a">低秩投影</span> + 两个专属参数',
  sub: '4096 → 128 → 8192 的瓶颈让门的容量很小：实测 1,581,120 个参数，只有同层 q/k/v 投影的 1.57%。',
  caption: 'A_log 每头一个数（64 个），dt_bias 每 (头, 通道) 一个数（8192 个）—— 形状不同，广播方式就不同。',
  lang: 'python',
  codeStart: 348,
  code: `        self.f_a_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.dt_bias = nn.Parameter(torch.empty(self.qkv_dim))
        self.A_log = nn.Parameter(torch.empty(self.num_heads))

        self.safe_gate_lower_bound = config.linear_lower_bound`,
  codeNote: 'torch.empty 而不是 zeros —— 这两个参数的初值全部交给 _init_weights 的专门分支。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap14 vizgrow hstart', style: 'width:100%' });
    wrap.appendChild(viz);

    /* 低秩链路 */
    const chain = U.el('div', { class: 'row gap10 center', style: 'width:100%' });
    const bH = bx307('hidden_states', '(b, s, 4096)', 0, 168);
    const bM = bx307('head_dim 128', '瓶颈', 1, 120);
    const bO = bx307('qkv_dim 8192', '64 头 x 128', 2, 168);
    const a1 = ar307('f_a_proj  524,288', 150);
    const a2 = ar307('f_b_proj  1,048,576', 160);
    [bH, a1, bM, a2, bO].forEach(e => chain.appendChild(e));
    viz.appendChild(U.el('div', { class: 'col gap6', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: '低秩投影：门的容量被压到 128 维' }), chain,
    ]));

    /* 两个参数 */
    const pr = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cA = W.card({ cc: 1, tint: 1, style: 'flex:1',
      title: '<span class="mono" style="font-size:11px">A_log (64,)</span>',
      sub: '每个头一个时间尺度。forward 里 view(1,1,64,1)，一个值铺满这一头的 128 个通道。' });
    const cD = W.card({ cc: 2, tint: 2, style: 'flex:1',
      title: '<span class="mono" style="font-size:11px">dt_bias (8192,)</span>',
      sub: '每个 (头, 通道) 一个加性偏置。forward 里 view(1,1,-1)，逐格相加，是这个门唯一的逐通道参数。' });
    pr.appendChild(cA); pr.appendChild(cD);
    viz.appendChild(pr);

    /* 参数量对比 */
    const bars = W.bars([
      { label: '遗忘门', value: M307.fgParams, cc: 1, valueText: '1,581,120' },
      { label: 'q / k / v 三投影', value: M307.qkvParams, cc: 4, valueText: '100,663,296' },
    ], { max: M307.qkvParams });
    viz.appendChild(U.el('div', { class: 'col gap6', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: '参数量实测对比（条长 = 参数量）' }), bars,
    ]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    [bH, a1, bM, a2, bO].forEach(e => { e.style.opacity = '.2'; });
    [cA, cD].forEach(e => { e.style.opacity = '.2'; });
    bars.setAll(0);
    msg.innerHTML = '<span class="cm">// 先看投影：4096 -> 128 -> 8192</span>';
    tl.at(1200, () => { bH.style.opacity = '1'; a1.style.opacity = '1'; bM.style.opacity = '1'; });
    tl.at(4000, () => {
      a2.style.opacity = '1'; bO.style.opacity = '1';
      msg.innerHTML = '为什么敢用 128 的瓶颈：门只需要吐出「每头每通道的衰减分数」，不需要 q/k/v 那样的容量';
    });
    tl.at(7200, () => {
      cA.style.opacity = '1';
      msg.innerHTML = '<em>A_log</em> 只有 <em>64</em> 个数：一个头一个 —— 所以它是「每头一个时间尺度」';
    });
    tl.at(10200, () => {
      cD.style.opacity = '1';
      msg.innerHTML = '<em>dt_bias</em> 有 <em>8192</em> 个数：每个 (头, 通道) 一个 —— 逐通道的加性偏置';
    });
    tl.at(13200, () => {
      bars.setTo(0, 1); bars.setTo(1, 1);
      msg.innerHTML = '实测 <em>1,581,120</em> vs <em>100,663,296</em>：遗忘门只占 <em>1.57%</em>';
    });
  },
},

/* ------------------------------------------------- 3 前向算账 */
{
  kicker: '阶段 2 · 形状',
  title: '★ <span class="hl-a">A 是每头一个</span>，<span class="hl-b">dt_bias 是每通道一个</span>',
  sub: '同样一行的两个量，广播方式完全不同；这也是「逐通道时间尺度」这个说法的来源。',
  caption: '记住这两个形状，第六节看递推时就不会把 g 的形状读错。',
  lang: 'python',
  codeStart: 356,
  code: `        hidden_shape = (*hidden_states.shape[:2], -1, self.head_dim)

        forget_gate = self.f_b_proj(self.f_a_proj(hidden_states))
        g = (forget_gate.float() + self.dt_bias.float().view(1, 1, -1)).view(hidden_shape)
        A_log = self.A_log.float().view(1, 1, self.num_heads, 1)
        decay_rate = torch.exp(A_log)`,
  codeNote: '三处 .float() 是必须的：bf16 的 8 位有效位吃不下 -4 量级的偏置加 1e-2 量级的分数。',
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow', style: 'width:100%' });
    wrap.appendChild(viz);

    const stage = U.el('div', { class: 'row gap20 center', style: 'width:100%' });
    viz.appendChild(stage);

    /* 左：heads x channels 示意网格 */
    const left = U.el('div', { class: 'col gap6 center', style: 'flex:none' });
    left.appendChild(U.el('div', { class: 'klabel', text: 'g 的形状示意：4 头 x 8 通道（实际 64 x 128）' }));
    const T = W.tensor(4, 8, (r) => r, { cell: 32 });
    left.appendChild(T);
    left.appendChild(U.el('div', { class: 'mono dim', style: 'font-size:11.5px', text: '(b, s, 64, 128)' }));
    stage.appendChild(left);

    /* 右：两张卡 */
    const right = U.el('div', { class: 'col gap12', style: 'flex:1' });
    const cA = W.card({ cc: 1, tint: 1,
      title: '<span class="mono" style="font-size:11px">A_log.view(1, 1, 64, 1)</span>',
      sub: '每头一个值，铺满这一头的 128 个通道 —— 颜色按行走。' });
    const cD = W.card({ cc: 2, tint: 2,
      title: '<span class="mono" style="font-size:11px">dt_bias.view(1, 1, -1)</span>',
      sub: '8192 个值逐格相加 —— 每一格都有自己的偏置。' });
    right.appendChild(cA); right.appendChild(cD);
    stage.appendChild(right);

    /* 形状链 */
    const ch = U.el('div', { class: 'row gap8 center wrap', style: 'width:100%' });
    ch.innerHTML =
      '<span class="chip c0">(b, s, 4096)</span><span class="dim mono" style="font-size:11px">f_a_proj</span>'
      + '<span class="chip c1">(b, s, 128)</span><span class="dim mono" style="font-size:11px">f_b_proj</span>'
      + '<span class="chip c2">(b, s, 8192)</span><span class="dim mono" style="font-size:11px">view</span>'
      + '<span class="chip c4">(b, s, 64, 128)</span>';
    viz.appendChild(U.el('div', { class: 'col gap6', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: '形状链（实测：8192 = 64 x 128）' }), ch,
    ]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    /* 点亮：只用 inset 阴影 + 亮度，不用 scale（.tcell.on 的 scale(1.16)
       会把最左一列挤出 #visual —— 逐幕溢出巡检抓到过） */
    const lit = (c, on) => {
      c.style.boxShadow = on ? 'inset 0 0 0 2px var(--accent)' : '';
      c.style.filter = on ? 'brightness(1.55)' : 'brightness(.7)';
    };
    const litRows = (pred) => T.cells.forEach((c, k) => lit(c, pred(Math.floor(k / 8))));

    msg.innerHTML = '<span class="cm">// 先看形状：hidden 折成 (b, s, 64, 128)</span>';
    tl.at(1200, () => {
      cA.classList.add('ac'); cD.classList.remove('ac');
      litRows(h => h === 0);
      msg.innerHTML = '<em>A_log</em> 一行一个值 → <span class="fn">decay_rate = exp(A_log)</span> 是<em>每头</em>的时间尺度';
    });
    tl.at(4400, () => {
      cA.classList.remove('ac'); cD.classList.add('ac');
      litRows(() => false);
      msg.innerHTML = '<em>dt_bias</em> 逐格相加 → 这是<em>每通道</em>的偏置，也是唯一的逐通道参数';
    });
    tl.at(7600, () => {
      cD.classList.remove('ac');
      litRows(() => true);
      msg.innerHTML = 'raw 分数 + dt_bias 之后才 <em>view</em> 成 <em>(b, s, 64, 128)</em>';
    });
    tl.at(10800, () => {
      litRows(() => false);
      msg.innerHTML = '三处 <em>.float()</em>：这一步加法必须在 fp32 里做';
    });
    tl.at(13400, () => {
      msg.innerHTML = '<span class="cm">// 回顾 L3-06：mHC 的门形状是 (b,s,4,4096)，这里换成 (b,s,64,128)，都靠广播铺开</span>';
    });
  },
},

/* ------------------------------------------------- 4 ★ 下界截断 */
{
  kicker: '阶段 3 · 值域',
  title: '★ <span class="hl-a">-5.0</span>：把 log-decay 关进 (-5, 0)',
  sub: 'g = -5 x sigmoid(decay_rate x raw)：无论打分多大，门都出不了这个区间，衰减率永远到不了 0。',
  caption: '手算表来自 probe_l307.py 第 10 节（fp32 实测）。条长 = |g|，右侧 = 该点的 decay。',
  lang: 'python',
  codeStart: 363,
  code: `        # Safe lower bound decay
        if self.safe_gate_lower_bound is not None:
            return self.safe_gate_lower_bound * torch.sigmoid(decay_rate * g)`,
  codeNote: '只要下界不是 None 就直接 return —— linear_lower_bound 决定的是走哪条公式，不只是取什么值。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart', style: 'width:100%' });
    wrap.appendChild(viz);

    const rows = [
      { raw: -15, g: -0.0000015, d: 0.999998450 },
      { raw: -8, g: -0.0016768, d: 0.998324654 },
      { raw: -4, g: -0.0899310, d: 0.913994193 },
      { raw: 0, g: -2.5000000, d: 0.082084998 },
      { raw: 4, g: -4.9100690, d: 0.007371980 },
      { raw: 8, g: -4.9983232, d: 0.006749254 },
      { raw: 15, g: -4.9999981, d: 0.006737960 },
      { raw: 30, g: -5.0000000, d: 0.006737947 },
    ];
    const bars = W.bars(rows.map(r => ({
      label: 'raw = ' + (r.raw > 0 ? '+' : '') + r.raw + '.0',
      value: Math.abs(r.g), cc: 1,
      valueText: 'decay ' + r.d.toFixed(9),
    })), { max: 5 });
    viz.appendChild(U.el('div', { class: 'col gap6', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: 'g = -5 x sigmoid(decay_rate x raw)，decay_rate = 1（A_log = 0）' }),
      bars,
    ]));

    const sr = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const s1 = W.card({ cc: 1, tint: 1, style: 'flex:1',
      title: '值域 (-5, 0)', sub: 'sigmoid 的值域是 (0, 1)，乘 -5.0 之后无论 raw 多大都出不去。' });
    const s2 = W.card({ cc: 4, tint: 4, style: 'flex:1',
      title: 'floor = e^-5 = 0.006738', sub: '单步衰减永远 >= 这个数，永远不等于 0。' });
    const s3 = W.card({ cc: 2, tint: 2, style: 'flex:1',
      title: 'raw > 15 之后饱和', sub: 'raw=+15 与 raw=+60 的 decay 在第 6 位小数上才分得开。' });
    [s1, s2, s3].forEach(e => sr.appendChild(e));
    viz.appendChild(sr);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    bars.setAll(0);
    msg.innerHTML = '<span class="cm">// raw 是 f_b_proj(f_a_proj(x)) 给出的分数</span>';
    tl.at(1200, () => { [0, 1, 2, 3].forEach(i => bars.setTo(i, 1)); });
    tl.at(4200, () => {
      [4, 5, 6, 7].forEach(i => bars.setTo(i, 1));
      msg.innerHTML = 'raw 越正，门越负 → 衰减越狠；到 <em>raw = +15</em> 附近就压到 floor 上了';
    });
    tl.at(7600, () => {
      s3.classList.add('ac');
      msg.innerHTML = '这不是 <span class="fn">clamp()</span>，是 <em>sigmoid 自己的饱和</em> —— 函数形式自带的截断';
    });
    tl.at(10800, () => {
      s1.classList.add('ac'); s3.classList.remove('ac');
      msg.innerHTML = '手算一例：raw = 0 <span class="op">→</span> g = <em>-2.5</em> <span class="op">→</span> decay = <em>0.082085</em>（忘掉 91.79%）';
    });
    tl.at(13800, () => {
      s1.classList.remove('ac'); s2.classList.add('ac');
      msg.innerHTML = 'floor <em>e^-5 = 0.006737947</em>：单步最多忘掉 99.33%，<span class="hl">但绝不会一步清零</span>';
    });
    tl.at(16200, () => {
      s2.classList.remove('ac');
      msg.innerHTML = '<span class="cm">// 这句话只保证「一步」—— 累积会怎样，见第六节的实测</span>';
    });
  },
},

/* ------------------------------------------------- 5 没有下界的那一支 */
{
  kicker: '阶段 4 · 对照',
  title: '另一支：<span class="hl-a">A 是乘法增益</span>，门可以掉到 -320',
  sub: 'g = -decay_rate x softplus(raw)。A 与 raw 都不封顶，所以这一支允许模型一步之内把状态乘成 0。',
  caption: '同一个 A_log，在两种公式里的角色完全不同 —— 这正是初始化必须跟着分支走的原因。',
  lang: 'python',
  codeStart: 367,
  code: `        # Softplus "log(1 + exp(x))" with uper bound restraint to avoid overflows
        # NOTE: Softplus for larger values (e.g. 20+), Softplus(x) == x
        g_softplus = torch.where(g > 20.0, g, torch.log(1.0 + torch.exp(g)))

        return -decay_rate * g_softplus`,
  codeNote: '手写 softplus：g > 20 之后直接取 g，因为 exp(g) 在 g > 88 时 fp32 上溢。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart', style: 'width:100%' });
    wrap.appendChild(viz);

    /* 两条公式对照 */
    const cmp = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cSafe = W.card({ cc: 4, tint: 4, style: 'flex:1', title: '有下界（默认）',
      sub: 'A 是 sigmoid 内部的<em>斜率</em>。输出恒在 (-5, 0)，与 A 的大小无关。' });
    const cUnsafe = W.card({ cc: 3, tint: 3, style: 'flex:1', title: '没有下界',
      sub: 'A 是<em>乘法增益</em>。A = exp(A_log) 不封顶，raw 也不封顶 → 门无下界。' });
    const f1 = U.el('div', { class: 'formula', style: 'width:100%;margin-top:8px' });
    f1.innerHTML = 'g = <em>-5.0</em> x sigmoid(decay_rate x raw)';
    const f2 = U.el('div', { class: 'formula', style: 'width:100%;margin-top:8px' });
    f2.innerHTML = 'g = <em>-decay_rate</em> x softplus(raw)';
    cSafe.appendChild(f1); cUnsafe.appendChild(f2);
    cmp.appendChild(cSafe); cmp.appendChild(cUnsafe);
    viz.appendChild(cmp);

    /* 饱和表 */
    const tb = W.table([
      ['1.0', '-4.5', '-0.054934714', '1.087e-02'],
      ['4.0', '-4.5', '-0.000000076', '1.523e-08'],
      ['16.0', '-4.5', '-0.000000000', '<span class="hlbad">5.380e-32</span>'],
    ], { head: ['decay_rate', 'raw 分数', 'g = -5 x sigmoid(A x raw)', 'sigmoid 的导数'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 1, style: 'padding:10px 12px' }, [
      U.el('div', { class: 'klabel', text: '实测：A 一大，sigmoid 就饱和，门被冻住（probe 第 12 节）' }), tb,
    ]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    f1.style.opacity = '.15'; f2.style.opacity = '.15';
    msg.innerHTML = '<span class="cm">// 先看两条公式：同一个 A，位置不同</span>';
    tl.at(1200, () => { f1.style.opacity = '1'; msg.innerHTML = '有下界时，A 在 <em>sigmoid 里面</em> —— 它是斜率'; });
    tl.at(4400, () => {
      f2.style.opacity = '1';
      msg.innerHTML = '没有下界时，A 在 <em>外面做乘法</em> —— 实测 A=16、raw=20 时 g = <span class="hlbad">-320.0</span>';
    });
    tl.at(7600, () => {
      msg.innerHTML = '而 <em>exp(-320)</em> 在 fp32 里恰好是 <em>0.0</em>：这一支允许一步把状态乘成 0';
    });
    tl.at(10800, () => {
      msg.innerHTML = '回头看斜率那一支：A=16、raw=-4.5 时 sigmoid 导数只有 <span class="hlbad">5.38e-32</span> — 门冻住';
    });
    tl.at(13800, () => {
      msg.innerHTML = '<span class="cm">// 结论：A_log 的初始化必须和 forward 走哪一支配套（下一幕）</span>';
    });
  },
},

/* ------------------------------------------------- 6 ★ 递推里的位置 */
{
  kicker: '阶段 5 · 落点',
  title: '★ 遗忘门的位置：<span class="hl-a">状态 = 状态 x exp(g)</span>',
  sub: '递推里唯一用到门的地方，就是更新状态之前的那一次乘法 —— 它决定上一步的记忆留多少。',
  caption: '验收点 2：能指着这一行说出遗忘门在 KDA 递推里的位置。展开推导见 L4-02。',
  lang: 'python',
  codeStart: 501,
  code: `    for i in range(sequence_length):
        q_i = query[:, i]
        k_i = key[:, i]
        v_i = value[:, i]
        g_i = g[:, i][..., None].exp()
        b_i = beta[:, i][..., None]

        last_recurrent_state = last_recurrent_state * g_i
        kv_mem = (last_recurrent_state * k_i[..., None]).sum(dim=-2)
        delta = (v_i - kv_mem) * b_i

        last_recurrent_state = last_recurrent_state + k_i.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, i] = (last_recurrent_state * q_i.unsqueeze(-1)).sum(dim=-2)`,
  codeNote: '传进来的 g 是 log-decay，.exp() 之后才是衰减率；顺序是先衰减、再读取、最后写入。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow', style: 'width:100%' });
    wrap.appendChild(viz);

    const fl = W.flow([
      { t: '状态 x exp(g)', s: '遗忘', cc: 1 },
      { t: 'kv_mem', s: '(状态·k).sum', cc: 0 },
      { t: 'delta', s: '(v - kv_mem)·beta', cc: 4 },
      { t: '状态 += k x delta', s: '写入', cc: 5 },
      { t: 'out = (状态·q).sum', s: '读出', cc: 3 },
    ], { style: 'width:100%' });
    viz.appendChild(U.el('div', { class: 'col gap6', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: '递推的 5 步（顺序不能换）' }), fl,
    ]));

    const tb = W.table([
      ['-5.000', '0.000000e+00', 'exp(-120) = 7.667648e-53'],
      ['-1.000', '3.775137e-11', 'exp(-24) = 3.775135e-11'],
      ['-0.050', '3.011944e-01', 'exp(-1.2) = 3.011942e-01'],
    ], { head: ['门 g', '24 步后的状态（实测）', '手算'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 2, style: 'padding:10px 12px' }, [
      U.el('div', { class: 'klabel', text: '实测：k=v=0、beta=0、初始状态全 1，只让这一行乘法起作用' }), tb,
    ]));

    const fix = W.card({ cc: 3, tint: 3, title: '实测订正：分块形式（L4-03 的范围）',
      sub: 'decay_mask 恒在 [0, 1]（减法方向与掩码配套，gate=-5 时非对角最大正好是 e^-5），'
         + '不会上溢；真正归零的是 exp(cumsum)：gate=-5 时 64 个位置里有 44 个精确为 0。' });
    viz.appendChild(fix);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    fl.focus(0);
    fix.style.opacity = '.25';
    msg.innerHTML = '<span class="cm">// 第 505 行：门在这里进递推</span>';
    tl.at(1200, () => { msg.innerHTML = '<em>g_i = g[:, i][..., None].exp()</em> —— 传进来的是 log-decay，这里才变成衰减率'; });
    tl.at(4000, () => {
      fl.focus(1);
      msg.innerHTML = '<span class="hl">状态 = 状态 x g_i</span>：遗忘发生的唯一一行，旧状态整体乘一个 <= 1 的数';
    });
    tl.at(6800, () => { fl.focus(2); msg.innerHTML = '用衰减后的状态读记忆：<em>kv_mem</em>'; });
    tl.at(9600, () => { fl.focus(3); msg.innerHTML = '<em>delta = (v - kv_mem) x beta</em> —— beta 就是第一幕那个写入门'; });
    tl.at(12400, () => {
      fl.focus(4);
      msg.innerHTML = '写入是<em>加法</em>：衰减只作用在旧记忆上，这一步刚写进去的不会被同一行衰减';
    });
    tl.at(15200, () => {
      fl.focus(-1); fix.style.opacity = '1';
      msg.innerHTML = 'gate=-5 那一行 24 步后<em>精确为 0</em>：不是下界失效，是 fp32 撑不住（exp(-104) 实测就是 0）';
    });
  },
},

/* ------------------------------------------------- 7 ★ 初始化 */
{
  kicker: '阶段 6 · 初始化',
  title: '★ <span class="hl-a">A_log 的初始化跟着分支走</span>，dt_bias 跟着 FLA 惯例',
  sub: '有下界就把 A_log 置 0（decay_rate = 1），没有下界才用 log-uniform —— 这两行是一对，不能拆开改。',
  caption: '源码里那句 NOTE: This is incredibly important so keep it this way at all costs 就是写给这个 if/else 的。',
  lang: 'python',
  codeStart: 1385,
  code: `        if isinstance(module, Glm5NextTextForgetGate):
            # Following FLA initialization
            # NOTE: This is incredibly important so keep it this way at all costs
            if module.safe_gate_lower_bound is not None:
                init.zeros_(module.A_log)
            else:
                init.copy_(
                    module.A_log,
                    init.uniform_(module.A_log, a=1.0, b=16.0).log(),
                )`,
  codeNote: 'init.copy_(p, f(p)) 是「先取值、再变换、再写回」的惯用写法：这里就是把 U(1,16) 取 log。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart', style: 'width:100%' });
    wrap.appendChild(viz);

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cOn = W.card({ cc: 4, tint: 4, style: 'flex:1',
      title: '有下界：init.zeros_(A_log)',
      sub: 'A_log = 0 <span class="op">→</span> decay_rate = exp(0) = <em>1.0</em>'
         + '<br>实测：默认 config 下 4 个头全是 1.0' });
    const cOff = W.card({ cc: 2, tint: 2, style: 'flex:1',
      title: '没有下界：log-uniform(1, 16)',
      sub: 'A_log = log(U(1,16)) <span class="op">→</span> decay_rate ~ <em>U(1, 16)</em>'
         + '<br>实测采样：[3.606, 2.113, 10.985, 15.361]' });
    row.appendChild(cOn); row.appendChild(cOff);
    viz.appendChild(row);

    const why = W.card({ cc: 1, tint: 1, title: '为什么不能对调',
      sub: '有下界那一支里 A 是 sigmoid 的斜率：A=16 时梯度只剩 5.38e-32（门冻住）；A=1 时 raw=-4.5 '
         + '给出 g=-0.0549、梯度 1.087e-02。没有下界那一支里 A 是乘法增益，A ∈ [1,16] 才给得出'
         + '每头不同的时间尺度初值。' });
    viz.appendChild(why);

    const tb = W.table([
      ['0.001', '-6.907255', '实测 softplus(dt_bias) 落在 [1.05e-3, 9.64e-2]'],
      ['0.010', '-4.600166', '实测 dt_bias 落在 [-6.857, -2.290]'],
      ['0.100', '-2.252168', '初始 decay 均值 0.911（每步记住约 91%）'],
    ], { head: ['先抽到的 dt', 'softplus^-1(dt) 写进 dt_bias', '实测'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 2, style: 'padding:10px 12px' }, [
      U.el('div', { class: 'klabel', text: 'dt_bias 的三步逆 softplus（两种分支共用，没有 if）' }), tb,
    ]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    [cOn, cOff, why].forEach(e => { e.style.opacity = '.22'; });
    msg.innerHTML = '<span class="cm">// 初始化与 forward 的分支是一对</span>';
    tl.at(1200, () => { cOn.style.opacity = '1'; msg.innerHTML = '有下界 → <em>A_log = 0</em> → decay_rate = 1.0：让门落在 sigmoid 的线性区'; });
    tl.at(4400, () => { cOff.style.opacity = '1'; msg.innerHTML = '没有下界 → <em>log(U(1,16))</em> → decay_rate ~ U(1,16)：给 64 个头不同的时间尺度初值'; });
    tl.at(7600, () => { why.style.opacity = '1'; msg.innerHTML = '对调会怎样：门要么一上来就饱和冻住，要么 64 个头挤在同一个时间尺度上'; });
    tl.at(10800, () => { msg.innerHTML = '<em>dt_bias</em> 走的是另一套：先抽 <em>dt ~ U(1e-3, 1e-1)</em>，再做<em>逆 softplus</em>写回'; });
    tl.at(13800, () => { msg.innerHTML = '结果是一串 <em>-2.3 ~ -6.9</em> 的负偏置 —— 把初始的门推向「慢遗忘」一侧'; });
    tl.at(16200, () => {
      msg.innerHTML = '<span class="cm">// 注意：forward 里 dt_bias 是直接相加、不过 softplus —— FLA 的语义在这里已经变了</span>';
    });
  },
},

/* ------------------------------------------------- 8 门控范数 */
{
  kicker: '阶段 7 · 第二个门',
  title: '门控范数：<span class="hl-a">先归一化，再逐通道放行</span>',
  sub: 'RMSNorm 在 fp32 里算（连权重都不降精度），门是 sigmoid(gate)；顺序反了分母就不是这一步激活的均方了。',
  caption: '对比 L3-01：Glm5NextTextRMSNorm 是先转回 input_dtype 再乘权重，这里把权重也升到 fp32。',
  lang: 'python',
  codeStart: 383,
  code: `    def forward(self, hidden_states, gate=None) -> torch.Tensor:
        input_dtype = hidden_states.dtype

        # Strict FP32 norm (do not downcast on the weights)
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight.to(torch.float32) * hidden_states

        # Apply gating
        hidden_states = hidden_states * ACT2FN[self.activation](gate.to(torch.float32))

        return hidden_states.to(input_dtype)`,
  codeNote: 'gate=None 会直接抛 AttributeError（实测）—— 这个函数不允许不传门。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow', style: 'width:100%' });
    wrap.appendChild(viz);

    const fl = W.flow([
      { t: 'core_attn_out', s: '(b, s, 64, 128)', cc: 0 },
      { t: 'x rsqrt(mean(x^2)+eps)', s: 'fp32', cc: 1 },
      { t: 'x weight', s: 'fp32，不降精度', cc: 4 },
      { t: 'x sigmoid(gate)', s: '逐通道放行', cc: 2 },
      { t: 'o_proj', s: '(b, s, 4096)', cc: 3 },
    ], { style: 'width:100%' });
    viz.appendChild(U.el('div', { class: 'col gap6', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: '门控范数的 5 步' }), fl,
    ]));

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const c1 = W.card({ cc: 4, tint: 4, style: 'flex:1', title: '严格 fp32',
      sub: 'hidden_states 与 weight 都升到 fp32，最后才 to(input_dtype)。实测 bf16 进、bf16 出。' });
    const c2 = W.card({ cc: 2, tint: 2, style: 'flex:1', title: 'activation 写死 "sigmoid"',
      sub: '不吃 config.hidden_act（对比 L3-02 的 MLP：那里 act_fn 来自 config）。门必须是 0~1 的放行系数。' });
    const c3 = W.card({ cc: 3, tint: 3, style: 'flex:1', title: 'gate 不能省',
      sub: '实测 gate=None 直接抛 AttributeError: NoneType object has no attribute to。' });
    [c1, c2, c3].forEach(e => row.appendChild(e));
    viz.appendChild(row);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    fl.focus(0);
    msg.innerHTML = '<span class="cm">// 输入是 KDA 的输出，不是状态</span>';
    tl.at(1200, () => { fl.focus(1); msg.innerHTML = '只算<em>均方</em>、不减均值 —— 这就是 RMSNorm 与 LayerNorm 的全部区别'; });
    tl.at(4000, () => { fl.focus(2); msg.innerHTML = '注意 <em>weight</em> 也升到 fp32：注释里的 Strict 说的就是这一步'; });
    tl.at(6800, () => { fl.focus(3); msg.innerHTML = '门控在<em>最后</em>：sigmoid(gate) 逐 (头, 通道) 放行'; });
    tl.at(9600, () => { fl.focus(4); msg.innerHTML = '然后才 <em>o_proj</em>，回到 4096'; });
    tl.at(12400, () => {
      fl.focus(-1);
      msg.innerHTML = '顺序不能反：归一化是<em>逐 token</em> 的统计量，门控是<em>逐通道</em> 的系数 —— 先门控会把分母也改掉';
    });
    tl.at(14800, () => {
      msg.innerHTML = '<span class="cm">// 第 767 行那个 gate 来自 g_b_proj(g_a_proj(h))：和遗忘门完全独立的第二个门</span>';
    });
  },
},

/* ------------------------------------------------- 9 收束 */
{
  kicker: '收束',
  title: '把两个门放在一起看',
  sub: '三个放行系数、两种参数化、一套跟着分支走的初始化 —— 这一层 KDA 的门就这些。',
  caption: '下一课 L3-08：把线性注意力、MLA 与 MoE 三个子块按顺序装回一层。',
  lang: 'python',
  codeStart: 766,
  code: `        # Final gated norm and proj
        gate = self.g_b_proj(self.g_a_proj(hidden_states)).view(hidden_shape)
        output = self.o_norm(core_attn_out, gate).reshape(batch_size, seq_len, -1)
        output = self.o_proj(output)`,
  codeNote: '门控范数站在 KDA 的出口：归一化 → 门控 → 输出投影。',
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart', style: 'width:100%' });
    wrap.appendChild(viz);

    const tb = W.table([
      ['<span class="hl2">遗忘门 g</span>', '-5.0 x sigmoid(exp(A_log) x score)',
       '(-5, 0)', 'A_log (64,) / dt_bias (8192,)', 'L341-371'],
      ['写入门 beta', 'sigmoid(b_proj(h))', '(0, 1)', 'b_proj', 'L735'],
      ['<span class="hl4">门控范数 gate</span>', 'sigmoid(g_b_proj(g_a_proj(h)))',
       '(0, 1)', 'g_a/g_b_proj + weight', 'L766-768'],
    ], { head: ['门', '公式', '值域', '专属参数', '源码行'] });
    const tcard = U.el('div', { class: 'card', cc: 0, style: 'padding:10px 12px' }, [
      U.el('div', { class: 'klabel', text: '本课拆开的三个放行系数' }), tb,
    ]);
    viz.appendChild(tcard);

    viz.appendChild(W.exercise(
      '关掉下界时（<code class="inl">linear_attn_config={"safe_gate": False, "gate_lower_bound": None}</code>），'
      + '为什么 <code class="inl">A_log</code> 要改成 log-uniform(1, 16)？两种初始化对调会怎样？',
      '因为两种写法里 <b>A 的角色不同</b>：有下界时 A 是 <code class="inl">sigmoid</code> 内部的斜率，'
      + 'A=16 会把 sigmoid 推饱和，梯度只剩 <b>5.38e-32</b>（实测）—— 门冻住、学不动；'
      + '所以必须置 0（decay_rate = 1），让门落在 sigmoid 的线性区（raw=-4.5 时 g=-0.0549，梯度 1.087e-02）。'
      + '没有下界时 A 是<b>乘法增益</b>，A ∈ [1,16] 才给得出每个头不同的时间尺度初值；'
      + '对调的话，要么门一上来就饱和冻住，要么 64 个头挤在同一个时间尺度上。'));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    const trs = Array.from(tb.querySelectorAll('tbody tr'));
    trs.forEach(r => { r.style.opacity = '.3'; });
    msg.innerHTML = '<span class="cm">// 三行，逐个点亮</span>';
    tl.at(1200, () => {
      trs[0].style.opacity = '1';
      msg.innerHTML = '<em>遗忘门</em>：唯一带下界的门，值域 (-5, 0)，管递推状态';
    });
    tl.at(4000, () => {
      trs[1].style.opacity = '1';
      msg.innerHTML = '<em>写入门 beta</em>：最简单的一个，只是 b_proj + sigmoid';
    });
    tl.at(6800, () => {
      trs[2].style.opacity = '1';
      msg.innerHTML = '<em>门控范数</em>：作用在 KDA 的输出上，先归一化再放行';
    });
    tl.at(9600, () => {
      trs.forEach(r => { r.style.opacity = '1'; });
      msg.innerHTML = '一句话：下界 -5.0 保证的是<em>单步</em>衰减 >= e^-5，不是<em>永不遗忘</em>，也不是防分块上溢';
    });
    tl.at(12800, () => {
      msg.innerHTML = '而 <em>A_log</em> 的初始化与 forward 的分支严格配套 —— 这是本课最该记住的一条工程约束';
    });
    tl.at(15800, () => {
      msg.innerHTML = '<span class="cm">// 下一课 L3-08：三个子块按顺序装回一层（mHC -> 注意力 -> mHC -> MoE）</span>';
    });
  },
},

];
