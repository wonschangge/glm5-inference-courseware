/* ==========================================================================
   L3-01 · 三种 RMSNorm
   --------------------------------------------------------------------------
   覆盖：models/glm5_next/modeling_glm5_next.py（2444 行）
         pytorch_utils.py（255 行）
   目标：看到一层里的任何一个 norm，能立刻说出它是三种形态里的哪一种、
         以及它为什么不能换成另外两种。
   排版基准：#visual 可视区 = 994 x 662.7 舞台像素（无头浏览器实测）。
   实测来源：_data/recon/probe_L3-01.py（脚本不计入覆盖率）。
   ========================================================================== */
'use strict';

/* 实测常量（probe_L3-01.py 输出，不是估算） */
const N = {
  inst: 237, weighted: 113, unweighted: 90, gated: 34,
  normParams: 399616, totalParams: 312692428286,
  gateSigmoidMean: 0.5, gateSigmoidMin: 0.219, gateSigmoidMax: 0.803,
  bf16Diff: 1017, elems: 1048576,
};

const SCENES = [

/* ------------------------------------------------- 1 全景：三种形态 */
{
  kicker: '第 3 层 · 文本主干',
  title: '一层里的归一化，只有 <span class="hl-a">三种形态</span>',
  sub: '主干里 237 个 norm 实例，按「有没有 weight」「有没有 gate」只分成三类 —— 形态不是风格问题，是下游需求决定的。',
  caption: '本课以算子为单位：先看形态，再看每一种形态被谁需要。回顾 L0-01：那一课把 45 层压成一张图，这一课开始拆图里的一块。',
  lang: 'python',
  codeStart: 66,
  code: `@use_kernel_forward_from_hub("RMSNorm")
class Glm5NextTextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Glm5NextTextRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps`,
  codeNote: 'Glm5NextTextRMSNorm —— 加权版。三类里最常见的一个，主干里 113 个实例。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '三种形态 —— 主干里 237 个 norm 实例全部落在这三类里' }));

    const DEFS = [
      { cc: 0, t: '加权版', n: '113 个', cls: 'Glm5NextTextRMSNorm',
        u: '每个子层的入口与出口、q_a / kv_a 投影前、主干出口' },
      { cc: 4, t: '无权重版', n: '90 个', cls: 'Glm5NextTextUnweightedRMSNorm',
        u: '只出现在 mHC 内部：4 条残差流拼接之后做尺度对齐' },
      { cc: 1, t: '门控版', n: '34 个', cls: 'Glm5NextTextRMSNormGated',
        u: '只出现在 KDA 层的 o_norm：逐 head 归一化 + sigmoid 门' },
    ];
    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cards = DEFS.map(d => {
      const c = W.card({ cc: d.cc, tint: d.cc,
        title: d.t + ' <span class="mono" style="font-size:10px;color:var(--ink-faint)">' + d.n + '</span>',
        sub: '<span class="mono" style="font-size:10.5px">' + d.cls + '</span>'
             + '<div style="margin-top:6px">' + d.u + '</div>',
        style: 'flex:1' });
      row.appendChild(c); return c;
    });
    viz.appendChild(row);

    const STATS = [
      { k: 'norm 实例', v: N.inst, s: '45 层内合计', cc: 0 },
      { k: '形态 / 类', v: '3 / 4', s: '有 4 个类，只有 3 种式子', cc: 1 },
      { k: 'norm 参数占比', v: '0.000128%', s: '399,616 / 312,692,428,286', cc: 2 },
      { k: '最贵的形态', v: '4096', s: '加权版每实例的 weight 数', cc: 4 },
    ];
    const srow = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const stats = STATS.map(st => {
      const c = W.card({ cc: st.cc,
        title: '<span class="mono" style="font-size:10.5px">' + st.k + '</span>',
        body: U.el('div', { class: 'n big', text: String(st.v) }),
        sub: '<span style="font-size:10px">' + st.s + '</span>', style: 'flex:1' });
      srow.appendChild(c); return c;
    });
    viz.appendChild(srow);

    const tb = W.table([
      ['加权', '每个通道一个可学的乘数', '下游要方向自由度'],
      ['无权重', 'weight 恒等于 1', '下游只要尺度'],
      ['门控', 'weight × sigmoid(gate)', '下游要在逐通道上「关掉」'],
    ], { head: ['形态', '多了什么', '为什么需要它'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 0, style: 'padding:9px 12px' }, [tb]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    cards.forEach(c => { c.style.opacity = '.3'; });
    msg.innerHTML = '<span class="cm">// 237 个实例，先分类，再逐个拆</span>';
    tl.at(1800, () => {
      cards[0].style.opacity = '1'; cards[0].classList.add('ac');
      msg.innerHTML = '加权版 <em>113</em> 个 <span class="cm">// 每个子层入口/出口各一个，是主干的主力</span>';
    });
    tl.at(5200, () => {
      cards[0].classList.remove('ac');
      cards[1].style.opacity = '1'; cards[1].classList.add('ac');
      msg.innerHTML = '无权重版 <em>90</em> 个 <span class="cm">// 一个参数都没有，全部在 mHC 内部</span>';
    });
    tl.at(8600, () => {
      cards[1].classList.remove('ac');
      cards[2].style.opacity = '1'; cards[2].classList.add('ac');
      msg.innerHTML = '门控版 <em>34</em> 个 <span class="cm">// 34 正好等于 linear_attention 的层数</span>';
    });
    tl.at(12200, () => {
      cards.forEach(c => { c.style.opacity = '1'; c.classList.remove('ac'); });
      stats.forEach(c => c.classList.add('ac'));
      msg.innerHTML = '三种形态的参数量加起来只有 <em>399,616</em>，占全模型 <em>0.000128%</em>';
    });
    tl.at(15200, () => {
      stats.forEach(c => c.classList.remove('ac'));
      msg.innerHTML = '<span class="cm">// 参数这么少，所以真正的约束不是显存，而是「下游要什么」</span>';
    });
  },
},

/* ------------------------------------------------- 2 基准形态：逐行 + 手算 */
{
  kicker: '形态 ① · 加权版',
  title: '五行数学：<span class="hl-a">升精度 → 求标量 → rsqrt → 乘回去</span>',
  sub: '先手算一个 4 维的例子，再回头读那五行。注意归约之后剩下什么形状 —— 它决定了这个算子能不能分块。',
  caption: '回顾 L0-01：hidden_states 是 (batch, seq, 4096)；这一行 mean(-1) 把最后那维压成一个数。',
  lang: 'python',
  codeStart: 76,
  code: `    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"`,
  codeNote: 'Glm5NextTextRMSNorm.forward —— 三个精度约定全在这五行里。',
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '手算：一个 4 维 token（真实代码走的路径，数字逐步算出来）' }));

    const tb = W.table([
      ['x', '<span class="mono">[3, 4, 0, 0]</span>', 'RMS = 2.5', 0],
      ['x.pow(2)', '<span class="mono">[9, 16, 0, 0]</span>', '逐元素平方', 0],
      ['.mean(-1, keepdim=True)', '<span class="hl">6.25</span>', '形状 (b, s, 1) —— 每个 token 只剩一个标量', 1],
      ['+ variance_epsilon', '<span class="mono">6.25001</span>', 'eps = 1e-5（config 值，不是类默认的 1e-6）', 2],
      ['torch.rsqrt(...)', '<span class="hl">0.39999968</span>', '= 1 / 2.5000016，开方与取倒数一次算完', 2],
      ['x * that', '<span class="mono">[1.19999904, 1.59999872, 0, 0]</span>', 'RMS 从 2.5 归一到 1', 3],
      ['* weight（初始全 1）', '<span class="mono">[1.19999904, 1.59999872, 0, 0]</span>', '逐通道乘 —— 这就是「方向自由度」', 4],
    ], { head: ['步骤', '值', '说明'] });
    const tbCard = U.el('div', { class: 'card', cc: 0, style: 'padding:9px 12px' }, [tb]);
    viz.appendChild(tbCard);

    const CONV = [
      { cc: 4, t: '① 无条件升 fp32', s: 'bf16 只有 8 位尾数，4096 项求和的舍入误差会随维度累积；升到 fp32 之后可以忽略' },
      { cc: 2, t: '② 归约后只剩 (b, s, 1)', s: '逐 token 独立 —— 这个性质决定了它可以沿 seq 任意分块（第 8 幕）' },
      { cc: 1, t: '③ 先降精度、再乘 weight', s: '加权乘法发生在输入精度里；门控版特意把这一步挪回 fp32（第 5 幕）' },
    ];
    const crow = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cc = CONV.map(d => {
      const c = W.card({ cc: d.cc, tint: d.cc, title: d.t, sub: d.s, style: 'flex:1' });
      crow.appendChild(c); return c;
    });
    viz.appendChild(crow);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    const rowEls = Array.from(tb.querySelectorAll('tbody tr'));
    rowEls.forEach(r => { r.style.opacity = '.25'; });
    cc.forEach(c => { c.style.opacity = '.25'; });
    msg.innerHTML = '<span class="cm">// 拿一个 4 维 token 走一遍真实代码</span>';
    const light = (k) => rowEls.forEach((r, i) => { r.style.opacity = (i === k) ? '1' : '.25'; });
    tl.at(1600, () => { light(0); msg.innerHTML = 'x = <em>[3, 4, 0, 0]</em> &nbsp;<span class="cm">// RMS = 2.5</span>'; });
    tl.at(4200, () => { light(2); msg.innerHTML = 'mean(x²) = <em>6.25</em> &nbsp;<span class="cm">// 4096 维也一样：每个 token 只留一个标量</span>'; });
    tl.at(7000, () => { light(3); msg.innerHTML = '6.25 + <em>1e-5</em> = 6.25001 &nbsp;<span class="cm">// eps 在这里只值小数点后第 5 位</span>'; });
    tl.at(9800, () => { light(4); msg.innerHTML = 'rsqrt(6.25001) = <em>0.39999968</em>'; });
    tl.at(12600, () => { light(5); msg.innerHTML = '输出 <em>[1.2, 1.6, 0, 0]</em> —— 尺度被拉到 1，方向没变'; });
    tl.at(15400, () => {
      rowEls.forEach(r => { r.style.opacity = '1'; });
      cc[0].style.opacity = '1'; cc[0].classList.add('ac');
      msg.innerHTML = '三个精度约定：<em>① 升 fp32 求均值</em>';
    });
    tl.at(18000, () => {
      cc[0].classList.remove('ac');
      cc[1].style.opacity = '1'; cc[2].style.opacity = '1';
      cc[1].classList.add('ac');
      msg.innerHTML = '② 归约成 (b,s,1) → 逐 token 独立；③ 先降精度再乘 weight（第 5 幕看它值多少）';
    });
  },
},

/* ------------------------------------------------- 3 无权重版 */
{
  kicker: '形态 ② · 无权重版',
  title: '★ <span class="hl4">0 个参数</span>的归一化：去掉的不是成本，是自由度',
  sub: 'Glm5NextTextUnweightedRMSNorm 的一个实例，parameters() 是空集合。它与加权版的关系只有一句话：weight 全 1 的加权版。',
  caption: '验收点一：「无权重版本用于只需要尺度、不需要方向的场合」—— 下一幕看那个场合具体在哪。',
  lang: 'python',
  codeStart: 211,
  code: `class Glm5NextTextUnweightedRMSNorm(nn.Module):
    def __init__(self, eps: float = 1.0e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)`,
  codeNote: '整个类只有两行有效代码：__init__ 存一个 eps，forward 一行算完。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '两种形态并排：差别只在 weight 那一排数字' }));

    const mkStrip = (cc, vals, on) => {
      const row = U.el('div', { class: 'row gap6', style: 'width:100%;margin:8px 0' });
      vals.forEach(v => {
        row.appendChild(U.el('div', {
          class: 'n sm',
          style: 'flex:1;height:40px;border-radius:8px;display:flex;align-items:center;'
               + 'justify-content:center;border:1px solid var(--c' + cc + ');background:'
               + (on ? 'rgba(52,211,153,.16)' : 'rgba(56,189,248,.13)') + ';color:#fff',
          text: v }));
      });
      return row;
    };
    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const left = W.card({ cc: 0, tint: 0, title: '加权版：weight 是可学的',
      sub: '每个通道一个乘数，训练之后不再是 1 —— 下游因此能拿到「非单位尺度」',
      body: U.el('div', {}, [mkStrip(0, ['1', '2', '1', '0.5', '1.5', '1', '0.75', '1'], false)]) });
    const right = W.card({ cc: 4, tint: 4, title: '无权重版：weight 恒为 1',
      sub: '实测 <code class="inl">parameters()</code> 为空，<code class="inl">numel() == 0</code>',
      body: U.el('div', {}, [mkStrip(4, ['1', '1', '1', '1', '1', '1', '1', '1'], true)]) });
    left.style.flex = '1'; right.style.flex = '1';
    row.appendChild(left); row.appendChild(right);
    viz.appendChild(row);

    const tb = W.table([
      ['parameters()', '<span class="hl">4096</span>', '<span class="hl4">0</span>'],
      ['weight = 1 时两者输出', '<span class="hl4">allclose = True</span>', '同一个式子'],
      ['Unweighted(x) × weight', '<span class="hl4">== Norm(x) 逐元素成立</span>', '只差一个乘数'],
      ['手算 x = [3,4,0,0]', '<span class="mono">[1.2, 1.6, 0, 0]</span>', 'weight=1 时两版输出相同'],
    ], { head: ['量', '加权版', '无权重版'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 4, style: 'padding:9px 12px' }, [tb]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    right.style.opacity = '.28';
    msg.innerHTML = '<span class="cm">// 左边是主干里最常见的形态</span>';
    tl.at(2200, () => { left.classList.add('ac'); msg.innerHTML = '加权版：<em>4096</em> 个可学乘数 <span class="cm">// 逐个通道放大或压小</span>'; });
    tl.at(5600, () => {
      left.classList.remove('ac'); right.style.opacity = '1'; right.classList.add('ac');
      msg.innerHTML = '无权重版：<em>0</em> 个参数 &nbsp;<span class="cm">// forward 一行：x * rsqrt(mean(x²) + eps)</span>';
    });
    tl.at(9200, () => {
      msg.innerHTML = '它不是「另一种归一化」，而是 <em>weight 全 1 的加权版</em> —— 实测 allclose 为 True';
    });
    tl.at(12600, () => {
      msg.innerHTML = '★ 去掉的不是参数（0.000001% 而已），是 <em>方向自由度</em>';
    });
    tl.at(15200, () => {
      msg.innerHTML = '<span class="cm">// 那么谁会需要一个「不想要自由度」的归一化？下一幕</span>';
    });
  },
},

/* ------------------------------------------------- 4 无权重版用在哪 */
{
  kicker: '形态 ② · 调用现场',
  title: '用在哪：<span class="hl-b">mHC 内部</span>，把 4 条流拼成 16384 维再对齐',
  sub: '90 个实例全部来自同一个构造点 —— 45 层 × (attn_hc, ffn_hc)。它归一化的不是 hidden_states，而是 4 条流拼接后的向量。',
  caption: '回顾 L0-01 第五节：hidden_states 一开始就是 (batch, seq, 4, 4096)。展开见 L3-06。',
  lang: 'python',
  codeStart: 276,
  code: `        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.input_norm = Glm5NextTextUnweightedRMSNorm(eps=config.rms_norm_eps)`,
  codeNote: '全模型唯一一处实例化无权重版的地方。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: 'input_norm 在 mHC 里的位置：先拼平、再对齐尺度、最后算混合系数' }));

    const stage = U.el('div', { class: 'row gap10 center', style: 'width:100%' });
    const streams = U.el('div', { class: 'col gap4' });
    for (let s = 0; s < 4; s++) {
      const t = W.tensor(1, 8, () => s, { cell: 15 });
      streams.appendChild(t);
    }
    const lcol = U.el('div', { class: 'col gap4 center', style: 'flex:none' }, [
      U.el('div', { class: 'klabel', text: 'hidden_streams (b,s,4,4096)' }), streams,
    ]);
    stage.appendChild(lcol);

    const arr = (txt) => U.el('div', { class: 'col center gap4', style: 'flex:none;padding:0 2px' }, [
      U.el('div', { style: 'font-size:17px;color:var(--accent)', text: '⟹' }),
      U.el('div', { class: 'mono', style: 'font-size:9.5px;color:var(--ink-faint);text-align:center', html: txt }),
    ]);
    stage.appendChild(arr('view(b,s,-1)'));
    const flat = U.el('div', { class: 'col gap4 center', style: 'flex:none' }, [
      U.el('div', { class: 'klabel', text: 'flattened (b,s,16384)' }),
      W.tensor(1, 12, () => 3, { cell: 14 }),
      U.el('div', { class: 'mono', style: 'font-size:9.5px;color:var(--c4)', text: '4 x 4096 = 16384 维' }),
    ]);
    stage.appendChild(flat);
    stage.appendChild(arr('input_norm'));
    const nm = W.card({ cc: 4, tint: 4, title: 'UnweightedRMSNorm', sub: '0 参数',
      style: 'flex:none;width:122px;text-align:center' });
    stage.appendChild(nm);
    stage.appendChild(arr('fn 投影'));
    const out = W.card({ cc: 1, tint: 1, title: 'pre / post / comb', sub: '混合系数',
      style: 'flex:none;width:122px;text-align:center' });
    stage.appendChild(out);
    viz.appendChild(stage);

    const STATS = [
      { cc: 4, k: '90 个实例', s: '45 层 x (attn_hc / ffn_hc)' },
      { cc: 2, k: '16384 维', s: '4 条 4096 的流拼接之后' },
      { cc: 3, k: '省下 1,474,560', s: '若给它加权重：90 x 16384 个参数（0.00047%）' },
    ];
    const srow = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const stats = STATS.map(st => {
      const c = W.card({ cc: st.cc, title: st.k, sub: st.s, style: 'flex:1;text-align:center' });
      srow.appendChild(c); return c;
    });
    viz.appendChild(srow);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 归一化的对象不是一条 4096 维的 hidden，而是 4 条流的拼接</span>';
    tl.at(2000, () => { nm.classList.add('ac'); msg.innerHTML = '先把 <em>4 x 4096</em> 拼成一条 <em>16384</em> 维向量，再一次性对齐尺度'; });
    tl.at(5400, () => {
      nm.classList.remove('ac'); out.classList.add('ac');
      msg.innerHTML = '对齐之后只做一件事：算出 <em>pre / post / comb</em> 三组混合系数';
    });
    tl.at(8800, () => {
      out.classList.remove('ac'); stats[0].classList.add('ac');
      msg.innerHTML = '全模型 <em>90</em> 个无权重 norm 全部来自这一个构造点';
    });
    tl.at(12200, () => {
      stats[0].classList.remove('ac'); stats[2].classList.add('ac');
      msg.innerHTML = '若换成加权版会多出 <em>1,474,560</em> 个参数 —— 但这只占 0.00047%';
    });
    tl.at(15000, () => {
      stats.forEach(s => s.classList.remove('ac'));
      msg.innerHTML = '★ 所以真正的理由是：<em>它的输出不进主干计算，方向自由度没有意义</em>';
    });
  },
},

/* ------------------------------------------------- 5 门控版 forward */
{
  kicker: '形态 ③ · 门控版',
  title: '<span class="hl-b">sigmoid(gate)</span>：在 128 维上逐通道开合',
  sub: '门控版 = 加权版 + 一个与输入相关的门。它多出来的两处 fp32 不是洁癖 —— 实测有 19.04% 的元素会因为省掉它们而变值。',
  caption: '验收点二的答案在这一幕与下一幕：门控范数给 KDA 的读出加了「与内容相关的闸门」。',
  lang: 'python',
  codeStart: 374,
  code: `# NOTE: the FLA package does not re-cast to \`input_dtype\` in its implementation, maybe we should do the same
@use_kernel_forward_from_hub("RMSNormGated")
class Glm5NextTextRMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, **kwargs) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.activation = "sigmoid"

    def forward(self, hidden_states, gate=None) -> torch.Tensor:
        input_dtype = hidden_states.dtype

        # Strict FP32 norm (do not downcast on the weights)
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight.to(torch.float32) * hidden_states

        # Apply gating
        hidden_states = hidden_states * ACT2FN[self.activation](gate.to(torch.float32))

        return hidden_states.to(input_dtype)`,
  codeNote: '注意 self.activation = "sigmoid" 是硬编码字符串，不是 config 项；两处 .to(torch.float32) 是刻意的。',
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '一个 head 的 128 维：每个通道一个 0~1 的闸门（实测范围 0.219 ~ 0.803）' }));

    const row = U.el('div', { class: 'row gap14', style: 'width:100%;align-items:stretch' });
    const mtx = W.matrix(6, 16, (r, c) => {
      const g = 0.5 + 0.16 * Math.sin(r * 2.7 + c * 1.3) + 0.08 * Math.cos(c * 0.9 - r * 1.7);
      return Math.max(0.15, Math.min(0.85, g));
    }, { cell: 12 });
    const mcard = W.card({ cc: 1, tint: 1, title: 'sigmoid(gate) 的分布',
      sub: '越亮 = 门开得越大。<span class="mono">(b, s, head, 128)</span> 上每个元素一个系数',
      body: mtx, style: 'flex:none' });
    row.appendChild(mcard);

    const CALC = [
      ['gate = +2.0', 'sigmoid = <span class="hl4">0.8808</span>', '归一化后的 1.6 → <span class="mono">1.4093</span>'],
      ['gate = 0.0', 'sigmoid = <span class="hl3">0.5000</span>', '归一化后的 1.6 → <span class="mono">0.8000</span>'],
      ['gate = -2.0', 'sigmoid = <span class="hlbad">0.1192</span>', '归一化后的 1.6 → <span class="mono">0.1907</span>'],
    ];
    const tb = W.table(CALC, { head: ['某个通道的门', '门开度', '该通道的输出'] });
    const ccard = W.card({ cc: 4, tint: 4, title: '手算：门如何改变一个通道',
      sub: '归一化只能把尺度拉回 1；门才能把某一维压到接近 0',
      body: tb, style: 'flex:1' });
    row.appendChild(ccard);
    viz.appendChild(row);

    const CONV = [
      { cc: 0, t: '① weight 留在 fp32',
        s: 'self.weight.to(torch.float32) * hidden_states；注释写明 Strict FP32 norm (do not downcast on the weights)' },
      { cc: 2, t: '② gate 也升 fp32',
        s: 'ACT2FN["sigmoid"](gate.to(torch.float32))，最后整个结果一次性 .to(input_dtype)' },
      { cc: 3, t: '③ 实测代价：19.04%',
        s: '改成「先降 bf16 再乘 weight」，1,048,576 个元素里 199,675 个结果不同，最大相对差 6.7e-3（约 1.7 个 bf16 ulp）' },
    ];
    const crow = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cc = CONV.map(d => {
      const c = W.card({ cc: d.cc, tint: d.cc, title: d.t, sub: d.s, style: 'flex:1' });
      crow.appendChild(c); return c;
    });
    viz.appendChild(crow);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    mcard.classList.add('ac');
    msg.innerHTML = '<span class="cm">// 初始化时 sigmoid(gate) 均值 0.5000 —— 整体乘 0.5，被 o_proj 吸收</span>';
    tl.at(2200, () => { msg.innerHTML = '门是逐通道的：<em>128</em> 维里每一维有自己的开度'; });
    tl.at(5600, () => {
      mcard.classList.remove('ac'); ccard.classList.add('ac');
      msg.innerHTML = 'gate = -2 时这一维被乘 <em>0.1192</em> —— 相当于把这个通道关掉';
    });
    tl.at(9000, () => {
      ccard.classList.remove('ac'); cc[0].classList.add('ac');
      msg.innerHTML = '与加权版的第一处差别：<em>weight 乘法留在 fp32</em>';
    });
    tl.at(12400, () => {
      cc[0].classList.remove('ac'); cc[1].classList.add('ac');
      msg.innerHTML = '第二处：<em>gate 也升 fp32</em>，最后一次性降回 input_dtype';
    });
    tl.at(15600, () => {
      cc[1].classList.remove('ac'); cc[2].classList.add('ac');
      msg.innerHTML = '实测：省掉这两处，<em>19.04%</em> 的元素会变值，最大相对差 6.7e-3';
    });
    tl.at(18200, () => {
      cc.forEach(c => c.classList.remove('ac'));
      msg.innerHTML = '<span class="cm">// KDA 每层只有 128 个权重，多留一份 fp32 几乎不花钱</span>';
    });
  },
},

/* ------------------------------------------------- 6 门控解决什么 */
{
  kicker: '形态 ③ · 为什么需要它',
  title: '★ 门控范数：用<span class="hl-a">当下的输入</span>决定要不要相信刚读出的记忆',
  sub: 'core_attn_out 来自递推状态 S 的读出，gate 来自当前 hidden_states。两路汇合在 o_norm —— 这就是 KDA 需要门控的全部理由。',
  caption: '这一块回答验收点二。KDA 的递推展开见 L4-02 ~ L4-05；门的瓶颈宽度为什么取 head_dim 也在那里讲。',
  lang: 'python',
  codeStart: 766,
  code: `        # Final gated norm and proj
        gate = self.g_b_proj(self.g_a_proj(hidden_states)).view(hidden_shape)
        output = self.o_norm(core_attn_out, gate).reshape(batch_size, seq_len, -1)
        output = self.o_proj(output)`,
  codeNote: '两路输入、一个出口：core_attn_out（记忆读出）× sigmoid(gate)（当前输入算出的闸门）。',
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap10 vizgrow hstart' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: 'o_norm 的两路输入（gate 支路与 KDA 支路并行计算，不增加串行深度）' }));

    const top = W.card({ cc: 0, tint: 0, title: 'hidden_states', sub: '(b, s, 4096)',
      style: 'width:300px;text-align:center;align-self:center' });
    viz.appendChild(top);

    const two = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const gateBox = W.card({ cc: 2, tint: 2, title: 'gate 支路（瓶颈）',
      sub: '<span class="mono">g_a_proj 4096→128</span><br><span class="mono">g_b_proj 128→8192</span>',
      body: U.el('div', { class: 'mono', style: 'font-size:11px;color:var(--c2)', text: '1.57M 参数（直连要 33.5M）' }),
      style: 'flex:1' });
    const kdaBox = W.card({ cc: 1, tint: 1, title: 'KDA 支路（递推）',
      sub: '<span class="mono">q/k/v → 卷积 → S_t = S_{t-1} ⊙ decay + write</span>',
      body: U.el('div', { class: 'mono', style: 'font-size:11px;color:var(--c1)', text: 'core_attn_out：从状态 S 读出' }),
      style: 'flex:1' });
    two.appendChild(gateBox); two.appendChild(kdaBox);
    viz.appendChild(two);

    const mid = U.el('div', { class: 'row gap10 center', style: 'width:100%' });
    mid.innerHTML = '<span class="mono" style="color:var(--ink-faint);font-size:11px">两路汇合</span>'
      + '<span style="font-size:16px;color:var(--accent)">▼</span>';
    viz.appendChild(mid);

    const on = W.card({ cc: 3, tint: 3, title: 'o_norm(core_attn_out, gate)',
      sub: '逐 head 归一化 + 逐通道 sigmoid 门 → <span class="mono">o_proj</span> 回到 4096',
      style: 'width:520px;text-align:center;align-self:center' });
    viz.appendChild(on);

    const WHY = [
      { cc: 2, t: '尺度漂移', s: 'S 的幅度随递推步数累积：写得多、衰减得慢，读出就大。RMSNorm 只能拉回尺度，拉不回「这一维该不该有值」' },
      { cc: 4, t: '基准版做不到', s: '加权版的 weight 是与输入无关的常数 —— 它无法对「这一次的读出」说不要' },
    ];
    const wrow = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    WHY.forEach(d => wrow.appendChild(W.card({ cc: d.cc, tint: d.cc, title: d.t, sub: d.s, style: 'flex:1' })));
    viz.appendChild(wrow);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    /* 数据包：两路各发一个，汇到 o_norm */
    FLOW.attach(U.q('#visual'));
    FLOW.pathBetween('g', gateBox, on);
    FLOW.pathBetween('k', kdaBox, on);
    FLOW.send('k', { at: 3200, dur: 1100, color: '167,139,250', label: 'core_attn_out' });
    FLOW.send('g', { at: 4600, dur: 1100, color: '251,191,36', label: 'gate' });

    msg.innerHTML = '<span class="cm">// 两条支路都只依赖 hidden_states —— 可以并行算</span>';
    tl.at(1800, () => { gateBox.classList.add('ac'); msg.innerHTML = 'gate 支路：<em>4096 → 128 → 8192</em> 的瓶颈，最终给出逐通道的闸门'; });
    tl.at(5200, () => {
      gateBox.classList.remove('ac'); kdaBox.classList.add('ac');
      msg.innerHTML = 'KDA 支路：读出的 <em>core_attn_out</em> 尺度会随递推步数累积';
    });
    tl.at(8600, () => {
      kdaBox.classList.remove('ac'); on.classList.add('ac');
      msg.innerHTML = '两路在 <em>o_norm</em> 汇合：归一化管尺度，门管「要不要」';
    });
    tl.at(12000, () => { msg.innerHTML = '★ 门 = <em>用当下的输入，决定要不要相信刚刚读出的记忆</em>'; });
    tl.at(15200, () => {
      wrow.children[0].classList.add('ac');
      msg.innerHTML = '一个已经写坏的通道，归一化之后照样输出单位尺度 —— 只有门能把它压到 0.12';
    });
    tl.at(18000, () => {
      wrow.children[0].classList.remove('ac'); wrow.children[1].classList.add('ac');
      msg.innerHTML = '基准版做不到这件事：它的 weight 是<em>与输入无关的常数</em>';
    });
  },
},

/* ------------------------------------------------- 7 eps */
{
  kicker: '数值 · eps',
  title: 'eps：类默认 <span class="hl-a">1e-6</span>，模型实际用 <span class="hl-b">1e-5</span>',
  sub: '三个类的默认值都是 1e-6，但 12 个实例化点全部显式传 config.rms_norm_eps —— 默认值在这个模型里从不生效。',
  caption: '回顾 L1-02 / L1-03：rms_norm_eps 就写在 config 里（1e-5）。configuration_glm5_next.py 不计入本课覆盖率，只作实测证据。',
  lang: 'python',
  codeStart: 1435,
  code: `        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Glm5NextTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Glm5NextTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.hc_head = Glm5NextTextHyperHead()`,
  codeNote: '出口的 norm 也显式传了 eps=config.rms_norm_eps —— 位置参数与关键字参数两种写法，传的是同一个东西。',
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: 'eps 的物理含义：放大上限 = 1/sqrt(eps)' }));

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const c5 = W.card({ cc: 0, tint: 0, title: 'eps = 1e-5（模型实际用的）',
      body: U.el('div', { class: 'n big', text: '316 倍' }),
      sub: '放大上限 1/sqrt(1e-5)。门槛 sqrt(eps) = 3.162e-3：token RMS 低于它才开始受影响', style: 'flex:1' });
    const c6 = W.card({ cc: 5, tint: 5, title: 'eps = 1e-6（三个类的默认值）',
      body: U.el('div', { class: 'n big', text: '1000 倍' }),
      sub: '放大上限 1/sqrt(1e-6)，门槛 1.000e-3。默认值在本模型里从不生效', style: 'flex:1' });
    row.appendChild(c5); row.appendChild(c6);
    viz.appendChild(row);

    const tb = W.table([
      ['RMS = 1', '<span class="mono">4.552e-06</span>', '收缩 0.0005%', '正常 token 的量级'],
      ['RMS = 0.1', '<span class="mono">4.496e-04</span>', '收缩 0.0500%', ''],
      ['RMS = 0.01', '<span class="mono">4.178e-02</span>', '收缩 4.6537%', '开始能看出来'],
      ['RMS = 1e-3', '<span class="mono">5.736e-01</span>', '收缩 69.8489%', '接近全零的 token'],
    ], { head: ['该 token 的 RMS', '两种 eps 的最大相对差', 'eps 造成的收缩', '说明'] });
    const tbCard = U.el('div', { class: 'card', cc: 3, style: 'padding:9px 12px' }, [tb]);
    viz.appendChild(tbCard);

    const bf = W.card({ cc: 2, tint: 2, title: 'bf16 下还有区别吗',
      sub: '实测 1,048,576 个元素里只有 <span class="hl3">1,017 个</span>（0.10%）不同。'
         + 'bf16 的相对精度 2^-8 = 3.9e-3，比两个候选 eps 的差别（4.6e-6）大三个数量级 —— '
         + '<b>调 eps 不是精度旋钮，是数值防御</b>',
      style: 'width:100%' });
    viz.appendChild(bf);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// out = x * rsqrt(mean(x^2) + eps)</span>';
    tl.at(2000, () => { c5.classList.add('ac'); msg.innerHTML = '1e-5 把最坏情况下的放大压到 <em>316 倍</em>（1e-6 是 1000 倍）'; });
    tl.at(5400, () => {
      c5.classList.remove('ac'); c6.classList.add('ac');
      msg.innerHTML = '但三个类的默认值 <em>1e-6</em> 在本模型里从不生效：12 个实例化点全部显式传 config 值';
    });
    tl.at(8800, () => {
      c6.classList.remove('ac'); tbCard.classList.add('ac');
      msg.innerHTML = '门槛 <em>sqrt(eps) = 3.162e-3</em>：只有 RMS 掉到这个量级以下，eps 才看得见';
    });
    tl.at(12200, () => {
      tbCard.classList.remove('ac'); bf.classList.add('ac');
      msg.innerHTML = 'bf16 下两者有 <em>99.9%</em> 的元素完全一样 —— 实测 1017 / 1048576';
    });
    tl.at(15600, () => {
      bf.classList.remove('ac');
      msg.innerHTML = '★ 常量输入 1e-6 时输出 3.16e-4 —— 正好是 1/sqrt(1e-5) 倍的放大上限';
    });
    tl.at(18200, () => {
      msg.innerHTML = '<span class="cm">// 真正会爆的是 eps = 0，而这里没有这个选项</span>';
    });
  },
},

/* ------------------------------------------------- 8 分布与代价 */
{
  kicker: '横切 · 分布与代价',
  title: '自研 norm 的<span class="hl-a">代价</span>与<span class="hl4">红利</span>',
  sub: '代价：237 个 norm 不在 transformers 的归一化登记表里，通用逻辑照顾不到。红利：逐 token 独立，所以可以沿 seq 任意分块。',
  caption: '这一幕用到 pytorch_utils.py 的 ALL_LAYERNORM_LAYERS 与 apply_chunking_to_forward（同属本课覆盖的两个文件之一）。',
  lang: 'python',
  codeStart: 1295,
  code: `        self.input_layernorm = Glm5NextTextRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Glm5NextTextRMSNorm(config.hidden_size, config.rms_norm_eps)`,
  codeNote: '每个解码层两个加权 norm —— 90 个实例就是这么来的（45 x 2）。',
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '237 个实例的构成（meta device 实测）' }));

    const row = U.el('div', { class: 'row gap14', style: 'width:100%;align-items:stretch' });
    const bars = W.bars([
      { label: 'Glm5NextTextRMSNorm', value: 113, valueText: '113', cc: 0 },
      { label: 'UnweightedRMSNorm', value: 90, valueText: '90', cc: 4 },
      { label: 'RMSNormGated', value: 34, valueText: '34', cc: 1 },
    ], { max: 113, cc: 0 });
    const bcard = W.card({ cc: 0, title: '三类实例数', sub: '合计 237', body: bars, style: 'flex:1' });
    row.appendChild(bcard);

    const reg = W.card({ cc: 3, tint: 3, title: '代价：不在登记表里',
      sub: 'pytorch_utils.py 的 <span class="mono">ALL_LAYERNORM_LAYERS = [nn.LayerNorm]</span>；'
         + '实测与模型里三个 norm 类型的<b>交集为空</b> —— '
         + '权重初始化、weight decay 分组、量化登记都要自己写（第 5 幕那行 init.ones_ 就是证据）',
      style: 'flex:1' });
    row.appendChild(reg);
    viz.appendChild(row);

    const tb = W.table([
      ['沿 seq(dim=1) 分块', '<span class="hl4">逐位相同（最大绝对差 0.0）</span>', '逐 token 独立'],
      ['沿 hidden(dim=-1) 分块', '<span class="hlbad">RuntimeError</span>', 'weight(64) 与切出的 16 维对不上'],
      ['无权重版沿 hidden 分块', '<span class="hlbad">不报错，但结果错 27.8%</span>', '少一个参数 = 少一道保险'],
      ['沿 seq 做 softmax 再分块', '<span class="hlbad">结果错（最大绝对差 0.888）</span>', '跨 token 的算子不能这么切'],
    ], { head: ['用 apply_chunking_to_forward 切', '与整块相比', '原因'] });
    const tbCard = U.el('div', { class: 'card', cc: 4, style: 'padding:9px 12px' }, [tb]);
    viz.appendChild(tbCard);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    tbCard.classList.add('ac');
    msg.innerHTML = '<span class="cm">// 先看分布，再看自研的代价与红利</span>';
    tl.at(1800, () => { reg.classList.add('ac'); msg.innerHTML = '登记表里只有 nn.LayerNorm，而 GLM-5 的 237 个 norm <em>一个都没登记</em>'; });
    tl.at(5200, () => {
      reg.classList.remove('ac'); bcard.classList.add('ac');
      msg.innerHTML = '红利：归一化只在最后一维归约 → <em>逐 token 独立</em>';
    });
    tl.at(8600, () => {
      bcard.classList.remove('ac'); tbCard.classList.add('ac');
      msg.innerHTML = '实测沿 seq 分块与整块<em>逐位相同</em>（最大绝对差 0.0）—— 可以随便切、可以融合';
    });
    tl.at(12000, () => {
      msg.innerHTML = '反例：沿 seq 的 softmax 再沿 seq 分块，结果就错了 —— 它跨 token 混合';
    });
    tl.at(15200, () => {
      msg.innerHTML = '★ 一层里唯一能随便切、能融合、能逐块流式算的，就是归一化这一步';
    });
  },
},

/* ------------------------------------------------- 9 收束 */
{
  kicker: '收束',
  title: '四个类，三种形态，一张表',
  sub: '视觉塔里那个同名的 Glm5NextRMSNorm，去掉类名与 docstring 后与加权版 17 行逐字相同 —— 数类名是 4，数式子是 3。',
  caption: '下一课 L3-02：SwiGLU 的 gate/up/down 与 swiglu_limit 截断 —— 又一个「限制幅度」的设计。',
  lang: 'python',
  codeStart: 1558,
  code: `@use_kernel_forward_from_hub("RMSNorm")
class Glm5NextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Glm5NextRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps`,
  codeNote: '第四个类：与 Glm5NextTextRMSNorm 逐字相同的那 17 行（实测 a == b 为 True）。',
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const tb = W.table([
      ['加权 <span class="mono dim">x113</span>', '<span class="hl">4096</span>', '子层入口/出口、q_a、kv_a、主干出口', '下游要方向自由度'],
      ['无权重 <span class="mono dim">x90</span>', '<span class="hl4">0</span>', 'mHC 的 input_norm（4 流拼接后）', '下游只要尺度'],
      ['门控 <span class="mono dim">x34</span>', '<span class="hl2">128 + σ(gate)</span>', 'KDA 的 o_norm（逐 head）', '要在逐通道上关掉记忆'],
      ['视觉塔同名类', '<span class="mono">4096</span>', 'Glm5NextRMSNorm', '与加权版 17 行逐字相同'],
    ], { head: ['形态 / 实例数', 'weight', '用在哪', '一句话理由'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 0, style: 'padding:10px 12px' }, [
      U.el('div', { class: 'klabel', text: '本课总表（数字全部来自 probe_L3-01.py 实测）' }), tb,
    ]));

    viz.appendChild(W.exercise(
      '① <code class="inl">Glm5NextTextUnweightedRMSNorm</code> 用在哪、为什么不用加权版？'
      + '② 门控范数在 KDA 里解决什么问题？'
      + '③ 模型实际跑的 <code class="inl">eps</code> 是 1e-5 还是 1e-6？',
      '① mHC 的 <code class="inl">input_norm</code>（90 个实例）：它归一化的是 4 条流拼接后的 16384 维向量，'
      + '输出只用来算 <code class="inl">pre/post/comb</code> 混合系数、不进主干计算 —— '
      + '所以只需要「尺度」不需要「方向」，<code class="inl">parameters()</code> 实为空集。'
      + '<br>② KDA 的读出 <code class="inl">core_attn_out</code> 尺度随递推步数累积；'
      + 'RMSNorm 只能把尺度拉回 1，<b>拉不回「这一维该不该有值」</b>。'
      + '门控给每个通道一个 0~1 的闸门（<code class="inl">sigmoid(gate)</code>，实测 init 均值 0.5），'
      + '可以把写坏的通道压到接近 0 —— 这是与输入无关的常数 weight 做不到的。'
      + '<br>③ <b>1e-5</b>：三个类的默认值是 1e-6，但 12 个实例化点全部显式传 '
      + '<code class="inl">config.rms_norm_eps</code>，默认值从不生效。'));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 三种形态 = 三种「下游要什么」</span>';
    tl.at(2000, () => { msg.innerHTML = '归一化都把尺度拉回 1；差别只在<em>拉回之后还给不给自由度</em>'; });
    tl.at(5600, () => { msg.innerHTML = 'eps 决定最坏情况放大多少倍：<em>316</em>（1e-5）vs 1000（1e-6）；bf16 下 99.9% 元素无差别'; });
    tl.at(9200, () => { msg.innerHTML = '★ 逐 token 独立 ⇒ 归一化是可分块、可融合的那一步'; });
    tl.at(12800, () => { msg.innerHTML = '<span class="cm">// 下一课 L3-02：MLP 与 SwiGLU 截断 —— 另一个「限制幅度」的设计</span>'; });
    tl.at(16000, () => { msg.innerHTML = '<span class="cm">// 再往后 L3-06：mHC 的 pre / post / comb 具体怎么算出来</span>'; });
  },
},

];
