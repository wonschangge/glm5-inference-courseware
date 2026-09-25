/* ==========================================================================
   L3-02 · MLP 与 SwiGLU 截断
   --------------------------------------------------------------------------
   覆盖：models/glm5_next/modeling_glm5_next.py + activations.py（2 个文件 / 2813 行）
   目标：看完能写出 SwiGLU 的前向公式、手算一次截断，并说清 10.0 这把钳子
        在数值上挡住了什么、代价是什么。
   排版基准：#visual 可视区 = 994 x 662.7 舞台像素（无头浏览器实测）。
   本文件里的所有代码块都是 source.md 里引用过的逐字片段（由脚本从源文件切片注入）。
   ========================================================================== */
'use strict';

/* 实测常量（用仓库自带的 .venv/bin/python 跑出，不是估算） */
const K = {
  hidden: 4096, denseInter: 12288, moeInter: 2048,
  experts: 288, topk: 8, shared: 1,
  limit: 10.0,
  denseParams: 150994944,          // 3 × 4096 × 12288
  moeParams: 7274102784,           // 一个稀疏层的全部 MLP 相关参数（实测）
  actParams: 226492416,            // 每 token 真正参与计算的部分
  rawMax: 2419.451,                // 未截断时 max|SiLU(gate)·up|（std=18，4000 样本）
  clampedMax: 99.995,              // 截断后同一个量
  initMaxGate: 6.44,               // 默认初始化下实测到的 max|gate|
  actKeys: 24,                     // len(ACT2CLS)
};

/* 代码栏标题跟着当前幕的源文件走（SHELL.boot 只设一次，这里按幕覆盖） */
function setName(n) { const e = U.q('#sideName'); if (e) e.textContent = n; }

const SCENES = [

/* ------------------------------------------------- 1 全景：最后一个算子 */
{
  kicker: '第 3 层 · 文本主干',
  title: '一层里<span class="hl-a">最后一个算子</span>：前馈网络',
  sub: '注意力换的是「token 之间怎么通信」，MLP 换的是「每个 token 自己的内容」。本课只讲后者。',
  caption: '回顾 L0-01：层内顺序是「mHC 混合 → 归一化 → 子层 → mHC 写回」，这里讲第二个子层。',
  lang: 'python',
  codeStart: 1341,
  code: `        residual = hidden_states
        post, comb, hidden_states = self.ffn_hc(hidden_states)
        # Feed forward
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)`,
  codeNote: 'Glm5NextTextDecoderLayer.forward 的后半段 —— self.mlp 是这一层里最后一个换内容的算子。',
  duration: 17000,
  build(root, tl) {
    setName('modeling_glm5_next.py');
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap14 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '一层之内的两个子层 —— 本课讲后面那个' }));
    const fl = W.flow([
      { t: 'attn_hc', s: '4 条流 → 1', cc: 5 },
      { t: '注意力', s: 'KDA / MLA', cc: 0 },
      { t: '写回', s: 'post ⊙ + combᵀ', cc: 5 },
      { t: 'ffn_hc', s: '4 条流 → 1', cc: 5 },
      { t: 'MLP', s: '4096 → 4096', cc: 4 },
      { t: '写回', s: 'post ⊙ + combᵀ', cc: 5 },
    ], { style: 'width:100%' });
    viz.appendChild(fl);

    const stats = [
      { k: '进去的形状', v: '(b, s, 4096)', cc: 0, s: 'hidden_size' },
      { k: '出来的形状', v: '(b, s, 4096)', cc: 0, s: '一个都不改' },
      { k: '稠密 MLP 中间维', v: '12288', cc: 4, s: '= 3 × 4096' },
      { k: '截断阈值', v: '10.0', cc: 2, s: 'swiglu_limit' },
    ];
    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const statEls = stats.map(st => {
      const c = W.card({ cc: st.cc, tint: st.cc, title: '<span class="mono" style="font-size:11px">' + st.k + '</span>',
        body: U.el('div', { class: 'n big', style: 'font-size:17px', text: st.v }),
        sub: st.s, style: 'flex:1' });
      row.appendChild(c); return c;
    });
    viz.appendChild(row);

    const qs = [
      { cc: 4, q: '① 为什么是两个投影？', a: 'gate 当闸门、up 当内容；乘法把两条线性变换耦合起来' },
      { cc: 2, q: '② 为什么要钳位？', a: '10.0 把中间激活按在 100 以内，越界处的梯度直接为 0' },
      { cc: 1, q: '③ 激活从哪来？', a: 'ACT2FN[config.hidden_act] —— 一张 24 项的注册表' },
    ];
    const qrow = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    qs.forEach(x => qrow.appendChild(W.card({ cc: x.cc, title: x.q, sub: x.a, style: 'flex:1' })));
    viz.appendChild(qrow);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 第 1345 行：这一层最后做的事就是把 hidden 交给 self.mlp</span>';
    tl.at(2000, () => { fl.focus(4); msg.innerHTML = 'MLP：<em>两个投影 + 一次门控乘法 + 一个投影</em>，形状进出不变'; });
    tl.at(5200, () => { fl.focus(3); msg.innerHTML = '它前面的 <span class="fn">ffn_hc</span> 把 4 条残差流压成 1 条 —— 所以 MLP 看到的是 3 维张量'; });
    tl.at(8400, () => {
      fl.focus(-1); statEls[2].classList.add('ac'); statEls[3].classList.add('ac');
      msg.innerHTML = '中间维 <em>12288</em>、阈值 <em>10.0</em>：本课要拆的就是这两个数';
    });
    tl.at(11800, () => {
      statEls.forEach(e => e.classList.remove('ac'));
      msg.innerHTML = '<span class="cm">// 回顾 L0-01：self.mlp 是个二选一的槽位 —— 前 3 层稠密，其余 42 层稀疏</span>';
    });
    tl.at(14200, () => { msg.innerHTML = '稀疏层里的<em>共享专家</em>也是同一个类，所以这一课讲的东西在 45 层里都成立'; });
  },
},

/* ------------------------------------------------- 2 三个投影 */
{
  kicker: '结构 · 三投影',
  title: '三个投影：<span class="hl-a">gate</span> / <span class="hl-c">up</span> / <span class="hl-b">down</span>',
  sub: '两个把 4096 抬到 12288（一个当闸门、一个当内容），一个把 12288 压回 4096。三个都不带偏置。',
  caption: 'bias=False 是参数量能口算的原因：3 × 4096 × 12288 = 150,994,944。',
  lang: 'python',
  codeStart: 87,
  code: `class Glm5NextTextMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.swiglu_limit = config.swiglu_limit`,
  codeNote: 'Glm5NextTextMLP.__init__ —— 三个投影 + 一个激活实例 + 一个阈值，共 11 行。',
  duration: 17000,
  build(root, tl) {
    setName('modeling_glm5_next.py');
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap14 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '三个投影的形状与分工' }));
    const defs = [
      { cc: 4, t: 'gate_proj', shape: '4096 → 12288', role: '闸门：经过 SiLU 后决定「放多少过去」', tint: 4 },
      { cc: 2, t: 'up_proj', shape: '4096 → 12288', role: '内容：决定「过去的是什么」', tint: 2 },
      { cc: 1, t: 'down_proj', shape: '12288 → 4096', role: '投回主干：把 12288 维求和回 4096 维', tint: 1 },
    ];
    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cards = defs.map(d => {
      const c = W.card({ cc: d.cc, tint: d.tint,
        title: '<span class="mono" style="font-size:12px">' + d.t + '</span>',
        body: U.el('div', { class: 'col gap6' }, [
          U.el('div', { class: 'n', style: 'font-size:15px;color:#fff', text: d.shape }),
          U.el('div', { class: 'cs', text: d.role }),
        ]),
        sub: 'bias=False',
        style: 'flex:1' });
      row.appendChild(c); return c;
    });
    viz.appendChild(row);

    const tb = W.table([
      ['每个投影', '<span class="hl">4096 × 12288</span>', '50,331,648'],
      ['三个合计', '<span class="hl">150,994,944</span>', '≈ 151.0M（实测 sum(p.numel())）'],
      ['偏置', '<span class="hl">0</span>', 'bias=False，一个都没有'],
      ['中间维', '<span class="hl">3 ×</span>', '12288 = 3 × 4096'],
    ], { head: ['量', '值', '说明'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 0, style: 'padding:10px 12px' }, [tb]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 三个 nn.Linear，都是 bias=False</span>';
    tl.at(2200, () => { cards[0].classList.add('ac'); msg.innerHTML = '<span class="fn">gate_proj</span>：闸门那一份，它会被 SiLU 压到 (-0.2785, +∞)'; });
    tl.at(5400, () => {
      cards[0].classList.remove('ac'); cards[1].classList.add('ac');
      msg.innerHTML = '<span class="fn">up_proj</span>：内容那一份，两侧都无界 —— 所以它需要<em>双侧</em>钳位';
    });
    tl.at(8800, () => {
      cards[1].classList.remove('ac'); cards[2].classList.add('ac');
      msg.innerHTML = '<span class="fn">down_proj</span>：12288 维<em>求和</em>回 4096 —— 中间激活的量级直接决定这个和的范围';
    });
    tl.at(12200, () => {
      cards.forEach(c => c.classList.remove('ac'));
      msg.innerHTML = '中间维 <em>12288 = 3 × 4096</em>：门控多了一个矩阵，所以宽度取 3 倍而不是 4 倍';
    });
    tl.at(14600, () => {
      msg.innerHTML = '第 92 行 <em>intermediate_size</em> 可被构造参数覆盖 —— 共享专家就是靠这个钩子复用的';
    });
  },
},

/* ------------------------------------------------- 3 ★ SwiGLU 前向 */
{
  kicker: '★ 核心 · 前向',
  title: '★ 一行 SwiGLU：<span class="hl-a">SiLU(gate) ⊙ up</span>',
  sub: '门控乘法的意思是：gate 决定放多少，up 决定放什么。乘法把它们耦合起来，比单个逐元素非线性强。',
  caption: '这就是「SwiGLU」这个词的全部内容：Swish(SiLU) + Gated Linear Unit。',
  lang: 'python',
  codeStart: 99,
  code: `    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # Key difference using clamping
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)`,
  codeNote: 'Glm5NextTextMLP.forward —— 全部 7 行，真正做事的只有最后一行。',
  duration: 18000,
  build(root, tl) {
    setName('modeling_glm5_next.py');
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const fml = U.el('div', { class: 'formula', style: 'width:100%;border-left-color:var(--accent)' });
    fml.innerHTML = 'out = <span class="fn">W_down</span>( <em>SiLU</em>(clamp(W_gate·x, max = L)) '
      + '<span class="op">⊙</span> clamp(W_up·x, -L, +L) ) &nbsp;&nbsp;'
      + '<span class="cm">// L = swiglu_limit = 10.0，SiLU(z) = z·σ(z)</span>';
    viz.appendChild(fml);

    const fl = W.flow([
      { t: 'x', s: '(b, s, 4096)', cc: 0 },
      { t: 'gate / up', s: '两个投影', cc: 2 },
      { t: 'clamp', s: '10.0', cc: 3 },
      { t: 'SiLU ⊙', s: '逐元素相乘', cc: 4 },
      { t: 'down_proj', s: '(b, s, 4096)', cc: 1 },
    ], { style: 'width:100%' });
    viz.appendChild(fl);

    viz.appendChild(U.el('div', { class: 'klabel', text: '手算五组（数字由 .venv/bin/python 跑出）' }));
    const tb = W.table([
      ['3.0', '2.0', '<span class="dim">3.0 / 2.0</span>', '2.857722', '<span class="hl4">5.715445</span>', '5.715445'],
      ['12.0', '9.0', '<span class="hl3">10.0 / 9.0</span>', '9.999546', '<span class="hl3">89.995914</span>', '107.999336'],
      ['0.5', '15.0', '<span class="hl3">0.5 / 10.0</span>', '0.311230', '<span class="hl3">3.112297</span>', '4.668445'],
      ['-8.0', '4.0', '<span class="dim">-8.0 / 4.0</span>', '-0.002683', '<span class="hl4">-0.010731</span>', '-0.010731'],
      ['12.0', '15.0', '<span class="hl3">10.0 / 10.0</span>', '9.999546', '<span class="hl3">99.995460</span>', '179.998894'],
    ], { head: ['gate', 'up', 'clamp 后', 'SiLU(gate)', '输出', '不截断'] });
    const tbRow = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    tbRow.appendChild(U.el('div', { class: 'card', cc: 4, style: 'padding:9px 12px;flex:1' }, [tb]));
    const side = W.card({ cc: 2, style: 'flex:0 0 250px',
      title: '怎么看这张表',
      body: U.el('div', { class: 'col gap8' }, [
        U.el('div', { class: 'cs', html: '第 <b>1</b>、<b>4</b> 行：正常值，钳子<b>完全不干预</b>（两列相同）。' }),
        U.el('div', { class: 'cs', html: '第 <b>2</b>、<b>3</b>、<b>5</b> 行：越界，输出被钉在边界上。' }),
        U.el('div', { class: 'cs', html: 'SiLU 负侧自带有界（最小 <b>-0.278465</b>），正侧却近似线性 —— 所以 <code class="inl">gate</code> 只需要上界。' }),
      ]) });
    tbRow.appendChild(side);
    viz.appendChild(tbRow);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 7 行里，第 3 行是注释，第 5、6 行是钳子，第 7 行才是 SwiGLU 本体</span>';
    tl.at(2400, () => { fl.focus(2); msg.innerHTML = '<em>clamp</em> 在乘法之前 —— 这是 GLM-5 相对「教科书 SwiGLU」多出来的两步'; });
    tl.at(5800, () => { fl.focus(3); msg.innerHTML = '<span class="op">⊙</span> 是关键：<em>gate 与 up 逐元素相乘</em>，两条线性变换在这里耦合'; });
    tl.at(9200, () => {
      fl.focus(-1);
      msg.innerHTML = '手算第 2 行：gate=12 被钉到 <em>10</em> → SiLU(10) = <em>9.999546</em>，乘 up=9 → <em>89.995914</em>';
    });
    tl.at(12600, () => {
      msg.innerHTML = '同一个 gate 若 <em>不截断</em>：SiLU(12) = 11.999926，乘 9 → <em>107.999336</em>';
    });
    tl.at(15600, () => {
      msg.innerHTML = '<span class="cm">// 第 5 行更极端：99.995460 vs 179.998894 —— 钳子第二次都拦住了</span>';
    });
  },
},

/* ------------------------------------------------- 4 ★ 那把钳子 */
{
  kicker: '★ 核心 · 数值',
  title: '★ <span class="hl-c">10.0</span> 这把钳子：为什么它能提升稳定性',
  sub: '它不改正常值，只把「会跑到 2400 的中间激活」按在 100 以内 —— 而且顺手切断了越界处的梯度。',
  caption: '钳位阈值的实测位置：默认初始化下 max|gate| = 6.44，10.0 大约是它的 1.55 倍。',
  lang: 'python',
  codeStart: 102,
  code: `        # Key difference using clamping
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)`,
  codeNote: '只有三行：一句注释 + 两把钳子。注意两把钳子的范围是不对称的。',
  duration: 18000,
  build(root, tl) {
    setName('modeling_glm5_next.py');
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap14 vizgrow hstart' });
    wrap.appendChild(viz);

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const c1 = W.card({ cc: 3, tint: 3, title: '钳位：把越界值钉在边界上',
      body: U.el('div', { class: 'col gap6' }, [
        U.el('div', { class: 'mono', style: 'font-size:11px', html: 'gate.clamp(min=None, max=10)' }),
        U.el('div', { class: 'mono', style: 'font-size:11px', html: 'up.clamp(min=-10, max=10)' }),
      ]),
      sub: 'gate 只封上界（SiLU 负侧本来就有下界 -0.2785）；up 两侧都封', style: 'flex:1' });
    const c2 = W.card({ cc: 2, tint: 2, title: '上界是可以手推的',
      body: U.el('div', { class: 'mono', style: 'font-size:11.5px;color:#fde68a',
        html: '|SiLU(clamp(gate))| &lt; 10<br>|clamp(up)| ≤ 10<br>⇒ |中间激活| &lt; 100' }),
      style: 'flex:1' });
    const c3 = W.card({ cc: 0, tint: 0, title: '梯度：越界即 0',
      body: U.el('div', { class: 'col gap6' }, [
        U.el('div', { class: 'mono', style: 'font-size:11px', html: 'x = [-3, 5, 12, -21]' }),
        U.el('div', { class: 'mono', style: 'font-size:11px', color: '#bae6fd', html: '∂clamp/∂x = [1, 1, 0, 0]' }),
      ]),
      sub: '12 和 -21 拿不到任何梯度 —— 钳子也是刹车', style: 'flex:1' });
    row.appendChild(c1); row.appendChild(c2); row.appendChild(c3);
    viz.appendChild(row);

    const bwrap = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const bars = W.bars([
      { label: '不截断 max', value: K.rawMax, cc: 3, valueText: '2419.45' },
      { label: '截断后 max', value: K.clampedMax, cc: 4, valueText: '99.99' },
    ], { max: K.rawMax });
    bwrap.appendChild(U.el('div', { class: 'card', cc: 4, style: 'padding:10px 12px;flex:1' }, [
      U.el('div', { class: 'klabel', text: 'max|SiLU(gate)·up|（std=18 的随机 gate/up，各 4000 个样本，实测）' }),
      bars,
    ]));
    const c4 = W.card({ cc: 4, tint: 4, style: 'flex:0 0 300px', title: '阈值放在哪',
      body: U.el('div', { class: 'col gap6' }, [
        U.el('div', { class: 'mono', style: 'font-size:11px', html: 'initializer_range = 0.02' }),
        U.el('div', { class: 'mono', style: 'font-size:11px', html: '实测 max|gate| = <span class="hl">6.44</span>' }),
        U.el('div', { class: 'mono', style: 'font-size:11px', html: '10.0 / 6.44 ≈ <span class="hl">1.55×</span>' }),
      ]),
      sub: '10.0 就在「出厂健康值」的 1.55 倍处：正常不触发，跑飞了才兜底' });
    bwrap.appendChild(c4);
    viz.appendChild(bwrap);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 先看这行注释：作者把这件事标成了 Key difference</span>';
    tl.at(2600, () => {
      c1.classList.add('ac');
      msg.innerHTML = '钳子只在<em>越界处</em>生效：手算表第 1、4 行显示，正常值一个都没被改';
    });
    tl.at(6000, () => {
      c1.classList.remove('ac'); c2.classList.add('ac');
      msg.innerHTML = '上界公式：SiLU 正侧近似线性（实测 SiLU(20) = 20.000000），所以<em>不加钳位就没有上界</em>';
    });
    tl.at(9400, () => {
      c2.classList.remove('ac'); c3.classList.add('ac');
      msg.innerHTML = '第二重作用：越界元素的梯度<em>恰好是 0</em> —— 想跑飞的那一维拿不到「再大一点」的梯度';
    });
    tl.at(12800, () => {
      c3.classList.remove('ac');
      msg.innerHTML = '实测同一组随机数：max|输出| 从 <em>2419.451</em> 降到 <em>99.995</em>（降了 24 倍）';
    });
    tl.at(16000, () => {
      msg.innerHTML = '<span class="cm">// 为什么这对 45 层的大模型重要：down_proj 要把 12288 项求和，单项的量级决定这个和的范围</span>';
    });
  },
},

/* ------------------------------------------------- 5 同一套截断出现 4 次 */
{
  kicker: '结构 · 一致性',
  title: '同一套截断，在这份文件里出现 <span class="hl-a">4 次</span>',
  sub: '稠密 MLP、路由专家、视觉 MLP、patch merger —— 走的代码路径不同，钳子的位置与范围完全一样。',
  caption: '实测：grep -c "clamp(min=None" 命中 4 处 —— 行 103 / 140 / 1530 / 1553。',
  lang: 'python',
  codeStart: 138,
  code: `    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        # Simple swiglu instead of alpha
        return F.silu(gate) * up`,
  codeNote: 'Glm5NextTextExperts._apply_gate —— 路由专家的那一路。注意最后一行注释：不用 alpha。',
  duration: 17000,
  build(root, tl) {
    setName('modeling_glm5_next.py');
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap14 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '一个稀疏层里，三次 mlp 调用用同一套数值约定' }));
    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const mk = (cc, tint, t, shape, note) => W.card({ cc: cc, tint: tint,
      title: '<span class="mono" style="font-size:12px">' + t + '</span>',
      body: U.el('div', { class: 'col gap6' }, [
        U.el('div', { class: 'n', style: 'font-size:13px;color:#fff', text: shape }),
        U.el('div', { class: 'cs', text: note }),
      ]), style: 'flex:1' });
    const cA = mk(4, 4, 'shared_experts', 'Glm5NextTextMLP 实例', '中间维 2048×1；always on');
    const cB = mk(0, 0, 'experts._apply_gate', '打包张量 chunk(2)', 'top-8 路由专家；每层 288 份权重');
    const cC = mk(1, 1, 'down_proj', '12288 / 2048 → 4096', '两边都在钳位之后才做乘法和降维');
    row.appendChild(cA); row.appendChild(cB); row.appendChild(cC);
    viz.appendChild(row);

    const tb = W.table([
      ['行 <span class="hl">103</span>', '<code class="inl">Glm5NextTextMLP.forward</code>', '稠密层 + 共享专家', 'gate max / up ±10'],
      ['行 <span class="hl">140</span>', '<code class="inl">Glm5NextTextExperts._apply_gate</code>', 'top-8 路由专家', 'gate max / up ±10'],
      ['行 <span class="hl">1530</span>', '<code class="inl">Glm5NextVisionMLP.forward</code>', '视觉塔 FFN', 'gate max / up ±10'],
      ['行 <span class="hl">1553</span>', '<code class="inl">Glm5NextVisionPatchMerger.forward</code>', '图像 token 汇入主干', 'gate max / up ±10'],
    ], { head: ['位置', '函数', '谁在用', '钳位范围'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 2, style: 'padding:10px 12px' }, [
      U.el('div', { class: 'klabel', text: '实测的四个钳位点（grep 行号）' }), tb,
    ]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 路由专家与稠密 MLP 的区别只有两处：切分方式、激活写法</span>';
    tl.at(2400, () => { cB.classList.add('ac'); msg.innerHTML = '专家把 gate/up <em>打包成一个张量</em>，所以要先 <span class="fn">chunk(2, dim=-1)</span> 现场切开'; });
    tl.at(5600, () => {
      cB.classList.remove('ac'); cA.classList.add('ac');
      msg.innerHTML = '共享专家直接复用 <span class="fn">Glm5NextTextMLP</span>，只是中间维换成 <em>2048 × 1</em>';
    });
    tl.at(9000, () => {
      cA.classList.remove('ac'); cC.classList.add('ac');
      msg.innerHTML = '它同样调用 <em>ACT2FN[hidden_act]</em> 与 <em>swiglu_limit</em> —— 三条路共享同一个 config 约定';
    });
    tl.at(12400, () => {
      cC.classList.remove('ac');
      msg.innerHTML = '一个稀疏层的 MLP 相关参数：<em>7.27B</em> 存储、每 token 只用 <em>226.5M</em>（32 倍差距）';
    });
    tl.at(14800, () => {
      msg.innerHTML = '<span class="cm">// 打包布局与 grouped GEMM 的关系见 L3-04；谁被选中见 L3-03</span>';
    });
  },
},

/* ------------------------------------------------- 6 ACT2FN 注册表 */
{
  kicker: '依赖 · 注册表',
  title: '<span class="hl-b">ACT2FN</span>：字符串怎么变成算子',
  sub: 'self.act_fn 不是硬编码的 silu，而是拿 config.hidden_act 去查一张 24 项的注册表。',
  caption: '同一个注册表也被这一层的 RMSNormGated 用到（它的 self.activation = "sigmoid"），见 L3-01。',
  lang: 'python',
  codeStart: 324,
  code: `ACT2CLS = {
    "gelu": GELUActivation,
    "gelu_10": (ClippedGELUActivation, {"min": -10, "max": 10}),
    "gelu_fast": FastGELUActivation,
    "gelu_new": NewGELUActivation,
    "gelu_python": (GELUActivation, {"use_gelu_python": True}),
    "gelu_pytorch_tanh": GELUTanh,
    "gelu_python_tanh": (GELUTanh, {"use_gelu_tanh_python": True}),
    "gelu_accurate": AccurateGELUActivation,
    "hardswish": nn.Hardswish,
    "laplace": LaplaceActivation,
    "leaky_relu": nn.LeakyReLU,
    "linear": LinearActivation,
    "mish": MishActivation,
    "quick_gelu": QuickGELUActivation,
    "relu": nn.ReLU,
    "relu2": ReLUSquaredActivation,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "silu": SiLUActivation,
    "sqrtsoftplus": SqrtSoftplusActivation,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
    "prelu": nn.PReLU,
    "xielu": XIELUActivation,
}
ACT2FN = ClassInstantier(ACT2CLS)`,
  codeNote: 'activations.py 的 ACT2CLS：24 个键，两种值形态 —— 裸类，或（类, kwargs）二元组。',
  duration: 17000,
  build(root, tl) {
    setName('activations.py');
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const fl = W.flow([
      { t: 'config.hidden_act', s: '"silu"', cc: 0 },
      { t: 'ACT2FN["silu"]', s: '查表', cc: 1 },
      { t: 'ClassInstantier', s: 'cls(**kwargs)', cc: 5 },
      { t: 'SiLUActivation()', s: '实例，存在 self 上', cc: 4 },
    ], { style: 'width:100%' });
    viz.appendChild(fl);

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const tb = W.table([
      ['<code class="inl">"silu"</code>', '<span class="hl4">SiLUActivation</span>', '本次被选中的那一项'],
      ['<code class="inl">"gelu_10"</code>', '(ClippedGELUActivation, {min, max})', '带参数的注册：夹在 [-10, 10]'],
      ['<code class="inl">"gelu_new"</code>', 'NewGELUActivation', 'tanh 近似'],
      ['<code class="inl">"sqrtsoftplus"</code>', 'SqrtSoftplusActivation', 'docstring 写明是 DeepSeek V4 的路由打分'],
      ['<code class="inl">"xielu"</code>', 'XIELUActivation', '带 nn.Parameter 的激活'],
      ['…', '<span class="dim">共 24 个键</span>', '<span class="dim">gelu / relu / swish 各家族的演化史</span>'],
    ], { head: ['键', '值', '说明'] });
    row.appendChild(U.el('div', { class: 'card', cc: 1, style: 'padding:10px 12px;flex:1' }, [
      U.el('div', { class: 'klabel', text: 'ACT2CLS 里与本课有关的几项' }), tb,
    ]));
    const side = U.el('div', { class: 'col gap12', style: 'flex:0 0 280px' });
    side.appendChild(W.card({ cc: 2, tint: 2, title: '两种值形态',
      sub: '裸类：<code class="inl">"silu": SiLUActivation</code><br>带参数：<code class="inl">"gelu_10": (类, {"min": -10, "max": 10})</code>',
      body: U.el('div', { class: 'cs', html: '后者就是「同一个类、不同参数」的注册方式 —— 第 4 节的钳位先例靠它落地。' }) }));
    side.appendChild(W.card({ cc: 4, tint: 4, title: 'SiLU 是什么',
      body: U.el('div', { class: 'mono', style: 'font-size:11.5px;color:#a7f3d0', html: 'SiLU(x) = x · σ(x)<br>= x / (1 + e^(-x))' }),
      sub: '实测 nn.functional.silu(x) 与 x·sigmoid(x) 逐位相同' }));
    row.appendChild(side);
    viz.appendChild(row);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// ACT2CLS 是字典；ACT2FN 是它的实例化版本</span>';
    tl.at(2400, () => { fl.focus(0); msg.innerHTML = '第 96 行：<em>ACT2FN[config.hidden_act]</em> —— 所有算子都从这一个字符串出发'; });
    tl.at(5600, () => { fl.focus(1); msg.innerHTML = '查表的结果<em>不是函数、不是字符串</em>，而是一个 nn.Module 实例'; });
    tl.at(9000, () => {
      fl.focus(-1);
      msg.innerHTML = '实测 <em>len(ACT2CLS) = 24</em>：从 <span class="fn">gelu</span> 的三个近似到 <span class="fn">xielu</span>';
    });
    tl.at(12400, () => {
      msg.innerHTML = '<span class="hl3">gelu_10</span> 这一项值得记住：夹在 [-10, 10] 的 GELU —— 和本课的钳子同一个想法';
    });
    tl.at(14800, () => {
      msg.innerHTML = '<span class="cm">// 注册表怎么把「类」变成「实例」，是下一幕的内容</span>';
    });
  },
},

/* ------------------------------------------------- 7 ★ ClassInstantier */
{
  kicker: '依赖 · 实例化',
  title: '★ 表里存的不是实例，是<span class="hl-a">「类 + 参数」</span>',
  sub: '每次下标访问都新建一个实例 —— 因为 nn.Module 不能共享，两个算子必须各拿一份。',
  caption: '实测：ACT2FN["silu"] is not ACT2FN["silu"] 为 True —— 两次取出的是两个对象。',
  lang: 'python',
  codeStart: 224,
  code: `class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)`,
  codeNote: 'ClassInstantier —— 整个类只有 5 行，第 5 行是全部的魔法。',
  duration: 16000,
  build(root, tl) {
    setName('activations.py');
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '两种登记形态，同一个出口' }));
    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const p1 = W.card({ cc: 4, tint: 4, title: '裸类登记',
      body: U.el('div', { class: 'col gap6' }, [
        U.el('div', { class: 'mono', style: 'font-size:11px', html: '"silu": <span class="hl4">SiLUActivation</span>' }),
        U.el('div', { class: 'dim mono', style: 'font-size:11px', html: '↓ isinstance(content, tuple) 为 False' }),
        U.el('div', { class: 'mono', style: 'font-size:11px', html: 'cls, kwargs = cls, {}' }),
        U.el('div', { class: 'mono', style: 'font-size:11px;color:#a7f3d0', html: 'SiLUActivation()' }),
      ]), style: 'flex:1' });
    const p2 = W.card({ cc: 2, tint: 2, title: '带参数登记',
      body: U.el('div', { class: 'col gap6' }, [
        U.el('div', { class: 'mono', style: 'font-size:11px', html: '"gelu_10": (<span class="hl3">ClippedGELU</span>, {...})' }),
        U.el('div', { class: 'dim mono', style: 'font-size:11px', html: '↓ isinstance(content, tuple) 为 True' }),
        U.el('div', { class: 'mono', style: 'font-size:11px', html: 'cls, kwargs = content' }),
        U.el('div', { class: 'mono', style: 'font-size:11px;color:#fde68a', html: 'ClippedGELUActivation(min=-10, max=10)' }),
      ]), style: 'flex:1' });
    row.appendChild(p1); row.appendChild(p2);
    viz.appendChild(row);

    const tb = W.table([
      ['<code class="inl">type(ACT2FN)</code>', 'ClassInstantier', 'OrderedDict 的子类'],
      ['<code class="inl">ACT2FN["silu"] is ACT2FN["silu"]</code>', '<span class="hlbad">False</span>', '每次访问都新建实例'],
      ['<code class="inl">ACT2FN["gelu_10"].min / .max</code>', '<span class="hl">-10 / 10</span>', '二元组里的 kwargs 被真的用上了'],
      ['<code class="inl">ACT2FN["no_such_act"]</code>', '<span class="hlbad">KeyError</span>', '报错信息里带 24 个合法键'],
      ['写错激活名的后果', '<span class="hl">建模时就炸</span>', '不在第 30 层才炸'],
    ], { head: ['实测', '结果', '说明'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 5, style: 'padding:10px 12px' }, [
      U.el('div', { class: 'klabel', text: '用 .venv/bin/python 实测到的五条' }), tb,
    ]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 为什么不能把实例直接放进表里？</span>';
    tl.at(2200, () => { p1.classList.add('ac'); msg.innerHTML = '裸类登记：<em>cls(**{})</em> —— 等价于无参构造'; });
    tl.at(5400, () => {
      p1.classList.remove('ac'); p2.classList.add('ac');
      msg.innerHTML = '二元组登记：kwargs 原样传进构造函数 —— <em>同一个类、不同参数</em>可以各占一个键';
    });
    tl.at(8600, () => {
      p2.classList.remove('ac');
      msg.innerHTML = '为什么每次新建：<em>nn.Module 不能共享</em>；像 xielu 那样带参数的激活被两处引用会串味';
    });
    tl.at(11600, () => {
      msg.innerHTML = '还有一层保险：键写错时抛 <em>KeyError</em> 并列出全部键，而不是静默换一个默认激活';
    });
    tl.at(13400, () => {
      msg.innerHTML = '<span class="cm">// 这行 ACT2FN[...] 写在 __init__ 里，所以配置错误在建模那一刻就暴露</span>';
    });
  },
},

/* ------------------------------------------------- 8 视觉支路 */
{
  kicker: '结构 · 跨模态',
  title: '视觉支路：同一段代码的<span class="hl-a">第二份拷贝</span>',
  sub: 'Glm5NextVisionMLP 与文本 MLP 几乎逐行相同 —— 连注释都一样。差异只有两处。',
  caption: 'patch merger 是第三份：前面多了 proj + LayerNorm + GELU，后面同样两把钳子。',
  lang: 'python',
  codeStart: 1526,
  code: `    def forward(self, hidden_state):
        gate = self.gate_proj(hidden_state)
        up = self.up_proj(hidden_state)
        # Key difference using clamping
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)`,
  codeNote: 'Glm5NextVisionMLP.forward —— 与文本 MLP 的差异：入参叫 hidden_state，bias 默认为 True。',
  duration: 16000,
  build(root, tl) {
    setName('modeling_glm5_next.py');
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const mk = (cc, tint, t, sub, items) => W.card({ cc: cc, tint: tint,
      title: '<span class="mono" style="font-size:11.5px">' + t + '</span>', sub: sub,
      body: U.el('div', { class: 'col gap6' }, items.map(x => U.el('div', { class: 'cs', html: x }))),
      style: 'flex:1' });
    row.appendChild(mk(4, 4, 'Glm5NextTextMLP', '文本主干 · 行 87',
      ['中间维 <b>12288</b>（= 3 × 4096）', '<b>bias=False</b>', 'act_fn 来自 <b>ACT2FN[config.hidden_act]</b>', '共享专家也用它，维 2048']));
    row.appendChild(mk(0, 0, 'Glm5NextVisionMLP', '视觉塔 · 行 1515',
      ['中间维 <b>4096</b>（config.intermediate_size）', '<b>bias=True</b>（config.attention_bias）', '同样两把钳子、同一个阈值', '入参名 <b>hidden_state</b>（无 s）']));
    row.appendChild(mk(1, 1, 'Glm5NextVisionPatchMerger', '汇入主干 · 行 1535',
      ['dim → <b>projection_intermediate_size</b> → dim', '前面多 <b>proj + LayerNorm + GELU</b>', '构造参数里直接收 <b>swiglu_limit</b>', '行 1553 是第四个钳位点']));
    viz.appendChild(row);

    const tb = W.table([
      ['文本 LLM', '10.0', '<code class="inl">text_config.swiglu_limit</code>'],
      ['视觉塔', '10.0', '<code class="inl">vision_config.swiglu_limit</code>'],
      ['文本 hidden_act', '<code class="inl">"silu"</code>', '查 ACT2FN 得到 SiLUActivation'],
      ['视觉 hidden_act', '<code class="inl">"silu"</code>', '同一张注册表'],
    ], { head: ['一侧', '实测默认值', '来源'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 2, style: 'padding:10px 12px' }, [
      U.el('div', { class: 'klabel', text: '实测：两侧的约定完全一致' }), tb,
    ]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 第 1529 行的注释和第 102 行一模一样：Key difference using clamping</span>';
    tl.at(2200, () => {
      msg.innerHTML = '两处差异：入参名（<em>hidden_state</em>）与 <em>bias</em>（视觉塔带偏置，文本不带）';
    });
    tl.at(5400, () => {
      msg.innerHTML = '钳子的<em>位置、范围、阈值</em>三项全部相同 —— 跨模态的统一约定';
    });
    tl.at(8800, () => {
      msg.innerHTML = '文件头写明：这份 <em>modeling_glm5_next.py</em> 由 modular 文件生成 —— 重复是管理一致性的手段';
    });
    tl.at(12000, () => {
      msg.innerHTML = '<span class="cm">// 图像 token 汇入主干时用的也是同一把钳子，所以主干里没有「另一种数值约定」</span>';
    });
  },
},

/* ------------------------------------------------- 9 收束 */
{
  kicker: '收束',
  title: '把这一课压成一张表',
  sub: '三个投影、一次门控乘法、两把钳子、一张注册表 —— 45 层里每一层都重复这一套。',
  caption: '下一课 L3-03：gate_proj 的兄弟 —— 288 个专家分数怎么变成 8 条路径。',
  lang: 'python',
  codeStart: 99,
  code: `    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # Key difference using clamping
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)`,
  codeNote: '回到开头：现在这 7 行里的每一个名字你都能说出它属于哪一节。',
  duration: 18000,
  build(root, tl) {
    setName('modeling_glm5_next.py');
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const steps = [
      { n: '①', t: '两个投影', d: 'gate（闸门）+ up（内容），4096 → 12288', c: 4 },
      { n: '②', t: '两把钳子', d: 'gate 封上界 10；up 封 ±10', c: 2 },
      { n: '③', t: '一次门控乘积', d: 'SiLU(gate) ⊙ up，把两条线性变换耦合', c: 0 },
      { n: '④', t: '投回主干', d: 'down_proj：12288 → 4096 求和', c: 1 },
      { n: '⑤', t: '查表得到的激活', d: 'ACT2FN[config.hidden_act]，每次新建实例', c: 5 },
    ];
    const row = U.el('div', { class: 'row gap8', style: 'width:100%;align-items:stretch' });
    const els = steps.map(s => {
      const e = W.card({ cc: s.c, tint: s.c, title: '<span class="mono" style="font-size:11px">' + s.n + ' ' + s.t + '</span>',
        sub: s.d, style: 'flex:1;text-align:center' });
      row.appendChild(e); return e;
    });
    viz.appendChild(row);

    const tb = W.table([
      ['稠密 MLP 参数量', '<span class="hl">151.0M</span>', '3 × 4096 × 12288，零偏置'],
      ['稀疏层 MLP 参数量', '<span class="hl">7.27B</span>', '存储；每 token 只用 226.5M（32×）'],
      ['中间激活上界', '<span class="hl">&lt; 100</span>', '未截断时实测 2419.451 → 截断后 99.995'],
      ['阈值位置', '<span class="hl">1.55×</span>', '默认初始化下 max|gate| = 6.44，阈值 10.0'],
      ['钳位点', '<span class="hl">4 处</span>', '行 103 / 140 / 1530 / 1553'],
      ['注册表规模', '<span class="hl">24 键</span>', 'ACT2CLS → ACT2FN（ClassInstantier）'],
    ], { head: ['量', '实测值', '说明'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 0, style: 'padding:10px 12px' }, [
      U.el('div', { class: 'klabel', text: '本课用到的实测数字（.venv/bin/python 实跑）' }), tb,
    ]));

    viz.appendChild(W.exercise(
      '<code class="inl">gate = 12.0</code>、<code class="inl">up = 15.0</code> 时，'
      + '这一维经过 SwiGLU 之后的中间激活是多少？不加 <code class="inl">swiglu_limit</code> 呢？',
      '先钳位：gate → <b>10.0</b>，up → <b>10.0</b>。'
      + 'SiLU(10) = 10/(1+e^-10) = <b>9.999546</b>，乘 10 → <b>99.995460</b>。'
      + '<br>不钳位：SiLU(12) = <b>11.999926</b>，乘 15 → <b>179.998894</b>。'
      + '<br>上界还能直接手推：|SiLU(clamp(gate))| &lt; 10 且 |clamp(up)| ≤ 10，'
      + '所以中间激活恒 <b>&lt; 100</b> —— 与输入无关。'
      + '<br>再想一步：越界元素的梯度是 <b>0</b>（实测 [1, 1, 0, 0]），'
      + '所以这个上界不只挡住数值，也挡住了「继续变大」的学习信号。'));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    els.forEach(e => e.style.opacity = '.28');
    msg.innerHTML = '<span class="cm">// 五个要点，逐个点亮</span>';
    steps.forEach((s, i) => {
      tl.at(1500 + i * 2100, () => {
        els.forEach((e, k) => { e.style.opacity = (k === i) ? '1' : '.28'; });
        msg.innerHTML = '<em>' + s.n + '</em> ' + s.t + ' &nbsp;<span class="cm">// ' + s.d + '</span>';
      });
    });
    tl.at(12600, () => {
      els.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = 'SwiGLU 的前向 = <em>W_down( SiLU(clamp(W_gate·x, 10)) ⊙ clamp(W_up·x, ±10) )</em>';
    });
    tl.at(15200, () => {
      msg.innerHTML = '<span class="cm">// 下一课 L3-03：同一层的 gate_proj 有个兄弟 —— 路由器，它把 288 个分数压成 8 条路径</span>';
    });
  },
},

];
