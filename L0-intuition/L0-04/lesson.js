/* ==========================================================================
   L0-04 · MoE 经济学：288 选 8
   --------------------------------------------------------------------------
   覆盖：models/glm5_next/modeling_glm5_next.py
   目标：看完能算出稀疏层与稠密层的参数量比，说清「3% 激活」的来历，
         并解释共享专家为什么必须存在。
   排版基准：#visual 可视区 = 994 x 662.7 舞台像素（无头浏览器实测）。
   ========================================================================== */
'use strict';

/* 实测常量（.venv/bin/python + meta 设备建层数出来的，不是估算） */
const M = {
  layers: 45, dense: 3, sparse: 42,
  experts: 288, topk: 8, shared: 1,
  hidden: 4096, moeInter: 2048, denseInter: 12288,
  scaling: 2.5, nGroup: 1, topkGroup: 1,
  denseParams: 150994944,      /* 151.0M  = 3 * 4096 * 12288 */
  moeParams: 7274102784,       /* 7274.1M */
  routedParams: 7247757312,    /* 7247.8M = 288 * 25165824 */
  expertParams: 25165824,      /* 25.17M  = 2*2048*4096 + 4096*2048 */
  sharedParams: 25165824,      /* 25.2M   = 3 * 4096 * 2048 */
  gateParams: 1179648,         /* 1.18M   = 288 * 4096 */
  activeMoe: 227672064,        /* 227.7M  = 8*25.17M + 25.17M + 1.18M */
  total: 313786889214,         /* 313.79B */
  activeTotal: 17842343902,    /* 17.84B */
  ratioParams: 48.17,          /* 7274.1M / 151.0M */
  ratioCompute: 1.51,          /* 227.7M / 151.0M */
  activePct: 3.13,             /* (8 + 1) / 288 */
  picks: [7, 42, 88, 130, 171, 205, 246, 287],
};

const SCENES = [

/* ------------------------------------------------- 1 全景：一个稀疏层 */
{
  kicker: '预备篇 · 直觉',
  title: 'MoE 经济学：<span class="hl-a">288 选 8</span>',
  sub: '45 层里有 42 层是 MoE。每层 288 个路由专家，每个 token 只叫醒 8 个，再额外过 1 个永远在线的共享专家。',
  caption: '这一课只讲 MLP 这一块；注意力那半（34 层 KDA + 11 层 MLA）在 L0-03。',
  lang: 'python',
  codeStart: 192,
  code: `    def __init__(self, config: Glm5NextConfig):
        super().__init__()
        self.config = config
        self.experts = Glm5NextTextExperts(config)
        self.gate = Glm5NextTextTopkRouter(config)
        self.shared_experts = Glm5NextTextMLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )`,
  codeNote: 'Glm5NextTextMoE.__init__ —— 一个稀疏层的全部零件都在这里。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap14 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '一个稀疏 MLP 层的内部结构（Glm5NextTextMoE）' }));
    const fl = W.flow([
      { t: 'hidden_states', s: '(b, s, 4096)', cc: 4 },
      { t: 'gate', s: '288 个分数', cc: 2 },
      { t: 'experts', s: '288 选 8', cc: 0 },
      { t: 'Σ w · E', s: '8 个专家', cc: 0 },
      { t: '+ shared', s: '常开 1 个', cc: 1 },
      { t: 'out', s: '(b, s, 4096)', cc: 3 },
    ]);
    viz.appendChild(fl);

    const statRow = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const statEls = [
      { s: 'n_routed_experts', v: '288', cc: 0 },
      { s: 'num_experts_per_tok', v: '8', cc: 2 },
      { s: 'n_shared_experts', v: '1', cc: 1 },
    ].map(d => {
      const c = W.card({ cc: d.cc, tint: d.cc, style: 'flex:1;text-align:center',
        title: '<span class="mono" style="font-size:11px">' + d.s + '</span>',
        body: U.el('div', { class: 'n big', text: d.v }) });
      statRow.appendChild(c); return c;
    });
    viz.appendChild(statRow);

    const budget = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const bA = W.card({ cc: 3, tint: 3, title: '总参数（meta 设备实测）', style: 'flex:1',
      body: U.el('div', { class: 'n big', text: '313.79 B' }),
      sub: 'Glm5NextForConditionalGeneration 默认配置全部参数' });
    const bB = W.card({ cc: 4, tint: 4, title: '每 token 激活（实测）', style: 'flex:1',
      body: U.el('div', { class: 'n big', text: '17.84 B' }),
      sub: '占总量 5.68% —— 对外口径「320B 总参 / 18B 激活」' });
    budget.appendChild(bA); budget.appendChild(bB);
    viz.appendChild(budget);

    /* 本课路线：四幕各解决一个「为什么」 */
    const road = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const roadEls = [
      { cc: 0, t: '第一幕 · 稀疏', s: '288 个盒子里只叫醒 8 个' },
      { cc: 4, t: '第二幕 · 预算', s: '48.17× 的参数 vs 1.51× 的算力' },
      { cc: 1, t: '第三幕 · 共享专家', s: '它凭什么永远开着' },
      { cc: 3, t: '第七幕 · 训练/推理', s: '辅助损失不进 decode' },
    ].map(d => {
      const e = W.card({ cc: d.cc, title: '<span class="mono" style="font-size:11px">' + d.t + '</span>',
        sub: d.s, style: 'flex:1' });
      road.appendChild(e); return e;
    });
    viz.appendChild(road);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    roadEls.forEach(e => { e.style.opacity = '.3'; });

    fl.focus(0);
    msg.innerHTML = '<span class="cm">// 一个稀疏层三板斧：路由器 + 专家池 + 共享专家</span>';
    tl.at(1500, () => { fl.focus(1); msg.innerHTML = '<em>gate</em>：把 4096 维的 hidden 投成 <em>288</em> 个分数（只有 1.18M 参数）'; });
    tl.at(4200, () => { fl.focus(2); msg.innerHTML = '<em>experts</em>：288 个专家的参数都在显存里，这一次前向只算被选中的 8 个'; });
    tl.at(7000, () => { fl.focus(4); msg.innerHTML = '<em>shared_experts</em>：<span class="hl2">不参与竞争</span>，每个 token 都过它'; });
    tl.at(10000, () => {
      fl.focus(-1); statEls.forEach(e => e.classList.add('ac'));
      msg.innerHTML = '每 token 只为 <em>9 / 288 = 3.125%</em> 的专家参数付算力';
    });
    tl.at(13500, () => {
      statEls.forEach(e => e.classList.remove('ac'));
      bA.classList.add('ac'); bB.classList.add('ac');
      msg.innerHTML = '<span class="cm">// 容量按 288 份买，账单按 9 份付 —— 这就是本课的标题</span>';
    });
    tl.at(15500, () => {
      bA.classList.remove('ac'); bB.classList.remove('ac');
      roadEls.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = '<span class="cm">// 接下来四幕：稀疏 → 预算 → 共享专家 → 训练/推理的边界</span>';
    });
  },
},

/* ------------------------------------------------- 2 288 个盒子亮 8 个 */
{
  kicker: '第一幕 · 稀疏',
  title: '288 个专家里，每个 token 只叫醒 <span class="hl-a">8</span> 个',
  sub: '专家不是「一个更大的 MLP」，而是 288 个并排的小 MLP；参数打包成一个 3D 张量。',
  caption: '8 / 288 = 2.78%；再加 1 个共享专家，每 token 约 3% 的专家参数在动。',
  lang: 'python',
  codeStart: 112,
  code: `    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.swiglu_limit = config.swiglu_limit`,
  codeNote: 'gate_up_proj 与 down_proj 是 3D 参数：288 份权重打包在一起。',
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '288 个路由专家 = 24 列 × 12 行（每格 1 个专家，悬停看编号）' }));
    const grid = U.el('div', { style: 'display:grid;grid-template-columns:repeat(24,1fr);gap:3px;width:100%' });
    const cells = [];
    for (let i = 0; i < 288; i++) {
      const e = U.el('div', {
        class: 'mono',
        title: 'expert ' + i,
        style: 'height:26px;border-radius:5px;display:flex;align-items:center;justify-content:center;'
             + 'font-size:8.5px;min-width:0;border:1px solid rgba(150,180,255,.14);'
             + 'background:rgba(150,180,255,.05);color:var(--ink-faint);'
             + 'transition:background .4s cubic-bezier(.16,1,.3,1),border-color .4s,color .4s,transform .4s cubic-bezier(.34,1.56,.64,1)',
      });
      grid.appendChild(e); cells.push(e);
    }
    viz.appendChild(grid);

    const legend = U.el('div', { class: 'row gap12 wrap', style: 'width:100%' });
    legend.innerHTML =
      '<span class="chip c0"><i class="sw sw-c0"></i>被选中的 8 个</span>' +
      '<span class="chip"><i class="sw" style="background:rgba(150,180,255,.35)"></i>这次前向不参与计算</span>' +
      '<span class="dim mono" style="margin-left:auto;font-size:11px">8 / 288 = 2.78%</span>';
    viz.appendChild(legend);

    const statRow = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    [
      { k: '单个专家', v: '25.17 M', s: '2 × 2048 × 4096 + 4096 × 2048', cc: 0 },
      { k: '8 个专家', v: '201.33 M', s: '8 × 25.17M', cc: 2 },
      { k: '+ 1 个共享专家', v: '25.17 M', s: '3 × 4096 × 2048（一个专家宽）', cc: 1 },
      { k: '288 个专家', v: '7247.8 M', s: '这才是装在显存里的', cc: 3 },
    ].forEach(d => {
      statRow.appendChild(W.card({ cc: d.cc, style: 'flex:1;text-align:center',
        title: '<span class="mono" style="font-size:11px">' + d.k + '</span>',
        body: U.el('div', { class: 'n', style: 'font-size:16px', text: d.v }),
        sub: '<span class="mono" style="font-size:10px">' + d.s + '</span>' }));
    });
    viz.appendChild(statRow);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 288 个格子都装了权重，但一个 token 只碰其中 8 个</span>';
    M.picks.forEach((pi, k) => {
      tl.at(1800 + k * 850, () => {
        const c = cells[pi];
        c.style.borderColor = 'var(--c0)';
        c.style.background = 'rgba(56,189,248,.34)';
        c.style.color = '#fff';
        c.style.boxShadow = '0 0 0 2px var(--c0), 0 0 14px -1px var(--c0)';
        c.style.zIndex = '3';
        c.textContent = String(pi);
        msg.innerHTML = '叫醒第 <em>' + (k + 1) + '</em> / 8 个 &nbsp;<span class="cm">// expert ' + pi + '</span>';
      });
    });
    tl.at(9000, () => {
      msg.innerHTML = '剩下 <em>280</em> 个专家的权重这次一动不动 <span class="cm">// 但它们仍占着显存</span>';
    });
    tl.at(12000, () => {
      msg.innerHTML = '单个专家的 25.17M 参数，比稠密层 MLP 的 151.0M <em>小 6 倍</em> —— 稀疏换来的是数量';
    });
    tl.at(15500, () => {
      msg.innerHTML = '<span class="cm">// (8 + 1) / 288 = 3.125% —— 这就是「约 3% 激活」的来历</span>';
    });
  },
},

/* ------------------------------------------------- 3 参数预算 */
{
  kicker: '第二幕 · 预算',
  title: '同一层 MLP：参数 <span class="hl-c">48.17×</span>，算力只 <span class="hl4">1.51×</span>',
  sub: '稠密层 151.0M 参数；MoE 层 7274.1M 参数，但每个 token 只用到 227.7M。',
  caption: '这两个倍数的落差就是 MoE 的全部经济学：容量按 288 份买，账单按 9 份付。',
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
  codeNote: '三个投影都是 bias=False，所以参数量可以直接手算：3 × hidden × intermediate。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '单层 MLP 参数量（meta 设备实测，同一横轴）' }));
    const layerBars = W.bars([
      { label: '稠密 MLP 层', value: M.denseParams, cc: 4, valueText: '151.0 M' },
      { label: 'MoE 层全部', value: M.moeParams, cc: 3, valueText: '7274.1 M' },
      { label: 'MoE 层激活', value: M.activeMoe, cc: 0, valueText: '227.7 M' },
    ], { max: M.moeParams });
    viz.appendChild(U.el('div', { class: 'card', cc: 4, style: 'padding:11px 13px;width:100%' }, [layerBars]));

    viz.appendChild(U.el('div', { class: 'klabel', text: '整个模型（Glm5NextForConditionalGeneration 默认配置）' }));
    const modelBars = W.bars([
      { label: '总参数', value: M.total, cc: 1, valueText: '313.79 B' },
      { label: '每 token 激活', value: M.activeTotal, cc: 4, valueText: '17.84 B' },
    ], { max: M.total });
    viz.appendChild(U.el('div', { class: 'card', cc: 1, style: 'padding:11px 13px;width:100%' }, [modelBars]));

    const chips = U.el('div', { class: 'row gap10 wrap', style: 'width:100%' });
    chips.innerHTML =
      '<span class="chip c3">稠密层 151.0M = 3 × 4096 × 12288</span>' +
      '<span class="chip c2">MoE 层 7274.1M = 288 × 25.17M</span>' +
      '<span class="chip c0">每 token 227.7M = 8×25.17M + 25.17M + 1.18M</span>';
    viz.appendChild(chips);

    /* 两个倍数的读法 */
    const read = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const rA = W.card({ cc: 2, tint: 2, title: '为什么参数能到 48 倍', style: 'flex:1',
      sub: '专家池把中间维从 12288 降到 2048（单专家只有稠密层的 1/6），但买了 288 份。'
         + '<span class="mono" style="font-size:10.5px">288 / 6 = 48</span>' });
    const rB = W.card({ cc: 4, tint: 4, title: '为什么算力只有 1.5 倍', style: 'flex:1',
      sub: '每个 token 只走 9 份（8 路由 + 1 共享），所以账单与「选中几个」成正比，'
         + '与「装了多少个」无关。' });
    read.appendChild(rA); read.appendChild(rB);
    viz.appendChild(read);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 先立基准：前 3 层是稠密 MLP，每 token 都要付这 151.0M</span>';
    tl.at(1600, () => { layerBars.setTo(0, 1); msg.innerHTML = '稠密层：<em>151.0 M</em> 参数（3 × 4096 × 12288，bias 全关）'; });
    tl.at(4600, () => { layerBars.setTo(1, 1); msg.innerHTML = 'MoE 层：<em>7274.1 M</em> 参数 —— 是稠密层的 <span class="hl3">48.17 倍</span>'; });
    tl.at(7800, () => { layerBars.setTo(2, 1); msg.innerHTML = '每 token 只激活：<em>227.7 M</em> —— 只有稠密层的 <span class="hl4">1.51 倍</span>'; });
    tl.at(10800, () => {
      modelBars.setTo(0, 1); modelBars.setTo(1, 1);
      msg.innerHTML = '整模型：<em>313.79 B</em> 里每次前向只走 <em>17.84 B</em>（5.68%）';
    });
    tl.at(14000, () => {
      rA.classList.add('ac');
      msg.innerHTML = '<span class="cm">// 48 倍的容量、1.5 倍的账单 —— 差距全部来自「只算被选中的」</span>';
    });
    tl.at(16500, () => {
      rA.classList.remove('ac'); rB.classList.add('ac');
      msg.innerHTML = '买 288 份容量、每 token 只结算 9 份 —— 这就是 MoE 把「参数」和「算力」解耦的方式';
    });
  },
},

/* ------------------------------------------------- 4 共享专家 */
{
  kicker: '第三幕 · 共享专家',
  title: '★ 共享专家：<span class="hl-b">永远开着</span>的那一个',
  sub: '路由专家学「分工」，共享专家学「公共知识」。它不参与竞争，每个 token 都要过它。',
  caption: '注意它用的是 Glm5NextTextMLP，宽度 = moe_intermediate_size × n_shared_experts —— 恰好一个专家宽。',
  lang: 'python',
  codeStart: 201,
  code: `    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        _, topk_weights, topk_indices = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states`,
  codeNote: '最后一行是加法：共享专家的输出直接叠在路由专家的加权和上。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    /* 注意：.vizgrow 自带 flex-direction:column，这里必须用行内样式抢回 row */
    const viz = U.el('div', { class: 'row gap12 vizgrow',
      style: 'flex-direction:row;align-items:center' });
    wrap.appendChild(viz);

    const left = W.card({ cc: 4, tint: 4, title: 'hidden_states', style: 'flex:1',
      body: U.el('div', { class: 'mono', style: 'font-size:11px;line-height:1.7;color:var(--ink-dim)' },
        [U.el('div', { html: '(b, s, 4096)' }),
         U.el('div', { class: 'dim', html: '↓ residuals = hidden_states' }),
         U.el('div', { class: 'hl4', html: '留一份给共享专家' })]) });

    const midA = W.card({ cc: 0, tint: 0, title: '① 路由专家 · 8 个', style: 'width:100%',
      sub: 'gate → top-8 → Σ wᵢ · Eᵢ<br><span class="mono" style="font-size:10px">竞争上岗，每个 token 选的都不一样</span>' });
    const midB = W.card({ cc: 1, tint: 1, title: '② 共享专家 · 1 个', style: 'width:100%',
      sub: 'Glm5NextTextMLP(residuals)<br><span class="mono" style="font-size:10px">不参与竞争，每个 token 都过</span>' });
    const mid = U.el('div', { class: 'col gap12', style: 'flex:2.1;min-width:0' }, [midA, midB]);

    const mid2 = U.el('div', { class: 'col center gap8', style: 'flex:none;padding:0 4px' });
    mid2.innerHTML = '<div style="font-size:22px;color:var(--accent)">⟹</div>'
      + '<div class="mono" style="font-size:10px;color:var(--c2)">相加</div>';

    const right = W.card({ cc: 3, tint: 3, title: '+ 输出', style: 'flex:1',
      body: U.el('div', { class: 'mono', style: 'font-size:11px;line-height:1.7;color:var(--ink-dim)' },
        [U.el('div', { html: 'routed + shared' }),
         U.el('div', { class: 'dim', html: '↓' }),
         U.el('div', { html: '(b, s, 4096)' })]) });

    viz.appendChild(left); viz.appendChild(mid); viz.appendChild(mid2); viz.appendChild(right);

    const bar = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(bar);
    bar.innerHTML = '<span class="cm">// hidden_states = experts(...) + shared_experts(residuals)</span>';

    /* 数据包：两条路径都汇到「+」。
       canvas 的尺寸由引擎的 FLOW.resize() 按**舞台坐标**统一处理
       （见 engine.js 里关于「屏幕像素 vs 舞台像素」的注释），
       本课不需要再自己钉一遍。 */
    try {
      FLOW.attach(U.q('#visual'));
      FLOW.pathBetween('routed', midA, right);
      FLOW.pathBetween('shared', midB, right);
      FLOW.rail('routed', 1200, '56,189,248');
      FLOW.rail('shared', 6200, '167,139,250');
      tl.at(5000, () => { FLOW.send('routed', { at: 5000, dur: 1500, color: '56,189,248', label: '8 个' }); });
      tl.at(9500, () => { FLOW.send('shared', { at: 9500, dur: 1500, color: '167,139,250', label: 'always' }); });
    } catch (e) { /* FLOW 只做装饰，失败不影响本幕 */ }

    tl.at(1500, () => {
      midA.classList.add('ac');
      bar.innerHTML = '<span class="cm">// ① 路由专家的输出被 8 个权重加权求和</span>';
    });
    tl.at(5000, () => {
      midB.classList.add('ac');
      bar.innerHTML = '② <em>self.shared_experts(residuals)</em>：吃的是**原始输入**，不是路由专家的输出';
    });
    tl.at(9500, () => {
      right.classList.add('ac');
      bar.innerHTML = '两路<em>相加</em>后才是这一层的输出 <span class="cm">// 加法，不是门控、不是拼接</span>';
    });
    tl.at(13000, () => {
      midA.classList.remove('ac');
      bar.innerHTML = '没有共享专家，公共知识就要被 8 个专家各自重复学 —— 共享专家是<span class="hl2">不参与竞争的快车道</span>';
    });
    tl.at(16000, () => {
      midA.classList.add('ac');
      bar.innerHTML = '<span class="cm">// 8 + 1 里的那个 1，全部理由就在这里</span>';
    });
  },
},

/* ------------------------------------------------- 5 打分 */
{
  kicker: '第四幕 · 打分',
  title: '先打分：<span class="hl-a">sigmoid</span>，不是 softmax',
  sub: 'gate 是一个 288 × 4096 的矩阵，把 hidden 投成 288 个互相独立的分数。',
  caption: '分数用 fp32 算：288 路的 top-k 对数值误差很敏感，bf16 抖动会让同一 token 两次选到不同专家。',
  lang: 'python',
  codeStart: 159,
  code: `    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        scores = router_logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias`,
  codeNote: 'scores 用 sigmoid；选谁用 scores_for_choice（带 bias），权重用 scores（不带）。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow' });
    wrap.appendChild(viz);

    /* 确定性伪随机：保证每次重放形状一模一样 */
    let seed = 20250925;
    const rnd = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; };
    const gauss = () => {
      const u = Math.max(1e-6, rnd()), v = rnd();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    };
    const z = [];
    for (let i = 0; i < 288; i++) z.push(gauss() * 0.95);
    const sc = z.map(x => 1 / (1 + Math.exp(-x)));
    const order = sc.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
    const topSet = new Set(order.slice(0, M.topk).map(p => p[1]));

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:center;justify-content:center' });

    const colL = U.el('div', { class: 'col gap6 center', style: 'flex:none' });
    colL.appendChild(U.el('div', { class: 'klabel', text: 'self.weight (288, 4096) 的示意切面' }));
    const gm = W.matrix(12, 24, (r, c) => Math.abs(Math.sin((r * 24 + c) * 0.7)), { cell: 10, gap: 1.5 });
    colL.appendChild(gm);
    colL.appendChild(U.el('div', { class: 'mono dim', style: 'font-size:10.5px', text: '1.18 M 参数 —— 路由本身很便宜' }));
    row.appendChild(colL);

    const mid = U.el('div', { class: 'col center gap6', style: 'flex:none;padding:0 6px' });
    mid.innerHTML = '<div class="mono" style="font-size:10px;color:var(--c2)">F.linear + sigmoid</div>'
      + '<div style="font-size:20px;color:var(--accent)">⟹</div>'
      + '<div class="mono" style="font-size:10px;color:var(--ink-faint)">fp32</div>';
    row.appendChild(mid);

    const colR = U.el('div', { class: 'col gap6 center', style: 'flex:none' });
    colR.appendChild(U.el('div', { class: 'klabel', text: '288 个分数（越亮越高，方框 = top-8）' }));
    const gs = W.matrix(12, 24, (r, c) => sc[r * 24 + c], { cell: 10, gap: 1.5 });
    colR.appendChild(gs);
    colR.appendChild(U.el('div', { class: 'mono dim', style: 'font-size:10.5px', text: '288 个分数互相独立，和不为 1' }));
    row.appendChild(colR);

    row.style.opacity = '0';
    viz.appendChild(row);

    const chips = U.el('div', { class: 'row gap10 wrap', style: 'width:100%;justify-content:center' });
    chips.innerHTML =
      '<span class="chip c0">scores = router_logits.sigmoid()</span>' +
      '<span class="chip c2">scores_for_choice = scores + e_score_correction_bias</span>';
    viz.appendChild(chips);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 路由是逐 token 的：view(-1, hidden_dim) 之后看不到邻居</span>';
    tl.at(1600, () => {
      row.style.transition = 'opacity .6s cubic-bezier(.16,1,.3,1)';
      row.style.opacity = '1';
      msg.innerHTML = 'gate 只有 <em>288 × 4096 = 1.18M</em> 参数（专家池是 7274.1M）—— <span class="cm">路由廉价，专家昂贵</span>';
    });
    tl.at(5200, () => {
      order.slice(0, M.topk).forEach(p => {
        const e = gs.at(Math.floor(p[1] / 24), p[1] % 24);
        if (e) { e.style.boxShadow = '0 0 0 1.5px #fff, 0 0 10px -1px var(--accent)'; e.style.zIndex = '3'; }
      });
      msg.innerHTML = 'sigmoid 之后，每个专家<em>独立</em>回答「我有多想做这个 token」 <span class="cm">// softmax 会把 288 路耦合成一个分布</span>';
    });
    tl.at(8800, () => {
      msg.innerHTML = '选谁用 <em>带 bias</em> 的分数；权重取自<em>不带 bias</em> 的 scores —— bias 只调负载，不改数值尺度';
    });
    tl.at(12000, () => {
      msg.innerHTML = '分数用 <em>fp32</em> 算：bf16 下 top-k 会抖，同一个 token 两次可能选到不同专家';
    });
    tl.at(14500, () => {
      msg.innerHTML = '<span class="cm">// 但 sigmoid 的分数天然偏小，所以后面还要归一化和缩放（见第七幕）</span>';
    });
  },
},

/* ------------------------------------------------- 6 分组路由 */
{
  kicker: '第五幕 · 分组',
  title: '★ 两级筛选：先在<span class="hl-c">组</span>里挑，再在组内选专家',
  sub: 'n_group 把 288 个专家切成若干组，每组取前 2 名分数之和当「组分数」，只保留 topk_group 个组。',
  caption: '默认 n_group=1 / topk_group=1 —— 只有一组，这段代码退化成空操作。',
  lang: 'python',
  codeStart: 164,
  code: `        group_scores = (
            scores_for_choice.view(-1, self.num_group, self.num_experts // self.num_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)`,
  codeNote: 'group_scores 用 top-2 之和而不是最大值：不让一个高分单独决定整组命运。',
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '示意 n_group = 4 / topk_group = 2：4 个组各 72 个专家（每格代表 6 个）' }));
    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:flex-start' });
    const groups = [];
    for (let g = 0; g < 4; g++) {
      const col = U.el('div', { class: 'col gap6', style: 'flex:1 1 0;min-width:0' });
      col.appendChild(U.el('div', { class: 'chip c1', style: 'align-self:flex-start', text: '组 ' + g + ' · 72 个' }));
      const bx = U.el('div', { style: 'display:grid;grid-template-columns:repeat(6,1fr);gap:5px;width:100%' });
      const cells = [];
      for (let i = 0; i < 12; i++) {
        const e = U.el('div', {
          title: '组 ' + g + ' 的专家（示意）',
          style: 'height:30px;border-radius:5px;min-width:0;border:1px solid rgba(150,180,255,.16);'
               + 'background:rgba(150,180,255,.07);'
               + 'transition:background .45s cubic-bezier(.16,1,.3,1),border-color .45s,opacity .45s,transform .45s' });
        bx.appendChild(e); cells.push(e);
      }
      col.appendChild(bx);
      const pill = U.el('div', { class: 'pill', style: 'align-self:stretch;justify-content:center', text: '组分数 = ?' });
      col.appendChild(pill);
      row.appendChild(col);
      groups.push({ col, cells, pill });
    }
    viz.appendChild(row);

    const bottom = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const b1 = W.card({ cc: 2, tint: 2, title: '每组取 top-2 之和', style: 'flex:1',
      sub: '<span class="mono" style="font-size:10.5px">.topk(2, dim=-1)[0].sum(dim=-1)</span>' });
    const b2 = W.card({ cc: 3, tint: 3, title: '落选组整组屏蔽', style: 'flex:1',
      sub: '<span class="mono" style="font-size:10.5px">masked_fill(~score_mask, float("-inf"))</span>' });
    const b3 = W.card({ cc: 0, tint: 0, title: '组内再 topk(top_k)', style: 'flex:1',
      sub: '<span class="mono" style="font-size:10.5px">torch.topk(scores_for_choice, k=self.top_k)</span>' });
    bottom.appendChild(b1); bottom.appendChild(b2); bottom.appendChild(b3);
    viz.appendChild(bottom);

    const why = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const w1 = W.card({ cc: 1, title: '动机 ① 通信', style: 'flex:1',
      sub: '288 选 8 是全对全比较；专家并行下每一轮路由都是一次 all-to-all。先在组粒度收敛范围，通信模式才可预测。' });
    const w2 = W.card({ cc: 5, title: '动机 ② 负载', style: 'flex:1',
      sub: '只按分数选，容易出现「8 个都落在同一台设备」或「每台都碰一点」的抖动 —— 组相当于给路由加的配额。' });
    why.appendChild(w1); why.appendChild(w2);
    viz.appendChild(why);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    const pick2 = [1, 4];
    msg.innerHTML = '<span class="cm">// 第一步：每个组各自算一个「组分数」</span>';
    tl.at(1600, () => {
      groups.forEach((G, gi) => pick2.forEach((k, n) => {
        const c = G.cells[k + gi];
        c.style.borderColor = 'var(--c2)';
        c.style.background = 'rgba(251,191,36,.30)';
        c.style.transform = 'translateY(-2px)';
      }));
      b1.classList.add('ac');
      msg.innerHTML = '每组选出<em>最高分</em>的两个专家，把它们的分数<em>相加</em>当作这个组的代表分';
    });
    tl.at(5200, () => {
      groups.forEach((G, gi) => {
        G.pill.className = 'pill ' + (gi < 2 ? 'ok' : '');
        G.pill.textContent = '组分数 ' + (0.84 - gi * 0.13).toFixed(2);
      });
      msg.innerHTML = '组分数出场：<em>0.84 / 0.71 / 0.58 / 0.45</em> —— 现在比较的粒度从 288 降到了 4';
    });
    tl.at(8800, () => {
      b1.classList.remove('ac'); b2.classList.add('ac');
      groups.forEach((G, gi) => {
        if (gi < 2) { G.pill.className = 'pill ok'; return; }
        G.col.style.opacity = '.26';
        G.cells.forEach(c => { c.style.background = 'rgba(150,180,255,.04)'; c.style.borderColor = 'rgba(150,180,255,.08)'; });
        G.pill.className = 'pill bad';
      });
      msg.innerHTML = 'topk_group = 2：落选的<em>整组</em>被打成 -inf，后面的 top-k 不可能再碰到它们';
    });
    tl.at(12400, () => {
      b2.classList.remove('ac'); b3.classList.add('ac');
      msg.innerHTML = '最后才在<em>选中的 2 个组里</em>做 288 选 8 <span class="cm">// 两级筛选：组 → 专家</span>';
    });
    tl.at(15800, () => {
      b3.classList.remove('ac');
      msg.innerHTML = '<span class="cm">// 默认 n_group=1 / topk_group=1：只有一组，整组必然入选 —— 这段是空操作</span>';
    });
    tl.at(17800, () => {
      w1.classList.add('ac'); w2.classList.add('ac');
      msg.innerHTML = '分组的代价是路由自由度变小，换来的是<em>可预测的通信</em>与<em>更稳的负载</em>';
    });
  },
},

/* ------------------------------------------------- 7 归一化 + 2.5 倍 */
{
  kicker: '第六幕 · 缩放',
  title: 'norm_topk_prob 之后，权重和正好 <span class="hl-c">2.5</span>',
  sub: '先归一化让 8 个权重和为 1，再乘 routed_scaling_factor=2.5 —— 因为 top-k 只留下了一小部分概率。',
  caption: '实测（8 专家 / top-2）：[0.736, 0.606] → [0.548, 0.452] → [1.371, 1.129]，和恒等于 2.5。',
  lang: 'python',
  codeStart: 179,
  code: `        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return router_logits, topk_weights, topk_indices`,
  codeNote: '权重取自 scores（不带 bias）；归一化之后再乘 2.5。',
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const mk = (title, cc, vals, sumText, note) => {
      const c = W.card({ cc: cc, style: 'flex:1 1 0;min-width:0', title: title, sub: note });
      const box = U.el('div', { class: 'col gap6', style: 'width:100%;margin-top:8px' });
      const fills = vals.map((v, i) => {
        const r = U.el('div', { class: 'row gap8', style: 'width:100%' });
        r.appendChild(U.el('div', { class: 'mono', style: 'font-size:10.5px;color:var(--ink-dim);width:52px;flex:none',
          text: i === 0 ? '专家 #7' : '专家 #5' }));
        const track = U.el('div', { style: 'flex:1;height:12px;border-radius:5px;background:rgba(150,180,255,.09);'
          + 'border:1px solid rgba(150,180,255,.10);overflow:hidden;min-width:0' });
        const fill = U.el('div', { style: 'height:100%;width:0;border-radius:4px;'
          + 'background:linear-gradient(90deg,var(--c' + cc + '),rgba(255,255,255,.7));'
          + 'transition:width .85s cubic-bezier(.16,1,.3,1)' });
        track.appendChild(fill); r.appendChild(track);
        r.appendChild(U.el('div', { class: 'mono', style: 'font-size:10.5px;color:#fff;width:46px;flex:none;text-align:right',
          text: v.toFixed(3) }));
        box.appendChild(r);
        return fill;
      });
      c.appendChild(box);
      const s = U.el('div', { class: 'mono', style: 'font-size:11.5px;margin-top:9px;color:var(--ink-dim)' });
      c.appendChild(s);
      return { c, fills, s, sumText };
    };

    const MAXV = 1.5;
    const s1 = mk('① scores（sigmoid）', 0, [0.736, 0.606], '8 个分数之和 = 1.342',
      'sigmoid 的分数天然偏小：top-k 只留下了一部分');
    const s2 = mk('② norm_topk_prob = True', 2, [0.548, 0.452], '归一化后之和 = 1.000',
      '除以自己的和：只看相对比例，不看绝对大小');
    const s3 = mk('③ × routed_scaling_factor', 4, [1.371, 1.129], '× 2.5 之后之和 = 2.500',
      '补回 top-k 丢掉的那部分，抬到与残差可比的量级');
    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    [s1, s2, s3].forEach(o => row.appendChild(o.c));
    viz.appendChild(row);

    const why = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const wA = W.card({ cc: 3, title: '不补回来会怎样', style: 'flex:1',
      sub: 'sigmoid 的分数在 0.5 附近，top-8 又只留下一小部分；不放大，这一路的输出相对残差流'
         + '就几乎可以忽略，42 层稀疏层等于白加。' });
    const wB = W.card({ cc: 4, title: '怎么检查它生效了', style: 'flex:1',
      sub: '把 <span class="mono" style="font-size:10.5px">topk_weights</span> 沿最后一维求和：'
         + '开了 norm_topk_prob 时应恒等于 <span class="mono" style="font-size:10.5px">routed_scaling_factor</span>。'
         + '实测 8 专家 / top-2 的权重和正好 2.500。' });
    why.appendChild(wA); why.appendChild(wB);
    viz.appendChild(why);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    const grow = (o, t) => o.fills.forEach(f => { f.style.width = (t / MAXV * 100).toFixed(1) + '%'; });
    const show = (o) => { o.s.innerHTML = o.sumText; };

    msg.innerHTML = '<span class="cm">// 从 288 个分数里 gather 出被选中的 8 个权重</span>';
    tl.at(1600, () => {
      grow(s1, 0.736); show(s1);
      /* 两条不同高度 */
      s1.fills[1].style.width = (0.606 / MAXV * 100).toFixed(1) + '%';
      msg.innerHTML = '① 被选中的两个分数：<em>0.736</em> 与 <em>0.606</em>，和只有 <em>1.342</em>';
    });
    tl.at(5200, () => {
      grow(s2, 0.548); s2.fills[1].style.width = (0.452 / MAXV * 100).toFixed(1) + '%';
      show(s2);
      msg.innerHTML = '<em>norm_topk_prob = True</em>：除以自己的和 → <em>[0.548, 0.452]</em>，和恰好 <em>1.000</em>';
    });
    tl.at(9000, () => {
      grow(s3, 1.371); s3.fills[1].style.width = (1.129 / MAXV * 100).toFixed(1) + '%';
      show(s3);
      msg.innerHTML = '③ 乘 <em>2.5</em> → <em>[1.371, 1.129]</em>：<span class="hl3">权重和恒等于 routed_scaling_factor</span>';
    });
    tl.at(12800, () => {
      msg.innerHTML = '为什么要补回来：这一路的输出要与<em>残差流</em>直接相加，整体偏小就等于这 42 层几乎没贡献';
    });
    tl.at(15800, () => {
      wA.classList.add('ac');
      msg.innerHTML = '<span class="cm">// 检查这一行有没有生效：把 topk_weights 求和，应当等于 2.5</span>';
    });
    tl.at(17600, () => {
      wA.classList.remove('ac'); wB.classList.add('ac');
      msg.innerHTML = '实测权重和 <em>2.500</em> —— 归一化 + 缩放这两步都生效了，这是最快的自检';
    });
  },
},

/* ------------------------------------------------- 8 训练 vs 推理 */
{
  kicker: '第七幕 · 训练 vs 推理',
  title: '推理时 <span class="hlbad">load_balancing_loss_func</span> 根本不会被调用',
  sub: '它是训练用的辅助损失：只在 output_router_logits 为真时才算，而推理默认是 False。',
  caption: '实测 load_balancing_loss_func(None, ...) 与传单层张量都返回 0（不是抛错）—— 别靠「没报错」判断它生效了。',
  lang: 'python',
  codeStart: 2248,
  code: `        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device`,
  codeNote: '两层开关：output_router_logits（默认 False）与 labels is not None。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const tr = W.card({ cc: 2, tint: 2, title: '训练：labels 不为 None', style: 'flex:1',
      sub: 'output_router_logits = True<br><span class="mono" style="font-size:10.5px">aux_loss 会乘上 router_aux_loss_coef（默认 0.001）加进 loss</span>',
      body: U.el('div', { class: 'mono', style: 'font-size:11px;line-height:1.8;color:var(--ink-dim);margin-top:6px' },
        [U.el('div', { html: 'router_logits（45 层）' }),
         U.el('div', { class: 'dim', html: '↓' }),
         U.el('div', { html: 'tokens_per_expert × router_prob' }),
         U.el('div', { class: 'dim', html: '↓ 惩罚不均衡的路由' }),
         U.el('div', { class: 'hl3', html: 'loss += 0.001 × aux_loss' })]) });
    const inf = W.card({ cc: 4, tint: 4, title: '推理：labels 为 None', style: 'flex:1',
      sub: 'output_router_logits 默认 False（实测）<br><span class="mono" style="font-size:10.5px">aux_loss = None，不进任何计算</span>',
      body: U.el('div', { class: 'mono', style: 'font-size:11px;line-height:1.8;color:var(--ink-dim);margin-top:6px' },
        [U.el('div', { html: 'logits = lm_head(...)' }),
         U.el('div', { class: 'dim', html: '↓ 采样' }),
         U.el('div', { html: 'next_token' }),
         U.el('div', { class: 'dim', html: '↓' }),
         U.el('div', { class: 'hl4', html: '这一项不存在' })]) });
    row.appendChild(tr); row.appendChild(inf);
    viz.appendChild(row);

    const gate = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:center' });
    gate.appendChild(W.card({ cc: 0, style: 'flex:1',
      title: '<span class="mono" style="font-size:11.5px">output_router_logits = False</span>',
      sub: '实测 Glm5NextTextConfig() 的默认值。它同时决定「要不要收集 router_logits」——_can_record_outputs 里挂的钩子，要的时候才收。' }));
    viz.appendChild(gate);

    const chips = U.el('div', { class: 'row gap10 wrap', style: 'width:100%' });
    chips.innerHTML =
      '<span class="chip c3">load_balancing_loss_func(None, ...) → 0</span>' +
      '<span class="chip c3">load_balancing_loss_func(单层张量, ...) → 0</span>' +
      '<span class="chip c1">router_aux_loss_coef = 0.001</span>';
    viz.appendChild(chips);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 这个函数的名字里有 loss —— 它属于训练循环，不属于前向</span>';
    tl.at(1600, () => {
      tr.classList.add('ac');
      msg.innerHTML = '训练：<em>output_router_logits=True</em> 且有 labels → 追加一项辅助损失，惩罚「路由太不均衡」';
    });
    tl.at(5200, () => {
      tr.classList.remove('ac'); inf.classList.add('ac');
      msg.innerHTML = '推理：<em>labels 为 None</em> → 第二层保险直接挡住，<code class="inl">loss +=</code> 那行不会执行';
    });
    tl.at(8800, () => {
      msg.innerHTML = '第一层开关 <em>output_router_logits</em> 默认 <span class="hlbad">False</span> → <code class="inl">aux_loss</code> 就是 <em>None</em>';
    });
    tl.at(12000, () => {
      msg.innerHTML = '还有个坑：<em>gate_logits 不是 tuple 就直接 return 0</em> —— 实测两种错传都静默返回 0';
    });
    tl.at(14800, () => {
      msg.innerHTML = '<span class="cm">// 结论：decode 的算力里没有这一项。但它解释了训练时为什么要分组路由</span>';
    });
  },
},

/* ------------------------------------------------- 9 收束 */
{
  kicker: '收束',
  title: '把「288 选 8」压成一张表',
  sub: '现在你应该能算出稀疏层与稠密层的参数比，并说出共享专家为什么存在。',
  caption: '下一课 L0-05：mHC —— 4 条残差流，共享专家的输出加到哪一条。',
  lang: 'python',
  codeStart: 2052,
  code: `def load_balancing_loss_func(
    gate_logits: torch.Tensor | tuple[torch.Tensor] | None,
    num_experts: int | None = None,
    top_k=2,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor | int:`,
  codeNote: '回到训练那一侧：这个损失就是「8 个专家怎么排班」的指挥棒。',
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap10 vizgrow hstart' });
    wrap.appendChild(viz);

    const tb = W.table([
      ['专家池', '<span class="hl">288</span>', 'n_routed_experts，每层一份'],
      ['每 token 选中', '<span class="hl">8</span>', 'num_experts_per_tok'],
      ['共享专家', '<span class="hl2">1</span>', 'n_shared_experts，宽度 = 1 个专家'],
      ['稀疏 : 稠密 参数', '<span class="hl3">48.17×</span>', '7274.1 M : 151.0 M'],
      ['稀疏 : 稠密 算力', '<span class="hl4">1.51×</span>', '227.7 M : 151.0 M'],
      ['每 token 激活比', '<span class="hl">3.125%</span>', '(8 + 1) / 288'],
      ['整模型 总参 / 激活', '<span class="hl">313.79 B / 17.84 B</span>', '5.68%'],
      ['routed_scaling_factor', '<span class="hl3">2.5</span>', 'topk_weights 之和恒等于它'],
    ], { head: ['量', '实测值', '来源'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 0, style: 'padding:10px 12px;width:100%' }, [
      U.el('div', { class: 'klabel', text: '本课用到的实测数字（meta 设备 + 真实 TopkRouter）' }), tb,
    ]));

    viz.appendChild(W.exercise(
      '一个稀疏层的 MLP 部分有 <b>7274.1M</b> 参数，每个 token 实际算多少？'
      + '相对稠密层的 <b>151.0M</b>，参数比与算力比各是多少？',
      '每 token 激活 = 8 × 25.17M（路由专家）+ 25.17M（共享专家）+ 1.18M（gate）= <b>227.7M</b>。'
      + '<br>参数比 = 7274.1 / 151.0 = <b>48.17×</b>；算力比 = 227.7 / 151.0 = <b>1.51×</b>。'
      + '<br>「3% 激活」的来历：被激活的有 8 个路由专家 + 1 个共享专家，'
      + '(8 + 1) / 288 = <b>3.125%</b>。'
      + '<br>共享专家的理由：路由专家在竞争中被逼向专精，公共知识需要一条'
      + '<b>不参与竞争</b>的通道去承接，否则要么被重复学 8 遍、要么谁都学不好。'));

    const chips = U.el('div', { class: 'row gap10 wrap', style: 'width:100%' });
    chips.innerHTML =
      U.chip('容量按 288 份买', 2) + U.chip('账单按 9 份付', 0) +
      U.chip('共享专家 = 不竞争的快车道', 1) +
      '<span class="dim mono" style="margin-left:auto;font-size:11px">下一课 L0-05 · mHC 的 4 条残差流</span>';
    viz.appendChild(chips);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 一张表就是这一课的全部</span>';
    tl.at(1800, () => { msg.innerHTML = '参数看 <em>288</em>，算力看 <em>9</em>：48.17× 与 1.51× 的落差就是 MoE 经济学'; });
    tl.at(5200, () => { msg.innerHTML = '每 token 激活 <em>17.84 B</em> / 总参 <em>313.79 B</em> —— 这就是「18B 激活 / 320B 总参」'; });
    tl.at(8800, () => { msg.innerHTML = '共享专家不是「第 289 个候选」，它是<em>每个 token 的必经之路</em>'; });
    tl.at(12400, () => { msg.innerHTML = '训练用 <em>load_balancing_loss_func</em> 给路由排班；推理路径上它<em>不存在</em>'; });
    tl.at(15600, () => { msg.innerHTML = '<span class="cm">// 下一课 L0-05：这 42 层稀疏层的输出，加到 mHC 的哪一条残差流上</span>'; });
  },
},

];
