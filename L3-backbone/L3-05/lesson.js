/* ==========================================================================
   L3-05 · MoE 装配与共享专家
   --------------------------------------------------------------------------
   覆盖：models/glm5_next/modeling_glm5_next.py（1 个文件 / 2444 行）
   目标：能画出 MoE 前向的数据流（含共享专家的加法支路），并说清
         output_router_logits 在推理时到底改变了什么（实测：什么都不改变）。
   排版基准：#visual 可视区 = 994 x 662.7 舞台像素（无头浏览器实测）。
   ========================================================================== */
'use strict';

/* 实测常量：_recon/probe_moe.py / probe_moe2.py / probe_moe3.py 跑出，不是估算 */
const M = {
  H: 4096, I: 2048, E: 288, topk: 8, shared: 1, denseI: 12288,
  routedOne: 25165824,        // 2*I*H + H*I
  routedAll: 7247757312,      // E * routedOne
  sharedOne: 25165824,        // 3*I*H（n_shared_experts = 1）
  denseOne: 150994944,        // 3*denseI*H
  scale: 2.5,                 // routed_scaling_factor
  layers: 45, denseLayers: 3, sparseLayers: 42,
};

const SCENES = [

/* ------------------------------------------------- 1 全景：三条支路 */
{
  kicker: '第 3 层 · 文本主干 · 装配',
  title: '一个稀疏层里，<span class="hl-a">三条支路</span>',
  sub: 'Glm5NextTextMoE 只做一件事：把「被选中的 8 个专家」和「每层都开的共享专家」相加。先看这张总图。',
  caption: '这一课只回答一个问题：一个 token 进了稀疏层以后，走了几条路、在哪一行合流。',
  lang: 'python',
  codeStart: 187,
  code: `class Glm5NextTextMoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config: Glm5NextConfig):
        super().__init__()
        self.config = config
        self.experts = Glm5NextTextExperts(config)
        self.gate = Glm5NextTextTopkRouter(config)
        self.shared_experts = Glm5NextTextMLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )`,
  codeNote: '三个子模块：gate 只出分数，experts 出 8 选 1 的加权和，shared_experts 出无损旁路。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart', style: 'width:100%;padding-top:4px;justify-content:space-evenly' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '一个稀疏层（Glm5NextTextMoE）的三条支路' }));

    const row = U.el('div', { class: 'row gap8', style: 'width:100%;align-items:stretch' });
    const mk = (cc, title, lines, flex) => {
      const c = W.card({ cc: cc, num: null, title: title,
        body: U.el('div', { class: 'mono', style: 'font-size:11px;line-height:1.75;color:var(--ink-dim)' },
          lines.map(t => U.el('div', { html: t }))) });
      c.style.flex = flex || '1 1 0';
      row.appendChild(c);
      return c;
    };
    const cIn = mk(0, '输入 hidden_states', [
      '(batch, seq, 4096)',
      '<span class="cm">// 第 2 行：原样留一份</span>',
      '<span style="color:var(--c0)">residuals = hidden</span>',
    ], '0.95 1 0');
    const cGate = mk(1, '① gate', [
      '<span style="color:var(--c1)">Glm5NextTextTopkRouter</span>',
      '(tokens, 288) → top-8',
      '<span class="cm">// 只出分数，不碰 hidden</span>',
    ]);
    const cRoute = mk(2, '② experts', [
      '<span style="color:var(--c2)">Glm5NextTextExperts</span>',
      '8 × (2048 → 2048)',
      '<span class="cm">// 按 topk_indices 分组算</span>',
    ]);
    const cShared = mk(4, '③ shared_experts', [
      '<span style="color:var(--c4)">Glm5NextTextMLP</span>',
      '2048 = moe_intermediate_size',
      '<span class="cm">// 无 top-k、无分组</span>',
    ]);
    const cOut = mk(5, '输出', [
      '<span style="color:var(--c5)">routed + shared</span>',
      '(batch, seq, 4096)',
      '<span class="cm">// 无权相加</span>',
    ], '0.95 1 0');

    viz.appendChild(row);

    const arrowRow = U.el('div', { class: 'row gap8', style: 'width:100%' });
    arrowRow.innerHTML =
      '<div class="mono" style="flex:1;text-align:center;font-size:11px;color:var(--c1)">└─ ① 决定「谁被选中」 ─┐</div>' +
      '<div class="mono" style="flex:1;text-align:center;font-size:11px;color:var(--c2)">② 只用 indices + weights ─┤</div>' +
      '<div class="mono" style="flex:1;text-align:center;font-size:11px;color:var(--c4)">③ 吃 original hidden ────┘</div>';
    viz.appendChild(arrowRow);

    const stats = [
      { k: 'num_experts_per_tok', v: M.topk, cc: 2, note: '每 token 选中的路由专家数' },
      { k: 'n_shared_experts', v: M.shared, cc: 4, note: '每层都开的共享专家数' },
      { k: 'routed_scaling_factor', v: M.scale, cc: 1, note: 'topk_weights 每行之和' },
      { k: 'mlp_layer_types', v: M.sparseLayers, cc: 0, note: '45 层里有 42 层是稀疏' },
    ];
    const srow = U.el('div', { class: 'row gap10', style: 'width:100%;align-items:center' });
    const sEls = stats.map(st => {
      const c = W.card({ cc: st.cc, tint: st.cc, style: 'flex:1 1 0',
        title: '<span class="mono" style="font-size:10.5px">' + st.k + '</span>',
        body: U.el('div', { class: 'n big', text: U.fmt(st.v) }),
        sub: '<span style="font-size:10.5px">' + st.note + '</span>' });
      srow.appendChild(c); return c;
    });
    viz.appendChild(srow);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    const all = [cIn, cGate, cRoute, cShared, cOut];
    msg.innerHTML = '<span class="cm">// 先记住三块分工，下面逐块拆</span>';
    tl.at(1600, () => {
      cGate.classList.add('ac');
      msg.innerHTML = '① <span class="fn">gate</span> 是唯一读 <em>hidden_states</em> 去算「选谁」的模块 —— 它的输出是分数，不是特征';
    });
    tl.at(4800, () => {
      cGate.classList.remove('ac'); cRoute.classList.add('ac');
      msg.innerHTML = '② <span class="fn">experts</span> 只拿到 <em>topk_indices</em> 与 <em>topk_weights</em>，按专家号把 token 分组做 FFN';
    });
    tl.at(8000, () => {
      cRoute.classList.remove('ac'); cShared.classList.add('ac');
      msg.innerHTML = '③ <span class="fn">shared_experts</span> 的中间维被显式设成 <em>2048</em>：一个路由专家那么大，但每层无条件执行';
    });
    tl.at(11200, () => {
      cShared.classList.remove('ac'); cOut.classList.add('ac');
      msg.innerHTML = '合流：<em>routed + shared</em>，<span class="hlbad">没有系数</span> —— 不是加权融合，是加法';
    });
    tl.at(14000, () => {
      all.forEach(c => c.classList.remove('ac')); sEls.forEach(c => c.classList.add('ac'));
      msg.innerHTML = '每 token 用到 <em>9</em> 个专家（8 路由 + 1 共享）／共 <em>289</em> 个 → <span class="hl">3.11%</span>';
    });
    tl.at(15800, () => { sEls.forEach(c => c.classList.remove('ac')); });
  },
},

/* ------------------------------------------------- 2 前向 8 行 */
{
  kicker: '装配 · 主前向',
  title: '整个 MoE 前向只有 <span class="hl-a">8 行</span>',
  sub: '分支与合流都在这 8 行里。第 2 行留下的那份 residuals 是理解共享专家的钥匙。',
  caption: '注意第 4 行：gate 返回三元组，第一个用 _ 接住 —— 那是 router_logits，第五幕展开。',
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
  codeNote: '第 2 行在任何 reshape 之前留一份原样的输入，第 7 行把它喂给 shared_experts。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap10 vizgrow hstart', style: 'width:100%;padding-top:4px;justify-content:space-evenly' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '逐行追踪：形状怎么变（实测小配置 batch=2, seq=5, H=64）' }));

    const steps = [
      { ln: 202, code: 'residuals = hidden_states', shp: '(2, 5, 64)', cc: 4, note: '原样留一份，供共享专家使用' },
      { ln: 203, code: 'orig_shape = hidden_states.shape', shp: 'torch.Size([2, 5, 64])', cc: 0, note: '记下形状，最后掰回去' },
      { ln: 204, code: '_, topk_weights, topk_indices = self.gate(hidden_states)', shp: 'w (10, 2) / idx (10, 2)', cc: 1, note: '第 0 个返回值被 _ 丢掉' },
      { ln: 205, code: 'hidden_states.view(-1, hidden_states.shape[-1])', shp: '(10, 64)', cc: 2, note: 'batch 与 seq 合并成 token 维' },
      { ln: 206, code: 'self.experts(...).view(*orig_shape)', shp: '(10, 64) → (2, 5, 64)', cc: 2, note: '按专家分组算完再拼回来' },
      { ln: 207, code: 'hidden_states + self.shared_experts(residuals)', shp: '(2, 5, 64) + (2, 5, 64)', cc: 5, note: '★ 合流：无权相加' },
    ];
    const rows = steps.map(s => {
      const c = U.el('div', {
        class: 'row gap8',
        style: 'width:100%;align-items:center;padding:5px 9px;border-radius:9px;'
             + 'border:1px solid var(--panel-brd);background:rgba(150,180,255,.05);'
             + 'transition:background .34s var(--ease-out),border-color .34s,opacity .34s',
      });
      c.appendChild(U.el('div', { class: 'n sm', style: 'flex:none;width:34px;color:var(--c' + s.cc + ')', text: 'L' + s.ln }));
      c.appendChild(U.el('div', { class: 'mono', style: 'flex:1 1 0;font-size:11.5px;color:var(--ink)', text: s.code }));
      c.appendChild(U.el('div', { class: 'mono', style: 'flex:none;font-size:11px;color:var(--c' + s.cc + ')', text: s.shp }));
      c.appendChild(U.el('div', { class: 'mono', style: 'flex:none;width:210px;font-size:10.5px;color:var(--ink-faint)', text: s.note }));
      viz.appendChild(c);
      return c;
    });

    const add = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch;margin-top:2px' });
    const cA = W.card({ cc: 4, tint: 4, style: 'flex:1 1 0', title: '共享专家这一路',
      body: U.el('div', { class: 'mono', style: 'font-size:11.5px;line-height:1.75;color:var(--ink-dim)' }, [
        U.el('div', { html: 'residuals <span class="cm">// 未展平、未路由</span>' }),
        U.el('div', { html: '↓ Glm5NextTextMLP(residuals)' }),
        U.el('div', { html: '<span style="color:var(--c4)">shared_out (2, 5, 64)</span>' }),
      ]) });
    const cB = W.card({ cc: 2, tint: 2, style: 'flex:1 1 0', title: '路由专家这一路',
      body: U.el('div', { class: 'mono', style: 'font-size:11.5px;line-height:1.75;color:var(--ink-dim)' }, [
        U.el('div', { html: 'view(-1, 64) → experts → view(2,5,64)' }),
        U.el('div', { html: '↓ 只有 8 个专家参与' }),
        U.el('div', { html: '<span style="color:var(--c2)">routed_out (2, 5, 64)</span>' }),
      ]) });
    add.appendChild(cA); add.appendChild(cB);
    viz.appendChild(add);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    steps.forEach((s, i) => {
      tl.at(1400 + i * 1800, () => {
        rows.forEach((r, k) => {
          r.style.background = (k === i) ? 'rgba(34,211,238,.10)' : 'rgba(150,180,255,.05)';
          r.style.borderColor = (k === i) ? 'var(--accent)' : 'var(--panel-brd)';
          r.style.opacity = (k === i) ? '1' : '.55';
        });
        msg.innerHTML = '<span class="cm">// L' + s.ln + '</span> ' + s.note;
      });
    });
    tl.at(12400, () => {
      rows.forEach(r => { r.style.opacity = '1'; r.style.background = 'rgba(150,180,255,.05)'; r.style.borderColor = 'var(--panel-brd)'; });
      msg.innerHTML = '两条路在 <em>L207</em> 合流 —— 一行加法，没有 alpha、没有 gate、没有 scale';
    });
    tl.at(14200, () => {
      cA.classList.add('ac'); cB.classList.remove('ac');
      msg.innerHTML = '实测：<span class="fn">allclose</span>(moe(x), experts(x) + shared_experts(x)) = <em>True</em>';
    });
    tl.at(16000, () => {
      cA.classList.remove('ac'); cB.classList.add('ac');
      msg.innerHTML = '共享专家吃的是 <em>residuals</em>，路由专家吃的是展平后的 hidden —— 两条路的输入就不同';
    });
  },
},

/* ------------------------------------------------- 3 分支条件 */
{
  kicker: '装配 · 分支条件',
  title: '稠密还是稀疏：<span class="hl-a">构造期</span>就定死了',
  sub: 'Glm5NextTextMoE 里没有任何「我是第几层」的概念。选不选它，发生在 decoder 层构造的时候。',
  caption: '这是实例化期的二选一，不是前向期的 if —— 构件完成后永不再变。',
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
  codeNote: '同一个文件里两条互相独立的排班轴：注意力看 layer_types，FFN 看 mlp_layer_types。',
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap10 vizgrow hstart', style: 'width:100%;padding-top:4px;justify-content:space-evenly' });
    wrap.appendChild(viz);
    /* 顶部留 4px：卡片选中时 .ac 会上移 3px，不能让首个子元素贴死在 0 处 */
    viz.appendChild(U.el('div', { class: 'klabel', style: 'margin:0;height:4px',
      text: 'mlp_layer_types[layer_idx] —— 两条分支的构造代码' }));

    const twoCol = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    viz.appendChild(twoCol);

    const left = W.card({ cc: 4, tint: 4, style: 'flex:1 1 0;display:flex;flex-direction:column',
      title: 'mlp_layer_types[i] == "dense"',
      sub: '<code class="inl">self.mlp = Glm5NextTextMLP(config)</code>' });
    const lk = (t, s, cc) => U.el('div', { class: 'row gap8', style: 'width:100%;align-items:center;padding:5px 9px;'
      + 'border-radius:9px;border:1px solid rgba(150,180,255,.16);background:rgba(150,180,255,.05);margin-top:7px' }, [
      U.el('div', { class: 'mono', style: 'flex:1 1 0;font-size:11.5px;color:var(--ink)', html: t }),
      U.el('div', { class: 'mono', style: 'flex:none;font-size:10.5px;color:var(--c' + cc + ')', text: s }),
    ]);
    left.appendChild(lk('gate_proj 4096 → 12288', '1 个', 4));
    left.appendChild(lk('act_fn(gate) * up', 'SwiGLU', 4));
    left.appendChild(lk('down_proj 12288 → 4096', '1 个', 4));
    left.appendChild(U.el('div', { class: 'formula', style: 'width:100%;margin-top:10px;font-size:12px;--cc:var(--c4)',
      html: '参数 = 3 × 12288 × 4096 = <em>150,994,944</em> &nbsp;<span class="cm">// 每层一份</span>' }));

    const right = W.card({ cc: 2, tint: 2, style: 'flex:1 1 0;display:flex;flex-direction:column',
      title: 'mlp_layer_types[i] == "sparse"',
      sub: '<code class="inl">self.mlp = Glm5NextTextMoE(config)</code>' });
    right.appendChild(lk('gate: 4096 → 288 分 → top-8', '路由器', 1));
    right.appendChild(lk('experts: 2048 → 2048 × 288', '288 个', 2));
    right.appendChild(lk('shared_experts: 2048 → 2048', '1 个', 4));
    right.appendChild(U.el('div', { class: 'formula', style: 'width:100%;margin-top:10px;font-size:12px;--cc:var(--c2)',
      html: '参数 = 7,247,757,312 + 25,165,824 ≈ <em>7.27 B</em> &nbsp;<span class="cm">// 每层一份</span>' }));

    twoCol.appendChild(left); twoCol.appendChild(right);

    const tl_ = W.table([
      ['mlp_layer_types 实测默认值', "<code class=\"inl\">['dense', 'dense', 'dense', 'sparse', …]</code>", '前 3 层稠密'],
      ['dense 层数 / sparse 层数', '<span class="hl4">3</span> / <span class="hl3">42</span>', 'num_hidden_layers = 45'],
      ['专家中间维 vs 稠密中间维', '2048 vs 12288', '稀疏化的第一刀：把专家做窄'],
    ], { head: ['项', '实测值', '含义'] });
    const tb = W.card({ cc: 0, style: 'width:100%;padding:10px 12px', kids: [tl_] });
    viz.appendChild(tb);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    left.style.opacity = '.45'; right.style.opacity = '.45';
    msg.innerHTML = '<span class="cm">// config.mlp_layer_types[layer_idx] 只看一个字符串</span>';
    tl.at(1600, () => {
      msg.innerHTML = '判据只有 <em>"sparse"</em> 一个字符串，相等就是 MoE，否则是稠密 MLP';
    });
    tl.at(4600, () => {
      left.style.opacity = '1'; left.classList.add('ac');
      msg.innerHTML = '稠密层：<em>layer 0/1/2</em> —— 一个完整的 12288 宽 FFN，没有任何路由';
    });
    tl.at(7800, () => {
      left.classList.remove('ac'); left.style.opacity = '.45';
      right.style.opacity = '1'; right.classList.add('ac');
      msg.innerHTML = '稀疏层：<em>layer 3…44</em> —— 288 个窄专家 + 1 个共享专家，但每 token 只碰 9 个';
    });
    tl.at(11000, () => {
      right.classList.remove('ac');
      tb.classList.add('ac');
      msg.innerHTML = '回顾 L3-03：<em>layer_types</em> 与 <em>mlp_layer_types</em> 是两条互不对齐的排班轴，这里看到的是 MLP 那一轴';
    });
    tl.at(13600, () => { tb.classList.remove('ac'); });
  },
},

/* ------------------------------------------------- 4 数据流与参数量 */
{
  kicker: '主视觉 · 数据流',
  title: 'MoE 前向的<span class="hl-a">数据流</span>与形状',
  sub: '把上一幕的 8 行压成三张牌子：shape → (288 分) → top-8 → 分组计算 → 无权相加。',
  caption: '验收点 1 就落在这张图上：能照着它把 4096 → 288 → 8 → 4096 讲一遍。',
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
  codeNote: '同一段代码，这一幕换成「节点图」视角重看一遍。',
  duration: 19800,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart', style: 'width:100%;padding-top:4px;justify-content:space-evenly' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: 'MoE 前向数据流（数字为默认配置实测值）' }));

    const fl = W.flow([
      { t: 'hidden_states', s: '(B, S, 4096)', cc: 0 },
      { t: 'gate', s: '线性层 → 288 分', cc: 1 },
      { t: 'top-8', s: 'indices + weights', cc: 3 },
      { t: 'experts', s: '8 × (2048→2048)', cc: 2 },
      { t: 'shared_experts', s: '2048→2048 恒开', cc: 4 },
      { t: 'out', s: '(B, S, 4096)', cc: 5 },
    ], { style: 'width:100%' });
    viz.appendChild(fl);

    viz.appendChild(U.el('div', { class: 'klabel', text: '每 token 的专家参数账（默认配置）', style: 'margin-top:2px' }));
    const rows = [
      ['可用的路由专家', M.E + ' 个', '288 × 25,165,824 = 7.248 B', '全部要驻留显存'],
      ['每 token 选中', M.topk + ' 个', '8 × 25,165,824 = 201.3 M', '路由分支的实际计算量'],
      ['共享专家', M.shared + ' 个', '25,165,824 = 25.2 M', '每 token 无条件执行'],
      ['每 token 合计', '9 / 289 个', '≈ 226.5 M', '<span class="hl">3.11%</span> 的专家参数被激活'],
    ];
    const tb = W.table(rows.map(r => [r[0], r[1], r[2], r[3]]),
      { head: ['项', '个数', '参数量', '说明'] });
    const tbc = W.card({ cc: 3, style: 'width:100%;padding:10px 12px', kids: [tb] });
    viz.appendChild(tbc);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    fl.focus(0);
    msg.innerHTML = '<span class="cm">// 从 4096 维的 hidden 开始</span>';
    tl.at(1500, () => { fl.focus(1); msg.innerHTML = 'gate 把 <em>4096</em> 投影成 <em>288</em> 个分数 —— 这一步是稠密的，每个 token 都算'; });
    tl.at(4300, () => { fl.focus(2); msg.innerHTML = '取 top-<em>8</em>：得到 8 个专家号（int64）与 8 个权重（每行和 = <em>2.5</em>）'; });
    tl.at(7100, () => { fl.focus(3); msg.innerHTML = 'experts 按专家号把 token <em>分组</em>，每组做一次 2048→2048 的 SwiGLU（grouped GEMM，见 L3-04）'; });
    tl.at(9900, () => { fl.focus(4); msg.innerHTML = '<em>共享专家</em>与路由无关：它拿的是 residuals，走的是一个 2048 宽的普通 MLP'; });
    tl.at(12700, () => { fl.focus(5); msg.innerHTML = '两项<em>逐元素相加</em>，回到 <em>(B, S, 4096)</em> —— 数据流闭合'; });
    tl.at(16200, () => {
      fl.focus(-1); tbc.classList.add('ac');
      msg.innerHTML = '记住这张账：<em>7.248 B</em> 参数驻留，每 token 只用 <em>226.5 M</em> —— MoE 的全部意义在这两个数的差里';
    });
    tl.at(18000, () => tbc.classList.remove('ac'));
  },
},

/* ------------------------------------------------- 5 ★ 共享专家 */
{
  kicker: '★ 洞察一 · 共享专家',
  title: '共享专家是<span class="hl-a">纯加法旁路</span>',
  sub: '「共享专家」很容易被理解成「一个权重较小的专家」。两个消融实验说明：它不吃路由结果，也没有系数。',
  caption: '消融做法：分别把「路由专家权重」和「共享专家权重」置零，看输出变成什么。',
  lang: 'python',
  codeStart: 99,
  code: `    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # Key difference using clamping
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)`,
  codeNote: '共享专家执行的**就是**这段代码 —— 没有 top-k、没有分组、没有修正偏置。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart', style: 'width:100%;padding-top:4px;justify-content:space-evenly' });
    wrap.appendChild(viz);

    const top = W.card({ cc: 4, tint: 4, style: 'width:100%', title: 'shared_experts 的定义（L197–L199）',
      body: U.el('div', { class: 'mono', style: 'font-size:12px;line-height:1.8;color:var(--ink-dim)' }, [
        U.el('div', { html: 'Glm5NextTextMLP(config=config, <span style="color:var(--c4)">intermediate_size=config.moe_intermediate_size * config.n_shared_experts</span>)' }),
        U.el('div', { class: 'cm', html: '// 默认 = 2048 × 1 = 2048，与一个路由专家等宽（实测比值 1.000）' }),
      ]) });
    viz.appendChild(top);

    const exps = [
      { n: 'A', cc: 2, title: '把 288 个路由专家的权重置零',
        body: 'moe(x) 与 shared_experts(x) <span class="hl4">逐元素相等</span>（allclose == True）',
        concl: '共享专家输出里不含任何路由信息' },
      { n: 'B', cc: 4, title: '把共享专家 down_proj 置零',
        body: '剩下 routed 部分，范数 <span class="mono">1148.19</span>；共享部分范数 <span class="mono">826.50</span>',
        concl: '两项量级相当，不是「小补丁」' },
      { n: 'C', cc: 3, title: '把路由门 weight 置零（≠ 关掉路由）',
        body: 'sigmoid(0)=0.5 让所有专家同分，top-k 仍选出 2 个并各拿 <span class="mono">1.25</span>',
        concl: '门置零 ≠ 路由消失；要关路由得置零专家权重' },
    ];
    const row = U.el('div', { class: 'row gap10', style: 'width:100%;align-items:stretch' });
    const els = exps.map(e => {
      const c = W.card({ cc: e.cc, num: e.n, style: 'flex:1 1 0', title: e.title,
        body: U.el('div', { class: 'col gap6' }, [
          U.el('div', { style: 'font-size:11.5px;color:var(--ink-dim);line-height:1.55', html: e.body }),
          U.el('div', { class: 'mono', style: 'font-size:10.5px;color:var(--c' + e.cc + ');line-height:1.5', html: '→ ' + e.concl }),
        ]) });
      row.appendChild(c); return c;
    });
    viz.appendChild(row);

    const why = W.card({ cc: 1, tint: 1, style: 'width:100%', title: '为什么共享专家能稳住训练',
      body: U.el('div', { class: 'col gap6' }, [
        U.el('div', { style: 'font-size:11.5px;color:var(--ink-dim);line-height:1.6',
          html: '路由是一个<em>离散选择</em>：没被选中的专家这一步收不到梯度。路由器一旦退化，路由分支会从「多专家混合」塌成「单专家」。' }),
        U.el('div', { style: 'font-size:11.5px;color:var(--ink-dim);line-height:1.6',
          html: '共享专家给每个 token 一份<em>不经过选择</em>的稠密变换 —— 无论路由怎么抖，这一层都还有一条参数固定、梯度连续的通路。' }),
        U.el('div', { class: 'row gap10 wrap', style: 'width:100%;margin-top:2px' }, [
          U.el('span', { class: 'chip c4', html: '<i class="sw sw-c4"></i>9 个专家里的 1 个 = 11.1% 计算量' }),
          U.el('span', { class: 'chip c2', html: '<i class="sw sw-c2"></i>25.2 M 参数 vs 路由分支 7.248 B' }),
          U.el('span', { class: 'chip c1', html: '<i class="sw sw-c1"></i>0.35% 的参数买一条兜底通路' }),
        ]),
      ]) });
    viz.appendChild(why);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 共享专家执行的代码：一段普通的 SwiGLU MLP</span>';
    tl.at(1600, () => { els[0].classList.add('ac'); msg.innerHTML = '实验 A：路由专家全置零 → 输出<em>恰好等于</em>共享专家的输出'; });
    tl.at(4600, () => {
      els[0].classList.remove('ac'); els[1].classList.add('ac');
      msg.innerHTML = '实验 B：共享专家置零 → 剩下 routing 部分，范数 <em>1148</em> vs <em>827</em>，两者对等';
    });
    tl.at(7600, () => {
      els[1].classList.remove('ac'); els[2].classList.add('ac');
      msg.innerHTML = '实验 C：门置零<span class="hlbad">不等于</span>路由关闭 —— 同分时 top-k 照样选出专家、各拿 1.25';
    });
    tl.at(10800, () => {
      els[2].classList.remove('ac'); why.classList.add('ac');
      msg.innerHTML = '所以「稳住训练」的机制不是加权，而是<em>无条件执行</em>：总有一路不依赖选择的梯度';
    });
    tl.at(14000, () => {
      why.classList.remove('ac');
      msg.innerHTML = '一句话：<em>routed + shared</em>，加法，<span class="hlbad">没有系数</span>';
    });
  },
},

/* ------------------------------------------------- 6 路由器三元组 */
{
  kicker: '★ 洞察二 · 被丢掉的返回值',
  title: '路由器的三个返回值，<span class="hl-a">第一个没人要</span>',
  sub: 'self.gate(...) 返回 (router_logits, topk_weights, topk_indices)，调用方用 _ 接住第一个。这不是省略，是前向真的不需要它。',
  caption: '它需要它的地方在别处：_can_record_outputs 这张表告诉框架「第 0 个返回值才是 logits」。',
  lang: 'python',
  codeStart: 183,
  code: `        topk_weights = topk_weights * self.routed_scaling_factor
        return router_logits, topk_weights, topk_indices`,
  codeNote: '返回顺序就是契约：index=0 必须是 router_logits，否则辅助损失会静默地算错。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart', style: 'width:100%;padding-top:4px;justify-content:space-evenly' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '三个返回值：形状、去向、要不要' }));

    const trips = [
      { i: 0, name: 'router_logits', shp: '(tokens, 288) float32', use: '辅助损失 load balancing',
        alive: false, cc: 3, note: 'MoE.forward 用 _ 接住 —— 前向不需要它' },
      { i: 1, name: 'topk_weights', shp: '(tokens, 8)', use: '加权求和，交 experts',
        alive: true, cc: 2, note: '每行和 = routed_scaling_factor = 2.5' },
      { i: 2, name: 'topk_indices', shp: '(tokens, 8) int64', use: '按专家号分组，交 experts',
        alive: true, cc: 1, note: 'grouped GEMM 的排序依据（L3-04）' },
    ];
    const row = U.el('div', { class: 'row gap10', style: 'width:100%;align-items:stretch' });
    const els = trips.map(t => {
      const c = W.card({ cc: t.cc, num: t.i, style: 'flex:1 1 0;display:flex;flex-direction:column', title: t.name,
        body: U.el('div', { class: 'col gap6' }, [
          U.el('div', { class: 'mono', style: 'font-size:11px;color:var(--ink)', text: t.shp }),
          U.el('div', { style: 'font-size:11px;color:var(--ink-dim);line-height:1.5', html: '去向：' + t.use }),
          U.el('div', { class: 'mono', style: 'font-size:10.5px;line-height:1.5;color:var(--c' + t.cc + ')', text: '→ ' + t.note }),
        ]) });
      row.appendChild(c); return c;
    });
    viz.appendChild(row);

    const und = U.el('div', { class: 'card', cc: 3, tint: 3, style: 'width:100%' });
    und.appendChild(U.el('div', { class: 'ct', html: 'MoE.forward 里那一行' }));
    und.appendChild(U.el('div', { class: 'mono', style: 'font-size:12.5px;color:var(--ink);line-height:1.7' ,
      html: '_<span class="hlbad">,</span> topk_weights, topk_indices = self.gate(hidden_states)' }));
    und.appendChild(U.el('div', { class: 'cm mono', style: 'font-size:10.5px;margin-top:5px',
      html: '// hook 实测：无论 output_router_logits 怎么设，TopkRouter 每次前向都被调用且只调用 1 次' }));
    viz.appendChild(und);

    const rec = U.el('div', { class: 'card', cc: 1, style: 'width:100%' });
    rec.appendChild(U.el('div', { class: 'ct', html: '_can_record_outputs（L1374–L1378）' }));
    rec.appendChild(U.el('div', { class: 'mono', style: 'font-size:11.5px;line-height:1.75;color:var(--ink-dim)' , html:
      'OutputRecorder(Glm5NextTextTopkRouter, <span class="hl2">index=0</span>)' }));
    rec.appendChild(U.el('div', { style: 'font-size:11px;color:var(--ink-dim);line-height:1.55;margin-top:5px',
      html: '前向丢弃的 logits 在这里被<em>登记</em>：跑完前向由 capture_outputs 收集起来，交给辅助损失。' }));
    viz.appendChild(rec);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 三个返回值，三个完全不同的去向</span>';
    tl.at(1600, () => { els[1].classList.add('ac'); msg.innerHTML = '<em>topk_weights</em>：归一化后乘 2.5，每行和恒为 <em>2.5</em>，交给 experts 做加权'; });
    tl.at(4600, () => { els[1].classList.remove('ac'); els[2].classList.add('ac'); msg.innerHTML = '<em>topk_indices</em>：int64 的专家号，决定 token 被分到哪一组 —— 它才是 grouped GEMM 的输入'; });
    tl.at(7600, () => {
      els[2].classList.remove('ac'); els[0].classList.add('ac');
      msg.innerHTML = '<em>router_logits</em>：前向<em>一点用都没有</em>，被 <span class="hlbad">_</span> 接住扔掉';
    });
    tl.at(10800, () => {
      und.classList.add('ac');
      msg.innerHTML = '但它<em>确实被算出来了</em> —— 这不是「省略一个不存在的返回值」';
    });
    tl.at(13800, () => {
      und.classList.remove('ac'); rec.classList.add('ac');
      msg.innerHTML = '它在别处被登记：<em>index=0</em> 是位置约定 —— 调换 return 的顺序不会报错，只会让辅助损失变成没意义的数';
    });
    tl.at(16000, () => { rec.classList.remove('ac'); els.forEach(e => e.classList.remove('ac')); });
  },
},

/* ------------------------------------------------- 7 output_router_logits */
{
  kicker: '★ 洞察三 · 开关',
  title: '<span class="hl-b">output_router_logits</span> 关掉了什么',
  sub: '默认 False。关掉它不改变任何一个数值 —— 它只决定「收不收集 logits、算不算辅助损失」。',
  caption: '实测：同一份输入跑两次，last_hidden_state 的 allclose(atol=0) 为 True，逐元素完全相同。',
  lang: 'python',
  codeStart: 2219,
  code: `        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.text_config.output_router_logits
        )`,
  codeNote: '运行期参数优先于 config；两者都不给时用 config 的默认值 False。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart', style: 'width:100%;padding-top:4px;justify-content:space-evenly' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '同一份输入的两次前向（probe_moe2.py 实测）' }));
    const tb = W.table([
      ['<code class="inl">False</code>（默认）', '1 次 / 层', '<span class="hlbad">没有</span> router_logits', '<span class="hl4">与 True 逐元素相同</span>'],
      ['<code class="inl">True</code>', '1 次 / 层', '每层一份 (tokens, 288)', '<span class="hl4">与 False 逐元素相同</span>'],
    ], { head: ['output_router_logits', 'TopkRouter 调用次数', 'router_logits 随输出返回', 'last_hidden_state'] });
    const tbc = W.card({ cc: 1, style: 'width:100%;padding:10px 12px', kids: [tb] });
    viz.appendChild(tbc);

    const row = U.el('div', { class: 'row gap10', style: 'width:100%;align-items:stretch' });
    const c1 = W.card({ cc: 0, tint: 0, style: 'flex:1 1 0', title: '关闭时（推理默认）',
      body: U.el('div', { class: 'col gap5' }, [
        U.el('div', { class: 'mono', style: 'font-size:11.5px;color:var(--ink-dim);line-height:1.7', html: 'aux_loss = <span class="hlbad">None</span>' }),
        U.el('div', { style: 'font-size:11.5px;color:var(--ink-dim);line-height:1.6', html: '不算负载均衡损失，不动 loss；路由照常工作。' }),
      ]) });
    const c2 = W.card({ cc: 1, tint: 1, style: 'flex:1 1 0', title: '打开时（训练要观测)',
      body: U.el('div', { class: 'col gap5' }, [
        U.el('div', { class: 'mono', style: 'font-size:11.5px;color:var(--ink-dim);line-height:1.7', html: 'aux_loss = load_balancing_loss_func(...)' }),
        U.el('div', { style: 'font-size:11.5px;color:var(--ink-dim);line-height:1.6', html: 'labels 不为 None 时：<code class="inl">loss += 0.001 * aux_loss</code>' }),
      ]) });
    row.appendChild(c1); row.appendChild(c2);
    viz.appendChild(row);

    viz.appendChild(U.el('div', { class: 'klabel', text: '打开它的代价：router_logits 本身要留在显存里（float32, 288 个专家）' }));
    const bars = W.bars([
      { label: '1 层 × 4k token', value: 4.5, valueText: '4.5 MiB', cc: 4 },
      { label: '1 层 × 64k token', value: 72, valueText: '72 MiB', cc: 2 },
      { label: '45 层 × 64k token', value: 3240, valueText: '3.16 GiB', cc: 1 },
      { label: '45 层 × 1M token', value: 49400, valueText: '48.3 GiB', cc: 3 },
    ], { max: 49400, cc: 3 });
    viz.appendChild(bars);
    bars.setAll(0);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 默认 config.output_router_logits = False</span>';
    tl.at(1600, () => {
      c1.classList.add('ac');
      msg.innerHTML = '关闭：<em>generate()</em> 这条路永远不会进那个 if —— 不算 aux_loss、不改 loss';
    });
    tl.at(4600, () => {
      c1.classList.remove('ac'); c2.classList.add('ac'); tbc.classList.add('ac');
      msg.innerHTML = '★ 但它<em>不改变路由</em>：两次前向的 last_hidden_state <span class="hl4">allclose(atol=0) == True</span>';
    });
    tl.at(7600, () => {
      c2.classList.remove('ac');
      msg.innerHTML = '原因在上一幕：TopkRouter 的<em>调用次数与开关无关</em>（都是 1 次/层），开关只管收不收集';
    });
    tl.at(10600, () => {
      tbc.classList.remove('ac');
      bars.setTo(0, 1); bars.setTo(1, 1);
      msg.innerHTML = '代价一：每层 64k token 要留 <em>72 MiB</em> 的 logits 张量';
    });
    tl.at(13400, () => {
      bars.setTo(2, 1); bars.setTo(3, 1);
      msg.innerHTML = '代价二：<em>45 层 × 1M token ≈ 48.3 GiB</em> —— 与 L5 的显存账本直接冲突';
    });
    tl.at(16200, () => {
      msg.innerHTML = '所以它是一个<em>训练期观测开关</em>；推理路径上唯一的后果是变慢、变占内存';
    });
  },
},

/* ------------------------------------------------- 8 辅助损失 */
{
  kicker: '★ 洞察三之补 · 辅助损失',
  title: '负载均衡损失：<span class="hl-a">均匀时等于 top_k</span>',
  sub: 'load_balancing_loss_func 的实测三个极端值 —— 最小值恰好是 top_k，最大值恰好是 num_experts。',
  caption: 'E × ⟨专家使用率, 专家概率质量⟩：两个量都归一化过，所以与 token 数无关。',
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
  codeNote: 'router_aux_loss_coef 默认 0.001 —— 惩罚项量级在 0.002 ~ 0.008，是温和的矫正力。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap10 vizgrow hstart', style: 'width:100%;padding-top:4px;justify-content:space-evenly' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '实测：E = 8, top_k = 2, 2000 个 token，直接构造 logits' }));

    const bars = W.bars([
      { label: '均匀 (logits 全 0)', value: 2.0, valueText: '2.0  = top_k', cc: 4 },
      { label: '全挤在 1 个专家', value: 8.0, valueText: '8.0  = num_experts', cc: 3 },
      { label: '× coef 后（均匀）', value: 0.002, valueText: '0.002', cc: 0 },
      { label: '× coef 后（集中）', value: 0.008, valueText: '0.008', cc: 2 },
    ], { max: 8, cc: 1 });
    viz.appendChild(bars);
    bars.setAll(0);

    const row = U.el('div', { class: 'row gap10', style: 'width:100%;align-items:stretch' });
    const cA = W.card({ cc: 4, tint: 4, style: 'flex:1 1 0', title: '均匀时 = top_k（可手推）',
      body: U.el('div', { class: 'col gap4' }, [
        U.el('div', { class: 'mono', style: 'font-size:11px;color:var(--ink-dim);line-height:1.7',
          html: '使用率 = top_k/E，概率质量 = 1/E' }),
        U.el('div', { class: 'mono', style: 'font-size:11px;color:var(--ink);line-height:1.7',
          html: 'E × Σ (top_k/E)(1/E) = <span class="hl4">top_k</span>' }),
      ]) });
    const cB = W.card({ cc: 3, tint: 3, style: 'flex:1 1 0', title: '全挤一个 = E（可手推）',
      body: U.el('div', { class: 'col gap4' }, [
        U.el('div', { class: 'mono', style: 'font-size:11px;color:var(--ink-dim);line-height:1.7',
          html: '使用率与概率质量同时集中在同一个专家' }),
        U.el('div', { class: 'mono', style: 'font-size:11px;color:var(--ink);line-height:1.7',
          html: 'E × (1 × 1) = <span class="hlbad">E</span> &nbsp;<span class="cm">// 与 token 数无关</span>' }),
      ]) });
    row.appendChild(cA); row.appendChild(cB);
    viz.appendChild(row);

    const note = W.card({ cc: 1, style: 'width:100%', title: '与 L3-06 的 Sinkhorn 是两种思路',
      body: U.el('div', { style: 'font-size:11.5px;color:var(--ink-dim);line-height:1.6',
        html: '辅助损失是<em>软惩罚</em>：引导路由器自己走向均衡；mHC 的 20 次 Sinkhorn 迭代是<em>硬投影</em>：每步都把矩阵压回双随机。一个改梯度，一个改数值。' }) });
    viz.appendChild(note);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 构造两个极端，直接调 load_balancing_loss_func</span>';
    tl.at(1600, () => { bars.setTo(0, 1); msg.innerHTML = '所有 logits = 0（完全均匀）→ aux_loss = <em>2.0</em>，恰好等于 <em>top_k = 2</em>'; });
    tl.at(4400, () => { bars.setTo(1, 1); msg.innerHTML = '全部 token 挤在 1 个专家 → aux_loss = <em>8.0</em>，恰好等于 <em>num_experts = 8</em>'; });
    tl.at(7200, () => {
      cA.classList.add('ac'); cB.classList.add('ac');
      msg.innerHTML = '两端都能手推出来 —— 说明这个损失是<em>归一化过的比值</em>，不是计数';
    });
    tl.at(10200, () => {
      cA.classList.remove('ac'); cB.classList.remove('ac');
      bars.setTo(2, 1); bars.setTo(3, 1);
      msg.innerHTML = '乘 <em>router_aux_loss_coef = 0.001</em> 之后：惩罚项只有 <em>0.002 ~ 0.008</em> 量级，是温和的矫正力';
    });
    tl.at(13200, () => {
      note.classList.add('ac');
      msg.innerHTML = '对照 L3-06：辅助损失改<em>梯度</em>，Sinkhorn 20 次迭代改<em>数值</em> —— 两种不同的均衡手段';
    });
    tl.at(15400, () => {
      note.classList.remove('ac');
      msg.innerHTML = '另：文件 L2084–L2085 的注释说明它逐层累加、最后归一化，<em>峰值显存与层数无关</em>';
    });
  },
},

/* ------------------------------------------------- 9 收束 */
{
  kicker: '收束 · 一句话记住',
  title: '一页总表：<span class="hl-a">两条专家路径</span>',
  sub: 'MoE 前向的数据流、共享专家的加法性质、router_logits 的去向 —— 压成一张表。',
  caption: '下一课 L3-06 讲 mHC：这两条路径的输出怎么写回 4 条残差流。',
  lang: 'python',
  codeStart: 206,
  code: `        hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states`,
  codeNote: '整课的落点就是这两行：一行路由、一行无权相加。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart', style: 'width:100%;padding-top:4px;justify-content:space-evenly' });
    wrap.appendChild(viz);

    const tb = W.table([
      ['L202', '<code class="inl">residuals = hidden_states</code>', '原样留一份', '给共享专家用的输入'],
      ['L204', '<code class="inl">_ , topk_weights, topk_indices</code>', '(tokens,8) × 2', '<span class="hlbad">router_logits 被丢掉</span>'],
      ['L205', '<code class="inl">view(-1, shape[-1])</code>', '(B,S,4096) → (B·S,4096)', '专家计算要 2D'],
      ['L206', '<code class="inl">experts(...).view(*orig_shape)</code>', '8 个专家加权和', 'topk_weights 每行和 = 2.5'],
      ['L207', '<code class="inl">+ self.shared_experts(residuals)</code>', '<span class="hl4">无权相加</span>', '★ 合流点'],
      ['L2249', '<code class="inl">if output_router_logits:</code>', '默认 False', '不改数值，只加显存'],
    ], { head: ['行号', '代码', '张量 / 条件', '要点'] });
    const tbc = W.card({ cc: 0, style: 'width:100%;padding:10px 12px', kids: [tb] });
    viz.appendChild(tbc);

    const ans = U.el('div', { class: 'row gap10', style: 'width:100%;align-items:stretch' });
    const keep = W.card({ cc: 4, tint: 4, style: 'flex:1 1 0', title: '三句话',
      body: U.el('div', { class: 'col gap5' }, [
        U.el('div', { style: 'font-size:11.5px;color:var(--ink-dim);line-height:1.6', html: '① 稀疏层 = 路由专家（8/288）+ 共享专家（1/1）。' }),
        U.el('div', { style: 'font-size:11.5px;color:var(--ink-dim);line-height:1.6', html: '② 共享专家是<em>无权加法旁路</em>，不吃 topk_weights。' }),
        U.el('div', { style: 'font-size:11.5px;color:var(--ink-dim);line-height:1.6', html: '③ <code class="inl">output_router_logits</code> 只影响收集，不影响路由。' }),
      ]) });
    const ex = W.exercise(
      '若把 <code class="inl">n_shared_experts</code> 从 1 改成 2，共享专家会发生什么变化？路由分支会变吗？',
      '共享专家的中间维变成 <code class="inl">2048 × 2 = 4096</code>（第一节那行显式传参），参数量变成 <code class="inl">3 × 4096 × 4096 = 50,331,648</code>，'
      + '是原来的 2 倍；每个 token 的专家计算量从 9 份变成 10 份。<br>'
      + '<b>路由分支完全不变</b>：<code class="inl">n_shared_experts</code> 只出现在 <code class="inl">shared_experts</code> 的构造里，'
      + '既不进 <code class="inl">Glm5NextTextTopkRouter</code>，也不进 <code class="inl">Glm5NextTextExperts</code> —— '
      + '这正是「两条路径互相独立」的最直接证据。'
    );
    ans.appendChild(keep); ans.appendChild(ex);
    viz.appendChild(ans);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 全课落点：L206 与 L207</span>';
    tl.at(1500, () => { msg.innerHTML = '<em>L206</em>：路由专家 —— 按 <span class="fn">topk_indices</span> 分组，用 <span class="fn">topk_weights</span> 加权'; });
    tl.at(4200, () => { msg.innerHTML = '<em>L207</em>：<span class="hl4">+ self.shared_experts(residuals)</span> —— 加权项与无权项的一次加法'; });
    tl.at(7000, () => { msg.innerHTML = '一个 token 的专家计算量 = <em>8 + 1 = 9</em> 份；参数驻留 = <em>7.248 B</em> + <em>25.2 M</em>'; });
    tl.at(9800, () => { msg.innerHTML = '验收点 1：照着数据流图说一遍 <em>4096 → 288 → 8 → 4096</em>；验收点 2：说出共享专家为什么能稳住训练'; });
    tl.at(12600, () => { msg.innerHTML = '<span class="cm">// 下一课 L3-06：mHC —— post / comb 怎么把结果写回 4 条残差流</span>'; });
    tl.at(15200, () => { msg.innerHTML = '回顾 L0-01：稀疏层的输出最终要经过 <span class="fn">ffn_hc</span> 的写回，那一步在下一课'; });
  },
},

];
