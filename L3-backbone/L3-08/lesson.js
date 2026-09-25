/* ==========================================================================
   L3-08 · 解码层装配：三个子块的顺序
   --------------------------------------------------------------------------
   覆盖：models/glm5_next/modeling_glm5_next.py, modeling_layers.py,
         modeling_outputs.py（3 个文件 / 4788 行）
   目标：看完能默写一层的执行顺序，并说出 mHC 下的 residual 与标准 Transformer 差在哪。
   排版基准：#visual 可视区 = 994 x 662.7 舞台像素（无头浏览器实测）。
   ★ 实测订正：作业书写 2445 / 683 / 1663 行，实测 wc -l 是 2444 / 682 / 1662。
   ========================================================================== */
'use strict';

/* 本课用到的实测常量（_data/recon/probe_l308.py 与 probe_l308b.py 的输出；脚本不计入覆盖率） */
const M = {
  hc: 4,               /* hc_mult：残差流条数 */
  iters: 20,           /* hc_sinkhorn_iters */
  eps: 1e-6,           /* hc_eps */
  swiglu: 10.0,        /* swiglu_limit */
  hcParams: 393243,    /* 单个 mHC 参数量 = fn(24x16384) + base(24) + scale(3) */
  preStd: 2.49,        /* 投影输出的 std（input_norm 之后流的 std = 1.0000） */
  postMean: 1.004, postMin: 0.005, postMax: 1.968,
  rowDev: 1.77e-2, colDev: 1.07e-6,
  layers: 45,
};

const SCENES = [

/* ============================================================ 1 全景 */
{
  kicker: 'L3 · 文本主干 · 装配',
  title: '一层 = 三个子块，<span class="hl-a">六次调用</span>',
  sub: 'mHC 不是残差的名字，是一个真的子模块；它在一层里出现两次，把注意力与 MoE 各缝一次。',
  caption: '回顾 L0-01：45 层循环里每层返回 (hidden_states, topk_indices)。本课把"一层之内"彻底拆开。',
  lang: 'python',
  codeStart: 1280,
  code: `class Glm5NextTextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Glm5NextTextConfig, layer_idx: int):
        super().__init__()
        self.block_type = config.layer_types[layer_idx]
        self.hidden_size = config.hidden_size
        self.self_attn = (
            Glm5NextTextLinearAttention(config, layer_idx)
            if self.block_type == "linear_attention"
            else Glm5NextTextAttention(config, layer_idx)
        )

        self.mlp = (
            Glm5NextTextMoE(config) if config.mlp_layer_types[layer_idx] == "sparse" else Glm5NextTextMLP(config)
        )

        self.input_layernorm = Glm5NextTextRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Glm5NextTextRMSNorm(config.hidden_size, config.rms_norm_eps)

        self.attn_hc = Glm5NextTextHyperConnection(config)
        self.ffn_hc = Glm5NextTextHyperConnection(config)`,
  codeNote: 'Glm5NextTextDecoderLayer —— 三个子块的清单：self_attn / mlp / attn_hc / ffn_hc，加两个 norm。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const fl = W.flow([
      { t: 'mHC', s: 'attn_hc', cc: 1 },
      { t: '注意力', s: 'self_attn', cc: 4 },
      { t: 'mHC', s: 'ffn_hc', cc: 1 },
      { t: 'MoE / MLP', s: 'mlp', cc: 2 },
    ], { style: 'width:100%' });
    wrap.appendChild(U.el('div', { class: 'col gap8', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: '四个阶段：mHC 出现两次，两次之间夹一个子层' }), fl,
    ]));

    const viz = U.el('div', { class: 'col gap10 vizgrow', style: 'width:100%' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel',
      text: '实测：子模块调用次序（钩子记录，一个都不多）' }));
    const seq = ['attn_hc', 'input_layernorm', 'self_attn', 'ffn_hc', 'post_attention_layernorm', 'mlp'];
    const seqRow = U.el('div', { class: 'row gap6', style: 'width:100%;align-items:stretch' });
    const seqEls = seq.map((n, i) => {
      const e = U.el('div', { class: 'card cc' + (i === 2 || i === 5 ? 4 : 1),
        style: 'flex:1 1 0;padding:7px 5px;text-align:center;opacity:.22;'
             + 'transition:opacity .45s var(--ease-out),transform .45s var(--ease-out),box-shadow .4s' }, [
        U.el('div', { class: 'n sm', style: 'color:var(--ink-faint)', text: String(i + 1) }),
        U.el('div', { class: 'mono', style: 'font-size:9.5px;line-height:1.25;color:#fff;'
          + 'overflow-wrap:anywhere;word-break:break-all;margin-top:3px', text: n }),
      ]);
      seqRow.appendChild(e); return e;
    });
    viz.appendChild(seqRow);

    /* 写回标记：模块列表里没有它们 —— 它们是 forward 里的两行内联运算 */
    const wbRow = U.el('div', { class: 'row gap6', style: 'width:100%' });
    const wbEls = [];
    for (let i = 0; i < 6; i++) {
      const marked = (i === 2 || i === 5);
      const e = U.el('div', { style: 'flex:1 1 0;text-align:center;opacity:.18;'
        + 'transition:opacity .45s var(--ease-out)' }, [
        U.el('div', { class: 'chip c' + (marked ? 2 : 0),
          style: 'display:' + (marked ? 'inline-flex' : 'none'),
          text: marked ? '写回 ' + (i === 2 ? '①' : '②') : '' }),
      ]);
      wbRow.appendChild(e); wbEls.push(e);
    }
    viz.appendChild(wbRow);

    const cards = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const c1 = W.card({ cc: 1, tint: 1, style: 'flex:1', title: '① mHC &times;2',
      sub: '<code class="inl">attn_hc</code> 与 <code class="inl">ffn_hc</code> 是两个独立对象'
        + '（实测 <code class="inl">attn_hc is ffn_hc</code> 为 <b>False</b>），各持一份 fn/base/scale。' });
    const c2 = W.card({ cc: 4, tint: 4, style: 'flex:1', title: '② 注意力 &times;1',
      sub: '按 <code class="inl">config.layer_types[i]</code> 二选一：34 层 KDA、11 层 MLA+DSA。' });
    const c3 = W.card({ cc: 2, tint: 2, style: 'flex:1', title: '③ MoE / 稠密 MLP &times;1',
      sub: '按 <code class="inl">config.mlp_layer_types[i]</code> 二选一：3 层稠密、42 层 MoE（内部还分稀疏 + 共享）。' });
    [c1, c2, c3].forEach(c => cards.appendChild(c));
    viz.appendChild(cards);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    const mark = k => {
      fl.focus(k);
      seqEls.forEach((e, i) => { e.style.opacity = (Math.floor(i / 2) === k || (k === 3 && i === 5)) ? '1' : '.22'; });
    };
    msg.innerHTML = '<span class="cm">// 三个子块：mHC、注意力、MoE/MLP；装配模板只有一套</span>';
    tl.at(1500, () => {
      mark(0); wbEls[0].style.opacity = '.6';
      msg.innerHTML = '第 <em>1</em> 拍：<em>attn_hc</em> 先算权重、把 4 条流收拢成 1 条';
    });
    tl.at(4200, () => {
      seqEls.forEach((e, i) => { e.style.opacity = i <= 2 ? '1' : '.22'; });
      msg.innerHTML = '第 <em>2</em>、<em>3</em> 拍：<em>input_layernorm</em> 只归一化收拢后的那 1 条，再进 <em>self_attn</em>';
    });
    tl.at(7200, () => {
      wbEls[2].style.opacity = '1'; seqEls[2].classList.add('ac');
      msg.innerHTML = '注意力之后是<em>写回 ①</em> —— 它<em>不在模块列表里</em>，是 forward 里的两行张量运算';
    });
    tl.at(10200, () => {
      seqEls[2].classList.remove('ac'); mark(2); wbEls[4].style.opacity = '.6';
      msg.innerHTML = '第 <em>4</em> 拍：换成 <em>ffn_hc</em>，同样的三件事再做一遍';
    });
    tl.at(13200, () => {
      seqEls.forEach(e => { e.style.opacity = '1'; });
      wbEls[5].style.opacity = '1'; seqEls[5].classList.add('ac'); c3.classList.add('ac');
      msg.innerHTML = '<em>post_attention_layernorm</em> → <em>mlp</em> → <em>写回 ②</em>：一层结束';
    });
    tl.at(15600, () => {
      seqEls[5].classList.remove('ac');
      [c1, c2, c3].forEach(c => c.classList.add('ac'));
      msg.innerHTML = '记住这条链：<em>mHC → norm → 子层 → 写回</em>，跑两遍';
    });
  },
},

/* ============================================================ 2 顺序① */
{
  kicker: 'L3-08 · 第 2 幕 · 顺序 ①',
  title: 'mHC 在 norm <span class="hl-a">之前</span>',
  sub: 'post / comb / collapsed 一起算出来：权重从"未归一化的 4 条流"上估计，只有收拢后的那 1 条进 RMSNorm。',
  caption: '回顾 L3-06：mHC 内部自带一个 0 参数的 unweighted RMSNorm，所以"权重不归一化"不是疏忽。',
  lang: 'python',
  codeStart: 1312,
  code: `        dtype = hidden_states.dtype

        residual = hidden_states
        post, comb, hidden_states = self.attn_hc(hidden_states)
        # Self attn
        hidden_states = self.input_layernorm(hidden_states)
        topk_indices = None`,
  codeNote: 'Glm5NextTextDecoderLayer.forward 的前 6 行 —— 顺序与形状都在这里。',
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col vizgrow', style: 'width:100%' });
    wrap.appendChild(viz);
    /* 注意：.vizgrow 自带 flex-direction:column，所以"行"要作为它的子元素，
       不能把 row 与 vizgrow 写在同一个 div 上（后者会覆盖 flex-direction）。 */
    const lane = U.el('div', { class: 'row gap14', style: 'width:100%;align-items:center' });
    viz.appendChild(lane);

    /* 左：4 条流 */
    const left = U.el('div', { class: 'col gap5', style: 'flex:none;width:250px' });
    left.appendChild(U.el('div', { class: 'klabel', style: 'text-transform:none;letter-spacing:.02em',
      text: 'hidden_states (B, S, 4, D)' }));
    const bars = [];
    for (let s = 0; s < M.hc; s++) {
      const b = U.el('div', { class: 'n sm', style: 'height:30px;border-radius:8px;display:flex;'
        + 'align-items:center;justify-content:center;border:1px solid var(--c1);'
        + 'background:rgba(167,139,250,.16);color:#ddd6fe;opacity:.25;'
        + 'transition:opacity .5s var(--ease-out)', text: '流 ' + s });
      left.appendChild(b); bars.push(b);
    }
    left.appendChild(U.el('div', { class: 'mono dim', style: 'font-size:10.5px', text: 'residual 留的是这一份（未归一化）' }));
    lane.appendChild(left);

    /* 中：mHC */
    const mid = U.el('div', { class: 'col gap6 center', style: 'flex:none;width:230px' });
    mid.innerHTML = '<div class="card cc1" style="width:100%;text-align:center;padding:9px 11px">'
      + '<div class="mono" style="font-size:12px;color:#fff">self.attn_hc(...)</div>'
      + '<div class="mono" style="font-size:10px;color:var(--ink-faint);margin-top:3px">算权重 + 收拢</div></div>'
      + '<div style="font-size:16px;color:var(--accent)">&#10233;</div>';
    const outs = U.el('div', { class: 'col gap4', style: 'width:100%' });
    const o1 = U.el('div', { class: 'chip c0', style: 'opacity:.2;transition:opacity .45s', text: 'post  (B, S, 4)' });
    const o2 = U.el('div', { class: 'chip c2', style: 'opacity:.2;transition:opacity .45s', text: 'comb  (B, S, 4, 4)' });
    const o3 = U.el('div', { class: 'chip c4', style: 'opacity:.2;transition:opacity .45s', text: 'collapsed  (B, S, D)' });
    [o1, o2, o3].forEach(o => outs.appendChild(o));
    mid.appendChild(outs);
    lane.appendChild(mid);

    /* 右：collapsed 才进 norm */
    const right = U.el('div', { class: 'col gap6', style: 'flex:1;min-width:0' });
    right.appendChild(U.el('div', { class: 'klabel', text: '只有这一条往下走' }));
    const col4 = U.el('div', { class: 'n sm', style: 'height:30px;border-radius:8px;display:flex;'
      + 'align-items:center;justify-content:center;border:1px solid var(--c4);'
      + 'background:rgba(52,211,153,.18);color:#a7f3d0;opacity:.2;transition:opacity .5s',
      text: 'collapsed (B, S, D)' });
    right.appendChild(col4);
    right.appendChild(U.el('div', { class: 'mono dim', style: 'font-size:10.5px;text-align:center', text: '↓ input_layernorm' }));
    const attn = U.el('div', { class: 'card cc4', style: 'text-align:center;padding:8px 10px;opacity:.2;'
      + 'transition:opacity .5s' }, [
      U.el('div', { class: 'mono', style: 'font-size:11.5px;color:#fff', text: 'self_attn(norm(collapsed))' }),
    ]);
    right.appendChild(attn);
    lane.appendChild(right);

    const cards = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const k1 = W.card({ cc: 1, tint: 1, style: 'flex:1', title: '为什么 mHC 在前',
      sub: '归一化只服务"子层输入"。权重需要的是 4 条流之间的<em>相对</em>关系，先归一化会把流压成一条，混合就无从谈起。' });
    const k2 = W.card({ cc: 4, tint: 4, style: 'flex:1', title: '两条归一化各有其主',
      sub: 'mHC 内部的 <code class="inl">input_norm</code>（0 参数）管权重估计；主干这个 RMSNorm 管子层输入 —— 互不替代。' });
    const k3 = W.card({ cc: 0, tint: 0, style: 'flex:1', title: '实测形状',
      sub: 'post (1,5,4) · comb (1,5,4,4) · collapsed (1,5,64)（小配置 B=1,S=5,hidden=64）。' });
    [k1, k2, k3].forEach(c => cards.appendChild(c));
    wrap.appendChild(cards);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 第一步不是归一化，是 mHC</span>';
    tl.at(1800, () => {
      bars.forEach(b => { b.style.opacity = '1'; });
      msg.innerHTML = '进来的是 <em>4 条流</em>：<em>(B, S, 4, D)</em>，不是教科书里的 (B, S, D)';
    });
    tl.at(4800, () => {
      o1.style.opacity = '1'; o2.style.opacity = '1';
      msg.innerHTML = '<em>attn_hc</em> 一次给出两个权重：<em>post</em> 管写多少、<em>comb</em> 管 4 条旧流怎么混合';
    });
    tl.at(7800, () => {
      o3.style.opacity = '1'; col4.style.opacity = '1';
      msg.innerHTML = '同时把 4 条流按 <em>pre</em> 收拢成 1 条 <em>collapsed (B, S, D)</em> —— 它才是子层的输入';
    });
    tl.at(10800, () => {
      attn.style.opacity = '1'; k1.classList.add('ac'); k2.classList.add('ac');
      msg.innerHTML = '所以顺序是 <em>mHC → norm → 子层</em>：归一化作用在收拢后的 1 条上';
    });
    tl.at(13400, () => {
      k1.classList.remove('ac'); k2.classList.remove('ac'); k3.classList.add('ac');
      msg.innerHTML = '<span class="cm">// 权重是在"原始 4 条流"上估计的 —— mHC 内部有它自己的归一化</span>';
    });
  },
},

/* ============================================================ 3 ★ 写回 */
{
  kicker: 'L3-08 · 第 3 幕 · 顺序 ② ★',
  title: '★ 写回：<span class="hl-a">post &odot; 子层输出 + combᵀ &middot; 旧流</span>',
  sub: '标准 Transformer 是 x + f(norm(x))；这里是"门控 &times; 子层输出"再加"4 条旧流的混合"。',
  caption: '验收点 2 的答案在本幕：residual 在 mHC 下是门控 + 流间混合，不是一条加法。',
  lang: 'python',
  codeStart: 1337,
  code: `        hidden_states = post.to(dtype).unsqueeze(-1) * hidden_states.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual
        )`,
  codeNote: '两行张量运算，就是全部残差逻辑 —— 它不在任何子模块里。',
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const f = U.el('div', { class: 'formula', style: 'width:100%;font-size:15px;text-align:center' });
    wrap.appendChild(f);

    const viz = U.el('div', { class: 'col gap10 vizgrow hstart', style: 'width:100%' });
    wrap.appendChild(viz);

    /* 三个输入 -> 一个输出 */
    const io = U.el('div', { class: 'row gap10 center', style: 'width:100%' });
    const chip = (t, cc) => U.el('div', { class: 'card cc' + cc, style: 'padding:7px 10px;text-align:center;flex:none' },
      [U.el('div', { class: 'mono', style: 'font-size:11px;color:#fff', text: t })]);
    const cp = chip('post (B, S, 4)', 0);
    const ca = chip('子层输出 (B, S, D)', 4);
    const cr = chip('旧流 residual (B, S, 4, D)', 1);
    const co = chip('新流 (B, S, 4, D)', 2);
    io.appendChild(cp); io.appendChild(U.el('div', { class: 'mono', style: 'color:var(--c2)', html: '&odot;' }));
    io.appendChild(ca); io.appendChild(U.el('div', { class: 'mono', style: 'color:var(--c2)', html: '+' }));
    io.appendChild(cr); io.appendChild(U.el('div', { class: 'mono', style: 'color:var(--accent)', html: '&rArr;' }));
    io.appendChild(co);
    viz.appendChild(io);

    const tbEl = W.table([
      ['残差条数', '1 条', '<span class="hl">4 条</span>（hc_mult）'],
      ['合并方式', 'x + f(norm(x))，加法', 'post &odot; f + comb<sup>T</sup> &middot; x，门控 + 混合'],
      ['子层输出的去向', '原样加到 x 上', '先被 post 逐流缩放，再广播到 4 条流'],
      ['旧流的去向', '原样保留', '被 comb 混合后才保留'],
      ['每层出现次数', '2 次', '2 次（同一段代码出现两次）'],
      ['写回幅度', '恒为 1', 'post &isin; (0, 2)：实测 mean 1.004、范围 [0.005, 1.968]'],
    ], { head: ['', '教科书 residual', 'mHC（本课）'] });
    const tbCard = U.el('div', { class: 'card cc1', style: 'padding:9px 11px;width:100%' }, [
      U.el('div', { class: 'klabel', text: '差异逐条对照（数字来自实测脚本）' }), tbEl,
    ]);
    viz.appendChild(tbCard);

    /* 逐流展开的手算核对 */
    const chk = U.el('div', { class: 'mono', style: 'font-size:11px;color:var(--ink-dim);'
      + 'background:rgba(6,12,26,.6);border:1px solid var(--panel-brd);border-radius:9px;padding:7px 11px;width:100%' });
    chk.innerHTML = '新流[i] = post[i] * 子层输出 + &Sigma;<sub>j</sub> comb[j, i] * 旧流[j]'
      + ' &nbsp;&nbsp;<span class="hl4">实测与源码写法逐元素一致 = True</span>';
    viz.appendChild(chk);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    const trs = U.qa('tbody tr', tbEl);
    trs.forEach(r => { r.style.opacity = '.3'; });
    f.innerHTML = '新流 = <em>post</em> &odot; 子层输出 &nbsp;<span class="cm">+</span>&nbsp; <em>combᵀ</em> &middot; 旧流';
    msg.innerHTML = '<span class="cm">// 第一项：post 逐元素门控"写多少"</span>';
    tl.at(1800, () => {
      cp.classList.add('ac'); co.classList.add('ac');
      msg.innerHTML = '第一项 <em>post &odot; f</em>：子层输出被复制到 4 条流上，每条流各缩放一次';
    });
    tl.at(4800, () => {
      cp.classList.remove('ac'); cr.classList.add('ac');
      msg.innerHTML = '第二项 <em>combᵀ &middot; 旧流</em>：每条新流拿到的是 4 条旧流的一个线性组合';
    });
    tl.at(7800, () => {
      cr.classList.remove('ac'); co.classList.add('ac');
      trs.forEach((r, i) => { r.style.opacity = i < 2 ? '1' : '.3'; });
      msg.innerHTML = '于是"残差"从一条加法，变成了 <em>门控 + 4 &times; 4 的流间混合</em>';
    });
    tl.at(10800, () => {
      trs.forEach((r, i) => { r.style.opacity = (i === 2 || i === 3) ? '1' : '.3'; });
      msg.innerHTML = '子层输出与旧流<em>都不再原样保留</em> —— 这是与教科书最本质的两处不同';
    });
    tl.at(13400, () => {
      trs.forEach(r => { r.style.opacity = '1'; });
      msg.innerHTML = '<span class="cm">// 写回幅度不是恒等的 1：post 从第一步起就是活的（下一幕讲它的来源）</span>';
    });
  },
},

/* ============================================================ 4 pre/post/comb */
{
  kicker: 'L3-08 · 第 4 幕 · 权重从哪来',
  title: 'pre / post / comb：<span class="hl-a">一次线性</span>切三段',
  sub: 'flatten 后一次 Linear 出 (2 + N) x N 个数，切成 pre(N) / post(N) / comb(N x N)；三段的激活函数各不相同。',
  caption: '回顾 L3-06：mHC 的全部可学参数就是 fn / base / scale 三个。',
  lang: 'python',
  codeStart: 314,
  code: `        # All weights are computed with a one layer perceptron. For pre and post, this is it.
        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)
        comb = torch.softmax(comb_w * comb_scale + comb_b, dim=-1) + self.hc_eps`,
  codeNote: 'Glm5NextTextHyperConnection.forward 的三个权重 —— 一行一条。',
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap10 vizgrow', style: 'width:100%' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', style: 'text-transform:none;letter-spacing:.02em',
      text: '(2 + N) x N = 24 个数（N = hc_mult = 4）：前 4 个 pre、中间 4 个 post、后 16 个 comb' }));
    const seg = W.tensor(1, 24, (r, c) => (c < 4 ? 1 : (c < 8 ? 0 : 2)), { cell: 30, gap: 3 });
    seg.querySelectorAll('.tcell').forEach(c => { c.style.transition = 'opacity .45s var(--ease-out)'; c.style.opacity = '.18'; });
    viz.appendChild(seg);

    const cards = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const k1 = W.card({ cc: 1, tint: 1, style: 'flex:1', title: 'pre：收拢权重',
      sub: '<code class="inl">sigmoid(w &middot; s + b) + hc_eps</code> &rarr; (0, 1+&epsilon;]。加 &epsilon; 是为了不让任何一条流被彻底清零。' });
    const k2 = W.card({ cc: 0, tint: 0, style: 'flex:1', title: 'post：写回门控',
      sub: '<code class="inl">2 &middot; sigmoid(...)</code> &rarr; <b>(0, 2)</b>。取 2 倍是为了让<em>平均</em>写回幅度落在 1 附近。' });
    const k3 = W.card({ cc: 2, tint: 2, style: 'flex:1', title: 'comb：流间混合',
      sub: '<code class="inl">softmax(..., dim=-1) + hc_eps</code> &rarr; 先按<em>行</em>归一，再交给 Sinkhorn 把列也拉平。' });
    [k1, k2, k3].forEach(c => cards.appendChild(c));
    viz.appendChild(cards);

    const scale = U.el('div', { class: 'mono', style: 'font-size:11px;color:var(--ink-dim);width:100%;'
      + 'background:rgba(6,12,26,.6);border:1px solid var(--panel-brd);border-radius:9px;padding:7px 11px' });
    scale.innerHTML = '实测尺度账：流经 <span class="hl">input_norm</span> 后 std = 1.0000 '
      + '&rarr; 投影维度 N&middot;D = 16384、fn 初始化 std = 0.02 &rarr; pre-activation std = '
      + '<span class="hl">' + M.preStd + '</span> &rarr; <span class="hlbad">post 实测 mean 1.004、范围 [0.005, 1.968]</span>';
    scale.style.opacity = '.25'; scale.style.transition = 'opacity .5s';
    viz.appendChild(scale);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    const paint = (a, b, on) => {
      seg.cells.forEach((c, i) => { if (i >= a && i < b) c.style.opacity = on ? '1' : '.18'; });
    };
    msg.innerHTML = '<span class="cm">// 24 个数就是全部：一次 F.linear 出三个权重</span>';
    tl.at(1800, () => {
      paint(0, 4, true); k1.classList.add('ac');
      msg.innerHTML = '<em>pre</em>（4 个）：4 条流按它加权求和成 1 条 —— 只在 mHC 内部用掉';
    });
    tl.at(4800, () => {
      k1.classList.remove('ac'); paint(4, 8, true); k2.classList.add('ac');
      msg.innerHTML = '<em>post</em>（4 个）：<em>2 &middot; sigmoid</em>，范围 (0, 2)，就是上一幕那个逐流门控';
    });
    tl.at(7800, () => {
      k2.classList.remove('ac'); paint(8, 24, true); k3.classList.add('ac');
      msg.innerHTML = '<em>comb</em>（16 个）：4 &times; 4 的矩阵，<em>softmax(dim=-1)</em> 先保证行和为 1';
    });
    tl.at(10800, () => {
      paint(0, 24, true); scale.style.opacity = '1';
      msg.innerHTML = '★ 实测订正：<em>base=0 / scale=1</em> 只让 post 的<em>均值</em>等于 1；'
        + '逐 token 逐流可以从 0.005 到 1.968';
    });
    tl.at(13400, () => {
      k3.classList.remove('ac');
      msg.innerHTML = '参数量：<em>fn 393,216 + base 24 + scale 3 = ' + U.fmt(M.hcParams) + '</em>，一层两个 = '
        + U.fmt(2 * M.hcParams);
    });
  },
},

/* ============================================================ 5 ★ Sinkhorn */
{
  kicker: 'L3-08 · 第 5 幕 · 数值 ★',
  title: 'comb 必须<span class="hl-a">双随机</span>：Sinkhorn 手算',
  sub: '只有行和、列和都为 1 的混合矩阵才不会放大范数。20 次迭代之后，它其实只是"近似"双随机。',
  caption: '本幕是本课的数值幕：手算一个 2x2，再拿真实 4x4 的实测偏差对照。',
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
  codeNote: 'Sinkhorn-Knopp：先列归一，再循环 (hc_sinkhorn_iters - 1) 次「行、列」。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap10 vizgrow', style: 'width:100%' });
    wrap.appendChild(viz);

    /* 四个 2x2 小矩阵：M0 / iters=1 / iters=2 / iters=20 */
    const mats = [
      { t: 'M0（softmax 之后）', v: [['0.9000', '0.4000'], ['0.3000', '0.8000']], rs: ['1.3000', '1.1000'], cs: ['1.2000', '1.2000'] },
      { t: 'iters = 1（只列归一）', v: [['0.7500', '0.3333'], ['0.2500', '0.6667']], rs: ['1.0833', '0.9167'], cs: ['1.0000', '1.0000'] },
      { t: 'iters = 2', v: [['0.7174', '0.2973'], ['0.2826', '0.7027']], rs: ['1.0147', '0.9853'], cs: ['1.0000', '1.0000'] },
      { t: 'iters = 20', v: [['0.7101', '0.2899'], ['0.2899', '0.7101']], rs: ['1.0000', '1.0000'], cs: ['1.0000', '1.0000'] },
    ];
    const row = U.el('div', { class: 'row gap10', style: 'width:100%;align-items:flex-start' });
    const matEls = mats.map((m, k) => {
      const box = U.el('div', { class: 'card cc' + (k === 3 ? 4 : 0), style: 'flex:1 1 0;padding:8px 9px;'
        + 'opacity:.3;transition:opacity .5s var(--ease-out),transform .5s var(--ease-out)' });
      box.appendChild(U.el('div', { class: 'mono', style: 'font-size:10px;color:var(--ink-faint);text-align:center', text: m.t }));
      const g = U.el('div', { style: 'display:grid;grid-template-columns:1fr 1fr;gap:3px;margin:6px 0 4px' });
      m.v.forEach(r => r.forEach(x => g.appendChild(U.el('div', { class: 'n sm',
        style: 'height:24px;border-radius:6px;display:flex;align-items:center;justify-content:center;'
             + 'background:rgba(56,189,248,.13);color:#e0f4ff', text: x }))));
      box.appendChild(g);
      box.appendChild(U.el('div', { class: 'mono', style: 'font-size:9.5px;color:var(--ink-faint);text-align:center',
        text: '行和 ' + m.rs.join(' / ') }));
      box.appendChild(U.el('div', { class: 'mono', style: 'font-size:9.5px;text-align:center;color:'
        + (m.cs[0] === '1.0000' ? 'var(--c4)' : 'var(--c3)'), text: '列和 ' + m.cs.join(' / ') }));
      row.appendChild(box); return box;
    });
    viz.appendChild(row);

    const cards = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const a1 = W.card({ cc: 0, tint: 0, style: 'flex:1', title: '迭代次数要数对',
      sub: '先列归一 <b>1</b> 次，再循环 <code class="inl">iters - 1</code> 次（每次行+列），'
        + '所以 <code class="inl">hc_sinkhorn_iters = 20</code> 实际是 <b>1 + 19 x 2 = 39</b> 次归一化。' });
    const a2 = W.card({ cc: 3, tint: 3, style: 'flex:1', title: '★ 实测：只是近似双随机',
      sub: '真实 4x4（hidden 4096）实测：<b>列和偏差 1.07e-06</b>（最后一步就是列归一，被 &epsilon; 钉死），'
        + '<b>行和偏差 1.77e-02</b> —— 还差 1.8%。' });
    const a3 = W.card({ cc: 1, tint: 1, style: 'flex:1', title: '为什么慢',
      sub: 'pre-activation 的 std 实测 2.49，softmax 出来很尖（4 类的行熵只有 0.159，均匀是 1.386）；'
        + '越尖的矩阵 Sinkhorn 收敛越慢。' });
    [a1, a2, a3].forEach(c => cards.appendChild(c));
    viz.appendChild(cards);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 手势：先把列和压到 1，再轮流压行和、列和</span>';
    tl.at(1800, () => {
      matEls[0].style.opacity = '1';
      msg.innerHTML = '起点 M0：行和 <em>1.30 / 1.10</em>、列和 <em>1.20 / 1.20</em>，两边都不是 1';
    });
    tl.at(4800, () => {
      matEls[1].style.opacity = '1'; matEls[1].style.transform = 'translateY(-4px)';
      msg.innerHTML = '第 1 步只做<em>列归一</em>：列和立刻变成 1，行和被推成 <em>1.0833 / 0.9167</em>';
    });
    tl.at(7800, () => {
      matEls[1].style.transform = 'none'; matEls[2].style.opacity = '1'; matEls[2].style.transform = 'translateY(-4px)';
      msg.innerHTML = '第 2 步<em>行、列各来一次</em>：行和误差从 8% 掉到 <em>1.5%</em> —— 每轮几何式缩小';
    });
    tl.at(11000, () => {
      matEls[2].style.transform = 'none'; matEls[3].style.opacity = '1';
      a1.classList.add('ac');
      msg.innerHTML = '到 <em>iters = 20</em>（实际 39 次归一化）：2x2 这个例子里两边都到 1e-6';
    });
    tl.at(14000, () => {
      a1.classList.remove('ac'); a2.classList.add('ac'); a3.classList.add('ac');
      msg.innerHTML = '★ 但真实 4x4 实测：<em>列和偏差 1.07e-06</em>、<em>行和偏差 1.77e-02</em> —— 是<em>近似</em>双随机';
    });
    tl.at(16600, () => {
      a2.classList.remove('ac'); a3.classList.remove('ac');
      msg.innerHTML = '<span class="cm">// 源码注释的意图是"约束范数不放大"；hc_sinkhorn_iters 就是精度旋钮</span>';
    });
  },
},

/* ============================================================ 6 顺序③ */
{
  kicker: 'L3-08 · 第 6 幕 · 顺序 ③',
  title: 'FFN 子块：<span class="hl-a">同一套模板</span>再缝一次',
  sub: '把注意力换成 mlp，只改了三个名字；写回那两行一字不差。层返回 (hidden_states, topk_indices)。',
  caption: '回顾 L0-05：mHC 在每个子层前后各出现一次 —— 这个"各出现一次"就是本幕的代码形态。',
  lang: 'python',
  codeStart: 1341,
  code: `        residual = hidden_states
        post, comb, hidden_states = self.ffn_hc(hidden_states)
        # Feed forward
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = post.to(dtype).unsqueeze(-1) * hidden_states.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual
        )

        return hidden_states, topk_indices`,
  codeNote: 'Glm5NextTextDecoderLayer.forward 的后半段 —— 与前半段逐行同构。',
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap10 vizgrow', style: 'width:100%' });
    wrap.appendChild(viz);

    const mk = (name, cc, steps) => {
      const box = W.card({ cc: cc, tint: cc, style: 'flex:1;opacity:.3;'
        + 'transition:opacity .5s var(--ease-out),transform .5s var(--ease-out)' });
      box.appendChild(U.el('div', { class: 'ct', html: name }));
      steps.forEach((s, i) => box.appendChild(U.el('div', { class: 'mono',
        style: 'font-size:11px;color:' + (i === steps.length - 1 ? 'var(--c2)' : 'var(--ink-dim)')
             + ';padding:2px 0;overflow-wrap:anywhere', text: (i + 1) + '. ' + s })));
      return box;
    };
    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const b1 = mk('子块 ①：注意力', 4, ['attn_hc(...)  → post, comb, collapsed',
      'input_layernorm(collapsed)', 'self_attn(norm)', '写回：post ⊙ out + combᵀ · residual']);
    const b2 = mk('子块 ②：MoE / MLP', 2, ['ffn_hc(...)  → post, comb, collapsed',
      'post_attention_layernorm(collapsed)', 'mlp(norm)', '写回：post ⊙ out + combᵀ · residual']);
    row.appendChild(b1); row.appendChild(b2);
    viz.appendChild(row);

    const same = U.el('div', { class: 'mono', style: 'font-size:11px;color:var(--ink-dim);width:100%;'
      + 'background:rgba(6,12,26,.6);border:1px solid var(--panel-brd);border-radius:9px;padding:8px 11px;'
      + 'opacity:.3;transition:opacity .5s' });
    same.innerHTML = '两次只有三处名字不同：<span class="hl">attn_hc → ffn_hc</span> &nbsp; '
      + '<span class="hl">input_layernorm → post_attention_layernorm</span> &nbsp; '
      + '<span class="hl">self_attn → mlp</span><br>'
      + '写回那两行（<span class="hl3">post.to(dtype).unsqueeze(-1) * ... + torch.matmul(combᵀ, residual)</span>）完全相同';
    viz.appendChild(same);

    const ret = U.el('div', { class: 'row gap10 center', style: 'width:100%' });
    ret.innerHTML = '<span class="chip c1">返回 hidden_states (B, S, 4, D)</span>'
      + '<span class="mono" style="color:var(--accent)">&nbsp;+&nbsp;</span>'
      + '<span class="chip c0">返回 topk_indices（可为 None）</span>';
    viz.appendChild(ret);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 注意力分支结束后，立刻写回</span>';
    tl.at(1800, () => {
      b1.style.opacity = '1'; b1.style.transform = 'translateY(-3px)';
      msg.innerHTML = '子块 ①：<em>mHC → norm → 注意力 → 写回</em>，四拍走完';
    });
    tl.at(4800, () => {
      b1.style.transform = 'none'; b2.style.opacity = '1'; b2.style.transform = 'translateY(-3px)';
      msg.innerHTML = '子块 ②：把 <em>self.attn_hc</em> 换成 <em>self.ffn_hc</em>，同一套模板再走四拍';
    });
    tl.at(7800, () => {
      b2.style.transform = 'none'; same.style.opacity = '1';
      msg.innerHTML = '差异只有<em>三处名字</em>；写回的两行代码一字不差 —— 这就是"装配模板只有一套"';
    });
    tl.at(10800, () => {
      ret.classList.add('ac');
      msg.innerHTML = '层交出两个东西：<em>4 维的流</em> + <em>topk_indices</em>（DSA 的稀疏选择，交给下一层）';
    });
    tl.at(13400, () => {
      ret.classList.remove('ac');
      msg.innerHTML = '<span class="cm">// 一层的语义边界是"子层 + 写回"，不是"子层"——写回才是把流接回去的那一步</span>';
    });
  },
},

/* ============================================================ 7 MoE 内部 */
{
  kicker: 'L3-08 · 第 7 幕 · 子块内部',
  title: 'MoE 子块内部：<span class="hl-a">稀疏 + 共享</span>，还有被裁两次的 SwiGLU',
  sub: '它是 42 层里的 mlp：路由选 8 个专家，再无条件加一路共享专家；两条路的 SwiGLU 都吃 swiglu_limit = 10.0。',
  caption: '回顾 L3-05：路由的分组细节在那一课；这里只看它在装配里的位置与裁剪手算。',
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
  codeNote: 'Glm5NextTextMoE.forward —— 内部还有一次残差：稀疏专家 + 共享专家。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const fl = W.flow([
      { t: 'hidden', s: 'norm 之后', cc: 0 },
      { t: 'gate', s: 'TopkRouter', cc: 2 },
      { t: 'experts', s: 'top-8 稀疏', cc: 1 },
      { t: '共享专家', s: 'Glm5NextTextMLP', cc: 4 },
      { t: '相加输出', s: 'hidden + shared', cc: 3 },
    ], { style: 'width:100%' });
    wrap.appendChild(U.el('div', { class: 'col gap8', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: 'MoE 子块内部：先存一份 residual，最后再加回来' }), fl,
    ]));

    const viz = U.el('div', { class: 'col gap10 vizgrow hstart', style: 'width:100%' });
    wrap.appendChild(viz);

    const tbEl = W.table([
      ['gate', '12.0 &nbsp; -3.0 &nbsp; 0.5', '10.0 &nbsp; -3.0 &nbsp; 0.5', '只裁上界'],
      ['up', '15.0 &nbsp; -12.0 &nbsp; 2.0', '10.0 &nbsp; -10.0 &nbsp; 2.0', '双边都裁'],
      ['silu(gate) &middot; up', '180.00 &nbsp; 1.7073 &nbsp; 0.6225', '<span class="hl">99.9955</span> &nbsp; 1.4228 &nbsp; 0.6225', '第 1 个元素差 1.8 倍'],
    ], { head: ['量', '原始值', '裁剪后（限幅 10.0）', '规则'] });
    const tbCard = U.el('div', { class: 'card cc2', style: 'padding:9px 11px;width:100%;opacity:.3;'
      + 'transition:opacity .5s' }, [
      U.el('div', { class: 'klabel', text: '手算例子（实测输出）' }), tbEl,
    ]);
    viz.appendChild(tbCard);

    const cards = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const c1 = W.card({ cc: 4, tint: 4, style: 'flex:1', title: '共享专家那一路',
      sub: '走 <code class="inl">Glm5NextTextMLP.forward</code>：<code class="inl">gate.clamp(min=None, max=limit)</code>、'
        + '<code class="inl">up.clamp(-limit, limit)</code>。' });
    const c2 = W.card({ cc: 1, tint: 1, style: 'flex:1', title: '路由专家那一路',
      sub: '走 <code class="inl">Glm5NextTextExperts._apply_gate</code>：同样的两条 clamp，注释写着 '
        + '<code class="inl">Simple swiglu instead of alpha</code>。' });
    [c1, c2].forEach(c => cards.appendChild(c));
    viz.appendChild(cards);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    fl.focus(0);
    msg.innerHTML = '<span class="cm">// 子块的输入是 norm 之后的 (B, S, D)</span>';
    tl.at(1800, () => { fl.focus(1); msg.innerHTML = '路由 <em>gate</em> 给出 (router_logits, topk_weights, topk_indices)'; });
    tl.at(4600, () => { fl.focus(2); msg.innerHTML = '稀疏专家只对选中的 <em>8</em> 个专家做 <em>index_add_</em> 累加'; });
    tl.at(7400, () => { fl.focus(3); msg.innerHTML = '同时<em>每个 token</em>都走一路共享专家（1 个，宽度 = moe_intermediate_size）'; });
    tl.at(10200, () => {
      fl.focus(4); tbCard.style.opacity = '1';
      msg.innerHTML = '两路都要过 SwiGLU：<em>gate 只裁上界、up 双边裁</em>，限幅 <em>10.0</em>';
    });
    tl.at(13000, () => {
      c1.classList.add('ac'); c2.classList.add('ac');
      msg.innerHTML = '同一个 <em>swiglu_limit</em> 在两个地方各裁一次 —— 两处都不能少';
    });
    tl.at(15000, () => {
      c1.classList.remove('ac'); c2.classList.remove('ac'); fl.focus(-1);
      msg.innerHTML = '最后 <em>hidden + shared</em>：MoE 子块内部也有一次残差，别和层级的 mHC 写回搞混';
    });
  },
},

/* ============================================================ 8 checkpointing */
{
  kicker: 'L3-08 · 第 8 幕 · 推理行为',
  title: '推理时 <span class="hl-a">GradientCheckpointingLayer</span> 什么都不做',
  sub: '分支条件是 gradient_checkpointing and self.training；eval 之后第二个条件就是 False，直接走普通 __call__。',
  caption: '本幕只讲推理侧：所有 checkpoint 的复杂度都留在训练侧，推理路径上它退化成 nn.Module。',
  lang: 'python',
  codeStart: 76,
  code: `    gradient_checkpointing = False
    # Layers that only read the KV cache can set this to keep it under gradient checkpointing (the recompute reads the
    # same states). Writers must leave it \`False\`, otherwise the cache is updated a second time on the backward replay.
    _can_checkpoint_with_cache = False

    def __call__(self, *args, **kwargs):
        if self.gradient_checkpointing and self.training:
            do_warn = False
            layer_name = self.__class__.__name__
            message = f"Caching is incompatible with gradient checkpointing in {layer_name}. Setting"

            if "use_cache" in kwargs and kwargs["use_cache"]:
                kwargs["use_cache"] = False
                message += " \`use_cache=False\`,"
                do_warn = True

            if not self._can_checkpoint_with_cache:
                # different names for the same thing in different layers
                if "past_key_values" in kwargs and kwargs["past_key_values"] is not None:
                    kwargs["past_key_values"] = None
                    message += " \`past_key_values=None\`,"
                    do_warn = True

                if "layer_past" in kwargs and kwargs["layer_past"] is not None:
                    kwargs["layer_past"] = None
                    message += " \`layer_past=None\`,"
                    do_warn = True

            # warn if anything was changed
            if do_warn:
                message = message.rstrip(",") + "."
                logger.warning_once(message)

            return self._gradient_checkpointing_func(partial(super().__call__, **kwargs), *args)
        return super().__call__(*args, **kwargs)`,
  codeNote: 'modeling_layers.py 的 GradientCheckpointingLayer —— 解码层的基类，全文只有这两段。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap10 vizgrow hstart', style: 'width:100%' });
    wrap.appendChild(viz);

    /* 分支图 */
    const br = U.el('div', { class: 'row gap10 center', style: 'width:100%' });
    const cond = U.el('div', { class: 'card cc0', style: 'padding:9px 12px;text-align:center;flex:none' }, [
      U.el('div', { class: 'mono', style: 'font-size:11.5px;color:#fff', text: 'def __call__(self, *args, **kwargs)' }),
      U.el('div', { class: 'mono', style: 'font-size:10.5px;color:var(--c2);margin-top:3px',
        text: 'if self.gradient_checkpointing and self.training:' }),
    ]);
    const git = U.el('div', { class: 'card cc1', style: 'padding:9px 12px;text-align:center;flex:1;opacity:.3;'
      + 'transition:opacity .5s' }, [
      U.el('div', { class: 'mono', style: 'font-size:11px;color:#ddd6fe', text: '训练分支：_gradient_checkpointing_func(...)' }),
      U.el('div', { class: 'mono', style: 'font-size:10px;color:var(--ink-faint);margin-top:3px',
        text: 'torch.utils.checkpoint：用重算换显存' }),
    ]);
    const inf = U.el('div', { class: 'card cc4', style: 'padding:9px 12px;text-align:center;flex:1;opacity:.3;'
      + 'transition:opacity .5s' }, [
      U.el('div', { class: 'mono', style: 'font-size:11px;color:#a7f3d0', text: '否则：super().__call__(*args, **kwargs)' }),
      U.el('div', { class: 'mono', style: 'font-size:10px;color:var(--ink-faint);margin-top:3px',
        text: '普通 nn.Module.__call__ —— 推理走这条' }),
    ]);
    br.appendChild(cond);
    br.appendChild(U.el('div', { class: 'mono', style: 'color:var(--accent);flex:none', html: '&rarr;' }));
    br.appendChild(git);
    br.appendChild(inf);
    viz.appendChild(br);

    const tbEl = W.table([
      ['<code class="inl">gradient_checkpointing</code>（类属性）', '<span class="hl4">False</span>', '构造后就是 False，没人改它'],
      ['<code class="inl">_gradient_checkpointing_func</code>', '<span class="hl4">属性不存在</span>', '只有 gradient_checkpointing_enable() 才 setattr'],
      ['eval + no_grad 调用一次', '<span class="hl4">0 次</span>', '把该属性换成计数器，实测 0'],
      ['train + gradient_checkpointing=True', '<span class="hlbad">1 次</span>', '对照组：同一个层，只差 training 标志'],
      ['<code class="inl">_can_checkpoint_with_cache</code>', 'False', '训练时才用它把 use_cache 置 False'],
    ], { head: ['实测项', '结果', '说明'] });
    const tbCard = U.el('div', { class: 'card cc1', style: 'padding:9px 11px;width:100%' }, [
      U.el('div', { class: 'klabel', text: '实测：推理路径上没有任何 checkpoint 开销' }), tbEl,
    ]);
    viz.appendChild(tbCard);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    const trs = U.qa('tbody tr', tbEl);
    trs.forEach(r => { r.style.opacity = '.3'; });
    msg.innerHTML = '<span class="cm">// 解码层继承它，只为训练期服务</span>';
    tl.at(1800, () => {
      git.style.opacity = '1';
      msg.innerHTML = '训练分支：前向包进 <em>torch.utils.checkpoint</em>，用重算换显存';
    });
    tl.at(4800, () => {
      git.style.opacity = '.25'; inf.style.opacity = '1'; inf.classList.add('ac');
      msg.innerHTML = '推理：<em>self.training = False</em>，条件不成立 &rarr; 直接 <em>super().__call__</em>';
    });
    tl.at(7800, () => {
      inf.classList.remove('ac'); trs[0].style.opacity = '1'; trs[1].style.opacity = '1';
      msg.innerHTML = '两道保险：类属性 <em>False</em>；而且<em>那个函数属性根本不存在</em>（没 enable 就不会挂上）';
    });
    tl.at(10800, () => {
      trs[2].style.opacity = '1'; trs[3].style.opacity = '1';
      msg.innerHTML = '实测：eval + no_grad 调用 <em>0 次</em>；同一层 train + 开关打开时 <em>1 次</em>';
    });
    tl.at(13600, () => {
      trs.forEach(r => { r.style.opacity = '1'; }); trs[4].style.opacity = '1';
      msg.innerHTML = '那条 <em>_can_checkpoint_with_cache = False</em> 只在训练分支里起作用：它会强制 use_cache=False';
    });
    tl.at(15800, () => {
      msg.innerHTML = '<span class="cm">// 结论：推理时它是一条普通前向链，KV cache 照常累积（L5 不再需要考虑它）</span>';
    });
  },
},

/* ============================================================ 9 收束 */
{
  kicker: 'L3-08 · 收束',
  title: '把一层默写出来：<span class="hl-a">四拍</span>跑两遍',
  sub: '返回值是元组，主干出口才包装成数据类；三条数据类约定决定了下游该怎么读它。',
  caption: '下一课 L4-01：为什么要混合两种注意力 —— 34 层 KDA 与 11 层 MLA+DSA 的分工。',
  lang: 'python',
  codeStart: 345,
  code: `    last_hidden_state: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    router_logits: tuple[torch.FloatTensor] | None = None`,
  codeNote: 'MoeModelOutputWithPast 的字段 —— 主干 forward 最后一行构造的就是它。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const exit = U.el('div', { class: 'mono', style: 'font-size:11.5px;color:var(--ink-dim);width:100%;'
      + 'background:rgba(6,12,26,.6);border:1px solid var(--panel-brd);border-radius:9px;padding:8px 11px' });
    exit.innerHTML = '主干出口（逐字，L1511-1512）：<br>'
      + 'hidden_states = self.norm(self.hc_head(hidden_states))<br>'
      + 'return <span class="hl">MoeModelOutputWithPast</span>(last_hidden_state=hidden_states, past_key_values=past_key_values)';
    wrap.appendChild(exit);

    const tbEl = W.table([
      ['每个字段都有默认值', '<code class="inl">X | None = None</code>', '只给两个字段也能构造（主干就是这么写的）'],
      ['字段顺序 = 位置参数顺序', '<code class="inl">o[0] is o.last_hidden_state</code> &rarr; True', '@dataclass：改顺序就是破坏性改动'],
      ['None 字段被丢掉', "<code class=\"inl\">keys()</code> &rarr; ['last_hidden_state', 'past_key_values']", '长度会随开关变，别按位置解包'],
      ['它是 dict 的子类', '<code class="inl">isinstance(o, dict)</code> &rarr; True', '<code class="inl">**outputs</code> 合法'],
    ], { head: ['约定', '实测', '后果'] });
    const tbCard = U.el('div', { class: 'card cc4', style: 'padding:9px 11px;width:100%;opacity:.35;'
      + 'transition:opacity .5s' }, [
      U.el('div', { class: 'klabel', text: '数据类约定（三条都是实测）' }), tbEl,
    ]);
    wrap.appendChild(tbCard);

    const steps = ['attn_hc', 'input_layernorm', 'self_attn', '写回 ①', 'ffn_hc',
      'post_attention_layernorm', 'mlp', '写回 ②', 'return (hidden, topk)'];
    const strip = U.el('div', { class: 'row gap4', style: 'width:100%' });
    const stepEls = steps.map((s, i) => {
      const e = U.el('div', { class: 'n sm', style: 'flex:1 1 0;min-width:0;height:30px;border-radius:7px;'
        + 'display:flex;align-items:center;justify-content:center;text-align:center;font-size:10px;'
        + 'border:1px solid ' + (s.indexOf('写回') === 0 ? 'var(--c2)' : 'var(--panel-brd)') + ';'
        + 'background:' + (s.indexOf('写回') === 0 ? 'rgba(251,191,36,.14)' : 'rgba(150,180,255,.09)') + ';'
        + 'color:' + (s.indexOf('写回') === 0 ? '#fde68a' : 'var(--ink-dim)') + ';'
        + 'overflow-wrap:anywhere;word-break:break-all;padding:0 3px;line-height:1.15' },
        [U.el('span', { text: s })]);
      strip.appendChild(e); return e;
    });
    wrap.appendChild(U.el('div', { class: 'col gap6', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', style: 'text-transform:none;letter-spacing:.02em',
        text: '默写用：mHC → norm → 子层 → 写回，跑两遍' }), strip, 
    ]));

    const exRow = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const ex1 = W.exercise(
      '不看笔记，写出 <code class="inl">Glm5NextTextDecoderLayer.forward</code> 的 6 次子模块调用。',
      '<b>attn_hc → input_layernorm → self_attn → ffn_hc → post_attention_layernorm → mlp</b>。'
      + '<br>两次写回不在这个列表里 —— 它们是 forward 里的两行内联张量运算。'
      + '<br>骨架：<code class="inl">mHC → norm → 子层 → 写回</code> 跑两遍，'
      + '第二遍把 <code class="inl">attn_hc / input_layernorm / self_attn</code> '
      + '换成 <code class="inl">ffn_hc / post_attention_layernorm / mlp</code>。');
    const ex2 = W.exercise(
      '标准 Transformer 的 residual 与 mHC 下的 residual，差在哪三处？',
      '① <b>条数</b>：1 条 vs <b>4 条</b>（<code class="inl">hc_mult</code>）；'
      + '② <b>合并方式</b>：<code class="inl">x + f(norm(x))</code> vs '
      + '<code class="inl">post &odot; f + combᵀ &middot; x</code> —— 子层输出先被门控、旧流先被混合；'
      + '③ <b>幅度</b>：恒为 1 vs <code class="inl">post &isin; (0, 2)</code>（实测 mean 1.004、范围 [0.005, 1.968]）。'
      + '<br>约束 <code class="inl">comb</code> 双随机（Sinkhorn）就是为了让第 ② 步的混合不放大范数。');
    [ex1, ex2].forEach(e => { e.style.flex = '1'; exRow.appendChild(e); });
    wrap.appendChild(exRow);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 一层返回元组，主干出口才包装成数据类</span>';
    tl.at(1800, () => {
      msg.innerHTML = '主干出口：<em>norm(hc_head(...))</em> 把 4 条流收成 1 条，装进 <em>MoeModelOutputWithPast</em>';
    });
    tl.at(4600, () => {
      tbCard.style.opacity = '1'; U.qa('tbody tr', tbEl)[0].classList.add('ac');
      U.qa('tbody tr', tbEl)[1].classList.add('ac');
      msg.innerHTML = '约定 ① 字段全可省、② 顺序即位置 —— 所以<em>只给两个字段</em>也合法';
    });
    tl.at(7400, () => {
      U.qa('tbody tr', tbEl).forEach(r => r.classList.remove('ac'));
      U.qa('tbody tr', tbEl)[2].classList.add('ac'); U.qa('tbody tr', tbEl)[3].classList.add('ac');
      msg.innerHTML = '约定 ③ <em>None 字段被丢掉</em>：打开 output_hidden_states 后返回值会变长，别按位置解包';
    });
    tl.at(10200, () => {
      U.qa('tbody tr', tbEl).forEach(r => r.classList.remove('ac'));
      stepEls.forEach((e, i) => {
        e.style.transition = 'opacity .3s var(--ease-out)';
        e.style.opacity = i < 4 ? '1' : '.3';
      });
      msg.innerHTML = '默写第一遍：<em>attn_hc → input_layernorm → self_attn → 写回 ①</em>';
    });
    tl.at(13000, () => {
      stepEls.forEach((e, i) => { e.style.opacity = i >= 4 ? '1' : '.45'; });
      msg.innerHTML = '第二遍：<em>ffn_hc → post_attention_layernorm → mlp → 写回 ②</em> → return';
    });
    tl.at(15600, () => {
      stepEls.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = '<span class="cm">// 下一课 L4-01：34 层 KDA 与 11 层 MLA+DSA 为什么要混着排</span>';
    });
  },
},

];
