/* ==========================================================================
   L3-03 · 路由器：288 个分数如何变成 8 条路径
   --------------------------------------------------------------------------
   覆盖：models/glm5_next/modeling_glm5_next.py（1 个文件 / 2444 行）
   目标：看完能手算一次 top-8 选择，并说清分组路由在分布式下省下了什么。
   排版基准：#visual 可视区 = 994 x 662.7 舞台像素（无头浏览器实测）。
   数据来源：_data/recon/probe_l303.py 的实跑输出（不是估算）。
   ========================================================================== */
'use strict';

/* 实测常量 —— 全部来自 probe_l303.py 的输出 */
const M = {
  experts: 288,        /* n_routed_experts */
  topk: 8,             /* num_experts_per_tok */
  groups: 8,           /* n_group（默认是 1） */
  perGroup: 36,        /* 288 / 8 */
  topkGroup: 2,        /* topk_group（默认是 1） */
  factor: 2.5,         /* routed_scaling_factor */
  /* 默认配置、随机权重、1 个 token 的实跑结果 */
  idx: [164, 26, 287, 215, 114, 147, 128, 284],
  w: [0.306771, 0.319676, 0.307620, 0.312870, 0.315230, 0.319281, 0.312806, 0.305747],
  wNoNorm: [2.294565, 2.391088, 2.300918, 2.340181, 2.357833, 2.388132, 2.339705, 2.286903],
  sumNoNorm: 18.699324,
  /* n_group=8 / topk_group=2 的实跑结果 */
  gscore: [1.7871, 1.7063, 1.7538, 1.8790, 1.8731, 1.7889, 1.7762, 1.8351],
  gwin: [3, 4],
  gidx: [126, 128, 147, 114, 164, 136, 173, 125],
  /* 偏置实验：e164 加 −5、e7 加 +5 */
  idxBiased: [215, 26, 114, 147, 7, 287, 128, 284],
  /* 8 专家 / 2 组的最小例子 */
  tiny: {
    logit: [1.0, 2.0, -1.0, 1.5, 3.0, 2.5, 2.0, -2.0],
    sig: [0.731059, 0.880797, 0.268941, 0.817574, 0.952574, 0.924142, 0.880797, 0.119203],
    gsum: [1.698371, 1.876716],
    pick: [4, 5],
    raw: [0.952574, 0.924142],
    norm: [0.507575, 0.492425],
    final: [1.268938, 1.231063],
  },
  /* sigmoid vs softmax 的对照（同一组 5 个 logit） */
  cmp: {
    sig: [0.880797, 0.924142, 0.731059, 0.817574, 0.622459],
    sm: [0.259993, 0.428656, 0.095646, 0.157694, 0.058012],
    sig2: [0.982014, 0.924142, 0.731059, 0.817574, 0.622459],
    sm2: [0.721918, 0.161082, 0.035942, 0.059259, 0.021800],
  },
};

/* 确定性伪随机：build 必须幂等，所以不能用 Math.random */
const pix = i => ((i * 2654435761) % 1000) / 1000;

/* 小工具：数值卡片（值在上、说明在下） */
const statCard = (st) => W.card({
  cc: st.cc, tint: st.cc, style: 'flex:1;min-width:0',
  title: '<span class="mono" style="font-size:10.5px">' + st.k + '</span>',
  body: U.el('div', { class: 'col gap4' }, [
    U.el('div', { class: 'n big', text: st.v }),
    U.el('div', { style: 'font:400 11px/1.45 var(--sans);color:var(--ink-dim)', html: st.d }),
  ]),
});

const SCENES = [

/* ------------------------------------------------- 1 全景：288 → 8 */
{
  kicker: '文本主干 · 全景',
  title: '288 个分数，<span class="hl-a">8 条路径</span>',
  sub: '一个 token 的 4096 维向量进入 MoE 层后，先被压成 288 个分数，再被压成 8 个「专家号 + 权重」对。'
     + '路由器只做这一件事，它自己不算任何专家。',
  caption: '这一课只讲分诊台。专家怎么算、权重怎么存，在 L3-04；共享专家在 L3-05。',
  lang: 'python',
  codeStart: 159,
  code: `    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        scores = router_logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias`,
  codeNote: 'Glm5NextTextTopkRouter.forward 的前 4 行 —— 整条路由链路的入口。',
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart', style: 'width:100%' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '一个 token 在 MoE 层里被分诊的全过程' }));

    const fl = W.flow([
      { t: 'h', s: '(N, 4096)', cc: 0 },
      { t: 'F.linear', s: '4096 → 288', cc: 4 },
      { t: 'sigmoid', s: '∈ (0, 1)', cc: 2 },
      { t: 'top-8', s: 'sorted=False', cc: 1 },
      { t: '8 条路径', s: '(专家号, 权重)', cc: 5 },
    ], { style: 'width:100%' });
    viz.appendChild(fl);

    const stats = [
      { k: 'n_routed_experts', v: '288', d: '候选专家总数', cc: 0 },
      { k: 'num_experts_per_tok', v: '8', d: '每个 token 只走 8 条', cc: 4 },
      { k: 'routed_scaling_factor', v: '2.5', d: '8 个权重之和恒等于它', cc: 3 },
      { k: 'n_group（默认）', v: '1', d: '★ 所以组筛选当前在空转', cc: 2 },
    ];
    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cards = stats.map(st => { const c = statCard(st); row.appendChild(c); return c; });
    viz.appendChild(row);

    /* 候选池的三次收缩：288 → 72 → 8 */
    const stk = W.stack([
      { label: '216 个被 -inf 淘汰', value: 216, cc: 3 },
      { label: '64 个落选', value: 64, cc: 2 },
      { label: '', value: 8, cc: 4 },
    ]);
    viz.appendChild(U.el('div', { class: 'col gap5', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: '候选池的三次收缩 288 → 72 → 8（实测：n_group=8 / topk_group=2 时）' }), stk,
    ]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);
    cards.forEach(c => { c.style.opacity = '.32'; });
    stk.reveal(0);

    fl.focus(0);
    msg.innerHTML = '<span class="cm">// 288 个候选挑 8 个 —— 但这不是一次 top-8，中间还藏着两级筛选</span>';
    tl.at(1600, () => {
      fl.focus(1);
      msg.innerHTML = '第一步 <em>F.linear(h, W)</em> &nbsp;<span class="cm">// W 的形状 (288, 4096)</span>';
    });
    tl.at(4400, () => {
      fl.focus(2);
      msg.innerHTML = '第二步 <em>sigmoid</em> &nbsp;<span class="cm">// ★ 不是 softmax，本课第一节就订正这个</span>';
    });
    tl.at(7200, () => {
      fl.focus(3);
      msg.innerHTML = '第三步 <em>topk(8)</em> &nbsp;<span class="cm">// sorted=False：返回的 8 个下标是无序的</span>';
    });
    tl.at(10000, () => {
      fl.focus(4);
      msg.innerHTML = '第四步 <em>归一化 × 2.5</em> &nbsp;<span class="cm">// 权重和恒为常数，与选到谁无关</span>';
    });
    tl.at(13000, () => {
      fl.focus(-1);
      cards.forEach(c => { c.style.opacity = '1'; c.classList.add('ac'); });
      stk.reveal(1);
      msg.innerHTML = '<span class="cm">// 288 → 72 → 8：记住这四个数 288 / 8 / 2.5 / 1</span>';
    });
  },
},

/* ------------------------------------------------- 2 打分层 */
{
  kicker: '第一步 · 打分',
  title: 'gate 就是一层 <span class="hl-a">Linear(4096 → 288)</span>',
  sub: 'self.weight 的形状是 (num_experts, hidden_dim)。所谓「288 个分数」，'
     + '就是一次矩阵乘法的输出 —— 没有第二个来源。',
  caption: '回顾 L0-01：同一份 hidden，在这里看到的是 288 个不同的分数。',
  lang: 'python',
  codeStart: 146,
  code: `class Glm5NextTextTopkRouter(nn.Module):
    def __init__(self, config: Glm5NextTextConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))
        self.routed_scaling_factor = config.routed_scaling_factor
        self.num_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.e_score_correction_bias = nn.Buffer(torch.zeros(self.num_experts))`,
  codeNote: '路由器持有的全部状态：一个 (288, 4096) 的权重，一个 288 维的偏置。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart', style: 'width:100%' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '一次矩阵乘法：(N, 4096) × (4096, 288) → (N, 288)' }));

    const stage = U.el('div', { class: 'row gap20 center', style: 'width:100%;flex:1;min-height:0' });
    viz.appendChild(stage);

    /* 左：h */
    const left = U.el('div', { class: 'col gap6 center' });
    left.appendChild(U.el('div', { class: 'klabel', text: 'h（投影前）' }));
    const hbar = U.el('div', {
      style: 'width:48px;height:170px;border-radius:10px;display:flex;align-items:center;'
           + 'justify-content:center;border:1px solid var(--c0);'
           + 'background:linear-gradient(180deg,rgba(56,189,248,.34),rgba(56,189,248,.10));'
           + 'transition:opacity .45s,transform .45s;font:600 11px var(--mono);color:#bae6fd',
      text: '4096',
    });
    left.appendChild(hbar);
    stage.appendChild(left);

    const mid = U.el('div', { class: 'col center gap4' });
    mid.innerHTML = '<div style="font-size:20px;color:var(--accent)">×</div>'
      + '<div class="mono" style="font-size:10.5px;color:var(--c4)">Wᵀ</div>'
      + '<div class="mono" style="font-size:10px;color:var(--ink-faint)">(4096, 288)</div>';
    stage.appendChild(mid);

    /* 右：288 个 logits */
    const right = U.el('div', { class: 'col gap6 center' });
    right.appendChild(U.el('div', { class: 'klabel', text: 'router_logits —— 288 个分数' }));
    const tg = W.tensor(16, 18, () => -1, { cell: 13, gap: 2 });
    tg.cells.forEach((c, i) => {
      c.style.background = 'rgba(52,211,153,' + (0.14 + 0.62 * pix(i)).toFixed(3) + ')';
      c.style.borderColor = 'rgba(52,211,153,.34)';
      c.style.opacity = '0';
      c.style.transform = 'scale(.5)';
      c.style.transition = 'opacity .3s,transform .3s';
    });
    right.appendChild(tg);
    stage.appendChild(right);

    const cards = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    [
      { k: 'self.weight', v: '(288, 4096)', d: '裸 Parameter，不是 nn.Linear', cc: 4 },
      { k: '.view(-1, self.hidden_dim)', v: '(B, S, 4096) → (N, 4096)', d: '路由逐 token 独立，batch/seq 无意义', cc: 0 },
      { k: '.type(torch.float32)', v: '强制 fp32', d: 'bf16 下 288 路排序会抖动，抖动会被 MoE 放大', cc: 3 },
    ].forEach(st => { const c = statCard(st); c.style.opacity = '.3'; cards.appendChild(c); });
    viz.appendChild(cards);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 先看输入：一个 token 的 4096 维向量</span>';
    tl.at(1800, () => {
      hbar.style.opacity = '1';
      msg.innerHTML = 'h 是一个 <em>4096</em> 维行向量 &nbsp;<span class="cm">// (batch, seq, 4096) 拍平后是 (N, 4096)</span>';
    });
    tl.at(4600, () => {
      msg.innerHTML = '<em>W</em> 的形状是 (288, 4096) —— <span class="cm">// 288 行，每一行是一个专家的"考卷"</span>';
    });
    tl.at(7400, () => {
      tg.cells.slice(0, 144).forEach((c, i) => {
        c.style.transitionDelay = (i * 3) + 'ms'; c.style.opacity = '1'; c.style.transform = 'none';
      });
      msg.innerHTML = '前 <em>144</em> 个 logits 算完 &nbsp;<span class="cm">// 每个分数 = h 与某一行 W 的点积</span>';
    });
    tl.at(10200, () => {
      tg.cells.slice(144).forEach((c, i) => {
        c.style.transitionDelay = (i * 3) + 'ms'; c.style.opacity = '1'; c.style.transform = 'none';
      });
      msg.innerHTML = '全部 <em>288</em> 个 logits 就绪 &nbsp;<span class="cm">// 形状 (N, 288)</span>';
    });
    tl.at(13400, () => {
      cards.forEach(c => { c.style.opacity = '1'; });
      msg.innerHTML = '<span class="cm">// 三个细节：裸 Parameter / 逐 token / 强制 fp32</span>';
    });
  },
},

/* ------------------------------------------------- 3 ★ 订正 sigmoid */
{
  kicker: '★ 订正 · 打分函数',
  title: '不是 <span class="hl-b">softmax</span>，是 <span class="hl-a">sigmoid</span>',
  sub: '作业书把这条要点写成「gate 线性层 + softmax」。实测：整个 Glm5NextTextTopkRouter 里'
     + '一次 softmax 都没有 —— 源码用的是 router_logits.sigmoid()。',
  caption: '凡「文件里写着」的说法都要跑一遍确认 —— 作业书也不例外。',
  lang: 'python',
  codeStart: 161,
  code: `        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        scores = router_logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias`,
  codeNote: '第 2 行是订正的关键：sigmoid，逐元素，与其他 287 个专家无关。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap10 vizgrow hstart', style: 'width:100%' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '同一组 5 个 logit = [2.0, 2.5, 1.0, 1.5, 0.5]，两种归一化的实跑值' }));

    const mkBars = (title, sub, vals, cc) => {
      const bars = W.bars(vals.map((v, i) => ({
        label: 'e' + (i + 1), value: v, cc: cc, valueText: v.toFixed(6),
      })), { max: 1.0 });
      const c = W.card({ cc: cc, tint: cc, style: 'flex:1;min-width:0',
        title: title, sub: sub, body: bars });
      c._bars = bars;
      return c;
    };

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cA = mkBars('sigmoid —— 本课实际用的', '每个分数只看自己那一个 logit', M.cmp.sig, 0);
    const cB = mkBars('softmax —— 这里没有用', '每个分数都被其余 287 个 logit 影响', M.cmp.sm, 2);
    row.appendChild(cA); row.appendChild(cB);
    viz.appendChild(row);

    const tb = W.table([
      ['单个分数取决于', '<span class="hl">只有自己那一个 logit</span>', '全部 288 个 logit'],
      ['改动其中一个专家', '<span class="hl">其余分数一个都不动</span>', '<span class="hlbad">其余 287 个全部变化</span>'],
      ['语义', '「我有多够格」', '「我占全体的比例」'],
    ], { head: ['对比项', 'sigmoid（实际）', 'softmax（不存在）'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 0, style: 'padding:9px 12px' }, [tb]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    const setW = (card, vals) => card._bars.fills.forEach((f, i) => {
      f.bf.style.width = (vals[i] * 100).toFixed(1) + '%';
      f.bv.textContent = vals[i].toFixed(6);
    });
    setW(cA, M.cmp.sig.map(() => 0));
    setW(cB, M.cmp.sm.map(() => 0));
    cA.style.opacity = '1'; cB.style.opacity = '.32';

    msg.innerHTML = '<span class="cm">// 先看 sigmoid：5 个分数互不相干</span>';
    tl.at(1800, () => {
      setW(cA, M.cmp.sig);
      msg.innerHTML = 'sigmoid 的和 = <em>3.976031</em> &nbsp;<span class="cm">// 不是 1，也不需要是 1</span>';
    });
    tl.at(4600, () => {
      setW(cA, M.cmp.sig2);
      msg.innerHTML = '把 e1 的 logit 从 <em>2.0</em> 抬到 <em>4.0</em>：只有 e1 变（Δ=0.101217），'
        + '<span class="cm">// 其余 4 个 Δ 全是 0</span>';
    });
    tl.at(7600, () => {
      cB.style.opacity = '1';
      setW(cB, M.cmp.sm);
      msg.innerHTML = '换成 softmax：和 = <em>1.000000</em> &nbsp;<span class="cm">// 漂亮，但它付出了耦合的代价</span>';
    });
    tl.at(10600, () => {
      setW(cB, M.cmp.sm2);
      msg.innerHTML = '同样把 e1 抬到 4.0：Δ = <span class="hlbad">+0.461925 / −0.267574 / −0.059704 / '
        + '−0.098435 / −0.036212</span> <span class="cm">// 全部 5 个都动了</span>';
    });
    tl.at(13800, () => {
      msg.innerHTML = '<span class="cm">// 实测 grep：softmax 只出现在 Indexer(L1002)、注意力(L1094)、'
        + '辅助统计(L2096) —— 一行都不在 TopkRouter 里</span>';
    });
  },
},

/* ------------------------------------------------- 4 第一级筛选 */
{
  kicker: '第二步 · 第一级筛选',
  title: '先把 288 个专家切成 <span class="hl-a">n_group</span> 个组',
  sub: '每组只留分数最高的 2 个之和，作为该组的代表分。于是 (N, 288) 被压成 (N, n_group) ——'
     + '第一级筛选的全部意义就在这个形状变化里。',
  caption: '★ 陷阱：默认 n_group=1，这一级什么都没筛。下面有实测。',
  lang: 'python',
  codeStart: 164,
  code: `        group_scores = (
            scores_for_choice.view(-1, self.num_group, self.num_experts // self.num_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )`,
  codeNote: 'num_group=8 时形状是 (N, 8, 36)，再 topk(2).sum(-1) → (N, 8)。',
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap10 vizgrow hstart', style: 'width:100%' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '实测：n_group=8 → 8 组 × 每组 36 个专家，每组只报一个数' }));

    const row = U.el('div', { class: 'row gap8', style: 'width:100%;align-items:stretch' });
    const cells = M.gscore.map((v, i) => {
      const c = W.card({ cc: i, style: 'flex:1 1 0;min-width:0;padding:9px 8px;text-align:center',
        title: '<span class="mono" style="font-size:11px">组 ' + i + '</span>',
        body: U.el('div', { class: 'col gap3 center' }, [
          U.el('div', { class: 'dim', style: 'font:400 10px/1.3 var(--mono)', text: '36 个专家' }),
          U.el('div', { class: 'n', style: 'font-size:14.5px', text: v.toFixed(4) }),
        ]) });
      c.style.opacity = '0'; c.style.transform = 'translateY(10px)';
      c.style.transition = 'opacity .4s,transform .4s';
      row.appendChild(c); return c;
    });
    viz.appendChild(row);

    const shape = U.el('div', { class: 'formula', style: 'width:100%' });
    shape.innerHTML = 'group_scores &nbsp;<em>(N, 288)</em> <span class="op">→</span> '
      + '<em>(N, 8, 36)</em> <span class="op">→</span> <em>(N, 8)</em> '
      + '&nbsp;<span class="cm">// 288 个数被压成 8 个数</span>';
    viz.appendChild(shape);

    const warn = U.el('div', { class: 'card cc2', style: 'width:100%;padding:9px 12px' });
    warn.innerHTML = '<div class="klabel">★ 默认配置的实测（probe_l303.py 第 2 节）</div>'
      + '<div class="mono" style="font-size:11.5px;line-height:1.7;color:var(--ink-dim)">'
      + 'n_group = 1（默认）&nbsp;→&nbsp; group_scores.shape = <span class="hl">(1, 1)</span>'
      + ' &nbsp;·&nbsp; score_mask 全 1 = <span class="hl4">True</span>'
      + ' &nbsp;·&nbsp; masked_fill 后还有 -inf = <span class="hlbad">False</span><br>'
      + '<span class="cm">// 288 个专家挤在一个组里 → 第一级筛选是一个彻底的 no-op</span></div>';
    viz.appendChild(warn);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 288 个专家的分数先被切成 8 组</span>';
    const rev = (a, b, at) => tl.at(at, () => {
      cells.slice(a, b).forEach((c, k) => {
        c.style.transitionDelay = (k * 60) + 'ms'; c.style.opacity = '1'; c.style.transform = 'none';
      });
      msg.innerHTML = '组 <em>' + a + '–' + (b - 1) + '</em> 报出自己的代表分 &nbsp;'
        + '<span class="cm">// 每组 = 该组分数最高的两个之和</span>';
    });
    rev(0, 2, 1400); rev(2, 4, 3600); rev(4, 6, 5800); rev(6, 8, 8000);
    tl.at(10600, () => {
      [3, 4].forEach(i => cells[i].classList.add('ac'));
      msg.innerHTML = '代表分最高的两组是 <em>组 3</em>（1.8790）与 <em>组 4</em>（1.8731）'
        + ' &nbsp;<span class="cm">// 八组分数非常接近，这正是负载均衡想要的样子</span>';
    });
    tl.at(13200, () => {
      cells.forEach(c => c.classList.remove('ac'));
      msg.innerHTML = '<span class="cm">// ★ 但默认 n_group=1：只有 1 个组，(N,288) 被压成 (N,1) —— 什么也没筛掉</span>';
    });
  },
},

/* ------------------------------------------------- 5 第二级筛选 */
{
  kicker: '第三步 · 第二级筛选',
  title: '没被选中的组，整个置成 <span class="hl-c">-inf</span>',
  sub: '组级决策在这里被翻译成专家级掩码。用 -inf 而不是 0：0 仍可能被 topk 选中（若整组为负），'
     + '-inf 在数值上保证不可能。',
  caption: '这一步之后，"选 8 个"才真正带上了分组约束。',
  lang: 'python',
  codeStart: 169,
  code: `        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)`,
  codeNote: '取 [1] 是只要下标；sorted=False 因为排序对 mask 没有意义，而它在热路径上。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap10 vizgrow hstart', style: 'width:100%' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '288 个专家 = 8 组 × 36 个（网格每 2 行 = 1 组）；实测选中组 3 与组 4' }));

    const tg = W.tensor(16, 18, () => -1, { cell: 13, gap: 2 });
    tg.style.width = (18 * 13 + 17 * 2) + 'px';
    tg.style.margin = '0 auto';
    const grpOf = i => Math.floor(i / 36);
    const isWin = i => M.gwin.indexOf(grpOf(i)) >= 0;
    tg.cells.forEach((c, i) => {
      c.style.transition = 'background .4s,opacity .4s,box-shadow .4s';
      c.style.background = 'rgba(150,180,255,.10)';
      c.style.borderColor = 'rgba(150,180,255,.13)';
      c.style.opacity = '.35';
    });
    viz.appendChild(tg);

    const legend = U.el('div', { class: 'row gap10 wrap center', style: 'width:100%' });
    legend.innerHTML = U.chip('组 3 · 36 个专家', 0) + U.chip('组 4 · 36 个专家', 4)
      + U.chip('其余 6 组 · 216 个 → -inf', 3)
      + '<span class="dim mono" style="font-size:11px">候选池 288 → 72</span>';
    viz.appendChild(legend);

    const idxRow = U.el('div', { class: 'row gap6 wrap center', style: 'width:100%' });
    const idxChips = M.gidx.map(v => {
      const e = U.el('span', { class: 'chip c4', style: 'height:19px;font-size:10.5px', text: 'e' + v });
      e.style.opacity = '.3';
      idxRow.appendChild(e); return e;
    });
    viz.appendChild(U.el('div', { class: 'col gap5', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: '最终入选的 8 个专家（实测 topk_indices，无序）' }), idxRow,
    ]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 288 个候选，全部还活着</span>';
    tl.at(1500, () => {
      tg.cells.forEach((c, i) => {
        if (!isWin(i)) return;
        c.style.background = 'rgba(' + (grpOf(i) === 3 ? '56,189,248' : '52,211,153') + ',.34)';
        c.style.borderColor = grpOf(i) === 3 ? 'rgba(56,189,248,.7)' : 'rgba(52,211,153,.7)';
        c.style.opacity = '1';
      });
      msg.innerHTML = 'group_mask 把 <em>组 3</em> 与 <em>组 4</em> 标成 1 '
        + '<span class="cm">// scatter_ 而不是 ==，因为 topk 不保证下标唯一</span>';
    });
    tl.at(4500, () => {
      tg.cells.forEach((c, i) => {
        if (isWin(i)) return;
        c.style.background = 'rgba(251,113,133,.09)';
        c.style.borderColor = 'rgba(251,113,133,.22)';
        c.style.opacity = '.42';
      });
      msg.innerHTML = '<em>216</em> 个专家被置成 -inf &nbsp;<span class="cm">// 候选池 288 → 72</span>';
    });
    tl.at(7500, () => {
      msg.innerHTML = 'score_mask 是 (N, n_group) —— <span class="cm">// 先 unsqueeze(-1).expand(每组 36).reshape(288) 才变成专家级掩码</span>';
    });
    tl.at(10500, () => {
      idxChips.forEach((e, k) => {
        e.style.transition = 'opacity .3s'; e.style.transitionDelay = (k * 90) + 'ms';
        e.style.opacity = '1';
      });
      msg.innerHTML = '最后 topk(8) 就在这 <em>72</em> 个里挑 &nbsp;<span class="cm">// 实测 8 个全部落在组 3、组 4</span>';
    });
    tl.at(13800, () => {
      msg.innerHTML = '<span class="cm">// 用 -inf 而不是 0：0 仍可能被选中（若整组为负），-inf 保证不可能</span>';
    });
  },
},

/* ------------------------------------------------- 6 ★ 三种取值来源 */
{
  kicker: '★ 落点 · 选择 vs 计分',
  title: '选择用<span class="hl-a">含偏置</span>的分，权重用<span class="hl-b">无偏</span>的分',
  sub: 'topk_indices 从 scores_for_choice 来，topk_weights 却从 scores gather ——'
     + '这是整课最容易被写错的一行。',
  caption: 'L179 是这一课最贵的一行：选是一回事，计分是另一回事。',
  lang: 'python',
  codeStart: 178,
  code: `        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return router_logits, topk_weights, topk_indices`,
  codeNote: '第 1 行与第 2 行取自两个不同的变量 —— 这就是全部要点。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);
    SHELL.hotLines([]);

    const viz = U.el('div', { class: 'col gap10 vizgrow hstart', style: 'width:100%' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '两条并行的流水线：一条决定「选谁」，一条决定「算多少」' }));
    const flA = W.flow([
      { t: 'scores_for_choice', s: '= scores + bias', cc: 0 },
      { t: 'topk(8, sorted=False)', s: '第一级掩码已生效', cc: 0 },
      { t: 'topk_indices', s: '(N, 8) 无序', cc: 0 },
    ], { style: 'width:100%' });
    const flB = W.flow([
      { t: 'scores', s: '无偏', cc: 1 },
      { t: 'gather(1, indices)', s: '按位取值', cc: 1 },
      { t: '÷ sum(+1e-20)', s: 'norm_topk_prob', cc: 1 },
      { t: '× 2.5', s: 'routed_scaling_factor', cc: 1 },
      { t: 'topk_weights', s: '和 = 2.5', cc: 1 },
    ], { style: 'width:100%' });
    viz.appendChild(U.el('div', { class: 'col gap6', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: '① 选择链（L178）—— 用带偏置的分' }), flA,
    ]));
    viz.appendChild(U.el('div', { class: 'col gap6', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: '② 计分链（L179-L183）—— 用不带偏置的分' }), flB,
    ]));

    const tb = W.table([
      ['topk_indices', '<span class="mono hl">' + M.idx.join(' · ') + '</span>'],
      ['topk_weights', '<span class="mono hl2">' + M.w.map(v => v.toFixed(6)).join(' · ') + '</span>'],
      ['sum', '<span class="hl4">2.500000</span> &nbsp;<span class="cm">= routed_scaling_factor</span>'],
    ], { head: ['实测输出（288 专家 / 默认配置）', '值'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 1, style: 'padding:9px 12px' }, [tb]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    flA.focus(-1); flB.focus(-1);
    msg.innerHTML = '<span class="cm">// 两条链共享同一个 topk_indices，但取值来源不同</span>';
    tl.at(1800, () => {
      flA.focus(0); SHELL.hotLines([1]);
      msg.innerHTML = '选择看的是 <em>scores_for_choice</em>（含 e_score_correction_bias）'
        + ' &nbsp;<span class="cm">// L178</span>';
    });
    tl.at(4800, () => {
      flA.focus(2); SHELL.hotLines([1, 2]);
      msg.innerHTML = '得到 <em>8 个下标</em>，而且是无序的 &nbsp;<span class="cm">// sorted=False：'
        + '下游不依赖顺序，排序是白花的代价</span>';
    });
    tl.at(7800, () => {
      flB.focus(1); SHELL.hotLines([2]);
      msg.innerHTML = '★ 权重却从 <em>scores</em> 里 gather &nbsp;<span class="cm">// L179：'
        + '偏置只当选拔委员，不当计分员</span>';
    });
    tl.at(10800, () => {
      flB.focus(3); SHELL.hotLines([3, 4, 5]);
      msg.innerHTML = '只在<em>选中的 8 个</em>里归一化，再乘 <em>2.5</em> &nbsp;'
        + '<span class="cm">// 权重和是一个与输入无关的常数</span>';
    });
    tl.at(13800, () => {
      flB.focus(0); SHELL.hotLines([6]);
      msg.innerHTML = '<span class="cm">// 返回值三元组：router_logits 也交出来了 —— 训练时算辅助损失用</span>';
    });
  },
},

/* ------------------------------------------------- 7 ★ 手算 */
{
  kicker: '★ 手算 · 最小例子',
  title: '手算一次 <span class="hl-a">top-8</span> 选择',
  sub: '真机 288 个专家算不动。缩到 8 个专家 / 2 组（每组 4 个）/ topk_group=1 / top_k=2，'
     + '用同一份代码跑出每一步 —— 拿计算器就能核对。',
  caption: '注意第 ⑦ 步：e1 的分数比 e5 低吗？不 —— 它更高，但它的组输了，连参选资格都没有。',
  lang: 'python',
  codeStart: 164,
  code: `        group_scores = (
            scores_for_choice.view(-1, self.num_group, self.num_experts // self.num_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.num_group, self.num_experts // self.num_group)
            .reshape(-1, self.num_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return router_logits, topk_weights, topk_indices`,
  codeNote: '手算用到的全部代码就是这个函数的第 ⑤ 步到第 ⑨ 步，一行不少。',
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap10', style: 'width:100%;height:100%' });
    root.appendChild(wrap);
    SHELL.hotLines([]);

    const viz = U.el('div', { class: 'col gap10 vizgrow hstart', style: 'width:100%' });
    wrap.appendChild(viz);

    /* 两组对决 */
    const duel = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const gA = W.card({ cc: 3, style: 'flex:1;min-width:0', title: '组 0 = e0..e3',
      body: U.el('div', { class: 'col gap4' }, [
        U.el('div', { class: 'mono', style: 'font-size:11.5px', text: '0.880797(e1) + 0.817574(e3)' }),
        U.el('div', { class: 'n', style: 'font-size:19px', text: '= 1.698371' }),
        U.el('div', { class: 'hlbad', style: 'font:600 11px var(--mono)', text: '输 —— 全组被置 -inf' }),
      ]) });
    const gB = W.card({ cc: 4, style: 'flex:1;min-width:0', title: '组 1 = e4..e7',
      body: U.el('div', { class: 'col gap4' }, [
        U.el('div', { class: 'mono', style: 'font-size:11.5px', text: '0.952574(e4) + 0.924142(e5)' }),
        U.el('div', { class: 'n', style: 'font-size:19px', text: '= 1.876716' }),
        U.el('div', { class: 'hl4', style: 'font:600 11px var(--mono)', text: '赢 —— 成为唯一候选池' }),
      ]) });
    duel.appendChild(gA); duel.appendChild(gB);
    viz.appendChild(duel);

    /* 逐专家表 */
    const rows = M.tiny.sig.map((s, i) => {
      const win = i >= 4;
      const res = (i === 4 || i === 5)
        ? '<span class="hl4">' + M.tiny.final[i - 4].toFixed(6) + '</span>' : '<span class="faint">·</span>';
      return ['e' + i, M.tiny.logit[i].toFixed(1), s.toFixed(6),
        win ? '<span class="mono">' + s.toFixed(6) + '</span>'
            : '<span class="hlbad mono">-inf</span>', res];
    });
    const tb = W.table(rows, { head: ['专家', 'logit', 'sigmoid', '掩码后候选', '最终权重'] });
    const tbCard = U.el('div', { class: 'card', cc: 4, style: 'padding:9px 12px' }, [tb]);
    viz.appendChild(tbCard);

    const steps = U.el('div', { class: 'row gap6 wrap center', style: 'width:100%' });
    const stepNames = ['① 打分', '② 组得分', '③ 选组', '④ 掩码', '⑤ top-2', '⑥ 归一化 ×2.5'];
    const stepEls = stepNames.map(t => {
      const e = U.el('span', { class: 'chip', style: 'height:20px;font-size:10.5px', text: t });
      steps.appendChild(e); return e;
    });
    viz.appendChild(steps);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    const light = k => stepEls.forEach((e, i) => {
      e.className = 'chip' + (i === k ? ' c4' : '');
      e.style.opacity = (i === k) ? '1' : '.4';
    });

    light(0);
    msg.innerHTML = 'h = [1, 1, 0, 0] &nbsp;<span class="cm">// logits = W·h = [1.0, 2.0, -1.0, 1.5, 3.0, 2.5, 2.0, -2.0]</span>';
    tl.at(1400, () => {
      light(0); SHELL.hotLines([]);
      msg.innerHTML = '① <em>sigmoid</em>：例 sigmoid(2.5) = 1/(1+e<sup>-2.5</sup>) = <em>0.924142</em> '
        + '<span class="cm">// 8 个分数互不相干</span>';
    });
    tl.at(3800, () => {
      light(1); gA.classList.add('ac'); gB.classList.add('ac'); SHELL.hotLines([1, 2, 3, 4]);
      msg.innerHTML = '② 每组取前二求和：组 0 = 0.880797+0.817574 = <em>1.698371</em>，'
        + '组 1 = 0.952574+0.924142 = <em>1.876716</em>';
    });
    tl.at(6200, () => {
      light(2); gA.classList.remove('ac'); SHELL.hotLines([5, 6, 7]);
      msg.innerHTML = '③ <em>topk_group=1</em> → 只有组 1 胜出 &nbsp;<span class="cm">// 组 0 的四行马上全变 -inf</span>';
    });
    tl.at(8600, () => {
      light(3); SHELL.hotLines([8, 9, 10, 11, 12]);
      msg.innerHTML = '④ 掩码展开成专家级：score_mask = [0,0,0,0,1,1,1,1] '
        + '<span class="cm">// 组级决策 → 专家级掩码的唯一桥梁</span>';
    });
    tl.at(11000, () => {
      light(4); SHELL.hotLines([13, 14]);
      msg.innerHTML = '⑤ 在剩下的 4 个里取 top-2 → <em>indices = [4, 5]</em>（无序），'
        + 'raw = <em>[0.952574, 0.924142]</em> <span class="cm">// 从无偏的 scores gather</span>';
    });
    tl.at(13600, () => {
      light(5); SHELL.hotLines([15, 16, 17, 18]);
      msg.innerHTML = '⑥ 归一化：÷(0.952574+0.924142) → <em>[0.507575, 0.492425]</em>，'
        + '再 ×2.5 → <em>[1.268938, 1.231063]</em>，和 = <em>2.5</em>';
    });
    tl.at(16600, () => {
      light(-1);
      msg.innerHTML = '★ 关键：<em>e1</em> 的分数（0.880797）比 e2 高得多，'
        + '但它的组输了 —— <span class="cm">分组筛选是硬约束，不是加分项</span>';
    });
  },
},

/* ------------------------------------------------- 8 数值开关与因子 */
{
  kicker: '数值 · 开关与因子',
  title: '<span class="hl-a">norm_topk_prob</span> 与 <span class="hl-b">routed_scaling_factor</span>',
  sub: '开关决定要不要在选中的 8 个里重新归一化；因子决定归一化之后乘多少。'
     + '两者相乘的结果是：权重和恒为 2.5。',
  caption: '权重和是常数，意味着下游专家输出的尺度可预测 —— 对训练是稳定性，对推理是可预期的数值范围。',
  lang: 'python',
  codeStart: 180,
  code: `        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor`,
  codeNote: '同一组 logits，只改 norm_topk_prob —— 权重和从 2.5 变成 18.699324。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap10 vizgrow hstart', style: 'width:100%' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '同一组 logits，同一组选中专家，唯一变量是 norm_topk_prob' }));

    const mk = (flag, vals, cc, note) => {
      const bars = W.bars(M.idx.map((e, i) => ({
        label: 'e' + e, value: vals[i], cc: cc, valueText: vals[i].toFixed(6),
      })), { max: 2.5 });
      const c = W.card({ cc: cc, tint: cc, style: 'flex:1;min-width:0',
        title: 'norm_topk_prob = <span class="mono">' + flag + '</span>', sub: note, body: bars });
      c._bars = bars;
      return c;
    };
    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cT = mk('True', M.w, 4, '在选中的 8 个里重新归一化（默认）');
    const cF = mk('False', M.wNoNorm, 3, '直接拿 sigmoid 值当权重，不归一化');
    row.appendChild(cT); row.appendChild(cF);
    viz.appendChild(row);

    const stack = W.stack(M.idx.map((e, i) => ({ label: 'e' + e, value: M.w[i], cc: i })));
    viz.appendChild(U.el('div', { class: 'col gap5', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: 'True 的 8 个权重（相对比例）—— 堆叠条总长 = 权重和 = 2.5' }), stack,
    ]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    cT.style.opacity = '.3'; cF.style.opacity = '.3';
    stack.reveal(0);
    msg.innerHTML = '<span class="cm">// 先看默认：norm_topk_prob = True</span>';
    tl.at(1800, () => {
      cT.style.opacity = '1';
      cT._bars.fills.forEach((f, i) => { f.bf.style.width = (M.w[i] / 2.5 * 100).toFixed(1) + '%'; });
      stack.reveal(1);
      msg.innerHTML = 'True：权重 = raw ÷ sum(raw) × 2.5 &nbsp;→&nbsp; 和 = <em>2.500000</em> '
        + '<span class="cm">// 恒等于 routed_scaling_factor</span>';
    });
    tl.at(5000, () => {
      cF.style.opacity = '1';
      cF._bars.fills.forEach((f, i) => { f.bf.style.width = (M.wNoNorm[i] / 2.5 * 100).toFixed(1) + '%'; });
      msg.innerHTML = 'False：8 个 sigmoid 值直接相加 &nbsp;→&nbsp; 和 = <em>' + M.sumNoNorm.toFixed(6) + '</em> '
        + '<span class="cm">// 与输入相关，不可预测</span>';
    });
    tl.at(8200, () => {
      msg.innerHTML = '同一个 token，权重尺度差了 <em>7.48</em> 倍 &nbsp;'
        + '<span class="cm">// 18.699324 / 2.5 —— 下游专家输出的尺度也随之漂移</span>';
    });
    tl.at(11200, () => {
      msg.innerHTML = '<span class="cm">// L181 的 + 1e-20 是防零除：sigmoid 恒大于 0，理论上不会零除，'
        + '但这是 fp32 下免费的保险</span>';
    });
    tl.at(14000, () => {
      cT.classList.add('ac'); cF.classList.remove('ac');
      msg.innerHTML = '★ 结论：<em>norm_topk_prob=True</em> 让「8 条路径的权重和」成为一个'
        + '<em>与输入无关的常数</em> —— 这正是 2.5 这个数字的全部意义';
    });
  },
},

/* ------------------------------------------------- 9 收束 */
{
  kicker: '收束',
  title: '8 条路径去哪，以及<span class="hl-a">分组为什么能省通信</span>',
  sub: '专家侧按「命中的专家」组织循环，而不是按 token 逐个遍历 288 个专家；'
     + '分布式下，分组把 8 份激活限制在少数几张卡上。',
  caption: '下一课 L3-04：专家的权重怎么存，grouped GEMM 怎么用。',
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
  codeNote: '注意下划线：router_logits 在推理时算完就扔，只有训练/监控才捕获它。',
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap10', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap8 vizgrow hstart', style: 'width:100%' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '专家并行 EP=8：一个 token 的 8 份激活最多落在几张卡上？' }));
    const mesh = W.mesh([['ep', 1], ['card', 8]], { cell: 58 });
    mesh.style.width = (8 * 58 + 7 * 8) + 'px';
    mesh.style.margin = '0 auto';
    const devs = mesh.devs;
    devs.forEach((d, i) => {
      d.style.setProperty('--c', 'var(--c' + U.ci(i) + ')');
      d.style.transition = 'opacity .4s,transform .4s,box-shadow .4s';
    });
    viz.appendChild(mesh);

    const tb = W.table([
      ['① 打分', 'L161', 'F.linear → (N, 288)，fp32'],
      ['② 打分函数', 'L162', 'sigmoid（★ 不是 softmax）'],
      ['③ 第一级筛选', 'L164-L168', '每组前 2 之和 → (N, n_group)'],
      ['④ 掩码', 'L169-L177', '未选中的组 → -inf'],
      ['⑤ 第二级筛选', 'L178', 'topk(8, sorted=False)'],
      ['⑥ 权重', 'L179-L183', 'gather 自无偏 scores → 归一化 → ×2.5'],
      ['⑦ 消费', 'L201-L208', '专家循环 index_add_ + 共享专家'],
    ], { head: ['步骤', '代码行', '产出'] });
    const tbCard = U.el('div', { class: 'card', cc: 0, style: 'padding:9px 12px' }, [tb]);
    viz.appendChild(tbCard);

    viz.appendChild(W.exercise(
      '一个 token 的 <code class="inl">topk_weights</code> 是 '
      + '<code class="inl">[0.3068, 0.3197, 0.3076, 0.3129, 0.3152, 0.3193, 0.3128, 0.3057]</code>。'
      + '不看源码，你能说出它们的和是多少吗？又是哪两行代码决定的？',
      '和 = <b>2.500000</b>，正好等于 <code class="inl">routed_scaling_factor</code>。'
      + '决定它的两行是 <code class="inl">topk_weights /= denominator</code> 与 '
      + '<code class="inl">topk_weights = topk_weights * self.routed_scaling_factor</code>：'
      + '先在选中的 8 个里归一化（和 = 1），再乘 2.5。'
      + '<br>★ 所以只要 <code class="inl">norm_topk_prob=True</code>，'
      + '<b>权重和就与输入无关</b> —— 这是可预测性的来源。'));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 不分组时，8 个选中专家理论上可能落在 8 张卡上</span>';
    tl.at(1800, () => {
      devs.forEach(d => { d.classList.add('off'); d.classList.remove('ac'); });
      devs[3].classList.remove('off'); devs[4].classList.remove('off');
      devs[3].classList.add('ac'); devs[4].classList.add('ac');
      msg.innerHTML = '限定 <em>topk_group=2</em>：8 个专家必然落在 <em>≤2</em> 个组里 '
        + '&nbsp;<span class="cm">// 实测 8 个全部落在组 3、组 4</span>';
    });
    tl.at(5000, () => {
      msg.innerHTML = '通信扇出从 <em>≤8</em> 降到 <em>≤2</em> &nbsp;'
        + '<span class="cm">// 代价：可能选不到全局最优的 8 个（第 7 幕的 e1 就是这样被牺牲的）</span>';
    });
    tl.at(8000, () => {
      tbCard.classList.add('ac');
      msg.innerHTML = '<span class="cm">// 全课七步：打分 → 两级筛选 → 权重 → 消费</span>';
    });
    tl.at(11000, () => {
      msg.innerHTML = '★ 但必须回到实测：默认 <em>n_group=1 / topk_group=1</em>，'
        + '<span class="cm">这段分组能力目前是"装好了但没通电"</span>';
    });
    tl.at(14000, () => {
      msg.innerHTML = '<em>288</em> 个分数 <span class="op">→</span> 两级筛选 '
        + '<span class="op">→</span> <em>8</em> 条路径 &nbsp;'
        + '<span class="cm">// 权重和恒为 2.5</span>';
    });
    tl.at(17000, () => {
      msg.innerHTML = '<span class="cm">// 下一课 L3-04：这 8 条路径在专家侧怎么变成一次 grouped GEMM</span>';
    });
  },
},

];
