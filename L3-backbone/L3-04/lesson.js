/* ==========================================================================
   L3-04 · 专家层与 grouped GEMM
   --------------------------------------------------------------------------
   覆盖：models/glm5_next/modeling_glm5_next.py（1 个文件 / 2444 行）
   目标：看完能说出打包存储省下了什么、swiglu_limit 截了什么，
        以及为什么专家权重必须按 (E, ...) 排布（grouped GEMM 的组轴 = EP 的切分轴）。
   排版基准：#visual 可视区 = 994 x 662.7 舞台像素（无头浏览器实测）。
   ========================================================================== */
'use strict';

/* 实测常量：来自 _data/recon/probe_l304.py 的输出，不是估算 */
const M4 = {
  experts: 288, topk: 8, inter: 2048, hidden: 4096, moeLayers: 42,
  gateUp: 16777216, down: 8388608, perExpert: 25165824,
  perLayer: 7247757312, allLayers: 304405807104,
  swiglu: 10.0, outCap: 99.995,
};

const SCENES = [

/* ------------------------------------------------- 1 全景：三块积木 */
{
  kicker: '第 3 层 · 文本主干',
  title: '一个 MoE 模块里，装着<span class="hl-a">三块积木</span>',
  sub: '路由专家、路由器、共享专家。这一课只拆第一块 —— 但会发现它的权重长什么样，'
     + '直接决定了它能不能用 grouped GEMM。',
  caption: '回顾 L0-01：45 层里 42 层是 MoE，每层一份这样的装配。',
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
  codeNote: 'Glm5NextTextMoE.__init__ —— 装配，不做数学。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap14 vizgrow', style: 'width:100%' });
    wrap.appendChild(viz);

    /* ---- 三条支路 ---- */
    const fl = W.flow([
      { t: 'hidden_states', s: '(tokens, 4096)', cc: 0 },
      { t: 'gate 288→8', s: '路由器 · L3-03', cc: 1 },
      { t: 'experts ×288', s: '路由专家', cc: 2 },
      { t: 'shared ×1', s: '稠密 MLP', cc: 4 },
      { t: '相加', s: '(tokens, 4096)', cc: 3 },
    ], { style: 'width:100%' });
    viz.appendChild(U.el('div', { class: 'col gap8', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: 'Glm5NextTextMoE 的三个部件' }), fl,
    ]));

    /* ---- 四个实测数字 ---- */
    const stats = [
      { k: 'n_routed_experts', v: '288', s: '每层路由专家数', cc: 2 },
      { k: 'num_experts_per_tok', v: '8', s: '每 token 命中几个', cc: 1 },
      { k: 'moe_intermediate_size', v: '2048', s: '单专家中间维', cc: 0 },
      { k: 'n_shared_experts', v: '1', s: '= 1 个专家大小', cc: 4 },
    ];
    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const statEls = stats.map(st => {
      const c = W.card({ cc: st.cc, tint: st.cc, style: 'flex:1',
        title: '<span class="mono" style="font-size:10.5px">' + st.k + '</span>',
        body: U.el('div', { class: 'n big', text: st.v }),
        kids: U.el('div', { class: 'cs', text: st.s }) });
      row.appendChild(c); return c;
    });
    viz.appendChild(row);

    /* ---- 规模 ---- */
    const scale = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const scA = W.card({ cc: 5, title: '共享专家 = 1 个路由专家',
      sub: '实测参数量 25,165,824，与单个路由专家一模一样 —— 它就是"第 289 个不参加选拔的专家"。',
      style: 'flex:1' });
    const scB = W.card({ cc: 3, title: '42 层合计 304.41 B',
      sub: '一层 7,247,757,312 个参数；bf16 下 <span class="mono">gate_up_proj</span> 单独就要 9 GiB/卡。',
      style: 'flex:1' });
    scale.appendChild(scA); scale.appendChild(scB);
    viz.appendChild(scale);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    fl.focus(0);
    msg.innerHTML = '<span class="cm">// hidden_states 先被复制成两份：一份去路由，一份留给共享专家</span>';
    tl.at(1800, () => { fl.focus(1); msg.innerHTML = '路由器把 4096 维压成 <em>288</em> 个分数，选出 <em>8</em> 条路径'; });
    tl.at(4600, () => { fl.focus(2); statEls[0].classList.add('ac'); statEls[1].classList.add('ac');
      msg.innerHTML = '路由专家：<em>288</em> 选 <em>8</em>，只有被选中的专家才做计算'; });
    tl.at(7600, () => { fl.focus(3); statEls[0].classList.remove('ac'); statEls[1].classList.remove('ac');
      statEls[3].classList.add('ac'); msg.innerHTML = '共享专家：<em>每个 token 都走</em>，参数量正好等于 1 个专家'; });
    tl.at(10800, () => { fl.focus(4); statEls[3].classList.remove('ac');
      msg.innerHTML = '两条支路<em>逐元素相加</em>，回到 <span class="fn">(tokens, 4096)</span>'; });
    tl.at(13800, () => {
      fl.focus(-1); scA.classList.add('ac'); scB.classList.add('ac');
      msg.innerHTML = '<span class="cm">// 304 B 参数全在这 42 层里 —— 它们怎么存，是这一课的主题</span>';
    });
  },
},

/* ------------------------------------------------- 2 ★ 布局：两个 3D 参数 */
{
  kicker: '专家层 · 存储',
  title: '★ 不是 864 个 <span class="hl-b">nn.Linear</span>，而是 2 个 <span class="hl-a">3D 参数</span>',
  sub: '288 个专家 × 3 个投影 = 864 个权重矩阵。这里把它们压成两个张量，第 0 维是专家编号。',
  caption: '第 0 维永远是专家轴 —— 整课后面所有结论都从这一行形状出发。',
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
  codeNote: 'Glm5NextTextExperts.__init__ —— "Collection of expert weights stored as 3D tensors."',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow', style: 'width:100%' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '同一种东西的两种存法（示意：真实 E = 288）' }));

    /* ---- 一叠专家切片 ---- */
    const row = U.el('div', { class: 'row gap10', style: 'width:100%;align-items:center;justify-content:center' });
    const slab = (label, cc) => {
      const col = U.el('div', { class: 'col gap6 center', style: 'flex:none' });
      col.appendChild(U.el('div', { class: 'klabel', text: label }));
      const t = W.tensor(3, 4, () => cc, { cell: 24, gap: 3 });
      t.querySelectorAll('.tcell').forEach(c => { c.style.opacity = '.55'; });
      col.appendChild(t);
      return col;
    };
    const s0 = slab('专家 0', 0), s1 = slab('专家 1', 1);
    const dots = U.el('div', { class: 'mono dim', style: 'flex:none;font-size:20px', text: '⋯' });
    const s2 = slab('专家 287', 5);
    [s0, s1, dots, s2].forEach(e => row.appendChild(e));
    viz.appendChild(U.el('div', { class: 'card', cc: 2, style: 'padding:11px 13px' }, [
      row,
      U.el('div', { class: 'mono', style: 'font-size:11px;color:var(--ink-faint);text-align:center;margin-top:8px',
        html: '每个切片是一整个专家的权重 —— 它们<b>连续</b>躺在一个张量里' }),
    ]));

    /* ---- 两个形状 ---- */
    const shapes = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const shA = W.card({ cc: 0, tint: 0, style: 'flex:1',
      title: '<span class="mono" style="font-size:11px">gate_up_proj</span>',
      sub: '<span class="mono">(288, 4096, 4096)</span> —— 第 1 维是 <span class="mono">2 * 2048</span>，gate 与 up 打包在一起',
      body: U.el('div', { class: 'n sm', style: 'margin-top:6px', text: '16,777,216 参数/专家' }) });
    const shB = W.card({ cc: 4, tint: 4, style: 'flex:1',
      title: '<span class="mono" style="font-size:11px">down_proj</span>',
      sub: '<span class="mono">(288, 4096, 2048)</span> —— 输出回 4096 维，同样按专家排布',
      body: U.el('div', { class: 'n sm', style: 'margin-top:6px', text: '8,388,608 参数/专家' }) });
    shapes.appendChild(shA); shapes.appendChild(shB);
    viz.appendChild(shapes);

    const tb = W.table([
      ['单个专家', '<span class="hl">25,165,824</span>', '16,777,216 + 8,388,608'],
      ['一层（288 个）', '<span class="hl">7,247,757,312</span>', '7.248 B'],
      ['42 个 MoE 层', '<span class="hl">304,405,807,104</span>', '304.41 B'],
    ], { head: ['量', '实测值', '拆解'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 3, style: 'padding:10px 12px' }, [
      U.el('div', { class: 'klabel', text: '参数量（probe_l304.py 在 meta 设备上实测）' }), tb,
    ]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 两个 nn.Parameter，不是 864 个</span>';
    tl.at(2200, () => { s0.classList.add('ac');
      msg.innerHTML = '第 0 维 = <em>专家编号</em>：<span class="fn">gate_up_proj[e]</span> 就是一个普通权重矩阵'; });
    tl.at(5200, () => { s1.classList.add('ac'); shA.classList.add('ac');
      msg.innerHTML = '第 1 维 = <em>2 * intermediate_dim</em>：gate 与 up 的权重<b>拼在一起</b>（下一幕展开）'; });
    tl.at(8600, () => { s2.classList.add('ac'); shB.classList.add('ac');
      msg.innerHTML = '<span class="fn">down_proj</span> 同样是 3D —— 两个参数就够了，不需要 ModuleList'; });
    tl.at(12000, () => {
      [s0, s1, s2].forEach(e => e.classList.remove('ac'));
      msg.innerHTML = '<span class="cm">// 304.41 B 参数全部由这一行形状规定</span>';
    });
    tl.at(14800, () => {
      shA.classList.remove('ac'); shB.classList.remove('ac');
      msg.innerHTML = '对照稠密 MLP：三个独立 <span class="fn">nn.Linear</span>、三个 Parameter（见 source.md 第二节）';
    });
  },
},

/* ------------------------------------------------- 3 ★ 打包：一次 GEMM 顶两次 */
{
  kicker: '专家层 · 打包',
  title: '★ 打包省下的是<span class="hl-a">调用次数</span>，不是算力',
  sub: 'gate 和 up 共用同一个输入，于是它们的权重可以拼成一个矩阵：一次 GEMM 出两半，chunk 只是视图。',
  caption: '注意：FLOPs 一点没省 —— 省的是 kernel 启动次数与权重张量个数。',
  lang: 'python',
  codeStart: 132,
  code: `            top_k_pos, token_idx = torch.where(mask[expert_idx])
            current = self._apply_gate(F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx]))
            current = F.linear(current, self.down_proj[expert_idx]) * top_k_weights[token_idx, top_k_pos, None]`,
  codeNote: '每个专家恰好两次 F.linear：第一次的权重是打包过的。',
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow', style: 'width:100%' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '一个专家的 gate_up_proj[e]：(4096 + 4096, 4096)' }));

    /* ---- 打包条 ---- */
    const bar = U.el('div', { class: 'row', style: 'width:100%;height:74px;border-radius:10px;'
      + 'border:1px solid var(--panel-brd);overflow:hidden;background:rgba(150,180,255,.05)' });
    const halfG = U.el('div', { class: 'col center', style: 'flex:1;height:100%;background:rgba(56,189,248,.20);'
      + 'border-right:2px dashed rgba(255,255,255,.45);transition:opacity .4s',
      html: '<div class="mono" style="font-size:11.5px;color:#bae6fd">gate 权重 · 2048 行</div>' });
    const halfU = U.el('div', { class: 'col center', style: 'flex:1;height:100%;background:rgba(251,191,36,.18);transition:opacity .4s',
      html: '<div class="mono" style="font-size:11.5px;color:#fde68a">up 权重 · 2048 行</div>' });
    bar.appendChild(halfG); bar.appendChild(halfU);
    viz.appendChild(bar);

    /* ---- 两个口径 ---- */
    const cmp = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cA = W.card({ cc: 4, tint: 4, style: 'flex:1', title: '打包（本课）',
      sub: '<span class="mono">F.linear(x, gate_up_proj[e])</span> → <span class="mono">chunk(2, dim=-1)</span>'
         + ' —— 1 次 GEMM + 1 次零拷贝切片' });
    const cB = W.card({ cc: 3, tint: 3, style: 'flex:1', title: '不打包（教科书）',
      sub: '<span class="mono">F.linear(x, W_gate[e])</span> 与 <span class="mono">F.linear(x, W_up[e])</span>'
         + ' —— 同一个 x、同一份 FLOPs，却是 2 次 GEMM' });
    cmp.appendChild(cA); cmp.appendChild(cB);
    viz.appendChild(cmp);

    /* ---- 调用次数 ---- */
    const bars = W.bars([
      { label: '打包 · 每个专家', value: 10, cc: 4, valueText: '10 次' },
      { label: '不打包 · 每个专家', value: 15, cc: 3, valueText: '15 次' },
    ], { max: 15 });
    viz.appendChild(U.el('div', { class: 'card', cc: 0, style: 'padding:10px 12px' }, [
      U.el('div', { class: 'klabel', text: '命中 5 个专家时的 GEMM 调用次数（探针实测计数）' }), bars,
    ]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    halfG.style.opacity = '.25'; halfU.style.opacity = '.25';
    bars.setAll(0);
    msg.innerHTML = '<span class="cm">// 两半权重本来就是同时被用到的</span>';
    tl.at(2000, () => { halfG.style.opacity = '1'; halfU.style.opacity = '1';
      msg.innerHTML = '把两半拼成 <span class="fn">(2 * intermediate_dim, hidden)</span>，一次 GEMM 出两个结果'; });
    tl.at(5200, () => { cA.classList.add('ac');
      msg.innerHTML = '<span class="fn">chunk(2, dim=-1)</span> 是<em>视图</em>：与源张量共享同一块 storage，零拷贝、零显存'; });
    tl.at(8400, () => { cA.classList.remove('ac'); cB.classList.add('ac'); bars.setAll(1);
      msg.innerHTML = '不打包的话：命中 5 个专家要多 <em>5</em> 次调用；288 个专家全命中就是 <em>576 vs 864</em>'; });
    tl.at(11800, () => { cB.classList.remove('ac');
      msg.innerHTML = '<span class="cm">// 数值完全一致（allclose=True）—— 省的是调用次数与 Parameter 个数</span>'; });
    tl.at(14000, () => {
      msg.innerHTML = '顺带省下第二个东西：EP 切分、量化、加载都只需处理 <em>一个</em> 参数对象';
    });
  },
},

/* ------------------------------------------------- 4 逐专家前向 */
{
  kicker: '专家层 · eager 实现',
  title: '先摊开成 mask，再<span class="hl-a">按专家</span>循环',
  sub: '默认实现不追求吞吐，它把"谁该算"写成一张 one-hot 表，然后只遍历真正被命中的专家。',
  caption: '哨兵那一列是留给 EP 的：非本卡的槽位会被写成 num_local_experts，正好落进最后一类。',
  lang: 'python',
  codeStart: 124,
  code: `        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            current = self._apply_gate(F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx]))
            current = F.linear(current, self.down_proj[expert_idx]) * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final`,
  codeNote: 'Glm5NextTextExperts.forward —— 循环次数 = 命中专家数，不是 288。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart', style: 'width:100%' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: 'mask（one_hot 后 permute(2,1,0)）· 行 = 专家，列 = token' }));

    /* 5 个 token、每 token 2 个专家（探针里的真实索引） */
    const IDX = [[0, 3], [1, 3], [5, 0], [2, 2], [4, 1]];
    const sel = (e, t) => IDX[t].indexOf(e) >= 0;

    const grid = U.el('div', { class: 'row gap10', style: 'width:100%;justify-content:center' });
    const labels = U.el('div', { class: 'col gap3', style: 'flex:none' });
    for (let r = 0; r < 7; r++) {
      labels.appendChild(U.el('div', {
        class: 'mono', style: 'height:24px;line-height:24px;font-size:10px;text-align:right;width:52px;color:'
          + (r === 6 ? 'var(--bad)' : 'var(--ink-faint)'),
        text: r === 6 ? '哨兵 6' : ('专家 ' + r) }));
    }
    grid.appendChild(labels);
    const g = W.tensor(7, 5, (r, c) => (sel(r, c) ? r : -1), { cell: 24, gap: 3 });
    g.querySelectorAll('.tcell').forEach(c => { c.style.fontSize = '11px'; });
    g.cells.forEach((c, k) => { c.textContent = sel(Math.floor(k / 5), k % 5) ? '●' : ''; });
    grid.appendChild(g);
    const marks = U.el('div', { class: 'col gap3', style: 'flex:none;margin-left:4px' });
    for (let r = 0; r < 7; r++) {
      marks.appendChild(U.el('div', { class: 'mono', style: 'height:24px;line-height:24px;font-size:10px;color:var(--ink-faint)',
        text: r === 6 ? '永远不会被命中' : (r === 2 ? '← 被 2 个 token 选中' : '') }));
    }
    grid.appendChild(marks);
    viz.appendChild(U.el('div', { class: 'card', cc: 1, style: 'padding:11px 13px' }, [grid]));

    /* index_add_ */
    const acc = U.el('div', { class: 'row gap8', style: 'width:100%;justify-content:center' });
    const accEls = [];
    for (let t = 0; t < 5; t++) {
      const e = U.el('div', { class: 'card', cc: 5, style: 'flex:1;padding:8px 6px;text-align:center;transition:opacity .4s,transform .4s' });
      e.innerHTML = '<div class="mono" style="font-size:10px;color:var(--ink-faint)">token ' + t + '</div>'
        + '<div class="n sm" style="margin-top:4px">' + IDX[t].join(' + ') + '</div>';
      acc.appendChild(e); accEls.push(e);
    }
    viz.appendChild(U.el('div', { class: 'col gap6', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: 'index_add_：8 条路径往同一行累加' }), acc,
    ]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// num_classes = num_experts + 1 —— 多出来的那一类是哨兵</span>';
    tl.at(2200, () => { msg.innerHTML = '<span class="fn">one_hot</span> 之后每行是一个专家、每列是一个 token，命中位置为 1'; });
    tl.at(5400, () => {
      g.querySelectorAll('.tcell').forEach(c => { c.style.boxShadow = ''; });
      msg.innerHTML = '<span class="fn">hit</span> 只保留"被选中过"的专家 → 循环次数 <em>5 vs 288</em>；哨兵行被 <span class="fn">continue</span> 跳过';
    });
    tl.at(8800, () => {
      marks.children[6].style.color = 'var(--bad)';
      msg.innerHTML = '实测：把索引写成 6（= num_experts）时，该 token 的输出<em>全 0</em> —— 哨兵真的被丢掉了';
    });
    tl.at(12200, () => {
      msg.innerHTML = '<span class="fn">torch.where(mask[e])</span> 同时给出 (top_k 位置, token 下标)，供两次 GEMM 使用';
    });
    tl.at(15500, () => {
      accEls.forEach(e => e.classList.add('ac'));
      msg.innerHTML = '<span class="cm">// 一个 token 的 8 个专家各算各的，最后累加到同一行</span>';
    });
  },
},

/* ------------------------------------------------- 5 ★ (E, out, in) 与 grouped GEMM */
{
  kicker: '专家层 · 布局即接口',
  title: '★ 为什么必须是 <span class="hl-a">(E, …)</span>：分组靠第 0 维',
  sub: 'grouped GEMM 一次调用算完所有专家，分组边界由 offsets 给出 —— 而 offsets 的长度必须等于权重第 0 维的长度。',
  caption: '模型文件里看不到 grouped_mm：调用在 integrations/moe.py（L8-04），但能不能用由这里的布局决定。',
  lang: 'python',
  codeStart: 108,
  code: `@use_experts_implementation
class Glm5NextTextExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""`,
  codeNote: '@use_experts_implementation 是调度点：默认 None → 回落到类里的 eager 实现。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart', style: 'width:100%' });
    wrap.appendChild(viz);

    /* ---- 三个轴 ---- */
    viz.appendChild(U.el('div', { class: 'klabel', text: 'gate_up_proj 的三个轴，各有各的下游' }));
    const axes = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const ax = (cc, name, size, role) => W.card({ cc: cc, tint: cc, style: 'flex:1',
      title: '<span class="mono" style="font-size:11px">dim ' + name + '</span>',
      body: U.el('div', { class: 'n big', text: size }),
      kids: U.el('div', { class: 'cs', html: role }) });
    const a0 = ax(2, '0', 'E = 288', '专家轴 → <b>grouped GEMM 的组轴</b>、<b>EP 的切分轴</b>');
    const a1 = ax(0, '1', '2I = 4096', '输出轴 → gate 与 up 打包在一起');
    const a2 = ax(4, '2', 'H = 4096', '输入轴 → 转置只是 stride 变换');
    [a0, a1, a2].forEach(e => axes.appendChild(e));
    viz.appendChild(axes);

    /* ---- offsets 手算 ---- */
    const S = U.el('div', { class: 'row gap4', style: 'width:100%' });
    const owner = [0, 0, 2, 2, 2];
    const sEls = owner.map((e, i) => {
      const b = U.el('div', { class: 'n sm', style: 'flex:1;height:36px;border-radius:8px;border:1px solid var(--c'
        + (e === 0 ? 0 : 4) + ';background:rgba(150,180,255,.10);display:flex;align-items:center;justify-content:center;'
        + 'color:#fff;transition:opacity .4s,transform .4s',
        text: 'S' + i });
      S.appendChild(b); return b;
    });
    viz.appendChild(U.el('div', { class: 'card', cc: 1, style: 'padding:10px 12px' }, [
      U.el('div', { class: 'klabel', text: 'offsets 手算：3 个专家，5 个 (token, 专家) 对' }),
      S,
      U.el('div', { class: 'mono', style: 'font-size:11px;color:var(--ink-dim);margin-top:8px',
        html: '专家 0 吃 2 行、专家 1 吃 <b>0</b> 行、专家 2 吃 3 行 &nbsp;→&nbsp; '
            + 'offsets = <span class="hl">[2, 2, 5]</span>&nbsp;&nbsp;(int32，长度 = 专家数)' }),
    ]));

    /* ---- 实测报错 ---- */
    const errs = U.el('div', { class: 'row wrap gap8', style: 'width:100%' });
    [
      '专家轴不在第 0 维 → contraction dimension of mat_a and mat_b must match',
      'offsets 长度 ≠ 专家数 → matrix batch sizes have to match',
      'offsets 不是 int32 → Offsets have to be int32',
    ].forEach(t => {
      errs.appendChild(U.el('span', { class: 'pill bad', style: 'font-size:10px', text: t }));
    });
    viz.appendChild(U.el('div', { class: 'col gap6', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: '实测：违反布局约定时 grouped_mm 的报错原文' }), errs,
    ]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 一份布局，三套实现都要吃</span>';
    tl.at(2400, () => { a0.classList.add('ac');
      msg.innerHTML = 'dim 0 是<em>组轴</em>：grouped GEMM 靠它把 S 行输入切成 E 组'; });
    tl.at(5600, () => { a1.classList.add('ac');
      msg.innerHTML = 'dim 1 是输出轴：<em>打包</em>与<em>分组</em>是两个正交的维度约定'; });
    tl.at(8800, () => {
      a0.classList.remove('ac'); a1.classList.remove('ac'); a2.classList.add('ac');
      sEls.forEach(e => { e.style.opacity = '1'; e.style.transform = 'none'; });
      msg.innerHTML = 'dim 2 是输入轴：转置零拷贝（<span class="cm">transpose(-2,-1)</span> 与源张量共享 storage）';
    });
    tl.at(12200, () => {
      sEls[2].style.opacity = '.3'; sEls[3].style.opacity = '.3'; sEls[4].style.opacity = '.3';
      sEls[3].style.transform = 'translateY(4px)';
      msg.innerHTML = '<span class="fn">offsets=[2,2,5]</span> 里的 <em>2</em> 出现了两次 —— 中间那个专家这一轮是空的，照样合法';
    });
    tl.at(15200, () => {
      a2.classList.remove('ac');
      msg.innerHTML = '★ 如果一开始存成 864 个 nn.Linear，想用 grouped GEMM 就得先 stack 一遍：<em>布局就是接口</em>';
    });
  },
},

/* ------------------------------------------------- 6 swiglu_limit 的两次截断 */
{
  kicker: '专家层 · 数值',
  title: '两次截断，而且<span class="hl-a">不对称</span>',
  sub: 'swiglu_limit = 10.0：gate 只截上界，up 上下都截。于是单个专家的输出幅值被钉在 99.995。',
  caption: '回顾 L3-02：稠密 MLP 上是同一对 clamp，连注释都一样。',
  lang: 'python',
  codeStart: 140,
  code: `        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        # Simple swiglu instead of alpha
        return F.silu(gate) * up`,
  codeNote: 'gate 有天然下界（silu 的最小值是 −0.278465），up 没有。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart', style: 'width:100%' });
    wrap.appendChild(viz);

    /* ---- 两条数轴：红格 = 被剪掉的值 ---- */
    const axis = (label, walls, cc) => {
      const col = U.el('div', { class: 'col gap6', style: 'width:100%' });
      col.appendChild(U.el('div', { class: 'klabel', text: label }));
      const row = U.el('div', { class: 'row gap3', style: 'width:100%' });
      for (let v = -16; v <= 32; v += 4) {
        const cut = walls.some(w => v > w[0] && v <= w[1]);
        row.appendChild(U.el('div', {
          class: 'mono', style: 'flex:1;height:34px;border-radius:7px;display:flex;align-items:center;justify-content:center;'
            + 'font-size:10px;border:1px solid ' + (cut ? 'var(--bad)' : 'var(--c' + cc + ')') + ';'
            + 'background:' + (cut ? 'rgba(251,113,133,.16)' : 'rgba(150,180,255,.10)') + ';'
            + 'color:' + (cut ? 'var(--bad)' : '#fff'),
          text: (cut ? '✂ ' : '') + v }));
      }
      col.appendChild(row);
      return col;
    };
    viz.appendChild(axis('gate：只截上界 —— 超过 10 的值被剪掉', [[10, 40]], 0));
    viz.appendChild(axis('up：上下都截 —— 低于 −10、高于 10 都被剪掉', [[-40, -10], [10, 40]], 2));

    /* ---- 手算表 ---- */
    const tb = W.table([
      ['0.5', '0.5', '0.5', '0.5', '0.31123', '<span class="hl">0.156</span>'],
      ['10.0', '10.0', '10.0', '10.0', '9.99955', '<span class="hl">99.995</span>'],
      ['40.0', '40.0', '<span class="hlbad">10.0</span>', '<span class="hlbad">10.0</span>', '9.99955', '<span class="hl">99.995</span>'],
      ['−40.0', '−40.0', '−40.0', '<span class="hlbad">−10.0</span>', '−0.00000', '<span class="hl">0.000</span>'],
      ['2.0', '−13.0', '2.0', '<span class="hlbad">−10.0</span>', '1.76159', '<span class="hl">−17.616</span>'],
    ], { head: ['gate', 'up', "gate′", "up′", 'silu(gate′)', '输出'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 3, style: 'padding:10px 12px' }, [
      U.el('div', { class: 'klabel', text: '手算（实测，swiglu_limit = 10.0）' }), tb,
    ]));

    /* ---- 上界对照 ---- */
    const bars = W.bars([
      { label: '不截断 (40,40)', value: 1600, cc: 3, valueText: '1600' },
      { label: '截断后', value: 99.995, cc: 4, valueText: '99.995' },
    ], { max: 1600 });
    viz.appendChild(U.el('div', { class: 'card', cc: 0, style: 'padding:10px 12px' }, [
      U.el('div', { class: 'klabel', text: '单专家输出幅值：silu(10) × 10 = 99.995 是硬上界' }), bars,
    ]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    bars.setAll(0);
    msg.innerHTML = '<span class="cm">// 两行 clamp，但两边的规则不一样</span>';
    tl.at(2200, () => { bars.setAll(1);
      msg.innerHTML = '<span class="fn">gate</span> 只截上界：因为 <em>silu 有全局下界 −0.278465</em>（在 x ≈ −1.278 处）'; });
    tl.at(5600, () => {
      msg.innerHTML = '<span class="fn">up</span> 两侧都截：它是纯乘子，<em>没有自带下界</em>，不拦就会上下都炸'; });
    tl.at(9000, () => {
      msg.innerHTML = '读表第 3 行：<em>(40, 40) → (10, 10)</em>，输出从 <span class="hlbad">1600</span> 掉到 <span class="hl">99.995</span>'; });
    tl.at(12400, () => {
      msg.innerHTML = '读表第 4 行：<em>gate = −40 没有被截</em>（只截上界），但 silu(−40) ≈ 0，输出 0 —— 负方向本来就无害'; });
    tl.at(15000, () => {
      msg.innerHTML = '<span class="cm">// 注释 "Simple swiglu instead of alpha"：上一代带学习缩放，这一代直接用 silu(gate) * up</span>';
    });
  },
},

/* ------------------------------------------------- 7 EP：切的就是第 0 维 */
{
  kicker: '专家层 · 分布式',
  title: 'EP 切分：<span class="hl-a">Shard(0)</span> 切的就是专家那一维',
  sub: '切完之后每张卡拿到 (E/ep, 2I, H) —— 形状约定没变，只是 E 变小了，grouped GEMM 的代码一行都不用改。',
  caption: '展开见 L8-03（张量并行样式表）与 L8-04（专家并行与 MoE 集成）。',
  lang: 'python',
  codeStart: 92,
  code: `    base_model_ep_plan = {
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
    }`,
  codeNote: 'models/glm5_next/configuration_glm5_next.py 第 92–97 行（借来的一处证据，不计入本课 coverage）。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow', style: 'width:100%' });
    wrap.appendChild(viz);

    const top = U.el('div', { class: 'row gap16', style: 'width:100%;align-items:center;justify-content:center' });
    const mesh = W.mesh([['x', 2], ['y', 4]], { cell: 72, cell2: () => '36 专家' });
    top.appendChild(mesh);
    const card = W.card({ cc: 2, style: 'flex:1;max-width:430px', title: 'ep_router 做三件事',
      sub: '① 非本卡专家的分数清零<br>② 全局专家号 → 本地号（288 → 36 取模）'
         + '<br>③ 被丢弃的槽位写成哨兵 <span class="mono">num_local_experts</span>，正好落进 mask 的最后一类' });
    top.appendChild(card);
    viz.appendChild(U.el('div', { class: 'col gap8', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: 'EP = 8：每张卡持有 288 / 8 = 36 个专家（实测可整除）' }), top,
    ]));

    const tb = W.table([
      ['1', '288', '<span class="hl">9.00 GiB</span>'],
      ['8', '36', '<span class="hl">1.12 GiB</span>'],
      ['16', '18', '<span class="hl">0.56 GiB</span>'],
      ['32', '9', '<span class="hl">0.28 GiB</span>'],
    ], { head: ['ep_size', '每卡专家数', 'gate_up_proj 每卡（bf16）'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 1, style: 'padding:10px 12px' }, [
      U.el('div', { class: 'klabel', text: '实测：每卡专家数与显存（只算 gate_up_proj）' }), tb,
    ]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    mesh.devs.forEach(d => { d.style.opacity = '.35'; });
    msg.innerHTML = '<span class="cm">// 288 个专家摊到 8 张卡上</span>';
    tl.at(2400, () => { card.classList.add('ac');
      msg.innerHTML = '并行样式 <span class="fn">grouped_gemm</span> 的实测实现是 <span class="mono">MoEParamShard(Shard(0))</span> —— 切第 0 维'; });
    tl.at(5800, () => {
      mesh.devs.forEach(d => { d.style.opacity = '1'; });
      msg.innerHTML = '每卡仍是 <span class="fn">(E_local, 2I, H)</span>：<em>形状约定没变，只是 E 变小了</em>'; });
    tl.at(9200, () => {
      mesh.devs.forEach((d, i) => { d.style.opacity = (i % 4 === 0) ? '1' : '.3'; });
      msg.innerHTML = '于是同一份 grouped GEMM 代码不用改 —— 只是 <span class="fn">offsets</span> 的长度变成 36'; });
    tl.at(12600, () => {
      mesh.devs.forEach(d => { d.style.opacity = '1'; });
      msg.innerHTML = '代价在路由侧：<span class="fn">gate</span> 每卡都是全量的 288 个分数，只有本卡的 36 个专家能被真正执行'; });
    tl.at(15800, () => {
      msg.innerHTML = '<span class="cm">// 约束：288 % ep_size == 0；1 / 8 / 16 / 32 实测都能整除</span>';
    });
  },
},

/* ------------------------------------------------- 8 出口：加回共享专家 */
{
  kicker: '专家层 · 出口',
  title: '出口只有一行：<span class="hl-a">routed + shared</span>',
  sub: '共享专家用的是进来时的 residuals，不是路由输出 —— 两条支路并行，最后逐元素相加。',
  caption: '共享专家的装配细节与缩放口径在 L3-05。',
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
  codeNote: 'Glm5NextTextMoE.forward —— 注意 residuals 是入口处的副本。',
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow', style: 'width:100%' });
    wrap.appendChild(viz);

    const inBar = U.el('div', { class: 'card cc0', style: 'width:100%;text-align:center;padding:9px 12px' });
    inBar.innerHTML = '<div class="mono" style="font-size:11.5px;color:#bae6fd">hidden_states (tokens, 4096)</div>';
    viz.appendChild(inBar);

    const paths = U.el('div', { class: 'row gap16', style: 'width:100%;align-items:stretch;justify-content:center' });
    const pA = W.card({ cc: 2, tint: 2, style: 'flex:1', title: 'experts（288 选 8）',
      sub: '只算被选中的专家，输出乘 <span class="mono">topk_weights</span> 与 <span class="mono">routed_scaling_factor = 2.5</span>',
      body: U.el('div', { class: 'mono', style: 'font-size:11px;margin-top:6px;color:var(--c2)', text: '7.248 B 参数 / 层' }) });
    const pB = W.card({ cc: 4, tint: 4, style: 'flex:1', title: 'shared_experts（1 个稠密 MLP）',
      sub: '每个 token 都走，无路由、无权重缩放 —— 输出直接相加',
      body: U.el('div', { class: 'mono', style: 'font-size:11px;margin-top:6px;color:var(--c4)', text: '25,165,824 参数 / 层 = 1 个专家' }) });
    paths.appendChild(pA); paths.appendChild(pB);
    viz.appendChild(paths);

    const outBar = U.el('div', { class: 'card ccA', style: 'width:100%;text-align:center;padding:9px 12px;transition:opacity .45s,transform .45s' });
    outBar.innerHTML = '<div class="mono" style="font-size:11.5px;color:#c9f7ff">'
      + 'hidden_states + shared_experts(residuals) → (tokens, 4096)</div>';
    viz.appendChild(outBar);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    outBar.style.opacity = '0'; outBar.style.transform = 'scale(.94)';
    msg.innerHTML = '<span class="cm">// 入口处先把 hidden_states 留一份</span>';
    tl.at(2400, () => { pA.classList.add('ac');
      msg.innerHTML = '路由那条路：288 选 8，<em>只有被选中的专家算</em>，占 99.65% 的参数'; });
    tl.at(5800, () => { pA.classList.remove('ac'); pB.classList.add('ac');
      msg.innerHTML = '共享那条路：<em>每个 token 都算</em>，参数量恰好 1 个专家 —— 实测 out == routed + shared 为 True'; });
    tl.at(9200, () => {
      pB.classList.remove('ac'); outBar.style.opacity = '1'; outBar.style.transform = 'none';
      msg.innerHTML = '合并方式是<em>逐元素相加</em>，没有额外的混合权重 —— 不对称是设计的一部分'; });
    tl.at(12400, () => {
      msg.innerHTML = '<span class="cm">// 下一课 L3-05：共享专家为什么要独立存在</span>';
    });
  },
},

/* ------------------------------------------------- 9 收束 */
{
  kicker: '收束',
  title: '三个问题，三句话',
  sub: '能把这三句说完整，这一课的验收点就达到了。',
  caption: '下一课 L3-05：MoE 装配与共享专家。',
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
  codeNote: '回到最开始那两个 Parameter —— 现在你能说出每一维为什么是它。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const tb = W.table([
      ['打包省下了什么',
       '省掉每个专家一次 GEMM 调用（576 vs 864）与一个 Parameter 对象；<span class="hlbad">FLOPs 一点没省</span>'],
      ['swiglu_limit 截了什么',
       'gate 只截上界（silu 自带 −0.278 下界），up 两侧都截；输出上界 = silu(10) × 10 = <span class="hl">99.995</span>'],
      ['为什么是 (E, …)',
       'dim 0 既是 grouped GEMM 的组轴（offsets 长度 = E），也是 EP 的切分轴（Shard(0)）'],
    ], { head: ['问题', '答案'] });
    const tbCard = U.el('div', { class: 'card', cc: 0, style: 'padding:10px 12px' }, [
      U.el('div', { class: 'klabel', text: '本课的三条结论' }), tb,
    ]);
    wrap.appendChild(tbCard);

    const tb2 = W.table([
      ['n_routed_experts', '288', '每层路由专家'],
      ['num_experts_per_tok', '8', '每 token 命中'],
      ['单专家参数', '25,165,824', 'gate_up 16,777,216 + down 8,388,608'],
      ['一层 288 个专家', '7.248 B', '42 层合计 304.41 B'],
      ['swiglu_limit', '10.0', '输出上界 99.995'],
    ], { head: ['量', '实测值', '说明'] });
    const card2 = U.el('div', { class: 'card', cc: 4, style: 'padding:10px 12px' }, [
      U.el('div', { class: 'klabel', text: '本课用到的实测数字（probe_l304.py）' }), tb2,
    ]);
    wrap.appendChild(card2);

    wrap.appendChild(W.exercise(
      '把 <code class="inl">gate_up_proj</code> 改存成 <code class="inl">(2I, E, H)</code>（专家轴挪到中间），'
      + 'grouped GEMM 还能用吗？要改哪些地方？',
      '不能直接用。grouped_mm 的 <code class="inl">offsets</code> 长度必须等于权重第 0 维的长度'
      + '（实测报错 <code class="inl">matrix batch sizes have to match</code>），'
      + '而且 <code class="inl">transpose(-2,-1)</code> 得到的张量收缩维也会对不上'
      + '（实测报错 <code class="inl">contraction dimension of mat_a and mat_b must match</code>）。'
      + '<br>要改的话至少三处：<b>EP 切分</b>要改成 <code class="inl">Shard(1)</code>（每卡拿到的那一维变了）；'
      + '<b>ep_router</b> 的本地专家号映射要跟着换轴；'
      + '<b>eager 实现</b>里的 <code class="inl">gate_up_proj[expert_idx]</code> 也要改写法。'
      + '换句话说：这一维不是"存哪儿方便"，而是三套下游共同约定的接口。'));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 三个问题，逐个点亮</span>';
    tb.querySelectorAll('tr').forEach((tr, i) => { if (i) tr.style.opacity = '.3'; });
    tl.at(1800, () => { tb.querySelectorAll('tr')[1].style.opacity = '1';
      msg.innerHTML = '<em>打包</em>省下调用次数与参数对象，不省 FLOPs'; });
    tl.at(5200, () => { tb.querySelectorAll('tr')[1].style.opacity = '.3';
      tb.querySelectorAll('tr')[2].style.opacity = '1';
      msg.innerHTML = '<em>截断</em>是不对称的：gate 一处、up 两处，输出上界 99.995'; });
    tl.at(8600, () => { tb.querySelectorAll('tr')[2].style.opacity = '.3';
      tb.querySelectorAll('tr')[3].style.opacity = '1';
      msg.innerHTML = '<em>(E, …)</em> 同时满足 grouped GEMM 与 EP —— 布局就是接口'; });
    tl.at(12000, () => {
      tb.querySelectorAll('tr').forEach(tr => { tr.style.opacity = '1'; });
      card2.classList.add('ac');
      msg.innerHTML = '<span class="cm">// 上面每个数字都来自 probe_l304.py 的实测输出</span>';
    });
    tl.at(15400, () => {
      msg.innerHTML = '<span class="cm">// 下一课 L3-05：MoE 装配与共享专家</span>';
    });
  },
},

];
