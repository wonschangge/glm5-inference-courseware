/* ==========================================================================
   L0-03 · 层类型排布：34 层 KDA + 11 层 MLA
   --------------------------------------------------------------------------
   覆盖：models/glm5_next/configuration_glm5_next.py（1 个文件 / 321 行）
   目标：看完能凭 num_hidden_layers 写出 layer_types；能解释 indexer_types 的
         full / shared 各干什么、shared 省掉了什么；能说出 3:1 的成本结构；
         能说出 index_topk % index_kpool == 0 为什么必须存在。
   排版基准：#visual 可视区 = 994 x 662.7 舞台像素（无头浏览器实测）。
   ========================================================================== */
'use strict';

/* 实测常量：由 .venv/bin/python 实例化 Glm5NextTextConfig() 跑出，不是估算 */
const M = {
  layers: 45, kda: 34, mla: 11,
  dense: 3, sparse: 42,
  indexTopk: 2048, kpool: 16, pools: 128,
  indexHeadDim: 128, indexNHeads: 32,
  kdaHeads: 64, kdaHeadDim: 128,
  kdaState: 1048576,          /* 64 × 128 × 128：KDA 每层的常数递推状态 */
  kvLoraRank: 512,            /* MLA 每 token 每层缓存的潜变量宽度 */
  seq: 131072,                /* 128K 上下文 */
};
M.mlaPerLayer = M.kvLoraRank * M.seq;          /* 67,108,864 */
M.allMla = M.layers * M.mlaPerLayer;           /* 3,019,898,880 */
M.realMla = M.mla * M.mlaPerLayer;             /* 738,197,504 */
M.kdaTotal = M.kda * M.kdaState;               /* 35,651,584 */

const SCENES = [

/* ------------------------------------------------- 1 全景：45 格排班表 */
{
  kicker: '预备篇 · 排班表',
  title: '45 层，<span class="hl-a">34 + 11</span>：先看一张排班表',
  sub: 'config 里两张同长的字符串列表决定每一层跑什么。这一课把这两张表彻底拆开。',
  caption: '回顾 L0-01：主干循环里 causal_mask_mapping[self.config.layer_types[i]] 取的就是这张表。',
  lang: 'python',
  codeStart: 106,
  code: `    num_hidden_layers: int = 45
    num_attention_heads: int = 64
    num_key_value_heads: int = 64`,
  codeNote: '所有排班表的长度都等于 num_hidden_layers 这一行；64 头是 MLA 的前提（不靠减少 KV 头省显存）。',
  duration: 12000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '45 层的注意力排班（实测：34 层 KDA + 11 层 MLA + DSA）' }));

    /* 45 格条带：格子宽度 = (994 - 44×3) / 45 ≈ 19px */
    const strip = U.el('div', { class: 'row gap3', style: 'width:100%' });
    const cells = [];
    for (let i = 0; i < M.layers; i++) {
      const mla = (i % 4 === 3);
      const e = U.el('div', {
        class: 'n sm',
        style: 'flex:1 1 0;min-width:14px;height:56px;border-radius:8px;display:flex;'
             + 'align-items:center;justify-content:center;font-size:9.5px;'
             + 'border:1px solid ' + (mla ? 'var(--c1)' : 'rgba(56,189,248,.5)') + ';'
             + 'background:' + (mla ? 'rgba(167,139,250,.22)' : 'rgba(56,189,248,.11)') + ';'
             + 'color:' + (mla ? '#ddd6fe' : '#bae6fd') + ';'
             + 'transition:box-shadow .3s var(--ease-out),transform .3s var(--ease-out)',
        title: 'layer ' + i + ' → ' + (mla ? 'indexed_attention (MLA + DSA)' : 'linear_attention (KDA)'),
        text: String(i),
      });
      strip.appendChild(e); cells.push(e);
    }
    viz.appendChild(strip);

    const legend = U.el('div', { class: 'row gap12 wrap', style: 'width:100%' });
    legend.innerHTML =
      '<span class="chip c0"><i class="sw sw-c0"></i>linear_attention · KDA</span>' +
      '<span class="chip c1"><i class="sw sw-c1"></i>indexed_attention · MLA + DSA</span>' +
      '<span class="dim mono" style="margin-left:auto;font-size:11px">周期 4：3 层 KDA + 1 层 MLA</span>';
    viz.appendChild(legend);

    const stats = [
      { k: 'num_hidden_layers', v: M.layers, cc: 0, s: '主干层数' },
      { k: 'linear_attention', v: M.kda, cc: 0, s: 'KDA · 常数状态' },
      { k: 'indexed_attention', v: M.mla, cc: 1, s: 'MLA + DSA · 索引器' },
      { k: '周期', v: 4, cc: 2, s: 'idx % 4' },
    ];
    const statRow = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const statEls = stats.map(st => {
      const c = W.card({ cc: st.cc, tint: st.cc,
        title: '<span class="mono" style="font-size:10.5px">' + st.k + '</span>',
        body: U.el('div', { class: 'n big', text: U.fmt(st.v) }),
        sub: st.s, style: 'flex:1' });
      statRow.appendChild(c); return c;
    });
    viz.appendChild(statRow);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    const light = (list) => cells.forEach((c, k) => {
      const on = list.indexOf(k) >= 0;
      c.style.boxShadow = on ? '0 0 0 2px var(--accent), 0 0 16px -2px var(--accent)' : 'none';
      c.style.transform = on ? 'translateY(-4px)' : 'none';
    });

    msg.innerHTML = '<span class="cm">// 45 个格子 = 45 层，每格写着自己的注意力类型</span>';
    tl.at(1400, () => { light([0, 1, 2]); msg.innerHTML = 'KDA 层：<em>3 个一组</em> —— 走线性注意力，缓存是常数大小'; });
    tl.at(3800, () => { light([3]); msg.innerHTML = '第 <em>4</em> 层换成 <span class="hl2">MLA + DSA</span>：只有这种层才带索引器'; });
    tl.at(6200, () => { light([7, 11, 15]); msg.innerHTML = '周期重复：<em>3, 7, 11, 15 …</em> 每隔 4 层出现一个 MLA 层'; });
    tl.at(8600, () => {
      light([]); statEls.forEach(e => e.classList.add('ac'));
      msg.innerHTML = '数出来：<span class="hl">34</span> 层 KDA + <span class="hl2">11</span> 层 MLA = 45';
    });
    tl.at(10400, () => {
      statEls.forEach(e => e.classList.remove('ac'));
      msg.innerHTML = '<span class="cm">// 这 45 个字符串是算出来的，规则只有一行</span>';
    });
  },
},

/* ------------------------------------------------- 2 生成规律 */
{
  kicker: '第 ① 步 · 生成',
  title: '规律只有一条：<span class="hl-a">idx % 4 != 3</span>',
  sub: '先挑出 KDA 的层号集合，再给每一层贴标签 —— 两者互补，穷尽全部 45 层。',
  caption: '紧随其后还有一个兼容层：外部的 "full_attention" 会被改名为 "indexed_attention"。',
  lang: 'python',
  codeStart: 165,
  code: `        if self.layer_types is None:
            kda_layers = [idx for idx in range(self.num_hidden_layers) if idx % 4 != 3]
            self.layer_types = [
                "linear_attention" if layer_idx in kda_layers else "indexed_attention"
                for layer_idx in range(self.num_hidden_layers)
            ]
        self.layer_types = [
            "indexed_attention" if layer_type == "full_attention" else layer_type for layer_type in self.layer_types
        ]`,
  codeNote: '第二个赋值没有 if 保护：它对「生成出来的表」和「外部传进来的表」都执行，只负责统一旧命名。',
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '逐层推演：idx % 4 只有 4 种取值，只有第 4 种会掉出 KDA 集合' }));

    const rows = [
      ['0', '0', '✓', '<span class="hl">linear_attention</span>'],
      ['1', '1', '✓', '<span class="hl">linear_attention</span>'],
      ['2', '2', '✓', '<span class="hl">linear_attention</span>'],
      ['3', '<span class="hl3">3</span>', '<span class="hlbad">✗</span>', '<span class="hl2">indexed_attention</span>'],
      ['4', '0', '✓', '<span class="hl">linear_attention</span>'],
      ['5', '1', '✓', '<span class="hl">linear_attention</span>'],
      ['6', '2', '✓', '<span class="hl">linear_attention</span>'],
      ['7', '<span class="hl3">3</span>', '<span class="hlbad">✗</span>', '<span class="hl2">indexed_attention</span>'],
      ['…', '…', '…', '<span class="dim">… 每 4 层重复 …</span>'],
      ['43', '<span class="hl3">3</span>', '<span class="hlbad">✗</span>', '<span class="hl2">indexed_attention</span>'],
      ['44', '0', '✓', '<span class="hl">linear_attention</span>'],
    ];
    const tb = W.table(rows, { head: ['idx', 'idx % 4', 'idx % 4 != 3', 'layer_types[idx]'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 0, style: 'padding:8px 12px;width:100%' }, [tb]));
    const trs = Array.from(tb.querySelectorAll('tbody tr'));

    const cards = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const f1 = W.card({ cc: 1, tint: 1, title: 'MLA 层数 = 45 // 4',
      body: U.el('div', { class: 'n big', text: '11' }),
      sub: 'idx 3, 7, 11, … 43 —— 整除的副产品', style: 'flex:1' });
    const f2 = W.card({ cc: 0, tint: 0, title: 'KDA 层数 = 45 − 11',
      body: U.el('div', { class: 'n big', text: '34' }),
      sub: '剩下的层全部落在 KDA 集合里', style: 'flex:1' });
    cards.appendChild(f1); cards.appendChild(f2);
    viz.appendChild(cards);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    const pick = (k) => trs.forEach((tr, i) => tr.classList.toggle('ac', i === k));
    msg.innerHTML = '<span class="cm">// 第 0 层：0 % 4 = 0 ≠ 3 → 进入 kda_layers</span>';
    tl.at(1200, () => { pick(0); msg.innerHTML = 'idx <em>0</em>：<em>0 % 4 = 0</em> ≠ 3 → 属于 KDA 集合'; });
    tl.at(3200, () => { pick(3); msg.innerHTML = 'idx <em>3</em>：<em>3 % 4 = 3</em> → <span class="hlbad">掉出集合</span>，于是被判为 <span class="hl2">indexed_attention</span>'; });
    tl.at(5400, () => { pick(4); msg.innerHTML = 'idx <em>4</em>：取模回到 <em>0</em> —— 周期重新开始，又是 KDA'; });
    tl.at(7600, () => { pick(7); msg.innerHTML = 'idx <em>7</em>：又一次 <em>7 % 4 = 3</em> → 第二个 MLA 层'; });
    tl.at(9800, () => { pick(9); msg.innerHTML = 'idx <em>43</em>：<em>43 % 4 = 3</em> → 第 11 个、也是最后一个 MLA 层'; });
    tl.at(12000, () => {
      pick(10);
      msg.innerHTML = 'idx <em>44</em>：<em>44 % 4 = 0</em> → 尾巴那一层回到 KDA（45 不是 4 的倍数）';
    });
    tl.at(13800, () => {
      pick(-1); f1.classList.add('ac'); f2.classList.add('ac');
      msg.innerHTML = 'KDA <span class="hl">34</span> : MLA <span class="hl2">11</span> —— 换成 8 层就是 6 : 2，公式不变';
    });
  },
},

/* ------------------------------------------------- 3 ★ 成本结构 */
{
  kicker: '第 ② 步 · 为什么',
  title: '★ 为什么是 3:1：<span class="hl-c">增长的那部分只剩 11/45</span>',
  sub: '两种层的缓存形态完全不同：KDA 是常数状态，MLA 的 KV 随序列线性增长。',
  caption: '这里的关键数字都是实测维度推算出来的：64×128×128 与 kv_lora_rank = 512。',
  lang: 'python',
  codeStart: 143,
  code: `    linear_head_dim: int = 128
    linear_num_heads: int = 64`,
  codeNote: 'KDA 每层状态 = 64 × 128 × 128 = 1,048,576 个数（形状里没有序列维）；MLA 每 token 每层 = kv_lora_rank = 512 个数。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const kda = W.card({ cc: 0, tint: 0, title: 'KDA 层 · 常数状态',
      body: U.el('div', { class: 'mono', style: 'font-size:11.5px;line-height:1.7;color:var(--ink-dim)' }, [
        U.el('div', { html: '64 头 × 128 × 128 = <span class="hl">1,048,576</span> 个数/层' }),
        U.el('div', { html: '× 34 层 = <span class="hl">35,651,584</span> 个数' }),
        U.el('div', { html: '<span class="cm">// 与 seq 无关，长上下文不会把它变大</span>' }),
      ]), style: 'flex:1' });
    const mla = W.card({ cc: 1, tint: 1, title: 'MLA 层 · 线性增长',
      body: U.el('div', { class: 'mono', style: 'font-size:11.5px;line-height:1.7;color:var(--ink-dim)' }, [
        U.el('div', { html: '每 token 每层 = kv_lora_rank = <span class="hl2">512</span> 个数' }),
        U.el('div', { html: 'seq 2,048 → 1,048,576 个数 <span class="cm">（与 KDA 打平）</span>' }),
        U.el('div', { html: 'seq 128K → <span class="hl2">67,108,864</span> 个数/层' }),
      ]), style: 'flex:1' });
    row.appendChild(kda); row.appendChild(mla);
    viz.appendChild(row);

    /* 交点 */
    const cross = U.el('div', { class: 'formula', style: 'width:100%' });
    cross.innerHTML = '交点：1,048,576 / 512 = <em>2,048</em> 个 token &nbsp;'
      + '<span class="cm">// 上下文短于 2048 时，KDA 的常数状态反而更贵</span>';
    viz.appendChild(cross);

    /* 45 层的账（seq = 128K） */
    const bars = W.bars([
      { label: '假设全 MLA', value: M.allMla, cc: 1, valueText: U.fmt(M.allMla, { si: true }) },
      { label: '实际 11 层 MLA', value: M.realMla, cc: 1, valueText: U.fmt(M.realMla, { si: true }) },
      { label: 'KDA 常数 34 层', value: M.kdaTotal, cc: 0, valueText: U.fmt(M.kdaTotal, { si: true }) },
    ]);
    bars.setAll(0);
    viz.appendChild(U.el('div', { class: 'col gap6', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: 'seq = 131,072 时 45 层的缓存元素总数（实测维度推算）' }),
      bars,
    ]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    kda.style.opacity = '.34'; mla.style.opacity = '.34';
    msg.innerHTML = '<span class="cm">// 一张是平的，一张是斜的 —— 这就是 3:1 的全部理由</span>';
    tl.at(1500, () => { kda.style.opacity = '1'; msg.innerHTML = 'KDA：<em>常数</em>状态。64 头各持一个 128×128 的递推状态，与上下文长度无关'; });
    tl.at(4400, () => { mla.style.opacity = '1'; msg.innerHTML = 'MLA：每 token 每层只存 <em>512</em> 个数，但会<em>一直涨</em>'; });
    tl.at(7400, () => {
      cross.classList.add('ac');
      msg.innerHTML = '两者在 <em>2,048</em> 个 token 处打平 —— 超过之后，MLA 每多一个 token 就多付一份钱';
    });
    tl.at(10600, () => {
      cross.classList.remove('ac'); bars.setAll(1);
      msg.innerHTML = '128K 上下文：实际 MLA 部分 <em>0.74G</em> 个数，KDA 常数部分只有 <em>0.036G</em> 个数';
    });
    tl.at(13400, () => {
      msg.innerHTML = '全 MLA 需要 <em>3.02G</em> → 实际约 <em>1/3.9</em>，因为增长项只来自 <em>11/45 = 24.4%</em> 的层';
    });
    tl.at(15600, () => {
      msg.innerHTML = '<span class="cm">// 保留 11 层 MLA 是推理而非实测：KDA 压历史是有损的，精确检索需要 DSA</span>';
    });
  },
},

/* ------------------------------------------------- 4 indexer_types 是什么 */
{
  kicker: '第 ③ 步 · 索引器排班',
  title: '第二张表：默认 <span class="hl-a">45 层全 full</span>',
  sub: 'indexer_types 决定每层要不要算那 2048 个位置："full" 自己跑索引器，"shared" 复用上一层选好的。',
  caption: '两张表长度都是 45，但只有 indexed_attention 的层号上它们才真正相交。',
  lang: 'python',
  codeStart: 134,
  code: `    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 32
    head_dim: int = 0
    layer_types: list[str] | None = None
    # \`"full"\` runs the indexer, \`"shared"\` reuses the previous full layer's index mask.
    indexer_types: list[str] | None = None`,
  codeNote: '第 139 行的注释就是这张表的说明书；两个值都是 None，真正的表在 __post_init__ 里现算。',
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: 'indexer_types（实测默认）：45 个 full，MLA 层上才有作用' }));

    const strip = U.el('div', { class: 'row gap3', style: 'width:100%' });
    const cells = [];
    for (let i = 0; i < M.layers; i++) {
      const mla = (i % 4 === 3);
      const e = U.el('div', {
        style: 'flex:1 1 0;min-width:14px;height:50px;border-radius:8px;display:flex;flex-direction:column;'
             + 'align-items:center;justify-content:center;gap:3px;'
             + 'border:1px solid ' + (mla ? 'var(--c1)' : 'rgba(150,180,255,.18)') + ';'
             + 'background:' + (mla ? 'rgba(167,139,250,.18)' : 'rgba(150,180,255,.06)') + ';'
             + 'transition:box-shadow .3s var(--ease-out),opacity .3s',
        title: 'layer ' + i + ' → indexer_types[' + i + '] = full',
      }, [
        U.el('div', { class: 'n', style: 'font-size:9.5px;color:' + (mla ? '#ddd6fe' : 'var(--ink-faint)'), text: String(i) }),
        U.el('div', { class: 'mono', style: 'font-size:8px;color:var(--c4)', text: mla ? 'F' : '·' }),
      ]);
      strip.appendChild(e); cells.push(e);
    }
    viz.appendChild(strip);

    const cards = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const full = W.card({ cc: 4, tint: 4, title: 'full —— 跑索引器',
      sub: '用 32 头 × 128 维的小注意力给历史池打分，挑出 128 个池（= 2048 个位置），并把结果交给下一层。',
      style: 'flex:1' });
    const shared = W.card({ cc: 2, tint: 2, title: 'shared —— 复用上一层',
      sub: '不跑索引器，直接用"上一个 full 层"选好的位置；实现里连索引器子模块都不会被建出来。',
      style: 'flex:1' });
    cards.appendChild(full); cards.appendChild(shared);
    viz.appendChild(cards);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 45 个格子里写的都是 full —— 但只有带索引器的层才在乎</span>';
    tl.at(1400, () => {
      cells.forEach((c, k) => { c.style.opacity = (k % 4 === 3) ? '1' : '.3'; });
      msg.innerHTML = '只有 <span class="hl2">indexed_attention</span> 的 11 层真的有索引器；KDA 层那一项写了也没用';
    });
    tl.at(4200, () => {
      cells.forEach(c => { c.style.opacity = '1'; });
      full.classList.add('ac');
      msg.innerHTML = '<em>full</em>：这一层自己算 —— 索引器是一次小注意力（32 头 × 128 维），成本不小';
    });
    tl.at(7200, () => {
      full.classList.remove('ac'); shared.classList.add('ac');
      msg.innerHTML = '<em>shared</em>：直接用上一个 full 层的结果，省掉整个索引器前向';
    });
    tl.at(10200, () => {
      shared.classList.remove('ac');
      msg.innerHTML = '<span class="cm">// 默认全 full，是因为 freq 默认 1 —— 下一幕看那条判据</span>';
    });
    tl.at(12800, () => {
      msg.innerHTML = 'KDA 层没有索引器：它在 indexer_types 里的取值<em>永远不被读</em>';
    });
  },
},

/* ------------------------------------------------- 5 ★ shared 省掉了什么 */
{
  kicker: '第 ④ 步 · 复用',
  title: '★ <span class="hl-b">shared</span> 省掉了什么',
  sub: '省掉的不只是一次前向：索引器模块、它自己的缓存、以及那 2048 个位置的重复计算。',
  caption: '实测在缩小的孪生模型上完成（同样的代码路径，只是层数与维度变小）。',
  lang: 'python',
  codeStart: 41,
  code: `    layer_types (\`list[str]\`, *optional*):
        Per-layer attention cache schedule. Values are \`"linear_attention"\` for
        KDA layers and \`"indexed_attention"\` for MLA (DSA) layers.
    indexer_types (\`list[str]\`, *optional*):
        Per-layer DSA indexer mode. Values are \`"full"\` (run the indexer) or \`"shared"\`
        (reuse the previous full layer's top-k selection).`,
  codeNote: '注意措辞：reuse the previous full layer —— 复用的是"上一个 full 层"，不是"上一层"。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const prod = W.card({ cc: 4, tint: 4, title: 'full 层（生产者）',
      body: U.el('div', { class: 'mono', style: 'font-size:11.5px;line-height:1.75;color:var(--ink-dim)' }, [
        U.el('div', { html: '· 建索引器子模块（实测孪生：<span class="hl4">151,808</span> 个参数）' }),
        U.el('div', { html: '· 跑一次索引器前向，挑出 128 个池' }),
        U.el('div', { html: '· 自己的压缩键缓存被写入（<span class="hl4">indexer_keys</span>）' }),
        U.el('div', { html: '· next_skip_topk = True：把结果传给下一层' }),
      ]), style: 'flex:1' });
    const cons = W.card({ cc: 2, tint: 2, title: 'shared 层（消费者）',
      body: U.el('div', { class: 'mono', style: 'font-size:11.5px;line-height:1.75;color:var(--ink-dim)' }, [
        U.el('div', { html: '· <span class="hl3">indexer = None</span>：模块根本不实例化' }),
        U.el('div', { html: '· 没有索引器前向，也就没有打分与 top-k' }),
        U.el('div', { html: '· 索引器缓存保持未初始化（实测 <span class="hl3">None</span>）' }),
        U.el('div', { html: '· 直接复用上一层选好的 2048 个位置' }),
      ]), style: 'flex:1' });
    row.appendChild(prod); row.appendChild(cons);
    viz.appendChild(row);

    /* 相邻性约束 */
    const chain = U.el('div', { class: 'row gap8 center', style: 'width:100%' });
    chain.innerHTML =
      '<span class="chip c4">L3 · MLA + full</span><span class="arrow" style="max-width:40px"></span>' +
      '<span class="chip c0">L4 · KDA</span><span class="arrow" style="max-width:40px"></span>' +
      '<span class="chip c0">L5 · KDA</span><span class="arrow" style="max-width:40px"></span>' +
      '<span class="chip c0">L6 · KDA</span><span class="arrow" style="max-width:40px"></span>' +
      '<span class="chip c2">L7 · MLA + shared</span>';
    viz.appendChild(U.el('div', { class: 'col gap6', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: '默认排班下的传递链：KDA 层会把 topk_indices 重置为 None' }),
      U.el('div', { class: 'card', cc: 3, style: 'padding:9px 11px' }, [chain]),
    ]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    cons.style.opacity = '.3';
    msg.innerHTML = '<span class="cm">// 先看生产者：full 层一个前向里多做了什么</span>';
    tl.at(1500, () => { prod.classList.add('ac'); msg.innerHTML = 'full 层要多跑一次小注意力 + 维护自己的一份压缩键缓存'; });
    tl.at(4500, () => {
      prod.classList.remove('ac'); cons.style.opacity = '1'; cons.classList.add('ac');
      msg.innerHTML = 'shared 层把上面三件事<em>全省掉</em>：模块不建、前向不跑、缓存不写';
    });
    tl.at(7800, () => {
      chain.querySelectorAll('.chip')[4].style.boxShadow = '0 0 0 2px var(--bad)';
      chain.querySelectorAll('.chip')[0].style.boxShadow = '0 0 0 2px var(--ok)';
      msg.innerHTML = '但这份结果只传<em>一层</em>：L3 的选择在 L4 就被重置成 <span class="hlbad">None</span>';
    });
    tl.at(11200, () => {
      msg.innerHTML = '所以实测（8 层孪生、freq=4）会直接抛：<span class="hlbad">Shared DSA layers require top-k indices from a previous full indexer layer.</span>';
    });
    tl.at(14200, () => {
      msg.innerHTML = '<span class="cm">// 结论：默认 3:1 排班里 MLA 层永远相邻不了，所以默认只能全 full</span>';
    });
  },
},

/* ------------------------------------------------- 6 freq / offset */
{
  kicker: '第 ⑤ 步 · 默认从哪来',
  title: '判据：<span class="hl-a">max(i − offset + 1, 0) % freq == 0</span>',
  sub: 'index_topk_freq 默认 1 —— 任何整数取模 1 都是 0，于是 45 层全被判为 full。',
  caption: 'freq 与 offset 都不在 dataclass 字段里，是从 **kwargs 里取的"可调但非标准"参数。',
  lang: 'python',
  codeStart: 182,
  code: `            else:
                freq = max(kwargs.get("index_topk_freq", 1), 1)
                offset = kwargs.get("index_skip_topk_offset", 2)
                self.indexer_types = [
                    "full" if (max(i - offset + 1, 0) % freq) == 0 else "shared" for i in range(self.num_hidden_layers)
                ]`,
  codeNote: 'max(..., 0) 让开头几层不会因为减 offset 变成负数：i=0 时括号里是 0，仍然是 full。',
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const mkStrip = (label, isFull) => {
      const col = U.el('div', { class: 'col gap6', style: 'width:100%' });
      col.appendChild(U.el('div', { class: 'klabel', text: label }));
      const r = U.el('div', { class: 'row gap3', style: 'width:100%' });
      const cs = [];
      for (let i = 0; i < M.layers; i++) {
        const f = isFull(i);
        const mla = (i % 4 === 3);
        const e = U.el('div', {
          class: 'n',
          style: 'flex:1 1 0;min-width:14px;height:34px;border-radius:7px;display:flex;'
               + 'align-items:center;justify-content:center;font-size:9px;'
               + 'font-family:var(--mono);'
               + 'border:1px solid ' + (f ? 'var(--c2)' : 'rgba(150,180,255,.2)') + ';'
               + 'background:' + (f ? 'rgba(251,191,36,.20)' : 'rgba(150,180,255,.05)') + ';'
               + 'color:' + (f ? '#fde68a' : 'var(--ink-faint)') + ';'
               + 'transition:box-shadow .3s var(--ease-out),transform .3s var(--ease-out)',
          title: 'layer ' + i + ' → ' + (f ? 'full' : 'shared') + (mla ? '（MLA 层）' : '（KDA 层）'),
          text: f ? 'F' : 's',
        });
        r.appendChild(e); cs.push(e);
      }
      col.appendChild(r);
      return { col, cs };
    };

    const s1 = mkStrip('freq = 1（默认）· index_skip_topk_offset = 2', () => true);
    const s2 = mkStrip('freq = 4 · index_skip_topk_offset = 2（实测：full 落在 0, 1, 5, 9, …, 41）',
      i => ((Math.max(i - 2 + 1, 0) % 4) === 0));
    viz.appendChild(s1.col);
    viz.appendChild(s2.col);

    const warn = U.el('div', { class: 'card cc3', style: 'padding:9px 11px;width:100%' });
    warn.innerHTML = '<div class="mono" style="font-size:11.5px;color:#fecdd3">'
      + '⚠ freq=4 时 12 个 full <b>全落在 KDA 层</b>上，11 个 MLA 层全是 shared —— '
      + '实测这种组合建模即抛 ValueError: Shared DSA layers require top-k indices from a previous full indexer layer.'
      + '</div>';
    viz.appendChild(warn);
    warn.style.opacity = '.28';

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 默认：freq = 1 → (…) % 1 恒为 0 → 45 个 full</span>';
    tl.at(1400, () => { s1.cs.forEach((c, i) => { if (i % 4 === 3) c.style.transform = 'translateY(-3px)'; }); msg.innerHTML = 'freq = 1：每一层都满足判据 → 45 层全 <em>full</em>（F 只标在 MLA 层上才有意义）'; });
    tl.at(4400, () => { s2.cs.forEach(c => { c.style.opacity = '0'; }); msg.innerHTML = '换成 <em>freq = 4</em>：full 变成稀疏的 —— 隔 4 层才有一个'; });
    tl.at(7000, () => { s2.cs.forEach(c => { c.style.opacity = '1'; }); msg.innerHTML = '实测 full 落在 <em>[0, 1, 5, 9, …, 41]</em>，共 12 层 full、33 层 shared'; });
    tl.at(10200, () => {
      warn.style.opacity = '1';
      s2.cs.forEach((c, i) => { if (i % 4 === 3) c.style.boxShadow = '0 0 0 2px var(--bad)'; });
      msg.innerHTML = '但这份排班<em>不看层类型</em>：12 个 full 全部落在 KDA 层，MLA 层一个 full 都没有';
    });
    tl.at(13200, () => {
      msg.innerHTML = '<span class="cm">// 排班必须和 layer_types 对齐 —— 否则 shared 层拿不到任何人的 top-k</span>';
    });
  },
},

/* ------------------------------------------------- 7 pattern */
{
  kicker: '第 ⑥ 步 · 显式模式',
  title: '逃生口：<span class="hl-b">index_topk_pattern</span> 直接指定',
  sub: '"F" → full，"S" → shared；模式串的长度就是表的长度，不会补齐。',
  caption: '优先级：显式传的 indexer_types > index_topk_pattern > freq/offset。',
  lang: 'python',
  codeStart: 175,
  code: `        # Per-layer indexer mode: a pattern (e.g. \`"FSSF..."\`) overrides the freq/offset schedule.
        if self.indexer_types is None:
            pattern = kwargs.get("index_topk_pattern")
            if pattern is not None:
                self.indexer_types = (
                    [{"F": "full", "S": "shared"}[c] for c in pattern] if isinstance(pattern, str) else list(pattern)
                )`,
  codeNote: '逐字符映射；传 list 则原样收下。代码里没有 range(num_hidden_layers)，所以长度不会被检查。',
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '实测（num_hidden_layers = 8）：模式串逐字符变成 indexer_types' }));

    const mkMap = (pat, types) => {
      const col = U.el('div', { class: 'col gap4', style: 'width:100%' });
      const r1 = U.el('div', { class: 'row gap4', style: 'width:100%' });
      const r2 = U.el('div', { class: 'row gap4', style: 'width:100%' });
      for (let i = 0; i < pat.length; i++) {
        const f = pat[i] === 'F';
        r1.appendChild(U.el('div', {
          class: 'n sm', style: 'flex:1;height:30px;border-radius:7px;display:flex;align-items:center;'
            + 'justify-content:center;font-family:var(--mono);border:1px solid '
            + (f ? 'var(--c2)' : 'rgba(150,180,255,.24)') + ';background:'
            + (f ? 'rgba(251,191,36,.18)' : 'rgba(150,180,255,.06)') + ';color:'
            + (f ? '#fde68a' : 'var(--ink-dim)'), text: pat[i] }));
        r2.appendChild(U.el('div', {
          class: 'n sm', style: 'flex:1;height:30px;border-radius:7px;display:flex;align-items:center;'
            + 'justify-content:center;font-family:var(--mono);border:1px solid var(--panel-brd);'
            + 'background:rgba(150,180,255,.05);color:' + (f ? 'var(--c2)' : 'var(--ink-dim)'),
          text: types[i] }));
      }
      col.appendChild(r1); col.appendChild(r2);
      return col;
    };

    const box = U.el('div', { class: 'card cc1', style: 'padding:9px 11px;width:100%' }, [
      U.el('div', { class: 'klabel', text: 'index_topk_pattern = "FSSFSSSF"  →  indexer_types = [\'full\', \'shared\', \'shared\', \'full\', \'shared\', \'shared\', \'shared\', \'full\']' }),
      mkMap('FSSFSSSF', ['full', 'shared', 'shared', 'full', 'shared', 'shared', 'shared', 'full']),
    ]);
    viz.appendChild(box);

    const bad = U.el('div', { class: 'card cc3', style: 'padding:9px 11px;width:100%' });
    bad.innerHTML = '<div class="mono" style="font-size:11.5px;color:#fecdd3;line-height:1.7">'
      + '⚠ 45 层的 config 传 <b>"FSSF"</b> 实测得到的是<b>长度 4</b> 的列表 —— 不补齐、不循环。<br>'
      + '真拿去建模：第 4 层之后 <b>IndexError: list index out of range</b>（config.indexer_types[layer_idx] 越界）。'
      + '</div>';
    viz.appendChild(bad);
    bad.style.opacity = '.28';

    const prio = U.el('div', { class: 'row gap8 wrap', style: 'width:100%' });
    prio.innerHTML =
      '<span class="chip c4">① 显式传 indexer_types</span>' +
      '<span class="dim mono">优先于</span>' +
      '<span class="chip c2">② index_topk_pattern</span>' +
      '<span class="dim mono">优先于</span>' +
      '<span class="chip c0">③ index_topk_freq / index_skip_topk_offset</span>';
    viz.appendChild(prio);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 逐字符映射：F → full，S → shared</span>';
    tl.at(1600, () => { msg.innerHTML = 'F → <em>full</em>，S → <em>shared</em>；字符串的每个字符对应一层'; });
    tl.at(4400, () => { bad.style.opacity = '1'; msg.innerHTML = '陷阱：<em>模式串的长度就是表的长度</em> —— 代码里没有按 num_hidden_layers 补齐'; });
    tl.at(7800, () => { msg.innerHTML = '45 层的模型要写满 45 个字符（或直接传等长 list），否则建模时越界'; });
    tl.at(11000, () => { prio.classList.add('ac'); msg.innerHTML = '三个入口的优先级：显式 indexer_types → 模式串 → freq / offset'; });
    tl.at(13800, () => { msg.innerHTML = '<span class="cm">// 默认走 freq 分支，是因为没人愿意手写 45 个字母</span>'; });
  },
},

/* ------------------------------------------------- 8 ★ 整除约束 */
{
  kicker: '第 ⑦ 步 · 校验',
  title: '★ <span class="hl-c">index_topk % index_kpool == 0</span> 为什么必须存在',
  sub: '索引器是"挑池子再展开"：池数用整除算，而掩码宽度写死了 index_topk。不能整除，两者就对不上。',
  caption: '同节的另外两条检查：num_attention_heads == num_key_value_heads，以及 qk_rope_head_dim 必须为 0。',
  lang: 'python',
  codeStart: 207,
  code: `    def validate_architecture(self):
        """Part of \`@strict\`-powered validation. Validates the architecture of the config."""
        if self.num_attention_heads != self.num_key_value_heads:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be the same as "
                f"num_key_value_heads ({self.num_key_value_heads})."
            )

        if self.index_kpool < 1:
            raise ValueError(f"index_kpool must be positive, got {self.index_kpool}.")

        if self.index_topk % self.index_kpool != 0:
            raise ValueError(f"index_topk ({self.index_topk}) must be divisible by index_kpool ({self.index_kpool}).")`,
  codeNote: '@strict 让这些检查在构造 config 时就跑：实测 index_topk=2050 直接抛 ValueError，而不是等到前向。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '索引器的预算切分：128 个池 × 每池 16 个 token = 2048 个位置' }));

    const pools = U.el('div', { class: 'row gap8 center', style: 'width:100%' });
    const poolEls = [];
    for (let p = 0; p < 8; p++) {
      const g = U.el('div', { style: 'display:grid;grid-template-columns:repeat(4,9px);gap:2px' });
      for (let c = 0; c < 16; c++) {
        g.appendChild(U.el('div', {
          style: 'width:9px;height:9px;border-radius:2px;background:rgba(34,211,238,.35);'
               + 'border:1px solid rgba(34,211,238,.5);transition:background .3s,transform .3s',
        }));
      }
      const b = U.el('div', { class: 'col gap4 center' }, [
        g, U.el('div', { class: 'mono', style: 'font-size:9px;color:var(--ink-faint)', text: '池 ' + p }),
      ]);
      pools.appendChild(b); poolEls.push(b);
    }
    pools.appendChild(U.el('div', { class: 'col center gap4' }, [
      U.el('div', { class: 'mono', style: 'font-size:14px;color:var(--ink-faint)', text: '…' }),
      U.el('div', { class: 'mono', style: 'font-size:9px;color:var(--ink-faint)', text: '共 128 池' }),
    ]));
    viz.appendChild(pools);

    const eq = U.el('div', { class: 'formula', style: 'width:100%' });
    eq.innerHTML = '2048 = 16 × <em>128</em> &nbsp;&nbsp;'
      + '<span class="cm">// select_k = index_topk // index_kpool，每池展开 index_kpool 个位置</span>';
    viz.appendChild(eq);

    const cases = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const okc = W.card({ cc: 4, tint: 4, title: '✓ 2048 % 16 == 0',
      sub: '128 个池刚好铺满 2048 个位置，掩码宽度与实际可选位置完全一致。', style: 'flex:1' });
    const noc = W.card({ cc: 3, tint: 3, title: '✗ 2050 % 16 == 2',
      sub: '池数仍是 2050 // 16 = 128，展开只有 2048 个位置；掩码按 2050 补，多出的 2 个位置永远填 -1。',
      style: 'flex:1' });
    cases.appendChild(okc); cases.appendChild(noc);
    viz.appendChild(cases);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    noc.style.opacity = '.3';
    msg.innerHTML = '<span class="cm">// 池化：每 16 个 token 压成一组，索引器只对组打分</span>';
    tl.at(1600, () => {
      poolEls.forEach((b, i) => { if (i < 3) b.style.transform = 'translateY(-4px)'; });
      msg.innerHTML = '索引器挑的是<em>池</em>：2048 / 16 = <em>128</em> 个池，每池展开 16 个 token 位置';
    });
    tl.at(4600, () => {
      poolEls.forEach(b => { b.style.transform = 'none'; });
      eq.classList.add('ac');
      msg.innerHTML = '两处都用<em>整除</em>：池数由 // 得出，而下游掩码宽度写死了 index_topk';
    });
    tl.at(7800, () => {
      eq.classList.remove('ac'); okc.classList.add('ac'); noc.style.opacity = '1';
      msg.innerHTML = '不能整除时：池数被截断，实际可选位置<em>少于</em>预算，错配还<em>不会报错</em>';
    });
    tl.at(11000, () => {
      okc.classList.remove('ac'); noc.classList.add('ac');
      msg.innerHTML = '实测：index_topk=2050 在<em>构造 config 时</em>就抛 ValueError —— 把静默错配变成异常';
    });
    tl.at(14200, () => {
      noc.classList.remove('ac');
      msg.innerHTML = 'index_kpool &lt; 1 也抛：池长为 0 的索引器等于除以零，同样在 config 阶段拦下';
    });
  },
},

/* ------------------------------------------------- 9 收束 */
{
  kicker: '收束',
  title: '两张表合起来，就是一次前向的排班',
  sub: '给定 num_hidden_layers 能写出 layer_types；给定一个 shared 层，能说出它省掉了什么。',
  caption: '下一课 L0-04：288 选 8 —— 稀疏层里的 MoE 经济学。',
  lang: 'python',
  codeStart: 165,
  code: `        if self.layer_types is None:
            kda_layers = [idx for idx in range(self.num_hidden_layers) if idx % 4 != 3]
            self.layer_types = [
                "linear_attention" if layer_idx in kda_layers else "indexed_attention"
                for layer_idx in range(self.num_hidden_layers)
            ]`,
  codeNote: '回到开头：现在这一段的每一行你都能推演出来 —— 包括 34 与 11 这两个数字。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const tb = W.table([
      ['layer_types', '<span class="hl">34 KDA + 11 MLA</span>', 'idx % 4 != 3 → linear_attention；MLA 层数 = n // 4'],
      ['indexer_types', '<span class="hl4">45 全 full</span>', 'freq=1 时取模恒为 0；freq=4/offset=2 → 12 full + 33 shared'],
      ['mlp_layer_types', '<span class="hl">3 dense + 42 sparse</span>', '前 3 层稠密（另一条独立的轴，见 L0-02）'],
      ['index_topk', '<span class="hl">2048 = 16 × 128</span>', '必须被 index_kpool 整除，否则静默错配'],
      ['缓存形态', '<span class="hl">常数 vs 线性</span>', 'KDA 1,048,576 个数/层；MLA 512 个数/token/层'],
    ], { head: ['排班表 / 参数', '实测值', '规则'] });
    wrap.appendChild(U.el('div', { class: 'card cc0', style: 'padding:10px 12px;width:100%' }, [
      U.el('div', { class: 'klabel', text: '本课的两张表 + 两个数字' }), tb,
    ]));

    const cards = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const c1 = W.card({ cc: 2, tint: 2, title: 'shared 省掉了什么（一句话）',
      sub: '索引器子模块不实例化、索引器前向不跑、索引器自己的压缩键缓存不写入 —— 直接复用"上一个 full 层"选好的 2048 个位置。',
      style: 'flex:1' });
    const c2 = W.card({ cc: 0, tint: 0, title: '3:1 的账（一句话）',
      sub: '增长的那部分只剩 11/45 的层：每 token 5,632 个数 vs 全 MLA 的 23,040 个数，约 4 倍；KDA 那部分是常数。',
      style: 'flex:1' });
    cards.appendChild(c1); cards.appendChild(c2);
    wrap.appendChild(cards);

    wrap.appendChild(W.exercise(
      '给定 <code class="inl">num_hidden_layers = 12</code>，写出 <code class="inl">layer_types</code>；'
      + '再说明为什么这个排班里<strong>不可能</strong>有 shared 索引器层。',
      '12 层 = 3 个完整周期：KDA 在 idx <b>0,1,2,4,5,6,8,9,10</b>（共 <b>9</b> 层），'
      + 'MLA 在 idx <b>3,7,11</b>（共 <b>3</b> 层 = 12 // 4）。<br>'
      + '默认 <code class="inl">index_topk_freq=1</code> → indexer_types 全是 <b>full</b>。'
      + '若强行把某个 MLA 层设成 shared：MLA 层之间总隔着 3 层 KDA，'
      + '而 KDA 层会把 <code class="inl">topk_indices</code> 重置为 <b>None</b>，'
      + '于是那层拿不到"上一个 full 层"的选择，建模时抛 '
      + '<code class="inl">ValueError: Shared DSA layers require top-k indices from a previous full indexer layer.</code>'
      + '<br>结论：<b>shared 只有在 layer_types 里出现相邻的两个 indexed_attention 层时才有意义。</b>'));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 三件事：规则、成本、约束</span>';
    tl.at(1600, () => { msg.innerHTML = '① 规则：<em>idx % 4 != 3</em> → KDA，其余 → MLA；MLA = n // 4'; });
    tl.at(4200, () => { c2.classList.add('ac'); msg.innerHTML = '② 成本：<em>增长的那部分只剩 11/45</em> —— 这是 3:1 而不是 1:1 的理由'; });
    tl.at(7200, () => {
      c2.classList.remove('ac'); c1.classList.add('ac');
      msg.innerHTML = '③ 复用：<em>shared</em> 省掉索引器模块、前向与它自己的缓存，但必须紧跟在 full 层之后';
    });
    tl.at(10400, () => {
      c1.classList.remove('ac');
      msg.innerHTML = '④ 约束：<em>index_topk % index_kpool == 0</em> —— 把静默的预算错配提前成 config 异常';
    });
    tl.at(13400, () => {
      msg.innerHTML = '下一课 <em>L0-04</em>：288 选 8 —— 这些稀疏层里的 MoE 经济学';
    });
    tl.at(15800, () => {
      msg.innerHTML = '<span class="cm">// 排班表不产生计算，但它决定 45 次循环每次走哪条分支</span>';
    });
  },
},

];
