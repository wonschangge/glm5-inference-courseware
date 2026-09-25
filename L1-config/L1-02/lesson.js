/* ==========================================================================
   L1-02 · Glm5NextTextConfig 逐字段精读
   --------------------------------------------------------------------------
   覆盖：models/glm5_next/configuration_glm5_next.py（1 个文件 / 321 行）
   目标：看完能把 59 个字段按 7 组分完类，说出 4 道硬约束各防什么错，
         并知道「声明的字段」与「只认 kwargs 的名字」是两张表。
   排版基准：#visual 可视区 = 994 x 662.7 舞台像素（无头浏览器实测）。
   ========================================================================== */
'use strict';

/* 实测常量（.venv/bin/python 内省 Glm5NextTextConfig()，不是估算） */
const G = {
  fields: 59, declared: 49, inherited: 10,
  attn: 10, moe: 12, kda: 5, dsa: 6, mhc: 3, trunk: 13,
  hidden: 4096, inter: 12288, moeInter: 2048, layers: 45, heads: 64,
  topk: 2048, kpool: 16, pools: 128, streams: 4,
  importT: '0.81 s', importM: '67 MB', modelT: '3.07 s', modelM: '390 MB', modelFiles: 505,
};

const SCENES = [

/* ------------------------------------------------- 1 全景：59 个字段 */
{
  kicker: 'L1 · 配置与分派',
  title: '一个 config，<span class="hl-a">59 个字段</span>，7 个组',
  sub: '逐字段精读的第一步不是读值，而是分组：哪些字段属于注意力 / MoE / KDA / DSA / mHC，哪些只是主干或基类。',
  caption: '回顾 L0-02：那一课从默认值读出形状；这一课把同一个文件拆成字段清单。',
  lang: 'python',
  code: `@auto_docstring(checkpoint="zai-org/GLM-5.3-Flash")
@strict
class Glm5NextTextConfig(PreTrainedConfig):`,
  codeStart: 27,
  codeNote: '@auto_docstring 管文档，@strict 管字段契约 —— 字段在这里第一次变成「有类型的东西」。',
  duration: 15000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    /* ---- 三个数字 ---- */
    const stats = [
      { n: String(G.fields), k: '字段总数', s: 'dataclasses.fields()', cc: 0 },
      { n: String(G.declared), k: '本类声明', s: 'L101 – L154', cc: 4 },
      { n: String(G.inherited), k: '基类继承', s: 'return_dict / dtype …', cc: 1 },
    ];
    const srow = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const sEls = stats.map(st => {
      const c = W.card({ cc: st.cc, tint: st.cc, style: 'flex:1',
        title: '<span class="mono" style="font-size:11px">' + st.k + '</span>',
        body: U.el('div', { class: 'row gap8', style: 'align-items:baseline' }, [
          U.el('div', { class: 'n big', text: st.n }),
          U.el('div', { class: 'mono dim', style: 'font-size:10.5px', text: st.s }),
        ]) });
      srow.appendChild(c); return c;
    });
    viz.appendChild(srow);

    /* ---- 7 组表 ---- */
    const rows = [
      { cls_k: true, cells: ['注意力 MLA', '<span class="hl">10</span>', 'kv_lora_rank=512 · q_lora_rank=1536 · head_dim=0'] },
      { cls_k: true, cells: ['MoE', '<span class="hl">12</span>', 'n_routed_experts=288 · num_experts_per_tok=8 · mlp_layer_types'] },
      { cls_k: true, cells: ['KDA 线性注意力', '<span class="hl">5</span>', 'layer_types · linear_head_dim=128 · linear_lower_bound=-5.0'] },
      { cls_k: true, cells: ['DSA 稀疏索引器', '<span class="hl">6</span>', 'index_topk=2048 · index_kpool=16 · indexer_types'] },
      { cls_k: true, cells: ['mHC 多流残差', '<span class="hl">3</span>', 'hc_mult=4 · hc_eps=1e-6 · hc_sinkhorn_iters=20'] },
      { cls_k: true, cells: ['主干（通用）', '<span class="hl">13</span>', 'hidden_size=4096 · num_hidden_layers=45 · vocab_size=154880'] },
      { cls_k: true, cells: ['基类继承', '<span class="hl">10</span>', 'return_dict · dtype · architectures …'] },
    ];
    const tb = W.table(rows, { head: ['家族', '字段数', '代表字段'] });
    const trs = Array.from(tb.querySelectorAll('tbody tr'));
    viz.appendChild(U.el('div', { class: 'card', cc: 0, style: 'padding:10px 12px;width:100%' }, [tb]));
    viz.appendChild(U.el('div', { class: 'mono dim', style: 'font-size:10.5px;width:100%',
      text: '分组规则：改了这个字段，谁的行为跟着变 —— 不按行号（第 3 幕会看到反例）' }));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    const pick = (k) => {
      trs.forEach((tr, i) => {
        tr.classList.toggle('ac', i === k);
        tr.style.opacity = (k < 0 || i === k) ? '1' : '.4';
      });
    };
    const texts = [
      '注意力组 <em>10</em> 个：头数、低秩瓶颈、头内维度、以及 <span class="cm">head_dim=0</span> 这个「不做 RoPE」的编码',
      'MoE 组 <em>12</em> 个：专家数、每 token 激活数、路由分组、以及稠密/稀疏排班表',
      'KDA 组 <em>5</em> 个：一张排班表 + 4 个 <span class="fn">linear_*</span> —— 第 5 幕专门讲',
      'DSA 组 <em>6</em> 个：选多少位置、池子多大、索引器自己的头宽与头数',
      'mHC 组 <em>3</em> 个：流条数、数值下限、Sinkhorn 迭代次数',
      '主干 <em>13</em> 个：宽度与深度 —— 改了它们，所有子模块一起变，所以不归任何家族',
      '基类 <em>10</em> 个：来自 <span class="fn">PreTrainedConfig</span> 的公共契约（回顾 L1-01）',
    ];
    pick(-1);
    msg.innerHTML = '<span class="cm">// 59 个字段，逐个判组</span>';
    texts.forEach((t, i) => {
      tl.at(1400 + i * 1750, () => { pick(i); msg.innerHTML = t; });
    });
    tl.at(14000, () => {
      pick(-1);
      msg.innerHTML = '<em>59 = 49 + 10</em> &nbsp;—— 分组不是目录朗读，是「谁会跟着变」的判断';
    });
  },
},

/* ------------------------------------------------- 2 身份与别名 */
{
  kicker: 'L1 · 配置与分派',
  title: '三行类级声明：<span class="hl-a">它是谁</span>、忽略什么、别名指向谁',
  sub: 'model_type 是字符串查找的键；keys_to_ignore_at_inference 把运行时状态排除在配置之外；attribute_map 让旧名字继续可用。',
  caption: '「字符串如何找到类」的完整链条是 L1-05；本课只钉住 config 这一端的两个键。',
  lang: 'python',
  code: `    model_type = "glm5_next_text"
    keys_to_ignore_at_inference = ["past_key_values"]`,
  codeStart: 69,
  codeNote: '两行元信息紧挨在类体开头 —— 它们不是数值，却决定了这个类怎么被找到。',
  duration: 13000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    /* ---- 字符串 -> 类 的链 ---- */
    const fl = W.flow([
      { t: 'config.json', s: '"model_type"', cc: 0 },
      { t: 'CONFIG_MAPPING', s: '_LazyConfigMapping', cc: 4 },
      { t: 'Glm5NextConfig', s: '顶层外壳', cc: 1 },
      { t: 'text_config', s: 'Glm5NextTextConfig', cc: 1 },
    ], { style: 'width:100%' });
    viz.appendChild(U.el('div', { class: 'col gap6', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: '字符串如何找到类：取值的那一刻才 import' }), fl,
    ]));

    /* ---- 三张卡片 ---- */
    const crow = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cards = [
      { cc: 0, t: 'model_type', s: '字符串查找的键。<code class="inl">from_pretrained</code> 读 config.json 的 model_type，'
          + '再回 <code class="inl">CONFIG_MAPPING</code> 取类 —— 实测它的类型是 <code class="inl">_LazyConfigMapping</code>。' },
      { cc: 4, t: 'keys_to_ignore_at_inference', s: '把 <code class="inl">past_key_values</code> 排除在「配置对比」之外：'
          + '它是每步前向都在变的运行时对象，不是配置。' },
      { cc: 2, t: 'attribute_map', s: '旧名 → 正名。读写 <code class="inl">num_local_experts</code> 都落到 '
          + '<code class="inl">n_routed_experts</code>；序列化只写正名。' },
    ];
    const cEls = cards.map(c => {
      const e = W.card({ cc: c.cc, tint: c.cc, style: 'flex:1',
        title: '<span class="mono" style="font-size:11.5px">' + c.t + '</span>', sub: c.s });
      crow.appendChild(e); return e;
    });
    viz.appendChild(crow);

    const lazy = U.el('div', { class: 'formula', style: 'width:100%' });
    lazy.innerHTML = '惰性 import 实测：<em>import transformers</em> = ' + G.importT + ' / ' + G.importM
      + '（不含 torch）&nbsp;｜&nbsp;<em>modeling_glm5_next</em> = ' + G.modelT + ' / ' + G.modelM
      + '&nbsp;｜&nbsp;仓库里 <em>' + G.modelFiles + '</em> 个 modeling_*.py';
    viz.appendChild(lazy);

    const alias = U.el('div', { class: 'card', cc: 5, style: 'padding:9px 12px;width:100%' });
    alias.innerHTML = '<div class="mono" style="font-size:11.5px;line-height:1.7">'
      + 'c = Glm5NextTextConfig(<span class="hl3">num_local_experts</span>=99) &nbsp;→&nbsp; '
      + 'c.n_routed_experts = <span class="hl">99</span> &nbsp;·&nbsp; '
      + 'c.num_local_experts = <span class="hl">99</span> &nbsp;·&nbsp; '
      + '\'num_local_experts\' in c.to_dict() = <span class="hlbad">False</span></div>';
    viz.appendChild(alias);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    cEls.forEach(e => { e.style.opacity = '.3'; });
    alias.style.opacity = '.3';
    fl.focus(-1);
    msg.innerHTML = '<span class="cm">// config.json 里那个字符串，就是这条链的起点</span>';
    tl.at(1600, () => { fl.focus(0); msg.innerHTML = 'checkpoint 里写着 <em>"model_type": "glm5_next"</em>'; });
    tl.at(3400, () => { fl.focus(1); msg.innerHTML = '<em>CONFIG_MAPPING</em> 是惰性的：取这个键才 import 对应的 configuration_*.py'; });
    tl.at(5200, () => { fl.focus(2); msg.innerHTML = '拿到顶层 <em>Glm5NextConfig</em>，再由它的 <span class="fn">sub_configs</span> 走到子配置'; });
    tl.at(7000, () => {
      fl.focus(3);
      msg.innerHTML = '本课读的就是最后一环：<em>Glm5NextTextConfig</em>（它自己的 model_type 是 glm5_next_text）';
    });
    tl.at(8800, () => {
      cEls.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = '三行类级声明各管一件事：<em>身份</em> / <em>忽略键</em> / <em>别名</em>';
    });
    tl.at(10600, () => {
      alias.style.opacity = '1';
      msg.innerHTML = '<span class="cm">// 别名只影响读写，不影响序列化 —— 正名才是唯一的真相</span>';
    });
  },
},

/* ------------------------------------------------- 3 ★ 顺序 != 分组 */
{
  kicker: 'L1 · 逐字段',
  title: '★ 12 行声明里，<span class="hl-a">MoE 被 MLA 劈成两段</span>',
  sub: '文件的书写顺序不是分组顺序。按行号读，你会以为 kv_lora_rank 是 MoE 的字段。',
  caption: '分组只能按语义：改这个字段，谁的行为跟着变。这是「逐字段精读」最容易踩的坑。',
  lang: 'python',
  code: `    n_shared_experts: int = 1
    n_routed_experts: int = 288
    routed_scaling_factor: float = 2.5
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 0
    v_head_dim: int = 256
    qk_nope_head_dim: int = 256
    n_group: int = 1
    topk_group: int = 1
    num_experts_per_tok: int = 8
    norm_topk_prob: bool = True`,
  codeStart: 109,
  codeNote: 'L109–L120 是连续 12 行：MoE(3) → MLA(5) → MoE(4)。',
  duration: 11500,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const names = [
      ['n_shared_experts', 0], ['n_routed_experts', 0], ['routed_scaling_factor', 0],
      ['kv_lora_rank', 1], ['q_lora_rank', 1], ['qk_rope_head_dim', 1],
      ['v_head_dim', 1], ['qk_nope_head_dim', 1],
      ['n_group', 0], ['topk_group', 0], ['num_experts_per_tok', 0], ['norm_topk_prob', 0],
    ];
    const C = [
      { brd: 'var(--c2)', bg: 'rgba(251,191,36,.14)', fg: '#fde68a' },
      { brd: 'var(--c1)', bg: 'rgba(167,139,250,.20)', fg: '#ddd6fe' },
    ];
    const grid = U.el('div', { style: 'display:grid;grid-template-columns:repeat(6,1fr);gap:4px;width:100%' });
    const cells = names.map(([nm, fam]) => {
      const e = U.el('div', {
        class: 'mono',
        style: 'height:44px;border-radius:8px;display:flex;align-items:center;justify-content:center;'
             + 'font-size:9.5px;text-align:center;padding:2px 4px;overflow-wrap:anywhere;'
             + 'border:1px solid ' + C[fam].brd + ';background:' + C[fam].bg + ';color:' + C[fam].fg
             + ';transition:opacity .3s,transform .3s,box-shadow .3s',
        text: nm });
      grid.appendChild(e); return e;
    });
    viz.appendChild(U.el('div', { class: 'col gap6', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: '源文件 L109 – L120（按书写顺序，6 列一行）' }), grid,
    ]));

    const legend = U.el('div', { class: 'row gap12 wrap', style: 'width:100%' });
    legend.innerHTML = U.chip('MoE 组', 2) + U.chip('注意力 MLA 组', 1)
      + '<span class="dim mono" style="margin-left:auto;font-size:10.5px">同色 = 同一家族</span>';
    viz.appendChild(legend);

    /* ---- 重新分组的两个盒子 ---- */
    const brow = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const boxA = W.card({ cc: 2, tint: 2, style: 'flex:1',
      title: 'MoE 组 · 7 个字段（被切成 3 + 4）',
      sub: '<span class="mono" style="font-size:10.5px">n_shared_experts · n_routed_experts · routed_scaling_factor'
         + ' · n_group · topk_group · num_experts_per_tok · norm_topk_prob</span>' });
    const boxB = W.card({ cc: 1, tint: 1, style: 'flex:1',
      title: 'MLA 组 · 5 个字段（夹在中间）',
      sub: '<span class="mono" style="font-size:10.5px">kv_lora_rank · q_lora_rank · qk_rope_head_dim'
         + ' · v_head_dim · qk_nope_head_dim</span>' });
    brow.appendChild(boxA); brow.appendChild(boxB);
    viz.appendChild(brow);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    const light = (idxs, on) => idxs.forEach(i => {
      cells[i].style.boxShadow = on ? '0 0 0 2px currentColor, 0 0 16px -4px currentColor' : 'none';
      cells[i].style.transform = on ? 'translateY(-3px)' : 'none';
      cells[i].style.opacity = '1';
    });
    boxA.style.opacity = '.3'; boxB.style.opacity = '.3';
    msg.innerHTML = '<span class="cm">// 一段连续声明，三个家族混在一起</span>';
    tl.at(1400, () => { light([0, 1, 2], true); msg.innerHTML = 'L109–L111 是 <span class="hl3">MoE</span>：专家数、共享专家、缩放系数'; });
    tl.at(3300, () => { light([3, 4, 5, 6, 7], true); msg.innerHTML = 'L112–L116 是 <span class="hl2">MLA</span>：KV/Q 的低秩瓶颈与头内维度'; });
    tl.at(5200, () => { light([8, 9, 10, 11], true); msg.innerHTML = 'L117–L120 又回到 <span class="hl3">MoE</span>：路由分组与 top-k 概率'; });
    tl.at(7100, () => {
      boxA.style.opacity = '1'; boxB.style.opacity = '1';
      msg.innerHTML = '把同族的格子收拢：<em>MoE 7 个</em>、<em>MLA 5 个</em> —— 位置完全对不上';
    });
    tl.at(9100, () => {
      msg.innerHTML = '★ 所以逐字段精读的判据只有一条：<em>改了这个字段，谁的行为跟着变</em>';
    });
  },
},

/* ------------------------------------------------- 4 主干骨架 */
{
  kicker: 'L1 · 逐字段',
  title: '主干骨架：<span class="hl-a">4 个宽度</span> + 3 个深度 / 头数',
  sub: '这 8 个字段不属于任何一个「优化家族」—— 改了它们，注意力、MoE、KDA、mHC 会一起变。',
  caption: '回顾 L0-02：4096 → 1536 → 16384 的手算，就是从这几个字段出发的。',
  lang: 'python',
  code: `    vocab_size: int = 154880
    hidden_size: int = 4096
    intermediate_size: int = 12288
    moe_intermediate_size: int = 2048

    num_hidden_layers: int = 45
    num_attention_heads: int = 64
    num_key_value_heads: int = 64`,
  codeStart: 101,
  codeNote: 'L101–L108：4 个宽度 + 深度 + 两个头数。',
  duration: 11000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const rows = [
      { cls_k: true, cells: ['vocab_size', '<span class="hl">154880</span>', '词表宽度 —— lm_head 的输出维（L6）'] },
      { cls_k: true, cells: ['hidden_size', '<span class="hl">4096</span>', '残差流宽度 = 所有子模块的输入输出'] },
      { cls_k: true, cells: ['intermediate_size', '<span class="hl">12288</span>', '稠密 FFN 隐藏宽度 = 3 × hidden_size'] },
      { cls_k: true, cells: ['moe_intermediate_size', '<span class="hl">2048</span>', '单个专家的宽度 = hidden_size / 2'] },
      { cls_k: true, cells: ['num_hidden_layers', '<span class="hl">45</span>', '深度；也是三张「每层一张表」的长度'] },
      { cls_k: true, cells: ['num_attention_heads', '<span class="hl">64</span>', '注意力头数'] },
      { cls_k: true, cells: ['num_key_value_heads', '<span class="hl">64</span>', '与上一行相等 → <span class="hlbad">不是 GQA</span>'] },
    ];
    const tb = W.table(rows, { head: ['字段', '默认值', '它决定什么'] });
    const trs = Array.from(tb.querySelectorAll('tbody tr'));
    viz.appendChild(U.el('div', { class: 'card', cc: 4, style: 'padding:10px 12px;width:100%' }, [tb]));

    const f = U.el('div', { class: 'formula', style: 'width:100%' });
    f.innerHTML = '实测算术：<em>12288 = 3 × 4096</em>&nbsp;｜&nbsp;<em>2048 = 4096 / 2</em>'
      + '&nbsp;｜&nbsp;<em>64 == 64</em> → 不是 GQA&nbsp;｜&nbsp;<em>45</em> = layer_types / mlp_layer_types / indexer_types 的长度';
    viz.appendChild(f);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    const pick = (idxs) => trs.forEach((tr, i) => {
      tr.classList.toggle('ac', idxs.indexOf(i) >= 0);
      tr.style.opacity = idxs.length === 0 || idxs.indexOf(i) >= 0 ? '1' : '.38';
    });
    pick([]);
    msg.innerHTML = '<span class="cm">// 这四个宽度是一切的起点</span>';
    tl.at(1600, () => { pick([0, 1, 2, 3]); msg.innerHTML = '4 个宽度：词表 <em>154880</em>、隐藏 <em>4096</em>、稠密 FFN <em>12288</em>、专家 <em>2048</em>'; });
    tl.at(3800, () => { pick([4]); msg.innerHTML = '深度 <em>45</em> —— 三张「每层一张表」的字段长度都是它'; });
    tl.at(5800, () => { pick([5, 6]); msg.innerHTML = '两个头数都是 <em>64</em>：KV 头没有减少，所以这不是 GQA'; });
    tl.at(8000, () => {
      pick([]);
      msg.innerHTML = '★ 它们不归任何家族，因为<em>改了谁都跟着变</em> —— 这就是「主干」这一组的判据';
    });
  },
},

/* ------------------------------------------------- 5 KDA 的 5 个字段 */
{
  kicker: 'L1 · 逐字段',
  title: 'KDA 的 <span class="hl-a">5 个字段</span>：4 个 linear_* + 1 张排班表',
  sub: 'layer_types 决定哪些层是 KDA，另外 4 个 linear_* 决定 KDA 层内部怎么算 —— 而它们并不挨着。',
  caption: '验收点 2 的答案在这一幕。KDA 层内部展开在 L4-02 ~ L4-05。',
  lang: 'python',
  code: `    linear_head_dim: int = 128
    linear_num_heads: int = 64
    linear_conv_kernel_dim: int = 4
    linear_lower_bound: float | None = -5.0`,
  codeStart: 143,
  codeNote: '这 4 行是全部 linear_* 字段；第 5 个字段 layer_types 在 L138，不在这段里。',
  duration: 12000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const defs = [
      { cc: 0, t: 'layer_types', v: '45 项', s: '哪些层是 KDA（默认 34 / 45）。声明在 L138 —— 位置不等于分组。' },
      { cc: 4, t: 'linear_head_dim', v: '128', s: '递推状态下每个头的宽度。与注意力的 qk_nope_head_dim=256 不是一回事。' },
      { cc: 4, t: 'linear_num_heads', v: '64', s: '头数与注意力相同，但走的是递推，不是 Q·Kᵀ。' },
      { cc: 2, t: 'linear_conv_kernel_dim', v: '4', s: '线性注意力里那段短卷积的核宽。' },
      { cc: 5, t: 'linear_lower_bound', v: '-5.0', s: '遗忘门衰减的下界。类型是 float | None —— 允许 None 是有意的。' },
    ];
    const row = U.el('div', { class: 'row gap8', style: 'width:100%;align-items:stretch' });
    const els = defs.map(d => {
      const e = W.card({ cc: d.cc, tint: d.cc, style: 'flex:1',
        title: '<span class="mono" style="font-size:10.5px">' + d.t + '</span>',
        body: U.el('div', { class: 'col gap6' }, [
          U.el('div', { class: 'n', text: d.v }),
          U.el('div', { class: 'cs', html: d.s }),
        ]) });
      row.appendChild(e); return e;
    });
    viz.appendChild(row);

    const f = U.el('div', { class: 'formula', style: 'width:100%' });
    f.innerHTML = 'KDA 5 字段 = <em>layer_types</em>（哪些层）+ <em>4 × linear_*</em>（层内怎么算）';
    viz.appendChild(f);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    els.forEach(e => { e.style.opacity = '.3'; });
    msg.innerHTML = '<span class="cm">// 第 5 个字段其实不在这段代码里</span>';
    tl.at(1500, () => { els[0].style.opacity = '1'; msg.innerHTML = '第 1 个：<em>layer_types</em> —— 排班表，声明在 L138，和这 4 行隔着 indexer_types'; });
    tl.at(3300, () => { els[1].style.opacity = '1'; msg.innerHTML = '第 2 个：<em>linear_head_dim = 128</em> —— KDA 的每头宽度'; });
    tl.at(5100, () => { els[2].style.opacity = '1'; msg.innerHTML = '第 3 个：<em>linear_num_heads = 64</em> —— 头数，不是 softmax 注意力'; });
    tl.at(6900, () => { els[3].style.opacity = '1'; msg.innerHTML = '第 4 个：<em>linear_conv_kernel_dim = 4</em> —— 短卷积核宽'; });
    tl.at(8700, () => { els[4].style.opacity = '1'; msg.innerHTML = '第 5 个：<em>linear_lower_bound = -5.0</em> —— 衰减下界（第 7 幕会看到它怎么被补回来）'; });
    tl.at(10500, () => {
      els.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = '★ 4 个 <span class="fn">linear_*</span> 是连续 4 行，但 KDA 的第 5 个字段在别处 —— 再次说明<em>分组不按位置</em>';
    });
  },
},

/* ------------------------------------------------- 6 DSA 的 6 个字段 */
{
  kicker: 'L1 · 逐字段',
  title: 'DSA 的 <span class="hl-a">6 个字段</span>：从 2048 个位置到 128 个池',
  sub: '选多少位置、池子多大、索引器自己多大 —— 这三件事各有一组字段，而它们被拆在两段声明里。',
  caption: '2048 ÷ 16 = 128 个池。索引器怎么选，展开在 L4-06。',
  lang: 'python',
  code: `    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 32
    head_dim: int = 0
    layer_types: list[str] | None = None
    # \`"full"\` runs the indexer, \`"shared"\` reuses the previous full layer's index mask.
    indexer_types: list[str] | None = None
    base_config_key = "text_config"`,
  codeStart: 134,
  codeNote: '这里 4 个 DSA 字段 + base_config_key；另外 2 个池化字段在 L153–L154。',
  duration: 12000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const fl = W.flow([
      { t: '序列', s: '每 16 个 token 一池', cc: 0 },
      { t: '128 个 KPool', s: '2048 ÷ 16', cc: 2 },
      { t: '索引器', s: '32 头 × 128 维', cc: 1 },
      { t: 'top-2048', s: '每层保留的位置', cc: 4 },
    ], { style: 'width:100%' });
    viz.appendChild(U.el('div', { class: 'col gap6', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: 'DSA 的选择链（数字全部来自默认值）' }), fl,
    ]));

    const rows = [
      { cls_k: true, cells: ['index_topk', '<span class="hl">2048</span>', '每层最多保留的稀疏位置数'] },
      { cls_k: true, cells: ['index_head_dim', '<span class="hl">128</span>', '索引器每个头的宽度'] },
      { cls_k: true, cells: ['index_n_heads', '<span class="hl">32</span>', '索引器的头数 —— 它是一次小注意力'] },
      { cls_k: true, cells: ['indexer_types', '<span class="hl">45 项</span>', 'full = 跑索引器；shared = 复用上一层选好的位置'] },
      { cls_k: true, cells: ['index_kpool', '<span class="hl">16</span>', '池化窗口（声明在 L153）'] },
      { cls_k: true, cells: ['index_kpool_always_select_tail', '<span class="hl">True</span>', '尾部不足一池的残尾是否永远入选（L154）'] },
    ];
    const tb = W.table(rows, { head: ['字段', '默认值', '作用'] });
    const trs = Array.from(tb.querySelectorAll('tbody tr'));
    viz.appendChild(U.el('div', { class: 'card', cc: 1, style: 'padding:10px 12px;width:100%' }, [tb]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    const pick = (idxs) => trs.forEach((tr, i) => {
      tr.classList.toggle('ac', idxs.indexOf(i) >= 0);
      tr.style.opacity = idxs.length === 0 || idxs.indexOf(i) >= 0 ? '1' : '.38';
    });
    pick([]);
    fl.focus(-1);
    msg.innerHTML = '<span class="cm">// 6 个字段，三件事：选多少 / 池多大 / 索引器多大</span>';
    tl.at(1600, () => { fl.focus(3); pick([0]); msg.innerHTML = '<em>index_topk = 2048</em> —— 每层最多保留 2048 个位置'; });
    tl.at(3600, () => { fl.focus(1); pick([4, 5]); msg.innerHTML = '<em>index_kpool = 16</em> → 2048 ÷ 16 = <em>128</em> 个整池；尾部残池由 always_select_tail 决定'; });
    tl.at(5600, () => { fl.focus(2); pick([1, 2]); msg.innerHTML = '索引器自己的规模：<em>32 头 × 128 维</em>，与注意力层的 64 × 256 无关'; });
    tl.at(7600, () => { pick([3]); msg.innerHTML = '<em>indexer_types</em> 是排班：full 跑索引器，shared 白拿上一层的位置'; });
    tl.at(9600, () => {
      fl.focus(-1); pick([]);
      msg.innerHTML = '★ 6 个字段被拆在 L134–L140 与 L153–L154 两段 —— 又一次「位置不等于分组」';
    });
  },
},

/* ------------------------------------------------- 7 mHC 与借住者 */
{
  kicker: 'L1 · 逐字段',
  title: 'mHC 只有 <span class="hl-a">3 个字段</span>，另外 5 个是「借住」的',
  sub: '同一段里挤着三个家族：mHC 3 个、MoE 2 个、DSA 2 个。逐字段精读必须一个个判组。',
  caption: 'hc_mult = 4 就是 L0-01 里 hidden_states 为什么是 4 维的那个 4；展开见 L0-05 / L3-06。',
  lang: 'python',
  code: `    hc_mult: int = 4
    hc_eps: float = 1e-6
    hc_sinkhorn_iters: int = 20
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001

    index_kpool: int = 16
    index_kpool_always_select_tail: bool = True`,
  codeStart: 147,
  codeNote: 'L147–L154：mHC(3) + MoE(2) + DSA(2)。',
  duration: 11000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    /* ---- mHC 三件套 ---- */
    const mhc = [
      { t: 'hc_mult', v: '4', s: '残差流条数。hidden_states 因此是 (batch, seq, 4, 4096)。' },
      { t: 'hc_eps', v: '1e-6', s: 'Sinkhorn 归一化的数值下限，防止除零。' },
      { t: 'hc_sinkhorn_iters', v: '20', s: 'Sinkhorn 迭代次数 —— 流间组合矩阵算多少轮。' },
    ];
    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const mEls = mhc.map(d => {
      const e = W.card({ cc: 2, tint: 2, style: 'flex:1',
        title: '<span class="mono" style="font-size:11px">' + d.t + '</span>',
        body: U.el('div', { class: 'col gap6' }, [
          U.el('div', { class: 'n', text: d.v }),
          U.el('div', { class: 'cs', html: d.s }),
        ]) });
      row.appendChild(e); return e;
    });
    viz.appendChild(U.el('div', { class: 'col gap6', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: 'mHC 组 · 3 个字段' }), row,
    ]));

    /* ---- 借住的 4 个 ---- */
    const borrowed = [
      { cc: 2, t: 'output_router_logits', g: 'MoE', s: '是否把路由 logits 也返回（训练用）' },
      { cc: 2, t: 'router_aux_loss_coef', g: 'MoE', s: '负载均衡辅助损失的系数' },
      { cc: 1, t: 'index_kpool', g: 'DSA', s: '池化窗口大小' },
      { cc: 1, t: 'index_kpool_always_select_tail', g: 'DSA', s: '残池是否永远入选' },
    ];
    const row2 = U.el('div', { class: 'row gap10', style: 'width:100%;align-items:stretch' });
    const bEls = borrowed.map(d => {
      const e = W.card({ cc: d.cc, style: 'flex:1;padding:9px 11px',
        title: '<span class="mono" style="font-size:10px">' + d.t + '</span>'
             + ' <span class="chip c' + d.cc + '" style="height:17px;font-size:9.5px">' + d.g + '</span>',
        sub: '<span style="font-size:10.5px">' + d.s + '</span>' });
      row2.appendChild(e); return e;
    });
    viz.appendChild(U.el('div', { class: 'col gap6', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: '同一段里借住的 4 个字段（不属于 mHC）' }), row2,
    ]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    mEls.forEach(e => { e.style.opacity = '.3'; });
    bEls.forEach(e => { e.style.opacity = '.3'; });
    msg.innerHTML = '<span class="cm">// 先认出这一段的真主人</span>';
    tl.at(1600, () => { mEls[0].style.opacity = '1'; msg.innerHTML = '<em>hc_mult = 4</em> —— 残差流条数，L0-01 里那个「4 维 hidden」的来源'; });
    tl.at(3400, () => { mEls[1].style.opacity = '1'; msg.innerHTML = '<em>hc_eps = 1e-6</em> —— Sinkhorn 的数值下限'; });
    tl.at(5200, () => { mEls[2].style.opacity = '1'; msg.innerHTML = '<em>hc_sinkhorn_iters = 20</em> —— 流间组合矩阵的迭代次数'; });
    tl.at(7000, () => {
      bEls.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = '另外 4 个是<span class="hl3">借住</span>的：2 个 MoE + 2 个 DSA —— 看标签，不看位置';
    });
    tl.at(9000, () => {
      msg.innerHTML = '★ 一段 8 行的声明里三个家族：<em>mHC 3</em> / <em>MoE 2</em> / <em>DSA 2</em>';
    });
  },
},

/* ------------------------------------------------- 8 ★ 四道闸门 */
{
  kicker: 'L1 · 逐字段',
  title: '★ 四道硬约束，<span class="hl-a">外加一道 NoPE 断言</span>',
  sub: 'validate_architecture 一共 5 处 raise：4 条参数契约 + 1 条架构语义断言。每条都对应一种「能构造成功、但前向必错」的配置。',
  caption: '验收点 1 的答案在这一幕。实测：第 4 条在当前 @strict 下根本走不到。',
  lang: 'python',
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
            raise ValueError(f"index_topk ({self.index_topk}) must be divisible by index_kpool ({self.index_kpool}).")

        if self.q_lora_rank is None:
            raise ValueError("For DSA usage in the attention layers, the \`q_lora_rank\` is strictly required!")

        if self.qk_rope_head_dim > 0:
            raise ValueError(
                f"Expecting NoPE for the DSA attention layers, but got {self.qk_rope_head_dim} as RoPE dim."
            )`,
  codeStart: 207,
  codeNote: 'L207–L227：四道参数契约 + 一道 NoPE 断言，顺序也有意义。',
  duration: 14000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const rows = [
      { cls_k: true, cells: ['1', '<span class="mono">heads != kv_heads</span>', '按 GQA 思路少配 KV 头；MLA 的 K/V 由同一条压缩潜向量展开，头数必须 1:1', '<span class="hlbad">ValueError</span>'] },
      { cls_k: true, cells: ['2', '<span class="mono">index_kpool &lt; 1</span>', '池化窗口为 0；必须排在取模之前，否则抛的是 ZeroDivisionError 而不是人话', '<span class="hlbad">ValueError</span>'] },
      { cls_k: true, cells: ['3', '<span class="mono">index_topk % index_kpool</span>', '2048 个稀疏位置切不整（池化是整池操作）', '<span class="hlbad">ValueError</span>'] },
      { cls_k: true, cells: ['4', '<span class="mono">q_lora_rank is None</span>', 'DSA 层没有压缩 Q，低秩路径断掉', '<span class="hl3">走不到</span>'] },
      { cls_k: true, cells: ['5', '<span class="mono">qk_rope_head_dim &gt; 0</span>', '在 NoPE 架构上打开 RoPE —— 这条是语义断言，不是参数契约', '<span class="hlbad">ValueError</span>'] },
    ];
    const tb = W.table(rows, { head: ['#', '检查', '它防的是什么错误', '实测'] });
    const trs = Array.from(tb.querySelectorAll('tbody tr'));
    viz.appendChild(U.el('div', { class: 'card', cc: 3, style: 'padding:10px 12px;width:100%' }, [tb]));

    const card = W.card({ cc: 4, style: 'width:100%',
      title: '顺序也有意义',
      sub: '实测同时给 <code class="inl">index_kpool=0</code> 与 <code class="inl">q_lora_rank=None</code>，'
         + '报的是 <code class="inl">q_lora_rank</code> 的<b>字段</b>错 —— <code class="inl">@strict</code> '
         + '先跑字段校验，再跑类级校验器。而第 4 条正是被字段校验提前拦下的那一条。' });
    viz.appendChild(card);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    const pick = (idxs) => trs.forEach((tr, i) => {
      tr.classList.toggle('ac', idxs.indexOf(i) >= 0);
      tr.style.opacity = idxs.length === 0 || idxs.indexOf(i) >= 0 ? '1' : '.34';
    });
    pick([]);
    card.style.opacity = '.3';
    msg.innerHTML = '<span class="cm">// 四道参数契约，逐个过</span>';
    tl.at(1600, () => { pick([0]); msg.innerHTML = '① <em>头数必须 1:1</em> —— 实测 <span class="mono">num_key_value_heads=32</span> 直接抛错'; });
    tl.at(3800, () => { pick([1]); msg.innerHTML = '② <em>index_kpool ≥ 1</em> —— 它必须排在取模之前，否则先炸的是 ZeroDivisionError'; });
    tl.at(6000, () => { pick([2]); msg.innerHTML = '③ <em>index_topk 能被 index_kpool 整除</em> —— 默认 2048 % 16 == 0 正好过'; });
    tl.at(8200, () => { pick([3]); msg.innerHTML = '④ <em>q_lora_rank 必须存在</em> —— 但实测走不到：@strict 的字段校验先以 TypeError 拦下'; });
    tl.at(10400, () => {
      pick([4]);
      msg.innerHTML = '第 5 条不是参数契约：<em>qk_rope_head_dim &gt; 0</em> 是「这个架构不做 RoPE」的语义断言';
    });
    tl.at(12400, () => {
      pick([]); card.style.opacity = '1';
      msg.innerHTML = '★ 闸门只覆盖「会算错」的组合：<em>layer_types / indexer_types 的长度没有任何检查</em>';
    });
  },
},

/* ------------------------------------------------- 9 收束 */
{
  kicker: '收束',
  title: '把 59 个字段压成一张表',
  sub: '7 组 / 49 + 10 / 4 道闸门 / 5 个只在构造时认识的名字 —— 下一课读 vision_config 与顶层嵌套。',
  caption: '下一课 L1-03：Glm5NextVisionConfig 与 Glm5NextConfig 如何把 text + vision 拼起来。',
  lang: 'python',
  code: `    model_type = "glm5_next"
    sub_configs = {"vision_config": Glm5NextVisionConfig, "text_config": Glm5NextTextConfig}`,
  codeStart: 291,
  codeNote: '顶层外壳把本课的 config 登记在 text_config 这个键上 —— base_config_key 正是它。',
  duration: 12000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const rows = [
      { cls_k: true, cells: ['注意力 MLA', '<span class="hl">10</span>', '头数 1:1、低秩瓶颈 1536 / 512、head_dim=0 表示不做 RoPE'] },
      { cls_k: true, cells: ['MoE', '<span class="hl">12</span>', '288 选 8、+1 共享专家、稠密/稀疏排班表'] },
      { cls_k: true, cells: ['KDA', '<span class="hl">5</span>', 'layer_types（哪些层）+ 4 个 linear_*（层内怎么算）'] },
      { cls_k: true, cells: ['DSA', '<span class="hl">6</span>', '2048 个位置、16 一池 → 128 池、索引器 32 × 128'] },
      { cls_k: true, cells: ['mHC', '<span class="hl">3</span>', 'hc_mult=4 条流、1e-6 下限、20 次 Sinkhorn'] },
      { cls_k: true, cells: ['主干', '<span class="hl">13</span>', '4096 / 12288 / 2048 / 45 / 64 —— 改了谁都跟着变'] },
      { cls_k: true, cells: ['基类继承', '<span class="hl">10</span>', 'PreTrainedConfig 的公共契约（L1-01）'] },
    ];
    const tb = W.table(rows, { head: ['家族', '字段数', '一句话'] });
    const trs = Array.from(tb.querySelectorAll('tbody tr'));
    viz.appendChild(U.el('div', { class: 'card', cc: 0, style: 'padding:9px 12px;width:100%' }, [tb]));

    const ex1 = W.exercise(
      'KDA 相关的 <b>5 个字段</b>是哪 5 个？它们为什么不在同一段代码里？',
      '<b>layer_types</b>（哪些层是 KDA）+ 4 个 <code class="inl">linear_*</code>：'
      + '<code class="inl">linear_head_dim=128</code>、<code class="inl">linear_num_heads=64</code>、'
      + '<code class="inl">linear_conv_kernel_dim=4</code>、<code class="inl">linear_lower_bound=-5.0</code>。'
      + '<br>它们不挨着，是因为 <code class="inl">layer_types</code> 声明在 L138，'
      + '与 4 个 <code class="inl">linear_*</code>（L143–L146）之间隔着 <code class="inl">indexer_types</code> 与 '
      + '<code class="inl">base_config_key</code> —— <b>分组按语义，不按行号</b>。');
    ex1.style.width = '100%';
    viz.appendChild(ex1);

    const ex2 = W.exercise(
      '<code class="inl">validate_architecture</code> 的四道硬约束各自防什么错？',
      '① <b>头数必须 1:1</b>：MLA 不是 GQA，少配 KV 头会让压缩潜向量对不上；'
      + '② <b>index_kpool ≥ 1</b>：池化窗口为 0 会让下一行的取模变成除零；'
      + '③ <b>index_topk 能被 index_kpool 整除</b>：2048 ÷ 16 = 128 个整池；'
      + '④ <b>q_lora_rank 必须存在</b>：DSA 需要压缩 Q 这条低秩路径。'
      + '<br>第 5 条 <code class="inl">qk_rope_head_dim &gt; 0</code> 是 NoPE 语义断言，不属于四道参数契约；'
      + '而第 ④ 条在当前 <code class="inl">@strict</code> 下已被字段校验提前拦下。');
    ex2.style.width = '100%';
    viz.appendChild(ex2);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    trs.forEach(tr => { tr.style.opacity = '.4'; });
    msg.innerHTML = '<span class="cm">// 七组，逐个点亮</span>';
    rows.forEach((r, i) => {
      tl.at(1200 + i * 1300, () => {
        trs.forEach((tr, k) => { tr.style.opacity = (k === i) ? '1' : '.4'; });
        msg.innerHTML = '<em>' + (i + 1) + '/7</em> ' + rows[i].cells[0] + ' &nbsp;<span class="cm">// '
          + rows[i].cells[1].replace(/<[^>]+>/g, '') + ' 个字段</span>';
      });
    });
    tl.at(10600, () => {
      trs.forEach(tr => { tr.style.opacity = '1'; });
      msg.innerHTML = '<em>59 = 49 + 10</em>；另有 5 个名字不是字段（无类型、无校验，却会被序列化）—— 下一课读 vision_config';
    });
  },
},

];
