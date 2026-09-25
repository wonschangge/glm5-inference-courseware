/* ==========================================================================
   L0-02 · 配置即架构：从 config 读出全部形状
   --------------------------------------------------------------------------
   覆盖：models/glm5_next/configuration_glm5_next.py（1 个文件 / 321 行）
   目标：看完能从 Glm5NextTextConfig() 的默认值手算出 MLA 层的 Q/K/V 形状，
        并说清 head_dim 为什么是 0、validate_architecture 的五处 raise 各防什么。
   排版基准：#visual 可视区 = 994 x 662.7 舞台像素（无头浏览器实测）。
   ========================================================================== */
'use strict';

/* 实测常量：全部来自 .venv/bin/python 内省 Glm5NextTextConfig()，不是抄文档 */
const CFG = {
  hidden: 4096, inter: 12288, moeInter: 2048,
  layers: 45, heads: 64, kvHeads: 64, dense: 3, sparse: 42,
  qLora: 1536, kvLora: 512, qkRope: 0, qkNope: 256, vHead: 256,
  qkHead: 256, headDim: 0,
  qProj: 64 * 256, kvProj: 64 * (256 + 256), kvComp: 512,
  indexTopk: 2048, indexKpool: 16, indexHeadDim: 128, indexNHeads: 32,
  tpPlan: 13, epPlan: 4,
};

const SCENES = [

/* ------------------------------------------------- 1 全局：这张施工图 */
{
  kicker: '预备篇 · 直觉',
  title: '一个 config，<span class="hl-a">钉死 45 层的全部形状</span>',
  sub: '本课只读一个文件：configuration_glm5_next.py。它不含一行前向计算，却规定了每一层的张量宽度、MLP 排班，以及哪些配置会被直接拒绝。',
  caption: '回顾 L0-01：那张「45 层排班表」上的每一个形状，最后都要回到这个文件对账。',
  lang: 'python',
  codeStart: 27,
  code: `@auto_docstring(checkpoint="zai-org/GLM-5.3-Flash")
@strict
class Glm5NextTextConfig(PreTrainedConfig):`,
  codeNote: '整个文件只有一个类承担「形状」职责：Glm5NextTextConfig。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const nodes = [
      { t: '默认值字段', s: '4096 / 64 / 256', cc: 0 },
      { t: '__post_init__', s: '生成三张表', cc: 2 },
      { t: 'validate_architecture', s: '5 处 raise', cc: 3 },
      { t: '45 层形状', s: '与后续课程对账', cc: 1 },
    ];
    const fl = W.flow(nodes, { style: 'width:100%' });
    wrap.appendChild(U.el('div', { class: 'col gap8', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: '本课要走的四步' }), fl,
    ]));

    const stats = [
      { s: 'hidden_size', v: CFG.hidden, cc: 4 },
      { s: 'num_attention_heads', v: CFG.heads, cc: 0 },
      { s: 'head_dim', v: CFG.headDim, cc: 3 },
      { s: 'num_hidden_layers', v: CFG.layers, cc: 1 },
    ];
    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const statEls = stats.map(st => {
      const c = W.card({ cc: st.cc, tint: st.cc,
        title: '<span class="mono" style="font-size:11px">' + st.s + '</span>',
        body: U.el('div', { class: 'n big', text: String(st.v) }) });
      c.style.flex = '1';
      row.appendChild(c); return c;
    });
    wrap.appendChild(row);

    const two = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    two.appendChild(W.card({ cc: 4, title: '本课给你什么',
      sub: '一套可手算的推导：从 <code class="inl">4096</code> / <code class="inl">64</code> / '
         + '<code class="inl">256</code> 直接写出 MLA 层 Q、K、V 的投影形状。', style: 'flex:1' }));
    two.appendChild(W.card({ cc: 3, title: '本课不给你什么',
      sub: '任何前向逻辑。config 里没有矩阵乘法 —— 它只负责把形状<b>钉死</b>，算在别的文件里。', style: 'flex:1' }));
    wrap.appendChild(two);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    fl.focus(0);
    msg.innerHTML = '<span class="cm">// 配置即架构：这张施工图决定了后面每一层</span>';
    tl.at(1800, () => { fl.focus(1); msg.innerHTML = '默认值先立起来：<em>4096</em> 隐藏维、<em>64</em> 个头、<em>45</em> 层'; });
    tl.at(4800, () => { fl.focus(2); msg.innerHTML = '配置组合先过闸门：<em>5</em> 处 raise，第 7 幕逐条看'; });
    tl.at(8000, () => { fl.focus(3); msg.innerHTML = '形状定下来之后，最反直觉的一个数是 <em>head_dim = 0</em>'; });
    tl.at(11500, () => {
      fl.focus(-1); statEls[2].classList.add('ac');
      msg.innerHTML = '<span class="cm">// 先记住这四个数：4096 / 64 / 0 / 45</span>';
    });
    tl.at(14200, () => {
      statEls.forEach(e => e.classList.remove('ac'));
      msg.innerHTML = '下一步：把「宽度」和「深度」分开量一遍';
    });
  },
},

/* ------------------------------------------------- 2 宽度与深度 */
{
  kicker: '第 1 步 · 量宽度',
  title: '三条宽度：<span class="hl-a">4096</span> / <span class="hl4">12288</span> / <span class="hl3">2048</span>',
  sub: '稠密 FFN 宽 3 倍、专家 FFN 只有一半 —— 这两个比例就是 L0-04「288 选 8 为什么省算力」的算术基础。',
  caption: '这一幕只读 config 里 11 行；层数与头数在右侧对账，注意 num_key_value_heads 与 Q 头数相等。',
  lang: 'python',
  codeStart: 101,
  code: `    vocab_size: int = 154880
    hidden_size: int = 4096
    intermediate_size: int = 12288
    moe_intermediate_size: int = 2048

    num_hidden_layers: int = 45
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    n_shared_experts: int = 1
    n_routed_experts: int = 288
    routed_scaling_factor: float = 2.5`,
  codeNote: '宽度（前四行）+ 深度与专家（后七行）—— 注意力形状与它们无关。',
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });

    const left = W.card({ cc: 0, title: '三条宽度', sub: 'FFN 的隐藏宽度（实测默认值）', style: 'flex:1.3' });
    const bars = W.bars([
      { label: 'hidden_size', value: 4096, cc: 0, valueText: '4096' },
      { label: 'intermediate_size', value: 12288, cc: 4, valueText: '12288' },
      { label: 'moe_intermediate_size', value: 2048, cc: 2, valueText: '2048' },
    ], { max: 12288 });
    left.appendChild(U.el('div', { style: 'width:100%;margin-top:10px' }, [bars]));
    row.appendChild(left);

    const right = W.card({ cc: 1, title: '深度与头数', style: 'flex:1' });
    right.appendChild(U.el('div', { style: 'width:100%;margin-top:2px' }, [
      W.table([
        ['num_hidden_layers', '<span class="hl">45</span>', '主干层数'],
        ['num_attention_heads', '<span class="hl">64</span>', 'Q 的头数'],
        ['num_key_value_heads', '<span class="hl">64</span>', 'K/V 头数，与上一行<b>相等</b>'],
      ], { head: ['字段', '值', '含义'] }),
    ]));
    row.appendChild(right);
    viz.appendChild(row);

    const chips = U.el('div', { class: 'row gap12 wrap', style: 'width:100%' });
    chips.innerHTML =
      '<span class="chip c4"><i class="sw sw-c4"></i>稠密 FFN = 3 × hidden</span>' +
      '<span class="chip c2"><i class="sw sw-c2"></i>专家 FFN = hidden / 2</span>' +
      '<span class="chip c1"><i class="sw sw-c1"></i>K/V 头数 == Q 头数 → 不是 GQA</span>';
    viz.appendChild(chips);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    bars.setTo(0, 0); bars.setTo(1, 0); bars.setTo(2, 0);
    msg.innerHTML = '<span class="cm">// 先看 hidden_size：整条残差流的宽度</span>';
    tl.at(1600, () => {
      bars.setTo(0, 1);
      msg.innerHTML = 'hidden_size = <em>4096</em> &nbsp;<span class="cm">// 后面所有形状的基准</span>';
    });
    tl.at(4400, () => {
      bars.setTo(1, 1);
      msg.innerHTML = 'intermediate_size = <em>12288</em> = <em>3 × 4096</em> —— 稠密层的 FFN 宽 3 倍';
    });
    tl.at(7400, () => {
      bars.setTo(2, 1);
      msg.innerHTML = 'moe_intermediate_size = <em>2048</em> = <em>4096 / 2</em> —— 单个专家只有一半宽';
    });
    tl.at(10400, () => {
      right.classList.add('ac'); left.classList.remove('ac');
      msg.innerHTML = '两个头数<em>相等</em> → 这是 MHA 式的排布，没有 GQA 的分组';
    });
    tl.at(13200, () => {
      right.classList.remove('ac');
      msg.innerHTML = '<span class="cm">// 这三个比例先记住，注意到形状的第二幕用到的是另外五个字段</span>';
    });
  },
},

/* ------------------------------------------------- 3 ★ MLA 形状（FLOW） */
{
  kicker: '第 2 步 · ★ 形状',
  title: '★ <span class="hl-a">4096 → 1536 → 16384</span>：手算 MLA 的 Q/K/V',
  sub: '四个字段决定一切：q_lora_rank=1536、kv_lora_rank=512、qk_nope_head_dim=256、v_head_dim=256。',
  caption: '每层实际的 nn.Linear 形状已在 modeling_glm5_next.py:1128-1149 实测核对（该文件不计入本课覆盖率）。',
  lang: 'python',
  codeStart: 112,
  code: `    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 0
    v_head_dim: int = 256
    qk_nope_head_dim: int = 256
    n_group: int = 1
    topk_group: int = 1
    num_experts_per_tok: int = 8`,
  codeNote: '这五行就是 MLA 的全部维度来源；qk_rope_head_dim = 0 是下一幕的主角。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const nodes = [
      { t: 'hidden_states', s: '4096', cc: 0 },
      { t: 'q_a_proj / kv_a_proj', s: '1536 / 512', cc: 4 },
      { t: 'q_b_proj / kv_b_proj', s: '16384 / 32768', cc: 2 },
      { t: '注意力', s: '64 头 × 256 维', cc: 1 },
      { t: 'o_proj', s: '16384 → 4096', cc: 5 },
    ];
    const fl = W.flow(nodes, { style: 'width:100%' });
    viz.appendChild(U.el('div', { class: 'col gap8', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: 'MLA 的形状链 —— 每个数字都能从 config 默认值算出来' }), fl,
    ]));

    const mk = (cc, title, lines) => W.card({ cc: cc, tint: cc, title: title,
      body: U.el('div', { class: 'mono', style: 'font-size:11px;line-height:1.75;margin-top:4px' },
        lines.map(h => U.el('div', { html: h }))) });

    const cards = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cq = mk(4, '① Q 路径', [
      'hidden 4096',
      '<span class="dim">↓ q_a_proj（压缩）</span>',
      '1536 = q_lora_rank',
      '<span class="dim">↓ q_b_proj（展开）</span>',
      '64 × <span class="hl">256</span> = <span class="hl">16384</span>',
    ]);
    const ckv = mk(2, '② K/V 路径（共用一条潜向量）', [
      'hidden 4096',
      '<span class="dim">↓ kv_a_proj_with_mqa</span>',
      '512 + 0 = 512',
      '<span class="dim">↓ kv_b_proj</span>',
      '64 × (256 + 256) = <span class="hl">32768</span>',
      '<span class="dim">↓ 切成 k_nope 256 / v 256</span>',
    ]);
    const co = mk(1, '③ 注意力与输出', [
      'scores (batch, 64, seq, seq)',
      'scale = 1 / √256 = 1/16',
      '64 × v_head_dim 256 = 16384',
      '<span class="dim">↓ o_proj</span>',
      'hidden 4096',
    ]);
    [cq, ckv, co].forEach(c => { c.style.flex = '1'; cards.appendChild(c); });
    viz.appendChild(cards);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    /* 数据流层：让「包」沿着形状链走一遍 */
    FLOW.attach(U.q('#visual'));
    for (let k = 0; k < fl.nodes.length - 1; k++) FLOW.pathBetween('p' + k, fl.nodes[k], fl.nodes[k + 1]);
    const dim = (k) => [cq, ckv, co].forEach((c, i) => { c.style.opacity = (i === k) ? '1' : '.3'; });

    fl.focus(0); [cq, ckv, co].forEach(c => { c.style.opacity = '.3'; });
    msg.innerHTML = '<span class="cm">// 从 hidden_states 开始，一条路走到底</span>';
    tl.at(1600, () => {
      fl.focus(1); FLOW.send('p0', { at: 1600, dur: 1300, color: '52,211,153', label: '4096' });
      msg.innerHTML = '先<em>压缩</em>：Q → <em>1536</em>，KV → <em>512</em>（这就是 MLA 的低秩瓶颈）';
    });
    tl.at(4800, () => {
      fl.focus(2); FLOW.send('p1', { at: 4800, dur: 1300, color: '251,191,36', label: '1536 / 512' });
      dim(0);
      msg.innerHTML = '再<em>展开</em>：q_b_proj → <em>64 × 256 = 16384</em>，kv_b_proj → <em>64 × 512 = 32768</em>';
    });
    tl.at(8000, () => {
      fl.focus(3); FLOW.send('p2', { at: 8000, dur: 1300, color: '167,139,250', label: '64 × 256' });
      dim(1);
      msg.innerHTML = '注意力：<em>64</em> 头，每头 <em>qk_head_dim = 0 + 256 = 256</em> 维';
    });
    tl.at(11200, () => {
      fl.focus(4); FLOW.send('p3', { at: 11200, dur: 1300, color: '244,114,182', label: '16384' });
      dim(2);
      msg.innerHTML = '输出：<em>64 × v_head_dim 256 = 16384</em> → o_proj 回到 <em>4096</em>';
    });
    tl.at(14400, () => {
      fl.focus(-1); [cq, ckv, co].forEach(c => { c.style.opacity = '1'; });
      msg.innerHTML = '<span class="cm">// 4096 / 1536 / 16384 / 512 / 32768 —— 全部是 config 默认值的算术</span>';
    });
    tl.at(16300, () => {
      msg.innerHTML = '★ 唯一还没解释的数字是那个 <em>0</em>：qk_head_dim 为什么是 0 + 256？';
    });
  },
},

/* ------------------------------------------------- 4 ★ head_dim = 0 */
{
  kicker: '第 3 步 · ★ 语义',
  title: '★ 为什么 <span class="hl-a">head_dim</span> 是 0 而不是 256',
  sub: '在本仓库的约定里，head_dim 表示「会被 RoPE 旋转的那部分维度」。DSA 层是 NoPE，所以那一部分是 0 —— 真正的头宽叫 qk_head_dim = 256。',
  caption: 'head_dim=0 不等于「头宽为 0」：命名约定比数值更容易骗人，这一幕就是为了拆掉这个错觉。',
  lang: 'python',
  codeStart: 134,
  code: `    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 32
    head_dim: int = 0
    layer_types: list[str] | None = None
    # `"full"` runs the indexer, `"shared"` reuses the previous full layer's index mask.
    indexer_types: list[str] | None = None`,
  codeNote: 'index_head_dim 是索引器的头宽，与注意力头的 256 无关 —— 别把这两个数混起来。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    /* 头内维度分解条：0 维 RoPE + 256 维 NoPE */
    const bar = U.el('div', { class: 'row gap10', style: 'width:100%;align-items:stretch' });
    const rope = U.el('div', { class: 'col center',
      style: 'flex:0 0 118px;height:62px;border-radius:11px;border:1.5px dashed rgba(251,113,133,.65);'
           + 'background:rgba(251,113,133,.10);transition:opacity .4s,transform .4s' });
    rope.innerHTML = '<div class="n" style="color:var(--c3);font-size:17px">0</div>'
      + '<div class="klabel" style="margin:3px 0 0">RoPE 段</div>';
    const nope = U.el('div', { class: 'col center',
      style: 'flex:1;height:62px;border-radius:11px;border:1.5px solid rgba(167,139,250,.6);'
           + 'background:rgba(167,139,250,.14);transition:opacity .4s,transform .4s' });
    nope.innerHTML = '<div class="n" style="color:#ddd6fe;font-size:17px">256</div>'
      + '<div class="klabel" style="margin:3px 0 0">qk_nope_head_dim · 不旋转</div>';
    bar.appendChild(rope); bar.appendChild(nope);
    viz.appendChild(U.el('div', { class: 'col gap6', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: '一个头的 Q/K 宽度 = RoPE 段 + NoPE 段（色块宽度为示意，非等比）' }), bar,
    ]));

    const grid = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const tb = W.card({ cc: 1, title: '四个名字，别混', style: 'flex:1.35' });
    tb.appendChild(W.table([
      ['qk_rope_head_dim', '<span class="hlbad">0</span>', '会被 RoPE 旋转的维度 → 没有'],
      ['qk_nope_head_dim', '<span class="hl2">256</span>', '不旋转的那部分'],
      ['head_dim', '<span class="hlbad">0</span>', '被强制 = qk_rope_head_dim'],
      ['qk_head_dim', '<span class="hl">256</span>', '真正参与点积的头宽 = 0 + 256'],
    ], { head: ['字段', '值', '含义'] }));
    grid.appendChild(tb);

    const ev = W.card({ cc: 4, tint: 4, title: '实测证据', style: 'flex:1' });
    ev.appendChild(U.el('div', { class: 'mono', style: 'font-size:11px;line-height:1.7;margin-top:4px' }, [
      U.el('div', { html: 'C()<span class="dim">.head_dim</span> → <span class="hl">0</span>' }),
      U.el('div', { html: 'C()<span class="dim">.qk_head_dim</span> → <span class="hl">256</span>' }),
      U.el('div', { html: 'C(qk_rope_head_dim=64)' }),
      U.el('div', { class: 'dim', html: '&nbsp;&nbsp;→ ValueError: Expecting NoPE' }),
    ]));
    grid.appendChild(ev);
    viz.appendChild(grid);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 先把一个头的宽度拆开看</span>';
    tl.at(2200, () => {
      rope.style.transform = 'scale(1.06)';
      msg.innerHTML = '旋转的那一段是 <em>0</em> 维 —— 这个架构<b>不做 RoPE</b>';
    });
    tl.at(5200, () => {
      rope.style.transform = 'none'; nope.style.transform = 'scale(1.03)';
      msg.innerHTML = '不旋转的那一段是 <em>256</em> 维：这才是头的真实宽度';
    });
    tl.at(8200, () => {
      nope.style.transform = 'none'; tb.classList.add('ac');
      msg.innerHTML = '<em>head_dim</em> 记的是「可旋转维度」，所以它是 <em>0</em>；总宽记在 <em>qk_head_dim</em>';
    });
    tl.at(11500, () => {
      tb.classList.remove('ac'); ev.classList.add('ac');
      msg.innerHTML = '实测：<em>C(head_dim=256).head_dim</em> 依然是 <em>0</em> —— 传进去也会被丢掉（下一幕）';
    });
    tl.at(14200, () => {
      ev.classList.remove('ac');
      msg.innerHTML = '★ 一句话：<em>0</em> 不是「头宽为零」，是「这一层没有 RoPE」的编码';
    });
  },
},

/* ------------------------------------------------- 5 显式 override */
{
  kicker: '第 4 步 · 机制',
  title: '★ 那三行是谁写的：<span class="hl-a">显式 override</span>',
  sub: '__post_init__ 里先 pop 掉传进来的 head_dim，再用 qk_rope_head_dim 重新赋值 —— 作者专门留了一句 NOTE 解释为什么必须这样做。',
  caption: 'kwargs.pop("head_dim", None)：就算 config.json 里写着 head_dim=256，也一样被丢掉。',
  lang: 'python',
  codeStart: 200,
  code: `        # NOTE: this forces an intentional override as we have the convention of head_dim being the RoPE based dim
        kwargs.pop("head_dim", None)
        self.head_dim = self.qk_rope_head_dim
        self.qk_head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim

        super().__post_init__(**kwargs)`,
  codeNote: 'NOTE 注释 + 三行赋值，写在 super().__post_init__() 之前。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const nodes = [
      { t: 'kwargs.pop("head_dim")', s: '先丢掉传进来的值', cc: 3 },
      { t: 'self.head_dim = self.qk_rope_head_dim', s: '= 0', cc: 0 },
      { t: 'self.qk_head_dim = rope + nope', s: '= 0 + 256 = 256', cc: 4 },
    ];
    const fl = W.flow(nodes, { style: 'width:100%' });
    viz.appendChild(U.el('div', { class: 'col gap8', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: '三行代换 —— 顺序不能换：先丢，再写，再算' }), fl,
    ]));

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const tb = W.card({ cc: 1, title: '你传的 vs 实际生效的', style: 'flex:1.25' });
    tb.appendChild(W.table([
      ['head_dim', '256（哪怕 config.json 里写了）', '<span class="hl">0</span>'],
      ['qk_rope_head_dim', '0（默认）', '<span class="hl">0</span>'],
      ['qk_head_dim', '不可传（算出来的）', '<span class="hl">256</span>'],
    ], { head: ['字段', '传入值', '生效值'] }));
    row.appendChild(tb);

    const note = W.card({ cc: 2, tint: 2, title: 'NOTE 说了什么', style: 'flex:1' });
    note.appendChild(U.el('div', { class: 'mono', style: 'font-size:10.5px;line-height:1.65;margin-top:4px' }, [
      U.el('div', { class: 'dim', html: '// this forces an intentional' }),
      U.el('div', { class: 'dim', html: '// override as we have the' }),
      U.el('div', { class: 'dim', html: '// convention of head_dim being' }),
      U.el('div', { class: 'dim', html: '// the RoPE based dim' }),
      U.el('div', { style: 'margin-top:6px;color:#fde68a', html: '→ 故意覆盖，不是笔误' }),
    ]));
    row.appendChild(note);
    viz.appendChild(row);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    fl.focus(0);
    msg.innerHTML = '<span class="cm">// 为什么第 2 幕那个 0 丢不掉</span>';
    tl.at(2000, () => {
      fl.focus(1);
      msg.innerHTML = '第一步 <em>pop</em>：用户/checkpoint 给的 <em>head_dim</em> 直接丢弃，不进任何计算';
    });
    tl.at(5400, () => {
      fl.focus(2); note.classList.add('ac');
      msg.innerHTML = '第二步 <em>重新赋值</em>：head_dim := qk_rope_head_dim，于是恒等于 <em>0</em>';
    });
    tl.at(8800, () => {
      fl.focus(-1); note.classList.remove('ac'); tb.classList.add('ac');
      msg.innerHTML = '第三步顺手算出真实头宽：<em>qk_head_dim = 0 + 256 = 256</em>';
    });
    tl.at(12200, () => {
      tb.classList.remove('ac');
      msg.innerHTML = '<span class="cm">// 这三行写在 super().__post_init__() 之前：先改写字段，再交回父类</span>';
    });
    tl.at(14500, () => {
      msg.innerHTML = '收口在第 7 幕：<em>qk_rope_head_dim &gt; 0</em> 会被 validate_architecture 直接拒绝';
    });
  },
},

/* ------------------------------------------------- 6 mlp_layer_types */
{
  kicker: '第 5 步 · 排班',
  title: '前 <span class="hl-a">3</span> 层稠密，其余 <span class="hl3">42</span> 层稀疏',
  sub: 'mlp_layer_types 默认是 None —— 真正的排班表由 __post_init__ 现算：稠密前缀由 min(3, N) 保护，稀疏后缀没有保护。',
  caption: '回顾 L0-01 第 4 幕的 MLP 轴；注意力轴 layer_types 的生成在同一个 __post_init__ 里，展开见 L0-03。',
  lang: 'python',
  codeStart: 156,
  code: `    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] * min(3, self.num_hidden_layers) + ["sparse"] * (
                self.num_hidden_layers - 3
            )`,
  codeNote: '第一段是 KV 头数兜底，第二段就是「3 + 42」的出处 —— 没有任何一行硬编码 42。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const kd = U.el('div', { class: 'row between', style: 'width:100%' });
    kd.innerHTML = '<span class="klabel" style="margin:0">mlp_layer_types —— 45 层的 FFN 排班（实测默认值）</span>'
      + '<span class="dim mono" style="font-size:10.5px">层号 0 → 44</span>';
    viz.appendChild(kd);

    const strip = U.el('div', { class: 'row gap3', style: 'width:100%' });
    const cells = [];
    for (let i = 0; i < CFG.layers; i++) {
      const e = U.el('div', { class: 'n sm',
        style: 'flex:1 1 0;min-width:14px;height:58px;border-radius:8px;display:flex;align-items:center;'
             + 'justify-content:center;font-size:9.5px;border:1px solid rgba(150,180,255,.14);'
             + 'background:transparent;color:var(--ink-faint);transition:all .45s cubic-bezier(.16,1,.3,1)',
        title: 'layer ' + i, text: String(i) });
      strip.appendChild(e); cells.push(e);
    }
    viz.appendChild(strip);

    const dense = (c) => { c.style.borderColor = 'rgba(52,211,153,.75)'; c.style.background = 'rgba(52,211,153,.20)'; c.style.color = '#a7f3d0'; c.style.transform = 'translateY(-3px)'; };
    const sparse = (c) => { c.style.borderColor = 'rgba(251,191,36,.6)'; c.style.background = 'rgba(251,191,36,.15)'; c.style.color = '#fde68a'; };

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const fx = W.card({ cc: 0, title: '生成规则（逐字来自源码）', style: 'flex:1.5' });
    fx.appendChild(U.el('div', { class: 'mono', style: 'font-size:11px;line-height:1.75;margin-top:4px' }, [
      U.el('div', { html: '<span class="hl4">["dense"]</span> * min(3, N)' }),
      U.el('div', { html: '+ <span class="hl3">["sparse"]</span> * (N - 3)' }),
      U.el('div', { class: 'dim', style: 'margin-top:4px', html: 'N = 45 → 3 + 42' }),
    ]));
    row.appendChild(fx);

    const edge = W.card({ cc: 3, title: '边界：N < 3（实测）', style: 'flex:1' });
    edge.appendChild(U.el('div', { class: 'mono', style: 'font-size:10.5px;line-height:1.7;margin-top:4px' }, [
      U.el('div', { html: 'N=2 → <span class="hl4">[\'dense\', \'dense\']</span>' }),
      U.el('div', { class: 'dim', html: '[..] * (-1) 得到空列表' }),
      U.el('div', { class: 'dim', html: '不报错，只是没有稀疏层' }),
    ]));
    row.appendChild(edge);
    viz.appendChild(row);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 45 层的默认排班</span>';
    tl.at(1500, () => {
      for (let i = 0; i < CFG.dense; i++) dense(cells[i]);
      fx.classList.add('ac');
      msg.innerHTML = '前 <em>3</em> 层：<em>["dense"] * min(3, 45)</em> → 每层走完整的 12288 宽 FFN';
    });
    tl.at(4600, () => {
      for (let i = CFG.dense; i < CFG.layers; i++) sparse(cells[i]);
      fx.classList.remove('ac');
      msg.innerHTML = '其余 <em>42</em> 层：<em>["sparse"] * (45 - 3)</em> → 走 MoE（288 选 8）';
    });
    tl.at(8000, () => {
      msg.innerHTML = '<span class="cm">// 注意：没有一行硬编码 42，它是 45 - 3 算出来的</span>';
    });
    tl.at(11000, () => {
      edge.classList.add('ac');
      msg.innerHTML = '★ <em>min(3, N)</em> 保护了稠密前缀，但 <em>N - 3</em> 没保护：N=2 时后缀为空';
    });
    tl.at(14000, () => {
      edge.classList.remove('ac');
      msg.innerHTML = '另一条轴 layer_types（34 KDA : 11 MLA）在同一个 __post_init__ 里生成 → L0-03';
    });
  },
},

/* ------------------------------------------------- 7 ★ validate */
{
  kicker: '第 6 步 · ★ 闸门',
  title: '★ 五处 <span class="hl-a">raise</span>：四道形状契约 + 一道 NoPE 断言',
  sub: '校验不是形式主义 —— 每一条都对应一种「配置能构造成功、但前向一定算错」的组合。',
  caption: '实测：index_topk=2001 被拦下，默认的 2048 通过（2048 % 16 == 0）；qk_rope_head_dim=64 直接被拒。',
  lang: 'python',
  codeStart: 207,
  code: `    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
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
            raise ValueError("For DSA usage in the attention layers, the `q_lora_rank` is strictly required!")

        if self.qk_rope_head_dim > 0:
            raise ValueError(
                f"Expecting NoPE for the DSA attention layers, but got {self.qk_rope_head_dim} as RoPE dim."
            )`,
  codeNote: '@strict 驱动的类级校验：构造时就跑，不是在第一次前向时才炸。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap8 vizgrow hstart' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: 'validate_architecture 的五条闸门（按源码顺序）' }));

    const tb = W.table([
      ['1', '<span class="mono">num_attention_heads == num_key_value_heads</span>',
       '防止按 GQA 的思路少配 KV 头。MLA 的 K/V 由同一条压缩潜向量展开，头数必须 1:1',
       '传 num_key_value_heads=8 → 抛错'],
      ['2', '<span class="mono">index_kpool &gt;= 1</span>',
       '防止池化窗口为 0 或负数：2048 个位置要按 kpool 分组，窗口为 0 会除零',
       'index_kpool=0 → 抛错'],
      ['3', '<span class="mono">index_topk % index_kpool == 0</span>',
       '防止 2048 个稀疏位置切不整（最后一块残缺）',
       'index_topk=2001 → 抛错；2048 % 16 == 0 → 通过'],
      ['4', '<span class="mono">q_lora_rank is not None</span>',
       '防止 DSA 层没有压缩 Q —— 少了它，q_a_proj / q_b_proj 这条低秩路径就断了',
       'q_lora_rank=None 先被 @strict 的类型检查以 TypeError 拦下（这条 raise 是兜底）'],
      ['5', '<span class="mono">qk_rope_head_dim == 0</span>',
       '防止有人在 NoPE 架构上打开 RoPE —— 第 3 幕那个 0 的另一面',
       'qk_rope_head_dim=64 → ValueError: Expecting NoPE'],
    ], { head: ['#', '检查', '防的是什么错误', '实测'] });
    viz.appendChild(tb);
    const rows = Array.from(tb.querySelectorAll('tbody tr'));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 五条闸门，逐条点亮</span>';
    const show = (k, text) => {
      rows.forEach((r, i) => r.classList.toggle('ac', i === k));
      msg.innerHTML = text;
    };
    tl.at(1500, () => show(0, '① <em>头数必须相等</em> &nbsp;<span class="cm">// 不是 GQA，是 1:1</span>'));
    tl.at(4500, () => show(1, '② <em>kpool ≥ 1</em> &nbsp;<span class="cm">// 池化窗口不能为 0</span>'));
    tl.at(7500, () => show(2, '③ <em>2048 % 16 == 0</em> &nbsp;<span class="cm">// 2048 个位置必须切得整整齐齐</span>'));
    tl.at(10500, () => show(3, '④ <em>q_lora_rank 必须有值</em> &nbsp;<span class="cm">// DSA 靠它压缩 Q</span>'));
    tl.at(13500, () => show(4, '⑤ <em>qk_rope_head_dim 必须为 0</em> &nbsp;<span class="cm">// NoPE 是硬要求</span>'));
    tl.at(16000, () => {
      rows.forEach(r => r.classList.remove('ac'));
      msg.innerHTML = '★ 注意：没有一条检查 hidden_size % num_attention_heads —— 头宽不是除出来的，是直接给的';
    });
  },
},

/* ------------------------------------------------- 8 并行计划 */
{
  kicker: '第 7 步 · 接口',
  title: 'config 里还写着「<span class="hl-a">怎么切开</span>」',
  sub: 'base_model_tp_plan 13 条、base_model_ep_plan 4 条 —— 形状之外，切分策略也归 config 管。这是 L8 的入口。',
  caption: 'attribute_map 与「full_attention 改名」是同一类兼容层：同一个概念两个名字，构造时统一。',
  lang: 'python',
  codeStart: 72,
  code: `    base_model_tp_plan = {
        "layers.*.self_attn.q_b_proj": "colwise",
        "layers.*.self_attn.kv_a_proj_with_mqa": "mla_kv_a_proj",
        "layers.*.self_attn.kv_b_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }`,
  codeNote: '13 条张量并行计划（实测 len(base_model_tp_plan) == 13）。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const tp = W.card({ cc: 0, title: 'base_model_tp_plan · 13 条', style: 'flex:1.4' });
    tp.appendChild(W.table([
      ['self_attn.q_b_proj', 'colwise'],
      ['self_attn.kv_a_proj_with_mqa', '<span class="hl">mla_kv_a_proj</span>'],
      ['self_attn.kv_b_proj', 'colwise'],
      ['self_attn.o_proj', 'rowwise'],
      ['mlp.experts', 'moe_tp_experts'],
    ], { head: ['权重', '切法'] }));
    row.appendChild(tp);

    const ep = W.card({ cc: 1, title: 'base_model_ep_plan · 4 条', style: 'flex:1' });
    ep.appendChild(W.table([
      ['mlp.gate', 'ep_router'],
      ['experts.gate_up_proj', 'grouped_gemm'],
      ['experts.down_proj', 'grouped_gemm'],
      ['experts', 'moe_tp_experts'],
    ], { head: ['权重', '切法'] }));
    row.appendChild(ep);
    viz.appendChild(row);

    const am = W.card({ cc: 4, tint: 4, title: 'attribute_map —— 旧名字的兼容层',
      body: U.el('div', { class: 'mono', style: 'font-size:11.5px;margin-top:4px' },
        [U.el('div', { html: '{<span class="st">"num_local_experts"</span>: <span class="st">"n_routed_experts"</span>}' })]) });
    viz.appendChild(am);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 形状定完了，接下来是「怎么放到多张卡上」</span>';
    tl.at(2000, () => {
      tp.classList.add('ac');
      msg.innerHTML = '<em>13</em> 条张量并行：colwise / rowwise / packed_colwise，加一条自定义 <em>mla_kv_a_proj</em>';
    });
    tl.at(5200, () => {
      tp.classList.remove('ac'); ep.classList.add('ac');
      msg.innerHTML = 'MLA 的压缩 KV 形状特殊 → <em>切法也得特殊</em>；专家则走单独的 EP 计划';
    });
    tl.at(8600, () => {
      ep.classList.remove('ac'); am.classList.add('ac');
      msg.innerHTML = '别名：<em>num_local_experts</em> 读写的其实是 <em>n_routed_experts</em>';
    });
    tl.at(11800, () => {
      am.classList.remove('ac');
      msg.innerHTML = '<span class="cm">// config = 形状 + 排班 + 校验 + 切分计划，四件事一个文件</span>';
    });
    tl.at(14200, () => {
      msg.innerHTML = 'tp / ep 计划怎么变成真的切分 → <em>L8</em>；多模态外壳 Glm5NextConfig 拼 text + vision → L2-06';
    });
  },
},

/* ------------------------------------------------- 9 收束 */
{
  kicker: '收束',
  title: '把这一课压成一张表',
  sub: '两条验收点：能从默认值手算 MLA 的 Q/K/V 形状；能解释 head_dim 为什么是 0 而不是 256。',
  caption: '下一课 L0-03：layer_types 的 34 : 11 是怎么排出来的。',
  lang: 'python',
  codeStart: 101,
  code: `    vocab_size: int = 154880
    hidden_size: int = 4096
    intermediate_size: int = 12288
    moe_intermediate_size: int = 2048

    num_hidden_layers: int = 45
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    n_shared_experts: int = 1
    n_routed_experts: int = 288
    routed_scaling_factor: float = 2.5`,
  codeNote: '回到开头：现在这 11 行里的每一个数，你都能说出它变成了哪个形状。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap10 vizgrow hstart' });
    wrap.appendChild(viz);

    const tb = W.table([
      ['宽度', '<span class="hl">4096</span>', 'hidden_size —— 残差流宽度'],
      ['MLA 压缩', '<span class="hl">1536 / 512</span>', 'q_lora_rank 压缩 Q；kv_lora_rank 压缩 KV（K 与 V 共用）'],
      ['头内维度', '<span class="hl">0 + 256</span>', 'qk_head_dim = qk_rope_head_dim + qk_nope_head_dim'],
      ['Q / KV 展开', '<span class="hl">16384 / 32768</span>', '64×256 与 64×(256+256)'],
      ['head_dim', '<span class="hlbad">0</span>', '被强制 = qk_rope_head_dim → DSA 层完全没有 RoPE'],
      ['MLP 排班', '<span class="hl">3 : 42</span>', 'min(3, N) 个 dense + (N-3) 个 sparse'],
      ['校验', '<span class="hl">5 处 raise</span>', '四道形状契约 + 一道 NoPE 断言'],
    ], { head: ['量', '实测值', '一句话'] });
    viz.appendChild(tb);
    const rows = Array.from(tb.querySelectorAll('tbody tr'));

    const ex = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const e1 = W.exercise(
      '默认 config 下，<code class="inl">q_b_proj</code> 的权重形状是多少？一次注意力的分数矩阵最后一维是多少？',
      'Q 先压到 <b>q_lora_rank = 1536</b>，再展开成 <b>64 × qk_head_dim(256) = 16384</b>，'
      + '所以 <code class="inl">q_b_proj</code> 是 <b>Linear(1536, 16384)</b>（不是直接从 4096 投到 16384）。'
      + '<br>分数矩阵形状 <code class="inl">(batch, 64, seq, seq)</code>，缩放系数 '
      + '<code class="inl">1/√256 = 1/16</code>。');
    const e2 = W.exercise(
      '如果 checkpoint 的 <code class="inl">config.json</code> 里写着 <code class="inl">head_dim: 256</code>，加载后 head_dim 是多少？为什么？',
      '<b>0</b>。<code class="inl">__post_init__</code> 先 <code class="inl">kwargs.pop("head_dim", None)</code> 丢掉传入值，'
      + '再 <code class="inl">self.head_dim = self.qk_rope_head_dim</code>。'
      + '实测 <code class="inl">Glm5NextTextConfig(head_dim=256).head_dim</code> → <b>0</b>。'
      + '<br>原因是命名约定：本仓库用 <code class="inl">head_dim</code> 表示「RoPE 维度」，DSA 层是 NoPE。'
      + '真正的头宽是 <code class="inl">qk_head_dim = 256</code>。');
    e1.style.flex = '1'; e2.style.flex = '1';
    ex.appendChild(e1); ex.appendChild(e2);
    viz.appendChild(ex);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 七行汇总，逐行点亮</span>';
    rows.forEach((r, i) => {
      tl.at(1200 + i * 1700, () => {
        rows.forEach((x, k) => x.classList.toggle('ac', k === i));
        msg.innerHTML = '<em>' + (i + 1) + '/7</em> ' + String(r.cells[2].textContent);
      });
    });
    tl.at(13500, () => {
      rows.forEach(r => r.classList.remove('ac'));
      msg.innerHTML = '★ config 默认值 = 架构：<em>4096</em> 宽、<em>64</em> 头、每头 <em>256</em> 维、<em>head_dim = 0</em>';
    });
    tl.at(16200, () => {
      msg.innerHTML = '<span class="cm">// 下一课 L0-03：34 层 KDA + 11 层 MLA 是怎么排出来的</span>';
    });
  },
},

];
