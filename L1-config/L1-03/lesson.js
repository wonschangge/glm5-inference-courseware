/* ==========================================================================
   L1-03 · 视觉配置与顶层嵌套
   --------------------------------------------------------------------------
   覆盖：models/glm5_next/configuration_glm5_next.py（1 个文件 / 321 行）
   目标：看完能手算 336x336 图像产生多少个视觉 token（144），说清 patch_size 与
         spatial_merge_size 各管什么，并解释顶层 config 如何用 sub_configs 把两个子配置
         缝起来、扁平 checkpoint 的兼容分支为什么必须存在。
   排版基准：#visual 可视区 = 994 x 662.7 舞台像素（无头浏览器实测）。
   ========================================================================== */
'use strict';

/* ---- 实测常量（.venv 里跑 Glm5NextVisionConfig / Glm5NextConfig / Glm5NextImageProcessor 得到） ---- */
const V = {
  depth: 24, hidden: 1024, heads: 16,
  patch: 14, merge: 2, image: 336,
  grid: 24, patches: 576, tokens: 144, pxPerTok: 28,
  outHidden: 1536, projMid: 10240, interm: 4096, patchDim: 1176,
  minTok: 16, maxTok: 8000,
  imgTok: 154854, vidTok: 154855,
  imgStart: 154830, imgEnd: 154831, vidStart: 154832, vidEnd: 154833,
  vocab: 154880, pad: 154820, flatLayers: 45,
};

const SCENES = [

/* ------------------------------------------------- 1 全景：三个类 */
{
  kicker: 'L1 · 配置与分派',
  title: '一个文件，<span class="hl-a">三个 config 类</span>',
  sub: 'L1-02 已经把文本那一半讲透了；本课只加两块新东西 —— 视觉塔的形状，和把两半缝在一起的顶层容器。',
  caption: '三个类共用一个 checkpoint 锚点：zai-org/GLM-5.3-Flash。',
  lang: 'python',
  codeStart: 27,
  code: `@auto_docstring(checkpoint="zai-org/GLM-5.3-Flash")
@strict
class Glm5NextTextConfig(PreTrainedConfig):`,
  codeNote: '第一个类：文本主干配置（45 层 / 4096 宽 / 64 头），逐字段见 L1-02。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);
    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '321 行，三个公开类（__all__ 也只列了三个）' }));
    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const defs = [
      { n: 'Glm5NextTextConfig', at: 'L29', d: '文本主干：45 层 / 4096 宽 / 64 头', m: 'model_type = "glm5_next_text"', c: 0, tag: 'L1-02 已讲' },
      { n: 'Glm5NextVisionConfig', at: 'L232', d: '视觉塔：24 层 / 1024 宽 / 16 头', m: 'model_type = "glm5_next_vision"', c: 1, tag: '本课新增' },
      { n: 'Glm5NextConfig', at: 'L269', d: '顶层容器：两个子配置 + 六个 token 编号', m: 'model_type = "glm5_next"', c: 2, tag: '本课新增' },
    ];
    const cards = defs.map(d => {
      const c = W.card({ cc: d.c, tint: d.c, title: U.esc(d.n), style: 'flex:1' });
      c.appendChild(U.el('div', { class: 'cs', text: d.at + ' · ' + d.d }));
      c.appendChild(U.el('div', { class: 'mono', style: 'font-size:10.5px;color:var(--ink-dim);margin-top:6px;overflow-wrap:anywhere', text: d.m }));
      c.appendChild(U.el('div', { class: 'chip c' + d.c, style: 'margin-top:8px', html: '<i class="sw sw-c' + d.c + '"></i>' + d.tag }));
      row.appendChild(c); return c;
    });
    viz.appendChild(row);

    const tree = U.el('div', { class: 'card cc2', style: 'padding:10px 13px;width:100%' });
    tree.appendChild(U.el('div', { class: 'klabel', text: '嵌套关系：sub_configs 是唯一的那根缝线' }));
    const line = U.el('div', { class: 'row gap8 center wrap', style: 'width:100%' });
    line.innerHTML = U.chip('Glm5NextConfig', 2)
      + '<span class="dim mono" style="font-size:12px">&#8835;</span>'
      + U.chip('vision_config &rarr; Glm5NextVisionConfig', 1)
      + U.chip('text_config &rarr; Glm5NextTextConfig', 0);
    tree.appendChild(line);
    viz.appendChild(tree);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    cards.forEach(c => { c.style.opacity = '.3'; });
    msg.innerHTML = '<span class="cm">// L1-02 讲完了第一个类；本课补上后两个</span>';
    tl.at(1800, () => {
      cards[0].style.opacity = '1';
      msg.innerHTML = '回顾 <span class="fn">L1-02</span>：文本配置的 45 层 / 4096 宽 / 64 头，是本文件里唯一讲过的一块';
    });
    tl.at(5000, () => {
      cards[1].style.opacity = '1'; cards[1].classList.add('ac');
      msg.innerHTML = '新增第一块：<em>Glm5NextVisionConfig</em> —— 一台「像素 &rarr; token」的换算器';
    });
    tl.at(8200, () => {
      cards[1].classList.remove('ac');
      cards[2].style.opacity = '1'; cards[2].classList.add('ac');
      msg.innerHTML = '新增第二块：<em>Glm5NextConfig</em> —— 一个只装子配置与 token 编号的容器';
    });
    tl.at(11400, () => {
      cards[2].classList.remove('ac'); tree.classList.add('ac');
      msg.innerHTML = '<span class="fn">sub_configs</span> 把两半缝进顶层：键名既是字段名，也是 config.json 里的键名';
    });
    tl.at(14300, () => {
      tree.classList.remove('ac');
      msg.innerHTML = '★ 先给答案：336 × 336 的图 = <em>144</em> 个视觉 token。第三幕手算给你看';
    });
  },
},

/* ------------------------------------------------- 2 视觉塔的规模 */
{
  kicker: '视觉塔 · 规模',
  title: '视觉塔：<span class="hl-b">24 层</span>、1024 宽、16 头',
  sub: 'Glm5NextVisionConfig 的文档字符串只列了 3 个字段 —— 那是它真的想让用户调的东西，其余都是规模。',
  caption: 'default_rope_type = "axial" 是本文件里唯一一句「视觉塔的位置编码与文本主干不同」的声明。',
  lang: 'python',
  codeStart: 230,
  code: `@auto_docstring(checkpoint="zai-org/GLM-5.3-Flash")
@strict
class Glm5NextVisionConfig(PreTrainedConfig):
    r"""
    out_hidden_size (\`int\`, *optional*, defaults to 1536):
        The output hidden size of the vision model.
    projection_intermediate_size (\`int\`, *optional*, defaults to 10240):
        The projection_intermediate_size size for the vision patch merger.
    swiglu_limit (\`float\`, *optional*, defaults to 10.0):
        Clamp limit applied to the vision SwiGLU gate/up projections.
    """

    model_type = "glm5_next_vision"
    base_config_key = "vision_config"
    default_rope_type = "axial"
    attribute_map = {"num_attention_heads": "num_heads"}`,
  codeNote: '类头 + 文档字符串 + 四个身份字符串，全部在 L230–L245。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);
    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const row = U.el('div', { class: 'row gap20 center', style: 'width:100%' });
    const left = U.el('div', { class: 'col gap6 center', style: 'width:280px;flex:none' });
    left.appendChild(U.el('div', { class: 'klabel', text: 'depth = 24 层，每层 1024 宽' }));
    const bars = [];
    for (let i = 0; i < V.depth; i++) {
      const b = U.el('div', {
        style: 'width:250px;height:9px;flex:none;border-radius:3px;border:1px solid rgba(167,139,250,.30);'
             + 'background:rgba(167,139,250,.07);transition:background .34s var(--ease-out),transform .34s var(--ease-out)',
      });
      left.appendChild(b); bars.push(b);
    }
    left.appendChild(U.el('div', { class: 'mono', style: 'font-size:10.5px;color:var(--ink-faint);text-align:center', text: 'hidden_size = 1024 · num_heads = 16' }));
    row.appendChild(left);

    const right = U.el('div', { class: 'col gap10', style: 'flex:1;min-width:0' });
    right.appendChild(U.el('div', { class: 'klabel', text: '四个身份字符串 —— 全部在 L242–L245' }));
    const tbCard = W.card({ cc: 1, style: 'padding:8px 11px' });
    tbCard.appendChild(W.table([
      ['model_type', '<span class="hl2">glm5_next_vision</span>', 'AutoConfig 注册表的键（L1-05）'],
      ['base_config_key', '<span class="hl2">vision_config</span>', '在顶层文件里住在哪个键下'],
      ['default_rope_type', '<span class="hl2">axial</span>', '高 / 宽两套频率（L2-07）'],
      ['attribute_map', '<span class="hl2">num_attention_heads</span> &rarr; <span class="hl2">num_heads</span>', '别名：两个名字同一个值'],
    ], { head: ['字段', '值', '管什么'] }));
    right.appendChild(tbCard);

    const cmp = U.el('div', { class: 'row gap10', style: 'width:100%;align-items:stretch' });
    const cA = W.card({ cc: 1, title: '视觉塔', sub: '<code class="inl">attention_bias = True</code>', style: 'flex:1' });
    const cB = W.card({ cc: 0, title: '文本主干', sub: '<code class="inl">attention_bias = False</code>', style: 'flex:1' });
    cmp.appendChild(cA); cmp.appendChild(cB);
    right.appendChild(cmp);
    row.appendChild(right);
    viz.appendChild(row);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 24 层，一层一层亮起来</span>';
    bars.forEach((b, i) => {
      tl.at(900 + i * 80, () => {
        b.style.background = 'rgba(167,139,250,.34)';
        b.style.transform = 'translateX(3px)';
      });
    });
    tl.at(4600, () => {
      tbCard.classList.add('ac');
      msg.innerHTML = '四个字符串各管一件事：注册表的键 / 在顶层住哪 / 位置编码 / 别名';
    });
    tl.at(7800, () => {
      msg.innerHTML = '实测：<span class="fn">getattr(v, "num_attention_heads")</span> 读到 <em>16</em> —— 别名指向 <span class="fn">num_heads</span>';
    });
    tl.at(11000, () => {
      cA.classList.add('ac');
      msg.innerHTML = '对比：视觉塔留注意力偏置（<em>True</em>），文本主干不要（<em>False</em>，L132）—— 不是笔误';
    });
    tl.at(14200, () => {
      cA.classList.remove('ac'); tbCard.classList.remove('ac');
      msg.innerHTML = '★ 但真正决定「一个 token 覆盖多大一块图」的，是下一幕那两个数字';
    });
  },
},

/* ------------------------------------------------- 3 ★ 336 -> 576 -> 144 */
{
  kicker: '视觉塔 · 换算',
  title: '★ <span class="hl-a">336 × 336</span> 的图 &rarr; <span class="hl-c">144</span> 个视觉 token',
  sub: '三个数字决定一次换算：patch_size = 14、spatial_merge_size = 2，以及输入的像素尺寸。',
  caption: '验收点一就在这一幕：336 / 14 = 24 → 24 × 24 = 576 patch → 576 / 2² = 144 token。',
  lang: 'python',
  codeStart: 247,
  code: `    depth: int = 24
    hidden_size: int = 1024
    hidden_act: str = "silu"
    attention_bias: bool = True
    attention_dropout: float | int = 0.0
    num_heads: int = 16
    in_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 336
    patch_size: int | list[int] | tuple[int, int] = 14
    rms_norm_eps: float = 1e-05
    spatial_merge_size: int = 2
    temporal_patch_size: int | list[int] | tuple[int, int] = 2`,
  codeNote: '几何字段全在 L247–L258；其中 patch_size 与 spatial_merge_size 是换算的两个因子。',
  duration: 17500,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);
    const viz = U.el('div', { class: 'col gap10 vizgrow hstart' });
    wrap.appendChild(viz);

    const row = U.el('div', { class: 'row gap14 center', style: 'width:100%' });

    /* A: 输入图像 */
    const A = U.el('div', { class: 'col gap6 center', style: 'flex:none' });
    A.appendChild(U.el('div', { class: 'klabel', text: '① 输入图像' }));
    const sq = U.el('div', {
      style: 'width:238px;height:238px;flex:none;border-radius:10px;border:1px solid var(--panel-brd-hi);'
           + 'background:linear-gradient(150deg,rgba(56,189,248,.22),rgba(56,189,248,.05));'
           + 'display:flex;flex-direction:column;align-items:center;justify-content:center;gap:5px;'
           + 'transition:box-shadow .4s var(--ease-out),transform .4s var(--ease-out)',
    });
    sq.innerHTML = '<div class="n big" style="color:#e0f4ff">336 &times; 336</div>'
      + '<div class="mono dim" style="font-size:10.5px">像素 · image_size 的默认值</div>';
    A.appendChild(sq);
    row.appendChild(A);

    const arrow = (t) => {
      const e = U.el('div', { class: 'col center gap4', style: 'flex:none;width:74px' });
      e.innerHTML = '<div class="mono" style="font-size:19px;color:var(--accent)">&#10233;</div>'
        + '<div class="mono" style="font-size:10px;color:var(--c2);text-align:center;line-height:1.35">' + t + '</div>';
      return e;
    };
    row.appendChild(arrow('patch_size<br>= 14'));

    /* B: patch 网格 24x24 */
    const B = U.el('div', { class: 'col gap6 center', style: 'flex:none' });
    B.appendChild(U.el('div', { class: 'klabel', class: 'mono', style: 'font-size:10.5px;color:var(--ink-faint)', text: '② patch 网格 = 24 × 24' }));
    const gB = W.tensor(24, 24, (r, c) => ((r < 2 && c < 2) ? 2 : 0), { cell: 8 });
    B.appendChild(gB);
    row.appendChild(B);

    row.appendChild(arrow('merge_size<br>= 2'));

    /* C: 合并后的 token 网格 12x12 */
    const C = U.el('div', { class: 'col gap6 center', style: 'flex:none' });
    C.appendChild(U.el('div', { class: 'klabel', class: 'mono', style: 'font-size:10.5px;color:var(--ink-faint)', text: '③ 视觉 token = 12 × 12' }));
    const gC = W.tensor(12, 12, () => 4, { cell: 16 });
    C.appendChild(gC);
    row.appendChild(C);
    viz.appendChild(row);

    const stats = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const mkStat = (k, v, u, cc) => {
      const c = W.card({ cc: cc, style: 'flex:1;text-align:center;padding:9px 10px' });
      c.appendChild(U.el('div', { class: 'klabel', text: k }));
      c.appendChild(U.el('div', { class: 'n big', style: 'color:var(--c' + cc + ')', text: v }));
      c.appendChild(U.el('div', { class: 'mono dim', style: 'font-size:10.5px;margin-top:3px', text: u }));
      return c;
    };
    const s1 = mkStat('patch 数', '576', '24 × 24', 0);
    const s2 = mkStat('视觉 token', '144', '576 / 2²', 4);
    const s3 = mkStat('每个 token 覆盖', '28 × 28', '14 × 2 像素一条边', 2);
    [s1, s2, s3].forEach(s => { s.style.opacity = '.28'; stats.appendChild(s); });
    viz.appendChild(stats);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    const cellsB = Array.from(gB.querySelectorAll('.tcell'));
    const cellsC = Array.from(gC.querySelectorAll('.tcell'));
    const group = cellsB.slice(0, 2).concat(cellsB.slice(24, 26));
    cellsB.forEach(c => { c.style.opacity = '.12'; });
    cellsC.forEach(c => { c.style.opacity = '.12'; });
    sq.style.opacity = '.45';

    msg.innerHTML = '<span class="cm">// 从一张图开始</span>';
    tl.at(1600, () => {
      sq.style.opacity = '1'; sq.style.boxShadow = '0 0 0 2px var(--c0), 0 14px 34px -10px var(--c0)';
      msg.innerHTML = '输入 <em>336 &times; 336</em> 像素：这是 <span class="fn">image_size</span> 的默认值，也是训练分辨率';
    });
    tl.at(4800, () => {
      sq.style.boxShadow = 'none';
      /* 逐行扫出：用 transitionDelay 排队，不新增时间轴事件（保持 build 幂等） */
      cellsB.forEach((c, i) => { c.style.transitionDelay = (Math.floor(i / 24) * 40) + 'ms'; c.style.opacity = '1'; });
      s1.style.opacity = '1';
      msg.innerHTML = '<span class="fn">patch_size = 14</span>：336 / 14 = <em>24</em>，于是 24 &times; 24 = <em>576</em> 个 patch';
    });
    tl.at(8200, () => {
      cellsC.forEach((c, i) => { c.style.transitionDelay = (Math.floor(i / 12) * 55) + 'ms'; c.style.opacity = '1'; });
      s2.style.opacity = '1';
      msg.innerHTML = '<span class="fn">spatial_merge_size = 2</span>：2 &times; 2 个相邻 patch 合成 <em>1</em> 个 token &rarr; 576 / 4 = <em>144</em>';
    });
    tl.at(11600, () => {
      group.forEach(c => { c.style.transitionDelay = '0ms'; c.style.transform = 'scale(1.35)'; c.style.zIndex = '3'; });
      s3.style.opacity = '1';
      msg.innerHTML = '被合并的那 4 格（高亮）= <em>28 &times; 28</em> 像素 &rarr; 一个视觉 token 覆盖 28 像素一条边';
    });
    tl.at(14600, () => {
      group.forEach(c => { c.style.transform = 'none'; });
      msg.innerHTML = '<span class="cm">// 实测 get_number_of_image_patches(336, 336) = 576；preprocess 后 image_grid_thw = [[1, 24, 24]]</span>';
    });
  },
},

/* ------------------------------------------------- 4 四个宽度 */
{
  kicker: '视觉塔 · 宽度',
  title: '合并块交付 <span class="hl-c">1536</span> 维，而不是文本的 4096',
  sub: 'vision config 里有四个宽度，分属四个不同位置 —— 混起来就会算错参数量与显存。',
  caption: 'out_hidden_size 是「一个视觉 token 交付时多少维」；它和文本流的 4096 不是同一个宽度。',
  lang: 'python',
  codeStart: 259,
  code: `    out_hidden_size: int = 1536
    intermediate_size: int = 4096
    initializer_range: float = 0.02
    rope_parameters: dict | None = None
    projection_intermediate_size: int = 10240
    swiglu_limit: float = 10.0`,
  codeNote: '输出与投影侧的四个宽度（L259–L264），加一个 SwiGLU 夹紧值。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);
    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '一个 patch 从进塔到交付：宽度在哪些地方变' }));
    const fl = W.flow([
      { t: 'pixel patch', s: U.fmt(V.patchDim) + ' 维', cc: 3 },
      { t: 'patch_embed', s: '&rarr; ' + U.fmt(V.hidden), cc: 1 },
      { t: '24 层', s: U.fmt(V.hidden) + ' 宽', cc: 1 },
      { t: '合并 2&times;2', s: U.fmt(V.hidden) + ' &rarr; ' + U.fmt(V.outHidden), cc: 4 },
      { t: 'merger SwiGLU', s: '中间 ' + U.fmt(V.projMid), cc: 2 },
    ], { style: 'width:100%' });
    viz.appendChild(fl);

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const defs = [
      { k: 'hidden_size', v: '1024', d: '塔内每个 patch 的宽度', c: 1 },
      { k: 'intermediate_size', v: '4096', d: '塔内 FFN 的宽度', c: 4 },
      { k: 'out_hidden_size', v: '1536', d: '合并块交付宽度 = 一个 token 的维数', c: 2 },
      { k: 'projection_intermediate_size', v: '10240', d: '合并块里那层 SwiGLU 的中间宽度', c: 0 },
    ];
    const cards = defs.map(d => {
      const c = W.card({ cc: d.c, style: 'flex:1;padding:9px 11px' });
      c.appendChild(U.el('div', { class: 'mono', style: 'font-size:10.5px;color:var(--c' + d.c + ');overflow-wrap:anywhere', text: d.k }));
      c.appendChild(U.el('div', { class: 'n big', style: 'margin-top:3px', text: d.v }));
      c.appendChild(U.el('div', { class: 'cs', style: 'margin-top:3px', text: d.d }));
      row.appendChild(c); return c;
    });
    viz.appendChild(row);

    const note = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const nA = W.card({ cc: 3, title: 'swiglu_limit = 10.0', sub: '与文本侧的 swiglu_limit 同名同值，但是<b>两份独立字段</b>：一个夹视觉塔，一个夹文本 FFN。', style: 'flex:1' });
    const nB = W.card({ cc: 4, title: '1536 &ne; 4096', sub: '交付宽度是 1536，文本流宽 4096 —— 接上是另一层的事（L2-06 / L2-07），本文件只把数写清楚。', style: 'flex:1' });
    note.appendChild(nA); note.appendChild(nB);
    viz.appendChild(note);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    fl.focus(-1);
    msg.innerHTML = '<span class="cm">// 一个 patch 的宽度之旅</span>';
    tl.at(1600, () => { fl.focus(0); msg.innerHTML = '进塔前：<span class="fn">pixel_values</span> 每行 <em>1176</em> 个数（3 &times; 2 &times; 14 &times; 14，实测）'; });
    tl.at(4800, () => { fl.focus(1); msg.innerHTML = '<span class="fn">patch_embed</span> 把它投到塔内宽度 <em>1024</em>'; });
    tl.at(8000, () => { fl.focus(2); msg.innerHTML = '<em>24</em> 层里宽度不变，一直保持 1024（<span class="fn">depth</span> 只改深度，不改宽度）'; });
    tl.at(11200, () => {
      fl.focus(3); cards[2].classList.add('ac');
      msg.innerHTML = '2 &times; 2 合并时宽度被抬到 <em>1536</em> —— 这就是 <span class="fn">out_hidden_size</span>，一个 token 的交付维数';
    });
    tl.at(14200, () => {
      fl.focus(4); cards[2].classList.remove('ac'); cards[3].classList.add('ac');
      msg.innerHTML = '<span class="fn">projection_intermediate_size = 10240</span>：合并块内部的 SwiGLU 比塔内 FFN 还宽（1536 &rarr; 10240 &rarr; 1536）';
    });
  },
},

/* ------------------------------------------------- 5 ★ 顶层容器 */
{
  kicker: '顶层 · 嵌套',
  title: '★ 顶层 config 是一个<span class="hl-c">容器</span>',
  sub: 'Glm5NextConfig 自己不含任何形状：只有两个子配置字段，和六个特殊 token 的编号。',
  caption: '键名必须逐字一致：顶层靠 sub_configs 往下建，子配置靠 base_config_key 往上找。',
  lang: 'python',
  codeStart: 291,
  code: `    model_type = "glm5_next"
    sub_configs = {"vision_config": Glm5NextVisionConfig, "text_config": Glm5NextTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 154854
    video_token_id: int = 154855
    image_start_token_id: int = 154830
    image_end_token_id: int = 154831
    video_start_token_id: int = 154832
    video_end_token_id: int = 154833
    tie_word_embeddings: bool = False`,
  codeNote: 'sub_configs 一行就是全部的嵌套声明（L292）；六个编号紧跟在两个子配置字段后面。',
  duration: 17500,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap10', style: 'width:100%;height:100%' });
    root.appendChild(wrap);
    const viz = U.el('div', { class: 'col gap8 vizgrow', style: 'width:100%' });
    wrap.appendChild(viz);

    const top = W.card({ cc: 2, tint: 2, style: 'width:420px;padding:9px 12px;text-align:center' });
    top.appendChild(U.el('div', { class: 'mono', style: 'font-size:12.5px;color:#fff', text: 'Glm5NextConfig' }));
    top.appendChild(U.el('div', { class: 'mono dim', style: 'font-size:10.5px;margin-top:3px', text: 'model_type = "glm5_next" · sub_configs 有两个键' }));
    viz.appendChild(top);

    const fork = U.el('div', { style: 'position:relative;width:560px;height:18px;flex:none' });
    fork.innerHTML =
      '<div style="position:absolute;left:50%;top:0;width:1px;height:9px;background:var(--panel-brd-hi)"></div>'
      + '<div style="position:absolute;left:25%;right:25%;top:9px;height:1px;background:var(--panel-brd-hi)"></div>'
      + '<div style="position:absolute;left:25%;top:9px;width:1px;height:9px;background:var(--panel-brd-hi)"></div>'
      + '<div style="position:absolute;right:25%;top:9px;width:1px;height:9px;background:var(--panel-brd-hi)"></div>';
    viz.appendChild(fork);

    const row = U.el('div', { class: 'row gap12 center', style: 'width:100%' });
    const mkSide = (key, cls, cc, lines) => {      const c = W.card({ cc: cc, style: 'width:274px;padding:9px 11px' });
      c.appendChild(U.el('div', { class: 'mono', style: 'font-size:11.5px;color:var(--c' + cc + ')', text: key }));
      c.appendChild(U.el('div', { class: 'cs', text: cls }));
      lines.forEach(t => c.appendChild(U.el('div', { class: 'mono', style: 'font-size:10.5px;color:var(--ink-dim);margin-top:5px;overflow-wrap:anywhere', html: t })));
      return c;
    };
    const vis = mkSide('vision_config', 'Glm5NextVisionConfig · L232', 1, [
      'model_type = <span class="hl2">glm5_next_vision</span>',
      'base_config_key = <span class="hl2">"vision_config"</span>',
      'depth = 24 · patch_size = 14',
    ]);
    const txt = mkSide('text_config', 'Glm5NextTextConfig · L29', 0, [
      'model_type = <span class="hl">glm5_next_text</span>',
      'base_config_key = <span class="hl">"text_config"</span>',
      'num_hidden_layers = 45',
    ]);
    row.appendChild(vis); row.appendChild(txt);
    viz.appendChild(row);

    const back = U.el('div', { class: 'card cc4', style: 'padding:9px 12px;width:100%' });
    back.appendChild(U.el('div', { class: 'klabel', text: '反向也通：子配置靠 base_config_key 找到自己（L141 / L243）' }));
    back.appendChild(U.el('div', {
      class: 'mono', style: 'font-size:11px;color:var(--ink-dim);overflow-wrap:anywhere',
      html: 'Glm5NextVisionConfig.from_pretrained("…/config.json") &nbsp;<span class="hl4">&rarr; 自动下钻到 config.json["vision_config"]</span>',
    }));
    viz.appendChild(back);

    const ids = U.el('div', { class: 'card cc2', style: 'padding:9px 12px;width:100%' });
    ids.appendChild(U.el('div', { class: 'klabel', text: '另外六个编号平铺在顶层（第八幕展开）' }));
    const idRow = U.el('div', { class: 'row gap8 wrap center', style: 'width:100%' });
    idRow.innerHTML = U.chip('image_token_id 154854', 0) + U.chip('video_token_id 154855', 0)
      + U.chip('image_start_token_id 154830', 2) + U.chip('image_end_token_id 154831', 2)
      + U.chip('video_start_token_id 154832', 3) + U.chip('video_end_token_id 154833', 3);
    ids.appendChild(idRow);
    viz.appendChild(ids);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 顶层类的字段只有两种</span>';
    tl.at(1600, () => {
      top.classList.add('ac');
      msg.innerHTML = '两种字段：<em>两个子配置</em>（往下装）+ <em>六个 token 编号</em>（平铺在顶层）';
    });
    tl.at(4800, () => {
      top.classList.remove('ac'); vis.classList.add('ac');
      msg.innerHTML = '第一个键：<span class="fn">vision_config</span> &rarr; <em>Glm5NextVisionConfig</em>（24 层的视觉塔）';
    });
    tl.at(8000, () => {
      vis.classList.remove('ac'); txt.classList.add('ac');
      msg.innerHTML = '第二个键：<span class="fn">text_config</span> &rarr; <em>Glm5NextTextConfig</em>（45 层的文本主干）';
    });
    tl.at(11200, () => {
      txt.classList.remove('ac'); back.classList.add('ac');
      msg.innerHTML = '反过来也行：同一个 config.json，子类靠 <span class="fn">base_config_key</span> 钻进去取自己那一层';
    });
    tl.at(14500, () => {
      back.classList.remove('ac');
      msg.innerHTML = '★ 两个键名在两处各写了一遍（L292 的字典键、L141/L243 的 base_config_key）—— 必须逐字对齐';
    });
  },
},

/* ------------------------------------------------- 6 ★ 扁平 checkpoint */
{
  kicker: '顶层 · 兼容',
  title: '★ 扁平 checkpoint 的<span class="hl-a">向后兼容分支</span>',
  sub: '早期只有纯文本权重，config.json 里没有 text_config 这一层 —— 那条 elif 就是为它准备的。',
  caption: 'forward（转交），不是 move（搬走）：实测转交之后字段在两个位置同时存在。',
  lang: 'python',
  codeStart: 305,
  code: `    def __post_init__(self, **kwargs):
        if isinstance(self.text_config, dict):
            self.text_config = self.sub_configs["text_config"](**self.text_config)
        elif self.text_config is None:
            # Flat (text-only) GLM-5.3-Flash checkpoints store the text fields at the
            # top level; forward them so \`text_config\` is populated for BC.
            self.text_config = self.sub_configs["text_config"](**kwargs)`,
  codeNote: '三个分支：字典 → 建对象；None → 转交 kwargs；已是对象 → 不动。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap10', style: 'width:100%;height:100%' });
    root.appendChild(wrap);
    const viz = U.el('div', { class: 'col gap8 vizgrow hstart' });
    wrap.appendChild(viz);

    const mkJson = (title, cc, body, badge) => {
      const c = W.card({ cc: cc, style: 'flex:1;padding:9px 11px' });
      c.appendChild(U.el('div', { class: 'ct', html: title }));
      if (badge) c.appendChild(U.el('div', { class: 'chip c' + cc, style: 'margin-bottom:6px', html: badge }));
      c.appendChild(U.el('div', { class: 'mono', style: 'font-size:10.5px;line-height:1.55;color:var(--ink-dim);white-space:pre-wrap;overflow-wrap:anywhere', text: body }));
      return c;
    };
    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const nested = mkJson('① 嵌套形态（现在的 checkpoint）', 0,
      '{\n  "model_type": "glm5_next",\n  "text_config": {\n    "num_hidden_layers": 45,\n    "hidden_size": 4096\n  },\n  "vision_config": { "depth": 24 }\n}',
      '<i class="sw sw-c0"></i>两个子配置各占一个键');
    const flat = mkJson('② 扁平形态（早期纯文本权重）', 2,
      '{\n  "model_type": "glm5_next",\n  "num_hidden_layers": 45,\n  "hidden_size": 4096,\n  "layer_types": [ … ],\n  "vocab_size": 154880\n}',
      '<i class="sw sw-c2"></i>文本字段摊在顶层');
    row.appendChild(nested); row.appendChild(flat);
    viz.appendChild(row);

    const join = U.el('div', { class: 'row gap10 center', style: 'width:100%' });
    join.innerHTML = '<div class="mono dim" style="font-size:11px;flex:none">两条路汇到同一行：</div>'
      + '<code class="inl">self.text_config = self.sub_configs["text_config"](**kwargs)</code>';
    viz.appendChild(join);

    const meas = W.card({ cc: 4, style: 'width:100%;padding:9px 12px' });
    meas.appendChild(U.el('div', { class: 'klabel', text: '实测：扁平 config.json 读进内存之后' }));
    meas.appendChild(U.el('div', {
      class: 'mono', style: 'font-size:11.5px;line-height:1.7;color:var(--ink-dim)',
      html: 'c.text_config.num_hidden_layers &nbsp;=&nbsp; <span class="hl4">45</span><br>'
          + 'c.num_hidden_layers &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=&nbsp; <span class="hl4">45</span>'
          + ' &nbsp;<span class="cm">// 转交 ≠ 搬走：两处都在</span>',
    }));
    viz.appendChild(meas);

    const warn = W.card({ cc: 3, style: 'width:100%;padding:9px 12px' });
    warn.appendChild(U.el('div', { class: 'ct', html: '会咬人的地方（实测）' }));
    warn.appendChild(U.el('div', {
      class: 'cs',
      html: '<code class="inl">Glm5NextConfig(depth=48, patch_size=16)</code> 建出来：'
          + '<code class="inl">vision_config.depth</code> 仍是 <b>24</b>、<code class="inl">patch_size</code> 仍是 <b>14</b>'
          + ' —— kwargs 被<strong>整包</strong>转交给 text_config，视觉字段不会被识别，而且<b>不报错</b>。'
          + '要给视觉塔换尺寸只能走嵌套：<code class="inl">Glm5NextConfig(vision_config={"depth": 48})</code>。',
    }));
    viz.appendChild(warn);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 同一个 __post_init__，两种 checkpoint</span>';
    tl.at(1600, () => {
      nested.classList.add('ac');
      msg.innerHTML = '嵌套形态：<span class="fn">text_config</span> 是字典，走第一个分支 &rarr; <em>cls(**dict)</em>';
    });
    tl.at(4800, () => {
      nested.classList.remove('ac'); flat.classList.add('ac');
      msg.innerHTML = '扁平形态：扁平文件里<em>根本没有</em> text_config 这个键 &rarr; 字段是 None，走第二个分支';
    });
    tl.at(8000, () => {
      join.style.opacity = '1';
      msg.innerHTML = '第二个分支做的事只有一件：把顶层剩下的 <span class="fn">**kwargs</span> 原样交给 <em>Glm5NextTextConfig</em>';
    });
    tl.at(11400, () => {
      meas.classList.add('ac');
      msg.innerHTML = '实测：转交之后 <em>两个位置同时是 45</em> —— 注释里的关键词是 forward，不是 move';
    });
    tl.at(14600, () => {
      meas.classList.remove('ac'); warn.classList.add('ac');
      msg.innerHTML = '★ 它是整包转交、不按前缀派发：扁平文件里的视觉字段会被 text_config 收下，视觉塔默默用默认值';
    });
  },
},

/* ------------------------------------------------- 7 视觉那一半 */
{
  kicker: '顶层 · 不对称',
  title: '视觉那一半：形状相似，但<span class="hl-b">少一条回退</span>',
  sub: 'elif self.vision_config is None: 后面是一对空括号 —— 直接建一份全默认的视觉配置。',
  caption: '文本配置「可从顶层反向填充」，视觉配置「必须显式嵌套」—— 这份不对称本身就是文档。',
  lang: 'python',
  codeStart: 313,
  code: `        if isinstance(self.vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        super().__post_init__(**kwargs)`,
  codeNote: '视觉侧只有两条分支：字典 → 建对象；None → 建默认塔。没有 kwargs 回退。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);
    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '同一个 __post_init__ 里，两半的分支对照' }));
    const tbCard = W.card({ cc: 1, style: 'padding:9px 11px' });
    tbCard.appendChild(W.table([
      ['<code class="inl">dict</code>', '<code class="inl">cls(**dict)</code>', '<code class="inl">cls(**dict)</code>', '配置文件形态：两半对称'],
      ['<code class="inl">None</code>', '<span class="hl">cls(**kwargs)</span>', '<span class="hl2">cls()</span>', '两条路在这里分岔'],
      ['<code class="inl">已是对象</code>', '原样保留', '原样保留', '代码手搓的形态'],
    ], { head: ['传进来的值', 'text_config', 'vision_config', '备注'] }));
    viz.appendChild(tbCard);

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cA = W.card({ cc: 2, style: 'flex:1' });
    cA.appendChild(U.el('div', { class: 'ct', html: '为什么视觉侧没有 kwargs 回退' }));
    cA.appendChild(U.el('div', { class: 'cs', html: '扁平 checkpoint 是<b>纯文本</b>的，里面根本不存在视觉字段 —— 「回退」无从谈起；而多模态模型<b>必须有</b>视觉塔，所以这里给的是「默认塔」而不是「空」。' }));
    const cB = W.card({ cc: 4, style: 'flex:1' });
    cB.appendChild(U.el('div', { class: 'ct', html: 'super().__post_init__(**kwargs)' }));
    cB.appendChild(U.el('div', { class: 'cs', html: '<code class="inl">@strict</code> 的类型校验、dtype 归一化都在父类里发生（L1-01）。这也解释了上一幕的实测：kwargs 一路上交，沿途被记成了对象属性。' }));
    row.appendChild(cA); row.appendChild(cB);
    viz.appendChild(row);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 三条入口，两半各两条</span>';
    tl.at(1600, () => {
      tbCard.classList.add('ac');
      msg.innerHTML = '字典分支两半<em>完全对称</em>：都是 <span class="fn">cls(**dict)</span>';
    });
    tl.at(4800, () => {
      msg.innerHTML = '<span class="fn">None</span> 分支才是不对称的地方：文本侧转交 <em>**kwargs</em>，视觉侧只给一对空括号';
    });
    tl.at(8000, () => {
      cA.classList.add('ac');
      msg.innerHTML = '为什么？扁平 checkpoint 里没有视觉字段可回退，但多模态模型必须有塔 &rarr; <em>默认塔</em>';
    });
    tl.at(11200, () => {
      cA.classList.remove('ac'); cB.classList.add('ac');
      msg.innerHTML = '<span class="fn">super().__post_init__(**kwargs)</span> 是两条路的共同出口：校验都在父类（L1-01）';
    });
    tl.at(14200, () => {
      cB.classList.remove('ac'); tbCard.classList.remove('ac');
      msg.innerHTML = '★ 一句话记住这份不对称：文本可反向填充，视觉必须显式嵌套';
    });
  },
},

/* ------------------------------------------------- 8 六个 token 编号 */
{
  kicker: '顶层 · 编号',
  title: '六个编号：<span class="hl-a">占位符</span>与<span class="hl-c">跨度标记</span>的分工',
  sub: '336×336 的图要写 144 个占位符；而「有几张图、是图还是视频」靠的是 start / end 标记。',
  caption: '实测：image_end_token_id 与 video_token_id 在当前建模代码里引用数为 0 —— 是预留席位。',
  lang: 'python',
  codeStart: 267,
  code: `@auto_docstring(checkpoint="zai-org/GLM-5.3-Flash")
@strict
class Glm5NextConfig(PreTrainedConfig):
    r"""
    image_token_id (\`int\`, *optional*, defaults to 154854):
        The image token index to encode the image prompt.
    video_token_id (\`int\`, *optional*, defaults to 154855):
        The video token index to encode the video prompt.
    image_start_token_id (\`int\`, *optional*, defaults to 154830):
        The image start token index to encode the start of image.
    image_end_token_id (\`int\`, *optional*, defaults to 154831):
        The image end token index to encode the end of image.
    video_start_token_id (\`int\`, *optional*, defaults to 154832):
        The video start token index to encode the start of video.
    video_end_token_id (\`int\`, *optional*, defaults to 154833):
        The video end token index to encode the end of video.`,
  codeNote: '六个编号的文档字符串（L267–L282）：它们全部落在词表尾部 154830–154855。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);
    const viz = U.el('div', { class: 'col gap10 vizgrow', style: 'width:100%' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '一段图像输入在 input_ids 里的长相' }));
    const strip = U.el('div', { class: 'row gap12 center', style: 'width:100%' });
    const mkTag = (name, id, cc) => {
      const e = U.el('div', { class: 'card cc' + cc, style: 'padding:7px 10px;text-align:center;flex:none' });
      e.appendChild(U.el('div', { class: 'mono', style: 'font-size:10.5px;color:var(--c' + cc + ')', text: name }));
      e.appendChild(U.el('div', { class: 'n sm', style: 'margin-top:3px', text: id }));
      return e;
    };
    const tagS = mkTag('<image_start>', String(V.imgStart), 2);
    strip.appendChild(tagS);

    const mid = U.el('div', { class: 'col gap5 center', style: 'flex:none' });
    const gT = W.tensor(12, 12, () => 0, { cell: 8 });
    mid.appendChild(gT);
    mid.appendChild(U.el('div', { class: 'mono', style: 'font-size:10.5px;color:var(--c0);text-align:center', text: '144 个占位符（336×336 的图）' }));
    strip.appendChild(mid);

    const tagE = mkTag('<image_end>', String(V.imgEnd), 2);
    strip.appendChild(tagE);
    viz.appendChild(strip);

    const tbCard = W.card({ cc: 1, style: 'padding:9px 11px;width:100%' });
    tbCard.appendChild(W.table([
      ['image_token_id', '154854', '<span class="hl">占位符：要填几个</span>，前向时被换成视觉向量', '2 处'],
      ['image_start_token_id', '154830', '跨度起点：<span class="hl3">数出有几张图</span>', '2 处'],
      ['image_end_token_id', '154831', '跨度终点（当前无消费者）', '<span class="hlbad">0 处</span>'],
      ['video_token_id', '154855', '视频占位符（当前无消费者）', '<span class="hlbad">0 处</span>'],
      ['video_start_token_id', '154832', '视频跨度开：区分模态', '4 处'],
      ['video_end_token_id', '154833', '视频跨度闭：累积和判据', '4 处'],
    ], { head: ['字段', '值', '角色', '建模代码引用'] }));
    viz.appendChild(tbCard);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    const cells = Array.from(gT.querySelectorAll('.tcell'));
    cells.forEach(c => { c.style.opacity = '.14'; });
    tagE.style.opacity = '.3'; tbCard.style.opacity = '.35';

    msg.innerHTML = '<span class="cm">// 一张图在 input_ids 里的三段结构</span>';
    tl.at(1600, () => {
      msg.innerHTML = '第一段：<span class="fn">image_start_token_id</span> = <em>154830</em> —— 一个跨度标记';
    });
    tl.at(4800, () => {
      cells.forEach((c, i) => { c.style.transitionDelay = (Math.floor(i / 12) * 45) + 'ms'; c.style.opacity = '1'; });
      msg.innerHTML = '第二段：<em>144</em> 个 <span class="fn">image_token_id</span> = 154854 —— 一个占位符对应一个视觉 token';
    });
    tl.at(8000, () => {
      tagE.style.opacity = '1';
      msg.innerHTML = '第三段：<span class="fn">image_end_token_id</span> = <em>154831</em> —— 与 start 配成一对，用来数出有几张图';
    });
    tl.at(11400, () => {
      tbCard.style.opacity = '1'; tbCard.classList.add('ac');
      msg.innerHTML = '实测引用次数（modeling_glm5_next.py）：<em>image_end</em> 与 <em>video_token_id</em> 都是 <span class="hlbad">0 处</span> —— 预留席位';
    });
    tl.at(14600, () => {
      tbCard.classList.remove('ac');
      msg.innerHTML = '★ 分工三重：占位符决定<em>填几个</em>，start 决定<em>有几张</em>，video 的 start/end 决定<em>哪种模态</em>';
    });
  },
},

/* ------------------------------------------------- 9 收束 */
{
  kicker: '收束',
  title: '把视觉配置与顶层嵌套压成一张表',
  sub: '两个验收点在这里各就各位：144 个视觉 token，以及占位符与跨度标记的分工。',
  caption: '下一课 L1-04：配置怎么被序列化、继承，以及文档字符串是怎么被机械改写的。',
  lang: 'python',
  codeStart: 321,
  code: `__all__ = ["Glm5NextConfig", "Glm5NextTextConfig", "Glm5NextVisionConfig"]`,
  codeNote: '文件最后一行的对外清单：三个类，多一个都没有。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap10', style: 'width:100%;height:100%' });
    root.appendChild(wrap);
    const viz = U.el('div', { class: 'col gap10 vizgrow hstart', style: 'width:100%' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '本课的五个要点（数字全部来自实测）' }));
    const tbRows = [
      ['Glm5NextVisionConfig', 'L232', '24 层 / 1024 宽 / 16 头；patch 14 · merge 2 &rarr; 每 token 覆盖 28×28 像素', '本课'],
      ['336×336 &rarr; 144', '实测', '(336/14)&sup2; = 576 patch &rarr; ÷2&sup2; = 144 token；pixel_values (576, 1176)', '验收点 1'],
      ['sub_configs', 'L292', '往下建用 sub_configs，往上找用 base_config_key，键名必须逐字一致', '本课'],
      ['扁平兼容分支', 'L308', 'kwargs 整包转交给 text_config（转交 ≠ 搬走）；视觉侧没有回退', '本课'],
      ['六个 token 编号', 'L297', 'image_token_id 填几个；start 数几张；video span 分模态', '验收点 2'],
    ];
    viz.appendChild(W.table(tbRows, { head: ['要点', '位置', '一句话', '状态'] }));

    const ex = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:flex-start' });
    const exA = W.exercise(
      '一张 <b>336×336</b> 的图产生多少视觉 token？如果一次喂 <b>8</b> 张这样的图，'
      + '<code class="inl">image_token_id</code> 占位符一共要写多少个？',
      '<b>144</b> 个。代数只有三步：336 / 14 = <b>24</b> &rarr; 24 &times; 24 = <b>576</b> 个 patch &rarr; '
      + '576 / 2&sup2; = <b>144</b> 个视觉 token。<br>'
      + '8 张图：144 &times; 8 = <b>1152</b> 个占位符（patch 张量是 576 &times; 8 = 4608 行）。'
      + '<br>注意：占位符个数必须<b>精确</b>等于视觉特征个数，否则前向直接抛 '
      + '<code class="inl">Image features and image tokens do not match</code>。');
    exA.style.flex = '1';
    ex.appendChild(exA);
    const exB = W.exercise(
      '<code class="inl">image_token_id</code> 与 <code class="inl">image_start/end_token_id</code> 各管什么？',
      '<code class="inl">image_token_id</code> 是<b>占位符</b>：一个占位符对应一个视觉 token，前向时被整体换成向量 —— '
      + '它决定「填几个」。<code class="inl">image_start_token_id</code> / <code class="inl">image_end_token_id</code> 是'
      + '<b>跨度标记</b>：标记一张图从哪开始到哪结束，用来「数出有几张」，不参与填向量。<br>'
      + '另外两个事实（实测）：图片与视频<b>共用</b>同一个占位符，靠 video 的 start/end 跨度分模态；'
      + '而 <code class="inl">image_end_token_id</code> 与 <code class="inl">video_token_id</code> 在当前建模代码里'
      + '引用数为 <b>0</b> —— 属于预留席位。');
    exB.style.flex = '1';
    ex.appendChild(exB);
    viz.appendChild(ex);

    const nx = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    nx.appendChild(W.card({ cc: 0, style: 'flex:1', title: '回顾 L1-02',
      sub: '文本配置的 45 层 / 4096 宽 / 64 头；它与本课的视觉配置合起来，才是完整的一份 <code class="inl">Glm5NextConfig</code>。' }));
    nx.appendChild(W.card({ cc: 1, style: 'flex:1', title: '下一课 L1-04',
      sub: '配置的序列化、继承与文档生成：<code class="inl">to_dict()</code> 写出什么、<code class="inl">@auto_docstring</code> 在类定义时改了什么。' }));
    viz.appendChild(nx);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 从三个类回到这一行</span>';
    tl.at(1600, () => { msg.innerHTML = '这个文件对外只有三个名字：<em>Glm5NextConfig</em> / <em>Glm5NextTextConfig</em> / <em>Glm5NextVisionConfig</em>'; });
    tl.at(4800, () => { msg.innerHTML = '视觉那一半 = 一台换算器：<span class="fn">patch_size</span> 切块，<span class="fn">spatial_merge_size</span> 合并'; });
    tl.at(8000, () => { msg.innerHTML = '336 &times; 336 &rarr; <em>576</em> patch &rarr; <em>144</em> token &rarr; 每个覆盖 <em>28 &times; 28</em> 像素'; });
    tl.at(11200, () => { msg.innerHTML = '顶层那一半 = 一个容器：<span class="fn">sub_configs</span> 往下建，<span class="fn">base_config_key</span> 往上找'; });
    tl.at(14500, () => { msg.innerHTML = '<span class="cm">// 下一课 L1-04：配置怎么被序列化、继承，文档字符串怎么被改写</span>'; });
  },
},

];
