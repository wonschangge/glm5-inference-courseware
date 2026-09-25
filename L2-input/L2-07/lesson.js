/* ==========================================================================
   L2-07 · 视觉塔：从 336x336 像素到 27 个 token
   --------------------------------------------------------------------------
   覆盖：models/glm5_next/modeling_glm5_next.py, backbone_utils.py,
         utils/backbone_utils.py（3 个文件 / 2850 行）
   目标：看完能手算一张图产出多少视觉 token（144），并说清轴向 RoPE 与一维 RoPE 的差别。
   排版基准：#visual 可视区 = 994 x 662.7 舞台像素（无头浏览器实测）。
   ★ 实测订正：课程标题里的「27 个 token」与源码不符 —— 实测是 144。见第 3 幕。
   ========================================================================== */
'use strict';

/* 本课用到的实测常量（_data/recon/probe_l207.py 跑出，脚本不计入覆盖率） */
const V = {
  imageSize: 336, patch: 14, merge: 2, temporal: 2,
  grid: 24, patches: 576, tokens: 144,
  hidden: 1024, depth: 24, heads: 16, headDim: 64, outHidden: 1536,
  freqs: 16, ctx: 10240, imageTokenId: 154854, swigluLimit: 10.0,
};

const SCENES = [

/* ============================================================ 1 全景 */
{
  kicker: 'L2 · 输入流水线 · 全景',
  title: '一张图进塔，<span class="hl-a">144</span> 个 token 出来',
  sub: '336x336 的 RGB 图 → 24x24 个 patch 位置（576）→ 每 2x2 合并成 1 个 → 144 个 1536 维向量。',
  caption: '★ 课程标题写的是「27 个 token」—— 与源码不符，实测是 144，证据在第三幕。',
  lang: 'python',
  codeStart: 1857,
  code: `        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size, kwargs=kwargs)
        cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid_thw, self.config, kwargs=kwargs)

        hidden_states = self.patch_embed(hidden_states)
        position_embeddings = self.rotary_pos_emb(hidden_states, position_ids)`,
  codeNote: 'Glm5NextVisionModel.forward 的前 5 行：整座塔只有四件事。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const fl = W.flow([
      { t: '像素', s: '336x336x3', cc: 0 },
      { t: 'patchify', s: '28 的整数倍', cc: 4 },
      { t: 'patch 位置', s: '24x24 = 576', cc: 2 },
      { t: '24 层视觉塔', s: '非因果 · 轴向 RoPE', cc: 1 },
      { t: '视觉 token', s: '(144, 1536)', cc: 3 },
    ], { style: 'width:100%' });
    wrap.appendChild(U.el('div', { class: 'col gap8', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: '视觉塔主链路' }), fl,
    ]));

    /* vizgrow 自带 flex-direction:column，横向排布必须再套一层 .row */
    const midOuter = U.el('div', { class: 'vizgrow', style: 'width:100%' });
    wrap.appendChild(midOuter);
    const mid = U.el('div', { class: 'row gap16 center', style: 'width:100%' });
    midOuter.appendChild(mid);

    const col = (t, s, node) => U.el('div', { class: 'col gap4 center' }, [
      U.el('div', { class: 'klabel', text: t }), node,
      U.el('div', { class: 'mono dim', style: 'font-size:11px;text-align:center', text: s }),
    ]);

    const g24 = W.tensor(24, 24, (r, c) => ((Math.floor(r / 2) + Math.floor(c / 2)) % 2) ? 0 : 4,
      { cell: 10, gap: 1 });
    mid.appendChild(col('24 x 24 个 patch 位置', '每格 = 一个 14x14x2 的管子', g24));

    const arr = U.el('div', { class: 'col center gap4', style: 'flex:none;width:152px' });
    arr.innerHTML = '<div style="font-size:18px;color:var(--accent)">&#10233;</div>'
      + '<div class="mono" style="font-size:10.5px;color:var(--c2);text-align:center">'
      + '每 2x2 一块<br>Conv2d(k=2, s=2)</div>';
    mid.appendChild(arr);

    const g12 = W.tensor(12, 12, (r, c) => ((r + c) % 2) ? 0 : 4, { cell: 21, gap: 2 });
    mid.appendChild(col('12 x 12 个视觉 token', '144 个 1536 维向量', g12));

    const stats = [
      { k: 'image_size', v: '336', s: '输入边长（像素）', cc: 0 },
      { k: 'patch_size', v: '14', s: '每个 patch 14x14', cc: 4 },
      { k: 'patch 网格', v: '24 x 24', s: '336 / 14 = 24', cc: 2 },
      { k: '视觉 token', v: '144', s: '576 / 2^2', cc: 3 },
    ];
    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const statEls = stats.map(st => {
      const c = W.card({ cc: st.cc, tint: st.cc, sub: st.s, style: 'flex:1',
        title: '<span class="mono" style="font-size:11px">' + st.k + '</span>',
        body: U.el('div', { class: 'n big', text: st.v }) });
      row.appendChild(c); return c;
    });
    wrap.appendChild(row);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    const clearSel = () => {
      g24.cells.forEach(c => c.classList.remove('sel'));
      g12.cells.forEach(c => c.classList.remove('sel'));
    };
    const selBlock = (br, bc) => {
      [0, 1].forEach(dr => [0, 1].forEach(dc => g24.at(2 * br + dr, 2 * bc + dc).classList.add('sel')));
    };

    fl.focus(0);
    msg.innerHTML = '<span class="cm">// 一张 336x336 的 RGB 图：先 resize 到 28 的整数倍（28 = 14 x 2）</span>';
    tl.at(1800, () => {
      fl.focus(1);
      msg.innerHTML = '图处理器把它 patch 化成一条序列：<em>pixel_values (576, 1176)</em>';
    });
    tl.at(4600, () => {
      fl.focus(2); selBlock(0, 0);
      msg.innerHTML = '576 = <em>24 x 24</em> 个 patch 位置 &nbsp;<span class="cm">// 1176 = 3 x 2 x 14 x 14</span>';
    });
    tl.at(7400, () => {
      fl.focus(3); clearSel(); selBlock(3, 7); g12.at(3, 7).classList.add('sel');
      msg.innerHTML = '每 <em>2x2</em> 个 patch 合成 1 个视觉 token &nbsp;<span class="cm">// Conv2d(kernel=2, stride=2)</span>';
    });
    tl.at(10400, () => {
      fl.focus(4); clearSel();
      statEls.forEach(e => e.classList.add('ac'));
      msg.innerHTML = '得到 <em>144</em> 个 1536 维向量 &nbsp;<span class="cm">// 12 x 12 = 144</span>';
    });
    tl.at(13600, () => {
      statEls.forEach(e => e.classList.remove('ac'));
      statEls[3].classList.add('ac');
      msg.innerHTML = '★ 课程标题写的 <span class="hlbad">27</span> 与源码不符：实测是 <em>144</em>';
    });
  },
},

/* ============================================================ 2 patch embed */
{
  kicker: 'L2-07 · 第 2 幕 · patch 嵌入',
  title: 'patch embed：<span class="hl-a">kernel == stride</span> 的卷积 = 线性层',
  sub: 'Conv3d(kernel = stride = (2,14,14)) 把每个不重叠的 patch 管独立映射到 1024 维 —— 逐个位置看，它就是 Linear(1176 -> 1024)。',
  caption: '回顾 L2-05：序列 (576, 1176) 是 patchify 的产物；1176 = in_channels x temporal_patch_size x patch_size^2。',
  lang: 'python',
  codeStart: 1718,
  code: `class Glm5NextVisionPatchEmbed(nn.Module):
    def __init__(self, config: Glm5NextVisionConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states`,
  codeNote: 'Glm5NextVisionPatchEmbed —— 整座塔里唯一碰原始像素的地方。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const cards = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const mk = (cc, title, mono, sub) => W.card({ cc: cc, tint: cc, style: 'flex:1', title: title,
      sub: sub, body: U.el('div', { class: 'mono', style: 'font-size:11.5px;color:#fff', text: mono }) });
    const k1 = mk(0, '权重形状（实测）', 'proj.weight (1024, 3, 2, 14, 14)', 'Conv3d 的权重是 (out, in, T, P, P)');
    const k2 = mk(2, '滑动方式', 'kernel = stride = (2, 14, 14)', '互不重叠：每个输出位置只吃一个管');
    const k3 = mk(1, '等价写法', 'unfold -> Linear(1176, 1024)', '切窗交给框架，不必手写 reshape');
    [k1, k2, k3].forEach(k => cards.appendChild(k));

    /* vizgrow 自带 flex-direction:column，横向排布必须再套一层 .row */
    const midOuter = U.el('div', { class: 'vizgrow', style: 'width:100%' });
    wrap.appendChild(midOuter);
    const mid = U.el('div', { class: 'row gap16 center', style: 'width:100%' });
    midOuter.appendChild(mid);

    const col = (t, s, node) => U.el('div', { class: 'col gap4 center' }, [
      U.el('div', { class: 'klabel', text: t }), node,
      U.el('div', { class: 'mono dim', style: 'font-size:11px;text-align:center', text: s }),
    ]);

    const gi = W.tensor(5, 6, () => -1, { cell: 26, gap: 2, text: () => '1176' });
    mid.appendChild(col('输入：576 个 patch 管', '每个管里 1176 个数', gi));

    const arr = U.el('div', { class: 'col center gap4', style: 'flex:none;width:190px' });
    arr.innerHTML = '<div style="font-size:18px;color:var(--accent)">&#10233;</div>'
      + '<div class="mono" style="font-size:10.5px;color:var(--c2);text-align:center">'
      + '每个位置一次<br>1176 -> 1024 的线性映射</div>';
    mid.appendChild(arr);

    const go = W.tensor(5, 6, () => -1, { cell: 26, gap: 2, text: () => '1024' });
    mid.appendChild(col('输出：576 个 1024 维向量', '位置一一对应，不混合邻域', go));
    wrap.appendChild(cards);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    const pick = (r, c) => {
      gi.cells.forEach(x => x.classList.remove('sel'));
      go.cells.forEach(x => x.classList.remove('sel'));
      gi.at(r, c).classList.add('sel');
      go.at(r, c).classList.add('sel');
    };

    msg.innerHTML = '<span class="cm">// 卷积核与步长一样大 —— 滑窗一次跳一整格</span>';
    tl.at(2200, () => {
      k1.classList.add('ac');
      msg.innerHTML = '权重 <em>(1024, 3, 2, 14, 14)</em>：输出维、输入通道、时间 2 帧、14x14 空间';
    });
    tl.at(5200, () => {
      k1.classList.remove('ac'); k2.classList.add('ac'); pick(0, 0);
      msg.innerHTML = '第 1 个位置：<em>kernel == stride</em> &rArr; 只用一个管，不与邻域重叠';
    });
    tl.at(8200, () => {
      pick(2, 3);
      msg.innerHTML = '第 15 个位置：同样只用一个管 &nbsp;<span class="cm">// 等价于对每个位置独立做 Linear</span>';
    });
    tl.at(11200, () => {
      k2.classList.remove('ac'); k3.classList.add('ac');
      msg.innerHTML = '所以 <em>Conv3d</em> 在这里只是一个"会自己切窗的 Linear"';
    });
    tl.at(13800, () => {
      k3.classList.remove('ac');
      msg.innerHTML = '<span class="cm">// 前向第一步 .view(-1, 3, 2, 14, 14) 就是把这个管还原成卷积要的形状</span>';
    });
  },
},

/* ============================================================ 3 订正 144 */
{
  kicker: 'L2-07 · 第 3 幕 · 实测订正',
  title: '★ 是 <span class="hl-a">144</span> 个 token，不是 27',
  sub: '三种独立算法都给 144；运行时就是按 grid_thw.prod() // merge_size^2 现算的。',
  caption: '证据：_data/recon/probe_l207.py 的真实输出（脚本，不计入覆盖率）。',
  lang: 'python',
  codeStart: 1934,
  code: `        pixel_values = pixel_values.type(self.visual.dtype)
        vision_outputs = self.visual(pixel_values, grid_thw=image_grid_thw, **kwargs)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(vision_outputs.pooler_output, split_sizes)
        vision_outputs.pooler_output = image_embeds

        return vision_outputs`,
  codeNote: 'Glm5NextModel.get_image_features —— 「一张图几个 token」的运行时定义。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const f = U.el('div', { class: 'formula', style: 'width:100%;font-size:16px;text-align:center' });
    wrap.appendChild(f);

    const tb = W.table([
      ['算法一', '先数 patch 再合并', '(336/14)^2 / 2^2 = 576 / 4', '<span class="hl">144</span>'],
      ['算法二', '先合并再数格', '(336/14/2)^2 = 12 x 12', '<span class="hl">144</span>'],
      ['算法三', '直接跑代码', 'grid_thw.prod() // merge_size**2', '<span class="hl">144</span>'],
    ], { head: ['', '思路', '算式', '结果'] });
    const tbCard = U.el('div', { class: 'card cc0', style: 'padding:10px 12px' }, [
      U.el('div', { class: 'klabel', text: '三种独立算法' }), tb,
    ]);
    wrap.appendChild(tbCard);

    const bad = W.card({ cc: 3, tint: 3, title: '标题里的 27',
      body: U.el('div', { class: 'mono', style: 'font-size:11.5px;color:#fff',
        text: '576 / 27 = 21.33 —— 连整除都不成立；也不是任何 576 / 2^k' }),
      sub: '所以本课一律用 144；「27」只作为标题里的错误出现一次。' });
    wrap.appendChild(bad);

    const ev = U.el('div', { class: 'mono', style: 'font-size:11px;color:var(--ink-dim);'
      + 'background:rgba(6,12,26,.6);border:1px solid var(--panel-brd);border-radius:9px;padding:8px 12px' });
    ev.innerHTML = '[6] 真实图像处理器：一张 336x336 的图产出多少 token<br>'
      + '&nbsp;&nbsp;&nbsp;&nbsp;pixel_values (576, 1176)&nbsp; image_grid_thw = [[1, 24, 24]]<br>'
      + '&nbsp;&nbsp;&nbsp;&nbsp;num_image_tokens = 576 // 2**2 = <span class="hl">144</span>';
    wrap.appendChild(ev);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    f.innerHTML = 'patch 数 = (336 / 14)<sup>2</sup> = 24 x 24 = <em>576</em>';
    bad.style.opacity = '.3'; ev.style.opacity = '.3';
    msg.innerHTML = '<span class="cm">// 「一张图几个 token」不是一个常数，而是每次前向现算的</span>';
    tl.at(2400, () => {
      f.innerHTML = 'patch 数 = (336 / 14)<sup>2</sup> = 24 x 24 = <em>576</em>'
        + ' &nbsp;<span class="cm">// 336/14 = 24</span>';
      msg.innerHTML = '图处理器实测产出 <em>image_grid_thw = [[1, 24, 24]]</em>';
    });
    tl.at(5600, () => {
      f.innerHTML = 'patch 数 = 576 &nbsp;&rarr;&nbsp; 每 2x2 合并 &nbsp;&rarr;&nbsp; 576 / 4 = <em>144</em>';
      bad.classList.add('ac');
      msg.innerHTML = '每个 2x2 块合成 1 个 token &nbsp;<span class="cm">// merge_size = 2</span>';
    });
    tl.at(9000, () => {
      bad.classList.remove('ac');
      f.innerHTML = '(336/14)<sup>2</sup> / 2<sup>2</sup> = 576 / 4 = <span class="hl">144</span> 个视觉 token';
      msg.innerHTML = '源码里就是这一行：<em>image_grid_thw.prod(-1) // spatial_merge_size ** 2</em>';
    });
    tl.at(12400, () => {
      bad.classList.add('ac'); ev.style.opacity = '1';
      msg.innerHTML = '★ 标题里的 <span class="hlbad">27</span>：576 / 27 = 21.33，<em>连整除都不成立</em>';
    });
    tl.at(15400, () => {
      bad.classList.remove('ac');
      msg.innerHTML = '<span class="cm">// 这个数必须与处理器的占位符数量一致，否则 masked_scatter 会报错（第 8 幕）</span>';
    });
  },
},

/* ============================================================ 4 16 个频率 */
{
  kicker: 'L2-07 · 第 4 幕 · 轴向 RoPE（上）',
  title: '轴向 RoPE：<span class="hl-a">16</span> 个频率，H / W 各一套',
  sub: 'head_dim = 64 时，频率个数是 head_dim // 4 = 16 —— 比一维 RoPE 少一半，因为另一半要留给 width。',
  caption: '作者在类注释里写得很直白：freqs 用 head-dim//4 预算，之后 concat H 与 W 的位置。',
  lang: 'python',
  codeStart: 1773,
  code: `        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        spatial_dim = dim // 2

        attention_factor = 1.0  # Unused in this type of RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim))`,
  codeNote: 'compute_axial_rope_parameters —— 频率表就是这么算出来的。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const doc = W.card({ cc: 1, style: 'width:100%',
      title: '<span class="mono" style="font-size:11px">class Glm5NextVisionRotaryEmbedding 的 docstring</span>',
      body: U.el('div', { class: 'mono', style: 'font-size:10.5px;line-height:1.6;color:var(--ink-dim)',
        html: 'Simple axial 2D rope with same freqs used for H and W grids.<br>'
            + 'The freqs are pre-computed using <span class="hl">head-dim//4</span> which is later used to concat H and W positions.<br>'
            + 'The final angles rotate over the whole head dim, no partial rotation involved.' }) });
    wrap.appendChild(doc);

    const chain = U.el('div', { class: 'row gap10 center', style: 'width:100%' });
    const chip = (t, cc) => U.el('div', { class: 'card cc' + cc, style: 'padding:7px 12px;text-align:center' },
      [U.el('div', { class: 'mono', style: 'font-size:11.5px;color:#fff', text: t })]);
    chain.appendChild(chip('head_dim = 1024 / 16 = 64', 0));
    chain.appendChild(U.el('div', { class: 'arrow', style: 'max-width:60px' }));
    chain.appendChild(chip('spatial_dim = 64 // 2 = 32', 2));
    chain.appendChild(U.el('div', { class: 'arrow', style: 'max-width:60px' }));
    const lastChip = chip('arange(0, 32, 2) -> 16 个频率', 4);
    chain.appendChild(lastChip);
    wrap.appendChild(chain);

    const vals = [1.0, 0.562341, 0.316228, 0.177828, 0.1, 0.0562341, 0.0316228, 0.0177828,
      0.01, 0.00562341, 0.00316228, 0.00177828, 0.001, 0.000562341, 0.000316228, 0.000177828];
    const txt = ['1.000', '0.562', '0.316', '0.178', '0.100', '0.0562', '0.0316', '0.0178',
      '0.0100', '0.00562', '0.00316', '0.00178', '0.00100', '0.000562', '0.000316', '0.000178'];
    const bars = W.bars(vals.map((v, i) => ({
      label: 'f' + i, value: v, valueText: txt[i], cc: (i % 2) ? 0 : 1,
    })), { max: 1.0 });
    wrap.appendChild(U.el('div', { class: 'col gap6', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: 'inv_freq = 1 / 10000^(i / 32)，16 个值（实测）' }), bars,
    ]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    bars.setAll(0);
    msg.innerHTML = '<span class="cm">// 一维 RoPE 会算 head_dim/2 = 32 个频率；这里只算一半</span>';
    tl.at(2400, () => {
      lastChip.classList.add('ac');
      msg.innerHTML = '频率个数 = <em>head_dim // 4 = 16</em> &nbsp;<span class="cm">// arange 的步长是 2</span>';
    });
    tl.at(5600, () => {
      lastChip.classList.remove('ac');
      bars.setTo(0, 1); bars.setTo(1, 1); bars.setTo(2, 1); bars.setTo(3, 1);
      msg.innerHTML = '前几个频率：<em>1.000 / 0.562 / 0.316 / 0.178</em> —— 每 4 个降一个数量级';
    });
    tl.at(9000, () => {
      bars.setAll(1);
      msg.innerHTML = '全部 16 个频率 &nbsp;<span class="cm">// 从 1.0 到 1.78e-4</span>';
    });
    tl.at(12400, () => {
      doc.classList.add('ac');
      msg.innerHTML = '★ 一维 RoPE 用 <em>32</em> 个频率（覆盖整个 head_dim）；这里只有 <em>16</em> 个';
    });
    tl.at(15200, () => {
      doc.classList.remove('ac');
      msg.innerHTML = '<span class="cm">// 少掉的那一半，是留给 width 轴的 —— 下一幕看它怎么拼回去</span>';
    });
  },
},

/* ============================================================ 5 重组 */
{
  kicker: 'L2-07 · 第 5 幕 · 轴向 RoPE（下）',
  title: '★ 重组：H / W 两套频率拼成 <span class="hl-a">64</span> 维',
  sub: 'cat([h, w]) 得到 32 维，再复制一份铺满 head_dim —— 复制是为了让 rotate_half 把第 i 维与第 i+32 维配对。',
  caption: '实测：只改 w 时 cos 的前 16 维不变，只改 h 时 16..32 维不变。',
  lang: 'python',
  codeStart: 1795,
  code: `    def recomposition_frequencies(self, freq):
        """
        Recompose the frequencies into the final spatial layout used per each grid.
        """
        freq_h, freq_w = freq[:, 0], freq[:, 1]
        freq_hw = torch.cat([freq_h, freq_w], dim=-1)
        return torch.cat([freq_hw, freq_hw], dim=-1)`,
  codeNote: 'recomposition_frequencies —— 本课第二个★，只有 4 行有效代码。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const grid = W.tensor(2, 32, (r, c) => (c < 16) ? 0 : 1, { cell: 22, gap: 2 });
    wrap.appendChild(U.el('div', { class: 'col center gap6', style: 'width:100%' }, [
      U.el('div', { class: 'row', style: 'width:100%' }, [
        U.el('div', { class: 'klabel', style: 'flex:1;text-align:center', text: '前 16 维 ← h 坐标' }),
        U.el('div', { class: 'klabel', style: 'flex:1;text-align:center', text: '后 16 维 ← w 坐标' }),
      ]),
      grid,
      U.el('div', { class: 'mono dim', style: 'font-size:11px;text-align:center',
        text: '(N, 64) 的 cos / sin：上排是第 0..31 维，下排是第 32..63 维（完全相同的一份拷贝）' }),
    ]));

    const cmp = W.table([
      ['位置输入', '一个标量 p', '两个坐标 (h, w)'],
      ['频率个数', 'head_dim / 2 = <span class="hl">32</span>', 'head_dim / 4 = <span class="hl">16</span>，H/W 共用'],
      ['角度来源', 'p x inv_freq[i]', '前 16 维 h x f[i]，后 16 维 w x f[i]'],
      ['语义', '第几个 token', '第几行、第几列'],
    ], { head: ['', '一维 RoPE（文本）', '轴向 RoPE（本课）'] });
    const cmpCard = U.el('div', { class: 'card cc1', style: 'padding:10px 12px' }, [cmp]);
    wrap.appendChild(cmpCard);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    const rows = [0, 1].map(r => [0, 1].map(c => grid.at(r, c)));
    rows[1].forEach(c => { c.style.opacity = '0'; });
    cmpCard.style.opacity = '.35';

    msg.innerHTML = '<span class="cm">// freq 的形状是 (N, 2, 16)：第 0 列 h，第 1 列 w</span>';
    tl.at(2200, () => {
      msg.innerHTML = '先 <em>cat([freq_h, freq_w])</em> &rArr; <em>(N, 32)</em>：前 16 维只随 h 变，后 16 维只随 w 变';
    });
    tl.at(5400, () => {
      rows[1].forEach(c => { c.style.opacity = '1'; });
      msg.innerHTML = '再复制一份 <em>cat([freq_hw, freq_hw])</em> &rArr; <em>(N, 64)</em>，铺满 head_dim';
    });
    tl.at(8600, () => {
      [0, 1].forEach(r => { grid.at(r, 3).classList.add('sel'); grid.at(r, 19).classList.add('sel'); });
      msg.innerHTML = '<em>rotate_half</em> 把第 i 维与第 i+32 维配对 —— 复制之后，两边角度相同';
    });
    tl.at(11800, () => {
      msg.innerHTML = '于是每个频率得到一次标准二维旋转 &nbsp;<span class="cm">// "整个 head_dim 都在转"</span>';
    });
    tl.at(14600, () => {
      cmpCard.style.opacity = '1'; cmpCard.classList.add('ac');
      msg.innerHTML = '与一维 RoPE 的区别：<em>一维只有一个坐标</em>，这里是 (h, w) 两个方向各自编码';
    });
  },
},

/* ============================================================ 6 patch merger */
{
  kicker: 'L2-07 · 第 6 幕 · 合并与投影',
  title: 'patch merger：空间 <span class="hl-a">2x2</span> 合并，先投影到 1536',
  sub: 'view(-1,2,2,1024) 敢这么写，是因为序列顺序本来就是"块主序"；Conv2d(k=2,s=2) 再把它变成一次块内线性投影。',
  caption: '顺序要点：downsample 先做 1024 -> 1536，merger 的 dim 是 out_hidden_size（1536），不是 1024。',
  lang: 'python',
  codeStart: 1547,
  code: `    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.proj(hidden_state)
        hidden_state = self.act1(self.post_projection_norm(hidden_state))
        gate = self.gate_proj(hidden_state)
        up = self.up_proj(hidden_state)
        # Key difference using clamping
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)`,
  codeNote: 'Glm5NextVisionPatchMerger.forward —— proj -> LayerNorm -> GELU -> SwiGLU。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    /* vizgrow 自带 flex-direction:column，横向排布必须再套一层 .row */
    const midOuter = U.el('div', { class: 'vizgrow', style: 'width:100%' });
    wrap.appendChild(midOuter);
    const mid = U.el('div', { class: 'row gap16 center', style: 'width:100%' });
    midOuter.appendChild(mid);

    const col = (t, s, node) => U.el('div', { class: 'col gap4 center' }, [
      U.el('div', { class: 'klabel', text: t }), node,
      U.el('div', { class: 'mono dim', style: 'font-size:11px;text-align:center', text: s }),
    ]);

    const big = W.tensor(6, 6, (r, c) => (Math.floor(r / 2) * 3 + Math.floor(c / 2)) % 6, { cell: 30, gap: 4 });
    mid.appendChild(col('(576, 1024) 块主序', 'view(-1, 2, 2, 1024)', big));

    const arr = U.el('div', { class: 'col center gap4', style: 'flex:none;width:220px' });
    arr.innerHTML = '<div style="font-size:18px;color:var(--accent)">&#10233;</div>'
      + '<div class="mono" style="font-size:10.5px;color:var(--c2);text-align:center">'
      + 'permute(0,3,1,2)<br>Conv2d(kernel=2, stride=2)<br>1024 -> 1536</div>';
    mid.appendChild(arr);

    const small = W.tensor(3, 3, (r, c) => (r * 3 + c) % 6, { cell: 60, gap: 8 });
    mid.appendChild(col('(144, 1536)', '每块 4 个 patch -> 1 个 token', small));

    const cards = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const c1 = W.card({ cc: 2, tint: 2, style: 'flex:1', title: '先投影，再 merger',
      sub: 'downsample 把 1024 抬到 out_hidden_size = 1536；merger 的 dim 拿到的是 1536，不是 1024。' });
    const c2 = W.card({ cc: 3, tint: 3, style: 'flex:1', title: '名字有坑：last_hidden_state',
      sub: '出口的 last_hidden_state 已经是 (144, 1536)，不是 576 个 patch；真正拼进文本的是 pooler_output。' });
    cards.appendChild(c1); cards.appendChild(c2);
    wrap.appendChild(cards);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    const clear = () => {
      big.cells.forEach(c => c.classList.remove('sel'));
      small.cells.forEach(c => c.classList.remove('sel'));
    };
    const selBlock = (br, bc) => [0, 1].forEach(dr => [0, 1].forEach(dc =>
      big.at(2 * br + dr, 2 * bc + dc).classList.add('sel')));

    msg.innerHTML = '<span class="cm">// 24 层跑完，手上还是 576 个 1024 维 token</span>';
    tl.at(2000, () => {
      selBlock(0, 0);
      msg.innerHTML = '序列里<em>连续的 4 个 token</em>就是一个 2x2 块 &nbsp;<span class="cm">// 位置编号是块主序排好的</span>';
    });
    tl.at(5200, () => {
      clear(); selBlock(2, 1); small.at(2, 1).classList.add('sel');
      msg.innerHTML = '块内 4 个 patch &rarr; <em>1 个 token</em>：576 / 4 = 144';
    });
    tl.at(8400, () => {
      clear(); c1.classList.add('ac');
      msg.innerHTML = '这一步的 Conv2d 是 <em>1024 -> 1536</em> 的块内线性投影';
    });
    tl.at(11400, () => {
      c1.classList.remove('ac'); c2.classList.add('ac');
      msg.innerHTML = '出口 <em>pooler_output (144, 1536)</em> 才是送进文本的那一串';
    });
    tl.at(14000, () => {
      c2.classList.remove('ac');
      msg.innerHTML = '<span class="cm">// merger 内部：proj -> LayerNorm -> GELU -> SwiGLU（gate 上截断、up 双边截断，limit=10.0）</span>';
    });
  },
},

/* ============================================================ 7 packed 注意力 */
{
  kicker: 'L2-07 · 第 7 幕 · 塔内',
  title: '24 层都是一次 <span class="hl-a">qkv</span>：packed 变长、非因果',
  sub: '一次线性出 q/k/v，reshape 成 (seq, 3, 16, 64) 后把 3 当 batch 维用；q/k 各做一次逐头 RMSNorm，旋转在 fp32 里算。',
  caption: 'cu_seqlens 取代 attention_mask：段内全可见、段间不可见。后端分派见 L8-01 / L8-02。',
  lang: 'python',
  codeStart: 1624,
  code: `        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)`,
  codeNote: 'Glm5NextVisionAttention.forward 的前 13 行 —— 整座塔的形状枢纽。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const fl = W.flow([
      { t: '(576, 1024)', s: 'block 输出', cc: 0 },
      { t: 'qkv', s: 'Linear -> (576, 3072)', cc: 4 },
      { t: 'reshape', s: '(576, 3, 16, 64)', cc: 2 },
      { t: 'permute+unbind', s: '3 x (576, 16, 64)', cc: 1 },
      { t: 'q_norm / k_norm', s: '逐头 RMSNorm(64)', cc: 5 },
    ], { style: 'width:100%' });
    wrap.appendChild(U.el('div', { class: 'col gap8', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: '一次线性，没有第二次投影' }), fl,
    ]));

    const cards = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const mk = (cc, title, sub) => W.card({ cc: cc, tint: cc, style: 'flex:1', title: title, sub: sub });
    const a1 = mk(1, 'q/k 逐头 RMSNorm', '归一化的是最后一维 head_dim = 64，不是 1024 —— 头部之间互不影响。');
    const a2 = mk(4, 'is_causal = False', '一张图内部全可见：这是视觉塔与文本塔最本质的行为差异。');
    const a3 = mk(2, '旋转在 fp32 里算', 'apply_rotary_pos_emb_vision 内部 q.float() / k.float() 后再转回原 dtype。');
    [a1, a2, a3].forEach(c => cards.appendChild(c));
    wrap.appendChild(cards);

    const segs = [
      { t: '图 1', n: '576', w: 576, cc: 0 },
      { t: '图 2', n: '424', w: 424, cc: 1 },
      { t: '图 3', n: '400', w: 400, cc: 4 },
    ];
    const strip = U.el('div', { class: 'row gap4', style: 'width:100%' });
    const segEls = segs.map(s => {
      const e = U.el('div', { class: 'n sm', style: 'flex:' + s.w + ' 1 0;height:34px;border-radius:8px;'
        + 'display:flex;align-items:center;justify-content:center;'
        + 'border:1px solid var(--c' + s.cc + ');background:rgba(150,180,255,.10);color:#fff',
        text: s.t + ' · ' + s.n });
      strip.appendChild(e); return e;
    });
    wrap.appendChild(U.el('div', { class: 'col gap6', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: 'packed：cu_seqlens = [0, 576, 1000, 1400]' }), strip,
    ]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    fl.focus(0);
    msg.innerHTML = '<span class="cm">// 24 层 block 都是标准 pre-norm 残差，没有 mHC 的 4 条流</span>';
    tl.at(2000, () => { fl.focus(1); msg.innerHTML = '一次 <em>qkv</em> 线性：(576, 1024) &rarr; (576, 3072)'; });
    tl.at(5000, () => { fl.focus(2); msg.innerHTML = 'reshape 成 <em>(576, 3, 16, 64)</em> &nbsp;<span class="cm">// 把"3"当 batch 维借用</span>'; });
    tl.at(8200, () => { fl.focus(3); msg.innerHTML = 'permute + unbind 切出 <em>q / k / v</em> 三份 (576, 16, 64)'; });
    tl.at(11400, () => {
      fl.focus(4); a1.classList.add('ac');
      msg.innerHTML = 'q/k 各过一次 <em>RMSNorm(head_dim = 64)</em> —— 逐头归一化';
    });
    tl.at(14200, () => {
      fl.focus(-1); a1.classList.remove('ac'); a2.classList.add('ac');
      segEls.forEach((e, i) => { e.style.boxShadow = i === 1 ? '0 0 0 2px var(--accent)' : 'none'; });
      msg.innerHTML = '非因果 + packed：<em>段内全可见、段间不可见</em>，不需要 attention_mask';
    });
  },
},

/* ============================================================ 8 回填序列 */
{
  kicker: 'L2-07 · 第 8 幕 · 拼回文本',
  title: '144 个视觉 token 怎样<span class="hl-a">落到位置上</span>',
  sub: 'prompt 里先摆好 144 个占位符，再用 masked_scatter 把视觉向量原地填进去 —— 主干从此只看得到一条序列。',
  caption: '回顾 L0-01：主干入口二选一，多模态走 inputs_embeds。回顾 L2-06：占位符是处理器写进 prompt 的。',
  lang: 'python',
  codeStart: 2016,
  code: `        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw, **kwargs).pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)`,
  codeNote: 'Glm5NextModel.forward 里的图片分支 —— 塔与主干的接缝。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const mkRow = (label, bigText, bigCls, smallColor) => {
      const r = U.el('div', { class: 'row gap6', style: 'width:100%' });
      const cellStyle = 'flex:none;width:76px;height:34px;border-radius:8px;display:flex;'
        + 'align-items:center;justify-content:center;border:1px solid var(--panel-brd);'
        + 'background:rgba(150,180,255,.08);font:600 10.5px var(--mono);color:' + smallColor;
      ['BOS', '文本', '文本'].forEach(t => r.appendChild(U.el('div', { style: cellStyle, text: t })));
      const big = U.el('div', { class: 'n sm ' + bigCls, style: 'flex:1 1 0;height:34px;border-radius:8px;'
        + 'display:flex;align-items:center;justify-content:center;' + bigText.style, text: bigText.text });
      r.appendChild(big);
      ['文本', '文本', '文本', 'EOS'].forEach(t => r.appendChild(U.el('div', { style: cellStyle, text: t })));
      const box = U.el('div', { class: 'col gap5', style: 'width:100%' }, [
        U.el('div', { class: 'klabel', text: label }), r,
      ]);
      box._big = big;
      return box;
    };

    const before = mkRow('input_ids（处理器写好的 prompt）', {
      text: '<|image|> x 144', style: 'border:1px solid var(--c2);background:rgba(251,191,36,.12);color:#fde68a',
    }, '');
    const after = mkRow('inputs_embeds（masked_scatter 之后）', {
      text: '视觉向量 (144, 1536)', style: 'border:1px solid var(--c4);background:rgba(52,211,153,.16);color:#a7f3d0',
    }, 'color:var(--ink-dim)');
    wrap.appendChild(U.el('div', { class: 'col gap12', style: 'width:100%' }, [before, after]));

    const cards = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const mk = (cc, title, sub) => W.card({ cc: cc, tint: cc, style: 'flex:1', title: title, sub: sub });
    const b1 = mk(0, 'dtype / device 在这里对齐', '.to(inputs_embeds.device, inputs_embeds.dtype)：塔与主干的精度在此统一。');
    const b2 = mk(2, '图和视频共用 image_token_id', '实测 154854；靠 begin/end span 的 cumsum 区间区分，所以掩码要两段一起看。');
    const b3 = mk(3, '数量与宽度都要对', 'masked_scatter 要求元素数正好是「占位符数 x 文本宽度」：不只 144 对 144，'
      + 'out_hidden_size 还必须等于文本 hidden_size（实测不等就报 Image features and image tokens do not match）。');
    [b1, b2, b3].forEach(c => cards.appendChild(c));
    wrap.appendChild(cards);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    after.style.opacity = '.25';
    msg.innerHTML = '<span class="cm">// prompt 里先摆 144 个占位符 —— 位置是处理器（L2-06）算好的</span>';
    tl.at(2200, () => {
      before._big.style.boxShadow = '0 0 0 2px var(--c2)';
      msg.innerHTML = '视觉塔输出 <em>(144, 1536)</em>，占位符也正好 <em>144</em> 个';
    });
    tl.at(5400, () => {
      before._big.style.boxShadow = 'none';
      after.style.opacity = '1'; after._big.style.boxShadow = '0 0 0 2px var(--c4)';
      msg.innerHTML = '<em>masked_scatter</em> 把 144 个向量原地填进占位符的位置';
    });
    tl.at(8600, () => {
      b3.classList.add('ac');
      msg.innerHTML = '数量对不上就抛 <em>Image features and image tokens do not match</em>';
    });
    tl.at(11800, () => {
      b3.classList.remove('ac'); b2.classList.add('ac');
      msg.innerHTML = '★ 图和视频<em>共用同一个占位符 id</em>：靠 begin/end span 区分';
    });
    tl.at(14600, () => {
      b2.classList.remove('ac'); b1.classList.add('ac');
      msg.innerHTML = '替换完成后主干拿到一条 <em>text_len + 144</em> 的序列，它不知道图在哪'
        + ' <span class="cm">// 前提：out_hidden_size == 文本 hidden_size</span>';
    });
  },
},

/* ============================================================ 9 收束 */
{
  kicker: 'L2-07 · 收束',
  title: '把形状串起来：<span class="hl-a">576</span> 进，<span class="hl-a">144</span> 出',
  sub: '一座塔的全部形状只有六个阶段；而 backbone_utils 那套东西，视觉塔一样都没用上。',
  caption: '下一课 L3-01：文本侧的三种 RMSNorm —— 视觉塔里的 q_norm 与它们是同一族。',
  lang: 'python',
  codeStart: 1,
  code: `import warnings

from ..backbone_utils import BackboneConfigMixin, BackboneMixin


class BackboneConfigMixin(BackboneConfigMixin):
    warnings.warn(
        "Importing \`BackboneConfigMixin\` from \`utils/backbone_utils.py\` is deprecated and will be removed in "
        "Transformers v5.10. Import as \`from transformers.backbone_utils import BackboneConfigMixin\` instead.",
        FutureWarning,
    )`,
  codeNote: 'utils/backbone_utils.py 全文 —— import 路径的弃用垫片，视觉塔完全绕开了它。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const tb = W.table([
      ['pixel_values', '(576, 1176)', '1176 = 3 x 2 x 14 x 14'],
      ['patch_embed', '(576, 1024)', 'Conv3d kernel=stride=(2,14,14)'],
      ['24 x VisionBlock', '(576, 1024)', 'packed 非因果 · 轴向 RoPE'],
      ['2x2 合并 + downsample', '(144, 1536)', 'Conv2d kernel=stride=2'],
      ['merger', '(144, 1536)', 'proj -> LN -> GELU -> SwiGLU'],
      ['masked_scatter', '(text_len + 144, 4096)', '主干再也分不出图在哪'],
    ], { head: ['阶段', '形状', '关键参数'] });
    const tbCard = U.el('div', { class: 'card cc0', style: 'padding:10px 12px' }, [
      U.el('div', { class: 'klabel', text: '本课全部形状（实测）' }), tb,
    ]);
    wrap.appendChild(tbCard);

    const bc = W.card({ cc: 1, tint: 1, title: 'backbone 与 vision tower 是两套契约',
      sub: '实测：modeling_glm5_next.py 里 "backbone" 出现 <b>0</b> 次，视觉配置里 stage_names / out_features 也是 0 次。'
        + '看到 stage_names 就往 backbone 想，看到 grid_thw 就往 tower 想。' });
    wrap.appendChild(bc);

    const ex1 = W.exercise(
      '一张 336x336 的图，进塔时是多少个 patch？出塔时是多少个 token？请写出算式。',
      '<b>(336/14)^2 = 24 x 24 = 576</b> 个 patch 位置；每 2x2 合并 1 个，'
      + '<b>576 / 2^2 = 144</b> 个 token。等价算法：<b>(336/14/2)^2 = 12 x 12 = 144</b>。'
      + '<br>实测核对：图处理器给 <code class="inl">image_grid_thw = [[1, 24, 24]]</code>，'
      + '视觉塔 <code class="inl">pooler_output</code> 的形状是 <code class="inl">(144, 1536)</code>。'
      + '<br>★ 课程标题写的 27 是错的（576 / 27 不整除）。');
    const ex2 = W.exercise(
      '轴向 RoPE 与一维 RoPE 的区别是什么？为什么图像需要前者？',
      '一维 RoPE 只有一个位置标量 <code class="inl">p</code>，用 <code class="inl">head_dim/2</code> 个频率；'
      + '轴向 RoPE 有 <code class="inl">(h, w)</code> 两个坐标，用 <code class="inl">head_dim/4</code> 个频率，'
      + 'H 与 W <b>共用同一张频率表</b>，角度按 <code class="inl">cat([freq_h, freq_w])</code> 拼成 32 维再复制成 64 维。'
      + '<br>实测判据：只改 w 时 cos 的前 16 维不变、后 16 维变；只改 h 时相反。'
      + '<br>图像需要它，是因为"上下"与"左右"不是同一个方向；一维编号会把二维邻域压成一条线。');
    const exRow = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    [ex1, ex2].forEach(e => { e.style.flex = '1'; exRow.appendChild(e); });
    wrap.appendChild(exRow);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 六个阶段，逐个点亮</span>';
    const rows = U.qa('tbody tr', tb);
    rows.forEach(r => { r.style.opacity = '.35'; });
    const marks = [0, 1, 2, 3, 4, 5];
    marks.forEach((k, i) => {
      tl.at(1600 + i * 2000, () => {
        rows.forEach((r, j) => { r.style.opacity = (j === k) ? '1' : '.35'; });
      });
    });
    tl.at(13800, () => {
      rows.forEach(r => { r.style.opacity = '1'; });
      msg.innerHTML = '576 进、144 出：<em>(336/14)^2 / 2^2 = 144</em>';
    });
    tl.at(16200, () => {
      bc.classList.add('ac');
      msg.innerHTML = '<span class="cm">// 下一课 L3-01：文本侧的三种 RMSNorm（视觉塔的 q_norm 与它们同族）</span>';
    });
  },
},

];
