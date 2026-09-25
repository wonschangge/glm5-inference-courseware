/* ==========================================================================
   L2-05 · 图像变换、视频与通用视觉工具
   --------------------------------------------------------------------------
   覆盖：image_transforms.py / image_utils.py / video_processing_utils.py
         video_utils.py / vision_utils.py / audio_utils.py（6 个文件 / 5799 行）
   目标：看完能手算 336x336 图的视觉 token 数，并说清 temporal_patch_size=2
         对视频帧数提出了什么要求、由谁兜住。
   分幕：10 幕（作业书写的是 9 幕 —— 见 README 的「与作业书的偏差」一节）。
   排版基准：#visual 可视区 = 994 x 662.7 舞台像素（无头浏览器实测）。
   ========================================================================== */
'use strict';

/* 实测常量（来自 GLM-5 图像/视频处理器默认值，不是估算） */
const M = {
  imageSize: 336,
  patch: 14,
  merge: 2,
  temporal: 2,
  gridHW: 24,          /* 336 / 14 */
  patches: 576,        /* 24 * 24 */
  mergeLen: 4,         /* 2 ** 2 */
  tokens: 144,         /* 576 / 4 */
  patchDim: 588,       /* 3 * 1 * 14 * 14 */
};

const SCENES = [

/* ------------------------------------------------- 1 全景：三步 */
{
  kicker: '第 2 层 · 输入流水线',
  title: '像素怎么变成 <span class="hl-a">token</span>：三步',
  sub: '第 2 层走到这里才碰到真正的张量级变换。六个文件、5799 行，要记住的只有三步：重采样 → 切块 → 合并。',
  caption: '回顾 L0-01：文本那条路是 input_ids → embed_tokens；视觉这条路多绕了三步，但最后的落点一样 —— 同一条 seq 轴。',
  lang: 'python',
  codeStart: 839,
  code: `def divide_to_patches(
    image: Union[np.ndarray, "torch.Tensor"], patch_size: int | tuple[int, int]
) -> list[Union[np.ndarray, "torch.Tensor"]]:
    """
    Divides an image into patches of a specified size.

    Args:
        image (\`np.array | "torch.Tensor"\`):
            The input image.
        patch_size (\`int\` or \`tuple[int, int]\`):
            The size of each patch. If an int, patches are square. If a tuple,
            it is interpreted as \`(patch_height, patch_width)\`.
    Returns:
        list: A list of \`np.array | "torch.Tensor"\` representing the patches.
    """`,
  codeNote: 'image_transforms.divide_to_patches —— 最朴素的一版切块：两个步进循环，返回 patch 列表。',
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    wrap.appendChild(U.el('div', { class: 'klabel', text: '一张 336×336 的图，走完三步之后的形状' }));
    const fl = W.flow([
      { t: '读入', s: '(3, 336, 336)', cc: 0 },
      { t: '重采样', s: '尺寸对齐', cc: 4 },
      { t: '切块', s: '24 × 24 = 576', cc: 2 },
      { t: '合并', s: '÷ 4', cc: 1 },
      { t: '视觉 token', s: '144 个', cc: 3 },
    ], { style: 'width:100%' });
    wrap.appendChild(fl);

    const viz = U.el('div', { class: 'col gap10 vizgrow hstart' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '三步的形状' }));
    const tb = W.table([
      ['1 重采样', 'resize', '<span class="hl">(3, 336, 336)</span> → <span class="hl">(3, 336, 336)</span>', '只改数值，不改形状'],
      ['2 切块', 'patchify', '<span class="hl">(3, 336, 336)</span> → <span class="hl">(576, 588)</span>', '只改形状，不改数值'],
      ['3 合并', 'merge', '<span class="hl">576</span> → <span class="hl">144</span>', '2×2 个 patch 合成 1 个 token'],
    ], { head: ['步', '谁做的', '形状', '关键'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 0, style: 'padding:10px 12px;width:100%' }, [tb]));

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const c1 = W.card({ cc: 2, tint: 2, num: '3', title: '336 / 14 = 24',
      sub: 'patch 尺寸是 <code class="inl">14</code>，所以一条边切成 24 段 —— 这就是 <code class="inl">grid_h = grid_w = 24</code> 的来历。' });
    const c2 = W.card({ cc: 1, tint: 1, num: '4', title: 'merge_size² = 4',
      sub: '2×2 个 patch 合成一个 token。切块时 <code class="inl">permute</code> 已经把它们排到连续位置了。' });
    c1.style.flex = '1'; c2.style.flex = '1';
    row.appendChild(c1); row.appendChild(c2);
    viz.appendChild(row);

    const cost = W.table([
      ['重采样', '<span class="hl">有损</span>', 'resize 的插值是纯损失：尺寸已经对上了就别再做一次'],
      ['切块', '<span class="hl4">无损</span>', '纯内存重排，一个数都不变，只是换了个顺序'],
      ['合并', '<span class="hl4">无损</span>', '只是换分组：真正的降维发生在视觉塔的 merger 层'],
    ], { head: ['步', '有没有丢信息', '为什么'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 4, style: 'padding:10px 12px;width:100%' }, [
      U.el('div', { class: 'klabel', text: '★ 三步里只有第一步会丢信息' }), cost,
    ]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    const focus = (k) => fl.focus(k);
    focus(0); msg.innerHTML = '<span class="cm">// 一个 token 的旅程从这里开始</span>';
    tl.at(1800, () => { focus(1); msg.innerHTML = '重采样只动数值：<em>(3, 336, 336)</em> 进、<em>(3, 336, 336)</em> 出'; });
    tl.at(4600, () => { focus(2); msg.innerHTML = '切块只动形状：<em>(576, 588)</em> —— 576 个 patch，每个展平 588 个数'; });
    tl.at(7600, () => { c1.classList.add('ac'); msg.innerHTML = '<em>336 / 14 = 24</em>，所以是 24×24 = <em>576</em> 个 patch'; });
    tl.at(10600, () => { c1.classList.remove('ac'); c2.classList.add('ac'); msg.innerHTML = '<em>576 / 4 = 144</em> —— 这就是要插进序列的视觉 token 数'; });
    tl.at(13400, () => { focus(4); c2.classList.remove('ac'); msg.innerHTML = '★ 记住这条链：<em>grid_thw → prod() → // merge_length</em>'; });
    tl.at(15600, () => {
      focus(-1);
      msg.innerHTML = '<span class="cm">// 下面八幕把这 144 拆开：每一步到底在哪一行发生</span>';
    });
  },
},

/* ------------------------------------------------- 2 resize 双后端 */
{
  kicker: '步 ① · 重采样',
  title: '同一个 <span class="hl-a">resize</span>，<span class="hl-b">两份实现</span>',
  sub: 'image_transforms.py 是"函数式 numpy 版"，image_utils.py 是"类方法 PIL 版"。名字一样、签名很像，代价却付在不同地方。',
  caption: '这一层最容易读糊的就是这个：两个 resize 同名不同源，改错一个不会报错，只会悄悄换掉插值实现。',
  lang: 'python',
  codeStart: 353,
  code: `    # For all transformations, we want to keep the same data format as the input image unless otherwise specified.
    # The resized image from PIL will always have channels last, so find the input format first.
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)
    data_format = input_data_format if data_format is None else data_format

    # To maintain backwards compatibility with the resizing done in previous image feature extractors, we use
    # the pillow library to resize the image and then convert back to numpy
    do_rescale = False
    if not isinstance(image, PIL.Image.Image):
        do_rescale = _rescale_for_pil_conversion(image)
        image = to_pil_image(image, do_rescale=do_rescale, input_data_format=input_data_format)
    height, width = size
    # PIL images are in the format (width, height)
    resized_image = image.resize((width, height), resample=resample, reducing_gap=reducing_gap)

    if return_numpy:
        resized_image = np.array(resized_image)
        # If the input image channel dimension was of size 1, then it is dropped when converting to a PIL image
        # so we need to add it back if necessary.
        resized_image = np.expand_dims(resized_image, axis=-1) if resized_image.ndim == 2 else resized_image
        # The image is always in channels last format after converting from a PIL image
        resized_image = to_channel_dimension_format(
            resized_image, data_format, input_channel_dim=ChannelDimension.LAST
        )
        # If an image was rescaled to be in the range [0, 255] before converting to a PIL image, then we need to
        # rescale it back to the original range.
        resized_image = rescale(resized_image, 1 / 255) if do_rescale else resized_image
    return resized_image`,
  codeNote: 'image_transforms.resize 的尾段：numpy → PIL → resize → numpy，量纲要记账。',
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '两条落地路径 —— 输入都是 (3, H, W)' }));

    const mk = (cc, title, sub, steps) => {
      const c = W.card({ cc: cc, tint: cc, title: title, sub: sub });
      const box = U.el('div', { class: 'col gap6', style: 'width:100%;margin-top:9px' });
      const els = steps.map((s, i) => {
        const e = U.el('div', {
          class: 'mono', style: 'font-size:11px;padding:6px 9px;border-radius:8px;'
            + 'border:1px solid rgba(150,180,255,.16);background:rgba(150,180,255,.05)',
          html: s,
        });
        box.appendChild(e);
        if (i < steps.length - 1) {
          box.appendChild(U.el('div', { class: 'mono faint', style: 'font-size:10px;padding-left:10px', text: '↓' }));
        }
        return e;
      });
      c.appendChild(box);
      c._els = els;
      return c;
    };

    const left = mk(0, 'image_transforms.resize',
      '吃 <code class="inl">np.ndarray</code>。为了复用 PIL 的插值实现，内部**绕一趟** PIL。',
      ['np.ndarray (3, 336, 336) → 判断要不要 ×255',
       'to_pil_image → uint8 PIL（**有损**）',
       'image.resize((w, h), resample)',
       'np.array → 还原通道格式与量纲（÷255）']);
    const right = mk(4, 'image_utils.ImageFeatureExtractionMixin.resize',
      '吃 <code class="inl">PIL.Image.Image</code> 或先转 PIL，**省掉整个往返**。',
      ['PIL.Image.Image（或先 to_pil_image）',
       '算目标尺寸：短边对齐 / 直接 (size, size)',
       'short == 目标？→ 提前 return，否则 resize']);
    const lt = W.table([
      ['输入', 'np.ndarray', 'PIL.Image'],
      ['通道格式', '显式推断 + 还原', 'PIL 天生 channels-last'],
      ['量纲往返', '<span class="hlbad">要记账 ×255 / ÷255</span>', '<span class="hl4">不用</span>'],
      ['返回', 'np.ndarray', '<span class="hl">PIL.Image</span>'],
      ['谁在用', '图像处理器后端', '旧版特征提取器'],
    ], { head: ['', '张量版', 'PIL 版'] });
    left.appendChild(U.el('div', { style: 'width:100%;margin-top:10px' }, [lt]));
    const rt = W.table([
      ['真做尺寸的步数', '<span class="hlbad">2 / 4</span>'],
      ['提前返回', '<code class="inl">short == requested_new_short</code>'],
    ], { head: ['张量版的一段', '代价'] });
    right.appendChild(U.el('div', { style: 'width:100%;margin-top:10px' }, [rt]));

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    left.style.flex = '1'; right.style.flex = '1';
    row.appendChild(left); row.appendChild(right);
    viz.appendChild(row);

    const bar = U.el('div', { class: 'card', cc: 2, style: 'padding:9px 12px;width:100%' });
    bar.innerHTML = '<span class="klabel">两条路都绕不开的一行（函数入口处）</span>'
      + '<div class="mono" style="font-size:11.5px;color:#cfe6ff">'
      + 'data_format = input_data_format if data_format is None else data_format</div>'
      + '<div class="dim" style="font-size:11px;margin-top:4px">'
      + '变换不改变通道顺序，除非你显式要求 —— 这句是本层所有函数的共同前提。</div>';
    viz.appendChild(bar);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    left._els.forEach(e => { e.style.opacity = '.25'; });
    right._els.forEach(e => { e.style.opacity = '.25'; });
    msg.innerHTML = '<span class="cm">// 左边那份：形状不变，但数值往返了一趟 uint8</span>';
    left._els.forEach((e, i) => {
      tl.at(1800 + i * 2000, () => {
        left._els.forEach(x => { x.style.opacity = '.25'; });
        e.style.opacity = '1';
        e.style.borderColor = 'var(--c0)';
        msg.innerHTML = '左路第 <em>' + (i + 1) + '</em> 步 &nbsp;<span class="cm">// 4 步里有 2 步在"转格式"</span>';
      });
    });
    tl.at(9800, () => {
      left._els.forEach(e => { e.style.borderColor = 'rgba(150,180,255,.16)'; });
      msg.innerHTML = '右路（PIL 版）：同一次 resize，<em>只有 2 步是真的在做尺寸</em>';
    });
    tl.at(12000, () => {
      right._els.forEach((e, i) => {
        if (i === 1 || i === 2) { e.style.opacity = '1'; e.style.borderColor = 'var(--c4)'; }
      });
      msg.innerHTML = '★ <em>short == 目标就提前 return</em> —— 重采样是纯损失，能不做就不做';
    });
    tl.at(14500, () => {
      right._els.forEach(e => { e.style.opacity = '1'; e.style.borderColor = 'rgba(150,180,255,.16)'; });
      msg.innerHTML = '★ 分组的意义：<em>不同尺寸的图不能堆在同一个张量里</em>（下一幕用 grid 数证明）';
    });
    tl.at(16800, () => {
      msg.innerHTML = '<span class="cm">// 记住：同名函数分两份，改错一个不会报错，只会悄悄换实现</span>';
    });
  },
},

/* ------------------------------------------------- 3 normalize */
{
  kicker: '步 ①\u00b7补 · 数值',
  title: '为什么 <span class="hl-a">normalize</span> 第一件事是 cast',
  sub: '尺寸对齐之后是数值对齐。这里藏着两行注释，说明了两类真实的数值事故。',
  caption: 'uint8 回绕和 float16 静默提升，都不会抛异常 —— 只会让 loss 曲线看起来"有点怪"。',
  lang: 'python',
  codeStart: 411,
  code: `    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)

    channel_axis = get_channel_dimension_axis(image, input_data_format=input_data_format)
    num_channels = image.shape[channel_axis]

    # We cast to float32 to avoid errors that can occur when subtracting uint8 values.
    # We preserve the original dtype if it is a float type to prevent upcasting float16.
    if not np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32)

    if isinstance(mean, Collection):
        if len(mean) != num_channels:
            raise ValueError(f"mean must have {num_channels} elements if it is an iterable, got {len(mean)}")
    else:
        mean = [mean] * num_channels
    mean = np.array(mean, dtype=image.dtype)`,
  codeNote: 'image_transforms.normalize 的前半段：先定 dtype，再定广播形状。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart', style: 'justify-content:center' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '(x - mean) / std 在三种 dtype 下会发生什么' }));

    const g = U.el('div', { class: 'col gap8', style: 'width:100%' });
    const cases = [
      { cc: 3, t: 'uint8 直接算', code: 'np.uint8(3) - 5', out: '254', ok: false,
        d: '回绕。像素"变亮"而不是变负 —— 不报错，不警告。' },
      { cc: 4, t: '先 cast float32 再算（这一行的作用）', code: 'np.float32(3) - 5', out: '-2.0', ok: true,
        d: '正确。代价是一次 dtype 转换，142 万个像素也只是一次内存遍历。' },
      { cc: 2, t: 'float16 输入被顺手提升', code: 'image.astype(np.float32)', out: 'float32', ok: false,
        d: '注释第二句专门防这个：已经是浮点就**不动它**，否则显存翻倍而没人发现。' },
    ];
    const els = cases.map(c => {
      const card = W.card({ cc: c.cc, tint: c.ok ? 4 : c.cc,
        title: '<span class="mono" style="font-size:11px">' + c.t + '</span>' });
      const line = U.el('div', { class: 'row gap10 center', style: 'width:100%;margin:6px 0 2px' });
      line.appendChild(U.el('div', { class: 'mono', style: 'font-size:11.5px;color:var(--ink-dim)', text: c.code }));
      line.appendChild(U.el('div', { class: 'mono', style: 'font-size:10px;color:var(--ink-faint)', text: '→' }));
      line.appendChild(U.el('div', { class: 'n', style: 'font-size:14px;color:' + (c.ok ? 'var(--c4)' : 'var(--c3)'), text: c.out }));
      card.appendChild(line);
      card.appendChild(U.el('div', { class: 'cs', html: c.d }));
      g.appendChild(card);
      return { card: card, c: c };
    });
    viz.appendChild(g);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    els.forEach(e => { e.card.style.opacity = '.3'; });
    msg.innerHTML = '<span class="cm">// 三种 dtype，同一条算式</span>';
    els.forEach((e, i) => {
      tl.at(1800 + i * 2450, () => {
        els.forEach(x => { x.card.style.opacity = '.3'; x.card.classList.remove('ac'); });
        e.card.style.opacity = '1'; e.card.classList.add('ac');
        msg.innerHTML = e.c.ok
          ? '★ <em>astype(np.float32)</em> —— 就为了这一行不出错'
          : '<span class="cm">// ' + e.c.t + '</span>';
      });
    });
    tl.at(10000, () => {
      els.forEach(x => { x.card.style.opacity = '1'; x.card.classList.remove('ac'); });
      msg.innerHTML = 'cast 完才轮到广播：<em>mean = [mean] * num_channels</em>，标量写法被显式支持';
    });
    tl.at(12600, () => {
      msg.innerHTML = '两处 <em>raise ValueError</em> 都带上了元素个数 —— 对不上就报错，不静默截断';
    });
    tl.at(15200, () => {
      msg.innerHTML = '<span class="cm">// 形状一路不变：(C, H, W) 进、(C, H, W) 出</span>';
    });
  },
},

/* ------------------------------------------------- 4 两版切块 */
{
  kicker: '步 ② · 切块',
  title: '两版切块：<span class="hl-a">光栅序列表</span> vs <span class="hl-b">张量重排</span>',
  sub: 'divide_to_patches 是两个 for 循环的朴素版；模型真正用的是 patchify —— 一次 reshape 加一次 permute。两者等价，但只有一个能 batch。',
  caption: '要跳过若干行就必须拆成两个引用块 —— 这两段在原文件里相隔 1111 行，中间隔着整个 bbox / pad 工具区。',
  lang: 'python',
  codeStart: 854,
  code: `    patch_h, patch_w = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
    patches = []
    height, width = get_image_size(image, channel_dim=ChannelDimension.FIRST)
    for i in range(0, height, patch_h):
        for j in range(0, width, patch_w):
            patch = image[..., i : i + patch_h, j : j + patch_w]
            patches.append(patch)

    return patches`,
  codeNote: 'divide_to_patches 的主体：行优先遍历，返回 patch 列表（光栅序）。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap10 vizgrow hstart' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '同一个 6×6 图：左 = 逐个取子块，右 = 一次 reshape' }));

    const row = U.el('div', { class: 'row gap20', style: 'width:100%;justify-content:center;align-items:flex-start' });

    /* 左：6x6 网格，2x2 一组，逐块点亮 */
    const leftCol = U.el('div', { class: 'col gap6 center' });
    leftCol.appendChild(U.el('div', { class: 'mono dim', style: 'font-size:10.5px', text: 'divide_to_patches' }));
    const gt = W.tensor(6, 6, (r, c) => (Math.floor(r / 2) + Math.floor(c / 2)) % 6, { cell: 44, gap: 3 });
    leftCol.appendChild(gt);
    leftCol.appendChild(U.el('div', { class: 'mono faint', style: 'font-size:10px', text: '列表顺序 = 光栅序' }));
    row.appendChild(leftCol);

    const mid = U.el('div', { class: 'col center gap6', style: 'flex:none' });
    mid.innerHTML = '<div style="font-size:20px;color:var(--accent)">⟹</div>'
      + '<div class="mono" style="font-size:10px;color:var(--c2)">等价</div>';
    row.appendChild(mid);

    /* 右：一维 36 格，2x2 分组 */
    const rightCol = U.el('div', { class: 'col gap6 center' });
    rightCol.appendChild(U.el('div', { class: 'mono dim', style: 'font-size:10.5px', text: 'patchify → permute(0,2,5,3,6,1,4,7)' }));
    const strip = U.el('div', { class: 'row wrap gap4', style: 'width:282px' });
    const cells = [];
    for (let i = 0; i < 36; i++) {
      const grp = Math.floor(i / 4);
      const e = U.el('div', {
        class: 'n sm',
        style: 'width:44px;height:44px;border-radius:7px;display:flex;align-items:center;'
             + 'justify-content:center;font-size:11px;border:1px solid var(--c' + (grp % 6) + ');flex:none;'
             + 'background:rgba(150,180,255,.07);transition:opacity .3s,transform .3s',
        title: '位置 ' + i,
        text: String(i),
      });
      strip.appendChild(e); cells.push(e);
    }
    rightCol.appendChild(U.el('div', { class: 'mono faint', style: 'font-size:10px;text-align:center', text: '展平后的位置编号：每 4 格 = 1 个 token' }));
    rightCol.appendChild(strip);
    rightCol.appendChild(U.el('div', { class: 'mono faint', style: 'font-size:10px',
      text: '行优先：同一行的 2×2 块先排完，再排下一行块' }));
    row.appendChild(rightCol);
    viz.appendChild(row);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    const paint = (k) => {
      gt.cells.forEach((e, i) => {
        const r = Math.floor(i / 6), c = i % 6;
        const on = (Math.floor(r / 2) * 3 + Math.floor(c / 2)) === k;
        e.style.opacity = on ? '1' : '.22';
        e.style.transform = on ? 'scale(1.1)' : 'none';
      });
      cells.forEach((e, i) => {
        const on = Math.floor(i / 4) === k;
        e.style.opacity = on ? '1' : '.22';
        e.style.transform = on ? 'translateY(-3px)' : 'none';
      });
    };
    paint(0);
    msg.innerHTML = '<span class="cm">// 第 0 个 2×2 块：左图 4 个格子 → 右列 4 个连续位置</span>';
    tl.at(1600, () => { paint(3); msg.innerHTML = '第 <em>3</em> 块 &nbsp;<span class="cm">// 左图是跳着的，右列永远连续</span>'; });
    tl.at(4400, () => { paint(6); msg.innerHTML = '第 <em>6</em> 块 &nbsp;<span class="cm">// 这就是"合并"：合并的准备工作在切块时就做完了</span>'; });
    tl.at(7200, () => {
      gt.cells.forEach(e => { e.style.opacity = '1'; e.style.transform = 'none'; });
      cells.forEach(e => { e.style.opacity = '1'; e.style.transform = 'none'; });
      msg.innerHTML = '两种实现<em>结果等价</em>：都是"块优先、块内其次"的顺序';
    });
    tl.at(10400, () => {
      msg.innerHTML = '区别只在<em>能不能 batch</em>：列表版要自己 <span class="fn">torch.stack</span>，张量版天生带 batch 维';
    });
    tl.at(13200, () => {
      msg.innerHTML = '★ <em>8 维 reshape + 1 次 permute</em>，没有循环 —— 这是 patchify 比列表版快的原因';
    });
    tl.at(15600, () => {
      msg.innerHTML = '<span class="cm">// 下一幕：把 576 和 4 代进真正的参数，算一次 token 数</span>';
    });
  },
},

/* ------------------------------------------------- 5 144 */
{
  kicker: '步 ③ · 合并',
  title: '算一遍：336×336 的图是 <span class="hl-c">144</span> 个 token',
  sub: '把 patch_size=14、spatial_merge_size=2 代进去，两个除法就得出序列要给视觉留多少位置。',
  caption: '验收点 1 就是这一步：576 / 4 = 144。数字不是估的，是 grid_thw 算出来的。',
  lang: 'python',
  codeStart: 1096,
  code: `    if not is_nested:
        return [
            processed_images[grouped_images_index[i][0]][grouped_images_index[i][1]]
            for i in range(len(grouped_images_index))
        ]

    return _reconstruct_nested_structure(grouped_images_index, processed_images)`,
  codeNote: 'image_transforms.reorder_images 的主体 —— 把算完的 144 个 token 按原顺序还回"第几张图"。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '336 × 336 的三次除法' }));

    const f = (txt, cc) => {
      const e = U.el('div', {
        class: 'formula cc' + cc,
        style: 'width:100%;font-size:13.5px;opacity:.28',
        html: txt,
      });
      viz.appendChild(e);
      return e;
    };
    const f1 = f('grid_h = grid_w = <em>336 / 14</em> = <em>24</em> &nbsp;<span class="cm">// patch_size = 14</span>', 0);
    const f2 = f('patches = grid_h * grid_w = <em>24 × 24</em> = <em>576</em> &nbsp;<span class="cm">// 一张图的 patch 数</span>', 2);
    const f3 = f('merge_length = merge_size² = <em>2 × 2</em> = <em>4</em> &nbsp;<span class="cm">// spatial_merge_size = 2</span>', 4);
    const f4 = f('视觉 token = <em>576 / 4</em> = <em>144</em> &nbsp;<span class="cm">// ← 序列要留的位置数</span>', 3);

    /* 24x24 缩略网格，4 格一组点亮 */
    const gridWrap = U.el('div', { class: 'col gap6 center', style: 'width:100%' });
    gridWrap.appendChild(U.el('div', { class: 'klabel', text: 'grid 24×24 = 576 个 patch，每 2×2 合成 1 个 token' }));
    const g = W.tensor(24, 24, (r, c) => Math.floor(r / 2) * 12 + Math.floor(c / 2), { cell: 9, gap: 1 });
    g.cells.forEach(e => { e.style.opacity = '.14'; });
    gridWrap.appendChild(g);
    const cnt = U.el('div', { class: 'n', style: 'font-size:15px;color:var(--accent)', text: 'token 计数：0' });
    gridWrap.appendChild(cnt);
    gridWrap.appendChild(U.el('div', { class: 'mono faint', style: 'font-size:10.5px;text-align:center',
      text: '24 × 24 个 patch → 12 × 12 个 token 位置　（每 2×2 格合并成 1 个）' }));
    viz.appendChild(gridWrap);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    const light = (n) => {
      g.cells.forEach((e, i) => {
        const r = Math.floor(i / 24), c = i % 24;
        const k = Math.floor(r / 2) * 12 + Math.floor(c / 2);
        e.style.opacity = k < n ? '1' : '.14';
      });
      cnt.textContent = 'token 计数：' + n;
    };

    msg.innerHTML = '<span class="cm">// 从 patch_size 开始推</span>';
    tl.at(1600, () => { f1.style.opacity = '1'; msg.innerHTML = '一条边 336 像素、一个 patch <em>14</em> 像素 → <em>24</em> 段'; });
    tl.at(4400, () => { f2.style.opacity = '1'; msg.innerHTML = '于是 <em>24 × 24 = 576</em> 个 patch —— 右下角的格子数'; });
    tl.at(7600, () => {
      f3.style.opacity = '1'; light(12);
      msg.innerHTML = '<em>2×2</em> 个 patch 合成一个 token，所以先亮 12 个 <span class="cm">// 一行</span>';
    });
    tl.at(10800, () => {
      light(144);
      msg.innerHTML = '★ 全部点亮 = <em>144</em> 个 token。这就是序列里要插的视觉占位符个数';
    });
    tl.at(13400, () => {
      f4.style.opacity = '1';
      msg.innerHTML = '<span class="cm">// 一行代码：grid_thw.prod() // merge_length</span>';
    });
    tl.at(15600, () => {
      msg.innerHTML = '★ 记住这个数：<em>144</em>。它在 L4 的注意力里会变成 144 个 KV 位置';
    });
  },
},

/* ------------------------------------------------- 6 批量分组 */
{
  kicker: '工程 · 批量',
  title: '<span class="hl-a">group_images_by_shape</span>：一批图怎么一起算',
  sub: '一张图算完了，一批尺寸不同的图怎么办？答案是"按形状分组 → 组内堆叠 → 一起算 → 还原顺序"。',
  caption: '注意 pair 出去的第二个字典：它让跟着图片一起重排的附属数据（比如 grid_thw）不会错位。',
  lang: 'python',
  codeStart: 1029,
  code: `    # If disable grouping is not explicitly provided, we favor disabling it if the images are on CPU, and enabling it otherwise.
    if disable_grouping is None:
        device = _get_device_from_images(images, is_nested)
        disable_grouping = device.type == "cpu"

    if disable_grouping:
        grouped_images_index = {key: (key, 0) for key, _ in _iterate_items(images, is_nested)}
        if is_nested:
            grouped_images_index["_num_sublists"] = len(images)

        grouped_images = {key: img.unsqueeze(0) for key, img in _iterate_items(images, is_nested)}
        paired_grouped_values = [
            dict.fromkeys(grouped_images, None)
            if paired_list is None
            else {key: [item] for key, item in _iterate_items(paired_list, is_nested)}
            for paired_list in paired_inputs
        ]

        return grouped_images, *paired_grouped_values, grouped_images_index`,
  codeNote: 'group_images_by_shape 的 disable_grouping 分支：CPU 上每个图各自成一个 batch=1 的组。',
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    wrap.appendChild(U.el('div', { class: 'klabel', text: '3 张不同尺寸的图，两种分支的走法' }));
    const fl = W.flow([
      { t: '输入', s: '3 张图', cc: 0 },
      { t: '按 shape 分组', s: 'shape → [图]', cc: 2 },
      { t: '组内 stack', s: 'batch=组内张数', cc: 4 },
      { t: '一起 forward', s: '一次 kernel', cc: 1 },
      { t: 'reorder_images', s: '还原顺序', cc: 3 },
    ], { style: 'width:100%' });
    wrap.appendChild(fl);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    /* 两组卡：CPU 分支 / GPU 分支 */
    const mkBranch = (cc, title, sub, rows) => {
      const c = W.card({ cc: cc, tint: cc, title: title, sub: sub });
      const t = W.table(rows, { head: ['图', 'shape', '分组结果'] });
      c.appendChild(U.el('div', { style: 'width:100%;margin-top:8px' }, [t]));
      return c;
    };
    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cpu = mkBranch(2, 'disable_grouping = True（CPU）',
      '不分组。每张图 <code class="inl">unsqueeze(0)</code>，索引表退化成 <code class="inl">{i: (i, 0)}</code>。',
      [['#0', '(3, 336, 336)', '<span class="hl">batch=1</span>'],
       ['#1', '(3, 512, 512)', '<span class="hl">batch=1</span>'],
       ['#2', '(3, 336, 336)', '<span class="hl">batch=1</span>']]);
    const gpu = mkBranch(4, 'disable_grouping = False（GPU）',
      '按 <code class="inl">image.shape[1:]</code> 分组，同形状的叠成一个 batch。',
      [['#0', '(3, 336, 336)', '<span class="hl4">batch=2</span>'],
       ['#2', '(3, 336, 336)', '<span class="hl4">batch=2</span>'],
       ['#1', '(3, 512, 512)', '<span class="hl">batch=1</span>']]);
    cpu.style.flex = '1'; gpu.style.flex = '1';
    row.appendChild(cpu); row.appendChild(gpu);
    viz.appendChild(row);

    const sizes = W.table([
      ['第 0、2 张', '336 / 14 = 24', '<span class="hl">24 × 24 = 576</span>', '分成一组，batch=2'],
      ['第 1 张', '512 / 14 = 36.57…', '<span class="hl">36 × 36 = 1296</span>', '各自一组，batch=1'],
    ], { head: ['图', 'grid_h = H / 14', 'grid_h × grid_w', '分组结果'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 4, style: 'padding:10px 12px;width:100%' }, [
      U.el('div', { class: 'klabel', text: '★ 为什么必须分组：不同尺寸的 grid 不能堆在同一个张量里' }), sizes,
      U.el('div', { class: 'cs', style: 'margin-top:7px;font-size:11px',
        html: '整除掉的部分被丢掉（<code class="inl">//</code>）—— 这正是真实流程里'
            + '必须先 <code class="inl">smart_resize</code> 把边补到 28 的倍数的原因' }),
    ]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    cpu.style.opacity = '.3'; gpu.style.opacity = '.3';
    msg.innerHTML = '<span class="cm">// 默认值不是拍脑袋定的</span>';
    tl.at(1800, () => {
      fl.focus(0); cpu.style.opacity = '1';
      msg.innerHTML = 'CPU：<em>device.type == "cpu"</em> → 不分组 <span class="cm">// 注释里写了依据：PR #38157</span>';
    });
    tl.at(5200, () => {
      fl.focus(2); gpu.style.opacity = '1';
      msg.innerHTML = 'GPU：分组后 <span class="fn">torch.stack</span>，把同形状的图叠成一个 batch';
    });
    tl.at(8600, () => {
      msg.innerHTML = '★ 为什么 CPU 不分组：<em>stack 的拷贝比省下的批处理收益还贵</em> —— 这是实测结论';
    });
    tl.at(11600, () => {
      fl.focus(4);
      msg.innerHTML = '两种分支的<em>输出结构完全一致</em>：都是"字典 + 索引表"，所以下游只有一份代码';
    });
    tl.at(14600, () => {
      fl.focus(-1);
      msg.innerHTML = '还原交给 <span class="fn">reorder_images</span>：<em>分组只影响计算顺序，不影响输出顺序</em>';
    });
    tl.at(16800, () => {
      msg.innerHTML = '<span class="cm">// 下一幕：把同样的机制搬到视频上 —— 多了一个"抽哪几帧"的决策</span>';
    });
  },
},

/* ------------------------------------------------- 7 视频采样 */
{
  kicker: '视频 · 采样',
  title: '视频 = 同一套变换 + 先决定 <span class="hl-a">抽哪几帧</span>',
  sub: '视频处理器不新写一套 resize，它只在前面加一步采样；采样结果和帧索引一起被打包进 metadata。',
  caption: '视频抽帧是有损的：30fps 抽 16 帧之后，模型看到的时间密度和原始视频完全不同 —— 所以采样率必须留档。',
  lang: 'python',
  codeStart: 285,
  code: `def get_uniform_frame_indices(total_num_frames: int, num_frames: int | None = None):
    """
    Creates a numpy array for uniform sampling of \`num_frame\` frames from \`total_num_frames\`
    when loading a video.

    Args:
        total_num_frames (\`int\`):
            Total number of frames that a video has.
        num_frames (\`int\`, *optional*):
            Number of frames to sample uniformly. If not specified, all frames are sampled.

    Returns:
        np.ndarray: np array of frame indices that will be sampled.
    """
    if num_frames is not None:
        indices = np.arange(0, total_num_frames, total_num_frames / num_frames).astype(int)
    else:
        indices = np.arange(0, total_num_frames).astype(int)
    return indices`,
  codeNote: 'video_utils.get_uniform_frame_indices —— 步长不取整，结果再截断成 int。',
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    /* 帧条：30 格代表 30 帧，按 6 帧采样（演示比例） */
    const TOTAL = 30, PICK = 6;
    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '30 帧抽 6 帧：arange(0, 30, 30/6) = 0,5,10,15,20,25' }));

    const strip = U.el('div', { class: 'row gap3', style: 'width:100%;align-items:flex-end;height:80px' });
    const cells = [];
    for (let i = 0; i < TOTAL; i++) {
      const picked = (i * PICK) % TOTAL === 0;
      const e = U.el('div', {
        class: 'n sm',
        style: 'flex:1 1 0;height:64px;border-radius:5px;display:flex;align-items:center;'
             + 'justify-content:center;font-size:10px;'
             + 'border:1px solid rgba(150,180,255,.18);background:rgba(150,180,255,.06);'
             + 'color:var(--ink-dim);transition:height .5s var(--ease-out),background .5s,color .5s,border-color .5s',
        title: 'frame ' + i,
        text: String(i),
      });
      strip.appendChild(e); cells.push(e);
    }
    viz.appendChild(strip);

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cA = W.card({ cc: 4, tint: 4, title: 'VideoMetadata 留下了什么',
      sub: '<code class="inl">total_num_frames</code> / <code class="inl">fps</code> / <code class="inl">frames_indices</code> —— 抽到的索引是原视频里的绝对位置。' });
    const cB = W.card({ cc: 2, tint: 2, title: 'sampled_fps = len(indices)/total × fps',
      sub: '30fps 抽 6 帧 ≈ 6fps。时间密度变了，这个数字必须留档，否则 mrope 的时间轴会算错。' });
    cA.style.flex = '1'; cB.style.flex = '1';
    row.appendChild(cA); row.appendChild(cB);
    viz.appendChild(row);

    const cC = W.card({ cc: 1, tint: 1, title: '算一遍：30 帧的 16fps 视频抽 6 帧',
      sub: '<code class="inl">arange(0, 30, 30/6)</code> = <code class="inl">[0, 5, 10, 15, 20, 25]</code> —— 步长恰好整除时是"每组头一个"。' });
    const t2 = W.table([
      ['步长', '30 / 6 = 5.0', '整除 → 索引均匀'],
      ['时间戳', 'idx / fps = 0, 0.167, …', '还原成原视频里的秒数'],
      ['sampled_fps', '6 / 30 × 30 = 6.0', '模型看到的有效帧率'],
    ], { head: ['量', '计算', '含义'] });
    cC.appendChild(U.el('div', { style: 'width:100%;margin-top:8px' }, [t2]));
    cC.style.width = '100%'; cC.style.opacity = '.3';
    viz.appendChild(cC);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    const paint = (idx) => {
      cells[idx].style.borderColor = 'var(--c4)';
      cells[idx].style.background = 'rgba(52,211,153,.30)';
      cells[idx].style.color = '#fff';
      cells[idx].style.height = '88px';
    };
    msg.innerHTML = '<span class="cm">// 先看要抽哪些帧</span>';
    tl.at(1800, () => {
      /* 六个抽中的帧一起亮，靠 transition-delay 错峰 —— 一次事件，六帧依次弹起 */
      [0, 5, 10, 15, 20, 25].forEach((idx, k) => {
        cells[idx].style.transitionDelay = (k * 260) + 'ms';
        paint(idx);
      });
      msg.innerHTML = '抽中的 6 帧依次弹起 &nbsp;<span class="cm">// arange(0, 30, 5) = 0,5,10,15,20,25</span>';
    });
    tl.at(4200, () => {
      cA.classList.add('ac'); cA.style.opacity = '1';
      msg.innerHTML = '★ 抽到的帧号<em>不是</em> 0..5，而是 0,5,10,15,20,25 —— 存的是原视频里的位置';
    });
    tl.at(6600, () => {
      cA.classList.remove('ac'); cB.classList.add('ac'); cB.style.opacity = '1';
      msg.innerHTML = '★ 所以 <em>sampled_fps</em> 远小于原始 fps：模型看到的是"稀疏的时间"';
    });
    tl.at(9000, () => {
      cB.classList.remove('ac'); cC.classList.add('ac'); cC.style.opacity = '1';
      msg.innerHTML = '三个量算一遍：<em>步长 5</em> / <em>时间戳 = idx / fps</em> / <em>sampled_fps = 6</em>';
    });
    tl.at(11400, () => {
      [cA, cB, cC].forEach(c => { c.style.opacity = '1'; c.classList.remove('ac'); });
      msg.innerHTML = '★ 采样是有损的：<em>模型看到的时间密度和原始视频不同</em>，必须留档';
    });
    tl.at(13800, () => {
      msg.innerHTML = '<span class="cm">// 抽完帧才轮到 temporal patch：帧数本身还有一条硬约束</span>';
    });
  },
},

/* ------------------------------------------------- 8 采样接口 */
{
  kicker: '视频 · 采样接口',
  title: '采样的默认值、<span class="hl-a">互斥</span>与硬边界',
  sub: '样本数从哪来？类属性给默认值，fps 与 num_frames 互斥，超过总帧数直接报错 —— 三条规则都在同一个函数的前 20 行里。',
  caption: '同一个决策在 video_utils 里还有一份 numpy 实现（get_uniform_frame_indices）—— 本课第二次看到"同名不同源"。',
  lang: 'python',
  codeStart: 135,
  code: `    def sample_frames(
        self,
        metadata: VideoMetadata,
        num_frames: int | None = None,
        fps: int | float | None = None,
        **kwargs,
    ):
        """
        Default sampling function which uniformly samples the desired number of frames between 0 and total number of frames.
        If \`fps\` is passed along with metadata, \`fps\` frames per second are sampled uniformly. Arguments \`num_frames\`
        and \`fps\` are mutually exclusive.

        Args:
            metadata (\`VideoMetadata\`):
                Metadata of the video containing information about total duration, fps and total number of frames.
            num_frames (\`int\`, *optional*):
                Maximum number of frames to sample. Defaults to \`self.num_frames\`.
            fps (\`int\` or \`float\`, *optional*):
                Target frames to sample per second. Defaults to \`self.fps\`.

        Returns:
            np.ndarray:
                Indices to sample video frames.
        """`,
  codeNote: 'BaseVideoProcessor.sample_frames 的签名与 docstring —— "uniformly" 与 "mutually exclusive" 是两个关键词。',
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '三张卡片：默认值 / 互斥 / 边界' }));

    const mk = (cc, title, sub, rows) => {
      const c = W.card({ cc: cc, tint: cc, title: title, sub: sub });
      const t = W.table(rows, { head: ['参数', '取值', '含义'] });
      c.appendChild(U.el('div', { style: 'width:100%;margin-top:8px' }, [t]));
      return c;
    };
    const c1 = mk(0, '① 从哪来',
      '显式参数 → 类属性 → metadata，三级回退。基类那一排 <code class="inl">None</code> 类属性就是给子类覆盖用的。',
      [['num_frames', 'self.num_frames', '类属性默认（GLM-5 视频处理器 = 16）'],
       ['fps', 'self.fps', '类属性默认（= 2）'],
       ['total_num_frames', 'metadata', '<span class="hlbad">必须由解码器给出</span>']]);
    const c2 = mk(3, '② 互斥规则',
      '<code class="inl">num_frames</code>、<code class="inl">fps</code>、<code class="inl">sample_indices_fn</code> 三者只能给一个。',
      [['同给两个', '<span class="hlbad">raise</span>', 'ValueError：mutually exclusive'],
       ['只给 fps', '<span class="hl4">换算</span>', 'int(total / metadata.fps × fps)'],
       ['都不给', '全都要', '退化成"不采样"']]);
    const c3 = mk(2, '③ 硬边界',
      '要的帧数超过总帧数**直接报错**，不静默截断、不重复补帧。',
      [['num_frames > total', '<span class="hlbad">raise</span>', '宁可在处理器层炸，也不让形状玄学流到模型里'],
       ['采样方式', '<span class="hl4">等间隔</span>', 'arange(0, N, N/n)，步长不取整'],
       ['结果', 'int 索引', '原视频里的绝对帧号']]);
    [c1, c2, c3].forEach(c => { c.style.width = '100%'; viz.appendChild(c); });

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    [c1, c2, c3].forEach(c => { c.style.opacity = '.3'; });
    const pick = (k) => {
      [c1, c2, c3].forEach((c, i) => {
        c.style.opacity = (i === k) ? '1' : '.3';
        c.classList.toggle('ac', i === k);
      });
    };
    pick(0);
    msg.innerHTML = '<span class="cm">// 三级回退：先看你传了什么</span>';
    tl.at(1800, () => { pick(0); msg.innerHTML = '① 什么都不传就用类属性；<em>total_num_frames 只能来自 metadata</em> —— 所以解码必须先跑一步'; });
    tl.at(4800, () => { pick(1); msg.innerHTML = '② ★ 互斥：<em>fps 与 num_frames 只能给一个</em>，两个都给直接 raise'; });
    tl.at(7800, () => { msg.innerHTML = '只给 <em>fps</em> 时反推帧数：<em>int(total / metadata.fps × fps)</em> —— 这就是 metadata 必须带 fps 的原因'; });
    tl.at(10800, () => { pick(2); msg.innerHTML = '③ ★ 硬边界：<em>要的帧数超过总帧数 → raise</em>，不截断、不补帧'; });
    tl.at(13800, () => {
      [c1, c2, c3].forEach(c => { c.style.opacity = '1'; c.classList.remove('ac'); });
      msg.innerHTML = '三条规则合起来保证一件事：<em>采样结果一定是"总帧数以内的、等间隔的 n 帧"</em>';
    });
    tl.at(16400, () => {
      msg.innerHTML = '<span class="cm">// 但 n 是奇数时还有一道坎 —— 下一幕：temporal patch</span>';
    });
  },
},

/* ------------------------------------------------- 9 temporal patch */
{
  kicker: '视频 · 折时间维',
  title: '<span class="hl-c">temporal_patch_size = 2</span> 对帧数的要求',
  sub: '视频切块多一维：T 帧被折进 patch 的宽度，而不是变成新 token。所以帧数必须是 temporal_patch_size 的整数倍。',
  caption: '奇数帧不会报错 —— 会补最后一帧。这个"静默补齐"是理解 temporal patch 的关键。',
  lang: 'python',
  codeStart: 325,
  code: `        # Check that videos have \`num_frames\` divisible by \`temporal_patch_size\`
        if pad := -num_frames % temporal_patch_size:
            repeats = videos[:, -1:].expand(-1, pad, -1, -1, -1)
            videos = torch.cat((videos, repeats), dim=1)
            num_frames += pad

        grid_t = num_frames // temporal_patch_size
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size`,
  codeNote: 'Glm5NextVideoProcessor.patchify 的开头七行 —— 帧数不足整数倍就补，补的是最后一帧。不计入本课覆盖率（属 L2-06）。',
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: 'T 帧 → grid_t：每 temporal_patch_size 帧合成 1 格' }));

    const mk = (label, n, cc) => {
      const col = U.el('div', { class: 'col gap6 center', style: 'flex:1' });
      col.appendChild(U.el('div', { class: 'mono dim', style: 'font-size:10.5px', text: label }));
      const r = U.el('div', { class: 'row gap4 center', style: 'width:100%' });
      const cs = [];
      for (let i = 0; i < n; i++) {
        const e = U.el('div', {
          class: 'n sm',
          style: 'width:34px;height:34px;border-radius:7px;display:flex;align-items:center;'
               + 'justify-content:center;font-size:10px;'
               + 'border:1px solid var(--c' + cc + ');background:rgba(150,180,255,.09)',
          text: 'T' + i,
        });
        r.appendChild(e); cs.push(e);
      }
      col.appendChild(r);
      col._cells = cs;
      return col;
    };

    const stage = U.el('div', { class: 'row gap16 center', style: 'width:100%' });
    const cEven = mk('偶数：num_frames = 8', 8, 4);
    stage.appendChild(cEven);
    viz.appendChild(stage);

    const pair = U.el('div', { class: 'col gap6 center', style: 'width:100%' });
    pair.appendChild(U.el('div', { class: 'mono faint', style: 'font-size:10.5px', text: 'grid_t = 8 / 2 = 4 格' }));
    const pr = U.el('div', { class: 'row gap8 center' });
    for (let i = 0; i < 4; i++) {
      pr.appendChild(U.el('div', {
        class: 'n', style: 'width:76px;height:34px;border-radius:8px;display:flex;align-items:center;'
          + 'justify-content:center;border:1px solid var(--c1);background:rgba(167,139,250,.20);color:#ddd6fe',
        text: 'grid ' + i,
      }));
    }
    pair.appendChild(pr);
    viz.appendChild(pair);

    /* 奇数补帧演算 */
    const calc = U.el('div', { class: 'card', cc: 3, style: 'padding:10px 12px;width:100%' });
    calc.innerHTML = '<span class="klabel">帧数与 token 数（视频）</span>'
      + '<div class="mono" style="font-size:11px;line-height:1.85;color:var(--ink-dim)">'
      + 'grid_t = num_frames // temporal_patch_size<br>'
      + 'tokens = <span style="color:#cfe6ff">grid_t × grid_h × grid_w // merge_length</span><br>'
      + '　　　 = 8 × 24 × 24 // 4 = <span class="hl3">1152</span>　（16 帧，每帧 72 个）</div>';
    viz.appendChild(calc);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    cEven._cells.forEach((e, i) => { e.style.opacity = i < 2 ? '1' : '.2'; });
    msg.innerHTML = '<span class="cm">// temporal_patch_size = 2，所以两帧一组</span>';
    tl.at(1800, () => {
      msg.innerHTML = '第 <em>0、1</em> 帧合成一个 grid 格 &nbsp;<span class="cm">// T 被折进 patch 的宽度，不是新 token</span>';
    });
    tl.at(4200, () => {
      cEven._cells.forEach((e, i) => { e.style.opacity = i < 4 ? '1' : '.2'; });
      msg.innerHTML = '第 <em>2、3</em> 帧 → 第 2 个 grid 格；8 帧一共 <em>4</em> 格';
    });
    tl.at(6600, () => {
      cEven._cells.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = '★ <em>grid_thw 里的 t 是 grid_t，不是原始帧数</em>：16 帧的视频 t = 8';
    });
    tl.at(9000, () => {
      calc.classList.add('ac');
      msg.innerHTML = '于是 token 数是 <em>grid_t × grid_h × grid_w // merge_length</em>，不是按原始帧数算';
    });
    tl.at(11400, () => {
      calc.classList.remove('ac');
      msg.innerHTML = '★ 奇数帧呢？<em>-num_frames % 2</em> = 1 → <em>重复最后一帧</em>补成偶数';
    });
    tl.at(13800, () => {
      msg.innerHTML = '★ 补的是<em>最后一帧</em>而不是补零 —— 补零会在时间维上造出一段假的"黑场"';
    });
    tl.at(16000, () => {
      msg.innerHTML = '<span class="cm">// 不报错、不丢帧 —— 静默补齐。这就是 temporal_patch_size=2 的要求</span>';
    });
  },
},

/* ------------------------------------------------- 10 收束 */
{
  kicker: '收束',
  title: '把三步压成一张表',
  sub: '六个文件、5799 行，最后落到两个数上：336×336 的图是 144 个 token；视频的帧数必须是 temporal_patch_size 的整数倍。',
  caption: '下一课 L2-06：这些视觉 token 和文本 token 到底怎么被拼进同一条序列。',
  lang: 'python',
  codeStart: 1096,
  code: `    if not is_nested:
        return [
            processed_images[grouped_images_index[i][0]][grouped_images_index[i][1]]
            for i in range(len(grouped_images_index))
        ]

    return _reconstruct_nested_structure(grouped_images_index, processed_images)`,
  codeNote: 'image_transforms.reorder_images —— 分组、堆叠、还原，最后一步只有 7 行。',
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const tb = W.table([
      ['1 重采样', 'resize', '<span class="hl">(3, 336, 336)</span>', '只改数值', 'image_transforms / image_utils 各一份'],
      ['2 归一化', 'normalize', '<span class="hl">(3, 336, 336)</span>', '先 cast 再算', 'uint8 回绕 / float16 提升'],
      ['3 切块', 'patchify', '<span class="hl">(576, 588)</span>', '只改形状', '8 维 reshape + permute'],
      ['4 分组', 'group_by_shape', 'batch 维可变', '设备相关默认值', 'CPU 不分组（PR #38157）'],
      ['5 采样', 'sample_frames', 'T 帧 → n 帧', '有损且要留档', 'sampled_fps 必须记下来'],
      ['6 折时间', 'temporal patch', 'T → grid_t', '帧数必须整除', '奇数补最后一帧'],
      ['7 Token', 'prod() // merge_length', '<span class="hl3">144</span>', '序列要留的位置', '本课验收点 1'],
    ], { head: ['步', '函数', '形状', '性质', '坑'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 0, style: 'padding:10px 12px;width:100%' }, [tb]));

    viz.appendChild(W.exercise(
      '一张 <b>672×336</b> 的图，用 patch_size=14、spatial_merge_size=2，'
      + '会有多少个视觉 token？',
      'grid_h = 672/14 = <b>48</b>，grid_w = 336/14 = <b>24</b>；'
      + 'patches = 48×24 = <b>1152</b>；merge_length = 2² = <b>4</b>；'
      + 'tokens = 1152/4 = <b>288</b>。<br>'
      + '注意：这里假设尺寸已经对齐到 patch×merge 的整数倍（14×2=28 的倍数）。'
      + '真实流程里如果边不是 28 的倍数，会先补到 28 的倍数再切 —— '
      + '这就是 <code class="inl">smart_resize</code> 存在的原因。'));

    viz.appendChild(U.el('div', { class: 'card', cc: 1, style: 'padding:10px 12px;width:100%' }, [
      U.el('div', { class: 'klabel', text: '★ 本课三个数字，都可以从公式独立复算' }),
      W.table([
        ['336 × 336 单帧', '576 / 4', '<span class="hl3">144</span>', '图像占位符个数'],
        ['16 帧视频', '8 × 24 × 24 / 4', '<span class="hl3">1152</span>', '每帧 72 个 token'],
        ['patch 展平宽度', '3 × 1 × 14 × 14', '<span class="hl3">588</span>', '视觉塔的输入特征维'],
      ], { head: ['输入', '算式', '结果', '用途'] }),
    ]));

    viz.appendChild(W.exercise(
      '一个视频抽出了 <b>17</b> 帧，temporal_patch_size=2，会发生什么？'
      + '如果 temporal_patch_size 改成 4 呢？',
      '17 是奇数，<code class="inl">-17 % 2 = 1</code>，所以 <b>重复最后一帧</b>补成 18 帧，'
      + 'grid_t = 18/2 = <b>9</b>，不报错。<br>'
      + '若 <code class="inl">temporal_patch_size=4</code>：'
      + '<code class="inl">-17 % 4 = 3</code>，补 <b>3</b> 帧成 20，grid_t = 5。<br>'
      + '★ 关键不是"必须是偶数"，而是<b>必须是 temporal_patch_size 的整数倍</b> —— '
      + 'temporal_patch_size=2 时"偶数"只是它的一个特例。'));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 七个格子，逐个点亮</span>';
    tl.at(1800, () => { msg.innerHTML = '重采样和归一化<em>不改形状</em> —— 它们只让数值进入模型好学的范围'; });
    tl.at(4600, () => { msg.innerHTML = '切块<em>只改形状</em> —— 一次纯重排，不损失任何数值'; });
    tl.at(7400, () => { msg.innerHTML = '★ <em>576 / 4 = 144</em>：一个除法决定了文本序列要留多少位置'; });
    tl.at(10200, () => { msg.innerHTML = '视频多两步：<em>先采样再折时间</em>，而且两步都会丢信息，都要留档'; });
    tl.at(12000, () => { msg.innerHTML = '★ 视频帧数必须是 <em>temporal_patch_size 的整数倍</em>，不是就补最后一帧'; });
    tl.at(14200, () => { msg.innerHTML = '★ 144 / 1152 / 588 三个数都能从公式独立复算 —— 这才是"读懂了"'; });
    tl.at(16400, () => { msg.innerHTML = '下一课 L2-06：这 144 个位置怎么和文本 token 拼成同一条序列'; });
  },
},

];
