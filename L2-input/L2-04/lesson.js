/* ==========================================================================
   L2-04 · 图像处理基座：尺寸、归一化与后端
   --------------------------------------------------------------------------
   覆盖：image_processing_base.py / image_processing_utils.py /
        image_processing_backends.py（3 个文件 / 1874 行，实测口径）
   目标：看完能说出 preprocess 的四步编排、后端的选择顺序，
        并能自己把一张 336x336 的图算到 144 个视觉 token。
   实测环境：仓库自带 venv（transformers 5.18.0.dev0 / torch 2.14.0+cpu /
            torchvision 0.29.0+cpu），脚本 _data/recon/probe_l204.py。
   排版基准：#visual 可视区 = 994 x 662.7 舞台像素（无头浏览器实测）。
   ========================================================================== */
'use strict';

/* 本课实测常量（probe_l204.py 的真实输出，不是估算） */
const M = {
  linesBase: 510, linesUtils: 697, linesBackends: 667,
  imgH: 336, imgW: 336, ch: 3,
  patch: 14, merge: 2, temporal: 2,
  gridH: 24, gridW: 24,          // image_grid_thw = [[1, 24, 24]]
  patchDim: 1176,                // C * temporal * patch * patch = 3*2*14*14
  nPatches: 576,                 // 24 * 24
  nTokens: 144,                  // 24*24/4
  bigPatches: 3600, bigTokens: 900,   // 700x1000 的那张
  backends: 2,                   // 只有 torchvision / pil 两个后端类
};

const SCENES = [

/* ------------------------------------------------- 1 全景 */
{
  kicker: 'L2 · 输入流水线',
  title: '一张图，怎么变成<span class="hl-a">一串向量</span>',
  sub: '文本 token 和视觉 token 最终要排在同一条序列上。这一课管的是前半段：像素 → 归一化张量 → patch → 视觉 token。',
  caption: '回顾 L0-01：forward 的入口是 input_ids 与 inputs_embeds 二选一。这一课产出的 pixel_values，就是合流之前视觉侧交出来的物料。',
  lang: 'python',
  codeStart: 69,
  code: `    def __init__(self, **kwargs):
        """Set elements of \`kwargs\` as attributes."""
        # This key was saved while we still used \`XXXFeatureExtractor\` for image processing. Now we use
        # \`XXXImageProcessor\`, this attribute and its value are misleading.
        kwargs.pop("feature_extractor_type", None)
        # Pop "processor_class", should not be saved with image processing config anymore
        kwargs.pop("processor_class", None)
        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err`,
  codeNote: 'ImageProcessingMixin.__init__ —— 整个基座的根。两个 pop 是历史包袱的化石。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    /* ---- 主链路 ---- */
    const fl = W.flow([
      { t: '像素', s: '(H, W, 3)', cc: 0 },
      { t: '尺寸', s: 'SizeDict', cc: 2 },
      { t: '缩放 + 归一化', s: '×1/255, −μ/σ', cc: 4 },
      { t: 'patchify', s: '14×14 切块', cc: 1 },
      { t: '视觉 token', s: '144 / 张图', cc: 5 },
    ], { style: 'width:100%' });
    wrap.appendChild(U.el('div', { class: 'col gap8', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: '本课覆盖的那条路' }),
      fl,
    ]));

    /* ---- 三个数字 ---- */
    const stats = [
      { k: '引用块', v: 14, cc: 0, s: '逐字来自 3 个源文件' },
      { k: '后端类', v: M.backends, cc: 4, s: 'torchvision / pil' },
      { k: '336×336 →', v: M.nTokens, cc: 5, s: '视觉 token（merge 2×2）' },
      { k: 'pixel_values', v: M.patchDim, cc: 1, s: '每个 patch 展平成 1176 维' },
    ];
    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const statEls = stats.map(st => {
      const c = W.card({ cc: st.cc, tint: st.cc,
        title: `<span class="mono" style="font-size:11px">${st.s}</span>`,
        body: U.el('div', { class: 'n big', text: U.fmt(st.v) }),
        style: 'flex:1' });
      row.appendChild(c); return c;
    });
    wrap.appendChild(row);

    /* ---- 两条契约 ---- */
    const cmp = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cA = W.card({ cc: 0, title: '契约（换不掉）',
      sub: '<code class="inl">preprocess</code> 的四步编排 + 类属性当默认值。所有模型处理器共用同一条。',
      body: U.el('div', { class: 'mono', style: 'font-size:11.5px;color:var(--c0)', text: 'validate → default → size → dispatch' }) });
    const cB = W.card({ cc: 4, title: '实现（可以换）',
      sub: 'resize / pad / rescale / normalize / center_crop 重写在哪一层，就用哪一层的实现。',
      body: U.el('div', { class: 'mono', style: 'font-size:11.5px;color:var(--c4)', text: 'torchvision: torch.Tensor   pil: np.ndarray' }) });
    cA.style.flex = '1'; cB.style.flex = '1';
    cmp.appendChild(cA); cmp.appendChild(cB);
    wrap.appendChild(cmp);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    fl.focus(0);
    msg.innerHTML = '<span class="cm">// 从一张最普通的图片开始</span>';
    tl.at(2000, () => { fl.focus(1); msg.innerHTML = '尺寸先归一：<em>224</em>、<em>(336,336)</em> 都要变成同一个 <em>SizeDict</em>'; });
    tl.at(5200, () => { fl.focus(2); msg.innerHTML = '缩放 <span class="fn">×1/255</span> 再减均值除标准差 —— torchvision 后端会把两步<em>融合成一步</em>'; });
    tl.at(8600, () => { fl.focus(3); msg.innerHTML = '切块：<em>14×14</em> 一块，再按 <em>2×2</em> 合并 —— 合并后才是视觉 token'; });
    tl.at(12000, () => {
      fl.focus(4); statEls.forEach(e => e.classList.add('ac'));
      msg.innerHTML = '<em>24×24/4 = 144</em> 个视觉 token &nbsp;<span class="cm">// 序列长度由文本决定，视觉只填满占位符</span>';
    });
    tl.at(15200, () => {
      statEls.forEach(e => e.classList.remove('ac'));
      cA.classList.add('ac'); cB.classList.remove('ac');
      msg.innerHTML = '★ 记住这个二分：<em>契约在基座，实现交后端</em>';
    });
  },
},

/* ------------------------------------------------- 2 ★ 契约：preprocess 四步 */
{
  kicker: '第一步 · 契约',
  title: '★ <span class="hl-a">preprocess</span>：四步编排，一行像素运算都不写',
  sub: '校验类型 → 补默认值 → 标准化尺寸 → 委派给后端。所有模型的图像处理器共用这一段。',
  caption: '注意第三步在校验之前：size=224 和 {"height":224,"width":224} 必须先归一，后面的校验才有一条唯一判据。',
  lang: 'python',
  codeStart: 393,
  code: `        # Perform type validation on received kwargs
        validate_typed_dict(self.valid_kwargs, kwargs)

        # Set default kwargs from self
        for kwarg_name in self._valid_kwargs_names:
            kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))

        # Update kwargs that need further processing before being validated
        kwargs = self._standardize_kwargs(**kwargs)

        # Validate kwargs
        self._validate_preprocess_kwargs(**kwargs)

        image_like_kwargs = {} if image_like_kwargs is None else image_like_kwargs

        return self._preprocess_image_like_inputs(images, *args, **image_like_kwargs, **kwargs)`,
  codeNote: 'BaseImageProcessor.preprocess 的全部函数体 —— 四步 + 一次委派。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap10 vizgrow hstart' });
    wrap.appendChild(viz);

    /* 四步流水线 */
    viz.appendChild(U.el('div', { class: 'klabel', text: 'preprocess 的四步（每一步都有源码里的注释对应）' }));
    const steps = [
      { n: '①', t: 'validate_typed_dict', s: 'valid_kwargs 是这个类声明的 TypedDict', cc: 0 },
      { n: '②', t: 'kwargs.setdefault', s: '默认值住在类属性上：rescale_factor=1/255', cc: 4 },
      { n: '③', t: '_standardize_kwargs', s: 'size / crop_size / pad_size → SizeDict', cc: 2 },
      { n: '④', t: '_validate_preprocess_kwargs', s: '组合合法性：do_resize 就必须有 size', cc: 1 },
    ];
    const grid = U.el('div', { style: 'display:grid;grid-template-columns:1fr 1fr;gap:9px;width:100%' });
    const cards = steps.map(st => {
      const e = W.card({ cc: st.cc, title: '<span class="mono" style="font-size:11.5px">' + st.n + ' ' + st.t + '</span>',
        sub: st.s, style: 'min-height:76px' });
      grid.appendChild(e); return e;
    });
    viz.appendChild(grid);

    /* 委派 */
    const dv = W.card({ cc: 5, tint: 5, title: '→ _preprocess_image_like_inputs',
      sub: '契约到此结束。后面全是后端的事：基座的 process_image / _preprocess 都只 raise NotImplementedError。',
      body: U.el('div', { class: 'mono', style: 'font-size:11.5px;color:var(--c5)', text: 'prepare structure → process_image(每张) → _preprocess(批量)' }) });
    viz.appendChild(dv);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    cards.forEach(e => { e.style.opacity = '.3'; });
    msg.innerHTML = '<span class="cm">// 四步，逐个点亮</span>';
    const notes = [
      '校验的是<em>这个类声明过的键</em> —— <span class="fn">valid_kwargs</span> 是类属性，模型可以换',
      '取默认值用的是 <span class="fn">getattr(self, key, None)</span> —— 默认值只有一份，在类属性上',
      '<span class="fn">_standardize_kwargs</span> 在校验<em>之前</em>：先归一，才有一条唯一判据',
      '到这里一行像素运算都没做。真东西在<em>后端</em>的 <span class="fn">_preprocess</span> 里',
    ];
    steps.forEach((s, i) => {
      tl.at(1500 + i * 2600, () => {
        cards.forEach((e, k) => { e.style.opacity = (k === i) ? '1' : '.3'; });
        msg.innerHTML = '<em>' + s.n + '</em> ' + notes[i];
      });
    });
    tl.at(12400, () => {
      cards.forEach(e => { e.style.opacity = '1'; });
      dv.classList.add('ac');
      msg.innerHTML = '★ 最后一行是<em>分水岭</em>：契约负责编排，后端负责算';
    });
    tl.at(15400, () => {
      dv.classList.remove('ac');
      msg.innerHTML = '<span class="cm">// 所以 BaseImageProcessor 不能被当成可用的处理器实例化</span>';
    });
  },
},

/* ------------------------------------------------- 3 size 归一 */
{
  kicker: '第二步 · 尺寸',
  title: 'size 有<span class="hl-a">四种写法</span>，必须先归一',
  sub: 'size=224 到底是正方形，还是"短边 224"？歧义必须在校验之前消灭掉。',
  caption: '合法的键集合只有 6 组；不属于任何一组的字典会直接 raise。这项检查在 get_size_dict 里。',
  lang: 'python',
  codeStart: 328,
  code: `        if kwargs is None:
            kwargs = {}
        if size is not None and not isinstance(size, SizeDict):
            size = SizeDict(**get_size_dict(size=size, default_to_square=default_to_square))
        if crop_size is not None and not isinstance(crop_size, SizeDict):
            crop_size = SizeDict(**get_size_dict(crop_size, param_name="crop_size"))
        if pad_size is not None and not isinstance(pad_size, SizeDict):
            pad_size = SizeDict(**get_size_dict(size=pad_size, param_name="pad_size"))
        if isinstance(image_mean, list):
            image_mean = tuple(image_mean)
        if isinstance(image_std, list):
            image_std = tuple(image_std)

        kwargs["size"] = size
        kwargs["crop_size"] = crop_size
        kwargs["pad_size"] = pad_size
        kwargs["image_mean"] = image_mean
        kwargs["image_std"] = image_std`,
  codeNote: '_standardize_kwargs 的尾部：归一 + 无条件写回。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap10 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '实测：四种入参 → 同一个 SizeDict（probe_l204.py）' }));
    const tb = W.table([
      ['<code class="inl">size=224</code>', "<code class='inl'>default_to_square=True</code>", '<span class="hl4">{"height":224,"width":224}</span>', '方图'],
      ['<code class="inl">size=224</code>', "<code class='inl'>default_to_square=False</code>", '<span class="hl4">{"shortest_edge":224}</span>', '短边 224'],
      ['<code class="inl">size=(336,336)</code>', "<code class='inl'>height_width_order=True</code>", '<span class="hl4">{"height":336,"width":336}</span>', '按 (H,W)'],
      ['<code class="inl">size={...}</code>', '已经是字典', '<span class="hl">原样保留</span>', '但要过键集合检查'],
    ], { head: ['入参', '开关', '归一结果', '含义'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 0, style: 'padding:10px 12px' }, [tb]));

    /* 两处容易看漏的细节 */
    const r = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const c1 = W.card({ cc: 2, title: '① 参数名不对称',
      sub: 'size 走 <code class="inl">get_size_dict(size=size, …)</code>，crop_size / pad_size 走第一个位置参数。' +
           'param_name 只影响报错措辞，不影响结果。',
      style: 'flex:1' });
    const c2 = W.card({ cc: 4, title: '② list → tuple',
      sub: 'image_mean / image_std 从 list 转 tuple 是为了<em>可哈希</em>：to_dict 与归一化都要用它当键。',
      style: 'flex:1' });
    r.appendChild(c1); r.appendChild(c2);
    viz.appendChild(r);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    viz.querySelectorAll('.tbl tr').forEach((tr, i) => { if (i > 0) tr.style.opacity = '.32'; });
    msg.innerHTML = '<span class="cm">// 四种写法，四种命运</span>';
    const rows = viz.querySelectorAll('.tbl tr');
    const hi = k => rows.forEach((tr, i) => { tr.style.opacity = (i === k || i === 0) ? '1' : '.32'; });
    tl.at(1800, () => { hi(1); msg.innerHTML = '整数 + 默认开关 → <em>正方形</em>。这是最常见的写法'; });
    tl.at(4800, () => { hi(2); msg.innerHTML = '整数 + <span class="fn">default_to_square=False</span> → <em>shortest_edge</em>：短边缩到 224，长边按比例'; });
    tl.at(8000, () => { hi(3); msg.innerHTML = '元组 → 默认按 <em>(height, width)</em> 解释，可用 <span class="fn">height_width_order</span> 翻转'; });
    tl.at(11200, () => {
      hi(4);
      msg.innerHTML = '已经是字典？<em>原样保留</em> —— 但仍要过 <span class="fn">is_valid_size_dict</span> 的 6 组键检查';
    });
    tl.at(14200, () => {
      rows.forEach(tr => { tr.style.opacity = '1'; });
      c1.classList.add('ac');
      msg.innerHTML = '<span class="cm">// 最后五个 key 是无条件写回的：保证下游一定拿得到值</span>';
    });
  },
},

/* ------------------------------------------------- 4 ★ 后端是类级选择 */
{
  kicker: '第三步 · 后端',
  title: '★ 后端是<span class="hl-a">类级</span>的选择，不是调用级',
  sub: 'resize / normalize 重写在哪一层，就用哪一层的实现。基座故意不调 _set_attributes —— 这是为老代码留的后门。',
  caption: '那行注释写明了原因：backward compatibility with remote code。后果是基座自己不能被实例化来用，而且失败是静默的。',
  lang: 'python',
  codeStart: 189,
  code: `    valid_kwargs = ImagesKwargs

    default_to_square = True
    rescale_factor = 1 / 255
    model_input_names = ["pixel_values"]

    def __init__(self, **kwargs: Unpack[ImagesKwargs]):
        super().__init__(**kwargs)
        # We don't call self._set_attributes in BaseImageProcessor for backward compatibility with remote code
        # We call it instead in the backend subclasses' __init__ methods.`,
  codeNote: 'BaseImageProcessor 的四个类属性 + 那个"什么都不做"的 __init__。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap10 vizgrow hstart' });
    wrap.appendChild(viz);

    /* 继承链 */
    viz.appendChild(U.el('div', { class: 'klabel', text: '两个后端类，两条继承链（源文件里的实测类清单）' }));
    const fl = W.flow([
      { t: 'ImageProcessingMixin', s: '存取 / JSON / Hub', cc: 0 },
      { t: 'BaseImageProcessor', s: 'preprocess 四步编排', cc: 1 },
      { t: 'TorchvisionBackend', s: '@requires(torch, torchvision)', cc: 4 },
      { t: 'Glm5NextImageProcessor', s: '重写 resize/patchify', cc: 5 },
    ], { style: 'width:100%' });
    viz.appendChild(fl);
    const fl2 = W.flow([
      { t: 'ImageProcessingMixin', s: '同一个基座', cc: 0 },
      { t: 'BaseImageProcessor', s: '同一个契约', cc: 1 },
      { t: 'PilBackend', s: '@requires(vision)', cc: 2 },
      { t: 'Glm5NextImageProcessorPil', s: 'np.ndarray 路径', cc: 3 },
    ], { style: 'width:100%' });
    viz.appendChild(fl2);

    /* 类属性 = 默认值 */
    const r = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cA = W.card({ cc: 0, title: '默认值住在类属性上',
      body: U.el('div', { class: 'mono', style: 'font-size:11.5px;line-height:1.75;color:var(--ink-dim)' },
        [U.el('div', { html: 'valid_kwargs &nbsp;<span class="dim">= ImagesKwargs</span>' }),
         U.el('div', { html: 'default_to_square &nbsp;<span class="dim">= True</span>' }),
         U.el('div', { html: 'rescale_factor &nbsp;<span class="dim">= 1/255</span>' }),
         U.el('div', { html: 'model_input_names &nbsp;<span class="dim">= ["pixel_values"]</span>' })]),
      style: 'flex:1' });
    const cB = W.card({ cc: 5, title: '静默失败的组合',
      sub: '继承基座 → 不继承任何后端 → 不自己调 <code class="inl">_set_attributes</code>：' +
           '<code class="inl">self.size</code> 不存在，而 <code class="inl">getattr(self, key, None)</code> 直接给 None，<span class="hlbad">不报错</span>。',
      style: 'flex:1' });
    r.appendChild(cA); r.appendChild(cB);
    viz.appendChild(r);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 一个基座，两个实现</span>';
    tl.at(1800, () => { fl.focus(2); msg.innerHTML = 'torchvision 后端：<em>torch.Tensor</em> 运算，可以 <span class="fn">.to(device)</span>，批量分组'; });
    tl.at(4800, () => { fl2.focus(2); msg.innerHTML = 'pil 后端：<em>np.ndarray</em> 运算，只走 CPU，逐张处理'; });
    tl.at(8000, () => {
      fl.focus(-1); fl2.focus(-1); cA.classList.add('ac');
      msg.innerHTML = '★ 默认值不在签名里 —— 想改行为就传 <span class="fn">do_normalize=False</span> 这类开关';
    });
    tl.at(11400, () => {
      cA.classList.remove('ac'); cB.classList.add('ac');
      msg.innerHTML = '<span class="hlbad">注意</span>：基座的 __init__ 只做 super()，<em>_set_attributes 在每个后端里各调一次</em>';
    });
    tl.at(14400, () => {
      cB.classList.remove('ac');
      msg.innerHTML = '<span class="cm">// 这就是"契约被拆成两半"的代价：忘了补那一句，错得很安静</span>';
    });
  },
},

/* ------------------------------------------------- 5 rescale+normalize 融合 */
{
  kicker: '第四步 · 归一化',
  title: '<span class="hl-a">rescale</span> 与 <span class="hl-b">normalize</span> 会被融合成一步',
  sub: '(x·f − μ)/σ ≡ (x − μ/f)/(σ/f)。把均值方差各除以 f，就能省掉一次全图乘除。',
  caption: '实测：融合与分开做在 atol=1e-5 下一致，但最后一位有效数字不同 —— 不是数学上逐位相同的两条路径。',
  lang: 'python',
  codeStart: 299,
  code: `    @lru_cache(maxsize=10)
    def _fuse_mean_std_and_rescale_factor(
        self,
        do_normalize: bool | None = None,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        do_rescale: bool | None = None,
        rescale_factor: float | None = None,
        device: Optional["torch.device"] = None,
    ) -> tuple:
        if do_rescale and do_normalize:
            # Fused rescale and normalize
            image_mean = torch.tensor(image_mean, device=device) * (1.0 / rescale_factor)
            image_std = torch.tensor(image_std, device=device) * (1.0 / rescale_factor)
            do_rescale = False
        return image_mean, image_std, do_rescale`,
  codeNote: '三个细节都藏在这 5 行里：do_rescale 被改写、mean/std 变张量、lru_cache 挂在 self 上。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap10 vizgrow hstart' });
    wrap.appendChild(viz);

    /* 数值走一遍 */
    viz.appendChild(U.el('div', { class: 'klabel', text: '用一个真实像素走一遍（取值 128/255 ≈ 0.502，OpenAI CLIP 的 μ/σ）' }));
    const numRow = U.el('div', { class: 'row gap10', style: 'width:100%;align-items:stretch' });
    const nA = W.card({ cc: 4, title: '分开做',
      body: U.el('div', { class: 'mono', style: 'font-size:11.5px;line-height:1.75;color:var(--c4)' },
        [U.el('div', { html: 'x = <span class="hl">0.502</span>' }),
         U.el('div', { html: '① × (1/255) → 0.001969' }),
         U.el('div', { html: '② (x−0.4815)/0.2686' }),
         U.el('div', { html: '→ <span class="hl">−1.7855</span>' })]),
      style: 'flex:1' });
    const nB = W.card({ cc: 5, title: '融合后',
      body: U.el('div', { class: 'mono', style: 'font-size:11.5px;line-height:1.75;color:var(--c5)' },
        [U.el('div', { html: 'μ′ = 0.4815 × 255 = 122.79' }),
         U.el('div', { html: 'σ′ = 0.2686 × 255 = 68.50' }),
         U.el('div', { html: '(128 − 122.79)/68.50' }),
         U.el('div', { html: '→ <span class="hl">−1.7855</span>' })]),
      style: 'flex:1' });
    numRow.appendChild(nA); numRow.appendChild(nB);
    viz.appendChild(numRow);

    /* 三个细节 */
    const detail = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const d1 = W.card({ cc: 2, num: '①', title: 'do_rescale 被改成 False',
      sub: '必须告诉调用方"别再单独缩放"，否则会缩放两次。',
      style: 'flex:1' });
    const d2 = W.card({ cc: 1, num: '②', title: 'μ/σ 变成张量并上 device',
      sub: '返回值类型<em>取决于分支</em>：融合时是 Tensor，不融合时还是 list。',
      style: 'flex:1' });
    const d3 = W.card({ cc: 3, num: '③', title: '@lru_cache(maxsize=10)',
      sub: 'self 也是缓存键 → 缓存按实例；改类属性不会让它失效。',
      style: 'flex:1' });
    detail.appendChild(d1); detail.appendChild(d2); detail.appendChild(d3);
    viz.appendChild(detail);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    [d1, d2, d3].forEach(e => { e.style.opacity = '.3'; });
    nB.style.opacity = '.3';
    msg.innerHTML = '<span class="cm">// 同一张图，两条路径</span>';
    tl.at(1800, () => { msg.innerHTML = '先缩放再归一化：<em>(x·f − μ)/σ</em>，两次全图运算'; });
    tl.at(4600, () => {
      nB.style.opacity = '1';
      msg.innerHTML = '融合：把 <em>μ、σ 各除以 f</em>，让归一化在原图上跑一次 → 结果相同，少一趟';
    });
    tl.at(8000, () => {
      d1.style.opacity = '1';
      msg.innerHTML = '<em>①</em> 函数<em>返回</em>改写后的 <span class="fn">do_rescale</span>，不靠调用方记着';
    });
    tl.at(11000, () => {
      d2.style.opacity = '1';
      msg.innerHTML = '<em>②</em> <span class="fn">device=</span> 参数说明归一化可以在 <span class="hl2">GPU</span> 上做';
    });
    tl.at(14000, () => {
      d3.style.opacity = '1';
      msg.innerHTML = '<em>③</em> 缓存按实例 —— 稳态推理省下的是<em>张量构造</em>的开销';
    });
    tl.at(16200, () => {
      msg.innerHTML = '<span class="cm">// 实测：allclose(atol=1e-5) = True，但末位有效数字不同</span>';
    });
  },
},

/* ------------------------------------------------- 6 ★ 形状演算 */
{
  kicker: '第二步 · 形状演算',
  title: '★ 336×336 → <span class="hl-a">144 个视觉 token</span>',
  sub: 'patchify 的输出是 (576, 1176)：576 个 patch，每个展平成 1176 维。2×2 合并之后才是 144 个 token。',
  caption: '序列长度永远由文本决定；视觉只负责填满被 <|image|> 预定好的那 144 个位置。',
  lang: 'python',
  codeStart: 390,
  code: `        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(image=stacked_images, size=size, resample=resample)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)`,
  codeNote: 'torchvision 后端的批处理：按形状分组 → 批量 resize → 还原顺序。',
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap10 vizgrow hstart' });
    wrap.appendChild(viz);

    /* 形状链条 */
    viz.appendChild(U.el('div', { class: 'klabel', text: '实测形状链（Glm5NextImageProcessor，patch=14 / merge=2 / temporal=2）' }));
    const chain = [
      { t: '(336, 336, 3)', s: '输入 HWC', cc: 0 },
      { t: '(3, 336, 336)', s: '通道在前', cc: 4 },
      { t: '(576, 1176)', s: 'patchify 输出', cc: 1 },
      { t: '(1, 24, 24)', s: 'grid_thw', cc: 2 },
      { t: '144', s: 'token', cc: 5 },
    ];
    const fl = W.flow(chain, { style: 'width:100%' });
    viz.appendChild(fl);

    /* 分解 */
    const tb = W.table([
      ['patch 数', '<span class="hl">24 × 24 = 576</span>', 'grid_h × grid_w，就是 pixel_values 的第 0 维'],
      ['grid', '<span class="hl">(1, 24, 24)</span>', 't × h × w，图片只有 1 帧'],
      ['展平维', '<span class="hl">3 × 2 × 14 × 14 = 1176</span>', 'C × temporal × patch × patch'],
      ['视觉 token', '<span class="hl4">1×24×24 / 2² = 144</span>', 't×h×w / merge_size²'],
      ['另一张 700×1000', '<span class="hl">(1, 50, 72)</span>', '3600 个 patch → 900 个 token'],
    ], { head: ['量', '实测值', '怎么来的'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 1, style: 'padding:10px 12px' }, [tb]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 从一张 336×336 的 RGB 图开始</span>';
    tl.at(1800, () => { fl.focus(1); msg.innerHTML = 'process_image 只做两件事：<em>通道转到最前</em>、<em>变成 torch.Tensor</em>'; });
    tl.at(5000, () => {
      fl.focus(2);
      msg.innerHTML = '切成 <em>14×14</em> 的块 → <span class="fn">24×24 = 576</span> 块；每块展平成 <span class="fn">1176</span> 维';
    });
    tl.at(8600, () => {
      fl.focus(3);
      msg.innerHTML = '<span class="fn">image_grid_thw</span> 把几何信息单独带出来 —— <em>(1, 24, 24)</em>，不是藏在形状里';
    });
    tl.at(12000, () => {
      fl.focus(4);
      msg.innerHTML = '★ 每 <em>2×2</em> 个 patch 合并成 1 个视觉 token：<span class="hl4">576 / 4 = 144</span>';
    });
    tl.at(15200, () => {
      fl.focus(-1);
      msg.innerHTML = '<span class="cm">// 文本侧：input_ids (batch, seq)  视觉侧：这 144 个位置被预定了</span>';
    });
    tl.at(17400, () => {
      msg.innerHTML = '<em>序列长度是文本说了算</em>，视觉只负责填满占位符 —— 这就是两种 token 的对齐方式';
    });
  },
},

/* ------------------------------------------------- 7 from_pretrained */
{
  kicker: '第三步 · 存取契约',
  title: '从一个 JSON 到一个处理器对象',
  sub: 'from_pretrained 拆成"找文件"和"造对象"两半；resolve_revision 只调一次，保证所有文件来自同一个仓库快照。',
  caption: '存盘的是 image_processor_type（类名）而不是后端名 —— 后端留给加载时的环境去决定。这条线索在最后一幕闭环。',
  lang: 'python',
  codeStart: 174,
  code: `        kwargs["cache_dir"] = cache_dir
        kwargs["force_download"] = force_download
        kwargs["local_files_only"] = local_files_only
        # Resolve the revision once, so that all the files of this load come from the same repository state.
        kwargs["revision"] = resolve_revision(
            pretrained_model_name_or_path,
            revision,
            token=token,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )

        if token is not None:
            kwargs["token"] = token

        image_processor_dict, kwargs = cls.get_image_processor_dict(pretrained_model_name_or_path, **kwargs)

        return cls.from_dict(image_processor_dict, **kwargs)`,
  codeNote: 'ImageProcessingMixin.from_pretrained 的全部真代码：15 行，两半。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap10 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '加载与保存是一条闭环' }));
    const fl = W.flow([
      { t: 'preprocessor_config.json', s: '磁盘上的一个 JSON', cc: 0 },
      { t: 'get_image_processor_dict', s: '找文件 + 读 JSON', cc: 1 },
      { t: 'from_dict', s: '造对象', cc: 4 },
      { t: '处理器实例', s: '可以跑了', cc: 5 },
    ], { style: 'width:100%' });
    viz.appendChild(fl);

    const r = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cA = W.card({ cc: 2, title: '为什么 revision 只解析一次',
      sub: '注释写明了：保证这次加载涉及的<em>所有文件来自同一个仓库快照</em>。分开解析会读到不同 commit 的文件。',
      style: 'flex:1' });
    const cB = W.card({ cc: 4, title: 'kwargs 进得去出得来',
      sub: '<code class="inl">get_image_processor_dict</code> 返回 <code class="inl">(字典, 剩下的 kwargs)</code>；' +
           '用户显式传的覆盖值一路活到 <code class="inl">from_dict</code>。',
      style: 'flex:1' });
    r.appendChild(cA); r.appendChild(cB);
    viz.appendChild(r);

    viz.appendChild(W.exercise(
      '谁决定这份 <code class="inl">preprocessor_config.json</code> 会被哪个后端类加载？',
      '不是文件里的内容决定的 —— 存盘的字段是 <code class="inl">image_processor_type</code>（类名，' +
      'PIL 变体存盘时还会被砍掉结尾的 <code class="inl">Pil</code>）。' +
      '<b>后端由加载时的环境决定</b>：先看有没有显式 <code class="inl">backend=</code>，' +
      '再看 <code class="inl">use_fast</code>，再看强制 PIL 的名单，最后才是' +
      '<code class="inl">torchvision</code> 可用与否。同一个文件在两台机器上可以是两个类。'));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 一次加载，两个阶段</span>';
    tl.at(2000, () => { fl.focus(1); msg.innerHTML = '阶段一：<em>找</em> —— 本地目录？单文件？还是去 Hub 取'; });
    tl.at(5200, () => { fl.focus(2); msg.innerHTML = '阶段二：<em>造</em> —— 按 <span class="fn">valid_kwargs</span> 声明更新，再补兼容键'; });
    tl.at(8600, () => { cA.classList.add('ac'); msg.innerHTML = '<em>resolve_revision</em> 只调一次：所有文件必须来自同一个快照'; });
    tl.at(11800, () => {
      cA.classList.remove('ac'); cB.classList.add('ac'); fl.focus(3);
      msg.innerHTML = '用户传的覆盖值不会被吞掉 —— 这是"<em>配置即代码</em>"能成立的前提';
    });
    tl.at(14800, () => {
      cB.classList.remove('ac'); fl.focus(-1);
      msg.innerHTML = '<span class="cm">// 下一幕：这个对象拿到图之后，两个后端各走各的路</span>';
    });
  },
},

/* ------------------------------------------------- 8 两个后端的 _preprocess */
{
  kicker: '第四步 · 后端分叉',
  title: '同一个契约，<span class="hl-a">两种写法</span>',
  sub: 'pil 后端是最朴素的逐张循环；torchvision 后端按形状分组做批处理。顺序都一样，代价不一样。',
  caption: '顺序不能换：crop 在 resize 之后，rescale/normalize 在几何变换之后，pad 放最后（0 是"归一化后的 0"）。',
  lang: 'python',
  codeStart: 641,
  code: `        processed_images = []
        for image in images:
            if do_resize:
                image = self.resize(image=image, size=size, resample=resample)
            if do_center_crop:
                image = self.center_crop(image, crop_size)
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)
            processed_images.append(image)

        if do_pad:
            processed_images = self.pad(processed_images, pad_size=pad_size)

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)`,
  codeNote: 'PilBackend._preprocess —— 五步顺序一目了然，每步都调 self 的方法（可被子类重写）。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap10 vizgrow hstart' });
    wrap.appendChild(viz);

    /* 顺序链 */
    viz.appendChild(U.el('div', { class: 'klabel', text: '两个后端共用的操作顺序（PilBackend 源码里的 if 顺序）' }));
    const fl = W.flow([
      { t: 'resize', s: 'do_resize', cc: 0 },
      { t: 'center_crop', s: 'do_center_crop', cc: 1 },
      { t: 'rescale', s: 'do_rescale', cc: 4 },
      { t: 'normalize', s: 'do_normalize', cc: 5 },
      { t: 'pad', s: 'do_pad', cc: 2 },
    ], { style: 'width:100%' });
    viz.appendChild(fl);

    /* 两个后端对照 */
    viz.appendChild(U.el('div', { class: 'klabel', text: '同一件事的两种实现' }));
    const tb = W.table([
      ['循环粒度', '<span class="hl2">for image in images（逐张）</span>', '<span class="hl4">group_images_by_shape（按形状分组批量）</span>'],
      ['rescale+normalize', '两步分开调 <code class="inl">self.rescale</code> / <code class="inl">self.normalize</code>', '融合成 <code class="inl">rescale_and_normalize</code> 一次做完'],
      ['数据类型', 'np.ndarray', 'torch.Tensor'],
      ['设备', '只走 CPU', '可 <code class="inl">.to(device)</code> 上 GPU'],
    ], { head: ['维度', 'PilBackend', 'TorchvisionBackend'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 4, style: 'padding:10px 12px' }, [tb]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 为什么是这个顺序</span>';
    tl.at(1800, () => { fl.focus(0); msg.innerHTML = '<em>resize 在最前</em> —— 先把图变成目标尺寸，后面所有运算都在小图上做'; });
    tl.at(4800, () => { fl.focus(1); msg.innerHTML = '<em>crop 紧跟 resize</em>：反过来裁的位置就不对了'; });
    tl.at(7800, () => { fl.focus(2); msg.innerHTML = '<em>rescale</em> 把 0-255 压到 0-1 —— 必须在几何变换<em>之后</em>'; });
    tl.at(10800, () => { fl.focus(3); msg.innerHTML = '<em>normalize</em> 用 μ/σ 把数值拉到 0 附近。torchvision 后端把 ③④ 合成一步'; });
    tl.at(13800, () => {
      fl.focus(4);
      msg.innerHTML = '<em>pad 放最后</em>：填进去的 <span class="fn">0</span> 是"归一化之后的 0"，含义和黑边一致';
    });
    tl.at(15800, () => {
      fl.focus(-1);
      msg.innerHTML = '<span class="cm">// 出口永远一样：BatchFeature({"pixel_values": [...]})</span>';
    });
  },
},

/* ------------------------------------------------- 9 收束 */
{
  kicker: '收束',
  title: '默认后端的选择顺序 —— <span class="hl-a">四步</span>',
  sub: '本课的验收点：不看笔记说出 backend=None 时到底会选谁。顺序是显式 backend → use_fast → 强制 PIL 名单 → torchvision/pil 兜底。',
  caption: '下一课 L2-05：resize / center_crop / normalize 的 numpy 实现细节，以及 SizeDict、ImageInput 这些类型的定义。',
  lang: 'python',
  codeStart: 85,
  code: `        kwargs["cache_dir"] = cache_dir
        kwargs["force_download"] = force_download
        kwargs["local_files_only"] = local_files_only
        # Resolve the revision once, so that all the files of this load come from the same repository state.
        kwargs["revision"] = resolve_revision(
            pretrained_model_name_or_path,
            revision,
            token=token,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )

        if token is not None:
            kwargs["token"] = token

        image_processor_dict, kwargs = cls.get_image_processor_dict(pretrained_model_name_or_path, **kwargs)

        return cls.from_dict(image_processor_dict, **kwargs)`,
  codeNote: '回到起点：现在这 15 行里的每一个名字，你都能说出它落在流水线的哪一步。',
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    /* 选择顺序 */
    const steps = [
      { n: '①', t: 'backend=', c: 0, d: '显式指定 → 原样返回（最高优先级）' },
      { n: '②', t: 'use_fast', c: 2, d: '已弃用：True→torchvision / False→pil' },
      { n: '③', t: '强制 PIL 名单', c: 1, d: 'Lanczos 系处理器（torchvision<0.27 时才非空）' },
      { n: '④', t: '兜底', c: 4, d: 'torchvision 可用就它，否则 pil' },
    ];
    const row = U.el('div', { class: 'row gap8', style: 'width:100%;align-items:stretch' });
    const els = steps.map(s => {
      const e = W.card({ cc: s.c, tint: s.c,
        title: '<span class="mono" style="font-size:11px">' + s.n + ' ' + s.t + '</span>',
        sub: s.d, style: 'flex:1;text-align:center' });
      row.appendChild(e); return e;
    });
    wrap.appendChild(row);

    /* 实测数字表 */
    const tb = W.table([
      ['源文件行数', '<span class="hl">510 / 697 / 667</span>', '作业书写 511/698/668，实测少 1（计数口径）'],
      ['后端类', '<span class="hl">2</span>', 'TorchvisionBackend / PilBackend，没有 NumpyBackend'],
      ['默认后端', '<span class="hl4">torchvision</span>', 'is_torchvision_available() 为真时'],
      ['336×336 →', '<span class="hl">(576, 1176)</span>', 'grid (1,24,24) → 144 个视觉 token'],
      ['融合等价', '<span class="hl">allclose 1e-5</span>', 'rescale+normalize 与分开做一致，末位不同'],
    ], { head: ['量', '实测值', '说明'] });
    wrap.appendChild(U.el('div', { class: 'card', cc: 0, style: 'padding:10px 12px' }, [
      U.el('div', { class: 'klabel', text: '本课用到的实测数字（probe_l204.py）' }), tb,
    ]));

    wrap.appendChild(W.exercise(
      '一台机器上装了 torchvision 0.29；另一台没装。同一份 <code class="inl">preprocessor_config.json</code>'
      + '（<code class="inl">image_processor_type = "Glm5NextImageProcessor"</code>）分别会加载成什么类？',
      '装了 torchvision 的那台：<code class="inl">_resolve_backend(None, None, …)</code> 走到第 ④ 步，'
      + '<code class="inl">is_torchvision_available() = True</code> → <b>torchvision</b> 后端 → '
      + '<code class="inl">Glm5NextImageProcessor</code>。'
      + '<br>没装的那台：第 ④ 步回落到 <b>pil</b> → <code class="inl">Glm5NextImageProcessorPil</code>。'
      + '<br><b>关键</b>：配置文件一个字都没变 —— 后端是<em>加载时</em>由环境决定的。'
      + '这也是 <code class="inl">PilBackend.to_dict()</code> 要砍掉结尾 <code class="inl">Pil</code> 的原因：'
      + '存盘不固化后端，才能在两种机器上都能加载。'));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    els.forEach(e => e.style.opacity = '.28');
    msg.innerHTML = '<span class="cm">// backend=None 时，顺序从高到低</span>';
    steps.forEach((s, i) => {
      tl.at(1600 + i * 2200, () => {
        els.forEach((e, k) => { e.style.opacity = (k === i) ? '1' : '.28'; });
        msg.innerHTML = '<em>' + s.n + '</em> ' + s.t + ' &nbsp;<span class="cm">// ' + s.d + '</span>';
      });
    });
    tl.at(11000, () => {
      els.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = '★ 顺序记法：<em>显式 → 旧参数 → 强制名单 → 可用性兜底</em>';
    });
    tl.at(14200, () => {
      msg.innerHTML = '<span class="cm">// 选完之后还有一层兜底：目标后端不可用时，换另一个后端再试一次</span>';
    });
    tl.at(17200, () => {
      msg.innerHTML = '<span class="cm">// 下一课 L2-05：numpy 实现与 SizeDict / ImageInput 的类型定义</span>';
    });
  },
},

];
