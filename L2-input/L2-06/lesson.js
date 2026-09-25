/* ==========================================================================
   L2-06 · 多模态处理器装配
   --------------------------------------------------------------------------
   覆盖：processing_utils.py(2430) / models/glm5_next/processing_glm5_next.py(193)
         models/glm5_next/image_processing_glm5_next.py(322)
         models/glm5_next/image_processing_pil_glm5_next.py(322)
         models/glm5_next/video_processing_glm5_next.py(429)
         feature_extraction_utils.py(688)   —— 6 个文件 / 4384 行
   目标：看完能默写一次图像对话的 inputs 有哪五个 key，并说出每个 key 由谁产生；
         能算出 336x336 的图会占几个位置，以及这 144 个位置是怎么被填进同一条序列的。
   排版基准：#visual 可视区 = 994 x 662.7 舞台像素（无头浏览器实测）。
   ========================================================================== */
'use strict';

/* 实测常量（全部由 _data/recon/probe_l206.py 真跑出来，不是估算） */
const M = {
  keys: ['input_ids', 'attention_mask', 'mm_token_type_ids', 'pixel_values', 'image_grid_thw'],
  seq: 146, tokens: 144, patches: 576, patchDim: 1176,
  grid: 24, merge: 2, patch: 14, temporal: 2, channel: 3,
  imgTokens: 16, canvas: 112,
  sizes: [['336x336', 144], ['512x512', 361], ['1024x768', 1036], ['2048x1536', 4070]],
  twoImgs: 2020,
  frames: 16, gridT: 8, videoPatches: 4608, videoTokens: 1152, perFrame: 72,
  mmTypes: [0, 2, 2, 0, 1, 0],
  attrs: ['image_processor', 'tokenizer', 'video_processor'],
};

const SCENES = [

/* ================================================================ 1 全景 */
{
  kicker: 'L2 · 输入流水线 · 全景',
  title: '一句「描述这张图」是怎么变成 <span class="hl-a">inputs</span> 的',
  sub: '本课只做一件事：把 processor(...) 内部的装配顺序摊开。图片变成几个位置、坐在哪一行代码上，全部有实测答案。',
  caption: '验收点：看完要能默写一次图像对话的 inputs 字典有哪五个 key。',
  lang: 'python',
  codeStart: 636,
  code: `        # Sanitize args and kwargs
        for key in kwargs:
            if key not in self.get_attributes():
                raise TypeError(f"Unexpected keyword argument {key}.")
        for arg, attribute_name in zip(args, self.get_attributes()):
            if attribute_name in kwargs:
                raise TypeError(f"Got multiple values for argument {attribute_name}.")
            else:
                kwargs[attribute_name] = arg

        if len(kwargs) != len(self.get_attributes()):
            raise ValueError(
                f"This processor requires {len(self.get_attributes())} arguments: {', '.join(self.get_attributes())}. Got "
                f"{len(args)} arguments instead."
            )

        # Check each arg is of the proper class (this will also catch a user initializing in the wrong order)
        for attribute_name, arg in kwargs.items():
            self.check_argument_for_proper_class(attribute_name, arg)
            setattr(self, attribute_name, arg)`,
  codeNote: 'ProcessorMixin.__init__ 的核心 20 行：折参数、数个数、验类型、挂属性。',
  duration: 21000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const g = U.el('div', { class: 'vizgrow' });
    wrap.appendChild(g);
    const inner = U.el('div', { class: 'col gap12', style: 'width:100%' });
    g.appendChild(inner);

    const fl = W.flow([
      { t: '对话', s: 'text + image', cc: 0 },
      { t: 'apply_chat_template', s: '渲染 1 个占位符', cc: 1 },
      { t: 'processor(...)', s: '__call__ 分发', cc: 2 },
      { t: '三个子处理器', s: 'img / tok / video', cc: 4 },
      { t: 'BatchFeature', s: 'inputs 字典', cc: 3 },
    ], { style: 'width:100%' });
    inner.appendChild(U.el('div', { class: 'col gap8', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: '一次图像对话的装配链 —— 五步全部在下面九幕里拆开' }), fl,
    ]));

    inner.appendChild(U.el('div', { class: 'klabel', text: '返回值就是这五个 key（实测：一次 336x336 的图像调用）' }));
    const keyRow = W.cardRow(M.keys.map((k, i) => ({
      cc: i, tint: i, num: i + 1,
      title: '<span class="mono" style="font-size:11px">' + k + '</span>',
      sub: ['(1, 146)', '(1, 146)', '(1, 146)', '(576, 1176)', '(1, 3)'][i],
    })), { flex: true });
    inner.appendChild(keyRow);

    const statRow = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const stats = [
      { n: M.tokens, t: '占位符个数', s: '24x24 / 2^2', c: 4 },
      { n: M.patches, t: 'patch 个数', s: 'grid_h x grid_w', c: 0 },
      { n: M.patchDim, t: '每个 patch 的像素', s: '3 x 2 x 14 x 14', c: 2 },
      { n: 5, t: 'inputs 的 key', s: '验收点', c: 3 },
    ];
    const statEls = stats.map(st => {
      const e = W.card({ cc: st.c, tint: st.c, title: st.t,
        sub: '<span class="mono" style="font-size:10.5px">' + st.s + '</span>',
        body: U.el('div', { class: 'n big', text: U.fmt(st.n) }), style: 'flex:1;text-align:center' });
      statRow.appendChild(e); return e;
    });
    inner.appendChild(statRow);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    fl.focus(0);
    statEls.forEach(e => { e.style.opacity = '.25'; });
    let k = 0;
    while (k < 5) { keyRow.children[k].style.opacity = k === 0 ? '1' : '.25'; k++; }
    msg.innerHTML = '<span class="cm">// 从一句对话开始</span>';
    tl.at(1800, () => { fl.focus(1); msg.innerHTML = '模板只写 <em>1 个</em> &lt;|image|&gt; <span class="cm">// 谁把它变成 144 个？</span>'; });
    tl.at(5200, () => { fl.focus(2); msg.innerHTML = '<em>processor.__call__</em> 按模态分发：图像走 image_processor、文本走 tokenizer'; });
    tl.at(8600, () => {
      fl.focus(3);
      msg.innerHTML = '三个子处理器各自产出自己的 key <span class="cm">// 合并规则见第三幕</span>';
      let i = 0;
      while (i < 5) { keyRow.children[i].style.opacity = '1'; i++; }
    });
    tl.at(12200, () => {
      fl.focus(4);
      msg.innerHTML = '全部塞进 <em>BatchFeature</em>：既能 <span class="mono">inputs["input_ids"]</span>，也能 <span class="mono">inputs.input_ids</span>';
    });
    tl.at(15800, () => {
      fl.focus(-1);
      statEls.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = '<span class="cm">// 记住四个数：</span><em>144</em> 个位置 / <em>576</em> 个 patch / <em>1176</em> 维 / <em>5</em> 个 key';
    });
    tl.at(18500, () => {
      msg.innerHTML = '本课的问题：这 <em>144</em> 个位置是在<em>哪一行</em>被写进序列的？';
    });
  },
},

/* ================================================================ 2 三件套 */
{
  kicker: '第一幕 · 装配',
  title: '处理器不是"一个类"，是<span class="hl-a">三件套 + 一次挂载</span>',
  sub: 'Glm5NextProcessor 的签名就是装配清单。它继承 ProcessorMixin，于是"按名字查类型再 setattr"这套机制直接生效。',
  caption: '河道已经铺好：谁被拼进来由签名决定，拼接方式由 ProcessorMixin 决定。',
  lang: 'python',
  codeStart: 46,
  code: `    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):
        self.image_token = "<|image|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.video_token = "<|video|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )
        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)
        self.video_start_id = tokenizer.convert_tokens_to_ids("<|begin_of_video|>")
        self.video_end_id = tokenizer.convert_tokens_to_ids("<|end_of_video|>")`,
  codeNote: '三个子处理器并列；两个 token 字符串"先问 tokenizer，没有就用字面量兜底"。',
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const g = U.el('div', { class: 'vizgrow' });
    wrap.appendChild(g);
    const inner = U.el('div', { class: 'col gap12', style: 'width:100%' });
    g.appendChild(inner);

    inner.appendChild(U.el('div', { class: 'klabel', text: '实测 get_attributes() —— 由 __init__ 签名推出来，不是写死的' }));
    const trio = [
      { t: 'image_processor', s: 'Glm5NextImageProcessor / …Pil', d: 'model_input_names = [pixel_values, image_grid_thw]', c: 0 },
      { t: 'tokenizer', s: 'PreTrainedTokenizerBase', d: 'model_input_names = [input_ids, attention_mask]', c: 4 },
      { t: 'video_processor', s: 'Glm5NextVideoProcessor', d: 'model_input_names = [pixel_values_videos, video_grid_thw]', c: 2 },
    ];
    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cards = trio.map(x => {
      const card = W.card({ cc: x.c, tint: x.c,
        title: '<span class="mono" style="font-size:11.5px">' + x.t + '</span>',
        sub: x.s, style: 'flex:1',
        body: U.el('div', { class: 'mono', style: 'font-size:10.5px;color:var(--ink-faint);margin-top:6px;line-height:1.5', text: x.d }) });
      row.appendChild(card); return card;
    });
    inner.appendChild(row);

    inner.appendChild(U.el('div', { class: 'klabel', text: '两个 token 是"猜"出来的 —— 词表里必须有同名特殊 token' }));
    const tb = W.table([
      ['self.image_token', '<span class="mono">&lt;|image|&gt;</span>', 'tokenizer 有 image_token 就用它的，否则用字面量'],
      ['self.video_token', '<span class="mono">&lt;|video|&gt;</span>', '同上；但 GLM-5 实际用 &lt;|image|&gt; 占位，见第八幕'],
      ['self.video_start_id', 'convert_tokens_to_ids("&lt;|begin_of_video|&gt;")', '视频段的分界，在 super().__init__ <b>之后</b>取'],
      ['self.video_end_id', 'convert_tokens_to_ids("&lt;|end_of_video|&gt;")', '同上'],
    ], { head: ['属性', '实测值', '为什么'] });
    inner.appendChild(U.el('div', { class: 'card', cc: 0, style: 'padding:10px 12px' }, [tb]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    cards.forEach(c => { c.style.opacity = '.25'; });
    msg.innerHTML = '<span class="cm">// __init__ 只做验收与挂载，不负责造子处理器</span>';
    tl.at(1800, () => { cards[0].style.opacity = '1'; msg.innerHTML = '① 图像处理器：<em>patchify</em> 与 <em>smart_resize</em> 都在它肚子里（第七幕）'; });
    tl.at(5200, () => { cards[1].style.opacity = '1'; msg.innerHTML = '② tokenizer：负责把<em>已经展开好</em>的文本变成 id'; });
    tl.at(8400, () => { cards[2].style.opacity = '1'; msg.innerHTML = '③ 视频处理器：多一个时间维，每帧的时间戳也要写进文本'; });
    tl.at(11800, () => {
      cards.forEach(c => c.classList.add('ac'));
      msg.innerHTML = '三个都按名字查过类型才 <span class="mono">setattr</span> —— 传错位置会当场报错，不会错位';
    });
    tl.at(15200, () => {
      cards.forEach(c => c.classList.remove('ac'));
      msg.innerHTML = '<span class="cm">// 注意：chat_template 不是属性，它走 kwargs.pop</span> 所以 get_attributes() 里没有它';
    });
    tl.at(18000, () => { msg.innerHTML = '下一个问题：这三路结果，最后是怎么拼成<em>同一个</em>字典的？'; });
  },
},

/* ================================================================ 3 五个 key */
{
  kicker: '第二幕 · 验收点',
  title: '五个 key 分别在<span class="hl-a">哪一行</span>产生',
  sub: '__call__ 的后半段只有三件事：展开文本、tokenize、拼字典。inputs 的 key 集合完全可以从子处理器的 model_input_names 推出来。',
  caption: '这一幕就是验收点本身：能默写这五个 key，并说出每个由谁产生。',
  lang: 'python',
  codeStart: 694,
  code: `            text, text_replacement_offsets = self.get_text_with_replacements(
                text,
                images_replacements,
                videos_replacements,
                audio_replacements,
            )
            text_inputs = self.tokenizer(text, **merged_kwargs["text_kwargs"])
            self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video", "audio"])

            if return_text_replacement_offsets:
                text_inputs["text_replacement_offsets"] = text_replacement_offsets

            if return_mm_token_type_ids:
                text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(text_inputs["input_ids"])

        # Pop unused keys from the inputs, e.g. inputs used only to compute number of image tokens
        data = {**text_inputs, **processed_images, **processed_videos, **processed_audio}
        data = {k: v for k, v in data.items() if k not in self.unused_input_names}

        if not kwargs.get("return_metadata"):
            data.pop("video_metadata", None)

        return BatchFeature(data, tensor_type=return_tensors, skip_tensor_conversion=self.skip_tensor_conversion)`,
  codeNote: '第 694 行展开占位符 → 第 700 行 tokenize → 第 707 行加 mm_token_type_ids → 第 710 行拼字典。',
  duration: 22000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const g = U.el('div', { class: 'vizgrow' });
    wrap.appendChild(g);
    const inner = U.el('div', { class: 'col gap12', style: 'width:100%' });
    g.appendChild(inner);

    const srcRow = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const srcs = [
      { t: 'tokenizer', n: 2, k: 'input_ids · attention_mask', c: 4 },
      { t: 'image_processor', n: 2, k: 'pixel_values · image_grid_thw', c: 0 },
      { t: 'processor 自己', n: 1, k: 'mm_token_type_ids', c: 2 },
    ];
    const srcCards = srcs.map(x => {
      const e = W.card({ cc: x.c, tint: x.c, num: x.n, title: x.t,
        sub: '<span class="mono" style="font-size:10.5px">' + x.k + '</span>', style: 'flex:1' });
      srcRow.appendChild(e); return e;
    });
    inner.appendChild(U.el('div', { class: 'klabel', text: '三个来源 —— 各出各的 key，最后由 processor 合并' }));
    inner.appendChild(srcRow);

    const tb = W.table([
      ['input_ids', '(1, 146)', 'tokenizer', '<span class="hl4">144</span> 个占位符 + 2 个文本 token'],
      ['attention_mask', '(1, 146)', 'tokenizer', '默认 padding=False，所以是 ragged 的 list'],
      ['mm_token_type_ids', '(1, 146)', 'processor', '默认 <span class="mono">return_mm_token_type_ids=True</span>'],
      ['pixel_values', '(576, 1176)', 'image_processor', '每行一个 patch：<span class="mono">3x2x14x14 = 1176</span>'],
      ['image_grid_thw', '(1, 3)', 'image_processor', '<span class="mono">[[1, 24, 24]]</span> —— 占位符个数的唯一来源'],
    ], { head: ['key', '形状（实测）', '谁给的', '说明'] });
    inner.appendChild(U.el('div', { class: 'card', cc: 3, style: 'padding:10px 12px' }, [tb]));
    const trs = Array.from(tb.querySelectorAll('tbody tr'));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    trs.forEach(tr => { tr.style.opacity = '.25'; });
    srcCards.forEach(c => { c.style.opacity = '.3'; });
    msg.innerHTML = '<span class="cm">// 三路并行的产出，最终在一个字典字面量里相遇</span>';
    tl.at(1800, () => { srcCards[0].style.opacity = '1'; msg.innerHTML = '文本侧：<em>input_ids</em> + <em>attention_mask</em>（tokenizer 的默认产物）'; });
    tl.at(5400, () => { srcCards[1].style.opacity = '1'; msg.innerHTML = '视觉侧：<em>pixel_values</em>（像素）+ <em>image_grid_thw</em>（形状）—— 两者缺一不可'; });
    tl.at(9000, () => { srcCards[2].style.opacity = '1'; msg.innerHTML = '处理器自己加的第五个：<em>mm_token_type_ids</em>，标记每个位置属于哪个模态'; });
    tl.at(12600, () => {
      srcCards.forEach(c => c.classList.add('ac'));
      msg.innerHTML = '<span class="mono">data = {**text_inputs, **processed_images, ...}</span> &nbsp;<span class="cm">// 第 710 行</span>';
    });
    let i = 0;
    tl.at(15600, () => { trs.forEach(tr => { tr.style.opacity = '1'; }); trs[0].classList.add('ac'); msg.innerHTML = '五个 key：<em>input_ids / attention_mask / mm_token_type_ids / pixel_values / image_grid_thw</em>'; });
    tl.at(19000, () => {
      srcCards.forEach(c => c.classList.remove('ac'));
      msg.innerHTML = '想拦住多余的 key，得覆写 <span class="mono">unused_input_names</span>；默认是空的';
    });
  },
},

/* ================================================================ 4 ★ 1->N */
{
  kicker: '第三幕 · ★ 洞察',
  title: '模板写 <span class="hl-a">1 个</span>占位符，序列里有 <span class="hl-b">144 个</span>',
  sub: '把 1 变成 N 的是处理器，不是模板。N = image_grid_thw.prod() / merge_size^2，而 grid 来自真实像素尺寸。',
  caption: '回顾 L0-01：多模态前向走 inputs_embeds，正是因为这里的 N 个位置在外面就被换成了向量。',
  lang: 'python',
  codeStart: 63,
  code: `    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        merge_length = self.image_processor.merge_size**2
        num_image_tokens = image_inputs["image_grid_thw"][image_idx].prod() // merge_length
        return self.image_token * num_image_tokens

    def replace_video_token(self, video_inputs: dict, video_idx: int, **kwargs) -> str:
        merge_length = self.video_processor.merge_size**2
        num_frames = video_inputs["video_grid_thw"][video_idx][0]
        num_image_tokens = video_inputs["video_grid_thw"][video_idx].prod() // merge_length // num_frames`,
  codeNote: '三行算出 N：prod() 把 (t, h, w) 乘起来，再除以 merge_size 的平方。',
  duration: 21000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const g = U.el('div', { class: 'vizgrow' });
    wrap.appendChild(g);
    const inner = U.el('div', { class: 'col gap12', style: 'width:100%' });
    g.appendChild(inner);

    inner.appendChild(U.el('div', { class: 'klabel', text: 'prompt 里（模板产出）—— 1 个方块' }));
    const one = W.tensor(1, 12, (r, c) => (c === 0 ? 2 : -1), { cell: 22, gap: 4, text: (r, c) => (c === 0 ? '' : '') });
    one.style.justifySelf = 'start';
    const oneRow = U.el('div', { class: 'row gap10', style: 'width:100%' });
    oneRow.appendChild(one);
    oneRow.appendChild(U.el('div', { class: 'mono', style: 'font-size:11.5px;color:var(--ink-dim)', html: '&lt;|image|&gt; describe &nbsp;<span class="cm">// 占位符 1 个</span>' }));
    inner.appendChild(oneRow);

    inner.appendChild(U.el('div', { class: 'klabel', text: 'tokenize 之前（处理器产出）—— 144 个方块' }));
    const many = W.tensor(1, 144, () => 2, { cell: 5, gap: 1 });
    const manyRow = U.el('div', { class: 'row gap10', style: 'width:100%;align-items:center' });
    manyRow.appendChild(many);
    const cnt = U.el('div', { class: 'n', style: 'flex:none;width:52px;text-align:right', text: '0' });
    manyRow.appendChild(cnt);
    inner.appendChild(manyRow);

    const mid = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cA = W.card({ cc: 4, tint: 4, title: '分子：patch 数',
      body: U.el('div', { class: 'mono', style: 'font-size:12px;line-height:1.7', html: 'grid = <b>24 x 24</b><br>prod = 1 x 24 x 24 = <b>576</b>' }) });
    const cB = W.card({ cc: 2, tint: 2, title: '分母：merge_size^2',
      body: U.el('div', { class: 'mono', style: 'font-size:12px;line-height:1.7', html: 'merge_size = <b>2</b><br>2 x 2 = <b>4</b>' }) });
    const cC = W.card({ cc: 3, tint: 3, title: '结果：位置数 N',
      body: U.el('div', { class: 'mono', style: 'font-size:12px;line-height:1.7', html: '576 / 4 = <b>144</b><br>字符串长度 144 x 9 = <b>1296</b>' }) });
    [cA, cB, cC].forEach(c => { c.style.flex = '1'; mid.appendChild(c); });
    inner.appendChild(mid);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    many.cells.forEach(c => { c.style.opacity = '.12'; });
    cnt.textContent = '0';
    [cA, cB, cC].forEach(c => { c.style.opacity = '.3'; });
    msg.innerHTML = '<span class="cm">// 图片还没进 tokenizer，位置数就已经定死了</span>';
    tl.at(1800, () => { msg.innerHTML = '模板渲染出 <em>1 个</em> &lt;|image|&gt; &nbsp;<span class="cm">// 模板不认识图片尺寸</span>'; });
    tl.at(5200, () => { cA.style.opacity = '1'; msg.innerHTML = '图像处理器先给出 grid：这张 336x336 的图分块后是 <em>24 x 24</em>'; });
    tl.at(8600, () => { cB.style.opacity = '1'; msg.innerHTML = 'merger 会把 <em>2 x 2</em> 个 patch 并成 1 个 token —— 所以除以 4'; });
    tl.at(11800, () => {
      cC.style.opacity = '1';
      msg.innerHTML = '<span class="mono">576 / 4 = <em>144</em></span> &nbsp;<span class="cm">// 这就是 replace_image_token 的全部内容</span>';
    });
    tl.at(14800, () => {
      let i = 0;
      while (i < 144) { many.cells[i].style.opacity = '1'; i++; }
      cnt.textContent = '144';
      msg.innerHTML = '1 个占位符被复制成 <em>144 个</em> —— 它们在 tokenizer 眼里就是<em>普通文本 token</em>';
    });
    tl.at(18200, () => {
      msg.innerHTML = '所以视觉 token 与文本 token <em>共用一条序列、一张词表</em>；错位只可能发生在<em>个数</em>上';
    });
  },
},

/* ================================================================ 5 chat template */
{
  kicker: '第四幕 · 装配顺序',
  title: 'chat template 只负责<span class="hl-a">渲染</span>，装配是 processor 的事',
  sub: 'apply_chat_template(tokenize=True) 先收集内容块里的图片文件，渲染文本，然后回到那个 __call__。它没有第二条代码路径。',
  caption: '模板与处理器的契约只有一个：第 i 个 image 块 ↔ batch_images 里第 i 个文件。',
  lang: 'python',
  codeStart: 2149,
  code: `        if tokenize:
            batch_images, batch_videos = [], []
            batch_audios = []
            for conversation in conversations:
                images, videos = [], []
                for message in conversation:
                    content = message.get("content") or []
                    if isinstance(content, str):
                        continue
                    visuals = [
                        content_block for content_block in content if content_block["type"] in ["image", "video"]
                    ]`,
  codeNote: 'apply_chat_template 里真正"看图"的只有这几行：从 content 块里挑出 image / video。',
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const g = U.el('div', { class: 'vizgrow' });
    wrap.appendChild(g);
    const inner = U.el('div', { class: 'col gap12', style: 'width:100%' });
    g.appendChild(inner);

    const conv = [
      'conversation = [',
      '  { "role": "user", "content": [',
      '      { "type": "image", "image": <PIL> },',
      '      { "type": "text",  "text": "describe" } ] } ]',
    ].join('\n');
    inner.appendChild(U.el('div', { class: 'klabel', text: '输入：一次对话 —— 图和文是并列的内容块' }));
    inner.appendChild(U.el('div', { class: 'card', cc: 1, style: 'padding:10px 12px' }, [U.codeEl(conv, { lang: 'text' }).el]));

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const steps = [
      { t: '① 模板渲染', d: 'jinja 把 image 块写成 1 个 <|image|>；尺寸未知', c: 0 },
      { t: '② 收集素材', d: '处理器按同样顺序收 batch_images；顺序错就错位', c: 2 },
      { t: '③ 回到 __call__', d: 'out = self(text=prompt, images=batch_images, ...)', c: 4 },
    ];
    const cards = steps.map(s => {
      const e = W.card({ cc: s.c, tint: s.c, title: s.t,
        sub: '<span class="mono" style="font-size:10.5px">' + U.esc(s.d) + '</span>', style: 'flex:1' });
      row.appendChild(e); return e;
    });
    inner.appendChild(row);

    const ar = U.el('div', { class: 'formula', style: 'width:100%' });
    inner.appendChild(ar);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    cards.forEach(c => { c.style.opacity = '.25'; });
    ar.innerHTML = '<span class="cm">// 渲染结果（未 tokenize）</span><br>user:&lt;|image|&gt;describe &nbsp;&nbsp;<span class="op">↓</span>&nbsp;&nbsp;144 个占位符进了序列';
    msg.innerHTML = '<span class="cm">// 一次调用里其实发生了两次"对齐"</span>';
    tl.at(1800, () => { cards[0].style.opacity = '1'; msg.innerHTML = '第一次对齐：<em>模板里第 i 个 image 块</em> ↔ <em>第 i 张图</em>（靠顺序，不靠 id）'; });
    tl.at(5400, () => { cards[1].style.opacity = '1'; msg.innerHTML = '第二次对齐：<em>第 i 个占位符</em> ↔ <em>第 i 个替换串</em>（第六幕的迭代器）'; });
    tl.at(9000, () => { cards[2].style.opacity = '1'; msg.innerHTML = '<span class="mono">out = self(text=prompt, images=batch_images, videos=batch_videos, ...)</span>'; });
    tl.at(12600, () => {
      cards.forEach(c => c.classList.add('ac'));
      msg.innerHTML = '所以 <em>apply_chat_template 不是另一条路径</em>：它渲染完就把活交回 __call__';
    });
    tl.at(16000, () => {
      cards.forEach(c => c.classList.remove('ac'));
      msg.innerHTML = '<span class="cm">// GLM-5 的模板与处理器各管一半：模板管"1 个"，处理器管"N 个"</span>';
    });
  },
},

/* ================================================================ 6 展开循环 */
{
  kicker: '第五幕 · 展开与记账',
  title: '按<span class="hl-a">出现顺序</span>消费替换串，并记下每一笔 offset',
  sub: '第 i 个占位符换成第 i 个替换串 —— 顺序错了不会报错，只会静默错位。offset 是为了在"被撑长的序列"上反推位置。',
  caption: '这段循环的产物有两份：展开后的文本，和一张"谁换成了什么、换到哪儿"的账。',
  lang: 'python',
  codeStart: 898,
  code: `            for m in re.finditer(regex_special_mm_tokens, text[batch_idx]):
                start, end = m.span()
                expanded_sample.append(text[batch_idx][last:start])

                # adjust spans using running offset if one sample has several MM data associated
                start_with_offset = start + offset

                mm_type = m.lastgroup
                replacement_text = next(replacements_iters[mm_type])
                replacement_offsets.append(
                    {
                        "type": mm_type,
                        "span": (start, end),
                        "new_span": (start_with_offset, start_with_offset + len(replacement_text)),
                        "text": m.group(),
                        "replacement": replacement_text,
                    }
                )
                expanded_sample.append(replacement_text)
                # update the offsets and the last position
                offset += len(replacement_text) - (end - start)
                last = end`,
  codeNote: '命名分组分模态、迭代器按序消费、offset 增量维护 —— 三件事各占几行。',
  duration: 21000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const g = U.el('div', { class: 'vizgrow' });
    wrap.appendChild(g);
    const inner = U.el('div', { class: 'col gap12', style: 'width:100%' });
    g.appendChild(inner);

    inner.appendChild(U.el('div', { class: 'klabel', text: '句子里的占位符（按出现顺序编号）' }));
    const ph = U.el('div', { class: 'row gap8', style: 'width:100%;align-items:center' });
    const phEls = [];
    ['', 'A', '', 'B', '', 'C'].forEach((x, i) => {
      const isPh = i % 2 === 1;
      const e = U.el('div', {
        class: 'pill' + (isPh ? ' acc' : ''),
        style: 'flex:none;min-width:54px;justify-content:center;font-size:11px',
        html: isPh ? '&lt;|image|&gt; <b>' + x + '</b>' : '文本',
      });
      ph.appendChild(e); phEls.push(e);
    });
    inner.appendChild(ph);

    inner.appendChild(U.el('div', { class: 'klabel', text: '替换串迭代器 —— next() 一次取一个' }));
    const it = U.el('div', { class: 'row gap8', style: 'width:100%;align-items:center' });
    const itEls = ['144 个 token (A)', '361 个 token (B)', '144 个 token (C)'].map(t => {
      const e = W.card({ cc: 2, title: '<span class="mono" style="font-size:10.5px">' + t + '</span>', style: 'flex:1;padding:8px 10px' });
      it.appendChild(e); return e;
    });
    inner.appendChild(it);

    const tb = W.table([
      ['0', 'image', 'span (0, 9)', 'new_span (0, 1296)', '144 个 &lt;|image|&gt;'],
      ['1', 'image', 'span (18, 27)', 'new_span (1305, 2601)', '361 个 &lt;|image|&gt;'],
      ['2', 'image', 'span (33, 42)', 'new_span (2610, 3906)', '144 个 &lt;|image|&gt;'],
    ], { head: ['#', 'type', '原文位置', '展开后位置', '替换成'] });
    inner.appendChild(U.el('div', { class: 'card', cc: 4, style: 'padding:10px 12px' }, [
      U.el('div', { class: 'klabel', text: 'replacement_offsets —— 这段循环产出的账（形状示意，数字随样本变化）' }), tb,
    ]));
    const trs = Array.from(tb.querySelectorAll('tbody tr'));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    itEls.forEach(e => { e.style.opacity = '.25'; });
    trs.forEach(tr => { tr.style.opacity = '.25'; });
    phEls.forEach(e => { e.style.opacity = '.4'; });
    msg.innerHTML = '<span class="cm">// 三张图，三个占位符，按顺序消费</span>';
    tl.at(1800, () => { phEls[1].style.opacity = '1'; itEls[0].style.opacity = '1'; trs[0].style.opacity = '1'; msg.innerHTML = '第 1 个占位符 ← <em>第 1 个替换串</em> &nbsp;<span class="cm">// next(iter) 的次序就是文件次序</span>'; });
    tl.at(5600, () => { phEls[3].style.opacity = '1'; itEls[1].style.opacity = '1'; trs[1].style.opacity = '1'; msg.innerHTML = '第 2 个占位符 ← 第 2 个替换串 &nbsp;<span class="cm">// 尺寸不同，长度也不同（361 ≠ 144）</span>'; });
    tl.at(9400, () => { phEls[5].style.opacity = '1'; itEls[2].style.opacity = '1'; trs[2].style.opacity = '1'; msg.innerHTML = '第 3 个占位符 ← 第 3 个替换串 &nbsp;<span class="cm">// 谁多谁少都不会报错，只会错位</span>'; });
    tl.at(13000, () => {
      trs.forEach(tr => tr.classList.add('ac'));
      msg.innerHTML = '<span class="mono">offset += len(replacement) - (end - start)</span> &nbsp;<span class="cm">// 把原文位置换算成展开后位置</span>';
    });
    tl.at(16400, () => {
      trs.forEach(tr => tr.classList.remove('ac'));
      msg.innerHTML = '为什么要记这笔账？<em>return_assistant_tokens_mask</em> 要在撑长后的序列上反推"哪段是模型生成的"';
    });
    tl.at(19200, () => { msg.innerHTML = '账目是给推理引擎用的；模型真正需要的只有一件事：<em>个数要对得上</em>'; });
  },
},

/* ================================================================ 7 ★ 形状 */
{
  kicker: '第六幕 · ★ 形状演算',
  title: '336 x 336 的图：<span class="hl-a">24 x 24</span> 个 patch，<span class="hl-b">144</span> 个位置',
  sub: 'pixel_values 不是 (B, C, H, W)，而是"每个 patch 一行"的二维表 —— 最后一维 1176 = 3 x 2 x 14 x 14，那个 2 是补出来的时间维。',
  caption: '回顾 L2-05：resize / patchify 的张量级实现；下一课 L2-07 接手这张二维表。',
  lang: 'python',
  codeStart: 193,
  code: `        batch_size, channel, resized_height, resized_width = images.shape
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        patches = images.reshape(
            batch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
        flatten_patches = (
            patches.unsqueeze(6)
            .expand(-1, -1, -1, -1, -1, -1, temporal_patch_size, -1, -1)
            .reshape(
                batch_size,
                grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )
        )
        return flatten_patches, grid_h, grid_w`,
  codeNote: 'reshape 拆空间维 → permute 把 patch 的两维挪到末尾 → expand 补时间维 → reshape 拉平成表。',
  duration: 23000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const g = U.el('div', { class: 'vizgrow hstart' });
    wrap.appendChild(g);
    const inner = U.el('div', { class: 'row gap16', style: 'width:100%;align-items:center;justify-content:center' });
    g.appendChild(inner);

    const left = U.el('div', { class: 'col gap8', style: 'flex:none' });
    left.appendChild(U.el('div', { class: 'klabel', text: 'grid_h x grid_w = 24 x 24（每格 = 1 个 14x14 的 patch）' }));
    const t = W.tensor(24, 24, (r, c) => (Math.floor(r / 2) * 12 + Math.floor(c / 2)) % 6, { cell: 14, gap: 2 });
    left.appendChild(t);
    left.appendChild(U.el('div', { class: 'mono', style: 'font-size:11px;color:var(--ink-faint)', text: '同色 = 会被 merger 并成同一个 token 的 2x2 组' }));
    inner.appendChild(left);

    const right = U.el('div', { class: 'col gap10', style: 'flex:1;min-width:0' });
    const tb = W.table([
      ['原图', '336 x 336 x 3', 'RGB'],
      ['对齐画布', '336 x 336', 'align 到 28 的倍数（这里正好）'],
      ['grid', '24 x 24', '336 / 14'],
      ['patch 数', '<span class="hl">576</span>', '24 x 24'],
      ['每 patch 像素', '1176', '3 x <b>2</b> x 14 x 14（2 = temporal）'],
      ['位置数', '<span class="hl3">144</span>', '576 / 4（merge 2x2）'],
      ['pixel_values', '(576, 1176)', '实测 float32'],
    ], { head: ['量', '实测值', '怎么来的'] });
    right.appendChild(U.el('div', { class: 'card', cc: 3, style: 'padding:10px 12px' }, [tb]));
    const trs = Array.from(tb.querySelectorAll('tbody tr'));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    right.appendChild(msg);
    inner.appendChild(right);

    trs.forEach(tr => { tr.style.opacity = '.3'; });
    msg.innerHTML = '<span class="cm">// 一次性 reshape 到 (gh/2, 2, 14, gw/2, 2, 14)</span>';
    tl.at(1800, () => { trs[0].style.opacity = '1'; trs[1].style.opacity = '1'; msg.innerHTML = '第一步 <em>smart_resize</em>：向上对齐到 28 的倍数（336 正好，512 会变成 <em>532</em>）'; });
    tl.at(5400, () => { trs[2].style.opacity = '1'; msg.innerHTML = '第二步切 patch：每 <em>14 x 14</em> 像素一个块 → <em>24 x 24</em> 个'; });
    tl.at(9000, () => {
      trs[3].style.opacity = '1';
      let i = 0;
      while (i < 144) {
        t.at(2 * Math.floor(i / 12), 2 * (i % 12)).classList.add('on');
        i++;
      }
      msg.innerHTML = '576 个 patch 在内存里是一维的；<em>reshape</em> 只是换个看法，不搬数据';
    });
    tl.at(12600, () => {
      trs[4].style.opacity = '1';
      msg.innerHTML = '<em>expand</em> 给静态图补一个时间维：<span class="mono">3 x 2 x 14 x 14 = 1176</span> —— 与视频同一种布局';
    });
    tl.at(16200, () => {
      trs[5].style.opacity = '1';
      msg.innerHTML = '相邻的 <em>2 x 2</em> 个 patch 会并成一个 token → <em>576 / 4 = 144</em>';
    });
    tl.at(19800, () => {
      trs[6].style.opacity = '1';
      msg.innerHTML = '第三幕那个 144 到这里闭环：<em>grid</em> 是像素算出来的，<em>位置数</em> 是 grid 算出来的';
    });
  },
},

/* ================================================================ 8 视频 */
{
  kicker: '第七幕 · 视频',
  title: '时间维先<span class="hl-a">补齐</span>，再折进 patch',
  sub: 'temporal_patch_size = 2 要求帧数是偶数：patchify 会复制最后一帧兜底，采样阶段也会先把帧数凑成偶数。',
  caption: '同一个约束在两个层面各落实一次 —— 这是"形状决定代码"的典型例子。',
  lang: 'python',
  codeStart: 325,
  code: `        # Check that videos have \`num_frames\` divisible by \`temporal_patch_size\`
        if pad := -num_frames % temporal_patch_size:
            repeats = videos[:, -1:].expand(-1, pad, -1, -1, -1)
            videos = torch.cat((videos, repeats), dim=1)
            num_frames += pad

        grid_t = num_frames // temporal_patch_size
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size`,
  codeNote: '海象运算符一边算"补几帧"一边判断；补的是最后一帧，不是黑帧。',
  duration: 21000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const g = U.el('div', { class: 'vizgrow' });
    wrap.appendChild(g);
    const inner = U.el('div', { class: 'col gap12', style: 'width:100%' });
    g.appendChild(inner);

    inner.appendChild(U.el('div', { class: 'klabel', text: '16 帧 —— 两帧一组折成 1 个时间 patch' }));
    const fr = W.tensor(1, 16, (r, c) => Math.floor(c / 2) % 6, { cell: 20, gap: 5, text: (r, c) => String(c + 1) });
    const frRow = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:center' });
    frRow.appendChild(fr);
    frRow.appendChild(U.el('div', { class: 'mono', style: 'font-size:11.5px;color:var(--ink-dim)', html: '同色两帧 → 1 个时间 patch<br><span class="cm">// grid_t = 16 / 2 = 8</span>' }));
    inner.appendChild(frRow);

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cards = [
      { t: 'patchify 的输出', v: '(1, 4608, 1176)', d: 'grid = (8, 24, 24)', c: 0 },
      { t: '整段视频的 token', v: '1152', d: '4608 / 4（merge 2x2）', c: 2 },
      { t: '每帧的占位符', v: '72', d: '24x24 / (4 x 2)', c: 4 },
    ].map(x => {
      const e = W.card({ cc: x.c, tint: x.c, title: x.t,
        body: U.el('div', { class: 'col gap4' }, [
          U.el('div', { class: 'n big', text: x.v }),
          U.el('div', { class: 'mono', style: 'font-size:10.5px;color:var(--ink-dim)', text: x.d }),
        ]), style: 'flex:1;text-align:center' });
      row.appendChild(e); return e;
    });
    inner.appendChild(row);

    const pad = U.el('div', { class: 'formula', style: 'width:100%' });
    inner.appendChild(pad);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    cards.forEach(c => { c.style.opacity = '.25'; });
    pad.innerHTML = '<span class="cm">// 15 帧呢？</span> <span class="mono">-15 % 2 = 1</span> → 复制最后一帧 → <span class="mono">num_frames = 16</span> → grid_t = <em>8</em>（实测与 16 帧一致）';
    msg.innerHTML = '<span class="cm">// 帧数不管是 15 还是 16，grid_t 都是 8</span>';
    tl.at(1800, () => { cards[0].style.opacity = '1'; msg.innerHTML = 'patchify 的输出是"每行一个 patch"：<em>(1, 4608, 1176)</em>，时间维被折进 grid_t'; });
    tl.at(5600, () => { cards[1].style.opacity = '1'; msg.innerHTML = '整段视频占 <em>1152</em> 个位置 —— 这才是加到文本序列里的长度'; });
    tl.at(9000, () => { cards[2].style.opacity = '1'; msg.innerHTML = '但文本里是<em>逐帧</em>写的：每帧 72 个 = <span class="mono">24x24 / (4 x 2)</span>，分母多一个 <em>2</em>'; });
    tl.at(12600, () => {
      cards.forEach(c => c.classList.add('ac'));
      msg.innerHTML = '<span class="mono">replace_video_token</span> 里那句 <span class="mono">// num_frames</span> 就是在做这个除法（第三幕）';
    });
    tl.at(16000, () => {
      cards.forEach(c => c.classList.remove('ac'));
      msg.innerHTML = '奇数帧不是错误、也不会崩：<em>patchify 兜底补一帧</em>，采样阶段（第 234 行）也会先凑偶数';
    });
  },
},

/* ================================================================ 9 收束 */
{
  kicker: '收束 · ★ 序列对齐',
  title: '图像和视频<span class="hl-a">共用一个 token</span>，靠括号区分',
  sub: 'GLM-5 没有独立的视频占位符：同一串 <|image|> 落在 <|begin_of_video|> ... <|end_of_video|> 之内就是 2，之外就是 1。',
  caption: '下一课 L2-07：这些位置最终被视觉塔输出的向量填满。',
  lang: 'python',
  codeStart: 168,
  code: `    def create_mm_token_type_ids(self, input_ids: list) -> list[list[int]]:
        # We have to iterate for each list separately because inputs
        # might be non-padded lists and we can't cast numpy on that!
        # Then cast numpy as each input for faster indexing
        mm_token_type_ids = []
        for input in input_ids:
            array_ids = np.array(input)
            mm_token_types = np.zeros_like(input)

            # Replace 0 -> 2 only inside video segments because Glm5Next
            # uses the same special token to denote images and video
            # Otherwise replace 0 -> 1 for image modality
            starts = np.cumsum(array_ids == self.video_start_id, axis=0)
            ends = np.cumsum(array_ids == self.video_end_id, axis=0)
            is_video_modality = starts > ends

            mm_token_types[(array_ids == self.image_token_id) & is_video_modality] = 2
            mm_token_types[(array_ids == self.image_token_id) & (~is_video_modality)] = 1
            mm_token_type_ids.append(mm_token_types.tolist())
        return mm_token_type_ids`,
  codeNote: '两个 cumsum 之差 = "是否在视频段内"；模型侧用的是同一套规则，所以两边不会错位。',
  duration: 24000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const g = U.el('div', { class: 'vizgrow' });
    wrap.appendChild(g);
    const inner = U.el('div', { class: 'col gap10', style: 'width:100%' });
    g.appendChild(inner);

    inner.appendChild(U.el('div', { class: 'klabel', text: '实测：create_mm_token_type_ids([begin_video, img, img, end_video, img, text])' }));
    const strip = U.el('div', { class: 'row gap6', style: 'width:100%;align-items:center' });
    const names = ['&lt;|begin_of_video|&gt;', '&lt;|image|&gt;', '&lt;|image|&gt;', '&lt;|end_of_video|&gt;', '&lt;|image|&gt;', 'text'];
    const chips = names.map((n, i) => {
      const e = U.el('div', { class: 'pill', style: 'flex:none;font-size:10px;padding:4px 7px', html: n });
      strip.appendChild(e);
      const v = U.el('div', { class: 'mono', style: 'flex:none;font-size:10px;color:var(--ink-faint);min-width:12px;text-align:center', text: '?' });
      strip.appendChild(v);
      return { chip: e, val: v };
    });
    inner.appendChild(strip);

    const tb = W.table([
      ['input_ids', '五个 key 之一', '第 710 行拼进来'],
      ['attention_mask', 'tokenizer 默认产物', 'padding=False 时不补齐'],
      ['mm_token_type_ids', '处理器自己加', '默认 return_mm_token_type_ids=True'],
      ['pixel_values', '每个 patch 一行', '(576, 1176)'],
      ['image_grid_thw', '占位符个数的来源', '[[1, 24, 24]]'],
    ], { head: ['key', '谁给的', '备注'] });
    inner.appendChild(U.el('div', { class: 'card', cc: 3, style: 'padding:10px 12px' }, [
      U.el('div', { class: 'klabel', text: '回到验收点：一次图像对话的 inputs 有这五个 key' }), tb,
    ]));

    inner.appendChild(W.exercise(
      '一次图像对话的 <code class="inl">inputs</code> 有哪五个 key？'
      + '其中哪个是处理器自己加的、哪个决定了占位符个数？',
      '<b>五个 key</b>：<code class="inl">input_ids</code>、<code class="inl">attention_mask</code>、'
      + '<code class="inl">mm_token_type_ids</code>、<code class="inl">pixel_values</code>、<code class="inl">image_grid_thw</code>。'
      + '<br><code class="inl">mm_token_type_ids</code> 是处理器自己加的（默认 '
      + '<code class="inl">return_mm_token_type_ids=True</code>）；'
      + '<code class="inl">image_grid_thw</code> 决定占位符个数，公式 '
      + '<code class="inl">grid_t x grid_h x grid_w / merge_size^2</code>。'
      + '<br>336x336 的图：24 x 24 / 4 = <b>144</b> 个位置 —— 这就是序列里那 144 个 '
      + '<code class="inl">&lt;|image|&gt;</code>。'));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    chips.forEach(c => { c.val.textContent = '?'; c.chip.classList.remove('acc'); });
    msg.innerHTML = '<span class="cm">// 光看 token id 分不出图像还是视频</span>';
    tl.at(2000, () => {
      chips.forEach((c, i) => {
        c.val.textContent = String(M.mmTypes[i]);
        if (M.mmTypes[i] === 2) { c.chip.classList.add('acc'); c.chip.style.borderColor = 'rgba(251,191,36,.6)'; }
        if (M.mmTypes[i] === 1) { c.chip.classList.add('acc'); }
      });
      msg.innerHTML = '实测结果 <span class="mono">[0, 2, 2, 0, 1, 0]</span> &nbsp;<span class="cm">// 段内 = 2（视频），段外 = 1（图像）</span>';
    });
    tl.at(5600, () => { msg.innerHTML = '两个 <em>cumsum</em> 之差就是"在不在视频段里" —— 一次 O(n) 前缀和代替区间循环'; });
    tl.at(9200, () => { msg.innerHTML = '为什么必须这样？因为模型侧要用<em>同一条规则</em>把占位符切成两堆再填向量'; });
    tl.at(12800, () => { msg.innerHTML = '规则一旦不一致，填进去的向量会<em>整体错位</em> —— 展开见 L2-07 的 <span class="mono">get_placeholder_mask</span>'; });
    tl.at(16400, () => { msg.innerHTML = '<span class="cm">// 一句话总结</span> 模板写 1 个 → 处理器展开成 N 个 → 与文本同一序列 → 视觉塔按同一规则填回'; });
    tl.at(20200, () => { msg.innerHTML = '下一课 <em>L2-07</em>：从这张 (576, 1176) 的表，到填满那 144 个位置的向量'; });
  },
},

];
