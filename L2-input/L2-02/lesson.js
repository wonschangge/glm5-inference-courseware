/* ==========================================================================
   L2-02 · 快分词器、纯 Python 分词器与慢→快转换
   --------------------------------------------------------------------------
   覆盖：tokenization_utils_tokenizers.py / tokenization_python.py / convert_slow_tokenizer.py
        （3 个文件 / 5033 行，行数按实测 wc -l）
   目标：看完能说出文本变 id 时「快」与「慢」分别在哪、GLM-5 实际走哪条路，
        并把「文本 token + 视觉 token 在同一条序列上对齐」算成具体形状。
   实测来源：_data/recon/probe_l202.py（GLM-5.3-Flash 真 tokenizer + 真图像处理器）。
   排版基准：#visual 可视区 = 994 x 662.7 舞台像素（无头浏览器实测）。
   ========================================================================== */
'use strict';

const M = {
  vocab: 154820,        // tok.vocab_size（基础词表，实测）
  total: 154856,        // len(tok)，含 36 个 added token
  image: 154854,        // "<|image|>"，= Glm5NextConfig.image_token_id
  maxlen: 1048576,      // tokenizer_config.json 的 model_max_length
  pad: 154820,          // pad = eos = "<|endoftext|>"
  batch: [1, 4, 40],    // 3 条测试文本的真实 token 数
  vision: { '448x448': 256, '672x336': 288, '224x224': 64 },
};

const SCENES = [

/* ------------------------------------------------- 1 全景：三条路 */
{
  kicker: 'L2 · 输入流水线',
  title: '三个文件，<span class="hl-a">三条</span>从文本到 id 的路',
  sub: '同一个输入，可能走 Rust 的 fast 后端、纯 Python 的 slow 后端，或者「先翻译再跑」的转换器。',
  caption: '本课覆盖的 3 个文件 / 5033 行；下一课 L2-03 展开其中的 SentencePiece 部分。',
  lang: 'python',
  codeStart: 85,
  code: `@add_end_docstrings(INIT_TOKENIZER_DOCSTRING)
class TokenizersBackend(PreTrainedTokenizerBase):
    """
    Base class for all fast tokenizers (wrapping HuggingFace tokenizers library).

    Inherits from [\`~tokenization_utils_base.PreTrainedTokenizerBase\`].

    Handles all the shared methods for tokenization and special tokens, as well as methods for
    downloading/caching/loading pretrained tokenizers, as well as adding tokens to the vocabulary.

    This class also contains the added tokens in a unified way on top of all tokenizers so we don't have to handle the
    specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece...).
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model = None
    _tokenizer = None`,
  codeNote: 'TokenizersBackend 的类声明：vocab_files_names 同时声明了两种序列化格式的槽位。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap14 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '字符串 → 整数 id：三条路的入口分别是' }));
    const fl = W.flow([
      { t: '"你好，世界"', s: 'str', cc: 0 },
      { t: 'tokenizer.json', s: 'Rust 序列化', cc: 4 },
      { t: 'encode_batch', s: '一次调用', cc: 2 },
      { t: 'input_ids', s: '(B, L)', cc: 1 },
    ], { style: 'width:100%' });
    viz.appendChild(fl);

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cFast = W.card({ cc: 4, tint: 4, title: 'fast · TokenizersBackend',
      sub: '<span class="mono">tokenization_utils_tokenizers.py</span><br>1514 行 · 循环在 Rust 里' });
    const cSlow = W.card({ cc: 1, tint: 1, title: 'slow · PythonBackend',
      sub: '<span class="mono">tokenization_python.py</span><br>1424 行 · 循环在 Python 里' });
    const cConv = W.card({ cc: 2, tint: 2, title: '转换器 · 56 个类',
      sub: '<span class="mono">convert_slow_tokenizer.py</span><br>2095 行 · 只在有 .model 时才进场' });
    [cFast, cSlow, cConv].forEach(c => { c.style.flex = '1'; row.appendChild(c); });
    viz.appendChild(row);

    const stats = [
      { k: 'vocab_size', v: M.vocab, cc: 0 },
      { k: 'len(tok)', v: M.total, cc: 1 },
      { k: 'image_token_id', v: M.image, cc: 2 },
      { k: 'model_max_length', v: M.maxlen, cc: 3 },
    ];
    const srow = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const sEls = stats.map(st => {
      const c = W.card({ cc: st.cc, tint: st.cc,
        title: '<span class="mono" style="font-size:10.5px">' + st.k + '</span>',
        body: U.el('div', { class: 'n big', text: U.fmt(st.v) }), style: 'flex:1' });
      srow.appendChild(c); return c;
    });
    viz.appendChild(srow);
    viz.appendChild(U.el('div', { class: 'klabel', text: '上面四个数字全部来自 _data/recon/probe_l202.py 的实测输出' }));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    sEls.forEach(e => { e.style.opacity = '.25'; });
    fl.focus(0);
    msg.innerHTML = '<span class="cm">// 三个文件不是三个模型，是同一件事的三种实现</span>';
    tl.at(2000, () => { fl.focus(1); cSlow.style.opacity = '.3'; cConv.style.opacity = '.3';
      msg.innerHTML = 'fast：<em>tokenizer.json</em> 直接喂给 Rust 后端 —— GLM-5 走的是这条'; });
    tl.at(5200, () => { fl.focus(2); cFast.style.opacity = '.3'; cSlow.style.opacity = '1'; cConv.style.opacity = '.3';
      msg.innerHTML = 'slow：<em>纯 Python</em> 逐条分词，没有 tokenizers 二进制也能跑'; });
    tl.at(8600, () => { fl.focus(3); cFast.style.opacity = '.3'; cSlow.style.opacity = '.3'; cConv.style.opacity = '1';
      msg.innerHTML = '转换器：把 <em>SentencePiece 的 .model</em> 翻译成 Rust 组件 —— 只在没有 tokenizer.json 时'; });
    tl.at(12000, () => {
      fl.focus(-1); cFast.style.opacity = '1'; cSlow.style.opacity = '1'; cConv.style.opacity = '1';
      msg.innerHTML = 'GLM-5.3-Flash 的 checkpoint 里<em>只有 tokenizer.json</em>：另外两条是退路与对照物';
    });
    tl.at(14600, () => {
      sEls.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = '<span class="cm">// 记住 154820 / 154856 / 154854 / 1048576 这四个实测数</span>';
    });
  },
},

/* ------------------------------------------------- 2 ★ 走哪一支 */
{
  kicker: '阶段 ① · 分支',
  title: '★ GLM-5 走的是<span class="hl-a">最短的那一支</span>',
  sub: 'convert_to_native_format 是加载期的唯一分叉点：磁盘上有什么文件，决定后端从哪来。',
  caption: '「慢→快转换」对 GLM-5 是冷路径 —— 它的 checkpoint 里没有 .model 文件。',
  lang: 'python',
  codeStart: 104,
  code: `    @classmethod
    def convert_to_native_format(cls, trust_remote_code=False, **kwargs):
        """
        Build a \`tokenizers.Tokenizer\` backend from the available serialization files (tokenizer.json, sentencepiece
        models, tekken.json, vocab/merges).
        """
        # Preserve kwargs for possible downstream use
        local_kwargs = dict(kwargs)
        fast_tokenizer_file = local_kwargs.pop("tokenizer_file", None)

        if (
            fast_tokenizer_file is not None
            and os.path.isfile(fast_tokenizer_file)
            and (cls is TokenizersBackend or "__init__" not in cls.__dict__ or trust_remote_code)
        ):
            local_kwargs["tokenizer_object"] = TokenizerFast.from_file(fast_tokenizer_file)
            return local_kwargs`,
  codeNote: '第一个分支只有一行：TokenizerFast.from_file —— 不解析词表、不重算 merges。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'vizgrow hstart',
      style: 'display:flex;flex-direction:row;gap:14px;width:100%;align-items:stretch' });
    wrap.appendChild(viz);

    const left = U.el('div', { class: 'col gap8', style: 'flex:1.25' });
    left.appendChild(U.el('div', { class: 'klabel', text: 'convert_to_native_format 的四个出口' }));
    const branches = [
      { t: '① tokenizer.json 存在，且类没有自定义 __init__', s: 'TokenizerFast.from_file(path)——GLM-5 命中', cc: 4 },
      { t: '② tokenizer.json 存在，但类有自定义 __init__', s: '从 JSON 里抽 vocab / merges，交给子类重建', cc: 0 },
      { t: '③ tekken.json（Mistral）', s: 'MistralConverter(vocab_file).converted()', cc: 2 },
      { t: '④ vocab_file 以 .model 结尾', s: '_convert_from_sentencepiece → convert_slow_tokenizer', cc: 1 },
    ];
    const cards = branches.map(b => {
      const c = W.card({ cc: b.cc, title: b.t, sub: b.s });
      left.appendChild(c); return c;
    });
    viz.appendChild(left);

    const right = U.el('div', { class: 'col gap8', style: 'flex:1' });
    right.appendChild(U.el('div', { class: 'klabel', text: '实测：这个目录里有什么' }));
    right.appendChild(U.el('div', { class: 'card cc4', style: 'padding:10px 12px' }, [
      U.el('div', { class: 'mono', style: 'font-size:11.5px;line-height:1.85' , html:
        'tokenizer.json&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="dim">20,217,442 B</span><br>' +
        'tokenizer_config.json<br>processor_config.json' }),
    ]));
    right.appendChild(U.el('div', { class: 'klabel', text: '实测：加载结果' }));
    right.appendChild(U.el('div', { class: 'card cc1', style: 'padding:10px 12px' }, [
      U.el('div', { class: 'mono', style: 'font-size:11.5px;line-height:1.9', html:
        'is_fast = <span class="hl4">True</span><br>' +
        'vocab_size = <span class="hl">154820</span><br>' +
        'len(tok) = <span class="hl">154856</span><br>' +
        'can_save_slow_tokenizer = <span class="hlbad">False</span><br>' +
        'PythonBackend.from_pretrained() → <span class="hlbad">NotImplementedError</span>' }),
    ]));
    viz.appendChild(right);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    cards.forEach(c => c.classList.add('dimmed'));
    msg.innerHTML = '<span class="cm">// 磁盘上有哪种序列化文件，就走哪一支</span>';
    tl.at(2200, () => { cards[0].classList.remove('dimmed'); cards[0].classList.add('ac');
      msg.innerHTML = '① 命中：<em>一行 from_file</em>，1.06 s 加载完 20 MB —— 词表、merges、pre_tokenizer 全在文件里'; });
    tl.at(5600, () => { cards[0].classList.remove('ac'); cards[1].classList.remove('dimmed');
      msg.innerHTML = '② 给自定义 <em>__init__</em> 的类留的：从 JSON 里抽 vocab/merges 再自己重建后端'; });
    tl.at(9000, () => { cards[1].classList.add('dimmed'); cards[3].classList.remove('dimmed');
      msg.innerHTML = '④ 只有 <em>tokenizer.model</em> 时才走这条 —— 需要 sentencepiece + protobuf，GLM-5 用不到'; });
    tl.at(12400, () => {
      cards.forEach(c => { c.classList.remove('dimmed'); c.classList.remove('ac'); });
      cards[0].classList.add('ac');
      msg.innerHTML = '实测：<em>can_save_slow_tokenizer = False</em>，这份 checkpoint 没有慢分词器的原料';
    });
    tl.at(15000, () => { msg.innerHTML = '<span class="cm">// 另外两个出口（tekken / tiktoken）见 source.md 第一节</span>'; });
  },
},

/* ------------------------------------------------- 3 fast 批量 */
{
  kicker: '阶段 ② · fast 批量',
  title: '一次 <span class="hl-a">encode_batch</span>，B 和 L 就都定了',
  sub: 'fast 路径的 Python 侧没有任何逐条逻辑：单条也是「长度为 1 的批量」，交给 Rust 一起算。',
  caption: '慢路径在这里是一个 Python for 循环 —— 见第 6 幕。',
  lang: 'python',
  codeStart: 1062,
  code: `        # Direct rust backend call
        encodings = self._tokenizer.encode_batch(
            batch_text_or_text_pairs,
            add_special_tokens=add_special_tokens,
            is_pretokenized=is_split_into_words,
        )`,
  codeNote: '函数名是 encode_batch，不是 encode：单条也会被包成单元素 batch。',
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '3 条不等长文本 → 一次 Rust 调用 → 一个对齐的 (3, 40)' }));
    const bars = W.bars([
      { label: '第 0 条 "短"', value: M.batch[0], cc: 0 },
      { label: '第 1 条 一句话', value: M.batch[1], cc: 1 },
      { label: '第 2 条 长句 ×4', value: M.batch[2], cc: 2 },
    ], { max: M.batch[2] });
    bars.setAll(0);
    viz.appendChild(U.el('div', { class: 'card cc0', style: 'padding:10px 12px' }, [bars]));

    const fl = W.flow([
      { t: '3 条文本', s: 'list[str]', cc: 0 },
      { t: 'encode_batch', s: 'Rust · 一次', cc: 4 },
      { t: 'encodings', s: '3 个 Encoding', cc: 2 },
      { t: 'BatchEncoding', s: '(3, 40) 张量', cc: 1 },
    ], { style: 'width:100%' });
    viz.appendChild(fl);

    const shp = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    shp.appendChild(W.card({ cc: 4, title: '实测形状', style: 'flex:1',
      body: U.el('div', { class: 'mono', style: 'font-size:11.5px;line-height:1.8', html:
        'input_ids&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(3, 40)<br>' +
        'attention_mask&nbsp;(3, 40)<br>' +
        '<span class="dim">每行真实长度 [1, 4, 40]</span>' }) }));
    shp.appendChild(W.card({ cc: 2, title: '左侧填充（padding_side = "left"）', style: 'flex:1',
      body: U.el('div', { class: 'mono', style: 'font-size:11.5px;line-height:1.8', html:
        '第 0 行的前 39 位 = 154820<br>' +
        '<span class="dim">&lt;|endoftext|&gt; 同时是 pad_token 与 eos_token</span>' }) }));
    shp.appendChild(W.card({ cc: 3, title: '256 条一次调用', style: 'flex:1',
      body: U.el('div', { class: 'mono', style: 'font-size:11.5px;line-height:1.8', html:
        '<span class="n big" style="font-size:17px">4 ms</span><br>' +
        '<span class="dim">分开调 256 次 = 7 ms；差的是 Python 侧开销</span>' }) }));
    viz.appendChild(shp);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    fl.focus(0);
    msg.innerHTML = '<span class="cm">// 三条长度 1 / 4 / 40 的文本，先看长度分布</span>';
    tl.at(2200, () => { bars.setTo(0, 1); bars.setTo(1, 1); bars.setTo(2, 1);
      msg.innerHTML = '三条真实长度：<em>1</em> / <em>4</em> / <em>40</em> —— 不补齐就没法组成一个张量'; });
    tl.at(5200, () => { fl.focus(1);
      msg.innerHTML = 'Python 侧只做一件事：把 batch 交给 <em>encode_batch</em>，一次调用'; });
    tl.at(8400, () => { fl.focus(2);
      msg.innerHTML = 'Rust 返回 <em>3 个 Encoding</em>（ids / attention_mask / offsets 都在里面）'; });
    tl.at(11600, () => { fl.focus(3);
      msg.innerHTML = '拼成 <em>(3, 40)</em>：行 = 样本、列 = 该 batch 的最长序列 —— <span class="cm">两者都由这一次调用决定</span>'; });
    tl.at(13600, () => { msg.innerHTML = '<span class="cm">// 慢路径做不到「一次调用」：见第 6 幕的 Python for 循环</span>'; });
  },
},

/* ------------------------------------------------- 4 形状：降维 */
{
  kicker: '阶段 ③ · 形状',
  title: '单条为什么会被<span class="hl-a">降维</span>，批量为什么不会',
  sub: '降维条件写得很精确：不是批量 + 不张量化 + 没有 overflow，三条同时成立才把 (1, L) 压成 (L,)。',
  caption: '这是 L2 这一层「形状」的第一处陷阱：return_tensors 与降维是两个独立的开关。',
  lang: 'python',
  codeStart: 1101,
  code: `        batched_output = BatchEncoding(sanitized_tokens, sanitized_encodings, tensor_type=return_tensors)

        # If single input, remove the batch dimension (unless returning overflowing tokens)
        if not is_batched and return_tensors is None and not return_overflowing_tokens:
            batched_output = BatchEncoding(
                {
                    key: (value[0] if len(value) > 0 and isinstance(value[0], list) else value)
                    for key, value in batched_output.items()
                },
                batched_output.encodings,
            )

        return batched_output`,
  codeNote: 'tensor_type=return_tensors 在构造 BatchEncoding 时才生效，编码早就做完了。',
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '同一句话，三种调用 → 三种形状（实测）' }));
    const tb = W.table([
      ['tok("…一句话…")', '单条 · 不张量化', '<span class="hl4">(14,)</span>', '降维：去掉 batch 维'],
      ['tok("…一句话…", return_tensors="pt")', '单条 · 张量化', '<span class="hl">(1, 14)</span>', '不降维：return_tensors 不是 None'],
      ['tok([a, b, c], padding=True, return_tensors="pt")', '批量', '<span class="hl2">(3, 40)</span>', '本来就有 batch 维'],
    ], { head: ['调用', '输入形态', 'input_ids 形状', '原因'] });
    const tbCard = U.el('div', { class: 'card cc0', style: 'padding:9px 11px' }, [tb]);
    viz.appendChild(tbCard);
    const rows = Array.from(tb.querySelectorAll('tbody tr'));

    const strip = U.el('div', { class: 'row gap4', style: 'width:100%;align-items:center' });
    strip.appendChild(U.el('div', { class: 'mono faint', style: 'font-size:10.5px;width:120px', text: '14 个 token' }));
    const cells = [];
    for (let i = 0; i < 14; i++) {
      const c = U.el('div', { class: 'n sm', style: 'flex:1 1 0;max-width:44px;height:30px;border-radius:6px;'
        + 'display:flex;align-items:center;justify-content:center;'
        + 'border:1px solid rgba(56,189,248,.5);background:rgba(56,189,248,.12);color:#bae6fd', text: '·' });
      strip.appendChild(c); cells.push(c);
    }
    viz.appendChild(U.el('div', { class: 'col gap6', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: '同一串 id，加不加 batch 维' }), strip,
    ]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    rows.forEach(r => { r.style.opacity = '.3'; });
    msg.innerHTML = '<span class="cm">// 编码本身不关心形状，形状是最后一步拼出来的</span>';
    tl.at(2000, () => { rows[0].style.opacity = '1'; rows[1].style.opacity = '.3'; rows[2].style.opacity = '.3';
      msg.innerHTML = '单条 + 不张量化 → <em>(14,)</em>：batch 维被显式去掉'; });
    tl.at(5200, () => { rows[1].style.opacity = '1'; rows[0].style.opacity = '.3';
      msg.innerHTML = '单条 + <em>return_tensors="pt"</em> → <em>(1, 14)</em>：这条最容易踩，模型要的就是它'; });
    tl.at(8400, () => { rows[2].style.opacity = '1';
      cells.forEach((c, i) => { if (i % 2) { c.style.background = 'rgba(56,189,248,.28)'; c.textContent = String(i); } });
      msg.innerHTML = '批量 → <em>(3, 40)</em>：列数由最长的那条决定，短的左侧补 154820'; });
    tl.at(11800, () => {
      msg.innerHTML = 'overflow 时也不降维：<em>return_overflowing_tokens</em> 会让一条样本变多行';
    });
    viz.appendChild(W.exercise(
      '单条文本 + <code class="inl">return_tensors="pt"</code>，<code class="inl">input_ids</code> 的形状是 '
      + '<code class="inl">(14,)</code> 还是 <code class="inl">(1, 14)</code>？',
      '是 <b>(1, 14)</b>。降维那一段的条件是 <code class="inl">not is_batched and return_tensors is None</code>；'
      + '传了 <code class="inl">return_tensors</code> 就不满足第二个条件。<br>'
      + '反过来：<b>不带 tensor 参数的单条调用</b>才会被压成 <code class="inl">(14,)</code>。'));
  },
},

/* ------------------------------------------------- 5 慢路径 Trie */
{
  kicker: '阶段 ④ · 慢路径',
  title: '纯 Python 的第一件：<span class="hl-b">Trie</span> 单遍切分',
  sub: 'added token 必须整体成为一个 id。慢路径用一棵字符树，在单遍扫描里切出所有边界。',
  caption: 'fast 那边由 Rust 内部的 added-token 表负责同一件事。',
  lang: 'python',
  codeStart: 98,
  code: `    def split(self, text: str) -> list[str]:
        """
        Will look for the words added to the trie within \`text\`. Output is the original string split along the
        boundaries of the words found.

        This trie will match the longest possible word first !

        Example:

        \`\`\`python
        >>> trie = Trie()
        >>> trie.split("[CLS] This is a extra_id_100")
        ["[CLS] This is a extra_id_100"]

        >>> trie.add("[CLS]")
        >>> trie.add("extra_id_1")
        >>> trie.add("extra_id_100")
        >>> trie.split("[CLS] This is a extra_id_100")
        ["[CLS]", " This is a ", "extra_id_100"]
        \`\`\``,
  codeNote: 'docstring 里的例子就是规格说明书：extra_id_1 与 extra_id_100 必须吃满更长的那个。',
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const text = '前缀[gMASK]<sop>中间<|image|>后缀[gMASK]';
    const parts = ['前缀', '[gMASK]', '<sop>', '中间', '<|image|>', '后缀', '[gMASK]'];
    viz.appendChild(U.el('div', { class: 'klabel', text: '实测：把三个控制 token 装进 Trie 后 split 这串文本' }));
    const strip = U.el('div', { class: 'row gap2', style: 'width:100%' });
    const chars = Array.from(text).map(ch => {
      const e = U.el('div', { style: 'flex:1 1 0;min-width:12px;height:30px;border-radius:5px;'
        + 'display:flex;align-items:center;justify-content:center;font:600 11px var(--mono);'
        + 'border:1px solid rgba(150,180,255,.16);background:rgba(150,180,255,.05);'
        + 'color:var(--ink-dim);transition:background .3s,color .3s,border-color .3s', text: ch });
      strip.appendChild(e); return e;
    });
    viz.appendChild(U.el('div', { class: 'card cc1', style: 'padding:9px 11px' }, [strip]));

    viz.appendChild(U.el('div', { class: 'klabel', text: '切分结果（逐个出现）' }));
    const out = U.el('div', { class: 'row gap6 wrap', style: 'width:100%;min-height:26px' });
    const chips = [];
    parts.forEach(p => {
      const c = U.el('span', { class: 'chip c1', style: 'opacity:0;transition:opacity .3s', text: p });
      chips.push(c); out.appendChild(c);
    });
    viz.appendChild(out);
    viz.appendChild(U.el('div', { class: 'mono dim', style: 'font-size:11px', html:
      '这串文本里控制 token 出现 <span class="hl">4</span> 次；切出来的 7 段里，' +
      '<span class="hl2">4 段是控制 token、3 段才是交给 BPE 的文本</span>' }));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    let cursor = 0;
    parts.forEach((p, k) => {
      const start = cursor; cursor += p.length;
      tl.at(1550 + k * 1800, () => {
        for (let i = 0; i < chars.length; i++) {
          const on = i >= start && i < cursor;
          chars[i].style.background = on ? 'rgba(167,139,250,.30)' : 'rgba(150,180,255,.05)';
          chars[i].style.borderColor = on ? 'rgba(167,139,250,.8)' : 'rgba(150,180,255,.16)';
          chars[i].style.color = on ? '#fff' : 'var(--ink-dim)';
        }
        chips[k].style.opacity = '1';
        msg.innerHTML = '第 <em>' + (k + 1) + '</em> 段：<span class="hl2">' + U.esc(p) + '</span> '
          + '<span class="cm">// 下标 [' + start + ', ' + cursor + ')</span>';
      });
    });
    tl.at(13500, () => {
      msg.innerHTML = '<span class="cm">// 单遍扫描 + 最长匹配 = 控制 token 永远不会被后面的分词器切碎</span>';
    });
  },
},

/* ------------------------------------------------- 6 慢路径批量循环 */
{
  kicker: '阶段 ⑤ · 慢路径',
  title: '慢路径的批量 = Python 里的 <span class="hl-c">for 循环</span>',
  sub: '逐条递归回自己，padding 一律留到最后统一做 —— 三个注释写死了这个分工。',
  caption: '对照第 3 幕：fast 的 Python 调用次数是 1，慢路径正比于条数。',
  lang: 'python',
  codeStart: 761,
  code: `                current_output = self._encode_plus(
                    text=current_text,
                    text_pair=current_pair,
                    add_special_tokens=add_special_tokens,
                    padding_strategy=PaddingStrategy.DO_NOT_PAD,  # we pad in batch afterward
                    truncation_strategy=truncation_strategy,
                    max_length=max_length,
                    stride=stride,
                    is_split_into_words=is_split_into_words,
                    pad_to_multiple_of=None,  # we pad in batch afterward
                    padding_side=None,  # we pad in batch afterward
                    return_tensors=None,  # We convert the whole batch to tensors at the end
                    return_token_type_ids=return_token_type_ids,
                    return_attention_mask=False,  # we pad in batch afterward
                    return_overflowing_tokens=return_overflowing_tokens,
                    return_special_tokens_mask=return_special_tokens_mask,
                    return_length=return_length,
                    verbose=verbose,
                    **kwargs,
                )
                for key, value in current_output.items():
                    batch_outputs.setdefault(key, []).append(value)`,
  codeNote: '三个 "# we pad in batch afterward" 是慢路径的分工声明：逐条只切词。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '同一次批量调用，两条路的调用次数' }));
    const lanes = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const mk = (name, cc, sub) => {
      const c = W.card({ cc: cc, title: name, sub: sub, style: 'flex:1' });
      const row = U.el('div', { class: 'row gap4', style: 'width:100%;margin-top:7px' });
      const bs = [];
      for (let i = 0; i < 4; i++) {
        const b = U.el('div', { class: 'n sm', style: 'flex:1;height:26px;border-radius:6px;'
          + 'display:flex;align-items:center;justify-content:center;'
          + 'border:1px solid var(--c' + cc + ');background:rgba(150,180,255,.06);color:var(--ink-dim)',
          text: '条' + i });
        row.appendChild(b); bs.push(b);
      }
      c.appendChild(row);
      return { card: c, cells: bs };
    };
    const A = mk('fast：1 次调用', 4, 'Python 侧调用一次 encode_batch');
    const B = mk('slow：N 次调用', 1, '每条一次递归 _encode_plus');
    lanes.appendChild(A.card); lanes.appendChild(B.card);
    viz.appendChild(lanes);

    const tb = W.table([
      ['循环在哪', '<span class="hl4">Rust</span>', '<span class="hl2">Python</span>'],
      ['Python 侧调用次数', '1', '∝ 批量条数'],
      ['padding 时机', 'Rust 内部一次做完', '循环之后 self.pad(...)'],
      ['逐条产出', 'Encoding 对象', 'dict（再 setdefault 追加成 list）'],
      ['offsets 这类逐字符账本', '有（Rust 记）', '没有'],
    ], { head: ['', 'fast', 'slow'] });
    viz.appendChild(U.el('div', { class: 'card cc2', style: 'padding:9px 11px' }, [tb]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    A.cells.forEach(c => { c.style.opacity = '.3'; });
    B.cells.forEach(c => { c.style.opacity = '.3'; });
    msg.innerHTML = '<span class="cm">// 慢路径也支持 batch —— 但批量的循环体在 Python 里</span>';
    tl.at(2200, () => { A.cells.forEach(c => { c.style.opacity = '1'; c.style.background = 'rgba(52,211,153,.16)'; });
      A.card.classList.add('ac');
      msg.innerHTML = 'fast：<em>一次 encode_batch</em> 覆盖整个 batch，Python 侧与条数无关'; });
    tl.at(5600, () => { A.card.classList.remove('ac'); B.card.classList.add('ac');
      msg.innerHTML = 'slow：<em>zip(text, pairs)</em> 逐条处理，每条都递归回 _encode_plus'; });
    B.cells.forEach((c, i) => {
      tl.at(7400 + i * 1800, () => {
        c.style.opacity = '1'; c.style.background = 'rgba(167,139,250,.18)';
      });
    });
    tl.at(14600, () => {
      msg.innerHTML = '逐条时传 <em>return_attention_mask=False</em>、<em>padding=DO_NOT_PAD</em> —— 只切词，不对齐';
    });
    tl.at(16600, () => {
      msg.innerHTML = '<span class="cm">// 对齐与张量化都推到循环之外：这是慢路径形状更晚成形的原因</span>';
    });
  },
},

/* ------------------------------------------------- 7 慢→快转换 */
{
  kicker: '阶段 ⑥ · 转换',
  title: '★ 把 SentencePiece <span class="hl-a">翻译</span>成 Rust 组件',
  sub: '模型（Unigram 还是 BPE）、归一化、解码三件套都从 proto 里翻译出来 —— 但预分词器不在这里装。',
  caption: '实测订正：build_tokenizer_from_spm_proto 只装 normalizer 与 decoder，pre_tokenizer = None。',
  lang: 'python',
  codeStart: 671,
  code: `        # normalizer
        _normalizers = [normalizers.Replace(" ", "▁")]
        if precompiled_charsmap:
            _normalizers.insert(0, normalizers.Precompiled(precompiled_charsmap))
        tokenizer.normalizer = normalizers.Sequence(_normalizers)

        # decoder
        if byte_fallback:
            tokenizer.decoder = decoders.Sequence(
                [decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse()]
            )
        else:
            tokenizer.decoder = decoders.Sequence([decoders.Replace("▁", " ")])

        return tokenizer`,
  codeNote: '编码侧 " " → "▁"，解码侧 "▁" → " "：三个组件必须成对设计。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '实测：5 个 piece 的 proto 装出来的 Rust 后端' }));
    const comp = U.el('div', { class: 'row gap10', style: 'width:100%;align-items:stretch' });
    const items = [
      { t: 'model', v: 'BPE / Unigram', d: 'vocab 是 dict → BPE；list[tuple] → Unigram', cc: 4 },
      { t: 'normalizer', v: 'Sequence', d: 'Replace(" ", "▁") [+ Precompiled]', cc: 0 },
      { t: 'pre_tokenizer', v: 'None', d: '★ 这个函数不装它：由 converted() 补', cc: 3 },
      { t: 'decoder', v: 'Sequence', d: 'Replace("▁"," ") + ByteFallback + Fuse', cc: 2 },
    ];
    const els = items.map(it => {
      const c = W.card({ cc: it.cc, title: '<span class="mono">' + it.t + '</span>',
        sub: '<span class="hl">' + it.v + '</span><br>' + it.d, style: 'flex:1' });
      comp.appendChild(c); return c;
    });
    viz.appendChild(comp);

    const rt = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    rt.appendChild(W.card({ cc: 5, title: '往返实测', style: 'flex:1',
      body: U.el('div', { class: 'mono', style: 'font-size:11.5px;line-height:1.8', html:
        'encode("▁你好▁世界") → [1, 2, 3]<br>' +
        'tokens = [\'▁你\', \'好\', \'▁世界\']<br>' +
        'decode([1, 2, 3]) → <span class="hl5">\' 你好 世界\'</span>' }) }));
    rt.appendChild(W.card({ cc: 2, title: '转换器的分发', style: 'flex:1',
      body: U.el('div', { class: 'mono', style: 'font-size:11.5px;line-height:1.8', html:
        'SLOW_TO_FAST_CONVERTERS[类名]<br>' +
        '<span class="dim">56 条 · BertConverter 复用 12 次<br>查不到 → 落到 tiktoken / tekken 分支</span>' }) }));
    viz.appendChild(rt);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    els.forEach(e => { e.style.opacity = '.3'; });
    msg.innerHTML = '<span class="cm">// 从 .model（protobuf）里抽 vocab / merges / charsmap</span>';
    tl.at(2200, () => { els[0].style.opacity = '1';
      msg.innerHTML = '模型：<em>vocab 的数据结构</em>决定算法 —— dict 走 BPE，list[tuple] 走 Unigram'; });
    tl.at(5600, () => { els[1].style.opacity = '1';
      msg.innerHTML = '归一化：<em>Replace(" ", "▁")</em> —— 空格在编码前就换成了 ▁'; });
    tl.at(8800, () => { els[2].style.opacity = '1'; els[2].classList.add('ac');
      msg.innerHTML = '★ 实测订正：这个组装函数<em>不装 pre_tokenizer</em>（实测为 NoneType），'
        + '完整的 <span class="cm">converted() 里才用 Metaspace 补上</span>'; });
    tl.at(12200, () => { els[2].classList.remove('ac'); els[3].style.opacity = '1';
      msg.innerHTML = '解码：<em>Replace("▁", " ")</em> + ByteFallback + Fuse —— 与编码侧严格对称'; });
    tl.at(14600, () => {
      msg.innerHTML = '<span class="cm">// 三个组件成对才解得回去；GLM-5 没有 .model，所以从不走这条</span>';
    });
  },
},

/* ------------------------------------------------- 8 ★ 视觉/文本对齐 */
{
  kicker: '阶段 ⑦ · 对齐',
  title: '★ 视觉 token 与文本 token 靠 <span class="hl-a">id 的位置</span>对齐',
  sub: '视觉 token 数由图像处理器决定；文本侧只负责放同样多个占位 id，对齐靠 id 出现的位置。',
  caption: '这是 L2 整层的核心形状变换：分辨率的自由度被 merge_size² 除掉，剩下的才是序列长度。',
  lang: 'python',
  codeStart: 163,
  code: `    def extract(self, model_type, **kwargs) -> tuple[dict[str, int], list[tuple]]:
        """
        By default will return vocab and merges with respect to their order, by sending \`vocab_scores\` we're going to
        order the merges with respect to the piece scores instead.
        """
        self.proto.trainer_spec.unk_id
        if model_type is None:
            from tokenizers.models import BPE, Unigram

            model_type = Unigram if self.proto.trainer_spec.model_type == 1 else BPE
        vocab = [(piece.piece, piece.score) for piece in self.proto.pieces]

        if model_type.__name__ != "BPE":
            kwargs["unk_id"] = self.proto.trainer_spec.unk_id
            kwargs["vocab"] = vocab
        else:
            from .tokenization_utils_base import generate_merges

            vocab = {word: i for i, (word, score) in enumerate(vocab)}
            merges = generate_merges(vocab)
            kwargs["vocab"] = vocab
            kwargs["merges"] = merges

        # control tokens are special
        # user defined symbols are not
        # both user and control tokens are AddedTokens
        # Add user defined symbols (type == 4) from sentencepiece (https://github.com/google/sentencepiece/blob/6225e08edb2577757163b3f5dbba4c0b670ef445/src/sentencepiece_model.proto#L299C29-L299C33)
        spm_added_tokens = [(id, p.piece, p.type == 3) for id, p in enumerate(self.proto.pieces) if p.type in [3, 4]]
        kwargs["additional_special_tokens"] = [
            AddedToken(token, normalized=False, special=special)
            for id, token, special in sorted(spm_added_tokens, key=lambda x: x[0])
        ]
        kwargs["_spm_precompiled_charsmap"] = getattr(self.proto.normalizer_spec, "precompiled_charsmap", None)
        return kwargs`,
  codeNote: 'type 3/4 的 piece 变成 AddedToken(normalized=False)：这是占位 id 不被切碎的前提。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap10 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '图像侧：分辨率的自由度被 merge_size² 除掉（实测）' }));
    const img = W.flow([
      { t: '448 × 448', s: 'pixel_values (1024, 1176)', cc: 0 },
      { t: 'grid_thw [1,32,32]', s: 'patch 14 · merge 2', cc: 2 },
      { t: '1024 / 4', s: 'merge_size²', cc: 5 },
      { t: '256 个视觉 token', s: '视觉塔输出', cc: 4 },
    ], { style: 'width:100%' });
    viz.appendChild(img);

    viz.appendChild(U.el('div', { class: 'klabel', text: '文本侧：同一串 input_ids 上放 256 个占位（实测总长 264）' }));
    const segs = [
      { w: 1, t: '1', n: '', cc: 2, tip: '154830 <|begin_of_image|>' },
      { w: 256, t: '256 × <|image|>', n: '视觉占位', cc: 4, tip: '154854 重复 256 次' },
      { w: 1, t: '1', n: '', cc: 2, tip: '154831 <|end_of_image|>' },
      { w: 6, t: '6 个文本 token', n: '"你好，描述这张图。"', cc: 0, tip: '6 个文本 token' },
    ];
    const bar = U.el('div', { class: 'row gap4', style: 'width:100%;height:46px' });
    const segEls = segs.map(s => {
      const e = U.el('div', { title: s.tip, style: 'flex:' + s.w + ' 1 0;min-width:34px;height:46px;'
        + 'border-radius:8px;display:flex;flex-direction:column;align-items:center;justify-content:center;'
        + 'overflow:hidden;border:1px solid var(--c' + s.cc + ');background:rgba(150,180,255,.07);'
        + 'transition:background .35s,box-shadow .35s',
        html: '<span class="mono" style="font-size:11px;color:#fff;white-space:nowrap">' + U.esc(s.t) + '</span>'
            + (s.n ? '<span class="mono" style="font-size:9.5px;color:var(--ink-faint);white-space:nowrap">'
              + U.esc(s.n) + '</span>' : '') });
      bar.appendChild(e); return e;
    });
    viz.appendChild(bar);
    viz.appendChild(U.el('div', { class: 'row gap10 wrap', style: 'width:100%', html:
      U.chip('154830 begin_of_image', 2) + U.chip('154854 &lt;|image|&gt; × 256', 4)
      + U.chip('154831 end_of_image', 2) + U.chip('6 个文本 token', 0)
      + '<span class="dim mono" style="font-size:10.5px;margin-left:auto">段宽 = token 数占比</span>' }));

    const shapes = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    shapes.appendChild(W.card({ cc: 4, title: '序列账', style: 'flex:1', body: U.el('div', {
      class: 'mono', style: 'font-size:11.5px;line-height:1.8', html:
      '1 + 256 + 1 + 6 = <span class="hl4">264</span><br>视觉槽位 = [1..256]<br>' +
      '<span class="dim">224×224 的图 → 64 个占位 → 总长 72</span>' }) }));
    shapes.appendChild(W.card({ cc: 1, title: '两条样本 batch', style: 'flex:1', body: U.el('div', {
      class: 'mono', style: 'font-size:11.5px;line-height:1.8', html:
      'input_ids (2, 262)<br>attention_mask (2, 262)<br>每行长度 [262, 4]（左侧补 154820）' }) }));
    shapes.appendChild(W.card({ cc: 2, title: '合流之后', style: 'flex:1', body: U.el('div', {
      class: 'mono', style: 'font-size:11.5px;line-height:1.8', html:
      'inputs_embeds (2, 262, 4096)<br>视觉槽 256 + 文本槽 268<br>' +
      '<span class="dim">主干再也分不出哪一列是图</span>' }) }));
    viz.appendChild(shapes);

    viz.appendChild(W.exercise(
      '换成 672×336 的图（<code class="inl">grid_thw = [[1, 24, 48]]</code>），视觉 token 数和序列总长各是多少？',
      '1152 / 4 = <b>288</b> 个视觉 token；序列总长 = 1 + 288 + 1 + 6 = <b>296</b>。<br>'
      + '注意：<b>图像变大不会让词表变大</b>，只会让占位 id 重复更多次 —— 这就是「同一串位置上的对齐」的代价模型。'));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    segEls.forEach(e => { e.style.opacity = '.3'; });
    msg.innerHTML = '<span class="cm">// 图像先被切成 patch，再被 merge 成视觉 token</span>';
    tl.at(2200, () => { img.focus(0); segEls[1].style.opacity = '1';
      msg.innerHTML = '448×448 → <em>1024 个 patch</em>；文本侧此刻还什么都不知道'; });
    tl.at(5400, () => { img.focus(2);
      msg.innerHTML = 'merge_size = 2 → 每 <em>2×2 个 patch</em> 合成一个视觉 token：1024 / 4 = <em>'
        + M.vision['448x448'] + '</em>'; });
    tl.at(8600, () => { img.focus(3); segEls[0].style.opacity = '1'; segEls[2].style.opacity = '1'; segEls[3].style.opacity = '1';
      msg.innerHTML = '文本侧放 <em>256 个 &lt;|image|&gt;</em>，两头各一个控制符，再接 6 个文本 token'; });
    tl.at(12000, () => {
      segEls.forEach((e, i) => { if (i === 1) { e.style.background = 'rgba(52,211,153,.22)';
        e.style.boxShadow = '0 0 0 1px rgba(52,211,153,.7)'; } });
      msg.innerHTML = '模型只看 <em>input_ids == 154854</em> 的位置，把 256 个视觉向量按顺序填进去';
    });
    tl.at(15400, () => {
      msg.innerHTML = '<span class="cm">// 对齐的保证是「占位 id 不被切碎」—— 这正是 Trie / AddedToken 存在的理由</span>';
    });
  },
},

/* ------------------------------------------------- 9 收束 */
{
  kicker: '收束',
  title: '三条路，一张表',
  sub: 'GLM-5 的日常路径只有一次 encode_batch；另外两条是退路与对照物。',
  caption: '下一课 L2-03：SentencePiece 后端 —— unigram / BPE 与 add_dummy_prefix。',
  lang: 'python',
  codeStart: 1062,
  code: `        # Direct rust backend call
        encodings = self._tokenizer.encode_batch(
            batch_text_or_text_pairs,
            add_special_tokens=add_special_tokens,
            is_pretokenized=is_split_into_words,
        )`,
  codeNote: '回到开头：GLM-5 的文本→id，日常就是这 6 行。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const tb = W.table([
      ['fast · TokenizersBackend', 'tokenizer.json 存在', 'Rust（encode_batch）', '<span class="hl4">GLM-5 走这条</span>'],
      ['slow · PythonBackend', '没有 tokenizers 二进制 / 需要改规则', 'Python（逐条 for）', 'GLM-5 的 checkpoint 里没有原料'],
      ['转换器 · convert_slow_tokenizer', '只有 SentencePiece 的 .model', '一次性翻译成 Rust 组件', 'GLM-5 不经过'],
    ], { head: ['后端', '什么时候用', '循环在哪', 'GLM-5'] });
    const tbCard = U.el('div', { class: 'card cc0', style: 'padding:9px 11px' }, [tb]);
    wrap.appendChild(U.el('div', { class: 'col gap6', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: '三条路的分工' }), tbCard,
    ]));
    const rows = Array.from(tb.querySelectorAll('tbody tr'));

    const nums = W.table([
      ['vocab_size / len(tok)', '<span class="hl">154820 / 154856</span>', 'BPE，backend = tokenizers'],
      ['pad = eos', '<span class="hl">154820</span>', '"<|endoftext|>"，padding_side = left'],
      ['image_token_id', '<span class="hl">154854</span>', '"<|image|>"，占位 id'],
      ['448×448 的图', '<span class="hl">256</span>', '1024 个 patch / merge_size²'],
      ['带图序列', '<span class="hl">(2, 262)</span>', 'inputs_embeds (2, 262, 4096)'],
    ], { head: ['量', '实测值', '说明'] });
    wrap.appendChild(U.el('div', { class: 'card cc2', style: 'padding:9px 11px' }, [
      U.el('div', { class: 'klabel', text: '本课用到的实测数字（probe_l202.py）' }), nums,
    ]));

    wrap.appendChild(W.exercise(
      '为什么在 GLM-5 的 checkpoint 目录上 <code class="inl">PythonBackend.from_pretrained()</code> 会失败？',
      '因为纯 Python 后端需要一个可读的词表（<code class="inl">vocab.json</code> / <code class="inl">merges.txt</code> / '
      + '<code class="inl">tokenizer.model</code>），而这份 checkpoint 里<strong>只有 <code class="inl">tokenizer.json</code></strong>。'
      + '实测结果是 <code class="inl">NotImplementedError</code>，'
      + '<code class="inl">can_save_slow_tokenizer = False</code>。<br>'
      + '结论：<b>「fast / slow」不是两种可选精度，而是两种后端实现</b>；GLM-5 只发布了 fast 的那一份原料。'));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    rows.forEach(r => { r.style.opacity = '.3'; });
    msg.innerHTML = '<span class="cm">// 三条路逐个点亮</span>';
    rows.forEach((r, i) => {
      tl.at(1600 + i * 2400, () => {
        rows.forEach((x, k) => { x.style.opacity = (k === i) ? '1' : '.3'; });
        msg.innerHTML = ['fast：<em>GLM-5 走这条</em>，形状由一次 encode_batch 定',
          'slow：<em>退路</em>，Python 循环逐条处理，offsets 这类账本不给',
          '转换器：<em>只有 .model 时</em>的翻译层，56 个类按类名分发'][i];
      });
    });
    tl.at(9200, () => {
      rows.forEach(x => { x.style.opacity = '1'; });
      msg.innerHTML = '文本→id = 一次 <em>encode_batch</em> 定 (B, L)；视觉→对齐 = <em>占位 id 的位置</em>';
    });
    tl.at(12600, () => {
      msg.innerHTML = '<span class="cm">// 下一课 L2-03：.model 里面的 unigram / BPE 与 add_dummy_prefix</span>';
    });
    tl.at(15600, () => {
      msg.innerHTML = '层 L2 到此收束：文本与像素已经变成同一串位置上的向量';
    });
  },
},

];
