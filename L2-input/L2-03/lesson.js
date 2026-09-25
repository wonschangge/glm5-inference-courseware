/* ==========================================================================
   L2-03 · SentencePiece 分词器
   --------------------------------------------------------------------------
   覆盖：tokenization_utils_sentencepiece.py（1 个文件 / 315 行）
   目标：看完能说清一份 tokenizer.model 怎么变成 ids，以及 legacy 开关为什么
         会让「首 token」和「序列长度」都变。
   排版基准：#visual 可视区 = 994 x 662.7 舞台像素（无头浏览器实测）。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------- 1 全景 */
{
  kicker: 'L2 · 输入流水线',
  title: 'SentencePiece：<span class="hl-a">慢分词器</span>的共同父类',
  sub: '一份 tokenizer.model 怎么变成一串 id。这一课只看这一件事，以及它为什么决定了「首 token 长什么样」。',
  caption: '回顾 L0-01：ids 之后就是 embed_tokens → (b, s, 4096)。这一课补上 ids 是怎么来的。',
  lang: 'python',
  codeStart: 39,
  code: `VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}

SPIECE_UNDERLINE = "▁"


@add_end_docstrings(INIT_TOKENIZER_DOCSTRING)
class SentencePieceBackend(PreTrainedTokenizer):`,
  codeNote: 'SentencePieceBackend —— 315 行里唯一被 import 出去的名字。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    /* ---- 主链路 ---- */
    const nodes = [
      { t: 'tokenizer.model', s: 'sentencepiece proto', cc: 2 },
      { t: 'SentencePieceProcessor', s: 'Load() / encode()', cc: 0 },
      { t: '_tokenize', s: 'legacy 分支', cc: 1 },
      { t: 'input_ids', s: '(1, seq)', cc: 4 },
      { t: 'inputs_embeds', s: '(1, seq, 4096)', cc: 5 },
    ];
    const fl = W.flow(nodes, { style: 'width:100%' });
    viz.appendChild(U.el('div', { class: 'col gap6', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: '本课只覆盖这条链的中间一段' }), fl,
    ]));

    /* ---- 四个数字 ---- */
    const stats = [
      { k: '覆盖文件', v: '1', s: '本课只读这一个文件', cc: 0 },
      { k: '行数（实测）', v: '315', s: 'wc -l，不是 316', cc: 2 },
      { k: '类', v: '2', s: 'Backend + Extractor', cc: 1 },
      { k: '直接子类（实测）', v: '6', s: '全 models/ 目录数出来', cc: 4 },
    ];
    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const statEls = stats.map(st => {
      const c = W.card({ cc: st.cc, tint: st.cc,
        title: '<span class="mono" style="font-size:11px">' + st.k + '</span>',
        body: U.el('div', { class: 'n big', text: st.v }),
        sub: st.s, style: 'flex:1 1 0' });
      row.appendChild(c); return c;
    });
    viz.appendChild(row);

    /* ---- 谁走这个基类 ---- */
    const who = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cA = W.card({ cc: 0, tint: 0, title: '这个基类现在是给谁的',
      sub: 'BertGeneration / GPTSw3 / Siglip / SpeechT5 / Bartpho / PLBart —— 实测就这 6 个直接子类。',
      style: 'flex:1 1 0' });
    const cB = W.card({ cc: 3, tint: 3, title: 'GLM-5 实测走的是',
      sub: 'TokenizersBackend（backend="tokenizers"，is_fast=True）。作业书那条「GLM-5 往往走这个基类」与实测相反，以实测为准。',
      style: 'flex:1 1 0' });
    who.appendChild(cA); who.appendChild(cB);
    viz.appendChild(who);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    /* ---- 时间轴 ---- */
    fl.focus(0);
    msg.innerHTML = '<span class="cm">// 一个字符串的旅程从这里开始</span>';
    tl.at(1800, () => { fl.focus(1); msg.innerHTML = 'proto 装进 <em>SentencePieceProcessor</em> <span class="cm">// unigram 与 BPE 是同一段代码</span>'; });
    tl.at(4600, () => { fl.focus(2); msg.innerHTML = '<em>_tokenize</em>：<span class="cm">// legacy 决定要不要走那条补词边界的岔路</span>'; });
    tl.at(7400, () => { fl.focus(3); msg.innerHTML = '文本 → <em>input_ids</em> <span class="op">→</span> 形状 <em>(1, seq)</em>'; });
    tl.at(10200, () => { fl.focus(4); msg.innerHTML = '与视觉 token 拼在同一条 seq 上 <span class="op">→</span> <em>(1, seq, 4096)</em>'; });
    tl.at(13000, () => {
      fl.focus(-1); statEls.forEach(e => e.classList.add('ac'));
      msg.innerHTML = '<span class="cm">// 315 行，2 个类，13 个引用块</span>';
    });
    tl.at(15200, () => {
      statEls.forEach(e => e.classList.remove('ac'));
      cB.classList.add('ac');
      msg.innerHTML = '<span class="hlbad">实测订正</span>：GLM-5 的 tokenizer 是 <em>TokenizersBackend</em>，不走这个基类';
    });
  },
},

/* ------------------------------------------------- 2 构造顺序 */
{
  kicker: '构造 · 顺序',
  title: '先把 <span class="hl-a">sp_model</span> 装好，再让父类上桌',
  sub: '与常见的「先 super() 再补自己的字段」相反：这个类必须在父类初始化之前把 sentencepiece 模型加载完。',
  caption: '父类初始化时会顺着 _add_tokens → len(self) → get_vocab() → sp_model 摸到它。',
  lang: 'python',
  codeStart: 58,
  code: `    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(self, **kwargs):
        # Ensure optional dependency is available before loading
        requires_backends(self, "sentencepiece")

        # Extract sentencepiece-specific parameters
        self.vocab_file = kwargs.get("vocab_file")
        self.legacy = kwargs.get("legacy", True)
        self.sp_model_kwargs = kwargs.pop("sp_model_kwargs", {})`,
  codeNote: 'get 与 pop 的区别，决定了哪个参数能活到 save_pretrained。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '构造函数的六个动作（顺序不能换）' }));
    const steps = [
      { t: '① 依赖检查', s: 'requires_backends(self, "sentencepiece")', cc: 3 },
      { t: '② 取三个参数', s: 'vocab_file / legacy / sp_model_kwargs', cc: 0 },
      { t: '③ 补 backend', s: '"sentencepiece"（用户传了就不覆盖）', cc: 0 },
      { t: '④ Load 模型', s: '在 super() 之前把 .model 装进 processor', cc: 4 },
      { t: '⑤ legacy 改写', s: 'not legacy → 改 proto 的 add_dummy_prefix', cc: 1 },
      { t: '⑥ 交给父类', s: 'super().__init__() → _update_trie()', cc: 2 },
    ];
    const rowEls = [];
    const row = U.el('div', { class: 'row gap10', style: 'width:100%;align-items:stretch' });
    steps.forEach(st => {
      const c = W.card({ cc: st.cc, tint: st.cc, title: st.t, sub: st.s, style: 'flex:1 1 0' });
      row.appendChild(c); rowEls.push(c);
    });
    viz.appendChild(row);

    /* ---- 父类为什么会摸到 sp_model ---- */
    const chain = U.el('div', { class: 'card cc4', style: 'width:100%;padding:10px 12px' });
    chain.appendChild(U.el('div', { class: 'klabel', text: '父类初始化里的调用链 —— 这就是「顺序不能反」的原因' }));
    chain.appendChild(U.el('div', { class: 'mono', style: 'font-size:11.5px;color:var(--ink-dim);line-height:1.9' },
      [U.el('div', { html: 'super().__init__() &nbsp;<span class="op">→</span>&nbsp; _add_tokens() &nbsp;<span class="op">→</span>&nbsp; <span class="hl">len(self)</span> &nbsp;<span class="op">→</span>&nbsp; get_vocab() &nbsp;<span class="op">→</span>&nbsp; convert_ids_to_tokens() &nbsp;<span class="op">→</span>&nbsp; <span class="hl4">sp_model</span>' })]));
    viz.appendChild(chain);

    /* ---- 两套字段的归属 ---- */
    const pair = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    pair.appendChild(W.card({ cc: 0, title: 'kwargs.get("vocab_file")',
      sub: '留在 kwargs 里交给父类 → 进 init_kwargs → 被写回 tokenizer_config.json。',
      style: 'flex:1 1 0' }));
    pair.appendChild(W.card({ cc: 2, title: 'kwargs.pop("sp_model_kwargs")',
      sub: '取走给 SentencePieceProcessor 用，第 91 行再手动塞回去 —— 不重复也不丢。',
      style: 'flex:1 1 0' }));
    viz.appendChild(pair);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 六个动作，④ 必须在 ⑥ 之前</span>';
    const focus = (k) => rowEls.forEach((e, i) => { e.classList.toggle('ac', i === k); e.classList.toggle('dimmed', k >= 0 && i !== k); });
    focus(-1);
    tl.at(1800, () => { focus(0); msg.innerHTML = '① 可选依赖没装 → <em>明确报错</em>，而不是后面莫名其妙的 AttributeError'; });
    tl.at(4400, () => { focus(1); msg.innerHTML = '② <em>legacy 默认 True</em> —— 这是第三幕的主角'; });
    tl.at(7000, () => { focus(2); msg.innerHTML = '③ <span class="mono">backend="sentencepiece"</span>：v5 用它标记走哪条实现路径'; });
    tl.at(9800, () => { focus(3); msg.innerHTML = '④ <em>Load</em> 必须在父类之前 —— 父类一上来就会顺着右边那条链摸到 sp_model'; });
    tl.at(12600, () => { focus(4); msg.innerHTML = '⑤ legacy=False 时顺手改写 proto（下一幕）'; });
    tl.at(15000, () => { focus(-1); msg.innerHTML = '<span class="cm">// ⑥ 父类上桌时，料已经备齐了</span>'; });
  },
},

/* ------------------------------------------------- 3 ★ legacy 改写 proto */
{
  kicker: '★ 洞察 1 · legacy',
  title: '★ legacy 改的不是代码，是<span class="hl-b">那份 proto</span>',
  sub: 'add_dummy_prefix 是归一化器的行为，没有运行时开关 —— 只能把模型反序列化、改一个布尔、再装回去。',
  caption: '同一份 .model，legacy=True 与 legacy=False 会加载出两个行为不同的对象。',
  lang: 'python',
  codeStart: 78,
  code: `        if not self.legacy:
            model_pb2 = import_protobuf()
            proto = model_pb2.ModelProto.FromString(tokenizer.serialized_model_proto())
            if proto.normalizer_spec.add_dummy_prefix:
                proto.normalizer_spec.add_dummy_prefix = False
                tokenizer.LoadFromSerializedProto(proto.SerializeToString())

        self.sp_model = tokenizer`,
  codeNote: '4 行完成「反序列化 → 改一个布尔 → 装回去」。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    /* ---- 流程 ---- */
    const nodes = [
      { t: 'serialized_model_proto()', s: '把已加载的模型导出成 proto', cc: 0 },
      { t: 'ModelProto.FromString', s: '反序列化', cc: 0 },
      { t: 'add_dummy_prefix = False', s: '唯一的改动', cc: 1 },
      { t: 'LoadFromSerializedProto', s: '装回去，行为永久改变', cc: 2 },
    ];
    const fl = W.flow(nodes, { style: 'width:100%' });
    viz.appendChild(U.el('div', { class: 'col gap6', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: '四步：从已加载的模型里把 proto 取出来再装回去' }), fl,
    ]));

    /* ---- 那个布尔 ---- */
    const flag = W.card({ cc: 1, tint: 1, title: 'normalizer_spec.add_dummy_prefix',
      style: 'width:100%' });
    const frow = U.el('div', { class: 'row gap12 center', style: 'width:100%;margin-top:6px' });
    const v1 = U.el('div', { class: 'n big', style: 'color:var(--c4);transition:opacity .5s,transform .5s', text: '1' });
    const arrow = U.el('div', { class: 'mono', style: 'font-size:18px;color:var(--ink-faint)', text: '⟹' });
    const v0 = U.el('div', { class: 'n big', style: 'color:var(--c3);opacity:.25;transition:opacity .5s,transform .5s', text: '0' });
    frow.appendChild(U.el('div', { class: 'col center gap4' }, [U.el('div', { class: 'klabel', text: '训练产物' }), v1]));
    frow.appendChild(arrow);
    frow.appendChild(U.el('div', { class: 'col center gap4' }, [U.el('div', { class: 'klabel', text: 'legacy=False 之后' }), v0]));
    frow.appendChild(U.el('div', { class: 'mono dim', style: 'font-size:11px;margin-left:14px', text: '实测：读回来确实是 0' }));
    flag.appendChild(frow);
    viz.appendChild(flag);

    /* ---- unigram / BPE ---- */
    const pair = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    pair.appendChild(W.card({ cc: 4, tint: 4, title: 'unigram 模型',
      sub: 'proto 里 trainer_spec.model_type = 1。加载它的 Python 代码与 BPE 完全同一段。',
      style: 'flex:1 1 0' }));
    pair.appendChild(W.card({ cc: 5, tint: 5, title: 'BPE 模型',
      sub: 'proto 里 trainer_spec.model_type = 2。算法写在 proto 里，不在这个文件里。',
      style: 'flex:1 1 0' }));
    viz.appendChild(pair);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// if not self.legacy:</span>';
    tl.at(1800, () => { fl.focus(0); msg.innerHTML = '把已经加载的模型<em>导出</em>成 proto <span class="cm">// 不是重新读文件</span>'; });
    tl.at(4400, () => { fl.focus(1); msg.innerHTML = '<em>ModelProto.FromString</em> 反序列化 —— 所以需要 <span class="mono">import_protobuf()</span>'; });
    tl.at(7200, () => {
      fl.focus(2);
      v1.style.opacity = '.25'; v1.style.transform = 'translateX(-6px)';
      v0.style.opacity = '1'; v0.style.transform = 'scale(1.25)';
      msg.innerHTML = '改的只有 <em>一个布尔</em>：<span class="hl">add_dummy_prefix</span> 1 → 0';
    });
    tl.at(10000, () => { fl.focus(3); msg.innerHTML = '<em>LoadFromSerializedProto</em> 装回去 —— 这个实例的行为从此永久改变'; });
    tl.at(12800, () => { fl.focus(-1); msg.innerHTML = '<span class="cm">// 为什么不能运行时切？add_dummy_prefix 属于归一化器，没有运行时开关</span>'; });
    tl.at(14800, () => { msg.innerHTML = '<em>unigram</em> 与 <em>BPE</em> 走的是同一段代码 —— 分支在 proto 的 trainer_spec 里'; });
  },
},

/* ------------------------------------------------- 4 _tokenize 的两个分支 */
{
  kicker: '运行时 · _tokenize',
  title: '关掉开关之后，<span class="hl-a">词边界要自己补回来</span>',
  sub: 'legacy=False 时 sentencepiece 会把开头的 ▁ 吃掉；这个类的对策是「先把 unk_token 拼在前面，切完再切掉它」。',
  caption: '实测：encode(" Hello") → [H, el, lo]；encode("&lt;unk&gt; Hello")[5:] → [▁Hello]。',
  lang: 'python',
  codeStart: 214,
  code: `        if self.legacy or not text.startswith((SPIECE_UNDERLINE, " ")):
            return self.sp_model.encode(text, out_type=str)

        # 1. Encode string + prefix ex: "<unk> Hey"
        tokens = self.sp_model.encode(self.unk_token + text, out_type=str)
        # 2. Remove self.unk_token from ['<','unk','>', '▁Hey']
        unk_token_length = len(self.sp_model.encode(str(self.unk_token)))
        return tokens[unk_token_length:] if len(tokens) >= unk_token_length else tokens`,
  codeNote: '两个分支 + 一个必须运行时算出来的 unk_token_length。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    /* ---- 判断条件 ---- */
    const cond = U.el('div', { class: 'card cc0', style: 'width:100%;padding:10px 12px' });
    cond.appendChild(U.el('div', { class: 'klabel', text: '唯一的判断（第 214 行）' }));
    cond.appendChild(U.el('div', { class: 'mono', style: 'font-size:12.5px;color:#cfe6ff', html: 'if self.<span class="hl">legacy</span> or not text.startswith((<span class="hl3">SPIECE_UNDERLINE</span>, <span class="hl3">" "</span>)):' }));
    cond.appendChild(U.el('div', { class: 'cs', style: 'margin-top:6px', text: '只有「非 legacy 且首字符是空格或 ▁」才走特殊路径。普通文本老老实实 encode。' }));
    viz.appendChild(cond);

    /* ---- 两条路 ---- */
    const lr = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const pA = W.card({ cc: 4, tint: 4, title: '分支 A：直接 encode',
      body: U.el('div', { class: 'mono', style: 'font-size:11.5px;line-height:1.85' }, [
        U.el('div', { html: 'encode("Hello") &nbsp;<span class="dim">legacy=False</span>' }),
        U.el('div', { class: 'dim', html: '↓' }),
        U.el('div', { html: '<span class="hlbad">[\'H\', \'el\', \'lo\']</span> &nbsp;<span class="dim">首字符没有 ▁</span>' }),
      ]), style: 'flex:1 1 0' });
    const pB = W.card({ cc: 1, tint: 1, title: '分支 B：先拼 unk_token',
      body: U.el('div', { class: 'mono', style: 'font-size:11.5px;line-height:1.85' }, [
        U.el('div', { html: 'encode("&lt;unk&gt; Hello")' }),
        U.el('div', { class: 'dim', html: '↓ 切掉前 5 个 piece' }),
        U.el('div', { html: '<span class="hl4">[\'▁Hello\']</span> &nbsp;<span class="dim">词边界救回来了</span>' }),
      ]), style: 'flex:1 1 0' });
    lr.appendChild(pA); lr.appendChild(pB);
    viz.appendChild(lr);

    /* ---- unk_token_length ---- */
    const unk = W.card({ cc: 2, tint: 2, title: 'unk_token_length 必须现算', style: 'width:100%' });
    unk.appendChild(U.el('div', { class: 'mono', style: 'font-size:11.5px;line-height:1.8;margin-top:4px' }, [
      U.el('div', { html: 'len(sp_model.encode("&lt;unk&gt;")) = <span class="hl3">5</span> &nbsp;<span class="dim">// 本课模型实测</span>' }),
      U.el('div', { html: 'pieces = [\'<\', \'u\', \'n\', \'k\', \'>\'] &nbsp;<span class="dim">// docstring 举的例子是 4</span>' }),
    ]));
    viz.appendChild(unk);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    pA.style.opacity = '.3'; pB.style.opacity = '.3'; unk.style.opacity = '.3';
    msg.innerHTML = '<span class="cm">// 两个分支，只为一个字符服务</span>';
    tl.at(2000, () => { msg.innerHTML = '判断条件：<em>legacy</em> 或「首字符不是空格/▁」→ 直接走分支 A'; });
    tl.at(5000, () => { pA.style.opacity = '1'; msg.innerHTML = '分支 A：<em>"Hello"</em> → 裸 <span class="hlbad">H</span> + el + lo <span class="cm">// 词边界丢了</span>'; });
    tl.at(8200, () => { pB.style.opacity = '1'; msg.innerHTML = '分支 B：把开头变成「不是开头」—— <em>"&lt;unk&gt; Hello"</em>，▁ 就不会被吃掉'; });
    tl.at(11400, () => { unk.style.opacity = '1'; msg.innerHTML = '<em>unk_token_length</em> = 5：写死成 1 或 4 都会切错位置'; });
    tl.at(14400, () => { msg.innerHTML = '<span class="cm">// docstring 那句「internals will always strip any SPIECE_UNDERLINE」我实测过，是真的</span>'; });
    tl.at(16600, () => { msg.innerHTML = '顺带：<em>_tokenize 看不到特殊 token</em> —— 父类先用 trie 把它们切走了（第七幕）'; });
  },
},

/* ------------------------------------------------- 5 ★ 首 token 实测 */
{
  kicker: '★ 验收点 · 首 token',
  title: '★ 首 token 是 <span class="hl-c">▁Hello</span> 还是 <span class="hlbad">H</span>',
  sub: 'docstring 里的断言，我用两个真模型跑了一遍：关掉 add_dummy_prefix，首 token 从「带词边界的整词」退化成裸字符。',
  caption: '这不是格式差异，是 token 数差异 —— input_ids 的形状会跟着变。',
  lang: 'python',
  codeStart: 208,
  code: `        We de-activated the \`add_dummy_prefix\` option, thus the sentencepiece internals will always strip any
        SPIECE_UNDERLINE. For example: \`self.sp_model.encode(f"{SPIECE_UNDERLINE}Hey", out_type = str)\` will give
        \`['H', 'e', 'y']\` instead of \`['▁He', 'y']\`. Thus we always encode \`f"{unk_token}text"\` and strip the
        \`unk_token\`. Here is an example with \`unk_token = "<unk>"\` and \`unk_token_length = 4\`.
        \`self.tokenizer.sp_model.encode("<unk> Hey", out_type = str)[4:]\`.`,
  codeNote: 'docstring 是断言，不是证据 —— 这一课把它跑成了下面这张表。',
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    /* ---- 首 token 本体 ---- */
    viz.appendChild(U.el('div', { class: 'klabel', text: '同一个模型，只换一个加载参数' }));
    const chipRow = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const mkTok = (label, cc, pre, core, note) => {
      const c = W.card({ cc: cc, tint: cc, title: label, style: 'flex:1 1 0' });
      const strip = U.el('div', { class: 'row gap4 center', style: 'width:100%;margin:6px 0 4px' });
      if (pre) strip.appendChild(U.el('div', { class: 'n', style: 'font-size:20px;color:var(--c2)', text: pre }));
      strip.appendChild(U.el('div', { class: 'n', style: 'font-size:20px;color:#fff', text: core }));
      c.appendChild(strip);
      c.appendChild(U.el('div', { class: 'cs', text: note }));
      return c;
    };
    const tA = mkTok('legacy=True 的首 token', 4, '▁', 'Hello', '前面那个 ▁ 是模型自己加的（add_dummy_prefix=1）');
    const tB = mkTok('legacy=False 的首 token', 3, '', 'H', '开关关掉后，首 token 是一个裸字符');
    chipRow.appendChild(tA); chipRow.appendChild(tB);
    viz.appendChild(chipRow);

    /* ---- 实测表 ---- */
    const tb = W.table([
      ['"Hello world"', '<span class="hl4">[\'▁Hello\', \'▁world\']</span>', '<span class="hlbad">[\'H\', \'el\', \'lo\', \'▁world\']</span>', '2 → 4'],
      ['"Hello"', '<span class="hl4">[\'▁Hello\']</span>', '<span class="hlbad">[\'H\', \'el\', \'lo\']</span>', '1 → 3'],
      ['" Hello"', '<span class="hl4">[\'▁Hello\']</span>', '<span class="hl4">[\'▁Hello\']</span>', '1 → 1'],
      ['BPE 模型 "Hello world"', '<span class="hl4">[\'▁Hello\', \'▁world\']</span>', '<span class="hlbad">[\'H\', \'ello\', \'▁world\']</span>', '2 → 3'],
    ], { head: ['输入（同一个模型）', 'legacy=True', 'legacy=False', '长度'] });
    const tcard = U.el('div', { class: 'card cc0', style: 'width:100%;padding:10px 12px' }, [tb]);
    viz.appendChild(tcard);

    /* ---- 形状后果 ---- */
    const shape = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const sA = W.card({ cc: 4, title: 'legacy=True 的 ids',
      body: U.el('div', { class: 'mono', style: 'font-size:11.5px', html: '[8, 12] &nbsp;<span class="dim">→ input_ids (1, 2)</span>' }),
      style: 'flex:1 1 0' });
    const sB = W.card({ cc: 3, title: 'legacy=False 的 ids',
      body: U.el('div', { class: 'mono', style: 'font-size:11.5px', html: '[93, 72, 74, 12] &nbsp;<span class="dim">→ input_ids (1, 4)</span>' }),
      style: 'flex:1 1 0' });
    shape.appendChild(sA); shape.appendChild(sB);
    viz.appendChild(shape);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    const rows = Array.from(tb.querySelectorAll('tbody tr'));
    rows.forEach(r => { r.style.transition = 'background .3s'; });
    const light = (k) => rows.forEach((r, i) => { r.classList.toggle('ac', i === k); });
    tA.classList.add('ac');
    msg.innerHTML = '<span class="cm">// 同一份 .model，只换一个加载参数</span>';
    tl.at(2000, () => { light(0); msg.innerHTML = '<em>"Hello world"</em>：2 个 token → <span class="hlbad">4 个 token</span>'; });
    tl.at(5000, () => { light(1); msg.innerHTML = '<em>"Hello"</em>：一个整词 → <span class="hlbad">三个裸字符</span>'; });
    tl.at(8000, () => { light(2); msg.innerHTML = '<em>" Hello"</em>：两条路一样 —— 因为分支 B 把词边界抢回来了'; });
    tl.at(11000, () => {
      light(3);
      tA.classList.remove('ac'); tB.classList.add('ac');
      msg.innerHTML = '换成 <em>BPE</em> 模型结论不变：<span class="hlbad">H</span> + ello <span class="cm">// 首 token 仍然是裸字符</span>';
    });
    tl.at(14000, () => { light(-1); msg.innerHTML = '首 token 的 id 从 <em>8</em> 变成 <em>93</em> —— embedding 表里查到的<b>不是同一行</b>'; });
    tl.at(16600, () => { msg.innerHTML = '<span class="cm">// 所以 add_dummy_prefix 影响的不只是「长什么样」，还有序列长度与 ids</span>'; });
  },
},

/* ------------------------------------------------- 6 词表两套口径 */
{
  kicker: '词表 · 两套口径',
  title: 'vocab_size 与 len()：<span class="hl-a">差在哪</span>',
  sub: '一个只数模型自带的 piece，一个是 base + added 的总数。后者才是 embedding 表该有多大。',
  caption: '回顾 L2-01：resize_token_embeddings(len(tokenizer)) 用的就是后者。',
  lang: 'python',
  codeStart: 99,
  code: `    @property
    def vocab_size(self) -> int:
        """Returns vocab size"""
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        """Returns vocab as a dict"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab`,
  codeNote: 'get_vocab() 的两行：先铺 base，再并 added。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    /* ---- 两个口径 ---- */
    viz.appendChild(U.el('div', { class: 'klabel', text: '同一个分词器上的两个数' }));
    const lr = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cV = W.card({ cc: 0, tint: 0, title: 'vocab_size（@property）',
      body: U.el('div', { class: 'mono', style: 'font-size:11.5px;line-height:1.8' }, [
        U.el('div', { html: 'sp_model.get_piece_size()' }),
        U.el('div', { html: '<span class="hl">99</span> &nbsp;<span class="dim">只数模型自带的 piece</span>' }),
      ]), style: 'flex:1 1 0' });
    const cL = W.card({ cc: 2, tint: 2, title: 'len(tokenizer)（父类 __len__）',
      body: U.el('div', { class: 'mono', style: 'font-size:11.5px;line-height:1.8' }, [
        U.el('div', { html: 'len(self.get_vocab())' }),
        U.el('div', { html: '<span class="hl3">99 → 100</span> &nbsp;<span class="dim">加了 1 个 token 之后</span>' }),
      ]), style: 'flex:1 1 0' });
    lr.appendChild(cV); lr.appendChild(cL);
    viz.appendChild(lr);

    /* ---- 堆叠条 ---- */
    const st = W.stack([
      { label: 'base 99', value: 99, cc: 0 },
      { label: 'added 1', value: 1, cc: 2 },
    ], { style: 'width:100%' });
    viz.appendChild(U.el('div', { class: 'col gap6', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: 'len(tokenizer) 的构成（实测：base 99 + added 1）' }), st,
    ]));

    /* ---- 实测表 ---- */
    const tb = W.table([
      ['vocab_size', '<span class="hl">99</span>', '只数 sentencepiece 自带的 piece'],
      ['len(tok) 加之前', '<span class="hl">99</span>', 'total_vocab_size = 99'],
      ['_add_tokens 返回', '<span class="hl3">1</span>', '真正新增了 1 个'],
      ['len(tok) 加之后', '<span class="hl3">100</span>', 'base + added'],
      ['vocab_size 加之后', '<span class="hl">99</span>', '<b>没变</b> —— 它不认识 added token'],
    ], { head: ['量', '实测值', '说明'] });
    viz.appendChild(U.el('div', { class: 'card cc2', style: 'width:100%;padding:10px 12px' }, [tb]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    st.reveal(0);
    cL.style.opacity = '.35';
    msg.innerHTML = '<span class="cm">// 两套口径，别混用</span>';
    tl.at(2000, () => { cV.classList.add('ac'); msg.innerHTML = '<em>vocab_size</em>：模型原始词表有多大 <span class="cm">// 99</span>'; });
    tl.at(5000, () => { cV.classList.remove('ac'); cL.style.opacity = '1'; cL.classList.add('ac'); st.reveal(1); msg.innerHTML = '<em>len(tok)</em>：base + added <span class="cm">// 99 + 1 = 100</span>'; });
    tl.at(8400, () => { msg.innerHTML = '父类 <em>__len__</em> 返回 total_vocab_size —— 这才是 embedding 表该有多大'; });
    tl.at(11400, () => { msg.innerHTML = '第 143 行 <span class="mono">next_index = len(self)</span>：新 token 从 <em>100</em> 开始编号'; });
    tl.at(14400, () => { msg.innerHTML = '<span class="cm">// 写成 vocab_size 的话，第二个新 token 会盖掉第一个的位置</span>'; });
  },
},

/* ------------------------------------------------- 7 ★ in_base_vocab */
{
  kicker: '★ 洞察 2 · 词表边界',
  title: '★ 为什么 <span class="hl-a">in_base_vocab</span> 要判两次',
  sub: 'piece_to_id 对不存在的 piece 返回 unk_id=0，而不是报错 —— 所以「id 比词表小」这半句几乎永远成立。',
  caption: '去掉后半句，所有新增 token 都会指向 &lt;unk&gt;，而且 num_added 永远是 0。',
  lang: 'python',
  codeStart: 165,
  code: `            # Check if token already exists in the SentencePiece base vocab
            tok_id = self.sp_model.piece_to_id(token.content)
            in_base_vocab = (
                tok_id < self.sp_model.get_piece_size() and self.sp_model.IdToPiece(tok_id) == token.content
            )

            if in_base_vocab:
                token_index = tok_id
            else:
                token_index = next_index
                next_index += 1
                num_added += 1`,
  codeNote: '两个条件，防的是「未知 piece 伪装成 id 0」。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    /* ---- 两个条件 ---- */
    const lr = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cA = W.card({ cc: 3, tint: 3, title: '条件 A：tok_id < get_piece_size()',
      body: U.el('div', { class: 'mono', style: 'font-size:11.5px;line-height:1.8' }, [
        U.el('div', { html: 'piece_to_id("definitely_not_a_piece")' }),
        U.el('div', { html: '= <span class="hlbad">0</span> = unk_id &nbsp;<span class="dim">// 0 &lt; 99，成立</span>' }),
      ]),
      sub: '几乎永远为真 —— 它挡不住任何东西。', style: 'flex:1 1 0' });
    const cB = W.card({ cc: 4, tint: 4, title: '条件 B：IdToPiece(tok_id) == content',
      body: U.el('div', { class: 'mono', style: 'font-size:11.5px;line-height:1.8' }, [
        U.el('div', { html: 'IdToPiece(0) = "&lt;unk&gt;" ≠ 原文' }),
        U.el('div', { html: 'IdToPiece(8) = "<span class="hl4">▁Hello</span>" = 原文 ✓' }),
      ]),
      sub: '真正干活的那半句：round-trip 一致才算「在词表里」。', style: 'flex:1 1 0' });
    lr.appendChild(cA); lr.appendChild(cB);
    viz.appendChild(lr);

    /* ---- 两种 token 的落位 ---- */
    const tb = W.table([
      ['<code class="inl">"▁Hello"</code>', '8', '"▁Hello"', '<span class="hl4">命中</span>', '8（不新增）'],
      ['<code class="inl">"&lt;|brand_new|&gt;"</code>', '0', '"&lt;unk&gt;"', '<span class="hlbad">不命中</span>', '99 → len 100'],
    ], { head: ['token', 'piece_to_id', 'IdToPiece 回来', 'in_base_vocab', '最终 id'] });
    const tcard = U.el('div', { class: 'card cc0', style: 'width:100%;padding:10px 12px' }, [tb]);
    viz.appendChild(tcard);

    /* ---- 去掉后半句的后果 ---- */
    const warn = W.card({ cc: 3, tint: 3, title: '如果只留条件 A',
      sub: '任何新 token 都会被判成「已在词表里」，token_index = 0 —— 所有新增 token 全部指向 &lt;unk&gt;，而 _add_tokens 永远返回 0。',
      style: 'width:100%' });
    viz.appendChild(warn);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    const rows = Array.from(tb.querySelectorAll('tbody tr'));
    rows.forEach(r => { r.style.transition = 'background .3s'; });
    cA.style.opacity = '.35'; cB.style.opacity = '.35'; warn.style.opacity = '.3';
    msg.innerHTML = '<span class="cm">// 一行看起来啰嗦的判断</span>';
    tl.at(2000, () => { cA.style.opacity = '1'; msg.innerHTML = '条件 A：<em>tok_id &lt; 词表大小</em> —— 未知 piece 拿到 0，照样通过'; });
    tl.at(5000, () => { cB.style.opacity = '1'; msg.innerHTML = '条件 B：<em>IdToPiece(tok_id) == content</em> —— 把编号换回字符串，确认一致'; });
    tl.at(8400, () => { rows[0].classList.add('ac'); msg.innerHTML = '实测：<em>"▁Hello"</em> 命中 → id <em>8</em>，<b>不占新编号</b>，_add_tokens 返回 0'; });
    tl.at(11400, () => { rows[0].classList.remove('ac'); rows[1].classList.add('ac'); msg.innerHTML = '实测：<em>"&lt;|brand_new|&gt;"</em> 不命中 → 新编号 <em>99</em>，_add_tokens 返回 1'; });
    tl.at(14400, () => { warn.style.opacity = '1'; msg.innerHTML = '<span class="hlbad">少了条件 B</span>：所有新增 token 都指向 &lt;unk&gt;，而且 num_added 永远是 0'; });
  },
},

/* ------------------------------------------------- 8 形状演算 */
{
  kicker: '本层重点 · 形状',
  title: '文本 token 与视觉 token：<span class="hl-a">同一条序列</span>上的对齐',
  sub: '448×448 的图 → 256 个视觉 token。它们和文本 token 共享同一条 seq 轴，只是那些位置上的向量换了来源。',
  caption: '256 个位置由 masked_scatter 写入视觉向量；其余位置走 embed_tokens 查表。',
  lang: 'python',
  codeStart: 223,
  code: `    def _convert_token_to_id(self, token):
        """Converts a token (str) to an id using the vocab."""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self.sp_model.IdToPiece(index)
        return token`,
  codeNote: '_convert_token_to_id / _convert_id_to_token —— 序列里每个位置都是一个 id。',
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const stage = U.el('div', { class: 'row gap16', style: 'width:100%;align-items:flex-start' });

    /* ---- 左：比例条 + 256 个视觉 token 的网格 ---- */
    const left = U.el('div', { class: 'col gap8', style: 'flex:1 1 0;min-width:0' });
    left.appendChild(U.el('div', { class: 'klabel', text: '一条真实的混合序列（实测 267 个位置）' }));
    const st = W.stack([
      { label: '文本 6', value: 6, cc: 0 },
      { label: '图像 258', value: 258, cc: 1 },
      { label: '文本 3', value: 3, cc: 0 },
    ], { style: 'width:100%' });
    left.appendChild(st);
    left.appendChild(U.el('div', { class: 'row gap6 wrap', style: 'width:100%' }, [
      U.el('div', { class: 'chip c0', html: '<i class="sw sw-c0"></i>文本 6（"你好，GLM-5"）' }),
      U.el('div', { class: 'chip c1', html: '<i class="sw sw-c1"></i>图像 258 = 1 + 256 + 1' }),
      U.el('div', { class: 'chip c0', html: '<i class="sw sw-c0"></i>文本 3（"描述这张图"）' }),
    ]));
    const grid = W.tensor(16, 16, () => -1, { cell: 15, gap: 2 });
    grid.querySelectorAll('.tcell').forEach(c => {
      c.style.background = 'rgba(167,139,250,.22)';
      c.style.borderColor = 'rgba(167,139,250,.38)';
      c.style.transition = 'opacity .45s var(--ease-out), transform .45s var(--ease-out)';
      c.style.opacity = '0'; c.style.transform = 'scale(.6)';
    });
    left.appendChild(U.el('div', { class: 'col gap4', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: '256 个视觉占位（16 x 16 示意）' }),
      U.el('div', { class: 'row center', style: 'width:100%' }, [grid]),
    ]));
    stage.appendChild(left);

    /* ---- 右：真实数字 + 形状 ---- */
    const right = U.el('div', { class: 'col gap8', style: 'flex:1 1 0;min-width:0' });
    right.appendChild(U.el('div', { class: 'klabel', text: '这些数字都是跑出来的' }));
    const tb = W.table([
      ['图像尺寸', '448 x 448', 'vision_config.image_size'],
      ['patch', '14', '448 / 14 = 32'],
      ['image_grid_thw', '[[1, 32, 32]]', '图像处理器实测输出'],
      ['pixel_values', '(1024, 1176)', '1024 = 32 x 32'],
      ['视觉 token', '<span class="hl2">256</span>', '1024 / merge_size² = 1024 / 4'],
    ], { head: ['量', '实测值', '算式'] });
    right.appendChild(U.el('div', { class: 'card cc1', style: 'width:100%;padding:9px 11px' }, [tb]));
    const sh = U.el('div', { class: 'card cc4', style: 'width:100%' });
    sh.appendChild(U.el('div', { class: 'klabel', text: '形状演算' }));
    sh.appendChild(U.el('div', { class: 'mono', style: 'font-size:11.5px;line-height:1.9;margin-top:4px' }, [
      U.el('div', { html: '6 + 258 + 3 = <span class="hl4">267</span>' }),
      U.el('div', { html: 'input_ids &nbsp;<span class="dim">(1, 267)</span>' }),
      U.el('div', { html: 'attention_mask &nbsp;<span class="dim">(1, 267)</span>' }),
      U.el('div', { html: 'inputs_embeds &nbsp;<span class="hl">(1, 267, 4096)</span>' }),
      U.el('div', { html: '256 x 4096 = <span class="hl3">1,048,576</span> = 2^20' }),
    ]));
    right.appendChild(sh);
    stage.appendChild(right);
    viz.appendChild(stage);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    st.reveal(0);
    msg.innerHTML = '<span class="cm">// 文本先进来</span>';
    tl.at(2000, () => { st.reveal(1); msg.innerHTML = '一张 448×448 图占 <em>258</em> 个位置：1 个开始标记 + <em>256</em> 个占位 + 1 个结束标记'; });
    tl.at(5200, () => {
      grid.querySelectorAll('.tcell').forEach((c, k) => {
        c.style.transitionDelay = ((k % 16) * 10) + 'ms';
        c.style.opacity = '1'; c.style.transform = 'none';
      });
      msg.innerHTML = '<em>256</em> 个视觉 token —— 它们不是额外的一维，而是<b>序列上的位置</b>';
    });
    tl.at(8600, () => { msg.innerHTML = '所以 <em>input_ids</em> 与 <em>attention_mask</em> 都是 <span class="mono">(1, 267)</span>'; });
    tl.at(11800, () => { msg.innerHTML = 'embed_tokens 之后 <em>inputs_embeds</em> = <span class="mono">(1, 267, 4096)</span>'; });
    tl.at(15000, () => { msg.innerHTML = '<span class="cm">// 其中 256 个位置被 masked_scatter 换成视觉向量（L2-06 / L2-07）</span>'; });
    tl.at(17400, () => { msg.innerHTML = '对齐的代价很实在：<em>267 x 4096 x 2B</em> ≈ 2.09 MiB（fp16 的一份激活）'; });
  },
},

/* ------------------------------------------------- 9 收束 */
{
  kicker: '收束',
  title: '把这一课压成一张表',
  sub: '315 行 = 一个「先备料再上桌」的构造函数 + 一个改写 proto 的开关 + 一套词表边界账 + 一个被忽略的参数。',
  caption: '下一课 L2-04：文本之后轮到像素 —— 图像处理基座。',
  lang: 'python',
  codeStart: 301,
  code: `    def extract(self, vocab_scores=None) -> tuple[dict[str, int], list[tuple[str, float]], list[tuple]]:
        """
        By default will return vocab and merges with respect to their order, by sending \`vocab_scores\` we're going to
        order the merges with respect to the piece scores instead.
        """
        sp = self.sp
        vocab_ids = {sp.id_to_piece(index): index for index in range(sp.GetPieceSize())}

        vocab_scores_dict = {sp.id_to_piece(i): sp.get_score(i) for i in range(sp.GetPieceSize())}

        merges = generate_merges(vocab_ids, vocab_scores_dict)

        vocab_scores_list = [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.GetPieceSize())]

        return vocab_ids, vocab_scores_list, merges`,
  codeNote: '回到开头那句话：签名里有的参数，不一定真的有用。',
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const steps = [
      { t: '① 常量', d: 'tokenizer.model + ▁', cc: 0 },
      { t: '② 备料', d: 'Load 在 super() 之前', cc: 0 },
      { t: '③ ★ legacy', d: '改写 proto 的 add_dummy_prefix', cc: 1 },
      { t: '④ _tokenize', d: 'unk 前缀补回词边界', cc: 1 },
      { t: '⑤ 词表账', d: 'vocab_size 与 len() 分开', cc: 2 },
      { t: '⑥ ★ 边界', d: 'in_base_vocab 判两次', cc: 4 },
    ];
    const row = U.el('div', { class: 'row gap8', style: 'width:100%;align-items:stretch' });
    const els = steps.map(s => {
      const e = W.card({ cc: s.cc, tint: s.cc, title: s.t, sub: s.d, style: 'flex:1 1 0;text-align:center' });
      row.appendChild(e); return e;
    });
    viz.appendChild(row);

    const tb = W.table([
      ['文件', '<span class="hl">315</span> 行', '作业书写 316，wc -l 实测 315'],
      ['直接子类', '<span class="hl">6</span> 个', 'Llama / Gemma 在 v5 已改继承 TokenizersBackend'],
      ['首 token', '<span class="hl3">▁Hello → H</span>', 'add_dummy_prefix 关掉之后'],
      ['序列长度', '<span class="hl3">2 → 4</span>', '同一个字符串，token 数变了'],
      ['extract(vocab_scores)', '<span class="hlbad">参数无效</span>', '传与不传，返回值完全相同'],
    ], { head: ['量', '实测值', '说明'] });
    viz.appendChild(U.el('div', { class: 'card cc0', style: 'width:100%;padding:10px 12px' }, [
      U.el('div', { class: 'klabel', text: '本课用到的实测数字（跑出来的，不是估算）' }), tb,
    ]));

    viz.appendChild(W.exercise(
      '同一个字符串，<code class="inl">legacy=True</code> 得到 <code class="inl">[\'▁Hello\', \'▁world\']</code>，'
      + '<code class="inl">legacy=False</code> 得到 <code class="inl">[\'H\', \'el\', \'lo\', \'▁world\']</code>。'
      + '这两串 id 送进 <code class="inl">embed_tokens</code> 之后，<b>形状</b>差多少？',
      '差 <b>2</b>：<code class="inl">(1, 2, 4096)</code> 对 <code class="inl">(1, 4, 4096)</code>。'
      + '<br>关键：<b>legacy 不是格式选项</b> —— 它改的是 sentencepiece 归一化器里的一个布尔，'
      + '而那个布尔决定首 token 是不是一个「带词边界的整词」。'
      + '<br>还要记住：<code class="inl">legacy=False</code> 时开头真的是空格的那条路'
      + '（<code class="inl">" Hello"</code>）会被 <code class="inl">_tokenize</code> 用 unk 前缀抢救回来，'
      + '所以两条路只在<b>首字符不是词边界</b>时才分岔。'));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    els.forEach(e => { e.style.opacity = '.28'; });
    msg.innerHTML = '<span class="cm">// 六个部件，逐个点亮</span>';
    steps.forEach((s, i) => {
      tl.at(1500 + i * 1900, () => {
        els.forEach((e, k) => { e.style.opacity = (k === i) ? '1' : '.28'; });
        msg.innerHTML = '<em>' + s.t + '</em> ' + s.d;
      });
    });
    tl.at(13200, () => {
      els.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = '★ 最后一个反例：<em>extract(vocab_scores)</em> 收下参数却从不使用 —— 文档与实现不一致';
    });
    tl.at(16200, () => {
      msg.innerHTML = '<span class="cm">// 下一课 L2-04：图像处理基座（尺寸 / 归一化 / 后端）</span>';
    });
  },
},

];
