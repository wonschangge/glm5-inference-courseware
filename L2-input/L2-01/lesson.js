/* ==========================================================================
   L2-01 · 分词器公共契约：慢与快共享的那一层
   --------------------------------------------------------------------------
   覆盖：tokenization_utils_base.py（1 个文件 / 3690 行）
   目标：看完能说出 input_ids 与 attention_mask 在哪一步产生，
        以及 return_tensors 影响的是哪一步。
   排版基准：#visual 可视区 = 994 x 662.7 舞台像素（无头浏览器实测）。
   ========================================================================== */
'use strict';

/* 实测常量（全部来自 _data/recon/probe_l201.py 的真实输出，不是估算）
   词表： [UNK]=0 [PAD]=1 [CLS]=2 [SEP]=3，带 "[CLS] $A [SEP]" 后处理器 */
const M = {
  /* tokenizer.pad({"input_ids": [4,5,6]}, padding="max_length", max_length=5) 的实测输出 */
  padRightIds: [4, 5, 6, 1, 1],
  padRightMask: [1, 1, 1, 0, 0],
  padLeftIds: [1, 1, 4, 5, 6],
  padLeftMask: [0, 0, 1, 1, 1],
  /* SPECIAL_TOKENS_ATTRIBUTES 的 7 个名字，顺序即遍历顺序 */
  attrs: ['bos_token', 'eos_token', 'unk_token', 'sep_token', 'pad_token', 'cls_token', 'mask_token'],
};

const SCENES = [

/* ------------------------------------------------- 1 全景：四段流水线 */
{
  kicker: '第 2 层 · 输入流水线',
  title: '一次 <span class="hl-a">tokenizer(text)</span> 的四段流水线',
  sub: '慢分词器与快分词器共享的就是这一层。四段里只有 ②③ 会碰形状，①④ 一个 token 都不改。',
  caption: '这一课不训练任何模型：只把「你写的那行调用」拆成四段可验证的手续。',
  lang: 'python',
  codeStart: 2421,
  code: `    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        text_pair: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        text_target: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        text_pair_target: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        add_special_tokens: bool = True,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool | str | TruncationStrategy | None = None,
        max_length: int | None = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: int | None = None,
        padding_side: str | None = None,
        return_tensors: str | TensorType | None = None,
        return_token_type_ids: bool | None = None,
        return_attention_mask: bool | None = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        tokenizer_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> BatchEncoding:`,
  codeNote: 'PreTrainedTokenizerBase.__call__ —— 所有分词器唯一的公共入口。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const nodes = [
      { t: '① 收参', s: '21 形参 → all_kwargs', cc: 0 },
      { t: '② 归一化', s: 'True → 枚举', cc: 2 },
      { t: '③ 编码 + padding', s: 'input_ids / attention_mask', cc: 4 },
      { t: '④ 装箱', s: 'BatchEncoding', cc: 1 },
    ];
    const fl = W.flow(nodes, { style: 'width:100%' });
    wrap.appendChild(U.el('div', { class: 'col gap8', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: '四段流水线 —— 只有 ②③ 会改变张量形状' }), fl,
    ]));

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '决定这一切的三个类属性（本课引用的第一段代码）' }));
    const stats = [
      { s: 'model_input_names', v: '["input_ids", "attention_mask"]', cc: 4, note: 'attention_mask 造不造，由它说了算' },
      { s: 'padding_side', v: '"right"', cc: 0, note: '补在哪一头；另一个合法值是 "left"' },
      { s: 'truncation_side', v: '"right"', cc: 1, note: '砍在哪一头；与 padding_side 互相独立' },
    ];
    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cards = stats.map(st => {
      const c = W.card({
        cc: st.cc, tint: st.cc,
        title: '<span class="mono" style="font-size:11px">' + st.s + '</span>',
        body: U.el('div', { class: 'col gap6' }, [
          U.el('div', { class: 'mono', style: 'font-size:11.5px;color:#fff;word-break:break-all', text: st.v }),
          U.el('div', { class: 'dim', style: 'font-size:10.5px;line-height:1.45', text: st.note }),
        ]),
        style: 'flex:1',
      });
      row.appendChild(c); return c;
    });
    viz.appendChild(row);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    fl.focus(0);
    msg.innerHTML = '<span class="cm">// 一次调用从这里开始</span>';
    tl.at(2000, () => { fl.focus(1); msg.innerHTML = '② 归一化：<em>padding</em> / <em>truncation</em> 在这里被翻译成枚举'; });
    tl.at(5200, () => { fl.focus(2); msg.innerHTML = '③ 编码与 padding：<em>input_ids</em> 与 <em>attention_mask</em> 都在这一段里产生'; });
    tl.at(8600, () => { fl.focus(3); msg.innerHTML = '④ 装箱：<em>return_tensors</em> 只在这一段起作用'; });
    tl.at(12000, () => {
      fl.focus(-1); cards.forEach(c => c.classList.add('ac'));
      msg.innerHTML = '<span class="cm">// 本课要回答的两个验收点，答案分别在 ③ 与 ④ 里</span>';
    });
    tl.at(14600, () => {
      cards.forEach(c => c.classList.remove('ac'));
      msg.innerHTML = '<em>model_input_names</em> 是隐藏的主角：它决定 mask 会不会被造出来';
    });
  },
},

/* ------------------------------------------------- 2 token 属性体系 */
{
  kicker: '契约 ① · 属性',
  title: '<span class="hl-a">7 个名字</span>：token 属性的全部名单',
  sub: 'SPECIAL_TOKENS_ATTRIBUTES 是一个 list 而不是 set —— 顺序有语义，special_tokens_map 按它遍历。',
  caption: '实测订正：SpecialTokensMixin 这个类在本仓库已经不存在，它的 API 被并入了 PreTrainedTokenizerBase。',
  lang: 'python',
  codeStart: 971,
  code: `    # first name has to correspond to main model input name
    # to make sure \`tokenizer.pad(...)\` works correctly
    model_input_names: list[str] = ["input_ids", "attention_mask"]
    padding_side: str = "right"
    truncation_side: str = "right"
    slow_tokenizer_class = None

    # Special tokens support (moved from SpecialTokensMixin)
    # V5: Clean separation of named special tokens from extra special tokens
    SPECIAL_TOKENS_ATTRIBUTES = [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
    ]`,
  codeNote: '第 8 行那句注释就是「SpecialTokensMixin 已经不在了」的证据。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '7 个固定名字 · 实测值来自 probe_l201.py 构造的分词器' }));
    const tb = W.table([
      ['<span class="mono">bos_token</span>', '<span class="hlbad">None</span>', '没设 → 读出来是 None，不抛错'],
      ['<span class="mono">eos_token</span>', '<span class="hlbad">None</span>', '同上'],
      ['<span class="mono">unk_token</span>', "'[UNK]'", 'id = 0'],
      ['<span class="mono">sep_token</span>', "'[SEP]'", 'id = 3'],
      ['<span class="mono">pad_token</span>', "'[PAD]'", 'id = 1'],
      ['<span class="mono">cls_token</span>', "'[CLS]'", 'id = 2'],
      ['<span class="mono">mask_token</span>', '<span class="hlbad">None</span>', 'add_special_tokens 之后才有'],
    ], { head: ['属性名', '实测值', '说明'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 0, style: 'padding:10px 12px' }, [tb]));

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const c1 = W.card({
      cc: 4, tint: 4, title: '_special_tokens_map',
      sub: '一个有 7 个键的 dict，值可以是 str 或 AddedToken；<b>未设置 = 键存在但值为 None</b>。',
      body: U.el('div', { class: 'mono', style: 'font-size:10.5px;color:var(--ink-dim)', text: 'dict.fromkeys(SPECIAL_TOKENS_ATTRIBUTES)' }),
      style: 'flex:1',
    });
    const c2 = W.card({
      cc: 2, tint: 2, title: '_extra_special_tokens',
      sub: '一个 list，装没有标准名字的模型专属 token（多模态的 <code class="inl">&lt;image&gt;</code> 就走这里）。',
      body: U.el('div', { class: 'mono', style: 'font-size:10.5px;color:var(--ink-dim)', text: 'V5: 命名 token 与额外 token 分开存' }),
      style: 'flex:1',
    });
    row.appendChild(c1); row.appendChild(c2);
    viz.appendChild(row);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    tb.querySelectorAll('tbody tr').forEach(tr => { tr.style.opacity = '.3'; tr.style.transition = 'opacity .4s'; });
    msg.innerHTML = '<span class="cm">// 名单是固定的 7 个 —— 但顺序有语义</span>';
    const marks = [2, 4, 5];
    marks.forEach((i, k) => {
      tl.at(2200 + k * 2400, () => {
        const rows = tb.querySelectorAll('tbody tr');
        rows.forEach((tr, j) => { tr.style.opacity = (j === i) ? '1' : '.3'; });
        msg.innerHTML = '第 <em>' + (i + 1) + '</em> 项 <span class="hl">' + M.attrs[i] + '</span>'
          + ' &nbsp;<span class="cm">// 顺序就是 special_tokens_map 的遍历顺序</span>';
      });
    });
    tl.at(10000, () => {
      tb.querySelectorAll('tbody tr').forEach(tr => { tr.style.opacity = '1'; });
      c1.classList.add('ac');
      msg.innerHTML = '实测 <em>all_special_tokens</em> = [\'[UNK]\', \'[SEP]\', \'[PAD]\', \'[CLS]\']'
        + ' &nbsp;<span class="cm">// 正是这张名单过滤掉 None 之后的顺序</span>';
    });
    tl.at(13400, () => {
      c1.classList.remove('ac'); c2.classList.add('ac');
      msg.innerHTML = '★ 订正：源码里只剩一句 <em>moved from SpecialTokensMixin</em> —— 那个类已经不存在了';
    });
  },
},

/* ------------------------------------------------- 3 属性翻译 */
{
  kicker: '契约 ① · 读写',
  title: 'token 与 id 是<span class="hl-b">同一份数据</span>的两个视图',
  sub: 'pad_token 给字符串、pad_token_id 给整数，但它们落在同一个 dict 的同一个键上。',
  caption: '实测：tok.pad_token_id = 3 之后，tok.pad_token == "[SEP]" —— 存的是 token，不是 id。',
  lang: 'python',
  codeStart: 1281,
  code: `    def __getattr__(self, key):
        # Handle _id/_ids suffix (eg. bos_token_id -> bos_token)
        key_without_id = key.removesuffix("_ids").removesuffix("_id") if key.endswith(("_id", "_ids")) else key

        # Named special tokens (bos_token, eos_token, etc.)
        if key_without_id in self.SPECIAL_TOKENS_ATTRIBUTES:
            # Use __dict__.get to avoid recursive __getattr__ when _special_tokens_map
            # is not yet initialized (e.g. during fast tokenizer __init__)
            token_value = self.__dict__.get("_special_tokens_map", {}).get(key_without_id)
            if token_value is None:
                if self.verbose:
                    logger.error(f"Using {key}, but it is not set yet.")
                return None
            return self.convert_tokens_to_ids(str(token_value)) if key != key_without_id else str(token_value)`,
  codeNote: '__getattr__ 与 __setattr__ 完全对称：带 _id 的读要翻译，不带的直接给字符串。',
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '一个存储，两个视图（实测值）' }));

    const stage = U.el('div', { class: 'row gap12 center', style: 'width:100%' });
    const mk = (title, cc, lines) => {
      const c = W.card({
        cc: cc, tint: cc, title: title, style: 'flex:1',
        body: U.el('div', { class: 'col gap4' }, lines.map(t =>
          U.el('div', { class: 'mono', style: 'font-size:11.5px;color:#fff', html: t }))),
      });
      return c;
    };
    const left = mk('读 <span class="mono">pad_token</span>', 0, [
      '<span class="cm"># 直接给字符串</span>',
      "'[PAD]'",
    ]);
    const mid = U.el('div', { class: 'col center gap6', style: 'flex:none;width:120px' });
    mid.innerHTML = '<div class="mono" style="font-size:18px;color:var(--accent)">⇄</div>'
      + '<div class="mono" style="font-size:10px;color:var(--ink-faint);text-align:center">同一个 _special_tokens_map</div>';
    const right = mk('读 <span class="mono">pad_token_id</span>', 4, [
      '<span class="cm"># convert_tokens_to_ids(str(v))</span>',
      '1',
    ]);
    stage.appendChild(left); stage.appendChild(mid); stage.appendChild(right);
    viz.appendChild(stage);

    const tb = W.table([
      ['<span class="mono">tok.pad_token</span>', '<span class="hl">\'[PAD]\'</span>', '不带 _id → 原样返回 str(value)'],
      ['<span class="mono">tok.pad_token_id</span>', '<span class="hl">1</span>', '带 _id → 每次查一次词表'],
      ['<span class="mono">tok.pad_token_id = 3</span>', '<span class="hl2">pad_token 变成 \'[SEP]\'</span>', '写进去的是 token，不是整数'],
      ['<span class="mono">tok.pad_token = None</span>', '<span class="hlbad">清空</span>', 'None 被显式放过；它会从 special_tokens_map 里消失'],
      ['<span class="mono">tok.bos_token</span>', '<span class="hlbad">None</span>', '未设置不抛 AttributeError，直接给 None'],
    ], { head: ['写法', '实测结果', '为什么'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 1, style: 'padding:10px 12px' }, [tb]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    right.style.opacity = '.25';
    msg.innerHTML = '<span class="cm">// 不带 _id 的读：直接给字符串</span>';
    tl.at(2200, () => { right.style.opacity = '1'; left.classList.add('ac'); msg.innerHTML = '加上 <em>_id</em> 后缀 → <span class="hl2">convert_tokens_to_ids(str(value))</span>'; });
    tl.at(5600, () => {
      left.classList.remove('ac'); right.classList.add('ac');
      msg.innerHTML = '反过来写也一样：<em>pad_token_id = 3</em> 存下来的是 <span class="hl2">\'[SEP]\'</span>';
    });
    tl.at(9000, () => {
      right.classList.remove('ac');
      msg.innerHTML = '<span class="cm">// 两个视图，一个 dict —— 所以必须自己实现 __getattr__ / __setattr__</span>';
    });
    tl.at(12200, () => {
      msg.innerHTML = '<em>None</em> 是「未设置」的唯一表示：读出来是 None，也不出现在 <em>special_tokens_map</em> 里';
    });
  },
},

/* ------------------------------------------------- 4 策略枚举 */
{
  kicker: '契约 ② · 枚举',
  title: '★ <span class="hl-a">4</span> 种截断在这里，<span class="hl-b">3</span> 种填充不在这里',
  sub: 'TruncationStrategy 定义在本文件；PaddingStrategy 定义在 utils/generic.py —— 两个枚举不都在同一份文件里。',
  caption: '作业书要求「完整枚举」，实测订正：padding 的那 3 个值属 L1 的覆盖域，本课只做汇总、不引用。',
  lang: 'python',
  codeStart: 154,
  code: `class TruncationStrategy(ExplicitEnum):
    """
    Possible values for the \`truncation\` argument in [\`PreTrainedTokenizerBase.__call__\`]. Useful for tab-completion in
    an IDE.
    """

    ONLY_FIRST = "only_first"
    ONLY_SECOND = "only_second"
    LONGEST_FIRST = "longest_first"
    DO_NOT_TRUNCATE = "do_not_truncate"`,
  codeNote: 'ExplicitEnum：成员能与字符串直接比较，所以 "longest_first" 和枚举成员走同一条路。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '两个枚举，两处定义（实测取值）' }));

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const mkEnum = (title, cc, here, items) => {
      const body = U.el('div', { class: 'col gap6' });
      items.forEach(it => body.appendChild(U.el('div', {
        class: 'row gap8', style: 'width:100%',
      }, [
        U.el('span', { class: 'chip c' + cc, html: '<i class="sw sw-c' + cc + '"></i>' + it.v }),
        U.el('span', { class: 'dim', style: 'font-size:10.5px', text: it.d }),
      ])));
      return W.card({
        cc: cc, tint: cc, style: 'flex:1',
        title: title + ' &nbsp;<span class="mono" style="font-size:10px;color:'
          + (here ? 'var(--c4)' : 'var(--c2)') + '">' + (here ? '本文件 L154' : 'utils/generic.py L579') + '</span>',
        body: body,
      });
    };
    const cT = mkEnum('<span class="mono">TruncationStrategy</span>', 1, true, [
      { v: 'only_first', d: '只砍第一句' },
      { v: 'only_second', d: '只砍第二句' },
      { v: 'longest_first', d: '两句轮流砍（True 的落点）' },
      { v: 'do_not_truncate', d: '不截断 —— 也是一个显式策略值' },
    ]);
    const cP = mkEnum('<span class="mono">PaddingStrategy</span>', 2, false, [
      { v: 'longest', d: '补到本批最长（True 的落点）' },
      { v: 'max_length', d: '补到 max_length / model_max_length' },
      { v: 'do_not_pad', d: '完全不补（False 的落点）' },
    ]);
    row.appendChild(cT); row.appendChild(cP);
    viz.appendChild(row);

    viz.appendChild(U.el('div', { class: 'klabel', text: '实测：四种 truncation 在成对输入上的真实差别（max_length=14）' }));
    const tb = W.table([
      ['<span class="mono">（不截断）</span>', '<span class="mono" style="font-size:10.5px">[2,4,5,6,7,8,9,3, 6,7,8,9,10,11,6,12,13,3]</span>', '18 个'],
      ['<span class="mono">only_first</span>', '<span class="mono" style="font-size:10.5px">[2,4,5,3, 6,7,8,9,10,11,6,12,13,3]</span>', '第一句砍到 3'],
      ['<span class="mono">only_second</span>', '<span class="mono" style="font-size:10.5px">[2,4,5,6,7,8,9,3, 6,7,8,9,10,3]</span>', '第二句砍到 5'],
      ['<span class="mono">longest_first</span>', '<span class="mono" style="font-size:10.5px">[2,4,5,6,7,8,3, 6,7,8,9,10,11,3]</span>', '两句轮流砍'],
    ], { head: ['truncation', '实测 input_ids', '结果'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 4, style: 'padding:10px 12px' }, [tb]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    cT.style.opacity = '.35'; cP.style.opacity = '.35';
    msg.innerHTML = '<span class="cm">// 两个枚举，两个文件</span>';
    tl.at(2200, () => { cT.style.opacity = '1'; cT.classList.add('ac'); msg.innerHTML = '截断的 <em>4</em> 个值定义在本课覆盖的文件里 —— 它们能被逐字引用'; });
    tl.at(5800, () => {
      cT.classList.remove('ac'); cP.style.opacity = '1'; cP.classList.add('ac');
      msg.innerHTML = '填充的 <em>3</em> 个值在 <span class="hl3">utils/generic.py</span> —— <span class="cm">// 属 L1 覆盖域，本课只汇总</span>';
    });
    tl.at(9600, () => {
      cP.classList.remove('ac');
      msg.innerHTML = '<em>do_not_truncate</em> 是枚举成员之一 —— 所以判空要写 <span class="hl2">!= DO_NOT_TRUNCATE</span>，不能写 is not None';
    });
    tl.at(13400, () => {
      msg.innerHTML = '<span class="cm">// 四个值只有在成对输入（text + text_pair）时才产生可见差别</span>';
    });
    tl.at(15800, () => {
      msg.innerHTML = '两种策略最终都会被归一化成枚举：<em>下游拿到的永远不是 True / False</em>';
    });
  },
},

/* ------------------------------------------------- 5 padding 归一化 */
{
  kicker: '契约 ② · 归一化',
  title: '<span class="hl-a">True</span> 不是策略值，是<span class="hl-b">别名</span>',
  sub: 'padding=True 被翻译成 PaddingStrategy.LONGEST；False 也不是「不设置」，而是与它平级的 DO_NOT_PAD。',
  caption: '这一段用的是同一性比较（is），所以 padding=1 不会走进 True 分支，而是直接抛错。',
  lang: 'python',
  codeStart: 2346,
  code: `        # Backward compatibility for previous behavior:
        # If you only set max_length, it activates truncation for max_length
        if max_length is not None and padding is False and truncation is None:
            truncation = "longest_first"

        # Get padding strategy
        if padding is not False:
            if padding is True:
                if verbose:
                    if max_length is not None and (
                        truncation is None or truncation is False or truncation == "do_not_truncate"
                    ):
                        warnings.warn(
                            "\`max_length\` is ignored when \`padding\`=\`True\` and there is no truncation strategy. "
                            "To pad to max length, use \`padding='max_length'\`."
                        )
                padding_strategy = PaddingStrategy.LONGEST  # Default to pad to the longest sequence in the batch
            elif not isinstance(padding, PaddingStrategy):
                padding_strategy = PaddingStrategy(padding)
            elif isinstance(padding, PaddingStrategy):
                padding_strategy = padding
        else:
            padding_strategy = PaddingStrategy.DO_NOT_PAD`,
  codeNote: '整个函数只做一件事：把调用方给的各种写法翻译成枚举。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '入口写法 → 出口枚举（全部实测）' }));

    const tb = W.table([
      ['<span class="mono">padding=True</span>', '<span class="hl">PaddingStrategy.LONGEST</span>', '别名翻译；与 "longest" 输出逐位相同'],
      ['<span class="mono">padding="longest"</span>', '<span class="hl">PaddingStrategy.LONGEST</span>', 'PaddingStrategy(padding) 直接构造'],
      ['<span class="mono">padding=False</span>', '<span class="hl">PaddingStrategy.DO_NOT_PAD</span>', '走 else 分支，与 LONGEST 平级'],
      ['<span class="mono">padding=PaddingStrategy.X</span>', '<span class="hl">原样透传</span>', 'isinstance 分支：已经是枚举就不动'],
      ['<span class="mono">padding=1</span>', '<span class="hlbad">ValueError</span>', 'is 比较不成立 → PaddingStrategy(1) 抛错'],
      ['<span class="mono">max_length=4（单独给）</span>', '<span class="hl3">truncation 被偷偷打开</span>', '向后兼容暗门：改成 "longest_first"'],
    ], { head: ['调用方写法', '归一化结果', '说明'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 0, style: 'padding:10px 12px' }, [tb]));

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const c1 = W.card({
      cc: 2, tint: 2, title: '为什么要归一化',
      sub: '调用方可以写 True / "longest" / 枚举三种写法；下游 <code class="inl">_encode_plus</code> 的形参标注是 <code class="inl">PaddingStrategy</code>。中间必须有且只有一次翻译。',
      style: 'flex:1',
    });
    const c2 = W.card({
      cc: 3, tint: 3, title: '向后兼容暗门',
      sub: '只给 max_length、既不说 padding 也不说 truncation 时，它会自作主张打开截断 —— 这个行为没有任何显式开关。',
      style: 'flex:1',
    });
    row.appendChild(c1); row.appendChild(c2);
    viz.appendChild(row);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    const rows = tb.querySelectorAll('tbody tr');
    rows.forEach(tr => { tr.style.opacity = '.28'; tr.style.transition = 'opacity .4s'; });
    const show = (i) => rows.forEach((tr, j) => { tr.style.opacity = (j === i) ? '1' : '.28'; });
    msg.innerHTML = '<span class="cm">// 三种写法，一个出口</span>';
    tl.at(2000, () => { show(0); msg.innerHTML = '<em>True</em> → <span class="hl">LONGEST</span> &nbsp;<span class="cm">// 别名，不是新策略</span>'; });
    tl.at(5000, () => { show(1); msg.innerHTML = '<em>"longest"</em> → 同一个枚举成员：<span class="cm">// 实测两种写法输出逐位相同</span>'; });
    tl.at(8200, () => { show(2); msg.innerHTML = '<em>False</em> → <span class="hl">DO_NOT_PAD</span> &nbsp;<span class="cm">// 也是策略值，不是「不设置」</span>'; });
    tl.at(11400, () => { show(4); msg.innerHTML = '<em>padding=1</em> → <span class="hlbad">ValueError</span>：<span class="cm">// is 比较要求值精确</span>'; });
    tl.at(14200, () => {
      rows.forEach(tr => { tr.style.opacity = '1'; });
      c2.classList.add('ac');
      msg.innerHTML = '<em>max_length</em> 单独出现 → 截断被静默打开 &nbsp;<span class="cm">// 见下一幕的两种失效方式</span>';
    });
  },
},

/* ------------------------------------------------- 6 max_length 静默失效 */
{
  kicker: '契约 ② · 陷阱',
  title: '<span class="hl-a">max_length</span> 的两种静默失效',
  sub: '默认的 model_max_length 是 int(1e30)，任何「补到最大长度」的请求都会被降级成「什么都不做」。',
  caption: '实测：padding="max_length" 在没设 model_max_length 的分词器上根本没补齐，形状还是 (2,4) 与 (2,7)。',
  lang: 'python',
  codeStart: 2370,
  code: `        # Get truncation strategy
        if truncation is not False and truncation is not None:
            if truncation is True:
                truncation_strategy = (
                    TruncationStrategy.LONGEST_FIRST
                )  # Default to truncate the longest sequences in pairs of inputs
            elif not isinstance(truncation, TruncationStrategy):
                truncation_strategy = TruncationStrategy(truncation)
            elif isinstance(truncation, TruncationStrategy):
                truncation_strategy = truncation
        else:
            truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE

        # Set max length if needed
        if max_length is None:
            if padding_strategy == PaddingStrategy.MAX_LENGTH:
                if self.model_max_length > LARGE_INTEGER:
                    padding_strategy = PaddingStrategy.DO_NOT_PAD
                else:
                    max_length = self.model_max_length

            if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE:
                if self.model_max_length > LARGE_INTEGER:
                    truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE
                else:
                    max_length = self.model_max_length`,
  codeNote: '注意是赋值覆盖而不是抛错 —— 调用方看不到任何异常。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '实测：同一个分词器，三种写法的真实输出' }));

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const mk = (cc, title, sub, code) => W.card({
      cc: cc, tint: cc, title: title, sub: sub, style: 'flex:1',
      body: U.el('div', { class: 'mono', style: 'font-size:10.5px;line-height:1.6;color:#fff;word-break:break-all', text: code }),
    });
    const cA = mk(3, 'padding="max_length"', 'model_max_length = int(1e30)（默认）→ 策略降级', '[[2,4,5,3], [2,6,7,8,9,10,3]]');
    const cB = mk(4, 'padding="max_length" + max_length=8', '显式给了 max_length → 真的补齐', '[[2,4,5,3,1,1,1,1], [2,6,7,8,9,10,3,1]]');
    const cC = mk(2, 'max_length=4（单独给）', '截断被偷偷打开', '[2,4,5,3]  ← 原本 8 个');
    row.appendChild(cA); row.appendChild(cB); row.appendChild(cC);
    viz.appendChild(row);

    const tb = W.table([
      ['<span class="mono">max_length is None</span>', '只看 model_max_length', '两个分支各判一次，都是 > LARGE_INTEGER'],
      ['<span class="mono">LARGE_INTEGER</span>', '<span class="mono">int(1e20)</span>', '门槛不是 == 1e30，而是「大得离谱就当没设过」'],
      ['<span class="mono">实测默认值</span>', '<span class="mono" style="font-size:10.5px">1000000000000000019884624838656</span>', 'int(1e30) 的 float 化结果'],
      ['<span class="mono">唯一抛错处</span>', '<span class="hlbad">要 padding 却没有 pad_token</span>', '紧随引用块之后的 5 行；判据是 None 或 id &lt; 0'],
    ], { head: ['判据', '实测', '说明'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 1, style: 'padding:10px 12px' }, [tb]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    [cA, cB, cC].forEach(c => { c.style.opacity = '.3'; });
    msg.innerHTML = '<span class="cm">// 三种写法，三种结果</span>';
    tl.at(2200, () => {
      cA.style.opacity = '1'; cA.classList.add('ac');
      msg.innerHTML = '默认 <em>model_max_length</em> 大得离谱 → 策略被<span class="hlbad">静默降级</span>，形状根本没对齐';
    });
    tl.at(5800, () => {
      cA.classList.remove('ac'); cB.style.opacity = '1'; cB.classList.add('ac');
      msg.innerHTML = '显式给 <em>max_length=8</em> → 才真的补到 8 &nbsp;<span class="cm">// 这才是你以为的那个行为</span>';
    });
    tl.at(9600, () => {
      cB.classList.remove('ac'); cC.style.opacity = '1'; cC.classList.add('ac');
      msg.innerHTML = '第二种失效：只给 <em>max_length</em> → 截断被偷偷打开，<span class="hl3">这个暗门没有开关</span>';
    });
    tl.at(13200, () => {
      [cA, cB, cC].forEach(c => { c.classList.remove('ac'); c.style.opacity = '1'; });
      msg.innerHTML = '两条路都指向同一句话：<em>形状没对齐，转张量时就会抛错</em> —— 见第八幕';
    });
    tl.at(16000, () => {
      msg.innerHTML = '<span class="cm">// 这个函数出口的 4 个值永远是枚举：padding_strategy / truncation_strategy / max_length / kwargs</span>';
    });
  },
},

/* ------------------------------------------------- 7 ★ attention_mask */
{
  kicker: '★ 洞察 ① · 形状',
  title: '<span class="hl-a">attention_mask</span> 与 <span class="hl4">input_ids</span> 同生同长',
  sub: '两行由同一个 difference 决定：mask 先补满 1，padding 时再在同样的位置补 0 —— 长度天然相等。',
  caption: 'mask 造不造，取决于类属性 model_input_names 里有没有 "attention_mask"，而不是调用方传了什么。',
  lang: 'python',
  codeStart: 2796,
  code: `        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        required_input = encoded_inputs[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        # Initialize attention mask if not present.
        if return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * len(required_input)`,
  codeNote: '_pad 是慢分词器与显式 tokenizer.pad() 走的那条路。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '实测：pad({"input_ids": [4,5,6]}, padding="max_length", max_length=5)' }));

    const mkStrip = (label, cc, ids, mask) => {
      const col = U.el('div', { class: 'col gap8', style: 'flex:1;min-width:0' });
      col.appendChild(U.el('div', { class: 'klabel', text: label }));
      const mkRow = (vals, isMask) => {
        const r = U.el('div', { class: 'row gap4', style: 'width:100%' });
        const cells = vals.map((v, i) => {
          const pad = isMask ? v === 0 : i >= 3;
          const e = U.el('div', {
            class: 'n sm',
            style: 'flex:1 1 0;height:40px;border-radius:8px;display:flex;align-items:center;'
              + 'justify-content:center;font:600 13px var(--mono);transition:opacity .5s,transform .5s;'
              + 'border:1px solid ' + (pad ? 'var(--c2)' : 'var(--c' + cc + ')')
              + ';background:' + (pad ? 'rgba(251,191,36,.18)' : 'rgba(150,180,255,.10)')
              + ';color:' + (pad ? '#fde68a' : '#fff'),
            text: String(v),
          });
          r.appendChild(e); return e;
        });
        return { el: r, cells };
      };
      const idsRow = mkRow(ids, false);
      const maskRow = mkRow(mask, true);
      col.appendChild(U.el('div', { class: 'col gap4', style: 'width:100%' }, [
        U.el('div', { class: 'mono dim', style: 'font-size:10.5px', text: 'input_ids' }), idsRow.el,
      ]));
      col.appendChild(U.el('div', { class: 'col gap4', style: 'width:100%' }, [
        U.el('div', { class: 'mono dim', style: 'font-size:10.5px', text: 'attention_mask' }), maskRow.el,
      ]));
      return { el: col, ids: idsRow.cells, mask: maskRow.cells };
    };

    const row = U.el('div', { class: 'row gap16', style: 'width:100%;align-items:stretch' });
    const R = mkStrip('padding_side = "right"（默认）', 4, M.padRightIds, M.padRightMask);
    const L = mkStrip('padding_side = "left"', 0, M.padLeftIds, M.padLeftMask);
    row.appendChild(R.el); row.appendChild(L.el);
    viz.appendChild(row);

    viz.appendChild(U.el('div', { class: 'row gap12 wrap', style: 'width:100%' }, [
      U.el('span', { class: 'chip c4', html: '<i class="sw sw-c4"></i>真实 token / mask=1' }),
      U.el('span', { class: 'chip c2', html: '<i class="sw sw-c2"></i>pad 位 / mask=0' }),
      U.el('span', { class: 'dim mono', style: 'margin-left:auto;font-size:10.5px', text: 'difference = max_length - len(required_input) = 2' }),
    ]));

    const tb = W.table([
      ['<span class="mono">return_attention_mask</span>', '<span class="mono">"attention_mask" in self.model_input_names</span>', '默认 True —— mask 是默认要造的'],
      ['<span class="mono">mask 初始化</span>', '<span class="mono">[1] * len(required_input)</span>', '先于 padding，且不覆盖已存在的 mask'],
      ['<span class="mono">needs_to_be_padded</span>', '<span class="mono">策略 != DO_NOT_PAD and len != max_length</span>', '唯一的分支开关'],
      ['<span class="mono">right / left</span>', '<span class="mono">[1]*n+[0]*d &nbsp;/&nbsp; [0]*d+[1]*n</span>', 'mask 与 ids 用同一个 difference'],
    ], { head: ['环节', '代码', '为什么'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 4, style: 'padding:10px 12px' }, [tb]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    [R.el, L.el].forEach(e => { e.style.opacity = '.25'; });
    R.ids.slice(3).concat(R.mask.slice(3)).forEach(c => { c.style.opacity = '0'; c.style.transform = 'translateY(-8px)'; });
    msg.innerHTML = '<span class="cm">// 原始输入：[4, 5, 6]（长度 3）</span>';
    tl.at(2200, () => {
      R.el.style.opacity = '1';
      R.ids.slice(3).concat(R.mask.slice(3)).forEach(c => { c.style.opacity = '1'; c.style.transform = 'none'; });
      msg.innerHTML = 'right：ids 尾部补 <em>pad_id=1</em>，mask 同位置补 <em>0</em>';
    });
    tl.at(5800, () => {
      L.el.style.opacity = '1';
      msg.innerHTML = 'left：<span class="cm">// 同一个 difference，补在头部</span> ids [1,1,4,5,6] / mask [0,0,1,1,1]';
    });
    tl.at(9400, () => {
      msg.innerHTML = '<em>difference</em> 是同一个数：<span class="cm">// 它同时决定 ids 补几个、mask 补几个</span>';
    });
    tl.at(12800, () => {
      msg.innerHTML = '<span class="cm">// 顺序：mask 先补满 1，padding 再补 0 —— 所以长度天然相等，不是两次独立计算</span>';
    });
    tl.at(15600, () => {
      msg.innerHTML = '★ 长度相等不是巧合：<em>它们在同两个分支里被同一个变量决定</em>';
    });
  },
},

/* ------------------------------------------------- 8 ★ return_tensors */
{
  kicker: '★ 洞察 ② · 装箱',
  title: '<span class="hl-a">return_tensors</span> 影响的是<span class="hl-b">装箱</span>，不是编码',
  sub: 'BatchEncoding.__init__ 的最后一行是唯一一次类型转换；batch 轴也是在那一步补出来的。',
  caption: '实测：list 与 tensor 的数值逐位相同 —— 差别只在容器，不在内容。',
  lang: 'python',
  codeStart: 222,
  code: `    def __init__(
        self,
        data: dict[str, Any] | None = None,
        encoding: EncodingFast | Sequence[EncodingFast] | None = None,
        tensor_type: None | str | TensorType = None,
        prepend_batch_axis: bool = False,
        n_sequences: int | None = None,
    ):
        super().__init__(data)

        # If encoding is not None, the fast tokenization is used
        if encoding is not None and isinstance(encoding, EncodingFast):
            encoding = [encoding]

        self._encodings = encoding

        if n_sequences is None and encoding is not None and encoding:
            n_sequences = encoding[0].n_sequences

        self._n_sequences = n_sequences

        self.convert_to_tensors(tensor_type=tensor_type, prepend_batch_axis=prepend_batch_axis)`,
  codeNote: 'tensor_type 是构造参数：它在 __init__ 返回之前就被消费掉了。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '同一批数据，三种容器（实测）' }));

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const mk = (cc, title, shape, lines) => W.card({
      cc: cc, tint: cc, title: title, style: 'flex:1',
      body: U.el('div', { class: 'col gap6' }, [
        U.el('div', { class: 'n sm mono', style: 'color:var(--c' + cc + ')', text: shape }),
        U.el('div', { class: 'mono', style: 'font-size:10.5px;line-height:1.6;color:var(--ink-dim);word-break:break-all', text: lines }),
      ]),
    });
    const cL = mk(0, 'return_tensors=None', 'list，无 batch 轴', '[2,4,5,3]');
    const cN = mk(4, 'return_tensors="np"', 'ndarray (1, 4)', '[[2,4,5,3]]');
    const cP = mk(1, 'return_tensors="pt"', 'Tensor (1, 4) int64', '[[2,4,5,3]]');
    row.appendChild(cL); row.appendChild(cN); row.appendChild(cP);
    viz.appendChild(row);

    viz.appendChild(U.el('div', { class: 'klabel', text: '转换点在哪里 —— 三行代码，一条链' }));
    const chain = U.el('div', { class: 'col gap6', style: 'width:100%' });
    [
      ['pad() 的出口', 'return BatchEncoding(batch_outputs, tensor_type=return_tensors)', 0],
      ['__init__ 最后一行', 'self.convert_to_tensors(tensor_type=tensor_type, prepend_batch_axis=prepend_batch_axis)', 1],
      ['convert_to_tensors 开头', 'if tensor_type is None: return self', 4],
    ].forEach(([k, v, cc]) => {
      chain.appendChild(U.el('div', { class: 'row gap10', style: 'width:100%' }, [
        U.el('div', { class: 'n sm', style: 'flex:none;width:150px;color:var(--c' + cc + ')', text: k }),
        U.el('div', { class: 'mono', style: 'flex:1;font-size:10.5px;color:#fff;word-break:break-all', text: v }),
      ]));
    });
    viz.appendChild(U.el('div', { class: 'card', cc: 1, style: 'padding:10px 12px' }, [chain]));

    const tb = W.table([
      ['<span class="mono">不传 return_tensors</span>', '函数第一句就 <span class="hl">return self</span>', '空操作 —— list 与 tensor 的差别只可能来自这里'],
      ['<span class="mono">prepend_batch_axis=True</span>', '<span class="hl">(3,) → (1, 3)</span>', 'batch 轴是在转换时补的'],
      ['<span class="mono">ragged 批次 + return_tensors</span>', '<span class="hlbad">ValueError</span>', '不补齐的批次无法变矩形 —— 先 padding，再装箱'],
    ], { head: ['情形', '实测', '为什么'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 4, style: 'padding:10px 12px' }, [tb]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    [cL, cN, cP].forEach(c => { c.style.opacity = '.3'; });
    msg.innerHTML = '<span class="cm">// 同一批数据，三种容器</span>';
    tl.at(2200, () => { cL.style.opacity = '1'; cL.classList.add('ac'); msg.innerHTML = '不传 <em>return_tensors</em> → 原样是 list，<span class="cm">// 连 batch 轴都没有</span>'; });
    tl.at(5600, () => {
      cL.classList.remove('ac'); cN.style.opacity = '1'; cP.style.opacity = '1'; cP.classList.add('ac');
      msg.innerHTML = '传了 <em>"pt"</em> / <em>"np"</em> → 数值<span class="hl">逐位相同</span>，只是容器换了';
    });
    tl.at(9200, () => {
      cP.classList.remove('ac');
      msg.innerHTML = '<em>batch 轴</em>是转换时补的：<span class="cm">// 单条文本 (4,) → (1, 4)</span>';
    });
    tl.at(12800, () => {
      [cL, cN, cP].forEach(c => { c.style.opacity = '1'; });
      msg.innerHTML = '<span class="cm">// 所以：return_tensors 一个 token 都不改，它只决定「装进什么」</span>';
    });
    tl.at(15400, () => {
      msg.innerHTML = '★ 顺序不能反：<em>先 padding 把形状对齐，再装箱</em> —— ragged 批次转张量必抛错';
    });
  },
},

/* ------------------------------------------------- 9 收束 */
{
  kicker: '收束',
  title: '把四段流水线连起来',
  sub: '参数 → 张量 的完整映射；两个验收点的答案都在最后这张表里。',
  caption: '下一课 L2-02：快分词器与纯 Python 分词器各自的 _encode_plus 怎么实现。',
  lang: 'python',
  codeStart: 2472,
  code: `        all_kwargs = {
            "add_special_tokens": add_special_tokens,
            "padding": padding,
            "truncation": truncation,
            "max_length": max_length,
            "stride": stride,
            "is_split_into_words": is_split_into_words,
            "pad_to_multiple_of": pad_to_multiple_of,
            "padding_side": padding_side,
            "return_tensors": return_tensors,
            "return_token_type_ids": return_token_type_ids,
            "return_attention_mask": return_attention_mask,
            "return_overflowing_tokens": return_overflowing_tokens,
            "return_special_tokens_mask": return_special_tokens_mask,
            "return_offsets_mapping": return_offsets_mapping,
            "return_length": return_length,
            "split_special_tokens": kwargs.pop("split_special_tokens", self.split_special_tokens),
            "verbose": verbose,
        }`,
  codeNote: '这 17 个键就是「参数 → 下游」的唯一映射表。',
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '参数 → 它影响的到底是哪一步' }));

    const tb = W.table([
      ['<span class="mono">padding</span>', '② 归一化 → ③ 补位', '<span class="hl4">决定 attention_mask 会不会出现 0</span>'],
      ['<span class="mono">truncation</span>', '② 归一化 → ③ 砍长度', '决定 max_length 是否生效'],
      ['<span class="mono">max_length</span>', '② 归一化（会被静默降级）', '单独出现时会偷偷打开 truncation'],
      ['<span class="mono">return_tensors</span>', '<span class="hl2">④ 装箱</span>', '数值一个都不改，只决定容器'],
      ['<span class="mono">return_attention_mask</span>', '③ padding', '默认值来自类属性 model_input_names'],
      ['<span class="mono">padding_side</span>', '③ padding', '决定 0 补在头部还是尾部'],
      ['<span class="mono">add_special_tokens</span>', '③ 编码之前', '决定 [CLS] / [SEP] 进不进序列'],
      ['<span class="mono">is_split_into_words</span>', '③ 编码之前', '输入已经是词序列时用'],
    ], { head: ['参数', '生效在哪一段', '一句话'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 0, style: 'padding:10px 12px' }, [tb]));

    viz.appendChild(U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' }, [
      W.card({
        cc: 4, tint: 4, title: '验收点 ①：input_ids 与 attention_mask 的产生时机',
        sub: '都在第三段。ids 来自编码；mask 由 <code class="inl">_pad</code> 先补满 1、再按同一个 <code class="inl">difference</code> 补 0 —— 长度天然相等。',
        style: 'flex:1',
      }),
      W.card({
        cc: 1, tint: 1, title: '验收点 ②：return_tensors 影响的是哪一步',
        sub: '第四段。它是 <code class="inl">BatchEncoding.__init__</code> 的构造参数，唯一一次类型转换在 <code class="inl">convert_to_tensors</code> 里。',
        style: 'flex:1',
      }),
    ]));

    viz.appendChild(W.exercise(
      '两条长度 5 与 9 的序列，<code class="inl">padding=True, return_tensors="pt"</code>，'
      + '<code class="inl">input_ids.shape</code> 是多少？<code class="inl">attention_mask</code> 里 0 有几个？',
      '先看第三段：<code class="inl">padding=True</code> → <b>LONGEST</b> → 补齐到本批最长 9。'
      + '所以 <code class="inl">shape = (2, 9)</code>。'
      + '短的那条补 4 个 pad，<code class="inl">attention_mask</code> 同位置补 0 → <b>4 个 0</b>。'
      + '<br>关键：<b>shape 与 0 的个数由同一个 <code class="inl">difference</code> 决定</b>，'
      + '而张量类型要到第四段才被决定 —— 两条序列先对齐，才可能装箱。'));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    const rows = tb.querySelectorAll('tbody tr');
    rows.forEach(tr => { tr.style.opacity = '.3'; tr.style.transition = 'opacity .4s'; });
    msg.innerHTML = '<span class="cm">// 参数 → 段 → 后果</span>';
    tl.at(2200, () => {
      rows.forEach((tr, j) => { tr.style.opacity = (j === 3) ? '1' : '.3'; });
      msg.innerHTML = '<em>return_tensors</em> 只出现在第四段 &nbsp;<span class="cm">// 这是第二个验收点的答案</span>';
    });
    tl.at(5600, () => {
      rows.forEach((tr, j) => { tr.style.opacity = (j <= 2) ? '1' : '.3'; });
      msg.innerHTML = '前三个参数全部在第二段被归一化 &nbsp;<span class="cm">// True / 字符串 / 枚举，出口只有枚举</span>';
    });
    tl.at(9200, () => {
      rows.forEach((tr, j) => { tr.style.opacity = (j === 4 || j === 5) ? '1' : '.3'; });
      msg.innerHTML = '<em>attention_mask</em> 与 <em>padding_side</em> 在第三段生效 &nbsp;<span class="cm">// 这是第一个验收点的答案</span>';
    });
    tl.at(12800, () => {
      rows.forEach(tr => { tr.style.opacity = '1'; });
      msg.innerHTML = '一次调用 = <em>归一化 → 编码 → padding → 装箱</em>；只有 ②③ 改形状，①④ 一个 token 都不改';
    });
    tl.at(16200, () => {
      msg.innerHTML = '<span class="cm">// 下一课 L2-02：快分词器与纯 Python 分词器各自的 _encode_plus</span>';
    });
  },
},

];
