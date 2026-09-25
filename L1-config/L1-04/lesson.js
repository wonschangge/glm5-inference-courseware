/* ==========================================================================
   L1-04 · 配置的序列化、继承与文档生成
   --------------------------------------------------------------------------
   覆盖：utils/hp_naming.py, utils/doc.py, utils/auto_docstring.py,
         utils/type_validators.py（4 个文件 / 6326 行）
   目标：看完能说出「字符串 → 类」这一步在三个地方怎么发生：
         名字（hp_naming）怎么压缩与反解、文档（doc / auto_docstring）
         在类定义时改了什么、注解（type_validators）怎么变成构造期校验。
   排版基准：#visual 可视区 = 994 x 662.7 舞台像素（无头浏览器实测）。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------- 1 全景：谁在定义期改东西 */
{
  kicker: 'L1 · 配置与分派',
  title: '四个文件，一个共同动作：<span class="hl-a">用字符串查类</span>',
  sub: 'L1 前几课读的是 config 里的字段。这一课翻到背面，看字符串是怎么变成可执行事实的。',
  caption: '回顾 L1-01：config 的 to_dict / to_json_string 才是「保类型的序列化」；本课的 hp_naming 是不保类型的命名压缩。',
  lang: 'python',
  codeStart: 145,
  code: `def tensor_type_validator(value: str | TensorType | None = None):
    possible_names = ["pt", "np", "mlx"]
    if value is None:
        pass
    elif not isinstance(value, str) or value not in possible_names:
        raise ValueError(f"The tensor type should be one of {possible_names} but got tensor_type={value}")`,
  codeNote: 'type_validators.py —— 最短的一个校验器：查一张手写表，不通过就 raise。',
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);
    const vg = U.el('div', { class: 'col gap14 vizgrow hstart' });
    wrap.appendChild(vg);

    vg.appendChild(U.el('div', { class: 'klabel', text: '本课的四个文件 —— 前三个在定义期动手，第四个在调用期动手' }));
    const files = [
      { f: 'type_validators.py', n: 271, c: 0, d: '把类型注解变成构造期校验', k: '调用期' },
      { f: 'hp_naming.py', n: 163, c: 1, d: '把参数名压成短名，再反解回来', k: '定义期' },
      { f: 'doc.py', n: 1092, c: 2, d: '一串装饰器：改 __doc__，不动函数体', k: '定义期' },
      { f: 'auto_docstring.py', n: 4800, c: 4, d: '照签名与类名重新生成整份文档', k: '定义期' },
    ];
    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const kids = files.map(x => {
      const c = W.card({ cc: x.c, tint: x.c, title: x.k,
        sub: '<span class="mono" style="font-size:11px">' + x.f + '</span>',
        body: U.el('div', { class: 'col gap6' }, [
          U.el('div', { class: 'n sm', html: '<em>' + x.n + '</em> 行' }),
          U.el('div', { class: 'dim', style: 'font-size:10.5px;line-height:1.5', text: x.d }),
        ]),
        style: 'flex:1 1 0;width:239px' });
      row.appendChild(c); return c;
    });
    vg.appendChild(row);

    const flow = U.el('div', { class: 'row gap10 center', style: 'width:100%;flex-wrap:wrap' });
    flow.innerHTML = U.chip('str / qualname', 0) + U.chip('param_name', 1)
      + U.chip('{placeholder}', 2) + U.chip('类 / 字段 / 文档', 4);
    vg.appendChild(flow);

    const facts = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    facts.appendChild(W.card({ cc: 2, title: '实测 · Glm5NextConfig.__doc__', style: 'flex:1 1 0',
      body: U.el('div', { class: 'mono', style: 'font-size:11.5px;line-height:1.8' }, [
        U.el('div', { html: '源码里只有 <em>r""" … """</em> 那段自定义参数' }),
        U.el('div', { html: '运行期长度 <span class="hl">1901</span> 字符' }),
      ]) }));
    facts.appendChild(W.card({ cc: 4, title: '实测 · 惰性注册表', style: 'flex:1 1 0',
      body: U.el('div', { class: 'mono', style: 'font-size:11.5px;line-height:1.8' }, [
        U.el('div', { html: 'import MODEL_MAPPING：新增 <em>0</em> 个模块' }),
        U.el('div', { html: 'len(MODEL_MAPPING)：新增 <em>2536</em> 个模块' }),
      ]) }));
    vg.appendChild(facts);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    vg.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 四个机制，全部发生在「类定义完成的那一瞬间」</span>';
    const lines = [
      'type_validators：注解 <em>Annotated[T, v()]</em> → 构造时 raise，不进 forward',
      'hp_naming：参数名 → <em>hp_lr0.5</em>，名字里只保留「与默认值不同」的部分',
      'doc.py：<em>fn.__doc__ = …</em>，纯字符串搬运，函数体一行都没被解释',
      '★ auto_docstring：<em>cls.__doc__</em> 整份重写 —— 签名与类名就是唯一事实来源',
    ];
    kids.forEach((c, i) => {
      tl.at(2200 + i * 2300, () => {
        kids.forEach((e, k) => e.classList.toggle('ac', k === i));
        msg.innerHTML = lines[i];
      });
    });
    tl.at(11800, () => {
      kids.forEach(e => e.classList.remove('ac'));
      msg.innerHTML = '<span class="cm">// 定义期改的是「元数据」，调用期改的是「值」—— 这两件事在本课都会遇到</span>';
    });
    tl.at(15200, () => {
      msg.innerHTML = '下一课 L1-05 沿着 <em>MODEL_MAPPING</em> 这条线，看字符串到底怎么找到类';
    });
  },
},

/* ------------------------------------------------- 2 ★ 名字是一张双向映射表 */
{
  kicker: '机制 ① · 命名',
  title: '★ 名字是一张<span class="hl-a">双向映射表</span>，不是一个字符串',
  sub: 'TrialShortNamer 只有三个类属性，但已经决定了整个机制的形状：谁参与、以谁为参照、什么时候发布。',
  caption: '回顾 L1-02：Glm5NextTextConfig 的字段名就是这里的 param_name；短名只是它的压缩形式。',
  lang: 'python',
  codeStart: 19,
  code: `class TrialShortNamer:
    PREFIX = "hp"
    DEFAULTS = {}
    NAMING_INFO = None

    @classmethod
    def set_defaults(cls, prefix, defaults):
        cls.PREFIX = prefix
        cls.DEFAULTS = defaults
        cls.build_naming_info()`,
  codeNote: 'hp_naming.py —— 三个类属性各管一件事，第四个（NAMING_INFO）是算好的映射表。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap16', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const row = U.el('div', { class: 'row gap14', style: 'width:100%;align-items:stretch' });
    const c1 = W.card({ cc: 0, title: 'PREFIX', body: U.el('div', { class: 'col gap6' }, [
      U.el('div', { class: 'mono', style: 'font-size:12px;color:var(--c0)', text: '"hp"' }),
      U.el('div', { class: 'dim', style: 'font-size:11px;line-height:1.5', text: '名字的固定开头，也是 parse_repr 切掉的第一段' })]) });
    const c2 = W.card({ cc: 1, title: 'DEFAULTS', body: U.el('div', { class: 'col gap6' }, [
      U.el('div', { class: 'mono', style: 'font-size:12px;color:var(--c1)', text: '{param: value}' }),
      U.el('div', { class: 'dim', style: 'font-size:11px;line-height:1.5', text: '「什么不写进名字」的判据，也是反解时的兜底值' })]) });
    const c3 = W.card({ cc: 2, title: 'NAMING_INFO', body: U.el('div', { class: 'col gap6' }, [
      U.el('div', { class: 'mono', style: 'font-size:12px;color:var(--c2)', text: 'None → dict' }),
      U.el('div', { class: 'dim', style: 'font-size:11px;line-height:1.5', text: '缓存哨兵：建好之前不发布，发布之后不再重建' })]) });
    [c1, c2, c3].forEach(c => { c.style.flex = '1 1 0'; row.appendChild(c); });
    wrap.appendChild(row);

    const mid = U.el('div', { class: 'col gap10 vizgrow hstart' });
    mid.appendChild(U.el('div', { class: 'klabel', text: '实测：短名里只出现「与默认值不同」的参数（DEFAULTS = {a: 0, b: 0}）' }));
    const tb = W.table([
      ['shortname({a: 0, b: 0})', '<span class="dim">一个参数都没传</span>', "<span class='hl'>'hp'</span>", '默认值不写进名字'],
      ['shortname({a: 4})', 'a 与默认值不同', "<span class='hl'>'hp_a4'</span>", '只写差异'],
      ['shortname({a: 4, b: -3})', '两个都不同', "<span class='hl'>'hp_a4_b-3'</span>", '数字贴着键、其余用 - 隔开'],
      ['parse_repr(…)', '反解 + DEFAULTS 兜底', "<span class='hl'>回填未出现的键</span>", '所以没有 DEFAULTS 就无法反解'],
    ], { head: ['调用', '差异', '结果', '为什么'] });
    mid.appendChild(U.el('div', { class: 'card', cc: 4, style: 'padding:10px 12px' }, [tb]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 三个类属性，三件不同的事</span>';
    tl.at(2400, () => { c1.classList.add('ac'); msg.innerHTML = '<em>PREFIX</em>：名字的固定开头，也是 parse_repr 切掉的第一段'; });
    tl.at(5200, () => { c2.classList.add('ac'); msg.innerHTML = '<em>DEFAULTS</em>：既是「什么不写进名字」的判据，也是反解时的兜底值'; });
    tl.at(8600, () => {
      c3.classList.add('ac');
      msg.innerHTML = '<em>NAMING_INFO = None</em> 是缓存哨兵：<span class="cm">// 建好之前不发布，发布之后不再重建</span>';
    });
    tl.at(12400, () => {
      msg.innerHTML = '所以 <em>shortname({a: 0, b: 0})</em> 只能得到 <em>hp</em> —— 一个「什么都没说」的名字';
    });
    tl.at(15000, () => {
      msg.innerHTML = '<span class="cm">// 名字里丢了信息，就必须靠 DEFAULTS 补回来：这是 parse_repr 的全部工作</span>';
    });
  },
},

/* ------------------------------------------------- 3 短名生成：两级回退 */
{
  kicker: '机制 ① · 命名',
  title: '短名生成：先试<span class="hl-a">无分隔</span>，撞了再退到<span class="hl-b">有分隔</span>',
  sub: '参数按 _ 拆成词，每个词缩到最短的、没被占用的前缀，再拼回去。冲突在「词」和「参数」两级分别处理。',
  caption: '这个两层回退解释了实测里的 lr / lru 之分：先到先得，后者退一位。',
  lang: 'python',
  codeStart: 70,
  code: `    def shortname_for_key(info, param_name):
        words = param_name.split("_")

        shortname_parts = [TrialShortNamer.shortname_for_word(info, word) for word in words]

        # We try to create a separatorless short name, but if there is a collision we have to fallback
        # to a separated short name
        separators = ["", "_"]

        for separator in separators:
            shortname = separator.join(shortname_parts)
            if shortname not in info["reverse_short_param"]:
                info["short_param"][param_name] = shortname
                info["reverse_short_param"][shortname] = param_name
                return shortname

        return param_name`,
  codeNote: '两级回退：词内缩前缀（shortname_for_word），词间换分隔符（separators）。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap16', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const pipe = W.flow([
      { t: 'learning_rate', s: 'param_name', cc: 1 },
      { t: "['learning','rate']", s: 'split("_")', cc: 4 },
      { t: "['l','r']", s: 'shortname_for_word', cc: 0 },
      { t: 'lr', s: 'join("")', cc: 2 },
    ], { style: 'width:100%' });
    wrap.appendChild(pipe);

    const mid = U.el('div', { class: 'col gap12 vizgrow hstart' });
    mid.appendChild(U.el('div', { class: 'klabel', text: '实测：同一个 DEFAULTS，短名分两步被压缩' }));
    const tb = W.table([
      ['learning_rate', 'learning → <em>l</em>', 'rate → <em>r</em>', "<span class='hl'>lr</span>"],
      ['learning_runtime', 'learning → <em>l</em>', "runtime → <em>ru</em>（'r' 已被占）", "<span class='hl'>lru</span>"],
      ['num_train_epochs', 'num → <em>n</em> · train → <em>t</em>', 'epochs → <em>e</em>', "<span class='hl'>nte</span>"],
      ['warmup_ratio', 'warmup → <em>w</em>', 'ratio → <em>ra</em>（"r" 已被占）', "<span class='hl'>wra</span>"],
    ], { head: ['param_name', '词级压缩', '词级压缩（续）', 'short_param'] });
    mid.appendChild(U.el('div', { class: 'card', cc: 0, style: 'padding:10px 12px' }, [tb]));

    const cmp = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    cmp.appendChild(W.card({ cc: 4, title: '为什么按词拆？', style: 'flex:1 1 0',
      sub: '短前缀只在「词」这一级唯一 —— learning 是词，rate 也是词，各缩各的不打架。' }));
    cmp.appendChild(W.card({ cc: 2, title: '为什么两级分隔符？', style: 'flex:1 1 0',
      sub: '分隔符影响可读性与长度。先试净拼（lr），被别人占了才退到 l_r。' }));
    mid.appendChild(cmp);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 第 2 行：先按 _ 拆词</span>';
    tl.at(2400, () => { pipe.focus(0); msg.innerHTML = '<em>learning_rate</em> —— 一个下划线连接的复合键'; });
    tl.at(4600, () => { pipe.focus(1); msg.innerHTML = '拆成 <em>两个词</em>：短前缀的正交性只在这一级成立'; });
    tl.at(7200, () => { pipe.focus(2); msg.innerHTML = '逐词缩到最短未被占用的前缀：<em>l</em> 与 <em>r</em>'; });
    tl.at(9800, () => { pipe.focus(3); msg.innerHTML = '净拼成 <em>lr</em>，登记进 <em>short_param</em> 与 <em>reverse_short_param</em>'; });
    tl.at(12600, () => {
      pipe.focus(-1);
      msg.innerHTML = '<span class="cm">// 冲突分两层处理：词内（前缀加长）与词间（分隔符回退）</span>';
    });
    tl.at(15200, () => {
      msg.innerHTML = '兜底 <em>return param_name</em>：宁可名字长，也不能留下查不到反解的短名';
    });
  },
},

/* ------------------------------------------------- 4 值的字符串化与类型丢失 */
{
  kicker: '机制 ① · 命名',
  title: '值一进名字就变成字符串，<span class="hl-c">类型再也回不来</span>',
  sub: 'sep 只有两种取值，等于给名字定义了一套只认「数字 / 非数字」的语法。parse_repr 必须照着这套语法反解。',
  caption: '这不是通用序列化格式：它的目标是「人类可读的 trial 目录名」，不是 round-trip 保真。',
  lang: 'python',
  codeStart: 126,
  code: `            key = cls.NAMING_INFO["short_param"][k]

            if isinstance(v, bool):
                v = 1 if v else 0

            sep = "" if isinstance(v, (int, float)) else "-"
            e = f"{key}{sep}{v}"
            name.append(e)

        return "_".join(name)`,
  codeNote: 'shortname 的循环体：先降级 bool，再用类型选分隔符。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap16', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const row = U.el('div', { class: 'row gap14', style: 'width:100%;align-items:stretch' });
    const cA = W.card({ cc: 0, title: 'sep = ""', sub: 'int / float —— 数字直接贴着键',
      body: U.el('div', { class: 'mono', style: 'font-size:12px;line-height:1.9' }, [
        U.el('div', { html: "a = 4 &nbsp;→&nbsp; <span class='hl'>hp_a4</span>" }),
        U.el('div', { html: "nte = 5 &nbsp;→&nbsp; <span class='hl'>hp_nte5</span>" }),
      ]), style: 'flex:1 1 0' });
    const cB = W.card({ cc: 3, title: 'sep = "-"', sub: 'bool / str —— 用减号隔开',
      body: U.el('div', { class: 'mono', style: 'font-size:12px;line-height:1.9' }, [
        U.el('div', { html: "b = -3 &nbsp;→&nbsp; <span class='hl'>hp_b-3</span>" }),
        U.el('div', { html: "flag = True &nbsp;→&nbsp; 先降级 &nbsp;→&nbsp; <span class='hl'>hp_flag1</span>" }),
      ]), style: 'flex:1 1 0' });
    row.appendChild(cA); row.appendChild(cB);
    wrap.appendChild(row);

    const mid = U.el('div', { class: 'col gap10 vizgrow hstart' });
    mid.appendChild(U.el('div', { class: 'klabel', text: '实测：parse_repr 把名字读回来时发生了什么' }));
    const tb = W.table([
      ['hp_a4', '<span class="hl">4.0</span>', 'float', '数字分支：int 变成 float'],
      ["hp_b-3", "<span class='hlbad'>'3'</span>", 'str', '减号分支：负号被当成分隔符吃掉'],
      ["hp_a4_b3", "<span class='hlbad'>'3'</span>", 'str', '仍是字符串 —— float() 只用在「无减号」分支'],
      ['hp_a4_b-3', 'ValueError', '—', "could not convert string to float: ''"],
    ], { head: ['名字', 'parse_repr 结果', '类型', '说明'] });
    mid.appendChild(U.el('div', { class: 'card', cc: 3, style: 'padding:10px 12px' }, [tb]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// bool 先降级成 int：isinstance(True, int) 为真，不处理就会生成 hp_flagTrue</span>';
    tl.at(2600, () => { cA.classList.add('ac'); msg.innerHTML = '数字：<em>sep = ""</em>，于是 <em>键与值粘在一起</em>'; });
    tl.at(5400, () => { cB.classList.add('ac'); msg.innerHTML = '其他：<em>sep = "-"</em> —— 与「负数的负号」共用同一个字符'; });
    tl.at(8800, () => {
      cA.classList.remove('ac');
      msg.innerHTML = '反解时用 <em>"-" in value</em> 判断分支：<span class="cm">// 它分不清「分隔符」和「负号」</span>';
    });
    tl.at(12000, () => {
      msg.innerHTML = "★ <em>hp_b-3</em> 单独出现时：float('') 直接抛 <span class='hlbad'>ValueError</span>";
    });
    tl.at(15000, () => {
      msg.innerHTML = '结论：这套命名只保证「自己写出去的名字自己能读回来」，不保证数学性质';
    });
  },
},

/* ------------------------------------------------- 5 doc.py：纯字符串搬运 */
{
  kicker: '机制 ② · 文档',
  title: 'doc.py：改的是 <span class="hl-a">__doc__</span>，函数体一行都没被解释',
  sub: '这些装饰器都返回同一个 fn，只改一个属性 —— 这就是「机械替换」四个字的准确含义。',
  caption: '这一层是纯字符串层：出错了不会有任何异常，只会让文档静默变形。',
  lang: 'python',
  codeStart: 82,
  code: `def add_end_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = (fn.__doc__ if fn.__doc__ is not None else "") + "".join(docstr)
        return fn

    return docstring_decorator`,
  codeNote: 'doc.py 里最典型的「加一串文字然后原样返回」。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap16', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const stage = U.el('div', { class: 'row gap16 vizgrow hstart', style: 'width:100%' });

    const left = U.el('div', { class: 'col gap10', style: 'flex:1 1 0;justify-content:center' });
    left.appendChild(U.el('div', { class: 'klabel', text: '@add_end_docstrings("…") 做的事' }));
    const steps = [
      { t: 'fn.__doc__', s: '读出来（可能是 None，所以要先兜底）', c: 0 },
      { t: '"" .join(docstr)', s: '把新文字接在后面：顺序固定，新内容在后', c: 2 },
      { t: 'return fn', s: '★ 同一个对象，没换也没包', c: 4 },
    ];
    steps.forEach(st => {
      left.appendChild(W.card({ cc: st.c, title: '<span class="mono" style="font-size:11.5px">' + st.t + '</span>',
        sub: st.s, style: 'width:100%' }));
    });
    stage.appendChild(left);

    const right = U.el('div', { class: 'col gap10', style: 'flex:1 1 0;justify-content:center' });
    right.appendChild(U.el('div', { class: 'klabel', text: '两个装饰器对「缺输入」的态度相反' }));
    right.appendChild(W.card({ cc: 4, title: 'add_end_docstrings',
      sub: '<code class="inl">None</code> → 兜底成空串，继续执行。',
      body: U.el('div', { class: 'dim', style: 'font-size:11px;line-height:1.6',
        text: '定位：给已有函数追加标准段落，宁缺勿炸。' }), style: 'width:100%' }));
    const cbad = W.card({ cc: 3, title: 'replace_return_docstrings',
      sub: '找不到 <code class="inl">Returns:</code> 占位行 → <span class="hlbad">raise ValueError</span>',
      body: U.el('div', { class: 'dim', style: 'font-size:11px;line-height:1.6',
        text: '定位：替换返回段必须先有占位，早早炸掉比静默缺文档好。' }), style: 'width:100%' });
    right.appendChild(cbad);
    stage.appendChild(right);
    wrap.appendChild(stage);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 装饰器只碰元数据，不碰行为</span>';
    tl.at(2600, () => { msg.innerHTML = '第一步：读 <em>fn.__doc__</em> —— 它可能是 <em>None</em>，所以要先兜底'; });
    tl.at(5600, () => { msg.innerHTML = '第二步：字符串相加，<em>顺序固定</em>（新内容在后）'; });
    tl.at(8800, () => {
      msg.innerHTML = '★ 第三步：<em>return fn</em> —— 同一个函数对象，调用者感觉不到区别';
    });
    tl.at(12000, () => {
      cbad.classList.add('ac');
      msg.innerHTML = '对照：<em>replace_return_docstrings</em> 找不到占位行会拒绝执行';
    });
    tl.at(14200, () => {
      msg.innerHTML = '<span class="cm">// 一容错一严格，是按用途选的，不是统一风格</span>';
    });
  },
},

/* ------------------------------------------------- 6 auto_docstring：缩进算术化 */
{
  kicker: '机制 ② · 文档',
  title: '缩进也被<span class="hl-a">算术化</span>了：两个方向相反的函数',
  sub: 'equalize_indent 砍到零再统一加；set_min_indent 量出最小缩进再整体平移。用错一个，多行描述就会歪。',
  caption: 'doc.py 用 inspect.getsource 逐字符数缩进；auto_docstring.py 直接算 —— 两代实现并存。',
  lang: 'python',
  codeStart: 2810,
  code: `def get_indent_level(func):
    # Use this instead of \`inspect.getsource(func)\` as getsource can be very slow
    return (len(func.__qualname__.split(".")) - 1) * 4


def equalize_indent(docstring: str, indent_level: int) -> str:
    """
    Adjust the indentation of a docstring to match the specified indent level.
    """
    prefix = " " * indent_level
    # Uses splitlines() (no keepends) to match previous behaviour that dropped
    # any trailing newline via the old splitlines() + "\\n".join() + textwrap.indent path.
    return "\\n".join(prefix + line.lstrip() if line.strip() else "" for line in docstring.splitlines())


def set_min_indent(docstring: str, indent_level: int) -> str:
    """
    Adjust the indentation of a docstring to match the specified indent level.
    """
    # Equivalent to textwrap.dedent + textwrap.indent but avoids the two regex
    # passes that textwrap uses internally (one per call in dedent, one in indent).
    lines = docstring.split("\\n")
    min_indent = min(
        (len(line) - len(line.lstrip()) for line in lines if line.strip()),
        default=0,
    )
    prefix = " " * indent_level
    return "\\n".join(prefix + line[min_indent:] if line.strip() else "" for line in lines)`,
  codeNote: 'get_indent_level 用 __qualname__ 里的点数算缩进：命名即位置。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap16', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const mid = U.el('div', { class: 'col gap12 vizgrow hstart' });
    mid.appendChild(U.el('div', { class: 'klabel', text: '同一个输入，两个函数给同一种结果 —— 但方向相反' }));
    const mkBox = (t, cc) => U.el('div', { class: 'card cc' + cc, style: 'padding:10px 12px;flex:1 1 0' }, [
      U.el('div', { class: 'ct', html: t }),
      U.el('pre', { style: 'font-family:var(--mono);font-size:11px;line-height:1.9;color:var(--ink-dim);margin:8px 0 0;white-space:pre;overflow:hidden' }),
    ]);
    const row = U.el('div', { class: 'row gap14', style: 'width:100%;align-items:stretch' });
    const bIn = mkBox('输入 <span class="dim">（一段多行 description）</span>', 0);
    const bA = mkBox('equalize_indent(s, 4)', 4);
    const bB = mkBox('set_min_indent(s, 4)', 2);
    [bIn, bA, bB].forEach(b => row.appendChild(b));
    mid.appendChild(row);
    const pre = [bIn, bA, bB].map(b => b.querySelector('pre'));
    pre[0].textContent = '描述的第一行：  缩进 8\n描述的后续行：  缩进 12\n（用户写的，缩了几格不知道）';
    pre[1].textContent = '全部 lstrip → 0\n再统一加 prefix = 4\n\n结果：\n    第一行 4\n    后续行 4';
    pre[2].textContent = '量出 min_indent = 8\n每一行整体平移 -8\n再加 prefix = 4\n结果：\n    第一行 4\n    后续行 8';
    bB.classList.add('ac');

    const note = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    note.appendChild(W.card({ cc: 0, title: 'lstrip 版：牺牲相对缩进', style: 'flex:1 1 0',
      sub: '适合「完全由程序生成」的文本 —— 本来就没有需要保留的相对结构。' }));
    note.appendChild(W.card({ cc: 2, title: '平移版：保留相对缩进', style: 'flex:1 1 0',
      sub: '适合「用户写的、不知道缩了几格」的文本 —— 段落内的对齐必须留着。' }));
    mid.appendChild(note);
    wrap.appendChild(mid);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 第 3 行：不用 getsource，因为 getsource 很慢</span>';
    tl.at(2400, () => { bA.classList.add('ac'); msg.innerHTML = '<em>equalize_indent</em>：每行先 <em>lstrip</em>，再加统一前缀'; });
    tl.at(5400, () => { bB.classList.remove('ac'); msg.innerHTML = '<em>set_min_indent</em>：先算全局最小缩进，再整体平移'; });
    tl.at(8600, () => {
      bA.classList.remove('ac'); bB.classList.add('ac');
      msg.innerHTML = '两者结果相同、方向相反：<span class="cm">// 一个做加法，一个做减法</span>';
    });
    tl.at(11800, () => {
      msg.innerHTML = '<em>get_indent_level</em> 数的是 <em>__qualname__</em> 里的点：<em>Glm5NextConfig.__init__</em> → 2 个点 → 4 空格';
    });
    tl.at(15000, () => {
      msg.innerHTML = '<span class="cm">// 名字决定位置 —— 这是本课第三次遇到同一个判据</span>';
    });
  },
},

/* ------------------------------------------------- 7 参数区解析：正则切段落 */
{
  kicker: '机制 ② · 文档',
  title: '参数区解析：<span class="hl-a">文档的语法就是数据结构</span>',
  sub: '四个捕获组切出名字、类型、附加信息、描述；optional / shape / default 都是从描述文字里再抠出来的。',
  caption: '*optional* 写错一个，下游的 return_tensors 判断就会变 —— 文档不是给人看的副产品，它是输入。',
  lang: 'python',
  codeStart: 2896,
  code: `        if max_indent_level == 0:
            param_pattern = _re_param
        else:
            param_pattern = re.compile(
                # |--- Group 1 ---|| Group 2 ||- Group 3 -||---------- Group 4 ----------|
                rf"^\\s{{0,{max_indent_level}}}(\\w+)\\s*\\(\\s*([^, \\)]*)(\\s*.*?)\\s*\\)\\s*:\\s*((?:(?!\\n^\\s{{0,{max_indent_level}}}\\w+\\s*\\().)*)",
                re.DOTALL | re.MULTILINE,
            )
        for match in param_pattern.finditer(args_section):
            param_name = match.group(1)
            param_type = match.group(2)
            additional_info = match.group(3)
            optional = "optional" in additional_info
            shape = parse_shape(additional_info)
            default = parse_default(additional_info)
            param_description = match.group(4).strip()`,
  codeNote: 'G1 名字 / G2 类型 / G3 附加信息 / G4 描述 —— 一行 Args 被拆成四个字段。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap16', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const mid = U.el('div', { class: 'col gap12 vizgrow hstart' });
    mid.appendChild(U.el('div', { class: 'klabel', text: '被解析的一行（来自 Glm5NextConfig 的 Args 段）' }));
    const g1 = '<span class="hl">image_token_id</span>';
    const g2 = '<span class="hl2">int</span>';
    const g3 = ' <span class="hl4">*optional*, defaults to 154854</span>';
    const g4 = '<span class="dim">The image token index to encode the image prompt.</span>';
    const lineBox = U.el('div', { class: 'card', cc: 0, style: 'padding:11px 13px;width:100%' });
    lineBox.innerHTML = '<div class="mono" style="font-size:12px;line-height:2">'
      + g1 + ' (' + g2 + g3 + '):<br>&nbsp;&nbsp;&nbsp;&nbsp;' + g4 + '</div>';
    mid.appendChild(lineBox);

    const tb = W.table([
      ['<span class="hl">1</span>', "(\\w+)", '<code class="inl">image_token_id</code>', '参数名 → 字典的键'],
      ['<span class="hl2">2</span>', '([^, \\)]*)', '<code class="inl">int</code>', '类型：不允许逗号和空格'],
      ['<span class="hl4">3</span>', '(\\s*.*?)', '<code class="inl">*optional*, defaults to 154854</code>', '附加信息 → 再抠三个字段'],
      ['<span class="hl-c">4</span>', '((?!…).)*', '<span class="dim">The image token index …</span>', '描述：吃到下一个参数名为止'],
    ], { head: ['组', '正则片段', '这次命中', '用途'] });
    mid.appendChild(U.el('div', { class: 'card', cc: 2, style: 'padding:10px 12px' }, [tb]));

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    row.appendChild(W.card({ cc: 4, title: 'optional = "optional" in additional_info',
      sub: '不是独立数据，是字符串包含判断。', style: 'flex:1 1 0' }));
    row.appendChild(W.card({ cc: 3, title: 'parse_shape / parse_default',
      sub: '各查一次正则，从同一段文字里再抠出形状与默认值。', style: 'flex:1 1 0' }));
    mid.appendChild(row);
    wrap.appendChild(mid);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 第 4 行那条负向先行断言：(?!\\n^\\s{0,N}\\w+\\s*\\()</span>';
    tl.at(2600, () => { msg.innerHTML = '「一直吃，直到下一行又是一个参数名」—— 用正则切段落，所以必须开 <em>DOTALL</em>'; });
    tl.at(5600, () => { msg.innerHTML = 'G2 <em>([^, \\)]*)</em> 不允许逗号：<em>Union[dict, PreTrainedConfig]</em> 会在这里被截断'; });
    tl.at(9000, () => { msg.innerHTML = 'G3 里再抠出 <em>optional</em> / <em>shape</em> / <em>default</em> —— 三个派生字段'; });
    tl.at(12400, () => {
      msg.innerHTML = '★ 所以：<em>*optional*</em> 少写一个星号，运行时行为就会变 —— 文档是输入，不是产物';
    });
    tl.at(15200, () => {
      msg.innerHTML = '<span class="cm">// max_indent_level == 0 时走预编译的 _re_param：这是会被每个类调用的热路径</span>';
    });
  },
},

/* ------------------------------------------------- 8 类名后缀 → 模板 → 占位符 */
{
  kicker: '机制 ② · 文档',
  title: '类名后缀 → 文档模板 → <span class="hl-a">{占位符}</span>',
  sub: 'ClassDocstring 的每个属性名都是类名后缀；模板里的 {model_name} 还要再查一次 auto 注册表才变成真实类名。',
  caption: '如果文档里出现字面量 {model_class}，那不是 bug —— 是「这个名字当时真的不存在」的诚实记录。',
  lang: 'python',
  codeStart: 2664,
  code: `class ClassDocstring:
    Config = r"""
    This is the configuration class to store the configuration of a {model_base_class}. It is used to instantiate a {model_name}
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the [{model_checkpoint}](https://huggingface.co/{model_checkpoint})

    Configuration objects inherit from [\`PreTrainedConfig\`] and can be used to control the model outputs. Read the
    documentation from [\`PreTrainedConfig\`] for more information.
    """`,
  codeNote: '属性名即后缀：cls.__name__ 以哪个后缀结尾，就取哪一段模板。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap16', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const pipe = W.flow([
      { t: 'Glm5NextConfig', s: 'cls.__name__', cc: 1 },
      { t: 'Config', s: 're.findall(…$)', cc: 4 },
      { t: 'ClassDocstring.Config', s: '取属性', cc: 0 },
      { t: '{model_name}', s: '占位符', cc: 2 },
      { t: 'Glm5NextModel', s: '查 auto 注册表', cc: 3 },
    ], { style: 'width:100%' });
    wrap.appendChild(pipe);

    const mid = U.el('div', { class: 'col gap12 vizgrow hstart' });
    mid.appendChild(U.el('div', { class: 'klabel', text: '实测：Glm5NextConfig.__doc__ 的四段内容（源码里只有最后一段是手写的）' }));
    const tb = W.table([
      ['开场白', '<span class="dim">This is the configuration class … of a</span> <em>Glm5NextModel</em>'],
      ['checkpoint', '<span class="dim">a similar configuration to that of the</span> <em>zai-org/GLM-5.3-Flash</em>'],
      ['继承说明', '<span class="dim">Configuration objects inherit from</span> <em>PreTrainedConfig</em>'],
      ['参数表', '<span class="dim">text_config / vision_config / image_token_id / … 从签名与注解推出</span>'],
      ['总长度', "<span class='hl'>1901</span> <span class='dim'>字符</span>"],
    ], { head: ['段落', '内容'] });
    mid.appendChild(U.el('div', { class: 'card', cc: 4, style: 'padding:10px 12px' }, [tb]));

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cOk = W.card({ cc: 4, title: '查到了', sub: 'PLACEHOLDER_TO_AUTO_MODULE → MODEL_MAPPING_NAMES → 真实类名。',
      style: 'flex:1 1 0' });
    const cNo = W.card({ cc: 3, title: '查不到 / 库没装', sub: '保留占位符原文，绝不抛异常 —— 文档生成不能成为 import 失败的原因。',
      style: 'flex:1 1 0' });
    row.appendChild(cOk); row.appendChild(cNo);
    mid.appendChild(row);
    wrap.appendChild(mid);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 字符串如何找到类：不是查手写的表，而是用类名自己的后缀去命中模板</span>';
    tl.at(2400, () => { pipe.focus(1); msg.innerHTML = '<em>Glm5NextConfig</em> 以 <em>Config</em> 结尾 → 命中配置类模板'; });
    tl.at(5400, () => { pipe.focus(2); msg.innerHTML = '<em>Glm5NextForCausalLM</em> 以 <em>ForCausalLM</em> 结尾 → 命中另一段'; });
    tl.at(8600, () => { pipe.focus(4); msg.innerHTML = '占位符 <em>{model_name}</em> 再查一次 <em>MODEL_MAPPING_NAMES</em> —— 接上 L1-05'; });
    tl.at(11800, () => {
      pipe.focus(-1); cNo.classList.add('ac');
      msg.innerHTML = '查不到就 <em>原样保留占位符</em>：把「当时不存在」写进文档，而不是掩盖它';
    });
    tl.at(15000, () => {
      msg.innerHTML = '★ 一次定义，两处消费：字段只写一遍（带默认值），文档从签名反推出来';
    });
  },
},

/* ------------------------------------------------- 9 分派 + 调用期校验 + 收束 */
{
  kicker: '收束',
  title: '一个判据贯穿全课：<span class="hl-a">名字决定分支</span>',
  sub: '类还是方法、配置类还是图像处理器 —— 全部由 __qualname__ / __name__ / 类型注解决定，而不是靠手写的注册表。',
  caption: '下一课 L1-05：沿着 MODEL_MAPPING 这条线，看字符串到底怎么找到类。',
  lang: 'python',
  codeStart: 4788,
  code: `    def auto_docstring_decorator(obj):
        if len(obj.__qualname__.split(".")) > 1:
            return auto_method_docstring(
                obj, custom_args=custom_args, custom_intro=custom_intro, checkpoint=checkpoint
            )
        else:
            return auto_class_docstring(obj, custom_args=custom_args, custom_intro=custom_intro, checkpoint=checkpoint)`,
  codeNote: '入口分派：与 get_indent_level 完全同一个判据 —— 数 __qualname__ 里的点。',
  duration: 20000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const cards = [
      { cc: 0, n: '①', t: '名字', d: 'shortname → hp_lr0.5', k: 'hp_naming.py' },
      { cc: 1, n: '②', t: '反解', d: 'parse_repr → 参数字典', k: 'hp_naming.py' },
      { cc: 2, n: '③', t: '文档', d: 'fn.__doc__ = 字符串串接', k: 'doc.py' },
      { cc: 4, n: '④', t: '生成', d: 'cls.__doc__ 整份重写', k: 'auto_docstring.py' },
      { cc: 3, n: '⑤', t: '校验', d: 'Annotated[T, v()] → 构造期 raise', k: 'type_validators.py' },
    ];
    const row = U.el('div', { class: 'row gap8', style: 'width:100%;align-items:stretch' });
    const els = cards.map(x => {
      const e = W.card({ cc: x.cc, tint: x.cc, num: null,
        title: '<span class="mono" style="font-size:11px">' + x.n + ' ' + x.t + '</span>',
        sub: '<span class="mono" style="font-size:10.5px">' + x.d + '</span>',
        style: 'flex:1 1 0;width:187px' });
      row.appendChild(e); return e;
    });
    wrap.appendChild(row);

    const mid = U.el('div', { class: 'col gap12 vizgrow hstart' });
    const tb = W.table([
      ['判据', '__qualname__ 里的点数', '类 or 方法 / 缩进层数', '本课 3 次出现'],
      ['判据', 'cls.__name__ 的后缀', 'ClassDocstring 取哪段模板', 'L1-05 继续'],
      ['判据', 'param_name 拆出的词', '短名与反查键', 'L1-07 权重名映射'],
      ['风险', '值被写进名字', "type 丢失：'3' / ValueError", '不是通用序列化'],
      ['风险', '占位符查不到', '原样保留 {model_class}', '诚实 > 美观'],
    ], { head: ['类别', '输入', '决定什么', '接口'] });
    mid.appendChild(U.el('div', { class: 'card', cc: 0, style: 'padding:10px 12px' }, [tb]));

    mid.appendChild(W.exercise(
      '一个配置类的文档里出现了字面量 <code class="inl">{model_class}</code>，'
      + '同时 <code class="inl">LlamaConfig(initializer_range=2.0)</code> 报'
      + ' <code class="inl">StrictDataclassFieldValidationError</code>。这两个现象分别说明什么？',
      '<b>前者</b>：<code class="inl">format_args_docstring</code> 查 '
      + '<code class="inl">PLACEHOLDER_TO_AUTO_MODULE</code> 没查到，走了'
      + '「保留占位符原文」的兜底分支 —— 这个名字当时真的不存在，不是渲染 bug。'
      + '<br><b>后者</b>：<code class="inl">initializer_range: float = interval(min=0.0, max=1.0)(default=0.02)</code>，'
      + '校验器被挂在 dataclass 字段的 <code class="inl">metadata["validator"]</code> 上，'
      + '<b>构造对象时</b>就执行 —— 所以错误在「模型还没建起来」时暴露，而不是在 forward 里。'));
    wrap.appendChild(mid);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    els.forEach(e => { e.style.opacity = '.3'; });
    cards.forEach((x, i) => {
      tl.at(1600 + i * 2400, () => {
        els.forEach((e, k) => { e.style.opacity = (k === i) ? '1' : '.3'; });
        msg.innerHTML = '<em>' + x.n + '</em> ' + x.t + ' &nbsp;<span class="cm">// ' + x.k + '</span>';
      });
    });
    tl.at(13800, () => {
      els.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = '★ 三个判据是同一个思想：<em>让名字自己携带分派信息</em>，不要另写一张表';
    });
    tl.at(16600, () => {
      msg.innerHTML = '但名字会丢信息（<em>hp_b-3</em>）和说谎（<em>{model_class}</em>）—— 所以校验必须放在构造期';
    });
    tl.at(18800, () => {
      msg.innerHTML = '<span class="cm">// 下一课 L1-05：MODEL_MAPPING 这张字符串表长什么样</span>';
    });
  },
},

];
