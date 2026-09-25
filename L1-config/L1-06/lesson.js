/* ==========================================================================
   L1-06 · 其余四张自动注册表
   --------------------------------------------------------------------------
   覆盖：models/auto/{modeling,tokenization,image_processing,video_processing,
        feature_extraction,processing}_auto.py（6 个文件 / 5718 行）
   目标：看完能说出「一个字符串在哪一步变成了一个类」，以及六张表的裁决顺序差在哪。
   排版基准：#visual 可视区 = 994 x 662.7 舞台像素（无头浏览器实测）。
   ========================================================================== */
'use strict';

/* 实测常量（_data/recon/probe_l106.py 的输出，不是估算） */
const M106 = {
  lines: 5718, autoCls: 49, lazy: 54,
  cfg: 718,
  imgAll: 211, vidAll: 45, featAll: 69, procAll: 180,
  tok: 309, tokBackend: 87, badHub: 46,
  model: 556, clm: 178, i2t: 88,
  mods0: 243, mods1: 423,
  hitMods: 3, hitMs: '12.1', warmMs: '0.23',
};

const SCENES = [

/* ------------------------------------------------- 1 全景：六个文件 */
{
  kicker: 'L1 · 配置与分派',
  title: '六个文件，<span class="hl-a">六张表</span>，一个动作',
  sub: 'L1-05 拆完 AutoConfig 与 AutoModel，models/auto/ 只剩这 6 个文件 —— 它们是同一件事的六次复制。',
  caption: '本课的总问题：屏幕上的一个字符串，是在哪一行变成了一个真正被 import 的类？',
  lang: 'python',
  codeStart: 2227,
  code: `class AutoModel(_BaseAutoModelClass):
    _model_mapping = MODEL_MAPPING


AutoModel = auto_class_update(AutoModel)


class AutoModelForPreTraining(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_PRETRAINING_MAPPING


AutoModelForPreTraining = auto_class_update(AutoModelForPreTraining, head_doc="pretraining")


class AutoModelForCausalLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_CAUSAL_LM_MAPPING`,
  codeNote: 'modeling_auto.py L2227~L2242 —— 49 个 Auto* 类里的三个。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const fl = W.flow([
      { t: 'model_type', s: '"glm5_next"', cc: 0 },
      { t: '表', s: 'str → str', cc: 1 },
      { t: '_LazyAutoMapping', s: '按 model_type 对齐', cc: 2 },
      { t: 'AutoXxx[Config]', s: '此刻才 import', cc: 4 },
      { t: '类', s: 'Glm5NextModel', cc: 3 },
    ], { style: 'width:100%' });
    wrap.appendChild(U.el('div', { class: 'col gap8', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: '六张表共用的一条链路' }), fl,
    ]));

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '本课覆盖的六个文件（行数按作业书口径）' }));

    const files = [
      { f: 'modeling_auto.py', n: '2689', d: '48 个任务类 + 54 张映射', c: 0 },
      { f: 'tokenization_auto.py', n: '1039', d: '309 条表 + 注册表', c: 1 },
      { f: 'image_processing_auto.py', n: '780', d: '值 = backend 字典', c: 2 },
      { f: 'video_processing_auto.py', n: '434', d: '整表随环境塌陷', c: 3 },
      { f: 'feature_extraction_auto.py', n: '393', d: '底表 29 + 补丁 40', c: 4 },
      { f: 'processing_auto.py', n: '383', d: '一个门面三个子处理器', c: 5 },
    ];
    const cards = [];
    for (let r = 0; r < 2; r++) {
      const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
      files.slice(r * 3, r * 3 + 3).forEach(x => {
        const c = W.card({ cc: x.c, tint: x.c,
          title: '<span class="mono" style="font-size:11px;overflow-wrap:anywhere">' + x.f + '</span>',
          sub: '<span class="mono dim" style="font-size:10px">' + x.n + ' 行</span>',
          body: U.el('div', { class: 'mono', style: 'font-size:10.5px;color:var(--ink-faint);margin-top:3px', text: x.d }),
          style: 'flex:1 1 0' });
        row.appendChild(c); cards.push(c);
      });
      viz.appendChild(row);
    }

    const stats = [
      { k: 'modeling_auto 里的 Auto* 类', v: M106.autoCls, s: '48 个 AutoModel* + AutoBackbone', c: 0 },
      { k: '_LazyAutoMapping 实例', v: M106.lazy, s: '同一个类，被 new 了 54 次', c: 2 },
      { k: '六个文件合计', v: M106.lines, s: '行（作业书口径）', c: 4 },
    ];
    const srow = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const sEls = stats.map(st => {
      const c = W.card({ cc: st.c, tint: st.c,
        title: '<span class="mono" style="font-size:10.5px">' + st.k + '</span>',
        body: U.el('div', { class: 'n big', text: U.fmt(st.v) }),
        sub: '<span class="mono dim" style="font-size:10px">' + st.s + '</span>', style: 'flex:1 1 0' });
      srow.appendChild(c); return c;
    });
    viz.appendChild(srow);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    fl.focus(-1);
    cards.forEach(c => { c.style.opacity = '.35'; });
    msg.innerHTML = '<span class="cm">// 六个文件、六张表，写法完全对称</span>';
    tl.at(1400, () => { fl.focus(0); msg.innerHTML = '起点永远是一个字符串：config 里的 <em>model_type</em>'; });
    tl.at(3800, () => { fl.focus(1); msg.innerHTML = '第一跳：查表 —— 表里<em>只有字符串</em>，没有任何类对象'; });
    tl.at(6400, () => { fl.focus(2); msg.innerHTML = '<em>_LazyAutoMapping</em> 把「config 名」和「类名」两张表按 model_type 对齐'; });
    tl.at(9200, () => { fl.focus(3); msg.innerHTML = '第二跳：真的要用这个类了，<em>才 import</em> 那个模型包'; });
    tl.at(12000, () => {
      fl.focus(4); cards.forEach(c => { c.style.opacity = '1'; });
      msg.innerHTML = '六个文件，就是这同一条链路的六次复制';
    });
    tl.at(14600, () => {
      fl.focus(-1); sEls.forEach(e => e.classList.add('ac'));
      msg.innerHTML = '<span class="cm">// 49 个类 / 54 张映射 / 5718 行 —— 先记住这三个数</span>';
    });
  },
},

/* ------------------------------------------------- 2 config 决定形状 */
{
  kicker: 'modeling_auto.py · 落点',
  title: 'config 决定形状：GLM-5 <span class="hl-a">落在哪张任务表</span>',
  sub: '同一个 Glm5NextConfig，进不同的 Auto 类会拿到不同的模型类 —— 这不是猜的，是三张表都查了一遍。',
  caption: '表描述的是「哪个 config 走哪个类」，不是「哪个类属于哪个模型」。',
  lang: 'python',
  codeStart: 2451,
  code: `class AutoModelForImageTextToText(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING

    # override to give better return typehint
    @classmethod
    def from_pretrained(
        cls: type["AutoModelForImageTextToText"],
        pretrained_model_name_or_path: str | os.PathLike[str],
        *model_args,
        **kwargs,
    ) -> "_BaseModelWithGenerate":
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


AutoModelForImageTextToText = auto_class_update(AutoModelForImageTextToText, head_doc="image-text-to-text modeling")`,
  codeNote: 'AutoModelForImageTextToText —— GLM-5 真正命中的那个门面。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '实测：三张表对 Glm5NextConfig 的三个答案' }));

    const tb = W.table([
      ['MODEL_MAPPING', '556 条', '<span class="hl4">✓ 命中</span>',
        '<span class="mono">Glm5NextModel</span>', '裸主干（L0-01 里那个 forward 的出口）'],
      ['MODEL_FOR_CAUSAL_LM_MAPPING', '178 条', '<span class="hlbad">✗ 不在</span>',
        '<span class="mono dim">——</span>', '它不是「纯文本因果 LM」'],
      ['MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING', '88 条', '<span class="hl4">✓ 命中</span>',
        '<span class="mono">Glm5NextForConditionalGeneration</span>', '带 lm_head，可 generate'],
    ], { head: ['表', '规模', 'Glm5NextConfig', '拿到的类', '是什么'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 1, style: 'padding:10px 12px' }, [tb]));
    const trs = Array.from(tb.querySelectorAll('tbody tr'));

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cA = W.card({ cc: 4, tint: 4, title: 'AutoModel.from_pretrained',
      sub: '只要 config 能对上就拿主干。L0-01 里的 Glm5NextTextModel.forward 就是它。', style: 'flex:1 1 0' });
    const cB = W.card({ cc: 2, tint: 2, title: 'AutoModelForImageTextToText.from_pretrained',
      sub: '同一份权重，多一个 lm_head。上面那段代码把返回类型收窄成 _BaseModelWithGenerate，正是为了它。',
      style: 'flex:1 1 0' });
    row.appendChild(cA); row.appendChild(cB);
    viz.appendChild(row);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 同一个 config，三种问法</span>';
    tl.at(1800, () => {
      trs.forEach((t, i) => t.classList.toggle('ac', i === 0));
      msg.innerHTML = 'AutoModel → <em>Glm5NextModel</em> &nbsp;<span class="cm">// 只要主干</span>';
    });
    tl.at(5200, () => {
      trs.forEach((t, i) => t.classList.toggle('ac', i === 1));
      msg.innerHTML = '<span class="hlbad">AutoModelForCausalLM 选不中它</span> &nbsp;<span class="cm">// 178 条表里没有 glm5_next</span>';
    });
    tl.at(8600, () => {
      trs.forEach((t, i) => t.classList.toggle('ac', i === 2));
      cA.classList.add('ac'); cB.classList.add('ac');
      msg.innerHTML = 'AutoModelForImageTextToText → <em>Glm5NextForConditionalGeneration</em>';
    });
    tl.at(12000, () => {
      cA.classList.remove('ac');
      msg.innerHTML = '<span class="cm">// 回顾 L0-01：hc_head 之后接 lm_head，就是这一层加上去的</span>';
    });
    tl.at(14600, () => {
      trs.forEach(t => t.classList.remove('ac'));
      msg.innerHTML = '「config 决定形状」= 同一个 config，进不同的表，长出不同的模型';
    });
  },
},

/* ------------------------------------------------- 3 ★ 惰性映射 */
{
  kicker: '★ 共用零件',
  title: '读表零 import，<span class="hl-a">取格子</span>才 import 那一个包',
  sub: '54 张映射全是 _LazyAutoMapping(CONFIG_MAPPING_NAMES, 某张表) —— 表里装的是名字，不是类。',
  caption: '省下的是「把 556 个模型全 import 一遍」的钱；代价是不能当普通 dict 遍历。',
  lang: 'python',
  codeStart: 2084,
  code: `MODEL_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_MAPPING_NAMES)
MODEL_FOR_PRETRAINING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_PRETRAINING_MAPPING_NAMES)
MODEL_FOR_CAUSAL_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)`,
  codeNote: '三行、三个映射 —— 左边那张表和右边那张表都是 str → str。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: 'transformers.* 模块数（实测，import 时 vs 全部摸一遍）' }));

    const bars = W.bars([
      { label: 'import transformers', value: M106.mods0, cc: 0, valueText: String(M106.mods0) },
      { label: '把用到的都取一遍', value: M106.mods1, cc: 3, valueText: String(M106.mods1) },
    ], { max: M106.mods1 });
    viz.appendChild(U.el('div', { class: 'card', cc: 0, style: 'padding:10px 12px' }, [bars]));

    const bills = [
      { t: '读表 MODEL_MAPPING_NAMES["glm5_next"]', v: '+0 个模块', d: '拿到的还是字符串 \'Glm5NextModel\'', c: 4 },
      { t: '取格子 AutoModel._model_mapping[Glm5NextConfig]', v: '+' + M106.hitMods + ' 个模块 / ' + M106.hitMs + ' ms', d: '这时候才 import glm5_next 那个包', c: 2 },
      { t: '紧接着取 Glm5NextTextConfig', v: '+0 个模块 / ' + M106.warmMs + ' ms', d: '归一化后是同一个包，模块已缓存', c: 1 },
    ];
    const srow = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const sEls = bills.map(b => {
      const c = W.card({ cc: b.c, tint: b.c,
        title: '<span class="mono" style="font-size:10.5px">' + b.t + '</span>',
        sub: '<span class="mono" style="font-size:11px">' + b.v + '</span> &nbsp;' + b.d,
        style: 'flex:1 1 0' });
      srow.appendChild(c); return c;
    });
    viz.appendChild(srow);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 243 个模块是起点，不是终点</span>';
    tl.at(1800, () => { bars.setTo(0, 1); msg.innerHTML = '<em>import transformers</em> 之后：' + M106.mods0 + ' 个 transformers.* 模块'; });
    tl.at(5200, () => { bars.setTo(1, 1); msg.innerHTML = '把本课要用的类都取一遍：<em>' + M106.mods1 + '</em> 个 &nbsp;<span class="cm">// 差值是惰性省下来的</span>'; });
    tl.at(8600, () => { sEls[0].classList.add('ac'); msg.innerHTML = '读表：<em>0</em> 个模块 —— 表里只有名字'; });
    tl.at(11600, () => {
      sEls[0].classList.remove('ac'); sEls[1].classList.add('ac');
      msg.innerHTML = '取格子：<em>+' + M106.hitMods + ' 个模块</em>、' + M106.hitMs + ' ms —— 这才是 import 真正发生的地方';
    });
    tl.at(14400, () => {
      sEls[1].classList.remove('ac'); sEls[2].classList.add('ac');
      msg.innerHTML = '第二次取同一个包：<em>0</em> 个模块 —— 惰性映射自己也缓存模块';
    });
  },
},

/* ------------------------------------------------- 4 ★ 底表 + 补丁 */
{
  kicker: '★ 表的来源',
  title: '底表是<span class="hl-a">生成的</span>，补丁是<span class="hl-b">手写的</span>',
  sub: 'auto_mappings.py 由各 config 的 model_type 自动生成；推导不出来的例外，由每个 auto 文件自己 update 进去。',
  caption: '补丁不是拷贝一份新表，而是打在同一个 dict 对象上 —— 所以「表有多少条」取决于哪些模块被 import 过。',
  lang: 'python',
  codeStart: 90,
  code: `FEATURE_EXTRACTOR_MAPPING_NAMES.update(MISSING_FEATURE_EXTRACTOR_MAPPING_NAMES)
FEATURE_EXTRACTOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, FEATURE_EXTRACTOR_MAPPING_NAMES)


def feature_extractor_class_from_name(class_name: str):
    for module_name, extractors in FEATURE_EXTRACTOR_MAPPING_NAMES.items():`,
  codeNote: 'feature_extraction_auto.py L90~L95 —— update 一行，底表从 29 条变 69 条。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '四张底表 import 各自的 auto 模块之后（实测长度）' }));

    const bars = W.bars([
      { label: 'IMAGE_PROCESSOR', value: M106.imgAll, cc: 2, valueText: '120 → 211' },
      { label: 'PROCESSOR', value: M106.procAll, cc: 1, valueText: '147 → 180' },
      { label: 'FEATURE_EXTRACTOR', value: M106.featAll, cc: 4, valueText: '29 → 69' },
      { label: 'VIDEO_PROCESSOR', value: M106.vidAll, cc: 3, valueText: '31 → 45' },
    ], { max: M106.imgAll });
    viz.appendChild(U.el('div', { class: 'card', cc: 2, style: 'padding:10px 12px' }, [bars]));

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    row.appendChild(W.card({ cc: 3, tint: 3, title: 'auto_mappings.py 的头部写着什么',
      sub: '这是一个带警告框的「本文件自动生成，不要手改」——生成它的是各 config 的 cls.model_type。',
      style: 'flex:1 1 0' }));
    row.appendChild(W.card({ cc: 4, tint: 4, title: '为什么还需要补丁表',
      sub: '「一个 config 对应多个处理器」「类名不符合推导规则」这些例外，机器推不出来，只能手写 MISSING_*。',
      style: 'flex:1 1 0' }));
    viz.appendChild(row);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 四条柱子，起点都是 auto_mappings.py 里的底表</span>';
    tl.at(1900, () => { bars.setTo(0, 1); msg.innerHTML = '<em>IMAGE_PROCESSOR</em>：120 → 211（+91）'; });
    tl.at(4600, () => { bars.setTo(1, 1); msg.innerHTML = '<em>PROCESSOR</em>：147 → 180（+33）'; });
    tl.at(7300, () => { bars.setTo(2, 1); msg.innerHTML = '<em>FEATURE_EXTRACTOR</em>：29 → 69（+40）'; });
    tl.at(10000, () => { bars.setTo(3, 1); msg.innerHTML = '<em>VIDEO_PROCESSOR</em>：31 → 45（+14）'; });
    tl.at(12800, () => {
      bars.setAll(1);
      msg.innerHTML = '<span class="cm">// 四行 update，改的是同一个 dict 对象 —— 不是拷贝</span>';
    });
    tl.at(15000, () => { msg.innerHTML = '所以「表里有多少条」是一个<em>随 import 顺序变化</em>的量'; });
  },
},

/* ------------------------------------------------- 5 ★ 命名冲突 */
{
  kicker: '★ 命名冲突',
  title: '同一个类名，<span class="hl-a">五级</span>裁决顺序',
  sub: 'tokenizer_class_from_name() 是全仓库优先级层次最多的反查函数：注册表 > 兜底 > 表。',
  caption: '默认相信 Hub 上的 tokenizer_config.json，但对 46 个 model_type，内置表说了算。',
  lang: 'python',
  codeStart: 497,
  code: `def tokenizer_class_from_name(class_name: str) -> type[Any] | None:
    # Bloom tokenizer classes were removed but should map to the fast backend for BC
    if class_name in {"BloomTokenizer", "BloomTokenizerFast"}:
        return TokenizersBackend

    if class_name in REGISTERED_FAST_ALIASES:
        return REGISTERED_FAST_ALIASES[class_name]

    if class_name in REGISTERED_TOKENIZER_CLASSES:
        return REGISTERED_TOKENIZER_CLASSES[class_name]

    if class_name == "TokenizersBackend":
        return TokenizersBackend`,
  codeNote: 'tokenization_auto.py L497~L509 —— 前四道闸门都在大表之前。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const vizg = U.el('div', { class: 'vizgrow hstart' });
    wrap.appendChild(vizg);
    const viz = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:flex-start' });
    vizg.appendChild(viz);

    const left = U.el('div', { class: 'col gap8', style: 'flex:1 1 0;min-width:0' });
    left.appendChild(U.el('div', { class: 'klabel', text: '一个类名进来，按这个顺序找' }));
    const levels = [
      { n: 1, t: 'BloomTokenizer 历史别名', d: 'v5 删掉了慢分词器，别名指向 TokenizersBackend', c: 0 },
      { n: 2, t: 'REGISTERED_FAST_ALIASES', d: '用户注册的别名 —— 最优先', c: 4 },
      { n: 3, t: 'REGISTERED_TOKENIZER_CLASSES', d: 'AutoTokenizer.register() 塞进来的类（实测初始规模 0）', c: 4 },
      { n: 4, t: 'TokenizersBackend 这个兜底常量', d: '表里 87 条的值就是它 —— 直接返回，不查表', c: 2 },
      { n: 5, t: '遍历 TOKENIZER_MAPPING_NAMES', d: '309 条大表逐条比类名 —— 最后才轮到它', c: 1 },
    ];
    const lEls = levels.map(L => {
      const c = W.card({ cc: L.c, num: L.n, title: '<span class="mono" style="font-size:11px">' + L.t + '</span>',
        sub: L.d, style: 'width:100%' });
      left.appendChild(c); return c;
    });
    viz.appendChild(left);

    const right = U.el('div', { class: 'col gap10', style: 'flex:0 0 300px' });
    right.appendChild(U.el('div', { class: 'klabel', text: '实测数字' }));
    const stats = [
      { v: M106.tok, k: 'TOKENIZER_MAPPING_NAMES 条数', c: 1 },
      { v: M106.tokBackend, k: '其中值 = TokenizersBackend', c: 2 },
      { v: M106.badHub, k: 'Hub 说的不算数的 model_type', c: 3 },
    ];
    const rEls = stats.map(s => {
      const c = W.card({ cc: s.c, tint: s.c,
        title: '<span class="mono" style="font-size:10px">' + s.k + '</span>',
        body: U.el('div', { class: 'n big', text: U.fmt(s.v) }), style: 'width:100%' });
      right.appendChild(c); return c;
    });
    right.appendChild(U.el('div', { class: 'card', cc: 3, style: 'padding:9px 11px' }, [
      U.el('div', { class: 'mono', style: 'font-size:10.5px;color:var(--ink-dim);overflow-wrap:anywhere',
        text: '末尾还有一手：XxxFast 找不到，就把 Fast 去掉再来一次（v5 合并 fast/slow 的兼容层）。' }),
    ]));
    viz.appendChild(right);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    lEls.forEach(e => { e.style.opacity = '.4'; });
    msg.innerHTML = '<span class="cm">// 五道闸门，从上往下</span>';
    lEls.forEach((e, i) => {
      tl.at(1600 + i * 2500, () => {
        lEls.forEach((x, k) => { x.style.opacity = (k === i) ? '1' : '.4'; });
        msg.innerHTML = '第 <em>' + (i + 1) + '</em> 级：' + levels[i].t
          + ' &nbsp;<span class="cm">// ' + levels[i].d + '</span>';
      });
    });
    tl.at(14200, () => {
      lEls.forEach(e => { e.style.opacity = '1'; });
      rEls[2].classList.add('ac');
      msg.innerHTML = '<em>' + M106.badHub + '</em> 个 model_type：Hub 上写的类名被认为有错，强制走内置表';
    });
    tl.at(16200, () => {
      rEls[2].classList.remove('ac');
      msg.innerHTML = '<span class="cm">// 注册表排在表前面 —— 这是「用户 > 库」的直接体现</span>';
    });
  },
},

/* ------------------------------------------------- 6 image：backend 字典 */
{
  kicker: 'image_processing_auto.py',
  title: '这张表的值不是字符串，是 <span class="hl-a">backend 字典</span>',
  sub: '同一个 model_type 有两个类：torchvision 版和 PIL 版。取类之前必须先回答「用哪个 backend」。',
  caption: 'backend 不是性能开关，是输出契约 —— 两套实现给不出逐位相同的张量。',
  lang: 'python',
  codeStart: 333,
  code: `def _resolve_backend(backend: str | None, use_fast: bool | None, base_class_name: str | None) -> str:
    """Resolve raw backend inputs to a concrete backend name ('torchvision' or 'pil').

    Handles, in order:
    - Deprecated \`\`use_fast\`\` flag: warns and converts to an explicit backend string when no
      explicit backend is given.
    - Explicit backend string: returned as-is.
    - None resolution: forces 'pil' for processors in DEFAULT_TO_PIL_BACKEND_IMAGE_PROCESSORS
      (Lanczos interpolation, unsupported by torchvision < 0.27); otherwise picks 'torchvision'
      when available, falling back to 'pil'.
    """
    if use_fast is not None:
        logger.warning_once(
            "The \`use_fast\` parameter is deprecated and will be removed in a future version. "
            'Use \`backend="torchvision"\` instead of \`use_fast=True\`, or \`backend="pil"\` instead of \`use_fast=False\`.'
        )
        if backend is None:
            backend = "torchvision" if use_fast else "pil"

    if backend is None:
        if base_class_name in DEFAULT_TO_PIL_BACKEND_IMAGE_PROCESSORS:
            return "pil"
        return "torchvision" if is_torchvision_available() else "pil"

    return backend`,
  codeNote: '_resolve_backend —— 三行判定，决定后面去查字典的哪个键。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '实测 IMAGE_PROCESSOR_MAPPING_NAMES["glm5_next"] 的两个键' }));

    const tb = W.table([
      ['"torchvision"', '<span class="mono">Glm5NextImageProcessor</span>', 'GPU 加速，默认首选（实测 _resolve_backend(None, None, ...) → torchvision）'],
      ['"pil"', '<span class="mono">Glm5NextImageProcessorPil</span>', '显式指定 backend="pil"，或处理器在 Lanczos 名单里'],
    ], { head: ['backend 键', '类', '什么时候选它'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 2, style: 'padding:10px 12px' }, [tb]));
    const trs = Array.from(tb.querySelectorAll('tbody tr'));

    const chain = U.el('div', { class: 'row gap10 wrap', style: 'width:100%' });
    chain.innerHTML =
      U.chip('① use_fast（已弃用）→ 换算成 backend', 3) +
      U.chip('② 显式 backend 原样返回', 0) +
      U.chip('③ Lanczos 名单 → 强制 pil', 4) +
      U.chip('④ 否则 torchvision 可用就用它', 2);
    viz.appendChild(U.el('div', { class: 'col gap6', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: '_resolve_backend 的判定顺序' }), chain,
    ]));

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    row.appendChild(W.card({ cc: 4, tint: 4, title: '选不到就往下退',
      sub: '拿到 backend 后由 _load_class_with_fallback 逐个试：先你要的，再另一个，最后任何自定义 backend。',
      style: 'flex:1 1 0' }));
    row.appendChild(W.card({ cc: 3, tint: 3, title: '退不动才报错',
      sub: '依赖缺失时延迟加载给出的是 Dummy 类，靠 is_dummy 标记跳过 —— 不会等到实例化才炸。',
      style: 'flex:1 1 0' }));
    viz.appendChild(row);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    trs.forEach(t => { t.style.opacity = '.45'; });
    msg.innerHTML = '<span class="cm">// 表里这一格，装的是「按环境二选一」的策略</span>';
    tl.at(2000, () => {
      trs.forEach((t, i) => { t.style.opacity = i === 0 ? '1' : '.45'; });
      msg.innerHTML = '默认：<em>torchvision</em> 键上的 Glm5NextImageProcessor';
    });
    tl.at(5400, () => {
      trs.forEach((t, i) => { t.style.opacity = i === 1 ? '1' : '.45'; });
      msg.innerHTML = '指定 <em>backend="pil"</em>（或环境没有 torchvision）→ 换成 Glm5NextImageProcessorPil';
    });
    tl.at(9000, () => {
      trs.forEach(t => { t.style.opacity = '1'; });
      msg.innerHTML = '<span class="cm">// 这是六张表里唯一一张「值有内部结构」的表</span>';
    });
    tl.at(12400, () => { msg.innerHTML = '为什么显式命名？因为两套 backend 的数值结果不同 —— 它是<em>输出契约</em>'; });
    tl.at(15000, () => { msg.innerHTML = '回顾 L2：图像预处理出来的张量形状，决定后面视觉塔看到什么'; });
  },
},

/* ------------------------------------------------- 7 video：环境写进表里 */
{
  kicker: 'video_processing_auto.py',
  title: '可用性被<span class="hl-a">烤进表里</span>：45 条一起塌',
  sub: 'video 表的值是纯字符串，但 import 时有一段循环，会按 torchvision 在不在，把整张表重写一遍。',
  caption: '本机 torchvision 可用：45 条全是类名，None 的条数为 0 —— 反面是把开关换成 False 再 reload 实测的。',
  lang: 'python',
  codeStart: 74,
  code: `    VIDEO_PROCESSOR_MAPPING_NAMES.update(MISSING_VIDEO_PROCESSOR_MAPPING_NAMES)

for model_type, video_processors in VIDEO_PROCESSOR_MAPPING_NAMES.items():
    fast_video_processor_class = video_processors

    # If the torchvision is not available, we set it to None
    if not is_torchvision_available():
        fast_video_processor_class = None

    VIDEO_PROCESSOR_MAPPING_NAMES[model_type] = fast_video_processor_class

VIDEO_PROCESSOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, VIDEO_PROCESSOR_MAPPING_NAMES)`,
  codeNote: 'video_processing_auto.py L74~L85 —— 没有 break，每一条都重新赋值。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: 'VIDEO_PROCESSOR_MAPPING_NAMES 的 45 条（一格一条）' }));

    const strip = U.el('div', { class: 'row gap3', style: 'width:100%;height:34px' });
    const cells = [];
    for (let i = 0; i < M106.vidAll; i++) {
      const e = U.el('div', {
        style: 'flex:1 1 0;min-width:0;height:30px;border-radius:5px;border:1px solid rgba(52,211,153,.55);'
             + 'background:rgba(52,211,153,.18);transition:background .45s,border-color .45s,opacity .45s',
        title: '第 ' + (i + 1) + ' 条' });
      strip.appendChild(e); cells.push(e);
    }
    viz.appendChild(strip);

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cOn = W.card({ cc: 4, tint: 4, title: 'torchvision 可用（本机实测）',
      body: U.el('div', { class: 'mono', style: 'font-size:11px;line-height:1.6' }, [
        U.el('div', { html: '值为 None 的条数：<span class="hl4">0 / 45</span>' }),
        U.el('div', { html: 'VIDEO_PROCESSOR_MAPPING[Glm5NextConfig]' }),
        U.el('div', { class: 'dim', html: '→ Glm5NextVideoProcessor' }),
      ]), style: 'flex:1 1 0' });
    const cOff = W.card({ cc: 3, tint: 3, title: '把开关换成 False 再 reload',
      body: U.el('div', { class: 'mono', style: 'font-size:11px;line-height:1.6' }, [
        U.el('div', { html: '值为 None 的条数：<span class="hlbad">45 / 45</span>' }),
        U.el('div', { html: '.get(Glm5NextConfig) → None' }),
        U.el('div', { class: 'dim', html: 'auto_mappings 里的同名对象也一起变' }),
      ]), style: 'flex:1 1 0' });
    row.appendChild(cOn); row.appendChild(cOff);
    viz.appendChild(row);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 45 条，本机全是类名</span>';
    tl.at(2200, () => {
      cOn.classList.add('ac');
      msg.innerHTML = '<em>is_torchvision_available() == True</em> → 表里的值原样保留';
    });
    tl.at(6000, () => {
      cOn.classList.remove('ac'); cOff.classList.add('ac');
      cells.forEach((c, i) => {
        c.style.transitionDelay = (i * 12) + 'ms';
        c.style.borderColor = 'rgba(251,113,133,.55)';
        c.style.background = 'rgba(251,113,133,.14)';
      });
      msg.innerHTML = '把开关换成 <em>False</em> 再 reload：45 条<em>全部</em>写成 None';
    });
    tl.at(10200, () => {
      cells.forEach(c => { c.style.transitionDelay = '0ms'; });
      msg.innerHTML = '<span class="cm">// 这不是运行时判断，是被烤进表的数据里</span>';
    });
    tl.at(13200, () => {
      cOff.classList.add('ac'); cOn.classList.add('ac');
      msg.innerHTML = '同一个 dict 对象：auto_mappings 里的那份也一起变成 <em>45/45 个 None</em>';
    });
    tl.at(15800, () => {
      cells.forEach(c => {
        c.style.borderColor = 'rgba(52,211,153,.55)';
        c.style.background = 'rgba(52,211,153,.18)';
      });
      cOn.classList.remove('ac'); cOff.classList.remove('ac');
      msg.innerHTML = '对比第 4 幕：feature 的补丁是<em>加键</em>，这里是<em>改值</em> —— 同样是表，一处是索引，一处是策略';
    });
  },
},

/* ------------------------------------------------- 8 AutoProcessor */
{
  kicker: 'processing_auto.py',
  title: 'AutoProcessor：先认<span class="hl-a">文件</span>，再认类',
  sub: '它要同时当「多模态处理器工厂」和「纯 tokenizer 工厂」，办法是把配置文件排出一个先后顺序。',
  caption: '越靠近「专门描述处理器」的文件越优先；三级都落空时，还有一条四连兜底。',
  lang: 'python',
  codeStart: 216,
  code: `        processor_class = None
        processor_auto_map = None

        # First, let's see if we have a processor or preprocessor config.
        # Filter the kwargs for \`cached_file\`.
        _hub_valid_kwargs = (
            "cache_dir",
            "force_download",
            "proxies",
            "token",
            "revision",
            "local_files_only",
            "subfolder",
            "repo_type",
            "user_agent",
        )
        cached_file_kwargs = {key: kwargs[key] for key in _hub_valid_kwargs if key in kwargs}
        # We don't want to raise
        cached_file_kwargs.update(
            {
                "_raise_exceptions_for_gated_repo": False,
                "_raise_exceptions_for_missing_entries": False,
                "_raise_exceptions_for_connection_errors": False,
            }
        )

        # Let's start by checking whether the processor class is saved in a processor config
        processor_config_file = cached_file(pretrained_model_name_or_path, PROCESSOR_NAME, **cached_file_kwargs)
        if processor_config_file is not None:
            config_dict, _ = ProcessorMixin.get_processor_dict(pretrained_model_name_or_path, **kwargs)
            processor_class = config_dict.get("processor_class")
            if "AutoProcessor" in config_dict.get("auto_map", {}):
                processor_auto_map = config_dict["auto_map"]["AutoProcessor"]`,
  codeNote: 'processing_auto.py L216~L248 —— 第一个被探测的文件就是 processor_config.json。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const vizg = U.el('div', { class: 'vizgrow hstart' });
    wrap.appendChild(vizg);
    const viz = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:flex-start' });
    vizg.appendChild(viz);

    const left = U.el('div', { class: 'col gap8', style: 'flex:1 1 0;min-width:0' });
    left.appendChild(U.el('div', { class: 'klabel', text: '文件优先级（从上往下探）' }));
    const files = [
      { n: 'processor_config.json', d: 'processor_class 直接写在里面 —— 命中就结束', c: 1 },
      { n: 'preprocessor_config.json', d: '图像处理器配置，顺带记着 processor_class', c: 2 },
      { n: 'video_processor.json', d: '只有视频处理器的模型走这条', c: 3 },
      { n: 'tokenizer_config.json', d: '纯文本模型常常只到这里', c: 0 },
      { n: 'config.json 的 processor_class 属性', d: '最后兜底：连处理器配置文件都没有时', c: 4 },
    ];
    const tb = W.table(files.map((x, i) => [
      '<span class="mono" style="font-size:11px">' + (i + 1) + '</span>',
      '<span class="mono" style="font-size:11px;overflow-wrap:anywhere">' + x.n + '</span>',
      '<span style="font-family:var(--sans);font-size:11px">' + x.d + '</span>',
    ]), { head: ['#', '先看哪个文件', '说明'] });
    left.appendChild(U.el('div', { class: 'card', cc: 1, style: 'padding:10px 12px' }, [tb]));
    const fEls = Array.from(tb.querySelectorAll('tbody tr'));
    left.appendChild(U.el('div', { class: 'card', cc: 5, style: 'padding:9px 11px' }, [
      U.el('div', { class: 'mono', style: 'font-size:10.5px;color:var(--ink-dim);overflow-wrap:anywhere',
        text: '探测用的 cached_file 关掉了三个「抛异常」开关 —— 找不到文件是正常情况，不是错误。' }),
    ]));
    viz.appendChild(left);

    const right = U.el('div', { class: 'col gap10', style: 'flex:0 0 320px' });
    right.appendChild(U.el('div', { class: 'klabel', text: '命中之后：一个类，三个子处理器' }));
    right.appendChild(W.card({ cc: 1, tint: 1, title: 'PROCESSOR_MAPPING[Glm5NextConfig]',
      sub: '<span class="mono" style="font-size:10.5px">Glm5NextProcessor</span>（实测）', style: 'width:100%' }));
    const subs = [
      { t: 'image_processor', d: 'Glm5NextImageProcessor 或 Pil 版（看 backend）', c: 2 },
      { t: 'tokenizer', d: 'glm5_next 不在分词表里 → 落到默认 TokenizersBackend', c: 0 },
      { t: 'video_processor', d: 'Glm5NextVideoProcessor —— 需要 torchvision，否则表里是 None', c: 3 },
    ];
    const subEls = subs.map(s => {
      const c = W.card({ cc: s.c, tint: s.c, title: '<span class="mono" style="font-size:11px">' + s.t + '</span>',
        sub: s.d, style: 'width:100%' });
      right.appendChild(c); return c;
    });
    viz.appendChild(right);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    fEls.forEach(e => { e.style.opacity = '.4'; });
    subEls.forEach(e => { e.style.opacity = '.4'; });
    msg.innerHTML = '<span class="cm">// 先定「哪个文件说话」，再定用哪个类</span>';
    [0, 1, 2, 3, 4].forEach((k, i) => {
      tl.at(1800 + i * 1900, () => {
        fEls.forEach((x, j) => { x.style.opacity = (j === k) ? '1' : '.4'; });
        msg.innerHTML = '第 <em>' + (k + 1) + '</em> 层：' + files[k].n;
      });
    });
    tl.at(12000, () => {
      fEls.forEach(x => { x.style.opacity = '1'; });
      msg.innerHTML = '拿到 processor_class 之后才去 <em>processor_class_from_name()</em> 换成类';
    });
    tl.at(14200, () => {
      subEls.forEach(x => { x.style.opacity = '1'; });
      msg.innerHTML = 'GLM-5 命中 <em>Glm5NextProcessor</em> —— 它按 __init__ 形参名加载<em>三个</em>子处理器';
    });
    tl.at(16400, () => {
      msg.innerHTML = '<span class="cm">// 子处理器的组合不在这个文件里做，由 ProcessorMixin 按形参名装配</span>';
    });
  },
},

/* ------------------------------------------------- 9 收束 */
{
  kicker: '收束',
  title: '六张表 → <span class="hl-a">一张对照表</span>',
  sub: '写法完全对称，只有「谁说了算」各不相同。最后一幕把这六张表压成一张表，并回答验收点。',
  caption: '下一课 L1-07：动态模块、类型契约与权重名映射表。',
  lang: 'python',
  codeStart: 353,
  code: `        # At this stage, there doesn't seem to be a \`Processor\` class available for this model.
        # Let's try the commonly available classes
        for klass in (AutoTokenizer, AutoImageProcessor, AutoVideoProcessor, AutoFeatureExtractor):
            try:
                return klass.from_pretrained(
                    pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
                )
            except Exception:
                continue

        raise ValueError(
            f"Unrecognized processing class in {pretrained_model_name_or_path}. Can't instantiate a processor, a "
            "tokenizer, an image processor, a video processor or a feature extractor for this model. "
            "Make sure the repository contains the files of at least one of those processing classes."
        )`,
  codeNote: 'processing_auto.py L353~L367 —— 三级落空之后的四连兜底，以及那句把四类文件都列一遍的报错。',
  duration: 19000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap10 vizgrow hstart' });
    wrap.appendChild(viz);
    viz.appendChild(U.el('div', { class: 'klabel', text: '六张表 · 实测规模与值的形状' }));

    const tb = W.table([
      ['modeling_auto.py', '54 张任务表（此处列 3 张：556 / 178 / 88 条）', 'str → str', 'AutoModelForCausalLM …'],
      ['tokenization_auto.py', 'TOKENIZER_MAPPING_NAMES · 309 条', 'str → str（87 条是兜底）', 'AutoTokenizer'],
      ['image_processing_auto.py', 'IMAGE_PROCESSOR · 120 + 91 条', 'str → {backend: str}', 'AutoImageProcessor'],
      ['video_processing_auto.py', 'VIDEO_PROCESSOR · 31 + 14 条', 'str → str，环境可整体置 None', 'AutoVideoProcessor'],
      ['feature_extraction_auto.py', 'FEATURE_EXTRACTOR · 29 + 40 条', 'str → str', 'AutoFeatureExtractor'],
      ['processing_auto.py', 'PROCESSOR · 147 + 33 条', 'str → str', 'AutoProcessor'],
    ], { head: ['文件', '表与实测条数', '值的形状', '门面'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 1, style: 'padding:10px 12px' }, [tb]));
    const trs = Array.from(tb.querySelectorAll('tbody tr'));

    const ans = W.card({ cc: 2, tint: 2,
      title: '验收点：AutoProcessor 对 GLM-5 会实例化哪几个子处理器？',
      sub: '<b>三个</b>：image_processor（Glm5NextImageProcessor / Pil 版）、tokenizer、video_processor'
         + '（Glm5NextVideoProcessor）。依据是实测的 Glm5NextProcessor.get_attributes()'
         + ' → [\'image_processor\', \'tokenizer\', \'video_processor\']。',
      style: 'width:100%' });
    viz.appendChild(ans);

    viz.appendChild(W.exercise(
      '如果某个仓库里<em>只有</em> tokenizer_config.json，'
      + '<code class="inl">AutoProcessor.from_pretrained()</code> 会返回什么？',
      '三级文件探测全部落空（没有 processor_config / preprocessor_config / video_processor），'
      + '于是走四连兜底 <code class="inl">(AutoTokenizer, AutoImageProcessor, AutoVideoProcessor, AutoFeatureExtractor)</code>：'
      + '<b>AutoTokenizer 先成功</b>，所以返回的其实是一个分词器，而不是 processor。'
      + '<br>这也解释了为什么它能把「多模态处理器」和「纯文本分词器」两种用法吃在一个入口里。'));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 六张表，一次看完</span>';
    trs.forEach((t, i) => {
      tl.at(1600 + i * 1900, () => {
        trs.forEach((x, k) => x.classList.toggle('ac', k === i));
        msg.innerHTML = '第 <em>' + (i + 1) + '</em> 行：' + tb.querySelectorAll('tbody tr')[i].cells[0].textContent;
      });
    });
    tl.at(13200, () => {
      trs.forEach(x => x.classList.remove('ac'));
      ans.classList.add('ac');
      msg.innerHTML = '验收点：<em>三个</em>子处理器 —— image_processor / tokenizer / video_processor';
    });
    tl.at(16200, () => {
      msg.innerHTML = '读表零 import；取格子才 import；表的内容 = <em>生成 + 补丁 + 环境</em>';
    });
  },
},

];
