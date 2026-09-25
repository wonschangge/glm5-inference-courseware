/* ==========================================================================
   L1-05 · AutoModel 分派：字符串如何找到类
   --------------------------------------------------------------------------
   覆盖：models/auto/__init__.py, models/auto/configuration_auto.py,
         models/auto/auto_factory.py, models/auto/auto_mappings.py（4 个文件 / 2566 行）
   目标：能画出 config.json 的 model_type 字符串到 config 类的完整链路，
         并说清惰性映射把哪一部分成本推迟了。
   排版基准：#visual 可视区 = 994 x 662.7 舞台像素（无头浏览器实测）。
   ========================================================================== */
'use strict';

/* 实测常量（_data/recon/probe_l105.py，transformers 5.18.0.dev0，冷启动一次跑完） */
const P = {
  mods0: 240,      // import transformers 之后的 transformers.* 模块数
  modsAuto: 312,   // 再拿到 AutoConfig / AutoModel 之后
  entries: 718,    // auto_mappings.py 里那张表的原始条目数
  entriesAfter: 723, // configuration_auto.py 就地 update 之后
  keys: 724,       // CONFIG_MAPPING.keys()（再算上 gpt-sw3）
  mmLen: 556,      // len(AutoModel._model_mapping)，零导入
  costCfg: 2,      // 冷访问 CONFIG_MAPPING['glm5_next'] 新增模块数
  costModel: 101,  // 首次取模型类新增模块数
  costAll: 1000,   // CONFIG_MAPPING.values() 新增模块数
  modsAll: 1415,   // 全量遍历之后的总数
  msCfg: 6.8, msModel: 172.4, msAll: 1553, msKeys: 0.2,
};

const SCENES = [

/* ------------------------------------------------- 1 全景：五跳 */
{
  kicker: '第 1 层 · 分派 / 全景',
  title: '一个字符串，怎么变成 <span class="hl-a">一个类</span>',
  sub: 'config.json 里的 "model_type" 是全链路唯一的线索。这一课把五跳走完 —— 其中只有一次真正的 import。',
  caption: '回顾 L1-01：分派交回来的对象，必须满足 PreTrainedConfig 的公共契约。',
  lang: 'python',
  codeStart: 397,
  code: `        config_dict, unused_kwargs = PreTrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
        has_remote_code = "auto_map" in config_dict and "AutoConfig" in config_dict["auto_map"]
        has_local_code = "model_type" in config_dict and config_dict["model_type"] in CONFIG_MAPPING
        explicit_local_code = has_local_code and not CONFIG_MAPPING[config_dict["model_type"]].__module__.startswith(
            "transformers."
        )
        if has_remote_code:`,
  codeNote: 'AutoConfig.from_pretrained 里决定分派的那 7 行：先把 config.json 读成字典，再判断这个 model_type 认不认识。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '五跳 —— 每一跳都在不同的文件里' }));
    const fl = W.flow([
      { t: 'config.json', s: '"model_type"', cc: 0 },
      { t: 'CONFIG_MAPPING_NAMES', s: '718 条 str → str', cc: 4 },
      { t: 'model_type_to_module_name', s: '纯字符串替换', cc: 1 },
      { t: 'importlib.import_module', s: '唯一一次 import', cc: 2 },
      { t: 'Glm5NextConfig', s: '终于是类了', cc: 3 },
    ], { style: 'width:100%' });
    viz.appendChild(fl);

    const stats = [
      { k: '表的条目', v: P.entries, s: '全是 str，一个类都没有', cc: 4 },
      { k: 'import 之后', v: P.mods0, s: 'glm5_next 一个都没加载', cc: 0 },
      { k: 'AutoModel 认识', v: P.mmLen, s: 'config 类，零导入', cc: 1 },
    ];
    const srow = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const sEls = stats.map(st => {
      const c = W.card({ cc: st.cc, tint: st.cc, title: st.k,
        body: U.el('div', { class: 'col' }, [
          U.el('div', { class: 'n big', text: U.fmt(st.v) }),
          U.el('div', { class: 'mono dim', style: 'font-size:10px;margin-top:4px', text: st.s }),
        ]), style: 'flex:1' });
      srow.appendChild(c); return c;
    });
    viz.appendChild(srow);

    const tb = W.table([
      ['① 读 config.json', '<code class="inl">configuration_auto.py:397</code>', '读文件', '<span class="dim">—</span>'],
      ['② 查表得类名', '<code class="inl">auto_mappings.py:250</code>', '纯查表', '<span class="hl4">0 个模块</span>'],
      ['③ 算包名', '<code class="inl">configuration_auto.py:64</code>', '字符串替换', '<span class="hl4">0 个模块</span>'],
      ['④ importlib + getattr', '<code class="inl">configuration_auto.py:109</code>', '<span class="hl3">唯一一次 import</span>', '<span class="hl3">+2 个模块</span>'],
      ['⑤ from_dict 建实例', '<code class="inl">configuration_auto.py:440</code>', '交给类自己', '<span class="dim">见 L1-01</span>'],
    ], { head: ['跳', '代码位置', '做什么', '代价'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 0, style: 'padding:10px 12px;width:100%' }, [
      U.el('div', { class: 'klabel', text: '本课路线图' }), tb,
    ]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    fl.focus(0);
    msg.innerHTML = '<span class="cm">// config.json 里只有字符串，没有类</span>';
    tl.at(2000, () => { fl.focus(1); msg.innerHTML = '第一跳：<em>查表</em> —— "glm5_next" → "Glm5NextConfig"，还是字符串'; });
    tl.at(4600, () => { fl.focus(2); msg.innerHTML = '第二跳：<em>算包名</em> —— 类名 → 可以 import 的包名'; });
    tl.at(7200, () => { fl.focus(3); msg.innerHTML = '第三跳：<span class="fn">importlib</span> 拉包 + <span class="fn">getattr</span> 取类 <span class="cm">// 全链路唯一一次 import</span>'; });
    tl.at(9800, () => { fl.focus(4); msg.innerHTML = '第四跳：终于拿到 <em>类</em>，交给 <span class="fn">from_dict</span> 建实例'; });
    tl.at(12400, () => {
      fl.focus(-1);
      sEls.forEach(e => e.classList.add('ac'));
      msg.innerHTML = '<span class="cm">// 记住这三个数：718 / 240 / 556</span>';
    });
    tl.at(14800, () => {
      sEls.forEach(e => e.classList.remove('ac'));
      msg.innerHTML = '★ 全程只有一次 import —— 这是本课要算清的那笔账';
    });
  },
},

/* ------------------------------------------------- 2 包本身也是惰性的 */
{
  kicker: '第 0 跳 · 包自己',
  title: '连 <span class="hl-b">auto 包自己</span>都是惰性的',
  sub: 'if TYPE_CHECKING 里的 8 行 star-import 运行时一次都不执行；else 分支把整个模块对象换成了 _LazyModule。',
  caption: '这不是 auto 包的特殊待遇 —— transformers 的每个子包都长这样。',
  lang: 'python',
  codeStart: 29,
  code: `else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)`,
  codeNote: 'sys.modules[__name__] = _LazyModule(...) —— 把缓存里的模块对象整个换掉。',
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cA = W.card({ cc: 1, title: 'if TYPE_CHECKING:',
      sub: '8 行 <code class="inl">from .xxx import *</code>。解释器里 <code class="inl">TYPE_CHECKING</code> 恒为 False —— 这段只在 mypy / pyright 眼里存在。',
      style: 'flex:1' });
    const cB = W.card({ cc: 0, title: 'else:',
      sub: '真正执行的只有 3 行：把 <code class="inl">sys.modules[__name__]</code> 换成 <code class="inl">_LazyModule</code>。',
      style: 'flex:1' });
    row.appendChild(cA); row.appendChild(cB);
    viz.appendChild(row);

    const fl = W.flow([
      { t: 'import ...models.auto', s: '拿到 _LazyModule', cc: 1 },
      { t: '第一次属性访问', s: '才去 import 子模块', cc: 2 },
      { t: 'AutoConfig', s: '真的类', cc: 4 },
    ], { style: 'width:100%' });
    viz.appendChild(fl);

    const tb = W.table([
      ['<code class="inl">import transformers</code> 之后', '<span class="hl">240</span> 个 <code class="inl">transformers.*</code> 模块', 'glm5_next 一个都没进来'],
      ['再拿到 AutoConfig / AutoModel', '<span class="hl">312</span> 个', '只多了 auto 那几个类'],
      ['<code class="inl">from .auto_factory import *</code>', '<span class="hlbad">运行时从不执行</span>', '它是声明，不是代码路径'],
    ], { head: ['动作', '实测模块数', '说明'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 1, style: 'padding:10px 12px;width:100%' }, [
      U.el('div', { class: 'klabel', text: '实测（probe_l105.py 冷启动）' }), tb,
    ]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    cB.style.opacity = '.34';
    msg.innerHTML = '<span class="cm">// 先看上面那 8 行</span>';
    tl.at(2000, () => {
      cA.classList.add('ac');
      msg.innerHTML = '这 8 行 <em>运行时一次都不执行</em> —— 只有类型检查器会读它';
    });
    tl.at(5200, () => {
      cA.classList.remove('ac'); cA.style.opacity = '.42';
      cB.style.opacity = '1'; cB.classList.add('ac');
      msg.innerHTML = '真正跑的是 <span class="fn">else</span>：把模块对象换成 <em>_LazyModule</em>';
    });
    tl.at(8400, () => { fl.focus(0); msg.innerHTML = '<code class="inl">import</code> 这个包本身：<em>零个子模块</em>被导入'; });
    tl.at(11000, () => { fl.focus(1); msg.innerHTML = '第一次访问 <span class="fn">AutoConfig</span> 这个名字时，才真的去 import 它所在的文件'; });
    tl.at(13400, () => { fl.focus(-1); msg.innerHTML = '<span class="cm">// 惰性是从包这一级开始做的 —— 实测 240 个模块里没有 glm5_next</span>'; });
  },
},

/* ------------------------------------------------- 3 表：718 条 str → str */
{
  kicker: '第一跳 · 表',
  title: '表里全是 <span class="hl-a">字符串</span>：718 条 str → str',
  sub: 'auto_mappings.py 是一张纯数据表，import 它不会导入任何模型。glm5_next 三个 model_type 指向三个 config 类名。',
  caption: '这张表由代码生成：文件头第 2 行写着 Do NOT edit this file manually。',
  lang: 'python',
  codeStart: 250,
  code: `        ("glm5_next", "Glm5NextConfig"),
        ("glm5_next_text", "Glm5NextTextConfig"),
        ("glm5_next_vision", "Glm5NextVisionConfig"),`,
  codeNote: '右边是类名，不是类 —— 这一点决定了后面必须有人去 import 并 getattr。',
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const tb = W.table([
      ['<code class="inl">"glm5_next"</code>', '<code class="inl">"Glm5NextConfig"</code>', '顶层复合配置', '<span class="hl">glm5_next</span>'],
      ['<code class="inl">"glm5_next_text"</code>', '<code class="inl">"Glm5NextTextConfig"</code>', '文本子配置', '<span class="hl">glm5_next_text</span>'],
      ['<code class="inl">"glm5_next_vision"</code>', '<code class="inl">"Glm5NextVisionConfig"</code>', '视觉子配置', '<span class="hl">glm5_next_vision</span>'],
    ], { head: ['model_type（键）', '类名（值，仍是 str）', '是什么', '反查实测'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 4, style: 'padding:10px 12px;width:100%' }, [
      U.el('div', { class: 'klabel', text: 'CONFIG_MAPPING_NAMES 里的 glm5_next 三条' }), tb,
    ]));

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    row.appendChild(W.card({ cc: 0, tint: 0, title: '读表不导入',
      sub: '<code class="inl">from ...auto_mappings import CONFIG_MAPPING_NAMES</code> 拿到的是一张 <b>OrderedDict</b>，键值类型实测都是 <code class="inl">str</code>。',
      style: 'flex:1' }));
    row.appendChild(W.card({ cc: 2, tint: 2, title: '表是生成的',
      sub: '文件头原话：<b>Do NOT edit this file manually</b>。要改就改 config 类的 <code class="inl">model_type</code>，再跑代码生成器重写整张表。',
      style: 'flex:1' }));
    viz.appendChild(row);

    const shape = U.el('div', { class: 'card', cc: 4, style: 'padding:10px 12px;width:100%' });
    shape.innerHTML = '<div class="klabel">表的形态（示意，非逐字引用）</div>'
      + '<div class="mono" style="font-size:11.5px;line-height:1.7;color:var(--ink-dim)">'
      + 'CONFIG_MAPPING_NAMES = OrderedDict([<br>'
      + '&nbsp;&nbsp;&nbsp;&nbsp;("afmoe", "AfmoeConfig"),<br>'
      + '&nbsp;&nbsp;&nbsp;&nbsp;<span class="hl">("glm5_next", "Glm5NextConfig"),</span> &nbsp;<span class="cm">// 718 条里的一条</span><br>'
      + '&nbsp;&nbsp;&nbsp;&nbsp;... ])</div>';
    viz.appendChild(shape);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 三条并列 = 三个独立的 model_type</span>';
    tl.at(2400, () => { msg.innerHTML = '键是 <em>model_type</em>，值是 <em>类名</em> —— 表里没有任何一个类对象'; });
    tl.at(5600, () => { msg.innerHTML = '反查也成立：<span class="fn">config_class_to_model_type("Glm5NextConfig")</span> → <em>glm5_next</em>'; });
    tl.at(9000, () => { msg.innerHTML = '718 条全部读进内存，代价是 <em>0 个模型模块</em> <span class="cm">// 纯字符串</span>'; });
    tl.at(12400, () => { msg.innerHTML = '<span class="cm">// 表是生成的，所以偶尔要打补丁 —— 见第九幕</span>'; });
  },
},

/* ------------------------------------------------- 4 第二跳：算包名 */
{
  kicker: '第二跳 · 模块名',
  title: '类名 → <span class="hl-b">包名</span>：<code class="inl">-</code> 换成 <code class="inl">_</code>，特例改名',
  sub: 'Python 的模块名不能有连字符，所以 transfo-xl 必须变成 transfo_xl；而 glm5_next_text 藏在 glm5_next 包里，靠一张特例表归一。',
  caption: 'DEPRECATED_MODELS 实测是空列表 —— 那两个 deprecated. 前缀分支目前是死代码。',
  lang: 'python',
  codeStart: 64,
  code: `def model_type_to_module_name(key) -> str:
    """Converts a config key to the corresponding module."""
    # Special treatment
    if key in SPECIAL_MODEL_TYPE_TO_MODULE_NAME:
        key = SPECIAL_MODEL_TYPE_TO_MODULE_NAME[key]

        if key in DEPRECATED_MODELS:
            key = f"deprecated.{key}"
        return key

    key = key.replace("-", "_")
    if key in DEPRECATED_MODELS:
        key = f"deprecated.{key}"

    return key`,
  codeNote: '两次 replace 的本质：把 model_type 映射成一个可以 import 的包名。',
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const mkRow = (k, v, note, cls) => [
      '<code class="inl">' + k + '</code>',
      '<span class="mono">→</span> <code class="inl">' + v + '</code>',
      note, cls,
    ];
    const tb = W.table([
      mkRow('transfo-xl', 'transfo_xl', '连字符换下划线', '<span class="hl4">默认路径</span>'),
      mkRow('bert-generation', 'bert_generation', '连字符换下划线', '<span class="hl4">默认路径</span>'),
      mkRow('audio-spectrogram-transformer', 'audio_spectrogram_transformer', '连字符换下划线', '<span class="hl4">默认路径</span>'),
      mkRow('glm5_next', 'glm5_next', '名字本身就是合法包名', '<span class="dim">默认路径</span>'),
      mkRow('glm5_next_text', 'glm5_next', '命中特例表', '<span class="hl2">特例表</span>'),
      mkRow('glm5_next_vision', 'glm5_next', '命中特例表', '<span class="hl2">特例表</span>'),
    ], { head: ['model_type', '包名（实测输出）', '为什么', '走哪条路'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 1, style: 'padding:10px 12px;width:100%' }, [
      U.el('div', { class: 'klabel', text: 'model_type_to_module_name 的实测输出' }), tb,
    ]));

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    row.appendChild(W.card({ cc: 0, tint: 0, title: '第一段：SPECIAL_MODEL_TYPE_TO_MODULE_NAME',
      sub: '只在两种情况下登记：名字里有连字符，或者一个包里住着多个 <code class="inl">model_type</code>。命中就直接返回，不再往下走。',
      style: 'flex:1' }));
    row.appendChild(W.card({ cc: 3, tint: 3, title: '第二段：DEPRECATED_MODELS 分支',
      sub: '实测 <code class="inl">DEPRECATED_MODELS = []</code> —— <b>目前是死代码</b>。分支和注释不一定是活的，要回文件里核对常量本身。',
      style: 'flex:1' }));
    viz.appendChild(row);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 三个 glm5_next_* 最终都指向同一个包</span>';
    tl.at(2400, () => { msg.innerHTML = '连字符 <em>-</em> 在模块名里非法 → 一律换成下划线'; });
    tl.at(5600, () => { msg.innerHTML = '但 <em>glm5_next_text</em> 换成下划线也还是找不到包 —— 它住在 <em>glm5_next</em> 里'; });
    tl.at(9000, () => { msg.innerHTML = '所以有了那张特例表：<em>三个 model_type，一个包</em>'; });
    tl.at(12400, () => { msg.innerHTML = '<span class="cm">// 这一跳仍然是纯字符串操作，0 个模块</span>'; });
  },
},

/* ------------------------------------------------- 5 ★ 核心 4 行 */
{
  kicker: '第三跳 · ★ 核心',
  title: '★ 「用时才导入」就发生在这 <span class="hl-a">4 行</span>里',
  sub: '_LazyConfigMapping.__getitem__：查表拿到类名，算出包名，importlib 拉包，getattr 取类 —— 全程只有这一处碰 import 系统。',
  caption: '对比同一个类：keys() 一行都不导入，values() 会把 700 多条全部导入。',
  lang: 'python',
  codeStart: 103,
  code: `    def __getitem__(self, key: str) -> type[PreTrainedConfig]:
        if key in self._extra_content:
            return self._extra_content[key]
        if key not in self._mapping:
            raise KeyError(key)
        value = self._mapping[key]
        module_name = model_type_to_module_name(key)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(f".{module_name}", "transformers.models")
        if hasattr(self._modules[module_name], value):
            return getattr(self._modules[module_name], value)

        # Some of the mappings have entries model_type -> config of another model type. In that case we try to grab the
        # object at the top level.
        transformers_module = importlib.import_module("transformers")
        return getattr(transformers_module, value)`,
  codeNote: 'self._modules 是包级缓存：glm5_next_text 与 glm5_next 会命中同一个缓存项。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '__getitem__ 的四步（第 104–118 行）' }));
    const steps = [
      { n: '①', t: '查 _extra_content', s: '用户 register 进来的类，直接返回', cc: 3 },
      { n: '②', t: '查 _mapping', s: '键不存在就 raise KeyError', cc: 1 },
      { n: '③', t: '算包名 + importlib', s: '唯一一次 import，结果进 _modules 缓存', cc: 2 },
      { n: '④', t: 'getattr 取类', s: '取不到就退到顶层 transformers', cc: 4 },
    ];
    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cards = steps.map(st => {
      const c = W.card({ cc: st.cc, tint: st.cc, num: st.n, title: st.t, sub: st.s,
        style: 'flex:1;min-height:104px' });
      row.appendChild(c); return c;
    });
    viz.appendChild(row);

    const tb = W.table([
      ['<code class="inl">_mapping</code>', '718 → 723 条', '字符串', '导入时就位，零模型导入'],
      ['<code class="inl">_modules</code>', '访问后才长出来', '模块对象', '包级缓存，同一包只 import 一次'],
      ['<code class="inl">_extra_content</code>', 'register() 写入', '<b>类对象</b>', '唯一"已经拿在手上"的一批'],
    ], { head: ['内部字典', '内容', '存的是什么', '什么时候有'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 2, style: 'padding:10px 12px;width:100%' }, [
      U.el('div', { class: 'klabel', text: '_LazyConfigMapping 的三个字典' }), tb,
    ]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    cards.forEach(c => { c.style.opacity = '.3'; });
    msg.innerHTML = '<span class="cm">// 一次 __getitem__ 走四步</span>';
    tl.at(1800, () => { cards[0].style.opacity = '1'; cards[0].classList.add('ac'); msg.innerHTML = '① <em>_extra_content</em>：用户注册过的类，最先查'; });
    tl.at(4600, () => { cards[0].classList.remove('ac'); cards[1].style.opacity = '1'; cards[1].classList.add('ac'); msg.innerHTML = '② <em>_mapping</em>：查表。这一步只看键，<span class="hl4">不导入任何东西</span>'; });
    tl.at(7400, () => { cards[1].classList.remove('ac'); cards[2].style.opacity = '1'; cards[2].classList.add('ac'); msg.innerHTML = '③ <span class="fn">importlib.import_module(".glm5_next", "transformers.models")</span> <span class="cm">// 就是这里</span>'; });
    tl.at(10200, () => { cards[2].classList.remove('ac'); cards[3].style.opacity = '1'; cards[3].classList.add('ac'); msg.innerHTML = '④ <em>getattr</em> 取类；取不到就退到顶层 <span class="fn">transformers</span> 再取一次'; });
    tl.at(13000, () => {
      cards.forEach(c => { c.style.opacity = '1'; c.classList.remove('ac'); });
      msg.innerHTML = '★ 记住边界：<em>keys()</em> 零导入，<em>values()</em> 全量导入 —— 惰性只推迟成本，不消除成本';
    });
    tl.at(15600, () => { msg.innerHTML = '<span class="cm">// 实测：keys() +0 个模块 / values() +1000 个模块</span>'; });
  },
},

/* ------------------------------------------------- 6 入口函数 */
{
  kicker: '入口 · from_pretrained',
  title: '字符串不是参数，是 <span class="hl-b">config.json 里的字段</span>',
  sub: 'from_pretrained 的第一个参数是仓库 id 或路径；model_type 是 get_config_dict 读出来的。混起来是这一课最常见的误解。',
  caption: '想直接按 model_type 造配置：AutoConfig.for_model("glm5_next")，实测返回 Glm5NextConfig。',
  lang: 'python',
  codeStart: 419,
  code: `        elif "model_type" in config_dict:
            # Apply heuristic: if model_type is mistral but layer_types is present, treat as ministral
            if config_dict["model_type"] == "mistral" and "layer_types" in config_dict:
                logger.info(
                    "Detected mistral model with layer_types, treating as ministral for alternating attention compatibility. "
                )
                config_dict["model_type"] = "ministral"

            try:
                config_class = CONFIG_MAPPING[config_dict["model_type"]]`,
  codeNote: '拿类那两行；上面还有一段启发式：mistral + layer_types 会被改写成 ministral。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const tb = W.table([
      ['<code class="inl">AutoConfig.from_pretrained(?)</code>', '仓库 id / 本地路径', '<span class="hlbad">不是</span> model_type', '<code class="inl">"google-bert/bert-base-uncased"</code>'],
      ['<code class="inl">config_dict["model_type"]</code>', 'config.json 里的字段', '<span class="hl4">这才是</span> model_type', '<code class="inl">"glm5_next"</code>'],
      ['<code class="inl">AutoConfig.for_model(?)</code>', '直接吃 model_type', '<span class="hl4">可以</span>这么用', '<code class="inl">"glm5_next"</code>'],
    ], { head: ['入口', '第一个参数是什么', '是 model_type 吗', '例子'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 3, style: 'padding:10px 12px;width:100%' }, [
      U.el('div', { class: 'klabel', text: '三个入口，别混' }), tb,
    ]));

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cA = W.card({ cc: 0, tint: 0, title: 'contains：不导入',
      sub: '<code class="inl">"model_type" in CONFIG_MAPPING</code> 只查键（<code class="inl">__contains__</code>），<b>0 个模块</b>。用来判断认不认识。',
      style: 'flex:1' });
    const cB = W.card({ cc: 2, tint: 2, title: 'getitem：导入',
      sub: '<code class="inl">CONFIG_MAPPING[...]</code> 才会真的拉包。<code class="inl">explicit_local_code</code> 为了看 <code class="inl">__module__</code> 就得付这笔钱。',
      style: 'flex:1' });
    row.appendChild(cA); row.appendChild(cB);
    viz.appendChild(row);

    const cC = W.card({ cc: 1, tint: 1, title: '字符串在分派之前还能被改写',
      sub: '第 420–425 行：<code class="inl">model_type == "mistral"</code> 且配置里有 <code class="inl">layer_types</code> 时，会被就地改成 <code class="inl">"ministral"</code>。说明 model_type 是运行时数据，不是常量。',
      style: 'width:100%' });
    viz.appendChild(cC);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 先纠正一个常见写法</span>';
    tl.at(2200, () => { cC.style.opacity = '.34'; msg.innerHTML = '<code class="inl">from_pretrained("glm5_next")</code> 是把 "glm5_next" 当<em>仓库名</em>去找 —— 语义不对'; });
    tl.at(5400, () => { msg.innerHTML = '真正的 model_type 来自 <em>config.json</em>，由 <span class="fn">get_config_dict</span> 读出来'; });
    tl.at(8600, () => { cA.classList.add('ac'); msg.innerHTML = '同一个函数里：<span class="fn">in CONFIG_MAPPING</span> 不导入，<span class="fn">CONFIG_MAPPING[...]</span> 才导入'; });
    tl.at(11600, () => { cA.classList.remove('ac'); cB.classList.add('ac'); msg.innerHTML = '两行之内，一次免费、一次付费 <span class="cm">// 第 399 行 vs 第 400 行</span>'; });
    tl.at(14200, () => {
      cB.classList.remove('ac'); cC.style.opacity = '1';
      msg.innerHTML = '★ 想按 model_type 直接造：<span class="fn">AutoConfig.for_model("glm5_next")</span> → <em>Glm5NextConfig</em>';
    });
  },
},

/* ------------------------------------------------- 7 ★ 第二次分派 */
{
  kicker: '第二次分派 · ★ 同一套手法',
  title: '★ config 类 → 模型类：<span class="hl-a">反查回字符串</span>再走一遍',
  sub: 'AutoModel 的键是 config 类，不是字符串。_LazyAutoMapping 先把类名反查成 model_type，然后复用同一套 importlib 手法。',
  caption: '第 595 行把 self 反向挂回 _model_mapping —— 自动生成 docstring 时要靠它。',
  lang: 'python',
  codeStart: 582,
  code: `class _LazyAutoMapping(OrderedDict[type[PreTrainedConfig], _LazyAutoMappingValue]):
    """
    A mapping config to object (model or tokenizer for instance) that will load keys and values when it is accessed.

    Args:
        - config_mapping: The map model type to config class
        - model_mapping: The map model type to model (or tokenizer) class
    """

    def __init__(self, config_mapping, model_mapping) -> None:
        self._config_mapping = config_mapping
        self._reverse_config_mapping = {v: k for k, v in config_mapping.items()}
        self._model_mapping = model_mapping
        self._model_mapping._model_mapping = self
        self._extra_content = {}
        self._modules = {}`,
  codeNote: '_reverse_config_mapping 是 {类名字符串: model_type}，所以键又回到了字符串。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cA = W.card({ cc: 0, title: '_LazyConfigMapping（第五幕）',
      sub: '键 = <code class="inl">"glm5_next"</code>（字符串）<br>值 = config 类<br>作用：model_type → config 类',
      style: 'flex:1' });
    const cB = W.card({ cc: 1, title: '_LazyAutoMapping（本幕）',
      sub: '键 = <code class="inl">Glm5NextConfig</code>（类）<br>值 = 模型类（或 tuple）<br>作用：config 类 → 模型类',
      style: 'flex:1' });
    row.appendChild(cA); row.appendChild(cB);
    viz.appendChild(row);

    const fl = W.flow([
      { t: 'Glm5NextConfig', s: '作为键传进来', cc: 1 },
      { t: '_reverse_config_mapping', s: '{类名: model_type}', cc: 2 },
      { t: '"glm5_next"', s: '又变回字符串', cc: 0 },
      { t: '_load_attr_from_module', s: '同一个 importlib 套路', cc: 4 },
      { t: 'Glm5NextModel', s: '模型类', cc: 3 },
    ], { style: 'width:100%' });
    viz.appendChild(fl);

    const tb = W.table([
      ['<code class="inl">_config_mapping</code>', '718 → 723 条字符串表', '用来反查与遍历'],
      ['<code class="inl">_reverse_config_mapping</code>', '<code class="inl">{类名: model_type}</code>', '把类键翻译回字符串'],
      ['<code class="inl">_model_mapping</code>', '模型名字表（也是纯字符串）', '真正被查的那张表'],
      ['<code class="inl">_modules</code>', '包级缓存', '与 _LazyConfigMapping 同款'],
    ], { head: ['__init__ 里的字段', '内容', '用途'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 1, style: 'padding:10px 12px;width:100%' }, [
      U.el('div', { class: 'klabel', text: '构造函数的四件事（第 592–597 行）' }), tb,
    ]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    fl.focus(-1);
    msg.innerHTML = '<span class="cm">// AutoModel 收到的是一个 config 对象</span>';
    tl.at(2000, () => { fl.focus(0); msg.innerHTML = '键是 <em>config 类</em>，不是字符串 —— 所以不能直接查那张字符串表'; });
    tl.at(4800, () => { fl.focus(1); msg.innerHTML = '构造函数先把它反过来：<span class="fn">{类名: model_type}</span>'; });
    tl.at(7600, () => { fl.focus(2); msg.innerHTML = '于是键<em>又回到字符串</em> —— 后面可以完全复用第五幕那套手法'; });
    tl.at(10400, () => { fl.focus(3); msg.innerHTML = '<span class="fn">_load_attr_from_module</span>：算包名 → importlib → getattr，连缓存都照抄一份'; });
    tl.at(13000, () => {
      fl.focus(-1);
      msg.innerHTML = '实测：<em>AutoModel._model_mapping[Glm5NextConfig]</em> → <span class="hl4">Glm5NextModel</span>（单个类，不是 tuple）';
    });
    tl.at(15400, () => { msg.innerHTML = '<span class="cm">// 一个 config 对应多个 model_type 时才是 tuple，最后由 architectures 裁决</span>'; });
  },
},

/* ------------------------------------------------- 8 成本账 */
{
  kicker: '成本 · 实测',
  title: '惰性省下了什么：<span class="hl-c">240 → +2 / +101 / +1000</span>',
  sub: '冷启动一次跑完：import transformers 之后一个模型包都没进来；点名一个 model_type 只拉它自己那一支；全量遍历才付 1000 个模块。',
  caption: '实测脚本 _data/recon/probe_l105.py —— 它是脚本，不计入覆盖率。',
  lang: 'python',
  codeStart: 603,
  code: `    def __getitem__(self, key: type[PreTrainedConfig]) -> _LazyAutoMappingValue:
        if key in self._extra_content:
            return self._extra_content[key]
        model_type = self._reverse_config_mapping[key.__name__]
        if model_type in self._model_mapping:
            model_name = self._model_mapping[model_type]
            return self._load_attr_from_module(model_type, model_name)

        # Maybe there was several model types associated with this config.
        model_types = [k for k, v in self._config_mapping.items() if v == key.__name__]
        for mtype in model_types:
            if mtype in self._model_mapping:
                model_name = self._model_mapping[mtype]
                return self._load_attr_from_module(mtype, model_name)
        raise KeyError(key)`,
  codeNote: '__getitem__ 只碰一个 model_type；真正贵的是 keys()/values() 这种"全都要"的调用。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '新增的 transformers.* 模块数（冷启动实测）' }));
    const bars = W.bars([
      { label: 'import 之后', value: P.mods0, cc: 0, valueText: U.fmt(P.mods0) },
      { label: '点名 config', value: P.costCfg, cc: 4, valueText: '+' + P.costCfg },
      { label: '点名模型类', value: P.costModel, cc: 2, valueText: '+' + P.costModel },
      { label: 'values() 全都要', value: P.costAll, cc: 3, valueText: '+' + U.fmt(P.costAll) },
    ], { max: P.costAll });
    bars.setAll(0);
    viz.appendChild(bars);

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cA = W.card({ cc: 4, tint: 4, title: '真的免费的那两次',
      body: U.el('div', { class: 'mono', style: 'font-size:11px;line-height:1.75;color:var(--ink-dim)' }, [
        U.el('div', { html: 'len(_model_mapping) = <span class="hl">556</span> &nbsp;<span class="cm">// +0 个模块，0.1 ms</span>' }),
        U.el('div', { html: 'CONFIG_MAPPING.keys() = <span class="hl">724</span> &nbsp;<span class="cm">// +0 个模块，0.2 ms</span>' }),
      ]), style: 'flex:1' });
    const cB = W.card({ cc: 3, tint: 3, title: '要付钱的那两次',
      body: U.el('div', { class: 'mono', style: 'font-size:11px;line-height:1.75;color:var(--ink-dim)' }, [
        U.el('div', { html: '首次取模型类 <span class="hlbad">172.4 ms</span> &nbsp;<span class="cm">// 这个包的真实体积</span>' }),
        U.el('div', { html: 'values() <span class="hlbad">1553 ms</span> &nbsp;<span class="cm">// 1415 个模块，全量</span>' }),
      ]), style: 'flex:1' });
    row.appendChild(cA); row.appendChild(cB);
    viz.appendChild(row);

    const tb = W.table([
      ['<code class="inl">len(mapping)</code>', '集合运算', '<span class="hl4">0</span>', '只对两张表的键求交集'],
      ['<code class="inl">keys()</code> / <code class="inl">__contains__</code>', '键遍历', '<span class="hl4">0</span>', '一个类都没取出来'],
      ['<code class="inl">mapping[一个 config 类]</code>', '单点取值', '<span class="hl">+101</span>', '只拉被点到的那个包'],
      ['<code class="inl">values()</code> / <code class="inl">items()</code>', '全量取值', '<span class="hlbad">+1000</span>', '700 多条一次导完'],
    ], { head: ['调用', '做什么', '新增模块', '为什么'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 2, style: 'padding:10px 12px;width:100%' }, [
      U.el('div', { class: 'klabel', text: '同一个对象，四种调用，四种价格' }), tb,
    ]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 从 import transformers 的 240 个模块出发</span>';
    tl.at(2000, () => { bars.setAll(0.25); msg.innerHTML = '只 import transformers：<em>240</em> 个模块，一个模型包都没进来'; });
    tl.at(5000, () => { bars.setAll(1); msg.innerHTML = '点名 <em>glm5_next</em> 的 config：只多了 <em>2</em> 个模块，6.8 ms <span class="cm">// 条形细到看不见，这正是重点</span>'; });
    tl.at(8000, () => { msg.innerHTML = '点名它的<em>模型类</em>：多了 <em>101</em> 个模块，172 ms —— 同一次分派的两跳，按模块数差 50 倍'; });
    tl.at(11000, () => { msg.innerHTML = '而 <em>values()</em> 这种"全都要"：<em>1000</em> 个模块、1.5 秒，一次付清'; });
    tl.at(13800, () => { msg.innerHTML = '★ 惰性不消除成本，只把成本从 <span class="fn">import transformers</span> 推迟到<em>第一次点名</em>'; });
    tl.at(16000, () => { msg.innerHTML = '<span class="cm">// 只服务一个模型的推理进程，那 1000 个模块永远不用付</span>'; });
  },
},

/* ------------------------------------------------- 9 收束 */
{
  kicker: '收束',
  title: '把五跳连起来',
  sub: '表的边界：718 条自动生成 + 5 条手工登记 + 1 条历史包袱；你的自定义模型走 CONFIG_MAPPING.register()，落进 _extra_content。',
  caption: '下一课 L1-06：其余四张自动注册表（modeling / tokenization / image_processing / processor）。',
  lang: 'python',
  codeStart: 36,
  code: `CONFIG_MAPPING_NAMES.update(
    {
        "EvollaModel": "EvollaConfig",
        "mlcd": "MLCDVisionConfig",
        "parakeet_tdt": "ParakeetTDTConfig",
        "vibevoice_acoustic_tokenizer_decoder": "VibeVoiceAcousticTokenizerDecoderConfig",
        "vibevoice_acoustic_tokenizer_encoder": "VibeVoiceAcousticTokenizerEncoderConfig",
    }
)`,
  codeNote: '自动生成之外，还有 5 个不按命名规则来的模型靠手工登记 —— 这就是表的边界。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap10 vizgrow hstart' });
    wrap.appendChild(viz);

    const steps = [
      { n: '①', t: '读 config.json', d: 'get_config_dict → dict', c: 0 },
      { n: '②', t: '查表', d: '718 条 str→str，零导入', c: 4 },
      { n: '③', t: '算包名', d: '纯字符串替换，零导入', c: 1 },
      { n: '④', t: 'importlib', d: '唯一一次 import', c: 2 },
      { n: '⑤', t: 'from_dict', d: '建实例，交给 L1-01', c: 3 },
    ];
    const row = U.el('div', { class: 'row gap8', style: 'width:100%;align-items:stretch' });
    const els = steps.map(s => {
      const e = W.card({ cc: s.c, tint: s.c, title: '<span class="mono" style="font-size:11px">' + s.n + ' ' + s.t + '</span>',
        sub: s.d, style: 'flex:1;text-align:center' });
      row.appendChild(e); return e;
    });
    viz.appendChild(row);

    const tb = W.table([
      ['自动生成', '<span class="hl">718</span> 条', 'auto_mappings.py，文件头写着 Do NOT edit this file manually'],
      ['手工登记', '<span class="hl">5</span> 条', 'configuration_auto.py 的 update({...})，就地修改同一张表'],
      ['历史包袱', '<span class="hl">1</span> 条', 'gpt-sw3，注释里留着 TODO: deprecate and remove'],
      ['用户注册', '<span class="hl4">_extra_content</span>', 'AutoConfig.register()，存的是<b>类对象</b>，不走惰性'],
      ['合计', '<span class="hl">724</span>', '实测 CONFIG_MAPPING.keys() 的长度'],
    ], { head: ['来源', '条数', '说明'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 0, style: 'padding:10px 12px;width:100%' }, [
      U.el('div', { class: 'klabel', text: '724 = 718 + 5 + 1' }), tb,
    ]));

    viz.appendChild(W.exercise(
      '从 <code class="inl">config.json</code> 里的 <code class="inl">"model_type": "glm5_next_vision"</code> 出发，'
      + '到类被取出来，中间经过哪几步？<b>哪一步真正触发 import？</b>',
      '① <code class="inl">CONFIG_MAPPING_NAMES["glm5_next_vision"]</code> → <code class="inl">"Glm5NextVisionConfig"</code>（纯查表）；'
      + '② <code class="inl">model_type_to_module_name</code> 命中 <code class="inl">SPECIAL_MODEL_TYPE_TO_MODULE_NAME</code>，归一成 <code class="inl">"glm5_next"</code>（纯字符串）；'
      + '③ <code class="inl">CONFIG_MAPPING["glm5_next_vision"]</code> → <code class="inl">__getitem__</code> → '
      + '<code class="inl">importlib.import_module(".glm5_next", "transformers.models")</code> ← <b>只有这一步 import</b>；'
      + '④ <code class="inl">getattr(module, "Glm5NextVisionConfig")</code> 取类；'
      + '⑤ <code class="inl">config_class.from_dict(config_dict)</code> 建实例。'
      + '<br>关键：①② 全是字符串操作，实测新增 <b>0</b> 个模块。'));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    els.forEach(e => { e.style.opacity = '.3'; });
    msg.innerHTML = '<span class="cm">// 五跳，逐个点亮</span>';
    steps.forEach((s, i) => {
      tl.at(1500 + i * 1900, () => {
        els.forEach((e, k) => { e.style.opacity = (k === i) ? '1' : '.3'; });
        msg.innerHTML = '<em>' + s.n + '</em> ' + s.t + ' &nbsp;<span class="cm">// ' + s.d + '</span>';
      });
    });
    tl.at(11200, () => {
      els.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = '★ 表里存的是 <em>str → str</em>，类只在被点名的那一刻由 <span class="fn">importlib</span> 取出来';
    });
    tl.at(14000, () => { msg.innerHTML = '<span class="cm">// 下一课 L1-06：同样的结构，另外四张表</span>'; });
  },
},

];
