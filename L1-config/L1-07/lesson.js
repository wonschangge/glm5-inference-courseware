/* ==========================================================================
   L1-07 · 动态模块、类型契约与权重名映射表
   --------------------------------------------------------------------------
   覆盖：dynamic_module_utils.py / _typing.py / conversion_mapping.py（3 个文件 / 3161 行）
   目标：看完能说出"一个字符串怎么变成类"，以及动态模块的缓存与校验判据是什么。
   排版基准：#visual 可视区 = 994 x 662.7 舞台像素（无头浏览器实测）。
   ========================================================================== */
'use strict';

/* 本课实测常量（仓库自带 venv，transformers 5.18.0.dev0） */
const M = {
  dmuLines: 849, typLines: 185, cmLines: 2127,
  relImports: 18,          // modeling_glm5_next.py 的相对导入目标数
  handKeys: 118,           // _build_checkpoint_conversion_mapping 手写键
  aliasKeys: 97,           // _MODEL_TO_CONVERSION_PATTERN 条数
  aliasNew: 92,            // 其中补齐的全新键
  totalKeys: 210,          // 118 + 92
  glmRenames: 10, glmConv: 3,
  timeout: 15,             // TIME_OUT_REMOTE_CODE 秒
};

const SCENES = [

/* ------------------------------------------------- 1 全景 */
{
  kicker: 'L1 · 配置与分派',
  title: '三个字符串问题，<span class="hl-a">三个文件</span>',
  sub: '仓库名 → 模块 → 类；注解 → 契约；磁盘权重名 → 本实现模块名。L0 与 L1 前六课都默认「代码已经在库里」，这一课补上这个前提。',
  caption: '回顾 L1-05：AutoModel 用一张惰性映射表把字符串变成类；表里没有、而仓库自带代码时，才轮到 dynamic_module_utils。',
  lang: 'python',
  codeStart: 66,
  code: `    new_name = name.replace(".", "_dot_").replace("-", "_hyphen_")
    if new_name and new_name[0].isdigit():
        new_name = f"_{new_name}"
    if keyword.iskeyword(new_name):
        logger.warning(
            f"The module name {new_name} (originally {name}) is a reserved keyword in Python. "
            "Please rename the original module to avoid import issues."
        )
    elif not new_name.isidentifier():
        logger.warning(
            f"The module name {new_name} (originally {name}) is not a valid Python identifier. "
            "Please rename the original module to avoid import issues."
        )
    return new_name`,
  codeNote: '_sanitize_module_name —— 整条链的第一步：把仓库名变成一个合法的模块名。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '一次 trust_remote_code 加载：字符串的六次变形' }));
    const fl = W.flow([
      { t: 'repo id', s: '字符串', cc: 0 },
      { t: 'sanitize', s: '合法模块名', cc: 4 },
      { t: 'modules 缓存目录', s: '进 sys.path', cc: 2 },
      { t: 'get_cached_module_file', s: '校验 + 拷贝', cc: 1 },
      { t: 'get_class_in_module', s: '哈希 → 类', cc: 5 },
      { t: 'class', s: '可实例化', cc: 3 },
    ], { style: 'width:100%' });
    viz.appendChild(fl);

    const cards = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const mk = (cc, name, desc, n) => {
      const c = W.card({ cc: cc, tint: cc, title: '<span class="mono" style="font-size:11.5px">' + name + '</span>',
        sub: desc,
        body: U.el('div', { class: 'mono', style: 'font-size:10.5px;color:var(--ink-faint);margin-top:6px', text: n + ' 个引用块' }) });
      c.style.flex = '1';
      cards.appendChild(c); return c;
    };
    const c1 = mk(0, 'dynamic_module_utils.py', '字符串 → 模块 → 类。缓存目录、依赖哈希、信任边界都在这里。', '8');
    const c2 = mk(1, '_typing.py', '把鸭子类型写成合同：别名统一注解，Protocol 只给类型检查器看。', '2');
    const c3 = mk(2, 'conversion_mapping.py', '磁盘权重名 → 本实现模块名：两张表合成 210 个 model_type。', '4');
    viz.appendChild(cards);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    fl.focus(0);
    msg.innerHTML = '<span class="cm">// 起点是一个字符串：仓库名，或者 class_reference</span>';
    const steps = [
      ['sanitize 把它变成<em>合法模块名</em>', '实测：THUDM/glm-5 → THUDM/glm_hyphen_5'],
      ['缓存目录是 <em>sys.path</em> 上的一格', '不然 transformers_modules.* 根本导不进来'],
      ['<em>校验</em>：缺包就报错，内容变了就重拷', 'check_imports + 文件哈希'],
      ['<em>执行</em>：哈希决定要不要重新 exec', '相同就复用 sys.modules 里的类对象'],
      ['拿到<em>真正的类</em>，可以做后续所有事', '这一课的终点，也是 AutoModel 的兜底路径'],
    ];
    steps.forEach((s, i) => {
      tl.at(1500 + i * 1900, () => {
        fl.focus(i + 1);
        msg.innerHTML = s[0] + ' &nbsp;<span class="cm">// ' + s[1] + '</span>';
      });
    });
    const cs = [c1, c2, c3];
    cs.forEach((c, i) => {
      tl.at(11200 + i * 2000, () => {
        fl.focus(-1);
        cs.forEach((x, k) => { x.classList.toggle('ac', k === i); });
        msg.innerHTML = ['第一个文件：<em>8 个引用块</em>，全部围绕"字符串怎么变成类"',
                         '第二个文件：<em>2 个引用块</em>，没有一个运行期对象，全是注解',
                         '第三个文件：<em>4 个引用块</em>，回答"名字怎么对应"'][i];
      });
    });
  },
},

/* ------------------------------------------------- 2 缓存目录 */
{
  kicker: '第一步 · 落地',
  title: '缓存目录：<span class="hl-a">sys.path</span> 上的一格',
  sub: '动态代码不能原地 import，得先被搬进一个固定目录，再让这个目录成为可导入的路径。init_hf_modules 只做三件事，而且它必须能被反复调用。',
  caption: '注意第一行的早退：「已经在 sys.path 里」就当做过 —— 这个函数在一次加载里会被调用很多次。',
  lang: 'python',
  codeStart: 89,
  code: `    # This function has already been executed if HF_MODULES_CACHE already is in the Python path.
    if HF_MODULES_CACHE in sys.path:
        return

    sys.path.append(HF_MODULES_CACHE)
    os.makedirs(HF_MODULES_CACHE, exist_ok=True)
    init_path = Path(HF_MODULES_CACHE) / "__init__.py"
    if not init_path.exists():
        init_path.touch()
        importlib.invalidate_caches()`,
  codeNote: 'init_hf_modules —— 三件事 + 一个早退，全部幂等。',
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: 'init_hf_modules 的三件事（外加一个只做一次的动作）' }));
    const acts = [
      { cc: 0, t: '① append sys.path', d: 'HF_MODULES_CACHE 成为可导入的一格' },
      { cc: 4, t: '② makedirs', d: 'exist_ok=True —— 重复调用不报错' },
      { cc: 2, t: '③ touch __init__.py', d: '让它成为真正的包，而不是裸目录' },
      { cc: 3, t: '④ invalidate_caches', d: '只在真的新建了文件时才调用' },
    ];
    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const els = acts.map(a => {
      const c = W.card({ cc: a.cc, tint: a.cc, title: a.t, sub: a.d });
      c.style.flex = '1';
      row.appendChild(c); return c;
    });
    viz.appendChild(row);

    const tree = U.el('div', { class: 'card', cc: 1, style: 'padding:12px 14px;width:100%' });
    tree.appendChild(U.el('div', { class: 'klabel', text: '实测目录结构（本课临时仓库跑出来的）' }));
    tree.appendChild(U.el('div', {
      class: 'mono',
      style: 'font-size:11.5px;line-height:1.75;white-space:pre;color:var(--ink-dim);overflow-wrap:anywhere',
      text: [
        '~/.cache/huggingface/modules/            ← HF_MODULES_CACHE',
        '├── __init__.py                           ← ③ 建的空文件',
        '└── transformers_modules/                 ← TRANSFORMERS_DYNAMIC_MODULE_NAME',
        '    └── demo_repo/                        ← 本地模型目录名（sanitize 过）',
        '        └── a32f4f65faa60c4e/             ← 源码内容哈希（16 位十六进制）',
        '            ├── modeling_demo.py',
        '            └── helper_demo.py            ← 相对导入的文件也被一起拷进来',
      ].join('\n'),
    }));
    viz.appendChild(tree);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    els.forEach(e => { e.style.opacity = '.3'; });
    msg.innerHTML = '<span class="cm">// 三个动作 + 一个只在新建文件时触发的动作</span>';
    acts.forEach((a, i) => {
      tl.at(1600 + i * 2400, () => {
        els.forEach((e, k) => { e.style.opacity = (k === i) ? '1' : '.3'; });
        msg.innerHTML = a.t + ' &nbsp;<span class="cm">// ' + a.d + '</span>';
      });
    });
    tl.at(11800, () => {
      els.forEach(e => { e.style.opacity = '1'; });
      tree.classList.add('ac');
      msg.innerHTML = '目录里那条 <em>demo_repo/哈希/</em> 路径，就是下一幕要讲的"内容身份"';
    });
    tl.at(14200, () => {
      tree.classList.remove('ac');
      msg.innerHTML = '<span class="cm">// 早退是常态：第二次调用直接 return，连 makedirs 都不做</span>';
    });
  },
},

/* ------------------------------------------------- 3 依赖闭包 */
{
  kicker: '第二步 · 找依赖',
  title: '依赖闭包：<span class="hl-a">两个正则</span> 换来一个哈希的输入集合',
  sub: '哈希要喂"整个依赖闭包"，闭包靠正则一层层问出来：谁被相对导入了，就顺着它继续问。',
  caption: '实测边界：把库内文件喂给递归版 get_relative_import_files 会抛 FileNotFoundError —— 这条路径只为自包含的远端仓库设计。',
  lang: 'python',
  codeStart: 133,
  code: `    with open(module_file, encoding="utf-8") as f:
        content = f.read()

    # Imports of the form \`import .xxx\`
    relative_imports = re.findall(r"^\\s*import\\s+\\.(\\S+)\\s*$", content, flags=re.MULTILINE)
    # Imports of the form \`from .xxx import yyy\`
    relative_imports += re.findall(r"^\\s*from\\s+\\.(\\S+)\\s+import", content, flags=re.MULTILINE)
    # Unique-ify
    return list(set(relative_imports))`,
  codeNote: 'get_relative_imports —— 正则抓、set 去重、顺序留给调用方补。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    const lr = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const left = W.card({ cc: 0, tint: 0, title: '正则只抓两种形态',
      body: U.el('div', { class: 'mono', style: 'font-size:11px;line-height:1.8;color:var(--ink-dim);overflow-wrap:anywhere' }, [
        U.el('div', { html: '<span class="hl">^\\s*import\\s+\\.(\\S+)\\s*$</span> &nbsp;←&nbsp; import .configuration_x' }),
        U.el('div', { html: '<span class="hl">^\\s*from\\s+\\.(\\S+)\\s+import</span> &nbsp;←&nbsp; from .utils import X' }),
        U.el('div', { class: 'dim', style: 'margin-top:6px', text: '名单进 set：去重，同时丢掉顺序' }),
      ]) });
    const right = W.card({ cc: 4, tint: 4, title: '递归：从 1 个文件到闭包',
      body: U.el('div', { class: 'mono', style: 'font-size:11px;line-height:1.8;color:var(--ink-dim);overflow-wrap:anywhere' }, [
        U.el('div', { html: 'modeling_demo.py <span class="dim">→ {configuration, utils}</span>' }),
        U.el('div', { html: 'configuration_demo.py <span class="dim">→ {utils}</span>' }),
        U.el('div', { html: 'utils_demo.py <span class="dim">→ {}</span> &nbsp;<span class="hl4">停止</span>' }),
        U.el('div', { class: 'dim', style: 'margin-top:6px', text: '实测闭包 = configuration_demo.py + utils_demo.py' }),
      ]) });
    left.style.flex = '1'; right.style.flex = '1';
    lr.appendChild(left); lr.appendChild(right);
    viz.appendChild(lr);

    const stats = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    stats.appendChild(W.card({ cc: 2, tint: 2,
      title: '<span class="mono" style="font-size:11.5px">list(set(...))</span>',
      sub: '去重但不保证顺序 —— 所以 get_class_in_module 里补一次 sorted()。哈希必须稳定。',
      style: 'flex:1' }));
    stats.appendChild(W.card({ cc: 5, tint: 5,
      title: '<span class="mono" style="font-size:11.5px">实测 18</span>',
      sub: 'modeling_glm5_next.py 的 get_relative_imports 返回 18 个目标：缓存/工具/生成/掩码都在里面。',
      style: 'flex:1' }));
    viz.appendChild(stats);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    left.style.opacity = '.3'; right.style.opacity = '.3';
    msg.innerHTML = '<span class="cm">// 第一步：从文件内容里把"相对导入的名字"捞出来</span>';
    tl.at(1800, () => { left.style.opacity = '1'; msg.innerHTML = '两条正则，对应两种写法。<em>\\S+ 会把 ..activations 也一起吞进来</em>'; });
    tl.at(4800, () => { right.style.opacity = '1'; msg.innerHTML = '第二步：拿名字拼路径，<em>递归</em>问下去，直到没有新文件'; });
    tl.at(8200, () => {
      msg.innerHTML = '实测：这个模型主干有 <em>18</em> 个相对导入目标，全部会被卷进哈希';
    });
    tl.at(11600, () => {
      msg.innerHTML = '<span class="cm">// 正则会误报（字符串里的 import 也算），取舍是"宁可多带一个路径"</span>';
    });
    tl.at(14000, () => {
      msg.innerHTML = '闭包算完，下一幕就是：<em>这些文件的字节怎么变成一个判据</em>';
    });
  },
},

/* ------------------------------------------------- 4 ★ 缓存与校验 */
{
  kicker: '第三步 · ★ 校验',
  title: '★ 校验的粒度：<span class="hl-a">依赖闭包的字节</span>',
  sub: '缓存判据只有一行比较：模块对象上记着的哈希，和这次算出来的哈希一样吗？一样就不重新执行。',
  caption: '两层缓存：sys.modules 管进程内复用，磁盘上的 transformers_modules/ 管跨进程复用 —— 后者靠目录名，前者靠这行比较。',
  lang: 'python',
  codeStart: 300,
  code: `        module: ModuleType
        if cached_module is None:
            module = importlib.util.module_from_spec(module_spec)
            # insert it into sys.modules before any loading begins
            sys.modules[name] = module
        else:
            module = cached_module
        # reload in both cases, unless the module is already imported and the hash hits
        if getattr(module, "__transformers_module_hash__", "") != module_hash:
            module_spec.loader.exec_module(module)
            module.__transformers_module_hash__ = module_hash
        return getattr(module, class_name)`,
  codeNote: 'get_class_in_module 的后半段 —— 整个"缓存与校验策略"的落点。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '哈希的输入：模块本身 + 它递归依赖的每一个文件' }));
    const files = ['modeling_demo.py', 'configuration_demo.py', 'utils_demo.py'];
    const frow = U.el('div', { class: 'row gap8 center', style: 'width:100%' });
    const fEls = files.map((f, i) => {
      const e = U.el('div', { class: 'chip c' + (i % 6), style: 'height:26px', text: f });
      frow.appendChild(e); return e;
    });
    frow.appendChild(U.el('div', { class: 'mono', style: 'font-size:14px;color:var(--accent);padding:0 4px', text: '⟹' }));
    const hashChip = U.el('div', { class: 'pill acc', style: 'height:26px', text: 'sha256 → 52a224fd204265ae' });
    frow.appendChild(hashChip);
    viz.appendChild(frow);

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const hit = W.card({ cc: 4, tint: 4, title: '哈希相等：不 exec',
      sub: '直接 getattr(module, class_name)，把内存里那个类交出去。跳过一次 exec_module，就是省下整条 import 的时间。',
      style: 'flex:1' });
    const miss = W.card({ cc: 3, tint: 3, title: '哈希不等：exec 并写回',
      sub: 'module_spec.loader.exec_module(module) 之后把新哈希记回模块对象 —— 下一轮比较才有依据。',
      style: 'flex:1' });
    row.appendChild(hit); row.appendChild(miss);
    viz.appendChild(row);

    const layers = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    layers.appendChild(W.card({ cc: 0, title: '进程内：sys.modules',
      sub: '__transformers_module_hash__ 挂在模块对象上，进程退出就没了。锁 _HF_REMOTE_CODE_LOCK 保护的正是它。',
      style: 'flex:1' }));
    layers.appendChild(W.card({ cc: 2, title: '跨进程：磁盘目录名',
      sub: 'transformers_modules/<name>/<16 位哈希>/ —— 目录名本身就是"这份代码的身份"。',
      style: 'flex:1' }));
    viz.appendChild(layers);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    fEls.forEach(e => { e.style.opacity = '.3'; });
    hashChip.style.opacity = '.3';
    msg.innerHTML = '<span class="cm">// 参与哈希的是内容，不是文件名：名字不进 sha256</span>';
    tl.at(1600, () => { fEls.forEach(e => { e.style.opacity = '1'; }); msg.innerHTML = '模块文件 + <em>递归依赖的每个文件</em>，字节依次喂进同一个 sha256'; });
    tl.at(4600, () => { hashChip.style.opacity = '1'; msg.innerHTML = '实测产物：<em>16 位十六进制</em>（本条是临时仓库跑出来的真值）'; });
    tl.at(7600, () => { hit.classList.add('ac'); miss.classList.remove('ac'); msg.innerHTML = '<em>相等</em> → 复用 sys.modules 里的模块对象，连 exec 都省了'; });
    tl.at(10800, () => { hit.classList.remove('ac'); miss.classList.add('ac'); msg.innerHTML = '<em>不等</em> → 重新 exec，并把新哈希写回模块对象'; });
    tl.at(14200, () => {
      miss.classList.remove('ac');
      msg.innerHTML = '<span class="cm">// 校验粒度是"整个闭包"：任何一个被依赖的文件改了，主模块就重新执行</span>';
    });
  },
},

/* ------------------------------------------------- 5 本地路径 */
{
  kicker: '第四步 · 落地命名',
  title: '本地代码：目录名就是<span class="hl-a">版本号</span>',
  sub: '远端仓库有 commit hash 可用，本地目录没有 —— 于是 transformers 自己把源码内容算成一个哈希，垫在目录名里。',
  caption: '目录级哈希决定"进不进这个版本"，文件级 filecmp 决定"这个文件还要不要重写"。',
  lang: 'python',
  codeStart: 438,
  code: `    # Check we have all the requirements in our environment
    modules_needed = check_imports(resolved_module_file)
    if is_local:
        local_model_name = _sanitize_module_name(os.path.basename(os.path.normpath(pretrained_model_name_or_path)))
        local_source_files_hash = _compute_local_source_files_hash(pretrained_model_name_or_path, resolved_module_file)
        if local_model_name:
            submodule = os.path.sep.join([local_model_name, local_source_files_hash])
        else:
            submodule = local_source_files_hash

    # Now we move the module inside our cached dynamic modules.
    full_submodule = TRANSFORMERS_DYNAMIC_MODULE_NAME + os.path.sep + submodule
    create_dynamic_module(full_submodule)
    submodule_path = Path(HF_MODULES_CACHE) / full_submodule`,
  codeNote: 'get_cached_module_file 的本地分支 —— 缺包检查 + 两级目录名。',
  duration: 16000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '实测路径：一次本地加载之后，类名会长成这样' }));
    const pathRow = U.el('div', { class: 'row gap6 center wrap', style: 'width:100%' });
    ['transformers_modules', 'demo_repo', 'a32f4f65faa60c4e', 'modeling_demo.py'].forEach((p, i) => {
      pathRow.appendChild(U.el('div', { class: 'chip c' + [0, 4, 2, 5][i], style: 'height:26px', text: p }));
      if (i < 3) pathRow.appendChild(U.el('div', { class: 'mono', style: 'color:var(--ink-faint);font-size:12px', text: '/' }));
    });
    viz.appendChild(pathRow);

    const cls = U.el('div', { class: 'formula', style: 'width:100%' });
    cls.innerHTML = '<span class="cm"># 实测：get_class_in_module 返回的类</span><br>'
      + 'DemoModel &nbsp;<span class="op">=</span>&nbsp; <em>transformers_modules.demo_repo.a32f4f65faa60c4e.modeling_demo</em>.DemoModel';
    viz.appendChild(cls);

    const tb = W.table([
      ['远端仓库', '<code class="inl">commit_hash</code>', 'git 已经给了内容身份'],
      ['本地目录', '<code class="inl">local_source_files_hash</code>', '16 位十六进制，自己算'],
      ['<span class="hl4">共同点</span>', '<span class="hl4">目录名 = 内容身份</span>', '换内容 = 换目录 = 不串味'],
    ], { head: ['代码从哪来', '第二级目录名', '理由'] });
    viz.appendChild(U.el('div', { class: 'card', cc: 1, style: 'padding:10px 12px;width:100%' }, [tb]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 先补齐依赖，再决定目录名</span>';
    tl.at(2000, () => { msg.innerHTML = '<em>check_imports</em> 先跑：缺哪个包就在这一步抛 ImportError，而不是等到 exec'; });
    tl.at(5200, () => { msg.innerHTML = '本地：<em>目录名 + 源码哈希</em> 两级；远端：<em>目录名 + commit</em> 两级'; });
    tl.at(8600, () => { msg.innerHTML = '同一份代码算两次哈希 → 同一个目录 → <em>直接复用</em>，不重复拷贝'; });
    tl.at(11800, () => {
      msg.innerHTML = '<span class="cm">// 文件级还有一道 filecmp：内容一样就不重写，避免多余的 invalidate_caches</span>';
    });
  },
},

/* ------------------------------------------------- 6 ★ 信任三态 */
{
  kicker: '第五步 · ★ 信任',
  title: '★ trust_remote_code 不是布尔，是<span class="hl-a">三个出口</span>',
  sub: '前面整条链都发生在"用户已经同意"之后。同意这件事本身也有三种结局：不问、问、拒绝。',
  caption: '实测：TIME_OUT_REMOTE_CODE = 15 秒；非交互环境下 input() 抛异常，被兜底成一句明确的 ValueError。',
  lang: 'python',
  codeStart: 754,
  code: `    if trust_remote_code is None:
        if has_local_code:
            trust_remote_code = False
        elif has_remote_code and TIME_OUT_REMOTE_CODE > 0:
            prev_sig_handler = None
            try:
                prev_sig_handler = signal.signal(signal.SIGALRM, _raise_timeout_error)
                signal.alarm(TIME_OUT_REMOTE_CODE)
                while trust_remote_code is None:
                    answer = input(
                        f"{error_message} You can inspect the repository content at https://hf.co/{model_name}.\\n"
                        f"You can avoid this prompt in future by passing the argument \`trust_remote_code=True\`.\\n\\n"
                        f"Do you wish to run the custom code? [y/N] "
                    )
                    if answer.lower() in ["yes", "y", "1"]:
                        trust_remote_code = True
                    elif answer.lower() in ["no", "n", "0", ""]:
                        trust_remote_code = False
                signal.alarm(0)`,
  codeNote: '三态判定 + signal 超时 + 提问循环，全部在 20 行里。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '四种组合的真实结局（实测，不是推测）' }));
    const tb = W.table([
      ['本地有实现', '任意', '<span class="hl4">False</span>', '根本不问，走库内代码'],
      ['仅远端', 'True', '<span class="hl4">True</span>', '执行远端代码'],
      ['仅远端', 'False', '<span class="hlbad">ValueError</span>', '最后一道闸拦下'],
      ['仅远端', 'None', '<span class="hl3">提问 15 秒</span>', '非 TTY 时抛 ValueError'],
    ], { head: ['has_local_code', 'trust_remote_code', '解析结果', '接下来'] });
    const tbBox = U.el('div', { class: 'card', cc: 3, style: 'padding:10px 12px;width:100%' }, [tb]);
    viz.appendChild(tbBox);
    const trs = Array.from(tb.querySelectorAll('tbody tr'));

    const timerBox = U.el('div', { class: 'card', cc: 2, style: 'padding:10px 12px;width:100%' });
    timerBox.appendChild(U.el('div', { class: 'klabel', text: '提问的硬上限：signal.alarm(TIME_OUT_REMOTE_CODE) —— 实测 15 秒' }));
    const rail = U.el('div', { style: 'width:100%;height:12px;border-radius:6px;background:rgba(150,180,255,.10);border:1px solid var(--panel-brd);overflow:hidden;margin-top:6px' });
    const fill = U.el('div', { style: 'width:0;height:100%;border-radius:5px;background:linear-gradient(90deg,var(--c4),var(--c2),var(--c3));transition:width 3.2s cubic-bezier(.65,0,.35,1)' });
    rail.appendChild(fill);
    timerBox.appendChild(rail);
    timerBox.appendChild(U.el('div', { class: 'mono', style: 'font-size:11px;color:var(--ink-faint);margin-top:6px', text: 'y / yes / 1 → True　　回车 / n / no / 0 → False（默认是 No）' }));
    viz.appendChild(timerBox);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 本地有实现时，连提问都不会发生</span>';
    tl.at(1800, () => {
      trs.forEach((r, i) => r.classList.toggle('ac', i === 0));
      msg.innerHTML = '第一行：<em>has_local_code</em> 为真 → 直接 False，这一整套流程根本不启动';
    });
    tl.at(5000, () => {
      trs.forEach((r, i) => r.classList.toggle('ac', i === 1));
      msg.innerHTML = '第二行：远端代码 + 用户显式 <em>True</em> → 放行';
    });
    tl.at(8200, () => {
      trs.forEach((r, i) => r.classList.toggle('ac', i === 2));
      msg.innerHTML = '第三行：<em>False</em> → 撞在最后一道闸上，抛 ValueError';
    });
    tl.at(11400, () => {
      trs.forEach((r, i) => r.classList.toggle('ac', i === 3));
      fill.style.width = '100%';
      msg.innerHTML = '第四行：<em>None</em> → 提问，最多等 15 秒；没人回答就报错，不会永久挂住';
    });
    tl.at(15200, () => {
      trs.forEach(r => r.classList.remove('ac'));
      msg.innerHTML = '<span class="cm">// has_local_code 来自 AutoModel 的注册表 —— 这就是 L1-05/L1-06 与本课的接缝</span>';
    });
  },
},

/* ------------------------------------------------- 7 _typing */
{
  kicker: '第二个文件 · 类型契约',
  title: '类型契约：<span class="hl-a">别名</span>统一写法，<span class="hl-b">Protocol</span> 只给检查器看',
  sub: '_typing.py 一共 185 行，运行期一个对象都不产生 —— 它的全部价值是"让几百个文件里的写法一致"。',
  caption: '实测纠偏：这个版本里没有 TensorType（它在 utils/generic.py），本文件的别名就是下面这三个。',
  lang: 'python',
  codeStart: 28,
  code: `# A few helpful type aliases
Level: TypeAlias = int
ExcInfo: TypeAlias = (
    None
    | bool
    | BaseException
    | tuple[type[BaseException], BaseException, object]  # traceback is \`types.TracebackType\`, but keep generic here
)
DeviceMeshLike: TypeAlias = Any  # PyTorch stubs do not model torch.distributed.device_mesh consistently yet.`,
  codeNote: '三个别名，三种绕开类型噪音的姿势。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '三个别名：把"为什么这么宽 / 这么窄"写成注释' }));
    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const alias = [
      { cc: 0, t: 'Level', s: 'TypeAlias = int', d: 'logging 的等级本来就是 int，起个名字让签名自解释。' },
      { cc: 4, t: 'ExcInfo', s: 'None | bool | BaseException | tuple', d: '四种合法形态写全；第三项用 object 而不是 TracebackType，宁可宽一点。' },
      { cc: 2, t: 'DeviceMeshLike', s: 'TypeAlias = Any', d: 'PyTorch stub 还没稳定建模它 —— 显式退化成 Any，并留注释说明为什么。' },
    ];
    const aEls = alias.map(a => {
      const c = W.card({ cc: a.cc, tint: a.cc,
        title: '<span class="mono" style="font-size:12px">' + a.t + '</span>',
        sub: '<span class="mono" style="font-size:10.5px">' + a.s + '</span><br>' + a.d });
      c.style.flex = '1';
      row.appendChild(c); return c;
    });
    viz.appendChild(row);

    const proto = U.el('div', { class: 'card', cc: 1, style: 'padding:11px 13px;width:100%' });
    proto.appendChild(U.el('div', { class: 'ct', html: 'GenerativePreTrainedModel &nbsp;<span class="mono dim" style="font-size:11px">(Protocol)</span>' }));
    const pcols = U.el('div', { class: 'row gap16', style: 'width:100%;align-items:flex-start' });
    pcols.appendChild(U.el('div', { class: 'col gap4', style: 'flex:1' }, [
      U.el('div', { class: 'klabel', text: '它会读的属性' }),
      U.el('div', { class: 'mono', style: 'font-size:11px;line-height:1.7;color:var(--ink-dim);overflow-wrap:anywhere',
        text: 'config · device · dtype · main_input_name · base_model_prefix · hf_quantizer · hf_device_map · generation_config' }),
    ]));
    pcols.appendChild(U.el('div', { class: 'col gap4', style: 'flex:1' }, [
      U.el('div', { class: 'klabel', text: '它会调的方法' }),
      U.el('div', { class: 'mono', style: 'font-size:11px;line-height:1.7;color:var(--ink-dim);overflow-wrap:anywhere',
        text: 'forward · __call__ · can_generate · get_input_embeddings · set_output_embeddings · _supports_logits_to_keep' }),
    ]));
    proto.appendChild(pcols);
    viz.appendChild(proto);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    aEls.forEach(e => { e.style.opacity = '.3'; });
    proto.style.opacity = '.3';
    msg.innerHTML = '<span class="cm">// TypeAlias 只是注解，运行期不产生对象</span>';
    alias.forEach((a, i) => {
      tl.at(1600 + i * 2400, () => {
        aEls.forEach((e, k) => { e.style.opacity = (k === i) ? '1' : '.3'; });
        msg.innerHTML = '<em>' + a.t + '</em> &nbsp;<span class="cm">// ' + a.d + '</span>';
      });
    });
    tl.at(9400, () => {
      aEls.forEach(e => { e.style.opacity = '1'; });
      proto.style.opacity = '1';
      msg.innerHTML = '<em>Protocol</em> 不是基类：文档字符串写明"运行期完全不用，只为类型检查器解析 self.&lt;attr&gt;"';
    });
    tl.at(13000, () => {
      msg.innerHTML = '<span class="cm">// 鸭子类型被写成合同：GenerationMixin 只要求宿主提供这两列名字</span>';
    });
    tl.at(15200, () => {
      msg.innerHTML = '同文件另外三个 Protocol 分工相同 —— <em>结构化契约</em>，运行时零成本';
    });
  },
},

/* ------------------------------------------------- 8 ★ 权重名映射 */
{
  kicker: '第三个文件 · ★ 映射表',
  title: '★ 权重名映射：<span class="hl-a">97 条别名</span> + 118 条手写 = 210 个 model_type',
  sub: '磁盘上的名字和内存里的名字不一样，中间靠两张表：一张说"我长得像谁"，一张写真正的改名与张量操作。',
  caption: '实测：未知 model_type 返回 None 而不是空列表 —— 这个区别决定了查找还能不能回退。',
  lang: 'python',
  codeStart: 1958,
  code: `    for model_type, base_pattern in _MODEL_TO_CONVERSION_PATTERN.items():
        if model_type in mapping:
            continue
        mapping[model_type] = mapping[base_pattern].copy()

    return mapping


_checkpoint_conversion_mapping_cache = None


def get_checkpoint_conversion_mapping(model_type):
    global _checkpoint_conversion_mapping_cache
    if _checkpoint_conversion_mapping_cache is None:
        _checkpoint_conversion_mapping_cache = _build_checkpoint_conversion_mapping()
    return deepcopy(_checkpoint_conversion_mapping_cache.get(model_type))`,
  codeNote: '两张表在这里合流：别名表补 92 个新键，加上 118 个手写键得 210。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '最终 210 个 model_type 是怎么凑出来的（实测）' }));
    const st = W.stack([
      { label: '手写 118', value: M.handKeys, cc: 0 },
      { label: '别名补齐 92', value: M.aliasNew, cc: 2 },
    ], { style: 'width:100%' });
    viz.appendChild(st);
    viz.appendChild(U.el('div', { class: 'mono', style: 'font-size:11px;color:var(--ink-faint)',
      text: '118 = 101 条字典字面量 + 17 条 mapping["X"] = [...]　｜　97 条别名里 5 条与手写键重名，被 continue 跳过' }));

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    row.appendChild(W.card({ cc: 2, tint: 2, title: '_MODEL_TO_CONVERSION_PATTERN',
      sub: '97 条"我长得像谁"。同一类专家布局被几十个模型共享，没有它就要抄几十份列表。当前文件的 key 混着 model_type 与类名两套命名空间。',
      style: 'flex:1' }));
    row.appendChild(W.card({ cc: 5, tint: 5, title: 'mapping["glm5_next"]',
      sub: '13 条变换：10 条 WeightRenaming（只改名）+ 3 条 WeightConverter（改名 + 张量操作）。GLM-5 文本主干就在这张表里。',
      style: 'flex:1' }));
    viz.appendChild(row);

    const conv = U.el('div', { class: 'card', cc: 1, style: 'padding:11px 13px;width:100%' });
    conv.appendChild(U.el('div', { class: 'klabel', text: 'glm5_next 的三条 WeightConverter：合并与堆叠' }));
    conv.appendChild(U.el('div', { class: 'mono', style: 'font-size:11px;line-height:1.8;color:var(--ink-dim);overflow-wrap:anywhere' }, [
      U.el('div', { html: 'experts.*.gate_proj + up_proj <span class="op">→</span> gate_up_proj &nbsp;<span class="cm">[MergeModulelist(dim=0), Concatenate(dim=1)]</span>' }),
      U.el('div', { html: 'experts.*.down_proj &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span class="op">→</span> down_proj &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="cm">[MergeModulelist(dim=0)]</span>' }),
      U.el('div', { html: 'q_conv1d + k_conv1d + v_conv1d <span class="op">→</span> conv1d &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="cm">[Concatenate(dim=0)]</span>' }),
    ]));
    viz.appendChild(conv);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    st.reveal(0);
    msg.innerHTML = '<span class="cm">// 先看手写的那部分：118 个键，每个都写明了改名或张量操作</span>';
    tl.at(1800, () => { st.reveal(1); msg.innerHTML = '别名表再补 <em>92</em> 个全新键 → 最终 <em>210</em> 个 model_type 有映射'; });
    tl.at(5200, () => { msg.innerHTML = '.copy() 是<em>浅拷贝</em>：list 各一份，但里面的变换对象是共享的，必须当只读'; });
    tl.at(8600, () => { msg.innerHTML = '读的入口 <em>get_checkpoint_conversion_mapping</em>：首次调用才构建，返回前 deepcopy'; });
    tl.at(12000, () => { msg.innerHTML = '实测：未知 model_type 返回 <em>None</em>，不是空列表 —— 调用方靠它决定要不要回退'; });
    tl.at(15200, () => { msg.innerHTML = '<span class="cm">// 缓存的代价：可变对象必须深拷贝出去，否则子模块的 scope_prefix 会互相覆盖</span>'; });
  },
},

/* ------------------------------------------------- 9 收束 */
{
  kicker: '收束',
  title: '收束：从字符串到类的<span class="hl-a">两条路</span>',
  sub: '动态模块那条路靠文件系统 + 哈希；权重名那条路靠两张表 + 类名优先。两条路合起来，就是 config 里一个字符串的下半生。',
  caption: '下一课 L2-01：输入流水线 —— 文本和像素如何变成同一串位置上的向量。',
  lang: 'python',
  codeStart: 2014,
  code: `    class_name = type(model).__name__
    model_type = model.config.model_type

    # Class name takes priority — allows ForXxx-specific overrides
    conversions = get_checkpoint_conversion_mapping(class_name)
    if conversions is None and model_type:
        conversions = get_checkpoint_conversion_mapping(model_type)
    return conversions`,
  codeNote: 'extract_weight_conversions_for_model —— 类名优先，model_type 兜底。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const tb = W.table([
      ['<span class="mono">dynamic_module_utils.py</span>', '字符串 → 模块 → 类',
       '依赖闭包的字节哈希；<span class="hl4">相同就复用</span>'],
      ['<span class="mono">_typing.py</span>', '注解 → 契约',
       'Protocol + 别名，<span class="hl4">运行期零成本</span>'],
      ['<span class="mono">conversion_mapping.py</span>', '磁盘名 → 内存名',
       '两张表；<span class="hl4">类名优先，model_type 兜底</span>'],
    ], { head: ['文件', '它回答的问题', '判据 / 策略'] });
    wrap.appendChild(U.el('div', { class: 'card', cc: 0, style: 'padding:10px 12px;width:100%' }, [tb]));
    const trs = Array.from(tb.querySelectorAll('tbody tr'));

    const fl = W.flow([
      { t: '类名', s: 'type(model).__name__', cc: 1 },
      { t: 'model_type', s: 'config.model_type', cc: 0 },
      { t: 'None', s: '没有专门规则', cc: 3 },
    ], { style: 'width:100%' });
    wrap.appendChild(U.el('div', { class: 'col gap6', style: 'width:100%' }, [
      U.el('div', { class: 'klabel', text: '权重名映射的查找顺序（glm5_next_text → glm5_next 走的就是第二条）' }), fl,
    ]));

    wrap.appendChild(W.exercise(
      '别名表写的是 <code class="inl">mapping[base_pattern].copy()</code>。'
      + '那么改 <code class="inl">mapping["glm5_next_text"]</code> 和改 '
      + '它里面第 0 个变换的 <code class="inl">source_patterns</code>，效果一样吗？',
      '不一样。<code class="inl">.copy()</code> 是<b>浅拷贝</b>：两个 model_type 拿到的是两个不同的 list 对象，'
      + '往 list 里 append 互不影响；但 list 里装的是<b>同一个变换对象</b>，'
      + '改它的 <code class="inl">source_patterns</code> 两边都会变。<br>'
      + '所以别名表共享的变换必须当成只读 —— 这也正是 '
      + '<code class="inl">get_checkpoint_conversion_mapping</code> 返回前要 <code class="inl">deepcopy</code> 的原因：'
      + '调用方会往变换上写 <code class="inl">scope_prefix</code>。'));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 三个文件，三种"名字的语法"</span>';
    trs.forEach((r, i) => {
      tl.at(1600 + i * 2200, () => {
        trs.forEach((x, k) => x.classList.toggle('ac', k === i));
        msg.innerHTML = ['<em>字符串 → 类</em>：哈希是判据，sys.modules 是缓存',
                         '<em>注解 → 契约</em>：Protocol 描述宿主，运行期不执行',
                         '<em>磁盘名 → 内存名</em>：类名优先，model_type 兜底'][i];
      });
    });
    tl.at(8600, () => {
      trs.forEach(x => x.classList.remove('ac'));
      fl.focus(0);
      msg.innerHTML = '先按<em>类名</em>查：任务头可以覆盖架构族的规则';
    });
    tl.at(11200, () => { fl.focus(1); msg.innerHTML = '查不到再按 <em>model_type</em>：GLM-5 文本主干就是走这条'; });
    tl.at(13800, () => {
      fl.focus(2);
      msg.innerHTML = '两条都没有 → <em>None</em>：不是"空规则"，而是"我这儿没规则"';
    });
    tl.at(16200, () => {
      msg.innerHTML = '<span class="cm">// 下一课 L2-01：字符串进了模型之后，文本和像素怎么排到同一条序列上</span>';
    });
  },
},

];
