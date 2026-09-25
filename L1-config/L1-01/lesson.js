/* ==========================================================================
   L1-01 · PreTrainedConfig：配置对象的公共契约
   --------------------------------------------------------------------------
   覆盖：configuration_utils.py（1 个文件 / 1566 行）
   目标：看完能画出 config 从 json 到对象再到模型的路径，并说清 @strict 在什么时机校验、
        嵌套配置靠哪两个 ClassVar 寻址。
   排版基准：#visual 可视区 = 994 x 662.7 舞台像素（无头浏览器实测）。
   ========================================================================== */
'use strict';

const SCENES = [

/* ------------------------------------------------- 1 全景：两条路 */
{
  kicker: '第 1 层 · 配置与分派',
  title: '一份 <span class="hl-a">config.json</span> 的两条路',
  sub: '进：文件 → 字典 → 对象 → 模型；出：对象 → 差异字典 → 文件。这一课只讲这台机器，不讲任何一个模型的字段。',
  caption: '回顾 L0-01：那一课里的 self.config，就是这一课造出来的对象。',
  lang: 'python',
  codeStart: 178,
  code: `@dataclass_transform(kw_only_default=True)
@strict(accept_kwargs=True)
@dataclass(repr=False)
class PreTrainedConfig(PushToHubMixin, RotaryEmbeddingConfigMixin, HeterogeneousConfigMixin):`,
  codeNote: '三个装饰器叠在类上 —— 本课第四节要拆的就是中间那个 @strict。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '同一个对象，两条路' }));
    const fl = W.flow([
      { t: 'config.json', s: '磁盘 / Hub', cc: 0 },
      { t: 'dict', s: 'json.loads', cc: 2 },
      { t: 'config 对象', s: '@strict 校验', cc: 4 },
      { t: 'model', s: '__init__(config)', cc: 1 },
      { t: '形状', s: '45 层 · 4 条流', cc: 3 },
    ], { style: 'width:100%' });
    viz.appendChild(fl);

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const cIn = W.card({ cc: 0, tint: 0, title: '进：from_pretrained',
      sub: 'cached_file → json.loads → _decode_special_floats → cls(**config_dict)',
      style: 'flex:1;overflow-wrap:anywhere' });
    const cChk = W.card({ cc: 4, tint: 4, title: '校验：@strict',
      sub: '构造完成时跑一遍 validate_*；save_pretrained 写盘前再跑一遍',
      style: 'flex:1;overflow-wrap:anywhere' });
    const cOut = W.card({ cc: 1, tint: 1, title: '出：to_json_string',
      sub: 'to_dict → to_diff_dict → _encode_special_floats → json.dumps',
      style: 'flex:1;overflow-wrap:anywhere' });
    row.appendChild(cIn); row.appendChild(cChk); row.appendChild(cOut);
    viz.appendChild(row);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 1566 行，只干三件事：读进来、查一遍、写回去</span>';
    tl.at(2000, () => { fl.focus(0); msg.innerHTML = '起点：<em>config.json</em> —— 纯文本，里面没有类，只有字符串与数字'; });
    tl.at(5200, () => { fl.focus(1); msg.innerHTML = '<span class="fn">json.loads</span> 之后是 dict；特殊浮点在文件这条路上被还原'; });
    tl.at(8600, () => { fl.focus(2); msg.innerHTML = '★ <em>cls(**config_dict)</em>：字典变成对象，<span class="hl">@strict 在构造末尾校验</span>'; });
    tl.at(12000, () => { fl.focus(3); msg.innerHTML = '<span class="fn">PreTrainedModel.__init__(config)</span> 之后，config 的每个字段都变成一层网络'; });
    tl.at(14600, () => {
      fl.focus(-1); cChk.classList.add('ac');
      msg.innerHTML = '<span class="cm">// 本课的主角是中间那个方块 —— 校验到底发生在什么时候</span>';
    });
  },
},

/* ------------------------------------------------- 2 入口：from_pretrained */
{
  kicker: '入口 · from_pretrained',
  title: '四个形参塞回 <span class="hl-a">kwargs</span>，然后只做三件事',
  sub: 'get_config_dict 拿字典、base_config_key 下钻、from_dict 造对象 —— 前面 80 行文档字符串背后只有这 8 行。',
  caption: 'base_config_key 是类属性：顶层是空串，子配置是 "text_config"。',
  lang: 'python',
  codeStart: 737,
  code: `        kwargs["cache_dir"] = cache_dir
        kwargs["force_download"] = force_download
        kwargs["local_files_only"] = local_files_only
        kwargs["revision"] = revision

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if cls.base_config_key and cls.base_config_key in config_dict:
            config_dict = config_dict[cls.base_config_key]`,
  codeNote: 'get_config_dict 返回的是 (dict, kwargs) 一对 —— 第二个返回值是还没被消费的 kwargs。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: 'from_pretrained 的尾部：四步走完' }));
    const row = U.el('div', { class: 'row gap10', style: 'width:100%;align-items:stretch' });
    const steps = [
      { cc: 0, t: '① 打包参数', s: '四个具名形参写回 kwargs；下游只认一个 **kwargs' },
      { cc: 2, t: '② 拿字典', s: 'cls.get_config_dict(...) 返回 (config_dict, kwargs)' },
      { cc: 4, t: '③ 下钻', s: 'cls.base_config_key 非空时，取字典里的那一格' },
      { cc: 1, t: '④ 造对象', s: 'cls.from_dict(config_dict, **kwargs)' },
    ];
    const els = steps.map(st => {
      const c = W.card({ cc: st.cc, tint: st.cc, title: st.t, sub: st.s,
        style: 'flex:1;overflow-wrap:anywhere' });
      row.appendChild(c); return c;
    });
    viz.appendChild(row);

    const drill = U.el('div', { class: 'card cc4', style: 'padding:11px 13px;width:100%' });
    drill.appendChild(U.el('div', { class: 'klabel', text: '实测：子配置类去读顶层文件，会自动下钻' }));
    const chain = U.el('div', { class: 'row gap8 center wrap', style: 'width:100%' });
    chain.innerHTML =
      '<span class="chip c0">Glm5NextTextConfig.from_pretrained(dir)</span>' +
      '<span class="arrow" style="max-width:28px"></span>' +
      '<span class="chip c2">config.json</span>' +
      '<span class="arrow" style="max-width:28px"></span>' +
      '<span class="chip c4">config_dict["text_config"]</span>' +
      '<span class="arrow" style="max-width:28px"></span>' +
      '<span class="chip c1">num_hidden_layers = 45</span>';
    drill.appendChild(chain);
    drill.appendChild(U.el('div', {
      class: 'mono', style: 'font-size:11px;color:var(--ink-faint);margin-top:7px',
      text: 'base_config_key = "text_config" 时这一跳自动发生；实测结果与 Glm5NextTextConfig() 相等。' }));
    viz.appendChild(drill);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 四个形参只是被搬了个家</span>';
    tl.at(2200, () => { els[0].classList.add('ac'); msg.innerHTML = 'cache_dir / force_download / local_files_only / revision <em>全部塞回 kwargs</em>'; });
    tl.at(5600, () => { els[1].classList.add('ac'); msg.innerHTML = '<span class="fn">get_config_dict</span> 返回一对：字典 + 尚未消费的 kwargs'; });
    tl.at(9000, () => { els[2].classList.add('ac'); msg.innerHTML = '★ <em>base_config_key</em> 非空 → <span class="hl">config_dict = config_dict[cls.base_config_key]</span>'; });
    tl.at(12400, () => { els[3].classList.add('ac'); msg.innerHTML = '最后一跳交给 <span class="fn">from_dict</span>：kwargs 在这里覆盖字典里的值'; });
    tl.at(15000, () => {
      els.forEach(e => e.classList.remove('ac'));
      msg.innerHTML = '<span class="cm">// 这 8 行之外，from_pretrained 全是文档字符串</span>';
    });
  },
},

/* ------------------------------------------------- 3 版本解析与二次读取 */
{
  kicker: '入口 · 解析版本',
  title: 'revision 只解析<span class="hl-a">一次</span>，配置文件可能读<span class="hl-b">两份</span>',
  sub: 'main 是会动的分支名。先把它钉成一个 commit，再读文件；读到 configuration_files 就换一份再读一次。',
  caption: '第二次读取用的是 deepcopy 出来的干净 kwargs，否则 cache_dir 与 token 会在第一次 pop 时丢掉。',
  lang: 'python',
  codeStart: 781,
  code: `        # Resolve the revision once, so that both config files below are read from the exact same repository state.
        kwargs["revision"] = resolve_revision(
            pretrained_model_name_or_path,
            kwargs.get("revision"),
            token=kwargs.get("token"),
            local_files_only=kwargs.get("local_files_only", False),
            cache_dir=kwargs.get("cache_dir"),
        )

        original_kwargs = copy.deepcopy(kwargs)
        # Get config dict associated with the base config file
        config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)`,
  codeNote: 'original_kwargs 是为第二次读取准备的未污染副本。',
  duration: 17000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '实测：get_configuration_file 在 5.18.0.dev0 下选哪一份' }));
    const tb = W.table([
      ['config.json', '基础文件', '总是候选'],
      ['config.4.0.json', '版本 4.0', '≤ 5.18 → 候选'],
      ['config.4.30.0.json', '版本 4.30', '≤ 5.18 → 候选'],
      ['config.5.0.0.json', '<span class="hl">版本 5.0.0</span>', '<span class="hl">实测选中</span>'],
      ['config.9.9.9.json', '版本 9.9.9', '<span class="hlbad">比自己新 → 不选</span>'],
    ], { head: ['仓库里的候选', '含义', '结果'] });
    viz.appendChild(U.el('div', { class: 'card cc0', style: 'padding:10px 12px;width:100%' }, [tb]));

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    row.appendChild(W.card({ cc: 2, tint: 2, title: '为什么先钉住 revision',
      sub: '第一次读 config.json、第二次读 config.5.0.0.json；如果 main 在两次之间动了，两份文件来自不同仓库状态。',
      style: 'flex:1;overflow-wrap:anywhere' }));
    row.appendChild(W.card({ cc: 1, tint: 1, title: '为什么 deepcopy',
      sub: '_get_config_dict 会 pop 掉 cache_dir / token / subfolder 等键。留一份副本，第二次读取才不至于变成匿名下载。',
      style: 'flex:1;overflow-wrap:anywhere' }));
    viz.appendChild(row);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 一次 from_pretrained，可能读两个文件</span>';
    tl.at(2400, () => { msg.innerHTML = '第一次：<em>_get_config_dict</em> 读基础文件，拿到 config_dict'; });
    tl.at(5800, () => { msg.innerHTML = '若字典里有 <span class="hl">configuration_files</span> → <em>get_configuration_file</em> 选一份新的'; });
    tl.at(9200, () => { msg.innerHTML = '第二次：用 <em>original_kwargs</em> 重读；<span class="cm">// 只有 _configuration_file 是新增的</span>'; });
    tl.at(12600, () => { msg.innerHTML = '两次都返回 <em>(config_dict, kwargs)</em>，最终只有第二份进入 from_dict'; });
    tl.at(15000, () => { msg.innerHTML = '<span class="cm">// revision 从 "main" 变成一个确定的 commit —— 两份文件来自同一状态</span>'; });
  },
},

/* ------------------------------------------------- 4 读文件：两个隐藏动作 */
{
  kicker: '路径 · 读文件',
  title: '从文件读和从字典读，<span class="hl-a">不是同一条路</span>',
  sub: 'from_json_file 不经过 from_dict；特殊浮点只在文件那条路上被还原 —— 这是实测出来的差别。',
  caption: 'Infinity / NaN 不是合法 JSON，写出去时被编码成 {"__float__": "Infinity"}。',
  lang: 'python',
  codeStart: 984,
  code: `        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file: str | os.PathLike):
        with open(json_file, encoding="utf-8") as reader:
            text = reader.read()
        config_dict = json.loads(text)

        return cls._decode_special_floats(config_dict)`,
  codeNote: 'from_json_file 的最后一行是 cls(**config_dict)，不是 cls.from_dict(...)。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '同一个 json 文件，两条入口' }));
    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });

    const mkPath = (cc, title, lines, out, outCc) => {
      const c = W.card({ cc: cc, tint: cc, title: title, style: 'flex:1;overflow-wrap:anywhere' });
      const body = U.el('div', { class: 'mono', style: 'font-size:11.5px;line-height:1.8' });
      lines.forEach((t, i) => body.appendChild(U.el('div', {
        html: i === 0 ? '<span class="dim">' + t + '</span>' : t })));
      c.appendChild(body);
      c.appendChild(U.el('div', {
        class: 'mono', style: 'font-size:11.5px;margin-top:8px;color:var(--c' + outCc + ')', html: out }));
      return c;
    };

    const A = mkPath(4, '① from_json_file', [
      '_dict_from_json_file(path)',
      '↓ json.loads(text)',
      '↓ <span class="hl4">_decode_special_floats(dict)</span>',
      '↓ cls(**config_dict)',
    ], '→ rope_theta = <b>-inf</b>', 4);

    const B = mkPath(3, '② from_dict(json.loads(...))', [
      'json.loads(open(path).read())',
      '↓ <span class="hlbad">没有解码这一步</span>',
      '↓ from_dict(d)',
      '↓ cls(**config_dict)',
    ], '→ rope_theta = <b>{"__float__": "-Infinity"}</b>', 3);

    row.appendChild(A); row.appendChild(B);
    viz.appendChild(row);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    A.style.opacity = '.34'; B.style.opacity = '.34';
    msg.innerHTML = '<span class="cm">// 两条路在 cls(**config_dict) 才汇合</span>';
    tl.at(2600, () => {
      A.style.opacity = '1';
      msg.innerHTML = '文件路径：<em>_dict_from_json_file</em> 里多一步 <span class="hl4">_decode_special_floats</span>';
    });
    tl.at(6200, () => {
      B.style.opacity = '1';
      msg.innerHTML = '字典路径：<span class="hlbad">不经过解码</span> —— 实测 rope_theta 停在 {"__float__": "-Infinity"} 这个字典上';
    });
    tl.at(10000, () => {
      msg.innerHTML = '<span class="cm">// 编码在 _encode_special_floats（第 996 行），解码只在读文件那一层</span>';
    });
    tl.at(13400, () => {
      msg.innerHTML = '所以"从文件读"和"从字典读"得到的对象<span class="hl">可能不同</span> —— 这是路径差异，不是 bug';
    });
    tl.at(16200, () => {
      msg.innerHTML = '<span class="cm">// 想从字典读：自己先调 cls._decode_special_floats(d)</span>';
    });
  },
},

/* ------------------------------------------------- 5 ★ @strict */
{
  kicker: '★ 机制 · @strict',
  title: '<span class="hl-a">@strict</span> 把校验挂在「构造完成」那一刻',
  sub: '它来自 huggingface_hub，不是 transformers 自己写的。装了三样东西：逐字段类型校验、收集 validate_*、包一层 __init__。',
  caption: '子类要在自己的文件里再写一次 @strict —— 实测 configuration_glm5_next.py 里有 3 处。',
  lang: 'python',
  codeStart: 400,
  code: `    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls_has_custom_init = "__init__" in cls.__dict__
        # kw_only=True ensures fields without defaults in subclasses can follow
        # parent fields that have defaults (Python dataclass ordering rule).
        # Config fields are always passed as keyword arguments, so this is safe.
        cls = dataclass(cls, repr=False, kw_only=True)

        if not cls_has_custom_init:
            # Wrap all subclasses to accept arbitrary kwargs for BC
            # only if the subclass has no custom \`__init__\`. Most
            # remote code has an init defined, but some model are not
            # See https://huggingface.co/hmellor/Ilama-3.2-1B/blob/main/configuration_ilama.py
            cls = wrap_init_to_accept_kwargs(cls)`,
  codeNote: '__init_subclass__ 在类定义执行完、装饰器还没跑的时候被 Python 自动调用。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '实测：@strict(accept_kwargs=True) 在类上装了三样东西' }));
    const row = U.el('div', { class: 'row gap10', style: 'width:100%;align-items:stretch' });
    const three = [
      { cc: 0, t: '① 重写 __setattr__', s: '按 dataclass 字段类型逐次校验；类型不对立即抛错' },
      { cc: 2, t: '② 收集 validate_*', s: '扫描类里所有只收 self 的 validate_ 方法，存进 __class_validators__' },
      { cc: 4, t: '③ 包一层 __init__', s: '构造结束后调用 cls.validate(self)，把 ValueError 包成严格异常' },
    ];
    const els = three.map(st => {
      const c = W.card({ cc: st.cc, tint: st.cc, title: st.t, sub: st.s,
        style: 'flex:1;overflow-wrap:anywhere' });
      row.appendChild(c); return c;
    });
    viz.appendChild(row);

    const row2 = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    const exCard = W.card({ cc: 3, tint: 3, title: '实测：两种失败',
      body: U.el('div', { class: 'mono', style: 'font-size:11px;line-height:1.75' }, [
        U.el('div', { html: '<span class="hlbad">类级</span> StrictDataclassClassValidationError' }),
        U.el('div', { class: 'dim', html: "validator 'validate_layer_type' → ValueError" }),
        U.el('div', { html: '<span class="hlbad">字段级</span> StrictDataclassFieldValidationError' }),
        U.el('div', { class: 'dim', html: "field 'num_hidden_layers' expected int, got str" }),
      ]), style: 'flex:1.2;overflow-wrap:anywhere' });
    const subCard = W.card({ cc: 5, tint: 5, title: '子类要再写一次 @strict',
      sub: '基类那份只收集基类的 validate_*。Glm5NextTextConfig / Glm5NextVisionConfig / Glm5NextConfig 各自带一个 @strict。',
      style: 'flex:1;overflow-wrap:anywhere' });
    row2.appendChild(exCard); row2.appendChild(subCard);
    viz.appendChild(row2);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 基类装好机器，子类被定义时自动再走一遍 __init_subclass__</span>';
    tl.at(2400, () => { els[0].classList.add('ac'); msg.innerHTML = '赋值时校验：<em>__setattr__</em> 被换掉，类型不对当场报错'; });
    tl.at(5800, () => { els[1].classList.add('ac'); msg.innerHTML = '<em>validate_*</em> 是约定：方法名以 validate_ 开头、只收 self，就会被收进表里'; });
    tl.at(9200, () => {
      els[2].classList.add('ac');
      msg.innerHTML = '★ 关键在第三条：<span class="hl">校验时机 = 构造完成的那一刻</span>，不是字段被赋值的那一刻';
    });
    tl.at(12800, () => {
      els.forEach(e => e.classList.remove('ac'));
      msg.innerHTML = '所以一个非法的 <em>layer_types</em> 组合，在 <span class="fn">Config(...)</span> 这一行就炸，不会拖到模型跑到一半';
    });
    tl.at(15800, () => {
      msg.innerHTML = '<span class="cm">// 装饰器顺序：dataclass 先、strict 后 —— strict 开头就会检查 __dataclass_fields__</span>';
    });
  },
},

/* ------------------------------------------------- 6 ★ validate_ 家族 */
{
  kicker: '★ 机制 · 四个 validate_',
  title: '同一个家族里，<span class="hl-a">谁抛错</span>、<span class="hl-b">谁只警告</span>',
  sub: '「形状错了」抛错，「取值可疑」只警告 —— validate_token_ids 把原因写在了代码注释里。',
  caption: 'Hub 上真的有 pad_token_id=-1 的 config，所以这里不能抛错。',
  lang: 'python',
  codeStart: 567,
  code: `    def validate_layer_type(self):
        """Check that \`mlp_layer_types\` and \`layer_types\` is correctly defined."""
        for allowed_types, layer_types in zip(
            [ALLOWED_ATTN_LAYER_TYPES, ALLOWED_MLP_LAYER_TYPES], ["layer_types", "mlp_layer_types"]
        ):
            layers = getattr(self, layer_types, None)
            if not (layers is not None and hasattr(self, "num_hidden_layers")):
                return

            if not all(layer_type in allowed_types for layer_type in layers):
                raise ValueError(f"The \`{layer_types}\` entries must be in {allowed_types} but got {layers}")
            elif self.num_hidden_layers is not None and self.num_hidden_layers != len(layers):
                raise ValueError(
                    f"\`num_hidden_layers\` ({self.num_hidden_layers}) must be equal to the number of \`{layer_types}\` "
                    f"({len(layers)})"
                )`,
  codeNote: 'validate_layer_type 的两条硬约束：名字在白名单里、长度等于 num_hidden_layers。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '本文件里的 4 个 + 继承来的 1 个' }));
    const tb = W.table([
      ['validate_layer_type', '层类型名在白名单里；长度 == num_hidden_layers', '<span class="hlbad">raise</span>'],
      ['validate_architecture', 'head_dim × num_heads == embed_dim', '<span class="hlbad">raise</span>'],
      ['validate_output_attentions', '要 attentions 就必须是 eager 实现', '<span class="hlbad">raise</span>'],
      ['validate_token_ids', '*_token_id 必须落在 [0, vocab_size) 内', '<span class="hl">warning_once</span>'],
      ['validate_rope', '继承自 RotaryEmbeddingConfigMixin，不在本文件', '<span class="hlbad">raise</span>'],
    ], { head: ['validate_*', '查什么', '失败时'] });
    const tbCard = U.el('div', { class: 'card cc1', style: 'padding:10px 12px;width:100%' }, [tb]);
    viz.appendChild(tbCard);

    const row = U.el('div', { class: 'row gap12', style: 'width:100%;align-items:stretch' });
    row.appendChild(W.card({ cc: 3, tint: 3, title: '抛错的那次（实测）',
      body: U.el('div', { class: 'mono', style: 'font-size:11px;line-height:1.7;overflow-wrap:anywhere' }, [
        U.el('div', { class: 'dim', text: 'Glm5NextTextConfig(num_hidden_layers=45,' }),
        U.el('div', { class: 'dim', text: '                   layer_types=[...44 个...])' }),
        U.el('div', { html: '<span class="hlbad">→ ValueError: `num_hidden_layers` (45) must be' }),
        U.el('div', { html: '&nbsp;&nbsp;equal to the number of `layer_types` (44)</span>' }),
      ]), style: 'flex:1.2' }));
    row.appendChild(W.card({ cc: 4, tint: 4, title: '只警告的那次（实测）',
      body: U.el('div', { class: 'mono', style: 'font-size:11px;line-height:1.7;overflow-wrap:anywhere' }, [
        U.el('div', { class: 'dim', text: 'Glm5NextTextConfig(pad_token_id=-1)' }),
        U.el('div', { html: '<span class="hl4">→ 构造成功</span>，只留下一条 warning_once' }),
        U.el('div', { class: 'dim', text: '// 否则 Hub 上那批 config 再也加载不起来' }),
      ]), style: 'flex:1' }));
    viz.appendChild(row);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 两条硬约束：白名单 + 长度</span>';
    tl.at(2400, () => { msg.innerHTML = '名字必须在 <em>ALLOWED_ATTN_LAYER_TYPES</em> / <em>ALLOWED_MLP_LAYER_TYPES</em> 里'; });
    tl.at(5800, () => { msg.innerHTML = '长度必须等于 <em>num_hidden_layers</em> —— 否则 L0-01 的 <span class="fn">layer_types[i]</span> 会越界'; });
    tl.at(9400, () => { msg.innerHTML = '★ <span class="hlbad">形状错了抛错</span>，<span class="hl4">取值可疑只警告</span> —— 这是两条不同的红线'; });
    tl.at(13000, () => { msg.innerHTML = '<span class="cm">// for name in text_config 能跑，是因为 PreTrainedConfig.__iter__（第 1048 行）</span>'; });
    tl.at(16000, () => { msg.innerHTML = '<span class="cm">// 校验器由 @strict 统一收集，方法名就是接口</span>'; });
  },
},

/* ------------------------------------------------- 7 ★ 嵌套寻址 */
{
  kicker: '★ 机制 · 嵌套寻址',
  title: '两个 <span class="hl-a">ClassVar</span> 决定嵌套配置怎么找、怎么发',
  sub: 'base_config_key 回答「我在别人的字典里叫什么」，sub_configs 回答「我下面有谁」。',
  caption: '两个属性都用 ClassVar 标注，不进 __dict__、不会被序列化 —— 展开见 L1-03。',
  lang: 'python',
  codeStart: 471,
  code: `        for subconfig_key in self.sub_configs:
            subconfig = getattr(self, subconfig_key, None)
            if subconfig is not None:
                current_subconfig_attn = getattr(subconfig, "_attn_implementation", None)
                sub_implementation = (
                    value if not isinstance(value, dict) else value.get(subconfig_key, current_subconfig_attn)
                )
                subconfig._attn_implementation = sub_implementation`,
  codeNote: '这是 _attn_implementation 的 setter：按 sub_configs 的键把实现递归下发。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '实测：GLM-5 三个 config 类的两个类属性' }));
    const tb = W.table([
      ['Glm5NextConfig', "<span class=\"hl\">''</span>", 'vision_config, text_config'],
      ['Glm5NextTextConfig', "<span class=\"hl2\">'text_config'</span>", '<span class="dim">（空 —— 它没有子配置）</span>'],
      ['Glm5NextVisionConfig', "<span class=\"hl2\">'vision_config'</span>", '<span class="dim">（空 —— 它没有子配置）</span>'],
    ], { head: ['类', 'base_config_key', 'sub_configs 的键'] });
    viz.appendChild(U.el('div', { class: 'card cc0', style: 'padding:10px 12px;width:100%' }, [tb]));

    const drill = U.el('div', { class: 'card cc4', style: 'padding:10px 12px;width:100%' });
    drill.appendChild(U.el('div', { class: 'klabel', text: '往下找：base_config_key 的唯一消费者（第一节那两行）' }));
    const chain = U.el('div', { class: 'row gap8 center wrap', style: 'width:100%' });
    chain.innerHTML =
      '<span class="chip c1">Glm5NextTextConfig</span>' +
      '<span class="arrow" style="max-width:26px"></span>' +
      '<span class="chip c4">.base_config_key = "text_config"</span>' +
      '<span class="arrow" style="max-width:26px"></span>' +
      '<span class="chip c0">config_dict["text_config"]</span>';
    drill.appendChild(chain);
    viz.appendChild(drill);

    const bcast = U.el('div', { class: 'card cc2', style: 'padding:10px 12px;width:100%' });
    bcast.appendChild(U.el('div', { class: 'klabel', text: '往下发：同一个 sub_configs，被 _attn_implementation 当成广播名单（实测）' }));
    bcast.appendChild(U.el('div', { class: 'mono', style: 'font-size:11.5px;line-height:1.7;overflow-wrap:anywhere' }, [
      U.el('div', { html: '<span class="dim">attn_implementation =</span> {"": "sdpa", "text_config": "eager", "vision_config": "flash_attention_2"}' }),
      U.el('div', { html: '→ 顶层 <span class="hl">sdpa</span> &nbsp;/&nbsp; text <span class="hl2">eager</span> &nbsp;/&nbsp; vision <span class="hl3">flash_attention_2</span>' }),
      U.el('div', { html: '<span class="dim">传字符串时：三处都是同一个值</span>' }),
    ]));
    viz.appendChild(bcast);

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// sub_configs 既是寻址表，也是广播名单</span>';
    tl.at(2400, () => { msg.innerHTML = '<em>base_config_key</em>：我在别人的字典里叫什么 —— 空串表示"我是顶层"'; });
    tl.at(5800, () => { msg.innerHTML = '<em>sub_configs</em>：我下面有谁 —— 键名必须与子类的 base_config_key 一致'; });
    tl.at(9200, () => { msg.innerHTML = '★ 键名改一个字母，<span class="hlbad">下钻与广播会同时失效</span>'; });
    tl.at(12600, () => { msg.innerHTML = '运行时还有第三种寻址：<span class="fn">get_text_config()</span> 按名字表找，找到多个就抛错'; });
    tl.at(15600, () => { msg.innerHTML = '<span class="cm">// 回顾 L0-05：hc_mult 就住在 text_config 这一格里</span>'; });
  },
},

/* ------------------------------------------------- 8 出口：序列化 */
{
  kicker: '出口 · 序列化',
  title: '写出去的是<span class="hl-a">差异</span>，不是全量',
  sub: 'to_dict 全量 22 个顶层键，to_diff_dict 只剩 11 个；use_diff 默认为 True，save_pretrained 落盘的就是后者。',
  caption: '先 diff 再编码特殊浮点 —— 顺序反了，{"__float__": ...} 会被当成普通字典参与比较。',
  lang: 'python',
  codeStart: 1125,
  code: `        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type

        # Transformers version when serializing the model
        output["transformers_version"] = __version__

        # Pop "kwargs" since they are unpacked and set in the post init
        output.pop("kwargs", None)`,
  codeNote: '序列化的输入是 self.__dict__，不是 dataclass 字段表。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap14', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const viz = U.el('div', { class: 'col gap12 vizgrow hstart' });
    wrap.appendChild(viz);

    viz.appendChild(U.el('div', { class: 'klabel', text: '回程四步' }));
    const fl = W.flow([
      { t: 'self.__dict__', s: '实例字典', cc: 0 },
      { t: 'to_dict', s: '全量 + model_type', cc: 2 },
      { t: 'to_diff_dict', s: '只留差异', cc: 4 },
      { t: '_encode_special_floats', s: 'Inf / NaN', cc: 3 },
      { t: 'json.dumps', s: 'sort_keys=True', cc: 1 },
    ], { style: 'width:100%' });
    viz.appendChild(fl);

    viz.appendChild(U.el('div', { class: 'klabel', text: '实测：Glm5NextConfig() 的一次往返' }));
    const tb = W.table([
      ['to_dict() 顶层键', '<span class="hl">22</span>', '全量'],
      ['to_diff_dict() 顶层键', '<span class="hl">11</span>', '只留与默认不同的键'],
      ['嵌套 text_config 键', '<span class="hl">62 → 51</span>', '嵌套走 recursive_diff_dict'],
      ['to_json_string 字节', '<span class="hl">6024 → 4894</span>', '落盘的 config.json 就是 4894 字节 / 226 行'],
    ], { head: ['量', '实测', '说明'] });
    viz.appendChild(U.el('div', { class: 'card cc4', style: 'padding:10px 12px;width:100%' }, [tb]));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    msg.innerHTML = '<span class="cm">// 序列化的输入是实例字典</span>';
    tl.at(2400, () => { fl.focus(0); msg.innerHTML = '<em>copy.deepcopy(self.__dict__)</em> —— 所以内部键必须靠 _remove_keys_not_serialized 清掉'; });
    tl.at(5800, () => { fl.focus(1); msg.innerHTML = '<em>model_type</em> 来自类（不是实例）：它是 L1-05 里"字符串找类"要用的那个字符串'; });
    tl.at(9200, () => { fl.focus(2); msg.innerHTML = '★ <em>use_diff=True</em> 是默认值：22 键 → <span class="hl">11 键</span>，6024 字节 → <span class="hl">4894 字节</span>'; });
    tl.at(12800, () => { fl.focus(3); msg.innerHTML = '<em>_encode_special_floats</em> 必须在 diff 之后 —— 否则 {"__float__": ...} 会被当成普通字典递归比较'; });
    tl.at(15600, () => { fl.focus(4); msg.innerHTML = '<span class="cm">// sort_keys=True + 末尾补 "\\n"：diff 稳定、git 不报 No newline</span>'; });
  },
},

/* ------------------------------------------------- 9 收束 */
{
  kicker: '收束',
  title: '把一条路径压成一张表',
  sub: '两个校验时机、两个异常类、一条故意不抛错的检查 —— 这就是 PreTrainedConfig 的公共契约。',
  caption: '下一课 L1-02：Glm5NextTextConfig 逐字段精读。',
  lang: 'python',
  codeStart: 640,
  code: `        if hasattr(self, "validate"):
            self.validate()
        self.to_json_file(output_config_file, use_diff=True)`,
  codeNote: 'save_pretrained 里的第二次校验：不让非法配置流出这台机器。',
  duration: 18000,
  build(root, tl) {
    const wrap = U.el('div', { class: 'col gap12', style: 'width:100%;height:100%' });
    root.appendChild(wrap);

    const steps = [
      { n: '①', t: '读', c: 0, d: 'cached_file → json.loads → 解码特殊浮点' },
      { n: '②', t: '钻', c: 2, d: 'base_config_key → config_dict[key]' },
      { n: '③', t: '造', c: 4, d: 'cls(**config_dict) → @strict 校验' },
      { n: '④', t: '写', c: 1, d: 'to_diff_dict → json.dumps' },
    ];
    const row = U.el('div', { class: 'row gap8', style: 'width:100%;align-items:stretch' });
    const els = steps.map(s => {
      const e = W.card({ cc: s.c, tint: s.c, title: '<span class="mono" style="font-size:11px">' + s.n + ' ' + s.t + '</span>',
        sub: s.d, style: 'flex:1;text-align:center;overflow-wrap:anywhere' });
      row.appendChild(e); return e;
    });
    wrap.appendChild(row);

    const tb = W.table([
      ['构造完成', 'cls.validate(self)', '<span class="hlbad">StrictDataclassClassValidationError</span>', '抛错'],
      ['字段赋值', '__strict_setattr__', '<span class="hlbad">StrictDataclassFieldValidationError</span>', '抛错'],
      ['save_pretrained', 'self.validate()', '<span class="hlbad">同上</span>', '抛错'],
      ['特殊 token', 'validate_token_ids', '<span class="hl">logger.warning_once</span>', '只警告'],
    ], { head: ['时机', '谁在跑', '失败时', '结果'] });
    wrap.appendChild(U.el('div', { class: 'card cc1', style: 'padding:10px 12px' }, [
      U.el('div', { class: 'klabel', text: '校验的时机与后果（全部实测）' }), tb,
    ]));

    wrap.appendChild(W.exercise(
      '一个 config 类没有写 <code class="inl">@strict</code>，它继承的 <code class="inl">validate_layer_type</code> '
      + '还会在构造时被调用吗？<br>另外：<code class="inl">Glm5NextTextConfig.from_pretrained(dir)</code> '
      + '读的是哪一层字典？',
      '都不会"自动"发生。<b>①</b> <code class="inl">validate()</code> 是 <code class="inl">@strict</code> '
      + '在类上现装的；不写 <code class="inl">@strict</code> 的类没有这一步 —— 这也是为什么 GLM-5 的三个 config 类'
      + '各自都带一个 <code class="inl">@strict</code>。<b>②</b> 它先读顶层 <code class="inl">config.json</code>，'
      + '然后因为 <code class="inl">base_config_key == "text_config"</code>，'
      + '把 <code class="inl">config_dict["text_config"]</code> 那一格取出来当输入；实测结果与直接构造的'
      + ' <code class="inl">Glm5NextTextConfig()</code> 相等。'));

    const msg = U.el('div', { class: 'formula', style: 'width:100%' });
    wrap.appendChild(msg);

    els.forEach(e => e.style.opacity = '.28');
    msg.innerHTML = '<span class="cm">// 四个动作，逐个点亮</span>';
    steps.forEach((s, i) => {
      tl.at(1800 + i * 2000, () => {
        els.forEach((e, k) => { e.style.opacity = (k === i) ? '1' : '.28'; });
        msg.innerHTML = '<em>' + s.n + ' ' + s.t + '</em> &nbsp;<span class="cm">// ' + s.d + '</span>';
      });
    });
    tl.at(11600, () => {
      els.forEach(e => { e.style.opacity = '1'; });
      msg.innerHTML = '★ 校验有两个时机：<span class="hl">构造完成</span> 与 <span class="hl">写盘之前</span>';
    });
    tl.at(14600, () => {
      msg.innerHTML = '<span class="cm">// 下一课 L1-02：这些机制作用在哪些字段上</span>';
    });
  },
},

];
