/* ==========================================================================
   GLM-5 推理课件 — 核心引擎
   --------------------------------------------------------------------------
   单一来源：全部课件共用本文件，禁止内联复制。
   设计要点：
   - 舞台固定 1600x900，按窗口等比缩放 —— 任何屏幕排版一致，便于像素级验收。
   - 时间轴只在"播放中"前进，因此暂停是真正的时间冻结（含幕内分步动画）。
   - 跳转 = 重建当前幕 + 快进重放事件（确定性），所以任意时刻画面可复现。
   - 经典脚本（非 ES module）：file:// 双击直接可用，不被 CORS 拦。
   ========================================================================== */
'use strict';

/* ==========================================================================
   1. 工具集 U
   ========================================================================== */
const U = {
  /* ---- DOM ---- */
  el(tag, attrs = {}, kids = []) {
    const e = document.createElement(tag);
    for (const k in attrs) {
      const v = attrs[k];
      if (v == null) continue;
      if (k === 'class') e.className = v;
      else if (k === 'html') e.innerHTML = v;
      else if (k === 'text') e.textContent = v;
      else if (k === 'style') e.setAttribute('style', v);
      else if (k === 'data') { for (const d in v) e.dataset[d] = v[d]; }
      else if (k.startsWith('on') && typeof v === 'function') e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v);
    }
    (Array.isArray(kids) ? kids : [kids]).forEach(c => {
      if (c == null || c === false) return;
      e.appendChild(typeof c === 'string' || typeof c === 'number'
        ? document.createTextNode(String(c)) : c);
    });
    return e;
  },
  q(sel, root = document) { return root.querySelector(sel); },
  qa(sel, root = document) { return Array.from(root.querySelectorAll(sel)); },

  /* 舞台缩放比：屏幕像素 / 舞台像素 */
  scale: 1,

  /* ---- 数值 ---- */
  clamp: (v, a, b) => Math.max(a, Math.min(b, v)),
  lerp: (a, b, t) => a + (b - a) * t,

  /* 千分位；opts.dp 小数位；opts.si 用 K/M/B 后缀 */
  fmt(n, opts = {}) {
    if (n == null || !isFinite(n)) return String(n);
    const dp = opts.dp ?? 0;
    if (opts.si) {
      const u = [[1e12, 'T'], [1e9, 'B'], [1e6, 'M'], [1e3, 'K']];
      for (const [m, s] of u) if (Math.abs(n) >= m) return (n / m).toFixed(dp || 1) + s;
    }
    return n.toLocaleString('en-US', { minimumFractionDigits: dp, maximumFractionDigits: dp });
  },
  /* 字节数 -> 人类可读 */
  bytes(n, dp = 2) {
    const u = ['B', 'KB', 'MB', 'GB', 'TB'];
    let i = 0;
    while (Math.abs(n) >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return n.toFixed(i === 0 ? 0 : dp) + ' ' + u[i];
  },
  pct: (a, b, dp = 1) => (b ? (a / b * 100).toFixed(dp) : '0') + '%',

  /* ---- 缓动（JS 侧补间用；与 CSS 曲线保持一致的观感） ---- */
  ease: {
    out: t => 1 - Math.pow(1 - t, 3),
    inOut: t => t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2,
    pop: t => { const c = 1.70158 + 1; return 1 + (c + 1) * Math.pow(t - 1, 3) + c * Math.pow(t - 1, 2); },
  },

  /* 数字滚动：把 el 的文本从 a 补间到 b（跟随 timeline，不自行计时） */
  countTo(el, a, b, fmtFn, durMs = 900) {
    return (now, t0) => {
      const t = U.clamp((now - t0) / durMs, 0, 1);
      el.textContent = (fmtFn || (x => U.fmt(Math.round(x))))(U.lerp(a, b, U.ease.out(t)));
    };
  },

  /* ---- 轴 / 色 ---- */
  axc: i => 'axc' + (((i % 6) + 6) % 6),
  ci: i => (((i % 6) + 6) % 6),
  chip(name, ci, extra = '') {
    return `<span class="chip c${U.ci(ci)} ${extra}"><i class="sw sw-c${U.ci(ci)}"></i>${name}</span>`;
  },

  /* ---- 元素几何：返回相对于 #visual 的舞台像素矩形 ---- */
  rectIn(el, host) {
    const h = host || U.q('#visual');
    const a = el.getBoundingClientRect(), b = h.getBoundingClientRect();
    const s = U.scale || 1;
    return { x: (a.left - b.left) / s, y: (a.top - b.top) / s, w: a.width / s, h: a.height / s,
             cx: (a.left - b.left + a.width / 2) / s, cy: (a.top - b.top + a.height / 2) / s };
  },

  /* ======================================================================
     2. 语法高亮 —— 单遍 tokenizer
     ----------------------------------------------------------------------
     ★ 必须单遍扫描。用多次 replace 依次染色，后一次替换会把前一次插入的
       标签再匹配一次，产出 "nm">@mesh 这类破碎 HTML。
     语言：python（默认） / text / bash / json
     ====================================================================== */
  _KW: new Set(('False None True and as assert async await break class continue def del elif else except ' +
    'finally for from global if import in is lambda nonlocal not or pass raise return try while with yield ' +
    'match case self cls').split(' ')),
  _BI: new Set(('len range print int float str bool list dict tuple set min max sum abs any all zip enumerate ' +
    'isinstance getattr setattr super type open repr sorted round next iter hasattr').split(' ')),

  hl(code, lang) {
    if (lang === 'text') return U.esc(code);
    const out = [];
    let i = 0;
    const n = code.length;
    const isId = c => /[A-Za-z0-9_]/.test(c);
    const isIdS = c => /[A-Za-z_]/.test(c);
    const isDig = c => /[0-9]/.test(c);

    while (i < n) {
      const c = code[i], c2 = code.substr(i, 2), c3 = code.substr(i, 3);

      /* 三引号字符串（含 docstring） */
      if (c3 === '"""' || c3 === "'''") {
        const q = c3; let j = i + 3;
        while (j < n && code.substr(j, 3) !== q) j++;
        j = Math.min(n, j + 3);
        out.push('<span class="st">' + U.esc(code.slice(i, j)) + '</span>');
        i = j; continue;
      }
      /* 行注释 */
      if (c === '#') {
        let j = code.indexOf('\n', i); if (j < 0) j = n;
        out.push('<span class="cm">' + U.esc(code.slice(i, j)) + '</span>');
        i = j; continue;
      }
      /* 单/双引号字符串（含 f 前缀已由标识符分支处理，这里处理裸串） */
      if (c === '"' || c === "'") {
        const q = c; let j = i + 1;
        while (j < n) { if (code[j] === '\\') { j += 2; continue; } if (code[j] === q) { j++; break; } j++; }
        out.push('<span class="st">' + U.esc(code.slice(i, j)) + '</span>');
        i = j; continue;
      }
      /* 装饰器 */
      if (c === '@' && isIdS(code[i + 1] || '')) {
        let j = i + 1; while (j < n && (isId(code[j]) || code[j] === '.')) j++;
        out.push('<span class="at">' + U.esc(code.slice(i, j)) + '</span>');
        i = j; continue;
      }
      /* 数字 */
      if (isDig(c) && !isIdS(code[i - 1] || '')) {
        let j = i;
        while (j < n && /[0-9a-fA-FxXoObB_.]/.test(code[j])) j++;
        out.push('<span class="nu">' + U.esc(code.slice(i, j)) + '</span>');
        i = j; continue;
      }
      /* 标识符 / 关键字 / 函数名 / 类名 */
      if (isIdS(c)) {
        let j = i; while (j < n && isId(code[j])) j++;
        const w = code.slice(i, j);
        /* f/r/b 前缀字符串 */
        if ((w === 'f' || w === 'r' || w === 'b' || w === 'fr' || w === 'rf') &&
            (code[j] === '"' || code[j] === "'")) {
          const q = code[j]; let k = j + 1; let depth = 0;
          while (k < n) {
            if (code[k] === '\\') { k += 2; continue; }
            if (code[k] === '{') depth++;
            else if (code[k] === '}') depth--;
            else if (code[k] === q && depth <= 0) { k++; break; }
            k++;
          }
          out.push('<span class="st">' + U.esc(code.slice(i, k)) + '</span>');
          i = k; continue;
        }
        let k = j; while (k < n && code[k] === ' ') k++;
        if (U._KW.has(w)) out.push('<span class="kw">' + U.esc(w) + '</span>');
        else if (code[k] === '(') out.push('<span class="fn2">' + U.esc(w) + '</span>');
        else if (U._BI.has(w)) out.push('<span class="fn2">' + U.esc(w) + '</span>');
        else if (/^[A-Z]/.test(w)) out.push('<span class="dc">' + U.esc(w) + '</span>');
        else out.push(U.esc(w));
        i = j; continue;
      }
      /* 运算符 */
      if ('+-*/%=<>!&|^~'.includes(c)) {
        let j = i; while (j < n && '+-*/%=<>!&|^~'.includes(code[j])) j++;
        out.push('<span class="op">' + U.esc(code.slice(i, j)) + '</span>');
        i = j; continue;
      }
      out.push(U.esc(c)); i++;
    }
    return out.join('');
  },

  esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  },

  /* 逐行代码元素：返回 {el, lines:HTMLElement[]}，每行带 data-no，可 .hot 高亮 */
  codeEl(code, opts = {}) {
    const pre = U.el('pre', { class: 'code' });
    const ls = String(code).replace(/\n+$/, '').split('\n');
    const els = ls.map((ln, i) => {
      const e = U.el('span', { class: 'ln', 'data-no': opts.startNo ? opts.startNo + i : i + 1 });
      e.innerHTML = ln === '' ? '&nbsp;' : U.hl(ln, opts.lang);
      pre.appendChild(e);
      return e;
    });
    return { el: pre, lines: els };
  },

  /* 热行切换：只点亮给定行号（1 基），其余熄灭 */
  hot(lines, idxs) {
    const set = new Set((Array.isArray(idxs) ? idxs : [idxs]).map(x => x - 1));
    lines.forEach((e, i) => e.classList.toggle('hot', set.has(i)));
  },
};

/* ==========================================================================
   3. 时间轴 TL
   --------------------------------------------------------------------------
   tl.at(ms, fn) 注册事件。播放时按虚拟时钟触发；seek 时重建场景后快进重放。
   ========================================================================== */
class Timeline {
  constructor() {
    this.events = [];
    this.t = 0;
    this.playing = false;
  }
  at(ms, fn) { this.events.push({ t: Math.max(0, ms | 0), fn }); return this; }
  /* 顺序批量：tl.seq(起始, 间隔, [fn...]) */
  seq(t0, step, fns) { fns.forEach((f, i) => this.at(t0 + i * step, f)); return this; }
  reset() { this.events = []; this.t = 0; }
  sort() { this.events.sort((a, b) => a.t - b.t); }
  /* 触发所有 <= t 且尚未触发的事件 */
  advance(t, fired) {
    this.t = t;
    while (fired.i < this.events.length && this.events[fired.i].t <= t) {
      const e = this.events[fired.i++];
      try { e.fn(); } catch (err) { console.error('[timeline] event @' + e.t + 'ms failed:', err); }
    }
  }
  replayTo(t) {
    const fired = { i: 0 };
    this.advance(t, fired);
    return fired;
  }
  /* 逐帧回调（补间用）：fns 收到 (nowMs, t0, totalMs) */
  frame(fn) { this._frames = this._frames || []; this._frames.push(fn); return this; }
  runFrames(now) { (this._frames || []).forEach(f => { try { f(now); } catch (e) { console.error(e); } }); }
}

/* ==========================================================================
   4. 数据流层（canvas）
   --------------------------------------------------------------------------
   在 #visual 之上画"数据包沿路径移动"的轨迹。由引擎统一驱动，因此暂停/跳转
   同样有效：seek 会 clear 后由重放的事件重新登记。
   ========================================================================== */
const FLOW = {
  cv: null, ctx: null, host: null,
  paths: {}, travels: [], _t0: 0,

  attach(host) {
    this.detach();
    this.host = host;
    const cv = U.el('canvas', { id: 'flowCanvas' });
    host.appendChild(cv);
    this.cv = cv; this.ctx = cv.getContext('2d');
    this.resize();
    this.paths = {}; this.travels = [];
    return this;
  },
  detach() { if (this.cv && this.cv.parentNode) this.cv.parentNode.removeChild(this.cv); this.cv = null; this.ctx = null; },
  resize() {
    if (!this.cv || !this.host) return;
    const r = this.host.getBoundingClientRect();
    const s = U.scale || 1;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    /* ★ 单位陷阱：getBoundingClientRect() 给的是**屏幕像素**，而 #stage 带着
       transform:scale(s)。canvas 的 CSS 盒子写在舞台内部，所以要除以 s 才是
       它在舞台坐标里的尺寸；位图则要按 dpr*s 放大才够清晰。
       早期版本把屏幕像素直接当舞台像素用，缩放比不为 1 时 canvas 会比
       #visual 宽出 s 倍，数据包画到可视区外面去。 */
    const W = Math.max(1, r.width / s);
    const H = Math.max(1, r.height / s);
    const k = dpr * s;
    this.cv.width = Math.round(W * k);
    this.cv.height = Math.round(H * k);
    this.cv.style.width = W + 'px';
    this.cv.style.height = H + 'px';
    this.ctx.setTransform(k, 0, 0, k, 0, 0);
    this.W = W; this.H = H;           /* 舞台坐标，与 FLOW.path 的坐标系一致 */
  },
  clear() { this.paths = {}; this.travels = []; if (this.ctx) this.ctx.clearRect(0, 0, this.W, this.H); },
  /* 登记一条路径：pts = 舞台坐标 [[x,y],...] */
  path(id, pts, opts = {}) { this.paths[id] = { pts, opts }; return id; },
  /* 从元素几何自动取路径 */
  pathBetween(id, a, b, opts = {}) {
    const A = (a.cx != null) ? a : U.rectIn(a, this.host);
    const B = (b.cx != null) ? b : U.rectIn(b, this.host);
    const p = opts.via === 'v'
      ? [[A.cx, A.y], [A.cx, (A.y + B.y + B.h) / 2], [B.cx, (A.y + B.y + B.h) / 2], [B.cx, B.y + B.h]]
      : [[A.x + A.w, A.cy], [B.x, B.cy]];
    return this.path(id, p, opts);
  },
  /* 沿路径发一个包；startMs/durMs 用时间轴毫秒 */
  send(pathId, o = {}) {
    this.travels.push({
      p: pathId, t0: o.at ?? 0, dur: o.dur ?? 1100,
      color: o.color ?? '34,211,238', r: o.r ?? 4.5,
      label: o.label || '', trail: o.trail !== false, oneway: o.oneway !== false,
      keep: !!o.keep, shape: o.shape || 'dot',
    });
  },
  /* 沿路径画静态发光参考线 */
  rail(pathId, t0, color) {
    this.travels.push({ p: pathId, t0, dur: 0, rail: true, color: color ?? '34,211,238' });
  },
  _pt(pts, t) {
    /* 等分弧长参数化 */
    let total = 0; const segs = [];
    for (let i = 1; i < pts.length; i++) {
      const d = Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
      segs.push(d); total += d;
    }
    if (total === 0) return pts[0];
    let want = t * total;
    for (let i = 0; i < segs.length; i++) {
      if (want <= segs[i] || i === segs.length - 1) {
        const u = segs[i] ? want / segs[i] : 0;
        return [U.lerp(pts[i][0], pts[i + 1][0], u), U.lerp(pts[i][1], pts[i + 1][1], u)];
      }
      want -= segs[i];
    }
    return pts[pts.length - 1];
  },
  render(now) {
    if (!this.ctx) return;
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.W, this.H);
    /* 先画参考线，再画包，保证包在上层 */
    for (const tr of this.travels) {
      const P = this.paths[tr.p]; if (!P) continue;
      const pts = P.pts;
      if (tr.rail) {
        if (now < tr.t0) continue;
        const a = U.clamp((now - tr.t0) / 500, 0, 1);
        ctx.save();
        ctx.globalAlpha = .3 * a;
        ctx.strokeStyle = 'rgba(' + tr.color + ',.9)';
        ctx.lineWidth = 1.5; ctx.setLineDash([5, 5]);
        ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
        for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
        ctx.stroke(); ctx.restore();
        continue;
      }
      const t = (now - tr.t0) / tr.dur;
      if (t < 0 || t > 1) continue;
      const e = U.ease.inOut(U.clamp(t, 0, 1));
      const [x, y] = this._pt(pts, e);
      /* 拖尾 */
      if (tr.trail) {
        for (let k = 1; k <= 7; k++) {
          const tt = U.clamp(e - k * 0.035, 0, 1);
          const [tx, ty] = this._pt(pts, tt);
          ctx.beginPath();
          ctx.fillStyle = 'rgba(' + tr.color + ',' + (0.20 * (1 - k / 8)).toFixed(3) + ')';
          ctx.arc(tx, ty, tr.r * (1 - k / 10), 0, 6.2832); ctx.fill();
        }
      }
      const g = ctx.createRadialGradient(x, y, 0, x, y, tr.r * 5);
      g.addColorStop(0, 'rgba(' + tr.color + ',.95)');
      g.addColorStop(.35, 'rgba(' + tr.color + ',.42)');
      g.addColorStop(1, 'rgba(' + tr.color + ',0)');
      ctx.beginPath(); ctx.fillStyle = g; ctx.arc(x, y, tr.r * 5, 0, 6.2832); ctx.fill();
      ctx.beginPath(); ctx.fillStyle = 'rgba(255,255,255,.96)';
      ctx.arc(x, y, tr.r * .62, 0, 6.2832); ctx.fill();
      if (tr.label) {
        ctx.font = '600 10px ui-monospace,monospace';
        ctx.fillStyle = 'rgba(233,240,255,.92)';
        ctx.textAlign = 'center';
        ctx.fillText(tr.label, x, y - tr.r - 6);
      }
    }
  },
};

/* ==========================================================================
   5. 舞台缩放
   ========================================================================== */
function fitStage() {
  const st = U.q('#stage'); if (!st) return;
  const W = 1600, H = 900;
  const s = Math.min(window.innerWidth / W, window.innerHeight / H);
  U.scale = s;
  st.style.transform = 'scale(' + s + ')';
  if (FLOW.cv) FLOW.resize();
}

/* ==========================================================================
   6. 分幕渲染器
   ========================================================================== */
const RENDER = {
  scenes: [], idx: 0, tl: new Timeline(), fired: { i: 0 },
  clock: 0, playing: true, raf: 0, last: 0, sceneT0: 0,
  total() { return this.scenes[this.idx] ? this.scenes[this.idx].duration : 0; },

  mount(i, seekMs) {
    const host = U.q('#visual'); if (!host) return;
    FLOW.clear();
    host.innerHTML = '';
    this.idx = i; this.clock = seekMs || 0;
    const sc = this.scenes[i];
    const root = U.el('div', { class: 'scene' });
    host.appendChild(root);
    this.tl = new Timeline();
    try {
      sc.build(root, this.tl);
    } catch (err) {
      console.error('[scene ' + (i + 1) + '] build failed:', err);
      root.appendChild(W.errScenes(String(err && err.stack || err)));
      return;
    }
    this.tl.sort();
    FLOW.render(0);
    this.fired = this.tl.replayTo(this.clock);
    this.tl.runFrames(this.clock);
    FLOW.render(this.clock);
    if (FLOW.cv) FLOW.cv.style.opacity = '1';
    this.paint();
  },

  /* 跳转到指定毫秒 */
  seek(ms) {
    const d = this.total();
    const t = U.clamp(ms, 0, Math.max(0, d - 1));
    this.mount(this.idx, t);
    if (SHELL.sync) SHELL.sync();
  },
  go(i, dir) {
    const n = this.scenes.length;
    const j = U.clamp(i, 0, n - 1);
    if (j === this.idx) return;
    /* 幕间过渡：整块淡出再换内容 */
    const host = U.q('#visual');
    const cur = host.firstElementChild;
    const swap = () => { this.mount(j, 0); if (SHELL.sync) SHELL.sync(); };
    if (cur && dir !== 0) {
      cur.style.transition = 'opacity .22s ease, transform .22s ease';
      cur.style.opacity = '0';
      cur.style.transform = dir > 0 ? 'translateX(-22px)' : 'translateX(22px)';
      setTimeout(swap, 210);
    } else swap();
  },
  next() { this.go(this.idx + 1, 1); },
  prev() { this.go(this.idx - 1, -1); },

  play() { if (this.playing) return; this.playing = true; this.last = performance.now(); SHELL.sync && SHELL.sync(); SHELL.syncPlay && SHELL.syncPlay(); },
  pause() { if (!this.playing) return; this.playing = false; SHELL.sync && SHELL.sync(); SHELL.syncPlay && SHELL.syncPlay(); },
  toggle() { this.playing ? this.pause() : this.play(); },

  tick(now) {
    this.raf = requestAnimationFrame(t => this.tick(t));
    /* 自动推进：到幕末停一拍后进下一幕 */
    if (this.playing) {
      const dt = Math.min(64, now - (this.last || now));
      this.last = now;
      this.clock += dt;
      const d = this.total();
      if (this.clock >= d + 900) {
        if (this.idx < this.scenes.length - 1) { this.last = now; this.next(); }
        else { this.playing = false; this.clock = d; SHELL.sync && SHELL.sync(); SHELL.syncPlay && SHELL.syncPlay(); }
      } else if (this.clock > d) {
        this.clock = d;
      }
    } else { this.last = now; }
    this.tl.advance(this.clock, this.fired);
    this.tl.runFrames(this.clock);
    FLOW.render(this.clock);
    if (SHELL.tick) SHELL.tick();
  },

  paint() { if (SHELL.sync) SHELL.sync(); },
};

/* ==========================================================================
   7. 启动
   ========================================================================== */
function bootEngine(scenes) {
  if (!scenes || !scenes.length) {
    const host = U.q('#visual');
    if (host) host.appendChild(W.errScenes('SCENES 为空或未定义'));
    return;
  }
  RENDER.scenes = scenes;
  fitStage();
  window.addEventListener('resize', fitStage);
  RENDER.mount(0, 0);
  RENDER.last = performance.now();
  RENDER.raf = requestAnimationFrame(t => RENDER.tick(t));
}
