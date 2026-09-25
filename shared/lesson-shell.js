/* ==========================================================================
   GLM-5 推理课件 — 课程外壳
   --------------------------------------------------------------------------
   负责：页面骨架、头部随幕更新、右侧代码栏、底部走带、键盘交互。
   课件只调用 SHELL.boot(SCENES, options)，不重复实现这些。
   ========================================================================== */
'use strict';

const SHELL = {
  opts: {}, codeLines: [], _lastIdx: -1, _revealTimers: [],

  boot(scenes, o = {}) {
    this.opts = o;
    W.skeleton({
      title: o.pageTitle || o.title || 'GLM-5 推理课件',
      css: o.css || null,          // null -> 沿用 index.html 里的相对路径
    });

    /* 校验必需容器：缺了就明确报错，而不是白屏 */
    const need = ['#stage', '#visual', '#side', '#sideBody', '#ftr', '#ticks', '#btns'];
    const miss = need.filter(s => !U.q(s));
    if (miss.length) {
      document.body.appendChild(W.errLayout('缺少容器: ' + miss.join(', ')));
      return;
    }

    U.q('#sideName').textContent = o.codeName || 'modeling_glm5_next.py';
    U.q('#hdrKicker').innerHTML = o.kicker || '';

    /* ---- 头部右上：返回门户 + 自定义链接 ---- */
    const hr = U.q('#hdrRight');
    (o.links || []).forEach(l => {
      hr.appendChild(U.el('a', { class: 'bt nav', href: l.href, html: l.label, title: l.title || '' }));
    });

    /* ---- 走带按钮 ---- */
    const btns = U.q('#btns');
    const mk = (cls, html, title, fn) => {
      const b = U.el('button', { class: 'bt ' + cls, title: title, html: html });
      b.addEventListener('click', fn);
      btns.appendChild(b); return b;
    };
    this.bFirst = mk('', W.ICON.first, '第一幕 (Home)', () => { RENDER.go(0, -1); });
    this.bPrev = mk('', W.ICON.prev, '上一幕 (←)', () => RENDER.prev());
    this.bPlay = mk('on', W.ICON.pause, '播放 / 暂停 (空格)', () => RENDER.toggle());
    this.bNext = mk('', W.ICON.next, '下一幕 (→)', () => RENDER.next());
    this.bLast = mk('', W.ICON.last, '最后一幕 (End)', () => { RENDER.go(RENDER.scenes.length - 1, 1); });

    /* ---- 进度刻度 ---- */
    const ticks = U.q('#ticks');
    scenes.forEach((s, i) => {
      const t = U.el('div', { class: 'tk', title: (i + 1) + '. ' + (s.title || '').replace(/<[^>]+>/g, '') });
      t.appendChild(U.el('div', { class: 'tkfill' }));
      t.addEventListener('click', () => RENDER.go(i, i > RENDER.idx ? 1 : -1));
      ticks.appendChild(t);
    });
    this.ticks = U.qa('.tk', ticks);

    /* ---- 键盘 ---- */
    document.addEventListener('keydown', e => {
      if (e.target && /INPUT|TEXTAREA/.test(e.target.tagName)) return;
      switch (e.key) {
        case 'ArrowRight': case 'l': RENDER.next(); e.preventDefault(); break;
        case 'ArrowLeft': case 'h': RENDER.prev(); e.preventDefault(); break;
        case ' ': RENDER.toggle(); e.preventDefault(); break;
        case 'Home': RENDER.go(0, -1); e.preventDefault(); break;
        case 'End': RENDER.go(scenes.length - 1, 1); e.preventDefault(); break;
        case 'r': RENDER.seek(0); e.preventDefault(); break;
      }
    });

    bootEngine(scenes);
  },

  /* ------------------------------------------------------------------
     每幕挂载后调用：更新头部、代码栏、说明、刻度
     ------------------------------------------------------------------ */
  sync() {
    const i = RENDER.idx, sc = RENDER.scenes[i]; if (!sc) return;
    const changed = i !== this._lastIdx;

    U.q('#hdrTitle').innerHTML = sc.title || '';
    U.q('#hdrSub').innerHTML = sc.sub || '';
    U.q('#cap').innerHTML = (sc.caption
      ? '<span class="k">' + (sc.capKey || '要点') + '</span><span>' + sc.caption + '</span>'
      : '');

    if (changed) {
      this._lastIdx = i;
      this.renderCode(sc);
      this.ticks.forEach((t, k) => {
        t.classList.toggle('cur', k === i);
        t.classList.toggle('done', k < i);
      });
      U.q('#counter').textContent = (i + 1) + ' / ' + RENDER.scenes.length;
      this.bPrev.classList.toggle('off', i === 0);
      this.bNext.classList.toggle('off', i === RENDER.scenes.length - 1);
      this.bFirst.classList.toggle('off', i === 0);
      this.bLast.classList.toggle('off', i === RENDER.scenes.length - 1);
      /* 幕内滚动位置复位 */
      const body = U.q('#sideBody'); if (body) body.scrollTop = 0;
    }
    this.syncPlay();
  },

  /* 代码栏渲染：逐行淡入 + 可热行 */
  renderCode(sc) {
    const body = U.q('#sideBody'); if (!body) return;
    this._revealTimers.forEach(t => clearTimeout(t)); this._revealTimers = [];
    body.innerHTML = '';
    const note = U.q('#sideNote');
    if (note) note.innerHTML = sc.codeNote || '';

    if (!sc.code) { this.codeLines = []; return; }
    const built = U.codeEl(sc.code, { lang: sc.lang || 'python', startNo: sc.codeStart || 1 });
    body.appendChild(built.el);
    this.codeLines = built.lines;
    built.lines.forEach((ln, k) => {
      ln.style.opacity = '0';
      ln.style.transform = 'translateX(9px)';
      const id = setTimeout(() => {
        ln.style.transition = 'opacity .3s ease, transform .3s cubic-bezier(.16,1,.3,1), background .3s';
        ln.style.opacity = '1'; ln.style.transform = 'none';
      }, 55 * k);
      this._revealTimers.push(id);
    });
  },

  /* 幕内可调用：点亮指定行（1 基），[] 取消全部 */
  hotLines(idxs) { U.hot(this.codeLines, idxs); },

  syncPlay() {
    if (!this.bPlay) return;
    const on = RENDER.playing;
    this.bPlay.innerHTML = on ? W.ICON.pause : W.ICON.play;
    this.bPlay.classList.toggle('on', on);
    this.bPlay.title = on ? '暂停 (空格)' : '播放 (空格)';
  },

  /* 每帧：进度条 + 当前刻度填充 */
  tick() {
    const d = RENDER.total() || 1;
    const p = U.clamp(RENDER.clock / d, 0, 1);
    const bf = U.q('#barFill'); if (bf) bf.style.width = (p * 100).toFixed(2) + '%';
    const cur = this.ticks && this.ticks[RENDER.idx];
    if (cur) { const f = cur.firstElementChild; if (f) f.style.width = (p * 100).toFixed(1) + '%'; }
  },
};
