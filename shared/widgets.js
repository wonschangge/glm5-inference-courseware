/* ==========================================================================
   GLM-5 推理课件 — 可视化与交互组件
   --------------------------------------------------------------------------
   单一来源：全部课件共用本文件。
   约定：所有组件返回**元素**（不是 HTML 字符串），除非名字以 S 结尾。
   ========================================================================== */
'use strict';

const W = {

  /* ======================================================================
     错误页：课件自身出问题时不要白屏
     ====================================================================== */
  errScenes(msg) {
    return U.el('div', { class: 'errbox' }, [
      U.el('div', { class: 'et', text: '⚠ 没有加载到任何场景' }),
      U.el('div', { class: 'ed', text: msg || '请检查 lesson.js 是否定义了非空的 SCENES 数组。' }),
    ]);
  },
  errLayout(msg) {
    return U.el('div', { class: 'errbox' }, [
      U.el('div', { class: 'et', text: '⚠ 缺少必需容器' }),
      U.el('div', { class: 'ed', text: msg || 'index.html 必须包含 #stage / #visual / #side / #ftr / #hdr。' }),
    ]);
  },

  /* ======================================================================
     卡片
     ====================================================================== */
  card(o = {}) {
    const cls = ['card'];
    if (o.cc != null) cls.push('cc' + U.ci(o.cc));
    if (o.tint != null) cls.push('tint' + U.ci(o.tint));
    if (o.cls) cls.push(o.cls);
    const e = U.el('div', { class: cls.join(' '), style: o.style || '' });
    if (o.num != null) {
      e.appendChild(U.el('div', { class: 'ct' }, [
        U.el('span', { class: 'num', text: String(o.num) }),
        U.el('span', { html: o.title || '' }),
      ]));
    } else if (o.title != null) {
      e.appendChild(U.el('div', { class: 'ct', html: o.title }));
    }
    if (o.sub != null) e.appendChild(U.el('div', { class: 'cs', html: o.sub }));
    if (o.body) e.appendChild(o.body);
    if (o.kids) (Array.isArray(o.kids) ? o.kids : [o.kids]).forEach(k => k && e.appendChild(k));
    return e;
  },

  /* 卡片行：等宽排布，自动处理 wrap */
  cardRow(defs, o = {}) {
    const r = U.el('div', { class: 'row wrap gap12', style: 'width:100%;align-items:stretch' });
    defs.forEach((d, i) => {
      const c = W.card(Object.assign({}, d, { num: d.num != null ? d.num : (o.numbered ? i + 1 : null) }));
      if (o.flex) c.style.flex = '1 1 ' + (o.basis || '0') + 'px';
      r.appendChild(c);
    });
    return r;
  },

  /* ======================================================================
     键值表
     ====================================================================== */
  table(rows, o = {}) {
    const t = U.el('table', { class: 'tbl' });
    if (o.head) {
      const tr = U.el('tr');
      o.head.forEach(h => tr.appendChild(U.el('th', { html: h })));
      t.appendChild(U.el('thead', {}, [tr]));
    }
    const tb = U.el('tbody');
    rows.forEach(r => {
      const tr = U.el('tr', { class: r.cls || '' });
      const cells = Array.isArray(r) ? r : r.cells;
      cells.forEach((c, i) => {
        const td = U.el('td', { class: (Array.isArray(r) ? (i === 0 ? 'k' : '') : (r.cls_k && i === 0 ? 'k' : '')) });
        if (c && c.nodeType) td.appendChild(c);
        else td.innerHTML = c == null ? '' : String(c);
        tr.appendChild(td);
      });
      tb.appendChild(tr);
    });
    t.appendChild(tb);
    return t;
  },

  /* ======================================================================
     条形图（宽度补间由 tl 驱动，或直接设置最终值）
     ====================================================================== */
  bars(items, o = {}) {
    const wrap = U.el('div', { class: 'barwrap', style: o.style || '' });
    const fills = [];
    items.forEach((it, i) => {
      const row = U.el('div', { class: 'brow' + (o.cc != null ? ' cc' + U.ci(o.cc) : '') });
      if (it.cc != null) row.className += ' cc' + U.ci(it.cc);
      row.appendChild(U.el('div', { class: 'bl', text: it.label, title: it.label }));
      const bt = U.el('div', { class: 'bt2' });
      const bf = U.el('div', { class: 'bf' });
      if (it.cc != null) bf.style.setProperty('--cc', 'var(--c' + U.ci(it.cc) + ')');
      bt.appendChild(bf); row.appendChild(bt);
      const bv = U.el('div', { class: 'bv', text: it.valueText != null ? it.valueText : '' });
      row.appendChild(bv);
      wrap.appendChild(row);
      fills.push({ bf, bv, it, max: o.max || Math.max(...items.map(x => x.value || 0)) });
    });
    /* setTo(0..1) 或 setTo(i, v) */
    wrap.setAll = (t) => fills.forEach(f => {
      const p = f.max ? (f.it.value / f.max) : 0;
      f.bf.style.width = (p * 100 * t).toFixed(1) + '%';
    });
    wrap.setTo = (i, v) => {
      const f = fills[i]; if (!f) return;
      const p = f.max ? (f.it.value / f.max) : 0;
      f.bf.style.width = (p * 100 * v).toFixed(1) + '%';
      if (f.it.valueText != null) f.bv.textContent = f.it.valueText;
    };
    wrap.fills = fills;
    return wrap;
  },

  /* 堆叠条：items = [{label, value, cc}] */
  stack(items, o = {}) {
    const s = U.el('div', { class: 'stack', style: o.style || '' });
    const total = items.reduce((a, b) => a + b.value, 0) || 1;
    items.forEach((it, i) => {
      const e = U.el('i', {
        style: 'flex:0 0 0;background:var(--c' + U.ci(it.cc != null ? it.cc : i) + ');'
             + 'color:#04121c;flex-grow:0',
        text: it.label || '',
        title: (it.label || '') + ' ' + U.fmt(it.value),
      });
      e.dataset.w = (it.value / total);
      s.appendChild(e);
    });
    s.reveal = (t) => Array.from(s.children).forEach(c => {
      c.style.flexGrow = (parseFloat(c.dataset.w) * t).toFixed(4);
    });
    return s;
  },

  /* ======================================================================
     流程线：nodes = [{t, s, cc}]
     ====================================================================== */
  flow(nodes, o = {}) {
    const r = U.el('div', { class: 'flowline', style: o.style || '' });
    const els = [];
    nodes.forEach((n, i) => {
      if (i > 0) {
        const a = U.el('div', { class: 'arrow' });
        if (n.label) a.title = n.label;
        r.appendChild(a); els.push({ arrow: a });
      }
      const e = U.el('div', { class: 'node' + (n.cc != null ? ' cc' + U.ci(n.cc) : '') });
      e.innerHTML = '<div class="nt">' + (n.t || '') + '</div>' + (n.s ? '<div class="ns">' + n.s + '</div>' : '');
      r.appendChild(e); els.push(e);
    });
    r.nodes = els.filter(e => !e.arrow);
    r.arrows = els.filter(e => e.arrow);
    r.focus = (k) => {
      r.nodes.forEach((e, i) => { e.classList.toggle('ac', i === k); e.classList.toggle('dimmed', k >= 0 && i !== k); });
    };
    return r;
  },

  /* ======================================================================
     张量网格：owner(r,c) -> 颜色索引 / -1；opts.text(r,c)
     ====================================================================== */
  tensor(rows, cols, owner, o = {}) {
    const cs = o.cell || 15, gap = o.gap == null ? 2 : o.gap;
    const g = U.el('div', { class: 'tgrid' });
    g.style.gridTemplateColumns = 'repeat(' + cols + ',' + cs + 'px)';
    g.style.gridTemplateRows = 'repeat(' + rows + ',' + cs + 'px)';
    g.style.gap = gap + 'px';
    const cells = [];
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const cell = U.el('div', { class: 'tcell' });
        const ow = owner ? owner(r, c) : -1;
        if (ow >= 0) cell.classList.add('ax' + U.ci(ow));
        if (o.text) { const t = o.text(r, c); if (t != null) cell.textContent = t; }
        g.appendChild(cell); cells.push(cell);
      }
    }
    g.cells = cells;
    g.at = (r, c) => cells[r * cols + c];
    return g;
  },

  /* ======================================================================
     矩阵热力图：val(r,c) -> 0..1
     ====================================================================== */
  matrix(rows, cols, val, o = {}) {
    const cs = o.cell || 9, gap = o.gap == null ? 1.5 : o.gap;
    const g = U.el('div', { class: 'mtx' });
    g.style.gridTemplateColumns = 'repeat(' + cols + ',' + cs + 'px)';
    g.style.gridTemplateRows = 'repeat(' + rows + ',' + cs + 'px)';
    g.style.gap = gap + 'px';
    const cells = [];
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const e = U.el('div', { class: 'mx' });
        const v = val(r, c);
        e.style.background = v > 0
          ? 'rgba(34,211,238,' + (0.10 + 0.85 * v).toFixed(3) + ')'
          : 'rgba(150,180,255,.07)';
        g.appendChild(e); cells.push(e);
      }
    }
    g.cells = cells;
    g.at = (r, c) => cells[r * cols + c];
    /* 逐格点亮：把 value 直接写成颜色 */
    g.set = (r, c, v) => {
      const e = g.at(r, c); if (!e) return;
      e.style.background = v > 0 ? 'rgba(34,211,238,' + (0.10 + 0.85 * v).toFixed(3) + ')' : 'rgba(150,180,255,.07)';
    };
    return g;
  },

  /* ======================================================================
     设备网格（mesh）：axes = [['x',2],['y',2]]
     ====================================================================== */
  mesh(axes, o = {}) {
    const [a0, a1] = axes;
    const cs = o.cell || 62;
    const g = U.el('div', { class: 'mesh' });
    g.style.gridTemplateColumns = 'repeat(' + a1[1] + ',' + cs + 'px)';
    g.style.gridTemplateRows = 'repeat(' + a0[1] + ',' + cs + 'px)';
    const devs = [];
    for (let i = 0; i < a0[1]; i++) {
      for (let j = 0; j < a1[1]; j++) {
        const id = i * a1[1] + j;
        const d = U.el('div', { class: 'dev' });
        d.style.setProperty('--c', 'var(--c' + U.ci(i) + ')');
        d.innerHTML = '<span class="dev-id">' + id + '</span>';
        if (o.cell2) { const t = o.cell2(id, i, j); if (t) d.insertAdjacentHTML('beforeend', '<span class="dev-id" style="margin-top:4px;opacity:.7">' + t + '</span>'); }
        g.appendChild(d); devs.push(d);
      }
    }
    g.devs = devs;
    g.at = id => devs[id];
    return g;
  },

  /* ======================================================================
     练习（折叠式自测）
     ====================================================================== */
  exercise(q, a, o = {}) {
    const box = U.el('div', { class: 'ex' + (o.cls ? ' ' + o.cls : '') });
    box.appendChild(U.el('div', { class: 'q' }, [
      U.el('span', { class: 'qm', text: 'Q' }),
      U.el('span', { html: q }),
    ]));
    const tog = U.el('div', { class: 'tog' }, [U.el('span', { text: '▸ 展开答案' })]);
    const ans = U.el('div', { class: 'a', html: a });
    tog.addEventListener('click', () => {
      box.classList.toggle('open');
      tog.firstElementChild.textContent = box.classList.contains('open') ? '▾ 收起答案' : '▸ 展开答案';
    });
    box.appendChild(tog); box.appendChild(ans);
    return box;
  },

  /* ======================================================================
     页面骨架 —— 生成一个符合引擎约定的课件页面
     ====================================================================== */
  skeleton(o) {
    document.title = o.title || '课件';
    const icon = o.icon || 'data:image/svg+xml,' + encodeURIComponent(
      '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="7" fill="#0b1222"/>' +
      '<path d="M8 21V11h3.4l4.6 6 4.6-6H24v10h-2.8v-5.6L17 21.4h-2l-4.2-5.6V21z" fill="#22d3ee"/></svg>');
    /* ★ 保留 index.html 里已经写好的相对路径。课件的目录深度不同（层内 vs 跨层），
       由 JS 猜路径必错；页面里已经有 <link rel=stylesheet> 就以它为准。 */
    const existing = document.querySelector('link[rel="stylesheet"]');
    const cssHref = o.css || (existing && existing.getAttribute('href')) || 'shared/theme.css';
    const head = document.head;
    head.innerHTML =
      '<meta charset="utf-8">' +
      '<meta name="viewport" content="width=device-width,initial-scale=1">' +
      '<title>' + U.esc(o.title || '课件') + '</title>' +
      '<link rel="icon" href="' + icon + '">' +
      '<link rel="stylesheet" href="' + cssHref + '">';
    document.body.innerHTML =
      '<div id="bgAurora"><i></i><i></i><i></i></div>' +
      '<div id="bgGrid"></div>' +
      '<div id="bgVignette"></div>' +
      '<div id="app"><div id="stage">' +
      '  <header id="hdr"><div id="hdrMain">' +
      '    <div id="hdrKicker"></div>' +
      '    <h1 id="hdrTitle"></h1>' +
      '    <div id="hdrSub"></div>' +
      '  </div><div id="hdrRight" class="row gap8" style="flex:none;padding-top:4px"></div></header>' +
      '  <main id="visual"></main>' +
      '  <aside id="side"><div id="sideHd">' +
      '    <i class="dot r"></i><i class="dot y"></i><i class="dot g"></i>' +
      '    <span class="nm" id="sideName"></span></div>' +
      '    <div id="sideBody"></div><div id="sideNote"></div></aside>' +
      '  <footer id="ftr">' +
      '    <div id="cap"></div>' +
      '    <div id="bar"><div id="barFill"></div></div>' +
      '    <div id="transport"><div id="btns"></div><div id="ticks"></div><div id="counter"></div></div>' +
      '  </footer>' +
      '</div></div>';
  },
};

/* 箭头图标（内联 SVG，避免任何外部请求） */
W.ICON = {
  prev: '<svg viewBox="0 0 16 16"><path d="M10.5 2.5 5 8l5.5 5.5 1.4-1.4L7.8 8l4.1-4.1z"/></svg>',
  next: '<svg viewBox="0 0 16 16"><path d="M5.5 2.5 11 8l-5.5 5.5-1.4-1.4L8.2 8 4.1 3.9z"/></svg>',
  play: '<svg viewBox="0 0 16 16"><path d="M4 2.6 13 8l-9 5.4z"/></svg>',
  pause: '<svg viewBox="0 0 16 16"><path d="M4 2.5h3v11H4zm5 0h3v11H9z"/></svg>',
  first: '<svg viewBox="0 0 16 16"><path d="M3 2.5h2v11H3zm3.5 5.5L14 2.6v10.8z"/></svg>',
  last: '<svg viewBox="0 0 16 16"><path d="M11 2.5h2v11h-2zM9.5 8 2 13.4V2.6z"/></svg>',
  home: '<svg viewBox="0 0 16 16"><path d="M8 1.6 1 7.3l1.2 1.5L3 8.1V14h4v-3.4h2V14h4V8.1l.8.7L15 7.3z"/></svg>',
};
