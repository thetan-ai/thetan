// Small SVG charts: thin marks, recessive grid, direct labels, hover tooltips.

const NS = 'http://www.w3.org/2000/svg';

function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

const tipEl = () => document.querySelector('.tooltip');
export function showTip(html, x, y) {
  const t = tipEl();
  t.innerHTML = html;
  t.classList.add('show');
  const r = t.getBoundingClientRect();
  let left = x + 14, top = y - r.height - 10;
  if (left + r.width > window.innerWidth - 8) left = x - r.width - 14;
  if (top < 8) top = y + 16;
  t.style.left = `${left}px`;
  t.style.top = `${top}px`;
}
export function hideTip() { tipEl().classList.remove('show'); }

// right-rounded bar anchored square at the baseline
function hbar(x0, x1, y, h, r = 4) {
  r = Math.min(r, h / 2, Math.max(0, x1 - x0));
  return `M${x0},${y}H${x1 - r}Q${x1},${y} ${x1},${y + r}V${y + h - r}Q${x1},${y + h} ${x1 - r},${y + h}H${x0}Z`;
}
function vbar(x, w, y0, y1, r = 4) {
  r = Math.min(r, w / 2, Math.max(0, y0 - y1));
  return `M${x},${y0}V${y1 + r}Q${x},${y1} ${x + r},${y1}H${x + w - r}Q${x + w},${y1} ${x + w},${y1 + r}V${y0}Z`;
}

// Horizontal bars on a log axis. rows: {label, value, display, tip, highlight}
export function logBars(container, rows, { domain, ticks, tickFmt }) {
  const W = 560, labelW = 168, valueW = 74, rowH = 34, barH = 14, top = 6;
  const H = top + rows.length * rowH + 28;
  const x0 = labelW, x1 = W - valueW;
  const lmin = Math.log10(domain[0]), lmax = Math.log10(domain[1]);
  const sx = (v) => x0 + ((Math.log10(v) - lmin) / (lmax - lmin)) * (x1 - x0);
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img' });

  const gy = top + rows.length * rowH;
  for (const t of ticks) {
    const x = sx(t);
    el('line', { x1: x, x2: x, y1: top - 2, y2: gy, class: 'grid' }, svg);
    el('text', { x, y: gy + 18, 'text-anchor': 'middle' }, svg).textContent = tickFmt(t);
  }
  el('line', { x1: x0, x2: x0, y1: top - 2, y2: gy, class: 'axis' }, svg);

  rows.forEach((r, i) => {
    const y = top + i * rowH + (rowH - barH) / 2;
    const g = el('g', {}, svg);
    el('text', { x: x0 - 12, y: y + barH / 2 + 4, 'text-anchor': 'end', class: `lbl${r.highlight ? ' hi' : ''}` }, g).textContent = r.label;
    const bar = el('path', { d: hbar(x0, sx(r.value), y, barH), class: 'bar', style: `fill: ${r.highlight ? 'var(--accent)' : 'var(--neutral-mark)'}` }, g);
    el('text', { x: sx(r.value) + 8, y: y + barH / 2 + 4, class: `val${r.highlight ? ' hi' : ''}` }, g).textContent = r.display;
    const hit = el('rect', { x: 0, y: top + i * rowH, width: W, height: rowH, class: 'hit' }, g);
    hit.addEventListener('pointermove', (e) => { bar.classList.add('hover'); showTip(r.tip, e.clientX, e.clientY); });
    hit.addEventListener('pointerleave', () => { bar.classList.remove('hover'); hideTip(); });
  });

  svg.setAttribute('aria-label', rows.map((r) => `${r.label}: ${r.display}`).join(', '));
  container.replaceChildren(svg);
}

// Propositions 1-2: measured short-edge probability by dimension, with the exponential fit.
export function pThetaChart(container) {
  const data = [[2, 0.065], [3, 0.037], [4, 0.02], [5, 0.01]];
  const fit = (d) => 0.217 * Math.exp(-0.594 * d);
  const W = 360, H = 220, l = 40, r = 12, t = 16, b = 34;
  const sx = (d) => l + ((d - 1.5) / 4) * (W - l - r);
  const sy = (v) => H - b - (v / 0.08) * (H - t - b);
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'p theta by dimension: d=2 0.065, d=3 0.037, d=4 0.020, d=5 0.010' });
  for (const v of [0, 0.02, 0.04, 0.06, 0.08]) {
    el('line', { x1: l, x2: W - r, y1: sy(v), y2: sy(v), class: v === 0 ? 'axis' : 'grid' }, svg);
    el('text', { x: l - 8, y: sy(v) + 4, 'text-anchor': 'end' }, svg).textContent = v.toFixed(2);
  }
  let path = '';
  for (let d = 1.6; d <= 5.4; d += 0.05) path += `${path ? 'L' : 'M'}${sx(d).toFixed(1)},${sy(fit(d)).toFixed(1)}`;
  const bw = 30;
  data.forEach(([d, v]) => {
    const bar = el('path', { d: vbar(sx(d) - bw / 2, bw, sy(0), sy(v)), class: 'bar', style: 'fill: var(--accent)' }, svg);
    el('text', { x: sx(d), y: sy(v) - 8, 'text-anchor': 'middle', class: 'val' }, svg).textContent = v.toFixed(3);
    el('text', { x: sx(d), y: H - b + 18, 'text-anchor': 'middle' }, svg).textContent = `d = ${d}`;
    const hit = el('rect', { x: sx(d) - 30, y: t, width: 60, height: H - t - b, class: 'hit' }, svg);
    hit.addEventListener('pointermove', (e) => { bar.classList.add('hover'); showTip(`<b>d = ${d}</b><br>p<sub>θ</sub> = ${v.toFixed(3)} · fit ${fit(d).toFixed(3)}`, e.clientX, e.clientY); });
    hit.addEventListener('pointerleave', () => { bar.classList.remove('hover'); hideTip(); });
  });
  el('path', { d: path, fill: 'none', stroke: 'var(--ink-2)', 'stroke-width': 1.5, 'stroke-dasharray': '4 4', 'pointer-events': 'none' }, svg);
  el('text', { x: sx(4.1), y: sy(fit(4.1)) - 26, class: 'val', 'text-anchor': 'start' }, svg).textContent = 'fit 0.217·e^(−0.594d)';
  container.replaceChildren(svg);
}

// K inferred across θ for TB and TS, log y, with a crosshair and click-to-set.
export function sweepChart(container, { thetas, tb, ts, trueK, onPick }) {
  const W = 380, H = 150, l = 34, r = 8, t = 8, b = 22;
  const all = [...tb, ...ts, trueK];
  const lo = Math.max(1, Math.min(...all) * 0.8), hi = Math.max(...all) * 1.25;
  const tx = (v) => l + ((v - thetas[0]) / (thetas[thetas.length - 1] - thetas[0])) * (W - l - r);
  const ty = (k) => t + (1 - (Math.log10(k) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * (H - t - b);
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Clusters inferred by TB and TS across theta' });

  for (const k of [1, 3, 10, 30, 100, 300, 1000, 3000]) {
    if (k < lo || k > hi) continue;
    el('line', { x1: l, x2: W - r, y1: ty(k), y2: ty(k), class: 'grid' }, svg);
    el('text', { x: l - 6, y: ty(k) + 4, 'text-anchor': 'end' }, svg).textContent = k;
  }
  for (let v = Math.ceil(thetas[0]); v <= thetas[thetas.length - 1]; v++) {
    el('text', { x: tx(v), y: H - 6, 'text-anchor': 'middle' }, svg).textContent = v;
  }
  el('line', { x1: l, x2: W - r, y1: H - b, y2: H - b, class: 'axis' }, svg);
  el('line', { x1: l, x2: W - r, y1: ty(trueK), y2: ty(trueK), stroke: 'var(--ink-2)', 'stroke-width': 1, 'stroke-dasharray': '3 3' }, svg);
  const poly = (ys) => ys.map((k, i) => `${i ? 'L' : 'M'}${tx(thetas[i]).toFixed(1)},${ty(Math.max(lo, k)).toFixed(1)}`).join('');
  el('path', { d: poly(ts), fill: 'none', stroke: 'var(--ts-mark)', 'stroke-width': 2, 'stroke-linejoin': 'round' }, svg);
  el('path', { d: poly(tb), fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2, 'stroke-linejoin': 'round' }, svg);

  const marker = el('g', { 'pointer-events': 'none' }, svg);
  const mLine = el('line', { y1: t, y2: H - b, stroke: 'var(--accent)', 'stroke-width': 1, opacity: 0.6 }, marker);
  const mDot = el('circle', { r: 4.5, fill: 'var(--accent)', stroke: 'var(--bg)', 'stroke-width': 2 }, marker);
  const hover = el('line', { y1: t, y2: H - b, stroke: 'var(--ink-2)', 'stroke-width': 1, opacity: 0 }, svg);

  const nearest = (clientX) => {
    const box = svg.getBoundingClientRect();
    const x = ((clientX - box.left) / box.width) * W;
    let best = 0;
    thetas.forEach((v, i) => { if (Math.abs(tx(v) - x) < Math.abs(tx(thetas[best]) - x)) best = i; });
    return best;
  };
  const hit = el('rect', { x: l, y: t, width: W - l - r, height: H - t - b, fill: 'transparent', style: 'cursor: crosshair' }, svg);
  hit.addEventListener('pointermove', (e) => {
    const i = nearest(e.clientX);
    hover.setAttribute('x1', tx(thetas[i])); hover.setAttribute('x2', tx(thetas[i])); hover.setAttribute('opacity', 0.5);
    showTip(`<b>θ = ${thetas[i].toFixed(2)}</b><br>TB K = ${tb[i]} · TS K = ${ts[i]} · true ${trueK}<br><span style="opacity:.7">click to set θ</span>`, e.clientX, e.clientY);
  });
  hit.addEventListener('pointerleave', () => { hover.setAttribute('opacity', 0); hideTip(); });
  hit.addEventListener('click', (e) => onPick(thetas[nearest(e.clientX)]));

  container.replaceChildren(svg);
  return {
    setTheta(theta) {
      const x = tx(Math.min(thetas[thetas.length - 1], Math.max(thetas[0], theta)));
      let i = 0;
      thetas.forEach((v, j) => { if (Math.abs(v - theta) < Math.abs(thetas[i] - theta)) i = j; });
      mLine.setAttribute('x1', x); mLine.setAttribute('x2', x);
      mDot.setAttribute('cx', x); mDot.setAttribute('cy', ty(Math.max(lo, tb[i])));
    },
  };
}
