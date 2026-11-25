// The ordering problem, live: TS and TB read the paper's 10 x 10 Gaussian
// grid (ell = 5, theta = 3.6) in a fresh random order every few seconds.

import * as T from './thetan.js';
import { CLUSTER, SPUR, assignColors, currentTheme } from './theme.js';

const THETA = 3.6;
const OFF_TOL = 1.0; // a centroid further than this from every true centre is "off-centre"
const INK = { light: '#0f1119', dark: '#eef0f6' };

export function mountDuel(root, { reduced = false } = {}) {
  const data = T.gridGaussians(10, 10, 5, 120, T.mulberry32(2024));
  const rng = T.mulberry32(77);
  const panels = ['ts', 'tb'].map((m) => {
    const fig = root.querySelector(`.duel-panel[data-method="${m}"]`);
    return {
      method: m,
      canvas: fig.querySelector('canvas'),
      k: fig.querySelector('[data-k]'),
      off: fig.querySelector('[data-off]'),
      agree: fig.querySelector('[data-agree]'),
      prev: null, sum: 0, cnt: 0, last: null,
    };
  });
  const section = root.closest('section');
  const countEl = section.querySelector('[data-duel-count]');
  const pauseBtn = section.querySelector('[data-duel-pause]');
  let count = 0, paused = reduced, visible = false, timer = null;
  if (reduced) { pauseBtn.setAttribute('aria-pressed', 'true'); pauseBtn.textContent = 'Play'; }

  function offCentre(r) {
    const flags = new Uint8Array(r.K);
    for (let k = 0; k < r.K; k++) {
      let best = Infinity;
      for (let t = 0; t < data.K; t++) {
        const dx = r.centroids[k * 2] - data.centers[t * 2], dy = r.centroids[k * 2 + 1] - data.centers[t * 2 + 1];
        best = Math.min(best, dx * dx + dy * dy);
      }
      flags[k] = best > OFF_TOL * OFF_TOL;
    }
    return flags;
  }

  function draw(p) {
    const r = p.last;
    if (!r) return;
    const c = p.canvas, dpr = Math.min(window.devicePixelRatio || 1, 2);
    const size = c.clientWidth * dpr;
    if (c.width !== size) { c.width = size; c.height = size; }
    const ctx = c.getContext('2d');
    const ext = 27, s = size / (2 * ext), theme = currentTheme(), pal = CLUSTER[theme];
    const X = (x) => (x + ext) * s, Y = (y) => (ext - y) * s;
    ctx.clearRect(0, 0, size, size);
    const dot = Math.max(1.4, 1.5 * dpr);
    ctx.globalAlpha = 0.8;
    for (let i = 0; i < data.n; i++) {
      ctx.fillStyle = pal[r.cols[r.labels[i]]];
      ctx.fillRect(X(data.X[i * 2]) - dot / 2, Y(data.X[i * 2 + 1]) - dot / 2, dot, dot);
    }
    ctx.globalAlpha = 1;
    for (let k = 0; k < r.K; k++) {
      const x = X(r.centroids[k * 2]), y = Y(r.centroids[k * 2 + 1]);
      if (r.off[k]) {
        ctx.save();
        if (theme === 'dark') { ctx.shadowColor = SPUR.dark; ctx.shadowBlur = 12 * dpr; }
        ctx.strokeStyle = SPUR[theme]; ctx.lineWidth = 2 * dpr;
        ctx.beginPath(); ctx.arc(x, y, 6.5 * dpr, 0, Math.PI * 2); ctx.stroke();
        ctx.restore();
      } else {
        ctx.fillStyle = INK[theme];
        ctx.beginPath(); ctx.arc(x, y, 2 * dpr, 0, Math.PI * 2); ctx.fill();
      }
    }
  }

  function step() {
    count++;
    countEl.textContent = count;
    const results = {
      ts: T.sequentialShuffled(data.X, data.n, data.d, THETA, rng),
      tb: T.berserker(data.X, data.n, data.d, THETA, rng),
    };
    for (const p of panels) {
      const r = results[p.method];
      r.off = offCentre(r);
      r.cols = assignColors(r.centroids, r.K, 2); // neighbouring clusters never share a hue
      p.k.textContent = r.K;
      p.off.textContent = r.off.reduce((a, b) => a + b, 0);
      if (p.prev) {
        const ari = T.adjustedRandIndex(p.prev, r.labels);
        p.sum += ari; p.cnt++;
        p.agree.innerHTML = `${ari.toFixed(3)} <span>avg ${(p.sum / p.cnt).toFixed(3)}</span>`;
      }
      p.prev = r.labels;
      p.last = r;
      draw(p);
      p.canvas.animate([{ opacity: 0.35 }, { opacity: 1 }], { duration: 450, easing: 'ease-out' });
    }
  }

  function schedule() {
    clearInterval(timer);
    if (visible && !paused) timer = setInterval(() => { if (!document.hidden) step(); }, 2600);
  }

  new IntersectionObserver(([e]) => {
    visible = e.isIntersecting;
    if (visible && count === 0) step();
    schedule();
  }, { rootMargin: '200px' }).observe(root);

  section.querySelector('[data-duel-shuffle]').addEventListener('click', () => { step(); schedule(); });
  pauseBtn.addEventListener('click', () => {
    paused = !paused;
    pauseBtn.setAttribute('aria-pressed', String(paused));
    pauseBtn.textContent = paused ? 'Play' : 'Pause';
    schedule();
  });
  new ResizeObserver(() => panels.forEach(draw)).observe(root);
  return { redraw: () => panels.forEach(draw) };
}
