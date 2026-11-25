// The θ playground: twelve Gaussians on a flat map. Move θ and Thetan reruns at
// once; every cluster wears a hue and shows its centroid and θ circle, to scale.

import * as T from './thetan.js';
import { CLUSTER, assignColors, currentTheme, rgbOf } from './theme.js';
import { sweepChart } from './charts.js';

const DATA_SEED = 11; // θ = 3.8 sits on the plateau where TB finds all twelve and TS does not
const TINY = 8;       // clusters with fewer points are outliers: a hollow ring, no θ circle
const TAU = Math.PI * 2;

// Twelve unit Gaussians on a jittered map, no two centres closer than 8.5.
export function makeMapData(seed, { K = 12, W = 24, H = 14, minSep = 8.5, lo = 120, hi = 220 } = {}) {
  const rng = T.mulberry32(seed);
  const centers = [];
  for (let guard = 0; centers.length < K && guard < 20000; guard++) {
    const c = [(rng() * 2 - 1) * W, (rng() * 2 - 1) * H];
    if (centers.every((o) => Math.hypot(o[0] - c[0], o[1] - c[1]) >= minSep)) centers.push(c);
  }
  const sizes = centers.map(() => Math.round(lo + rng() * (hi - lo)));
  const n = sizes.reduce((a, b) => a + b, 0);
  const X = new Float64Array(n * 2), y = new Int32Array(n);
  let i = 0;
  centers.forEach((c, k) => {
    for (let s = 0; s < sizes[k]; s++, i++) {
      X[i * 2] = c[0] + T.randn(rng);
      X[i * 2 + 1] = c[1] + T.randn(rng);
      y[i] = k;
    }
  });
  return { X, y, n, d: 2, K: centers.length };
}

// one run of TB, with TS (its first pass) alongside, both ready to draw
function cluster(data, theta, seed) {
  const t0 = performance.now();
  const res = T.berserker(data.X, data.n, 2, theta, T.mulberry32(seed));
  const ms = performance.now() - t0;
  const tr = res.trace;
  const tsLabels = new Int32Array(data.n);
  for (let j = 0; j < data.n; j++) tsLabels[tr.perm1[j]] = tr.labels1[j];
  const view = (labels, C, K) => {
    const size = new Int32Array(K);
    for (const l of labels) size[l]++;
    // big clusters choose hues first; neighbours never share one
    const order = Array.from({ length: K }, (_, k) => k).sort((a, b) => size[b] - size[a]);
    return { labels, C, K, size, hue: assignColors(C, K, 2, order) };
  };
  return { theta, ms, tb: view(res.labels, res.centroids, res.K), ts: view(tsLabels, tr.M1, tr.K1) };
}

export function mountPlayground(root) {
  const canvas = root.querySelector('[data-map] canvas');
  const ctx = canvas.getContext('2d');
  const slider = root.querySelector('#theta-slider');
  const out = root.querySelector('#theta-out');
  const segBtns = [...root.querySelectorAll('.seg button')];
  const field = (k) => root.querySelector(`[data-play="${k}"]`);
  const sweepBox = root.querySelector('[data-sweep]');

  let dataSeed = DATA_SEED, orderSeed = 5, method = 'tb';
  let data = makeMapData(dataSeed);
  let result = cluster(data, +slider.value, orderSeed);
  let sweep = null, sweepToken = 0, swept = false;

  // ----- the map -----

  let W = 0, H = 0, sc = 1, ox = 0, oy = 0, bounds = null;
  function fit() {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let i = 0; i < data.n; i++) {
      x0 = Math.min(x0, data.X[i * 2]); x1 = Math.max(x1, data.X[i * 2]);
      y0 = Math.min(y0, data.X[i * 2 + 1]); y1 = Math.max(y1, data.X[i * 2 + 1]);
    }
    const pad = 2.5;
    bounds = [x0 - pad, x1 + pad, y0 - pad, y1 + pad];
  }
  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = canvas.clientWidth; H = canvas.clientHeight;
    if (!W || !H) return;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const [x0, x1, y0, y1] = bounds;
    const Hd = H - 30; // a strip along the bottom keeps the scale bar clear of the points
    sc = Math.min(W / (x1 - x0), Hd / (y1 - y0));
    ox = (W - (x1 - x0) * sc) / 2 - x0 * sc;
    oy = (Hd - (y1 - y0) * sc) / 2 + y1 * sc;
    draw();
  }
  const px = (x) => ox + x * sc;
  const py = (y) => oy - y * sc;

  function draw() {
    if (!W) return;
    const theme = currentTheme(), dark = theme === 'dark';
    const hues = CLUSTER[theme].map(rgbOf);
    const ink = dark ? '238, 240, 246' : '15, 17, 25';
    const muted = dark ? '150, 158, 178' : '106, 112, 130';
    const v = result[method], theta = result.theta;
    ctx.clearRect(0, 0, W, H);

    // faint graph-paper dots, one every 2 units
    ctx.fillStyle = dark ? 'rgba(255, 255, 255, 0.045)' : 'rgba(15, 17, 26, 0.08)';
    const [x0, x1, y0, y1] = bounds;
    for (let gx = Math.ceil(x0 / 2) * 2; gx <= x1; gx += 2) {
      for (let gy = Math.ceil(y0 / 2) * 2; gy <= y1; gy += 2) ctx.fillRect(px(gx) - 0.75, py(gy) - 0.75, 1.5, 1.5);
    }

    // θ around every real cluster, to scale
    ctx.setLineDash([4, 5]);
    ctx.lineWidth = 1;
    for (let k = 0; k < v.K; k++) {
      if (v.size[k] < TINY) continue;
      const c = hues[v.hue[k]];
      ctx.beginPath();
      ctx.arc(px(v.C[k * 2]), py(v.C[k * 2 + 1]), theta * sc, 0, TAU);
      ctx.fillStyle = `rgba(${c}, ${dark ? 0.05 : 0.06})`;
      ctx.fill();
      ctx.strokeStyle = `rgba(${c}, 0.5)`;
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // points, in their cluster's hue
    const r = Math.max(1.4, Math.min(2.2, sc * 0.16));
    for (let i = 0; i < data.n; i++) {
      const l = v.labels[i];
      ctx.fillStyle = v.size[l] < TINY ? `rgba(${muted}, 0.9)` : `rgba(${hues[v.hue[l]]}, 0.85)`;
      ctx.beginPath();
      ctx.arc(px(data.X[i * 2]), py(data.X[i * 2 + 1]), r, 0, TAU);
      ctx.fill();
    }

    // centroids; an outlier cluster gets a hollow ring
    for (let k = 0; k < v.K; k++) {
      const x = px(v.C[k * 2]), y = py(v.C[k * 2 + 1]);
      if (v.size[k] < TINY) {
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, TAU);
        ctx.strokeStyle = `rgba(${muted}, 1)`;
        ctx.lineWidth = 1.5;
        ctx.stroke();
        continue;
      }
      const c = hues[v.hue[k]];
      ctx.save();
      if (dark) { ctx.shadowColor = `rgba(${c}, 0.9)`; ctx.shadowBlur = 12; }
      ctx.beginPath();
      ctx.arc(x, y, 5.5, 0, TAU);
      ctx.fillStyle = `rgb(${c})`;
      ctx.fill();
      ctx.restore();
      ctx.beginPath();
      ctx.arc(x, y, 2.2, 0, TAU);
      ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
      ctx.fill();
    }

    // scale bar: how long θ is on this map
    const bx = 16, by = H - 18, len = theta * sc;
    ctx.strokeStyle = `rgba(${ink}, 0.8)`;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(bx, by - 4); ctx.lineTo(bx, by); ctx.lineTo(bx + len, by); ctx.lineTo(bx + len, by - 4);
    ctx.stroke();
    ctx.font = '500 11px "Geist Mono", ui-monospace, monospace';
    ctx.fillStyle = `rgba(${ink}, 0.85)`;
    ctx.textAlign = 'left';
    ctx.fillText(`θ = ${theta.toFixed(1)}`, bx, by - 9);
  }

  // ----- readouts -----

  function readout() {
    const v = result[method];
    const real = v.size.reduce((a, s) => a + (s >= TINY ? 1 : 0), 0);
    field('k').textContent = v.K;
    field('truek').textContent = data.K;
    field('ari').textContent = T.adjustedRandIndex(v.labels, data.y).toFixed(3);
    field('ms').textContent = `${result.ms < 10 ? result.ms.toFixed(1) : Math.round(result.ms)} ms`;
    field('outliers').textContent = v.K - real ? `${v.K - real} of them ${v.K - real === 1 ? 'is an outlier' : 'are outliers'}` : '';
    canvas.setAttribute('aria-label', `${method === 'tb' ? 'Thetan Berserker' : 'Thetan Sequential'} at θ = ${result.theta.toFixed(1)}: ${v.K} clusters for ${data.K} true groups`);
  }

  function rerun() {
    result = cluster(data, +slider.value, orderSeed);
    readout();
    draw();
  }

  // ----- K(θ) for TB and TS, in small chunks so scrolling stays smooth -----

  function runSweep() {
    const token = ++sweepToken;
    const thetas = [];
    for (let t = 1.5; t <= 9.001; t += 0.25) thetas.push(+t.toFixed(2));
    const tb = [], ts = [];
    sweepBox.innerHTML = '<p class="sweep-wait">Sweeping θ…</p>';
    let i = 0;
    const next = () => {
      if (token !== sweepToken) return;
      const r = T.berserker(data.X, data.n, 2, thetas[i], T.mulberry32(5));
      tb.push(r.K);
      ts.push(r.trace.K1);
      if (++i < thetas.length) { setTimeout(next, 0); return; }
      sweep = sweepChart(sweepBox, {
        thetas, tb, ts, trueK: data.K,
        onPick: (v) => { slider.value = v; slider.dispatchEvent(new Event('input')); },
      });
      sweep.setTheta(+slider.value);
    };
    next();
  }

  // ----- controls -----

  function syncSliderFill() {
    const f = ((slider.value - slider.min) / (slider.max - slider.min)) * 100;
    slider.style.setProperty('--fill', `${f}%`);
  }

  let pending = 0;
  slider.addEventListener('input', () => {
    out.textContent = (+slider.value).toFixed(1);
    syncSliderFill();
    if (sweep) sweep.setTheta(+slider.value);
    if (!pending) pending = requestAnimationFrame(() => { pending = 0; rerun(); });
  });
  segBtns.forEach((b) => b.addEventListener('click', () => {
    method = b.dataset.method;
    segBtns.forEach((x) => x.setAttribute('aria-checked', String(x === b)));
    readout();
    draw();
  }));
  root.querySelectorAll('[data-play-action]').forEach((b) => b.addEventListener('click', () => {
    if (b.dataset.playAction === 'shuffle') orderSeed += 7;
    else {
      data = makeMapData(++dataSeed);
      fit();
      resize();
      runSweep();
    }
    rerun();
  }));

  fit();
  new ResizeObserver(resize).observe(canvas);
  new IntersectionObserver(([e]) => {
    if (e.isIntersecting && !swept) { swept = true; runSweep(); }
  }, { rootMargin: '400px' }).observe(root);
  syncSliderFill();
  readout();
  resize();

  return { redraw: draw };
}
