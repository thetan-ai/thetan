// The step-by-step figure: one real Thetan Berserker run on 150 points from
// five Gaussians (θ = 3.6), replayed as one short loop per step.

import * as T from './thetan.js';
import { ACCENT as ACCENT_HEX, CLUSTER, SPUR as SPUR_HEX, assignColors, currentTheme, rgbOf } from './theme.js';

const THETA = 3.6;
const CENTERS = [[-9.5, 2.5], [-3.5, -4], [1.5, 4], [7, -2.5], [10.5, 4.5]];
const PER = 30;
const SIGMA = 1.05;
const DATA_SEED = 2;
// a real run whose first pass makes two mistakes and whose second pass meets an outlier
const RUN_SEED = 3;

// colours for the current theme, as "r, g, b" (one figure per page, so module state is fine)
let INK, DUST, ACCENT, SPUR, HALO, GRID, GLOW, HUES;
function applyColors(theme) {
  const dark = theme === 'dark';
  INK = dark ? '238, 240, 246' : '15, 17, 25';
  DUST = dark ? '150, 158, 178' : '106, 112, 130';
  ACCENT = rgbOf(ACCENT_HEX[theme]);
  SPUR = rgbOf(SPUR_HEX[theme]);
  HALO = dark ? '6, 7, 12' : '255, 255, 255';
  GRID = dark ? 'rgba(255, 255, 255, 0.045)' : 'rgba(15, 17, 26, 0.09)';
  GLOW = dark;
  HUES = CLUSTER[theme].map(rgbOf);
}
applyColors(currentTheme());
const rgba = (c, a) => `rgba(${c}, ${Math.max(0, Math.min(1, a)).toFixed(3)})`;
const clamp01 = (t) => Math.min(1, Math.max(0, t));
const seg = (t, a, b) => clamp01((t - a) / (b - a)); // progress of t through [a, b]
const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const lerp = (a, b, t) => a + (b - a) * t;
const SUB = '₀₁₂₃₄₅₆₇₈₉';
const nameOf = (k) => `c${String(k).replace(/\d/g, (d) => SUB[d])}`;
const WORDS = ['no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'];
const word = (v) => WORDS[v] ?? String(v);
const TAU = Math.PI * 2;
// a distance next to θ: two decimals when one would round onto θ itself
const dist = (d) => (Math.abs(d - THETA) < 0.1 ? d.toFixed(2) : d.toFixed(1));

// ---------- the run, and the tables its replay needs ----------

// Centroid positions before every visit of one TS pass, so the figure can show
// what each point was measured against.
function replayPass(X, order, labels, K, prior) {
  const n = order.length;
  const sums = new Float64Array(K * 2), cnt = new Float64Array(K);
  let created = 0;
  if (prior) { sums.set(prior.sums); cnt.set(prior.cnt); created = prior.created; }
  const snaps = [], made = [];
  for (let j = 0; j <= n; j++) {
    const s = new Float64Array(K * 2);
    for (let k = 0; k < created; k++) { s[k * 2] = sums[k * 2] / cnt[k]; s[k * 2 + 1] = sums[k * 2 + 1] / cnt[k]; }
    snaps.push(s);
    made.push(created);
    if (j === n) break;
    const i = order[j], l = labels[j];
    sums[l * 2] += X[i * 2]; sums[l * 2 + 1] += X[i * 2 + 1]; cnt[l] += 1;
    if (l >= created) created = l + 1;
  }
  return { snaps, made };
}

function nearest(C, K, x, y) {
  let best = Infinity, arg = -1;
  for (let k = 0; k < K; k++) {
    const d = Math.hypot(C[k * 2] - x, C[k * 2 + 1] - y);
    if (d < best) { best = d; arg = k; }
  }
  return { k: arg, d: best };
}

function buildModel() {
  const rng = T.mulberry32(DATA_SEED);
  const n = CENTERS.length * PER, X = new Float64Array(n * 2), y = new Int32Array(n);
  let i = 0;
  CENTERS.forEach((c, k) => {
    for (let s = 0; s < PER; s++, i++) {
      X[i * 2] = c[0] + SIGMA * T.randn(rng);
      X[i * 2 + 1] = c[1] + SIGMA * T.randn(rng);
      y[i] = k;
    }
  });
  const res = T.berserker(X, n, 2, THETA, T.mulberry32(RUN_SEED));
  const tr = res.trace;
  const original = (perm, arr) => { const out = new Int32Array(n); for (let j = 0; j < n; j++) out[perm[j]] = arr[j]; return out; };
  const order = Int32Array.from({ length: n }, (_, j) => j);

  // pass 1: centroid k is called c_k
  const p1 = replayPass(X, tr.perm1, tr.labels1, tr.K1, null);
  const lab1 = original(tr.perm1, tr.labels1), vis1 = original(tr.perm1, order);
  const firstOf = new Int32Array(tr.K1).fill(-1); // the point that opened each centroid
  for (let j = 0; j < n; j++) if (firstOf[tr.labels1[j]] === -1) firstOf[tr.labels1[j]] = tr.perm1[j];

  // cleanup 1: survivors keep their pass-1 names
  const surv1 = [];
  for (let k = 0; k < tr.K1; k++) if (!tr.cand1.isCandidate[k]) surv1[tr.cand1.newIndex[k]] = k;
  const lab2 = original(tr.perm1, tr.labels2).map((s) => surv1[s]);
  const M2 = new Float64Array(tr.K1 * 2); // M⋆₂, indexed by pass-1 name
  surv1.forEach((k, s) => { M2[k * 2] = tr.M2star[s * 2]; M2[k * 2 + 1] = tr.M2star[s * 2 + 1]; });

  // pass 2: the prepended centroids are read first, as single points
  const prior = { sums: new Float64Array(tr.K3 * 2), cnt: new Float64Array(tr.K3), created: tr.priorK };
  const creator = new Int32Array(tr.priorK).fill(-1);
  for (let s = 0; s < tr.K2; s++) {
    const c = tr.priorLabels[s];
    prior.sums[c * 2] += tr.M2star[s * 2]; prior.sums[c * 2 + 1] += tr.M2star[s * 2 + 1]; prior.cnt[c] += 1;
    if (creator[c] === -1) creator[c] = s;
  }
  const p2 = replayPass(X, tr.perm2, tr.labels3, tr.K3, prior);
  const lab3 = original(tr.perm2, tr.labels3), vis2 = original(tr.perm2, order);
  // a prepended centroid keeps its name; one opened in pass 2 gets the next free name
  const name3 = Array.from({ length: tr.K3 }, (_, c) => (c < tr.priorK ? surv1[creator[c]] : tr.K1 + (c - tr.priorK)));

  // cleanup 2 and the final clustering, indexed by pass-2 centroid
  const finalOf = new Int32Array(tr.K3).fill(-1);
  for (let c = 0; c < tr.K3; c++) if (!tr.cand2.isCandidate[c]) finalOf[c] = tr.finalMap[tr.cand2.newIndex[c]];
  const lab4 = new Int32Array(n); // pass-2 centroid each point ends with
  for (let i2 = 0; i2 < n; i2++) {
    const c = lab3[i2];
    lab4[i2] = tr.cand2.isCandidate[c] ? tr.cand2.keeperOf[c] : c;
  }

  // every centroid name gets a hue; the clusters that survive choose first,
  // so the final answer shows distinct colours and each duplicate differs from its twin
  const names = tr.K1 + tr.K3 - tr.priorK;
  const P = new Float64Array(names * 2);
  for (let k = 0; k < tr.K1; k++) { P[k * 2] = tr.M1[k * 2]; P[k * 2 + 1] = tr.M1[k * 2 + 1]; }
  for (let c = tr.priorK; c < tr.K3; c++) { const k = name3[c]; P[k * 2] = tr.M3[c * 2]; P[k * 2 + 1] = tr.M3[c * 2 + 1]; }
  const survivors = name3.filter((_, c) => !tr.cand2.isCandidate[c]);
  const hueOf = assignColors(P, names, 2, [...new Set([...survivors, ...Array.from({ length: names }, (_, k) => k)])]);

  return {
    X, y, n, K: CENTERS.length, hueOf, tr, res, p1, p2, lab1, vis1, firstOf, lab2, M2, lab3, vis2, name3, finalOf, lab4,
    ari: T.adjustedRandIndex(y, res.labels),
  };
}

// ---------- the figure ----------

export function mountExplainer(root, { reduced = false } = {}) {
  const canvas = root.querySelector('canvas');
  const ctx = canvas.getContext('2d');
  const capEl = root.querySelector('[data-fig="caption"]');
  const statEl = root.querySelector('[data-fig="stats"]');
  const m = buildModel();
  const { X, n, tr } = m;
  const hue = (name) => HUES[m.hueOf[name]];

  // world -> canvas
  let W = 0, H = 0, sc = 1, ox = 0, oy = 0;
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (let i = 0; i < n; i++) {
    xmin = Math.min(xmin, X[i * 2]); xmax = Math.max(xmax, X[i * 2]);
    ymin = Math.min(ymin, X[i * 2 + 1]); ymax = Math.max(ymax, X[i * 2 + 1]);
  }
  const pad = 1.6;
  xmin -= pad; xmax += pad; ymin -= pad; ymax += pad + 1.2; // headroom for labels

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = canvas.clientWidth; H = canvas.clientHeight;
    if (!W || !H) return;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    sc = Math.min(W / (xmax - xmin), H / (ymax - ymin));
    ox = (W - (xmax - xmin) * sc) / 2 - xmin * sc;
    oy = (H - (ymax - ymin) * sc) / 2 + ymax * sc;
    draw();
  }
  const px = (x) => ox + x * sc;
  const py = (y) => oy - y * sc;

  // ----- drawing helpers -----

  function backdrop() {
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = GRID;
    for (let gx = Math.ceil(xmin / 2) * 2; gx <= xmax; gx += 2) {
      for (let gy = Math.ceil(ymin / 2) * 2; gy <= ymax; gy += 2) ctx.fillRect(px(gx) - 0.75, py(gy) - 0.75, 1.5, 1.5);
    }
  }

  function line(x1, y1, x2, y2, color, width = 1, dash = null) {
    ctx.beginPath();
    ctx.moveTo(px(x1), py(y1)); ctx.lineTo(px(x2), py(y2));
    ctx.strokeStyle = color; ctx.lineWidth = width;
    ctx.setLineDash(dash || []);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  function circle(x, y, r, stroke, fill = null, dash = null, width = 1) {
    if (r <= 0) return;
    ctx.beginPath();
    ctx.arc(px(x), py(y), r * sc, 0, TAU);
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) {
      ctx.strokeStyle = stroke; ctx.lineWidth = width;
      ctx.setLineDash(dash || []);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  function tag(x, y, text, color, dy = -16) {
    ctx.font = '500 11px "Geist Mono", ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    // kept inside the canvas, with a dark halo so it stays legible over points and spokes
    const half = ctx.measureText(text).width / 2 + 4;
    const X0 = Math.min(W - half, Math.max(half, px(x))), Y0 = py(y) + dy;
    ctx.lineJoin = 'round';
    ctx.lineWidth = 4;
    ctx.strokeStyle = rgba(HALO, 0.9);
    ctx.strokeText(text, X0, Y0);
    ctx.fillStyle = color;
    ctx.fillText(text, X0, Y0);
  }

  function centroid(x, y, color, scale = 1, label = null, labelColor = null) {
    if (scale <= 0.01) return;
    const X0 = px(x), Y0 = py(y);
    ctx.save();
    if (GLOW) { ctx.shadowColor = rgba(color, 0.9); ctx.shadowBlur = 14 * scale; }
    ctx.beginPath();
    ctx.arc(X0, Y0, 5 * scale, 0, TAU);
    ctx.fillStyle = rgba(color, 1);
    ctx.fill();
    ctx.restore();
    ctx.beginPath();
    ctx.arc(X0, Y0, 2 * scale, 0, TAU);
    ctx.fillStyle = GLOW ? 'rgba(255, 244, 236, 0.95)' : 'rgba(255, 255, 255, 0.95)';
    ctx.fill();
    if (label) tag(x, y, label, labelColor || rgba(INK, 0.85 * Math.min(1, scale)));
  }

  // every point with a spoke to the centroid it belongs to, in that cluster's hue;
  // `where(i)` returns the centroid as [x, y], or null while the point is unread
  function members(where, { alpha = 1, red = null, hueOf = null } = {}) {
    const colorOf = (i) => (red && red(i) ? SPUR : hueOf ? hueOf(i) : DUST);
    const groups = new Map();
    for (let i = 0; i < n; i++) {
      const c = where(i);
      if (!c) continue;
      const key = colorOf(i);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(i, c);
    }
    for (const [key, g] of groups) {
      ctx.beginPath();
      for (let a = 0; a < g.length; a += 2) {
        const i = g[a], c = g[a + 1];
        ctx.moveTo(px(X[i * 2]), py(X[i * 2 + 1]));
        ctx.lineTo(px(c[0]), py(c[1]));
      }
      ctx.strokeStyle = rgba(key, 0.2 * alpha);
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    for (let i = 0; i < n; i++) {
      const c = where(i);
      ctx.beginPath();
      ctx.arc(px(X[i * 2]), py(X[i * 2 + 1]), c ? 2.5 : 1.9, 0, TAU);
      ctx.fillStyle = c ? rgba(colorOf(i), 0.9 * alpha) : rgba(DUST, 0.22);
      ctx.fill();
    }
  }

  // a ring that bursts outward where a cluster has just opened
  function pop(x, y, age, color) {
    if (age < 0 || age > 0.6) return;
    const k = age / 0.6;
    circle(x, y, THETA * (0.25 + 0.75 * ease(k)), rgba(color, 0.7 * (1 - k)), null, null, 1.5);
  }

  const at = (C, k) => [C[k * 2], C[k * 2 + 1]];
  const mix = (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t)];

  // ----- step 1: pass 1 -----

  const S1 = { slow: 4, beat: 0.68, fastEnd: 4.4, dur: 5.8 };
  const fastStart1 = S1.slow * S1.beat;
  const visitTime1 = (j) => (j < S1.slow ? j * S1.beat + S1.beat * 0.62 : fastStart1 + ((j - S1.slow) / (n - S1.slow)) * (S1.fastEnd - fastStart1));

  function step1(t) {
    const { snaps, made } = m.p1;
    let mDone, j = -1, u = 0;
    if (t < fastStart1) { j = Math.floor(t / S1.beat); u = (t - j * S1.beat) / S1.beat; mDone = j; }
    else mDone = Math.min(n, S1.slow + Math.floor((n - S1.slow) * seg(t, fastStart1, S1.fastEnd)));
    const result = j >= 0 ? ease(seg(u, 0.62, 0.95)) : 0;
    const cur = j >= 0 ? tr.perm1[j] : -1;
    const isNew = j >= 0 && tr.labels1[j] >= made[j];
    const C = j >= 0 && u >= 0.62 ? snaps[j + 1] : snaps[mDone];
    const Cprev = j >= 0 ? snaps[j] : C;
    const K = j >= 0 && u >= 0.62 ? made[j + 1] : made[mDone];
    const pos = (k) => (j >= 0 && k < made[j] ? mix(at(Cprev, k), at(C, k), result) : at(C, k));

    members((i) => (i !== cur && m.vis1[i] < mDone ? pos(m.lab1[i]) : null), { hueOf: (i) => hue(m.lab1[i]) });
    for (let k = 0; k < K; k++) {
      const [x, y] = pos(k);
      const born = visitTime1(tr.labels1.indexOf(k));
      centroid(x, y, hue(k), 1 + 0.6 * Math.exp(-Math.max(0, t - born) / 0.18), nameOf(k));
      pop(x, y, t - born, hue(k));
    }

    let caption, read = mDone;
    if (j >= 0) {
      const x = X[cur * 2], y = X[cur * 2 + 1];
      const near = nearest(snaps[j], made[j], x, y);
      const grow = ease(seg(u, 0, 0.3));
      const fadeTheta = 1 - seg(u, 0.62, 0.9);
      circle(x, y, THETA * grow, rgba(INK, 0.5 * fadeTheta), rgba(INK, 0.035 * fadeTheta), [4, 4]);
      if (u < 0.3) circle(x, y, (16 - 13 * seg(u, 0, 0.3)) / sc, rgba(INK, 0.8));
      if (near.k >= 0 && u >= 0.3 && u < 0.62 + 0.3) {
        const [cx, cy] = at(snaps[j], near.k), within = near.d < THETA;
        const lc = within ? INK : ACCENT;
        line(x, y, cx, cy, rgba(lc, 0.75 * fadeTheta), 1.2, [3, 3]);
        ctx.font = '500 11px "Geist Mono", ui-monospace, monospace';
        ctx.fillStyle = rgba(lc, 0.95 * fadeTheta);
        ctx.textAlign = 'center';
        ctx.fillText(`${dist(near.d)} ${within ? '<' : '>'} θ`, px((x + cx) / 2), py((y + cy) / 2) - 8);
      }
      if (!isNew && u >= 0.62) {
        const [cx, cy] = pos(tr.labels1[j]);
        line(x, y, lerp(x, cx, result), lerp(y, cy, result), rgba(hue(tr.labels1[j]), 0.85), 1.4);
      }
      ctx.beginPath();
      ctx.arc(px(x), py(y), 3.2, 0, TAU);
      ctx.fillStyle = rgba(INK, 1);
      ctx.fill();
      read = j + (u >= 0.62 ? 1 : 0);

      const k = tr.labels1[j];
      if (near.k < 0) caption = `The first point has nothing to join, so it opens ${nameOf(k)}.`;
      else if (isNew) {
        const sameGroup = m.y[cur] === m.y[m.firstOf[near.k]];
        caption = `${dist(near.d)} from ${nameOf(near.k)}, farther than θ: it opens ${nameOf(k)}.` +
          (sameGroup ? ` Same group as ${nameOf(near.k)}, so this is a mistake.` : '');
      } else caption = `${dist(near.d)} from ${nameOf(k)}, within θ: it joins, and ${nameOf(k)} moves toward it.`;
    } else if (t < S1.fastEnd) caption = 'Every point is read once, in shuffled order. Early points decide where clusters start.';
    else caption = `Pass 1 ends with ${tr.K1} centroids for ${m.K} groups. ${tr.K1 > m.K ? `${word(tr.K1 - m.K)[0].toUpperCase() + word(tr.K1 - m.K).slice(1)} of them are mistakes.` : ''}`;
    return { caption, stats: `points read ${read} / ${n} · centroids ${K} · θ = ${THETA}` };
  }

  // ----- step 2: TS on the centroids finds the duplicates -----

  const S2 = { scanStart: 0.6, scanEnd: 3.1, dur: 4.6 };
  const isCand1 = (k) => tr.cand1.isCandidate[k] === 1;

  function step2(t) {
    const M1 = tr.M1, K1 = tr.K1;
    const scanned = Math.min(K1, Math.floor(seg(t, S2.scanStart, S2.scanEnd) * K1 + (t >= S2.scanStart ? 1 : 0)));
    const flagged = (k) => isCand1(k) && k < scanned;
    members((i) => at(M1, m.lab1[i]), { alpha: 0.75, red: (i) => flagged(m.lab1[i]), hueOf: (i) => hue(m.lab1[i]) });
    const ring = seg(t, 0, 0.5);
    for (let k = 0; k < K1; k++) circle(...at(M1, k), THETA, rgba(flagged(k) ? SPUR : hue(k), 0.4 * ring), null, [4, 5]);
    let last = -1;
    for (let k = 0; k < K1; k++) {
      const [x, y] = at(M1, k);
      if (flagged(k)) {
        last = k;
        const keep = tr.cand1.keeperOf[k];
        line(x, y, ...at(M1, keep), rgba(SPUR, 0.9), 1.5);
        centroid(x, y, SPUR, 1.15, `${nameOf(k)} · spurious`, rgba(SPUR, 1));
      } else centroid(x, y, hue(k), 1, nameOf(k));
    }
    const scanning = scanned - 1;
    if (t >= S2.scanStart && t < S2.scanEnd && scanning >= 0) circle(...at(M1, scanning), 12 / sc, rgba(INK, 0.9), null, null, 1.5);

    const count = tr.cand1.candidates.filter((k) => k < scanned).length;
    let caption;
    if (t < S2.scanStart || last < 0) caption = `Run Thetan Sequential again, on the ${K1} centroids alone, oldest first.`;
    else if (t < S2.scanEnd) {
      const keep = tr.cand1.keeperOf[last];
      const d = Math.hypot(M1[last * 2] - M1[keep * 2], M1[last * 2 + 1] - M1[keep * 2 + 1]);
      caption = `${nameOf(last)} is ${dist(d)} from the older ${nameOf(keep)}, within θ: the same cluster twice. ${nameOf(last)} is flagged spurious.`;
    } else caption = `${word(count)[0].toUpperCase() + word(count).slice(1)} of ${K1} centroids are flagged spurious (ζ₁).`;
    return { caption, stats: `centroids ${K1} · spurious ${count} · θ = ${THETA}` };
  }

  // ----- step 3: delete the spurious centroids, reassign, recompute -----

  const S3 = { fade: [0.3, 1.0], move: [0.8, 1.9], settle: [2.0, 3.0], dur: 4.4 };

  function step3(t) {
    const M1 = tr.M1, K1 = tr.K1;
    const gone = ease(seg(t, ...S3.fade)), swing = ease(seg(t, ...S3.move)), settle = ease(seg(t, ...S3.settle));
    const survPos = (k) => mix(at(M1, k), at(m.M2, k), settle);
    members((i) => {
      const k = m.lab1[i];
      return isCand1(k) ? mix(at(M1, k), survPos(m.lab2[i]), swing) : survPos(k);
    }, { red: (i) => isCand1(m.lab1[i]) && swing < 0.5, hueOf: (i) => hue(m.lab2[i]) });
    for (let k = 0; k < K1; k++) {
      if (isCand1(k)) {
        if (gone >= 1) continue;
        const [x, y] = at(M1, k);
        line(x, y, ...at(M1, tr.cand1.keeperOf[k]), rgba(SPUR, 0.9 * (1 - gone)), 1.5);
        centroid(x, y, SPUR, 1.15 * (1 - gone), gone < 0.5 ? `${nameOf(k)} · spurious` : null, rgba(SPUR, 1 - gone * 2));
      } else centroid(...survPos(k), hue(k), 1, nameOf(k));
    }
    const moved = m.lab1.reduce((a, k) => a + (isCand1(k) ? 1 : 0), 0);
    const names = tr.cand1.candidates.map(nameOf).join(' and ');
    const caption = t < S3.settle[0]
      ? `Delete ${names}. Their ${moved} points move to the nearest surviving centroid.`
      : `Recompute each centroid as the mean of its points: ${tr.K2} clean centroids, M⋆₂.`;
    return { caption, stats: `centroids ${t < S3.fade[1] ? K1 : tr.K2} · spurious removed ${tr.cand1.candidates.length}` };
  }

  // ----- step 4: prepend M⋆₂, then a second pass in a fresh order -----

  const S4 = { clear: [0, 0.5], claim: [0.4, 1.5], read: [1.6, 4.6], dur: 6.0 };
  const readVisits4 = (t) => Math.min(n, Math.floor(n * Math.pow(seg(t, ...S4.read), 1.35)));
  const visitTime4 = (j) => S4.read[0] + Math.pow(j / n, 1 / 1.35) * (S4.read[1] - S4.read[0]);

  function step4(t) {
    const { snaps, made } = m.p2;
    const mDone = t < S4.read[0] ? 0 : readVisits4(t);
    const C = snaps[mDone], K = made[mDone];
    const clear = seg(t, ...S4.clear), claim = ease(seg(t, ...S4.claim));
    // the end of step 3 fades out, then pass 2 reads the points again
    if (clear < 1) members((i) => at(m.M2, m.lab2[i]), { alpha: 1 - clear, hueOf: (i) => hue(m.lab2[i]) });
    if (t >= S4.read[0]) members((i) => (m.vis2[i] < mDone ? at(C, m.lab3[i]) : null), { hueOf: (i) => hue(m.name3[m.lab3[i]]) });
    else if (clear >= 1) members(() => null);
    for (let c = 0; c < tr.priorK; c++) {
      circle(...at(C, c), THETA * claim, rgba(hue(m.name3[c]), 0.6), rgba(hue(m.name3[c]), 0.08), null, 1.2);
    }
    let newest = null;
    for (let c = 0; c < K; c++) {
      const [x, y] = at(C, c);
      if (c >= tr.priorK) {
        const born = visitTime4(tr.labels3.indexOf(c));
        newest = { c, born };
        centroid(x, y, hue(m.name3[c]), 1 + 0.6 * Math.exp(-Math.max(0, t - born) / 0.18), `${nameOf(m.name3[c])} · new`);
        pop(x, y, t - born, hue(m.name3[c]));
      } else centroid(x, y, hue(m.name3[c]), 1, nameOf(m.name3[c]));
    }
    let caption;
    if (t < S4.claim[1] - 0.3) caption = `M⋆₂ goes first. Each of the ${tr.priorK} centroids claims its ground before any point is read.`;
    else if (newest && t - newest.born < 1.7) caption = `This point lands outside every territory, so pass 2 still opens ${nameOf(m.name3[newest.c])}.`;
    else if (t < S4.read[1]) caption = 'Points arrive in a fresh order and join the centroids that are already there.';
    else caption = `Pass 2 ends with ${tr.K3} centroids: the ${tr.priorK} that went first${tr.K3 > tr.priorK ? `, plus ${word(tr.K3 - tr.priorK)} opened by outlying points` : ''}.`;
    return { caption, stats: `points read ${mDone} / ${n} · centroids ${K} · θ = ${THETA}` };
  }

  // ----- step 5: second cleanup, then the final clustering -----

  const S5 = { ring: [0, 0.5], flag: 0.6, fade: [1.5, 2.1], swing: [1.5, 2.3], settle: [2.3, 3.1], dur: 5.8 };
  const isCand2 = (c) => tr.cand2.isCandidate[c] === 1;
  const finalPos = (c) => at(m.res.centroids, m.finalOf[c]);

  function step5(t) {
    const M3 = tr.M3, K3 = tr.K3;
    const ring = seg(t, ...S5.ring) * (1 - seg(t, ...S5.settle));
    const flagged = t >= S5.flag;
    const gone = ease(seg(t, ...S5.fade)), swing = ease(seg(t, ...S5.swing)), settle = ease(seg(t, ...S5.settle));
    const survPos = (c) => mix(at(M3, c), finalPos(c), settle);
    members((i) => {
      const c = m.lab3[i];
      return isCand2(c) ? mix(at(M3, c), survPos(m.lab4[i]), swing) : survPos(c);
    }, {
      red: (i) => flagged && isCand2(m.lab3[i]) && swing < 0.5,
      hueOf: (i) => hue(m.name3[isCand2(m.lab3[i]) && swing >= 0.5 ? m.lab4[i] : m.lab3[i]]),
    });
    for (let c = 0; c < K3; c++) if (!isCand2(c) || gone < 1) circle(...at(M3, c), THETA, rgba(isCand2(c) && flagged ? SPUR : hue(m.name3[c]), 0.4 * ring), null, [4, 5]);
    for (let c = 0; c < K3; c++) {
      if (isCand2(c) && flagged) {
        if (gone >= 1) continue;
        const [x, y] = at(M3, c);
        line(x, y, ...at(M3, tr.cand2.keeperOf[c]), rgba(SPUR, 0.9 * (1 - gone)), 1.5);
        centroid(x, y, SPUR, 1.15 * (1 - gone), gone < 0.5 ? `${nameOf(m.name3[c])} · spurious` : null, rgba(SPUR, 1 - gone * 2));
      } else if (!isCand2(c) || !flagged) centroid(...(isCand2(c) ? at(M3, c) : survPos(c)), hue(m.name3[c]), 1, nameOf(m.name3[c]));
    }
    const done = seg(t, S5.settle[1], S5.settle[1] + 0.5);
    if (done > 0) {
      ctx.globalAlpha = done;
      ctx.textAlign = 'left';
      ctx.fillStyle = rgba(INK, 1);
      ctx.font = 'italic 400 34px "Instrument Serif", Georgia, serif';
      ctx.fillText(`K = ${m.res.K}`, 16, H - 34);
      ctx.font = '500 11px "Geist Mono", ui-monospace, monospace';
      ctx.fillStyle = rgba(ACCENT, 1);
      ctx.fillText('FOUND, NOT GIVEN', 18, H - 16);
      ctx.globalAlpha = 1;
    }
    const spur = tr.cand2.candidates;
    let caption;
    if (t < S5.flag) caption = `One last cleanup: Thetan Sequential on the ${K3} centroids.`;
    else if (t < S5.settle[0]) {
      const c = spur[0], keep = tr.cand2.keeperOf[c];
      caption = spur.length
        ? `${nameOf(m.name3[c])} sits within θ of ${nameOf(m.name3[keep])}: flagged, deleted, and its points move over.`
        : 'No centroid sits within θ of another: nothing to remove.';
    } else if (t < S5.settle[1]) caption = 'Undo the shuffle and recompute every centroid from its points.';
    else caption = `Done: ${m.res.K} clusters, and TB was never told the number.` +
      (m.ari > 0.9999 ? ' Every point is in its true group.' : ` ARI against the truth: ${m.ari.toFixed(3)}.`);
    return { caption, stats: `centroids ${t < S5.fade[1] ? K3 : m.res.K} · spurious ${spur.length} · θ = ${THETA}` };
  }

  const STEPS = {
    1: { draw: step1, dur: S1.dur },
    2: { draw: step2, dur: S2.dur },
    3: { draw: step3, dur: S3.dur },
    4: { draw: step4, dur: S4.dur },
    5: { draw: step5, dur: S5.dur },
  };

  // ----- loop: the active step repeats while the figure is on screen -----

  let step = 1, start = performance.now(), visible = false, raf = 0, lastCaption = '', frozen = null;
  function draw(now = performance.now()) {
    if (!W) return;
    const s = STEPS[step];
    const t = frozen ?? (reduced ? s.dur - 0.01 : ((now - start) / 1000) % s.dur);
    backdrop();
    const info = s.draw(t);
    if (info.caption !== lastCaption) { capEl.textContent = info.caption; lastCaption = info.caption; }
    statEl.textContent = info.stats;
  }
  function loop(now) {
    raf = 0;
    draw(now);
    if (visible && !reduced) raf = requestAnimationFrame(loop);
  }
  const kick = () => { if (!raf && visible) raf = requestAnimationFrame(loop); };

  new IntersectionObserver(([e]) => { visible = e.isIntersecting; kick(); }).observe(canvas);
  new ResizeObserver(resize).observe(canvas);
  resize();

  return {
    setStep(s) {
      if (s === step || !STEPS[s]) return;
      step = s;
      start = performance.now();
      draw();
      kick();
    },
    // debugging: hold one step at a chosen time (null resumes the loop)
    render(s, t) { step = s; frozen = t; draw(); },
    get step() { return step; },
    setTheme(theme) { applyColors(theme); draw(); },
  };
}
