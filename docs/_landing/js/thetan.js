// Thetan in the browser: a faithful port of TS (Alg. 1) and TB (Alg. 2)
// from thetan/core.pyx and thetan/berserker.py, plus the trace the
// visualisation replays. Data are flat Float64Arrays of n * d values.

// ---------- randomness ----------

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function randn(rng) {
  let u = 0;
  while (u === 0) u = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

export function permutation(n, rng) {
  const p = new Int32Array(n);
  for (let i = 0; i < n; i++) p[i] = i;
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const t = p[i]; p[i] = p[j]; p[j] = t;
  }
  return p;
}

export function permute(X, d, perm) {
  const out = new Float64Array(perm.length * d);
  for (let i = 0; i < perm.length; i++) {
    const s = perm[i] * d, t = i * d;
    for (let k = 0; k < d; k++) out[t + k] = X[s + k];
  }
  return out;
}

function dist(A, a, B, b, d) {
  let s = 0;
  for (let k = 0; k < d; k++) {
    const v = A[a + k] - B[b + k];
    s += v * v;
  }
  return Math.sqrt(s);
}

// ---------- Algorithm 1: Thetan Sequential ----------
// `prior` (optional) holds centroids that are read before the data, as
// sequential_prelim does for the second pass of TB: they open clusters
// (or merge among themselves) but receive no labels.

export function sequential(X, n, d, theta, prior = null, priorCount = 0) {
  let cap = 64;
  let sums = new Float64Array(cap * d);
  let cents = new Float64Array(cap * d);
  let sizes = new Float64Array(cap);
  const labels = new Int32Array(n);
  let K = 0;

  const grow = () => {
    cap *= 2;
    const s2 = new Float64Array(cap * d); s2.set(sums); sums = s2;
    const c2 = new Float64Array(cap * d); c2.set(cents); cents = c2;
    const z2 = new Float64Array(cap); z2.set(sizes); sizes = z2;
  };

  const visit = (src, off) => {
    let min = Infinity, arg = -1;
    for (let k = 0; k < K; k++) {
      const v = dist(src, off, cents, k * d, d);
      if (v < min) { min = v; arg = k; }
    }
    if (arg >= 0 && min < theta) {
      sizes[arg] += 1;
      const o = arg * d;
      for (let j = 0; j < d; j++) {
        sums[o + j] += src[off + j];
        cents[o + j] = sums[o + j] / sizes[arg];
      }
      return arg;
    }
    if (K === cap) grow();
    const o = K * d;
    for (let j = 0; j < d; j++) { sums[o + j] = src[off + j]; cents[o + j] = src[off + j]; }
    sizes[K] = 1;
    return K++;
  };

  const priorLabels = new Int32Array(priorCount);
  for (let i = 0; i < priorCount; i++) priorLabels[i] = visit(prior, i * d);
  const priorK = K;
  for (let i = 0; i < n; i++) labels[i] = visit(X, i * d);

  return { centroids: cents.slice(0, K * d), K, labels, priorK, priorLabels };
}

// ---------- TB helpers (candidate_labels_c, reassign, centroids_from_labels) ----------

// cLabels: label of every centroid after TS on the centroids.
// Within each group the lowest index survives; the rest are candidates.
export function candidateLabels(cLabels, K, initialLabels) {
  const keeper = new Int32Array(K).fill(-1);
  for (let i = 0; i < K; i++) {
    const g = cLabels[i];
    if (keeper[g] === -1 || i < keeper[g]) keeper[g] = i;
  }
  const isCandidate = new Uint8Array(K);
  const keeperOf = new Int32Array(K);
  const candidates = [];
  for (let i = 0; i < K; i++) {
    keeperOf[i] = keeper[cLabels[i]];
    if (keeperOf[i] !== i) { isCandidate[i] = 1; candidates.push(i); }
  }
  // compact index of every surviving centroid
  const newIndex = new Int32Array(K).fill(-1);
  let s = 0;
  for (let i = 0; i < K; i++) if (!isCandidate[i]) newIndex[i] = s++;

  const n = initialLabels.length;
  const updated = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const l = initialLabels[i];
    updated[i] = isCandidate[l] ? -1 : newIndex[l];
  }
  return { candidates, isCandidate, keeperOf, newIndex, survivors: s, updated };
}

export function deleteRows(C, K, d, isCandidate) {
  const out = [];
  for (let i = 0; i < K; i++) if (!isCandidate[i]) for (let j = 0; j < d; j++) out.push(C[i * d + j]);
  return Float64Array.from(out);
}

export function reassign(X, n, d, C, K, labels) {
  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    if (labels[i] >= 0) { out[i] = labels[i]; continue; }
    let min = Infinity, arg = 0;
    for (let k = 0; k < K; k++) {
      const v = dist(X, i * d, C, k * d, d);
      if (v < min) { min = v; arg = k; }
    }
    out[i] = arg;
  }
  return out;
}

export function centroidsFromLabels(X, n, d, labels, K) {
  const C = new Float64Array(K * d);
  const cnt = new Float64Array(K);
  for (let i = 0; i < n; i++) {
    const l = labels[i];
    cnt[l] += 1;
    for (let j = 0; j < d; j++) C[l * d + j] += X[i * d + j];
  }
  for (let k = 0; k < K; k++) if (cnt[k] > 0) for (let j = 0; j < d; j++) C[k * d + j] /= cnt[k];
  return { C, cnt };
}

// Drop clusters that ended up with no points and renumber the rest.
function compact(labels, K) {
  const used = new Int32Array(K).fill(-1);
  let m = 0;
  for (let i = 0; i < labels.length; i++) if (used[labels[i]] === -1) used[labels[i]] = 0;
  for (let k = 0; k < K; k++) if (used[k] === 0) used[k] = m++;
  const out = new Int32Array(labels.length);
  for (let i = 0; i < labels.length; i++) out[i] = used[labels[i]];
  return { labels: out, K: m, map: used };
}

// ---------- Algorithm 2: Thetan Berserker ----------

export function berserker(X, n, d, theta, rng = Math.random) {
  // pass 1
  const perm1 = permutation(n, rng);
  const X1 = permute(X, d, perm1);
  const p1 = sequential(X1, n, d, theta);
  const M1 = p1.centroids, K1 = p1.K;

  // cleanup 1: TS on the centroids, delete spurious ones, reassign
  const c1 = sequential(M1, K1, d, theta);
  const cand1 = candidateLabels(c1.labels, K1, p1.labels);
  const M2 = deleteRows(M1, K1, d, cand1.isCandidate);
  const labels2 = reassign(X1, n, d, M2, cand1.survivors, cand1.updated);
  const M2star = centroidsFromLabels(X1, n, d, labels2, cand1.survivors).C;
  const K2 = cand1.survivors;

  // pass 2: prepend the cleaned centroids to a fresh permutation
  const perm2 = permutation(n, rng);
  const X2 = permute(X, d, perm2);
  const p2 = sequential(X2, n, d, theta, M2star, K2);
  const M3 = p2.centroids, K3 = p2.K;

  // cleanup 2
  const c2 = sequential(M3, K3, d, theta);
  const cand2 = candidateLabels(c2.labels, K3, p2.labels);
  const M3b = deleteRows(M3, K3, d, cand2.isCandidate);
  const labels4 = reassign(X2, n, d, M3b, cand2.survivors, cand2.updated);

  // invert the permutation and compute centroids from labels
  const unperm = new Int32Array(n);
  for (let i = 0; i < n; i++) unperm[perm2[i]] = labels4[i];
  const fin = compact(unperm, cand2.survivors);
  const final = centroidsFromLabels(X, n, d, fin.labels, fin.K);

  return {
    centroids: final.C, K: fin.K, labels: fin.labels,
    trace: {
      perm1, labels1: p1.labels, K1, M1,
      cand1, labels2, K2, M2star,
      perm2, labels3: p2.labels, K3, priorK: p2.priorK, priorLabels: p2.priorLabels, M3,
      cand2, labels4, finalMap: fin.map,
    },
  };
}

// TS on a fresh permutation, labels returned in the original order.
export function sequentialShuffled(X, n, d, theta, rng = Math.random) {
  const perm = permutation(n, rng);
  const r = sequential(permute(X, d, perm), n, d, theta);
  const labels = new Int32Array(n);
  for (let i = 0; i < n; i++) labels[perm[i]] = r.labels[i];
  return { centroids: r.centroids, K: r.K, labels };
}

// ---------- metrics ----------

function comb2(x) { return (x * (x - 1)) / 2; }

export function adjustedRandIndex(a, b) {
  const n = a.length;
  const table = new Map(), ra = new Map(), rb = new Map();
  for (let i = 0; i < n; i++) {
    const key = a[i] * 1048576 + b[i];
    table.set(key, (table.get(key) || 0) + 1);
    ra.set(a[i], (ra.get(a[i]) || 0) + 1);
    rb.set(b[i], (rb.get(b[i]) || 0) + 1);
  }
  let sij = 0, sa = 0, sb = 0;
  for (const v of table.values()) sij += comb2(v);
  for (const v of ra.values()) sa += comb2(v);
  for (const v of rb.values()) sb += comb2(v);
  const expected = (sa * sb) / comb2(n);
  const max = (sa + sb) / 2;
  return max === expected ? 1 : (sij - expected) / (max - expected);
}

// fraction of true centres with no estimated centroid within `tol` (#MC)
export function missedCentroids(trueC, Kt, estC, Ke, d, tol) {
  let missed = 0;
  for (let t = 0; t < Kt; t++) {
    let hit = false;
    for (let k = 0; k < Ke && !hit; k++) if (dist(trueC, t * d, estC, k * d, d) <= tol) hit = true;
    if (!hit) missed++;
  }
  return missed;
}

// ---------- data ----------

// The paper's simulation: an A x B grid of unit Gaussians, ell apart.
export function gridGaussians(A, B, ell, per, rng, sigma = 1) {
  const K = A * B, n = K * per, d = 2;
  const X = new Float64Array(n * d), y = new Int32Array(n), centers = new Float64Array(K * d);
  let i = 0;
  for (let a = 0; a < A; a++) for (let b = 0; b < B; b++) {
    const k = a * B + b;
    const cx = (a - (A - 1) / 2) * ell, cy = (b - (B - 1) / 2) * ell;
    centers[k * 2] = cx; centers[k * 2 + 1] = cy;
    for (let s = 0; s < per; s++, i++) {
      X[i * 2] = cx + sigma * randn(rng);
      X[i * 2 + 1] = cy + sigma * randn(rng);
      y[i] = k;
    }
  }
  return { X, y, n, d, K, centers };
}

// Unit Gaussians in 3D whose centres keep at least `minSep` apart.
export function blobs3D(K, minSep, radius, perMin, perMax, rng, sigma = 1) {
  const centers = [];
  let guard = 0;
  while (centers.length < K && guard++ < 20000) {
    const c = [0, 0, 0].map(() => (rng() * 2 - 1) * radius);
    c[1] *= 0.55; // flatter, galaxy-like volume
    if (Math.hypot(c[0], c[1] / 0.55, c[2]) > radius) continue;
    if (centers.every((o) => Math.hypot(o[0] - c[0], o[1] - c[1], o[2] - c[2]) >= minSep)) centers.push(c);
  }
  const sizes = centers.map(() => Math.round(perMin + rng() * (perMax - perMin)));
  const n = sizes.reduce((a, b) => a + b, 0), d = 3;
  const X = new Float64Array(n * d), y = new Int32Array(n);
  let i = 0;
  centers.forEach((c, k) => {
    for (let s = 0; s < sizes[k]; s++, i++) {
      for (let j = 0; j < 3; j++) X[i * 3 + j] = c[j] + sigma * randn(rng);
      y[i] = k;
    }
  });
  return { X, y, n, d, K: centers.length, centers: Float64Array.from(centers.flat()) };
}
