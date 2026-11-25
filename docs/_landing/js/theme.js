// Light and dark themes, and the colours clusters wear in each.
// The page follows the system setting until the viewer picks a theme.

const KEY = 'thetan-theme';
const root = document.documentElement;
const prefersLight = window.matchMedia('(prefers-color-scheme: light)');
const listeners = new Set();

export function currentTheme() {
  const t = root.dataset.theme;
  return t === 'light' || t === 'dark' ? t : prefersLight.matches ? 'light' : 'dark';
}

export function onThemeChange(fn) { listeners.add(fn); }

// the browser's own chrome follows the page
function syncMeta(t) {
  document.querySelector('meta[name="theme-color"]')?.setAttribute('content', t === 'light' ? '#f6f5f1' : '#04050a');
}
syncMeta(currentTheme());

function emit() {
  const t = currentTheme();
  syncMeta(t);
  listeners.forEach((fn) => fn(t));
}

export function setTheme(t) {
  root.dataset.theme = t;
  try { localStorage.setItem(KEY, t); } catch { /* storage blocked: the choice lasts this visit */ }
  emit();
}

export function toggleTheme() { setTheme(currentTheme() === 'light' ? 'dark' : 'light'); }

prefersLight.addEventListener('change', () => { if (!root.dataset.theme) emit(); });

// ---------- cluster colours ----------
// Blue, aqua, yellow, green and violet from the validated categorical palette:
// red is reserved for spurious centroids and orange is the accent. Each theme
// has its own steps (dark aqua re-stepped to clear the dark checks).

export const CLUSTER = {
  light: ['#2a78d6', '#1baf7a', '#eda100', '#008300', '#6250d6'],
  dark: ['#3987e5', '#12a88a', '#c98500', '#008300', '#9085e9'],
};
export const SPUR = { light: '#d92d3a', dark: '#ff3d6e' };
export const ACCENT = { light: '#c2470f', dark: '#ff7a3d' };

// How alike two hues look (1 = same hue). Blue and violet fail the colour-vision
// checks in both themes; yellow and green sit in the warning band on dark.
const SIM = Array.from({ length: 5 }, (_, a) => Array.from({ length: 5 }, (_, b) => (a === b ? 1 : 0)));
SIM[0][4] = SIM[4][0] = 0.8;
SIM[2][3] = SIM[3][2] = 0.5;

// Pick the hue whose closest look-alike sits farthest away, so neighbouring
// clusters never share a colour. `others` lists [position, colour] pairs.
export function pickColor(pos, others, d) {
  let best = 0, bestScore = -1;
  for (let c = 0; c < 5; c++) {
    let score = Infinity;
    for (const [p, col] of others) {
      const s = SIM[c][col];
      if (!s) continue;
      let dd = 0;
      for (let k = 0; k < d; k++) dd += (pos[k] - p[k]) ** 2;
      score = Math.min(score, Math.sqrt(dd) / s);
    }
    if (score > bestScore) { bestScore = score; best = c; }
  }
  return best;
}

// Colours for K centroids (flat array, d per row), in the order given.
export function assignColors(C, K, d, order = null) {
  const out = new Uint8Array(K), placed = [];
  for (const i of order || Array.from({ length: K }, (_, k) => k)) {
    const pos = Array.from({ length: d }, (_, k) => C[i * d + k]);
    out[i] = pickColor(pos, placed, d);
    placed.push([pos, out[i]]);
  }
  return out;
}

export const rgbOf = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)).join(', ');
