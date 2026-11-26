import { mountExplainer } from './explainer.js';
import { mountDuel } from './ordering.js';
import { mountPlayground } from './playground.js';
import { logBars, pThetaChart } from './charts.js';
import { currentTheme, onThemeChange, toggleTheme } from './theme.js';

const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

// ?debug exposes the figure so frames can be stepped from the console
const debug = new URLSearchParams(location.search).has('debug');

// ---------- the step-by-step figure ----------

const explainer = mountExplainer($('.fig-card'), { reduced });
if (debug) window.__explainer = explainer;
const figSteps = $$('.fig-steps button');
figSteps.forEach((b) => b.addEventListener('click', () => {
  $(`.step[data-step="${b.dataset.go}"]`).scrollIntoView({ behavior: reduced ? 'auto' : 'smooth', block: 'center' });
}));
const algoToggle = $('[data-algo-toggle]');
algoToggle.addEventListener('click', () => {
  const full = algoToggle.closest('.fig-algo').classList.toggle('full');
  algoToggle.setAttribute('aria-expanded', String(full));
  algoToggle.textContent = full ? 'Show this step only' : 'Show all 19 lines';
});

// ---------- scroll: which scene, which story step ----------

const sections = $$('[data-scene]');
const steps = $$('.step');
const algoLines = $$('.algo-lines li');
const STEP_LINES = { 1: [3, 4, 5], 2: [6, 7], 3: [8, 9, 10], 4: [11, 12, 13], 5: [14, 15, 16, 17, 18, 19] };
const navLinks = $$('.nav-links a');

function onScroll() {
  const mid = window.innerHeight * 0.5;
  // on narrow screens the figure pins to the top, so steps are read lower down
  const stepLine = window.innerWidth <= 900 ? window.innerHeight * 0.72 : mid;
  $('.nav').classList.toggle('scrolled', window.scrollY > 40);

  let mode = 'ambient', sectionId = null;
  for (const s of sections) {
    const r = s.getBoundingClientRect();
    if (r.top <= mid && r.bottom > mid) { mode = s.dataset.scene; sectionId = s.id; break; }
  }
  if (window.scrollY < 10) { mode = 'hero'; sectionId = 'top'; }
  navLinks.forEach((a) => a.classList.toggle('active', a.getAttribute('href') === `#${sectionId}`));

  // the step nearest the reading line; the first one until the section is reached
  let step = 1;
  if (mode === 'story') {
    let best = Infinity;
    for (const s of steps) {
      const r = s.getBoundingClientRect();
      const d = Math.abs(r.top + r.height / 2 - stepLine);
      if (d < best) { best = d; step = +s.dataset.step; }
    }
  }
  steps.forEach((s) => s.classList.toggle('active', +s.dataset.step === step));
  figSteps.forEach((b) => (+b.dataset.go === step ? b.setAttribute('aria-current', 'step') : b.removeAttribute('aria-current')));
  const on = new Set(STEP_LINES[step]);
  algoLines.forEach((li) => li.classList.toggle('on', on.has(+li.dataset.l)));
  explainer.setStep(step);

  document.body.dataset.mode = mode;
}

let scrollQueued = false;
const queueScroll = () => {
  if (scrollQueued) return;
  scrollQueued = true;
  requestAnimationFrame(() => { scrollQueued = false; onScroll(); });
};
window.addEventListener('scroll', queueScroll, { passive: true });
window.addEventListener('resize', queueScroll);
onScroll();
if (debug) window.__sync = onScroll;

// ---------- ordering duel ----------

const duel = mountDuel($('[data-duel]'), { reduced });

// ---------- θ playground ----------

const playground = mountPlayground($('#theta'));

// ---------- theme ----------

const themeBtn = $('[data-theme-toggle]');
function labelThemeButton(theme) {
  themeBtn.setAttribute('aria-label', theme === 'light' ? 'Switch to dark theme' : 'Switch to light theme');
}
themeBtn.addEventListener('click', toggleTheme);
labelThemeButton(currentTheme());
onThemeChange((theme) => {
  labelThemeButton(theme);
  explainer.setTheme(theme);
  duel.redraw();
  playground.redraw();
});

// ---------- charts ----------

logBars($('[data-chart="runtime"]'), [
  { label: 'Thetan Berserker', value: 2.7, display: '2.7 s', highlight: true, tip: '<b>Thetan Berserker</b><br>2.7 s · K inferred (300.4 on average)' },
  { label: 'Mini-Batch KMeans', value: 15, display: '15 s', tip: '<b>Mini-Batch KMeans</b><br>15 s · given K = 300' },
  { label: 'KMeans', value: 27, display: '27 s', tip: '<b>KMeans</b><br>27 s · given K = 300' },
  { label: 'HDBSCAN', value: 4980, display: '83 min', tip: '<b>HDBSCAN</b><br>83 min (4,980 s) · min samples 40' },
], { domain: [1, 10000], ticks: [1, 10, 100, 1000, 10000], tickFmt: (t) => `${t.toLocaleString('en-US')} s` });

logBars($('[data-chart="memory"]'), [
  { label: 'Thetan Berserker', value: 77, display: '77 MB', highlight: true, tip: '<b>Thetan Berserker</b><br>77 MB peak' },
  { label: 'Mini-Batch KMeans', value: 32, display: '32 MB', tip: '<b>Mini-Batch KMeans</b><br>32 MB peak · given K = 300' },
  { label: 'KMeans', value: 214, display: '214 MB', tip: '<b>KMeans</b><br>214 MB peak · given K = 300' },
  { label: 'HDBSCAN', value: 1200, display: '1.2 GB', tip: '<b>HDBSCAN</b><br>1.2 GB peak' },
], { domain: [10, 2000], ticks: [10, 100, 1000], tickFmt: (t) => (t >= 1000 ? '1 GB' : `${t} MB`) });

pThetaChart($('[data-chart="ptheta"]'));

// ---------- small interactions ----------

const toast = $('.toast');
let toastTimer;
function say(msg) {
  toast.textContent = msg;
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 2400);
}
$$('[data-soon]').forEach((a) => a.addEventListener('click', (e) => { e.preventDefault(); say(a.dataset.soon); }));
window.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); location.href = $('[data-search]').href; }
});

$$('.tabs [role="tab"]').forEach((tab) => tab.addEventListener('click', () => {
  const card = tab.closest('.code-card');
  $$('[role="tab"]', card).forEach((t) => t.setAttribute('aria-selected', String(t === tab)));
  $$('pre[data-panel]', card).forEach((p) => (p.hidden = p.dataset.panel !== tab.dataset.tab));
}));
$$('[data-copy]').forEach((btn) => btn.addEventListener('click', async () => {
  const card = btn.closest('.code-card');
  const pre = $$('pre', card).find((p) => !p.hidden);
  try {
    await navigator.clipboard.writeText(pre.innerText);
    btn.textContent = 'Copied';
  } catch {
    btn.textContent = 'Select & copy';
  }
  setTimeout(() => (btn.textContent = 'Copy'), 1600);
}));

// reveal on scroll
const revealables = $$('.section-head, .thm, .chart-card, .stat, .fam, .doc, .code-card, .duel, .play-map, .play-panel');
revealables.forEach((e) => e.classList.add('reveal'));
const io = new IntersectionObserver((entries) => entries.forEach((en) => {
  if (en.isIntersecting) { en.target.classList.add('in'); io.unobserve(en.target); }
}), { rootMargin: '0px 0px -8% 0px' });
revealables.forEach((e) => io.observe(e));
