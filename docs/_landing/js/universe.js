// The 3D canvas behind the hero: a live stream of points that Thetan Sequential
// reads one by one. Each lands inside a θ-sphere or opens a new cluster, and a
// TB-style cleanup merges duplicate centroids every second. Below the hero only
// a faint starfield remains (dark theme).

import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import * as T from './thetan.js';
import { CLUSTER, SPUR, currentTheme, pickColor } from './theme.js';

const MAX_CENTROIDS = 2048;
const POINT_SIZE = 0.2;

// ---------- colour ----------
// Every cluster wears one hue, shared by its points and its centroid; red marks
// a spurious centroid. Dark glows additively; light draws ink on paper.

const PAL = { light: CLUSTER.light.map((h) => new THREE.Color(h)), dark: CLUSTER.dark.map((h) => new THREE.Color(h)) };
const SPUR_C = { light: new THREE.Color(SPUR.light), dark: new THREE.Color(SPUR.dark) };
const LOOK = {
  dark: {
    clear: 0x04050a, blending: THREE.AdditiveBlending, bloom: true, stars: true,
    point: 0.55, flash: 2.2, size: 1, core: 2.2,
    flight: new THREE.Color(0.42, 0.5, 0.72), flightAlpha: 1,
  },
  light: {
    clear: 0xf6f5f1, blending: THREE.NormalBlending, bloom: false, stars: false,
    point: 1, flash: 1, size: 1.2, core: 0.85,
    flight: new THREE.Color(0.16, 0.18, 0.24), flightAlpha: 0.55,
  },
};

const clamp01 = (t) => Math.min(1, Math.max(0, t));

// ---------- the hero stream ----------

const STREAM = {
  theta: 3.8,
  sigma: 1,
  rate: 150,           // points per second
  life: 13,            // seconds a point stays once read
  travel: [2.1, 3.0],  // seconds in flight
  trail: 0.12,         // seconds of path each streak covers
  sources: 7,          // Gaussians feeding the stream at any moment
  swap: 6,             // seconds between one Gaussian drying up and a new one appearing
  zone: [17, 10, 10],  // half-extents of the region the Gaussians live in
  minSep: 8,
  cleanup: 1,          // seconds between TB-style cleanups that merge duplicate centroids
};
const MAX_SETTLED = 4096;
const MAX_FLYING = 1024;
const MAX_SEGMENTS = MAX_FLYING + 256; // streaks, plus short spokes to centroids

function bezier(p, u, out) {
  const v = 1 - u, a = v * v * v, b = 3 * v * v * u, c = 3 * v * u * u, d = u * u * u;
  for (let k = 0; k < 3; k++) out[k] = a * p.from[k] + b * p.c1[k] + c * p.c2[k] + d * p.to[k];
  return out;
}
// fast as it leaves the left edge, settling as it reaches its cluster
const flightU = (t, dur) => 1 - Math.pow(1 - clamp01(t / dur), 1.7);

class Stream {
  constructor(rng) {
    this.rng = rng;
    this.time = 0;
    this.startX = -90; // updated by the camera to sit just off the left edge
    this.sources = [];
    this.flying = [];
    this.spokes = [];
    this.merges = [];
    this.centroids = new Map(); // id -> { id, sum, n, born, pos }, in creation order
    this.nextId = 0;
    this.spawnDebt = 0;
    this.swapClock = 0;
    this.cleanClock = 0;
    // points already read: a ring in arrival order, so expiry is first in, first out
    this.pos = new Float32Array(MAX_SETTLED * 3);
    this.born = new Float32Array(MAX_SETTLED);
    this.label = new Int32Array(MAX_SETTLED);
    this.head = 0;
    this.tail = 0;
    for (let s = 0; s < STREAM.sources; s++) this._addSource();
  }

  _addSource() {
    const { zone, minSep } = STREAM, rng = this.rng;
    for (let tries = 0; tries < 300; tries++) {
      const c = zone.map((h) => (rng() * 2 - 1) * h);
      if ((c[0] / zone[0]) ** 2 + (c[1] / zone[1]) ** 2 + (c[2] / zone[2]) ** 2 > 1) continue;
      if (this.sources.every((s) => Math.hypot(s.c[0] - c[0], s.c[1] - c[1], s.c[2] - c[2]) >= minSep)) {
        this.sources.push({ c, active: true, retired: 0 });
        return true;
      }
    }
    return false;
  }

  _spawn() {
    const active = this.sources.filter((s) => s.active);
    const src = active[Math.floor(this.rng() * active.length)];
    const g = () => T.randn(this.rng);
    const to = src.c.map((v) => v + STREAM.sigma * g());
    const from = [this.startX, 2.5 * g(), 5 * g()];
    const span = to[0] - from[0];
    const c1 = [from[0] + span * 0.4, from[1] + 3 * g(), from[2] + 3 * g()];
    const c2 = [to[0] - 14, from[1] + (to[1] - from[1]) * 0.7, from[2] + (to[2] - from[2]) * 0.7];
    const [lo, hi] = STREAM.travel;
    this.flying.push({ from, c1, c2, to, t: 0, dur: lo + (hi - lo) * this.rng() });
  }

  _arrive(p) {
    if (this.tail - this.head >= MAX_SETTLED) this._expireOldest();
    const idx = this.tail % MAX_SETTLED;
    this.tail++;
    this.pos[idx * 3] = p.to[0]; this.pos[idx * 3 + 1] = p.to[1]; this.pos[idx * 3 + 2] = p.to[2];
    this.born[idx] = this.time;
    // Thetan Sequential: join the nearest centroid closer than θ, else open a new one
    let best = Infinity, arg = null;
    for (const c of this.centroids.values()) {
      const d = Math.hypot(c.pos.x - p.to[0], c.pos.y - p.to[1], c.pos.z - p.to[2]);
      if (d < best) { best = d; arg = c; }
    }
    if (!arg || best >= STREAM.theta) {
      // a new cluster takes the hue least like its neighbours'
      const near = [...this.centroids.values()].map((c) => [[c.pos.x, c.pos.y, c.pos.z], c.col]);
      arg = { id: this.nextId++, sum: [0, 0, 0], n: 0, born: this.time, pos: new THREE.Vector3(), col: pickColor(p.to, near, 3) };
      this.centroids.set(arg.id, arg);
    }
    arg.n++;
    for (let k = 0; k < 3; k++) arg.sum[k] += p.to[k];
    arg.pos.set(arg.sum[0] / arg.n, arg.sum[1] / arg.n, arg.sum[2] / arg.n);
    this.label[idx] = arg.id;
    this.spokes.push({ idx, t: this.time });
  }

  // the window slides: old points leave their cluster, and an empty cluster closes
  _expireOldest() {
    const idx = this.head % MAX_SETTLED;
    this.head++;
    const c = this.centroids.get(this.label[idx]);
    if (!c) return;
    c.n--;
    if (c.n <= 0) { this.centroids.delete(c.id); return; }
    for (let k = 0; k < 3; k++) c.sum[k] -= this.pos[idx * 3 + k];
    c.pos.set(c.sum[0] / c.n, c.sum[1] / c.n, c.sum[2] / c.n);
  }

  // TB's cleanup on the live centroids: TS over them, oldest first; one that
  // lands within θ of an older keeper is a duplicate and merges into it
  _cleanup() {
    const keepers = [];
    for (const c of this.centroids.values()) {
      let best = Infinity, keeper = null;
      for (const k of keepers) {
        const d = k.pos.distanceTo(c.pos);
        if (d < best) { best = d; keeper = k; }
      }
      if (!keeper || best >= STREAM.theta) { keepers.push(c); continue; }
      keeper.n += c.n;
      for (let k = 0; k < 3; k++) keeper.sum[k] += c.sum[k];
      keeper.pos.set(keeper.sum[0] / keeper.n, keeper.sum[1] / keeper.n, keeper.sum[2] / keeper.n);
      for (let j = this.head; j < this.tail; j++) {
        const idx = j % MAX_SETTLED;
        if (this.label[idx] === c.id) this.label[idx] = keeper.id;
      }
      this.centroids.delete(c.id);
      this.merges.push({ id: c.id, into: keeper, t: this.time });
    }
  }

  update(dt) {
    this.time += dt;
    this.cleanClock += dt;
    if (this.cleanClock >= STREAM.cleanup) { this.cleanClock = 0; this._cleanup(); }
    this.merges = this.merges.filter((m) => this.time - m.t < 0.7);
    // Gaussians come and go, so the stream keeps discovering clusters
    this.swapClock += dt;
    if (this.swapClock >= STREAM.swap) {
      this.swapClock = 0;
      const oldest = this.sources.find((s) => s.active);
      if (this._addSource() && oldest) { oldest.active = false; oldest.retired = this.time; }
    }
    this.sources = this.sources.filter((s) => s.active || this.time - s.retired < STREAM.life + STREAM.travel[1]);

    this.spawnDebt += STREAM.rate * dt;
    while (this.spawnDebt >= 1 && this.flying.length < MAX_FLYING) { this._spawn(); this.spawnDebt -= 1; }
    this.spawnDebt = Math.min(this.spawnDebt, 1);

    const keep = [];
    for (const p of this.flying) {
      p.t += dt;
      if (p.t >= p.dur) this._arrive(p);
      else keep.push(p);
    }
    this.flying = keep;
    while (this.tail > this.head && this.time - this.born[this.head % MAX_SETTLED] > STREAM.life) this._expireOldest();
    this.spokes = this.spokes.filter((s) => this.time - s.t < 0.5);
  }
}

// ---------- shaders ----------

const pointVert = /* glsl */ `
  attribute vec4 rgba;
  attribute float size;
  uniform float uScale;
  varying vec4 vColor;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = size > 0.0 ? max(1.0, size * uScale / -mv.z) : 0.0;
    vColor = rgba;
  }`;
const pointFrag = /* glsl */ `
  uniform float uAlpha;
  varying vec4 vColor;
  void main() {
    float d = length(gl_PointCoord - 0.5) * 2.0;
    float a = 1.0 - smoothstep(0.3, 1.0, d);
    gl_FragColor = vec4(vColor.rgb, a * vColor.a * uAlpha);
  }`;
const shellVert = /* glsl */ `
  attribute float aSh;
  varying vec3 vN;
  varying vec3 vV;
  varying vec3 vC;
  varying float vSh;
  void main() {
    mat4 m = modelMatrix * instanceMatrix;
    vec4 wp = m * vec4(position, 1.0);
    vN = normalize(mat3(m) * normal);
    vV = normalize(cameraPosition - wp.xyz);
    #ifdef USE_INSTANCING_COLOR
      vC = instanceColor;
    #else
      vC = vec3(1.0);
    #endif
    vSh = aSh;
    gl_Position = projectionMatrix * viewMatrix * wp;
  }`;
const shellFrag = /* glsl */ `
  uniform float uI;
  uniform float uLight;
  varying vec3 vN;
  varying vec3 vV;
  varying vec3 vC;
  varying float vSh;
  void main() {
    float f = 1.0 - abs(dot(normalize(vN), normalize(vV)));
    f = pow(f, 7.0); // a thin rim: θ reads as an outline, not a glowing ball
    if (uLight > 0.5) gl_FragColor = vec4(vC, clamp(f * vSh * uI * 2.2, 0.0, 1.0));
    // on dark the rim is the hue washed toward white, so it reads as a fine line
    else gl_FragColor = vec4((vC + 0.5) * 0.66 * f * vSh * uI, 1.0);
  }`;

const pointMaterial = (alpha) => new THREE.ShaderMaterial({
  uniforms: { uScale: { value: 1 }, uAlpha: { value: alpha } }, vertexShader: pointVert, fragmentShader: pointFrag,
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
});

// ---------- the scene ----------

export class Universe {
  constructor(canvas, { reducedMotion = false } = {}) {
    this.canvas = canvas;
    this.reduced = reducedMotion;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' });
    this.renderer.setClearColor(0x04050a, 1);
    this.pr = Math.min(window.devicePixelRatio || 1, 1.5);
    this.renderer.setPixelRatio(this.pr);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.1, 2000);
    this.root = new THREE.Group();
    this.scene.add(this.root);

    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.55, 0.35, 0.6);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());

    this.theme = currentTheme();
    this._buildBackdrop();
    this._buildCentroids();
    this._buildStream();

    this.slots = new Map();
    this.frame = 0;
    this.time = 0;
    this.tmp = new THREE.Vector3();
    this.mode = 'hero';
    this.vis = 1; // how much of the stream shows: it fades out below the hero
    this.cam = { az: 0, el: 0.14, r: 96, shift: 0.27, lift: 0, px: 0, py: 0, tpx: 0, tpy: 0 };
    this.camTarget = { r: 96, el: 0.14, shift: 0.27, lift: 0 };

    this.stream = new Stream(T.mulberry32(7));
    // open mid-flow rather than on an empty stage
    for (let i = 0; i < 300; i++) this.stream.update(1 / 20);
    this.setTheme(this.theme);

    this.resize();
    window.addEventListener('resize', () => this.resize());
    window.addEventListener('pointermove', (e) => {
      this.cam.tpx = e.clientX / window.innerWidth - 0.5;
      this.cam.tpy = e.clientY / window.innerHeight - 0.5;
    }, { passive: true });

    this.clock = new THREE.Clock();
    this._loop = this._loop.bind(this);
    requestAnimationFrame(this._loop);
  }

  get clustersFound() { return this.stream.centroids.size; }

  setMode(mode) { this.mode = mode; }

  setTheme(theme) {
    this.theme = theme;
    const look = LOOK[theme];
    this.renderer.setClearColor(look.clear, 1);
    this.bloom.enabled = look.bloom;
    this.stars.visible = look.stars;
    for (const m of [this.settledMat, this.headMat, this.lineMat, this.shellMat]) {
      m.blending = look.blending;
      m.needsUpdate = true;
    }
    this.shellMat.uniforms.uLight.value = theme === 'light' ? 1 : 0;
  }

  // ----- building -----

  _buildBackdrop() {
    const n = 2200, pos = new Float32Array(n * 3), col = new Float32Array(n * 4), size = new Float32Array(n);
    const rng = T.mulberry32(99);
    for (let i = 0; i < n; i++) {
      const u = rng() * 2 - 1, phi = rng() * Math.PI * 2, r = 180 + rng() * 420;
      const s = Math.sqrt(1 - u * u);
      pos[i * 3] = r * s * Math.cos(phi); pos[i * 3 + 1] = r * u; pos[i * 3 + 2] = r * s * Math.sin(phi);
      const b = 0.05 + rng() * 0.2;
      col[i * 4] = b * 0.85; col[i * 4 + 1] = b * 0.9; col[i * 4 + 2] = b; col[i * 4 + 3] = 1;
      size[i] = 0.8 + rng() * 2.2;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('rgba', new THREE.BufferAttribute(col, 4));
    g.setAttribute('size', new THREE.BufferAttribute(size, 1));
    this.starMat = pointMaterial(0.6);
    this.stars = new THREE.Points(g, this.starMat);
    this.scene.add(this.stars);
  }

  _buildStream() {
    // points already read
    const g = new THREE.BufferGeometry();
    this.sPos = new THREE.BufferAttribute(new Float32Array(MAX_SETTLED * 3), 3).setUsage(THREE.DynamicDrawUsage);
    this.sCol = new THREE.BufferAttribute(new Float32Array(MAX_SETTLED * 4), 4).setUsage(THREE.DynamicDrawUsage);
    this.sSize = new THREE.BufferAttribute(new Float32Array(MAX_SETTLED), 1).setUsage(THREE.DynamicDrawUsage);
    g.setAttribute('position', this.sPos); g.setAttribute('rgba', this.sCol); g.setAttribute('size', this.sSize);
    this.settledMat = pointMaterial(0.9);
    this.settled = new THREE.Points(g, this.settledMat);
    this.settled.frustumCulled = false;
    this.root.add(this.settled);

    // points in flight: a bright head and a streak behind it
    const h = new THREE.BufferGeometry();
    this.hPos = new THREE.BufferAttribute(new Float32Array(MAX_FLYING * 3), 3).setUsage(THREE.DynamicDrawUsage);
    this.hCol = new THREE.BufferAttribute(new Float32Array(MAX_FLYING * 4), 4).setUsage(THREE.DynamicDrawUsage);
    h.setAttribute('position', this.hPos); h.setAttribute('rgba', this.hCol);
    h.setAttribute('size', new THREE.BufferAttribute(new Float32Array(MAX_FLYING).fill(POINT_SIZE * 1.1), 1));
    this.headMat = pointMaterial(0.9);
    this.heads = new THREE.Points(h, this.headMat);
    this.heads.frustumCulled = false;
    this.root.add(this.heads);

    const l = new THREE.BufferGeometry();
    this.lPos = new THREE.BufferAttribute(new Float32Array(MAX_SEGMENTS * 6), 3).setUsage(THREE.DynamicDrawUsage);
    this.lCol = new THREE.BufferAttribute(new Float32Array(MAX_SEGMENTS * 8), 4).setUsage(THREE.DynamicDrawUsage);
    l.setAttribute('position', this.lPos); l.setAttribute('color', this.lCol);
    this.lineMat = new THREE.LineBasicMaterial({
      vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false,
    });
    this.lines = new THREE.LineSegments(l, this.lineMat);
    this.lines.frustumCulled = false;
    this.root.add(this.lines);
    this.bz = [0, 0, 0];
    this.bz2 = [0, 0, 0];
  }

  _buildCentroids() {
    const core = new THREE.SphereGeometry(0.3, 20, 14);
    this.cores = new THREE.InstancedMesh(core, new THREE.MeshBasicMaterial({ color: 0xffffff, toneMapped: false }), MAX_CENTROIDS);
    this.cores.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(MAX_CENTROIDS * 3), 3);
    this.cores.count = 0;
    this.cores.frustumCulled = false;
    this.root.add(this.cores);

    const shell = new THREE.SphereGeometry(1, 40, 24);
    this.shellMat = new THREE.ShaderMaterial({
      uniforms: { uI: { value: 1.1 }, uLight: { value: 0 } }, vertexShader: shellVert, fragmentShader: shellFrag,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    this.shells = new THREE.InstancedMesh(shell, this.shellMat, MAX_CENTROIDS);
    this.shells.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(MAX_CENTROIDS * 3), 3);
    this.shellSh = new THREE.InstancedBufferAttribute(new Float32Array(MAX_CENTROIDS), 1);
    shell.setAttribute('aSh', this.shellSh);
    this.shells.count = 0;
    this.shells.frustumCulled = false;
    this.root.add(this.shells);
    this.m4 = new THREE.Matrix4(); this.q = new THREE.Quaternion(); this.sv = new THREE.Vector3();
  }

  resize() {
    const w = window.innerWidth, h = window.innerHeight;
    if (!w || !h) return; // hidden pane or minimised window
    this.w = w; this.h = h;
    this.renderer.setSize(w, h, false);
    this.composer.setSize(w, h);
    this.bloom.resolution.set(w / 2, h / 2);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    const scale = (h * this.pr * 0.5) / Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2));
    for (const m of [this.settledMat, this.headMat]) m.uniforms.uScale.value = scale;
    this.starMat.uniforms.uScale.value = scale * 0.06;
  }

  // ----- per-frame wants: a centroid that should be on screen this frame -----

  want(key, pos, color, scale = 1, shell = 0, shellR = 1) {
    let s = this.slots.get(key);
    if (!s) {
      s = { p: pos.clone(), tp: pos.clone(), s: 0, ts: 0, c: color.clone(), tc: color.clone(), sh: 0, tsh: 0, shr: shellR * 0.2, tshr: shellR };
      this.slots.set(key, s);
    }
    s.tp.copy(pos); s.tc.copy(color); s.ts = scale; s.tsh = shell; s.tshr = shellR;
    s.seen = this.frame;
    return s;
  }

  // ----- the two scenes -----

  _drawStream(alpha) {
    const st = this.stream, now = st.time, look = LOOK[this.theme], pal = PAL[this.theme];
    // points already read: flare in their cluster's hue on arrival, then settle and fade out
    const col = this.sCol.array, size = this.sSize.array, pos = this.sPos.array;
    const count = st.tail - st.head, base = st.head % MAX_SETTLED;
    for (let idx = 0; idx < MAX_SETTLED; idx++) {
      const live = (idx - base + MAX_SETTLED) % MAX_SETTLED < count;
      if (!live) { size[idx] = 0; col[idx * 4 + 3] = 0; continue; }
      const age = now - st.born[idx];
      const f = Math.exp(-age / 0.25);
      const cent = st.centroids.get(st.label[idx]);
      const c = pal[cent ? cent.col : 0];
      const k = look.point + (look.flash - look.point) * f;
      col[idx * 4] = c.r * k; col[idx * 4 + 1] = c.g * k; col[idx * 4 + 2] = c.b * k;
      col[idx * 4 + 3] = clamp01((STREAM.life - age) / 1.4);
      size[idx] = POINT_SIZE * look.size * (1 + 1.6 * f);
      pos[idx * 3] = st.pos[idx * 3]; pos[idx * 3 + 1] = st.pos[idx * 3 + 1]; pos[idx * 3 + 2] = st.pos[idx * 3 + 2];
    }
    this.sPos.needsUpdate = this.sCol.needsUpdate = this.sSize.needsUpdate = true;

    // points in flight
    const hp = this.hPos.array, hc = this.hCol.array, lp = this.lPos.array, lc = this.lCol.array;
    let nl = 0;
    st.flying.forEach((p, i) => {
      // faint while crossing behind the headline, bright as it nears its cluster
      const u = flightU(p.t, p.dur);
      const a = clamp01(p.t / 0.35) * (0.22 + 0.78 * clamp01((u - 0.15) / 0.4));
      const head = bezier(p, u, this.bz);
      const tail = bezier(p, flightU(Math.max(0, p.t - STREAM.trail), p.dur), this.bz2);
      hp[i * 3] = head[0]; hp[i * 3 + 1] = head[1]; hp[i * 3 + 2] = head[2];
      const fc = look.flight, fa = a * look.flightAlpha;
      hc[i * 4] = fc.r; hc[i * 4 + 1] = fc.g; hc[i * 4 + 2] = fc.b; hc[i * 4 + 3] = fa;
      const o = nl * 6, oc = nl * 8;
      lp[o] = head[0]; lp[o + 1] = head[1]; lp[o + 2] = head[2];
      lp[o + 3] = tail[0]; lp[o + 4] = tail[1]; lp[o + 5] = tail[2];
      lc[oc] = lc[oc + 4] = fc.r; lc[oc + 1] = lc[oc + 5] = fc.g; lc[oc + 2] = lc[oc + 6] = fc.b;
      lc[oc + 3] = fa * 0.8; lc[oc + 7] = 0;
      nl++;
    });
    this.heads.geometry.setDrawRange(0, st.flying.length);
    // a short spoke from each newly read point to the centroid it joined
    for (const s of st.spokes) {
      if (nl >= MAX_SEGMENTS) break;
      const c = st.centroids.get(st.label[s.idx]);
      if (!c) continue;
      const k = 0.6 * (1 - (now - s.t) / 0.5), o = nl * 6, oc = nl * 8, h = pal[c.col];
      lp[o] = st.pos[s.idx * 3]; lp[o + 1] = st.pos[s.idx * 3 + 1]; lp[o + 2] = st.pos[s.idx * 3 + 2];
      lp[o + 3] = c.pos.x; lp[o + 4] = c.pos.y; lp[o + 5] = c.pos.z;
      lc[oc] = lc[oc + 4] = h.r; lc[oc + 1] = lc[oc + 5] = h.g; lc[oc + 2] = lc[oc + 6] = h.b;
      lc[oc + 3] = lc[oc + 7] = k;
      nl++;
    }
    this.lines.geometry.setDrawRange(0, nl * 2);
    this.hPos.needsUpdate = this.hCol.needsUpdate = this.lPos.needsUpdate = this.lCol.needsUpdate = true;

    this.settledMat.uniforms.uAlpha.value = 0.9 * alpha;
    this.headMat.uniforms.uAlpha.value = 0.9 * alpha;
    this.lineMat.opacity = alpha;

    // centroids pop when a cluster opens, then rest with a faint θ outline
    if (this.mode !== 'hero') return;
    for (const c of st.centroids.values()) {
      const age = now - c.born;
      this.want(`s${c.id}`, c.pos, pal[c.col], 1 + 1.2 * Math.exp(-age / 0.35), 0.12 + 0.7 * Math.exp(-age / 0.6));
    }
    // a merged duplicate turns red and slides into the centroid that keeps its points
    for (const mg of st.merges) {
      const k = clamp01((now - mg.t) / 0.7);
      this.want(`s${mg.id}`, mg.into.pos, SPUR_C[this.theme], 1.3 * (1 - k), 0.3 * (1 - k));
    }
  }

  // ----- loop -----

  _loop() {
    requestAnimationFrame(this._loop);
    this.tick(Math.min(this.clock.getDelta(), 0.05));
  }

  tick(dt) {
    if (!this.w) return;
    this.frame++;
    this.time += dt;
    this.vis += ((this.mode === 'hero' ? 1 : 0) - this.vis) * (1 - Math.exp(-dt * 3));
    const show = this.vis > 0.005;
    this.settled.visible = this.heads.visible = this.lines.visible = show;
    if (show) {
      this.stream.update(dt * (this.reduced ? 0.35 : 1));
      this._drawStream(this.vis);
    }

    this._updateSlots(dt);
    this._updateCamera(dt);
    this.composer.render();
  }

  _updateSlots(dt) {
    const a = 1 - Math.exp(-dt * 7), slow = 1 - Math.exp(-dt * 4);
    let i = 0;
    const cc = this.cores.instanceColor.array, sc = this.shells.instanceColor.array, sh = this.shellSh.array;
    const core = LOOK[this.theme].core;
    const theta = STREAM.theta;
    for (const [key, s] of this.slots) {
      if (s.seen !== this.frame) { s.ts = 0; s.tsh = 0; }
      s.p.lerp(s.tp, a);
      s.c.lerp(s.tc, a);
      s.s += (s.ts - s.s) * a;
      s.sh += (s.tsh - s.sh) * slow;
      s.shr += (s.tshr - s.shr) * slow;
      if (s.seen !== this.frame && s.s < 0.02) { this.slots.delete(key); continue; }
      if (i >= MAX_CENTROIDS) continue;
      this.m4.compose(s.p, this.q, this.sv.setScalar(Math.max(0.0001, s.s)));
      this.cores.setMatrixAt(i, this.m4);
      cc[i * 3] = s.c.r * core; cc[i * 3 + 1] = s.c.g * core; cc[i * 3 + 2] = s.c.b * core;
      this.m4.compose(s.p, this.q, this.sv.setScalar(Math.max(0.0001, theta * s.shr * Math.min(1, s.s))));
      this.shells.setMatrixAt(i, this.m4);
      sc[i * 3] = s.c.r; sc[i * 3 + 1] = s.c.g; sc[i * 3 + 2] = s.c.b;
      sh[i] = s.sh;
      i++;
    }
    this.cores.count = i; this.shells.count = i;
    this.cores.instanceMatrix.needsUpdate = true; this.cores.instanceColor.needsUpdate = true;
    this.shells.instanceMatrix.needsUpdate = true; this.shells.instanceColor.needsUpdate = true; this.shellSh.needsUpdate = true;
  }

  // ----- camera -----

  setCamera(target) { Object.assign(this.camTarget, target); }

  _updateCamera(dt) {
    const c = this.cam, t = this.camTarget;
    const a = 1 - Math.exp(-dt * 1.6);
    c.r += (t.r - c.r) * a;
    c.el += (t.el - c.el) * a;
    c.shift += (t.shift - c.shift) * a;
    c.lift += ((t.lift || 0) - c.lift) * a;
    // face the stream so it always enters from the left, with a slow sway
    c.az += (0.22 * Math.sin(this.time * 0.1) - c.az) * a;
    c.px += (c.tpx - c.px) * 0.04; c.py += (c.tpy - c.py) * 0.04;
    const az = c.az + c.px * 0.1;
    const el = Math.max(-0.2, Math.min(1.35, c.el + c.py * 0.12));
    this.camera.position.set(c.r * Math.cos(el) * Math.sin(az), c.r * Math.sin(el), c.r * Math.cos(el) * Math.cos(az));
    this.camera.lookAt(0, -1, 0);
    this.camera.setViewOffset(this.w, this.h, -c.shift * this.w, c.lift * this.h, this.w, this.h);
    // the stream enters just beyond the left edge of the screen
    const halfW = c.r * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)) * this.camera.aspect;
    this.stream.startX = -(1 + 2 * c.shift) * halfW - 8;
  }
}

