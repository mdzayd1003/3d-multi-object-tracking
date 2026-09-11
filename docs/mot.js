/* 3D multi-object tracking, ported from the Python package.
 *
 * The demo computes rather than replays, so this file carries the whole
 * deterministic pipeline: the RNG, the scene generator, oriented-box geometry,
 * the Kalman filter, both solvers, the tracker and CLEAR MOT.
 *
 * scripts/crosscheck.mjs holds it to the Python. The check is in two parts and
 * the split is deliberate: every DISCRETE output -- assignment pairs, track
 * identities, the ID switch, false positive and miss counts -- must match
 * EXACTLY, because those are the decisions the tracker actually makes. The
 * continuous values are compared to a tolerance, because the Kalman update
 * inverts a 7x7 matrix and numpy's LAPACK inverse and the Gauss-Jordan one
 * below do not have to agree in the last bits.
 */

const MASK = 0xffffffff;

export function mix(seed, index) {
  let x = (((seed & 0xffff) << 16) ^ (index & 0xffff) ^ 0x5bf03635) >>> 0;
  x = Math.imul(x, 2654435761) >>> 0;
  return (x ^ (x >>> 15)) >>> 0;
}

export class XorShift32 {
  constructor(seed) {
    this.s = (seed & MASK) >>> 0 || 0x9e3779b9;
  }
  nextU32() {
    let x = this.s;
    x = (x ^ (x << 13)) >>> 0;
    x = (x ^ (x >>> 17)) >>> 0;
    x = (x ^ (x << 5)) >>> 0;
    this.s = x >>> 0;
    return this.s;
  }
  random() { return this.nextU32() / 4294967296.0; }
  uniform(lo, hi) { return lo + (hi - lo) * this.random(); }
  normal(mu = 0, sigma = 1) {
    const u1 = Math.max(this.random(), 1e-12);
    const u2 = this.random();
    return mu + sigma * Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  }
}

/* ---------- oriented 3D boxes: [x, y, z, yaw, l, w, h] ---------- */

export function cornersBev(b) {
  const c = Math.cos(b[3]), s = Math.sin(b[3]);
  const dx = b[4] / 2, dy = b[5] / 2;
  const local = [[dx, dy], [-dx, dy], [-dx, -dy], [dx, -dy]];
  return local.map(([lx, ly]) => [b[0] + lx * c - ly * s, b[1] + lx * s + ly * c]);
}

export function polyArea(poly) {
  if (poly.length < 3) return 0;
  let a = 0;
  for (let i = 0; i < poly.length; i++) {
    const j = (i + 1) % poly.length;
    a += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1];
  }
  return Math.abs(a) / 2;
}

export function convexIntersection(subject, clip) {
  let output = subject.map((p) => [p[0], p[1]]);
  for (let i = 0; i < clip.length; i++) {
    if (output.length === 0) return [];
    const a = clip[i], b = clip[(i + 1) % clip.length];
    const ex = b[0] - a[0], ey = b[1] - a[1];
    const side = (p) => ex * (p[1] - a[1]) - ey * (p[0] - a[0]);
    const current = output;
    output = [];
    for (let j = 0; j < current.length; j++) {
      const p = current[j];
      const q = current[(j - 1 + current.length) % current.length];
      const sp = side(p), sq = side(q);
      if (sp >= 0) {
        if (sq < 0) {
          const t = sq / (sq - sp);
          output.push([q[0] + t * (p[0] - q[0]), q[1] + t * (p[1] - q[1])]);
        }
        output.push(p);
      } else if (sq >= 0) {
        const t = sq / (sq - sp);
        output.push([q[0] + t * (p[0] - q[0]), q[1] + t * (p[1] - q[1])]);
      }
    }
  }
  return output;
}

export function iou3d(a, b) {
  const top = Math.min(a[2] + a[6] / 2, b[2] + b[6] / 2);
  const bottom = Math.max(a[2] - a[6] / 2, b[2] - b[6] / 2);
  const dz = top - bottom;
  if (dz <= 0) return 0;
  const interArea = polyArea(convexIntersection(cornersBev(a), cornersBev(b)));
  if (interArea <= 0) return 0;
  const inter = interArea * dz;
  const union = a[4] * a[5] * a[6] + b[4] * b[5] * b[6] - inter;
  return union > 0 ? inter / union : 0;
}

export function centerDistance(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1]); }

export function costMatrix(tracks, dets, metric) {
  return tracks.map((t) => dets.map((d) => (metric === "iou" ? 1 - iou3d(t, d) : centerDistance(t, d))));
}

/* ---------- assignment ---------- */

export function hungarian(cost) {
  if (!cost.length || !cost[0].length) return [];
  let C = cost, n = cost.length, m = cost[0].length;
  const transposed = n > m;
  if (transposed) {
    C = Array.from({ length: m }, (_, i) => Array.from({ length: n }, (_, j) => cost[j][i]));
    [n, m] = [m, n];
  }
  const INF = Infinity;
  const u = new Float64Array(n + 1), v = new Float64Array(m + 1);
  const p = new Int32Array(m + 1), way = new Int32Array(m + 1);

  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Float64Array(m + 1).fill(INF);
    const used = new Uint8Array(m + 1);
    for (;;) {
      used[j0] = 1;
      const i0 = p[j0];
      let delta = INF, j1 = -1;
      for (let j = 1; j <= m; j++) {
        if (used[j]) continue;
        const cur = C[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
      }
      for (let j = 0; j <= m; j++) {
        if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
        else { minv[j] -= delta; }
      }
      j0 = j1;
      if (p[j0] === 0) break;
    }
    while (j0) { const j1 = way[j0]; p[j0] = p[j1]; j0 = j1; }
  }

  let pairs = [];
  for (let j = 1; j <= m; j++) if (p[j] > 0) pairs.push([p[j] - 1, j - 1]);
  if (transposed) pairs = pairs.map(([r, c]) => [c, r]);
  pairs.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  return pairs;
}

export function greedy(cost) {
  if (!cost.length || !cost[0].length) return [];
  const n = cost.length, m = cost[0].length;
  const flat = [];
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) flat.push([cost[i][j], i * m + j]);
  // Stable ascending sort by cost, ties by flat index -- matches numpy's
  // argsort(kind="stable") on a C-ordered array.
  flat.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const rows = new Set(), cols = new Set(), pairs = [];
  for (const [, k] of flat) {
    const i = Math.floor(k / m), j = k % m;
    if (rows.has(i) || cols.has(j)) continue;
    rows.add(i); cols.add(j); pairs.push([i, j]);
    if (pairs.length === Math.min(n, m)) break;
  }
  pairs.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  return pairs;
}

export function associate(cost, threshold, solver = "hungarian") {
  const n = cost.length, m = n ? cost[0].length : 0;
  const pairs = (solver === "greedy" ? greedy : hungarian)(cost);
  const matched = pairs.filter(([i, j]) => cost[i][j] <= threshold);
  const rows = new Set(matched.map(([i]) => i));
  const cols = new Set(matched.map(([, j]) => j));
  const unT = [], unD = [];
  for (let i = 0; i < n; i++) if (!rows.has(i)) unT.push(i);
  for (let j = 0; j < m; j++) if (!cols.has(j)) unD.push(j);
  return { matched, unmatchedTracks: unT, unmatchedDets: unD };
}

/* ---------- small dense linear algebra ---------- */

const matmul = (A, B) => A.map((row) => B[0].map((_, j) => row.reduce((s, v, k) => s + v * B[k][j], 0)));
const transpose = (A) => A[0].map((_, j) => A.map((r) => r[j]));
const eye = (n) => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
const matsub = (A, B) => A.map((r, i) => r.map((v, j) => v - B[i][j]));
const matadd = (A, B) => A.map((r, i) => r.map((v, j) => v + B[i][j]));
const matvec = (A, x) => A.map((r) => r.reduce((s, v, k) => s + v * x[k], 0));

export function inverse(M) {
  const n = M.length;
  const a = M.map((r, i) => [...r, ...eye(n)[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(a[r][col]) > Math.abs(a[piv][col])) piv = r;
    if (Math.abs(a[piv][col]) < 1e-300) throw new Error("singular matrix");
    [a[col], a[piv]] = [a[piv], a[col]];
    const d = a[col][col];
    for (let j = 0; j < 2 * n; j++) a[col][j] /= d;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = a[r][col];
      if (f === 0) continue;
      for (let j = 0; j < 2 * n; j++) a[r][j] -= f * a[col][j];
    }
  }
  return a.map((r) => r.slice(n));
}

/* ---------- Kalman ---------- */

export const DIM_X = 10, DIM_Z = 7;

export function wrapAngle(a) {
  const twoPi = 2 * Math.PI;
  return ((a + Math.PI) % twoPi + twoPi) % twoPi - Math.PI;
}

export class KalmanBox {
  constructor(box, dt = 0.1, posVar = 1.0, velVar = 100.0, q = 0.05, r = 0.35) {
    this.dt = dt;
    this.x = new Array(DIM_X).fill(0);
    for (let i = 0; i < DIM_Z; i++) this.x[i] = box[i];
    this.F = eye(DIM_X);
    this.F[0][7] = dt; this.F[1][8] = dt; this.F[2][9] = dt;
    this.H = Array.from({ length: DIM_Z }, (_, i) =>
      Array.from({ length: DIM_X }, (_, j) => (i === j ? 1 : 0)));
    this.P = eye(DIM_X).map((row, i) => row.map((val, j) => (i === j ? (i < DIM_Z ? posVar : velVar) : 0)));
    this.Q = eye(DIM_X).map((row, i) => row.map((val, j) => (i === j ? (i < DIM_Z ? q : q * 10) : 0)));
    this.R = eye(DIM_Z).map((row, i) => row.map((val, j) => (i === j ? r : 0)));
  }
  predict() {
    this.x = matvec(this.F, this.x);
    this.x[3] = wrapAngle(this.x[3]);
    this.P = matadd(matmul(matmul(this.F, this.P), transpose(this.F)), this.Q);
    return this.box();
  }
  update(z) {
    const hx = matvec(this.H, this.x);
    const y = z.map((v, i) => v - hx[i]);
    y[3] = wrapAngle(y[3]);
    const S = matadd(matmul(matmul(this.H, this.P), transpose(this.H)), this.R);
    const K = matmul(matmul(this.P, transpose(this.H)), inverse(S));
    const Ky = matvec(K, y);
    this.x = this.x.map((v, i) => v + Ky[i]);
    this.x[3] = wrapAngle(this.x[3]);
    const IKH = matsub(eye(DIM_X), matmul(K, this.H));
    this.P = matadd(matmul(matmul(IKH, this.P), transpose(IKH)),
                    matmul(matmul(K, this.R), transpose(K)));
    return this.box();
  }
  box() {
    const b = this.x.slice(0, DIM_Z);
    for (let i = 4; i < 7; i++) b[i] = Math.max(b[i], 1e-3);
    return b;
  }
  velocity() { return this.x.slice(7, 10); }
}

/* ---------- scenes ---------- */

export const SCENES = {
  sparse:   { n: 6,  spacing: 12.0, crossings: 0 },
  crossing: { n: 8,  spacing: 6.0,  crossings: 3 },
  dense:    { n: 16, spacing: 3.0,  crossings: 6 },
};

function trajectory(rng, lane, crosses, nFrames, dt, spacing) {
  const x0 = rng.uniform(-40, -10);
  const speed = rng.uniform(4, 11);
  const length = rng.uniform(3.6, 4.9);
  const width = rng.uniform(1.6, 2.1);
  const height = rng.uniform(1.4, 1.8);
  const z = height / 2;
  const duration = Math.max(nFrames * dt, 1e-6);
  let lateralRate = 0;
  if (crosses) {
    const lanes = rng.uniform(1.2, 2.4);
    lateralRate = (spacing * lanes / duration) * (rng.random() < 0.5 ? 1 : -1);
  }
  const wander = rng.uniform(0, 0.15);
  const phase = rng.uniform(0, 2 * Math.PI);
  const path = [];
  for (let t = 0; t < nFrames; t++) {
    const tt = t * dt;
    const x = x0 + speed * tt;
    const y = lane + lateralRate * (tt - nFrames * dt / 2) + wander * Math.sin(phase + tt);
    const dy = lateralRate + wander * Math.cos(phase + tt);
    path.push([x, y, z, Math.atan2(dy, speed), length, width, height]);
  }
  return path;
}

export function makeScene(opts = {}) {
  const {
    name = "crossing", nFrames = 60, seed = 7, dt = 0.1,
    missRate = 0.12, fpPerFrame = 0.6, noise = 0.25,
  } = opts;
  const cfg = SCENES[name];
  if (!cfg) throw new Error(`unknown scene ${name}`);
  const rng = new XorShift32(mix(seed, 1013));
  const n = cfg.n;
  const tracks = [];
  for (let i = 0; i < n; i++) {
    tracks.push(trajectory(rng, (i - n / 2) * cfg.spacing, i < cfg.crossings, nFrames, dt, cfg.spacing));
  }

  const frames = [];
  for (let t = 0; t < nFrames; t++) {
    const gt = new Map();
    for (let i = 0; i < n; i++) gt.set(i, tracks[i][t]);
    const detections = [];
    for (let i = 0; i < n; i++) {
      if (rng.random() < missRate) continue;
      const truth = tracks[i][t];
      const box = truth.slice();
      box[0] += rng.normal(0, noise);
      box[1] += rng.normal(0, noise);
      box[2] += rng.normal(0, noise * 0.4);
      box[3] += rng.normal(0, noise * 0.15);
      for (let k = 4; k < 7; k++) box[k] = Math.max(box[k] + rng.normal(0, noise * 0.25), 0.5);
      const err = Math.hypot(box[0] - truth[0], box[1] - truth[1]);
      const score = Math.max(0.05, Math.min(0.99, 0.95 - 0.45 * err + rng.normal(0, 0.05)));
      detections.push({ box, score, gtId: i });
    }
    let nFp = Math.floor(fpPerFrame);
    if (rng.random() < fpPerFrame % 1) nFp += 1;
    for (let k = 0; k < nFp; k++) {
      const box = [
        rng.uniform(-45, 45),
        rng.uniform(-n * cfg.spacing / 2 - 5, n * cfg.spacing / 2 + 5),
        0.8, rng.uniform(-Math.PI, Math.PI),
        rng.uniform(3, 5), rng.uniform(1.5, 2.2), rng.uniform(1.4, 1.8),
      ];
      detections.push({ box, score: Math.max(0.05, Math.min(0.9, rng.normal(0.28, 0.14))), gtId: -1 });
    }
    // Python sorts by descending score with a stable sort; ties keep insertion
    // order in both languages.
    const idx = detections.map((_, k) => k);
    idx.sort((a, b) => detections[b].score - detections[a].score || a - b);
    frames.push({ index: t, gt, detections: idx.map((k) => detections[k]) });
  }
  const totalGt = frames.reduce((s, f) => s + f.gt.size, 0);
  return { name, frames, nObjects: n, dt, totalGt };
}

/* ---------- tracker ---------- */

export const DEFAULT_GATE = { iou: 0.999, dist: 4.0 };

export function runTracker(scene, cfg = {}) {
  const {
    metric = "dist", solver = "hungarian", gate = null,
    maxAge = 3, minHits = 2, scoreThreshold = 0, dt = 0.1,
  } = cfg;
  const gateValue = gate === null ? DEFAULT_GATE[metric] : gate;
  let tracks = [];
  let nextId = 1;
  const outputs = [];

  for (const frame of scene.frames) {
    const dets = frame.detections.filter((d) => d.score >= scoreThreshold);
    for (const tr of tracks) { tr.kf.predict(); tr.age += 1; }

    const pred = tracks.map((t) => t.kf.box());
    const obs = dets.map((d) => d.box);
    let matched = [], unmatchedDets = [];
    if (pred.length && obs.length) {
      const C = costMatrix(pred, obs, metric);
      const res = associate(C, gateValue, solver);
      matched = res.matched;
      unmatchedDets = res.unmatchedDets;
    } else {
      unmatchedDets = dets.map((_, i) => i);
    }

    for (const [ti, di] of matched) {
      const tr = tracks[ti];
      tr.kf.update(dets[di].box);
      tr.hits += 1; tr.age = 0; tr.score = dets[di].score;
      if (tr.hits >= minHits) tr.confirmed = true;
    }
    for (const di of unmatchedDets) {
      tracks.push({
        id: nextId++, kf: new KalmanBox(dets[di].box, dt),
        hits: 1, age: 0, score: dets[di].score, confirmed: minHits <= 1,
      });
    }
    tracks = tracks.filter((t) => t.age <= maxAge);

    const boxes = new Map();
    for (const t of tracks) if (t.confirmed) boxes.set(t.id, t.kf.box());
    outputs.push({ frame: frame.index, boxes });
  }
  return outputs;
}

/* ---------- CLEAR MOT ---------- */

export const DEFAULT_MATCH = { dist: 2.0, iou: 0.75 };

export function clearMot(scene, outputs, metric = "dist", threshold = null) {
  const thr = threshold === null ? DEFAULT_MATCH[metric] : threshold;
  const cost = (a, b) => (metric === "dist" ? centerDistance(a, b) : 1 - iou3d(a, b));
  let fp = 0, fn = 0, ids = 0, matches = 0, distSum = 0, gtTotal = 0;
  let prev = new Map();

  for (let k = 0; k < scene.frames.length; k++) {
    const gt = scene.frames[k].gt;
    const hyp = outputs[k].boxes;
    gtTotal += gt.size;
    const gtIds = [...gt.keys()].sort((a, b) => a - b);
    const hypIds = [...hyp.keys()].sort((a, b) => a - b);

    const mapping = new Map();
    const usedHyp = new Set();
    for (const g of gtIds) {
      const h = prev.get(g);
      if (h !== undefined && hyp.has(h) && !usedHyp.has(h)) {
        const c = cost(gt.get(g), hyp.get(h));
        if (c <= thr) { mapping.set(g, h); usedHyp.add(h); matches++; distSum += c; }
      }
    }
    const freeGt = gtIds.filter((g) => !mapping.has(g));
    const freeHyp = hypIds.filter((h) => !usedHyp.has(h));
    if (freeGt.length && freeHyp.length) {
      const C = freeGt.map((g) => freeHyp.map((h) => cost(gt.get(g), hyp.get(h))));
      const { matched } = associate(C, thr, "hungarian");
      for (const [gi, hi] of matched) {
        mapping.set(freeGt[gi], freeHyp[hi]);
        usedHyp.add(freeHyp[hi]);
        matches++; distSum += C[gi][hi];
      }
    }
    for (const [g, h] of mapping) {
      const p = prev.get(g);
      if (p !== undefined && p !== h) ids++;
    }
    fn += gt.size - mapping.size;
    fp += hyp.size - usedHyp.size;
    const carried = new Map(prev);
    for (const [g, h] of mapping) carried.set(g, h);
    prev = new Map([...carried].filter(([g]) => gt.has(g) || mapping.has(g)));
  }

  const n = Math.max(gtTotal, 1);
  return {
    mota: 1 - (fp + fn + ids) / n,
    motp: distSum / Math.max(matches, 1),
    fp, fn, ids, matches, gt: gtTotal,
    recall: matches / n,
    precision: matches / Math.max(matches + fp, 1),
  };
}
