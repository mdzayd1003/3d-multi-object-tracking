/* Hold docs/mot.js to the Python package.
 *
 * Two standards, on purpose:
 *
 *   EXACT   -- every discrete output. Assignment pairs, the set of track ids
 *              reported in each frame, and the ID switch / false positive /
 *              miss counts. These are decisions, not measurements: if the two
 *              ports disagree about which detection belongs to which track,
 *              the demo is showing something the library would not do.
 *
 *   1e-9    -- the continuous values. The Kalman update inverts a 7x7 matrix,
 *              and numpy's LAPACK inverse and the Gauss-Jordan routine in the
 *              port are not obliged to agree in the last bits. Demanding
 *              equality there would be demanding that JavaScript reimplement
 *              LAPACK.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import * as mot from "../docs/mot.js";

const here = dirname(fileURLToPath(import.meta.url));
const ref = JSON.parse(readFileSync(join(here, "..", "results", "reference.json"), "utf8"));

const TOL = 1e-9;
let worst = 0, worstWhere = "-", checks = 0, failures = [];

function near(got, want, where) {
  checks++;
  const d = Math.abs(got - want);
  if (d > worst) { worst = d; worstWhere = where; }
  if (!(d <= TOL)) failures.push(`${where}: ${got} vs ${want} (|d| = ${d.toExponential(3)})`);
}

function exact(got, want, where) {
  checks++;
  if (JSON.stringify(got) !== JSON.stringify(want)) {
    failures.push(`${where}: ${JSON.stringify(got)} != ${JSON.stringify(want)}`);
  }
}

/* 1. box geometry */
ref.iou.forEach((c, i) => near(mot.iou3d(c.a, c.b), c.iou, `iou[${i}]`));

/* 2. both solvers, ties included */
ref.solvers.forEach((c, i) => {
  exact(mot.hungarian(c.cost), c.hungarian, `hungarian[${i}]`);
  exact(mot.greedy(c.cost), c.greedy, `greedy[${i}]`);
});

/* 3. the filter */
{
  const kf = new mot.KalmanBox([0, 0, 0, 0.1, 4, 2, 1.5]);
  ref.kalman.forEach((step, t) => {
    kf.predict();
    const box = kf.update(step.z);
    box.forEach((v, i) => near(v, step.box[i], `kalman[${t}].box[${i}]`));
    kf.velocity().forEach((v, i) => near(v, step.v[i], `kalman[${t}].v[${i}]`));
  });
}

/* 4. scenes, tracker identities and CLEAR MOT */
for (const s of ref.scenes) {
  const tag = `${s.case.name}@${s.case.noise}`;
  const scene = mot.makeScene({ name: s.case.name, seed: s.case.seed, noise: s.case.noise });

  exact(scene.frames.length, s.frames.length, `${tag}.frameCount`);
  s.frames.forEach((f, t) => {
    const mine = scene.frames[t];
    exact(mine.detections.length, f.det.length, `${tag}.f${t}.detCount`);
    for (const [k, v] of Object.entries(f.gt)) {
      v.forEach((x, i) => near(mine.gt.get(Number(k))[i], x, `${tag}.f${t}.gt${k}[${i}]`));
    }
    f.det.forEach((d, j) => {
      d.box.forEach((x, i) => near(mine.detections[j].box[i], x, `${tag}.f${t}.det${j}[${i}]`));
      near(mine.detections[j].score, d.score, `${tag}.f${t}.det${j}.score`);
    });
  });

  for (const r of s.runs) {
    const cfg = {
      metric: r.config.metric,
      solver: r.config.solver,
      maxAge: r.config.max_age ?? 3,
      minHits: r.config.min_hits ?? 2,
    };
    const label = `${tag}/${cfg.metric}-${cfg.solver}-a${cfg.maxAge}h${cfg.minHits}`;
    const outputs = mot.runTracker(scene, cfg);
    const ids = outputs.map((o) => [...o.boxes.keys()].sort((a, b) => a - b));
    exact(ids, r.ids, `${label}.trackIds`);

    const rep = mot.clearMot(scene, outputs, "dist");
    for (const k of ["fp", "fn", "ids", "matches", "gt"]) exact(rep[k], r.report[k], `${label}.${k}`);
    for (const k of ["mota", "motp", "recall", "precision"]) near(rep[k], r.report[k], `${label}.${k}`);
  }
}

console.log(`crosscheck: ${checks} comparisons`);
console.log(`  worst continuous difference ${worst.toExponential(3)} at ${worstWhere} (tolerance ${TOL.toExponential(0)})`);
if (failures.length) {
  console.error(`\n${failures.length} FAILED:`);
  for (const f of failures.slice(0, 25)) console.error("  " + f);
  if (failures.length > 25) console.error(`  ... and ${failures.length - 25} more`);
  process.exit(1);
}
console.log("  every discrete output identical; all continuous values within tolerance");
