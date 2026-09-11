"""Emit the reference the JavaScript port is checked against.

Covers the whole deterministic chain -- scene generation, both solvers, the
Kalman filter, the tracker's identity assignments and CLEAR MOT -- rather than
just the final score, so that a port that drifts in the middle and happens to
land on the same MOTA still fails.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mot.assign import greedy, hungarian                      # noqa: E402
from mot.boxes import cost_matrix, iou_3d                     # noqa: E402
from mot.kalman import KalmanBox                              # noqa: E402
from mot.metrics import clear_mot                             # noqa: E402
from mot.scenes import make_scene                             # noqa: E402
from mot.tracker import TrackerConfig, run                    # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "reference.json"

CASES = [
    dict(name="sparse", seed=7, noise=0.25),
    dict(name="crossing", seed=7, noise=0.25),
    dict(name="crossing", seed=7, noise=1.0),
    dict(name="dense", seed=11, noise=0.5),
]
CONFIGS = [
    dict(metric="dist", solver="hungarian"),
    dict(metric="dist", solver="greedy"),
    dict(metric="iou", solver="hungarian"),
    dict(metric="dist", solver="hungarian", max_age=1, min_hits=1),
]


def main() -> int:
    ref = {"scenes": [], "solvers": [], "kalman": [], "iou": []}

    # 1. Geometry, on cases chosen to include the awkward ones: identical boxes,
    #    right angles, pure vertical separation, and near-misses.
    rng = np.random.default_rng(17)
    pairs = [
        ([0, 0, 0, 0, 4, 2, 1.5], [0, 0, 0, 0, 4, 2, 1.5]),
        ([0, 0, 0, 0, 4, 2, 1.5], [0, 0, 0, np.pi / 2, 4, 2, 1.5]),
        ([0, 0, 0, 0, 4, 2, 1.5], [2, 0, 0, 0, 4, 2, 1.5]),
        ([0, 0, 0, 0, 4, 2, 1.5], [0, 0, 5, 0, 4, 2, 1.5]),
        ([0, 0, 0, 0, 4, 2, 1.5], [3.999, 0, 0, 0, 4, 2, 1.5]),
    ]
    for _ in range(40):
        pairs.append((
            list(np.concatenate([rng.uniform(-5, 5, 3), rng.uniform(-np.pi, np.pi, 1), rng.uniform(1, 5, 3)])),
            list(np.concatenate([rng.uniform(-5, 5, 3), rng.uniform(-np.pi, np.pi, 1), rng.uniform(1, 5, 3)])),
        ))
    ref["iou"] = [{"a": list(map(float, a)), "b": list(map(float, b)), "iou": iou_3d(np.array(a), np.array(b))}
                  for a, b in pairs]

    # 2. Both solvers, including a matrix of exact ties -- where any difference
    #    in tie-breaking between the two languages shows up immediately.
    mats = [np.ones((4, 4)), np.array([[1.0, 2.0], [1.0, 9.0]])]
    for _ in range(30):
        n, m = int(rng.integers(1, 7)), int(rng.integers(1, 7))
        mats.append(np.round(rng.random((n, m)) * 10, 6))
    ref["solvers"] = [
        {"cost": M.tolist(), "hungarian": hungarian(M), "greedy": greedy(M)} for M in mats
    ]

    # 3. The filter, stepped long enough that the covariance recursion matters.
    kf = KalmanBox(np.array([0.0, 0.0, 0.0, 0.1, 4.0, 2.0, 1.5]))
    steps = []
    for t in range(1, 26):
        kf.predict()
        z = np.array([0.3 * t, 0.1 * t, 0.0, 0.1 + 0.02 * t, 4.0, 2.0, 1.5])
        steps.append({"z": z.tolist(), "box": kf.update(z).tolist(), "v": kf.velocity().tolist()})
    ref["kalman"] = steps

    # 4. Scenes end to end: the generated data, the identities the tracker
    #    assigns frame by frame, and the resulting counts.
    for case in CASES:
        sc = make_scene(case["name"], seed=case["seed"], noise=case["noise"])
        frames = [{
            "gt": {str(k): v.tolist() for k, v in f.gt.items()},
            "det": [{"box": d.box.tolist(), "score": d.score} for d in f.detections],
        } for f in sc.frames]
        runs = []
        for cfg in CONFIGS:
            outputs = run(sc, TrackerConfig(**cfg))
            runs.append({
                "config": cfg,
                "ids": [sorted(int(i) for i in o.boxes) for o in outputs],
                "report": clear_mot(sc, outputs, metric="dist"),
            })
        ref["scenes"].append({"case": case, "frames": frames, "runs": runs})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(ref))
    n = sum(len(s["frames"]) for s in ref["scenes"])
    print(f"wrote {OUT.relative_to(OUT.parents[1])}: {len(ref['scenes'])} scenes, {n} frames, "
          f"{len(ref['solvers'])} cost matrices, {len(ref['iou'])} box pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
