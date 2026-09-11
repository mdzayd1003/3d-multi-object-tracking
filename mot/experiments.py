"""The four measurements this repository exists to make.

Each returns plain data, is deterministic given its seed, and is small enough to
run in CI.  The claims they support are stated in the README with the numbers
these produce, not with numbers from a paper.
"""
from __future__ import annotations

import numpy as np

from .boxes import center_distance, iou_3d
from .metrics import amota, clear_mot
from .scenes import make_scene
from .tracker import TrackerConfig, run

__all__ = [
    "cost_ceiling",
    "solver_gap",
    "age_tradeoff",
    "amota_disagreement",
    "ALL",
]

NOISES = (0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def cost_ceiling(scene: str = "crossing", seed: int = 7) -> dict:
    """IoU vs centre distance as localisation error grows, plus why IoU stops.

    Reports three things per noise level: the two trackers' scores, the fraction
    of true (ground truth, detection) pairs that do not overlap at all, and the
    fraction further apart than the distance gate.  The second is the number
    usually blamed for IoU's collapse; the point of measuring it is that it is
    still small where the collapse happens.
    """
    rows = []
    for noise in NOISES:
        sc = make_scene(scene, seed=seed, noise=noise)
        zero_iou = far = total = 0
        for f in sc.frames:
            for d in f.detections:
                if d.gt_id < 0:
                    continue
                total += 1
                zero_iou += iou_3d(f.gt[d.gt_id], d.box) == 0.0
                far += center_distance(f.gt[d.gt_id], d.box) > 4.0
        rows.append({
            "noise": noise,
            "zero_iou_frac": zero_iou / max(total, 1),
            "beyond_dist_gate_frac": far / max(total, 1),
            "iou": clear_mot(sc, run(sc, TrackerConfig(metric="iou"))),
            "dist": clear_mot(sc, run(sc, TrackerConfig(metric="dist"))),
        })

    # The ceiling: sweep each cost's gate at the noise where IoU has collapsed
    # but the geometry has not.  An IoU gate of 0.999 already admits any pair
    # that overlaps at all, so anything beyond it cannot change the answer.
    hard = make_scene(scene, seed=seed, noise=1.0)
    iou_gates = [0.7, 0.9, 0.95, 0.99, 0.999, 0.9999]
    dist_gates = [2.0, 3.0, 4.0, 5.0, 6.0]
    return {
        "scene": scene,
        "rows": rows,
        "gate_sweep": {
            "noise": 1.0,
            "iou": [{"gate": g, **clear_mot(hard, run(hard, TrackerConfig(metric="iou", gate=g)))}
                    for g in iou_gates],
            "dist": [{"gate": g, **clear_mot(hard, run(hard, TrackerConfig(metric="dist", gate=g)))}
                     for g in dist_gates],
        },
    }


def solver_gap(seed: int = 7) -> dict:
    """Hungarian against greedy, and a count of the pairs Hungarian gives up.

    Hungarian minimises the total cost of a complete matching.  The gate is
    applied afterwards, so pairs that will be discarded still pull on the
    optimum -- and Hungarian will happily give up one cheap pair to reduce the
    cost of two pairs that never survive gating.  `sacrificed` counts exactly
    those: pairs greedy kept, Hungarian dropped, and the gate would have passed.
    """
    from .assign import gate as apply_gate
    from .assign import greedy, hungarian
    from .boxes import cost_matrix
    from .tracker import Tracker

    rows = []
    for scene in ("sparse", "crossing", "dense"):
        for noise in (0.5, 1.0):
            sc = make_scene(scene, seed=seed, noise=noise)
            h = clear_mot(sc, run(sc, TrackerConfig(solver="hungarian")))
            g = clear_mot(sc, run(sc, TrackerConfig(solver="greedy")))

            tracker = Tracker(TrackerConfig(metric="dist"))
            disagreed = sacrificed = h_kept = g_kept = 0
            for f in sc.frames:
                pred = np.array([t.box() for t in tracker.tracks]).reshape(-1, 7)
                obs = np.array([d.box for d in f.detections]).reshape(-1, 7)
                if len(pred) and len(obs):
                    C = cost_matrix(pred, obs, "dist")
                    hp, gp = hungarian(C), greedy(C)
                    if sorted(hp) != sorted(gp):
                        disagreed += 1
                        hg = apply_gate(hp, C, 4.0)
                        gg = apply_gate(gp, C, 4.0)
                        h_kept += len(hg)
                        g_kept += len(gg)
                        sacrificed += len(set(gg) - set(hg))
                tracker.step(f.detections)

            rows.append({
                "scene": scene, "noise": noise,
                "hungarian": h, "greedy": g,
                "frames_disagreed": disagreed,
                "gated_pairs_hungarian": h_kept,
                "gated_pairs_greedy": g_kept,
                "sacrificed": sacrificed,
            })
    return {"rows": rows}


def age_tradeoff(scene: str = "crossing", noise: float = 0.6, seed: int = 7) -> dict:
    """What `max_age` actually buys.

    The usual description is that raising it trades false negatives for ID
    switches.  Only the first half of that holds here: past the knee, misses are
    flat and the entire cost of a longer coast is false positives.
    """
    sc = make_scene(scene, seed=seed, noise=noise)
    rows = [{"max_age": ma, **clear_mot(sc, run(sc, TrackerConfig(metric="dist", max_age=ma)))}
            for ma in (0, 1, 2, 3, 5, 8, 12)]
    return {"scene": scene, "noise": noise, "rows": rows}


def amota_disagreement(scene: str = "crossing", noise: float = 0.8, seed: int = 7) -> dict:
    """MOTA at one operating point against AMOTA over the recall sweep.

    Reported because they disagree: the configuration that wins on MOTA with the
    score threshold at zero is not the one that wins on AMOTA.
    """
    sc = make_scene(scene, seed=seed, noise=noise)
    rows = []
    for ma in (1, 3, 8):
        def go(t: float, ma=ma):
            return run(sc, TrackerConfig(metric="dist", max_age=ma, score_threshold=t))
        rows.append({
            "max_age": ma,
            "mota_at_0": clear_mot(sc, go(0.0))["mota"],
            **{k: v for k, v in amota(sc, go).items() if k != "sweep"},
        })
    best_mota = max(rows, key=lambda r: r["mota_at_0"])["max_age"]
    best_amota = max(rows, key=lambda r: r["amota"])["max_age"]
    return {
        "scene": scene, "noise": noise, "rows": rows,
        "best_by_mota": best_mota, "best_by_amota": best_amota,
        "disagree": best_mota != best_amota,
    }


ALL = {
    "cost_ceiling": cost_ceiling,
    "solver_gap": solver_gap,
    "age_tradeoff": age_tradeoff,
    "amota_disagreement": amota_disagreement,
}
