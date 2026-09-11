"""CLEAR MOT and the recall-averaged AMOTA/AMOTP.

Two things here are easy to get wrong and are therefore written out explicitly.

First, MOTA's identity accounting.  An ID switch is only counted when a ground
truth object that *was* matched to some hypothesis becomes matched to a
different one.  Getting that right requires carrying the previous frame's
mapping and preferring it before running the solver -- if you simply re-solve
every frame from scratch, two equally-good assignments alternate and the ID
switch count becomes a function of solver tie-breaking rather than of tracking
quality.  `test_metrics.py::test_sticky_matching_suppresses_phantom_switches`
pins the difference.

Second, AMOTA.  MOTA is reported at one operating point; AMOTA integrates over
recall by sweeping the confidence threshold, and the two can rank trackers
differently because a configuration tuned to look good at one threshold has no
obligation to hold up across the sweep.  That disagreement is measured in
`experiments.amota_disagreement` rather than asserted.
"""
from __future__ import annotations

import math

import numpy as np

from .assign import associate
from .boxes import center_distance, iou_3d

__all__ = ["MOTAccumulator", "clear_mot", "amota", "DEFAULT_MATCH"]

# Matching an output to ground truth is a separate decision from the tracker's
# own association, and is fixed at 2 m so that trackers using different internal
# costs are still scored on the same footing.
DEFAULT_MATCH = {"dist": 2.0, "iou": 0.75}


def _pair_cost(a, b, metric: str) -> float:
    return center_distance(a, b) if metric == "dist" else 1.0 - iou_3d(a, b)


class MOTAccumulator:
    """Frame-by-frame CLEAR MOT bookkeeping."""

    def __init__(self, metric: str = "dist", threshold: float | None = None):
        self.metric = metric
        self.threshold = DEFAULT_MATCH[metric] if threshold is None else threshold
        self.fp = 0
        self.fn = 0
        self.ids = 0
        self.matches = 0
        self.dist_sum = 0.0
        self.gt_total = 0
        self._prev: dict[int, int] = {}     # gt id -> hypothesis id

    def update(self, gt: dict, hyp: dict) -> None:
        self.gt_total += len(gt)
        gt_ids = sorted(gt)
        hyp_ids = sorted(hyp)

        # Step 1: keep last frame's pairs that are still valid, before solving.
        mapping: dict[int, int] = {}
        used_hyp = set()
        for g in gt_ids:
            h = self._prev.get(g)
            if h is not None and h in hyp and h not in used_hyp:
                c = _pair_cost(gt[g], hyp[h], self.metric)
                if c <= self.threshold:
                    mapping[g] = h
                    used_hyp.add(h)
                    self.matches += 1
                    self.dist_sum += c

        # Step 2: solve for whatever is left.
        free_gt = [g for g in gt_ids if g not in mapping]
        free_hyp = [h for h in hyp_ids if h not in used_hyp]
        if free_gt and free_hyp:
            cost = np.array(
                [[_pair_cost(gt[g], hyp[h], self.metric) for h in free_hyp] for g in free_gt]
            )
            pairs, _, _ = associate(cost, self.threshold, "hungarian")
            for gi, hi in pairs:
                g, h = free_gt[gi], free_hyp[hi]
                mapping[g] = h
                used_hyp.add(h)
                self.matches += 1
                self.dist_sum += cost[gi, hi]

        # Step 3: an object that was matched before and is matched to someone
        # else now is an ID switch.  Losing the object entirely is a miss, not a
        # switch, and is only charged as a switch if it comes back differently.
        for g, h in mapping.items():
            prev = self._prev.get(g)
            if prev is not None and prev != h:
                self.ids += 1

        self.fn += len(gt) - len(mapping)
        self.fp += len(hyp) - len(used_hyp)
        # Carry forward the objects that vanished this frame, so a track that
        # recovers the same identity after an occlusion is not charged.
        carried = dict(self._prev)
        carried.update(mapping)
        self._prev = {g: h for g, h in carried.items() if g in gt or g in mapping}

    def report(self) -> dict:
        n = max(self.gt_total, 1)
        return {
            "mota": 1.0 - (self.fp + self.fn + self.ids) / n,
            "motp": self.dist_sum / max(self.matches, 1),
            "fp": self.fp,
            "fn": self.fn,
            "ids": self.ids,
            "matches": self.matches,
            "gt": self.gt_total,
            "recall": self.matches / n,
            "precision": self.matches / max(self.matches + self.fp, 1),
        }


def clear_mot(scene, outputs, metric: str = "dist", threshold: float | None = None) -> dict:
    acc = MOTAccumulator(metric, threshold)
    for frame, out in zip(scene.frames, outputs):
        acc.update(frame.gt, out.boxes)
    return acc.report()


def amota(
    scene,
    run_fn,
    recalls=None,
    metric: str = "dist",
    threshold: float | None = None,
) -> dict:
    """Average MOTA/MOTP over a sweep of recall levels (the nuScenes protocol).

    `run_fn(score_threshold)` must return the tracker outputs for that operating
    point.  For each target recall the sweep picks the confidence threshold whose
    achieved recall is closest, then charges the shortfall:

        MOTAR = max(0, 1 - (IDS + FP + FN - (1 - r) * P) / (r * P))

    The `(1 - r) * P` term is what stops a tracker from scoring well simply by
    running at a recall where it makes few mistakes because it reports little.
    It assumes the operating point actually achieves recall `r`, so that misses
    roughly equal the allowance.  A discrete threshold sweep cannot hit every
    target exactly, and when the achieved recall overshoots the target the
    allowance exceeds the misses and the formula hands out credit for recall
    that was never asked for -- which is how MOTAR ends up above 1.  The sweep
    below is fine enough to keep the overshoot small, the result is clipped into
    [0, 1], and `recall_gap` reports the residual mismatch instead of hiding it.
    """
    recalls = list(recalls if recalls is not None else np.arange(1, 11) / 10.0)
    total_gt = scene.total_gt()

    sweep = []
    for s in np.arange(0.0, 0.99, 0.02):
        rep = clear_mot(scene, run_fn(float(s)), metric, threshold)
        sweep.append((float(s), rep))

    motars, motps, points = [], [], []
    for r in recalls:
        # Closest achieved recall; ties go to the higher threshold, which is the
        # more conservative operating point.
        best = min(sweep, key=lambda kv: (abs(kv[1]["recall"] - r), -kv[0]))
        thr, rep = best
        p = total_gt
        denom = r * p
        if denom <= 0:
            continue
        raw = 1.0 - (rep["ids"] + rep["fp"] + rep["fn"] - (1.0 - r) * p) / denom
        motar = min(1.0, max(0.0, raw))
        motars.append(motar)
        motps.append(rep["motp"])
        points.append({
            "recall": r, "threshold": thr, "achieved": rep["recall"],
            "motar": motar, "raw_motar": raw, "recall_gap": rep["recall"] - r,
        })

    return {
        "amota": float(np.mean(motars)) if motars else 0.0,
        "amotp": float(np.mean(motps)) if motps else float("nan"),
        "points": points,
        "max_recall_gap": max((abs(pt["recall_gap"]) for pt in points), default=0.0),
        "sweep": [{"threshold": t, **{k: v for k, v in rep.items() if k != "points"}} for t, rep in sweep],
    }
