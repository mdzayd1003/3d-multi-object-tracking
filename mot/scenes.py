"""Synthetic driving scenes and a detector that behaves like a real one.

Ground truth is generated first and detections are derived from it, so every
error the tracker makes is attributable: a miss is a miss because the detector
dropped the box, not because the scene was ambiguous.

The scenes are built around crossings on purpose.  Association only becomes
interesting when two objects are close enough that the cost matrix has to
choose, and in a scene of well-separated objects every solver and every cost
scores identically -- which is the most common way a tracking benchmark ends up
measuring nothing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .rng import XorShift32, mix

__all__ = ["Detection", "Frame", "Scene", "make_scene", "SCENES"]

# (n_objects, lane spacing in metres, how many pairs deliberately cross)
SCENES = {
    "sparse":   dict(n=6,  spacing=12.0, crossings=0),
    "crossing": dict(n=8,  spacing=6.0,  crossings=3),
    "dense":    dict(n=16, spacing=3.0,  crossings=6),
}


@dataclass
class Detection:
    box: np.ndarray
    score: float
    gt_id: int      # -1 for a false positive; kept only for diagnostics


@dataclass
class Frame:
    index: int
    gt: dict[int, np.ndarray] = field(default_factory=dict)
    detections: list[Detection] = field(default_factory=list)


@dataclass
class Scene:
    name: str
    frames: list[Frame]
    n_objects: int
    dt: float

    def total_gt(self) -> int:
        return sum(len(f.gt) for f in self.frames)


def _trajectory(rng: XorShift32, lane: float, crosses: bool, n_frames: int, dt: float,
                spacing: float):
    """One object's ground-truth path.

    A crossing object is given a lateral drift that carries it through at least
    one neighbouring lane over the sequence; everything else holds its lane with
    a little wander.  The drift is scaled by lane spacing and duration rather
    than fixed, because a "crossing" that covers less than one lane width never
    actually changes the lateral ordering -- and then no solver and no cost is
    ever forced to choose, which makes the whole comparison vacuous.
    `test_scenes.py::test_crossing_objects_actually_cross` holds this honest.
    """
    x0 = rng.uniform(-40.0, -10.0)
    speed = rng.uniform(4.0, 11.0)
    length = rng.uniform(3.6, 4.9)
    width = rng.uniform(1.6, 2.1)
    height = rng.uniform(1.4, 1.8)
    z = height / 2.0
    duration = max(n_frames * dt, 1e-6)
    if crosses:
        # Cover 1.2 to 2.4 lane widths end to end, so the ordering really flips.
        lanes = rng.uniform(1.2, 2.4)
        lateral_rate = (spacing * lanes / duration) * (1 if rng.random() < 0.5 else -1)
    else:
        lateral_rate = 0.0
    wander = rng.uniform(0.0, 0.15)
    phase = rng.uniform(0.0, 2.0 * math.pi)

    path = []
    for t in range(n_frames):
        tt = t * dt
        x = x0 + speed * tt
        y = lane + lateral_rate * (tt - n_frames * dt / 2.0) + wander * math.sin(phase + tt)
        # Heading follows the actual direction of travel, so a crossing object
        # really is turned, and an IoU that ignores yaw would be wrong.
        dy = lateral_rate + wander * math.cos(phase + tt)
        yaw = math.atan2(dy, speed)
        path.append(np.array([x, y, z, yaw, length, width, height]))
    return path


def make_scene(
    name: str = "crossing",
    n_frames: int = 60,
    seed: int = 7,
    dt: float = 0.1,
    miss_rate: float = 0.12,
    fp_per_frame: float = 0.6,
    noise: float = 0.25,
) -> Scene:
    """Build a scene and run the simulated detector over it.

    `noise` is the standard deviation in metres applied to the centre; the same
    figure scaled down is applied to extent and heading.  `miss_rate` and
    `fp_per_frame` are the detector's recall and clutter knobs.
    """
    if name not in SCENES:
        raise ValueError(f"unknown scene {name!r}; have {sorted(SCENES)}")
    cfg = SCENES[name]
    rng = XorShift32(mix(seed, 1013))

    n = cfg["n"]
    crossing_ids = set(range(min(cfg["crossings"], n)))
    tracks = {
        i: _trajectory(rng, (i - n / 2.0) * cfg["spacing"], i in crossing_ids, n_frames, dt,
                       cfg["spacing"])
        for i in range(n)
    }

    frames = []
    for t in range(n_frames):
        frame = Frame(index=t)
        for oid, path in tracks.items():
            frame.gt[oid] = path[t]
        for oid, path in tracks.items():
            if rng.random() < miss_rate:
                continue
            truth = path[t]
            box = truth.copy()
            box[0] += rng.normal(0.0, noise)
            box[1] += rng.normal(0.0, noise)
            box[2] += rng.normal(0.0, noise * 0.4)
            box[3] += rng.normal(0.0, noise * 0.15)
            box[4:7] += np.array([rng.normal(0.0, noise * 0.25) for _ in range(3)])
            box[4:7] = np.maximum(box[4:7], 0.5)
            # Score correlates with localisation quality, which is what makes
            # the confidence sweep in AMOTA mean anything.
            err = math.hypot(box[0] - truth[0], box[1] - truth[1])
            score = max(0.05, min(0.99, 0.95 - 0.45 * err + rng.normal(0.0, 0.05)))
            frame.detections.append(Detection(box=box, score=score, gt_id=oid))
        n_fp = int(fp_per_frame) + (1 if rng.random() < (fp_per_frame % 1.0) else 0)
        for _ in range(n_fp):
            box = np.array([
                rng.uniform(-45.0, 45.0),
                rng.uniform(-n * cfg["spacing"] / 2.0 - 5.0, n * cfg["spacing"] / 2.0 + 5.0),
                0.8, rng.uniform(-math.pi, math.pi),
                rng.uniform(3.0, 5.0), rng.uniform(1.5, 2.2), rng.uniform(1.4, 1.8),
            ])
            # False positives are low-scoring on average but overlap the true
            # range, so no threshold cleanly separates them.
            frame.detections.append(
                Detection(box=box, score=max(0.05, min(0.9, rng.normal(0.28, 0.14))), gt_id=-1)
            )
        rng_order = sorted(range(len(frame.detections)), key=lambda k: -frame.detections[k].score)
        frame.detections = [frame.detections[k] for k in rng_order]
        frames.append(frame)

    return Scene(name=name, frames=frames, n_objects=n, dt=dt)
