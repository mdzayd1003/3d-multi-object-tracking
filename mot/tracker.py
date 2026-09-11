"""Tracking-by-detection: predict, associate, update, birth and death.

The lifecycle rules are the SORT ones -- a track is only reported once it has
been confirmed by `min_hits` consecutive detections, and it is deleted after
`max_age` frames without one -- because they are the ones whose failure modes
the experiments here are about.  `max_age` in particular is a straight trade of
ID switches against false negatives, and `experiments.age_tradeoff` measures the
exchange rate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .assign import associate
from .boxes import cost_matrix
from .kalman import KalmanBox

__all__ = ["Track", "Tracker", "TrackerConfig", "run"]

# Gate thresholds are in the units of the cost: 1 - IoU is unitless and 0.999
# admits any pair that overlaps at all; distance is metres.
DEFAULT_GATE = {"iou": 0.999, "dist": 4.0}


@dataclass
class TrackerConfig:
    metric: str = "dist"
    solver: str = "hungarian"
    gate: float | None = None
    max_age: int = 3
    min_hits: int = 2
    score_threshold: float = 0.0
    dt: float = 0.1

    def gate_value(self) -> float:
        return DEFAULT_GATE[self.metric] if self.gate is None else self.gate


@dataclass
class Track:
    track_id: int
    kf: KalmanBox
    hits: int = 1
    age: int = 0            # frames since the last matched detection
    score: float = 0.0
    confirmed: bool = False
    history: list = field(default_factory=list)

    def box(self) -> np.ndarray:
        return self.kf.box()


@dataclass
class Output:
    frame: int
    boxes: dict[int, np.ndarray]   # track id -> box
    scores: dict[int, float]


class Tracker:
    def __init__(self, cfg: TrackerConfig | None = None):
        self.cfg = cfg or TrackerConfig()
        self.tracks: list[Track] = []
        self._next_id = 1

    def step(self, detections) -> Output:
        cfg = self.cfg
        dets = [d for d in detections if d.score >= cfg.score_threshold]

        for tr in self.tracks:
            tr.kf.predict()
            tr.age += 1

        pred = np.array([t.box() for t in self.tracks]).reshape(-1, 7)
        obs = np.array([d.box for d in dets]).reshape(-1, 7)
        if len(self.tracks) and len(dets):
            cost = cost_matrix(pred, obs, cfg.metric)
            matched, unmatched_t, unmatched_d = associate(cost, cfg.gate_value(), cfg.solver)
        else:
            matched, unmatched_t, unmatched_d = [], list(range(len(self.tracks))), list(range(len(dets)))

        for ti, di in matched:
            tr = self.tracks[ti]
            tr.kf.update(dets[di].box)
            tr.hits += 1
            tr.age = 0
            tr.score = dets[di].score
            if tr.hits >= cfg.min_hits:
                tr.confirmed = True

        for di in unmatched_d:
            self.tracks.append(
                Track(
                    track_id=self._next_id,
                    kf=KalmanBox(dets[di].box, dt=cfg.dt),
                    score=dets[di].score,
                    confirmed=cfg.min_hits <= 1,
                )
            )
            self._next_id += 1

        self.tracks = [t for t in self.tracks if t.age <= cfg.max_age]

        # A track coasting on prediction alone is still reported -- that is the
        # whole point of max_age -- but only once it has been confirmed.
        boxes = {t.track_id: t.box() for t in self.tracks if t.confirmed}
        scores = {t.track_id: t.score for t in self.tracks if t.confirmed}
        return Output(frame=-1, boxes=boxes, scores=scores)


def run(scene, cfg: TrackerConfig | None = None) -> list[Output]:
    """Track a whole scene and return one Output per frame."""
    tracker = Tracker(cfg)
    out = []
    for frame in scene.frames:
        o = tracker.step(frame.detections)
        o.frame = frame.index
        out.append(o)
    return out
