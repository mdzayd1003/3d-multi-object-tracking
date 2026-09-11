"""Oriented 3D boxes and the two association costs the tracker can use.

A box is [x, y, z, yaw, l, w, h]: centre, heading about the vertical axis, and
extent.  z is the centre height, so the vertical span is z +/- h/2.

Two costs live here because the choice between them is the main thing this
repository measures.  3D IoU is the usual one and is exactly 0 whenever the
boxes do not physically overlap; centre distance keeps a gradient out to any
range.  See `experiments.cost_degeneracy` for what that difference is worth.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "corners_bev",
    "poly_area",
    "convex_intersection",
    "iou_3d",
    "center_distance",
    "cost_matrix",
]


def corners_bev(box: np.ndarray) -> np.ndarray:
    """The four bird's-eye-view corners, counter-clockwise, as a (4, 2) array."""
    x, y, _z, yaw, length, width, _h = box
    c, s = math.cos(yaw), math.sin(yaw)
    dx = length / 2.0
    dy = width / 2.0
    local = np.array([[dx, dy], [-dx, dy], [-dx, -dy], [dx, -dy]])
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([x, y])


def poly_area(poly: np.ndarray) -> float:
    """Shoelace area of a simple polygon; 0 for fewer than three vertices."""
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def convex_intersection(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clip of one convex polygon by another.

    Both polygons must be counter-clockwise.  Returns the intersection as a
    (k, 2) array, possibly empty.
    """
    output = [tuple(p) for p in subject]
    n = len(clip)
    for i in range(n):
        if not output:
            return np.zeros((0, 2))
        a = clip[i]
        b = clip[(i + 1) % n]
        # Positive `side` means the point is to the left of a->b, i.e. inside.
        ex, ey = b[0] - a[0], b[1] - a[1]

        def side(p) -> float:
            return ex * (p[1] - a[1]) - ey * (p[0] - a[0])

        current = output
        output = []
        for j, p in enumerate(current):
            q = current[j - 1]
            sp, sq = side(p), side(q)
            if sp >= 0.0:
                if sq < 0.0:
                    t = sq / (sq - sp)
                    output.append((q[0] + t * (p[0] - q[0]), q[1] + t * (p[1] - q[1])))
                output.append(p)
            elif sq >= 0.0:
                t = sq / (sq - sp)
                output.append((q[0] + t * (p[0] - q[0]), q[1] + t * (p[1] - q[1])))
    return np.asarray(output, dtype=float).reshape(-1, 2)


def iou_3d(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two oriented 3D boxes.

    Exact for the vertical axis (boxes are axis-aligned in z) and exact in the
    BEV plane via polygon clipping, so this is the real 3D IoU and not the
    axis-aligned approximation that many trackers use.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    top = min(a[2] + a[6] / 2.0, b[2] + b[6] / 2.0)
    bottom = max(a[2] - a[6] / 2.0, b[2] - b[6] / 2.0)
    dz = top - bottom
    if dz <= 0.0:
        return 0.0
    inter_area = poly_area(convex_intersection(corners_bev(a), corners_bev(b)))
    if inter_area <= 0.0:
        return 0.0
    inter = inter_area * dz
    vol_a = a[4] * a[5] * a[6]
    vol_b = b[4] * b[5] * b[6]
    union = vol_a + vol_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def center_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Bird's-eye-view centre distance, in metres.

    BEV rather than full 3D because height estimates are the noisiest component
    of most detectors and adding them to the cost mostly adds noise.
    """
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def cost_matrix(tracks: np.ndarray, dets: np.ndarray, metric: str) -> np.ndarray:
    """Cost for every (track, detection) pair.  Lower is a better match.

    "iou" returns 1 - IoU, so a non-overlapping pair costs exactly 1.0 and every
    such pair is tied with every other.  "dist" returns metres.
    """
    tracks = np.asarray(tracks, dtype=float).reshape(-1, 7)
    dets = np.asarray(dets, dtype=float).reshape(-1, 7)
    out = np.zeros((len(tracks), len(dets)))
    for i, t in enumerate(tracks):
        for j, d in enumerate(dets):
            if metric == "iou":
                out[i, j] = 1.0 - iou_3d(t, d)
            elif metric == "dist":
                out[i, j] = center_distance(t, d)
            else:
                raise ValueError(f"unknown metric {metric!r}")
    return out
