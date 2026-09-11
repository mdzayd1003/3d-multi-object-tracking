import math

import numpy as np
import pytest

from mot.boxes import (center_distance, convex_intersection, corners_bev, cost_matrix,
                       iou_3d, poly_area)

CAR = np.array([0.0, 0.0, 0.0, 0.0, 4.0, 2.0, 1.5])


def test_self_iou_is_one():
    assert iou_3d(CAR, CAR) == pytest.approx(1.0)


def test_rotated_ninety_degrees_is_exactly_one_third():
    # A 4x2 box and a 2x4 box share a 2x2 square: 4 / (8 + 8 - 4).
    b = CAR.copy()
    b[3] = math.pi / 2
    assert iou_3d(CAR, b) == pytest.approx(1.0 / 3.0)


def test_half_length_shift_is_exactly_one_third():
    b = CAR.copy()
    b[0] = 2.0
    assert iou_3d(CAR, b) == pytest.approx(1.0 / 3.0)


def test_disjoint_boxes_are_zero():
    b = CAR.copy()
    b[0] = 9.0
    assert iou_3d(CAR, b) == 0.0


def test_vertical_separation_alone_zeroes_the_iou():
    # Same footprint, stacked out of contact: BEV overlap is total, IoU is not.
    b = CAR.copy()
    b[2] = 5.0
    assert iou_3d(CAR, b) == 0.0


def test_iou_is_symmetric():
    rng = np.random.default_rng(3)
    for _ in range(200):
        a = np.concatenate([rng.uniform(-4, 4, 3), rng.uniform(-math.pi, math.pi, 1),
                            rng.uniform(1, 5, 3)])
        b = np.concatenate([rng.uniform(-4, 4, 3), rng.uniform(-math.pi, math.pi, 1),
                            rng.uniform(1, 5, 3)])
        assert iou_3d(a, b) == pytest.approx(iou_3d(b, a), abs=1e-12)


def test_iou_stays_in_range():
    rng = np.random.default_rng(4)
    for _ in range(300):
        a = np.concatenate([rng.uniform(-6, 6, 3), rng.uniform(-math.pi, math.pi, 1),
                            rng.uniform(0.5, 6, 3)])
        b = np.concatenate([rng.uniform(-6, 6, 3), rng.uniform(-math.pi, math.pi, 1),
                            rng.uniform(0.5, 6, 3)])
        assert 0.0 <= iou_3d(a, b) <= 1.0 + 1e-12


def test_corners_and_area_agree_with_the_extent():
    c = corners_bev(CAR)
    assert c.shape == (4, 2)
    assert poly_area(c) == pytest.approx(8.0)


def test_yaw_does_not_change_area():
    b = CAR.copy()
    b[3] = 0.7
    assert poly_area(corners_bev(b)) == pytest.approx(8.0)


def test_clipping_a_square_by_a_larger_square_returns_the_square():
    small = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    big = small * 3
    assert poly_area(convex_intersection(small, big)) == pytest.approx(4.0)


def test_center_distance_ignores_height():
    b = CAR.copy()
    b[0], b[1], b[2] = 3.0, 4.0, 100.0
    assert center_distance(CAR, b) == pytest.approx(5.0)


def test_cost_matrix_shapes_and_polarity():
    tracks = np.stack([CAR, CAR + np.array([20, 0, 0, 0, 0, 0, 0])])
    dets = np.stack([CAR])
    c_iou = cost_matrix(tracks, dets, "iou")
    c_dist = cost_matrix(tracks, dets, "dist")
    assert c_iou.shape == c_dist.shape == (2, 1)
    # The matching pair is cheaper under both costs.
    assert c_iou[0, 0] < c_iou[1, 0]
    assert c_dist[0, 0] < c_dist[1, 0]
    # And the non-overlapping pair costs exactly 1.0 under IoU -- no gradient.
    assert c_iou[1, 0] == 1.0


def test_unknown_metric_is_rejected():
    with pytest.raises(ValueError):
        cost_matrix(np.stack([CAR]), np.stack([CAR]), "manhattan")
