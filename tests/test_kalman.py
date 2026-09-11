import math

import numpy as np
import pytest

from mot.kalman import KalmanBox, wrap_angle

BOX = np.array([0.0, 0.0, 0.0, 0.0, 4.0, 2.0, 1.5])


def test_wrap_angle_folds_into_the_principal_branch():
    assert wrap_angle(0.0) == pytest.approx(0.0)
    assert wrap_angle(3 * math.pi) == pytest.approx(-math.pi)
    assert wrap_angle(-3 * math.pi) == pytest.approx(-math.pi)
    for a in np.linspace(-20, 20, 200):
        assert -math.pi <= wrap_angle(a) < math.pi + 1e-12


def test_learns_a_constant_velocity_exactly():
    kf = KalmanBox(BOX, dt=0.1)
    for t in range(1, 40):
        kf.predict()
        z = BOX.copy()
        z[0], z[1] = 0.3 * t, 0.1 * t
        kf.update(z)
    assert kf.velocity()[:2] == pytest.approx([3.0, 1.0], abs=1e-6)


def test_predict_extrapolates_along_the_learned_velocity():
    kf = KalmanBox(BOX, dt=0.1)
    for t in range(1, 40):
        kf.predict()
        z = BOX.copy()
        z[0] = 0.5 * t
        kf.update(z)
    before = kf.box()[0]
    after = kf.predict()[0]
    assert after - before == pytest.approx(0.5, abs=1e-3)


def test_yaw_wraparound_does_not_throw_the_box():
    # Heading steps across the +/-pi branch cut. Without wrapping the innovation
    # the residual is nearly 2*pi and the correction drags every other state
    # component with it.
    kf = KalmanBox(np.array([0.0, 0.0, 0.0, math.pi - 0.05, 4.0, 2.0, 1.5]))
    kf.predict()
    box = kf.update(np.array([0.0, 0.0, 0.0, -math.pi + 0.05, 4.0, 2.0, 1.5]))
    assert abs(wrap_angle(box[3])) > math.pi - 0.2      # still pointing backwards
    assert abs(box[0]) < 1e-6 and abs(box[1]) < 1e-6    # position untouched


def test_covariance_stays_symmetric_and_positive_definite():
    # The Joseph form is used precisely so this survives a long track.
    kf = KalmanBox(BOX, dt=0.1)
    rng = np.random.default_rng(11)
    for t in range(500):
        kf.predict()
        z = BOX.copy()
        z[:3] += rng.normal(0, 0.3, 3)
        kf.update(z)
        assert np.allclose(kf.P, kf.P.T, atol=1e-9)
    assert np.min(np.linalg.eigvalsh(kf.P)) > 0.0


def test_extent_never_goes_non_positive():
    kf = KalmanBox(np.array([0.0, 0.0, 0.0, 0.0, 0.6, 0.6, 0.6]))
    for _ in range(50):
        kf.predict()
        kf.update(np.array([0.0, 0.0, 0.0, 0.0, 0.01, 0.01, 0.01]))
    assert np.all(kf.box()[4:7] > 0.0)


def test_a_stationary_object_keeps_zero_velocity():
    kf = KalmanBox(BOX, dt=0.1)
    for _ in range(60):
        kf.predict()
        kf.update(BOX)
    assert kf.velocity() == pytest.approx(np.zeros(3), abs=1e-6)
