"""Constant-velocity Kalman filter over an oriented 3D box.

State is [x, y, z, yaw, l, w, h, vx, vy, vz]: the box, plus translational
velocity.  Heading rate and size rate are deliberately absent -- with one
measurement per frame they are barely observable, and letting size drift is a
common way to make IoU-based association worse than it needs to be.

The one subtlety is yaw.  It is an angle, so the innovation has to be wrapped
into [-pi, pi] before it is used; without that, a track whose heading crosses
the +/-pi branch cut sees a residual of nearly 2*pi and the filter throws the
box across the map.  `test_kalman.py::test_yaw_wraparound` pins that.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["wrap_angle", "KalmanBox"]

DIM_X = 10
DIM_Z = 7


def wrap_angle(a: float) -> float:
    """Fold an angle into [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _transition(dt: float) -> np.ndarray:
    F = np.eye(DIM_X)
    F[0, 7] = dt
    F[1, 8] = dt
    F[2, 9] = dt
    return F


class KalmanBox:
    """One track's filter.

    `pos_var` / `vel_var` set the initial covariance, `q` the process noise and
    `r` the measurement noise.  The defaults are tuned for the synthetic scenes
    in `scenes.py` (metres, 10 Hz) and are not claimed to transfer.
    """

    def __init__(
        self,
        box: np.ndarray,
        dt: float = 0.1,
        pos_var: float = 1.0,
        vel_var: float = 100.0,
        q: float = 0.05,
        r: float = 0.35,
    ):
        box = np.asarray(box, dtype=float).reshape(DIM_Z)
        self.dt = float(dt)
        self.x = np.zeros(DIM_X)
        self.x[:DIM_Z] = box
        self.F = _transition(dt)
        self.H = np.zeros((DIM_Z, DIM_X))
        self.H[:, :DIM_Z] = np.eye(DIM_Z)
        self.P = np.diag([pos_var] * DIM_Z + [vel_var] * 3)
        self.Q = np.diag([q] * DIM_Z + [q * 10.0] * 3)
        self.R = np.eye(DIM_Z) * r

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.x[3] = wrap_angle(self.x[3])
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.box()

    def update(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float).reshape(DIM_Z)
        y = z - self.H @ self.x
        y[3] = wrap_angle(y[3])          # see the module docstring
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[3] = wrap_angle(self.x[3])
        # Joseph form: stays symmetric positive-definite under the repeated
        # predict/update cycling a long track does, where (I - KH)P does not.
        I_KH = np.eye(DIM_X) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return self.box()

    def box(self) -> np.ndarray:
        b = self.x[:DIM_Z].copy()
        b[4:7] = np.maximum(b[4:7], 1e-3)   # an extent must stay positive
        return b

    def velocity(self) -> np.ndarray:
        return self.x[7:10].copy()
