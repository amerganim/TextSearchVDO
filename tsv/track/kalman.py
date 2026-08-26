"""Constant-velocity Kalman filter over `(cx, cy, aspect, height)`.

The SORT/DeepSORT formulation. It matters more here than in a typical
benchmark: frames are sampled a few per second rather than at 25 fps, so a
walking person can move most of their own width between two samples and raw
IoU between consecutive detections collapses. Predicting where the box should
have gone is what keeps association working at that spacing.

Height, not area, carries the scale: a person's height is stable while their
width swings wildly as arms and legs move.
"""

from __future__ import annotations

import numpy as np

# Uncertainty is modelled as proportional to object height - a box twice as
# large is twice as uncertain in pixels. Standard DeepSORT weights.
STD_POSITION = 1.0 / 20
STD_VELOCITY = 1.0 / 160

_NDIM = 4


def _motion_matrix(dt: float = 1.0) -> np.ndarray:
    f = np.eye(2 * _NDIM, dtype=np.float64)
    for i in range(_NDIM):
        f[i, _NDIM + i] = dt
    return f


_UPDATE_MATRIX = np.eye(_NDIM, 2 * _NDIM, dtype=np.float64)


class KalmanFilter:
    def __init__(self, dt: float = 1.0) -> None:
        self._motion = _motion_matrix(dt)

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Start a track from one observation, with zero initial velocity."""
        mean = np.concatenate([measurement, np.zeros(_NDIM)])
        height = measurement[3]
        std = np.array([
            2 * STD_POSITION * height,
            2 * STD_POSITION * height,
            1e-2,
            2 * STD_POSITION * height,
            10 * STD_VELOCITY * height,
            10 * STD_VELOCITY * height,
            1e-5,
            10 * STD_VELOCITY * height,
        ])
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        height = mean[3]
        std = np.array([
            STD_POSITION * height, STD_POSITION * height, 1e-2, STD_POSITION * height,
            STD_VELOCITY * height, STD_VELOCITY * height, 1e-5, STD_VELOCITY * height,
        ])
        motion_cov = np.diag(np.square(std))
        mean = self._motion @ mean
        covariance = self._motion @ covariance @ self._motion.T + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        height = mean[3]
        std = np.array([
            STD_POSITION * height, STD_POSITION * height, 1e-1, STD_POSITION * height,
        ])
        innovation_cov = np.diag(np.square(std))
        projected_mean = _UPDATE_MATRIX @ mean
        projected_cov = _UPDATE_MATRIX @ covariance @ _UPDATE_MATRIX.T
        return projected_mean, projected_cov + innovation_cov

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        projected_mean, projected_cov = self.project(mean, covariance)

        # Solve rather than invert: the projected covariance is small but can
        # be poorly conditioned when a box has barely moved for many frames.
        kalman_gain = np.linalg.solve(
            projected_cov.T, (covariance @ _UPDATE_MATRIX.T).T
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + innovation @ kalman_gain.T
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_covariance
