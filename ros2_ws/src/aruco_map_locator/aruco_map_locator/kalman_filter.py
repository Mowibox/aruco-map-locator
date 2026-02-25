"""Kalman filter implementation for robot pose smoothing.

@file        kalman_filter.py
@author      Mowibox (Ousmane THIONGANE)
@version     1.0
@date        2026-02-24
"""

# Imports
import cv2
import numpy as np


class KalmanFilter:
    """
    Kalman filter for 2D robot pose tracking.

    State vector (5D): [x, y, vx, vy, theta]
    Measurement vector (3D): [x, y, theta]
    """

    def __init__(self, dt: float = 1 / 30) -> None:
        """Initialize the Kalman filter.

        @param dt: Time step (in seconds)
        """
        self.kf = cv2.KalmanFilter(5, 3)
        self.initialized = False

        # x_t = A * x_{t-1} + w
        self.kf.transitionMatrix = np.array(
            [[1, 0, dt, 0, 0], [0, 1, 0, dt, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]], dtype=np.float32
        )
        # z_t = C * x_t
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1]], dtype=np.float32)

        self.kf.processNoiseCov = np.diag([1e-4, 1e-4, 5e-3, 5e-3, 1e-3]).astype(np.float32)

        self.kf.measurementNoiseCov = np.diag([1e-4, 1e-4, 1e-2]).astype(np.float32)

        self.kf.errorCovPost = np.eye(5, dtype=np.float32) * 1e-2

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap an angle to the range [-pi, pi].

        @param angle: The angle to wrap (rad)
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def update(self, x: float, y: float, theta: float) -> tuple[float, float, float]:
        """Update the Kalman filter with a new measurement.

        @param x: Measured x position (m)
        @param y: Measured y position (m)
        @param theta: Measured orientation (rad)
        """
        if not self.initialized:
            self.kf.statePost = np.array([[x], [y], [0.0], [0.0], [theta]], dtype=np.float32)
            self.initialized = True
            return x, y, theta

        self.kf.predict()

        # Preventing 0/pi discontinuity
        predicted_theta = float(self.kf.statePre[4])
        theta_wrapped = float(predicted_theta) + self._wrap_angle(theta - float(predicted_theta))

        measurement = np.array([[x], [y], [theta_wrapped]], dtype=np.float32)
        corrected = self.kf.correct(measurement)

        return float(corrected[0]), float(corrected[1]), self._wrap_angle(float(corrected[4]))
