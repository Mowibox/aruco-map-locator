"""EKF implementation for differential drive robot pose smoothing.

@file        kalman_filter.py
@author      Mowibox (Ousmane THIONGANE)
@version     1.0
@date        2026-02-24
"""

# Imports
import numpy as np
from numpy.typing import NDArray


class KalmanFilter:
    """
    Extended Kalman Filter for differential drive robot pose tracking.

    State vector (5D):  µ = [x, y, theta, v, omega]
    Measurement vector (3D):  z = [x, y, theta]

    Transition function:
        x_{t+1} = x_t + v_t * cos(theta_t) * dt
        y_{t+1} = y_t + v_t * sin(theta_t) * dt
        θ_{t+1} = θ_t + ω_t * dt
        v_{t+1} = v_t
        ω_{t+1} = ω_t
    """

    def __init__(self, dt: float = 1 / 30) -> None:
        """Initialize the EKF.

        @param dt: Time step (in seconds)
        """
        self.dt = dt
        self.initialized = False

        # State vector [x, y, theta, v, omega]
        self.mu: NDArray[np.float64] = np.zeros((5, 1), dtype=np.float64)

        # State covariance
        self.Sigma: NDArray[np.float64] = np.diag([1e-3, 1e-3, 1e-1, 1e-2, 1e-2])

        # Process noise
        self.Sigma_x: NDArray[np.float64] = np.diag([1e-4, 1e-4, 1e-3, 5e-3, 5e-3])

        # Measurement noise
        self.Sigma_z: NDArray[np.float64] = np.diag([1e-4, 1e-4, 1e-2])

        # Observation matrix
        self.C: NDArray[np.float64] = np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap an angle to the range [-pi, pi].

        @param angle: The angle to wrap (rad)
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _f(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Transition function f(x).

        @param mu: Current state [x, y, theta, v, omega]
        """
        x, y, theta, v, omega = mu.flatten()
        dt = self.dt
        return np.array(
            [
                [x + v * np.cos(theta) * dt],
                [y + v * np.sin(theta) * dt],
                [theta + omega * dt],
                [v],
                [omega],
            ],
            dtype=np.float64,
        )

    def _jacobian_A(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Jacobian of f(x) wrt state.

        @param mu: Current state [x, y, theta, v, omega]
        """
        _, _, theta, v, _ = mu.flatten()
        dt = self.dt
        return np.array(
            [
                [1, 0, -v * np.sin(theta) * dt, np.cos(theta) * dt, 0],
                [0, 1, v * np.cos(theta) * dt, np.sin(theta) * dt, 0],
                [0, 0, 1, 0, dt],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ],
            dtype=np.float64,
        )

    def update(self, x: float, y: float, theta: float) -> tuple[float, float, float]:
        """Update the EKF with a new measurement.

        @param x: Measured x position (m)
        @param y: Measured y position (m)
        @param theta: Measured orientation (rad)
        """
        if not self.initialized:
            self.mu = np.array([[x], [y], [theta], [0.0], [0.0]], dtype=np.float64)
            self.initialized = True
            return x, y, theta

        # Predict
        A_t = self._jacobian_A(self.mu)
        self.mu = self._f(self.mu)
        self.mu[2, 0] = self._wrap_angle(self.mu[2, 0])
        self.Sigma = A_t @ self.Sigma @ A_t.T + self.Sigma_x

        # Update: Conditionning
        z_t = np.array([[x], [y], [theta]], dtype=np.float64)

        mu_z = self.C @ self.mu
        innovation = z_t - mu_z
        innovation[2, 0] = self._wrap_angle(float(innovation[2, 0]))

        S_t = self.Sigma_z + self.C @ self.Sigma @ self.C.T
        K_t = self.Sigma @ self.C.T @ np.linalg.inv(S_t)
        self.mu = self.mu + K_t @ innovation
        self.mu[2, 0] = self._wrap_angle(self.mu[2, 0])
        self.Sigma = (np.eye(5) - K_t @ self.C) @ self.Sigma

        return float(self.mu[0]), float(self.mu[1]), self._wrap_angle(float(self.mu[2]))
