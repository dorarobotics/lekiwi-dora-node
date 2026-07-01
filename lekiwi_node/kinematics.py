from __future__ import annotations

import numpy as np

from lekiwi_node.geometry import Twist


class KiwiDrive:
    """3-wheel omnidirectional (kiwi) kinematics, ported from LeRobot lekiwi.py.

    Wheels mounted at body angles [150, -90, 30] deg (= radians([240,0,120]) - 90).
    body_to_wheels: (vx, vy, omega) [m/s, m/s, rad/s] -> 3 wheel angular speeds [rad/s].
    wheels_to_body: inverse. The 3x3 mount matrix is invertible, so it round-trips.
    """

    def __init__(self, wheel_radius: float = 0.05, base_radius: float = 0.125,
                 max_wheel_omega: float = 30.0) -> None:
        self.wheel_radius = wheel_radius
        self.base_radius = base_radius
        self.max_wheel_omega = max_wheel_omega
        angles = np.radians(np.array([240.0, 0.0, 120.0]) - 90.0)  # [150, -90, 30] deg
        self._m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])
        self._m_inv = np.linalg.inv(self._m)

    def body_to_wheels(self, vx: float, vy: float, omega: float) -> tuple[float, float, float]:
        # Clamping is applied per-wheel independently, so under saturation the achieved
        # body velocity differs in direction from the command (a safety backstop, not a
        # motion primitive).
        wheel_linear = self._m.dot(np.array([vx, vy, omega], dtype=float))
        wheel_omega = wheel_linear / self.wheel_radius
        clamped = np.clip(wheel_omega, -self.max_wheel_omega, self.max_wheel_omega)
        return (float(clamped[0]), float(clamped[1]), float(clamped[2]))

    def wheels_to_body(self, w1: float, w2: float, w3: float) -> Twist:
        wheel_linear = np.array([w1, w2, w3], dtype=float) * self.wheel_radius
        vx, vy, omega = self._m_inv.dot(wheel_linear)
        return Twist(float(vx), float(vy), float(omega))
