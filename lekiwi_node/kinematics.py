from __future__ import annotations

from drive_kinematics import Kiwi

from lekiwi_node.geometry import Twist


class KiwiDrive:
    """Adapter over drive_kinematics.Kiwi, preserving the lekiwi KiwiDrive API.
    The kinematics live in the shared drive-kinematics library (sub-project A)."""

    def __init__(self, wheel_radius: float = 0.05, base_radius: float = 0.125,
                 max_wheel_omega: float = 30.0) -> None:
        self._k = Kiwi(wheel_radius=wheel_radius, base_radius=base_radius,
                       max_wheel_speed=max_wheel_omega)
        # Retained for API compatibility only — read-only snapshots of the ctor args;
        # mutating them does NOT change the active clamp (the kinematics live in self._k).
        self.wheel_radius = wheel_radius
        self.base_radius = base_radius
        self.max_wheel_omega = max_wheel_omega

    def body_to_wheels(self, vx: float, vy: float, omega: float) -> tuple[float, float, float]:
        w = self._k.body_to_wheels(vx, vy, omega)
        return (w[0], w[1], w[2])

    def wheels_to_body(self, w1: float, w2: float, w3: float) -> Twist:
        t = self._k.wheels_to_body(w1, w2, w3)
        return Twist(t.vx, t.vy, t.omega)
