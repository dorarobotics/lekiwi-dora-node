from __future__ import annotations

import math

from lekiwi_node.geometry import Pose2D, Twist, wrap_angle


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class HolonomicController:
    """Proportional go-to-pose controller for a holonomic (kiwi) base. Translation
    and yaw are controlled independently — no steering constraint, so lateral,
    reverse, and in-place spin targets are all directly reachable."""

    def __init__(self, *, xy_tol: float = 0.05, yaw_tol: float = 0.05,
                 k_lin: float = 1.5, k_ang: float = 1.5,
                 max_lin: float = 0.5, max_ang: float = 1.5) -> None:
        self.xy_tol = xy_tol
        self.yaw_tol = yaw_tol
        self.k_lin = k_lin
        self.k_ang = k_ang
        self.max_lin = max_lin
        self.max_ang = max_ang

    def step(self, current: Pose2D, target: Pose2D) -> tuple[Twist, bool]:
        dx = target.x - current.x
        dy = target.y - current.y
        dyaw = wrap_angle(target.yaw - current.yaw)
        if math.hypot(dx, dy) <= self.xy_tol and abs(dyaw) <= self.yaw_tol:
            return Twist(0.0, 0.0, 0.0), True
        # rotate world-frame error into the body frame (by -current.yaw)
        cs, sn = math.cos(current.yaw), math.sin(current.yaw)
        ex = cs * dx + sn * dy
        ey = -sn * dx + cs * dy
        # Per-axis saturation distorts direction when both axes clip, but the P loop still converges.
        vx = _clamp(self.k_lin * ex, -self.max_lin, self.max_lin)
        vy = _clamp(self.k_lin * ey, -self.max_lin, self.max_lin)
        omega = _clamp(self.k_ang * dyaw, -self.max_ang, self.max_ang)
        return Twist(vx, vy, omega), False
