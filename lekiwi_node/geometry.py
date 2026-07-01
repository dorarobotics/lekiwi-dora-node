from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float  # radians


@dataclass(frozen=True)
class Twist:
    vx: float   # m/s, body frame
    vy: float   # m/s, body frame (holonomic lateral)
    omega: float  # rad/s


def yaw_from_quat(qw: float, qx: float, qy: float, qz: float) -> float:
    """Z-axis yaw (radians) from a quaternion (w, x, y, z)."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(a: float) -> float:
    """Fold an angle into (-pi, pi]."""
    a = a % (2.0 * math.pi)
    if a > math.pi:
        a -= 2.0 * math.pi
    return a
