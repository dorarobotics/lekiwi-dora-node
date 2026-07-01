import math
from lekiwi_node.geometry import Pose2D, Twist, wrap_angle, yaw_from_quat


def test_yaw_from_identity_quat_is_zero():
    assert abs(yaw_from_quat(1.0, 0.0, 0.0, 0.0)) < 1e-9


def test_yaw_from_90deg_about_z():
    c = math.cos(math.pi / 4)
    assert abs(yaw_from_quat(c, 0.0, 0.0, c) - math.pi / 2) < 1e-6


def test_wrap_angle_folds_into_pi():
    assert abs(wrap_angle(3 * math.pi) - math.pi) < 1e-9
    assert abs(wrap_angle(-3 * math.pi) - math.pi) < 1e-9


def test_pose_and_twist_are_frozen_dataclasses():
    p = Pose2D(1.0, 2.0, 0.5)
    t = Twist(0.1, -0.2, 0.3)
    assert (p.x, p.y, p.yaw) == (1.0, 2.0, 0.5)
    assert (t.vx, t.vy, t.omega) == (0.1, -0.2, 0.3)
