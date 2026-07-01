import math
from lekiwi_node.base_controller import HolonomicController
from lekiwi_node.geometry import Pose2D


def _rollout(c, start, target, steps=4000, dt=0.05):
    """Ideal holonomic integration: body Twist rotated into world by current yaw."""
    p = start
    reached = False
    for _ in range(steps):
        tw, reached = c.step(p, target)
        if reached:
            break
        cs, sn = math.cos(p.yaw), math.sin(p.yaw)
        wx = cs * tw.vx - sn * tw.vy
        wy = sn * tw.vx + cs * tw.vy
        p = Pose2D(p.x + wx * dt, p.y + wy * dt, p.yaw + tw.omega * dt)
    return p, reached


def test_reached_within_tolerance_stops():
    c = HolonomicController(xy_tol=0.05, yaw_tol=0.05)
    tw, reached = c.step(Pose2D(1.0, 1.0, 0.0), Pose2D(1.02, 1.0, 0.0))
    assert reached is True
    assert (tw.vx, tw.vy, tw.omega) == (0.0, 0.0, 0.0)


def test_target_left_commands_positive_vy():
    # facing +x, target straight left in world (+y) -> body vy > 0, vx ~ 0.
    c = HolonomicController()
    tw, reached = c.step(Pose2D(0.0, 0.0, 0.0), Pose2D(0.0, 1.0, 0.0))
    assert reached is False
    assert tw.vy > 0 and abs(tw.vx) < 1e-9


def test_target_behind_commands_negative_vx():
    # target straight behind (world -x) -> body vx < 0 (reverse), trivially reachable.
    c = HolonomicController()
    tw, _ = c.step(Pose2D(0.0, 0.0, 0.0), Pose2D(-1.0, 0.0, 0.0))
    assert tw.vx < 0 and abs(tw.vy) < 1e-9


def test_pure_yaw_error_commands_omega_only():
    c = HolonomicController(xy_tol=0.05, yaw_tol=0.02)
    tw, reached = c.step(Pose2D(0.0, 0.0, 0.0), Pose2D(0.0, 0.0, 1.0))
    assert reached is False
    assert abs(tw.vx) < 1e-9 and abs(tw.vy) < 1e-9 and tw.omega > 0


def test_error_is_rotated_into_body_frame():
    # facing +y (yaw=pi/2); target ahead in world +y -> body vx > 0 (forward).
    c = HolonomicController()
    tw, _ = c.step(Pose2D(0.0, 0.0, math.pi / 2), Pose2D(0.0, 1.0, math.pi / 2))
    assert tw.vx > 0 and abs(tw.vy) < 1e-6


def test_converges_lateral_and_reverse_and_spin_targets():
    c = HolonomicController()
    for target in [Pose2D(0.0, 1.5, 0.0), Pose2D(-1.5, 0.0, 0.0), Pose2D(1.0, -1.0, math.pi / 2)]:
        p, reached = _rollout(c, Pose2D(0.0, 0.0, 0.0), target)
        assert reached is True
        assert math.hypot(p.x - target.x, p.y - target.y) <= 0.05
        assert abs(math.atan2(math.sin(p.yaw - target.yaw), math.cos(p.yaw - target.yaw))) <= 0.05
