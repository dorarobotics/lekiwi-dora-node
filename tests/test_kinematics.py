import math
from lekiwi_node.kinematics import KiwiDrive
from lekiwi_node.geometry import Twist


def test_pure_spin_all_wheels_equal():
    # omega only -> every wheel spins the same (pure rotation).
    k = KiwiDrive()
    w1, w2, w3 = k.body_to_wheels(0.0, 0.0, 1.0)
    assert abs(w1 - w2) < 1e-9 and abs(w2 - w3) < 1e-9
    assert w1 > 0


def test_pure_forward_back_wheel_is_zero():
    # +x forward: back wheel (mounted at -90 deg) contributes ~no drive.
    k = KiwiDrive()
    w1, w2, w3 = k.body_to_wheels(0.3, 0.0, 0.0)
    assert abs(w2) < 1e-9          # back wheel
    assert abs(w1 + w3) < 1e-9     # symmetric left/right


def test_pure_strafe_back_wheel_dominant():
    # +y strafe: back wheel carries the largest share, opposite sign to sides.
    k = KiwiDrive()
    w1, w2, w3 = k.body_to_wheels(0.0, 0.3, 0.0)
    assert w2 < 0 and w1 > 0 and w3 > 0
    assert abs(w2) > abs(w1)


def test_round_trip_recovers_body_velocity():
    k = KiwiDrive()
    for vx, vy, om in [(0.2, 0.0, 0.0), (0.0, 0.15, 0.0), (0.0, 0.0, 0.5), (0.1, -0.1, 0.3)]:
        w = k.body_to_wheels(vx, vy, om)
        t = k.wheels_to_body(*w)
        assert abs(t.vx - vx) < 1e-9 and abs(t.vy - vy) < 1e-9 and abs(t.omega - om) < 1e-9


def test_wheels_clamped_to_max():
    k = KiwiDrive(max_wheel_omega=5.0)
    w = k.body_to_wheels(100.0, 0.0, 0.0)
    assert all(abs(x) <= 5.0 + 1e-9 for x in w)
