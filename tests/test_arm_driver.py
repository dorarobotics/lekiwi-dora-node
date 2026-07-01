from lekiwi_node.arm_driver import ArmDriver

HOME = [0.0, 0.0, 0.0, 0.0, 0.0, 0.5]


def _driver():
    return ArmDriver(named_poses={"home": HOME}, joint_tol=0.05, dof=6)


def test_no_target_initially():
    assert _driver().target is None


def test_set_target_and_reached_tolerance():
    d = _driver()
    d.set_target([0.1, 0.2, 0.0, -0.3, 0.0, 0.4])
    assert d.target == [0.1, 0.2, 0.0, -0.3, 0.0, 0.4]
    assert d.reached([0.1, 0.2, 0.0, -0.3, 0.0, 0.4]) is True
    assert d.reached([0.1, 0.2, 0.0, -0.3, 0.0, 0.44]) is True   # within 0.05
    assert d.reached([0.1, 0.2, 0.0, -0.3, 0.0, 0.9]) is False   # gripper off


def test_set_target_wrong_length_rejected():
    d = _driver()
    try:
        d.set_target([0.0, 0.0, 0.0])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_named_home_sets_target():
    d = _driver()
    assert d.set_named("home") is True
    assert d.target == HOME


def test_unknown_named_returns_false_and_leaves_target():
    d = _driver()
    d.set_target([0.0] * 6)
    assert d.set_named("nope") is False
    assert d.target == [0.0] * 6


def test_reached_false_when_no_target():
    assert _driver().reached([0.0] * 6) is False


def test_clear_resets_target():
    d = _driver()
    d.set_target([0.0] * 6)
    d.clear()
    assert d.target is None
    assert d.reached([0.0] * 6) is False


def test_reached_allows_longer_measured_vector():
    d = _driver()
    d.set_target([0.0] * 6)
    assert d.reached([0.0] * 7) is True   # 7th channel ignored; first 6 match


def test_reached_boundary_is_inclusive():
    d = _driver()
    d.set_target([0.1, 0.2, 0.0, -0.3, 0.0, 0.4])
    assert d.reached([0.1, 0.2, 0.0, -0.3, 0.0, 0.45]) is True   # exactly target+tol
