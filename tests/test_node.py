from lekiwi_node.node import LekiwiNode
from lekiwi_node.geometry import Pose2D, Twist

HOME = [0.0, 0.0, 0.0, 0.0, 0.0, 0.5]


def _node():
    n = LekiwiNode(robot_id="lekiwi", named_arm_poses={"home": HOME})
    n.install_all_verbs()
    return n


def test_set_velocity_is_immediate_and_sets_twist():
    n = _node()
    r = n.dispatch("vendor.dora_nav.base.set_velocity", {"vx": 0.2, "vy": 0.1, "omega": 0.3})
    assert r["ok"] is True and r["code"] == "0"
    assert n.base_velocity == Twist(0.2, 0.1, 0.3)
    assert n.base_target is None


def test_go_to_pose_is_deferred_and_sets_target():
    n = _node()
    pose = {"position": [0.5, 1.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]}
    r = n.dispatch("vendor.dora_nav.base.go_to_pose", {"pose": pose})
    assert r["code"] == "DEFERRED"
    assert isinstance(n.base_target, Pose2D)
    assert (n.base_target.x, n.base_target.y) == (0.5, 1.0)
    assert n.base_velocity is None


def test_stop_clears_base_intents():
    n = _node()
    n.dispatch("vendor.dora_nav.base.set_velocity", {"vx": 1.0, "vy": 0.0, "omega": 0.0})
    r = n.dispatch("vendor.dora_nav.base.stop", {})
    assert r["ok"] is True
    assert n.base_velocity is None and n.base_target is None


def test_move_to_joint_state_deferred_builds_six_vector():
    n = _node()
    r = n.dispatch("vendor.lerobot.arm.move_to_joint_state",
                   {"joints": [0.1, 0.2, 0.3, 0.4, 0.5], "gripper": 0.6})
    assert r["code"] == "DEFERRED"
    assert n.arm_target == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def test_move_to_joint_state_defaults_gripper_open():
    n = _node()
    n.dispatch("vendor.lerobot.arm.move_to_joint_state", {"joints": [0.0, 0.0, 0.0, 0.0, 0.0]})
    assert n.arm_target == [0.0, 0.0, 0.0, 0.0, 0.0, n.gripper_open]


def test_move_to_named_home():
    n = _node()
    r = n.dispatch("vendor.lerobot.arm.move_to_named", {"name": "home"})
    assert r["code"] == "DEFERRED"
    assert n.arm_target == HOME


def test_unknown_named_pose_invalid_params():
    n = _node()
    r = n.dispatch("vendor.lerobot.arm.move_to_named", {"name": "nope"})
    assert r["ok"] is False and r["code"] == "INVALID_PARAMS"


def test_estop_blocks_motion():
    n = _node()
    n.dispatch("robot.estop", {"reason": "test"})
    r = n.dispatch("vendor.dora_nav.base.set_velocity", {"vx": 1.0, "vy": 0.0, "omega": 0.0})
    assert r["ok"] is False and r["code"] == "VENDOR_ERROR"


def test_capabilities_lists_both_namespaces():
    n = _node()
    verbs = {c["verb"] for c in n.capabilities_advert()["commands"]}
    assert "vendor.dora_nav.base.go_to_pose" in verbs
    assert "vendor.lerobot.arm.move_to_joint_state" in verbs
