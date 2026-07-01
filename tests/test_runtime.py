import json
import numpy as np
from lekiwi_node.node import LekiwiNode
from lekiwi_node.runtime import LekiwiRuntime, ACTUATOR_ORDER
from lekiwi_node.geometry import Pose2D


class _FakeDora:
    def __init__(self):
        self.outputs = []
    def send_output(self, oid, data):
        self.outputs.append((oid, data))


def _rt():
    n = LekiwiNode(robot_id="lekiwi", named_arm_poses={"home": [0, 0, 0, 0, 0, 0.5]})
    n.install_all_verbs()
    rt = LekiwiRuntime(
        n,
        base_pose_from=lambda arr: Pose2D(float(arr[0]), float(arr[1]), float(arr[2])),
        arm_joints_from=lambda arr: [float(x) for x in arr[3:9]],
        deadline_s=60.0,
    )
    return n, rt


def _req(verb, params):
    return {"type": "INPUT", "id": "cmd_request",
            "value": [json.dumps({"verb": verb, "request_id": "r", "params": params})]}


def _joints(arr):
    return {"type": "INPUT", "id": "joint_positions", "value": np.array(arr, dtype=float)}


def _resp_payload(data):
    first = data[0]
    s = first.as_py() if hasattr(first, "as_py") else first
    return json.loads(s)


def test_set_velocity_emits_wheel_controls():
    n, rt = _rt()
    rt.on_event(_req("vendor.dora_nav.base.set_velocity",
                     {"vx": 0.0, "vy": 0.0, "omega": 1.0}), _FakeDora())
    dn = _FakeDora()
    rt.on_event(_joints([0, 0, 0, 0, 0, 0, 0, 0, 0]), dn)
    oid, data = [o for o in dn.outputs if o[0] == "control"][0]
    vec = list(np.asarray(data).astype(float))
    assert abs(vec[0] - vec[1]) < 1e-6 and abs(vec[1] - vec[2]) < 1e-6 and vec[0] != 0.0


def test_go_to_pose_resolves_when_reached():
    n, rt = _rt()
    pose = {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]}
    dn = _FakeDora()
    assert rt.on_event(_req("vendor.dora_nav.base.go_to_pose", {"pose": pose}), dn) is True
    assert not [o for o in dn.outputs if o[0] == "cmd_response"]
    dn2 = _FakeDora()
    rt.on_event(_joints([0, 0, 0, 0, 0, 0, 0, 0, 0]), dn2)
    resp = [o for o in dn2.outputs if o[0] == "cmd_response"][0][1]
    payload = _resp_payload(resp)
    assert payload["ok"] is True and payload["code"] == "0"
    assert n.base_target is None


def test_arm_move_resolves_independently_of_base():
    n, rt = _rt()
    dn = _FakeDora()
    rt.on_event(_req("vendor.lerobot.arm.move_to_named", {"name": "home"}), dn)
    dn2 = _FakeDora()
    rt.on_event(_joints([0, 0, 0, 0, 0, 0, 0, 0, 0.5]), dn2)
    resp = [o for o in dn2.outputs if o[0] == "cmd_response"][0][1]
    payload = _resp_payload(resp)
    assert payload["ok"] is True
    assert n.arm_target is None


def test_actuator_order_has_nine_entries():
    assert len(ACTUATOR_ORDER) == 9


def test_controller_busy_on_second_go_to_pose():
    n, rt = _rt()
    pose = {"position": [5.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]}
    rt.on_event(_req("vendor.dora_nav.base.go_to_pose", {"pose": pose}), _FakeDora())
    dn = _FakeDora()
    rt.on_event(_req("vendor.dora_nav.base.go_to_pose", {"pose": pose}), dn)
    p = _resp_payload([o for o in dn.outputs if o[0] == "cmd_response"][0][1])
    assert p["ok"] is False and p["code"] == "CONTROLLER_BUSY"


def test_controller_busy_on_second_arm_move():
    n, rt = _rt()
    rt.on_event(_req("vendor.lerobot.arm.move_to_joint_state", {"joints": [1, 1, 1, 1, 1]}), _FakeDora())
    dn = _FakeDora()
    rt.on_event(_req("vendor.lerobot.arm.move_to_named", {"name": "home"}), dn)
    p = _resp_payload([o for o in dn.outputs if o[0] == "cmd_response"][0][1])
    assert p["code"] == "CONTROLLER_BUSY"


def test_stop_aborts_pending_go_to_pose_promptly():
    n, rt = _rt()
    pose = {"position": [5.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]}
    rt.on_event(_req("vendor.dora_nav.base.go_to_pose", {"pose": pose}), _FakeDora())
    dn = _FakeDora()
    rt.on_event(_req("vendor.dora_nav.base.stop", {}), dn)
    payloads = [_resp_payload(o[1]) for o in dn.outputs if o[0] == "cmd_response"]
    assert any(p["ok"] is False for p in payloads)          # go_to_pose aborted
    assert any(p.get("code") == "0" for p in payloads)      # stop ok
    assert rt._base_pending is None


def test_set_velocity_aborts_pending_go_to_pose():
    n, rt = _rt()
    pose = {"position": [5.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]}
    rt.on_event(_req("vendor.dora_nav.base.go_to_pose", {"pose": pose}), _FakeDora())
    dn = _FakeDora()
    rt.on_event(_req("vendor.dora_nav.base.set_velocity", {"vx": 0.1, "vy": 0.0, "omega": 0.0}), dn)
    payloads = [_resp_payload(o[1]) for o in dn.outputs if o[0] == "cmd_response"]
    assert any(p["ok"] is False for p in payloads)          # superseded go_to_pose aborted
    assert any(p.get("code") == "0" for p in payloads)      # set_velocity ok
    assert rt._base_pending is None


def test_estop_stops_arm_and_aborts_pending():
    n, rt = _rt()
    rt.on_event(_req("vendor.lerobot.arm.move_to_named", {"name": "home"}), _FakeDora())
    dn = _FakeDora()
    rt.on_event(_req("robot.estop", {"reason": "test"}), dn)
    payloads = [_resp_payload(o[1]) for o in dn.outputs if o[0] == "cmd_response"]
    assert any(p["ok"] is False for p in payloads)          # arm move aborted
    assert rt._arm_pending is None
    # after estop the arm holds measured, not the stale HOME target
    import numpy as np
    dn2 = _FakeDora()
    rt.on_event(_joints([0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), dn2)
    vec = list(np.asarray([o for o in dn2.outputs if o[0] == "control"][0][1]).astype(float))
    assert all(abs(vec[3 + i] - m) < 1e-6 for i, m in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))


def test_arm_pending_times_out_and_clears_driver():
    from lekiwi_node.node import LekiwiNode
    from lekiwi_node.runtime import LekiwiRuntime
    from lekiwi_node.geometry import Pose2D
    n = LekiwiNode(robot_id="lekiwi", named_arm_poses={"home": [9, 9, 9, 9, 9, 9]})
    n.install_all_verbs()
    rt = LekiwiRuntime(n, base_pose_from=lambda a: Pose2D(float(a[0]), float(a[1]), float(a[2])),
                       arm_joints_from=lambda a: [float(x) for x in a[3:9]], deadline_s=-1.0)
    rt.on_event(_req("vendor.lerobot.arm.move_to_named", {"name": "home"}), _FakeDora())
    dn = _FakeDora()
    rt.on_event(_joints([0, 0, 0, 0, 0, 0, 0, 0, 0]), dn)  # arm far from [9]*6; deadline -1 -> timeout
    p = _resp_payload([o for o in dn.outputs if o[0] == "cmd_response"][0][1])
    assert p["ok"] is False and p["code"] == "BRIDGE_TIMEOUT"
    assert rt._arm.target is None and n.arm_target is None


def test_pending_times_out_with_bridge_timeout():
    from lekiwi_node.node import LekiwiNode
    from lekiwi_node.runtime import LekiwiRuntime
    from lekiwi_node.geometry import Pose2D
    n = LekiwiNode(robot_id="lekiwi", named_arm_poses={"home": [0, 0, 0, 0, 0, 0.5]})
    n.install_all_verbs()
    rt = LekiwiRuntime(n, base_pose_from=lambda a: Pose2D(float(a[0]), float(a[1]), float(a[2])),
                       arm_joints_from=lambda a: [float(x) for x in a[3:9]], deadline_s=-1.0)
    pose = {"position": [5.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]}
    rt.on_event(_req("vendor.dora_nav.base.go_to_pose", {"pose": pose}), _FakeDora())
    dn = _FakeDora()
    rt.on_event(_joints([0, 0, 0, 0, 0, 0, 0, 0, 0]), dn)  # far from (5,0); deadline_s=-1 -> immediate timeout
    p = _resp_payload([o for o in dn.outputs if o[0] == "cmd_response"][0][1])
    assert p["ok"] is False and p["code"] == "BRIDGE_TIMEOUT"
