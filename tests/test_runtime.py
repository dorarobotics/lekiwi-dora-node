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
