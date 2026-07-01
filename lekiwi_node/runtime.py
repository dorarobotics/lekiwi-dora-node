from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pyarrow as pa

from lekiwi_node._envelope import (
    CmdRequest, InvalidEnvelope, build_cmd_response, parse_cmd_request,
)
from lekiwi_node.arm_driver import ArmDriver
from lekiwi_node.base_controller import HolonomicController
from lekiwi_node.control import assemble_control
from lekiwi_node.geometry import Pose2D, Twist
from lekiwi_node.kinematics import KiwiDrive
from lekiwi_node.node import ARM_MOTION_VERBS, LekiwiNode

# MJCF actuator order (from LeKiwi-sim/mjcf_lcmm_robot.xml); wheels then arm+gripper.
ACTUATOR_ORDER = ["drive_motor_1", "drive_motor_2", "drive_motor_3",
                  "Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
WHEEL_ACTUATORS = ACTUATOR_ORDER[:3]
ARM_ACTUATORS = ACTUATOR_ORDER[3:]


@dataclass
class PendingOp:
    request: CmdRequest
    started: float


def _decode_env(value: Any) -> dict[str, Any] | None:
    try:
        items = value.to_pylist() if hasattr(value, "to_pylist") else list(value)
    except Exception:  # noqa: BLE001
        return None
    if not items:
        return None
    first = items[0]
    try:
        return json.loads(first) if isinstance(first, str) else dict(first)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


class LekiwiRuntime:
    def __init__(self, node: LekiwiNode,
                 *, base_pose_from: Callable[[Any], Pose2D | None],
                 arm_joints_from: Callable[[Any], list[float]],
                 kiwi: KiwiDrive | None = None,
                 base_ctrl: HolonomicController | None = None,
                 arm: ArmDriver | None = None,
                 deadline_s: float = 60.0,
                 velocity_timeout_s: float = 0.5) -> None:
        self._node = node
        self._base_pose_from = base_pose_from
        self._arm_joints_from = arm_joints_from
        self._kiwi = kiwi or KiwiDrive()
        self._ctrl = base_ctrl or HolonomicController()
        self._arm = arm or ArmDriver(named_poses=node.named_arm_poses, dof=node.arm_dof)
        self._deadline_s = deadline_s
        self._velocity_timeout_s = velocity_timeout_s
        self._velocity_started: float | None = None
        self._base_pending: PendingOp | None = None
        self._arm_pending: PendingOp | None = None

    def handle_request(self, env: dict[str, Any], dn: Any) -> dict[str, Any] | None:
        try:
            req = parse_cmd_request(env)
        except InvalidEnvelope as e:
            bad = CmdRequest(request_id=str(env.get("request_id", "")), verb="",
                             params={}, target=None, spec_version="1.0.0",
                             trace_id=env.get("trace_id"))
            return build_cmd_response(bad, ok=False, code="INVALID_PARAMS", msg=str(e))
        if self._base_pending is not None and req.verb == "vendor.dora_nav.base.go_to_pose":
            return build_cmd_response(req, ok=False, code="CONTROLLER_BUSY",
                                      msg="base motion in progress")
        if req.verb in ARM_MOTION_VERBS and self._arm_pending is not None:
            return build_cmd_response(req, ok=False, code="CONTROLLER_BUSY",
                                      msg="arm motion in progress")
        result = self._node.dispatch(req.verb, req.params)
        if result.get("code") == "DEFERRED":
            if req.verb in ARM_MOTION_VERBS:
                self._arm.set_target(list(self._node.arm_target))
                self._arm_pending = PendingOp(req, time.monotonic())
            else:
                self._base_pending = PendingOp(req, time.monotonic())
            return None
        if req.verb == "vendor.dora_nav.base.set_velocity" and self._node.base_velocity is not None:
            self._velocity_started = time.monotonic()
        # An immediate verb (stop/estop/set_velocity) may have cleared a target a
        # pending op depends on; abort that op now so the bridge gets prompt closure
        # instead of a false BRIDGE_TIMEOUT, and reconcile the arm driver.
        self._reconcile_pending(dn)
        return build_cmd_response(req, ok=bool(result.get("ok", False)),
                                  code=str(result.get("code", "0")),
                                  data=result.get("data"), msg=str(result.get("msg", "")))

    def _reconcile_pending(self, dn: Any) -> None:
        if self._base_pending is not None and self._node.base_target is None:
            self._abort(self._base_pending, dn)
            self._base_pending = None
        if self._arm_pending is not None and self._node.arm_target is None:
            self._abort(self._arm_pending, dn)
            self._arm_pending = None
            self._arm.clear()

    def _abort(self, op: PendingOp, dn: Any) -> None:
        dn.send_output("cmd_response", pa.array([json.dumps(
            build_cmd_response(op.request, ok=False, code="VENDOR_ERROR",
                               msg="motion aborted (stopped or superseded)"))]))

    def on_event(self, event: dict[str, Any], dn: Any) -> bool:
        if event.get("type") == "STOP":
            return False
        if event.get("type") != "INPUT":
            return True
        if event.get("id") == "cmd_request":
            env = _decode_env(event.get("value"))
            if env is not None:
                tgt = env.get("target")
                if tgt is None or tgt == self._node.robot_id:
                    resp = self.handle_request(env, dn)
                    if resp is not None:
                        dn.send_output("cmd_response", pa.array([json.dumps(resp)]))
        elif event.get("id") == "joint_positions":
            self._drive(event.get("value"), dn)
        return True

    def _drive(self, value: Any, dn: Any) -> None:
        pose = self._base_pose_from(value)
        arm_meas = self._arm_joints_from(value)
        if self._node.base_target is not None and pose is not None:
            twist, base_reached = self._ctrl.step(pose, self._node.base_target)
        elif self._node.base_velocity is not None:
            if (self._velocity_started is not None
                    and time.monotonic() - self._velocity_started > self._velocity_timeout_s):
                self._node.base_velocity = None  # watchdog: stale set_velocity auto-stops
                twist, base_reached = Twist(0.0, 0.0, 0.0), False
            else:
                twist, base_reached = self._node.base_velocity, False
        else:
            twist, base_reached = Twist(0.0, 0.0, 0.0), False
        w1, w2, w3 = self._kiwi.body_to_wheels(twist.vx, twist.vy, twist.omega)
        arm_cmd = self._arm.target if self._arm.target is not None else list(arm_meas)
        if len(arm_cmd) < len(ARM_ACTUATORS):  # make the 0-fill explicit rather than relying on assemble_control's missing-key default
            arm_cmd = list(arm_cmd) + [0.0] * (len(ARM_ACTUATORS) - len(arm_cmd))
        values = {WHEEL_ACTUATORS[0]: w1, WHEEL_ACTUATORS[1]: w2, WHEEL_ACTUATORS[2]: w3}
        for name, v in zip(ARM_ACTUATORS, arm_cmd):
            values[name] = v
        dn.send_output("control", pa.array(np.array(
            assemble_control(ACTUATOR_ORDER, values), dtype=np.float32)))
        # deadlines are evaluated on joint_positions ticks; a stalled sensor stream defers timeout (acceptable in sim; revisit for hardware).
        if self._base_pending is not None:
            if self._node.base_target is not None and pose is not None and base_reached:
                self._resolve(self._base_pending, dn, pose=pose)
                self._base_pending = None
                self._node.base_target = None
            elif time.monotonic() - self._base_pending.started > self._deadline_s:
                self._timeout(self._base_pending, dn); self._base_pending = None; self._node.base_target = None
        if self._arm_pending is not None:
            if self._arm.reached(arm_meas):
                self._resolve(self._arm_pending, dn)
                self._arm_pending = None
                self._arm.clear()
                self._node.arm_target = None
            elif time.monotonic() - self._arm_pending.started > self._deadline_s:
                self._timeout(self._arm_pending, dn)
                self._arm_pending = None
                self._arm.clear()
                self._node.arm_target = None

    def _resolve(self, op: PendingOp, dn: Any, *, pose: Pose2D | None = None) -> None:
        data = None if pose is None else {"final_pose": {"x": pose.x, "y": pose.y, "yaw": pose.yaw}}
        dn.send_output("cmd_response", pa.array([json.dumps(
            build_cmd_response(op.request, ok=True, code="0", data=data))]))

    def _timeout(self, op: PendingOp, dn: Any) -> None:
        dn.send_output("cmd_response", pa.array([json.dumps(
            build_cmd_response(op.request, ok=False, code="BRIDGE_TIMEOUT",
                               msg="did not reach target in time"))]))
