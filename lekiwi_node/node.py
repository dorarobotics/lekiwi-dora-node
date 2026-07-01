from __future__ import annotations

from typing import Any, Callable

from lekiwi_node.geometry import Pose2D, Twist, yaw_from_quat

BASE_MOTION_VERBS = frozenset({
    "vendor.dora_nav.base.set_velocity", "vendor.dora_nav.base.go_to_pose",
})
ARM_MOTION_VERBS = frozenset({
    "vendor.lerobot.arm.move_to_joint_state", "vendor.lerobot.arm.move_to_named",
})


class LekiwiNode:
    """SPEC-V1 verbs for the LeKiwi robot (holonomic base + 5-DOF arm + gripper).
    Pure logic; the runtime drives I/O and resolves deferred motion."""

    def __init__(self, *, robot_id: str, named_arm_poses: dict[str, list[float]] | None = None,
                 gripper_open: float = 0.5, arm_dof: int = 6) -> None:
        self.robot_id = robot_id
        self.named_arm_poses = {k: list(v) for k, v in (named_arm_poses or {}).items()}
        self.gripper_open = gripper_open
        self.arm_dof = arm_dof
        self._verbs: dict[str, Callable[..., Any]] = {}
        self.base_target: Pose2D | None = None
        self.base_velocity: Twist | None = None
        self.arm_target: list[float] | None = None
        self.is_estopped = False
        self.estop_reason: str | None = None

    def register_verb(self, name: str, handler: Callable[..., Any]) -> None:
        if name in self._verbs:
            raise ValueError(f"verb already registered: {name}")
        self._verbs[name] = handler

    def dispatch(self, verb: str, args: dict[str, Any]) -> dict[str, Any]:
        if verb not in self._verbs:
            return {"ok": False, "code": "INVALID_PARAMS", "msg": f"unknown verb: {verb}"}
        try:
            return self._verbs[verb](**args)
        except TypeError as e:
            return {"ok": False, "code": "INVALID_PARAMS", "msg": f"bad args for {verb}: {e}"}

    def install_all_verbs(self) -> None:
        self.register_verb("robot.heartbeat", lambda: {"ok": True, "code": "0"})
        self.register_verb("robot.estop", self._verb_estop)
        self.register_verb("robot.get_capabilities",
                           lambda: {"ok": True, "code": "0", "data": self.capabilities_advert()})
        self.register_verb("vendor.dora_nav.base.set_velocity", self._verb_set_velocity)
        self.register_verb("vendor.dora_nav.base.go_to_pose", self._verb_go_to_pose)
        self.register_verb("vendor.dora_nav.base.stop", self._verb_stop)
        self.register_verb("vendor.lerobot.arm.move_to_joint_state", self._verb_move_joints)
        self.register_verb("vendor.lerobot.arm.move_to_named", self._verb_move_named)

    def _verb_estop(self, *, reason: str = "unspecified") -> dict[str, Any]:
        self.is_estopped = True
        self.estop_reason = reason
        self.base_target = None
        self.base_velocity = None
        self.arm_target = None
        return {"ok": True, "code": "0"}

    def _estop_guard(self) -> dict[str, Any] | None:
        if self.is_estopped:
            return {"ok": False, "code": "VENDOR_ERROR", "msg": f"estopped: {self.estop_reason}"}
        return None

    def _verb_set_velocity(self, *, vx: float, vy: float, omega: float,
                           control_source: str = "") -> dict[str, Any]:
        guard = self._estop_guard()
        if guard:
            return guard
        self.base_velocity = Twist(float(vx), float(vy), float(omega))
        self.base_target = None
        return {"ok": True, "code": "0"}

    def _verb_go_to_pose(self, *, pose: dict[str, Any], control_source: str = "") -> dict[str, Any]:
        guard = self._estop_guard()
        if guard:
            return guard
        if not isinstance(pose, dict) or "position" not in pose or "orientation" not in pose:
            return {"ok": False, "code": "INVALID_PARAMS",
                    "msg": "pose needs position[xyz] + orientation[xyzw]"}
        x, y = float(pose["position"][0]), float(pose["position"][1])
        qx, qy, qz, qw = (float(v) for v in pose["orientation"])
        self.base_target = Pose2D(x, y, yaw_from_quat(qw, qx, qy, qz))
        self.base_velocity = None
        return {"code": "DEFERRED"}

    def _verb_stop(self) -> dict[str, Any]:
        self.base_velocity = None
        self.base_target = None
        return {"ok": True, "code": "0"}

    def _verb_move_joints(self, *, joints: list[float], gripper: float | None = None,
                          control_source: str = "") -> dict[str, Any]:
        guard = self._estop_guard()
        if guard:
            return guard
        if len(joints) != self.arm_dof - 1:
            return {"ok": False, "code": "INVALID_PARAMS",
                    "msg": f"expected {self.arm_dof - 1} arm joints"}
        g = self.gripper_open if gripper is None else float(gripper)
        self.arm_target = [float(j) for j in joints] + [g]
        return {"code": "DEFERRED"}

    def _verb_move_named(self, *, name: str, control_source: str = "") -> dict[str, Any]:
        guard = self._estop_guard()
        if guard:
            return guard
        if name not in self.named_arm_poses:
            return {"ok": False, "code": "INVALID_PARAMS", "msg": f"unknown named pose: {name}"}
        self.arm_target = list(self.named_arm_poses[name])
        return {"code": "DEFERRED"}

    def capabilities_advert(self) -> dict[str, Any]:
        return {
            "spec_version": "1.0.0",
            "vendor": "lekiwi",
            "model": "lekiwi",
            "robot_id": self.robot_id,
            "heartbeat_timeout_ms": 0,
            "commands": [{"verb": v, "safety_tier": "emergency_override"}
                         for v in sorted(self._verbs.keys())],
            "streams": ["state", "capabilities"],
        }
