from __future__ import annotations


class ArmDriver:
    """Direct joint-space target tracking for the LeKiwi arm (5 joints + gripper).
    No IK/planning — replays calibrated joint vectors, like the LeRobot servo arm."""

    def __init__(self, *, named_poses: dict[str, list[float]] | None = None,
                 joint_tol: float = 0.05, dof: int = 6) -> None:
        self.named_poses = {k: list(v) for k, v in (named_poses or {}).items()}
        self.joint_tol = joint_tol
        self.dof = dof
        self.target: list[float] | None = None

    def set_target(self, joints: list[float]) -> None:
        if len(joints) != self.dof:
            raise ValueError(f"expected {self.dof} joint values, got {len(joints)}")
        self.target = [float(j) for j in joints]

    def set_named(self, name: str) -> bool:
        if name not in self.named_poses:
            return False
        self.set_target(self.named_poses[name])
        return True

    def reached(self, measured: list[float]) -> bool:
        if self.target is None or len(measured) < self.dof:
            return False
        return all(abs(measured[i] - self.target[i]) <= self.joint_tol
                   for i in range(self.dof))

    def clear(self) -> None:
        self.target = None
