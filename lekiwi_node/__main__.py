from __future__ import annotations

import numpy as np

from lekiwi_node.geometry import Pose2D, yaw_from_quat
from lekiwi_node.node import LekiwiNode
from lekiwi_node.runtime import LekiwiRuntime

HOME = [0.0, 0.0, 0.0, 0.0, 0.0, 0.5]  # 5 arm joints + gripper (open)


def _base_pose(value) -> Pose2D | None:
    arr = np.asarray(value)
    if arr.shape[0] < 7:
        return None
    x, y = float(arr[0]), float(arr[1])
    qw, qx, qy, qz = (float(arr[i]) for i in (3, 4, 5, 6))
    return Pose2D(x, y, yaw_from_quat(qw, qx, qy, qz))


def _arm_joints(value) -> list[float]:
    arr = np.asarray(value)
    # 6 arm/gripper joints follow the 7 base free-joint DOF (VERIFY: V2, live)
    return [float(v) for v in arr[7:13]]


def main() -> None:
    from dora import Node  # imported here so the module is importable without the dora runtime
    node = LekiwiNode(robot_id="lekiwi", named_arm_poses={"home": HOME})
    node.install_all_verbs()
    rt = LekiwiRuntime(node, base_pose_from=_base_pose, arm_joints_from=_arm_joints)
    dora = Node()
    for event in dora:
        if not rt.on_event(event, dora):
            break


if __name__ == "__main__":
    main()
