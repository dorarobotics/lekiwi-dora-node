from __future__ import annotations

import json
import os

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
    # Verified qpos layout (physics-ready scene, sim_scene.build_scene): nq=16 ==
    # base_free[0:7] + 3 wheel hinges[7:10] + 6 arm joints[10:16]. Confirmed by
    # loading the built scene in headless MuJoCo (Rotation..Jaw at qpos 10-15).
    arr = np.asarray(value)
    if arr.shape[0] < 16:
        return []  # malformed layout: runtime pads to length
    return [float(v) for v in arr[10:16]]


def main() -> None:
    from dora import Node  # imported here so the module is importable without the dora runtime
    import pyarrow as pa  # imported here so the module is importable without the dora runtime
    node = LekiwiNode(robot_id=os.environ.get("ROBOT_ID", "lekiwi"), named_arm_poses={"home": HOME})
    node.install_all_verbs()
    deadline = float(os.environ.get("MOTION_DEADLINE_S", "60.0"))
    rt = LekiwiRuntime(node, base_pose_from=_base_pose, arm_joints_from=_arm_joints, deadline_s=deadline,
                       velocity_timeout_s=float(os.environ.get("VELOCITY_TIMEOUT_S", "0.5")))
    dora = Node()
    dora.send_output("capabilities", pa.array([json.dumps(node.capabilities_advert())]))
    for event in dora:
        if not rt.on_event(event, dora):
            break


if __name__ == "__main__":
    main()
