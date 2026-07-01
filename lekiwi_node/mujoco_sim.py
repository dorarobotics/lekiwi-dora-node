"""LeKiwi MuJoCo sim backend — closes the base loop for the lekiwi dataflow.

Why a dedicated node (not the generic dora-mujoco runner): the physics-ready
scene disables contacts (the shipped single-body omniwheels cannot roll), so
spinning the wheel velocity actuators does NOT translate the free base — verified
in headless MuJoCo (base moved 0.000000 m). Base motion must therefore come from
a kinematic integrator, exactly as the proven octos-dora-bridge lekiwi reference
does. This node:

  * receives the runtime's ``control`` vector (3 wheel speeds + 6 arm positions),
  * applies it to the actuators and steps physics (wheels genuinely spin),
  * recovers the body twist from the 3 wheel speeds (``KiwiDrive.wheels_to_body``,
    the exact inverse of the runtime's IK) and forward-integrates the base pose,
  * writes the pose into the ``base_free`` qpos (0:7) and freezes its velocity,
  * streams the full ``joint_positions`` (nq=16) back — base pose at 0:6 (moving)
    and arm joints at 10:16 (real, actuated) — which the runtime extracts.

Headless by design (a live viewer stalls the dora loop).
"""
from __future__ import annotations

import json
import math
import os
from typing import Any

from lekiwi_node.geometry import Twist, wrap_angle
from lekiwi_node.kinematics import KiwiDrive
from lekiwi_node.sim_scene import BASE_Z, build_scene

DRIVE_ACTUATORS = ("drive_motor_1", "drive_motor_2", "drive_motor_3")
ARM_ACTUATORS = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
# Thin omniwheels have ~0 spin-axis inertia; a stiff velocity actuator drives QACC
# to NaN. Give the wheel DOFs armature + damping so mj_step stays stable.
WHEEL_JOINTS = (
    "ST3215_Servo_Motor-v1-2_Hub---Servo",
    "ST3215_Servo_Motor-v1-1_Hub-2---Servo",
    "ST3215_Servo_Motor-v1_Revolute-40",
)


def integrate_pose(x: float, y: float, theta: float, twist: Twist, dt: float
                   ) -> tuple[float, float, float]:
    """Holonomic dead-reckoning: advance a world pose by a body-frame twist.

    Body velocities (vx forward, vy left) rotate into the world by ``theta``.
    Pure — the load-bearing base kinematics, unit-tested without MuJoCo.
    """
    c, s = math.cos(theta), math.sin(theta)
    x += (twist.vx * c - twist.vy * s) * dt
    y += (twist.vx * s + twist.vy * c) * dt
    theta = wrap_angle(theta + twist.omega * dt)
    return x, y, theta


def _yaw_quat(theta: float) -> tuple[float, float, float, float]:
    return (math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0))


def main() -> None:  # pragma: no cover — needs a running dora daemon + mujoco
    import mujoco
    import numpy as np
    import pyarrow as pa
    from dora import Node

    dt = float(os.environ.get("SIM_DT", "0.05"))
    scene_xml = build_scene()  # vendored asset + its meshdir
    model = mujoco.MjModel.from_xml_string(scene_xml)
    data = mujoco.MjData(model)
    n_sub = max(1, int(round(dt / model.opt.timestep)))

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_free")
    base_qadr = model.jnt_qposadr[jid]
    base_vadr = model.jnt_dofadr[jid]
    drive_adr = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in DRIVE_ACTUATORS]
    arm_adr = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in ARM_ACTUATORS]
    for j in WHEEL_JOINTS:
        wdof = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
        model.dof_armature[wdof] = 0.05
        model.dof_damping[wdof] = 0.2

    kiwi = KiwiDrive()
    x = y = theta = 0.0
    data.qpos[base_qadr:base_qadr + 7] = [0.0, 0.0, BASE_Z, 1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    node = Node()

    def decode(value: Any) -> Any:
        items = value.to_pylist() if hasattr(value, "to_pylist") else list(value)
        return list(items) if items else []

    # Latest control vector (3 wheel speeds + 6 arm positions). Tick-driven: the
    # timer bootstraps the loop (emit joint_positions -> lekiwi_node emits control
    # -> we store it here), avoiding a control<->joints deadlock.
    wheel_cmd = [0.0, 0.0, 0.0]
    print("[lekiwi_mujoco_sim] virtual omni-drive up", flush=True)
    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue
        if event["id"] == "control":
            ctrl = decode(event["value"])
            if len(ctrl) < 9:
                continue
            wheel_cmd = [float(ctrl[0]), float(ctrl[1]), float(ctrl[2])]
            for a, q in zip(arm_adr, ctrl[3:9]):
                data.ctrl[a] = float(q)
        elif event["id"] == "tick":
            for a, w in zip(drive_adr, wheel_cmd):
                data.ctrl[a] = w
            for _ in range(n_sub):
                mujoco.mj_step(model, data)
            # recover the commanded twist from the wheel speeds and integrate the base
            twist = kiwi.wheels_to_body(wheel_cmd[0], wheel_cmd[1], wheel_cmd[2])
            x, y, theta = integrate_pose(x, y, theta, twist, dt)
            qw, qx, qy, qz = _yaw_quat(theta)
            data.qpos[base_qadr:base_qadr + 7] = [x, y, BASE_Z, qw, qx, qy, qz]
            data.qvel[base_vadr:base_vadr + 6] = 0.0
            mujoco.mj_forward(model, data)
            node.send_output("joint_positions", pa.array(np.asarray(data.qpos, dtype=np.float32)))


if __name__ == "__main__":
    main()
