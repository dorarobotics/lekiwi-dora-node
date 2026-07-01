"""Interactive MuJoCo viewer for the LeKiwi robot — a looped, watchable demo.

Opens a native MuJoCo window and drives the robot through a repeating sequence
(forward -> strafe -> spin -> arm up/down) using this package's own kinematics
(`sim_scene.build_scene` + `KiwiDrive` + `mujoco_sim.integrate_pose`). The wheels
genuinely spin under physics; the base pose is the kinematic integrator (contacts
are disabled — the shipped single-body omniwheels cannot roll). It resets to the
origin each cycle so the robot stays centered, and runs in real time.

macOS requires the MuJoCo GUI to run on the main thread, so launch with mjpython:

    mjpython examples/live_viewer.py

(Linux/Windows: `python examples/live_viewer.py` also works.) Requires `mujoco`
installed and `drive-kinematics` on the path (a declared dependency of this repo).
Close the window to stop. This is the standalone viewer — a live window blocks the
dora event loop, so it is not part of the dataflow.
"""
from __future__ import annotations

import os
import sys
import time

# Make `lekiwi_node` importable when run straight from a checkout.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import mujoco.viewer

from lekiwi_node.kinematics import KiwiDrive
from lekiwi_node.mujoco_sim import (
    ARM_ACTUATORS, DRIVE_ACTUATORS, WHEEL_JOINTS, _yaw_quat, integrate_pose,
)
from lekiwi_node.sim_scene import BASE_Z, build_scene

DT = 0.05
ARM_HOME = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ARM_RAISED = [0.6, -0.9, 0.9, 0.0, 0.0, 0.4]
# (label, vx, vy, omega, arm_target, ticks)
SEQUENCE = [
    ("forward", 0.25, 0.0, 0.0, ARM_HOME, 40),
    ("strafe", 0.0, 0.25, 0.0, ARM_HOME, 40),
    ("spin", 0.0, 0.0, 0.6, ARM_HOME, 40),
    ("arm up", 0.0, 0.0, 0.0, ARM_RAISED, 40),
    ("arm down", 0.0, 0.0, 0.0, ARM_HOME, 25),
]


def main() -> None:
    model = mujoco.MjModel.from_xml_string(build_scene())
    data = mujoco.MjData(model)
    n_sub = max(1, int(round(DT / model.opt.timestep)))

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_free")
    base_qadr = model.jnt_qposadr[jid]
    base_vadr = model.jnt_dofadr[jid]
    drive_adr = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in DRIVE_ACTUATORS]
    arm_adr = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in ARM_ACTUATORS]
    # Thin omniwheels have ~0 spin-axis inertia; give the wheel DOFs armature +
    # damping so a stiff velocity actuator does not drive QACC to NaN.
    for j in WHEEL_JOINTS:
        wdof = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
        model.dof_armature[wdof] = 0.05
        model.dof_damping[wdof] = 0.2

    kiwi = KiwiDrive()
    pose = {"x": 0.0, "y": 0.0, "th": 0.0}

    def reset() -> None:
        pose.update(x=0.0, y=0.0, th=0.0)
        data.qpos[:] = 0.0
        data.qpos[base_qadr:base_qadr + 7] = [0.0, 0.0, BASE_Z, 1.0, 0.0, 0.0, 0.0]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

    reset()
    print("[viewer] opening window — close it to stop", flush=True)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.3, 0.3, 0.1]
        viewer.cam.distance = 2.8
        viewer.cam.elevation = -25.0
        viewer.cam.azimuth = 130.0
        cycle = 0
        while viewer.is_running():
            cycle += 1
            reset()
            for _label, vx, vy, omega, arm, ticks in SEQUENCE:
                if not viewer.is_running():
                    break
                w1, w2, w3 = kiwi.body_to_wheels(vx, vy, omega)
                for _ in range(ticks):
                    if not viewer.is_running():
                        break
                    t0 = time.time()
                    for a, w in zip(drive_adr, (w1, w2, w3)):
                        data.ctrl[a] = w
                    for a, q in zip(arm_adr, arm):
                        data.ctrl[a] = q
                    for _ in range(n_sub):
                        mujoco.mj_step(model, data)
                    twist = kiwi.wheels_to_body(w1, w2, w3)
                    pose["x"], pose["y"], pose["th"] = integrate_pose(
                        pose["x"], pose["y"], pose["th"], twist, DT)
                    qw, qx, qy, qz = _yaw_quat(pose["th"])
                    data.qpos[base_qadr:base_qadr + 7] = [pose["x"], pose["y"], BASE_Z, qw, qx, qy, qz]
                    data.qvel[base_vadr:base_vadr + 6] = 0.0
                    mujoco.mj_forward(model, data)
                    viewer.sync()
                    remaining = DT - (time.time() - t0)
                    if remaining > 0:
                        time.sleep(remaining)
            print(f"[viewer] cycle {cycle} done", flush=True)
    print("[viewer] closed", flush=True)


if __name__ == "__main__":
    main()
