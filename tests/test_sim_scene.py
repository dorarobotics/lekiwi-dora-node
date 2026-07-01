"""Offline validation of the physics-ready LeKiwi scene + base integrator.

The MuJoCo-dependent test loads the built scene headless and pins the verified
qpos/actuator layout that the runtime and __main__ extractors rely on. It skips
cleanly where mujoco is not installed; the integrator tests are pure.
"""
from __future__ import annotations

import math

import pytest

from lekiwi_node.geometry import Twist
from lekiwi_node.kinematics import KiwiDrive
from lekiwi_node.mujoco_sim import integrate_pose
from lekiwi_node.sim_scene import build_scene


def test_build_scene_injects_freejoint_and_velocity_actuators():
    xml = build_scene()
    assert '<freejoint name="base_free"/>' in xml
    assert '<body name="chassis"' in xml
    assert '<velocity kv="8" name="drive_motor_1"' in xml
    assert '<motor name="drive_motor_' not in xml  # all wheel motors converted
    assert 'contact="disable"' in xml
    assert 'gravity="0 0 0"' in xml


def test_scene_layout_matches_extractor_indices():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(build_scene())
    assert model.nq == 16
    assert model.nu == 9

    def qadr(joint):
        return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)]

    assert qadr("base_free") == 0                       # base pose 0:7
    assert qadr("ST3215_Servo_Motor-v1-2_Hub---Servo") == 7  # wheels 7:10
    assert qadr("Rotation") == 10                       # arm 10:16
    assert qadr("Jaw") == 15

    # actuators: 0-2 wheels are velocity (trntype transmission is joint either way;
    # assert names + order, which the runtime's control vector depends on)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(model.nu)]
    assert names[:3] == ["drive_motor_1", "drive_motor_2", "drive_motor_3"]
    assert names[3:] == ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]


def test_wheel_spin_does_not_move_base_without_integrator():
    """The physics fact that forces a kinematic integrator: contacts off => the
    free base does not translate from wheel spin (base motion is integrated)."""
    mujoco = pytest.importorskip("mujoco")
    import numpy as np

    model = mujoco.MjModel.from_xml_string(build_scene())
    data = mujoco.MjData(model)
    for j in ("ST3215_Servo_Motor-v1-2_Hub---Servo",
              "ST3215_Servo_Motor-v1-1_Hub-2---Servo",
              "ST3215_Servo_Motor-v1_Revolute-40"):
        wdof = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
        model.dof_armature[wdof] = 0.05
        model.dof_damping[wdof] = 0.2
    data.qpos[0:7] = [0, 0, 0.06, 1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    for a in range(3):
        data.ctrl[a] = 10.0
    before = data.qpos[0:3].copy()
    for _ in range(200):
        mujoco.mj_step(model, data)
    after = data.qpos[0:3].copy()
    assert np.linalg.norm(after - before) < 1e-4   # base fixed
    assert abs(data.qpos[7]) > 0.1                 # wheels genuinely spun


def test_integrate_pose_forward_along_heading():
    # facing +x, drive forward 1 m/s for 1 s -> +1 x
    x, y, th = integrate_pose(0.0, 0.0, 0.0, Twist(1.0, 0.0, 0.0), 1.0)
    assert x == pytest.approx(1.0)
    assert y == pytest.approx(0.0)
    assert th == pytest.approx(0.0)


def test_integrate_pose_strafe_in_body_frame():
    # facing +90deg, body-forward vx=1 -> world +y (holonomic frame rotation)
    x, y, th = integrate_pose(0.0, 0.0, math.pi / 2, Twist(1.0, 0.0, 0.0), 1.0)
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(1.0)


def test_integrate_pose_matches_ik_roundtrip():
    """Sim recovers the commanded twist from wheel speeds exactly (M invertible),
    so the integrated pose equals integrating the original twist."""
    kiwi = KiwiDrive()
    cmd = Twist(0.3, -0.2, 0.4)
    w1, w2, w3 = kiwi.body_to_wheels(cmd.vx, cmd.vy, cmd.omega)
    recovered = kiwi.wheels_to_body(w1, w2, w3)
    a = integrate_pose(0.0, 0.0, 0.5, cmd, 0.1)
    b = integrate_pose(0.0, 0.0, 0.5, recovered, 0.1)
    assert a == pytest.approx(b)
