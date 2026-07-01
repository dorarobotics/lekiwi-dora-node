"""Build a physics-ready LeKiwi MuJoCo scene from the vendored (pristine) MJCF.

The shipped ``mjcf_lcmm_robot.xml`` is a CAD export that is NOT runnable as a
mobile robot: it bolts three separate top-level bodies to the world with no joint
(base_plate_layer1 = wheels, base_plate_layer2 = arm mount, drive_motor_mount-v4
= orphan), comments out its floor, and drives the wheels with force ``<motor>``
actuators. We read that file as text (never mutate the source) and inject the
minimal edits that make it step under MuJoCo:

  1. absolute ``meshdir`` so meshes resolve when loaded from a string;
  2. an ``<option>`` with gravity off + contacts disabled — the shipped omniwheels
     are single rigid bodies (no rollers), so wheel-ground contact cannot produce
     correct holonomic motion; the base is placed by a kinematic integrator (see
     ``mujoco_sim``) and the wheels spin purely for visual fidelity;
  3. the 3 wheel ``<motor>`` actuators become ``<velocity>`` so ``ctrl`` is a
     target wheel angular speed (rad/s) — matching the runtime's control vector;
  4. a floor plane (visual ground reference);
  5. one ``chassis`` body carrying a ``<freejoint>`` that wraps ALL THREE
     top-level bodies, so the whole robot is one rigid free body (a freejoint on
     just base_plate_layer1 would let the wheels drive away from the arm).

Resulting qpos layout (nq=16), verified in headless MuJoCo:
  [0:7]   base_free (x, y, z, qw, qx, qy, qz)
  [7:10]  3 wheel hinges
  [10:16] 6 arm joints (Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw)

This is a text transform of the vendored asset, ported from the proven
octos-dora-bridge lekiwi-sim reference (examples/lekiwi_scene.py).
"""
from __future__ import annotations

import os

ROOT_BODY_TAG = '<body name="base_plate_layer1-v5-1" pos="0.0 0.0 0.0" euler="-0.0 0.0 -0.0">'
COMPILER_TAG = '<compiler angle="radian" />'
WORLDBODY_OPEN = "<worldbody>"
WORLDBODY_CLOSE = "</worldbody>"

BASE_Z = 0.06  # m — free-joint z that rests the wheels at the floor plane

_OPTION = (
    '\n    <option timestep="0.002" gravity="0 0 0">'
    '\n      <flag contact="disable"/>'
    '\n    </option>'
)
_FLOOR = (
    '\n        <geom name="floor" type="plane" size="5 5 0.1" '
    'rgba="0.82 0.85 0.90 1" pos="0 0 0"/>'
)

# The vendored model lives next to this package's assets/ dir.
_DEFAULT_MJCF = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "assets", "mjcf_lcmm_robot.xml")


def build_scene(src_mjcf: str | None = None, meshdir: str | None = None) -> str:
    """Return the physics-ready scene XML string.

    ``src_mjcf`` defaults to the vendored asset; ``meshdir`` defaults to that
    asset's directory (its mesh ``file=`` paths already include the ``meshes/``
    prefix, so meshdir must point at the dir *containing* meshes/, not meshes/).
    """
    src_mjcf = src_mjcf or _DEFAULT_MJCF
    meshdir = meshdir if meshdir is not None else os.path.dirname(src_mjcf)

    with open(src_mjcf, "r", encoding="utf-8") as fh:
        xml = fh.read()

    xml = xml.replace(
        COMPILER_TAG,
        f'<compiler angle="radian" meshdir="{meshdir}" />{_OPTION}',
        1,
    )
    xml = xml.replace('<motor name="drive_motor_', '<velocity kv="8" name="drive_motor_')
    xml = xml.replace(WORLDBODY_OPEN, WORLDBODY_OPEN + _FLOOR, 1)
    xml = xml.replace(
        ROOT_BODY_TAG,
        f'<body name="chassis" pos="0 0 0">\n        <freejoint name="base_free"/>\n        {ROOT_BODY_TAG}',
        1,
    )
    xml = xml.replace(WORLDBODY_CLOSE, f"        </body>\n    {WORLDBODY_CLOSE}", 1)
    return xml
