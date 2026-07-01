# lekiwi-dora-node

A dora dataflow node that exposes the LeKiwi mobile-manipulation robot (3-wheel holonomic base + SO-ARM100 6-DOF arm + gripper) as a single octos SPEC robot.

## What it does

- **Base verbs** (octos SPEC, `vendor.dora_nav.base.*`):
  - `set_velocity` — continuous holonomic velocity (vx, vy, omega)
  - `stop` — zero all wheel actuators and cancel any pending `go_to_pose`
  - `go_to_pose` — closed-loop position control; resolves when the pose error falls inside the HolonomicController tolerance
- **Arm verbs** (octos SPEC, `vendor.lerobot.arm.*`):
  - `move_to_joint_state` — set absolute joint targets (up to 6 DOF + gripper)
  - `move_to_named` — move to a pre-configured named pose (e.g. `home`)
- **Emergency stop** (`robot.estop`) — zeroes base velocity, clears arm target, aborts any in-progress deferred operations with a `VENDOR_ERROR` response.

Control outputs are assembled in MJCF actuator order and emitted as `control` (float32 array) to the dora-mujoco sim node.

## Actuator order

```
drive_motor_1  drive_motor_2  drive_motor_3  Rotation  Pitch  Elbow  Wrist_Pitch  Wrist_Roll  Jaw
```

This matches the `<actuator>` block in `assets/mjcf_lcmm_robot.xml` (vendored from LeKiwi-sim).

## Running tests (offline)

```bash
cd lekiwi-dora-node
PYTHONPATH=. pytest tests -v
```

All 63 tests should pass without a running dora runtime. The `test_sim_scene.py`
MuJoCo-layout checks skip automatically where `mujoco` is not installed; the
scene-string and base-integrator assertions still run.

## Simulation dataflow

`dataflows/lekiwi-mujoco-bridge.yml` wires three nodes:

1. **mujoco_sim** — the lekiwi-specific sim node (`python -m lekiwi_node.mujoco_sim`), builds the physics-ready scene from `assets/mjcf_lcmm_robot.xml`, publishes `joint_positions`, consumes `control`.
2. **lekiwi_node** — this package (`python -m lekiwi_node`), consumes `joint_positions` + `cmd_request`, emits `control` + `cmd_response`.
3. **bridge** — `octos_spec_bridge` on HTTP port 8770, routes SPEC commands to/from lekiwi_node.

To run the dataflow on a sim host (venv-python needs `dora`, `mujoco`, `drive-kinematics`, `lekiwi_node`, `octos_spec_bridge`):

```bash
export LEKIWI_NODE=/path/to/lekiwi-dora-node
dora up
dora start dataflows/lekiwi-mujoco-bridge.yml
```

## Sim model & base loop (resolved offline)

The shipped `assets/mjcf_lcmm_robot.xml` is a CAD export that is **not** runnable
as a mobile robot; `lekiwi_node/sim_scene.py::build_scene()` makes it physics-ready
by text-injection (source never mutated), and `lekiwi_node/mujoco_sim.py` closes
the base loop. All of the following were verified in headless MuJoCo (see
`tests/test_sim_scene.py` and the offline forward/strafe/spin rollout):

- **A1 (no free joint)** — the source welds three separate top-level bodies
  (base_plate_layer1 = wheels, base_plate_layer2 = arm, drive_motor_mount-v4 =
  orphan) to the world. `build_scene` wraps all three in one `chassis` body with a
  `<freejoint name="base_free">`, so the whole robot is one rigid free body.
- **A2 (`<motor>` → `<velocity>`)** — the three `drive_motor_*` actuators are
  converted to `<velocity kv="8">`, so `ctrl` is a target wheel speed (rad/s).
- **Contacts/gravity off** — the shipped omniwheels are single rigid bodies (no
  rollers), so wheel-ground contact **cannot** produce holonomic motion; confirmed
  empirically that spinning the wheels moves the free base 0.000000 m. Base motion
  is therefore **kinematically integrated**: `mujoco_sim` recovers the body twist
  from the wheel speeds (`KiwiDrive.wheels_to_body`, exact inverse of the runtime
  IK), forward-integrates the pose, and writes it into the `base_free` qpos. The
  wheels still physically spin (visual fidelity). This is why the dataflow uses a
  lekiwi-specific sim node, not the generic dora-mujoco runner — and why the tick
  period must equal `SIM_DT`.
- **V2 (`joint_positions` layout)** — verified nq=16: `base_free` at qpos **0:7**
  (x, y, z, qw, qx, qy, qz), 3 wheel hinges at **7:10**, arm joints at **10:16**.
  `__main__.py` extractors read base 0:6 and arm 10:16 accordingly (the earlier
  7:12 arm assumption was wrong).
- **A3 (wheel-index ↔ angle mapping)** — **not load-bearing**: base pose is
  integrated, so the physical wheel assignment only affects the visual spin. The
  IK/FK round-trips exactly regardless of ordering.

### Still pending (needs a live sim host; asus currently unreachable)

- Full dataflow bring-up under a running `dora` daemon with `octos_spec_bridge`
  (the offline rollout exercises the exact command→wheels→recover→integrate path,
  but not the HTTP/SPEC front or dora transport).
- **A4** — estop is latching with no release verb (unchanged; a product decision).
