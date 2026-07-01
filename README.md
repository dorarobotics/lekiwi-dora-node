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

All 56 tests should pass without a running dora runtime or MuJoCo instance.

## Simulation dataflow

`dataflows/lekiwi-mujoco-bridge.yml` wires three nodes:

1. **mujoco_sim** — dora-mujoco runner, loads `assets/mjcf_lcmm_robot.xml`, publishes `joint_positions`, consumes `control`.
2. **lekiwi_node** — this package (`python -m lekiwi_node`), consumes `joint_positions` + `cmd_request`, emits `control` + `cmd_response`.
3. **bridge** — `octos_spec_bridge` on HTTP port 8770, routes SPEC commands to/from lekiwi_node.

To run the dataflow on a sim host:

```bash
# Resolve env vars and run
export DORA_MOVEIT2=/path/to/dora-moveit2
export LEKIWI_NODE=/path/to/lekiwi-dora-node
dora up
dora start dataflows/lekiwi-mujoco-bridge.yml
```

## Live bring-up (pending)

The following require a live sim host and are deferred. The vendored `assets/mjcf_lcmm_robot.xml` needs sim work (A1–A3 below) before the live demo can run.

- **V1 (wheel-index / angle correspondence)** — the MJCF mounts three omni-wheels at specific headings; the KiwiDrive mount angles (hard-coded inline in `lekiwi_node/kinematics.py` as `np.radians(np.array([240, 0, 120]) - 90)` = [150°, −90°, 30°]) must be verified against the physical layout and adjusted if the robot drifts instead of translating cleanly. See A3.
- **V2 (`joint_positions` layout)** — `__main__.py` assumes the free-joint DOF occupy indices 0–6 (x, y, z, qw, qx, qy, qz) and arm joints occupy indices 7–12. This must be confirmed against the actual dora-mujoco `joint_positions` output for this MJCF before deploying.
- **A1 (base has no free joint)** — the vendored `mjcf_lcmm_robot.xml` welds the base to the worldbody (only 9 hinge joints; no `<freejoint>`), so the base cannot translate and `joint_positions` won't contain the base pose the extractors assume. Add a base free joint (or float it in the runner) and re-confirm the `_base_pose` index layout.
- **A2 (wheel actuators are `<motor>`, i.e. force, not `<velocity>`)** — the code commands wheel angular velocities (rad/s), but the three `drive_motor_*` actuators are `<motor>` (force) actuators. Change them to `<velocity>` (or document the runner's force/velocity conversion).
- **A3 (wheel-index↔angle mapping is confirmed mismatched, V1)** — by actuator index the MJCF mounts are drive_motor_1 = −90°, _2 = +30°, _3 = +150°, but KiwiDrive emits rows [150°, −90°, 30°] to w1/w2/w3 — a cyclic reorder is required.
- **A4 (safety + extractor gaps)** — estop is latching with no release verb; and the extractor quaternion/slice assumptions (`_base_pose`/`_arm_joints`) are unit-untested (the V2 surface).
