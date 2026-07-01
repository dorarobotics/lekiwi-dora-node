# Design: `lekiwi-dora-node` — LeKiwi (arm + 3-wheel base) as one octos robot

**Date:** 2026-06-30
**Status:** approved (design); pending implementation plan
**Repo:** `lekiwi-dora-node` (new)
**Scope:** sub-project **B** of the LeKiwi effort (see §12). MuJoCo-sim-first.

## 1. Goal

One dora node — `lekiwi-dora-node` — that presents the **LeKiwi mobile manipulator (SO-arm + 3-wheel omnidirectional "kiwi" base) as a single octos robot** over one `SPEC-VENDOR-NODE-V1` endpoint, driving the existing LeKiwi MuJoCo model. An octos agent calls both the base and the arm through one `robot_id: lekiwi`, exactly as it calls any other SPEC vendor.

This is the standalone robot repo. It carries the **kiwi kinematics as the robot's own base driver** until the drive-agnostic nav work (sub-project A) lands in `dora-nav`; nothing here blocks on that.

## 2. Architecture

A single dora node exposing **two capability namespaces**, layered like the proven `hunter_base` testable split (pure logic units + a thin dora runtime), plus an arm driver. The MuJoCo sim is the **reused `dora-mujoco` runner** (as in the Hunter demo), pointed at the LeKiwi model.

```
octos agent
   │  POST /tools/<verb>  {"args":{…}}
   ▼
octos_spec_bridge            (one HTTP port)
   │  cmd_request / cmd_response  (dora topics)
   ▼
lekiwi-dora-node   (robot_id: lekiwi)
   ├─ base verbs: vendor.dora_nav.base.{set_velocity, stop, go_to_pose}
   │     go_to_pose → holonomic P-controller → Twist(vx,vy,ω)
   │     Twist → KiwiDrive.body_to_wheels → [w1,w2,w3] (rad/s)
   ├─ arm verbs:  vendor.lerobot.arm.{move_to_joint_state, move_to_named}
   │     joint targets (5-DOF + gripper)
   ▼  control vector (3 wheel vel + 6 arm pos)
dora-mujoco runner  ← LeKiwi MJCF (LeKiwi-sim/mjcf_lcmm_robot.xml)
   │  joint_positions (base pose + arm joints)  ▲ closes the loop
   └──────────────────────────────────────────┘
```

## 3. KiwiDrive kinematics (ported from LeRobot `lekiwi.py`)

Three omniwheels mounted at body angles **[150°, −90°, 30°]** (`np.radians([240, 0, 120] − 90)`), wheel radius **r = 0.05 m**, base radius **R = 0.125 m**.

- **body → wheels:** `M = [[cos aᵢ, sin aᵢ, R] for aᵢ in angles]`; `wheel_linear = M · [vx, vy, ω]` (m/s); `wheel_ω = wheel_linear / r` (rad/s).
- **wheels → body:** `[vx, vy, ω] = M⁻¹ · (wheel_ω · r)`.

`M` is 3×3 and invertible, so the mapping round-trips exactly (a core test). **Units (SI at the SPEC boundary):** `vx, vy` in m/s, `ω` in **rad/s**. (LeRobot's servo path uses deg/s and raw ticks; that is a hardware concern — the sim path is SI, commanding MuJoCo wheel-velocity actuators in rad/s.) Wheel commands are clamped to a configurable `max_wheel_omega`.

`kinematics.py` is pure: `body_to_wheels(vx, vy, omega) -> (w1, w2, w3)` and `wheels_to_body(w1, w2, w3) -> Twist`. The angle/r/R constants match the MJCF wheel layout (verified during implementation, §7).

## 4. SPEC verbs

| Namespace | Verb | Params | Deferred? | Notes |
|---|---|---|---|---|
| `vendor.dora_nav.base` | `set_velocity` | `{vx, vy, omega}` | no | holonomic body Twist → KiwiDrive → wheels; watchdog-guarded |
| | `stop` | `{}` | no | privileged; zero Twist |
| | `go_to_pose` | `{pose:{position, orientation}}` | **yes** | holonomic P-controller drives to target; resolves on reached / `BRIDGE_TIMEOUT` |
| `vendor.lerobot.arm` | `move_to_joint_state` | `{joints:[5], gripper?}` | **yes** | replay calibrated joint target; resolves on joint tolerance / timeout |
| | `move_to_named` | `{name}` | **yes** | named poses (`home`, …) from config |

Base uses the **`vendor.dora_nav.base.*`** namespace deliberately: LeKiwi drops straight into the existing octos inspection-agent pattern and inherits the future `dora-nav` contract widening (sub-project A). Arm uses a new **`vendor.lerobot.arm.*`** namespace (sibling to `vendor.moveit.arm.*`), reflecting direct joint control of a LeRobot arm rather than MoveIt planning.

## 5. Base holonomic controller

`base_controller.py` (pure): `step(current: Pose2D, target: Pose2D) -> (Twist, reached)`.

- World-frame error `(dx, dy, dyaw)` → rotate `(dx, dy)` into the **body frame** by `−current.yaw` → `Twist(vx, vy, ω)` via proportional gains, each clamped.
- `reached = hypot(dx,dy) ≤ xy_tol AND |dyaw| ≤ yaw_tol`. Because the base is **holonomic**, translation and rotation are controlled independently — no steering constraint, no arc-around. Targets in any direction (lateral, reverse, in-place spin) are directly reachable. This is the structural contrast with the Ackermann `hunter_base`, whose reverse/cornering cases are hard.

`set_velocity` bypasses the controller (direct Twist). `stop` emits zero Twist and cancels any pending `go_to_pose`.

## 6. Arm driver

`arm_driver.py` (pure): tracks the current 5-DOF + gripper target and reports `reached` when measured joints are within tolerance. `move_to_joint_state` sets the target from `joints` (+ optional `gripper`); `move_to_named` looks the target up in a small named-pose table (`home` required; others configurable). No MoveIt/IK — direct joint-space targets replayed to MuJoCo position actuators, matching the LeRobot/servo nature of the arm and the `hunter_base` inspect-pose pattern.

## 7. Sim integration (reuse `dora-mujoco`)

The MuJoCo model is **`LeKiwi-sim/mjcf_lcmm_robot.xml`** (the only local model with both arm and base), vendored into `assets/`. Its actuators:

| Logical (LeRobot) | MJCF actuator | Kind |
|---|---|---|
| base wheels ×3 | `drive_motor_1`, `drive_motor_2`, `drive_motor_3` | velocity |
| `arm_shoulder_pan` | `Rotation` | position |
| `arm_shoulder_lift` | `Pitch` | position |
| `arm_elbow_flex` | `Elbow` | position |
| `arm_wrist_flex` | `Wrist_Pitch` | position |
| `arm_wrist_roll` | `Wrist_Roll` | position |
| `arm_gripper` | `Jaw` | position |

The reused `dora-mujoco` runner loads this MJCF, streams `joint_positions` (base pose + arm joints), and applies a control vector. `runtime.py` assembles that vector in MJCF actuator order. **Two integration facts to verify during implementation** (flagged, not assumed): (a) the wheel-index→angle mapping — that `drive_motor_1/2/3` correspond to the `[150°, −90°, 30°]` rows (re-order the KiwiDrive rows to match if not); (b) the base pose channel the runner exposes for `go_to_pose` feedback (free-joint xy+yaw of the base body).

## 8. Components & isolation

| File | Responsibility | Depends on |
|---|---|---|
| `lekiwi_node/geometry.py` | `Pose2D`, `wrap_angle`, `yaw_from_quat` | stdlib |
| `lekiwi_node/kinematics.py` | `KiwiDrive.body_to_wheels` / `wheels_to_body` | numpy |
| `lekiwi_node/base_controller.py` | holonomic `step()` → Twist, reached | geometry |
| `lekiwi_node/arm_driver.py` | joint-target tracking, named poses | — |
| `lekiwi_node/_envelope.py` | SPEC-V1 parse/build (one contract) | — |
| `lekiwi_node/node.py` | verb registry + dispatch (both namespaces) + capabilities | kinematics, controllers, envelope |
| `lekiwi_node/runtime.py` | dora loop: dispatch, control-vector assembly, deferred resolution, state stream | node |
| `lekiwi_node/__main__.py` | dora entry point | runtime |
| `dataflows/lekiwi-mujoco-bridge.yml` | sim (`dora-mujoco`) + node + `octos_spec_bridge` | — |
| `assets/mjcf_lcmm_robot.xml` | vendored LeKiwi model | — |
| `tests/…` | unit tests per pure unit + runtime | — |

## 9. Error handling

- Unknown verb / bad params → SPEC `INVALID_PARAMS`. Motion during estop → `VENDOR_ERROR`. Concurrent motion → `CONTROLLER_BUSY` (per-namespace lock: base and arm may move independently but each is single-flight).
- `go_to_pose` / arm moves are deferred; they resolve `ok` on reached or `BRIDGE_TIMEOUT` on deadline. `set_velocity` is watchdog-guarded — a stale command auto-stops.
- Wheel commands clamped to `max_wheel_omega`; NaN/limit guards on the control vector.

## 10. Testing (TDD)

- **kinematics:** round-trip `wheels_to_body(body_to_wheels(v)) ≈ v`; known vectors — pure forward (`vx>0`), pure **strafe** (`vy>0`, the holonomic case Ackermann can't do), pure spin (`ω>0`) — against the LeRobot matrix; clamp behavior.
- **base controller:** convergence rollouts to forward / lateral / reverse / spin targets (all reachable — the Ackermann failure cases are trivial here); `reached` honors both `xy_tol` and `yaw_tol`; body-frame rotation of the error is correct.
- **arm driver:** target set/reached tolerance; `move_to_named('home')`; unknown named pose → failure.
- **envelope + node:** dispatch across both namespaces; per-namespace `CONTROLLER_BUSY`; capabilities advert lists all verbs.
- **runtime:** fake dora node — `cmd_request`→dispatch, `joint_positions`→controllers→control vector in MJCF actuator order, deferred resolution + timeout.

## 11. Footprint

New self-contained repo. Reuses: the `dora-mujoco` runner (sim), the `octos_spec_bridge` (HTTP front), the local `LeKiwi-sim` MJCF (vendored), and the KiwiDrive math ported from LeRobot `lekiwi.py`. Mirrors `hunter_base`'s structure and the SPEC contract used across the octos vendors. No changes to `nav-base-dora-node`, `dora-nav`, or the bridge in this sub-project.

## 12. Relationship to the other sub-projects (out of scope here)

- **A — drive-agnostic nav (`dora-nav`):** widen the nav velocity contract `{linear, angular}` → `{vx, vy, ω}` and add a `BaseKinematics` trait (`Diff/Kiwi/Mecanum/Ackermann`) in dora-nav's controller. LeKiwi's KiwiDrive here is the reference and the interim driver; when A lands, the robot can defer wheel-mapping to the nav stack. **Not built here.**
- **C — octos bridge integration:** register LeKiwi as a skill in `octos-dora-bridge` (one robot, both namespaces) and drive it from the octos inspection agent. Depends on B. **Not built here.**

## 13. Out of scope (YAGNI)

Real hardware (LeRobot/Feetech servos, `lekiwi_host`); MoveIt/IK for the arm; perception/VQA; multi-robot; the dora-nav trait work (sub-project A); bridge/agent wiring (sub-project C). Base uses SI (m/s, rad/s) in sim — the deg/s + raw-tick servo path is a hardware concern deferred with real-hardware support.
