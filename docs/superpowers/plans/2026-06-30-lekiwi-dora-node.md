# lekiwi-dora-node Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A single dora node presenting the LeKiwi mobile manipulator (SO-arm + 3-wheel holonomic "kiwi" base) as one octos robot over a `SPEC-VENDOR-NODE-V1` endpoint, driving the local LeKiwi MuJoCo model via the reused `dora-mujoco` runner.

**Architecture:** Pure logic units (geometry, KiwiDrive kinematics, holonomic controller, arm driver, SPEC envelope) + a `LekiwiNode` verb-dispatch object + a thin `LekiwiRuntime` dora loop — mirroring the proven `hunter_base` split, extended with an arm namespace. Base and arm are per-namespace single-flight and can move independently.

**Tech Stack:** Python 3.10, numpy, pyarrow, dora-rs 0.4.0, pytest. Spec: `docs/superpowers/specs/2026-06-30-lekiwi-dora-node-design.md`. Kinematics ported from LeRobot `lekiwi.py` (angles `[150°,−90°,30°]`, r=0.05 m, R=0.125 m). Sim model: `LeKiwi-sim/mjcf_lcmm_robot.xml`.

Run tests: `PYTHONPATH=. pytest tests -v` (any venv with numpy+pyarrow+pytest).

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | package metadata + deps (numpy, pyarrow) |
| `lekiwi_node/__init__.py` | package marker |
| `lekiwi_node/geometry.py` | `Pose2D`, `Twist`, `wrap_angle`, `yaw_from_quat` (pure) |
| `lekiwi_node/kinematics.py` | `KiwiDrive.body_to_wheels` / `wheels_to_body` (pure) |
| `lekiwi_node/base_controller.py` | `HolonomicController.step()` — pose+target → Twist, reached (pure) |
| `lekiwi_node/arm_driver.py` | `ArmDriver` — joint-target tracking + named poses (pure) |
| `lekiwi_node/_envelope.py` | SPEC-V1 `parse_cmd_request` / `build_cmd_response` |
| `lekiwi_node/control.py` | `assemble_control()` — logical wheel/arm values → MJCF-actuator-ordered vector (pure) |
| `lekiwi_node/node.py` | `LekiwiNode` — verb registry + dispatch (both namespaces) + capabilities |
| `lekiwi_node/runtime.py` | `LekiwiRuntime` — dora loop, per-namespace deferred resolution, state stream |
| `lekiwi_node/__main__.py` | dora entry point |
| `dataflows/lekiwi-mujoco-bridge.yml` | sim (`dora-mujoco`) + node + `octos_spec_bridge` |
| `assets/mjcf_lcmm_robot.xml` | vendored LeKiwi model |
| `tests/…` | one test module per unit |

---

### Task 1: Package scaffold + geometry

**Files:**
- Create: `pyproject.toml`, `lekiwi_node/__init__.py`, `tests/__init__.py`
- Create: `lekiwi_node/geometry.py`
- Test: `tests/test_geometry.py`

- [ ] **Step 1: Write `pyproject.toml` and empty package markers**

```toml
# pyproject.toml
[project]
name = "lekiwi-dora-node"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["numpy", "pyarrow"]

[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"
```

Create empty `lekiwi_node/__init__.py` and `tests/__init__.py`.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_geometry.py
import math
from lekiwi_node.geometry import Pose2D, Twist, wrap_angle, yaw_from_quat


def test_yaw_from_identity_quat_is_zero():
    assert abs(yaw_from_quat(1.0, 0.0, 0.0, 0.0)) < 1e-9


def test_yaw_from_90deg_about_z():
    c = math.cos(math.pi / 4)
    assert abs(yaw_from_quat(c, 0.0, 0.0, c) - math.pi / 2) < 1e-6


def test_wrap_angle_folds_into_pi():
    assert abs(wrap_angle(3 * math.pi) - math.pi) < 1e-9
    assert abs(wrap_angle(-3 * math.pi) - math.pi) < 1e-9


def test_pose_and_twist_are_frozen_dataclasses():
    p = Pose2D(1.0, 2.0, 0.5)
    t = Twist(0.1, -0.2, 0.3)
    assert (p.x, p.y, p.yaw) == (1.0, 2.0, 0.5)
    assert (t.vx, t.vy, t.omega) == (0.1, -0.2, 0.3)
```

Run: `PYTHONPATH=. pytest tests/test_geometry.py -v` → FAIL (no module).

- [ ] **Step 3: Implement `lekiwi_node/geometry.py`**

```python
# lekiwi_node/geometry.py
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float  # radians


@dataclass(frozen=True)
class Twist:
    vx: float   # m/s, body frame
    vy: float   # m/s, body frame (holonomic lateral)
    omega: float  # rad/s


def yaw_from_quat(qw: float, qx: float, qy: float, qz: float) -> float:
    """Z-axis yaw (radians) from a quaternion (w, x, y, z)."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(a: float) -> float:
    """Fold an angle into (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))
```

- [ ] **Step 4: Run tests** → `PYTHONPATH=. pytest tests/test_geometry.py -v` → PASS (4).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml lekiwi_node/__init__.py lekiwi_node/geometry.py tests/__init__.py tests/test_geometry.py
git commit -m "feat: package scaffold + geometry (Pose2D, Twist, quat/angle helpers)"
```

---

### Task 2: KiwiDrive kinematics

**Files:**
- Create: `lekiwi_node/kinematics.py`
- Test: `tests/test_kinematics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kinematics.py
import math
from lekiwi_node.kinematics import KiwiDrive
from lekiwi_node.geometry import Twist


def test_pure_spin_all_wheels_equal():
    # omega only -> every wheel spins the same (pure rotation).
    k = KiwiDrive()
    w1, w2, w3 = k.body_to_wheels(0.0, 0.0, 1.0)
    assert abs(w1 - w2) < 1e-9 and abs(w2 - w3) < 1e-9
    assert w1 > 0


def test_pure_forward_back_wheel_is_zero():
    # +x forward: back wheel (mounted at -90 deg) contributes ~no drive.
    k = KiwiDrive()
    w1, w2, w3 = k.body_to_wheels(0.3, 0.0, 0.0)
    assert abs(w2) < 1e-9          # back wheel
    assert abs(w1 + w3) < 1e-9     # symmetric left/right


def test_pure_strafe_back_wheel_dominant():
    # +y strafe: back wheel carries the largest share, opposite sign to sides.
    k = KiwiDrive()
    w1, w2, w3 = k.body_to_wheels(0.0, 0.3, 0.0)
    assert w2 < 0 and w1 > 0 and w3 > 0
    assert abs(w2) > abs(w1)


def test_round_trip_recovers_body_velocity():
    k = KiwiDrive()
    for vx, vy, om in [(0.2, 0.0, 0.0), (0.0, 0.15, 0.0), (0.0, 0.0, 0.5), (0.1, -0.1, 0.3)]:
        w = k.body_to_wheels(vx, vy, om)
        t = k.wheels_to_body(*w)
        assert abs(t.vx - vx) < 1e-9 and abs(t.vy - vy) < 1e-9 and abs(t.omega - om) < 1e-9


def test_wheels_clamped_to_max():
    k = KiwiDrive(max_wheel_omega=5.0)
    w = k.body_to_wheels(100.0, 0.0, 0.0)
    assert all(abs(x) <= 5.0 + 1e-9 for x in w)
```

Run → FAIL (no module).

- [ ] **Step 2: Implement `lekiwi_node/kinematics.py`**

```python
# lekiwi_node/kinematics.py
from __future__ import annotations

import math

import numpy as np

from lekiwi_node.geometry import Twist


class KiwiDrive:
    """3-wheel omnidirectional (kiwi) kinematics, ported from LeRobot lekiwi.py.

    Wheels mounted at body angles [150, -90, 30] deg (= radians([240,0,120]) - 90).
    body_to_wheels: (vx, vy, omega) [m/s, m/s, rad/s] -> 3 wheel angular speeds [rad/s].
    wheels_to_body: inverse. The 3x3 mount matrix is invertible, so it round-trips.
    """

    def __init__(self, wheel_radius: float = 0.05, base_radius: float = 0.125,
                 max_wheel_omega: float = 30.0) -> None:
        self.wheel_radius = wheel_radius
        self.base_radius = base_radius
        self.max_wheel_omega = max_wheel_omega
        angles = np.radians(np.array([240.0, 0.0, 120.0]) - 90.0)  # [150, -90, 30] deg
        self._m = np.array([[math.cos(a), math.sin(a), base_radius] for a in angles])
        self._m_inv = np.linalg.inv(self._m)

    def body_to_wheels(self, vx: float, vy: float, omega: float) -> tuple[float, float, float]:
        wheel_linear = self._m.dot(np.array([vx, vy, omega], dtype=float))
        wheel_omega = wheel_linear / self.wheel_radius
        clamped = np.clip(wheel_omega, -self.max_wheel_omega, self.max_wheel_omega)
        return (float(clamped[0]), float(clamped[1]), float(clamped[2]))

    def wheels_to_body(self, w1: float, w2: float, w3: float) -> Twist:
        wheel_linear = np.array([w1, w2, w3], dtype=float) * self.wheel_radius
        vx, vy, omega = self._m_inv.dot(wheel_linear)
        return Twist(float(vx), float(vy), float(omega))
```

- [ ] **Step 3: Run tests** → PASS (5). Note round-trip uses values whose wheels stay under `max_wheel_omega` (default 30), so clamping does not distort them.

- [ ] **Step 4: Commit**

```bash
git add lekiwi_node/kinematics.py tests/test_kinematics.py
git commit -m "feat: KiwiDrive kinematics (body<->wheels, ported from LeRobot lekiwi.py)"
```

---

### Task 3: Holonomic base controller

**Files:**
- Create: `lekiwi_node/base_controller.py`
- Test: `tests/test_base_controller.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_base_controller.py
import math
from lekiwi_node.base_controller import HolonomicController
from lekiwi_node.geometry import Pose2D


def _rollout(c, start, target, steps=4000, dt=0.05):
    """Ideal holonomic integration: body Twist rotated into world by current yaw."""
    p = start
    reached = False
    for _ in range(steps):
        tw, reached = c.step(p, target)
        if reached:
            break
        cs, sn = math.cos(p.yaw), math.sin(p.yaw)
        wx = cs * tw.vx - sn * tw.vy
        wy = sn * tw.vx + cs * tw.vy
        p = Pose2D(p.x + wx * dt, p.y + wy * dt, p.yaw + tw.omega * dt)
    return p, reached


def test_reached_within_tolerance_stops():
    c = HolonomicController(xy_tol=0.05, yaw_tol=0.05)
    tw, reached = c.step(Pose2D(1.0, 1.0, 0.0), Pose2D(1.02, 1.0, 0.0))
    assert reached is True
    assert (tw.vx, tw.vy, tw.omega) == (0.0, 0.0, 0.0)


def test_target_left_commands_positive_vy():
    # facing +x, target straight left in world (+y) -> body vy > 0, vx ~ 0.
    c = HolonomicController()
    tw, reached = c.step(Pose2D(0.0, 0.0, 0.0), Pose2D(0.0, 1.0, 0.0))
    assert reached is False
    assert tw.vy > 0 and abs(tw.vx) < 1e-9


def test_target_behind_commands_negative_vx():
    # target straight behind (world -x) -> body vx < 0 (reverse), trivially reachable.
    c = HolonomicController()
    tw, _ = c.step(Pose2D(0.0, 0.0, 0.0), Pose2D(-1.0, 0.0, 0.0))
    assert tw.vx < 0 and abs(tw.vy) < 1e-9


def test_pure_yaw_error_commands_omega_only():
    c = HolonomicController(xy_tol=0.05, yaw_tol=0.02)
    tw, reached = c.step(Pose2D(0.0, 0.0, 0.0), Pose2D(0.0, 0.0, 1.0))
    assert reached is False
    assert abs(tw.vx) < 1e-9 and abs(tw.vy) < 1e-9 and tw.omega > 0


def test_error_is_rotated_into_body_frame():
    # facing +y (yaw=pi/2); target ahead in world +y -> body vx > 0 (forward).
    c = HolonomicController()
    tw, _ = c.step(Pose2D(0.0, 0.0, math.pi / 2), Pose2D(0.0, 1.0, math.pi / 2))
    assert tw.vx > 0 and abs(tw.vy) < 1e-6


def test_converges_lateral_and_reverse_and_spin_targets():
    c = HolonomicController()
    for target in [Pose2D(0.0, 1.5, 0.0), Pose2D(-1.5, 0.0, 0.0), Pose2D(1.0, -1.0, math.pi / 2)]:
        p, reached = _rollout(c, Pose2D(0.0, 0.0, 0.0), target)
        assert reached is True
        assert math.hypot(p.x - target.x, p.y - target.y) <= 0.05
        assert abs(math.atan2(math.sin(p.yaw - target.yaw), math.cos(p.yaw - target.yaw))) <= 0.05
```

Run → FAIL (no module).

- [ ] **Step 2: Implement `lekiwi_node/base_controller.py`**

```python
# lekiwi_node/base_controller.py
from __future__ import annotations

import math

from lekiwi_node.geometry import Pose2D, Twist, wrap_angle


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class HolonomicController:
    """Proportional go-to-pose controller for a holonomic (kiwi) base. Translation
    and yaw are controlled independently — no steering constraint, so lateral,
    reverse, and in-place spin targets are all directly reachable."""

    def __init__(self, *, xy_tol: float = 0.05, yaw_tol: float = 0.05,
                 k_lin: float = 1.5, k_ang: float = 1.5,
                 max_lin: float = 0.5, max_ang: float = 1.5) -> None:
        self.xy_tol = xy_tol
        self.yaw_tol = yaw_tol
        self.k_lin = k_lin
        self.k_ang = k_ang
        self.max_lin = max_lin
        self.max_ang = max_ang

    def step(self, current: Pose2D, target: Pose2D) -> tuple[Twist, bool]:
        dx = target.x - current.x
        dy = target.y - current.y
        dyaw = wrap_angle(target.yaw - current.yaw)
        if math.hypot(dx, dy) <= self.xy_tol and abs(dyaw) <= self.yaw_tol:
            return Twist(0.0, 0.0, 0.0), True
        # rotate world-frame error into the body frame (by -current.yaw)
        cs, sn = math.cos(current.yaw), math.sin(current.yaw)
        ex = cs * dx + sn * dy
        ey = -sn * dx + cs * dy
        vx = _clamp(self.k_lin * ex, -self.max_lin, self.max_lin)
        vy = _clamp(self.k_lin * ey, -self.max_lin, self.max_lin)
        omega = _clamp(self.k_ang * dyaw, -self.max_ang, self.max_ang)
        return Twist(vx, vy, omega), False
```

- [ ] **Step 3: Run tests** → PASS (6).

- [ ] **Step 4: Commit**

```bash
git add lekiwi_node/base_controller.py tests/test_base_controller.py
git commit -m "feat: holonomic go-to-pose controller (lateral/reverse/spin reachable)"
```

---

### Task 4: Arm driver

**Files:**
- Create: `lekiwi_node/arm_driver.py`
- Test: `tests/test_arm_driver.py`

The arm has 6 controllable DOF: 5 joints + gripper, in the order
`[Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw]`. The driver deals in full
6-vectors; the node (Task 6) maps SPEC params to a 6-vector before calling it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_arm_driver.py
from lekiwi_node.arm_driver import ArmDriver

HOME = [0.0, 0.0, 0.0, 0.0, 0.0, 0.5]


def _driver():
    return ArmDriver(named_poses={"home": HOME}, joint_tol=0.05, dof=6)


def test_no_target_initially():
    assert _driver().target is None


def test_set_target_and_reached_tolerance():
    d = _driver()
    d.set_target([0.1, 0.2, 0.0, -0.3, 0.0, 0.4])
    assert d.target == [0.1, 0.2, 0.0, -0.3, 0.0, 0.4]
    assert d.reached([0.1, 0.2, 0.0, -0.3, 0.0, 0.4]) is True
    assert d.reached([0.1, 0.2, 0.0, -0.3, 0.0, 0.49]) is True   # within 0.05
    assert d.reached([0.1, 0.2, 0.0, -0.3, 0.0, 0.9]) is False   # gripper off


def test_set_target_wrong_length_rejected():
    d = _driver()
    try:
        d.set_target([0.0, 0.0, 0.0])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_named_home_sets_target():
    d = _driver()
    assert d.set_named("home") is True
    assert d.target == HOME


def test_unknown_named_returns_false_and_leaves_target():
    d = _driver()
    d.set_target([0.0] * 6)
    assert d.set_named("nope") is False
    assert d.target == [0.0] * 6


def test_reached_false_when_no_target():
    assert _driver().reached([0.0] * 6) is False
```

Run → FAIL (no module).

- [ ] **Step 2: Implement `lekiwi_node/arm_driver.py`**

```python
# lekiwi_node/arm_driver.py
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
```

- [ ] **Step 3: Run tests** → PASS (6).

- [ ] **Step 4: Commit**

```bash
git add lekiwi_node/arm_driver.py tests/test_arm_driver.py
git commit -m "feat: arm driver (joint-target tracking + named poses)"
```

---

### Task 5: SPEC envelope + control-vector assembly

**Files:**
- Create: `lekiwi_node/_envelope.py`, `lekiwi_node/control.py`
- Test: `tests/test_envelope.py`, `tests/test_control.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_envelope.py
from lekiwi_node._envelope import parse_cmd_request, build_cmd_response, InvalidEnvelope


def test_parse_minimal_request():
    req = parse_cmd_request({"verb": "vendor.dora_nav.base.stop", "request_id": "r1"})
    assert req.verb == "vendor.dora_nav.base.stop"
    assert req.request_id == "r1"
    assert req.params == {}


def test_parse_missing_verb_raises():
    try:
        parse_cmd_request({"request_id": "r1"})
        assert False
    except InvalidEnvelope:
        pass


def test_build_response_shape():
    req = parse_cmd_request({"verb": "v", "request_id": "r2", "trace_id": "t"})
    resp = build_cmd_response(req, ok=True, code="0", data={"k": 1})
    assert resp["ok"] is True and resp["code"] == "0"
    assert resp["request_id"] == "r2" and resp["trace_id"] == "t"
    assert resp["data"] == {"k": 1} and "ts" in resp
```

```python
# tests/test_control.py
from lekiwi_node.control import assemble_control


def test_assemble_orders_by_actuator_name():
    order = ["drive_motor_1", "drive_motor_2", "drive_motor_3",
             "Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
    values = {"drive_motor_1": 1.0, "drive_motor_2": 2.0, "drive_motor_3": 3.0,
              "Rotation": 0.1, "Pitch": 0.2, "Elbow": 0.3,
              "Wrist_Pitch": 0.4, "Wrist_Roll": 0.5, "Jaw": 0.6}
    vec = assemble_control(order, values)
    assert vec == [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def test_missing_actuator_defaults_zero():
    vec = assemble_control(["a", "b"], {"a": 1.5})
    assert vec == [1.5, 0.0]
```

Run → FAIL.

- [ ] **Step 2: Implement `lekiwi_node/_envelope.py`** (identical wire contract to the other octos vendors)

```python
# lekiwi_node/_envelope.py
"""SPEC-VENDOR-NODE-V1 envelope helpers — same wire contract as the octos vendors."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


class InvalidEnvelope(ValueError):
    pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class CmdRequest:
    request_id: str
    verb: str
    params: dict[str, Any]
    target: str | None
    spec_version: str
    trace_id: str | None


def parse_cmd_request(env: dict[str, Any]) -> CmdRequest:
    if "verb" not in env:
        raise InvalidEnvelope("cmd_request missing required field: verb")
    return CmdRequest(
        request_id=str(env.get("request_id", "")),
        verb=str(env["verb"]),
        params=dict(env.get("params") or {}),
        target=env.get("target"),
        spec_version=str(env.get("spec_version", "1.0.0")),
        trace_id=env.get("trace_id"),
    )


def build_cmd_response(request: CmdRequest, *, ok: bool, code: str,
                       data: dict[str, Any] | None = None, msg: str = "") -> dict[str, Any]:
    resp: dict[str, Any] = {
        "envelope_version": "1.0",
        "spec_version": request.spec_version,
        "request_id": request.request_id,
        "ok": bool(ok),
        "code": str(code),
        "msg": msg or "",
        "ts": _now_iso(),
        "data": data if data is not None else {},
    }
    if request.trace_id is not None:
        resp["trace_id"] = request.trace_id
    return resp
```

- [ ] **Step 3: Implement `lekiwi_node/control.py`**

```python
# lekiwi_node/control.py
from __future__ import annotations


def assemble_control(actuator_order: list[str], values: dict[str, float]) -> list[float]:
    """Build the MuJoCo control vector in actuator order; missing actuators -> 0.0."""
    return [float(values.get(name, 0.0)) for name in actuator_order]
```

- [ ] **Step 4: Run tests** → PASS (5). Commit:

```bash
git add lekiwi_node/_envelope.py lekiwi_node/control.py tests/test_envelope.py tests/test_control.py
git commit -m "feat: SPEC-V1 envelope + control-vector assembly"
```

---

### Task 6: LekiwiNode — verb dispatch (both namespaces)

**Files:**
- Create: `lekiwi_node/node.py`
- Test: `tests/test_node.py`

`LekiwiNode` is pure logic (mirrors `HunterBaseNode`). It holds the requested
intents; the runtime (Task 7) reads them and drives I/O. Base state is either a
velocity Twist (`set_velocity`) or a goal Pose2D (`go_to_pose`); arm state is a
6-vector goal. Motion verbs that must block return `{"code": "DEFERRED"}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_node.py
from lekiwi_node.node import LekiwiNode
from lekiwi_node.geometry import Pose2D, Twist

HOME = [0.0, 0.0, 0.0, 0.0, 0.0, 0.5]


def _node():
    n = LekiwiNode(robot_id="lekiwi", named_arm_poses={"home": HOME})
    n.install_all_verbs()
    return n


def test_set_velocity_is_immediate_and_sets_twist():
    n = _node()
    r = n.dispatch("vendor.dora_nav.base.set_velocity", {"vx": 0.2, "vy": 0.1, "omega": 0.3})
    assert r["ok"] is True and r["code"] == "0"
    assert n.base_velocity == Twist(0.2, 0.1, 0.3)
    assert n.base_target is None


def test_go_to_pose_is_deferred_and_sets_target():
    n = _node()
    pose = {"position": [0.5, 1.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]}
    r = n.dispatch("vendor.dora_nav.base.go_to_pose", {"pose": pose})
    assert r["code"] == "DEFERRED"
    assert isinstance(n.base_target, Pose2D)
    assert (n.base_target.x, n.base_target.y) == (0.5, 1.0)
    assert n.base_velocity is None


def test_stop_clears_base_intents():
    n = _node()
    n.dispatch("vendor.dora_nav.base.set_velocity", {"vx": 1.0, "vy": 0.0, "omega": 0.0})
    r = n.dispatch("vendor.dora_nav.base.stop", {})
    assert r["ok"] is True
    assert n.base_velocity is None and n.base_target is None


def test_move_to_joint_state_deferred_builds_six_vector():
    n = _node()
    r = n.dispatch("vendor.lerobot.arm.move_to_joint_state",
                   {"joints": [0.1, 0.2, 0.3, 0.4, 0.5], "gripper": 0.6})
    assert r["code"] == "DEFERRED"
    assert n.arm_target == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def test_move_to_joint_state_defaults_gripper_open():
    n = _node()
    n.dispatch("vendor.lerobot.arm.move_to_joint_state", {"joints": [0.0, 0.0, 0.0, 0.0, 0.0]})
    assert n.arm_target == [0.0, 0.0, 0.0, 0.0, 0.0, n.gripper_open]


def test_move_to_named_home():
    n = _node()
    r = n.dispatch("vendor.lerobot.arm.move_to_named", {"name": "home"})
    assert r["code"] == "DEFERRED"
    assert n.arm_target == HOME


def test_unknown_named_pose_invalid_params():
    n = _node()
    r = n.dispatch("vendor.lerobot.arm.move_to_named", {"name": "nope"})
    assert r["ok"] is False and r["code"] == "INVALID_PARAMS"


def test_estop_blocks_motion():
    n = _node()
    n.dispatch("robot.estop", {"reason": "test"})
    r = n.dispatch("vendor.dora_nav.base.set_velocity", {"vx": 1.0, "vy": 0.0, "omega": 0.0})
    assert r["ok"] is False and r["code"] == "VENDOR_ERROR"


def test_capabilities_lists_both_namespaces():
    n = _node()
    verbs = {c["verb"] for c in n.capabilities_advert()["commands"]}
    assert "vendor.dora_nav.base.go_to_pose" in verbs
    assert "vendor.lerobot.arm.move_to_joint_state" in verbs
```

Run → FAIL.

- [ ] **Step 2: Implement `lekiwi_node/node.py`**

```python
# lekiwi_node/node.py
from __future__ import annotations

from typing import Any, Callable

from lekiwi_node.geometry import Pose2D, Twist, yaw_from_quat

BASE_MOTION_VERBS = frozenset({
    "vendor.dora_nav.base.set_velocity", "vendor.dora_nav.base.go_to_pose",
})
ARM_MOTION_VERBS = frozenset({
    "vendor.lerobot.arm.move_to_joint_state", "vendor.lerobot.arm.move_to_named",
})


class LekiwiNode:
    """SPEC-V1 verbs for the LeKiwi robot (holonomic base + 5-DOF arm + gripper).
    Pure logic; the runtime drives I/O and resolves deferred motion."""

    def __init__(self, *, robot_id: str, named_arm_poses: dict[str, list[float]] | None = None,
                 gripper_open: float = 0.5, arm_dof: int = 6) -> None:
        self.robot_id = robot_id
        self.named_arm_poses = {k: list(v) for k, v in (named_arm_poses or {}).items()}
        self.gripper_open = gripper_open
        self.arm_dof = arm_dof
        self._verbs: dict[str, Callable[..., Any]] = {}
        self.base_target: Pose2D | None = None
        self.base_velocity: Twist | None = None
        self.arm_target: list[float] | None = None
        self.is_estopped = False
        self.estop_reason: str | None = None

    # ---- registry / dispatch ----
    def register_verb(self, name: str, handler: Callable[..., Any]) -> None:
        if name in self._verbs:
            raise ValueError(f"verb already registered: {name}")
        self._verbs[name] = handler

    def dispatch(self, verb: str, args: dict[str, Any]) -> dict[str, Any]:
        if verb not in self._verbs:
            return {"ok": False, "code": "INVALID_PARAMS", "msg": f"unknown verb: {verb}"}
        try:
            return self._verbs[verb](**args)
        except TypeError as e:
            return {"ok": False, "code": "INVALID_PARAMS", "msg": f"bad args for {verb}: {e}"}

    def install_all_verbs(self) -> None:
        self.register_verb("robot.heartbeat", lambda: {"ok": True, "code": "0"})
        self.register_verb("robot.estop", self._verb_estop)
        self.register_verb("robot.get_capabilities",
                           lambda: {"ok": True, "code": "0", "data": self.capabilities_advert()})
        self.register_verb("vendor.dora_nav.base.set_velocity", self._verb_set_velocity)
        self.register_verb("vendor.dora_nav.base.go_to_pose", self._verb_go_to_pose)
        self.register_verb("vendor.dora_nav.base.stop", self._verb_stop)
        self.register_verb("vendor.lerobot.arm.move_to_joint_state", self._verb_move_joints)
        self.register_verb("vendor.lerobot.arm.move_to_named", self._verb_move_named)

    # ---- common ----
    def _verb_estop(self, *, reason: str = "unspecified") -> dict[str, Any]:
        self.is_estopped = True
        self.estop_reason = reason
        self.base_target = None
        self.base_velocity = None
        self.arm_target = None
        return {"ok": True, "code": "0"}

    def _estop_guard(self) -> dict[str, Any] | None:
        if self.is_estopped:
            return {"ok": False, "code": "VENDOR_ERROR", "msg": f"estopped: {self.estop_reason}"}
        return None

    # ---- base ----
    def _verb_set_velocity(self, *, vx: float, vy: float, omega: float,
                           control_source: str = "") -> dict[str, Any]:
        guard = self._estop_guard()
        if guard:
            return guard
        self.base_velocity = Twist(float(vx), float(vy), float(omega))
        self.base_target = None
        return {"ok": True, "code": "0"}

    def _verb_go_to_pose(self, *, pose: dict[str, Any], control_source: str = "") -> dict[str, Any]:
        guard = self._estop_guard()
        if guard:
            return guard
        if not isinstance(pose, dict) or "position" not in pose or "orientation" not in pose:
            return {"ok": False, "code": "INVALID_PARAMS",
                    "msg": "pose needs position[xyz] + orientation[xyzw]"}
        x, y = float(pose["position"][0]), float(pose["position"][1])
        qx, qy, qz, qw = (float(v) for v in pose["orientation"])
        self.base_target = Pose2D(x, y, yaw_from_quat(qw, qx, qy, qz))
        self.base_velocity = None
        return {"code": "DEFERRED"}

    def _verb_stop(self) -> dict[str, Any]:
        self.base_velocity = None
        self.base_target = None
        return {"ok": True, "code": "0"}

    # ---- arm ----
    def _verb_move_joints(self, *, joints: list[float], gripper: float | None = None,
                          control_source: str = "") -> dict[str, Any]:
        guard = self._estop_guard()
        if guard:
            return guard
        if len(joints) != self.arm_dof - 1:
            return {"ok": False, "code": "INVALID_PARAMS",
                    "msg": f"expected {self.arm_dof - 1} arm joints"}
        g = self.gripper_open if gripper is None else float(gripper)
        self.arm_target = [float(j) for j in joints] + [g]
        return {"code": "DEFERRED"}

    def _verb_move_named(self, *, name: str, control_source: str = "") -> dict[str, Any]:
        guard = self._estop_guard()
        if guard:
            return guard
        if name not in self.named_arm_poses:
            return {"ok": False, "code": "INVALID_PARAMS", "msg": f"unknown named pose: {name}"}
        self.arm_target = list(self.named_arm_poses[name])
        return {"code": "DEFERRED"}

    # ---- advert ----
    def capabilities_advert(self) -> dict[str, Any]:
        return {
            "spec_version": "1.0.0",
            "vendor": "lekiwi",
            "model": "lekiwi",
            "robot_id": self.robot_id,
            "heartbeat_timeout_ms": 0,
            "commands": [{"verb": v, "safety_tier": "emergency_override"}
                         for v in sorted(self._verbs.keys())],
            "streams": ["state", "capabilities"],
        }
```

- [ ] **Step 3: Run tests** → PASS (9). Commit:

```bash
git add lekiwi_node/node.py tests/test_node.py
git commit -m "feat: LekiwiNode verb dispatch (base + arm namespaces)"
```

---

### Task 7: LekiwiRuntime — dora loop + per-namespace deferred resolution

**Files:**
- Create: `lekiwi_node/runtime.py`
- Test: `tests/test_runtime.py`

The runtime consumes `cmd_request` and `joint_positions`, drives the actuators,
and resolves the two independent deferred operations (base `go_to_pose`, arm move).
`joint_positions` layout is defined by the MJCF: indices 0,1 = base x,y; 3-6 =
base orientation quaternion (w,x,y,z); then the 6 arm joints in actuator order.
(The exact split is confirmed against the model in Task 8; the runtime takes the
base-pose and arm-joint slices via injected extractor callables so it is testable
without the model.)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runtime.py
import json
import numpy as np
from lekiwi_node.node import LekiwiNode
from lekiwi_node.runtime import LekiwiRuntime, ACTUATOR_ORDER
from lekiwi_node.geometry import Pose2D


class _FakeDora:
    def __init__(self):
        self.outputs = []
    def send_output(self, oid, data):
        self.outputs.append((oid, data))


def _rt():
    n = LekiwiNode(robot_id="lekiwi", named_arm_poses={"home": [0, 0, 0, 0, 0, 0.5]})
    n.install_all_verbs()
    # extractors: base pose + 6 arm joints from a plain list we control in the test
    rt = LekiwiRuntime(
        n,
        base_pose_from=lambda arr: Pose2D(float(arr[0]), float(arr[1]), float(arr[2])),
        arm_joints_from=lambda arr: [float(x) for x in arr[3:9]],
        deadline_s=60.0,
    )
    return n, rt


def _req(verb, params):
    return {"type": "INPUT", "id": "cmd_request",
            "value": [json.dumps({"verb": verb, "request_id": "r", "params": params})]}


def _joints(arr):
    return {"type": "INPUT", "id": "joint_positions", "value": np.array(arr, dtype=float)}


def test_set_velocity_emits_wheel_controls():
    n, rt = _rt()
    rt.on_event(_req("vendor.dora_nav.base.set_velocity",
                     {"vx": 0.0, "vy": 0.0, "omega": 1.0}), _FakeDora())
    dn = _FakeDora()
    # pose (0,0,0) + arm zeros
    rt.on_event(_joints([0, 0, 0, 0, 0, 0, 0, 0, 0]), dn)
    oid, data = [o for o in dn.outputs if o[0] == "control"][0]
    vec = list(np.asarray(data).astype(float))
    # pure spin -> the 3 wheels equal and non-zero
    assert abs(vec[0] - vec[1]) < 1e-6 and abs(vec[1] - vec[2]) < 1e-6 and vec[0] != 0.0


def test_go_to_pose_resolves_when_reached():
    n, rt = _rt()
    pose = {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]}
    dn = _FakeDora()
    assert rt.on_event(_req("vendor.dora_nav.base.go_to_pose", {"pose": pose}), dn) is True
    # no immediate cmd_response (deferred)
    assert not [o for o in dn.outputs if o[0] == "cmd_response"]
    # feed a pose already at target -> should resolve ok
    dn2 = _FakeDora()
    rt.on_event(_joints([0, 0, 0, 0, 0, 0, 0, 0, 0]), dn2)
    resp = [o for o in dn2.outputs if o[0] == "cmd_response"][0][1]
    payload = json.loads(resp[0]) if isinstance(resp[0], str) else resp[0]
    assert payload["ok"] is True and payload["code"] == "0"
    assert n.base_target is None


def test_arm_move_resolves_independently_of_base():
    n, rt = _rt()
    dn = _FakeDora()
    rt.on_event(_req("vendor.lerobot.arm.move_to_named", {"name": "home"}), dn)
    dn2 = _FakeDora()
    # arm already at home ([0,0,0,0,0,0.5]) -> resolves ok
    rt.on_event(_joints([0, 0, 0, 0, 0, 0, 0, 0, 0.5]), dn2)
    resp = [o for o in dn2.outputs if o[0] == "cmd_response"][0][1]
    payload = json.loads(resp[0]) if isinstance(resp[0], str) else resp[0]
    assert payload["ok"] is True
    assert n.arm_target is None


def test_actuator_order_has_nine_entries():
    assert len(ACTUATOR_ORDER) == 9
```

Run → FAIL.

- [ ] **Step 2: Implement `lekiwi_node/runtime.py`**

```python
# lekiwi_node/runtime.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pyarrow as pa

from lekiwi_node._envelope import (
    CmdRequest, InvalidEnvelope, build_cmd_response, parse_cmd_request,
)
from lekiwi_node.arm_driver import ArmDriver
from lekiwi_node.base_controller import HolonomicController
from lekiwi_node.control import assemble_control
from lekiwi_node.geometry import Pose2D, Twist
from lekiwi_node.kinematics import KiwiDrive
from lekiwi_node.node import ARM_MOTION_VERBS, BASE_MOTION_VERBS, LekiwiNode

# MJCF actuator order (from LeKiwi-sim/mjcf_lcmm_robot.xml); wheels then arm+gripper.
ACTUATOR_ORDER = ["drive_motor_1", "drive_motor_2", "drive_motor_3",
                  "Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
WHEEL_ACTUATORS = ACTUATOR_ORDER[:3]
ARM_ACTUATORS = ACTUATOR_ORDER[3:]


@dataclass
class PendingOp:
    request: CmdRequest
    started: float


def _decode_env(value: Any) -> dict[str, Any] | None:
    try:
        items = value.to_pylist() if hasattr(value, "to_pylist") else list(value)
    except Exception:  # noqa: BLE001
        return None
    if not items:
        return None
    first = items[0]
    try:
        return json.loads(first) if isinstance(first, str) else dict(first)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


class LekiwiRuntime:
    def __init__(self, node: LekiwiNode,
                 *, base_pose_from: Callable[[Any], Pose2D | None],
                 arm_joints_from: Callable[[Any], list[float]],
                 kiwi: KiwiDrive | None = None,
                 base_ctrl: HolonomicController | None = None,
                 arm: ArmDriver | None = None,
                 deadline_s: float = 60.0) -> None:
        self._node = node
        self._base_pose_from = base_pose_from
        self._arm_joints_from = arm_joints_from
        self._kiwi = kiwi or KiwiDrive()
        self._ctrl = base_ctrl or HolonomicController()
        self._arm = arm or ArmDriver(named_poses=node.named_arm_poses, dof=node.arm_dof)
        self._deadline_s = deadline_s
        self._base_pending: PendingOp | None = None
        self._arm_pending: PendingOp | None = None

    # ---- request handling ----
    def handle_request(self, env: dict[str, Any]) -> dict[str, Any] | None:
        try:
            req = parse_cmd_request(env)
        except InvalidEnvelope as e:
            return {"envelope_version": "1.0", "spec_version": "1.0.0",
                    "request_id": str(env.get("request_id", "")), "ok": False,
                    "code": "INVALID_PARAMS", "msg": str(e), "data": {}}
        if req.verb in BASE_MOTION_VERBS and self._base_pending is not None \
                and req.verb == "vendor.dora_nav.base.go_to_pose":
            return build_cmd_response(req, ok=False, code="CONTROLLER_BUSY",
                                      msg="base motion in progress")
        if req.verb in ARM_MOTION_VERBS and self._arm_pending is not None:
            return build_cmd_response(req, ok=False, code="CONTROLLER_BUSY",
                                      msg="arm motion in progress")
        result = self._node.dispatch(req.verb, req.params)
        if result.get("code") == "DEFERRED":
            if req.verb in ARM_MOTION_VERBS:
                self._arm.set_target(list(self._node.arm_target))
                self._arm_pending = PendingOp(req, time.monotonic())
            else:
                self._base_pending = PendingOp(req, time.monotonic())
            return None
        return build_cmd_response(req, ok=bool(result.get("ok", False)),
                                  code=str(result.get("code", "0")),
                                  data=result.get("data"), msg=str(result.get("msg", "")))

    # ---- dora event loop ----
    def on_event(self, event: dict[str, Any], dn: Any) -> bool:
        if event.get("type") == "STOP":
            return False
        if event.get("type") != "INPUT":
            return True
        if event.get("id") == "cmd_request":
            env = _decode_env(event.get("value"))
            if env is not None:
                tgt = env.get("target")
                if tgt is None or tgt == self._node.robot_id:
                    resp = self.handle_request(env)
                    if resp is not None:
                        dn.send_output("cmd_response", pa.array([json.dumps(resp)]))
        elif event.get("id") == "joint_positions":
            self._drive(event.get("value"), dn)
        return True

    def _drive(self, value: Any, dn: Any) -> None:
        pose = self._base_pose_from(value)
        arm_meas = self._arm_joints_from(value)
        # --- base command ---
        if self._node.base_target is not None and pose is not None:
            twist, base_reached = self._ctrl.step(pose, self._node.base_target)
        elif self._node.base_velocity is not None:
            twist, base_reached = self._node.base_velocity, False
        else:
            twist, base_reached = Twist(0.0, 0.0, 0.0), False
        w1, w2, w3 = self._kiwi.body_to_wheels(twist.vx, twist.vy, twist.omega)
        # --- arm command (hold target, else hold measured) ---
        arm_cmd = self._arm.target if self._arm.target is not None else arm_meas
        values = {WHEEL_ACTUATORS[0]: w1, WHEEL_ACTUATORS[1]: w2, WHEEL_ACTUATORS[2]: w3}
        for name, v in zip(ARM_ACTUATORS, arm_cmd):
            values[name] = v
        dn.send_output("control", pa.array(np.array(
            assemble_control(ACTUATOR_ORDER, values), dtype=np.float32)))
        # --- resolve deferred ops ---
        if self._base_pending is not None:
            if self._node.base_target is not None and pose is not None and base_reached:
                self._resolve(self._base_pending, dn, pose=pose)
                self._base_pending = None
                self._node.base_target = None
            elif time.monotonic() - self._base_pending.started > self._deadline_s:
                self._timeout(self._base_pending, dn); self._base_pending = None; self._node.base_target = None
        if self._arm_pending is not None:
            if self._arm.reached(arm_meas):
                self._resolve(self._arm_pending, dn)
                self._arm_pending = None
                self._arm.clear()
                self._node.arm_target = None
            elif time.monotonic() - self._arm_pending.started > self._deadline_s:
                self._timeout(self._arm_pending, dn); self._arm_pending = None; self._node.arm_target = None

    def _resolve(self, op: PendingOp, dn: Any, *, pose: Pose2D | None = None) -> None:
        data = None if pose is None else {"final_pose": {"x": pose.x, "y": pose.y, "yaw": pose.yaw}}
        dn.send_output("cmd_response", pa.array([json.dumps(
            build_cmd_response(op.request, ok=True, code="0", data=data))]))

    def _timeout(self, op: PendingOp, dn: Any) -> None:
        dn.send_output("cmd_response", pa.array([json.dumps(
            build_cmd_response(op.request, ok=False, code="BRIDGE_TIMEOUT",
                               msg="did not reach target in time"))]))
```

- [ ] **Step 3: Run tests** → `PYTHONPATH=. pytest tests/test_runtime.py -v` → PASS (4). Then run the whole suite `PYTHONPATH=. pytest tests -v` (expect all green).

- [ ] **Step 4: Commit**

```bash
git add lekiwi_node/runtime.py tests/test_runtime.py
git commit -m "feat: LekiwiRuntime dora loop + per-namespace deferred resolution"
```

---

### Task 8: Entry point, dataflow, vendored model (integration)

**Files:**
- Create: `lekiwi_node/__main__.py`
- Create: `dataflows/lekiwi-mujoco-bridge.yml`
- Create: `assets/mjcf_lcmm_robot.xml` (vendored)
- Create: `README.md`

This task wires the tested units to dora + the `dora-mujoco` runner + the
`octos_spec_bridge`. It has integration checks rather than unit tests, and two
model-specific facts to VERIFY before it runs (do not assume):

**(V1) Actuator order & wheel↔angle mapping.** Open the vendored MJCF; read the
`<actuator>` block order. Confirm `ACTUATOR_ORDER` in `runtime.py` matches it
exactly; if the wheels are ordered differently than `[150°,−90°,30°]`, reorder the
`KiwiDrive` angle rows (or the wheel actuator names) so wheel *i* receives the
speed for its physical mount angle. Verify with a one-shot: command `set_velocity
vx=0.2,vy=0,omega=0` and confirm the base drives **straight forward** in the viewer.

**(V2) `joint_positions` layout.** Confirm the runner publishes base free-joint
pose at indices `[0,1]=x,y` and `[3,4,5,6]=quat(w,x,y,z)`, with the 6 arm joints
following. Adjust the `base_pose_from` / `arm_joints_from` extractors in
`__main__.py` to match the actual layout.

- [ ] **Step 1: Vendor the model**

```bash
mkdir -p assets
cp ../LeKiwi-sim/mjcf_lcmm_robot.xml assets/mjcf_lcmm_robot.xml
```

- [ ] **Step 2: Write `lekiwi_node/__main__.py`**

```python
# lekiwi_node/__main__.py
from __future__ import annotations

import numpy as np
from dora import Node

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
    # 6 arm/gripper joints follow the 7 base free-joint DOF (VERIFY: V2)
    return [float(v) for v in arr[7:13]]


def main() -> None:
    node = LekiwiNode(robot_id="lekiwi", named_arm_poses={"home": HOME})
    node.install_all_verbs()
    rt = LekiwiRuntime(node, base_pose_from=_base_pose, arm_joints_from=_arm_joints)
    dora = Node()
    for event in dora:
        if not rt.on_event(event, dora):
            break


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write `dataflows/lekiwi-mujoco-bridge.yml`**

Model the topology on the Hunter demo's `hunter-inspection.yml`: (a) a `dora-mujoco`
sim node loading `assets/mjcf_lcmm_robot.xml`, publishing `joint_positions` and
consuming `control`; (b) `lekiwi-node` (this package's `__main__`) consuming
`cmd_request` + `joint_positions`, publishing `control` + `cmd_response`; (c) an
`octos_spec_bridge` exposing the SPEC HTTP port and bridging `cmd_request`/`cmd_response`.

```yaml
# dataflows/lekiwi-mujoco-bridge.yml
nodes:
  - id: mujoco_sim
    path: dora-mujoco            # reused runner; provide the model via env/args
    env:
      MJCF_PATH: assets/mjcf_lcmm_robot.xml
    inputs:
      control: lekiwi-node/control
    outputs:
      - joint_positions

  - id: lekiwi-node
    path: python
    args: -m lekiwi_node
    inputs:
      cmd_request: bridge/cmd_request
      joint_positions: mujoco_sim/joint_positions
    outputs:
      - control
      - cmd_response

  - id: bridge
    path: octos_spec_bridge       # exposes SPEC HTTP; port via env
    env:
      SPEC_HTTP_PORT: "8770"
    inputs:
      cmd_response: lekiwi-node/cmd_response
    outputs:
      - cmd_request
```

> The exact `path`/env keys for `dora-mujoco` and `octos_spec_bridge` must match
> those repos' node contracts (copy from `octos_inspection/dataflows/hunter-inspection.yml`,
> which wires the same two components). Adjust names to match.

- [ ] **Step 4: Import smoke test**

Run: `PYTHONPATH=. python -c "import lekiwi_node.__main__"` — Expected: imports
without error (the `dora` import may require the dora runtime; if unavailable in
the test env, guard the import or run inside the dataflow only). Also confirm the
whole unit suite is green: `PYTHONPATH=. pytest tests -v`.

- [ ] **Step 5: Live bring-up + verify V1/V2 (on a machine with the sim)**

`dora up` → `dora start dataflows/lekiwi-mujoco-bridge.yml`. Then, against the SPEC
port: `set_velocity vx=0.2` (V1: drives straight forward), `set_velocity vy=0.2`
(strafes sideways — the holonomic capability), `go_to_pose` to a lateral+reverse
target (reaches — the Ackermann failure case, trivial here), `move_to_named home`
(arm homes). Fix `ACTUATOR_ORDER` / extractor slices per V1/V2 if motion is wrong.

- [ ] **Step 6: Commit**

```bash
git add lekiwi_node/__main__.py dataflows/lekiwi-mujoco-bridge.yml assets/mjcf_lcmm_robot.xml README.md
git commit -m "feat: dora entry point + mujoco-bridge dataflow + vendored LeKiwi model"
```

---

## Plan Self-Review

**Spec coverage:** §2 architecture → Tasks 1–8 (layered units + node + runtime + dataflow). §3 KiwiDrive → Task 2 (exact matrix, constants, round-trip). §4 verbs (both namespaces, deferred flags) → Task 6 (dispatch) + Task 7 (deferred resolution). §5 holonomic controller → Task 3. §6 arm driver → Task 4. §7 sim reuse + actuator table → Task 8 (with V1/V2 verify steps). §8 file map → File Structure + one task per unit. §9 error handling (INVALID_PARAMS, VENDOR_ERROR on estop, CONTROLLER_BUSY per-namespace, BRIDGE_TIMEOUT, clamp) → Tasks 6–7. §10 testing → per-task TDD. **Full coverage.**

**No placeholders:** every code step is complete and runnable. The only deliberately deferred items are the two model-specific facts (actuator order, joint layout) — explicitly called out as **V1/V2 verify steps** in Task 8 with a concrete verification procedure, because they depend on inspecting the vendored MJCF and cannot be asserted from outside it. All types/functions used are defined in an earlier task.

**Type consistency:** `Pose2D(x,y,yaw)`, `Twist(vx,vy,omega)` (Task 1) used identically in Tasks 2,3,6,7. `KiwiDrive.body_to_wheels(vx,vy,omega)->tuple` / `wheels_to_body(w1,w2,w3)->Twist` (Task 2) consistent in Task 7. `HolonomicController.step(current,target)->(Twist,bool)` (Task 3) consistent in Task 7. `ArmDriver.set_target/set_named/reached/clear` + `dof` (Task 4) consistent in Task 7. `LekiwiNode` fields `base_target/base_velocity/arm_target/arm_dof/gripper_open/named_arm_poses` + `install_all_verbs()` (Task 6) consistent in Task 7. `assemble_control(order, values)` (Task 5) consistent in Task 7. `ACTUATOR_ORDER` (9 entries) shared by Task 7 and Task 8.

**Ordering:** geometry → kinematics → controller → arm → envelope/control → node → runtime → integration. Each task is independently testable and green before the next.
