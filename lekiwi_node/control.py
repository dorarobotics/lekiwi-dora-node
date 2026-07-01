from __future__ import annotations


def assemble_control(actuator_order: list[str], values: dict[str, float]) -> list[float]:
    """Build the MuJoCo control vector in actuator order; missing actuators -> 0.0."""
    return [float(values.get(name, 0.0)) for name in actuator_order]
