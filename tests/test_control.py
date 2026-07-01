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
