"""Shared coordinate, action, and gripper conventions for the VLA pipeline."""

from __future__ import annotations


WORLD_FRAME = "world"
END_EFFECTOR_FRAME = "hand"
POSITION_UNIT = "metres"
ANGLE_UNIT = "radians"

ACTION_NAMES = ("dx", "dy", "dz", "dyaw", "gripper")
ACTION_DIM = len(ACTION_NAMES)

GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 1.0
PANDA_GRIPPER_ACTUATOR = "actuator8"
PANDA_GRIPPER_OPEN_CONTROL = 255.0
PANDA_GRIPPER_CLOSED_CONTROL = 0.0


def gripper_command_to_panda_control(command: float) -> float:
    """Map the normalized project command to the Menagerie Panda actuator.

    The project interface uses ``0`` for open and ``1`` for closed. The Panda
    model's eighth actuator uses the reverse physical control range: ``255``
    opens both fingers and ``0`` closes them.
    """
    if command == GRIPPER_OPEN:
        return PANDA_GRIPPER_OPEN_CONTROL
    if command == GRIPPER_CLOSED:
        return PANDA_GRIPPER_CLOSED_CONTROL
    raise ValueError(f"gripper command must be {GRIPPER_OPEN} (open) or {GRIPPER_CLOSED} (closed), got {command}")
