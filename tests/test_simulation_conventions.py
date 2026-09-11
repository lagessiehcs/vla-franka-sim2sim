import pytest

from vla.simulation.conventions import (
    ACTION_DIM,
    ACTION_NAMES,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    PANDA_GRIPPER_CLOSED_CONTROL,
    PANDA_GRIPPER_OPEN_CONTROL,
    gripper_command_to_panda_control,
)


def test_action_format_and_gripper_adapter() -> None:
    assert ACTION_NAMES == ("dx", "dy", "dz", "dyaw", "gripper")
    assert ACTION_DIM == 5
    assert gripper_command_to_panda_control(GRIPPER_OPEN) == PANDA_GRIPPER_OPEN_CONTROL
    assert gripper_command_to_panda_control(GRIPPER_CLOSED) == PANDA_GRIPPER_CLOSED_CONTROL


def test_gripper_adapter_rejects_undefined_commands() -> None:
    with pytest.raises(ValueError, match="gripper command"):
        gripper_command_to_panda_control(0.5)
