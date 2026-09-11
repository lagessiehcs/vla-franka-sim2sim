import pytest

mujoco = pytest.importorskip("mujoco")

from vla.simulation.mujoco_franka_inspect import DEFAULT_SCENE, END_EFFECTOR_BODY, load_model


def test_official_franka_model_loads_when_submodule_is_initialized() -> None:
    if not DEFAULT_SCENE.is_file():
        pytest.skip("MuJoCo Menagerie submodule is not initialized")

    model = load_model()

    assert model.njnt == 9  # seven arm joints and two finger joints
    assert model.nu == 8  # seven arm actuators and one gripper actuator
    assert model.body(END_EFFECTOR_BODY).name == END_EFFECTOR_BODY
    assert model.actuator(7).name == "actuator8"
