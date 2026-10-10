import numpy as np
import pytest

pytest.importorskip("mujoco")

from vla.simulation import cartesian_controller
from vla.simulation.cartesian_controller import (
    BIN_FINGERTIP_CLEARANCE,
    MAX_GRASP_YAW_ADJUSTMENT,
    CartesianPandaController,
    WorkspaceLimits,
)
from vla.simulation.episode_logging import PickPlaceSuccessChecker
from vla.simulation.mujoco_tabletop import PANDA_XML, TabletopEnvironment


@pytest.fixture()
def environment() -> TabletopEnvironment:
    if not PANDA_XML.is_file():
        pytest.skip("MuJoCo Menagerie submodule is not initialized")
    return TabletopEnvironment()


def test_workspace_limits_are_derived_from_the_loaded_scene(environment: TabletopEnvironment) -> None:
    controller = CartesianPandaController(environment)
    limits = controller.workspace
    hand = environment.get_robot_state()["ee_position"]

    assert isinstance(limits, WorkspaceLimits)
    assert limits.contains((hand["x"], hand["y"], hand["z"]))
    with pytest.raises(ValueError, match="outside workspace"):
        limits.require((limits.x[1] + 0.01, 0.0, 0.5))


@pytest.mark.parametrize(
    ("seed", "object_id", "target_id"),
    [
        (0, "red_cube_1", "yellow_bin_1"),
        (1, "green_cube_1", "purple_bin_1"),
        (5, "blue_cube_1", "yellow_bin_1"),
        (9, "blue_cube_1", "purple_bin_1"),
    ],
)
def test_scripted_controller_completes_randomized_pick_and_place(
    environment: TabletopEnvironment, seed: int, object_id: str, target_id: str
) -> None:
    environment.reset(seed=seed)
    result = CartesianPandaController(environment).scripted_pick_and_place(
        object_id, target_id, scene_id=f"tabletop_{seed:05d}"
    )

    assert result["success"]["success"] is True
    assert [event["phase"] for event in result["episode"]["events"]] == [
        "open_gripper",
        "approach_object",
        "descend_to_object",
        "close_gripper",
        "lift_object",
        "move_above_target",
        "descend_to_target",
        "open_gripper",
        "retreat",
    ]
    assert result["episode"]["events"][3]["object"]["state"]["is_grasped"] is True
    assert result["episode"]["events"][3]["action"][4] == 1.0
    assert result["episode"]["events"][7]["action"][4] == 0.0
    assert result["episode"]["success"] == result["success"]


def test_success_check_requires_a_released_cube(environment: TabletopEnvironment) -> None:
    environment.reset(seed=0)
    environment.set_grasped_object("red_cube_1")

    result = PickPlaceSuccessChecker().check(environment, "red_cube_1", "yellow_bin_1")

    assert result["success"] is False
    assert result["released"] is False


def test_physical_grasp_lifts_a_cube_without_a_kinematic_attachment(environment: TabletopEnvironment) -> None:
    environment.reset(seed=0)
    controller = CartesianPandaController(environment)
    cube = environment.model.body("red_cube_1")
    cube_position = environment.data.xpos[cube.id].copy()

    controller.open_gripper()
    controller.move_to(cube_position + (0.0, 0.0, 0.18))
    controller.move_to(cube_position + (0.0, 0.0, 0.12))
    controller.close_gripper("red_cube_1")
    closed_position = environment.data.xpos[cube.id].copy()

    assert controller._cube_is_between_fingers("red_cube_1") is True
    hand = environment.model.body("hand")
    controller.move_to(environment.data.xpos[hand.id] + (0.0, 0.0, 0.16))

    assert environment.data.xpos[cube.id][2] > closed_position[2] + 0.10


def test_grasp_yaw_aligns_the_gripper_with_a_randomized_cube(environment: TabletopEnvironment) -> None:
    environment.reset(seed=0)
    controller = CartesianPandaController(environment)
    cube = environment.model.body("red_cube_1")
    cube_rotation = environment.data.xmat[cube.id].reshape(3, 3)
    cube_yaw = float(np.arctan2(cube_rotation[1, 0], cube_rotation[0, 0]))
    grasp_yaw = controller._grasp_yaw("red_cube_1")

    hand_yaw = environment.get_robot_state()["ee_yaw"]
    def alignment_error(yaw: float) -> float:
        return min(
            abs(controller._angle_difference(yaw, cube_yaw + np.pi / 2)),
            abs(controller._angle_difference(yaw, cube_yaw - np.pi / 2)),
        )

    assert abs(controller._angle_difference(grasp_yaw, hand_yaw)) <= MAX_GRASP_YAW_ADJUSTMENT + 1e-9
    assert alignment_error(grasp_yaw) <= 2e-3


def test_scripted_controller_keeps_fingertips_above_the_bin_walls(environment: TabletopEnvironment) -> None:
    environment.reset(seed=0)
    controller = CartesianPandaController(environment)
    target_id = "yellow_bin_1"
    target_top = controller._bin_top(target_id)

    result = controller.scripted_pick_and_place("red_cube_1", target_id)
    descend_event = next(event for event in result["episode"]["events"] if event["phase"] == "descend_to_target")

    hand_height = descend_event["robot"]["ee_position"]["z"]
    assert hand_height >= target_top + controller._fingertip_clearance_below_hand() + BIN_FINGERTIP_CLEARANCE - 0.04


def test_scripted_controller_preserves_grasp_yaw_after_closing(environment: TabletopEnvironment) -> None:
    environment.reset(seed=0)
    controller = CartesianPandaController(environment)
    grasp_yaw = controller._grasp_yaw("red_cube_1")

    result = controller.scripted_pick_and_place("red_cube_1", "yellow_bin_1")

    for event in result["episode"]["events"][3:]:
        event_yaw = event["robot"]["ee_yaw"]
        assert abs(controller._angle_difference(event_yaw, grasp_yaw)) < 0.04


def test_macos_viewer_restarts_with_the_matching_mjpython(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    python = tmp_path / "python"
    mjpython = tmp_path / "mjpython"
    python.touch()
    mjpython.touch()
    command: dict[str, object] = {}

    monkeypatch.setattr(cartesian_controller.sys, "platform", "darwin")
    monkeypatch.setattr(cartesian_controller.sys, "executable", str(python))
    monkeypatch.delenv("MJPYTHON_BIN", raising=False)
    monkeypatch.setattr(
        cartesian_controller.os,
        "execv",
        lambda path, arguments: command.update(path=path, arguments=arguments),
    )

    cartesian_controller._restart_with_mjpython_if_required(["--seed", "0", "--viewer"])

    assert command == {
        "path": str(mjpython),
        "arguments": [
            str(mjpython),
            "-m",
            "vla.simulation.cartesian_controller",
            "--seed",
            "0",
            "--viewer",
        ],
    }


