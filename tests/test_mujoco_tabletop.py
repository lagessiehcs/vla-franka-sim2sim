import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from vla.simulation.mujoco_tabletop import (
    CUBE_FRICTION,
    CUBE_HALF_SIZE,
    MAX_CUBE_DISTANCE_FROM_ROBOT_BASE,
    MAX_TARGET_DISTANCE_FROM_ROBOT_BASE,
    MIN_CUBE_FORWARD_FROM_ROBOT_BASE,
    PANDA_XML,
    RGB_CAMERA,
    ROBOT_PLATFORM_HEIGHT,
    ROBOT_PLATFORM_POSITION,
    ROBOT_PLATFORM_ROBOT_X_OFFSET,
    TABLE_EDGE_MARGIN,
    TabletopEnvironment,
)


@pytest.fixture()
def environment() -> TabletopEnvironment:
    if not PANDA_XML.is_file():
        pytest.skip("MuJoCo Menagerie submodule is not initialized")
    return TabletopEnvironment()


def test_reset_is_seeded_and_keeps_cubes_separate(environment: TabletopEnvironment) -> None:
    first = environment.reset(seed=42)
    second = environment.reset(seed=42)
    different = environment.reset(seed=43)

    def cube_positions(scene: dict) -> np.ndarray:
        cubes = [item for item in scene["objects"] if item["category"] == "rigid_object"]
        return np.array(
            [[item["pose"]["position"][axis] for axis in ("x", "y", "z")] for item in cubes]
        )

    first_positions = cube_positions(first)
    assert np.array_equal(first_positions, cube_positions(second))
    assert not np.array_equal(first_positions, cube_positions(different))
    assert all(
        np.linalg.norm(first_positions[left, :2] - first_positions[right, :2]) >= 0.09
        for left in range(3)
        for right in range(left + 1, 3)
    )

    table = environment.model.geom("table_top")
    table_position = environment.data.geom_xpos[table.id]
    table_size = environment.model.geom_size[table.id]
    robot_base = environment.data.xpos[environment.model.body("link0").id]
    assert np.all(first_positions[:, 0] >= table_position[0] - table_size[0] + TABLE_EDGE_MARGIN)
    assert np.all(first_positions[:, 0] <= table_position[0] + table_size[0] - TABLE_EDGE_MARGIN)
    assert np.all(first_positions[:, 1] >= table_position[1] - table_size[1] + TABLE_EDGE_MARGIN)
    assert np.all(first_positions[:, 1] <= table_position[1] + table_size[1] - TABLE_EDGE_MARGIN)
    assert np.all(np.linalg.norm(first_positions - robot_base, axis=1) <= MAX_CUBE_DISTANCE_FROM_ROBOT_BASE)
    assert np.all(first_positions[:, 0] >= robot_base[0] + MIN_CUBE_FORWARD_FROM_ROBOT_BASE)

    bin_positions = np.array(
        [
            [item["pose"]["position"][axis] for axis in ("x", "y", "z")]
            for item in first["objects"]
            if item["category"] == "target_zone"
        ]
    )
    assert np.all(np.linalg.norm(bin_positions - robot_base, axis=1) <= MAX_TARGET_DISTANCE_FROM_ROBOT_BASE)


def test_scene_and_robot_queries_have_shared_frames(environment: TabletopEnvironment) -> None:
    scene = environment.reset(seed=2)
    robot = environment.get_robot_state()

    assert scene["frame_id"] == "world"
    assert [item["id"] for item in scene["objects"]] == [
        "table",
        "red_cube_1",
        "green_cube_1",
        "blue_cube_1",
        "yellow_bin_1",
        "purple_bin_1",
    ]
    assert robot["frame_id"] == "world"
    assert robot["end_effector_frame"] == "hand"
    assert robot["gripper"] == 0.0


def test_rgb_camera_returns_an_image(environment: TabletopEnvironment) -> None:
    environment.reset(seed=1)
    image = environment.get_rgb(width=160, height=120)

    assert RGB_CAMERA == environment.model.camera(RGB_CAMERA).name
    assert image.shape == (120, 160, 3)
    assert image.dtype == np.uint8
    assert image.mean() > 20


def test_cubes_use_the_configured_friction(environment: TabletopEnvironment) -> None:
    for cube_id in ("red_cube_1", "green_cube_1", "blue_cube_1"):
        assert np.allclose(environment.model.geom(f"{cube_id}_geom").friction, CUBE_FRICTION)


def test_cubes_remain_on_the_table_when_simulated(environment: TabletopEnvironment) -> None:
    environment.reset(seed=42)
    for _ in range(500):
        mujoco.mj_step(environment.model, environment.data)

    table = environment.model.geom("table_top")
    minimum_cube_centre_height = environment.data.geom_xpos[table.id][2] + table.size[2] + CUBE_HALF_SIZE - 0.005
    for cube_id in ("red_cube_1", "green_cube_1", "blue_cube_1"):
        assert environment.data.xpos[environment.model.body(cube_id).id][2] >= minimum_cube_centre_height


def test_base_rotation_does_not_contact_the_table(environment: TabletopEnvironment) -> None:
    joint = environment.model.joint("joint1")
    for angle in np.linspace(-2.8, 2.8, 9):
        environment.reset(seed=0)
        environment.data.qpos[joint.qposadr[0]] = angle
        mujoco.mj_forward(environment.model, environment.data)
        contact_names = [
            (environment.model.geom(contact.geom1).name, environment.model.geom(contact.geom2).name)
            for contact in environment.data.contact
        ]
        assert all("table" not in first and "table" not in second for first, second in contact_names)


def test_robot_is_mounted_on_the_platform(environment: TabletopEnvironment) -> None:
    robot_base = environment.data.xpos[environment.model.body("link0").id]

    assert np.allclose(
        robot_base,
        (
            ROBOT_PLATFORM_POSITION[0] + ROBOT_PLATFORM_ROBOT_X_OFFSET,
            ROBOT_PLATFORM_POSITION[1],
            ROBOT_PLATFORM_HEIGHT,
        ),
    )


def test_platform_and_table_footprint_is_centered_on_the_floor(environment: TabletopEnvironment) -> None:
    platform = environment.model.geom("robot_platform_top")
    table = environment.model.geom("table_top")
    left_edge = min(
        environment.data.geom_xpos[platform.id][0] - platform.size[0],
        environment.data.geom_xpos[table.id][0] - table.size[0],
    )
    right_edge = max(
        environment.data.geom_xpos[platform.id][0] + platform.size[0],
        environment.data.geom_xpos[table.id][0] + table.size[0],
    )

    assert np.isclose((left_edge + right_edge) / 2, 0.0)
