"""Seeded MuJoCo tabletop scenes for collecting VLA observations.

Run ``python -m vla.simulation.mujoco_tabletop --output results/tabletop.png``
after installing the optional ``simulation`` dependencies and initializing the
MuJoCo Menagerie submodule.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import mujoco
import numpy as np
from PIL import Image
from mujoco import viewer

from .conventions import END_EFFECTOR_FRAME, WORLD_FRAME


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PANDA_DIRECTORY = PROJECT_ROOT / "assets" / "mujoco_menagerie" / "franka_emika_panda"
PANDA_XML = PANDA_DIRECTORY / "panda.xml"
RGB_CAMERA = "rgb_camera"
CUBE_HALF_SIZE = 0.025
CUBE_FRICTION = (2.0, 0.02, 0.001)
TABLE_EDGE_MARGIN = 0.075
BIN_CLEARANCE = 0.02
# A conservative workspace filter relative to the Panda base.
# Inverse kinematics and collision checking will provide the final reachability
# decision when a controller is added.
MAX_CUBE_DISTANCE_FROM_ROBOT_BASE = 0.78
MAX_TARGET_DISTANCE_FROM_ROBOT_BASE = 0.75
MIN_CUBE_FORWARD_FROM_ROBOT_BASE = 0.25
# The platform-and-table footprint spans x=-0.635 to x=0.635, centred on the floor origin.
ROBOT_PLATFORM_POSITION = (-0.415, 0.0, 0.0)
ROBOT_PLATFORM_HEIGHT = 0.15
ROBOT_PLATFORM_ROBOT_X_OFFSET = 0.04
TABLE_POSITION = (0.185, 0.0, 0.0)
RGB_CAMERA_POSITION = (1.085, -1.60, 1.50)

_CUBES = (
    ("red_cube_1", "red cube", (0.85, 0.08, 0.08, 1.0)),
    ("green_cube_1", "green cube", (0.10, 0.65, 0.15, 1.0)),
    ("blue_cube_1", "blue cube", (0.10, 0.30, 0.90, 1.0)),
)
_BINS = (
    ("yellow_bin_1", "yellow bin", (0.95, 0.75, 0.10, 1.0), (0.185, -0.18)),
    ("purple_bin_1", "purple bin", (0.55, 0.15, 0.75, 1.0), (0.185, 0.18)),
)


def _rgba(values: tuple[float, float, float, float]) -> str:
    return " ".join(str(value) for value in values)


def _tabletop_xml() -> str:
    """Return Panda XML extended with the table, objects, bins, and camera."""
    if not PANDA_XML.is_file():
        raise FileNotFoundError(
            f"MuJoCo Panda model not found: {PANDA_XML}. "
            "Run `git submodule update --init --recursive`."
        )

    cube_xml = "\n".join(
        f'''    <body name="{object_id}" pos="0 0 0">
      <freejoint name="{object_id}_free"/>
      <geom name="{object_id}_geom" type="box" size="{CUBE_HALF_SIZE} {CUBE_HALF_SIZE} {CUBE_HALF_SIZE}"
            mass="0.08" friction="{_rgba(CUBE_FRICTION)}" rgba="{_rgba(rgba)}"/>
    </body>'''
        for object_id, _, rgba in _CUBES
    )
    bin_xml = "\n".join(
        f'''    <body name="{bin_id}" pos="{position[0]} {position[1]} 0.32">
      <geom name="{bin_id}_base" type="box" pos="0 0 0.006" size="0.09 0.09 0.006"
            rgba="{_rgba(rgba)}"/>
      <geom name="{bin_id}_front" type="box" pos="-0.078 0 0.035" size="0.012 0.09 0.035"
            rgba="{_rgba(rgba)}"/>
      <geom name="{bin_id}_back" type="box" pos="0.078 0 0.035" size="0.012 0.09 0.035"
            rgba="{_rgba(rgba)}"/>
      <geom name="{bin_id}_left" type="box" pos="0 -0.078 0.035" size="0.09 0.012 0.035"
            rgba="{_rgba(rgba)}"/>
      <geom name="{bin_id}_right" type="box" pos="0 0.078 0.035" size="0.09 0.012 0.035"
            rgba="{_rgba(rgba)}"/>
    </body>'''
        for bin_id, _, rgba, position in _BINS
    )
    additions = f'''
    <light name="sun_light" pos="0 -0.8 2" dir="0.2 0.4 -1" directional="true"
           diffuse="1 0.95 0.85" ambient="0.25 0.24 0.22" specular="0.2 0.2 0.2" castshadow="true"/>
    <light name="sky_fill_light" pos="1.2 0.8 1.5" dir="-0.4 -0.3 -1" directional="true"
           diffuse="0.5 0.5 0.6" ambient="0.15 0.15 0.18" specular="0.1 0.1 0.1"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.15 0.15 0.15 1"/>
    <body name="robot_platform" pos="{ROBOT_PLATFORM_POSITION[0]} {ROBOT_PLATFORM_POSITION[1]} 0">
      <geom name="robot_platform_top" type="box" pos="0 0 {ROBOT_PLATFORM_HEIGHT / 2}"
            size="0.22 0.20 {ROBOT_PLATFORM_HEIGHT / 2}" rgba="0.20 0.20 0.20 1"/>
    </body>
    <body name="table" pos="{TABLE_POSITION[0]} {TABLE_POSITION[1]} {TABLE_POSITION[2]}">
      <geom name="table_top" type="box" pos="0 0 0.30" size="0.45 0.50 0.02"
            rgba="0.42 0.26 0.12 1"/>
      <geom name="table_leg_front_left" type="box" pos="-0.38 -0.43 0.15" size="0.03 0.03 0.15"
            rgba="0.20 0.20 0.20 1"/>
      <geom name="table_leg_front_right" type="box" pos="0.38 -0.43 0.15" size="0.03 0.03 0.15"
            rgba="0.20 0.20 0.20 1"/>
      <geom name="table_leg_back_left" type="box" pos="-0.38 0.43 0.15" size="0.03 0.03 0.15"
            rgba="0.20 0.20 0.20 1"/>
      <geom name="table_leg_back_right" type="box" pos="0.38 0.43 0.15" size="0.03 0.03 0.15"
            rgba="0.20 0.20 0.20 1"/>
    </body>
{cube_xml}
{bin_xml}
    <camera name="{RGB_CAMERA}" pos="{RGB_CAMERA_POSITION[0]} {RGB_CAMERA_POSITION[1]} {RGB_CAMERA_POSITION[2]}"
            xyaxes="0.836 0.548 0 -0.254 0.387 0.886"/>
'''
    panda_xml = PANDA_XML.read_text(encoding="utf-8")
    panda_xml = panda_xml.replace(
        '<body name="link0" childclass="panda">',
        f'<body name="link0" pos="{ROBOT_PLATFORM_POSITION[0] + ROBOT_PLATFORM_ROBOT_X_OFFSET} '
        f'{ROBOT_PLATFORM_POSITION[1]} {ROBOT_PLATFORM_HEIGHT}" '
        'childclass="panda">',
        1,
    )
    return panda_xml.replace("  </worldbody>", additions + "  </worldbody>", 1)


def _panda_assets() -> dict[str, bytes]:
    """Load Panda meshes for ``MjModel.from_xml_string`` without copying them."""
    asset_directory = PANDA_DIRECTORY / "assets"
    if not asset_directory.is_dir():
        raise FileNotFoundError(
            f"MuJoCo Panda assets not found: {asset_directory}. "
            "Run `git submodule update --init --recursive`."
        )
    return {
        str(asset.relative_to(PANDA_DIRECTORY)): asset.read_bytes()
        for asset in asset_directory.rglob("*")
        if asset.is_file()
    }


class TabletopEnvironment:
    """A resettable Panda tabletop environment with RGB and state queries."""

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_string(_tabletop_xml(), assets=_panda_assets())
        self.data = mujoco.MjData(self.model)
        self._cube_positions: dict[str, np.ndarray] = {}
        self.reset(seed=0)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """Reset the Panda and place the cubes deterministically for ``seed``."""
        rng = np.random.default_rng(seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        positions: list[np.ndarray] = []
        self._cube_positions = {}
        for object_id, _, _ in _CUBES:
            position = self._sample_cube_position(rng, positions)
            positions.append(position)
            self._cube_positions[object_id] = position
            joint = self.model.joint(f"{object_id}_free")
            qpos_address = joint.qposadr[0]
            self.data.qpos[qpos_address : qpos_address + 3] = position
            self.data.qpos[qpos_address + 3 : qpos_address + 7] = (1.0, 0.0, 0.0, 0.0)

        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self.get_scene_state()

    def _sample_cube_position(self, rng: np.random.Generator, occupied: list[np.ndarray]) -> np.ndarray:
        """Sample a cube centre from the current table, bin, and robot layout."""
        table = self.model.geom("table_top")
        table_centre = self.data.geom_xpos[table.id]
        table_half_size = self.model.geom_size[table.id]
        x_min = table_centre[0] - table_half_size[0] + TABLE_EDGE_MARGIN
        x_max = table_centre[0] + table_half_size[0] - TABLE_EDGE_MARGIN
        y_min = table_centre[1] - table_half_size[1] + TABLE_EDGE_MARGIN
        y_max = table_centre[1] + table_half_size[1] - TABLE_EDGE_MARGIN
        table_top_z = table_centre[2] + table_half_size[2]
        robot_base = self.data.xpos[self.model.body("link0").id]
        bins = [
            (self.data.xpos[self.model.body(bin_id).id], self.model.geom(f"{bin_id}_base").size[:2])
            for bin_id, _, _, _ in _BINS
        ]

        for _ in range(100):
            candidate = np.array(
                [rng.uniform(x_min, x_max), rng.uniform(y_min, y_max), table_top_z + CUBE_HALF_SIZE]
            )
            outside_bins = all(
                abs(candidate[0] - bin_position[0]) >= bin_half_size[0] + CUBE_HALF_SIZE + BIN_CLEARANCE
                or abs(candidate[1] - bin_position[1]) >= bin_half_size[1] + CUBE_HALF_SIZE + BIN_CLEARANCE
                for bin_position, bin_half_size in bins
            )
            cubes_separated = all(np.linalg.norm(candidate[:2] - position[:2]) >= 0.09 for position in occupied)
            within_robot_workspace = np.linalg.norm(candidate - robot_base) <= MAX_CUBE_DISTANCE_FROM_ROBOT_BASE
            in_front_of_robot = candidate[0] >= robot_base[0] + MIN_CUBE_FORWARD_FROM_ROBOT_BASE
            if outside_bins and cubes_separated and within_robot_workspace and in_front_of_robot:
                return candidate
        raise RuntimeError("could not sample a cube position that fits the table and conservative robot workspace")

    def get_rgb(self, width: int = 640, height: int = 480) -> np.ndarray:
        """Return the RGB image from the fixed named camera as ``uint8[H, W, 3]``."""
        with mujoco.Renderer(self.model, height=height, width=width) as renderer:
            renderer.update_scene(self.data, camera=RGB_CAMERA)
            return renderer.render().copy()

    def set_rgb_camera(self, position: Sequence[float], target: Sequence[float]) -> dict[str, list[float]]:
        """Point the named RGB camera at ``target`` from a world-frame position."""
        camera_position = np.asarray(position, dtype=float)
        target_position = np.asarray(target, dtype=float)
        if camera_position.shape != (3,) or target_position.shape != (3,):
            raise ValueError("camera position and target must each contain x, y, and z")

        forward = target_position - camera_position
        forward_norm = np.linalg.norm(forward)
        if forward_norm == 0:
            raise ValueError("camera position and target must be different")
        camera_z_axis = -forward / forward_norm
        camera_x_axis = np.cross((0.0, 0.0, 1.0), camera_z_axis)
        x_axis_norm = np.linalg.norm(camera_x_axis)
        if x_axis_norm < 1e-8:
            raise ValueError("camera cannot look exactly along the world z axis")
        camera_x_axis /= x_axis_norm
        camera_y_axis = np.cross(camera_z_axis, camera_x_axis)
        rotation = np.column_stack((camera_x_axis, camera_y_axis, camera_z_axis))
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, rotation.ravel())

        camera = self.model.camera(RGB_CAMERA)
        self.model.cam_pos[camera.id] = camera_position
        self.model.cam_quat[camera.id] = quaternion
        mujoco.mj_forward(self.model, self.data)
        return {"position": camera_position.tolist(), "target": target_position.tolist()}

    def randomize_rgb_camera(self, rng: np.random.Generator, position_jitter: float = 0.05) -> dict[str, list[float]]:
        """Apply small seeded camera variation around the default tabletop view."""
        if position_jitter < 0:
            raise ValueError("position_jitter must be non-negative")
        table = self.model.geom("table_top")
        table_centre = self.data.geom_xpos[table.id]
        camera_position = np.asarray(RGB_CAMERA_POSITION) + rng.uniform(-position_jitter, position_jitter, size=3)
        target = table_centre + np.array((0.0, 0.0, 0.15))
        target += rng.uniform(-position_jitter / 2, position_jitter / 2, size=3)
        return self.set_rgb_camera(camera_position, target)

    def save_rgb(self, output_path: Path, width: int = 640, height: int = 480) -> None:
        """Render the RGB camera frame to a PNG file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.get_rgb(width=width, height=height)).save(output_path)

    def get_object_state(self, object_id: str) -> dict[str, Any]:
        """Return one tabletop object state in the shared world frame."""
        if object_id == "table":
            table_top = self.model.geom("table_top")
            return {
                "id": "table",
                "label": "table",
                "category": "support_surface",
                "affordances": ["supporting_surface"],
                "state": {},
                "pose": {"position": self._position_dict(self.data.geom_xpos[table_top.id])},
            }

        cube = next((item for item in _CUBES if item[0] == object_id), None)
        if cube is not None:
            body = self.model.body(object_id)
            position = self.data.xpos[body.id]
            return {
                "id": object_id,
                "label": cube[1],
                "category": "rigid_object",
                "affordances": ["graspable", "placeable"],
                "state": {"is_grasped": False},
                "pose": {"position": self._position_dict(position)},
            }

        target = next((item for item in _BINS if item[0] == object_id), None)
        if target is not None:
            body = self.model.body(object_id)
            return {
                "id": object_id,
                "label": target[1],
                "category": "target_zone",
                "affordances": ["target_zone", "supporting_surface"],
                "state": {},
                "pose": {"position": self._position_dict(self.data.xpos[body.id])},
            }
        raise KeyError(f"unknown tabletop object: {object_id}")

    def get_scene_state(self) -> dict[str, Any]:
        """Return scene metadata suitable for instruction generation and logging."""
        object_ids = ["table"] + [cube[0] for cube in _CUBES] + [target[0] for target in _BINS]
        return {
            "frame_id": WORLD_FRAME,
            "objects": [self.get_object_state(object_id) for object_id in object_ids],
            "relationships": [],
        }

    def get_robot_state(self) -> dict[str, Any]:
        """Return the Panda end-effector pose and normalized gripper state."""
        hand = self.model.body(END_EFFECTOR_FRAME)
        rotation = self.data.xmat[hand.id].reshape(3, 3)
        finger_positions = [self.data.qpos[self.model.joint(name).qposadr[0]] for name in ("finger_joint1", "finger_joint2")]
        gripper = 0.0 if float(np.mean(finger_positions)) >= 0.02 else 1.0
        return {
            "frame_id": WORLD_FRAME,
            "end_effector_frame": END_EFFECTOR_FRAME,
            "ee_position": self._position_dict(self.data.xpos[hand.id]),
            "ee_yaw": float(np.arctan2(rotation[1, 0], rotation[0, 0])),
            "gripper": gripper,
        }

    @staticmethod
    def _position_dict(position: np.ndarray) -> dict[str, float]:
        return {axis: float(value) for axis, value in zip(("x", "y", "z"), position)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="randomization seed for cube placement")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "tabletop.png")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--viewer", action="store_true", help="open the reset scene in the MuJoCo viewer")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    environment = TabletopEnvironment()
    scene_state = environment.reset(seed=args.seed)
    environment.save_rgb(args.output, width=args.width, height=args.height)
    print(f"saved RGB camera frame: {args.output}")
    print(f"seed: {args.seed}; objects: {[item['id'] for item in scene_state['objects']]}")
    print(f"robot state: {environment.get_robot_state()}")
    if args.viewer:
        print("opening interactive MuJoCo viewer; close its window to return to the terminal")
        viewer.launch(environment.model, environment.data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
