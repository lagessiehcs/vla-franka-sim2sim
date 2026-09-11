"""Load and inspect the project Franka Panda MuJoCo scene.

Run with ``python -m vla.simulation.mujoco_franka_inspect`` after installing
the optional ``simulation`` dependencies and initializing Git submodules.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import mujoco
from mujoco import viewer
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCENE = PROJECT_ROOT / "assets" / "mujoco_menagerie" / "franka_emika_panda" / "scene.xml"
END_EFFECTOR_BODY = "hand"


def load_model(scene_path: Path = DEFAULT_SCENE) -> mujoco.MjModel:
    """Load the local scene and provide an actionable error for missing assets."""
    if not scene_path.is_file():
        raise FileNotFoundError(
            f"MuJoCo scene not found: {scene_path}. Run `git submodule update --init --recursive`."
        )
    return mujoco.MjModel.from_xml_path(str(scene_path))


def _joint_type_name(joint_type: int) -> str:
    return mujoco.mjtJoint(joint_type).name.removeprefix("mjJNT_").lower()


def make_third_person_camera() -> mujoco.MjvCamera:
    """Create the fixed third-person view used for initial RGB observations."""
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.0, 0.45)
    camera.distance = 2.0
    camera.azimuth = 135.0
    camera.elevation = -25.0
    return camera


def print_model_summary(model: mujoco.MjModel, data: mujoco.MjData, camera: mujoco.MjvCamera) -> None:
    """Print the arm joints, gripper joints, end-effector, actuators, and camera."""
    print("model: Franka Emika Panda (MuJoCo Menagerie)")
    print(f"degrees of freedom: nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print("joints:")
    for joint_id in range(model.njnt):
        joint = model.joint(joint_id)
        joint_range = model.jnt_range[joint_id]
        print(
            f"  {joint.name}: type={_joint_type_name(int(model.jnt_type[joint_id]))} "
            f"range=[{joint_range[0]:.4f}, {joint_range[1]:.4f}]"
        )

    ee = model.body(END_EFFECTOR_BODY)
    print(f"end effector: body={END_EFFECTOR_BODY}, world_position={data.xpos[ee.id].round(4).tolist()}")
    print("actuators:")
    for actuator_id in range(model.nu):
        actuator = model.actuator(actuator_id)
        control_range = model.actuator_ctrlrange[actuator_id]
        print(f"  {actuator.name}: ctrlrange=[{control_range[0]:.4f}, {control_range[1]:.4f}]")

    print(
        "camera: third-person free camera, "
        f"lookat={camera.lookat.round(4).tolist()}, distance={camera.distance:.2f}, "
        f"azimuth={camera.azimuth:.1f}, elevation={camera.elevation:.1f}"
    )


def render_camera(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera: mujoco.MjvCamera,
    output_path: Path,
    width: int,
    height: int,
) -> None:
    """Render the configured third-person camera and save it as a PNG image."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera=camera)
        Image.fromarray(renderer.render()).save(output_path)


def inspect(scene_path: Path, output_path: Path, width: int, height: int, launch_viewer: bool = False) -> None:
    """Load the home keyframe, print the requested inspection data, and render RGB."""
    model = load_model(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    camera = make_third_person_camera()
    print_model_summary(model, data, camera)
    render_camera(model, data, camera, output_path, width, height)
    print(f"saved RGB camera frame: {output_path}")
    if launch_viewer:
        print("opening interactive MuJoCo viewer; close its window to return to the terminal")
        viewer.launch(model, data)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE, help="MuJoCo XML scene to inspect")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "franka_inspection.png")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--viewer", action="store_true", help="open the interactive MuJoCo viewer after rendering")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    inspect(args.scene, args.output, args.width, args.height, args.viewer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
