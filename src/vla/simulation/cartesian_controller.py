"""Cartesian IK control and scripted Panda pick-and-place demonstrations.

Run ``python -m vla.simulation.cartesian_controller --seed 0`` after
installing the simulator dependencies and initializing the Menagerie submodule.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import mujoco
import numpy as np
from mujoco import viewer

from .conventions import (
    END_EFFECTOR_FRAME,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    PANDA_GRIPPER_ACTUATOR,
    gripper_command_to_panda_control,
)
from .episode_logging import EpisodeLogger, PickPlaceSuccessChecker
from .mujoco_tabletop import PROJECT_ROOT, TabletopEnvironment


ARM_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
# The physical position actuators settle within this tolerance under gravity.
POSITION_TOLERANCE = 0.04
# Matches the cube-yaw randomization range used by the tabletop environment.
MAX_GRASP_YAW_ADJUSTMENT = 0.50
# Keep the complete gripper clear of the bin walls.  The cube is released
# above the target and falls into it under physics.
BIN_TRANSPORT_CLEARANCE = 0.25
BIN_FINGERTIP_CLEARANCE = 0.12


class InverseKinematicsError(RuntimeError):
    """Raised when a Cartesian target cannot be reached within tolerance."""


class BinCollisionError(RuntimeError):
    """Raised when a robot body contacts a target-bin geometry."""


@dataclass(frozen=True)
class WorkspaceLimits:
    """Axis-aligned Cartesian target limits in the shared world frame."""

    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]

    @classmethod
    def from_environment(cls, environment: TabletopEnvironment) -> "WorkspaceLimits":
        """Derive conservative limits from the Panda base and table geometry."""
        table = environment.model.geom("table_top")
        table_centre = environment.data.geom_xpos[table.id]
        table_half_size = environment.model.geom_size[table.id]
        robot_base = environment.data.xpos[environment.model.body("link0").id]
        table_top = table_centre[2] + table_half_size[2]
        return cls(
            x=(float(robot_base[0] + 0.20), float(table_centre[0] + table_half_size[0] - 0.07)),
            y=(float(table_centre[1] - table_half_size[1] + 0.07), float(table_centre[1] + table_half_size[1] - 0.07)),
            z=(float(table_top + 0.07), float(table_top + 0.60)),
        )

    def contains(self, position: Sequence[float]) -> bool:
        """Return whether ``position`` is inside all three Cartesian limits."""
        x, y, z = position
        return self.x[0] <= x <= self.x[1] and self.y[0] <= y <= self.y[1] and self.z[0] <= z <= self.z[1]

    def require(self, position: Sequence[float]) -> np.ndarray:
        """Validate and return a 3-D target position."""
        target = np.asarray(position, dtype=float)
        if target.shape != (3,):
            raise ValueError("Cartesian target must contain x, y, and z")
        if not self.contains(target):
            raise ValueError(
                f"target {target.tolist()} is outside workspace "
                f"x={self.x}, y={self.y}, z={self.z}"
            )
        return target


class CartesianPandaController:
    """Position-controlled Cartesian Panda controller using damped-least-squares IK."""

    def __init__(self, environment: TabletopEnvironment) -> None:
        self.environment = environment
        self.model = environment.model
        self.data = environment.data
        self.workspace = WorkspaceLimits.from_environment(environment)
        self._hand_body_id = self.model.body(END_EFFECTOR_FRAME).id
        self._reference_hand_rotation = self.data.xmat[self._hand_body_id].reshape(3, 3).copy()
        self._arm_qpos_addresses = np.array(
            [self.model.joint(name).qposadr[0] for name in ARM_JOINT_NAMES], dtype=int
        )
        self._arm_dof_addresses = np.array(
            [self.model.joint(name).dofadr[0] for name in ARM_JOINT_NAMES], dtype=int
        )
        self._arm_actuator_ids = np.array([self.model.actuator(f"actuator{index}").id for index in range(1, 8)])
        self._gripper_actuator_id = self.model.actuator(PANDA_GRIPPER_ACTUATOR).id
        self._viewer: Any | None = None
        self._trajectory_recorder: Any | None = None
        self._robot_body_ids = {
            body_id
            for body_id in range(self.model.nbody)
            if self.model.body(body_id).name.startswith("link")
            or self.model.body(body_id).name in {"hand", "left_finger", "right_finger"}
        }
        self._bin_body_ids = {
            self.model.body(item["id"]).id
            for item in environment.get_scene_state()["objects"]
            if item["category"] == "target_zone"
        }

    def attach_viewer(self, live_viewer: Any) -> None:
        """Show every simulated physics step in a MuJoCo passive viewer."""
        self._viewer = live_viewer

    def attach_trajectory_recorder(self, recorder: Any | None) -> None:
        """Record one expert action for every scripted controller waypoint."""
        self._trajectory_recorder = recorder

    def solve_ik(
        self,
        target_position: Sequence[float],
        target_yaw: float | None = None,
        max_iterations: int = 200,
    ) -> np.ndarray:
        """Return arm joint targets for a reachable world-frame hand target.

        The solver uses the MuJoCo hand Jacobian and damped least squares.  It
        leaves the live simulation state unchanged; ``move_to`` applies the
        returned joint target through Panda's position actuators.
        """
        target = self.workspace.require(target_position)
        original_qpos = self.data.qpos.copy()
        original_qvel = self.data.qvel.copy()
        original_ctrl = self.data.ctrl.copy()
        joint_ranges = self.model.jnt_range[: len(ARM_JOINT_NAMES)]
        try:
            for _ in range(max_iterations):
                mujoco.mj_forward(self.model, self.data)
                position_error = target - self.data.xpos[self._hand_body_id]
                orientation_error = self._orientation_error(self._target_rotation(target_yaw))
                if np.linalg.norm(position_error) <= 1e-4 and np.linalg.norm(orientation_error) <= 2e-3:
                    return self.data.qpos[self._arm_qpos_addresses].copy()

                jacobian_position = np.zeros((3, self.model.nv))
                jacobian_rotation = np.zeros((3, self.model.nv))
                mujoco.mj_jacBody(
                    self.model, self.data, jacobian_position, jacobian_rotation, self._hand_body_id
                )
                jacobian = np.vstack(
                    (jacobian_position[:, self._arm_dof_addresses], jacobian_rotation[:, self._arm_dof_addresses])
                )
                error = np.concatenate((position_error, orientation_error))

                damping = 0.03
                joint_delta = jacobian.T @ np.linalg.solve(
                    jacobian @ jacobian.T + damping**2 * np.eye(jacobian.shape[0]), error
                )
                joint_delta = np.clip(joint_delta, -0.05, 0.05)
                candidate = self.data.qpos[self._arm_qpos_addresses] + joint_delta
                self.data.qpos[self._arm_qpos_addresses] = np.clip(
                    candidate, joint_ranges[:, 0] + 0.01, joint_ranges[:, 1] - 0.01
                )
            raise InverseKinematicsError(f"could not reach Cartesian target {target.tolist()}")
        finally:
            self.data.qpos[:] = original_qpos
            self.data.qvel[:] = original_qvel
            self.data.ctrl[:] = original_ctrl
            mujoco.mj_forward(self.model, self.data)

    def move_to(
        self,
        target_position: Sequence[float],
        target_yaw: float | None = None,
        max_cartesian_step: float = 0.035,
        physics_steps_per_waypoint: int = 45,
    ) -> dict[str, Any]:
        """Move the hand along a straight Cartesian path to ``target_position``."""
        target = self.workspace.require(target_position)
        if max_cartesian_step <= 0 or physics_steps_per_waypoint <= 0:
            raise ValueError("motion step sizes must be positive")
        start = self.data.xpos[self._hand_body_id].copy()
        start_yaw = self._hand_yaw()
        yaw_change = 0.0 if target_yaw is None else self._angle_difference(target_yaw, start_yaw)
        waypoint_count = max(
            1,
            int(np.ceil(np.linalg.norm(target - start) / max_cartesian_step)),
            int(np.ceil(abs(yaw_change) / 0.10)),
        )
        for index in range(1, waypoint_count + 1):
            waypoint = start + (target - start) * (index / waypoint_count)
            waypoint_yaw = None if target_yaw is None else start_yaw + yaw_change * (index / waypoint_count)
            current_position = self.data.xpos[self._hand_body_id].copy()
            current_yaw = self._hand_yaw()
            action = np.array(
                (*((waypoint - current_position).tolist()), 0.0 if waypoint_yaw is None else self._angle_difference(waypoint_yaw, current_yaw), self.environment.get_robot_state()["gripper"])
            )
            self._record_expert_action(action, "move")
            self.data.ctrl[self._arm_actuator_ids] = self.solve_ik(waypoint, waypoint_yaw)
            self._step_physics(physics_steps_per_waypoint)

        # Position actuators need a short settling interval after the final
        # waypoint; otherwise the reported hand state is still in transit.
        self._step_physics(250)
        error = float(np.linalg.norm(target - self.data.xpos[self._hand_body_id]))
        if error > POSITION_TOLERANCE:
            raise InverseKinematicsError(f"hand stopped {error:.3f} m from target {target.tolist()}")
        return self.environment.get_robot_state()

    def apply_action(self, action: Sequence[float]) -> dict[str, Any]:
        """Apply one project-format action: ``[dx, dy, dz, dyaw, gripper]``."""
        command = np.asarray(action, dtype=float)
        if command.shape != (5,):
            raise ValueError("action must contain [dx, dy, dz, dyaw, gripper]")
        robot = self.environment.get_robot_state()
        target = np.array([robot["ee_position"][axis] for axis in ("x", "y", "z")]) + command[:3]
        self.move_to(target, target_yaw=robot["ee_yaw"] + command[3])
        if command[4] == GRIPPER_OPEN:
            self.open_gripper()
        elif command[4] == GRIPPER_CLOSED:
            self.close_gripper()
        else:
            raise ValueError("gripper action must be 0 (open) or 1 (closed)")
        return self.environment.get_robot_state()

    def open_gripper(self) -> None:
        """Open the fingers."""
        self.environment.set_grasped_object(None)
        self._record_expert_action(np.array((0.0, 0.0, 0.0, 0.0, GRIPPER_OPEN)), "open_gripper")
        self.data.ctrl[self._gripper_actuator_id] = gripper_command_to_panda_control(GRIPPER_OPEN)
        self._step_physics(150)

    def close_gripper(self, object_id: str | None = None) -> None:
        """Close the fingers and verify a requested cube is physically grasped."""
        self._record_expert_action(np.array((0.0, 0.0, 0.0, 0.0, GRIPPER_CLOSED)), "close_gripper")
        self.data.ctrl[self._gripper_actuator_id] = gripper_command_to_panda_control(GRIPPER_CLOSED)
        self._step_physics(200)
        if object_id is not None:
            if not self._cube_is_between_fingers(object_id):
                raise RuntimeError(f"physical grasp failed for {object_id}")
            self.environment.set_grasped_object(object_id)

    def scripted_pick_and_place(
        self,
        object_id: str,
        target_id: str,
        scene_id: str = "scripted_tabletop",
        task_id: str | None = None,
        logger: EpisodeLogger | None = None,
        trajectory_recorder: Any | None = None,
    ) -> dict[str, Any]:
        """Execute approach, grasp, lift, place, and retreat for one cube/bin pair.

        The gripper closes around the cube before lifting it to the target.
        """
        self._validate_task_objects(object_id, target_id)
        previous_recorder = self._trajectory_recorder
        self.attach_trajectory_recorder(trajectory_recorder)
        task_id = task_id or f"place_{object_id}_in_{target_id}"
        logger = logger or EpisodeLogger(scene_id=scene_id, task_id=task_id)
        try:
            self.open_gripper()
            self._record(logger, "open_gripper", object_id, target_id, [0.0, 0.0, 0.0, 0.0, GRIPPER_OPEN])

            cube_position = self.data.xpos[self.model.body(object_id).id].copy()
            grasp_yaw = self._grasp_yaw(object_id)
            approach = cube_position + np.array((0.0, 0.0, 0.18))
            self.move_to(approach, target_yaw=grasp_yaw)
            self._record(logger, "approach_object", object_id, target_id)

            grasp_position = cube_position + np.array((0.0, 0.0, 0.12))
            self.move_to(grasp_position, target_yaw=grasp_yaw)
            self._record(logger, "descend_to_object", object_id, target_id)

            self.close_gripper(object_id)
            self._record(logger, "close_gripper", object_id, target_id, [0.0, 0.0, 0.0, 0.0, GRIPPER_CLOSED])

            bin_top = self._bin_top(target_id)
            transport_height = max(
                self.data.xpos[self._hand_body_id][2] + 0.18,
                bin_top + BIN_TRANSPORT_CLEARANCE,
            )
            lift_position = self.data.xpos[self._hand_body_id].copy()
            lift_position[2] = transport_height
            self.move_to(lift_position, target_yaw=grasp_yaw)
            self._record(logger, "lift_object", object_id, target_id)

            target_position = self.data.xpos[self.model.body(target_id).id].copy()
            above_target = target_position.copy()
            above_target[2] = transport_height
            self.move_to(above_target, target_yaw=grasp_yaw)
            self._record(logger, "move_above_target", object_id, target_id)

            place_hand_position = target_position.copy()
            place_hand_position[2] = bin_top + self._fingertip_clearance_below_hand() + BIN_FINGERTIP_CLEARANCE
            self.move_to(place_hand_position, target_yaw=grasp_yaw)
            self._record(logger, "descend_to_target", object_id, target_id)

            self.open_gripper()
            self._step_physics(300)
            self._record(logger, "open_gripper", object_id, target_id, [0.0, 0.0, 0.0, 0.0, GRIPPER_OPEN])

            retreat_position = self.data.xpos[self._hand_body_id].copy() + np.array((0.0, 0.0, 0.18))
            self.move_to(retreat_position, target_yaw=grasp_yaw)
            self._record(logger, "retreat", object_id, target_id)

            success = PickPlaceSuccessChecker().check(self.environment, object_id, target_id)
            return {"success": success, "episode": logger.to_dict(success)}
        finally:
            self.attach_trajectory_recorder(previous_recorder)

    def _step_physics(self, steps: int) -> None:
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
            self._raise_if_robot_contacts_bin()
            if self._viewer is not None and self._viewer.is_running():
                self._viewer.sync()
                # Run at approximately the model's simulation rate so the
                # approach, grasp, lift, and placement are visible.
                time.sleep(self.model.opt.timestep)

    def _cube_is_between_fingers(self, object_id: str) -> bool:
        """Return whether both physical fingertip bodies contact ``object_id``."""
        contacted_fingers: set[str] = set()
        for contact in self.data.contact:
            first_body = self.model.body(self.model.geom_bodyid[contact.geom1]).name
            second_body = self.model.body(self.model.geom_bodyid[contact.geom2]).name
            if object_id not in (first_body, second_body):
                continue
            contacted_fingers.update({first_body, second_body} & {"left_finger", "right_finger"})
        return contacted_fingers == {"left_finger", "right_finger"}

    def _record_expert_action(self, action: np.ndarray, phase: str) -> None:
        """Send one pre-action observation and small Cartesian command to a recorder."""
        if self._trajectory_recorder is not None:
            self._trajectory_recorder.record(self.environment, action, phase)

    def _bin_top(self, target_id: str) -> float:
        """Return the highest upper surface of one target bin."""
        geometry_names = ("base", "front", "back", "left", "right")
        return max(
            self.data.geom_xpos[self.model.geom(f"{target_id}_{name}").id][2]
            + self.model.geom_size[self.model.geom(f"{target_id}_{name}").id][2]
            for name in geometry_names
        )

    def _fingertip_clearance_below_hand(self) -> float:
        """Return the vertical distance from hand frame to the lower fingertip."""
        hand_height = self.data.xpos[self._hand_body_id][2]
        fingertip_heights = (
            self.data.xpos[self.model.body("left_finger").id][2],
            self.data.xpos[self.model.body("right_finger").id][2],
        )
        return float(hand_height - min(fingertip_heights))

    def _raise_if_robot_contacts_bin(self) -> None:
        """Fail fast if any Panda link, hand, or finger collides with a bin."""
        for contact in self.data.contact:
            first_body_id = self.model.geom_bodyid[contact.geom1]
            second_body_id = self.model.geom_bodyid[contact.geom2]
            robot_hits_bin = (
                first_body_id in self._robot_body_ids and second_body_id in self._bin_body_ids
            ) or (
                second_body_id in self._robot_body_ids and first_body_id in self._bin_body_ids
            )
            if robot_hits_bin:
                first_name = self.model.body(first_body_id).name
                second_name = self.model.body(second_body_id).name
                raise BinCollisionError(f"robot/bin collision: {first_name} with {second_name}")

    def _hand_yaw(self) -> float:
        rotation = self.data.xmat[self._hand_body_id].reshape(3, 3)
        return float(np.arctan2(rotation[1, 0], rotation[0, 0]))

    def _grasp_yaw(self, object_id: str) -> float:
        """Rotate the gripper toward the nearest cube-face alignment."""
        cube = self.model.body(object_id)
        cube_rotation = self.data.xmat[cube.id].reshape(3, 3)
        cube_yaw = float(np.arctan2(cube_rotation[1, 0], cube_rotation[0, 0]))
        # The hand's local y-axis is its finger-closing direction. A 180-degree
        # flip is physically equivalent, so first choose the closer cube axis.
        candidates = (cube_yaw + np.pi / 2, cube_yaw - np.pi / 2)
        current_yaw = self._hand_yaw()
        aligned_yaw = min(candidates, key=lambda candidate: abs(self._angle_difference(candidate, current_yaw)))
        # A large wrist rotation can disturb a cube already resting on the
        # table. Turn toward the alignment gradually within a safe limit.
        yaw_adjustment = np.clip(
            self._angle_difference(aligned_yaw, current_yaw),
            -MAX_GRASP_YAW_ADJUSTMENT,
            MAX_GRASP_YAW_ADJUSTMENT,
        )
        return current_yaw + yaw_adjustment

    def _target_rotation(self, target_yaw: float | None) -> np.ndarray:
        """Keep the downward hand orientation, with an optional world-z yaw."""
        if target_yaw is None:
            return self._reference_hand_rotation
        reference_yaw = float(np.arctan2(self._reference_hand_rotation[1, 0], self._reference_hand_rotation[0, 0]))
        yaw_delta = self._angle_difference(target_yaw, reference_yaw)
        cosine, sine = np.cos(yaw_delta), np.sin(yaw_delta)
        world_z_rotation = np.array(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))
        return world_z_rotation @ self._reference_hand_rotation

    def _orientation_error(self, target_rotation: np.ndarray) -> np.ndarray:
        """Return the MuJoCo 3-D rotational error from hand to target."""
        current_quaternion = np.empty(4)
        target_quaternion = np.empty(4)
        mujoco.mju_mat2Quat(current_quaternion, self.data.xmat[self._hand_body_id])
        mujoco.mju_mat2Quat(target_quaternion, target_rotation.ravel())
        error = np.empty(3)
        mujoco.mju_subQuat(error, target_quaternion, current_quaternion)
        # mju_subQuat expresses this rotation in the hand frame, while
        # mj_jacBody's rotational Jacobian is in the world frame.
        return self.data.xmat[self._hand_body_id].reshape(3, 3) @ error

    def _record(self, logger: EpisodeLogger, phase: str, object_id: str, target_id: str, action: list[float] | None = None) -> None:
        logger.record(
            phase=phase,
            simulation_time=float(self.data.time),
            robot=self.environment.get_robot_state(),
            object_state=self.environment.get_object_state(object_id),
            target_state=self.environment.get_object_state(target_id),
            action=action,
        )

    def _validate_task_objects(self, object_id: str, target_id: str) -> None:
        object_state = self.environment.get_object_state(object_id)
        target_state = self.environment.get_object_state(target_id)
        if object_state["category"] != "rigid_object":
            raise ValueError(f"object must be a cube, got {object_id}")
        if target_state["category"] != "target_zone":
            raise ValueError(f"target must be a bin, got {target_id}")

    @staticmethod
    def _angle_difference(target: float, current: float) -> float:
        return float((target - current + np.pi) % (2 * np.pi) - np.pi)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--object", default="red_cube_1", dest="object_id")
    parser.add_argument("--target", default="yellow_bin_1", dest="target_id")
    parser.add_argument("--log", type=Path, default=PROJECT_ROOT / "results" / "scripted_episode.json")
    parser.add_argument("--viewer", action="store_true", help="show the complete episode in the MuJoCo viewer")
    return parser.parse_args(argv)


def _restart_with_mjpython_if_required(command_arguments: Sequence[str]) -> None:
    """Restart a macOS live-viewer command through MuJoCo's GUI launcher."""
    if sys.platform != "darwin" or "MJPYTHON_BIN" in os.environ:
        return
    launcher = Path(sys.executable).with_name("mjpython")
    if not launcher.is_file():
        raise RuntimeError(
            "MuJoCo's live viewer on macOS requires mjpython, but it was not found beside "
            f"{sys.executable}. Activate the vla Conda environment and try again."
        )
    print("macOS live viewer: restarting with mjpython")
    os.execv(
        str(launcher),
        [str(launcher), "-m", "vla.simulation.cartesian_controller", *command_arguments],
    )


def main(argv: Sequence[str] | None = None) -> int:
    command_arguments = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(command_arguments)
    if args.viewer:
        _restart_with_mjpython_if_required(command_arguments)
    environment = TabletopEnvironment()
    environment.reset(seed=args.seed)
    controller = CartesianPandaController(environment)
    if args.viewer:
        print("opening the MuJoCo viewer; the scripted episode will begin immediately")
        with viewer.launch_passive(environment.model, environment.data) as live_viewer:
            controller.attach_viewer(live_viewer)
            result = controller.scripted_pick_and_place(
                args.object_id, args.target_id, scene_id=f"tabletop_{args.seed:05d}"
            )
            live_viewer.sync()
            print("episode complete; close the viewer window to return to the terminal")
            while live_viewer.is_running():
                live_viewer.sync()
                time.sleep(0.01)
    else:
        result = controller.scripted_pick_and_place(args.object_id, args.target_id, scene_id=f"tabletop_{args.seed:05d}")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(json.dumps(result["episode"], indent=2) + "\n", encoding="utf-8")
    print(f"success: {result['success']['success']}; episode log: {args.log}")
    return 0 if result["success"]["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
