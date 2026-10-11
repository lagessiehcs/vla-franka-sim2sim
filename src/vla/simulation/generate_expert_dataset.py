"""Generate validated RGB-and-action expert trajectories for behavior cloning."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from vla.instructions.generator import generate_instruction, validate_scene_task

from .cartesian_controller import CartesianPandaController
from .conventions import ACTION_DIM
from .mujoco_tabletop import PROJECT_ROOT, TabletopEnvironment


@dataclass(frozen=True)
class DatasetSplit:
    """Episode-level train, validation, and test assignments."""

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]


class ExpertTrajectoryRecorder:
    """Save one RGB observation and expert action before every controller step."""

    def __init__(
        self,
        output_directory: Path,
        episode_id: str,
        instruction: str,
        task_metadata: dict[str, Any],
        width: int,
        height: int,
    ) -> None:
        self.output_directory = output_directory
        self.episode_id = episode_id
        self.instruction = instruction
        self.task_metadata = task_metadata
        self.width = width
        self.height = height
        self.records: list[dict[str, Any]] = []

    def record(self, environment: TabletopEnvironment, action: np.ndarray, phase: str) -> None:
        """Save the current observation and the next expert Cartesian action."""
        command = np.asarray(action, dtype=float)
        if command.shape != (ACTION_DIM,):
            raise ValueError(f"expert action must contain {ACTION_DIM} values")
        timestep = len(self.records)
        image_relative_path = Path("images") / f"{self.episode_id}_{timestep:04d}.png"
        environment.save_rgb(self.output_directory / image_relative_path, self.width, self.height)
        robot = environment.get_robot_state()
        position = robot["ee_position"]
        self.records.append(
            {
                "episode_id": self.episode_id,
                "timestep": timestep,
                "phase": phase,
                "rgb": image_relative_path.as_posix(),
                "instruction": self.instruction,
                "ee_position": [position["x"], position["y"], position["z"]],
                "ee_yaw": robot["ee_yaw"],
                "gripper_state": robot["gripper"],
                "action_xyz": command[:3].tolist(),
                "action_yaw": float(command[3]),
                "action_gripper": float(command[4]),
                "task_metadata": self.task_metadata,
            }
        )


def _task_definition(scene: dict[str, Any], episode_id: str, rng: np.random.Generator) -> dict[str, Any]:
    """Select one valid cube/bin pick-and-place task from a reset scene."""
    cubes = [item for item in scene["objects"] if item["category"] == "rigid_object"]
    bins = [item for item in scene["objects"] if item["category"] == "target_zone"]
    cube = cubes[int(rng.integers(len(cubes)))]
    target = bins[int(rng.integers(len(bins)))]
    return {
        "schema_version": "1.0",
        "scene_id": episode_id,
        "task_id": f"place_{cube['id']}_in_{target['id']}_{episode_id}",
        "scene": scene,
        "task": {
            "goal": f"Move the {cube['label']} into the {target['label']}",
            "steps": [
                {"action": "pick", "object_id": cube["id"]},
                {"action": "place", "object_id": cube["id"], "target_id": target["id"], "relation": "in"},
            ],
            "constraints": ["Keep the cube above the table"],
            "safety_notes": ["Use a collision-free path"],
            "success_conditions": [
                {
                    "description": f"The {cube['label']} is inside the {target['label']}",
                    "object_id": cube["id"],
                    "target_id": target["id"],
                }
            ],
        },
    }


def split_episode_ids(episode_ids: Sequence[str], seed: int) -> DatasetSplit:
    """Assign whole episodes to deterministic train, validation, and test splits."""
    if len(episode_ids) < 3:
        raise ValueError("at least three episodes are required for train, validation, and test splits")
    ordered = [episode_ids[index] for index in np.random.default_rng(seed).permutation(len(episode_ids))]
    train_count = max(1, round(len(ordered) * 0.8))
    validation_count = max(1, round(len(ordered) * 0.1))
    test_count = len(ordered) - train_count - validation_count
    if test_count < 1:
        train_count -= 1
        test_count = 1
    return DatasetSplit(
        train=tuple(ordered[:train_count]),
        validation=tuple(ordered[train_count : train_count + validation_count]),
        test=tuple(ordered[train_count + validation_count :]),
    )


def _split_name(assignments: DatasetSplit, episode_id: str) -> str:
    if episode_id in assignments.train:
        return "train"
    if episode_id in assignments.validation:
        return "validation"
    if episode_id in assignments.test:
        return "test"
    raise KeyError(f"episode is not assigned to a split: {episode_id}")


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def generate_expert_dataset(
    output_directory: Path,
    count: int,
    seed: int,
    width: int = 128,
    height: int = 96,
) -> dict[str, int]:
    """Run seeded scripted demonstrations and store behavior-cloning samples."""
    if count < 3:
        raise ValueError("count must be at least 3")
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_directory}")

    output_directory.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, Any]] = []
    environment = TabletopEnvironment()
    try:
        for index in range(count):
            episode_id = f"episode_{index:05d}"
            episode_seed = seed + index
            scene = environment.reset(seed=episode_seed)
            definition = _task_definition(scene, episode_id, np.random.default_rng(episode_seed))
            issues = validate_scene_task(definition)
            if issues:
                raise ValueError(f"generated invalid task {episode_id}: {issues}")
            instruction = generate_instruction(definition)
            task = definition["task"]
            object_id = task["steps"][0]["object_id"]
            target_id = task["steps"][1]["target_id"]
            recorder = ExpertTrajectoryRecorder(
                output_directory,
                episode_id,
                instruction,
                {
                    "scene_id": definition["scene_id"],
                    "task_id": definition["task_id"],
                    "object_id": object_id,
                    "target_id": target_id,
                    "scene": definition["scene"],
                },
                width,
                height,
            )
            result = CartesianPandaController(environment).scripted_pick_and_place(
                object_id,
                target_id,
                scene_id=episode_id,
                task_id=definition["task_id"],
                trajectory_recorder=recorder,
            )
            if not result["success"]["success"]:
                raise RuntimeError(f"scripted expert failed for {episode_id}")
            completed.append(
                {
                    "episode_id": episode_id,
                    "seed": episode_seed,
                    "instruction": instruction,
                    "task_metadata": recorder.task_metadata,
                    "success": result["success"],
                    "records": recorder.records,
                }
            )
    finally:
        environment.close()

    assignments = split_episode_ids([episode["episode_id"] for episode in completed], seed)
    all_records: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    by_split = {"train": [], "validation": [], "test": []}
    for episode in completed:
        split = _split_name(assignments, episode["episode_id"])
        records = [{**record, "split": split} for record in episode["records"]]
        trajectory_relative_path = Path("trajectories") / f"{episode['episode_id']}.jsonl"
        _write_jsonl(output_directory / trajectory_relative_path, records)
        all_records.extend(records)
        by_split[split].extend(records)
        episodes.append(
            {
                "episode_id": episode["episode_id"],
                "seed": episode["seed"],
                "split": split,
                "instruction": episode["instruction"],
                "task_metadata": episode["task_metadata"],
                "success": episode["success"],
                "trajectory": trajectory_relative_path.as_posix(),
                "steps": len(records),
            }
        )
    _write_jsonl(output_directory / "manifest.jsonl", all_records)
    _write_jsonl(output_directory / "episodes.jsonl", episodes)
    for split, records in by_split.items():
        _write_jsonl(output_directory / "splits" / f"{split}.jsonl", records)
    return validate_expert_dataset(output_directory)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"dataset file not found: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def validate_expert_dataset(dataset_directory: Path) -> dict[str, int]:
    """Validate required trajectory fields, paths, and episode-level split isolation."""
    records = _read_jsonl(dataset_directory / "manifest.jsonl")
    episodes = _read_jsonl(dataset_directory / "episodes.jsonl")
    if not records or not episodes:
        raise ValueError("expert dataset must contain records and episodes")
    required = {
        "episode_id", "timestep", "rgb", "instruction", "ee_position", "ee_yaw", "gripper_state",
        "action_xyz", "action_yaw", "action_gripper", "task_metadata", "split",
    }
    seen_steps: set[tuple[str, int]] = set()
    episode_splits: dict[str, str] = {}
    split_counts = {"train": 0, "validation": 0, "test": 0}
    for record in records:
        missing = required.difference(record)
        if missing:
            raise ValueError(f"trajectory record is missing fields: {sorted(missing)}")
        key = (record["episode_id"], record["timestep"])
        if key in seen_steps:
            raise ValueError(f"duplicate trajectory step: {key}")
        seen_steps.add(key)
        if not (dataset_directory / record["rgb"]).is_file():
            raise FileNotFoundError(f"missing RGB image: {record['rgb']}")
        if len(record["ee_position"]) != 3 or len(record["action_xyz"]) != 3:
            raise ValueError("end-effector position and Cartesian action must each contain three values")
        if record["split"] not in split_counts:
            raise ValueError(f"unknown split: {record['split']}")
        previous_split = episode_splits.setdefault(record["episode_id"], record["split"])
        if previous_split != record["split"]:
            raise ValueError(f"episode appears in multiple splits: {record['episode_id']}")
        split_counts[record["split"]] += 1
    if set(episode_splits) != {episode["episode_id"] for episode in episodes}:
        raise ValueError("episode manifest and trajectory manifest disagree")
    for split, count in split_counts.items():
        split_records = _read_jsonl(dataset_directory / "splits" / f"{split}.jsonl")
        if len(split_records) != count or any(record["split"] != split for record in split_records):
            raise ValueError(f"invalid {split} split manifest")
    return {"episodes": len(episodes), "steps": len(records), **{f"{name}_steps": count for name, count in split_counts.items()}}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "expert_dataset")
    parser.add_argument("--count", type=int, default=200, help="number of randomized scripted episodes")
    parser.add_argument("--seed", type=int, default=0, help="first deterministic episode seed")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--validate", type=Path, metavar="DATASET", help="validate an existing dataset instead")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.validate is not None:
        metrics = validate_expert_dataset(args.validate)
        print(f"valid expert dataset: {metrics}")
        return 0
    metrics = generate_expert_dataset(args.output, args.count, args.seed, args.width, args.height)
    print(f"generated valid expert dataset: {metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
