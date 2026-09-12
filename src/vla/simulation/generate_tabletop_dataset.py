"""Generate seeded tabletop RGB scenes, valid task JSON, and language commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from vla.instructions.generator import generate_instruction, validate_scene_task

from .mujoco_tabletop import PROJECT_ROOT, TabletopEnvironment


def _task_definition(scene: dict[str, Any], sample_id: str, rng: np.random.Generator) -> dict[str, Any]:
    """Create one valid pick-and-place task for the current scene state."""
    cubes = [item for item in scene["objects"] if item["category"] == "rigid_object"]
    bins = [item for item in scene["objects"] if item["category"] == "target_zone"]
    cube = cubes[int(rng.integers(len(cubes)))]
    target = bins[int(rng.integers(len(bins)))]
    return {
        "schema_version": "1.0",
        "scene_id": sample_id,
        "task_id": f"place_{cube['id']}_in_{target['id']}_{sample_id}",
        "scene": scene,
        "task": {
            "goal": f"Move the {cube['label']} into the {target['label']}",
            "steps": [
                {"action": "pick", "object_id": cube["id"]},
                {"action": "place", "object_id": cube["id"], "target_id": target["id"], "relation": "in"},
            ],
            "constraints": ["Keep the cube above the table", "Do not contact the other cubes"],
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


def generate_dataset(
    output_directory: Path,
    count: int,
    seed: int,
    width: int = 640,
    height: int = 480,
    camera_jitter: float = 0.05,
) -> list[dict[str, Any]]:
    """Write ``count`` scene images, task definitions, and manifest entries.

    The output directory must be new or empty, preventing accidental mixing of
    two dataset runs.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_directory}")

    images_directory = output_directory / "images"
    definitions_directory = output_directory / "scene_tasks"
    images_directory.mkdir(parents=True, exist_ok=True)
    definitions_directory.mkdir(parents=True, exist_ok=True)

    environment = TabletopEnvironment()
    manifest: list[dict[str, Any]] = []
    for index in range(count):
        sample_seed = seed + index
        sample_id = f"tabletop_{index:05d}"
        scene = environment.reset(seed=sample_seed)
        rng = np.random.default_rng(sample_seed)
        camera = environment.randomize_rgb_camera(rng, position_jitter=camera_jitter)
        definition = _task_definition(scene, sample_id, rng)
        issues = validate_scene_task(definition)
        if issues:
            raise ValueError(f"generated invalid task {sample_id}: {issues}")
        instruction = generate_instruction(definition)

        image_relative_path = Path("images") / f"{sample_id}.png"
        definition_relative_path = Path("scene_tasks") / f"{sample_id}.json"
        environment.save_rgb(output_directory / image_relative_path, width=width, height=height)
        (output_directory / definition_relative_path).write_text(json.dumps(definition, indent=2) + "\n", encoding="utf-8")
        manifest.append(
            {
                "sample_id": sample_id,
                "seed": sample_seed,
                "image": image_relative_path.as_posix(),
                "scene_task": definition_relative_path.as_posix(),
                "instruction": instruction,
                "camera": camera,
            }
        )

    manifest_path = output_directory / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(item) + "\n" for item in manifest),
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "tabletop_dataset")
    parser.add_argument("--count", type=int, default=200, help="number of RGB scene/task pairs to create")
    parser.add_argument("--seed", type=int, default=0, help="first deterministic scene seed")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-jitter", type=float, default=0.05, help="camera-position variation in metres")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = generate_dataset(
        output_directory=args.output,
        count=args.count,
        seed=args.seed,
        width=args.width,
        height=args.height,
        camera_jitter=args.camera_jitter,
    )
    print(f"generated {len(manifest)} RGB scene/task pairs in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
