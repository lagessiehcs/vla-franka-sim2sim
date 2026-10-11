import json

import pytest

pytest.importorskip("mujoco")

from vla.simulation.generate_expert_dataset import generate_expert_dataset, validate_expert_dataset
from vla.simulation.mujoco_tabletop import PANDA_XML


def test_expert_dataset_contains_rgb_actions_and_episode_splits(tmp_path) -> None:
    if not PANDA_XML.is_file():
        pytest.skip("MuJoCo Menagerie submodule is not initialized")
    output_directory = tmp_path / "expert_dataset"

    metrics = generate_expert_dataset(output_directory, count=3, seed=12, width=64, height=48)
    repeated_metrics = validate_expert_dataset(output_directory)
    records = [json.loads(line) for line in (output_directory / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    episodes = [json.loads(line) for line in (output_directory / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]

    assert metrics == repeated_metrics
    assert metrics["episodes"] == 3
    assert metrics["steps"] > 100
    assert len(records) == metrics["steps"]
    assert {episode["split"] for episode in episodes} == {"train", "validation", "test"}
    assert {record["episode_id"] for record in records} == {episode["episode_id"] for episode in episodes}
    first = records[0]
    assert (output_directory / first["rgb"]).is_file()
    assert len(first["ee_position"]) == 3
    assert len(first["action_xyz"]) == 3
    assert {"scene_id", "task_id", "object_id", "target_id", "scene"}.issubset(first["task_metadata"])
    assert {record["action_gripper"] for record in records} == {0.0, 1.0}
