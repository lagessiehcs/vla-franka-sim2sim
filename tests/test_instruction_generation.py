import json
from pathlib import Path

import pytest

from vla.instructions import SceneDefinitionError, generate_instruction, load_scene_task, validate_scene_task


EXAMPLES = Path(__file__).parents[1] / "examples" / "scene_tasks"


def test_place_task_validates_and_generates_expected_instruction() -> None:
    definition = load_scene_task(EXAMPLES / "place_red_block_on_blue_plate.json")

    assert validate_scene_task(definition) == []
    assert generate_instruction(definition) == (
        "Goal: Move the red block onto the blue plate.\n"
        "Steps:\n"
        "1. Grasp the red block.\n"
        "2. Place the red block on the blue plate.\n"
        "Constraints: Keep the block upright. Do not contact the wooden divider.\n"
        "Safety: Use a collision-free path. Release only after the block is supported by the plate.\n"
        "Complete when: The red block is resting on the blue plate and the gripper is open."
    )


def test_duplicate_labels_are_disambiguated_with_ids() -> None:
    definition = load_scene_task(EXAMPLES / "place_red_block_on_blue_plate.json")
    duplicate = json.loads(json.dumps(definition))
    duplicate["scene"]["objects"].append(
        {"id": "red_block_2", "label": "red block", "pose": {"position": {"x": 0.1, "y": 0.1, "z": 0.03}}}
    )

    output = generate_instruction(duplicate)

    assert "the red block (red_block_1)" in output


def test_validation_reports_missing_reference() -> None:
    definition = load_scene_task(EXAMPLES / "place_red_block_on_blue_plate.json")
    definition["task"]["steps"][0]["object_id"] = "missing_handle"

    issues = validate_scene_task(definition)

    assert any(issue.path == "task.steps[0].object_id" for issue in issues)
    with pytest.raises(SceneDefinitionError, match="missing_handle"):
        generate_instruction(definition)
