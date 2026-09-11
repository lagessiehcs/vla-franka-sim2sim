"""Convert versioned scene/task metadata into robot-ready language.

The module deliberately depends only on Python's standard library so simulator
and data-collection code can use it before the full VLA stack is connected.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "1.0"
SUPPORTED_ACTIONS = {"pick", "place"}
SUPPORTED_RELATIONS = {"on", "in", "next_to", "left_of", "right_of", "behind", "in_front_of"}


@dataclass(frozen=True)
class ValidationIssue:
    """A human-readable problem in a scene/task definition."""

    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


class SceneDefinitionError(ValueError):
    """Raised when instruction generation receives invalid metadata."""


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _object_index(scene_task: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    objects = scene_task.get("scene", {}).get("objects", [])
    return {item["id"]: item for item in objects if isinstance(item, Mapping) and _non_empty_string(item.get("id"))}


def validate_scene_task(scene_task: Mapping[str, Any]) -> list[ValidationIssue]:
    """Return all structural and cross-reference errors in ``scene_task``.

    This is intentionally a focused contract validator, not a simulator-state
    validator. It ensures the data has enough information to create an
    unambiguous instruction and that every referenced object exists.
    """
    issues: list[ValidationIssue] = []
    if not isinstance(scene_task, Mapping):
        return [ValidationIssue("$", "must be a JSON object")]

    if scene_task.get("schema_version") != SCHEMA_VERSION:
        issues.append(ValidationIssue("schema_version", f"must be {SCHEMA_VERSION!r}"))
    for key in ("scene_id", "task_id"):
        if not _non_empty_string(scene_task.get(key)):
            issues.append(ValidationIssue(key, "must be a non-empty string"))

    scene = _mapping(scene_task.get("scene"))
    if scene is None:
        issues.append(ValidationIssue("scene", "must be an object"))
        return issues
    if not _non_empty_string(scene.get("frame_id")):
        issues.append(ValidationIssue("scene.frame_id", "must be a non-empty string"))
    objects = scene.get("objects")
    if not isinstance(objects, list) or not objects:
        issues.append(ValidationIssue("scene.objects", "must be a non-empty array"))
        return issues

    object_ids: set[str] = set()
    for index, obj_value in enumerate(objects):
        path = f"scene.objects[{index}]"
        obj = _mapping(obj_value)
        if obj is None:
            issues.append(ValidationIssue(path, "must be an object"))
            continue
        object_id = obj.get("id")
        if not _non_empty_string(object_id):
            issues.append(ValidationIssue(f"{path}.id", "must be a non-empty string"))
        elif object_id in object_ids:
            issues.append(ValidationIssue(f"{path}.id", f"duplicates object id {object_id!r}"))
        else:
            object_ids.add(object_id)
        if not _non_empty_string(obj.get("label")):
            issues.append(ValidationIssue(f"{path}.label", "must be a non-empty string"))
        pose = _mapping(obj.get("pose"))
        position = _mapping(pose.get("position")) if pose else None
        if position is None or any(not isinstance(position.get(axis), (int, float)) for axis in ("x", "y", "z")):
            issues.append(ValidationIssue(f"{path}.pose.position", "must provide numeric x, y, and z values"))

    task = _mapping(scene_task.get("task"))
    if task is None:
        issues.append(ValidationIssue("task", "must be an object"))
        return issues
    if not _non_empty_string(task.get("goal")):
        issues.append(ValidationIssue("task.goal", "must be a non-empty string"))
    steps = task.get("steps")
    if not isinstance(steps, list) or not steps:
        issues.append(ValidationIssue("task.steps", "must be a non-empty array"))
        return issues

    for index, step_value in enumerate(steps):
        path = f"task.steps[{index}]"
        step = _mapping(step_value)
        if step is None:
            issues.append(ValidationIssue(path, "must be an object"))
            continue
        action = step.get("action")
        if action not in SUPPORTED_ACTIONS:
            issues.append(ValidationIssue(f"{path}.action", f"must be one of {sorted(SUPPORTED_ACTIONS)}"))
        object_id = step.get("object_id")
        if object_id not in object_ids:
            issues.append(
                ValidationIssue(
                    f"{path}.object_id", f"must reference an object in scene.objects (got {object_id!r})"
                )
            )
        target_id = step.get("target_id")
        if target_id is not None and target_id not in object_ids:
            issues.append(
                ValidationIssue(
                    f"{path}.target_id", f"must reference an object in scene.objects (got {target_id!r})"
                )
            )
        if action == "place":
            if target_id not in object_ids:
                issues.append(ValidationIssue(f"{path}.target_id", "is required for a place action"))
            if step.get("relation") not in SUPPORTED_RELATIONS:
                issues.append(ValidationIssue(f"{path}.relation", f"must be one of {sorted(SUPPORTED_RELATIONS)} for place"))

    for index, condition in enumerate(task.get("success_conditions", [])):
        condition_path = f"task.success_conditions[{index}]"
        if not isinstance(condition, Mapping):
            issues.append(ValidationIssue(condition_path, "must be an object"))
            continue
        for reference in ("object_id", "target_id"):
            if reference in condition and condition[reference] not in object_ids:
                issues.append(ValidationIssue(f"{condition_path}.{reference}", "must reference an object in scene.objects"))
    return issues


def _display_name(object_id: str, objects: Mapping[str, Mapping[str, Any]]) -> str:
    """Use IDs only when labels repeat, preserving natural instructions."""
    label = str(objects[object_id]["label"])
    label_count = sum(item.get("label") == label for item in objects.values())
    return f"the {label} ({object_id})" if label_count > 1 else f"the {label}"


def _step_instruction(step: Mapping[str, Any], objects: Mapping[str, Mapping[str, Any]]) -> str:
    action = step["action"]
    subject = _display_name(step["object_id"], objects)
    if action == "place":
        target = _display_name(step["target_id"], objects)
        relation = str(step["relation"]).replace("_", " ")
        return f"Place {subject} {relation} {target}."
    return f"Grasp {subject}."


def _sentence_list(items: Iterable[str]) -> str:
    return " ".join(item.rstrip(".") + "." for item in items if _non_empty_string(item))


def generate_instruction(scene_task: Mapping[str, Any]) -> str:
    """Generate deterministic, concise instructions from a valid definition."""
    issues = validate_scene_task(scene_task)
    if issues:
        raise SceneDefinitionError("Invalid scene/task definition:\n- " + "\n- ".join(map(str, issues)))
    task = scene_task["task"]
    objects = _object_index(scene_task)
    lines = [f"Goal: {task['goal'].rstrip('.')}.", "Steps:"]
    lines.extend(f"{number}. {_step_instruction(step, objects)}" for number, step in enumerate(task["steps"], start=1))
    constraints = task.get("constraints", [])
    if constraints:
        lines.append("Constraints: " + _sentence_list(constraints))
    safety_notes = task.get("safety_notes", [])
    if safety_notes:
        lines.append("Safety: " + _sentence_list(safety_notes))
    success_conditions = task.get("success_conditions", [])
    if success_conditions:
        descriptions = [str(condition.get("description", "")) for condition in success_conditions]
        lines.append("Complete when: " + _sentence_list(descriptions))
    return "\n".join(lines)


def load_scene_task(path: Path) -> dict[str, Any]:
    """Load one JSON scene/task definition from disk."""
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise SceneDefinitionError(f"{path}: root JSON value must be an object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_task", type=Path, help="JSON file conforming to schemas/scene_task.schema.json")
    parser.add_argument("--validate-only", action="store_true", help="report validity without generating text")
    args = parser.parse_args(argv)
    definition = load_scene_task(args.scene_task)
    issues = validate_scene_task(definition)
    if issues:
        parser.error("Invalid scene/task definition:\n- " + "\n- ".join(map(str, issues)))
    if args.validate_only:
        print(f"Valid scene/task definition: {args.scene_task}")
    else:
        print(generate_instruction(definition))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
