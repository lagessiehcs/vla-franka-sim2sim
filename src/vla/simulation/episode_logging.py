"""Episode logs and task-success checks for scripted tabletop tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .mujoco_tabletop import CUBE_HALF_SIZE, TabletopEnvironment


def _json_value(value: Any) -> Any:
    """Convert NumPy values recursively so an episode can be written as JSON."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


@dataclass
class EpisodeLogger:
    """Collect controller phase events in a portable JSON-friendly format."""

    scene_id: str
    task_id: str
    events: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        phase: str,
        simulation_time: float,
        robot: dict[str, Any],
        object_state: dict[str, Any],
        target_state: dict[str, Any],
        action: list[float] | None = None,
    ) -> None:
        """Append one high-level controller event."""
        event: dict[str, Any] = {
            "index": len(self.events),
            "phase": phase,
            "simulation_time": simulation_time,
            "robot": robot,
            "object": object_state,
            "target": target_state,
        }
        if action is not None:
            event["action"] = action
        self.events.append(_json_value(event))

    def to_dict(self, success: dict[str, Any]) -> dict[str, Any]:
        """Return the complete episode record."""
        return {
            "schema_version": "1.0",
            "scene_id": self.scene_id,
            "task_id": self.task_id,
            "events": self.events,
            "success": _json_value(success),
        }

    def save(self, output_path: Path, success: dict[str, Any]) -> None:
        """Write the complete episode record to ``output_path``."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(success), indent=2) + "\n", encoding="utf-8")


class PickPlaceSuccessChecker:
    """Check whether a released cube is resting inside its requested bin."""

    def check(self, environment: TabletopEnvironment, object_id: str, target_id: str) -> dict[str, Any]:
        """Return an explainable success result for one pick-and-place task."""
        cube = environment.model.body(object_id)
        target = environment.model.body(target_id)
        base = environment.model.geom(f"{target_id}_base")
        cube_position = environment.data.xpos[cube.id].copy()
        target_position = environment.data.xpos[target.id].copy()
        base_half_size = environment.model.geom_size[base.id]
        base_top = environment.data.geom_xpos[base.id][2] + base_half_size[2]
        horizontal_limit = base_half_size[:2] - CUBE_HALF_SIZE
        xy_offset = np.abs(cube_position[:2] - target_position[:2])

        inside_xy = bool(np.all(xy_offset <= horizontal_limit))
        expected_rest_height = base_top + CUBE_HALF_SIZE
        supported = bool(abs(cube_position[2] - expected_rest_height) <= 0.015)
        released = environment.get_object_state(object_id)["state"]["is_grasped"] is False
        gripper_open = environment.get_robot_state()["gripper"] == 0.0
        return {
            "success": inside_xy and supported and released and gripper_open,
            "inside_target_xy": inside_xy,
            "supported": supported,
            "released": released,
            "gripper_open": gripper_open,
            "cube_position": cube_position.tolist(),
            "target_position": target_position.tolist(),
        }
