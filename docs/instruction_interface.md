# Scene Metadata and Instruction Interface (v1.0)

This is the scene/task interchange contract. A data producer (for example, a
simulator) writes one JSON scene/task definition; the instruction module
validates it and creates deterministic natural-language instructions.

The formal structural schema is [scene_task.schema.json](../schemas/scene_task.schema.json).
`vla.instructions.validate_scene_task` performs additional validation: each
object ID must be unique, and every object ID used in a task step must exist in
the scene.

## Minimal format

```json
{
  "schema_version": "1.0",
  "scene_id": "tabletop_blocks_001",
  "task_id": "place_red_block_on_blue_plate",
  "scene": {
    "frame_id": "world",
    "objects": [
      {
        "id": "red_block_1",
        "label": "red block",
        "pose": {"position": {"x": 0.42, "y": -0.18, "z": 0.03}}
      },
      {
        "id": "blue_plate_1",
        "label": "blue plate",
        "pose": {"position": {"x": 0.54, "y": 0.15, "z": 0.02}}
      }
    ]
  },
  "task": {
    "goal": "Move the red block onto the blue plate",
    "steps": [
      {"action": "pick", "object_id": "red_block_1"},
      {"action": "place", "object_id": "red_block_1", "target_id": "blue_plate_1", "relation": "on"}
    ]
  }
}
```

`scene.frame_id` names the coordinate frame used for every pose (normally
`world`). Positions are metres. `orientation_xyzw`, if supplied, is a unit
quaternion in `[x, y, z, w]` order. Object IDs must be stable, lowercase
identifiers; labels are human-facing terms. Include IDs in generated language
only when two objects have the same label.

## Instruction templates

The generator applies these templates in `task.steps` order:

| Action | Required fields | Output template |
| --- | --- | --- |
| `pick` | `object_id` | `Grasp the {object}.` |
| `place` | `object_id`, `target_id`, `relation` | `Place the {object} {relation} the {target}.` |

Allowed placement relations are `on`, `in`, `next_to`, `left_of`, `right_of`,
`behind`, and `in_front_of`. `constraints`, `safety_notes`, and
`success_conditions` are optional arrays, but should be supplied for any
physical manipulation task. They are appended as explicit sections so a VLA
can retain the task goal separately from motion limits and completion checks.

## Integration and versioning rules

- Producers must preserve `schema_version: "1.0"` and use metres in the stated frame.
- Add data only through optional object `category`, `affordances`, `state`, and
  scene `relationships` fields. Do not rename or change required fields in v1.0.
- A breaking change requires a new schema version and a converter; do not silently
  reuse the same version number.
- Before integration, run the validator on every produced JSON file.

## Use it

From the repository root:

```bash
python -m vla.instructions.generator examples/scene_tasks/place_red_block_on_blue_plate.json
python -m vla.instructions.generator examples/scene_tasks/place_red_block_on_blue_plate.json --validate-only
```

The first command produces the following pattern:

```text
Goal: Move the red block onto the blue plate.
Steps:
1. Grasp the red block.
2. Place the red block on the blue plate.
Constraints: Keep the block upright. Do not contact the wooden divider.
Safety: Use a collision-free path. Release only after the block is supported by the plate.
Complete when: The red block is resting on the blue plate and the gripper is open.
```
