# Robot Interface Conventions

All simulator code, datasets, controllers, and learned policies use these
conventions.

## Units and frames

| Item | Convention |
| --- | --- |
| Position and translation | metres |
| Angle and yaw | radians |
| World frame | `world` |
| World origin | Panda base body (`link0`) origin |
| World axes | `+x` forward into the workspace, `+y` robot left, `+z` upward |
| Coordinate system | right-handed |
| End-effector frame | Panda `hand` body frame |

The Panda hand pose is always reported relative to `world`. The controller
keeps roll and pitch fixed for initial tabletop tasks; `yaw` rotates around the
world `+z` axis.

## State

```text
ee_position  = [x, y, z]       # metres, world frame
ee_yaw       = yaw             # radians, relative to world frame
gripper_state = 0 or 1         # 0 = open, 1 = closed
```

## Action

Every policy and expert action uses this fixed ordering:

```text
[dx, dy, dz, dyaw, gripper]
```

| Field | Meaning |
| --- | --- |
| `dx`, `dy`, `dz` | End-effector translation in the world frame, metres per control step |
| `dyaw` | End-effector yaw change around world `+z`, radians per control step |
| `gripper` | `0` opens the gripper; `1` closes it |

The controller is responsible for clipping translation and yaw changes to its
configured safety limits.

## Panda gripper adapter

The project interface is normalized for the dataset and VLA model. The MuJoCo
Menagerie Panda model uses `actuator8` with a control range of `0` to `255`.
The controller maps the project command as follows:

| Project command | Meaning | Panda `actuator8` control |
| --- | --- | --- |
| `0` | open | `255` |
| `1` | closed | `0` |

Use `vla.simulation.conventions.gripper_command_to_panda_control` for this
conversion rather than placing this model-specific mapping in dataset or model
code.
