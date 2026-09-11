# VLA Franka

This repository contains the machine-learning foundation for a Vision-Language-Action (VLA) Franka manipulation project. It currently provides a small, reproducible PyTorch training pipeline that can be used to validate the ML environment and training workflow before connecting simulator-generated data.

## Current contents

The implemented supervised-learning example predicts a 5-D Cartesian action:

```text
[dx, dy, dz, dyaw, gripper]
```

It uses a synthetic robot/goal state and deterministic synthetic data, so it can validate the environment, training code, and checkpoint handling independently of a simulator.

The repository currently includes:

- A Python package under `src/vla/`.
- A tensor-backed `CartesianActionDataset` and PyTorch `DataLoader`s.
- A small MLP action-regression model.
- Training and validation loops with Smooth L1 loss.
- Automatic CUDA, Apple Silicon MPS, or CPU device selection.
- Reproducible random seeds.
- Best-model and last-model checkpoint saving.
- Checkpoint resume support.
- JSON metric output and automated smoke tests.

The current dataset is synthetic. It does not yet include simulator environments, RGB observations, language inputs, or Franka demonstration data.

The repository also includes a versioned scene/task metadata contract and a
dependency-free instruction generator. This can be developed and tested with
sample JSON before simulator data is available; see
[the instruction interface](docs/instruction_interface.md).

Shared coordinate frames, action units, and gripper encoding are defined in
[the robot interface conventions](docs/robot_interface.md).

## Project layout

```text
src/vla/
  training/
    supervised_sanity.py  # dataset, model, training, validation, checkpoints
  instructions/
    generator.py          # validate scene/task JSON and generate instructions
  simulation/
    mujoco_franka_inspect.py  # load/inspect/render the MuJoCo Franka Panda
    conventions.py            # shared frames, action format, gripper adapter
assets/mujoco_menagerie/  # official Franka model (Git submodule)
schemas/
  scene_task.schema.json  # scene/task interchange contract (v1.0)
examples/scene_tasks/     # valid, runnable scene/task definitions
tests/
  test_supervised_sanity.py
pyproject.toml            # package metadata and dependencies
requirements.txt          # runtime dependencies
```

## Installation

Use Python 3.9 or later. From the repository root, create and activate the
isolated Conda environment:

```bash
conda create -n vla python=3.10 pip
conda activate vla
python -m pip install -e '.[dev]'
```

For the MuJoCo Franka inspection tool, install the project-local simulation
dependencies as well:

```bash
python -m pip install -e '.[dev,simulation]'
git submodule update --init --recursive
```

The training script automatically selects CUDA when available, otherwise Apple Silicon MPS, then CPU. If the default PyTorch installation does not support your hardware, install the appropriate wheel from [PyTorch's installation guide](https://pytorch.org/get-started/locally/) before installing this project.

## Run the training example

```bash
python -m vla.training.supervised_sanity --epochs 30
```

The command prints training and validation loss for each epoch and creates:

- `checkpoints/best.pt` — state with the lowest validation loss.
- `checkpoints/last.pt` — most recent model and optimizer state.
- `results/supervised_sanity_metrics.json` — run configuration and final metrics.

Resume from the most recent checkpoint with a larger total epoch count:

```bash
python -m vla.training.supervised_sanity --epochs 50 --resume checkpoints/last.pt
```

## Verify the repository

```bash
pytest -q
```

## Generate task instructions

```bash
python -m vla.instructions.generator examples/scene_tasks/place_red_block_on_blue_plate.json
```

Use `--validate-only` to check a definition without producing language. The
schema, template behavior, required fields, and integration rules are in
[docs/instruction_interface.md](docs/instruction_interface.md).

The tests confirm the expected dataset tensor shapes and verify that training writes usable checkpoints and metrics.

## Inspect the MuJoCo Franka Panda

The official MuJoCo Menagerie model is included as the
`assets/mujoco_menagerie` Git submodule. The following command loads the Panda,
prints its joints, gripper, end-effector body, actuators, and camera details,
then writes a third-person RGB frame:

```bash
python -m vla.simulation.mujoco_franka_inspect --output results/franka_inspection.png
```

Install the optional simulator dependencies in the same Conda environment; no
system Python packages are required.

To run and view the Franka scene, start in the repository root and run:

```bash
python -m vla.simulation.mujoco_franka_inspect --viewer
```

The command first prints the Panda's joint, gripper, end-effector, actuator,
and camera details and saves `results/franka_inspection.png`. It then opens the
interactive viewer. The arm starts in its home pose and will not move until a
controller is added.

## Key implementation details

- `CartesianActionDataset` stores one feature tensor and action tensor per example.
- `DataLoader` shuffles training examples and keeps validation ordering stable.
- `train_one_epoch` performs the forward pass, loss calculation, gradient reset, backpropagation, and optimizer update.
- `validate` runs without gradients on held-out data.
- Checkpoints include the model state, optimizer state, epoch, best validation loss, and training configuration.
