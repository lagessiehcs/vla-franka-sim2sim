"""A small, reproducible PyTorch supervised-learning foundation.

The synthetic problem mirrors the later project interface: predict a Cartesian
delta action ``[dx, dy, dz, dyaw, gripper]`` from robot and goal state. It is a
sanity test for the training pipeline, not a robot controller.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset


FEATURE_DIM = 8
ACTION_DIM = 5


@dataclass
class TrainingConfig:
    seed: int = 7
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 2e-3
    train_samples: int = 2_048
    validation_samples: int = 512
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
    device: str = "auto"


class CartesianActionDataset(Dataset[Tuple[Tensor, Tensor]]):
    """Tensor-backed examples of state -> Cartesian delta action.

    Features contain ``[ee_x, ee_y, ee_z, ee_yaw, gripper, goal_x, goal_y,
    goal_z]``. Targets use the project's planned action ordering.
    """

    def __init__(self, features: Tensor, actions: Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
            raise ValueError(f"Expected features shaped [N, {FEATURE_DIM}], got {tuple(features.shape)}")
        if actions.ndim != 2 or actions.shape != (features.shape[0], ACTION_DIM):
            raise ValueError(f"Expected actions shaped [N, {ACTION_DIM}], got {tuple(actions.shape)}")
        self.features = features.float()
        self.actions = actions.float()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        return self.features[index], self.actions[index]


def make_synthetic_dataset(num_samples: int, seed: int) -> CartesianActionDataset:
    """Create a deterministic, learnable action-regression dataset."""
    generator = torch.Generator().manual_seed(seed)
    ee_position = torch.empty(num_samples, 3).uniform_(-0.35, 0.35, generator=generator)
    ee_position[:, 2].uniform_(0.10, 0.45, generator=generator)
    ee_yaw = torch.empty(num_samples, 1).uniform_(-torch.pi, torch.pi, generator=generator)
    gripper = torch.randint(0, 2, (num_samples, 1), generator=generator).float()
    goal_position = torch.empty(num_samples, 3).uniform_(-0.30, 0.30, generator=generator)
    goal_position[:, 2].uniform_(0.08, 0.35, generator=generator)

    relative_goal = goal_position - ee_position
    delta_xyz = relative_goal.clamp(min=-0.05, max=0.05)
    desired_yaw = torch.atan2(relative_goal[:, 1:2], relative_goal[:, 0:1])
    delta_yaw = torch.atan2(torch.sin(desired_yaw - ee_yaw), torch.cos(desired_yaw - ee_yaw)) / torch.pi
    distance = relative_goal.norm(dim=1, keepdim=True)
    # Closing near the goal and opening when away provides a simple gripper label.
    gripper_command = (distance < 0.13).float()

    features = torch.cat((ee_position, ee_yaw / torch.pi, gripper, goal_position), dim=1)
    actions = torch.cat((delta_xyz, delta_yaw, gripper_command), dim=1)
    return CartesianActionDataset(features, actions)


class ActionMLP(nn.Module):
    """Small baseline network used solely to validate the ML plumbing."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(FEATURE_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, ACTION_DIM),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mean_loss(total_loss: float, examples: int) -> float:
    return total_loss / max(examples, 1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Tuple[Tensor, Tensor]],
    optimizer: AdamW,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for features, targets in loader:
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(features)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * features.shape[0]
    return mean_loss(total_loss, len(loader.dataset))


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader[Tuple[Tensor, Tensor]],
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    for features, targets in loader:
        features, targets = features.to(device), targets.to(device)
        total_loss += loss_fn(model(features), targets).item() * features.shape[0]
    return mean_loss(total_loss, len(loader.dataset))


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    epoch: int,
    best_validation_loss: float,
    config: TrainingConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_validation_loss": best_validation_loss,
            "config": asdict(config),
        },
        path,
    )


def load_checkpoint(path: Path, model: nn.Module, optimizer: AdamW, device: torch.device) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def run_training(config: TrainingConfig, resume_path: Path | None = None) -> Dict[str, float]:
    """Train, validate, and write best/last checkpoints. Returns final metrics."""
    if config.epochs < 1:
        raise ValueError("epochs must be at least 1")
    set_seed(config.seed)
    device = choose_device(config.device)
    train_dataset = make_synthetic_dataset(config.train_samples, config.seed)
    validation_dataset = make_synthetic_dataset(config.validation_samples, config.seed + 1)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False)

    model = ActionMLP().to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.SmoothL1Loss()
    start_epoch, best_validation_loss = 1, float("inf")
    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path, model, optimizer, device)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_loss = float(checkpoint["best_validation_loss"])

    checkpoint_dir = Path(config.checkpoint_dir)
    final_train_loss = final_validation_loss = float("nan")
    for epoch in range(start_epoch, config.epochs + 1):
        final_train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        final_validation_loss = validate(model, validation_loader, loss_fn, device)
        if final_validation_loss < best_validation_loss:
            best_validation_loss = final_validation_loss
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch, best_validation_loss, config)
        save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, epoch, best_validation_loss, config)
        print(
            f"epoch={epoch:03d} train_loss={final_train_loss:.6f} "
            f"validation_loss={final_validation_loss:.6f} best={best_validation_loss:.6f}"
        )

    metrics = {
        "train_loss": final_train_loss,
        "validation_loss": final_validation_loss,
        "best_validation_loss": best_validation_loss,
        "device": str(device),
    }
    results_path = Path(config.results_dir) / "supervised_sanity_metrics.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps({"config": asdict(config), "metrics": metrics}, indent=2) + "\n")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--resume", type=Path, help="Path to a last.pt checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_training(
        TrainingConfig(
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            checkpoint_dir=args.checkpoint_dir,
            results_dir=args.results_dir,
            device=args.device,
        ),
        args.resume,
    )
    print(f"completed on {metrics['device']}; best validation loss: {metrics['best_validation_loss']:.6f}")


if __name__ == "__main__":
    main()
