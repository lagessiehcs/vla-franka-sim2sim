"""Train and run the semantic RGB-and-language tabletop baseline.

The baseline freezes pretrained CLIP image and text encoders, then trains a
small network to select one cube and one target bin.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from vla.training.supervised_sanity import choose_device, set_seed


OBJECT_IDS = ("red_cube_1", "green_cube_1", "blue_cube_1")
TARGET_IDS = ("yellow_bin_1", "purple_bin_1")


@dataclass(frozen=True)
class SemanticExample:
    """One RGB image, instruction, and its intended cube/bin pair."""

    image_path: Path
    instruction: str
    object_id: str
    target_id: str
    scene_id: str


@dataclass(frozen=True)
class SemanticPrediction:
    """The cube and bin selected from an RGB image and instruction."""

    object_id: str
    target_id: str
    object_confidence: float
    target_confidence: float


@dataclass
class SemanticTrainingConfig:
    """Configuration for frozen-feature semantic grounding training."""

    seed: int = 7
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    train_fraction: float = 0.8
    device: str = "auto"
    backbone: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"


class SemanticFeatureDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Precomputed frozen features and cube/bin classification labels."""

    def __init__(self, features: Tensor, object_labels: Tensor, target_labels: Tensor) -> None:
        if features.ndim != 2:
            raise ValueError("features must be shaped [samples, feature_dimension]")
        count = features.shape[0]
        if object_labels.shape != (count,) or target_labels.shape != (count,):
            raise ValueError("labels must contain one entry per feature row")
        self.features = features.float()
        self.object_labels = object_labels.long()
        self.target_labels = target_labels.long()

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.features[index], self.object_labels[index], self.target_labels[index]


class SemanticGroundingNetwork(nn.Module):
    """Small trainable classifier over frozen image-and-text features."""

    def __init__(self, feature_dimension: int, hidden_dimension: int = 256) -> None:
        super().__init__()
        if feature_dimension <= 0 or hidden_dimension <= 0:
            raise ValueError("feature dimensions must be positive")
        self.feature_dimension = feature_dimension
        self.hidden_dimension = hidden_dimension
        self.fusion = nn.Sequential(
            nn.Linear(feature_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.ReLU(),
        )
        self.object_head = nn.Linear(hidden_dimension, len(OBJECT_IDS))
        self.target_head = nn.Linear(hidden_dimension, len(TARGET_IDS))

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.fusion(features)
        return self.object_head(hidden), self.target_head(hidden)


class FrozenCLIPEncoder:
    """Frozen pretrained CLIP encoders used to make semantic features."""

    def __init__(self, backbone: str, pretrained: str, device: torch.device) -> None:
        try:
            import open_clip
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Install semantic dependencies with `python -m pip install -e '.[semantic]'`."
            ) from error

        self.device = device
        self.backbone = backbone
        self.pretrained = pretrained
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(backbone, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(backbone)
        self.model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, images: Sequence[Image.Image], instructions: Sequence[str]) -> Tensor:
        """Return normalized concatenated image and text features."""
        if len(images) != len(instructions) or not images:
            raise ValueError("images and instructions must be non-empty and have equal length")
        image_batch = torch.stack([self.preprocess(image.convert("RGB")) for image in images]).to(self.device)
        text_batch = self.tokenizer(list(instructions)).to(self.device)
        image_features = self.model.encode_image(image_batch)
        text_features = self.model.encode_text(text_batch)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return torch.cat((image_features, text_features), dim=-1).float()


class SemanticVLAPolicy:
    """Use frozen RGB/language features and trained heads to select a task pair."""

    def __init__(self, encoder: FrozenCLIPEncoder, model: SemanticGroundingNetwork) -> None:
        self.encoder = encoder
        self.model = model.to(encoder.device).eval()

    @torch.inference_mode()
    def predict(self, rgb: np.ndarray, instruction: str) -> SemanticPrediction:
        """Select one known cube and target bin from one camera image and command."""
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb must be a uint8 image shaped [height, width, 3]")
        if not instruction.strip():
            raise ValueError("instruction must not be empty")
        features = self.encoder.encode([Image.fromarray(rgb)], [instruction])
        object_logits, target_logits = self.model(features)
        object_probabilities = object_logits.softmax(dim=-1)[0]
        target_probabilities = target_logits.softmax(dim=-1)[0]
        object_index = int(object_probabilities.argmax())
        target_index = int(target_probabilities.argmax())
        return SemanticPrediction(
            object_id=OBJECT_IDS[object_index],
            target_id=TARGET_IDS[target_index],
            object_confidence=float(object_probabilities[object_index]),
            target_confidence=float(target_probabilities[target_index]),
        )


def _task_ids(definition: dict[str, Any]) -> tuple[str, str]:
    """Read the selected cube and target bin from one task definition."""
    for step in definition.get("task", {}).get("steps", []):
        if step.get("action") == "place":
            object_id, target_id = step.get("object_id"), step.get("target_id")
            if object_id in OBJECT_IDS and target_id in TARGET_IDS:
                return object_id, target_id
    raise ValueError("task must contain a place step for a known cube and target bin")


def load_semantic_examples(dataset_directory: Path) -> list[SemanticExample]:
    """Load RGB/instruction/label records from a generated tabletop dataset."""
    manifest_path = dataset_directory / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"dataset manifest not found: {manifest_path}")

    examples: list[SemanticExample] = []
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        entry = json.loads(line)
        image_path = dataset_directory / entry["image"]
        definition_path = dataset_directory / entry["scene_task"]
        if not image_path.is_file() or not definition_path.is_file():
            raise FileNotFoundError(f"manifest line {line_number} references a missing dataset file")
        object_id, target_id = _task_ids(json.loads(definition_path.read_text(encoding="utf-8")))
        instruction = str(entry["instruction"])
        if not instruction.strip():
            raise ValueError(f"manifest line {line_number} has an empty instruction")
        examples.append(
            SemanticExample(
                image_path=image_path,
                instruction=instruction,
                object_id=object_id,
                target_id=target_id,
                scene_id=str(entry["sample_id"]),
            )
        )
    if len(examples) < 2:
        raise ValueError("semantic training requires at least two examples")
    return examples


def split_examples(examples: Sequence[SemanticExample], train_fraction: float, seed: int) -> tuple[list[SemanticExample], list[SemanticExample]]:
    """Make deterministic, disjoint train and evaluation splits by scene."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    order = np.random.default_rng(seed).permutation(len(examples))
    train_count = min(max(1, round(len(examples) * train_fraction)), len(examples) - 1)
    train_indices = set(int(index) for index in order[:train_count])
    train = [example for index, example in enumerate(examples) if index in train_indices]
    evaluation = [example for index, example in enumerate(examples) if index not in train_indices]
    return train, evaluation


def _encode_examples(encoder: FrozenCLIPEncoder, examples: Sequence[SemanticExample], batch_size: int) -> SemanticFeatureDataset:
    features: list[Tensor] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        images: list[Image.Image] = []
        for example in batch:
            with Image.open(example.image_path) as image:
                images.append(image.convert("RGB").copy())
        features.append(encoder.encode(images, [example.instruction for example in batch]).cpu())
    return SemanticFeatureDataset(
        torch.cat(features),
        torch.tensor([OBJECT_IDS.index(example.object_id) for example in examples]),
        torch.tensor([TARGET_IDS.index(example.target_id) for example in examples]),
    )


def _loss_and_counts(model: SemanticGroundingNetwork, features: Tensor, object_labels: Tensor, target_labels: Tensor) -> tuple[Tensor, tuple[int, int, int]]:
    object_logits, target_logits = model(features)
    loss = nn.functional.cross_entropy(object_logits, object_labels) + nn.functional.cross_entropy(target_logits, target_labels)
    object_correct = object_logits.argmax(dim=-1).eq(object_labels)
    target_correct = target_logits.argmax(dim=-1).eq(target_labels)
    return loss, (int(object_correct.sum()), int(target_correct.sum()), int((object_correct & target_correct).sum()))


def train_semantic_heads(
    model: SemanticGroundingNetwork,
    train_dataset: SemanticFeatureDataset,
    evaluation_dataset: SemanticFeatureDataset,
    config: SemanticTrainingConfig,
) -> dict[str, float]:
    """Train the small grounding network and return held-out accuracies."""
    device = choose_device(config.device)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    for _ in range(config.epochs):
        model.train()
        for features, object_labels, target_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _loss_and_counts(model, features.to(device), object_labels.to(device), target_labels.to(device))
            loss.backward()
            optimizer.step()

    model.eval()
    total = object_correct = target_correct = joint_correct = 0
    with torch.inference_mode():
        loader = DataLoader(evaluation_dataset, batch_size=config.batch_size, shuffle=False)
        for features, object_labels, target_labels in loader:
            _, counts = _loss_and_counts(model, features.to(device), object_labels.to(device), target_labels.to(device))
            count = features.shape[0]
            total += count
            object_correct += counts[0]
            target_correct += counts[1]
            joint_correct += counts[2]
    return {
        "object_accuracy": object_correct / total,
        "target_accuracy": target_correct / total,
        "joint_accuracy": joint_correct / total,
        "evaluation_examples": float(total),
    }


def train_semantic_baseline(
    dataset_directory: Path,
    checkpoint_path: Path,
    config: SemanticTrainingConfig,
) -> dict[str, float]:
    """Train frozen-CLIP semantic grounding heads from generated RGB scenes."""
    set_seed(config.seed)
    examples = load_semantic_examples(dataset_directory)
    train_examples, evaluation_examples = split_examples(examples, config.train_fraction, config.seed)
    device = choose_device(config.device)
    encoder = FrozenCLIPEncoder(config.backbone, config.pretrained, device)
    train_dataset = _encode_examples(encoder, train_examples, config.batch_size)
    evaluation_dataset = _encode_examples(encoder, evaluation_examples, config.batch_size)
    model = SemanticGroundingNetwork(train_dataset.features.shape[1])
    metrics = train_semantic_heads(model, train_dataset, evaluation_dataset, config)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "feature_dimension": model.feature_dimension,
            "hidden_dimension": model.hidden_dimension,
            "config": asdict(config),
            "metrics": metrics,
            "object_ids": OBJECT_IDS,
            "target_ids": TARGET_IDS,
        },
        checkpoint_path,
    )
    return metrics


def load_semantic_policy(checkpoint_path: Path, device_name: str = "auto") -> SemanticVLAPolicy:
    """Load a trained grounding model and its frozen pretrained encoders."""
    device = choose_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if tuple(checkpoint["object_ids"]) != OBJECT_IDS or tuple(checkpoint["target_ids"]) != TARGET_IDS:
        raise ValueError("checkpoint uses unsupported cube or target IDs")
    configuration = checkpoint["config"]
    encoder = FrozenCLIPEncoder(configuration["backbone"], configuration["pretrained"], device)
    model = SemanticGroundingNetwork(checkpoint["feature_dimension"], checkpoint["hidden_dimension"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return SemanticVLAPolicy(encoder, model)


def execute_prediction(controller: Any, prediction: SemanticPrediction, scene_id: str) -> dict[str, Any]:
    """Run the scripted controller for the predicted cube/bin pair."""
    return controller.scripted_pick_and_place(
        prediction.object_id,
        prediction.target_id,
        scene_id=scene_id,
        task_id=f"place_{prediction.object_id}_in_{prediction.target_id}",
    )


def run_semantic_episode(
    checkpoint_path: Path,
    seed: int,
    instruction: str,
    device: str = "auto",
    show_viewer: bool = False,
) -> dict[str, Any]:
    """Predict a task pair from the current MuJoCo camera image and execute it."""
    from mujoco import viewer

    from vla.simulation.cartesian_controller import CartesianPandaController
    from vla.simulation.mujoco_tabletop import TabletopEnvironment

    environment = TabletopEnvironment()
    environment.reset(seed=seed)
    prediction = load_semantic_policy(checkpoint_path, device).predict(environment.get_rgb(), instruction)
    controller = CartesianPandaController(environment)
    if show_viewer:
        print("opening the MuJoCo viewer; the semantic episode will begin immediately")
        with viewer.launch_passive(environment.model, environment.data) as live_viewer:
            controller.attach_viewer(live_viewer)
            result = execute_prediction(controller, prediction, scene_id=f"tabletop_{seed:05d}")
            live_viewer.sync()
            print("episode complete; close the viewer window to return to the terminal")
            while live_viewer.is_running():
                live_viewer.sync()
                time.sleep(0.01)
    else:
        result = execute_prediction(controller, prediction, scene_id=f"tabletop_{seed:05d}")
    return {"prediction": asdict(prediction), "result": result}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="train object/bin grounding heads")
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--checkpoint", type=Path, default=Path("checkpoints/semantic_vla.pt"))
    train.add_argument("--epochs", type=int, default=SemanticTrainingConfig.epochs)
    train.add_argument("--batch-size", type=int, default=SemanticTrainingConfig.batch_size)
    train.add_argument("--learning-rate", type=float, default=SemanticTrainingConfig.learning_rate)
    train.add_argument("--seed", type=int, default=SemanticTrainingConfig.seed)
    train.add_argument("--device", default="auto")
    run = commands.add_parser("run", help="predict and execute one MuJoCo task")
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--instruction", required=True)
    run.add_argument("--device", default="auto")
    run.add_argument("--output", type=Path, default=Path("results/semantic_episode.json"))
    run.add_argument("--viewer", action="store_true", help="show the complete episode in the MuJoCo viewer")
    return parser.parse_args(argv)


def _restart_with_mjpython_if_required(command_arguments: Sequence[str]) -> None:
    """Restart the viewer process with MuJoCo's macOS Python launcher."""
    if sys.platform != "darwin" or "MJPYTHON_BIN" in os.environ:
        return
    launcher = Path(sys.executable).with_name("mjpython")
    if not launcher.is_file():
        raise RuntimeError(f"MuJoCo viewer launcher not found: {launcher}")
    print("macOS live viewer: restarting with mjpython")
    os.execv(str(launcher), [str(launcher), "-m", "vla.semantic.baseline", *command_arguments])


def main(argv: Sequence[str] | None = None) -> int:
    command_arguments = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    if args.command == "train":
        metrics = train_semantic_baseline(
            args.dataset,
            args.checkpoint,
            SemanticTrainingConfig(
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                device=args.device,
            ),
        )
        print(json.dumps(metrics, indent=2))
        return 0
    if args.viewer:
        _restart_with_mjpython_if_required(command_arguments)
    episode = run_semantic_episode(args.checkpoint, args.seed, args.instruction, args.device, args.viewer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(episode, indent=2) + "\n", encoding="utf-8")
    print(f"selected {episode['prediction']['object_id']} -> {episode['prediction']['target_id']}")
    print(f"success: {episode['result']['success']['success']}; episode log: {args.output}")
    return 0 if episode["result"]["success"]["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
