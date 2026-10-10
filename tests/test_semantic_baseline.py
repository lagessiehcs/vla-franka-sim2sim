import json

import numpy as np
import torch
import pytest

from vla.semantic import baseline

from vla.semantic.baseline import (
    OBJECT_IDS,
    TARGET_IDS,
    SemanticExample,
    SemanticFeatureDataset,
    SemanticGroundingNetwork,
    SemanticPrediction,
    SemanticTrainingConfig,
    SemanticVLAPolicy,
    execute_prediction,
    load_semantic_examples,
    split_examples,
    train_semantic_heads,
)


def _write_dataset_entry(directory, sample_id: str, object_id: str, target_id: str) -> dict[str, str]:
    images = directory / "images"
    tasks = directory / "scene_tasks"
    images.mkdir(exist_ok=True)
    tasks.mkdir(exist_ok=True)
    image = images / f"{sample_id}.png"
    image.write_bytes(b"placeholder")
    task = {
        "task": {
            "steps": [
                {"action": "pick", "object_id": object_id},
                {"action": "place", "object_id": object_id, "target_id": target_id, "relation": "in"},
            ]
        }
    }
    task_path = tasks / f"{sample_id}.json"
    task_path.write_text(json.dumps(task), encoding="utf-8")
    return {
        "sample_id": sample_id,
        "image": image.relative_to(directory).as_posix(),
        "scene_task": task_path.relative_to(directory).as_posix(),
        "instruction": f"Put the {object_id} in the {target_id}.",
    }


def test_load_semantic_examples_and_split_by_scene(tmp_path) -> None:
    entries = [
        _write_dataset_entry(tmp_path, "scene_0", "red_cube_1", "yellow_bin_1"),
        _write_dataset_entry(tmp_path, "scene_1", "green_cube_1", "purple_bin_1"),
        _write_dataset_entry(tmp_path, "scene_2", "blue_cube_1", "yellow_bin_1"),
    ]
    (tmp_path / "manifest.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")

    examples = load_semantic_examples(tmp_path)
    train, evaluation = split_examples(examples, train_fraction=2 / 3, seed=4)

    assert [(example.object_id, example.target_id) for example in examples] == [
        ("red_cube_1", "yellow_bin_1"),
        ("green_cube_1", "purple_bin_1"),
        ("blue_cube_1", "yellow_bin_1"),
    ]
    assert {example.scene_id for example in train}.isdisjoint(example.scene_id for example in evaluation)


def test_grounding_network_has_cube_and_target_outputs() -> None:
    model = SemanticGroundingNetwork(feature_dimension=8, hidden_dimension=12)

    object_logits, target_logits = model(torch.zeros(4, 8))

    assert object_logits.shape == (4, len(OBJECT_IDS))
    assert target_logits.shape == (4, len(TARGET_IDS))


def test_train_semantic_heads_returns_held_out_metrics() -> None:
    features = torch.eye(8)
    object_labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    target_labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    train = SemanticFeatureDataset(features[:6], object_labels[:6], target_labels[:6])
    evaluation = SemanticFeatureDataset(features[6:], object_labels[6:], target_labels[6:])
    metrics = train_semantic_heads(
        SemanticGroundingNetwork(feature_dimension=8, hidden_dimension=16),
        train,
        evaluation,
        SemanticTrainingConfig(epochs=2, batch_size=3, device="cpu"),
    )

    assert set(metrics) == {"object_accuracy", "target_accuracy", "joint_accuracy", "evaluation_examples"}
    assert metrics["evaluation_examples"] == 2.0
    assert all(0.0 <= metrics[name] <= 1.0 for name in ("object_accuracy", "target_accuracy", "joint_accuracy"))


def test_policy_decodes_predictions_and_controller_executes_them() -> None:
    class FakeEncoder:
        device = torch.device("cpu")

        def encode(self, images, instructions):
            assert len(images) == len(instructions) == 1
            return torch.zeros(1, 4)

    model = SemanticGroundingNetwork(feature_dimension=4, hidden_dimension=4)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.object_head.bias[1] = 3.0
        model.target_head.bias[0] = 3.0
    prediction = SemanticVLAPolicy(FakeEncoder(), model).predict(
        np.zeros((8, 8, 3), dtype=np.uint8), "Put the green cube in the yellow bin."
    )

    class FakeController:
        def scripted_pick_and_place(self, object_id, target_id, scene_id, task_id):
            return {"object_id": object_id, "target_id": target_id, "scene_id": scene_id, "task_id": task_id}

    result = execute_prediction(FakeController(), prediction, scene_id="scene_7")

    assert prediction.object_id == "green_cube_1"
    assert prediction.target_id == "yellow_bin_1"
    assert result["task_id"] == "place_green_cube_1_in_yellow_bin_1"


def test_semantic_prediction_runs_the_real_scripted_controller() -> None:
    pytest.importorskip("mujoco")
    from vla.simulation.cartesian_controller import CartesianPandaController
    from vla.simulation.mujoco_tabletop import PANDA_XML, TabletopEnvironment

    if not PANDA_XML.is_file():
        pytest.skip("MuJoCo Menagerie submodule is not initialized")
    environment = TabletopEnvironment()
    environment.reset(seed=0)
    prediction = SemanticPrediction("red_cube_1", "yellow_bin_1", 1.0, 1.0)

    result = execute_prediction(CartesianPandaController(environment), prediction, scene_id="tabletop_00000")

    assert result["success"]["success"] is True


def test_macos_viewer_restarts_with_the_matching_mjpython(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    python = tmp_path / "python"
    mjpython = tmp_path / "mjpython"
    python.touch()
    mjpython.touch()
    command: dict[str, object] = {}
    monkeypatch.setattr(baseline.sys, "platform", "darwin")
    monkeypatch.setattr(baseline.sys, "executable", str(python))
    monkeypatch.delenv("MJPYTHON_BIN", raising=False)
    monkeypatch.setattr(baseline.os, "execv", lambda path, arguments: command.update(path=path, arguments=arguments))

    baseline._restart_with_mjpython_if_required(["run", "--viewer"])

    assert command == {
        "path": str(mjpython),
        "arguments": [str(mjpython), "-m", "vla.semantic.baseline", "run", "--viewer"],
    }
