from pathlib import Path

from vla.training.supervised_sanity import (
    ACTION_DIM,
    FEATURE_DIM,
    TrainingConfig,
    make_synthetic_dataset,
    run_training,
)


def test_dataset_has_expected_tensor_shapes() -> None:
    dataset = make_synthetic_dataset(num_samples=11, seed=3)
    features, action = dataset[0]
    assert len(dataset) == 11
    assert features.shape == (FEATURE_DIM,)
    assert action.shape == (ACTION_DIM,)


def test_training_writes_resumable_checkpoints(tmp_path: Path) -> None:
    config = TrainingConfig(
        epochs=4,
        batch_size=32,
        train_samples=256,
        validation_samples=64,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        results_dir=str(tmp_path / "results"),
        device="cpu",
    )
    metrics = run_training(config)
    assert metrics["best_validation_loss"] < 0.10
    assert (tmp_path / "checkpoints" / "best.pt").is_file()
    assert (tmp_path / "checkpoints" / "last.pt").is_file()
    assert (tmp_path / "results" / "supervised_sanity_metrics.json").is_file()
