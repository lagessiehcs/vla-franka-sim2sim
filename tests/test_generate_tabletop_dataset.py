import json

import pytest

mujoco = pytest.importorskip("mujoco")

from vla.instructions.generator import validate_scene_task
from vla.simulation.generate_tabletop_dataset import generate_dataset
from vla.simulation.mujoco_tabletop import PANDA_XML


def test_dataset_generation_writes_valid_rgb_task_pairs(tmp_path) -> None:
    if not PANDA_XML.is_file():
        pytest.skip("MuJoCo Menagerie submodule is not initialized")

    output_directory = tmp_path / "tabletop_dataset"
    manifest = generate_dataset(output_directory, count=2, seed=7, width=80, height=60, camera_jitter=0.01)

    assert len(manifest) == 2
    assert manifest[0]["camera"]["position"] != manifest[1]["camera"]["position"]
    manifest_lines = (output_directory / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(manifest_lines) == 2
    for line, entry in zip(manifest_lines, manifest):
        assert json.loads(line) == entry
        assert (output_directory / entry["image"]).is_file()
        definition = json.loads((output_directory / entry["scene_task"]).read_text(encoding="utf-8"))
        assert validate_scene_task(definition) == []
        assert definition["scene_id"] == entry["sample_id"]
        assert "Goal:" in entry["instruction"]


def test_dataset_generation_refuses_to_mix_runs(tmp_path) -> None:
    output_directory = tmp_path / "existing_dataset"
    output_directory.mkdir()
    (output_directory / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        generate_dataset(output_directory, count=1, seed=0, width=80, height=60)
