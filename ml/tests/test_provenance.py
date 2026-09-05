import json

from bone_marrow_vision.data.dataset import sha256_file
from bone_marrow_vision.paths import portable_path
from bone_marrow_vision.provenance import build_provenance, canonical_json_sha256


def test_provenance_hashes_inputs_without_serializing_personal_absolute_paths(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    config = {"seed": 42, "architecture": "baseline_cnn"}
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 42\n", encoding="utf-8")

    provenance = build_provenance(
        config,
        input_files={"config": config_path},
        working_directory=tmp_path,
    )

    assert provenance["config_sha256"] == canonical_json_sha256(config)
    assert provenance["input_files"]["config"] == {
        "path": "config.yaml",
        "sha256": sha256_file(config_path),
    }
    assert str(tmp_path) not in json.dumps(provenance)


def test_external_display_paths_are_reduced_to_the_filename(tmp_path, monkeypatch):
    working = tmp_path / "working"
    working.mkdir()
    external = tmp_path / "external" / "input.json"
    monkeypatch.chdir(working)

    assert portable_path(external) == "input.json"
