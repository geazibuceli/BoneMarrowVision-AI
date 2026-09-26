import pytest

from bone_marrow_vision.training.orchestration import StageJournal


def test_resume_reuses_only_completed_unchanged_stages(tmp_path):
    artifact = tmp_path / "report.json"
    artifact.write_text("{}")
    journal = StageJournal(tmp_path, {"seed": 42}, resume=False)
    journal.finish("evaluate", [artifact])
    resumed = StageJournal(tmp_path, {"seed": 42}, resume=True)
    assert resumed.completed("evaluate")
    assert not resumed.completed("analyze")
    with pytest.raises(ValueError, match="inputs changed"):
        StageJournal(tmp_path, {"seed": 43}, resume=True)
    artifact.write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        resumed.completed("evaluate")


@pytest.mark.parametrize("failure_stage", ["train", "evaluate"])
def test_architecture_runner_resumes_after_failure(tmp_path, monkeypatch, failure_stage):
    import importlib.util
    import json
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("architecture_runner", scripts / "run_architecture_comparison.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    configuration = tmp_path / "ml/configs/resnet50.yaml"
    configuration.parent.mkdir(parents=True)
    configuration.write_text("architecture: resnet50\nepochs: 2\n")
    data = tmp_path / "data"
    data.mkdir()
    for name in ("train", "validation", "test"):
        (data / f"{name}.json").write_text(json.dumps([{"path": "fixture.png", "label": "A"}]))
    (data / "manifest.json").write_text("{}")
    monkeypatch.setattr(runner, "verify_split_manifest", lambda _: None)
    monkeypatch.setattr(runner, "audit_splits", lambda _: {"cross_split_duplicates": [], "conflicting_labels": []})
    output = tmp_path / "output"
    arguments = ["runner", "--architectures", "resnet50", "--full", "--data-dir", str(data), "--output-dir", str(output)]
    monkeypatch.setattr(sys, "argv", arguments)
    failed = False
    commands = []

    def run(command, **_kwargs):
        nonlocal failed
        commands.append(command)
        script = Path(command[2]).stem
        if script == "train":
            best = Path(command[command.index("--checkpoint") + 1])
            best.parent.mkdir(parents=True, exist_ok=True)
            best.with_name("best.latest.pt").write_bytes(b"recovery")
            if best.parent.name == "resnet50" and failure_stage == "train" and not failed:
                failed = True
                return SimpleNamespace(returncode=1)
            best.write_bytes(b"best")
            best.with_suffix(".history.json").write_text("[]")
        elif script == "evaluate":
            if failure_stage == "evaluate" and not failed:
                failed = True
                return SimpleNamespace(returncode=1)
            report = {key: 0.5 for key in ("accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")}
            report.update(class_names=["A"], provenance={"test_split": {"sha256": "fixture"}})
            Path(command[command.index("--output") + 1]).write_text(json.dumps(report))
        else:
            destination = Path(command[command.index("--output-dir") + 1])
            destination.mkdir()
            (destination / "analysis_summary.json").write_text("{}")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="failed"):
        runner.main()
    commands.clear()
    monkeypatch.setattr(sys, "argv", [*arguments, "--resume"])
    runner.main()
    assert json.loads((output / "status.json").read_text())["status"] == "completed"
    assert (output / "comparison.md").is_file()
    assert not any("smoke_resnet50" in " ".join(command) for command in commands)
    if failure_stage == "train":
        assert "--resume" in commands[0] and "--finalize-completed" in commands[0]
    else:
        assert not any(Path(command[2]).stem == "train" for command in commands)
