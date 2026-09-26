"""Persist completed stages and verify their artifacts before reuse."""

import json
from pathlib import Path

from bone_marrow_vision.data.dataset import sha256_file


class StageJournal:
    def __init__(self, output: Path, inputs: dict, *, resume: bool):
        self.path = output / "stages.json"
        if resume:
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
            if self.state["inputs"] != inputs:
                raise ValueError("Experiment inputs changed; use a new output directory")
        else:
            if self.path.exists():
                raise ValueError("Experiment already exists; use --resume")
            self.state = {"inputs": inputs, "completed": {}}
            self.save()

    def save(self):
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def completed(self, name: str) -> bool:
        artifacts = self.state["completed"].get(name)
        if artifacts is None:
            return False
        for value, expected in artifacts.items():
            path = Path(value)
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"Completed stage artifact changed or is missing: {name}")
        return True

    def finish(self, name: str, artifacts: list[Path]):
        self.state["completed"][name] = {str(path.resolve()): sha256_file(path) for path in artifacts}
        self.save()
