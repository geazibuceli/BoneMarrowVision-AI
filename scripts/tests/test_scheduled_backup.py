import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("scheduled_backup", SCRIPTS / "scheduled_backup.py")
backup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backup)


def snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(backup, "ROOT", tmp_path)
    directory = tmp_path / "models/backups/snapshot"
    directory.mkdir(parents=True)
    for name in ("database.dump", "uploads.tar"):
        (directory / name).write_bytes(b"fixture")
    evidence = {"status": "verified", "registered_artifacts": [],
                "database_dump_sha256": backup.digest(directory / "database.dump"),
                "uploads_archive_sha256": backup.digest(directory / "uploads.tar")}
    (directory / "verification.json").write_text(json.dumps(evidence))
    return directory


def test_backup_rejects_changed_verified_snapshot(tmp_path, monkeypatch):
    directory = snapshot(tmp_path, monkeypatch)
    assert backup.verified_snapshot(directory) == directory
    (directory / "database.dump").write_bytes(b"modified")
    with pytest.raises(ValueError, match="changed after verification"):
        backup.verified_snapshot(directory)


@pytest.mark.parametrize("failure", [None, "upload", "check"])
def test_remote_evidence_requires_successful_upload_and_check(tmp_path, monkeypatch, failure):
    directory = snapshot(tmp_path, monkeypatch)
    password = tmp_path / "password.txt"
    password.write_text("test-only-password")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"repository": "sftp:example:/backup", "password_file": str(password)}))
    monkeypatch.setattr(backup.sys, "argv", ["backup", "--config", str(config), "--snapshot", str(directory)])
    monkeypatch.setattr(backup.shutil, "which", lambda _: "restic")
    calls = []

    def invoke(arguments, environment):
        calls.append(arguments)
        assert "test-only-password" not in str(arguments)
        assert environment["RESTIC_PASSWORD_FILE"] == str(password)
        if (failure == "upload" and arguments[1] == "backup") or (failure == "check" and arguments[1] == "check"):
            raise RuntimeError("Remote operation failed")
        return json.dumps({"message_type": "summary", "snapshot_id": "abc123"})

    monkeypatch.setattr(backup, "invoke", invoke)
    if failure:
        with pytest.raises(RuntimeError):
            backup.main()
        assert not (directory / "offsite.json").exists()
    else:
        backup.main()
        evidence = json.loads((directory / "offsite.json").read_text())
        assert evidence["snapshot_id"] == "abc123"
        assert [call[1] for call in calls] == ["snapshots", "backup", "check"]


def test_incomplete_restic_backup_is_not_success(monkeypatch):
    monkeypatch.setattr(backup.subprocess, "run", Mock(return_value=Mock(returncode=3)))
    with pytest.raises(RuntimeError, match="exit code 3"):
        backup.invoke(["restic", "backup"], {})
