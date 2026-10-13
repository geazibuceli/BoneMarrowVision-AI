import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("scheduled_backup", SCRIPTS / "scheduled_backup.py")
backup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backup)


def snapshot(tmp_path):
    directory = tmp_path / "models/backups/snapshot"
    directory.mkdir(parents=True)
    for name in ("database.dump", "uploads.tar"):
        (directory / name).write_bytes(b"fixture")
    evidence = {"status": "verified", "registered_artifacts": [],
                "database_dump_sha256": backup.digest(directory / "database.dump"),
                "uploads_archive_sha256": backup.digest(directory / "uploads.tar")}
    (directory / "verification.json").write_text(json.dumps(evidence))
    return directory


class ScheduledBackupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp_path = Path(temporary.name)
        root_patch = patch.object(backup, "ROOT", self.tmp_path)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def test_backup_rejects_changed_verified_snapshot(self):
        directory = snapshot(self.tmp_path)
        self.assertEqual(backup.verified_snapshot(directory), directory)
        (directory / "database.dump").write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "changed after verification"):
            backup.verified_snapshot(directory)


    def test_remote_evidence_requires_successful_upload_and_check(self):
        for failure in (None, "upload", "check"):
            with self.subTest(failure=failure):
                self.check_remote_evidence(failure)

    def check_remote_evidence(self, failure):
        case_path = self.tmp_path / str(failure)
        directory = snapshot(case_path)
        password = case_path / "password.txt"
        password.write_text("test-only-password")
        config = case_path / "config.json"
        config.write_text(json.dumps({"repository": "sftp:example:/backup", "password_file": str(password)}))
        calls = []

        def invoke(arguments, environment):
            calls.append(arguments)
            self.assertNotIn("test-only-password", str(arguments))
            self.assertEqual(environment["RESTIC_PASSWORD_FILE"], str(password))
            if (failure == "upload" and arguments[1] == "backup") or (failure == "check" and arguments[1] == "check"):
                raise RuntimeError("Remote operation failed")
            return json.dumps({"message_type": "summary", "snapshot_id": "abc123"})

        with patch.object(backup, "ROOT", case_path), \
                patch.object(backup.sys, "argv", ["backup", "--config", str(config), "--snapshot", str(directory)]), \
                patch.object(backup.shutil, "which", return_value="restic"), \
                patch.object(backup, "invoke", side_effect=invoke):
            if failure:
                with self.assertRaises(RuntimeError):
                    backup.main()
                self.assertFalse((directory / "offsite.json").exists())
            else:
                backup.main()
                evidence = json.loads((directory / "offsite.json").read_text())
                self.assertEqual(evidence["snapshot_id"], "abc123")
                self.assertEqual([call[1] for call in calls], ["snapshots", "backup", "check"])

    def test_incomplete_restic_backup_is_not_success(self):
        with patch.object(backup.subprocess, "run", return_value=Mock(returncode=3)):
            with self.assertRaisesRegex(RuntimeError, "exit code 3"):
                backup.invoke(["restic", "backup"], {})
