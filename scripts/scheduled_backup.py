"""Create a verified Compose snapshot and send it to an encrypted Restic repository."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from verify_backup_restore import ROOT, digest


@contextmanager
def exclusive_backup():
    root = ROOT / "models/backups"
    root.mkdir(parents=True, exist_ok=True)
    with (root / "scheduler.lock").open("a+b") as stream:
        stream.write(b"0")
        stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def verified_snapshot(path: Path):
    path = path.resolve()
    if not path.is_relative_to((ROOT / "models/backups").resolve()):
        raise ValueError("Snapshot must be under models/backups")
    evidence = json.loads((path / "verification.json").read_text(encoding="utf-8"))
    if evidence.get("status") != "verified":
        raise ValueError("Snapshot recovery has not been verified")
    for name, key in (("database.dump", "database_dump_sha256"), ("uploads.tar", "uploads_archive_sha256")):
        if digest(path / name) != evidence[key]:
            raise ValueError("Snapshot changed after verification")
    for artifact in evidence["registered_artifacts"]:
        candidate = (path / "files" / artifact["path"]).resolve()
        if not candidate.is_relative_to(path / "files") or digest(candidate) != artifact["sha256"]:
            raise ValueError("Registered artifact changed after verification")
    return path


def invoke(arguments, environment):
    result = subprocess.run(arguments, cwd=ROOT, env=environment, capture_output=True,
                            text=True, timeout=24 * 3600,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if result.returncode:
        # Do not print repository URLs, provider responses, passwords, or private paths.
        raise RuntimeError(f"Backup command failed with exit code {result.returncode}")
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Private JSON with repository and password_file")
    parser.add_argument("--snapshot", type=Path, help="Upload an existing verified snapshot instead of creating one")
    parser.add_argument("--full-check", action="store_true", help="Read all repository data after backup")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not config.get("repository") or not Path(config.get("password_file", "")).is_file():
        raise ValueError("Configure repository and an existing password_file")
    executable = shutil.which("restic")
    if not executable:
        raise RuntimeError("Install Restic and add it to PATH before scheduling backups")
    environment = {**os.environ, "RESTIC_REPOSITORY": config["repository"],
                   "RESTIC_PASSWORD_FILE": str(Path(config["password_file"]).resolve())}
    with exclusive_backup():
        # Fail before stopping the app if repository access is unavailable.
        invoke([executable, "snapshots", "--json"], environment)
        snapshot = args.snapshot
        if snapshot is None:
            name = datetime.now(UTC).strftime("scheduled-%Y%m%dT%H%M%SZ-") + uuid4().hex[:8]
            snapshot = ROOT / "models/backups" / name
            invoke([sys.executable, str(ROOT / "scripts/verify_backup_restore.py"),
                    "--output", str(snapshot)], environment)
        snapshot = verified_snapshot(snapshot)
        output = invoke([executable, "backup", str(snapshot), "--tag", "bone-marrow-vision", "--json"], environment)
        summaries = [entry for line in output.splitlines() if line.strip()
                     and (entry := json.loads(line)).get("message_type") == "summary"]
        if not summaries or not summaries[-1].get("snapshot_id"):
            raise RuntimeError("Restic did not confirm a complete snapshot")
        invoke([executable, "check", "--read-data" if args.full_check else "--read-data-subset=5%"], environment)
        evidence = {"status": "uploaded_and_checked", "snapshot_id": summaries[-1]["snapshot_id"],
                    "verified_at_utc": datetime.now(UTC).isoformat(),
                    "check": "all_data" if args.full_check else "5_percent_sample"}
        (snapshot / "offsite.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        print("Encrypted backup uploaded; repository check passed.")


if __name__ == "__main__":
    if sys.stdout is None or sys.stderr is None:
        log_directory = ROOT / "models/backups"
        log_directory.mkdir(parents=True, exist_ok=True)
        sys.stdout = sys.stderr = (log_directory / "scheduler.log").open("a", encoding="utf-8", buffering=1)
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Backup failed ({type(error).__name__}). Check configuration and service availability.", file=sys.stderr)
        sys.exit(1)
