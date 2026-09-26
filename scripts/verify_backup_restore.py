"""Back up the local Compose application and prove recovery in isolated storage.

Briefly stops frontend/backend for a consistent database/media snapshot, then
restarts those same containers. Never restores into the application's database.
Backups include account data and private media and must remain outside Git.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
TABLES = ("users", "ml_models", "predictions", "experiments", "audit_logs", "alembic_version")


def command(*args, **kwargs):
    return subprocess.run(list(args), cwd=ROOT, check=True, timeout=300, **kwargs)


def output(*args):
    return command(*args, capture_output=True, text=True).stdout.strip()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def database_snapshot(container):
    result = {}
    for table in TABLES:
        sql = (
            "SELECT json_build_object('count', count(*), 'digest', "
            "md5(coalesce(string_agg(row_to_json(t)::text, E'\\n' "
            f"ORDER BY row_to_json(t)::text), ''))) FROM {table} t"
        )
        result[table] = json.loads(output(
            "docker", "exec", container, "psql", "-U", "app", "-d", "bone_marrow_vision",
            "-At", "-c", sql,
        ))
    return result


def project_file(value):
    path = Path(value)
    if str(value).startswith("/app/"):
        path = ROOT / str(value).removeprefix("/app/")
    elif not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError("A registered artifact is missing or outside the project; backup aborted")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory under models/backups")
    args = parser.parse_args()
    backup = args.output.resolve()
    if not backup.is_relative_to(ROOT / "models/backups"):
        parser.error("Backup output must be inside the ignored models/backups directory")
    database = output("docker", "compose", "ps", "-q", "db")
    backend = output("docker", "compose", "ps", "-q", "backend")
    frontend = output("docker", "compose", "ps", "-q", "frontend")
    if not all((database, backend, frontend)):
        raise RuntimeError("All application containers must be running before backup")
    backup.mkdir(parents=True, exist_ok=False)
    restore_container = f"bmv-restore-{uuid4().hex[:12]}"
    started_restore = False
    evidence = {"created_at_utc": datetime.now(UTC).isoformat(), "status": "started"}
    try:
        # Stop writers before pg_dump and media snapshot; retain the database and volumes.
        try:
            command("docker", "compose", "stop", "frontend", "backend", capture_output=True)
            before = database_snapshot(database)
            sql = "SELECT coalesce(json_agg(json_build_object('id', id, 'checkpoint_path', checkpoint_path, 'config_path', config_path, 'is_active', is_active)), '[]') FROM ml_models"
            models = json.loads(output("docker", "exec", database, "psql", "-U", "app",
                                       "-d", "bone_marrow_vision", "-At", "-c", sql))
            with (backup / "database.dump").open("xb") as stream:
                command("docker", "exec", database, "pg_dump", "-U", "app", "-d",
                        "bone_marrow_vision", "-Fc", stdout=stream, stderr=subprocess.PIPE)
            # The backend is stopped; use its volumes read-only in a disposable helper.
            image = output("docker", "inspect", "--format", "{{.Config.Image}}", backend)
            with (backup / "uploads.tar").open("xb") as stream:
                command("docker", "run", "--rm", "--network", "none", "--volumes-from",
                        f"{backend}:ro", "--entrypoint", "tar", image,
                        "-C", "/app/data/uploads", "-cf", "-", ".", stdout=stream, stderr=subprocess.PIPE)
            artifacts = []
            for model in models:
                for key in ("checkpoint_path", "config_path"):
                    source = project_file(model[key])
                    relative = source.relative_to(ROOT)
                    target = backup / "files" / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        shutil.copy2(source, target)
                    if digest(source) != digest(target):
                        raise RuntimeError("Artifact backup hash mismatch")
                    artifacts.append({"path": relative.as_posix(), "sha256": digest(target)})
            evidence.update(database_snapshot=before, registered_artifacts=artifacts,
                            active_model_ids=[m["id"] for m in models if m["is_active"]],
                            database_dump_sha256=digest(backup / "database.dump"),
                            uploads_archive_sha256=digest(backup / "uploads.tar"))
        finally:
            command("docker", "compose", "start", "backend", "frontend", capture_output=True)
        command("docker", "run", "--rm", "-d", "--name", restore_container,
                "--network", "none", "--tmpfs", "/var/lib/postgresql/data",
                "-e", "POSTGRES_DB=bone_marrow_vision", "-e", "POSTGRES_USER=app",
                "-e", "POSTGRES_HOST_AUTH_METHOD=trust", "postgres:16-alpine", capture_output=True)
        started_restore = True
        for _ in range(30):
            ready = subprocess.run(["docker", "exec", restore_container, "pg_isready", "-U", "app"],
                                   capture_output=True, timeout=10)
            if ready.returncode == 0:
                break
            time.sleep(1)
        else:
            raise RuntimeError("Isolated restore database did not become ready")
        with (backup / "database.dump").open("rb") as stream:
            command("docker", "exec", "-i", restore_container, "pg_restore", "--exit-on-error",
                    "-U", "app", "-d", "bone_marrow_vision", stdin=stream, capture_output=True)
        if database_snapshot(restore_container) != before:
            raise RuntimeError("Restored database contents differ from the snapshot")
        restored = backup / "restored_uploads"
        restored.mkdir()
        media_files = 0
        with tarfile.open(backup / "uploads.tar") as archive:
            members = archive.getmembers()
            if any(not (m.isfile() or m.isdir()) for m in members):
                raise ValueError("Uploads archive contains unsupported links or special files")
            archive.extractall(restored, filter="data")
            for member in members:
                if member.isfile():
                    with archive.extractfile(member) as stream:
                        expected = hashlib.file_digest(stream, "sha256").hexdigest()
                    if digest(restored / member.name) != expected:
                        raise RuntimeError("Restored media hash mismatch")
                    media_files += 1
        evidence.update(status="verified", restored_media_files=media_files,
                        verified_at_utc=datetime.now(UTC).isoformat())
        print(f"Backup recovery verified: {len(TABLES)} tables, {media_files} media files, {len(artifacts)} registered artifact references.")
        print(f"Private backup: {backup.relative_to(ROOT).as_posix()}")
    finally:
        if started_restore:
            command("docker", "stop", restore_container, capture_output=True)
        (backup / "verification.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
