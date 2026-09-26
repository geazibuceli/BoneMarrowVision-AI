# Local backup and recovery verification

Run from the project root while the local Compose database, backend, and frontend
are running:

```powershell
.venv/Scripts/python.exe scripts/verify_backup_restore.py --output models/backups/my-new-backup
```

Choose a new output directory for every run. The command briefly stops frontend
and backend to prevent application writes while collecting a PostgreSQL custom
dump, upload-volume archive, and copies of every registered checkpoint and
configuration. It restarts the same application containers even if backup fails.
Do not run other database writers during this snapshot window.

The command restores the dump into a disposable PostgreSQL 16 container with no
network access or published ports. It compares counts and content digests for
users, model registry, predictions, experiments, audit logs, and migration
version. It extracts media into a separate local directory and verifies each
file's SHA-256, and verifies copied model/configuration files against their
sources. It removes only the disposable database container. The application
database and volumes are never restoration targets.

`verification.json` records the outcome, snapshot digests, artifact hashes, and
active model IDs. Only `status: verified` establishes a successful exercise.
Keep the original files and this evidence together. These private backups
contain account data and potentially sensitive media; they remain under ignored
`models/backups/` and must not be committed. Do not publish their contents.

Secrets from `.env` are deliberately excluded. Preserve those through a separate
secure process. This is a local recovery test, not an off-host backup service,
scheduled retention policy, or automatic disaster failover. Store approved
encrypted copies off the workstation if recovery from disk loss is required.

## September 25 verification

The follow-up project update records its verified backup path and counts in
[the project review](project-review-2026-09-25.md). Deployment uses a rebuild of
backend/frontend only and retains PostgreSQL and upload volumes. Never use
`docker compose down -v` as part of this process.
