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

## Encrypted off-host backups and scheduling

`scripts/scheduled_backup.py` creates the same verified local snapshot, uploads it
with Restic, and checks the destination repository. It aborts before stopping the
application if the repository cannot be accessed. Only snapshots with successful
restore evidence and matching hashes are uploaded. Overlapping scheduled runs
are rejected by an operating-system file lock.

Install [Restic](https://restic.readthedocs.io/en/stable/020_installation.html),
choose an off-host repository (for example, SFTP or S3), and initialize it with
`restic init` using the provider's credentials. Keep the repository password in a
restricted file outside the repository and retain a separate secure copy: losing
it prevents recovery. See the official [scripting reference](https://github.com/restic/restic/blob/master/doc/075_scripting.rst)
for credential environment variables and exit codes.

Create a private `models/backups/backup-config.json`:

```json
{
  "repository": "sftp:backup-user@backup.example.org:/backups/bonemarrowvision",
  "password_file": "C:/private/bmv-restic-password.txt"
}
```

The example host is a placeholder. Configure SSH keys and known-host verification
or the selected provider's credentials in the environment of the scheduled user.
The wrapper does not create an account, initialize a repository, or embed secrets
in command arguments. Do not use a same-disk repository as workstation-loss protection.

```powershell
.venv/Scripts/python.exe scripts/scheduled_backup.py --config models/backups/backup-config.json
# Retry upload of an already verified snapshot without stopping the app again:
.venv/Scripts/python.exe scripts/scheduled_backup.py --config models/backups/backup-config.json --snapshot models/backups/my-new-backup
# Register a daily task after one successful manual run:
./scripts/install_backup_schedule.ps1 -Config models/backups/backup-config.json -At '03:00'
```

The Windows task runs without a visible console, under the current user's
interactive session, without elevated privileges. The user must be logged in and
Docker must be available. Task Scheduler records exit status; windowless runs
write sanitized messages to ignored `models/backups/scheduler.log`. On Linux,
schedule the same Python command through the host's scheduler with an explicit
working directory and credential environment.

`offsite.json` records the Restic snapshot ID only after upload and a repository
check succeed. By default, the check reads a random 5% data sample; pass
`--full-check` periodically to read all stored data. See Restic's
[integrity checks](https://github.com/restic/restic/blob/master/doc/045_working_with_repos.rst).
Restore that exact snapshot with `restic restore SNAPSHOT_ID --target NEW_DIRECTORY`
and exercise database/media recovery in isolation before relying on disaster recovery.

Local snapshots remain unencrypted and private under `models/backups/`; the
Restic destination is encrypted. No local or remote retention deletions are
automatic. Monitor disk capacity and define retention explicitly before enabling
a long-running schedule. Back up application secrets separately. No external
destination or scheduled task is enabled by checking these scripts into Git.
