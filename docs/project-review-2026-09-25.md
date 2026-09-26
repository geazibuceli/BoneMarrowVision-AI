# Project review: September 25, 2026

## Scope and confirmed correction

Reviewed the main authentication, upload, prediction/history, model-management,
experiment-viewing, and deployment paths. Executed the existing automated checks
and added coverage for a confirmed architecture integration failure. This is a
local engineering review, not an exhaustive security or accessibility audit.

ConvNeXt-Tiny was supported by the ML builder but rejected by the API registration
schema, response schema, and inference loader; it was also missing from the
administrator's architecture selector. All four boundaries now accept
`convnext_tiny`. Checkpoint architecture matching, digest validation, restricted
paths, administrator permissions, and separate activation remain enforced.

A regression test now creates a ConvNeXt checkpoint without downloading weights,
registers and activates it in an isolated database, uploads a synthetic image,
checks prediction attribution and a decodable Grad-CAM, rejects unauthenticated
media access, and verifies temporary input cleanup. Browser tests also verify
that administrators can select the architecture.

## Executed verification

| Check | Result |
| --- | --- |
| Backend on isolated SQLite | 55 tests passed |
| ML | 34 tests passed |
| Dependency audit script tests | 5 tests passed |
| Headless Edge browser integration | 6 tests passed, desktop and 360 px mobile |
| Frontend API tests | 9 tests passed |
| Backend on disposable PostgreSQL 16 | 55 tests passed after Alembic upgrade |
| Ruff, frontend lint and TypeScript | Passed |
| Frontend production build | Passed locally and in Docker |
| Dependency declarations versus locks | Passed |
| Installed Python dependency consistency | `pip check` passed |
| Python lock vulnerability query | OSV reported no known advisories for 80 locked versions |
| Frontend vulnerability query | npm audit reported zero vulnerabilities |
| Compose configuration | Passed |
| Updated backend and frontend Docker images | Built successfully |
| Existing Compose services after starting Docker | Backend, database, and frontend healthy |

The final combined Python/browser invocation passed 100 tests. PostgreSQL and
frontend tests are separate checks, not additional tests in that count. Browser
coverage includes registration/login, expired sessions, uploads and saved
results, ownership isolation, model activation/restore, experiment views, and
narrow layouts. It uses disposable accounts, SQLite, and a synthetic classifier.

The trained ConvNeXt checkpoint with SHA-256
`bb732a7e7b40f00c93102b238ce647386d1ce0464b628d6efb046305ed64cfc7`
also passed CPU inference and Grad-CAM using a validation image locally and a
synthetic image inside the updated read-only backend container. These are
execution checks, not a new accuracy evaluation. Local evidence is under the
run's ignored `backend_validation/` directory; images are not published.

Windows tests used the existing development environment and Edge; the backend
image uses its Linux CPU lock. No hosted CI execution was triggered. Initial
temporary-directory/subprocess failures were sandbox restrictions; approved
reruns passed without weakening application checks.

## Follow-up implementation and deployment

The four engineering follow-ups from the initial review were implemented:

- Backend/frontend containers were rebuilt and replaced using Compose, with the
  PostgreSQL and upload volumes retained. All three services are healthy.
- `scripts/verify_backup_restore.py` took a consistent private backup and restored
  it in a network-isolated, disposable PostgreSQL container. Six tables matched
  their snapshot counts and content digests. Two media files were extracted and
  verified, and ten checkpoint/configuration references across five registered
  models were copied and hash-verified. Evidence is stored locally under
  `models/backups/project-update-20260925-verified/verification.json`.
- The authenticated experiment view now discovers architecture runs under the
  model directory as well as conditional-GAN runs. Architecture details include
  test metrics, validation curves, class metrics, and filtered configurations.
  Host paths and runner commands are excluded from this new response. Missing
  optional results remain pending; incompatible or malformed reports fail safely.
- New classifier runs write atomic recovery files after each completed epoch.
  `train.py --resume` restores model/optimizer/scheduler/scaler, random and loader
  state, history, best checkpoint, and early-stopping counters. Compatibility
  checks reject changed data/configuration and unsupported legacy checkpoints.

The expanded combined backend/ML/scripts/browser suite passed 111 tests before
adding extra worker and CLI recovery cases. The recovery suite also verifies
interrupted versus uninterrupted training with worker processes and balanced
sampling. Frontend lint and all nine API tests passed; both production images
built successfully. Desktop and mobile browser cases cover architecture details
in addition to the pre-existing flows.

After deployment, read-only checks through the production Nginx frontend verified
that the completed ConvNeXt run appears with its actual 68.30% test macro-F1 at
1440 px and 360 px. Database table counts and active-model IDs matched the backup.
No application account, registered model, prediction, or active-model selection
was changed by verification. The real PostgreSQL service was retained rather
than recreated. The restore-test container was removed after verification.

The first backup attempt could not start its helper using the previous image ID
after an image rebuild. Services were restarted by the script's cleanup path;
that incomplete backup was not used for deployment. The verified retry used the
configured image reference. Both backups remain private and ignored by Git.

See [backup recovery](backup-recovery.md) and [training recovery](training-recovery.md)
for commands, assumptions, and limitations. No new full training run was started.
The September 25 trained weights and reported metrics remain unchanged.

## Remaining improvements

- Store approved encrypted backups off-host and schedule them if workstation-loss
  recovery is required. Secrets need a separate secure backup process.
- Classifier resume does not automatically resume the entire architecture/GAN
  orchestration or GAN generator/discriminator training. Evaluation and analysis
  must be run explicitly after resuming a classifier.
- Repeat matched model runs with additional seeds and investigate low-support
  classes. Patient/slide independence remains unverified and requires grouping
  metadata, not just more training runs.

Dependency queries are point-in-time results, not a guarantee against future
advisories. Browser checks cover selected interactions and layout assertions,
not a comprehensive assistive-technology assessment. Public deployment still
requires the environment-specific controls documented in `SECURITY.md`.
