# Project update review — September 17, 2026

## Remediation completed after the review

- Upgraded pytest to 9.1.1 and setuptools to 84.0.0 and regenerated both locks.
- Replaced python-jose with PyJWT 2.14.0; ecdsa is absent from the new locks.
  JWT validation enforces HS256, issuer, audience, expiration, and required claims.
- Added periodic artifact cleanup with configurable intervals, failure recovery,
  and graceful shutdown, plus regression tests.
- Added OSV auditing and scheduled generation of candidate lock patches for review.
- Updated React, React DOM, React types, React Router, Vite, oxlint, and the four
  GitHub Actions to the releases identified below. Added Alembic's path separator.

Post-change validation passed: 39 backend tests on SQLite and again on isolated
PostgreSQL, 20 ML tests, nine frontend tests, two Chromium browser workflows, and
five audit-script tests. Linters, frontend production build, lock consistency,
and Compose configuration validation passed. OSV reported no known advisories
for 80 locked Python package versions; npm audit also reported none.
The production API image built successfully from the runtime lock, including
editable ML installation and `pip check` with setuptools 84.0.0.

The existing Windows CUDA environment is separate: its torch 2.11.0+cu128
requires setuptools below 82. The attempted local setuptools update was reverted
to the previous 78.1.0, and native `pip check` passed. PyJWT was installed locally.
Do not apply the Linux CPU runtime lock to this environment. Updating that CUDA
toolchain requires a separate compatibility review; the clean OSV result above
applies to the committed Linux locks, not every package in the existing venv.

The original review below is retained as historical context. Its recommendations
describe the previous state, not additional outstanding implementation work.
The remaining Starlette/httpx and AnyIO deprecation warnings are upstream
compatibility maintenance items. Hosted GitHub Actions have not been executed
from this workspace; candidate lock patches still require review and full tests.

## Original review

This review inspected application dependencies, lock files, CI, retention behavior,
and local validation. Dependency versions were checked against npm, PyPI, OSV,
and official GitHub release metadata. No dependency upgrades were applied.

## Recommended work, in priority order

1. **Update pytest and regenerate the development lock.** The lock contains
   pytest 8.4.2 and `backend/requirements-dev.txt` restricts pytest to `<9.0`.
   Version 9.0.3 fixes insecure temporary-directory handling, CVE-2025-71176;
   the current release found was 9.1.1. Widen the constraint deliberately and
   validate backend, ML, and browser tests together. This affects test tooling;
   it does not establish a remotely exploitable API vulnerability.
   Source: [pytest 9.0.3 release](https://github.com/pytest-dev/pytest/releases/tag/9.0.3).

2. **Review the setuptools pin and update the build toolchain.** Both
   `requirements/runtime.in` and `scripts/compile-locks.sh` pin 79.0.1.
   OSV flags CVE-2026-59890, involving MANIFEST exclusions and Unicode filename
   normalization when building source distributions on macOS. This is different
   from the application's Linux installation of hashed binary wheels. OSV lists
   83.0.0 as fixed, while the upstream advisory still lists no patched version;
   resolve that metadata discrepancy before claiming remediation. PyPI reports
   84.0.0 as the latest release. Test editable ML installation and both lock files
   when changing the pin.
   Source: [setuptools advisory](https://github.com/pypa/setuptools/security/advisories/GHSA-h35f-9h28-mq5c).

3. **Make retention run periodically.** `backend/app/main.py` invokes
   `cleanup_expired_artifacts` only during application startup. A continuously
   running service therefore keeps artifacts past their retention period until
   its next restart. Add a scheduled cleanup mechanism and test expiration while
   the service remains running, including safe shutdown and overlapping runs.

4. **Maintain and audit the resolved Python locks automatically.** Dependabot
   configures Python checks under `/backend` and `/ml`, but has no explicit
   `/requirements` entry or lock-regeneration workflow. The existing lock checker
   verifies requirement consistency; it does not detect security advisories.
   Establish a tested process that regenerates the actual deployment locks and
   audits their resolved packages. Verify support for the custom `.lock` filenames
   before relying on Dependabot alone.

5. **Triage the unused ECDSA dependency.** `python-jose` brings in ecdsa 0.19.2,
   flagged for CVE-2024-23342. Upstream lists all versions as affected and no fix.
   The inspected JWT code explicitly signs and verifies with HS256, so it does
   not exercise the affected elliptic-curve signing operation. Consider removing
   this dependency through a tested JWT-library migration, or record a narrowly
   scoped exception based on the enforced algorithm. A routine ecdsa version bump
   cannot resolve the advisory.
   Source: [python-ecdsa advisory](https://github.com/tlsfuzzer/python-ecdsa/security/advisories/GHSA-wj6h-64fc-37mp).

## Routine maintenance candidates

These are available upgrades, not evidence of a current application failure.

| Component | Current | Latest found |
| --- | --- | --- |
| React / React DOM | 19.2.8 | 19.3.0 |
| React Router DOM | 7.18.3 | 7.18.4 |
| Vite | 8.2.2 | 8.3.0 |
| oxlint | 1.80.0 | 1.83.0 |
| actions/checkout | v4 | v7.0.1 |
| actions/setup-python | v5 | v7.0.0 |
| actions/setup-node | v4 | v7.0.0 |
| actions/upload-artifact | v4 | v7.0.1 |

Update React type packages alongside React. Review action migration notes and
runner requirements before changing major versions. Latest action releases were
checked through each repository's official GitHub releases API.

Do not blindly upgrade all Python packages: the latest NumPy, SciPy, contourpy,
and pyee releases found require Python 3.12, while the deployment targets 3.11.
Other transitive packages can have exact constraints imposed by their parents.

Backend tests also emit deprecation warnings about Alembic's missing
`path_separator=os`, Starlette's use of httpx, and an AnyIO portal alias. Address
the Alembic setting locally and track upstream compatibility for the others.

## Validation completed

- Backend: 20 tests passed in the existing Linux Python 3.11 validation image,
  using SQLite and isolated artifact directories.
- ML: 20 tests passed in the same image.
- Frontend: 9 tests passed; lint, TypeScript checking, and production build passed.
- Ruff checks passed for backend, ML, browser test code, and the lock checker.
- Lock consistency validation passed; native Windows `pip check` passed.
- Online npm audit reported zero known vulnerabilities.
- OSV queried 84 unique locked Python packages. Duplicate advisory identifiers
  were consolidated into the three issues discussed above.

The Windows build initially hit a sandbox subprocess restriction and passed when
rerun with permission. Docker also required permission to access its engine.
These were environment restrictions, not project build failures.

This review did not rerun browser E2E tests, PostgreSQL integration tests, hosted
GitHub Actions, or a fresh production image build. It did not audit container OS
packages or certify deployment configuration. Earlier runtime evidence remains
documented separately in `runtime-validation.md`.
