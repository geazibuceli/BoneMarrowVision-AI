# Reproducible Python installs and browser checks

## Supported locked environment

`requirements/runtime.lock` pins backend and ML runtime dependencies, including transitive dependencies and distribution hashes. `requirements/dev.lock` adds testing tools and Playwright while retaining the same runtime versions. Both files target **Linux x86_64, CPython 3.11, and CPU PyTorch**. The API Dockerfile and Python CI jobs install from these files with hash verification.

The dependency declarations remain in `backend/requirements*.txt` and `ml/pyproject.toml`. `requirements/runtime.in` selects the CPU PyTorch versions and build tooling; `requirements/dev.in` adds browser testing. Windows, macOS, other Python versions, and CUDA training are separate environments: do not use these Linux CPU locks as a universal workstation lock.

From a clean Python 3.11 environment on Linux x86_64:

```bash
python -m pip install --require-hashes --only-binary=:all: -r requirements/dev.lock
python -m pip install --no-deps --no-build-isolation -e ml
python -m pip check
python scripts/check_locked_requirements.py
```

Production uses `requirements/runtime.lock` instead of the development lock. Editable installation of the local ML package is deliberately separate: its dependencies and build tools are already installed from the lock. `--no-build-isolation` prevents an additional unpinned build environment.

The backend base image is pinned by digest. The Python packages and base image are fixed; these files do not claim that operating-system packages installed by browser setup, GitHub-hosted runner images, or build timestamps are bit-for-bit reproducible.

## Updating dependencies

Change the relevant declaration, then regenerate both locks from the repository root in Linux x86_64 with Python 3.11:

```bash
sh scripts/compile-locks.sh
```

The script verifies the platform, installs its compiler tooling, and reads the ML dependencies directly from `ml/pyproject.toml`. Existing compatible pins are preserved. To request updates explicitly, use `sh scripts/compile-locks.sh --upgrade` or `sh scripts/compile-locks.sh --upgrade-package PACKAGE`. Regenerating hashes can download large distributions.

On Windows, generate the locks in the same Linux image used by the backend, replacing the source path with the absolute checkout path:

```powershell
docker run --rm --mount "type=bind,source=$PWD,target=/workspace" -w /workspace python:3.11-slim@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534 sh scripts/compile-locks.sh
```

Review both lock diffs, run `python scripts/check_locked_requirements.py`, reinstall in a clean environment, run the backend, ML, and browser suites, and build the API image before merging. The declaration check detects incompatible or missing direct pins and runtime/development disagreements; `pip check` verifies installed dependency metadata. Neither check replaces executing the tests.

## Browser suite

The `browser` CI job runs `e2e/test_workflow.py` in Chromium at desktop and 360 px mobile widths. It starts the real FastAPI API and Vite frontend on isolated ports, applies migrations to a fresh SQLite database, and creates a deterministic small BaselineCNN checkpoint and synthetic image. It does not download a dataset, use real patient images, or require a trained checkpoint.

The checks cover registration, empty history, unavailable models, upload and actual inference, authenticated Grad-CAM, result reloads, pagination and return navigation, English dates under a Portuguese browser locale, invalid files, expired sessions, return after sign-in, and ownership isolation. The synthetic model validates integration, not classification accuracy. PostgreSQL backend tests run in their separate CI job.

After installing the Python development dependencies and running `npm ci` in `frontend`:

```bash
python -m playwright install --with-deps chromium
python -m pytest e2e -q
```

For an existing native Windows development environment, install `playwright==1.63.0` in that environment, then use its installed Chromium or optionally the system Edge browser:

```powershell
$env:E2E_BROWSER_CHANNEL = 'msedge'
python -m pytest e2e -q
```

Each run uses its own directory under ignored `.e2e-artifacts/`. Test servers and browser contexts are closed on completion, including failures. CI retains only server logs and failure screenshots for three days when a job fails; databases, checkpoints, and session credentials are excluded from uploaded artifacts. All test accounts and images are synthetic. The suite uses Vite for browser integration; the production Nginx smoke test is documented separately in [runtime validation](runtime-validation.md).

The workflow follows the official [pip-tools locking and hash guidance](https://pip-tools.readthedocs.io/en/stable/) and [Playwright Python CI setup](https://playwright.dev/python/docs/ci).

## Initial verification

The committed locks were installed with hash verification in a clean Python 3.11 Linux API image. `pip check` reported no conflicts. A separate Linux validation container using the development lock, Node 22.23.2, and Chromium passed 20 ML tests, 20 backend tests against isolated PostgreSQL, nine frontend tests, and both desktop/mobile browser cases. Lint and the frontend production build also passed. The API image built successfully with the runtime lock and its pinned base image.

Both browser cases also passed locally in headless Edge on Windows. They exposed a login redirect race: after a session expired, signing in could lead to the dashboard instead of the interrupted page. The authenticated redirect now respects the original destination, and the browser suite verifies that behavior.

The declaration checker was exercised with an intentionally incompatible dependency in a temporary copy and rejected it. CI workflow YAML was parsed locally. These are local validations of the workflow's commands; a hosted GitHub Actions run has not been triggered or observed in this session. The backend suite still emits dependency deprecation warnings, but no test failed.
