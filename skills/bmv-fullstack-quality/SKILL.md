---
name: bmv-fullstack-quality
description: Fix failures and improve end-to-end flows across React, FastAPI, and persistence in BoneMarrowVision. Use for integration bugs and functional reliability.
---

# BoneMarrowVision full-stack quality

Start with the expected behavior and an observable failure trigger. Follow the affected flow from the interface through the API and persistence rather than starting a general audit. Keep project content in English.

At the project root, consult frontend/src/api.ts, types.ts, and auth.tsx; backend/app/routers, schemas.py, models.py, and database.py; and corresponding tests in frontend/tests and backend/tests. Confirm the actual contract before changing calls or responses.

- Preserve prediction isolation between users and the distinction between researcher and administrator roles.
- For uploads and analysis, check error responses, temporary-file cleanup, and behavior without an active model. Use fixtures or mocks in contract tests; do not require a large checkpoint to test authentication or validation.
- For persistent schema changes, use backend/alembic and verify the migration. Do not replace migrations by recreating the user's database.
- Add regression tests for behavioral failures when they demonstrate the problem, avoiding tests that simply mirror the implementation.

Use commands from the current Makefile and manifests. In backend, run python -m pytest tests; in frontend, run npm test, npm run lint, and npm run build as appropriate for the affected layers. For ML changes, consult ml/tests and run the relevant suite from ml. On Windows without make, run individual commands in their respective directories.

Do not describe unit tests as end-to-end verification. Report the corrected flow, executed checks, and anything that required an unavailable database, browser, or model.
