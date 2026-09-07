# Changelog

## 0.2.0 - 2026-09-07

- Added configurable rate limiting for authentication and inference requests.
- Added trusted-proxy client identification for the bundled Nginx deployment.
- Added automatic retention of generated Grad-CAM files and temporary inference inputs.
- Added CPU-only PyTorch installation for the API container and CI jobs.
- Added PostgreSQL-backed API test execution in CI while preserving Alembic-managed schema.
- Hid expired Grad-CAM links from prediction responses.
- Expanded backend tests and updated operational documentation.