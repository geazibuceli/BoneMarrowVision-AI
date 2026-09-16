---
name: bmv-backend-security
description: Review or fix security in the BoneMarrowVision FastAPI API, including authentication, permissions, uploads, and checkpoints. Use for requested hardening or concrete failures at these boundaries.
---

# BoneMarrowVision backend security

Read SECURITY.md and the affected parts of backend/app/security.py, core/config.py, core/rate_limit.py, inference.py, and routers. Connect each finding to an execution path and demonstrable impact; distinguish confirmed vulnerabilities from hypotheses. Keep project content in English.

Preserve existing protections when fixing a problem:

- Administrative authorization for model management and owner-scoped queries for predictions and media. Test access by another user and without a session.
- Image-content validation and byte and pixel limits. A file extension or Content-Type alone does not prove a valid format.
- Model and configuration path restrictions, checkpoint validation, the registered digest, and torch.load with weights_only=True. Do not disable these checks to accept an incompatible file.
- File retention and cleanup on success and failure; do not expose uploads or Grad-CAM through a public static directory.
- Secrets outside code and logs. Do not print .env or resolved configuration containing credentials.

Current request limits are process-local. Assess trusted proxies and replica count before proposing a fix; do not add Redis or infrastructure without a need within the requested scope.

Use backend/tests, particularly authentication, contract, media, rate-limit, and retention tests as appropriate. Prefer temporary fixtures and test denied access as well as permitted paths. Run relevant tests from backend with python -m pytest.

Present evidence and proportionate fixes. A review does not authorize external penetration testing, secret rotation, or changes to real accounts.
