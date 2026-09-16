---
name: bmv-deploy-operations
description: Improve Docker, CI, configuration, logs, and health checks in BoneMarrowVision. Use to prepare or fix application execution and delivery.
---

# BoneMarrowVision deployment and operations

Read docker-compose.yml, backend/Dockerfile, frontend/Dockerfile, the frontend Nginx configuration, .github/workflows, .env.example, and Makefile as relevant. Confirm existing paths and commands before changing the workflow. Keep project content in English.

- The Compose setup described in README.md is a production-like local stack. Preparing deployment files does not establish public operational readiness; adapt to a destination only when one is defined.
- Preserve database and upload volumes, read-only model mounts, the non-root backend, and its read-only filesystem. Account for temporary directories required by PyTorch and Matplotlib.
- Check startup with migrations and without an active checkpoint. Distinguish process health, dependency availability, and inference readiness using the endpoints actually implemented.
- Keep secrets in external configuration. Prefer docker compose config --quiet when validating Compose to avoid printing resolved credentials.
- When changing proxies, check forwarded headers and TRUST_PROXY_HEADERS. For multiple replicas, assess process-local request limits, model memory, and shared storage.
- Logs should help diagnose failures without including tokens, passwords, or image contents. Do not introduce an entire observability platform to address one missing log.

Run checks for affected layers as specified in Makefile. Validate Compose when changing the stack; build or smoke-test containers when Docker is available and necessary to verify the change. Report checks blocked by missing tools or configuration.

Do not use down -v or recreate databases to fix startup. Before migrations over real data, consider backup and rollback in the authorized environment. External publication and infrastructure changes must respect the user's destination and authorization; creating this skill does not authorize deployment.
