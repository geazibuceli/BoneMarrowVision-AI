# Security policy

BoneMarrowVision AI is a research and educational prototype. It is not a medical device and is not approved for clinical or public production use.

## Reporting a vulnerability

Please do not publish exploitable details in a public issue. Use the repository's private GitHub security-advisory form when it is available, and include the affected component, a minimal reproduction, impact, and any suggested mitigation. Do not include real patient data, credentials, access tokens, model checkpoints, or private infrastructure details.

## Supported version

Security fixes are applied to the current default branch. No older release line is currently maintained.

## Deployment boundary

The provided Compose stack is for local, production-like validation. Before any networked deployment, add TLS, trusted-proxy configuration, request-rate limiting, centralized secret management, backups, monitoring, an upload-retention policy, and a formal privacy/LGPD assessment. Keep the backend port private, rotate the JWT and administrator bootstrap secrets, and never upload identifiable patient information.
