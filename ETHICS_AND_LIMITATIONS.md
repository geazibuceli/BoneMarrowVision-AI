# Ethics and Limitations

Research and Educational Use Only. This system is not a medical device and must not be used for clinical diagnosis, treatment, or decision-making.

The platform is not clinically validated. It must not be used for diagnosis, treatment, triage, screening, or any medical decision. Dataset bias, class imbalance, label noise, and hidden source correlations can produce misleading performance. Results may not generalize across hospitals, scanners, stains, preparation protocols, or populations. Neural networks can be overconfident, and Grad-CAM can highlight plausible but irrelevant regions; it is not evidence of causal reasoning or clinical correctness.

Responsible use requires external validation, uncertainty and calibration analysis, subgroup and source-shift analysis, expert hematologist review, and governance appropriate to the deployment context. A visually plausible Grad-CAM must never override low confidence, contradictory evidence, or professional review.

Avoid personally identifiable information in uploads. Apply data minimization, access control, deletion and retention rules, incident response, and an LGPD assessment before handling any personal data. The local stack does not supply TLS, centralized secrets, rate limiting, operational monitoring, backup policy, or a validated deletion workflow. This research prototype has security and privacy limits and is not a substitute for a regulated medical system. See [SECURITY.md](SECURITY.md).
