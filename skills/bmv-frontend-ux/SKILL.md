---
name: bmv-frontend-ux
description: Improve screens, navigation, responsiveness, and accessibility in the BoneMarrowVision React frontend. Use for interface and user experience tasks in this project.
---

# BoneMarrowVision frontend and UX

Treat the application as a software product. Prioritize the user's task rather than academic deliverables. Keep code, comments, documentation, and interface text in English.

Locate the root using README.md and frontend/package.json. Read frontend/src/main.tsx, styles.css, and the affected components before changing the interface. Consult api.ts, auth.tsx, and types.ts when the flow depends on data or sessions. Confirm paths because the structure may evolve.

- Preserve the existing visual language unless a redesign is requested. Extract components when doing so simplifies a concrete change; do not impose a general restructuring.
- When changing login, uploads, analysis, or history, cover the relevant loading, empty, success, error, and expired-session states. Prevent duplicate submissions and place feedback near the action.
- Present results and confidence using the actual API contract. Do not invent results, capabilities, or model availability to fill a screen.
- Preserve authenticated access to Grad-CAM images; do not turn private files into public URLs for previews.
- Check keyboard access, focus, labels, and narrow layouts on changed screens. Prefer native features and existing dependencies before adding libraries.

In frontend, run npm run lint and npm run build. Run npm test when changing integration or behavior covered by tests. If a browser is available, exercise the changed flow at narrow and wide viewport sizes. Disclose when visual verification was unavailable; a successful build does not prove usability.

Report implemented changes, checks performed, and any limitations without assuming the user requested a complete redesign.
