# Reports

This directory contains selected, de-identified outputs retained as evidence for the recorded TCC experiment. It must not contain raw images, uploads, patient identifiers, credentials, personal filesystem paths, or model checkpoints.

The existing `analysis_40ep/`, `metrics/`, and `dataset_summary.json` files predate the repository's strict split manifests and artifact-provenance format. They are historical results, not proof that the current pipeline was rerun successfully. Regenerate the dataset inspection, splits, evaluation, and artifact manifests before making a new reproducibility claim.

Generated Grad-CAM images are intentionally ignored by Git because an input image can contain sensitive information. Only deliberately reviewed, de-identified aggregate reports should be committed.
