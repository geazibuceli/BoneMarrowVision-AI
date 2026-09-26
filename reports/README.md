# Reports

This directory contains selected, de-identified outputs retained as evidence for the recorded experiment. It must not contain raw images, uploads, patient identifiers, credentials, personal filesystem paths, or model checkpoints.

The existing `analysis_40ep/`, `metrics/`, and `dataset_summary.json` files predate the repository's strict split manifests and artifact-provenance format. They are historical results, not proof that the current pipeline was rerun successfully. Regenerate the dataset inspection, splits, evaluation, and artifact manifests before making a new reproducibility claim.

Generated Grad-CAM images are intentionally ignored by Git because an input image can contain sensitive information. Only deliberately reviewed, de-identified aggregate reports should be committed.

## Completed architecture comparison

`architecture_comparison_2026-09-25.json` contains reviewed aggregate results for
ResNet-50, EfficientNet-B0, and ConvNeXt-Tiny on the same audited test split. It
retains configurations, class supports and metrics, selected validation epochs,
and source/checkpoint hashes. It contains no images, sample paths, credentials,
or checkpoint weights. Metrics are fractions, not percentages. Full source
reports remain in the ignored local run directories identified in the summary.
