# Model Card

## Model and intended use

This project provides a custom baseline CNN, ImageNet transfer learning with ResNet50, and EfficientNet-B0. The models are intended for educational and research analysis of bone marrow cell images only. They must not be used for diagnosis, treatment, triage, screening, or other clinical decisions.

## Completed ResNet-50 and EfficientNet-B0 comparison

The matched run in `models/architecture_comparison_seed42_run2` completed training, held-out evaluation, and per-class analysis on September 23, 2026. Both architectures used ImageNet weights, seed 42, 224 x 224 inputs, batch size 32, AdamW with learning rate 0.0003 and weight decay 0.0001, weighted cross-entropy, and a maximum of 100 epochs. Checkpoints were selected using validation macro-F1.

The audited splits in `data/experiments/resnet50_comparison` contain 119,947 training images, 25,701 validation images, and 25,703 test images across 21 classes. The audit found two byte-identical validation images with conflicting MON and PMO labels. Both occurrences were excluded from a separate copy, with the reason recorded in `exclusions.json`. Training and test split files were preserved byte-for-byte. The corrected audit found no exact content duplicates across splits or remaining conflicting labels; it does not establish patient/slide isolation or detect visually similar images.

| Metric | ResNet-50 | EfficientNet-B0 |
| --- | ---: | ---: |
| Accuracy | 85.95% | 86.49% |
| Balanced accuracy | 72.05% | 72.48% |
| Macro precision | 72.78% | 74.80% |
| Macro recall | 72.05% | 72.48% |
| Macro F1 | 71.56% | 73.11% |
| Weighted F1 | 86.02% | 86.46% |

Metrics are rounded percentages from each architecture's `test_report.json` under the run directory. Both reports record the same test split SHA-256, `f84457c9f582731b288f1100705c57d87bf7f5db92524b63067da0989f27a264`, and the same class vocabulary. The runner preserves configurations, input hashes, and execution status alongside the reports. Full checkpoints and evaluation artifacts remain local.

EfficientNet-B0 achieved the higher macro-F1 in this run. This single-seed comparison does not quantify training variability or establish statistical superiority. The historical 40-epoch result below belongs to a different training and validation protocol and should not be ranked as another matched run.

See [the comparison guide](docs/resnet50-comparison.md) for the audit procedure and integration evidence. To reproduce only these two architectures with the current runner, explicitly pass `--architectures resnet50 efficientnet_b0` and use a new output directory.

## Historical EfficientNet-B0 experiment

The recorded experiment used 171,353 valid images across 21 classes, ImageNet-pretrained EfficientNet-B0, 224 x 224 inputs, batch size 32, seed 42, weighted cross-entropy, and 40 epochs. Training used an NVIDIA GeForce RTX 5080 with CUDA. The held-out image-level test results were:

| Metric | Result |
| --- | ---: |
| Accuracy | 86.11% |
| Balanced accuracy | 72.46% |
| Macro precision | 75.57% |
| Macro recall | 72.46% |
| Macro F1 | 73.16% |
| Weighted F1 | 86.20% |

The best recorded validation macro F1 was 72.14% at epoch 30. The checkpoint is a local generated artifact and must not be committed. Selected, de-identified aggregate analysis outputs are retained under `reports/analysis_40ep/` for reference.

This run predates the current verified split manifests, portable provenance records, and strict duplicate policy. The recorded metrics are therefore historical evidence rather than proof of an end-to-end run with the current code. A new training and held-out evaluation run is required to make that reproducibility claim.

## Inference integration

The API loads an active checkpoint registered in the model registry, applies the evaluation preprocessing, returns the predicted class and top-three probabilities, and generates a Grad-CAM overlay. The endpoint rejects invalid image content and does not persist a prediction when the active checkpoint is missing or cannot be loaded. A `503` response means that model configuration or loading must be fixed before inference can run.

For the completed comparison, each selected checkpoint classified one validation image from each of the 21 classes and generated 21 decodable Grad-CAM overlays through the backend inference code. Both checkpoints also passed a real-image classification and Grad-CAM check inside the backend container. These checks verify execution, not additional accuracy or a browser upload workflow. Registration and activation are separate operations; completed offline evaluation does not imply that either model is active.

## Limitations

The test split was stratified at image level because the downloaded file layout contained no patient, slide, case, or source mapping. Patient-level independence cannot therefore be guaranteed. Related cells from one person or slide may occur in more than one split, which can inflate measured performance.

The classes are highly imbalanced, and results for categories with very small test support are unstable. The local download was incomplete and contained unreadable images plus duplicate content with conflicting class folders. The strict current pipeline stops on these conditions so that they must be repaired, reviewed, or explicitly documented.

Training uses configurable preprocessing, augmentation, class-imbalance strategies, deterministic controls, and a held-out test split. Exact numerical repeatability can still vary with hardware, drivers, PyTorch kernels, and third-party library versions. No external validation, calibration analysis, prospective evaluation, robustness study, or clinical validation was performed.
