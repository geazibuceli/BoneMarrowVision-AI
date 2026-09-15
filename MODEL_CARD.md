# Model Card

## Model and intended use

This project provides a custom baseline CNN, ImageNet transfer learning with ResNet50, and EfficientNet-B0. The models are intended for educational and research analysis of bone marrow cell images only. They must not be used for diagnosis, treatment, triage, screening, or other clinical decisions.

## Recorded EfficientNet-B0 experiment

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

## Limitations

The test split was stratified at image level because the downloaded file layout contained no patient, slide, case, or source mapping. Patient-level independence cannot therefore be guaranteed. Related cells from one person or slide may occur in more than one split, which can inflate measured performance.

The classes are highly imbalanced, and results for categories with very small test support are unstable. The local download was incomplete and contained unreadable images plus duplicate content with conflicting class folders. The strict current pipeline stops on these conditions so that they must be repaired, reviewed, or explicitly documented.

Training uses configurable preprocessing, augmentation, class-imbalance strategies, deterministic controls, and a held-out test split. Exact numerical repeatability can still vary with hardware, drivers, PyTorch kernels, and third-party library versions. No external validation, calibration analysis, prospective evaluation, robustness study, or clinical validation was performed.
