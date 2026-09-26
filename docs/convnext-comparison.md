# ConvNeXt-Tiny comparison

ConvNeXt-Tiny is supported by the shared model builder, training, evaluation,
checkpoint reconstruction, and Grad-CAM. Its configuration matches ResNet-50 and
EfficientNet-B0 except for the architecture. This is a matched initial experiment,
not an independently tuned optimum for each model.

## Completed run: September 25, 2026

`models/convnext_tiny_seed42_restart_20260924_214637` completed all 100 epochs,
held-out evaluation, and analysis successfully at 05:29:50 UTC (02:29:50 in
America/Sao_Paulo). The full run started from ImageNet weights; neither the
interrupted run nor the smoke-check weights were reused. The original run remains
preserved separately.

The configuration matches the September 23 ResNet-50/EfficientNet-B0 runs except
for architecture. All three share the same audited split manifest, test split
hash, and ordered 21-class vocabulary: 119,947 training, 25,701 validation, and
25,703 test images. The best validation macro-F1 was 0.713251 at epoch 28.

| Held-out test metric | ConvNeXt-Tiny |
| --- | ---: |
| Accuracy | 0.858110 |
| Balanced accuracy | 0.661788 |
| Macro precision | 0.724282 |
| Macro recall | 0.661788 |
| Macro F1 | 0.682981 |
| Weighted F1 | 0.858259 |

Evaluation used `convnext_tiny/best.pt`, selected by validation macro-F1, rather
than `convnext_tiny/best.final.pt` from epoch 100. The selected checkpoint SHA-256
is `bb732a7e7b40f00c93102b238ce647386d1ce0464b628d6efb046305ed64cfc7`.
The frozen configuration is `convnext_tiny.yaml` in the run directory; logs,
`status.json`, `convnext_tiny/test_report.json`, and `convnext_tiny/analysis/`
provide the local execution and evaluation records.

EfficientNet-B0 retained the highest test macro-F1 (0.731128), followed by
ResNet-50 (0.715575). ConvNeXt-Tiny had zero recall for ABE (one test image) and
LYI (ten test images); these estimates have very small supports. The reviewed
[aggregate record](../reports/architecture_comparison_2026-09-25.json) retains
per-class results for all three models. This single-seed result does not establish
statistical superiority or patient-level generalization.

This checkpoint subsequently passed local backend CPU inference and Grad-CAM on
a validation image and the same execution path on a synthetic image in the
updated backend container. See the [project review](project-review-2026-09-25.md)
for coverage and deployment status. It was not registered or activated in the
application during this review.

## Reproduction

Completed and partial architecture runs under `models/` now appear in
**Experiments > Local training runs**, alongside conditional-GAN runs. Open a
run to view test metrics, validation curves, per-class results, and configuration.
Missing evaluation files remain visible as pending; private runner commands and
host paths are not exposed. Comparing separate runs still requires matching
test split hashes and training conditions.

For new classifier runs, see [epoch-boundary recovery](training-recovery.md).

Run a one-epoch pretrained smoke check on at most two images per class per split:

```powershell
./.venv/Scripts/python.exe ml/scripts/run_architecture_comparison.py --architectures convnext_tiny --output-dir models/convnext_tiny_smoke_seed42
```

The first run downloads ImageNet weights into `models/torch-cache`. Use a new
output directory for every invocation; existing results are never overwritten.

Train and evaluate all three architectures on the same audited splits:

```powershell
./.venv/Scripts/python.exe ml/scripts/run_architecture_comparison.py --full --output-dir models/three_architecture_comparison_seed42
```

Use `--architectures resnet50 efficientnet_b0` to reproduce the two-model protocol,
or `--architectures convnext_tiny` to train only the new model. The default now
includes all three models. Full runs train for up to 100 epochs per model and
produce `comparison.md`, evaluation reports, and per-class analysis. Inspect
`status.json` and the per-step logs for progress and failures.

The runner uses the corrected splits described in [the split protocol](resnet50-comparison.md).
Patient/slide isolation remains unverified. Select adjustments using validation
data and reserve test data for final evaluation. Compare macro-F1 and per-class
recall alongside separately measured latency and memory on the target hardware.
No model is automatically registered or activated in the application.
