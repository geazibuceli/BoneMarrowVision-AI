# ConvNeXt-Tiny comparison

ConvNeXt-Tiny is supported by the shared model builder, training, evaluation,
checkpoint reconstruction, and Grad-CAM. Its configuration matches ResNet-50 and
EfficientNet-B0 except for the architecture. This is a matched initial experiment,
not an independently tuned optimum for each model.

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
