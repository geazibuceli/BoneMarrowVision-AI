# ResNet-50 comparison

Keep the historical EfficientNet-B0 checkpoints and reports. Register ResNet-50 as
a separate model version; activation changes the model used for future requests.
Only one model is active globally. Registration and activation require an administrator.

## Preserve and audit

Before training, archive existing checkpoints, their SHA-256 hashes, configurations,
reports, and split files under an ignored local directory. Never use the default
`models/best_model.pt` output for this experiment.

Audit the existing split membership without regenerating it:

```powershell
./.venv/Scripts/python.exe ml/scripts/audit_existing_splits.py --data-dir data/processed
```

The audit decodes images, hashes their file contents, checks class vocabularies,
and detects exact content duplicates. It writes a manifest only when there are no
cross-split duplicates or conflicting labels. It refuses to overwrite a manifest.
An unsuccessful duplicate audit writes `existing_split_audit.json` instead.
Original split JSON files are not changed. The audit does not detect visually
similar images or establish patient/slide isolation. Existing split rows do not
contain the metadata needed to establish that isolation.

Do not bypass verification for a controlled comparison. If leakage requires new
splits, retain historical results separately and retrain both architectures on the
corrected splits. Never remove test images based on model performance.

## Matched protocol

`ml/configs/resnet50.yaml` matches the recorded configuration of
`models/best_model_100ep.pt`, except for architecture: seed 42, 224-pixel images,
batch size 32, AdamW learning rate 0.0003, weighted cross entropy, and a maximum of
100 epochs. Select the best checkpoint using validation macro-F1. The historical
checkpoint records the current train and validation file hashes. Verify those
hashes again before reusing it. The legacy `best_model.pt` has 17 classes and is
not the reference for the current 21-class task.

A matched single-seed experiment is an initial comparison. Repeated runs with
different seeds are needed to quantify training variability. Use validation data
for tuning and reserve the test split for final evaluation.

## Execution

The initial audit found two byte-identical files with conflicting MON and PMO
labels, both in validation. No exact duplicates crossed split boundaries. Preserve
the original data and exclude both conflicting occurrences from a new copy:

```powershell
./.venv/Scripts/python.exe ml/scripts/prepare_comparison_splits.py
```

The corrected copy is `data/experiments/resnet50_comparison`: 119,947 training
images, 25,701 validation images, and 25,703 test images across 21 classes.
Train and test JSON files are copied byte-for-byte. `exclusions.json` records the
reason and source audit hash. Images are referenced in place, not deleted or relabeled.
The corrected copy is audited again before a manifest is created.

Run the automated sequence in a new output directory:

```powershell
./.venv/Scripts/python.exe ml/scripts/run_architecture_comparison.py --architectures resnet50 efficientnet_b0 --full --output-dir models/resnet50_efficientnet_b0_comparison_new
```

The runner refuses to reuse an existing output directory. Choose a new suffix for
a new run. It freezes and checks configurations, records split hashes, runs a
one-epoch smoke check for each architecture on at most two images per class per
split, then trains ResNet-50 and EfficientNet-B0 sequentially from pretrained
weights. Smoke weights are never used to initialize the full runs. Evaluation and
per-class analysis follow after both trainings, and `comparison.md` summarizes
metrics only after checking matching test hashes and class vocabularies.

Omit `--full` to run smoke checks only. Follow `status.json` and the per-step
`*.stdout.log` / `*.stderr.log` files inside the output directory. A failure stops
the sequence and records its error. The runner does not automatically resume its
sequence. New classifier runs can be continued with `train.py --resume`; see
[training recovery](training-recovery.md), then run evaluation and analysis explicitly.
Keep the host awake for the duration of the experiment. No registry activation is
performed automatically.

Pretrained ResNet-50 weights require an initial download. Measure training GPU
memory and time before estimating completion. The application currently performs
inference on CPU, so measure deployment latency on CPU as well, including Grad-CAM.

Use `ml/scripts/analyze_results.py` for per-class metrics and training curves.
Compare macro-F1, balanced accuracy, per-class precision/recall/F1, confusion
matrices, checkpoint size, and latency under identical hardware conditions.
Keep full local artifacts out of Git until reviewed for publication.

## Application validation

Register the new checkpoint under `models/` and its configuration under
`ml/configs/`, with an explicit name, version, architecture, and evaluation metrics.
Keep the existing EfficientNet entry. Verify ResNet prediction and Grad-CAM,
reactivation of EfficientNet, and the model identity on historical predictions.
Do not change the active model merely to record offline training results.

## Completed local run and checkpoint validation

The run in `models/architecture_comparison_seed42_run2` completed training,
evaluation, and analysis for both architectures on September 23, 2026.

| Test metric | ResNet-50 | EfficientNet-B0 |
| --- | ---: | ---: |
| Accuracy | 0.859549 | 0.864918 |
| Macro-F1 | 0.715575 | 0.731128 |
| Balanced accuracy | 0.720451 | 0.724813 |

The selected full-training checkpoints were subsequently loaded through the
backend inference code. Each model classified one validation image from each of
the 21 classes and generated 21 decodable Grad-CAM overlays. Checkpoint hashes and
class order matched the corresponding evaluation reports. These 42 execution
checks verify loading and inference behavior; they are not another accuracy
evaluation or an authenticated browser/API test.

Local artifacts are stored under the run's `final_validation/` directory:

- `validation.json`: execution-check results and checkpoint hashes.
- `resnet50_registration.json` and `efficientnet_b0_registration.json`: prepared
  request bodies for `POST /api/v1/models/register` using an authenticated admin.
- Per-class Grad-CAM overlays, retained locally and excluded from Git.

The local Docker Compose application was subsequently started at `http://localhost`.
The existing PostgreSQL volume was backed up before configuration and migration;
it contained no users or registered models. The first administrator was created
through the bootstrap API, and both models were registered through authenticated
API requests: ResNet-50 as model 1 and EfficientNet-B0 as model 2, both with version
`comparison-seed42-2026-09-23`. Returned checkpoint hashes matched the reports.
Neither model was activated; the initially empty active-model selection remains empty.

Both full checkpoints also passed classification and Grad-CAM checks on one real
image inside the backend container. Temporary images were removed and no prediction
history was created. Backend, frontend, and database health checks passed. These
checks do not constitute a browser upload-flow test. Registration and container
inference evidence are in `final_validation/docker_registration.json` and
`final_validation/docker_inference.json`. Local administrator credentials are kept
in an ignored local file, not in this document or source control.

## ConvNeXt-Tiny extension

ConvNeXt-Tiny completed the same audited splits and configuration on September 25,
2026. See the [run record](convnext-comparison.md) and the updated
[three-model comparison](../MODEL_CARD.md). The application checks above apply
only to the two September 23 checkpoints.
