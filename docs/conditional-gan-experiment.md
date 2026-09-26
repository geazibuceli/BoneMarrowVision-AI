# Conditional GAN augmentation experiment

Run from the repository root with the installed ML package and CUDA. New runs train
both a real-only control and a GAN-augmented classifier:

```powershell
.venv/Scripts/python.exe -u ml/scripts/run_gan_experiment.py --seed 42 --output data/experiments/paired-50ep-seed42
```

The output directory must not already exist. The runner preserves the source
splits and decodes and hashes every real image. The experiment configuration
quarantines every member of groups with cross-split pixel duplicates or conflicting
labels from its isolated copies of all splits; the originals remain unchanged.
`source_audit.json` records each exclusion. Setting `duplicate_policy: fail`
instead rejects these inputs. This audits image identity;
it does not establish patient independence without patient metadata.

The configuration is `ml/configs/conditional_gan_experiment.yaml`. The first stage
trains a class-conditioned 128px generator and a spectral-normalized projection
discriminator with hinge loss. One GAN epoch is 200 balanced, replacement-sampled
batches of 32 images, not a complete pass through the source dataset. Training
runs for 50 such epochs. Preview rows follow the checkpoint's sorted class names,
with four fixed-noise samples per row. GAN losses do not measure morphological
validity or classifier accuracy.

Generation uses the final GAN checkpoint, never the validation/test images.
Only classes below the median real training count receive synthetic images.
Each receives at most one synthetic image per real example, capped at that
median. This conservative rule is fixed before evaluation; a class with six real
examples receives at most six synthetic examples. It does not create evidence
of additional biological diversity. Outputs are 128px and are resized by the
classifier's existing 224px preprocessing.

Images have generated labels marked `unreviewed_experimental`. The runner checks
exact file duplicates among generated samples, but that is not a perceptual
memorization or morphology audit. Expert review and additional quality analysis
are required before treating them as validated training data. The automated run
is an exploratory experiment, with no automatic activation in the application.

The classifier stage trains two fresh ImageNet-initialized EfficientNet-B0 models
for exactly 50 epochs: first a real-only control, then a real-plus-synthetic model.
Both subprocesses use the same classifier configuration, seed, initialization
procedure, and real-only validation/test splits frozen after source quarantine.
Early stopping patience must exceed the run length. Best checkpoints are selected
by validation macro F1 before either classifier is evaluated on the test split.
Equal epochs do not imply equal optimizer steps because training sizes differ;
class weights also follow each arm's training counts using the same weighting rule.

Outputs in the selected local directory:

- `status.json`: active stage or failure details, UTC start/update timestamps,
  elapsed seconds at the last stage transition, and stage history.
- `config.json`, `provenance.json`, `real_class_counts.json`: run configuration,
  source-code hashes, and real training counts.
- `gan/latest.pt`, `gan/epoch_*.pt`, `gan/history.json`, `gan/preview_*.png`:
  generator/discriminator state, optimizer/RNG state, losses, and previews.
- `synthetic/manifest.json`: generated images, labels, hashes, seed, and generator digest.
- `classifier/epoch_001.pt` through `epoch_050.pt`: per-epoch classifier checkpoints.
- `classifier/best.pt`: selected by real validation macro F1.
- `classifier/best.final.pt`: final classifier, irrespective of the best epoch.
- `classifier/best.history.json`: metrics updated atomically after every epoch.
- `control/`: corresponding real-only checkpoints, history, CSV, and learning curves.
- `classifier_history.csv`, `classifier_curves.png`: tabular history and curves,
  refreshed during classifier training and finalized after the last epoch.
- `test_report.*`, `control_test_report.*`: each arm's real held-out test metrics and confusion matrix.
- `comparison.json`: real test accuracy, unweighted cross-entropy loss, macro
  precision, and macro F1 for both matched arms, GAN-minus-control deltas, seed,
  checkpoint and test-split hashes, and an optional historical baseline clearly
  identified as uncontrolled. Set `baseline_checkpoint: null` to omit the historical model.

Training loss uses class-weighted cross-entropy; validation/test loss is
unweighted cross-entropy. Their absolute values are not directly comparable.
The historical comparison also differs in training duration and possibly other
conditions. One matched run does not establish a consistent benefit. Repeat with
different seeds and compare paired changes; no improvement is assumed in advance:

```powershell
foreach ($experimentSeed in 42, 43, 44) {
    .venv/Scripts/python.exe -u ml/scripts/run_gan_experiment.py --seed $experimentSeed --output "data/experiments/paired-50ep-seed$experimentSeed"
    if ($LASTEXITCODE -ne 0) { throw "Experiment failed for seed $experimentSeed" }
}
```

Choose fresh output names if any run already exists. Each seed retrains the GAN
and both classifiers; budget time and disk space for all three training stages.

## View runs in the application

Sign in and open **Experiments → Local training runs**, then select a run.
The detail page shows comparison metrics, validation curves, per-class test
metrics, configuration, available artifacts, and stage history. In-progress
details refresh every 15 seconds; completed and failed runs have manual refresh.
Existing historical runs remain readable and are labeled as uncontrolled.
Missing optional files are expected while training; malformed files produce an error.

The authenticated `/api/v1/experiments/runs` endpoints read only supported report
files under `data/experiments/<run-id>`. Architecture-comparison runs under the
configured model directory are also listed with their own metric and curve view.
Local runs are shared research records,
visible to signed-in users like database experiments; they are not user-private
prediction records. Checkpoints, raw images, and arbitrary filesystem paths are
not exposed. Compose mounts this directory read-only into the API. Registered
database experiments remain listed separately.

Checkpoints contain optimizer/RNG states, but automatic resume is not implemented
by this runner. An interrupted run preserves its completed checkpoints; do not
reuse its output directory or represent partial training as a completed run.
All run artifacts are ignored by Git under `data/experiments/`.
