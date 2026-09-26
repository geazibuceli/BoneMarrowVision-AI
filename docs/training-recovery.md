# Resume classifier training

New runs of `ml/scripts/train.py` atomically replace a `best.latest.pt` recovery
file at the end of every completed epoch. It includes current weights, AdamW,
learning-rate scheduler, mixed-precision scaler, Python/NumPy/PyTorch random
states, data-loader and sampler generators, history, early-stopping counters,
and a complete copy of the best checkpoint so far.

Resume using the same output path, frozen configuration, and verified splits:

```powershell
.venv/Scripts/python.exe ml/scripts/train.py --config models/my_run/convnext_tiny.yaml --data-dir data/experiments/resnet50_comparison --checkpoint models/my_run/convnext_tiny/best.pt --resume models/my_run/convnext_tiny/best.latest.pt --epochs 100
```

Replace `my_run` with a run created by the updated trainer. `--epochs 100` means
100 total epochs, not 100 additional epochs. An interrupted partial epoch is
repeated from the previous completed epoch. The first recovery file becomes
available only after epoch 1 finishes. An interruption during an atomic write
leaves the previous recovery file intact.

The CLI refuses to overwrite an existing output without `--resume`. Resume
checks training configuration, class order, split/manifest hashes, loader
configuration, device type and CUDA device count. Only total target epochs and
the optional numbered-checkpoint interval may change. It refuses completed
targets, runs that already reached early stopping, unverified splits, and old
checkpoints without the new recovery state. It does not download pretrained
weights during resume.

Workers restart each epoch so their random seeds are reproducible from saved
generators. Direct callers of `train_model` must use non-persistent workers to
resume. Keep the same hardware, runtime, dataset images, and deterministic
settings; the checks cannot guarantee identical floating-point results across
different library versions or devices. Dataset identity is based on recorded
split/manifest hashes, not rehashing every image at each epoch.

Keep adequate disk space: the rolling recovery file contains both current and
best training state, and its atomic replacement temporarily requires another
file of comparable size. Numbered snapshots, best checkpoints, and final
checkpoints continue to be produced as before. Inference should use `best.pt`.

Previously completed September 25 results are unchanged and do not gain recovery state retroactively.

## Pipeline recovery

New architecture comparisons and GAN experiments record completed stages in
`stages.json`. Run the original command with `--resume` and the same output
directory, inputs, configuration, architecture list, seed, and `--full` setting:

```bash
python ml/scripts/run_architecture_comparison.py --full --output-dir models/my-comparison --resume
python ml/scripts/run_gan_experiment.py --output models/my-gan-experiment --resume
```

Use the original `--data-dir` and other options if they differ from defaults.
Completed stages are reused only when their recorded output hashes match.
Changed inputs, missing artifacts, and legacy runs without a stage journal are
rejected rather than treated as a verified continuation. Run only one process
per output directory.

Interrupted classifiers resume from their `.latest.pt` checkpoint. Pipelines use
`--finalize-completed` to reconstruct final artifacts if interrupted after the
last epoch but before finalization. Evaluation and report generation follow
automatically. GAN evaluation is repeated on resume; completed training is reused.

New GAN checkpoints also restore generator/discriminator weights, both optimizer
states, RNG state, sampler/loader generators, fixed preview inputs, and history.
Recovery starts after the last completed epoch; unfinished epochs repeat. Legacy
GAN checkpoints lack complete recovery state and cannot be resumed this way.
Use the same device environment for reproducibility. Synthetic generation may
repeat after interruption with its fixed seed; existing output belongs to the
same journal-verified run. Resuming never activates a model in the application.
