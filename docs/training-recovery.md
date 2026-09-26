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

Architecture-comparison and GAN orchestration do not automatically resume their
whole sequence. Resume an interrupted classifier with the CLI above, then run
evaluation and analysis explicitly; an old runner status is not proof of
completion. GAN generator/discriminator resume remains outside this classifier
feature. Previously completed September 25 results are unchanged and do not
gain recovery state retroactively.

Regression tests interrupt a run during an epoch and compare its resumed
history and weights with uninterrupted training, including shuffled sampling,
dropout, random input transforms, worker processes, and balanced sampling.
