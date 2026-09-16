---
name: bmv-inference-performance
description: Measure and improve latency, memory usage, and model loading in BoneMarrowVision PyTorch inference. Use for image-analysis performance issues without implying retraining.
---

# BoneMarrowVision inference performance

Read backend/app/inference.py, routers/predictions.py, and the modules used in ml/src/bone_marrow_vision/models and explainability. Identify the device, cache, locks, preprocessing, and Grad-CAM generation before proposing optimizations. Keep project content in English.

- Separate cold loading, warm-cache inference, and Grad-CAM generation. Record hardware, checkpoint, input size, repetition count, and concurrency for reproducible comparisons.
- Measure the bottleneck in the available environment. Without a compatible checkpoint, limit conclusions to inspection or isolated tests; do not invent performance gains.
- Preserve output classes and ordering, normalization, input size, and checkpoint integrity. Compare outputs before and after with tolerance appropriate to the numerical precision.
- Grad-CAM depends on gradients. Do not wrap the entire flow in inference_mode or remove locks without examining hooks, shared state, and concurrency.
- Consider memory per process and cache invalidation when switching models. More workers may duplicate models in memory instead of addressing the bottleneck.
- The API container uses CPU PyTorch. Do not automatically add CUDA, retraining, or architecture changes in response to a performance request.

Run relevant tests in backend/tests and ml/tests/test_explainability.py or test_checkpoints.py when those contracts change. Use temporary files for benchmarks and do not overwrite existing models or reports.

Deliver the change with measured comparisons where possible, output-quality checks, and measurement limitations. Label synthetic benchmarks as synthetic.
