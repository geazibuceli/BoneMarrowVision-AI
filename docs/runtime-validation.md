# Runtime validation

Validation date: September 16, 2026. Tested the working tree based on `5da08ff`, including the pending history pagination and saved-result changes. These results describe a local smoke test, not a production load test or a new model evaluation.

## Browser and application flows

Playwright drove headless Microsoft Edge against the real API and frontend. The local run used an isolated SQLite database, fresh migrations, generated test credentials, and separate upload and temporary-input directories.

The following checks passed:

- Register an account, authenticate, and open the analysis workspace.
- Submit a valid image without an active model and display the API's error.
- Register the existing checkpoint, activate it, and preserve the registry through a backend restart.
- Upload a real local image, run inference, and display a successfully decoded authenticated Grad-CAM image.
- Open the saved result and reload its URL without losing the prediction.
- Remove the temporary original image after inference.
- Navigate between history pages, open a result, and return to the originating page. Thirty synthetic history records were inserted into the isolated database to exercise pagination; they were not additional model evaluations.
- Render result and history screens at 360 px without horizontal page overflow; inspect desktop and mobile screenshots.
- Reject an unsupported file type in the browser and reject corrupt image contents in the API.
- Redirect an expired session to sign-in.
- Deny another account access to the saved result and display its empty history.
- Complete these flows without uncaught browser JavaScript errors.

Visual inspection found that dates inherited the browser's language. The formatter now explicitly uses English while retaining the user's local time zone. The production build was verified with a `pt-BR` browser locale.

## Real inference measurements

The checkpoint was `best_model_40ep_full.pt`, EfficientNet-B0, 21 classes, with 224 x 224 model inputs. Its SHA-256 was `0df298d962506e86860eacf766ae13ac50992aad4d473024ec7b0c7fe3c7c37c`.

The native benchmark ran on Windows with an AMD Ryzen 7 9800X3D, Python 3.14.0, and PyTorch 2.11.0+cu128. Inference used the **CPU**, with eight PyTorch threads. The benchmark used one existing local image, six sequential predictions, and two concurrent calls.

| Measurement | Observed result |
| --- | ---: |
| Model loading after Python imports | 68.6 ms |
| First prediction after model loading, including Grad-CAM and overlay | 117.2 ms |
| Five subsequent complete predictions, median | 61.0 ms |
| Subsequent complete prediction range | 58.3-64.7 ms |
| Warm Grad-CAM generation component | 31.8-34.6 ms |
| Warm overlay writing component | 9.4-10.3 ms |
| Classification-only forward pass, five calls | 11.3-18.3 ms |
| Two concurrent complete calls, combined elapsed time | 219.2 ms |
| Process RSS before loading the checkpoint | 692.3 MiB |
| Sampled peak process RSS | 1,335.5 MiB |
| Final process RSS | 924.3 MiB |

All eight complete predictions agreed on the predicted class and confidence within `1e-6`. Grad-CAM hooks and the inference lock remained unchanged. Memory was sampled every 10 ms and includes the Python process and imported libraries; it is not GPU memory or model weight size.

The first browser upload after restarting the local API took **9.571 seconds** until the saved-result link appeared. This includes request handling, lazy initialization, and browser timing, unlike the in-process warm measurements. It should not be presented as equivalent to the 61 ms warm benchmark.

## Docker stack

Both images built successfully, and Compose configuration validation passed. Tests used the separate Compose project `bmv-runtime-validation`, separate PostgreSQL and upload volumes, generated temporary credentials, and loopback ports 18000 and 18080. Existing application volumes were not changed.

The following checks passed:

- Fresh PostgreSQL migrations and healthy database, backend, and frontend containers.
- Database readiness without an active model, with an explicit 503 response for inference.
- Model registration and real CPU inference through Nginx.
- Authenticated PNG delivery for Grad-CAM, with unauthenticated access rejected.
- Prediction history persisted in PostgreSQL.
- Production frontend sign-in, saved-result navigation, authenticated blob images, English dates, and a 360 px layout.
- Prediction, authentication token, and Grad-CAM survival after restarting all containers.
- Backend execution as non-root UID 10001.

The container used Python 3.11.16 and PyTorch 2.14.0+cpu. Its first upload after model registration took **0.808 seconds**, with the model already loaded during registration. The returned class and confidence matched the native run for this image. This is a separate runtime and timing boundary, so it is not a direct comparison with the native warm benchmark.

Test containers and their network were removed after validation. Isolated test volumes and built images were retained; Docker Desktop was left running. Local test databases, screenshots, scripts, and detailed measurements remain under the ignored `.audit-runtime/` directory and are not part of the versioned report. Playwright and psutil were installed only in the local virtual environment for validation.

## Scope and remaining limits

Frontend tests and lint passed after the date-format fix; the final production frontend was built and exercised in Docker. Backend and ML implementation were not modified in this validation pass.

The browser exercise covered Edge on this machine, not a cross-browser matrix or a full keyboard and screen-reader audit. The measurements used one image and two concurrent calls, not sustained traffic. Checkpoint integrity and functioning inference were verified, but these checks do not establish model accuracy across a dataset. Docker's Python 3.11 runtime was exercised; the native timing environment used Python 3.14 rather than the README's recommended Python 3.11.

Docker resolves some dependencies from version ranges. The versions above describe this build; future builds may resolve different versions. The first-request delay and process memory measurements provide a baseline for any later startup or capacity work; no performance improvement is claimed here.
