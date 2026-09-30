# DeepTrace — Technical Development Plan

**Repository:** `C0dEr8627/deeptrace`  
**Branch:** `main`  
**Status:** Technical implementation baseline  
**Project type:** Semester-scale academic prototype  
**Primary deliverable:** Browser-based image and video visual deepfake screening

This document is the implementation contract for developers and coding agents. Use the specified stack, model, interfaces, and processing rules as the default. If a dependency version or dataset constraint forces a change, document the reason and update this plan before changing architecture.

## 1. Product and engineering objective

DeepTrace accepts a still image or a short video, applies a learned visual classifier, and returns a manipulation score with an uncertainty-aware interpretation. The system is a screening aid, not a forensic verification tool. It does not establish provenance, identify a person, or prove that media is real or fake.

The first release is deliberately single-modal: visual image classification, reused on sampled video frames. Do not implement audio, speech/voice-cloning detection, metadata-based authenticity decisions, identity recognition, live-stream analysis, or temporal neural networks in the MVP.

## 2. Locked MVP technology

| Area | Technology | Implementation decision |
|---|---|---|
| Primary language | Python 3.11 | Backend and ML code |
| ML framework | PyTorch 2.x + torchvision 0.x | Training, transforms, checkpoint loading |
| Baseline model | EfficientNet-B0, ImageNet-pretrained | One binary classifier; do not add model ensembles |
| Numerical/data utilities | NumPy, pandas | Arrays, manifests and experiment records |
| Evaluation | scikit-learn | Classification metrics, ROC/PR curves, confusion matrix |
| Image/video decoding | OpenCV 4.x + FFmpeg | Decode images and sample video frames |
| API | FastAPI + Pydantic v2 + Uvicorn | REST API and typed request/response schemas |
| Multipart uploads | python-multipart | FastAPI upload parsing |
| Frontend language | TypeScript | Browser application |
| Frontend | React 18+ with Vite | Single-page application |
| Frontend styling | CSS modules or project-level CSS | Responsive, accessible UI; no UI framework required for MVP |
| HTTP client | Native Fetch API | Typed API client |
| Frontend tests | Vitest + React Testing Library | Component and workflow tests |
| Backend tests | pytest + httpx | Unit and API tests |
| Formatting/linting | Ruff (Python), ESLint + Prettier (frontend) | Consistent code quality |
| Version control | Git + GitHub | Main branch and reviewed commits |
| Initial runtime | Local development machine | No cloud infrastructure required for MVP |

Use compatible stable releases and pin exact resolved versions in `backend/requirements.txt` (or a lock file) and the frontend lock file. Avoid unpinned dependencies in reproducible runs. CUDA is optional; CPU must remain supported for inference and basic tests.

## 3. Model and algorithm specification

### 3.1 Classification target

The model predicts two training classes:

- `real`: authentic media examples from the selected dataset.
- `fake`: media manipulated by one or more documented deepfake methods in that dataset.

The class mapping must be centralized in configuration and stored with the checkpoint. Do not infer label meaning from folder order. Dataset-specific label normalization belongs in the data-preparation stage.

### 3.2 Architecture

Use **torchvision EfficientNet-B0 with ImageNet weights** as the only MVP model.

- Load the torchvision pretrained EfficientNet-B0 weights.
- Replace the final classifier linear layer with a single output logit.
- Train with `BCEWithLogitsLoss` for binary classification.
- Apply `sigmoid(logit)` at inference to produce a bounded model score for the configured `fake` class.
- Use transfer learning: initially freeze the feature extractor and train the replacement head; then optionally unfreeze the final feature stages and fine-tune with a lower learning rate if validation results justify it.
- Save a state-dict checkpoint, class mapping, architecture name, preprocessing configuration, training seed, and selected validation threshold metadata.

Do not call the sigmoid output a calibrated probability. It is a model score unless calibration is separately evaluated.

### 3.3 Input preprocessing

All images and sampled video frames must use the same preprocessing pipeline:

1. Decode safely and reject empty/corrupt inputs.
2. Convert BGR (OpenCV) to RGB.
3. Resize to **224 × 224** using the selected torchvision transform.
4. Convert to a float tensor in `[0, 1]`.
5. Normalize using ImageNet channel statistics: mean `[0.485, 0.456, 0.406]`, standard deviation `[0.229, 0.224, 0.225]`.
6. Add a batch dimension and run inference under `torch.inference_mode()`.
7. Return the fake-class sigmoid score as a Python float.

For the baseline, classify the full decoded image/frame. Do not add face detection or face alignment until baseline performance is measured; these introduce an additional model, failure modes, and preprocessing decisions. Keep preprocessing implemented once and imported by both evaluation and serving code to prevent train/serve skew.

### 3.4 Training baseline

Initial reproducible baseline configuration (tune only through recorded experiments):

| Parameter | Initial value |
|---|---|
| Input size | 224 × 224 RGB |
| Batch size | 32, reduce if memory constrained |
| Optimizer | AdamW |
| Head learning rate | 1e-3 |
| Fine-tuning learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Epoch ceiling | 10 head-training epochs, then up to 5 fine-tuning epochs if justified |
| Loss | BCEWithLogitsLoss |
| Selection criterion | Validation PR-AUC; also inspect ROC-AUC and class-wise recall |
| Early stopping | Stop after 3 epochs without validation improvement |
| Seed | Fixed and recorded for each experiment |
| Checkpoint | Best validation checkpoint, not final epoch by default |

These are starting values, not guaranteed optimal hyperparameters. Track train/validation loss and metrics per epoch. Handle class imbalance using a documented sampler or positive-class weighting only when measured imbalance warrants it. Do not oversample or augment validation/test data.

Training augmentation should be conservative and label-preserving: horizontal flip and mild brightness/contrast changes may be used on training images. Avoid aggressive blur, compression, cropping, or geometric transforms until their impact is tested, since manipulation artifacts may be altered. Validation and test transforms are deterministic.

### 3.5 Dataset protocol

Select a dataset whose license/access conditions permit the intended academic use. Candidate datasets may include FaceForensics++ or another approved deepfake image/video corpus, subject to availability and license review. The final dataset choice must be recorded in the experiment manifest; this plan does not assume that a particular dataset has been downloaded or approved.

Maintain a CSV/Parquet manifest with at least:

`sample_id, path_or_source_id, label, dataset, source_video_id, identity_group, split`

- Keep source videos, identities, and derived frames grouped within a single split whenever metadata supports it.
- Split into train, validation, and held-out test partitions before extracting or augmenting frames.
- Prevent near-duplicate leakage; document grouping limitations if identity metadata is unavailable.
- Keep datasets outside Git. Commit only scripts, manifests without sensitive local paths where appropriate, and non-restricted aggregate results.
- Record dataset version, acquisition date, license, class counts, split seed, and exclusions.

A random frame-level split is prohibited when multiple frames originate from the same source video.

## 4. Video processing algorithm

Video support reuses the trained image classifier; it is not temporal modeling.

### 4.1 Initial processing limits

Store limits in one backend configuration module so they can be adjusted without changing the API contract:

- Maximum upload size: **100 MB**.
- Maximum duration: **60 seconds**.
- Maximum sampled frames: **16**.
- Sampling: evenly spaced timestamps across the decodable video duration, excluding duplicate timestamps.
- Maximum decoded frame dimension: resize frames to a maximum side of **1280 px** before classifier preprocessing.
- Unsupported, malformed, empty, or over-limit media must return a typed client error without crashing the worker.

These are conservative initial application limits for a semester prototype, not performance guarantees. Validate them on available hardware and revise them only with documented measurements.

### 4.2 Frame sampling and aggregation

1. Validate extension and detected media type; do not trust the filename alone.
2. Open with OpenCV/FFmpeg and read duration, frame rate, width, and height where available.
3. Generate up to 16 evenly spaced timestamps from the usable duration.
4. Seek/decode each timestamp; discard failed or empty frames and record the timestamp of each accepted frame.
5. Run each frame through the shared EfficientNet-B0 inference pipeline.
6. Aggregate valid frame fake scores using the **arithmetic mean** as the initial video score.
7. Return the number of requested and successfully analyzed frames, aggregate score, and per-frame timestamp/score entries.
8. If no frame can be decoded, return a processing error. If fewer than 4 frames are valid, return an `uncertain` outcome with a reason rather than presenting a normal video classification.

Do not average logits and sigmoid scores together; aggregate only the per-frame fake-class scores. Mean aggregation can dilute short manipulated segments and does not model temporal inconsistencies. State this limitation in the UI and report.

## 5. Decision policy and result schema

Let `s` be the fake-class model score in `[0, 1]`. Use two thresholds, `T_low` and `T_high`, selected on validation data and saved in model configuration:

- `s < T_low`: `no_manipulation_detected`
- `T_low <= s <= T_high`: `uncertain`
- `s > T_high`: `potentially_manipulated`

Choose thresholds using validation data to make the trade-off between false positives and false negatives explicit. Do not tune thresholds on the held-out test set or to improve a demo. If suitable thresholds cannot be justified, return `uncertain` rather than inventing cutoffs.

Use careful user-facing text:
- “No manipulation detected by this model”
- “Uncertain — model output is inconclusive”
- “Potentially manipulated — further verification recommended”

The API response should be versioned and consistent. Suggested Pydantic response:

```json
{
  "analysis_id": "uuid",
  "media_type": "image",
  "status": "completed",
  "prediction": "uncertain",
  "fake_score": 0.52,
  "score_type": "uncalibrated_model_score",
  "model": "efficientnet_b0",
  "frames_analyzed": null,
  "frame_results": [],
  "warnings": ["This result is an automated screening signal, not forensic proof."]
}
```

For video, `frames_analyzed` is an integer and `frame_results` contains timestamp and score objects. For images, `frames_analyzed` is null and `frame_results` is empty. Use explicit error response models with stable machine-readable codes.

## 6. Backend design

### 6.1 Responsibilities

FastAPI owns HTTP validation, temporary file lifecycle, media routing, orchestration, and response serialization. ML modules own transforms, checkpoint loading, prediction, and model metadata. Keep HTTP concerns out of training and model code.

Suggested endpoints:

| Method | Path | Responsibility |
|---|---|---|
| GET | `/api/health` | Process health and model-loaded status; no sensitive details |
| GET | `/api/model-info` | Public model name, supported media, score semantics |
| POST | `/api/analyze` | Multipart upload; analyze one image or video |

`POST /api/analyze` accepts one `file` field. Return HTTP 200 for completed analysis, 400 for unsupported media, 413 for size/duration limits, and 422 for corrupt or unprocessable media. Use 500 only for unexpected server errors; never return stack traces to clients.

### 6.2 Inference lifecycle

- Load the checkpoint once at application startup and reuse the model instance.
- Set model to `eval()`; use `torch.inference_mode()`.
- Select CUDA when available and configured; otherwise CPU.
- Avoid loading a new model for each request.
- Run CPU/GPU inference outside the async event loop using an appropriate threadpool/executor.
- Begin with single-process local serving. Do not claim multi-user concurrency or production-scale throughput.
- Add a configurable inference lock if the selected device/runtime requires serialized access.
- Keep training scripts separate from the API process.

### 6.3 Upload safety and privacy

- Enforce size limits while streaming the upload, not only after fully reading it.
- Generate a random temporary filename; never use the user-provided filename as a filesystem path.
- Validate extension, MIME hint, and actual decoder result.
- Store temporary media outside static/public directories.
- Delete temporary media in a `finally` cleanup path on success and failure.
- Do not log image/video bytes, local absolute paths, or unnecessary personal metadata.
- Do not persist uploads or results in a database for the MVP.
- Configure CORS only for the development frontend origin.
- Limit concurrent analysis requests to avoid memory exhaustion.

## 7. Frontend design

Build a React + TypeScript single-page interface with these states:

1. **Idle:** file picker, supported formats and limits.
2. **Selected:** filename, type, size, remove/replace action.
3. **Analyzing:** progress indicator and duplicate-submit prevention.
4. **Completed:** result category, score with explicit “model score” wording, warning, and video frame summary.
5. **Error:** readable message and retry path.

Implementation requirements:
- Use a typed API module around native `fetch` and `FormData`.
- Do not manually set the multipart `Content-Type`; let the browser add its boundary.
- Validate obvious client-side size/type issues for usability, while treating backend validation as authoritative.
- Render server errors using stable error codes/messages; do not display raw stack traces.
- Make keyboard navigation, visible focus, semantic labels, and responsive layouts part of the first UI pass.
- Do not imply that a higher score is a calibrated probability.
- Do not add authentication, history, cloud storage, or a database in the MVP.

## 8. Repository layout

Use a small monorepo layout:

```text
deeptrace/
├── README.md
├── DEVELOPMENT_PLAN.md
├── docs/
│   ├── SRS.md
│   ├── ARCHITECTURE.md
│   ├── DESIGN.md
│   └── TEST_PLAN.md
├── backend/
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   │   ├── routes.py
│   │   │   └── schemas.py
│   │   ├── core/
│   │   │   └── config.py
│   │   ├── services/
│   │   │   ├── image_service.py
│   │   │   └── video_service.py
│   │   └── ml/
│   │       ├── model.py
│   │       ├── transforms.py
│   │       └── predictor.py
│   └── tests/
├── frontend/
│   ├── package.json
│   └── src/
│       ├── api/
│       ├── components/
│       ├── pages/
│       └── types/
├── ml/
│   ├── configs/
│   ├── data/
│   │   ├── build_manifest.py
│   │   └── split_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── export_model.py
├── artifacts/
│   └── .gitkeep
└── .gitignore
```

Keep raw media, virtual environments, caches, logs, secrets, and large checkpoints out of Git. Store model artifacts locally for development; decide on Git LFS or external artifact storage only if repository delivery requires distributing a checkpoint.

## 9. Milestones and implementation order

| Milestone | Concrete output | Exit criteria |
|---|---|---|
| M0 — Repository foundation | Python/frontend skeleton, config, ignore rules, lint/test commands | Clean install and app health endpoint |
| M1 — Data pipeline | Dataset documentation, manifest builder, grouped split, integrity checks | No known source-video leakage; class counts recorded |
| M2 — Image baseline | EfficientNet-B0 training, checkpoint, inference CLI | Reproducible image score from saved checkpoint |
| M3 — Evaluation | Held-out metrics, confusion matrix, threshold selection record | Metrics and limitations documented |
| M4 — Image API | Upload validation, image decode, inference, typed response | Valid and invalid image API tests pass |
| M5 — Video pipeline | Duration validation, bounded sampling, frame inference, mean aggregation | Repeatable bounded video analysis |
| M6 — Frontend | Upload, analyzing, results, errors, accessibility | Browser workflow works against local API |
| M7 — Hardening/demo | Cleanup, edge-case tests, demo media, report evidence | Acceptance checklist passes |

Implement and validate the ML pipeline independently before integrating it with FastAPI. Complete image inference end-to-end before video. Do not start optional extensions until M0–M6 are stable.

## 10. Evaluation and reproducibility

Evaluate the final selected checkpoint once on a held-out test split. Report:

- Precision, recall, F1-score for both classes.
- ROC-AUC and PR-AUC.
- Confusion matrix, false-positive rate, and false-negative rate.
- Accuracy as a supplementary metric.
- Per-class support and dataset/source composition.
- Image results separately from video-level results.

Use scikit-learn with a fixed class order. Save predictions and labels in a machine-readable file, along with model checkpoint hash, config, seed, dataset manifest version, and software versions. Do not report only accuracy. Where possible, evaluate on a second dataset or unseen manipulation/compression condition and describe the result as a limited generalization check.

## 11. Test strategy

### Unit tests
- Label mapping and manifest validation.
- RGB conversion, tensor shape, dtype, and normalization.
- Checkpoint loading and finite score range.
- Threshold boundary behavior.
- Video timestamp generation, duplicate removal, and score aggregation.
- API schema serialization and error-code mapping.

### Integration tests
- Valid image upload returns a completed response.
- Valid short video returns bounded frame results.
- Oversized, unsupported, corrupt, empty, and over-duration uploads are rejected.
- Decoder failure and model-loading failure do not leak temporary files.
- Frontend handles success, API errors, network failure, and repeated submission.

Use small synthetic fixtures for routine CI tests. Keep real datasets and large model artifacts out of CI. Add a manual smoke test with representative real and manipulated samples, clearly marked as a demonstration rather than model validation.

## 12. Operational configuration

Keep adjustable values centralized, not scattered through route handlers:

- Upload size, duration, frame cap, minimum valid frames.
- Model artifact path, architecture identifier, device selection.
- Image dimensions and normalization constants.
- Decision thresholds and score semantics.
- CORS development origins and concurrency limit.

Fail fast at startup if the checkpoint is missing, incompatible, or has mismatched class/preprocessing metadata. Health status should distinguish process availability from model readiness.

## 13. Risks and explicit non-goals

| Risk | Engineering response |
|---|---|
| Dataset leakage | Group by source video/identity before split; document missing metadata |
| Domain shift | Hold out sources and report dataset-specific limitations |
| Model learns compression/background shortcuts | Inspect errors and test across available compression/source conditions |
| CPU inference is slow | Keep input size and frame count bounded; measure latency |
| Video frame mean misses short edits | Explain limitation; do not describe aggregation as temporal detection |
| Uncalibrated score is misunderstood | Label as model score; use uncertain band and explanatory notice |
| Upload security/privacy | Stream limits, random temp paths, cleanup, no media logging/persistence |
| Limited compute/time | One pretrained EfficientNet-B0; small controlled experiments |

Explicitly out of scope: audio detection, lip-sync analysis, face identity matching, facial landmark heuristics as authenticity proof, temporal CNN/Transformer/LSTM, ensemble fusion, blockchain/provenance verification, user accounts, persistent upload history, and production deployment scaling.

## 14. Definition of done

The MVP is complete when:

- The dataset source, access terms, class mapping, and split protocol are documented.
- The EfficientNet-B0 checkpoint loads with its preprocessing metadata and returns finite fake-class scores.
- Held-out evaluation results and threshold rationale are recorded.
- The image API and browser flow work for valid and invalid inputs.
- Video support, if included in the submitted scope, uses bounded repeatable sampling and documented mean aggregation.
- Uploads are cleaned up on all normal and error paths.
- UI wording clearly distinguishes a model screening score from proof or calibrated probability.
- Tests, limitations, and a reproducible demonstration procedure are available in the repository.

**Change policy:** Treat the model, preprocessing, output semantics, and API contract above as the baseline. A coding agent must not silently substitute architectures, datasets, libraries, or algorithms. Propose material changes with their rationale, trade-offs, and documentation updates before implementation.
