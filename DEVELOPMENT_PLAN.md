# DeepTrace — Development Plan

**Project:** DeepTrace — AI-Based Deepfake Detection System  
**Repository:** `C0dEr8627/deeptrace`  
**Target branch:** `main`  
**Document status:** Proposed implementation plan  
**Project type:** Semester-scale academic prototype

## 1. Purpose

This document defines a practical, incremental plan for developing DeepTrace. The priority is to deliver a working and demonstrable image-and-video screening prototype without expanding into a large multimodal forensic platform.

The plan is intentionally lightweight. Complete the image workflow first, then extend the same classifier to sampled video frames. Audio analysis, advanced temporal networks, and metadata-based authenticity decisions are outside the minimum viable product (MVP).

## 2. Project objective

Develop a web application that accepts an image or video, analyzes visual content with a fine-tuned pretrained image classifier, and presents a manipulation score with an uncertainty-aware interpretation.

DeepTrace provides an automated screening signal. It must not claim to prove that media is authentic or manipulated, and its model score must not be described as a real-world probability unless calibration has been evaluated.

## 3. MVP scope

### Included
- Upload one supported image or video.
- Validate file type, size, and video processing limits.
- Preprocess images consistently with the selected model.
- Fine-tune a pretrained CNN (initial candidate: EfficientNet or ResNet).
- Classify images.
- Sample a bounded number of video frames and classify each with the image model.
- Aggregate valid frame scores using a documented method.
- Display a result category, model score, and basic frame-level evidence for video.
- Evaluate with held-out data and report relevant metrics.
- Document privacy, uncertainty, and known limitations.

### Deferred
- Audio and voice-cloning detection.
- Dedicated temporal Transformer, LSTM, or 3D CNN.
- Multi-model score fusion.
- Metadata/provenance as a classifier signal.
- Real-time analysis, user accounts, and production-scale infrastructure.
- Forensic certainty or identity recognition.

## 4. Proposed technology

| Layer | Proposed technology | Purpose |
|---|---|---|
| Language | Python | ML and backend |
| ML framework | PyTorch, torchvision | Fine-tuning and inference |
| Media processing | OpenCV, FFmpeg | Image decoding and video frame sampling |
| Evaluation | scikit-learn | Metrics and evaluation utilities |
| API | FastAPI | Upload and analysis endpoints |
| Frontend | React | Upload and results interface |
| Version control | Git and GitHub | Source and change tracking |

Use a local development environment initially. GPU acceleration is helpful but not a prerequisite for all development; training and inference speed will depend on available hardware.

## 5. Development phases

The sequence below is milestone-based rather than tied to fixed calendar weeks. Map the milestones to the remaining semester after confirming deadlines and team availability.

| Phase | Work | Completion milestone |
|---|---|---|
| 1. Scope and setup | Confirm supported formats, size/duration limits, repository structure, and dataset access. Establish a minimal Python environment. | Requirements and project skeleton are ready. |
| 2. Dataset preparation | Select a permitted dataset, inspect labels, create a manifest, and split by source/identity where metadata allows. | Reproducible train/validation/test split exists. |
| 3. Image model baseline | Load a pretrained CNN, replace its classifier head, fine-tune on the training split, and evaluate on validation/test data. | Image inference returns a score and documented metrics. |
| 4. Video pipeline | Sample a limited number of frames, reuse image preprocessing and inference, and aggregate frame scores. | A video produces a bounded, repeatable analysis result. |
| 5. Backend API | Implement upload validation, preprocessing orchestration, inference calls, response schemas, and cleanup. | API handles valid and invalid inputs safely. |
| 6. Frontend | Build upload, processing, error, and result views; connect to the API. | A user can complete the end-to-end workflow in the browser. |
| 7. Integration and testing | Test image/video paths, errors, limits, and repeatability; fix integration issues. | MVP acceptance criteria pass. |
| 8. Evaluation and submission | Record metrics, test conditions, limitations, screenshots/demo, and project report material. | Reproducible results and final demonstration are ready. |

**Schedule guidance:** If time is limited, prioritize phases 1–3 and 5–8 for a reliable image MVP. Treat video support as the next increment and avoid beginning optional research extensions until the core workflow is stable.

## 6. Implementation workflow

### 6.1 Dataset and experiment
1. Review dataset access conditions and record the source and permitted use.
2. Keep raw datasets outside the application repository unless redistribution is explicitly allowed.
3. Create a manifest containing file path/reference, label, source, and identity/video grouping where available.
4. Avoid leakage between splits by keeping related identities, source videos, and near-duplicate frames in one split wherever the dataset supports this.
5. Record preprocessing, random seed, model version, hyperparameters, and checkpoint identifier.

### 6.2 Image classifier
1. Start with one pretrained architecture (EfficientNet or ResNet; choose one for the baseline).
2. Apply the model's expected resizing, color conversion, and normalization.
3. Replace the final classification layer for the project labels.
4. Fine-tune using the training split and monitor validation performance.
5. Save the selected checkpoint and its configuration.
6. Evaluate once on the held-out test split and preserve the results.

Do not select a model based only on training accuracy. Report failures and limitations as well as successful results.

### 6.3 Video screening
1. Validate duration and decodeability.
2. Sample a configured maximum number of frames at a repeatable interval or evenly spaced timestamps.
3. Run each valid frame through the same image pipeline.
4. Aggregate frame scores using a fixed initial method, such as the mean.
5. Return the number of frames analyzed and selected frame scores/timestamps where useful.
6. Mark the result uncertain or return a safe processing error if insufficient frames are available.

Frame aggregation is a lightweight baseline, not true temporal modeling. It may miss short manipulated segments or inconsistencies between frames.

### 6.4 API and interface
The backend should expose a small analysis endpoint, such as `POST /api/analyze`, and optionally `GET /api/health`. Return a consistent response containing media type, label, model score, score type, frames analyzed (for video), and a clear limitation notice.

The frontend should present:
- File picker and supported format/size guidance.
- Selected file information and processing state.
- Result category and model score.
- Frame-level information for video, if implemented.
- Clear errors and a notice that automated output is not proof.

## 7. Result interpretation

Use uncertainty-aware labels rather than definitive “real” or “fake” claims. Configure lower and upper decision thresholds using validation data:

- Below the lower threshold: **No manipulation detected by this model**
- Between thresholds: **Uncertain**
- Above the upper threshold: **Potentially manipulated**

The thresholds must be selected and documented using validation results, not adjusted to favor a demonstration. Unless calibration is separately performed and validated, display the output as a **model manipulation score**, not a literal probability of fakery.

## 8. Evaluation plan

Evaluate on held-out data, with source/identity-aware separation wherever possible. Report:
- Precision, recall, and F1-score.
- ROC-AUC and PR-AUC.
- False-positive and false-negative rates.
- Confusion matrix.
- Accuracy as a supplementary metric.

Where resources permit, include a small generalization check across a different dataset, manipulation method, compression level, or resolution. Clearly state the test population and conditions; results from one dataset do not establish real-world reliability.

## 9. Repository structure

Keep the initial structure simple and expand only when implementation requires it.

```text
DeepTrace/
├── README.md
├── DEVELOPMENT_PLAN.md
├── docs/
│   ├── SRS.md
│   ├── ARCHITECTURE.md
│   ├── DESIGN.md
│   └── TEST_PLAN.md
├── backend/
│   └── app/
├── frontend/
└── ml/
    ├── prepare_data.py
    ├── train.py
    ├── evaluate.py
    └── inference.py
```

Do not commit dataset media, uploaded user files, local environments, secrets, or large model checkpoints without an explicit storage and licensing decision.

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Limited time or compute | Transfer learning, one baseline model, bounded experiments and video frames. |
| Dataset access or license restrictions | Verify terms before use; document sources and avoid unauthorized redistribution. |
| Data leakage | Split by source/identity where possible and keep related frames grouped. |
| Poor performance on unseen fakes | Evaluate generalization where feasible and state limitations clearly. |
| Slow video processing | Restrict file size, duration, resolution, and sampled frame count. |
| Misleading confidence | Use validation-selected thresholds, uncertainty labels, and careful score wording. |
| Privacy concerns | Minimize upload retention, use temporary server-side files, and avoid logging media. |
| Integration delays | Validate the model independently before connecting API and frontend. |

## 11. MVP acceptance checklist

- [ ] Project scope, formats, and processing limits are documented.
- [ ] Dataset source, permitted use, labels, and split method are recorded.
- [ ] Image classifier can load the saved checkpoint and return a finite score.
- [ ] Held-out evaluation metrics are recorded.
- [ ] Video sampling is bounded and repeatable.
- [ ] Video scores are aggregated by a documented method.
- [ ] API validates inputs and handles decode/inference failures safely.
- [ ] Frontend supports upload, processing, errors, and results.
- [ ] Uncertainty and model limitations are visible to users.
- [ ] Temporary media is cleaned up according to the configured policy.
- [ ] Final test evidence, known limitations, and demo steps are documented.

## 12. Change control

Keep the MVP scope stable during implementation. Any proposed extension should be assessed against remaining time, compute, measurable benefit, and impact on testing. Update this plan when scope or major technical decisions change.

**Definition of done:** DeepTrace is considered complete for the semester prototype when a user can submit a supported image and, if video phase is completed, a supported video; receive a consistent uncertainty-aware result; and the team can demonstrate the workflow and explain its evaluation and limitations.
