# System Architecture

## 1. Architecture overview

The prototype follows a modular client-server architecture. The frontend
manages uploads and results; the backend validates media, invokes
preprocessing and inference, and returns a structured response.

``` mermaid
flowchart TD
    A[User] --> B[React Web Interface]
    B --> C[FastAPI]
    C --> D[File Validation]
    D --> E{Media Type}
    E -->|Image| F[Image Preprocessing]
    E -->|Video| G[Video Frame Sampling]
    F --> H[PyTorch Classifier]
    G --> H
    H --> I[Score Aggregation]
    I --> J[Decision and Uncertainty Rules]
    J --> K[API Response]
    K --> B
```

## 2. Components

### 2.1 React frontend

Responsibilities: - Select and upload a supported file. - Display
validation errors and analysis progress. - Render prediction, model
score, and supporting sampled-frame information. - Display limitations
and uncertainty messaging.

### 2.2 FastAPI backend

Responsibilities: - Expose analysis and health endpoints. - Validate
file extension, MIME type, size, and video duration where applicable. -
Coordinate preprocessing and inference. - Return structured responses
and safe error messages. - Ensure temporary files are cleaned up.

### 2.3 Preprocessing service

**Image path:** decode image, convert color format consistently,
resize/crop as required, and normalize according to the model's expected
preprocessing.

**Video path:** decode video using OpenCV or FFmpeg, sample a bounded
number of frames, and preprocess each frame using the same image
pipeline. Face detection/cropping may be added if the chosen model and
dataset pipeline support it consistently.

### 2.4 Inference module

Loads the fine-tuned pretrained model once per backend process where
practical. It performs inference without gradient calculation and
returns a score for each image or sampled frame.

### 2.5 Aggregation and decision module

For video, a simple initial aggregation method such as the mean of valid
frame scores can be used. The method must be fixed before evaluation and
documented. A threshold and uncertainty band should be selected using
validation data, not chosen to make demo results look favorable.

### 2.6 Dataset and training pipeline

A separate offline workflow prepares labelled data, creates source-aware
train/validation/test splits, fine-tunes the classifier, and evaluates
the saved model. Raw datasets should not be bundled with the application
repository unless their licenses permit it.

## 3. Data flow

1.  Frontend submits media to the API.
2.  Backend checks size, format, and processing limits.
3.  Media is decoded and preprocessed.
4.  Image input is classified once; video input is sampled and
    classified frame by frame.
5.  Scores are aggregated and passed through the decision rules.
6.  Backend returns a result payload.
7.  Frontend displays the result and a limitation notice.
8.  Temporary media and extracted frames are deleted according to the
    configured retention policy.

## 4. Proposed API response

``` json
{
  "analysis_id": "generated-id",
  "media_type": "video",
  "label": "uncertain",
  "score": 0.63,
  "score_type": "model_manipulation_score",
  "frames_analyzed": 16,
  "evidence": {
    "sampled_frame_scores": [0.42, 0.63, 0.71]
  },
  "notice": "Automated screening result; not proof of manipulation."
}
```

The values above are illustrative, not actual model output. The
implementation should define whether a higher score indicates stronger
evidence of manipulation and keep that convention consistent.

## 5. Deployment view

For the semester prototype, deploy the React frontend and FastAPI
backend locally or on one modest server. Keep model weights in a
controlled model directory. GPU support is optional. Containerization
can be added if it simplifies reproducibility.

## 6. Design decisions

-   Reuse one image classifier for image and sampled video-frame
    inference.
-   Prefer transfer learning over training from scratch.
-   Bound video duration, frame count, and upload size.
-   Keep training/evaluation separate from online inference.
-   Avoid audio, complex temporal models, and provenance claims in the
    first release.

## 7. Known limitations

The architecture detects learned visual patterns rather than proving
media history. It may fail on unseen manipulation methods, heavily
compressed media, non-face images, or content outside the training
distribution. Frame averaging can also hide a short manipulated segment;
this should be documented and considered during evaluation.
