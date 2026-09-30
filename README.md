# DeepTrace

**AI-Based Deepfake Detection System**

DeepTrace is a semester-scale research prototype for identifying signs of facial
manipulation in uploaded images and videos using a fine-tuned pretrained
image classifier.

> **Project status:** Proposed documentation and implementation plan.
> Features described here are intended scope, not claims about an
> already completed system.

## Objective

Build a minimal web application that accepts an image or video, analyzes
visual content, and reports a manipulation-likelihood score with an
uncertainty-aware result.

## Initial scope

-   Image upload and visual classification.
-   Video upload, frame sampling, frame-level classification, and score
    aggregation.
-   A simple results interface showing the prediction, model score, and
    sampled frames.
-   Evaluation using held-out data and standard classification metrics.

## Out of scope for the first version

Audio or voice-cloning analysis, advanced temporal neural networks,
metadata-based authenticity decisions, real-time monitoring, identity
recognition, and claims of forensic certainty.

## Proposed technology

-   **ML:** Python, PyTorch, torchvision, scikit-learn
-   **Media processing:** OpenCV, FFmpeg
-   **API:** FastAPI
-   **Frontend:** React
-   **Model:** Pretrained EfficientNet or ResNet, fine-tuned for binary
    classification
-   **Storage:** Temporary local storage for prototype uploads; database
    only if required

## High-level workflow

1.  User uploads image/video.
2.  Backend validates the file and preprocesses it.
3.  Image is classified directly; video is sampled into frames.
4.  The classifier returns scores for image/frame inputs.
5.  Video scores are aggregated using a documented rule.
6.  The interface displays a result with a caveat that model output is
    not proof.

## Responsible interpretation

The output is a model score, not a guaranteed probability that media is
fake. Performance may vary with compression, resolution, lighting,
manipulation technique, and dataset distribution. The application should
use "Potentially manipulated," "No manipulation detected by this model,"
and "Uncertain" rather than asserting authenticity.

## Suggested repository structure

``` text
DeepTrace/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── routes/
│   │   ├── services/
│   │   └── schemas/
│   ├── models/
│   └── tests/
├── frontend/
├── ml/
│   ├── data/
│   ├── notebooks/
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── docs/
│   ├── SRS.md
│   ├── ARCHITECTURE.md
│   ├── DESIGN.md
│   ├── TEST_PLAN.md
│   └── PROJECT_PLAN.md
└── README.md
```

## Run status

Implementation commands and deployment instructions should be added
after the development environment and actual implementation are
finalized.
