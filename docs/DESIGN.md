# Software Design Document

## 1. Design goals

-   Minimal implementation effort suitable for a semester project.
-   Clear separation between UI, API, media processing, and ML
    inference.
-   Reproducible model evaluation.
-   Understandable results with explicit uncertainty and limitations.
-   Safe handling of uploaded media.

## 2. Application screens

### 2.1 Upload screen

Elements: - Project title and short description. - Image/video file
picker and drag-and-drop area (optional). - Supported formats and upload
limits. - Analyze button. - Privacy and interpretation notice.

### 2.2 Processing state

Elements: - Selected file name, type, and size. - Progress indicator or
processing state. - Cancel/retry option only if supported by backend
behavior. - Clear error state for failed processing.

### 2.3 Results screen

Elements: - Result category: "Potentially manipulated," "No manipulation
detected by this model," or "Uncertain." - Model manipulation score,
labelled as a model score. - For video: number of sampled frames and a
compact frame-score view. - Brief explanation of what the score means
and does not mean. - Analyze another file action.

## 3. Backend module design

``` text
backend/app/
├── main.py                 # FastAPI app and middleware
├── routes/
│   └── analyze.py          # Upload/analyze endpoint
├── schemas/
│   └── result.py           # Request/response schemas
└── services/
    ├── validation.py       # File and media limits
    ├── image_processor.py  # Image decoding and transforms
    ├── video_processor.py  # Frame sampling
    ├── inference.py        # Model loading and prediction
    ├── aggregation.py      # Video score aggregation
    └── cleanup.py          # Temporary file cleanup
```

## 4. ML module design

``` text
ml/
├── prepare_data.py     # Dataset manifest and split creation
├── train.py            # Fine-tuning
├── evaluate.py         # Metrics and plots
├── inference.py        # Reusable prediction interface
└── config.py           # Model, image size, paths, thresholds
```

Keep dataset paths and model configuration outside source code where
practical. Record model architecture, input resolution, preprocessing,
dataset split, random seed, and checkpoint version.

## 5. Model design

Use a pretrained EfficientNet or ResNet from a maintained library.
Replace its final classification layer with a two-class output and
fine-tune on labelled authentic/manipulated samples. Begin with a small
controlled experiment and compare validation performance before
selecting a checkpoint.

Do not treat the model's raw sigmoid/softmax output as a calibrated
real-world probability. If calibration is not performed and validated,
call it a "model score."

## 6. Video design

-   Enforce a maximum duration and maximum frame count.
-   Sample frames at a fixed strategy, such as evenly spaced timestamps.
-   Reuse the image preprocessing and classifier.
-   Aggregate valid frame scores with a predetermined method (initially
    mean).
-   Preserve selected frame indices/timestamps for display.
-   Return an uncertain result if decoding fails or too few valid frames
    are available.

## 7. Decision design

Use two thresholds selected from validation data: - Score below lower
threshold: "No manipulation detected by this model." - Score between
thresholds: "Uncertain." - Score above upper threshold: "Potentially
manipulated."

Threshold selection should consider false-positive and false-negative
costs. The user interface must not label media as definitively real or
fake.

## 8. Privacy and security

-   Restrict upload size, extensions, and decoded media limits.
-   Do not trust client-provided MIME type alone.
-   Generate server-side temporary filenames.
-   Avoid executing uploaded content.
-   Clean up temporary files after processing.
-   Do not log raw media or unnecessary identifying metadata.
-   Do not retain uploads by default.

## 9. Accessibility and usability

Use clear labels, keyboard-accessible controls, readable contrast,
responsive layout, and plain-language explanations. Do not rely on color
alone to distinguish result categories.
