# DeepTrace

**AI-Based Deepfake Detection System**

DeepTrace is a semester-scale research prototype for screening uploaded images and videos for visual signs of manipulation using a fine-tuned pretrained image classifier. Its output is a model score, not forensic proof or a calibrated probability.

> **Status:** Repository foundation and dataset-preparation utilities are implemented. Model training, inference integration, and video analysis are not yet implemented.

## MVP scope

- Image-first visual classification, followed by sampled-frame video analysis.
- EfficientNet-B0 transfer learning with PyTorch and torchvision.
- FastAPI backend and React + TypeScript + Vite frontend.
- Held-out evaluation using precision, recall, F1, ROC-AUC, PR-AUC, and confusion matrix.
- Uncertainty-aware result wording; no claims of forensic certainty.

Audio analysis, temporal neural networks, identity recognition, metadata-based authenticity decisions, accounts, persistent history, and database storage are out of scope for the MVP.

## Technology

- **Language:** Python 3.11 and TypeScript
- **ML:** PyTorch 2.x, torchvision, EfficientNet-B0, scikit-learn, NumPy, pandas
- **Media:** OpenCV and FFmpeg
- **API:** FastAPI, Pydantic v2, Uvicorn
- **Frontend:** React 18+, Vite, native Fetch API, CSS
- **Tests and quality:** pytest, httpx, Vitest, React Testing Library, Ruff, ESLint, Prettier
- **Database:** None for the MVP; uploads are temporary and results are returned directly to the client.

See [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) for the technical implementation contract.

## Dataset preparation

No dataset is bundled. Dataset access and usage terms must be reviewed before downloading or using any media. Preparation utilities and the canonical manifest format are documented in [ml/data/README.md](ml/data/README.md). The manifest validator and group-aware splitter are available under `ml/data/` to reduce source leakage between train, validation, and test sets.

## Repository structure

```text
deeptrace/
├── backend/
│   ├── app/{api,core,ml,services}/
│   ├── tests/
│   ├── requirements.txt
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.ts
├── ml/
│   └── data/
│       ├── README.md
│       ├── validate_manifest.py
│       ├── split_dataset.py
│       └── test_split_dataset.py
├── artifacts/
├── DEVELOPMENT_PLAN.md
└── README.md
```

## Local development

See [backend/README.md](backend/README.md) for backend setup. For the frontend:

```bash
cd frontend
npm install
npm run dev
```

## Responsible interpretation

DeepTrace is a research screening aid. Scores are uncalibrated model outputs unless calibration is independently demonstrated. Performance can vary with compression, resolution, lighting, manipulation method, and dataset distribution. Interpret results as signals for further review, not proof that media is authentic or manipulated.
