# DeepTrace Backend

FastAPI service for the DeepTrace visual media screening prototype.

## Requirements

- Python 3.11
- pip and venv
- PyTorch-compatible CPU or NVIDIA CUDA environment (CPU is supported for the initial API scaffold)

## Local setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
$env:PYTHONPATH = "backend"
uvicorn app.main:app --reload --app-dir backend
```

Open http://127.0.0.1:8000/api/health to confirm the process responds. The model is not integrated yet, so model_ready is currently false by design.

## Tests and lint

```powershell
$env:PYTHONPATH = "backend"
python -m pytest backend/tests -q
ruff check backend
```

Install the PyTorch wheel appropriate for your installed NVIDIA driver/CUDA runtime using the official PyTorch selector if GPU training is required. Do not assume that a generic pip install enables CUDA.
