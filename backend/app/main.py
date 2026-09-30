"""DeepTrace FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="DeepTrace API",
    description="Visual deepfake screening API. Results are model screening signals, not forensic proof.",
    version="0.1.0",
)

# Local development origin; configure explicitly for deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/api/health", tags=["system"])
def health() -> dict[str, str | bool]:
    """Return process health separately from model readiness."""
    return {"status": "ok", "model_ready": False}
