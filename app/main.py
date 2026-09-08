from __future__ import annotations
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, float] = Field(min_length=1, max_length=512)

class AppState:
    bundle: dict[str, Any] | None = None
    error: str | None = None
state = AppState()

@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        state.bundle = joblib.load(Path(os.getenv("MODEL_PATH", "artifacts/production/model.joblib")))
        state.error = None
    except Exception as exc:
        state.bundle, state.error = None, str(exc)
    yield

app = FastAPI(title="Sensor Fault Detection API", version="1.0.0", lifespan=lifespan)


@app.get("/", include_in_schema=False, response_class=HTMLResponse)
def home() -> str:
    model_state = "ready" if state.bundle else "waiting for a trusted model artifact"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sensor Fault Detection API</title><link rel="icon" href="/favicon.svg" type="image/svg+xml">
<style>body{{font-family:system-ui,sans-serif;max-width:46rem;margin:4rem auto;padding:0 1.5rem;color:#0f172a}}code{{background:#e2e8f0;padding:.15rem .35rem;border-radius:.25rem}}a{{color:#0369a1}}.state{{padding:1rem;background:#f1f5f9;border-radius:.5rem}}</style>
</head><body><h1>Sensor Fault Detection API</h1><p class="state">Service is online; model state: <strong>{model_state}</strong>.</p>
<p>Use the interactive <a href="/docs">API documentation</a> to inspect endpoints and request schemas.</p>
<ul><li><a href="/health">Liveness status</a></li><li><a href="/ready">Model readiness</a></li><li><a href="/model/info">Model metadata</a></li></ul>
</body></html>"""

@app.get("/favicon.svg", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "favicon.svg", media_type="image/svg+xml")

@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "healthy", "model_loaded": state.bundle is not None}


@app.get("/ready")
def readiness() -> dict[str, str]:
    if not state.bundle:
        raise HTTPException(503, "Model is not ready")
    return {"status": "ready"}

@app.get("/model/info")
def model_info() -> dict[str, Any]:
    if not state.bundle: raise HTTPException(503, "Model is not ready")
    return state.bundle["metadata"]

@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    if not state.bundle: raise HTTPException(503, "Model is not ready")
    metadata = state.bundle["metadata"]; expected = metadata["features"]
    missing, extra = sorted(set(expected) - set(request.features)), sorted(set(request.features) - set(expected))
    if missing or extra: raise HTTPException(422, {"missing_features": missing, "unexpected_features": extra})
    x = pd.DataFrame([[request.features[f] for f in expected]], columns=expected)
    probability = float(state.bundle["model"].predict_proba(state.bundle["preprocessor"].transform(x))[0, 1])
    prediction = int(probability >= metadata["threshold"])
    return {"prediction": prediction, "label": "failure" if prediction else "non_failure", "failure_probability": probability, "model_version": metadata["model_version"]}
