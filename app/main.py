from __future__ import annotations
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
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

@app.get("/health")
def health() -> dict[str, Any]: return {"status": "healthy" if state.bundle else "unavailable", "model_loaded": state.bundle is not None}

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
