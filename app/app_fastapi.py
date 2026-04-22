from __future__ import annotations

from fastapi import FastAPI

from app.schemas import BatchPredictionRequest, BatchPredictionResponse, BeamInput, PredictionOutput
from app.service import get_predictor


app = FastAPI(
    title="FRP-RC PINN Predictor API",
    version="1.0.0",
    description="Inference API for the trained physics-informed FRP-RC flexural-capacity model.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict_one(payload: BeamInput) -> PredictionOutput:
    predictor = get_predictor()
    result = predictor.predict_records([payload.model_dump()])[0]
    return PredictionOutput(**result)


@app.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchPredictionRequest) -> BatchPredictionResponse:
    predictor = get_predictor()
    result = predictor.predict_records([item.model_dump() for item in payload.items])
    return BatchPredictionResponse(items=[PredictionOutput(**row) for row in result])

