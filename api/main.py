import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI

from api.schemas import PredictRequest, PredictResponse
from api.state import add_record, get_records
from lib.feature_engineering import RequestRecord, compute_features, FEATURE_NAMES
from lib.model_loader import load_artifacts

_model = None
_scaler = None
_metadata = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _scaler, _metadata
    _model, _scaler, _metadata = load_artifacts()
    yield


app = FastAPI(title="HTTP Anomaly Detection", lifespan=lifespan)


@app.get("/")
def root():
    return {
        "service": "HTTP Anomaly Detection",
        "endpoints": ["/predict", "/health", "/docs"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    t0 = time.perf_counter()

    record = RequestRecord(
        timestamp=req.timestamp,
        endpoint=req.endpoint,
        status_code=req.status_code,
        payload_size=req.payload_size,
        user_agent=req.user_agent,
        response_time=req.response_time,
    )
    add_record(req.ip, record)
    t_state = time.perf_counter()

    records_30s = get_records(req.ip, req.timestamp, 30.0)
    records_5min = get_records(req.ip, req.timestamp, 300.0)
    features = compute_features(records_30s, records_5min)
    t_features = time.perf_counter()

    X = np.array([features])
    X_scaled = _scaler.transform(X)
    t_scale = time.perf_counter()

    score = _model.score_samples(X_scaled)[0]
    prediction = _model.predict(X_scaled)[0]
    t_predict = time.perf_counter()

    feature_values = dict(zip(FEATURE_NAMES, features))
    sorted_features = dict(sorted(feature_values.items(), key=lambda x: abs(x[1]), reverse=True)[:5])

    return PredictResponse(
        ip=req.ip,
        is_anomaly=(prediction == -1),
        anomaly_score=round(score, 4),
        top_features=sorted_features,
        timing_ms={
            "state_update": round((t_state - t0) * 1000, 2),
            "feature_calc": round((t_features - t_state) * 1000, 2),
            "scaling": round((t_scale - t_features) * 1000, 2),
            "prediction": round((t_predict - t_scale) * 1000, 2),
            "total": round((t_predict - t0) * 1000, 2),
        },
    )
