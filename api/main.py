import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import Body, FastAPI

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


PREDICT_EXAMPLES = {
    "legitimate_user": {
        "summary": "Cenário 1 — Usuário legítimo",
        "description": "Navegação normal: GET espaçado, status 200, browser real",
        "value": {
            "ip": "203.0.113.10",
            "method": "GET",
            "status_code": 200,
            "timestamp": 1751500000.0,
            "endpoint": "/api/products",
            "payload_size": 5000,
            "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "response_time": 50,
        },
    },
    "credential_stuffing": {
        "summary": "Cenário 2 — Credential Stuffing",
        "description": "400 POSTs rápidos em /login, 90% status 401, bot UA",
        "value": {
            "ip": "198.51.100.77",
            "method": "POST",
            "status_code": 401,
            "timestamp": 1751500000.0,
            "endpoint": "/login",
            "payload_size": 256,
            "user_agent": "python-requests/2.31",
            "response_time": 45,
        },
    },
    "ddos": {
        "summary": "Cenário 3 — L7 DDoS",
        "description": "Rajada de 1200 requests, mix 200/503/504, browser UA",
        "value": {
            "ip": "192.0.2.200",
            "method": "GET",
            "status_code": 503,
            "timestamp": 1751500000.0,
            "endpoint": "/api/checkout",
            "payload_size": 500,
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "response_time": 800,
        },
    },
    "malicious_bot": {
        "summary": "Cenário 4 — Bot malicioso (scan)",
        "description": "Scan de endpoints sensíveis: /.env, /.git/config, /admin, etc.",
        "value": {
            "ip": "172.16.50.99",
            "method": "GET",
            "status_code": 404,
            "timestamp": 1751500000.0,
            "endpoint": "/.env",
            "payload_size": 2000,
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "response_time": 100,
        },
    },
}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest = Body(openapi_examples=PREDICT_EXAMPLES)):
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
