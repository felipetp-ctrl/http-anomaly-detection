import random
import time
from contextlib import asynccontextmanager
from enum import Enum

import numpy as np
from fastapi import Body, FastAPI, HTTPException

from api.schemas import DemoResponse, DemoSample, PredictRequest, PredictResponse
from api.state import add_record, get_records
from lib.feature_engineering import RequestRecord, compute_features, FEATURE_NAMES
from lib.model_loader import load_artifacts

_model = None
_scaler = None
_metadata = None
_feature_logger = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _scaler, _metadata
    _model, _scaler, _metadata = load_artifacts()

    global _feature_logger
    from monitoring.logger import FeatureLogger
    import os
    _feature_logger = FeatureLogger(
        connection_string=os.environ.get("AZURE_STORAGE_CONNECTION_STRING"),
        container_name=os.environ.get("FEATURE_LOG_CONTAINER", "feature-logs"),
    )
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

    if _feature_logger:
        try:
            _feature_logger.log(
                features=features,
                prediction=int(prediction),
                anomaly_score=round(score, 4),
                timestamp=req.timestamp,
            )
        except Exception:
            pass

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


# ---------------------------------------------------------------------------
# Demo scenarios
# ---------------------------------------------------------------------------

class ScenarioName(str, Enum):
    legitimate = "legitimate"
    credential_stuffing = "credential_stuffing"
    ddos = "ddos"
    malicious_bot = "malicious_bot"


SCENARIO_META = {
    ScenarioName.legitimate: {
        "title": "Usuário legítimo",
        "description": "10 GETs espaçados, status 200, browser real — deve permanecer NORMAL",
    },
    ScenarioName.credential_stuffing: {
        "title": "Credential Stuffing",
        "description": "400 POSTs rápidos em /login, 90% status 401, bot UA — deve virar ANOMALY",
    },
    ScenarioName.ddos: {
        "title": "L7 DDoS",
        "description": "1200 requests em rajada, mix 200/503/504, browser UA — deve virar ANOMALY",
    },
    ScenarioName.malicious_bot: {
        "title": "Bot malicioso (scan)",
        "description": "200 GETs em endpoints sensíveis (/.env, /admin, …), UA rotativo — deve virar ANOMALY",
    },
}


def _generate_scenario(name: ScenarioName) -> tuple[str, list[dict]]:
    rng = random.Random(42)
    base_ts = time.time()

    if name == ScenarioName.legitimate:
        ip = "demo-legit-203.0.113.10"
        endpoints = ["/api/products", "/api/users", "/home", "/about", "/api/search",
                     "/js/app.js", "/css/style.css", "/images/logo.png"]
        ua = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        reqs = []
        for i in range(10):
            reqs.append({
                "ip": ip, "timestamp": base_ts + i * 35.0,
                "endpoint": endpoints[i % len(endpoints)],
                "status_code": 200, "payload_size": 5000 + i * 3000,
                "user_agent": ua, "response_time": 50 + i * 40,
            })
        return ip, reqs

    if name == ScenarioName.credential_stuffing:
        ip = "demo-cred-198.51.100.77"
        ua = "python-requests/2.31"
        reqs = []
        for i in range(400):
            reqs.append({
                "ip": ip, "timestamp": base_ts + i * 0.05,
                "endpoint": "/login",
                "status_code": 401 if i % 10 != 9 else 200,
                "payload_size": 256 + (i % 3),
                "user_agent": ua, "response_time": 45 + (i % 5) * 0.5,
            })
        return ip, reqs

    if name == ScenarioName.ddos:
        ip = "demo-ddos-192.0.2.200"
        ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        endpoints = ["/api/checkout", "/api/search"]
        status_cycle = [200, 200, 200, 503, 504]
        reqs = []
        for i in range(1200):
            status = status_cycle[i % 5]
            rt = max(1, rng.gauss(50, 20)) if status == 200 else max(1, rng.gauss(800, 200))
            reqs.append({
                "ip": ip, "timestamp": base_ts + i * 0.015,
                "endpoint": endpoints[i % len(endpoints)],
                "status_code": status,
                "payload_size": max(50, rng.gauss(500, 115)),
                "user_agent": ua, "response_time": rt,
            })
        return ip, reqs

    # malicious_bot
    ip = "demo-bot-172.16.50.99"
    endpoints = ["/.env", "/.git/config", "/admin", "/wp-login.php", "/phpmyadmin",
                 "/api/users", "/.aws/credentials", "/server-status", "/actuator",
                 "/graphql", "/debug", "/console", "/.svn/entries", "/backup.sql",
                 "/wp-admin", "/config.php", "/.htaccess", "/xmlrpc.php",
                 "/api/v1/debug", "/metrics", "/health", "/trace", "/dump"]
    uas = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    ]
    status_cycle = [200, 404, 404, 403, 200, 404]
    reqs = []
    for i in range(200):
        reqs.append({
            "ip": ip, "timestamp": base_ts + i * 1.5,
            "endpoint": endpoints[i % len(endpoints)],
            "status_code": status_cycle[i % len(status_cycle)],
            "payload_size": max(10, rng.gauss(2000, 520)),
            "user_agent": uas[i % len(uas)],
            "response_time": max(1, rng.gauss(100, 30)),
        })
    return ip, reqs


def _run_prediction(ip: str, r: dict) -> tuple[bool, float, dict]:
    record = RequestRecord(
        timestamp=r["timestamp"], endpoint=r["endpoint"],
        status_code=r["status_code"], payload_size=r["payload_size"],
        user_agent=r["user_agent"], response_time=r["response_time"],
    )
    add_record(ip, record)
    records_30s = get_records(ip, r["timestamp"], 30.0)
    records_5min = get_records(ip, r["timestamp"], 300.0)
    features = compute_features(records_30s, records_5min)
    X = _scaler.transform(np.array([features]))
    score = _model.score_samples(X)[0]
    prediction = _model.predict(X)[0]
    feature_values = dict(zip(FEATURE_NAMES, features))
    top = dict(sorted(feature_values.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
    return prediction == -1, round(score, 4), top


@app.post("/demo/{scenario}", response_model=DemoResponse)
def run_demo(scenario: ScenarioName):
    """Executa um cenário completo de demo e retorna o resumo da detecção."""
    meta = SCENARIO_META[scenario]
    ip, reqs = _generate_scenario(scenario)

    total = len(reqs)
    sample_every = max(1, total // 20)
    samples: list[DemoSample] = []
    anomaly_count = 0
    first_anomaly: int | None = None

    for i, r in enumerate(reqs):
        is_anom, score, top = _run_prediction(ip, r)
        if is_anom:
            anomaly_count += 1
            if first_anomaly is None:
                first_anomaly = i + 1
        if i % sample_every == 0 or i == total - 1:
            samples.append(DemoSample(
                request_number=i + 1,
                is_anomaly=is_anom,
                anomaly_score=score,
                top_features=top,
            ))

    return DemoResponse(
        scenario=meta["title"],
        description=meta["description"],
        ip=ip,
        total_requests=total,
        anomalies_detected=anomaly_count,
        first_anomaly_at=first_anomaly,
        samples=samples,
    )
