# HTTP Anomaly Detection

Real-time HTTP anomaly detection using Isolation Forest. Detects credential stuffing, L7 DDoS, and malicious bots by analyzing per-IP behavioral patterns across dual time windows (30s / 5min).

Built as a practical demonstration for the **CloudWalk ML Engineer (Security)** position.

## Architecture

```
Request → /predict → State (deque per IP) → Feature Engineering (10 features) → Scaler → Isolation Forest → Response
```

### Features (per IP, per time window)

| Feature | Signal |
|---------|--------|
| `request_count_30s/5min` | Volume — DDoS, credential stuffing |
| `endpoint_entropy` | Concentration — stuffing targets `/login` |
| `status_code_entropy` | Distribution — stuffing generates mostly 401 |
| `status_401_ratio` | Auth failure rate |
| `interval_std` | Regularity — bots have near-zero std |
| `unique_ua_ratio` | UA rotation — sophisticated bots |
| `known_ua_ratio` | Real browser vs bot UA |
| `payload_size_std` | Payload uniformity — stuffing sends identical payloads |
| `response_time_std` | Response time uniformity |

### Design Decisions

- **Dual windows in single model**: 30s captures bursts, 5min captures slow attacks. The ratio between them carries information a single window misses.
- **`namedtuple` over `dict`**: Minimal memory footprint per record — under attack, a single IP may have thousands of records.
- **Time-based cleanup, not `maxlen`**: `maxlen` truncates true request counts, underestimating anomalies.
- **Sync endpoint**: CPU-bound work (features + inference) — `def` lets FastAPI run it in a thread pool without blocking the event loop.
- **Single feature source**: `lib/feature_engineering.py` is used by both training and inference, preventing train/serve skew.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Generate features + train
python -m training.prepare_features
python -m training.train_model
python -m training.evaluate

# Run API
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ip": "192.168.1.100",
    "timestamp": 1751500000.0,
    "endpoint": "/login",
    "method": "POST",
    "status_code": 401,
    "payload_size": 256,
    "user_agent": "python-requests/2.31",
    "response_time": 45.0
  }'
```

Response includes anomaly score, top contributing features, and per-stage latency breakdown.

## Azure Deployment

Deployed as an Azure Container App:

```bash
# Build and push to ACR
az acr login --name <acr-name>
docker build --platform linux/amd64 -t <acr-name>.azurecr.io/http-anomaly-api:latest .
docker push <acr-name>.azurecr.io/http-anomaly-api:latest

# Create Container App
az containerapp create \
  --name http-anomaly-api \
  --resource-group <rg> \
  --environment <env> \
  --image <acr-name>.azurecr.io/http-anomaly-api:latest \
  --target-port 8000 \
  --ingress external
```

**Live**: https://http-anomaly-api.blackplant-e06bfbe9.centralus.azurecontainerapps.io

## Latency (localhost, 100 requests)

| Metric | Value |
|--------|-------|
| Mean | 10.5 ms |
| p50 | 10.8 ms |
| p95 | 11.5 ms |
| p99 | 12.3 ms |

## Stack

Python 3.12+ · FastAPI · scikit-learn (Isolation Forest) · joblib · pandas (training only)
