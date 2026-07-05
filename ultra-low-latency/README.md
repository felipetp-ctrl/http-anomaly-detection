# Ultra-Low Latency Server (Rust)

Rust reimplementation of the HTTP anomaly detection API with native Isolation Forest inference. Achieves sub-millisecond end-to-end latency.

## Build

```bash
# 1. Export model parameters (one-time)
pip install joblib scikit-learn numpy
python ultra-low-latency/export_onnx.py

# 2. Build server
cd ultra-low-latency/server
cargo build --release

# 3. Run
cargo run --release
# Server starts on 0.0.0.0:8080

# 4. Run local smoke test / demo client
python ultra-low-latency/demo_local.py --base-url http://127.0.0.1:8080 --predict
```

## Local Demo Test

The Python helper in `ultra-low-latency/demo_local.py` checks the ultra-low-latency service end-to-end:

- `/health`
- `/openapi.json`
- `/docs`
- `/demo`
- `/demo/{scenario}`
- optional `/predict` smoke call
- DDoS run prints a small score chart so the anomaly progression is visible

If `http://127.0.0.1:8080` is not already running the Rust server, the helper can start it automatically on a free local port and continue the test flow.
If `/demo` is not exposed by the running service yet, the helper falls back to a local demo generator and still exercises `/predict` across the same scenarios.

Example:

```bash
python ultra-low-latency/demo_local.py --base-url http://127.0.0.1:8080 --scenario all --predict
```

Use `--no-auto-start` if you want the helper to fail instead of launching the Rust server itself.

## Latency Comparison (localhost, 100 requests)

### Client-side (curl round-trip)

| Metric | Python (FastAPI) | Rust (Actix-Web) | Speedup |
|--------|-----------------|------------------|---------|
| Mean   | 10.5 ms         | 0.621 ms         | 17x     |
| p50    | 10.8 ms         | 0.623 ms         | 17x     |
| p95    | 11.5 ms         | 0.748 ms         | 15x     |
| p99    | 12.3 ms         | 0.854 ms         | 14x     |

### Server-side (internal timing)

| Metric | Rust server-side |
|--------|-----------------|
| Mean   | 0.029 ms (29 µs) |
| p50    | 0.030 ms         |
| p95    | 0.040 ms         |
| p99    | 0.110 ms         |

## Architecture

```
HTTP Request → Actix-Web → serde JSON → State (HashMap) → Features (Rust) → Tree Traversal (native) → Response
```

- **No ONNX Runtime**: Isolation Forest trees are exported as JSON arrays and traversed natively in Rust — eliminates the ~2ms ONNX overhead.
- **No Mutex on inference**: Tree parameters are read-only after startup — only state management uses a Mutex.
- **StandardScaler in Rust**: `(x - mean) / scale` with hardcoded parameters.

## API

Same contract as the Python server — drop-in replacement on port 8080.

## Visual Docs

- Swagger UI: `/docs`
- OpenAPI spec: `/openapi.json`

The docs page uses the generated OpenAPI schema from the Rust service and exposes the existing `/predict`, `/health`, and demo endpoints in an interactive interface.
