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
```

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
