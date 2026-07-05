# MLOps Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a full MLOps pipeline with MLflow experiment tracking, Azure ML drift monitoring, GitHub Actions CI/CD with quality gates, and canary deployments to Azure Container Apps.

**Architecture:** Train model via Azure ML Compute, log experiments to MLflow hosted on Azure ML, register versioned models in MLflow Model Registry. Production features are logged to Azure Blob Storage, and Azure ML Data Drift Monitor compares them daily against the training baseline. When drift is detected, an Azure Monitor alert triggers a GitHub Actions retrain workflow that runs the full train → validate → canary deploy cycle.

**Tech Stack:** Python 3.12, scikit-learn, MLflow, Azure ML SDK v2, Azure Identity, Azure Storage Blob, GitHub Actions, Azure Container Apps

## Global Constraints

- Python ≥ 3.12
- scikit-learn 1.9.x (Isolation Forest)
- MLflow ≥ 2.15 (Azure ML integration)
- Azure ML SDK v2 (`azure-ai-ml` package)
- All Azure auth via `DefaultAzureCredential` (OIDC in CI, `az login` locally)
- Existing `lib/` module is read-only — no changes to `feature_engineering.py` or `known_user_agents.py`
- Feature names are defined in `lib.feature_engineering.FEATURE_NAMES` — always import from there
- Existing `artifacts/` directory stays for local dev; production uses MLflow registry

---

### Task 1: MLflow Training Wrapper

Modify `training/train_model.py` to log experiments to MLflow and register the model in the MLflow Model Registry.

**Files:**
- Modify: `training/train_model.py`
- Modify: `requirements.txt`
- Create: `tests/test_train_model.py`

**Interfaces:**
- Consumes: `lib.feature_engineering.FEATURE_NAMES` (list[str]), `data/features.csv`
- Produces: `training.train_model.train(tracking_uri: str | None = None, register: bool = False) -> dict` — returns `{"run_id": str, "silhouette_score": float, "anomaly_rate": float, "model_version": int | None}`

- [ ] **Step 1: Add dependencies to requirements.txt**

Append to `requirements.txt`:
```
mlflow==2.21.0
azure-ai-ml==1.25.0
azure-identity==1.21.0
azure-storage-blob==12.25.0
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_train_model.py
import json
import tempfile
from pathlib import Path

import mlflow
import pandas as pd

from lib.feature_engineering import FEATURE_NAMES


def _create_dummy_features(tmp_path: Path, n: int = 50) -> Path:
    """Create a minimal features.csv for testing."""
    import numpy as np

    rng = np.random.RandomState(42)
    data = {name: rng.randn(n) for name in FEATURE_NAMES}
    data["ip"] = [f"10.0.0.{i}" for i in range(n)]
    data["label"] = ["legitimate"] * (n - 5) + ["anomaly"] * 5
    df = pd.DataFrame(data)
    csv_path = tmp_path / "data"
    csv_path.mkdir()
    df.to_csv(csv_path / "features.csv", index=False)
    return tmp_path


def test_train_logs_to_mlflow(tmp_path):
    project_dir = _create_dummy_features(tmp_path)
    tracking_uri = f"file://{tmp_path / 'mlruns'}"

    from training.train_model import train

    result = train(
        tracking_uri=tracking_uri,
        data_dir=project_dir / "data",
        artifacts_dir=project_dir / "artifacts",
        register=False,
    )

    assert "run_id" in result
    assert "silhouette_score" in result
    assert "anomaly_rate" in result
    assert 0.0 <= result["anomaly_rate"] <= 1.0

    # Verify MLflow logged the run
    mlflow.set_tracking_uri(tracking_uri)
    run = mlflow.get_run(result["run_id"])
    assert run.data.params["n_estimators"] == "100"
    assert run.data.params["contamination"] == "0.05"
    assert "silhouette_score" in run.data.metrics

    # Verify artifacts were saved locally too
    assert (project_dir / "artifacts" / "model.joblib").exists()
    assert (project_dir / "artifacts" / "scaler.joblib").exists()


def test_train_produces_valid_metadata(tmp_path):
    project_dir = _create_dummy_features(tmp_path)
    tracking_uri = f"file://{tmp_path / 'mlruns'}"

    from training.train_model import train

    train(
        tracking_uri=tracking_uri,
        data_dir=project_dir / "data",
        artifacts_dir=project_dir / "artifacts",
        register=False,
    )

    metadata_path = project_dir / "artifacts" / "metadata.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["feature_names"] == FEATURE_NAMES
    assert metadata["n_estimators"] == 100
    assert "trained_at" in metadata
    assert "run_id" in metadata
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /Users/felipetomepereira/Projects/cloudwalk_fastapi_model && python -m pytest tests/test_train_model.py -v`
Expected: FAIL — `ImportError: cannot import name 'train' from 'training.train_model'`

- [ ] **Step 4: Implement the MLflow training wrapper**

Replace `training/train_model.py` with:

```python
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from lib.feature_engineering import FEATURE_NAMES

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

N_ESTIMATORS = 100
CONTAMINATION = 0.05
EXPERIMENT_NAME = "http-anomaly-isolation-forest"


def _dataset_hash(csv_path: Path) -> str:
    return hashlib.sha256(csv_path.read_bytes()).hexdigest()[:12]


def train(
    tracking_uri: str | None = None,
    data_dir: Path | None = None,
    artifacts_dir: Path | None = None,
    register: bool = False,
) -> dict:
    data_dir = data_dir or DATA_DIR
    artifacts_dir = artifacts_dir or ARTIFACTS_DIR
    csv_path = data_dir / "features.csv"

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(csv_path)
    X = df[FEATURE_NAMES].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=42,
    )

    with mlflow.start_run() as run:
        model.fit(X_scaled)
        predictions = model.predict(X_scaled)
        scores = model.score_samples(X_scaled)

        anomaly_rate = float((predictions == -1).mean())
        sil_score = float(silhouette_score(X_scaled, predictions))

        mlflow.log_params({
            "n_estimators": N_ESTIMATORS,
            "contamination": CONTAMINATION,
            "dataset_hash": _dataset_hash(csv_path),
            "training_samples": len(X),
        })
        mlflow.log_metrics({
            "silhouette_score": sil_score,
            "anomaly_rate": anomaly_rate,
            "mean_anomaly_score": float(scores.mean()),
            "std_anomaly_score": float(scores.std()),
        })

        artifacts_dir.mkdir(exist_ok=True)
        joblib.dump(model, artifacts_dir / "model.joblib")
        joblib.dump(scaler, artifacts_dir / "scaler.joblib")

        metadata = {
            "feature_names": FEATURE_NAMES,
            "n_estimators": N_ESTIMATORS,
            "contamination": CONTAMINATION,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "training_samples": len(X),
            "run_id": run.info.run_id,
            "dataset_hash": _dataset_hash(csv_path),
        }
        with open(artifacts_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        mlflow.log_artifacts(str(artifacts_dir))

        model_version = None
        if register:
            mv = mlflow.register_model(
                f"runs:/{run.info.run_id}/artifacts",
                "http-anomaly-detector",
            )
            model_version = int(mv.version)

        return {
            "run_id": run.info.run_id,
            "silhouette_score": sil_score,
            "anomaly_rate": anomaly_rate,
            "model_version": model_version,
        }


def main():
    result = train()
    print(f"Training complete. Run ID: {result['run_id']}")
    print(f"Silhouette score: {result['silhouette_score']:.4f}")
    print(f"Anomaly rate: {result['anomaly_rate']:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/felipetomepereira/Projects/cloudwalk_fastapi_model && python -m pytest tests/test_train_model.py -v`
Expected: 2 tests PASS

- [ ] **Step 6: Commit**

```bash
git add training/train_model.py requirements.txt tests/test_train_model.py
git commit -m "feat: add MLflow experiment tracking to training pipeline"
```

---

### Task 2: Model Loader with MLflow Registry Support

Extend `lib/model_loader.py` to load models from MLflow Model Registry by stage or version, falling back to local artifacts.

**Files:**
- Modify: `lib/model_loader.py`
- Create: `tests/test_model_loader.py`

**Interfaces:**
- Consumes: `lib.feature_engineering.FEATURE_NAMES` (list[str])
- Produces: `lib.model_loader.load_artifacts(artifacts_dir: Path | None = None, mlflow_model_name: str | None = None, mlflow_model_stage: str | None = None) -> tuple[IsolationForest, StandardScaler, dict]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model_loader.py
import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from lib.feature_engineering import FEATURE_NAMES
from lib.model_loader import load_artifacts


def _create_dummy_artifacts(tmp_path: Path) -> Path:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    scaler = StandardScaler()
    X = np.random.randn(20, len(FEATURE_NAMES))
    scaler.fit(X)
    model = IsolationForest(n_estimators=10, random_state=42)
    model.fit(scaler.transform(X))

    joblib.dump(model, artifacts_dir / "model.joblib")
    joblib.dump(scaler, artifacts_dir / "scaler.joblib")

    metadata = {
        "feature_names": FEATURE_NAMES,
        "n_estimators": 10,
        "contamination": 0.05,
        "trained_at": "2026-01-01T00:00:00+00:00",
        "training_samples": 20,
    }
    with open(artifacts_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    return artifacts_dir


def test_load_from_local_artifacts(tmp_path):
    artifacts_dir = _create_dummy_artifacts(tmp_path)
    model, scaler, metadata = load_artifacts(artifacts_dir=artifacts_dir)
    assert hasattr(model, "predict")
    assert hasattr(scaler, "transform")
    assert metadata["feature_names"] == FEATURE_NAMES


def test_load_raises_on_feature_mismatch(tmp_path):
    artifacts_dir = _create_dummy_artifacts(tmp_path)
    meta_path = artifacts_dir / "metadata.json"
    metadata = json.loads(meta_path.read_text())
    metadata["feature_names"] = ["wrong_feature"]
    meta_path.write_text(json.dumps(metadata))

    import pytest

    with pytest.raises(ValueError, match="Feature mismatch"):
        load_artifacts(artifacts_dir=artifacts_dir)
```

- [ ] **Step 2: Run tests to verify they pass (existing behavior)**

Run: `cd /Users/felipetomepereira/Projects/cloudwalk_fastapi_model && python -m pytest tests/test_model_loader.py -v`
Expected: 2 tests PASS (tests existing behavior before we extend)

- [ ] **Step 3: Add test for MLflow registry loading**

Append to `tests/test_model_loader.py`:

```python
def test_load_with_mlflow_params_falls_back_when_unavailable(tmp_path):
    """When mlflow_model_name is given but model not found, falls back to local."""
    artifacts_dir = _create_dummy_artifacts(tmp_path)
    model, scaler, metadata = load_artifacts(
        artifacts_dir=artifacts_dir,
        mlflow_model_name="nonexistent-model",
        mlflow_model_stage="Production",
    )
    assert hasattr(model, "predict")
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd /Users/felipetomepereira/Projects/cloudwalk_fastapi_model && python -m pytest tests/test_model_loader.py::test_load_with_mlflow_params_falls_back_when_unavailable -v`
Expected: FAIL — `TypeError: load_artifacts() got an unexpected keyword argument 'mlflow_model_name'`

- [ ] **Step 5: Implement MLflow registry support**

Replace `lib/model_loader.py` with:

```python
import json
import logging
from pathlib import Path

import joblib

from lib.feature_engineering import FEATURE_NAMES

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

logger = logging.getLogger(__name__)


def load_artifacts(
    artifacts_dir: Path | None = None,
    mlflow_model_name: str | None = None,
    mlflow_model_stage: str | None = None,
):
    if mlflow_model_name:
        try:
            return _load_from_registry(mlflow_model_name, mlflow_model_stage)
        except Exception as e:
            logger.warning("MLflow registry load failed (%s), falling back to local artifacts", e)

    return _load_local(artifacts_dir or ARTIFACTS_DIR)


def _load_from_registry(model_name: str, stage: str | None):
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    if stage:
        model_uri = f"models:/{model_name}/{stage}"
    else:
        model_uri = f"models:/{model_name}/latest"

    model_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
    model_dir = Path(model_path)

    model = joblib.load(model_dir / "model.joblib")
    scaler = joblib.load(model_dir / "scaler.joblib")

    with open(model_dir / "metadata.json") as f:
        metadata = json.load(f)

    _validate_features(metadata)
    return model, scaler, metadata


def _load_local(artifacts_dir: Path):
    model = joblib.load(artifacts_dir / "model.joblib")
    scaler = joblib.load(artifacts_dir / "scaler.joblib")

    with open(artifacts_dir / "metadata.json") as f:
        metadata = json.load(f)

    _validate_features(metadata)
    return model, scaler, metadata


def _validate_features(metadata: dict):
    expected = metadata["feature_names"]
    if expected != FEATURE_NAMES:
        raise ValueError(
            f"Feature mismatch. Model expects {expected}, code provides {FEATURE_NAMES}"
        )
```

- [ ] **Step 6: Run all tests to verify they pass**

Run: `cd /Users/felipetomepereira/Projects/cloudwalk_fastapi_model && python -m pytest tests/test_model_loader.py -v`
Expected: 3 tests PASS

- [ ] **Step 7: Commit**

```bash
git add lib/model_loader.py tests/test_model_loader.py
git commit -m "feat: add MLflow Model Registry support to model loader"
```

---

### Task 3: Production Feature Logger

Create `monitoring/logger.py` that logs computed features from each `/predict` call to Azure Blob Storage as daily append blobs. Integrate it into `api/main.py`.

**Files:**
- Create: `monitoring/__init__.py`
- Create: `monitoring/logger.py`
- Create: `tests/test_logger.py`
- Modify: `api/main.py`

**Interfaces:**
- Consumes: `lib.feature_engineering.FEATURE_NAMES` (list[str])
- Produces: `monitoring.logger.FeatureLogger` — class with `log(features: list[float], prediction: int, anomaly_score: float, timestamp: float) -> None` and `FeatureLogger(connection_string: str | None, container_name: str)`. When `connection_string` is `None`, logs to a local CSV file instead (for dev/test).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_logger.py
import csv
import time
from pathlib import Path

from lib.feature_engineering import FEATURE_NAMES


def test_logger_writes_local_csv(tmp_path):
    from monitoring.logger import FeatureLogger

    logger = FeatureLogger(connection_string=None, container_name="test", local_dir=tmp_path)

    features = [float(i) for i in range(len(FEATURE_NAMES))]
    ts = time.time()
    logger.log(features=features, prediction=1, anomaly_score=-0.35, timestamp=ts)
    logger.log(features=features, prediction=-1, anomaly_score=-0.55, timestamp=ts + 1)

    csv_files = list(tmp_path.glob("*.csv"))
    assert len(csv_files) == 1

    with open(csv_files[0]) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 2
    assert rows[0]["prediction"] == "1"
    assert rows[1]["prediction"] == "-1"
    for name in FEATURE_NAMES:
        assert name in rows[0]


def test_logger_creates_daily_files(tmp_path):
    from monitoring.logger import FeatureLogger

    logger = FeatureLogger(connection_string=None, container_name="test", local_dir=tmp_path)

    features = [0.0] * len(FEATURE_NAMES)
    logger.log(features=features, prediction=1, anomaly_score=-0.3, timestamp=1751500000.0)
    logger.log(features=features, prediction=1, anomaly_score=-0.3, timestamp=1751600000.0)

    csv_files = list(tmp_path.glob("*.csv"))
    assert len(csv_files) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/felipetomepereira/Projects/cloudwalk_fastapi_model && python -m pytest tests/test_logger.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'monitoring'`

- [ ] **Step 3: Implement the feature logger**

```python
# monitoring/__init__.py
```

```python
# monitoring/logger.py
import csv
import io
import logging
from datetime import datetime, timezone
from pathlib import Path

from lib.feature_engineering import FEATURE_NAMES

logger = logging.getLogger(__name__)

_CSV_FIELDS = ["timestamp", "prediction", "anomaly_score"] + FEATURE_NAMES


class FeatureLogger:
    def __init__(
        self,
        connection_string: str | None = None,
        container_name: str = "feature-logs",
        local_dir: Path | None = None,
    ):
        self._connection_string = connection_string
        self._container_name = container_name
        self._local_dir = local_dir

    def log(
        self,
        features: list[float],
        prediction: int,
        anomaly_score: float,
        timestamp: float,
    ) -> None:
        row = {
            "timestamp": timestamp,
            "prediction": prediction,
            "anomaly_score": anomaly_score,
        }
        for name, value in zip(FEATURE_NAMES, features):
            row[name] = value

        if self._connection_string:
            self._append_to_blob(row, timestamp)
        else:
            self._append_to_local(row, timestamp)

    def _blob_name(self, timestamp: float) -> str:
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        return f"features-{dt.strftime('%Y-%m-%d')}.csv"

    def _append_to_local(self, row: dict, timestamp: float) -> None:
        if self._local_dir is None:
            return
        filename = self._blob_name(timestamp)
        filepath = self._local_dir / filename
        is_new = not filepath.exists()

        with open(filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerow(row)

    def _append_to_blob(self, row: dict, timestamp: float) -> None:
        from azure.storage.blob import BlobServiceClient

        blob_name = self._blob_name(timestamp)
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=_CSV_FIELDS)

        blob_service = BlobServiceClient.from_connection_string(self._connection_string)
        container_client = blob_service.get_container_client(self._container_name)

        try:
            container_client.get_container_properties()
        except Exception:
            container_client.create_container()

        blob_client = container_client.get_blob_client(blob_name)

        try:
            blob_client.get_blob_properties()
        except Exception:
            writer.writeheader()

        writer.writerow(row)
        blob_client.upload_blob(buf.getvalue(), blob_type="AppendBlob", overwrite=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/felipetomepereira/Projects/cloudwalk_fastapi_model && python -m pytest tests/test_logger.py -v`
Expected: 2 tests PASS

- [ ] **Step 5: Integrate logger into api/main.py**

Add the logger initialization in the `lifespan` function and call it after prediction. Add these changes to `api/main.py`:

After the global variables block (`_model = None` etc.), add:
```python
_feature_logger = None
```

In the `lifespan` function, after `_model, _scaler, _metadata = load_artifacts()`, add:
```python
    global _feature_logger
    from monitoring.logger import FeatureLogger
    import os
    _feature_logger = FeatureLogger(
        connection_string=os.environ.get("AZURE_STORAGE_CONNECTION_STRING"),
        container_name=os.environ.get("FEATURE_LOG_CONTAINER", "feature-logs"),
    )
```

In the `predict` function, after the `return PredictResponse(...)` block, this won't work (return exits). Instead, capture the response before returning and log:

Replace the end of the `predict` function (from `feature_values = ...` to the return) with:
```python
    feature_values = dict(zip(FEATURE_NAMES, features))
    sorted_features = dict(sorted(feature_values.items(), key=lambda x: abs(x[1]), reverse=True)[:5])

    prediction = _model.predict(X_scaled)[0]

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
```

Note: the `prediction` variable is already computed above in the original code. The only actual addition is the `if _feature_logger:` block between the existing prediction and the return statement.

- [ ] **Step 6: Run full test suite to check nothing is broken**

Run: `cd /Users/felipetomepereira/Projects/cloudwalk_fastapi_model && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add monitoring/__init__.py monitoring/logger.py tests/test_logger.py api/main.py
git commit -m "feat: add production feature logger with Azure Blob Storage support"
```

---

### Task 4: Drift Monitoring Setup Scripts

Create scripts to register the baseline dataset and configure the Azure ML Data Drift Monitor.

**Files:**
- Create: `monitoring/register_baseline.py`
- Create: `monitoring/setup_drift_monitor.py`
- Create: `tests/test_drift_setup.py`

**Interfaces:**
- Consumes: `data/features.csv`, `lib.feature_engineering.FEATURE_NAMES`
- Produces: Two CLI scripts (`python -m monitoring.register_baseline`, `python -m monitoring.setup_drift_monitor`) that configure Azure ML resources

- [ ] **Step 1: Write test for baseline registration logic**

```python
# tests/test_drift_setup.py
import csv
from pathlib import Path

from lib.feature_engineering import FEATURE_NAMES


def test_baseline_csv_generation(tmp_path):
    """Test that baseline extraction produces correct CSV format."""
    from monitoring.register_baseline import extract_baseline_features

    # Create dummy features.csv
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    features_csv = data_dir / "features.csv"

    with open(features_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ip"] + FEATURE_NAMES + ["label"])
        writer.writeheader()
        for i in range(10):
            row = {"ip": f"10.0.0.{i}", "label": "legitimate"}
            for name in FEATURE_NAMES:
                row[name] = float(i)
            writer.writerow(row)

    output = tmp_path / "baseline.csv"
    extract_baseline_features(features_csv, output)

    with open(output) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 10
    assert set(reader.fieldnames) == set(FEATURE_NAMES + ["timestamp"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/felipetomepereira/Projects/cloudwalk_fastapi_model && python -m pytest tests/test_drift_setup.py -v`
Expected: FAIL — `ImportError: cannot import name 'extract_baseline_features'`

- [ ] **Step 3: Implement register_baseline.py**

```python
# monitoring/register_baseline.py
"""Register the training dataset as baseline for drift monitoring in Azure ML.

Usage:
  python -m monitoring.register_baseline  (requires Azure ML config)
  python -m monitoring.register_baseline --local-only  (just generates baseline CSV)
"""
import argparse
import csv
import time
from pathlib import Path

from lib.feature_engineering import FEATURE_NAMES

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def extract_baseline_features(features_csv: Path, output_path: Path) -> None:
    with open(features_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    with open(output_path, "w", newline="") as f:
        fields = FEATURE_NAMES + ["timestamp"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        base_ts = time.time()
        for i, row in enumerate(rows):
            out = {name: row[name] for name in FEATURE_NAMES}
            out["timestamp"] = base_ts + i
            writer.writerow(out)


def register_in_azure_ml(baseline_csv: Path) -> None:
    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import Data
    from azure.ai.ml.constants import AssetTypes
    from azure.identity import DefaultAzureCredential
    import os

    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_ML_WORKSPACE"],
    )

    baseline_data = Data(
        name="http-anomaly-baseline",
        path=str(baseline_csv),
        type=AssetTypes.URI_FILE,
        description="Baseline feature distributions for drift monitoring",
    )
    ml_client.data.create_or_update(baseline_data)
    print(f"Baseline dataset registered: {baseline_data.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()

    features_csv = DATA_DIR / "features.csv"
    baseline_csv = DATA_DIR / "baseline.csv"

    extract_baseline_features(features_csv, baseline_csv)
    print(f"Baseline CSV generated: {baseline_csv}")

    if not args.local_only:
        register_in_azure_ml(baseline_csv)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/felipetomepereira/Projects/cloudwalk_fastapi_model && python -m pytest tests/test_drift_setup.py -v`
Expected: PASS

- [ ] **Step 5: Implement setup_drift_monitor.py**

```python
# monitoring/setup_drift_monitor.py
"""Configure Azure ML Data Drift Monitor.

Usage: python -m monitoring.setup_drift_monitor
Requires: AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_ML_WORKSPACE env vars
"""
import os

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    DataDriftMonitor,
    MonitoringTarget,
    MonitorSchedule,
    CronTrigger,
    AlertNotification,
)
from azure.identity import DefaultAzureCredential

from lib.feature_engineering import FEATURE_NAMES

PSI_THRESHOLD = 0.2
DRIFT_FEATURE_COUNT_THRESHOLD = 2


def create_drift_monitor() -> None:
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_ML_WORKSPACE"],
    )

    monitor = DataDriftMonitor(
        name="http-anomaly-drift-monitor",
        baseline_data="http-anomaly-baseline:1",
        target_data="http-anomaly-production:latest",
        features=FEATURE_NAMES,
        frequency="Day",
        alert_config=AlertNotification(
            emails=[os.environ.get("ALERT_EMAIL", "")],
        ),
        compute=os.environ.get("AZURE_ML_COMPUTE", "cpu-cluster"),
        threshold=PSI_THRESHOLD,
    )

    schedule = MonitorSchedule(
        name="drift-daily-check",
        trigger=CronTrigger(expression="0 6 * * *"),
        create_monitor=monitor,
    )

    ml_client.schedules.begin_create_or_update(schedule)
    print("Drift monitor configured: daily at 06:00 UTC")
    print(f"PSI threshold: {PSI_THRESHOLD}")
    print(f"Features monitored: {', '.join(FEATURE_NAMES)}")


def main():
    create_drift_monitor()


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add monitoring/register_baseline.py monitoring/setup_drift_monitor.py tests/test_drift_setup.py
git commit -m "feat: add drift monitoring baseline registration and setup scripts"
```

---

### Task 5: Infrastructure Scripts

Create shell scripts that provision the Azure ML workspace, compute, and deploy Container Apps with canary support.

**Files:**
- Create: `infra/setup_workspace.sh`
- Create: `infra/deploy_container_app.sh`

**Interfaces:**
- Consumes: environment variables `AZURE_RESOURCE_GROUP`, `AZURE_ML_WORKSPACE`, `AZURE_SUBSCRIPTION_ID`, `AZURE_CONTAINER_APP_NAME`, `AZURE_CONTAINER_REGISTRY`
- Produces: two runnable shell scripts for infra provisioning and deployment

- [ ] **Step 1: Create setup_workspace.sh**

```bash
# infra/setup_workspace.sh
#!/usr/bin/env bash
set -euo pipefail

RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:?Set AZURE_RESOURCE_GROUP}"
WORKSPACE="${AZURE_ML_WORKSPACE:?Set AZURE_ML_WORKSPACE}"
LOCATION="${AZURE_LOCATION:-eastus}"
COMPUTE_NAME="${AZURE_ML_COMPUTE:-cpu-cluster}"
STORAGE_ACCOUNT="${AZURE_STORAGE_ACCOUNT:-httpanomalystorage}"

echo "=== Creating Resource Group ==="
az group create --name "$RESOURCE_GROUP" --location "$LOCATION"

echo "=== Creating Azure ML Workspace ==="
az ml workspace create \
  --name "$WORKSPACE" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION"

echo "=== Creating Compute Cluster ==="
az ml compute create \
  --name "$COMPUTE_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --workspace-name "$WORKSPACE" \
  --type AmlCompute \
  --size Standard_DS2_v2 \
  --min-instances 0 \
  --max-instances 2

echo "=== Creating Storage Account for Feature Logs ==="
az storage account create \
  --name "$STORAGE_ACCOUNT" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --sku Standard_LRS

az storage container create \
  --name feature-logs \
  --account-name "$STORAGE_ACCOUNT"

echo "=== Done ==="
echo "Workspace: $WORKSPACE"
echo "Compute: $COMPUTE_NAME"
echo "Storage: $STORAGE_ACCOUNT"
```

- [ ] **Step 2: Create deploy_container_app.sh**

```bash
# infra/deploy_container_app.sh
#!/usr/bin/env bash
set -euo pipefail

APP_NAME="${AZURE_CONTAINER_APP_NAME:?Set AZURE_CONTAINER_APP_NAME}"
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:?Set AZURE_RESOURCE_GROUP}"
REGISTRY="${AZURE_CONTAINER_REGISTRY:?Set AZURE_CONTAINER_REGISTRY}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CANARY_PERCENT="${CANARY_PERCENT:-10}"

IMAGE="${REGISTRY}.azurecr.io/http-anomaly-detection:${IMAGE_TAG}"

echo "=== Building and pushing image ==="
az acr build \
  --registry "$REGISTRY" \
  --image "http-anomaly-detection:${IMAGE_TAG}" \
  .

echo "=== Deploying canary revision (${CANARY_PERCENT}% traffic) ==="
az containerapp update \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --image "$IMAGE" \
  --revision-suffix "v${IMAGE_TAG}" \
  --set-env-vars \
    "AZURE_STORAGE_CONNECTION_STRING=secretref:storage-conn-string" \
    "FEATURE_LOG_CONTAINER=feature-logs"

NEW_REVISION=$(az containerapp revision list \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query "[-1].name" -o tsv)

PROD_REVISION=$(az containerapp revision list \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query "[-2].name" -o tsv)

az containerapp ingress traffic set \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --revision-weight \
    "${PROD_REVISION}=$((100 - CANARY_PERCENT))" \
    "${NEW_REVISION}=${CANARY_PERCENT}"

echo "=== Canary deployed ==="
echo "Production: ${PROD_REVISION} (${((100 - CANARY_PERCENT))}%)"
echo "Canary: ${NEW_REVISION} (${CANARY_PERCENT}%)"
echo ""
echo "To promote canary to 100%:"
echo "  az containerapp ingress traffic set \\"
echo "    --name $APP_NAME --resource-group $RESOURCE_GROUP \\"
echo "    --revision-weight ${NEW_REVISION}=100"
```

- [ ] **Step 3: Make scripts executable and commit**

```bash
chmod +x infra/setup_workspace.sh infra/deploy_container_app.sh
git add infra/
git commit -m "feat: add Azure infra provisioning and canary deploy scripts"
```

---

### Task 6: Azure ML Training Job Spec

Create the Azure ML job YAML that defines the training environment and command for remote compute.

**Files:**
- Create: `training/azureml_job.yml`
- Create: `training/conda_env.yml`

**Interfaces:**
- Consumes: `training/train_model.py` (the `train()` function)
- Produces: Azure ML job definition runnable via `az ml job create -f training/azureml_job.yml`

- [ ] **Step 1: Create conda environment file**

```yaml
# training/conda_env.yml
name: http-anomaly-training
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.12
  - pip
  - pip:
      - scikit-learn==1.9.0
      - pandas==3.0.3
      - numpy==2.5.0
      - mlflow==2.21.0
      - azure-ai-ml==1.25.0
      - azure-identity==1.21.0
      - joblib==1.4.2
```

- [ ] **Step 2: Create Azure ML job spec**

```yaml
# training/azureml_job.yml
$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json

experiment_name: http-anomaly-isolation-forest
display_name: train-isolation-forest
description: Train Isolation Forest model for HTTP anomaly detection

compute: azureml:cpu-cluster

environment:
  image: mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04
  conda_file: conda_env.yml

code: ../

command: >-
  python -c "
  from training.train_model import train;
  result = train(register=True);
  print(f'Run: {result[\"run_id\"]}');
  print(f'Silhouette: {result[\"silhouette_score\"]:.4f}');
  print(f'Anomaly rate: {result[\"anomaly_rate\"]:.4f}');
  print(f'Model version: {result[\"model_version\"]}');
  "

inputs:
  training_data:
    type: uri_file
    path: azureml:http-anomaly-baseline:1
```

- [ ] **Step 3: Commit**

```bash
git add training/azureml_job.yml training/conda_env.yml
git commit -m "feat: add Azure ML training job specification"
```

---

### Task 7: GitHub Actions — CI Workflow

Create the CI workflow that runs on every push and PR.

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `requirements.txt`, `tests/`
- Produces: CI status check on PRs

- [ ] **Step 1: Create CI workflow**

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install ruff mypy pytest

      - name: Lint
        run: ruff check .

      - name: Type check
        run: mypy api/ lib/ training/ monitoring/ --ignore-missing-imports

      - name: Unit tests
        run: pytest tests/ -v

  docker-build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build Docker image
        run: docker build -t http-anomaly-detection:test .
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add lint, test, and Docker build workflow"
```

---

### Task 8: GitHub Actions — Train & Deploy Workflow

Create the train-deploy workflow with quality gate and canary deployment.

**Files:**
- Create: `.github/workflows/train-deploy.yml`

**Interfaces:**
- Consumes: Azure OIDC credentials (GitHub secrets), `training/azureml_job.yml`, `infra/deploy_container_app.sh`
- Produces: Trained model in MLflow registry + canary deployment

- [ ] **Step 1: Create train-deploy workflow**

```yaml
# .github/workflows/train-deploy.yml
name: Train & Deploy

on:
  push:
    branches: [main]
    paths:
      - "training/**"
      - "data/**"
      - "lib/**"
  workflow_dispatch:

permissions:
  id-token: write
  contents: read
  pull-requests: write

env:
  AZURE_RESOURCE_GROUP: ${{ vars.AZURE_RESOURCE_GROUP }}
  AZURE_ML_WORKSPACE: ${{ vars.AZURE_ML_WORKSPACE }}

jobs:
  train:
    runs-on: ubuntu-latest
    outputs:
      run_id: ${{ steps.train.outputs.run_id }}
      silhouette_score: ${{ steps.train.outputs.silhouette_score }}
      anomaly_rate: ${{ steps.train.outputs.anomaly_rate }}
      model_version: ${{ steps.train.outputs.model_version }}
    steps:
      - uses: actions/checkout@v4

      - name: Azure login (OIDC)
        uses: azure/login@v2
        with:
          client-id: ${{ secrets.AZURE_CLIENT_ID }}
          tenant-id: ${{ secrets.AZURE_TENANT_ID }}
          subscription-id: ${{ secrets.AZURE_SUBSCRIPTION_ID }}

      - name: Install Azure ML CLI
        run: az extension add --name ml --yes

      - name: Submit training job
        id: train
        run: |
          JOB_NAME=$(az ml job create \
            --file training/azureml_job.yml \
            --resource-group "$AZURE_RESOURCE_GROUP" \
            --workspace-name "$AZURE_ML_WORKSPACE" \
            --query name -o tsv)

          echo "Waiting for job $JOB_NAME..."
          az ml job stream \
            --name "$JOB_NAME" \
            --resource-group "$AZURE_RESOURCE_GROUP" \
            --workspace-name "$AZURE_ML_WORKSPACE"

          STATUS=$(az ml job show \
            --name "$JOB_NAME" \
            --resource-group "$AZURE_RESOURCE_GROUP" \
            --workspace-name "$AZURE_ML_WORKSPACE" \
            --query status -o tsv)

          if [ "$STATUS" != "Completed" ]; then
            echo "::error::Training job failed with status: $STATUS"
            exit 1
          fi

          # Extract metrics from MLflow
          pip install mlflow azure-ai-ml azure-identity
          python -c "
          from azure.ai.ml import MLClient
          from azure.identity import DefaultAzureCredential
          import mlflow, os, json

          ml_client = MLClient(
              DefaultAzureCredential(),
              os.environ['AZURE_SUBSCRIPTION_ID'],
              os.environ['AZURE_RESOURCE_GROUP'],
              os.environ['AZURE_ML_WORKSPACE'],
          )
          tracking_uri = ml_client.workspaces.get(os.environ['AZURE_ML_WORKSPACE']).mlflow_tracking_uri
          mlflow.set_tracking_uri(tracking_uri)

          experiment = mlflow.get_experiment_by_name('http-anomaly-isolation-forest')
          runs = mlflow.search_runs([experiment.experiment_id], order_by=['start_time DESC'], max_results=1)
          run = runs.iloc[0]

          with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
              f.write(f'run_id={run.run_id}\n')
              f.write(f'silhouette_score={run[\"metrics.silhouette_score\"]}\n')
              f.write(f'anomaly_rate={run[\"metrics.anomaly_rate\"]}\n')
              f.write(f'model_version={run[\"tags.mlflow.model_version\"]}\n')
          "
        env:
          AZURE_SUBSCRIPTION_ID: ${{ secrets.AZURE_SUBSCRIPTION_ID }}

  quality-gate:
    needs: train
    runs-on: ubuntu-latest
    steps:
      - name: Check quality metrics
        run: |
          SILHOUETTE="${{ needs.train.outputs.silhouette_score }}"
          ANOMALY_RATE="${{ needs.train.outputs.anomaly_rate }}"

          echo "Silhouette score: $SILHOUETTE"
          echo "Anomaly rate: $ANOMALY_RATE"

          python3 -c "
          sil = float('$SILHOUETTE')
          rate = float('$ANOMALY_RATE')

          if sil < 0.1:
              print('::error::Silhouette score too low: {}'.format(sil))
              exit(1)

          if rate < 0.03 or rate > 0.07:
              print('::error::Anomaly rate out of range: {}'.format(rate))
              exit(1)

          print('Quality gate PASSED')
          print(f'  Silhouette: {sil:.4f} (>= 0.1)')
          print(f'  Anomaly rate: {rate:.4f} (0.03-0.07)')
          "

  deploy-canary:
    needs: [train, quality-gate]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Azure login (OIDC)
        uses: azure/login@v2
        with:
          client-id: ${{ secrets.AZURE_CLIENT_ID }}
          tenant-id: ${{ secrets.AZURE_TENANT_ID }}
          subscription-id: ${{ secrets.AZURE_SUBSCRIPTION_ID }}

      - name: Deploy canary (10% traffic)
        run: |
          CANARY_PERCENT=10 \
          IMAGE_TAG="${{ needs.train.outputs.model_version }}" \
          bash infra/deploy_container_app.sh
        env:
          AZURE_CONTAINER_APP_NAME: ${{ vars.AZURE_CONTAINER_APP_NAME }}
          AZURE_CONTAINER_REGISTRY: ${{ vars.AZURE_CONTAINER_REGISTRY }}

      - name: Monitor canary (10 minutes)
        run: |
          echo "Monitoring canary for 10 minutes..."
          for i in $(seq 1 10); do
            echo "Minute $i/10"
            HEALTH=$(curl -sf "https://${{ vars.AZURE_CONTAINER_APP_URL }}/health" || echo "FAIL")
            if [ "$HEALTH" = "FAIL" ]; then
              echo "::error::Health check failed at minute $i"
              exit 1
            fi
            sleep 60
          done
          echo "Canary monitoring passed"

      - name: Promote to 100%
        run: |
          NEW_REVISION=$(az containerapp revision list \
            --name "${{ vars.AZURE_CONTAINER_APP_NAME }}" \
            --resource-group "$AZURE_RESOURCE_GROUP" \
            --query "[-1].name" -o tsv)

          az containerapp ingress traffic set \
            --name "${{ vars.AZURE_CONTAINER_APP_NAME }}" \
            --resource-group "$AZURE_RESOURCE_GROUP" \
            --revision-weight "${NEW_REVISION}=100"

          echo "Canary promoted to 100%"
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/train-deploy.yml
git commit -m "ci: add train, quality gate, and canary deploy workflow"
```

---

### Task 9: GitHub Actions — Retrain on Drift Workflow

Create the webhook-triggered retrain workflow.

**Files:**
- Create: `.github/workflows/retrain-on-drift.yml`

**Interfaces:**
- Consumes: Azure Monitor webhook, same Azure secrets as Task 8
- Produces: Retrain pipeline execution triggered by drift detection

- [ ] **Step 1: Create retrain workflow**

```yaml
# .github/workflows/retrain-on-drift.yml
name: Retrain on Drift

on:
  repository_dispatch:
    types: [drift-detected]
  workflow_dispatch:
    inputs:
      reason:
        description: "Reason for manual retrain"
        required: false
        default: "Manual trigger"

permissions:
  id-token: write
  contents: read

env:
  AZURE_RESOURCE_GROUP: ${{ vars.AZURE_RESOURCE_GROUP }}
  AZURE_ML_WORKSPACE: ${{ vars.AZURE_ML_WORKSPACE }}

jobs:
  log-trigger:
    runs-on: ubuntu-latest
    steps:
      - name: Log drift event
        run: |
          echo "=== Retrain Triggered ==="
          echo "Event: ${{ github.event_name }}"
          if [ "${{ github.event_name }}" = "repository_dispatch" ]; then
            echo "Drift payload: ${{ toJSON(github.event.client_payload) }}"
          else
            echo "Reason: ${{ github.event.inputs.reason }}"
          fi

  retrain:
    needs: log-trigger
    uses: ./.github/workflows/train-deploy.yml
    secrets: inherit
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/retrain-on-drift.yml
git commit -m "ci: add drift-triggered retrain workflow"
```

---

### Task 10: Integration Test and Documentation

Add an integration test for the full predict → log flow and update the README with MLOps documentation.

**Files:**
- Create: `tests/test_integration.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: all previous tasks
- Produces: integration test + documentation

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
"""Integration test: predict endpoint logs features correctly."""
import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient


def test_predict_logs_features(tmp_path, monkeypatch):
    monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", "")
    monkeypatch.setenv("FEATURE_LOG_CONTAINER", "test")

    # Patch the logger to use local dir
    import monitoring.logger as logger_mod

    original_init = logger_mod.FeatureLogger.__init__

    def patched_init(self, connection_string=None, container_name="feature-logs", local_dir=None):
        original_init(self, connection_string=None, container_name=container_name, local_dir=tmp_path)

    monkeypatch.setattr(logger_mod.FeatureLogger, "__init__", patched_init)

    from api.main import app

    client = TestClient(app)

    response = client.post("/predict", json={
        "ip": "10.0.0.1",
        "timestamp": 1751500000.0,
        "endpoint": "/api/test",
        "method": "GET",
        "status_code": 200,
        "payload_size": 1000,
        "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "response_time": 50,
    })

    assert response.status_code == 200
    data = response.json()
    assert "is_anomaly" in data
    assert "anomaly_score" in data

    csv_files = list(tmp_path.glob("*.csv"))
    assert len(csv_files) >= 1


def test_predict_works_without_logger(monkeypatch):
    """Predict should work even if logger is not configured."""
    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)

    from api.main import app

    client = TestClient(app)

    response = client.post("/predict", json={
        "ip": "10.0.0.2",
        "timestamp": 1751500000.0,
        "endpoint": "/api/test",
        "method": "GET",
        "status_code": 200,
        "payload_size": 1000,
        "user_agent": "Mozilla/5.0",
        "response_time": 50,
    })

    assert response.status_code == 200
```

- [ ] **Step 2: Run integration tests**

Run: `cd /Users/felipetomepereira/Projects/cloudwalk_fastapi_model && python -m pytest tests/test_integration.py -v`
Expected: 2 tests PASS

- [ ] **Step 3: Add MLOps section to README**

Add the following section to `README.md` (after the existing content):

```markdown
## MLOps Pipeline

### Architecture

```
Code push → GitHub Actions CI → Train (Azure ML) → Quality Gate → Canary Deploy
                                                                      ↓
Drift detected ← Azure Monitor ← Data Drift Monitor ← Feature Logs ← Production
      ↓
Retrain (automated) → Quality Gate → Canary Deploy → Production
```

### Components

| Component | Tool | Purpose |
|-----------|------|---------|
| Experiment tracking | MLflow on Azure ML | Log params, metrics, artifacts per training run |
| Model registry | MLflow Model Registry | Version models with stages (Staging → Production) |
| CI | GitHub Actions | Lint, test, Docker build on every push |
| CD | GitHub Actions + Azure Container Apps | Quality gate → canary deploy (10% → 100%) |
| Drift monitoring | Azure ML Data Drift Monitor | Daily PSI check on production features |
| Auto-retrain | GitHub Actions webhook | Drift → retrain → validate → deploy |

### Running Locally

```bash
# Train with MLflow tracking (local)
python -m training.train_model

# Generate baseline for drift monitoring
python -m monitoring.register_baseline --local-only
```

### Azure Setup

```bash
# 1. Provision infrastructure
bash infra/setup_workspace.sh

# 2. Register baseline dataset
python -m monitoring.register_baseline

# 3. Configure drift monitor
python -m monitoring.setup_drift_monitor

# 4. Deploy
bash infra/deploy_container_app.sh
```

### GitHub Secrets Required

| Secret | Description |
|--------|-------------|
| `AZURE_CLIENT_ID` | App registration client ID (OIDC) |
| `AZURE_TENANT_ID` | Azure AD tenant ID |
| `AZURE_SUBSCRIPTION_ID` | Azure subscription ID |

### GitHub Variables Required

| Variable | Description |
|----------|-------------|
| `AZURE_RESOURCE_GROUP` | Resource group name |
| `AZURE_ML_WORKSPACE` | Azure ML workspace name |
| `AZURE_CONTAINER_APP_NAME` | Container App name |
| `AZURE_CONTAINER_REGISTRY` | ACR name |
| `AZURE_CONTAINER_APP_URL` | Container App FQDN |
```

- [ ] **Step 4: Run full test suite**

Run: `cd /Users/felipetomepereira/Projects/cloudwalk_fastapi_model && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_integration.py README.md
git commit -m "feat: add integration tests and MLOps pipeline documentation"
```
