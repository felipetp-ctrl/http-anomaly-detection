import json
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
