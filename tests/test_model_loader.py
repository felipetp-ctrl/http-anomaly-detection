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


def test_load_with_mlflow_params_falls_back_when_unavailable(tmp_path):
    """When mlflow_model_name is given but model not found, falls back to local."""
    artifacts_dir = _create_dummy_artifacts(tmp_path)
    model, scaler, metadata = load_artifacts(
        artifacts_dir=artifacts_dir,
        mlflow_model_name="nonexistent-model",
        mlflow_model_stage="Production",
    )
    assert hasattr(model, "predict")
