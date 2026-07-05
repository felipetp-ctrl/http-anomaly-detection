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

    MlflowClient()

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
