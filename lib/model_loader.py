import json
from pathlib import Path

import joblib

from lib.feature_engineering import FEATURE_NAMES

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def load_artifacts(artifacts_dir: Path | None = None):
    d = artifacts_dir or ARTIFACTS_DIR

    model = joblib.load(d / "model.joblib")
    scaler = joblib.load(d / "scaler.joblib")

    with open(d / "metadata.json") as f:
        metadata = json.load(f)

    expected = metadata["feature_names"]
    if expected != FEATURE_NAMES:
        raise ValueError(
            f"Feature mismatch. Model expects {expected}, code provides {FEATURE_NAMES}"
        )

    return model, scaler, metadata
