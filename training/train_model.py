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

        artifacts_dir.mkdir(exist_ok=True, parents=True)
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
