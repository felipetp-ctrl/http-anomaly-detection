import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from lib.feature_engineering import FEATURE_NAMES

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

N_ESTIMATORS = 100
CONTAMINATION = 0.05


def main():
    df = pd.read_csv(DATA_DIR / "features.csv")
    X = df[FEATURE_NAMES].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=42,
    )
    model.fit(X_scaled)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    # joblib is required here — scikit-learn models are not JSON-serializable
    joblib.dump(model, ARTIFACTS_DIR / "model.joblib")
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")

    metadata = {
        "feature_names": FEATURE_NAMES,
        "n_estimators": N_ESTIMATORS,
        "contamination": CONTAMINATION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_samples": len(X),
    }
    with open(ARTIFACTS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Trained on {len(X)} samples. Artifacts saved to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
