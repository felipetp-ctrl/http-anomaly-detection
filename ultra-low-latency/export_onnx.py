import json
from pathlib import Path

import joblib
import numpy as np

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
OUTPUT_DIR = Path(__file__).resolve().parent / "server" / "assets"


def _euler_mascheroni():
    return 0.5772156649015329


def _average_path_length(n):
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (np.log(n - 1) + _euler_mascheroni()) - 2.0 * (n - 1) / n


def main():
    # Safe: loading our own trained model artifacts, not untrusted data
    model = joblib.load(ARTIFACTS_DIR / "model.joblib")
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Export trees as JSON for native Rust inference
    trees = []
    for est in model.estimators_:
        t = est.tree_
        trees.append({
            "children_left": t.children_left.tolist(),
            "children_right": t.children_right.tolist(),
            "feature": t.feature.tolist(),
            "threshold": t.threshold.tolist(),
            "n_node_samples": t.n_node_samples.tolist(),
        })

    model_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "offset": float(model.offset_),
        "max_samples": int(model.max_samples_),
        "n_estimators": model.n_estimators,
        "average_path_length_max_samples": float(
            _average_path_length(model.max_samples_)
        ),
        "trees": trees,
    }

    params_path = OUTPUT_DIR / "model_params.json"
    with open(params_path, "w") as f:
        json.dump(model_params, f)
    print(f"Model params saved to {params_path}")

    # Validate
    X_test = np.array([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])
    X_scaled = scaler.transform(X_test)
    sk_score = model.score_samples(X_scaled)[0]
    sk_pred = model.predict(X_scaled)[0]
    print(f"sklearn  -> pred={sk_pred}, score={sk_score:.10f}")
    print("Export complete.")


if __name__ == "__main__":
    main()
