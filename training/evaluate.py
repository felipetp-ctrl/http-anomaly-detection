import pandas as pd
from sklearn.metrics import classification_report

from lib.feature_engineering import FEATURE_NAMES
from lib.model_loader import load_artifacts

DATA_DIR = __import__("pathlib").Path(__file__).resolve().parent.parent / "data"


def baseline_rule(row) -> int:
    if row["request_count_5min"] > 100:
        return -1
    if row["status_401_ratio"] > 0.5:
        return -1
    return 1


def main():
    model, scaler, _ = load_artifacts()
    df = pd.read_csv(DATA_DIR / "features.csv")

    X = df[FEATURE_NAMES].values
    X_scaled = scaler.transform(X)

    true_labels = (df["label"] == "legitimate").astype(int).map({1: 1, 0: -1})

    preds = model.predict(X_scaled)
    print("=== Isolation Forest ===")
    print(classification_report(true_labels, preds, target_names=["anomaly", "normal"]))

    baseline_preds = df.apply(baseline_rule, axis=1)
    print("=== Baseline (rules) ===")
    print(classification_report(true_labels, baseline_preds, target_names=["anomaly", "normal"]))


if __name__ == "__main__":
    main()
