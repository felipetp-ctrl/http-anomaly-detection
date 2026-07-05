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
