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
