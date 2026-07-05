import csv
import time
from pathlib import Path

from lib.feature_engineering import FEATURE_NAMES


def test_logger_writes_local_csv(tmp_path):
    from monitoring.logger import FeatureLogger

    logger = FeatureLogger(connection_string=None, container_name="test", local_dir=tmp_path)

    features = [float(i) for i in range(len(FEATURE_NAMES))]
    ts = time.time()
    logger.log(features=features, prediction=1, anomaly_score=-0.35, timestamp=ts)
    logger.log(features=features, prediction=-1, anomaly_score=-0.55, timestamp=ts + 1)

    csv_files = list(tmp_path.glob("*.csv"))
    assert len(csv_files) == 1

    with open(csv_files[0]) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 2
    assert rows[0]["prediction"] == "1"
    assert rows[1]["prediction"] == "-1"
    for name in FEATURE_NAMES:
        assert name in rows[0]


def test_logger_creates_daily_files(tmp_path):
    from monitoring.logger import FeatureLogger

    logger = FeatureLogger(connection_string=None, container_name="test", local_dir=tmp_path)

    features = [0.0] * len(FEATURE_NAMES)
    logger.log(features=features, prediction=1, anomaly_score=-0.3, timestamp=1751500000.0)
    logger.log(features=features, prediction=1, anomaly_score=-0.3, timestamp=1751600000.0)

    csv_files = list(tmp_path.glob("*.csv"))
    assert len(csv_files) == 2


def test_logger_no_local_dir_no_connection_is_noop():
    from monitoring.logger import FeatureLogger

    logger = FeatureLogger(connection_string=None, container_name="test")
    features = [0.0] * len(FEATURE_NAMES)
    # Should not raise even though there's nowhere to log
    logger.log(features=features, prediction=1, anomaly_score=-0.3, timestamp=time.time())
