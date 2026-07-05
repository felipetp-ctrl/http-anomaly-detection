import csv
import io
import logging
from datetime import datetime, timezone
from pathlib import Path

from lib.feature_engineering import FEATURE_NAMES

logger = logging.getLogger(__name__)

_CSV_FIELDS = ["timestamp", "prediction", "anomaly_score"] + FEATURE_NAMES


class FeatureLogger:
    def __init__(
        self,
        connection_string: str | None = None,
        container_name: str = "feature-logs",
        local_dir: Path | None = None,
    ):
        self._connection_string = connection_string
        self._container_name = container_name
        self._local_dir = local_dir

    def log(
        self,
        features: list[float],
        prediction: int,
        anomaly_score: float,
        timestamp: float,
    ) -> None:
        row = {
            "timestamp": timestamp,
            "prediction": prediction,
            "anomaly_score": anomaly_score,
        }
        for name, value in zip(FEATURE_NAMES, features):
            row[name] = value

        if self._connection_string:
            self._append_to_blob(row, timestamp)
        else:
            self._append_to_local(row, timestamp)

    def _blob_name(self, timestamp: float) -> str:
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        return f"features-{dt.strftime('%Y-%m-%d')}.csv"

    def _append_to_local(self, row: dict, timestamp: float) -> None:
        if self._local_dir is None:
            return
        filename = self._blob_name(timestamp)
        filepath = self._local_dir / filename
        is_new = not filepath.exists()

        with open(filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerow(row)

    def _append_to_blob(self, row: dict, timestamp: float) -> None:
        from azure.storage.blob import BlobServiceClient

        blob_name = self._blob_name(timestamp)
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=_CSV_FIELDS)

        blob_service = BlobServiceClient.from_connection_string(self._connection_string)
        container_client = blob_service.get_container_client(self._container_name)

        try:
            container_client.get_container_properties()
        except Exception:
            container_client.create_container()

        blob_client = container_client.get_blob_client(blob_name)

        try:
            blob_client.get_blob_properties()
        except Exception:
            writer.writeheader()

        writer.writerow(row)
        blob_client.upload_blob(buf.getvalue(), blob_type="AppendBlob", overwrite=False)
