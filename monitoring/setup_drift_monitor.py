"""Configure Azure ML Data Drift Monitor.

Usage: python -m monitoring.setup_drift_monitor
Requires: AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_ML_WORKSPACE env vars
"""
import os

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    DataDriftMonitor,
    MonitorSchedule,
    CronTrigger,
    AlertNotification,
)
from azure.identity import DefaultAzureCredential

from lib.feature_engineering import FEATURE_NAMES

PSI_THRESHOLD = 0.2
DRIFT_FEATURE_COUNT_THRESHOLD = 2


def create_drift_monitor() -> None:
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_ML_WORKSPACE"],
    )

    monitor = DataDriftMonitor(
        name="http-anomaly-drift-monitor",
        baseline_data="http-anomaly-baseline:1",
        target_data="http-anomaly-production:latest",
        features=FEATURE_NAMES,
        frequency="Day",
        alert_config=AlertNotification(
            emails=[os.environ.get("ALERT_EMAIL", "")],
        ),
        compute=os.environ.get("AZURE_ML_COMPUTE", "cpu-cluster"),
        threshold=PSI_THRESHOLD,
    )

    schedule = MonitorSchedule(
        name="drift-daily-check",
        trigger=CronTrigger(expression="0 6 * * *"),
        create_monitor=monitor,
    )

    ml_client.schedules.begin_create_or_update(schedule)
    print("Drift monitor configured: daily at 06:00 UTC")
    print(f"PSI threshold: {PSI_THRESHOLD}")
    print(f"Features monitored: {', '.join(FEATURE_NAMES)}")


def main():
    create_drift_monitor()


if __name__ == "__main__":
    main()
