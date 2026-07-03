import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from lib.feature_engineering import RequestRecord, compute_features, FEATURE_NAMES

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
WINDOW_30S = 30.0
WINDOW_5MIN = 300.0


def load_logs(csv_path: Path) -> dict[str, list[RequestRecord]]:
    records_by_ip: dict[str, list[RequestRecord]] = defaultdict(list)
    labels_by_ip: dict[str, str] = {}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = datetime.fromisoformat(row["timestamp"]).timestamp()
            rec = RequestRecord(
                timestamp=ts,
                endpoint=row["endpoint"],
                status_code=int(row["status_code"]),
                payload_size=float(row["payload_size"]),
                user_agent=row["user_agent"],
                response_time=float(row["response_time"]),
            )
            ip = row["ip"]
            records_by_ip[ip].append(rec)
            labels_by_ip[ip] = row["label"]

    return records_by_ip, labels_by_ip


def aggregate_features(records_by_ip, labels_by_ip):
    rows = []
    for ip, records in records_by_ip.items():
        records.sort(key=lambda r: r.timestamp)
        latest_ts = records[-1].timestamp

        records_30s = [r for r in records if latest_ts - r.timestamp <= WINDOW_30S]
        records_5min = [r for r in records if latest_ts - r.timestamp <= WINDOW_5MIN]

        features = compute_features(records_30s, records_5min)
        rows.append({
            "ip": ip,
            **dict(zip(FEATURE_NAMES, features)),
            "label": labels_by_ip[ip],
        })

    return rows


def main():
    csv_path = DATA_DIR / "http_logs.csv"
    records_by_ip, labels_by_ip = load_logs(csv_path)
    rows = aggregate_features(records_by_ip, labels_by_ip)

    out_path = DATA_DIR / "features.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ip"] + FEATURE_NAMES + ["label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} feature vectors → {out_path}")


if __name__ == "__main__":
    main()
