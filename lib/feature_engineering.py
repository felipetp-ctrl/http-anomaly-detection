import math
from collections import namedtuple

from lib.known_user_agents import is_known_user_agent

RequestRecord = namedtuple("RequestRecord", [
    "timestamp", "endpoint", "status_code",
    "payload_size", "user_agent", "response_time",
])

FEATURE_NAMES = [
    "request_count_30s",
    "request_count_5min",
    "endpoint_entropy",
    "status_code_entropy",
    "status_401_ratio",
    "interval_std",
    "unique_ua_ratio",
    "known_ua_ratio",
    "payload_size_std",
    "response_time_std",
]


def _entropy(values: list) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    counts: dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)


def _std(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def compute_features(records_30s: list[RequestRecord],
                     records_5min: list[RequestRecord]) -> list[float]:
    rc_30 = len(records_30s)
    rc_5 = len(records_5min)

    if rc_5 == 0:
        return [0.0] * len(FEATURE_NAMES)

    endpoints = [r.endpoint for r in records_5min]
    status_codes = [str(r.status_code) for r in records_5min]
    user_agents = [r.user_agent for r in records_5min]

    endpoint_ent = _entropy(endpoints)
    status_ent = _entropy(status_codes)

    status_401_count = sum(1 for r in records_5min if r.status_code == 401)
    status_401_ratio = status_401_count / rc_5

    timestamps = sorted(r.timestamp for r in records_5min)
    intervals = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
    interval_std = _std(intervals)

    unique_uas = len(set(user_agents))
    unique_ua_ratio = unique_uas / rc_5

    known_count = sum(1 for ua in user_agents if is_known_user_agent(ua))
    known_ua_ratio = known_count / rc_5

    payload_sizes = [r.payload_size for r in records_5min]
    payload_std = _std(payload_sizes)

    response_times = [r.response_time for r in records_5min]
    response_std = _std(response_times)

    return [
        float(rc_30),
        float(rc_5),
        endpoint_ent,
        status_ent,
        status_401_ratio,
        interval_std,
        unique_ua_ratio,
        known_ua_ratio,
        payload_std,
        response_std,
    ]
