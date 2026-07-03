from collections import deque

from lib.feature_engineering import RequestRecord

WINDOW_MAX = 300.0

_ip_records: dict[str, deque[RequestRecord]] = {}


def add_record(ip: str, record: RequestRecord) -> None:
    if ip not in _ip_records:
        _ip_records[ip] = deque()
    _ip_records[ip].append(record)
    _cleanup(ip, record.timestamp)


def get_records(ip: str, now: float, window: float) -> list[RequestRecord]:
    if ip not in _ip_records:
        return []
    cutoff = now - window
    return [r for r in _ip_records[ip] if r.timestamp >= cutoff]


def _cleanup(ip: str, now: float) -> None:
    q = _ip_records[ip]
    cutoff = now - WINDOW_MAX
    while q and q[0].timestamp < cutoff:
        q.popleft()
    if not q:
        del _ip_records[ip]
