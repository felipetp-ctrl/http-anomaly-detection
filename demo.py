"""
Live demo script for interview — simulates attack scenarios against the API.
Usage: python demo.py [url]
Default URL: http://localhost:8000
"""

import json
import sys
import time

import urllib.request

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
PREDICT_URL = f"{BASE_URL}/predict"

LEGITIMATE_USER = {
    "ip": "203.0.113.10",
    "method": "GET",
    "status_code": 200,
    "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
}

CREDENTIAL_STUFFING = {
    "ip": "198.51.100.77",
    "method": "POST",
    "status_code": 401,
    "user_agent": "python-requests/2.31",
}

DDOS = {
    "ip": "192.0.2.200",
    "method": "GET",
    "status_code": 200,
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
}

ENDPOINTS_NORMAL = ["/api/products", "/api/users", "/home", "/about", "/api/search", "/js/app.js"]
ENDPOINTS_STUFFING = ["/login", "/auth", "/login"]
ENDPOINTS_DDOS = ["/api/checkout", "/api/checkout", "/api/checkout"]


def send(payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(PREDICT_URL, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def print_result(label: str, result: dict):
    score = result["anomaly_score"]
    is_anom = result["is_anomaly"]
    bar_len = int(abs(score) * 30)
    bar = "█" * bar_len + "░" * (30 - bar_len)

    color = "\033[91m" if is_anom else "\033[92m"
    reset = "\033[0m"
    tag = f"{color}{'ANOMALY' if is_anom else 'NORMAL '}{reset}"

    features = result["top_features"]
    top3 = list(features.items())[:3]
    feat_str = ", ".join(f"{k}={v}" for k, v in top3)

    print(f"  {tag}  score={score:+.4f}  [{bar}]  {feat_str}")


def scenario(title: str, profile: dict, endpoints: list, count: int, payload_range: tuple, rt_range: tuple):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"  IP: {profile['ip']}  |  UA: {profile['user_agent'][:50]}...")
    print(f"{'='*70}")

    base_ts = time.time()
    for i in range(count):
        payload = {
            **profile,
            "timestamp": base_ts + i * 0.3,
            "endpoint": endpoints[i % len(endpoints)],
            "payload_size": payload_range[0] + (i % 3) * (payload_range[1] - payload_range[0]) // 3,
            "response_time": rt_range[0] + (i % 5) * (rt_range[1] - rt_range[0]) / 5,
        }
        result = send(payload)
        print(f"  req #{i+1:2d}/{count}", end="")
        print_result("", result)
        time.sleep(0.15)


def main():
    print("\n" + "="*70)
    print("  HTTP ANOMALY DETECTION — LIVE DEMO")
    print(f"  API: {BASE_URL}")
    print("="*70)

    input("\n  [Enter] Cenário 1: Tráfego legítimo")
    scenario(
        "CENÁRIO 1 — Usuário legítimo navegando normalmente",
        LEGITIMATE_USER,
        ENDPOINTS_NORMAL,
        count=8,
        payload_range=(5000, 30000),
        rt_range=(50, 400),
    )

    input("\n  [Enter] Cenário 2: Credential stuffing")
    scenario(
        "CENÁRIO 2 — Credential stuffing em /login",
        CREDENTIAL_STUFFING,
        ENDPOINTS_STUFFING,
        count=15,
        payload_range=(200, 300),
        rt_range=(40, 60),
    )

    input("\n  [Enter] Cenário 3: L7 DDoS")
    scenario(
        "CENÁRIO 3 — L7 DDoS em endpoints pesados",
        DDOS,
        ENDPOINTS_DDOS,
        count=40,
        payload_range=(100, 150),
        rt_range=(10, 15),
    )

    print(f"\n{'='*70}")
    print("  DEMO COMPLETA")
    print(f"{'='*70}")
    print("  O modelo diferencia tráfego legítimo de ataques usando:")
    print("  • Volume de requests em janelas de 30s e 5min")
    print("  • Entropia de endpoints e status codes")
    print("  • Regularidade dos intervalos (bots são metrônomos)")
    print("  • Rotação e tipo de user agents")
    print("  • Uniformidade de payload e response time")
    print()


if __name__ == "__main__":
    main()
