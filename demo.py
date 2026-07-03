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


def send(payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(PREDICT_URL, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def print_result(result: dict):
    score = result["anomaly_score"]
    is_anom = result["is_anomaly"]
    bar_len = int(abs(score) * 30)
    bar = "█" * bar_len + "░" * (30 - bar_len)

    color = "\033[91m" if is_anom else "\033[92m"
    reset = "\033[0m"
    tag = f"{color}{'ANOMALY' if is_anom else 'NORMAL '}{reset}"

    features = result["top_features"]
    top3 = list(features.items())[:3]
    feat_str = ", ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in top3)

    print(f"  {tag}  score={score:+.4f}  [{bar}]  {feat_str}")


def run_scenario(title: str, ip: str, requests: list[dict], show_every: int = 1):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"  IP: {ip}  |  {len(requests)} requests")
    print(f"{'='*70}")

    for i, req in enumerate(requests):
        result = send(req)
        if (i % show_every) == 0 or i == len(requests) - 1:
            print(f"  req #{i+1:4d}/{len(requests)}", end="")
            print_result(result)


def scenario_legitimate():
    ip = "203.0.113.10"
    base_ts = 1751500000.0
    endpoints = ["/api/products", "/api/users", "/home", "/about", "/api/search",
                 "/js/app.js", "/css/style.css", "/images/logo.png"]
    ua = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"

    requests = []
    for i in range(10):
        requests.append({
            "ip": ip, "method": "GET", "status_code": 200,
            "timestamp": base_ts + i * 35.0,
            "endpoint": endpoints[i % len(endpoints)],
            "payload_size": 5000 + i * 3000,
            "user_agent": ua,
            "response_time": 50 + i * 40,
        })
    return ip, requests


def scenario_credential_stuffing():
    ip = "198.51.100.77"
    base_ts = 1751500000.0
    ua = "python-requests/2.31"

    requests = []
    for i in range(400):
        status = 401 if i % 10 != 9 else 200
        requests.append({
            "ip": ip, "method": "POST",
            "status_code": status,
            "timestamp": base_ts + i * 0.05,
            "endpoint": "/login",
            "payload_size": 256 + (i % 3),
            "user_agent": ua,
            "response_time": 45 + (i % 5) * 0.5,
        })
    return ip, requests


def scenario_ddos():
    ip = "192.0.2.200"
    base_ts = 1751500000.0
    ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    endpoints = ["/api/checkout", "/api/search"]
    status_cycle = [200, 200, 200, 503, 504]

    requests = []
    for i in range(1200):
        requests.append({
            "ip": ip, "method": "GET",
            "status_code": status_cycle[i % 5],
            "timestamp": base_ts + i * 0.015,
            "endpoint": endpoints[i % len(endpoints)],
            "payload_size": 100 + (i % 3) * 2,
            "user_agent": ua,
            "response_time": 10 + (i % 5) * 0.3,
        })
    return ip, requests


def scenario_malicious_bot():
    ip = "172.16.50.99"
    base_ts = 1751500000.0
    endpoints = ["/.env", "/.git/config", "/admin", "/wp-login.php", "/phpmyadmin",
                 "/api/users", "/.aws/credentials", "/server-status", "/actuator",
                 "/graphql", "/debug", "/console", "/.svn/entries", "/backup.sql",
                 "/wp-admin", "/config.php", "/.htaccess", "/xmlrpc.php",
                 "/api/v1/debug", "/metrics", "/health", "/trace", "/dump"]
    uas = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    ]
    status_cycle = [200, 404, 404, 403, 200, 404]

    requests = []
    for i in range(200):
        requests.append({
            "ip": ip, "method": "GET",
            "status_code": status_cycle[i % len(status_cycle)],
            "timestamp": base_ts + i * 1.5,
            "endpoint": endpoints[i % len(endpoints)],
            "payload_size": 200 + (i % 5) * 150,
            "user_agent": uas[i % len(uas)],
            "response_time": 10 + (i % 7) * 5,
        })
    return ip, requests


def main():
    print("\n" + "="*70)
    print("  HTTP ANOMALY DETECTION — LIVE DEMO")
    print(f"  API: {BASE_URL}")
    print("="*70)

    input("\n  [Enter] Cenário 1: Tráfego legítimo (10 requests espaçados)")
    ip, reqs = scenario_legitimate()
    run_scenario("CENÁRIO 1 — Usuário legítimo navegando normalmente", ip, reqs)

    input("\n  [Enter] Cenário 2: Credential stuffing (400 POSTs em /login)")
    ip, reqs = scenario_credential_stuffing()
    run_scenario("CENÁRIO 2 — Credential stuffing: 400 POSTs, 90% status 401, bot UA", ip, reqs, show_every=50)

    input("\n  [Enter] Cenário 3: L7 DDoS (1200 requests em rajada)")
    ip, reqs = scenario_ddos()
    run_scenario("CENÁRIO 3 — L7 DDoS: 1200 requests, browser UA, mix 200/503/504", ip, reqs, show_every=100)

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
