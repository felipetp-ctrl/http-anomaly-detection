"""
Local smoke test and demo runner for the ultra-low-latency Rust server.

Usage:
  python ultra-low-latency/demo_local.py
  python ultra-low-latency/demo_local.py --base-url http://127.0.0.1:8080
  python ultra-low-latency/demo_local.py --scenario ddos --predict
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import subprocess
import uuid
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


DEFAULT_BASE_URL = "http://127.0.0.1:8080"
SCENARIOS = ["legitimate", "credential_stuffing", "ddos", "malicious_bot"]

SERVER_DIR = Path(__file__).resolve().parent / "server"
SERVER_COMMAND = ["cargo", "run", "--release"]
DEFAULT_STARTUP_TIMEOUT_SECONDS = 180
RUN_ID = uuid.uuid4().hex[:8]


def demo_ip(label: str) -> str:
    return f"demo-{RUN_ID}-{label}"


def request_json(method: str, url: str, payload: dict[str, Any] | None = None) -> Any:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def request_json_or_none(method: str, url: str, payload: dict[str, Any] | None = None) -> Any | None:
    try:
        return request_json(method, url, payload)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def request_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"Accept": "text/html"}, method="GET")
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.read().decode("utf-8")


def request_text_or_none(url: str) -> str | None:
    try:
        return request_text(url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def is_local_base_url(base_url: str) -> bool:
    hostname = urlparse(base_url).hostname
    return hostname in {"127.0.0.1", "localhost"}


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def launch_local_server() -> tuple[subprocess.Popen[str], str]:
    port = find_free_port()
    env = os.environ.copy()
    env["PORT"] = str(port)
    process = subprocess.Popen(
        SERVER_COMMAND,
        cwd=SERVER_DIR,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"

    deadline = time.time() + DEFAULT_STARTUP_TIMEOUT_SECONDS
    while time.time() < deadline:
        try:
            payload = request_json("GET", f"{base_url}/health")
            if payload.get("status") == "ok":
                print(f"started local ultra-low-latency server on {base_url}")
                return process, base_url
        except Exception:
            time.sleep(1.0)

    process.terminate()
    raise RuntimeError("timed out waiting for local ultra-low-latency server to start")


def probe_predict(base_url: str) -> bool:
    try:
        payload = request_json("POST", f"{base_url}/predict", build_predict_payload())
        print(f"/predict OK  anomaly={payload['is_anomaly']}  score={payload['anomaly_score']}")
        return True
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise


def assert_health(base_url: str) -> None:
    payload = request_json("GET", f"{base_url}/health")
    if payload.get("status") != "ok":
        raise RuntimeError(f"unexpected /health payload: {payload!r}")
    print("/health OK")


def assert_openapi(base_url: str) -> None:
    payload = request_json_or_none("GET", f"{base_url}/openapi.json")
    if payload is None:
        print("/openapi.json missing (skipping; rebuild the Rust server to enable docs)")
        return
    paths = payload.get("paths", {})
    expected_paths = {"/health", "/predict", "/demo", "/demo/{scenario}"}
    missing = sorted(expected_paths - set(paths))
    if missing:
        raise RuntimeError(f"missing OpenAPI paths: {missing}")
    print("/openapi.json OK")


def assert_docs(base_url: str) -> None:
    html = request_text_or_none(f"{base_url}/docs")
    if html is None:
        print("/docs missing (skipping; rebuild the Rust server to enable Swagger UI)")
        return
    markers = ["SwaggerUIBundle", "swagger-ui", "HTTP Anomaly Detection API Docs"]
    if not any(marker in html for marker in markers):
        raise RuntimeError("/docs does not look like Swagger UI")
    print("/docs OK")


def build_predict_payload() -> dict[str, Any]:
    return {
        "ip": demo_ip("predict-203.0.113.10"),
        "timestamp": 1751500000.0,
        "endpoint": "/api/products",
        "method": "GET",
        "status_code": 200,
        "payload_size": 5000.0,
        "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "response_time": 50.0,
    }


def build_local_demo_requests() -> dict[str, dict[str, Any]]:
    base_ts = 1_751_500_000.0
    browser_ua = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"

    legitimate_requests: list[dict[str, Any]] = []
    legitimate_endpoints = ["/api/products", "/api/users", "/home", "/about", "/api/search", "/js/app.js", "/css/style.css", "/images/logo.png"]
    for index in range(10):
        legitimate_requests.append({
            "ip": demo_ip("legit-203.0.113.10"),
            "method": "GET",
            "status_code": 200,
            "timestamp": base_ts + index * 35.0,
            "endpoint": legitimate_endpoints[index % len(legitimate_endpoints)],
            "payload_size": 5000 + index * 3000,
            "user_agent": browser_ua,
            "response_time": 50 + index * 40,
        })

    stuffing_requests: list[dict[str, Any]] = []
    bot_ua = "python-requests/2.31"
    for index in range(400):
        stuffing_requests.append({
            "ip": demo_ip("cred-198.51.100.77"),
            "method": "POST",
            "status_code": 401 if index % 10 != 9 else 200,
            "timestamp": base_ts + index * 0.05,
            "endpoint": "/login",
            "payload_size": 256 + (index % 3),
            "user_agent": bot_ua,
            "response_time": 45 + (index % 5) * 0.5,
        })

    ddos_requests: list[dict[str, Any]] = []
    ddos_endpoints = ["/api/checkout", "/api/search"]
    ddos_status_cycle = [200, 200, 200, 503, 504]
    for index in range(1200):
        status = ddos_status_cycle[index % len(ddos_status_cycle)]
        ddos_requests.append({
            "ip": demo_ip("ddos-192.0.2.200"),
            "method": "GET",
            "status_code": status,
            "timestamp": base_ts + index * 0.015,
            "endpoint": ddos_endpoints[index % len(ddos_endpoints)],
            "payload_size": 200 + ((index * 7919) % 600),
            "user_agent": browser_ua,
            "response_time": (30 + ((index * 37) % 40)) if status == 200 else (600 + ((index * 53) % 400)),
        })

    malicious_requests: list[dict[str, Any]] = []
    malicious_endpoints = ["/.env", "/.git/config", "/admin", "/wp-login.php", "/phpmyadmin", "/api/users", "/.aws/credentials", "/server-status", "/actuator", "/graphql", "/debug", "/console", "/.svn/entries", "/backup.sql", "/wp-admin", "/config.php", "/.htaccess", "/xmlrpc.php", "/api/v1/debug", "/metrics", "/health", "/trace", "/dump"]
    malicious_uas = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    ]
    malicious_status_cycle = [200, 404, 404, 403, 200, 404]
    for index in range(200):
        malicious_requests.append({
            "ip": demo_ip("bot-172.16.50.99"),
            "method": "GET",
            "status_code": malicious_status_cycle[index % len(malicious_status_cycle)],
            "timestamp": base_ts + index * 1.5,
            "endpoint": malicious_endpoints[index % len(malicious_endpoints)],
            "payload_size": 10 + ((index * 97) % 2490),
            "user_agent": malicious_uas[index % len(malicious_uas)],
            "response_time": 1 + ((index * 29) % 129),
        })

    return {
        "legitimate": {"ip": demo_ip("legit-203.0.113.10"), "requests": legitimate_requests},
        "credential_stuffing": {"ip": demo_ip("cred-198.51.100.77"), "requests": stuffing_requests},
        "ddos": {"ip": demo_ip("ddos-192.0.2.200"), "requests": ddos_requests},
        "malicious_bot": {"ip": demo_ip("bot-172.16.50.99"), "requests": malicious_requests},
    }


def run_predict(base_url: str) -> None:
    payload = request_json("POST", f"{base_url}/predict", build_predict_payload())
    required_fields = {"ip", "is_anomaly", "anomaly_score", "top_features", "timing_ms"}
    missing = sorted(required_fields - set(payload))
    if missing:
        raise RuntimeError(f"unexpected /predict response: missing {missing}")
    print(f"/predict OK  anomaly={payload['is_anomaly']}  score={payload['anomaly_score']}")


def print_demo_summary(scenario: str, payload: dict[str, Any]) -> None:
    samples = payload.get("samples", [])
    first_sample = samples[0] if samples else {}
    print(
        f"/demo/{scenario} OK  total={payload.get('total_requests')}  "
        f"anomalies={payload.get('anomalies_detected')}  first={payload.get('first_anomaly_at')}"
    )
    if first_sample:
        print(
            f"  sample#{first_sample.get('request_number')}  "
            f"anomaly={first_sample.get('is_anomaly')}  score={first_sample.get('anomaly_score')}"
        )


def run_demo_remote(base_url: str, scenario: str) -> None:
    payload = request_json("GET", f"{base_url}/demo/{scenario}")
    print_demo_summary(scenario, payload)


def run_demo_local(base_url: str, scenario: str) -> None:
    demo_data = build_local_demo_requests()[scenario]
    requests = demo_data["requests"]
    total = len(requests)
    anomaly_count = 0
    first_anomaly_at: int | None = None
    samples: list[dict[str, Any]] = []
    sample_every = max(1, total // 20)

    for index, request in enumerate(requests):
        payload = request_json("POST", f"{base_url}/predict", request)
        is_anomaly = bool(payload.get("is_anomaly"))
        if is_anomaly:
            anomaly_count += 1
            if first_anomaly_at is None:
                first_anomaly_at = index + 1
        if index % sample_every == 0 or index + 1 == total:
            samples.append({
                "request_number": index + 1,
                "is_anomaly": is_anomaly,
                "anomaly_score": payload.get("anomaly_score"),
            })

    print_demo_summary(scenario, {
        "total_requests": total,
        "anomalies_detected": anomaly_count,
        "first_anomaly_at": first_anomaly_at,
        "samples": samples,
    })


def list_demo_scenarios(base_url: str) -> tuple[list[str], bool]:
    payload = request_json_or_none("GET", f"{base_url}/demo")
    if payload is None:
        print("/demo missing (using local demo scenario generator)")
        return SCENARIOS, False
    scenarios = payload.get("available_scenarios", [])
    names = [item["name"] for item in scenarios if isinstance(item, dict) and item.get("name")]
    if not names:
        raise RuntimeError(f"unexpected /demo payload: {payload!r}")
    print("/demo OK  scenarios=" + ", ".join(names))
    return names, True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local smoke test for the ultra-low-latency Rust server")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="server base URL (default: http://127.0.0.1:8080)")
    parser.add_argument("--scenario", choices=SCENARIOS + ["all"], default="all", help="demo scenario to run")
    parser.add_argument("--predict", action="store_true", help="also call /predict with a sample payload")
    parser.add_argument("--skip-docs", action="store_true", help="skip /docs validation")
    parser.add_argument("--no-auto-start", action="store_true", help="do not start a local Rust server automatically")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    started_process: subprocess.Popen[str] | None = None

    print(f"Ultra-low-latency local demo smoke test")
    print(f"base_url={base_url}")
    started_at = time.perf_counter()

    try:
        assert_health(base_url)
        if not probe_predict(base_url):
            if args.no_auto_start or not is_local_base_url(base_url):
                raise RuntimeError(
                    f"{base_url} does not look like the ultra-low-latency Rust server; "
                    "start it with `cd ultra-low-latency/server && cargo run --release`"
                )
            started_process, base_url = launch_local_server()
            assert_health(base_url)
            if args.predict:
                run_predict(base_url)
        elif args.predict:
            # probe_predict already printed the result when the core route exists
            pass
        assert_openapi(base_url)
        if not args.skip_docs:
            assert_docs(base_url)

        scenarios, remote_demo_available = list_demo_scenarios(base_url)

        if args.scenario == "all":
            for scenario in scenarios:
                if remote_demo_available:
                    run_demo_remote(base_url, scenario)
                else:
                    run_demo_local(base_url, scenario)
        else:
            if remote_demo_available:
                run_demo_remote(base_url, args.scenario)
            else:
                run_demo_local(base_url, args.scenario)

    except urllib.error.URLError as exc:
        print(f"request failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"smoke test failed: {exc}", file=sys.stderr)
        if started_process is not None:
            started_process.terminate()
        return 1

    if started_process is not None:
        started_process.terminate()

    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    print(f"completed in {elapsed_ms:.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())