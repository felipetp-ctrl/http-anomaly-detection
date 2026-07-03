"""
Synthetic HTTP Log Generator
Generates realistic HTTP logs with four traffic profiles:
- Legitimate users (~95%)
- Credential stuffing attacks (~2%)
- L7 DDoS attacks (~2%)
- Malicious bots / scrapers (~1%)
"""

import csv
import random
import uuid
from datetime import datetime, timedelta

random.seed(42)

# --- Configuration ---
NUM_LEGITIMATE_IPS = 500
NUM_CREDENTIAL_STUFFING_IPS = 15
NUM_DDOS_IPS = 12
NUM_BOT_IPS = 8

SIMULATION_DURATION_MINUTES = 60
START_TIME = datetime(2026, 7, 1, 10, 0, 0)

OUTPUT_FILE = "http_logs.csv"
FIELDS = ["ip", "timestamp", "method", "endpoint", "status_code",
          "payload_size", "user_agent", "response_time", "label"]

# --- Realistic data pools ---
LEGITIMATE_ENDPOINTS = [
    "/", "/home", "/about", "/contact", "/products", "/products/list",
    "/product/123", "/product/456", "/product/789", "/cart", "/checkout",
    "/search", "/faq", "/terms", "/privacy", "/blog", "/blog/post-1",
    "/blog/post-2", "/account/profile", "/account/settings", "/api/products",
    "/api/search", "/images/logo.png", "/css/main.css", "/js/app.js"
]

REAL_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPad; CPU OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Mobile Safari/537.36",
]

BOT_USER_AGENTS = [
    "python-requests/2.31.0",
    "Go-http-client/1.1",
    "curl/8.4.0",
    "Java/17.0.8",
    "axios/1.6.0",
    "Scrapy/2.11.0",
    "node-fetch/3.3.2",
    "",
]

ROTATED_USER_AGENTS = REAL_USER_AGENTS + [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 11.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0",
]


def generate_ip():
    return f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"


def generate_legitimate_traffic(ip, start_time, duration_minutes):
    """
    Simulates a human browsing session:
    - Irregular intervals (exponential distribution)
    - 1-2 consistent user agents
    - Diverse endpoints in a natural navigation pattern
    - Mostly 200 status codes, occasional 301/304/404
    - Variable payload sizes and response times
    """
    requests = []
    user_agents = random.sample(REAL_USER_AGENTS, k=random.randint(1, 2))
    current_time = start_time + timedelta(seconds=random.uniform(0, duration_minutes * 60))
    
    # Human session: 3-30 requests over the simulation window
    num_requests = random.randint(3, 30)
    
    for _ in range(num_requests):
        # Irregular intervals: exponential with mean ~15-45 seconds
        interval = random.expovariate(1.0 / random.uniform(15, 45))
        current_time += timedelta(seconds=interval)
        
        if current_time > start_time + timedelta(minutes=duration_minutes):
            break
        
        endpoint = random.choice(LEGITIMATE_ENDPOINTS)
        method = "GET"
        if endpoint in ["/checkout", "/contact", "/account/settings"]:
            method = random.choice(["GET", "POST"])
        
        # Mostly 200, with realistic distribution
        status_code = random.choices(
            [200, 301, 304, 404, 500],
            weights=[0.80, 0.05, 0.08, 0.05, 0.02],
            k=1
        )[0]
        
        # Variable payload sizes depending on endpoint type
        if endpoint.startswith("/api/"):
            payload_size = random.randint(200, 5000)
        elif endpoint.endswith((".css", ".js", ".png")):
            payload_size = random.randint(5000, 150000)
        else:
            payload_size = random.randint(500, 25000)
        
        # Variable response time: 50-800ms, occasionally slower
        response_time = random.gauss(250, 150)
        response_time = max(30, min(response_time, 2000))
        
        requests.append({
            "ip": ip,
            "timestamp": current_time.isoformat(),
            "method": method,
            "endpoint": endpoint,
            "status_code": status_code,
            "payload_size": int(payload_size),
            "user_agent": random.choice(user_agents),
            "response_time": round(response_time, 1),
            "label": "legitimate"
        })
    
    return requests


def generate_credential_stuffing(ip, start_time, duration_minutes):
    """
    Simulates credential stuffing:
    - High volume of POST /login or /auth
    - Nearly uniform intervals (automated)
    - Overwhelmingly 401 status codes
    - Uniform payload sizes (username+password body)
    - Single bot user agent
    - Low response times (server rejects quickly)
    """
    requests = []
    target_endpoint = random.choice(["/login", "/auth", "/api/login", "/api/auth"])
    ua = random.choice(BOT_USER_AGENTS)
    
    # Attack starts at a random point in the window
    attack_start = start_time + timedelta(seconds=random.uniform(0, duration_minutes * 30))
    current_time = attack_start
    
    # 200-800 requests in a burst
    num_requests = random.randint(200, 800)
    # Very regular interval: 0.05-0.3 seconds with tiny jitter
    base_interval = random.uniform(0.05, 0.3)
    
    for _ in range(num_requests):
        jitter = random.gauss(0, base_interval * 0.05)  # ~5% jitter
        interval = max(0.01, base_interval + jitter)
        current_time += timedelta(seconds=interval)
        
        if current_time > start_time + timedelta(minutes=duration_minutes):
            break
        
        # 90-95% failures (401), occasional 200 (successful breach), rare 429 (rate limited)
        status_code = random.choices(
            [401, 200, 429, 403],
            weights=[0.92, 0.02, 0.04, 0.02],
            k=1
        )[0]
        
        # Uniform payload: login credentials are roughly same size
        payload_size = random.randint(120, 180)
        
        # Fast server response for auth failures
        response_time = random.gauss(15, 5)
        response_time = max(5, min(response_time, 50))
        
        requests.append({
            "ip": ip,
            "timestamp": current_time.isoformat(),
            "method": "POST",
            "endpoint": target_endpoint,
            "status_code": status_code,
            "payload_size": int(payload_size),
            "user_agent": ua,
            "response_time": round(response_time, 1),
            "label": "credential_stuffing"
        })
    
    return requests


def generate_l7_ddos(ip, start_time, duration_minutes):
    """
    Simulates L7 DDoS:
    - Extremely high request rate
    - Targets resource-heavy endpoints
    - Very regular intervals
    - Mix of methods to stress server
    - May use real-looking user agents to bypass simple filters
    - Uniform response times (server under stress)
    """
    requests = []
    # Target heavy endpoints that strain the server
    heavy_endpoints = ["/search", "/api/products", "/api/search",
                       "/products/list", "/checkout", "/"]
    target_endpoints = random.sample(heavy_endpoints, k=random.randint(1, 3))
    ua = random.choice(REAL_USER_AGENTS)  # Tries to look legitimate
    
    attack_start = start_time + timedelta(seconds=random.uniform(0, duration_minutes * 30))
    current_time = attack_start
    
    # Very high volume: 500-2000 requests
    num_requests = random.randint(500, 2000)
    # Extremely fast, regular intervals
    base_interval = random.uniform(0.02, 0.1)
    
    for _ in range(num_requests):
        jitter = random.gauss(0, base_interval * 0.03)  # ~3% jitter
        interval = max(0.005, base_interval + jitter)
        current_time += timedelta(seconds=interval)
        
        if current_time > start_time + timedelta(minutes=duration_minutes):
            break
        
        endpoint = random.choice(target_endpoints)
        method = random.choice(["GET", "GET", "GET", "POST"])
        
        # Server overwhelmed: mix of 200, 503, 504 (timeouts)
        status_code = random.choices(
            [200, 503, 504, 429],
            weights=[0.40, 0.30, 0.20, 0.10],
            k=1
        )[0]
        
        payload_size = random.randint(100, 500)
        
        # Under DDoS, response times are either very fast (cached/rejected)
        # or slow (server struggling)
        if status_code in [503, 504, 429]:
            response_time = random.gauss(10, 3)
        else:
            response_time = random.gauss(800, 200)
        response_time = max(3, min(response_time, 3000))
        
        requests.append({
            "ip": ip,
            "timestamp": current_time.isoformat(),
            "method": method,
            "endpoint": endpoint,
            "status_code": status_code,
            "payload_size": int(payload_size),
            "user_agent": ua,
            "response_time": round(response_time, 1),
            "label": "l7_ddos"
        })
    
    return requests


def generate_malicious_bot(ip, start_time, duration_minutes):
    """
    Simulates a scraping/crawling bot:
    - Hits many diverse endpoints systematically
    - Rotates user agents to evade detection
    - Semi-regular intervals (faster than human, but not as fast as DDoS)
    - Mostly 200 status codes (successfully scraping)
    - Accesses endpoints humans rarely visit
    """
    requests = []
    
    # Bots access everything, including paths humans rarely visit
    bot_endpoints = LEGITIMATE_ENDPOINTS + [
        "/robots.txt", "/sitemap.xml", "/admin", "/wp-login.php",
        "/.env", "/api/v1/users", "/api/v2/users", "/api/v1/config",
        "/backup.sql", "/debug", "/.git/config", "/product/001",
        "/product/002", "/product/003", "/product/004", "/product/005",
        "/product/006", "/product/007", "/product/008", "/product/009",
        "/product/010", "/product/011", "/product/012", "/product/013",
    ]
    
    attack_start = start_time + timedelta(seconds=random.uniform(0, duration_minutes * 30))
    current_time = attack_start
    
    num_requests = random.randint(100, 500)
    base_interval = random.uniform(0.5, 2.0)
    
    for i in range(num_requests):
        jitter = random.gauss(0, base_interval * 0.08)
        interval = max(0.1, base_interval + jitter)
        current_time += timedelta(seconds=interval)
        
        if current_time > start_time + timedelta(minutes=duration_minutes):
            break
        
        # Systematic crawling: cycles through endpoints
        endpoint = bot_endpoints[i % len(bot_endpoints)]
        
        status_code = random.choices(
            [200, 403, 404, 301],
            weights=[0.65, 0.15, 0.15, 0.05],
            k=1
        )[0]
        
        payload_size = random.randint(200, 2000)
        
        response_time = random.gauss(100, 30)
        response_time = max(20, min(response_time, 500))
        
        # Rotates user agents
        ua = random.choice(ROTATED_USER_AGENTS)
        
        requests.append({
            "ip": ip,
            "timestamp": current_time.isoformat(),
            "method": "GET",
            "endpoint": endpoint,
            "status_code": status_code,
            "payload_size": int(payload_size),
            "user_agent": ua,
            "response_time": round(response_time, 1),
            "label": "malicious_bot"
        })
    
    return requests


def main():
    all_requests = []
    
    print("Generating legitimate traffic...")
    for _ in range(NUM_LEGITIMATE_IPS):
        ip = generate_ip()
        reqs = generate_legitimate_traffic(ip, START_TIME, SIMULATION_DURATION_MINUTES)
        all_requests.extend(reqs)
    
    print("Generating credential stuffing attacks...")
    for _ in range(NUM_CREDENTIAL_STUFFING_IPS):
        ip = generate_ip()
        reqs = generate_credential_stuffing(ip, START_TIME, SIMULATION_DURATION_MINUTES)
        all_requests.extend(reqs)
    
    print("Generating L7 DDoS attacks...")
    for _ in range(NUM_DDOS_IPS):
        ip = generate_ip()
        reqs = generate_l7_ddos(ip, START_TIME, SIMULATION_DURATION_MINUTES)
        all_requests.extend(reqs)
    
    print("Generating malicious bot traffic...")
    for _ in range(NUM_BOT_IPS):
        ip = generate_ip()
        reqs = generate_malicious_bot(ip, START_TIME, SIMULATION_DURATION_MINUTES)
        all_requests.extend(reqs)
    
    # Sort all requests by timestamp (as they would appear in a real log)
    all_requests.sort(key=lambda x: x["timestamp"])
    
    # Write to CSV
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_requests)
    
    # Summary statistics
    total = len(all_requests)
    labels = {}
    for r in all_requests:
        labels[r["label"]] = labels.get(r["label"], 0) + 1
    
    print(f"\nDataset generated: {OUTPUT_FILE}")
    print(f"Total requests: {total}")
    for label, count in sorted(labels.items()):
        print(f"  {label}: {count} ({100*count/total:.1f}%)")


if __name__ == "__main__":
    main()