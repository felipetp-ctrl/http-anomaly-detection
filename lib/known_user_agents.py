KNOWN_BROWSER_PATTERNS = [
    "Mozilla/5.0",
    "Chrome/",
    "Safari/",
    "Firefox/",
    "Edg/",
    "OPR/",
    "Opera/",
]


def is_known_user_agent(ua: str) -> bool:
    return any(pattern in ua for pattern in KNOWN_BROWSER_PATTERNS)
