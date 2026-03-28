AUTO_RESOLVE_THRESHOLD = 0.8

PRIORITY_MAP = {
    "Login Issue": "P2",
    "Network Issue": "P1",
    "Application Error": "P2",
    "Access Issue": "P3",
    "Hardware Issue": "P3",
}

CRITICAL_KEYWORDS = ["outage", "down", "critical", "urgent", "production", "security breach", "ransomware", "data loss"]


def decide(confidence: float, ticket_text: str = "") -> dict:
    text_lower = ticket_text.lower()
    is_critical = any(kw in text_lower for kw in CRITICAL_KEYWORDS)

    if is_critical:
        return {
            "status": "Needs Human Review",
            "reason": "Critical keyword detected — escalated to human agent",
            "auto": False,
            "is_critical": True
        }

    if confidence >= AUTO_RESOLVE_THRESHOLD:
        return {
            "status": "Auto-Resolved",
            "reason": f"Confidence {confidence:.0%} ≥ threshold {AUTO_RESOLVE_THRESHOLD:.0%}",
            "auto": True,
            "is_critical": False
        }
    else:
        return {
            "status": "Needs Human Review",
            "reason": f"Confidence {confidence:.0%} < threshold {AUTO_RESOLVE_THRESHOLD:.0%}",
            "auto": False,
            "is_critical": False
        }


def get_priority(category: str) -> str:
    return PRIORITY_MAP.get(category, "P3")
