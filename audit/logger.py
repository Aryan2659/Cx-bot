import json
import uuid
import os
from datetime import datetime, timezone
from config import AUDIT_LOG_DIR


def _log_path() -> str:
    os.makedirs(AUDIT_LOG_DIR, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return os.path.join(AUDIT_LOG_DIR, f"audit_{date_str}.jsonl")


def write_audit(
    session_id: str,
    raw_query: str,
    redacted_query: str,
    routed_to: str,
    router_confidence: float,
    retrieved_sources: list,
    final_confidence: float,
    low_confidence_flag: bool,
    response_latency_ms: int,
    cache_hit: bool,
) -> str:
    audit_id = str(uuid.uuid4())
    record = {
        "audit_id": audit_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "raw_query": raw_query,
        "redacted_query": redacted_query,
        "routed_to": routed_to,
        "router_confidence": round(router_confidence, 4),
        "retrieved_sources": retrieved_sources,
        "final_confidence": round(final_confidence, 4),
        "low_confidence_flag": low_confidence_flag,
        "response_latency_ms": response_latency_ms,
        "cache_hit": cache_hit,
    }
    with open(_log_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return audit_id


def read_audit_logs(date_str: str = None, limit: int = 100) -> list:
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = os.path.join(AUDIT_LOG_DIR, f"audit_{date_str}.jsonl")
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records[-limit:]
