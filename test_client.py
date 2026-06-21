"""
Quick test client for the CX Bot API.
Usage: python test_client.py
"""
import httpx
import json

BASE_URL = "http://localhost:8000"

TEST_QUERIES = [
    ("session_billing_01", "Why was I charged twice on my last invoice?"),
    ("session_returns_01", "Can I return a product I bought 20 days ago?"),
    ("session_escalation_01", "I've contacted support 3 times and my issue is still not resolved. I need to speak with a manager."),
    ("session_billing_01", "What payment methods do you accept?"),  # same session, tests memory
]


def test_health():
    resp = httpx.get(f"{BASE_URL}/health")
    print(f"Health: {resp.json()}")


def test_query(session_id: str, query: str):
    resp = httpx.post(
        f"{BASE_URL}/query",
        json={"session_id": session_id, "query": query},
        timeout=120.0,
    )
    data = resp.json()
    print("\n" + "=" * 60)
    print(f"Query:      {query}")
    print(f"Agent:      {data.get('agent')}")
    print(f"Confidence: {data.get('confidence')} {'⚠ LOW' if data.get('low_confidence') else '✓'}")
    print(f"Cache Hit:  {data.get('cache_hit')}")
    print(f"Latency:    {data.get('latency_ms')}ms")
    print(f"Sources:    {data.get('sources')}")
    print(f"Answer:\n{data.get('answer')}")
    print(f"Disclaimer: {data.get('disclaimer')}")
    print(f"Audit ID:   {data.get('audit_id')}")


def test_audit():
    resp = httpx.get(f"{BASE_URL}/audit?limit=5")
    data = resp.json()
    print(f"\nAudit Log ({data['count']} records):")
    for r in data["records"]:
        print(f"  [{r['timestamp']}] {r['routed_to']} | conf={r['final_confidence']} | {r['response_latency_ms']}ms")


if __name__ == "__main__":
    print("Testing CX Bot API...")
    test_health()
    for session_id, query in TEST_QUERIES:
        test_query(session_id, query)
    test_audit()
