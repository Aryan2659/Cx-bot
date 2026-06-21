"""
Nightly Auto-Evaluator: samples audit logs, re-runs queries, computes metrics.

Fixes:
- Removed unused imports: time, CONFIDENCE_THRESHOLD
"""
import json
import random
import logging
import argparse
import os
from datetime import datetime, timezone, timedelta
from typing import List, Dict

from audit.logger import read_audit_logs
from config import AUDIT_LOG_DIR

logger = logging.getLogger(__name__)


def _load_samples(n: int, days_back: int = 7) -> List[Dict]:
    records = []
    for i in range(days_back):
        date = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d")
        records.extend(read_audit_logs(date_str=date, limit=500))
    random.shuffle(records)
    return records[:n]


def compute_metrics(records: List[Dict]) -> Dict:
    if not records:
        return {}

    total = len(records)
    cache_hits = sum(1 for r in records if r.get("cache_hit"))
    low_conf = sum(1 for r in records if r.get("low_confidence_flag"))
    latencies = [r.get("response_latency_ms", 0) for r in records]
    confidences = [r.get("final_confidence", 0) for r in records]

    sorted_lat = sorted(latencies)
    p95_idx = max(0, int(0.95 * len(sorted_lat)) - 1)

    routing_counts: Dict[str, int] = {}
    for r in records:
        agent = r.get("routed_to", "unknown")
        routing_counts[agent] = routing_counts.get(agent, 0) + 1

    return {
        "total_queries": total,
        "cache_hit_rate": round(cache_hits / total, 4),
        "low_confidence_rate": round(low_conf / total, 4),
        "avg_latency_ms": round(sum(latencies) / total, 1) if latencies else 0,
        "p95_latency_ms": sorted_lat[p95_idx] if sorted_lat else 0,
        "avg_confidence": round(sum(confidences) / total, 4) if confidences else 0,
        "routing_distribution": routing_counts,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }


def run_benchmark(n: int = 50) -> None:
    logger.info(f"Running benchmark on {n} samples...")
    records = _load_samples(n)
    if not records:
        logger.warning("No audit records found. Run some queries first.")
        return

    metrics = compute_metrics(records)

    targets = {
        "cache_hit_rate":       (0.40, "≥ 40%",   False),
        "low_confidence_rate":  (0.15, "≤ 15%",   True),
        "avg_latency_ms":       (1500, "< 1500ms", True),
        "p95_latency_ms":       (3500, "< 3500ms", True),
        "avg_confidence":       (0.80, "≥ 0.80",   False),
    }

    print("\n" + "=" * 60)
    print("CX BOT BENCHMARK RESULTS")
    print("=" * 60)
    for key, value in metrics.items():
        if key in targets:
            target_val, label, lower_is_better = targets[key]
            status = "✓ PASS" if (value <= target_val if lower_is_better else value >= target_val) else "✗ FAIL"
            print(f"  {key:<28} {str(value):<12} Target: {label:<12} {status}")
        else:
            print(f"  {key:<28} {value}")
    print("=" * 60 + "\n")

    os.makedirs(AUDIT_LOG_DIR, exist_ok=True)
    report_path = os.path.join(
        AUDIT_LOG_DIR,
        f"benchmark_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Benchmark report saved to {report_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["benchmark"], default="benchmark")
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()
    run_benchmark(args.samples)
