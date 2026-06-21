"""
Auto Prompt Optimizer: generates and tests prompt variations for low-confidence queries.

Fixes:
- Removed unused imports: List, CONFIDENCE_THRESHOLD
"""
import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict

import ollama
from config import OLLAMA_MODEL, OLLAMA_BASE_URL, AUDIT_LOG_DIR
from audit.logger import read_audit_logs

logger = logging.getLogger(__name__)

PROMPT_VARIANTS = [
    (
        "precise",
        "You are a precise customer support agent. Answer using ONLY the provided documents. "
        "Quote the exact policy section when applicable. Be concise and factual."
    ),
    (
        "empathetic",
        "You are a warm and empathetic customer support agent. Acknowledge the customer's concern, "
        "then answer using ONLY the provided documents. Use clear, simple language."
    ),
    (
        "structured",
        "You are a structured customer support agent. Answer in this format:\n"
        "1. Direct Answer\n2. Policy Reference\n3. Next Steps\n"
        "Use ONLY the provided documents."
    ),
]


def _test_prompt(system_prompt: str, query: str, context: str) -> float:
    prompt = (
        f"{system_prompt}\n\n"
        f"Documents:\n{context[:1000]}\n\n"
        f"Query: {query}\n\nAnswer:"
    )
    try:
        client = ollama.Client(host=OLLAMA_BASE_URL)
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2},
        )
        # FIX: attribute access, not dict syntax
        answer = response.message.content.strip()
        if "i don't have enough information" in answer.lower():
            return 0.3
        if len(answer) < 50:
            return 0.4
        return min(1.0, len(answer) / 500)
    except Exception as e:
        logger.warning(f"Prompt test failed: {e}")
        return 0.0


def optimize_prompts(n_samples: int = 20) -> None:
    records = read_audit_logs(limit=200)
    low_conf_records = [r for r in records if r.get("low_confidence_flag")][:n_samples]

    if not low_conf_records:
        logger.info("No low-confidence records to optimize against.")
        return

    scores: Dict[str, float] = {name: 0.0 for name, _ in PROMPT_VARIANTS}

    for record in low_conf_records:
        query = record.get("redacted_query", "")
        context = "Sample context for optimization."
        for name, prompt in PROMPT_VARIANTS:
            scores[name] += _test_prompt(prompt, query, context)

    for name in scores:
        scores[name] = round(scores[name] / len(low_conf_records), 4)

    best_name = max(scores, key=lambda k: scores[k])
    best_prompt = dict(PROMPT_VARIANTS)[best_name]

    logger.info(f"Best prompt variant: '{best_name}' with avg score {scores[best_name]}")

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scores": scores,
        "best_variant": best_name,
        "best_prompt": best_prompt,
        "samples_tested": len(low_conf_records),
    }
    os.makedirs(AUDIT_LOG_DIR, exist_ok=True)
    out_path = os.path.join(AUDIT_LOG_DIR, "prompt_optimization_latest.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Optimization result saved to {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    optimize_prompts()
