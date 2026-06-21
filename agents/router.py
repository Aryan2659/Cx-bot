"""
Router Agent: classifies incoming query into billing / returns / escalation.
"""
import json
import re
import logging
from typing import Tuple

import ollama
from config import OLLAMA_MODEL, OLLAMA_BASE_URL, CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)
_client = ollama.Client(host=OLLAMA_BASE_URL)

CATEGORIES = ["billing", "returns", "escalation"]

ROUTER_PROMPT = """You are a customer support query classifier.
Classify the customer query into EXACTLY one of these categories: billing, returns, escalation.

Category definitions:
- billing: questions about invoices, charges, payments, subscriptions, refunds, pricing
- returns: questions about returning products, exchanges, return eligibility, return status
- escalation: complaints, unresolved issues, requests to speak with a manager, urgent problems

Respond with ONLY valid JSON in this exact format (no other text):
{{"category": "<billing|returns|escalation>", "confidence": <float 0.0-1.0>}}

Customer Query: {query}"""


def _parse_router_response(text: str) -> Tuple[str, float]:
    """Parse category and confidence from LLM response."""
    try:
        data = json.loads(text.strip())
        category = data.get("category", "").lower().strip()
        confidence = float(data.get("confidence", 0.5))
        if category in CATEGORIES:
            return category, confidence
    except (json.JSONDecodeError, ValueError):
        pass

    cat_match = re.search(r'"category"\s*:\s*"(\w+)"', text, re.I)
    conf_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)

    category = "escalation"
    confidence = 0.5

    if cat_match:
        candidate = cat_match.group(1).lower()
        if candidate in CATEGORIES:
            category = candidate

    if conf_match:
        try:
            confidence = float(conf_match.group(1))
        except ValueError:
            pass

    return category, confidence


def route(query: str) -> Tuple[str, float]:
    """
    Classify query and return (category, confidence).
    Falls back to 'escalation' on any failure.
    """
    prompt = ROUTER_PROMPT.format(query=query)
    try:
        response = _client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_predict": 60},
            keep_alive="30m",
        )
        raw = response["message"]["content"].strip()
        category, confidence = _parse_router_response(raw)
    except Exception as e:
        logger.error(f"Router LLM call failed: {e}")
        return "escalation", 0.5

    if confidence < CONFIDENCE_THRESHOLD:
        logger.info(f"Low router confidence ({confidence:.2f}), routing to escalation.")
        return "escalation", confidence

    logger.info(f"Routed to '{category}' with confidence {confidence:.2f}")
    return category, confidence