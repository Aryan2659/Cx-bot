"""
Confidence Scorer: fuses three signals into a final confidence score.
"""
import re
import logging
from typing import List, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

W_ROUTER = 0.3
W_RETRIEVAL = 0.4
W_LLM = 0.3


def _avg_retrieval_score(chunks_with_scores: List[Tuple[Document, float]]) -> float:
    if not chunks_with_scores:
        return 0.0
    scores = [s for _, s in chunks_with_scores]
    weights = [1.0 / (i + 1) for i in range(len(scores))]
    return sum(s * w for s, w in zip(scores, weights)) / sum(weights)


def _parse_llm_rating(text: str) -> float:
    matches = re.findall(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", text)
    if matches:
        try:
            return min(1.0, max(0.0, float(matches[-1])))
        except ValueError:
            pass
    return 0.5


def compute_confidence(
    router_confidence: float,
    chunks_with_scores: List[Tuple[Document, float]],
    llm_self_rating_text: str = "",
) -> float:
    retrieval_score = _avg_retrieval_score(chunks_with_scores)
    llm_score = _parse_llm_rating(llm_self_rating_text) if llm_self_rating_text else 0.5
    fused = (
        W_ROUTER * router_confidence
        + W_RETRIEVAL * retrieval_score
        + W_LLM * llm_score
    )
    return float(round(min(1.0, max(0.0, fused)), 4))


def get_llm_self_rating_prompt(answer: str, context: str) -> str:
    return (
        f"Given the following context and answer, rate the answer's accuracy and groundedness "
        f"on a scale from 0.0 to 1.0 (1.0 = perfectly grounded, 0.0 = completely unsupported).\n\n"
        f"Context:\n{context[:800]}\n\nAnswer:\n{answer[:400]}\n\n"
        f"Respond with ONLY a single number between 0.0 and 1.0."
    )