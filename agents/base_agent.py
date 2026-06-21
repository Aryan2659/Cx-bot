"""
Base Agent: shared logic for all domain sub-agents.
"""
import re
import logging
from typing import List, Tuple, Dict, Any

from langchain_core.documents import Document
import ollama

from rag.retriever import hybrid_retrieve
from rag.graph_rag import graph_expand
from pipeline.confidence_scorer import compute_confidence
from memory.session_store import format_history_for_prompt
from config import OLLAMA_MODEL, OLLAMA_BASE_URL, CONFIDENCE_THRESHOLD, TOP_K_RETRIEVAL

logger = logging.getLogger(__name__)
_client = ollama.Client(host=OLLAMA_BASE_URL)

DOMAIN_DISCLAIMERS = {
    "billing": "This response is based on current billing policies. For account-specific issues, please contact your billing representative.",
    "returns": "This response is based on our standard returns policy. Actual eligibility may vary based on purchase date and item condition.",
    "escalation": "This case has been flagged for human review. A support specialist will follow up within one business day.",
}


class BaseAgent:
    domain: str = "base"
    system_prompt: str = "You are a helpful customer support assistant."

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Tuple[Document, float]]:
        return hybrid_retrieve(query, self.domain, top_k=top_k)

    def expand_graph(self, chunks: List[Document]) -> List[Document]:
        return graph_expand(chunks, self.domain, hops=1)

    def _build_context(self, chunks: List[Document]) -> str:
        parts = []
        for i, doc in enumerate(chunks[:5], 1):
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[{i}] (Source: {source})\n{doc.page_content}")
        return "\n\n".join(parts)

    def _build_prompt(self, query: str, context: str, history: str) -> str:
        history_block = f"\nConversation History:\n{history}\n" if history else ""
        return (
            f"{self.system_prompt}\n"
            f"{history_block}\n"
            f"Use ONLY the following documents to answer. Do not hallucinate.\n"
            f"If the answer is not in the documents, say 'I don't have enough information.'\n\n"
            f"Documents:\n{context}\n\n"
            f"Customer Query: {query}\n\n"
            f"Respond in EXACTLY this two-line format, nothing else:\n"
            f"ANSWER: <your answer to the customer>\n"
            f"CONFIDENCE: <single number 0.0-1.0 rating how well the documents support your answer>"
        )

    def _call_llm(self, prompt: str) -> str:
        try:
            response = _client.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2, "num_predict": 400},
                keep_alive="30m",
            )
            return response["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            return "I'm unable to process your request at the moment. Please try again."

    def _split_answer_and_rating(self, raw: str) -> Tuple[str, str]:
        ans_match = re.search(r"ANSWER:\s*(.*?)(?=\n\s*CONFIDENCE:|\Z)", raw, re.S | re.I)
        conf_match = re.search(r"CONFIDENCE:\s*(.*)", raw, re.S | re.I)
        answer = ans_match.group(1).strip() if ans_match else raw.strip()
        rating_text = conf_match.group(1).strip() if conf_match else ""
        return answer, rating_text

    def _extract_sources(self, chunks: List[Document]) -> List[str]:
        seen = set()
        sources = []
        for doc in chunks:
            src = doc.metadata.get("source", "unknown")
            if src not in seen:
                sources.append(src)
                seen.add(src)
        return sources

    def run(
        self,
        query: str,
        router_confidence: float,
        session_history: list,
    ) -> Dict[str, Any]:
        chunks_with_scores = self.retrieve(query)
        chunks = [doc for doc, _ in chunks_with_scores]

        expanded_chunks = self.expand_graph(chunks)

        context = self._build_context(expanded_chunks)
        history_str = format_history_for_prompt(session_history)

        prompt = self._build_prompt(query, context, history_str)
        raw = self._call_llm(prompt)
        answer, self_rating_text = self._split_answer_and_rating(raw)

        confidence = compute_confidence(router_confidence, chunks_with_scores, self_rating_text)
        low_confidence = bool(confidence < CONFIDENCE_THRESHOLD)

        sources = self._extract_sources(expanded_chunks)

        return {
            "answer": answer,
            "agent": self.domain,
            "sources": sources,
            "confidence": confidence,
            "low_confidence": low_confidence,
            "disclaimer": DOMAIN_DISCLAIMERS.get(self.domain, ""),
        }