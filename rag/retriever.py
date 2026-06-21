"""
Hybrid retriever: combines BM25 (keyword) + FAISS (semantic) results.
"""
import logging
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from rag.indexer import load_index
from rag.bm25_retriever import bm25_retrieve
from config import TOP_K_RETRIEVAL, BM25_WEIGHT, VECTOR_WEIGHT

logger = logging.getLogger(__name__)

_faiss_cache: dict = {}


def _get_faiss(domain: str) -> FAISS:
    if domain not in _faiss_cache:
        _faiss_cache[domain] = load_index(domain)
        logger.info(f"FAISS index loaded for domain '{domain}'")
    return _faiss_cache[domain]


def _normalize(scores: List[float]) -> List[float]:
    if not scores:
        return scores
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def hybrid_retrieve(query: str, domain: str, top_k: int = None) -> List[Tuple[Document, float]]:
    if top_k is None:
        top_k = TOP_K_RETRIEVAL

    faiss_store = _get_faiss(domain)
    try:
        vector_results = faiss_store.similarity_search_with_score(query, k=top_k)
        vector_docs = [doc for doc, _ in vector_results]
        vector_raw_scores = [1.0 / (1.0 + float(score)) for _, score in vector_results]
    except Exception as e:
        logger.warning(f"FAISS retrieval failed: {e}")
        vector_docs, vector_raw_scores = [], []

    try:
        bm25_results = bm25_retrieve(query, domain, top_k=top_k)
        bm25_docs = [doc for doc, _ in bm25_results]
        bm25_raw_scores = [score for _, score in bm25_results]
    except Exception as e:
        logger.warning(f"BM25 retrieval failed: {e}")
        bm25_docs, bm25_raw_scores = [], []

    vector_norm = _normalize(vector_raw_scores)
    bm25_norm = _normalize(bm25_raw_scores)

    score_map: dict = {}
    for doc, score in zip(vector_docs, vector_norm):
        fp = doc.page_content[:100]
        entry = score_map.get(fp, (doc, 0.0))
        score_map[fp] = (entry[0], entry[1] + VECTOR_WEIGHT * score)

    for doc, score in zip(bm25_docs, bm25_norm):
        fp = doc.page_content[:100]
        entry = score_map.get(fp, (doc, 0.0))
        score_map[fp] = (entry[0], entry[1] + BM25_WEIGHT * score)

    combined = sorted(score_map.values(), key=lambda x: x[1], reverse=True)[:top_k]

    docs = [d for d, _ in combined]
    merged_scores = _normalize([s for _, s in combined])
    return list(zip(docs, merged_scores))


def invalidate_faiss_cache(domain: str = None) -> None:
    if domain:
        _faiss_cache.pop(domain, None)
    else:
        _faiss_cache.clear()