"""
BM25 keyword retriever per domain.
"""
import logging
from typing import List, Tuple

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from rag.indexer import load_chunks

logger = logging.getLogger(__name__)

_bm25_cache: dict = {}


def _get_bm25(domain: str) -> Tuple[BM25Okapi, List[Document]]:
    if domain not in _bm25_cache:
        chunks = load_chunks(domain)
        tokenized = [doc.page_content.lower().split() for doc in chunks]
        bm25 = BM25Okapi(tokenized)
        _bm25_cache[domain] = (bm25, chunks)
        logger.info(f"BM25 index built for domain '{domain}' with {len(chunks)} chunks.")
    return _bm25_cache[domain]


def bm25_retrieve(query: str, domain: str, top_k: int = 5) -> List[Tuple[Document, float]]:
    """Return list of (Document, score) sorted by BM25 score descending."""
    bm25, chunks = _get_bm25(domain)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for idx, score in scored:
        if score > 0:
            results.append((chunks[idx], float(score)))
    return results


def invalidate_cache(domain: str = None) -> None:
    if domain:
        _bm25_cache.pop(domain, None)
    else:
        _bm25_cache.clear()
