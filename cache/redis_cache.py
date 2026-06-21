"""
Redis Semantic Cache: caches LLM responses keyed by query embedding.
Uses cosine similarity for lookup. Fully offline-compatible via embeddings module.
"""
import json
import logging
import hashlib
import numpy as np
from typing import Optional

import redis

from config import (
    REDIS_HOST, REDIS_PORT, REDIS_DB,
    CACHE_SIM_THRESHOLD, CACHE_TTL_SECONDS, EMBED_MODEL,
)

logger = logging.getLogger(__name__)

_client: redis.Redis = None
_embeddings_ready = False

CACHE_KEY_PREFIX = "semcache"


def _get_client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
            decode_responses=False,   # bytes mode — consistent for vec storage
            socket_connect_timeout=5,
        )
    return _client


def _ensure_embeddings():
    global _embeddings_ready
    if not _embeddings_ready:
        from rag.embeddings import initialize
        initialize(EMBED_MODEL)
        _embeddings_ready = True


def _embed(text: str) -> np.ndarray:
    _ensure_embeddings()
    from rag.embeddings import embed_query
    return embed_query(text)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for L2-normalized vectors = dot product."""
    return float(np.dot(a, b))


def _make_keys(domain: str, key_id: str):
    meta_key = f"{CACHE_KEY_PREFIX}:{domain}:{key_id}:meta".encode()
    vec_key  = f"{CACHE_KEY_PREFIX}:{domain}:{key_id}:vec".encode()
    return meta_key, vec_key


def lookup(query: str, domain: str) -> Optional[dict]:
    """Return cached response dict if a similar query exists, else None."""
    try:
        client = _get_client()
        query_vec = _embed(query)

        # Scan all meta keys for this domain (bytes pattern)
        pattern = f"{CACHE_KEY_PREFIX}:{domain}:*:meta".encode()
        keys = list(client.scan_iter(pattern))

        best_score = 0.0
        best_response = None

        for meta_key in keys:
            raw_meta = client.get(meta_key)
            if not raw_meta:
                continue
            # Derive vec_key from meta_key (both are bytes)
            vec_key = meta_key.replace(b":meta", b":vec")
            raw_vec = client.get(vec_key)
            if not raw_vec:
                continue
            cached_vec = np.frombuffer(raw_vec, dtype=np.float32)
            score = _cosine(query_vec, cached_vec)
            if score > best_score:
                best_score = score
                best_response = json.loads(raw_meta.decode("utf-8"))

        if best_score >= CACHE_SIM_THRESHOLD and best_response:
            logger.info(f"Cache HIT (score={best_score:.3f}) for domain '{domain}'")
            return best_response

    except redis.exceptions.ConnectionError:
        logger.debug("Redis unavailable — cache lookup skipped.")
    except Exception as e:
        logger.warning(f"Cache lookup failed: {e}")
    return None


def store(query: str, domain: str, response: dict) -> None:
    """Store a query-response pair in the semantic cache."""
    try:
        client = _get_client()
        query_vec = _embed(query)
        key_id = hashlib.md5(f"{domain}:{query}".encode()).hexdigest()
        meta_key, vec_key = _make_keys(domain, key_id)

        client.set(meta_key, json.dumps(response).encode("utf-8"), ex=CACHE_TTL_SECONDS)
        client.set(vec_key, query_vec.tobytes(), ex=CACHE_TTL_SECONDS)
        logger.debug(f"Cache STORE for domain '{domain}', key={key_id}")
    except redis.exceptions.ConnectionError:
        logger.debug("Redis unavailable — cache store skipped.")
    except Exception as e:
        logger.warning(f"Cache store failed: {e}")
