"""
Embedding provider with automatic offline fallback.
Primary: HuggingFace BGE (requires model download on first run).
Fallback: TF-IDF vectorizer (fully offline, no downloads needed).
"""
import logging
import os
import glob
import pickle
import numpy as np
from typing import List

logger = logging.getLogger(__name__)

_hf_encoder = None
_tfidf_vectorizer = None
_USE_TFIDF = False

_TFIDF_SEED = [
    "billing invoice charge payment subscription refund duplicate overpayment",
    "return exchange product window eligibility defective warranty unopened",
    "escalation complaint manager unresolved urgent priority sla ticket",
    "policy section refund timeline credit debit bank transfer paypal",
    "customer support order account email phone address receipt",
    "late fee retry suspend cancel annual monthly pro-rata",
    "dispute investigation hold resolution business days",
    "damage misuse final sale non-returnable condition packaging tags",
    "data privacy gdpr ccpa regulatory compliance officer",
    "compensation credit expedited shipping partial refund approval",
]


def _try_load_hf(model_name: str) -> bool:
    global _hf_encoder
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        enc = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        enc.embed_query("warm up")
        _hf_encoder = enc
        logger.info(f"HuggingFace embeddings loaded: {model_name}")
        return True
    except Exception as e:
        logger.warning(f"HuggingFace embeddings unavailable ({e}). Using TF-IDF fallback.")
        return False


def _load_real_corpus() -> List[str]:
    from config import DOCS_DIR, DOMAINS
    texts = []
    for domain in DOMAINS:
        for path in (glob.glob(os.path.join(DOCS_DIR, domain, "*.txt")) +
                      glob.glob(os.path.join(DOCS_DIR, domain, "*.md"))):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    texts.append(f.read())
            except OSError:
                continue
    return texts


def _get_tfidf():
    global _tfidf_vectorizer
    if _tfidf_vectorizer is None:
        tfidf_path = os.path.join("faiss_indexes", "tfidf_vectorizer.pkl")
        if os.path.exists(tfidf_path):
            with open(tfidf_path, "rb") as f:
                _tfidf_vectorizer = pickle.load(f)
            logger.info("TF-IDF vectorizer loaded from cache.")
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            corpus = _load_real_corpus() + _TFIDF_SEED
            _tfidf_vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), max_features=512, sublinear_tf=True,
                min_df=1,
            )
            _tfidf_vectorizer.fit(corpus)
            os.makedirs("faiss_indexes", exist_ok=True)
            with open(tfidf_path, "wb") as f:
                pickle.dump(_tfidf_vectorizer, f)
            logger.info(f"TF-IDF vectorizer fitted on {len(corpus)} real+seed docs and cached.")
    return _tfidf_vectorizer


def _tfidf_encode(texts: List[str]) -> np.ndarray:
    tfidf = _get_tfidf()
    matrix = tfidf.transform(texts).toarray().astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    zero_mask = (norms == 0).flatten()
    if zero_mask.any():
        dim = matrix.shape[1]
        matrix[zero_mask] = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / norms


def initialize(model_name: str) -> None:
    global _USE_TFIDF
    if _hf_encoder is not None or _USE_TFIDF:
        return
    if not _try_load_hf(model_name):
        _USE_TFIDF = True
        _get_tfidf()


def embed_texts(texts: List[str]) -> np.ndarray:
    if _USE_TFIDF or _hf_encoder is None:
        return _tfidf_encode(texts)
    try:
        vecs = _hf_encoder.embed_documents(texts)
        return np.array(vecs, dtype=np.float32)
    except Exception as e:
        logger.warning(f"HF embed_documents failed ({e}), using TF-IDF.")
        return _tfidf_encode(texts)


def embed_query(text: str) -> np.ndarray:
    if _USE_TFIDF or _hf_encoder is None:
        return _tfidf_encode([text])[0]
    try:
        vec = np.array(_hf_encoder.embed_query(text), dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
    except Exception as e:
        logger.warning(f"HF embed_query failed ({e}), using TF-IDF.")
        return _tfidf_encode([text])[0]


def get_langchain_embeddings():
    if not _USE_TFIDF and _hf_encoder is not None:
        return _hf_encoder
    return _TFIDFLangChainWrapper()


class _TFIDFLangChainWrapper:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return embed_texts(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return embed_query(text).tolist()

    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)