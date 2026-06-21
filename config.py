import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")

REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB: int = int(os.getenv("REDIS_DB", 0))

EMBED_MODEL: str = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
FAISS_INDEX_DIR: str = os.getenv("FAISS_INDEX_DIR", "faiss_indexes")
DOCS_DIR: str = os.getenv("DOCS_DIR", "documents")
AUDIT_LOG_DIR: str = os.getenv("AUDIT_LOG_DIR", "logs")

CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", 0.65))
CACHE_SIM_THRESHOLD: float = float(os.getenv("CACHE_SIM_THRESHOLD", 0.92))
CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", 3600))
SESSION_TTL_SECONDS: int = int(os.getenv("SESSION_TTL_SECONDS", 1800))
SESSION_MAX_TURNS: int = int(os.getenv("SESSION_MAX_TURNS", 10))

TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", 5))
BM25_WEIGHT: float = float(os.getenv("BM25_WEIGHT", 0.4))
VECTOR_WEIGHT: float = float(os.getenv("VECTOR_WEIGHT", 0.6))

DOMAINS = ["billing", "returns", "escalation"]
