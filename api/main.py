"""
FastAPI application: main entry point for the CX Bot.

Fixes:
- Removed unused HTTPException import
- Removed redundant local cache_hit = False variable (now set per-branch cleanly)
- POST /index now also invalidates in-memory FAISS + BM25 caches
"""
import os
import time
import uuid
import logging
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents.billing_agent import BillingAgent
from agents.returns_agent import ReturnsAgent
from agents.escalation_agent import EscalationAgent
from agents.router import route
from pipeline.pii_redactor import redact
from memory.session_store import get_history, append_turn, clear_session
from cache.redis_cache import lookup as cache_lookup, store as cache_store
from audit.logger import write_audit, read_audit_logs
from rag.indexer import build_all_indexes, build_index
from rag.retriever import invalidate_faiss_cache
from rag.bm25_retriever import invalidate_cache as invalidate_bm25
from config import DOCS_DIR, DOMAINS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Domain-Aware Multi-Agent CX Bot",
    description="Production-grade customer experience platform with RAG-powered sub-agents.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_AGENTS = {
    "billing": BillingAgent(),
    "returns": ReturnsAgent(),
    "escalation": EscalationAgent(),
}


# ─── Schemas ────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    query: str = Field(..., min_length=1, max_length=2000)


class QueryResponse(BaseModel):
    answer: str
    agent: str
    sources: list
    confidence: float
    low_confidence: bool
    disclaimer: str
    audit_id: str
    latency_ms: int
    cache_hit: bool


class AuditResponse(BaseModel):
    records: list
    count: int


class HealthResponse(BaseModel):
    status: str
    agents: list


class UploadResponse(BaseModel):
    status: str
    domain: str
    filename: str
    message: str


# ─── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", agents=list(_AGENTS.keys()))


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    start = time.time()

    # 1. Redact PII
    redacted_query = redact(req.query)

    # 2. Route
    category, router_confidence = route(redacted_query)

    # 3. Check semantic cache
    cached = cache_lookup(redacted_query, category)
    if cached:
        elapsed_ms = int((time.time() - start) * 1000)
        audit_id = write_audit(
            session_id=req.session_id,
            raw_query=req.query,
            redacted_query=redacted_query,
            routed_to=category,
            router_confidence=router_confidence,
            retrieved_sources=cached.get("sources", []),
            final_confidence=cached.get("confidence", 1.0),
            low_confidence_flag=cached.get("low_confidence", False),
            response_latency_ms=elapsed_ms,
            cache_hit=True,
        )
        return QueryResponse(
            answer=cached.get("answer", ""),
            agent=cached.get("agent", category),
            sources=cached.get("sources", []),
            confidence=cached.get("confidence", 1.0),
            low_confidence=cached.get("low_confidence", False),
            disclaimer=cached.get("disclaimer", ""),
            audit_id=audit_id,
            latency_ms=elapsed_ms,
            cache_hit=True,
        )

    # 4. Load session memory
    session_history = get_history(req.session_id)

    # 5. Invoke domain agent
    agent = _AGENTS.get(category, _AGENTS["escalation"])
    result = agent.run(redacted_query, router_confidence, session_history)

    # 6. Update session memory
    append_turn(req.session_id, "user", redacted_query)
    append_turn(req.session_id, "assistant", result["answer"])

    # 7. Cache result
    cache_store(redacted_query, category, result)

    # 8. Audit log
    elapsed_ms = int((time.time() - start) * 1000)
    audit_id = write_audit(
        session_id=req.session_id,
        raw_query=req.query,
        redacted_query=redacted_query,
        routed_to=category,
        router_confidence=router_confidence,
        retrieved_sources=result["sources"],
        final_confidence=result["confidence"],
        low_confidence_flag=result["low_confidence"],
        response_latency_ms=elapsed_ms,
        cache_hit=False,
    )

    return QueryResponse(
        **result,
        audit_id=audit_id,
        latency_ms=elapsed_ms,
        cache_hit=False,
    )


@app.get("/audit", response_model=AuditResponse)
def audit_endpoint(date: Optional[str] = None, limit: int = 100):
    records = read_audit_logs(date_str=date, limit=limit)
    return AuditResponse(records=records, count=len(records))


@app.delete("/session/{session_id}")
def clear_session_endpoint(session_id: str):
    clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.post("/index")
def index_endpoint(background_tasks: BackgroundTasks):
    """
    Trigger async re-indexing of all domain documents.
    Also clears in-memory FAISS and BM25 caches so agents
    immediately serve fresh indexes after rebuild.
    """
    def _reindex():
        build_all_indexes()
        invalidate_faiss_cache()
        invalidate_bm25()

    background_tasks.add_task(_reindex)
    return {"status": "indexing started", "message": "Re-indexing running in background."}


ALLOWED_DOC_EXTENSIONS = {".txt", ".md"}
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    domain: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Upload a policy document into a domain's document folder and
    trigger a background re-index of that domain only.
    """
    if domain not in DOMAINS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid domain '{domain}'. Must be one of: {DOMAINS}",
        )

    safe_name = os.path.basename(file.filename or "")
    ext = os.path.splitext(safe_name)[1].lower()
    if not safe_name or ext not in ALLOWED_DOC_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_DOC_EXTENSIONS)}",
        )

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File too large (max 5MB).")
    if not contents.strip():
        raise HTTPException(status_code=400, detail="File is empty.")

    domain_dir = os.path.join(DOCS_DIR, domain)
    os.makedirs(domain_dir, exist_ok=True)

    dest_path = os.path.join(domain_dir, safe_name)
    if os.path.exists(dest_path):
        stem, ext2 = os.path.splitext(safe_name)
        dest_path = os.path.join(domain_dir, f"{stem}_{uuid.uuid4().hex[:8]}{ext2}")

    with open(dest_path, "wb") as f:
        f.write(contents)

    final_name = os.path.basename(dest_path)
    logger.info(f"Uploaded '{final_name}' to domain '{domain}'. Triggering re-index.")

    def _reindex_domain():
        build_index(domain)
        invalidate_faiss_cache(domain)
        invalidate_bm25(domain)

    background_tasks.add_task(_reindex_domain)

    return UploadResponse(
        status="uploaded",
        domain=domain,
        filename=final_name,
        message="File saved. Re-indexing started in background.",
    )