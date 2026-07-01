# Domain-Aware Multi-Agent CX Bot

A local, fully self-hosted customer experience platform that routes incoming support
queries to domain-specific RAG agents (billing, returns, escalation), grounds every
answer in retrieved policy documents, scores its own confidence, and logs every
interaction for auditing and continuous evaluation.

Everything runs on your own machine — local LLM via Ollama, local vector index via
FAISS, local cache/session store via Redis. No external API keys required.

## Why this exists

Most "RAG chatbot" demos stop at retrieve → generate. This project adds the pieces
that make a support bot actually trustworthy and operable in production:

- **Routing instead of one giant prompt** — a lightweight classifier sends each query
  to a specialized agent with its own system prompt and document set.
- **Hybrid retrieval** — BM25 keyword search and FAISS semantic search are fused, so
  exact policy terms (e.g. "Section 5") aren't lost to pure embedding similarity.
- **Graph-based multi-hop expansion** — chunks that cross-reference each other
  ("see Section 3", "as per the returns policy") are linked in a graph so the agent
  can follow that reference automatically instead of missing it.
- **Confidence isn't a single number from the LLM** — it's a weighted fusion of router
  confidence, retrieval relevance, and an LLM self-rating, so low-confidence answers
  can be reliably flagged for escalation.
- **Every query is audited** — full request/response metadata is logged to disk as
  JSONL, which a nightly evaluator and prompt optimizer both read from.
- **PII is redacted before it touches the LLM or the logs**, with a regex fallback if
  Presidio/spaCy aren't installed.

## Stack

LangChain · FAISS · HuggingFace BGE embeddings · BM25 · NetworkX (graph RAG) ·
Ollama (Llama 3.2) · FastAPI · Redis · Presidio (PII redaction)

## Architecture

```
User Query
    │
    ▼
[PII Redactor] (Presidio, regex fallback)
    │
    ▼
[Redis Semantic Cache] ──HIT──→ Cached Response
    │ MISS
    ▼
[Session Memory] (Redis, last N turns)
    │
    ▼
[Router Agent] (Llama 3.2 zero-shot classification)
    ├──→ BillingAgent
    ├──→ ReturnsAgent
    └──→ EscalationAgent
              │
              ▼
    [Hybrid Retriever] BM25 + FAISS/BGE, score-fused
              │
              ▼
    [Graph RAG] NetworkX multi-hop expansion (cross-references)
              │
              ▼
    [Llama 3.2 via Ollama] grounded generation
              │
              ▼
    [Confidence Scorer] router + retrieval + LLM self-rating
              │
              ▼
    [Audit Logger] → logs/audit_<date>.jsonl
              │
              ▼
    Structured Response
```

## Project structure

```
cx_bot/
├── api/main.py              # FastAPI app — all HTTP endpoints
├── agents/
│   ├── base_agent.py        # Shared retrieve → expand → generate → score pipeline
│   ├── billing_agent.py     # Billing domain system prompt
│   ├── returns_agent.py     # Returns domain system prompt
│   ├── escalation_agent.py  # Escalation domain system prompt
│   └── router.py            # LLM-based query classifier
├── rag/
│   ├── indexer.py           # Loads docs, chunks, embeds, builds/saves FAISS indexes
│   ├── retriever.py         # Hybrid BM25 + FAISS retrieval with score fusion
│   ├── bm25_retriever.py    # BM25 keyword search per domain
│   ├── graph_rag.py         # Builds cross-reference graph, multi-hop expansion
│   └── embeddings.py        # Embedding model loader (HuggingFace BGE)
├── pipeline/
│   ├── pii_redactor.py      # Presidio-based PII redaction, regex fallback
│   └── confidence_scorer.py # Fuses router/retrieval/LLM signals into one score
├── cache/redis_cache.py     # Semantic response cache (cosine similarity over embeddings)
├── memory/session_store.py  # Per-session conversation history (Redis, TTL-bound)
├── audit/logger.py          # Append-only JSONL audit logging + read-back
├── eval/
│   ├── auto_eval.py         # Nightly benchmark: samples audit logs, computes metrics
│   └── prompt_optimizer.py  # Tests prompt variants against low-confidence queries
├── documents/                # Source-of-truth policy docs, by domain
│   ├── billing/
│   ├── returns/
│   └── escalation/
├── faiss_indexes/            # Generated — FAISS indexes + chunk pickles + graph cache
├── logs/                     # Generated — daily audit logs, benchmark/optimizer reports
├── config.py                 # All tunables, loaded from .env
├── docker-compose.yml        # Redis service
├── run.sh                    # One-shot startup script
└── test_client.py            # Manual smoke-test script
```

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and on PATH
- Docker (for Redis) — or a Redis instance you point `config.py` at
- ~4GB free disk for the Llama 3.2 model + BGE embedding model on first run

## Quick Start

### 1. Set up the environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # adjust values if needed
```

### 2. Pull and serve the Ollama model
```bash
ollama pull llama3.2
ollama serve
```

### 3. Start Redis
```bash
docker-compose up -d redis
```

### 4. Build document indexes
```bash
python rag/indexer.py --domain all
```

### 5. Start the API
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Or, on Linux/macOS, do steps 2–5 in one go:
```bash
chmod +x run.sh && ./run.sh
```

Once running, interactive API docs are available at `http://localhost:8000/docs`.

## Configuration

All settings live in `.env` (see `.env.example` for the full template) and are loaded
through `config.py`:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_MODEL` | `llama3.2` | Model used for routing and generation |
| `REDIS_HOST` / `REDIS_PORT` / `REDIS_DB` | `localhost` / `6379` / `0` | Redis connection for cache + session memory |
| `EMBED_MODEL` | `BAAI/bge-base-en-v1.5` | HuggingFace embedding model for FAISS |
| `FAISS_INDEX_DIR` | `faiss_indexes` | Where indexes/chunks/graphs are persisted |
| `DOCS_DIR` | `documents` | Root folder for domain policy documents |
| `AUDIT_LOG_DIR` | `logs` | Where audit logs and reports are written |
| `CONFIDENCE_THRESHOLD` | `0.65` | Below this, a response is flagged `low_confidence` |
| `CACHE_SIM_THRESHOLD` | `0.92` | Minimum cosine similarity for a cache hit |
| `CACHE_TTL_SECONDS` | `3600` | Cache entry lifetime |
| `SESSION_TTL_SECONDS` | `1800` | Session memory lifetime |
| `SESSION_MAX_TURNS` | `10` | Conversation turns retained per session |
| `TOP_K_RETRIEVAL` | `5` | Chunks retrieved per query |
| `BM25_WEIGHT` / `VECTOR_WEIGHT` | `0.4` / `0.6` | Fusion weights for hybrid retrieval |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/query` | Submit a customer query, get a routed, grounded, scored response |
| GET | `/audit` | View audit log records (`?date=YYYY-MM-DD&limit=100`) |
| POST | `/index` | Re-index all domain documents (runs in the background) |
| DELETE | `/session/{id}` | Clear a session's conversation memory |
| GET | `/health` | Health check; lists active agents |

### Example request
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"session_id": "user_001", "query": "Why was I charged twice?"}'
```

### Example response
```json
{
  "answer": "Based on our billing policy (Section 5)...",
  "agent": "billing",
  "sources": ["billing_policy.txt"],
  "confidence": 0.87,
  "low_confidence": false,
  "disclaimer": "This response is based on current billing policies...",
  "audit_id": "uuid-here",
  "latency_ms": 1240,
  "cache_hit": false
}
```

## Adding domain documents

Drop `.txt` or `.md` files into:
- `documents/billing/` — invoices, billing policies
- `documents/returns/` — return/exchange policies
- `documents/escalation/` — SLA documents, escalation procedures

Then re-index just that domain (or all of them):
```bash
python rag/indexer.py --domain billing
# or rebuild everything via the API (clears in-memory FAISS/BM25 caches too):
curl -X POST http://localhost:8000/index
```

To enable cross-document reference following, phrase relationships explicitly in
your docs (e.g. *"see Section 3"*, *"as per the returns policy"*) — `graph_rag.py`
detects these patterns to link chunks for multi-hop retrieval.

## Evaluation & continuous improvement

**Benchmark** — samples recent audit logs (across the last 7 days by default) and
checks them against target thresholds:
```bash
python eval/auto_eval.py --mode benchmark --samples 100
```

| Metric | Target |
|---|---|
| Cache hit rate | ≥ 40% |
| Low-confidence rate | ≤ 15% |
| Avg latency | < 1500ms |
| p95 latency | < 3500ms |
| Avg confidence | ≥ 0.80 |

**Prompt optimizer** — pulls queries that were flagged `low_confidence`, tests three
system-prompt variants (precise / empathetic / structured) against them, and writes
the best-performing variant to `logs/prompt_optimization_latest.json`:
```bash
python eval/prompt_optimizer.py
```

### Nightly cron (optional)
```cron
0 2 * * * cd /path/to/cx_bot && python eval/auto_eval.py >> logs/eval.log 2>&1
0 3 * * * cd /path/to/cx_bot && python eval/prompt_optimizer.py >> logs/optimizer.log 2>&1
```
## Deployment Notes

A live demo of this project is deployed on Hugging Face Spaces:

**Live demo:** https://huggingface.co/spaces/rnyx/cx-bot

### Current setup

The demo runs on Hugging Face's free CPU-basic tier as a Docker Space. Free-tier
Spaces sleep after 48 hours of no traffic, so a scheduled GitHub Actions workflow
(`.github/workflows/keep-alive.yml`) pings the Space's `/health` endpoint every
30 minutes to keep it warm and avoid cold starts.

This is a pragmatic choice for a portfolio/demo deployment, not a production
architecture. It has known limitations:

- **No persistent storage** — the Ollama model (`llama3.2`) is re-pulled on every
  container restart (redeploys, HF infra restarts), so a genuine restart still
  causes a slow first request.
- **No real uptime guarantee** — the keep-alive ping prevents the *inactivity*
  sleep, but doesn't recover the Space from a crash or a manual pause.
- **Single instance, no horizontal scaling.**

### What a production deployment would change

- Run on upgraded/paid Space hardware (or a dedicated host), which stays on by
  default and doesn't depend on an external keep-alive signal.
- Attach persistent storage so the model and FAISS indexes survive restarts.
- Add real health-check-based alerting instead of a pass/fail GitHub Action log.
- Move Redis to a managed instance rather than a container-local service.
- Add horizontal scaling / replicas if concurrent load is expected.
## Troubleshooting

- **Audit log line counts don't match a benchmark's `total_queries`** — the benchmark
  sums entries across the last 7 daily `audit_<date>.jsonl` files, not just today's.
  Check all files with:
  ```powershell
  Get-ChildItem logs\*.jsonl | ForEach-Object { "$($_.Name): $((Get-Content $_.FullName).Count) lines" }
  ```
- **Presidio/spaCy not installed** — PII redaction silently falls back to regex
  patterns (email, phone, credit card, SSN). Check logs for a "Presidio not
  available" warning.
- **Redis unreachable** — caching and session memory fail open (queries still work,
  just without caching/memory); check for "Redis unavailable" debug logs.
- **First query after a fresh clone is slow** — indexes are built lazily on first
  access if missing. Run `python rag/indexer.py --domain all` ahead of time to avoid
  this on a live request.

## License

MIT — see [LICENSE](LICENSE).
