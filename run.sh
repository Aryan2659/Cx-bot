#!/bin/bash
set -e

echo "=== CX Bot Startup ==="

# 1. Start Redis
echo "[1/4] Starting Redis..."
docker-compose up -d redis
sleep 2

# 2. Download spaCy model for Presidio (if available)
echo "[2/4] Setting up NLP models..."
python -m spacy download en_core_web_lg 2>/dev/null || echo "spaCy model not downloaded (Presidio will use regex fallback)"

# 3. Build document indexes
echo "[3/4] Building domain indexes..."
python rag/indexer.py --domain all

# 4. Start API server
echo "[4/4] Starting FastAPI server..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

echo "=== CX Bot running at http://localhost:8000 ==="
