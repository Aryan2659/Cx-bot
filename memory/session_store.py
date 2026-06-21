"""
Per-session conversation memory backed by Redis.
"""
import json
import logging
from typing import List, Dict

import redis

from config import REDIS_HOST, REDIS_PORT, REDIS_DB, SESSION_TTL_SECONDS, SESSION_MAX_TURNS

logger = logging.getLogger(__name__)

_client: redis.Redis = None


def _get_client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=5,
        )
    return _client


def _key(session_id: str) -> str:
    return f"session:{session_id}:history"


def get_history(session_id: str) -> List[Dict]:
    """Return conversation history for a session."""
    try:
        client = _get_client()
        raw = client.get(_key(session_id))
        if raw:
            return json.loads(raw)
    except Exception as e:
        logger.warning(f"Session get failed for {session_id}: {e}")
    return []


def append_turn(session_id: str, role: str, content: str) -> None:
    """Append a turn to session history and refresh TTL."""
    try:
        client = _get_client()
        history = get_history(session_id)
        history.append({"role": role, "content": content})
        # Keep only last N turns
        if len(history) > SESSION_MAX_TURNS * 2:
            history = history[-(SESSION_MAX_TURNS * 2):]
        client.set(_key(session_id), json.dumps(history), ex=SESSION_TTL_SECONDS)
    except Exception as e:
        logger.warning(f"Session append failed for {session_id}: {e}")


def clear_session(session_id: str) -> None:
    try:
        _get_client().delete(_key(session_id))
    except Exception as e:
        logger.warning(f"Session clear failed for {session_id}: {e}")


def format_history_for_prompt(history: List[Dict]) -> str:
    """Convert history list to a formatted string for prompt injection."""
    if not history:
        return ""
    lines = []
    for turn in history[-SESSION_MAX_TURNS * 2:]:
        role = turn.get("role", "user").capitalize()
        lines.append(f"{role}: {turn.get('content', '')}")
    return "\n".join(lines)
