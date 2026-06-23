import os
import json
import logging
import threading
import redis

logger = logging.getLogger(__name__)

SESSION_TTL = int(os.getenv("SESSION_TTL_SECONDS", 3600))
_client: redis.Redis | None = None
_lock = threading.Lock()


def _get_client() -> redis.Redis:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = redis.from_url(
                    os.environ.get("VALKEY_URL", "redis://localhost:6379"),
                    decode_responses=True,
                )
    return _client


def get_history(session_id: str) -> list[dict]:
    try:
        raw = _get_client().get(f"session:{session_id}")
        return json.loads(raw) if raw else []
    except Exception as e:
        logger.error("Failed to get session %s: %s", session_id, e)
        return []


def save_history(session_id: str, messages: list[dict]) -> None:
    try:
        _get_client().setex(
            f"session:{session_id}",
            SESSION_TTL,
            json.dumps(messages),
        )
    except Exception as e:
        logger.error("Failed to save session %s: %s", session_id, e)
