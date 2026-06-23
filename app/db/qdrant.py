import os
import logging
import threading
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

_client: QdrantClient | None = None
_lock = threading.Lock()


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = QdrantClient(
                    url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
                    api_key=os.environ.get("QDRANT_API_KEY"),  # None = no auth (local)
                    timeout=10,
                )
    return _client


def check_health() -> bool:
    try:
        get_client().get_collections()
        return True
    except Exception as e:
        logger.error("Qdrant health check failed: %s", e)
        return False
