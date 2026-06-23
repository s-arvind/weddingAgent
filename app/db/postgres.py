import os
import logging
import socket
import threading
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal = None
_lock = threading.Lock()


def _build_engine():
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL env var not set")
    return create_engine(
        url,
        poolclass=NullPool,
        connect_args={
            "connect_timeout": 5,
            "application_name": f"weddingagent@{socket.gethostname()}:{os.getpid()}",
        },
    )


def get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        with _lock:
            if _engine is None:
                _engine = _build_engine()
                _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def get_session():
    """FastAPI dependency — use with Depends(get_session).

    Handlers that write should call session.commit() explicitly; this
    dependency only manages lifecycle and rollback on error.
    """
    get_engine()
    session = _SessionLocal()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_health() -> bool:
    get_engine()
    session = _SessionLocal()
    try:
        session.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"DB health check failed: {e}")
        return False
    finally:
        session.close()
