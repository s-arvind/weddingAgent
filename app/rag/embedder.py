import os
import threading
import logging
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_model = None
_lock = threading.Lock()


def _resolve_device() -> str:
    """Pick the best available device. BGE-M3 is large; CPU is the slow path."""
    requested = os.getenv("EMBEDDING_DEVICE", "").strip().lower()
    if requested:
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def get_model() -> SentenceTransformer:
    """Lazily load the embedding model exactly once, thread-safely."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
                device = _resolve_device()
                logger.info("Loading embedding model %s on %s", model_name, device)
                _model = SentenceTransformer(model_name, device=device)
    return _model


def warmup() -> None:
    """Force model load + download before the ingestion loop. Fails fast."""
    get_model()


def embed(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = get_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
    )
    return vectors.tolist()
