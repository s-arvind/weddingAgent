import os
import threading
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_model = None
_lock = threading.Lock()


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                import torch
                torch.set_num_threads(os.cpu_count() or 4)
                model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                if device == "cpu":
                    torch.set_num_threads(os.cpu_count() or 4)
                logger.info("Loading embedding model %s on %s", model_name, device)
                _model = SentenceTransformer(model_name, device=device)
    return _model


def warmup() -> None:
    get_model()


def embed(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vectors = get_model().encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "64")),
    )
    return vectors.tolist()
