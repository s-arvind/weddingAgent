import os
import threading
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_model = None
_lock = threading.Lock()
_batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "512"))


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                import torch
                model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                if device == "cpu":
                    torch.set_num_threads(os.cpu_count() or 4)
                logger.info("Loading embedding model %s on %s", model_name, device)
                model = SentenceTransformer(model_name, device=device)
                if device == "cuda":
                    # T4 is Turing: fp16 (not bf16) gives ~2-4x throughput via
                    # tensor cores. Safe here since we L2-normalize the output.
                    model = model.half()
                    logger.info("Using fp16 inference on CUDA")
                _model = model
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
        batch_size=_batch_size,
    )
    return vectors.tolist()
